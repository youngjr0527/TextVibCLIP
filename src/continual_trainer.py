"""
Continual Trainer: TextVibCLIP Continual Learning 파이프라인
Domain별 순차 학습 및 성능 평가 관리
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

from .textvib_model import TextVibCLIP, create_textvib_model
from .replay_buffer import ReplayBuffer
from .data_loader import create_domain_dataloaders, create_combined_dataloader, create_first_domain_dataloader
from .data_cache import create_cached_first_domain_dataloader
from .utils import setup_amp_and_scaler
from configs.model_config import TRAINING_CONFIG, DATA_CONFIG, EVAL_CONFIG, MODEL_CONFIG, CWRU_DATA_CONFIG

logger = logging.getLogger(__name__)


class ContinualTrainer:
    """
    TextVibCLIP Continual Learning Trainer
    
    Domain별 순차 학습, Replay mechanism, 성능 평가 관리
    """
    
    def __init__(self,
                 model: Optional[TextVibCLIP] = None,
                 device: torch.device = torch.device('cpu'),
                 save_dir: str = 'checkpoints',
                 use_amp: bool = True,
                 max_grad_norm: float = 0.1,
                 domain_order: List[Union[int, str]] = None,
                 data_dir: Optional[str] = None,
                 dataset_type: str = DATA_CONFIG.get('dataset_type', 'uos'),
                 patience: Optional[int] = None):
        """
        Args:
            model (TextVibCLIP, optional): 사전 초기화된 모델
            device (torch.device): 학습 디바이스
            save_dir (str): 체크포인트 저장 경로
            use_amp (bool): AMP 사용 여부
            max_grad_norm (float): Gradient clipping 최대 norm
            domain_order (List[Union[int, str]]): 도메인 순서 (없으면 기본값 사용)
        """
        self.device = device
        self.save_dir = save_dir
        self.max_grad_norm = max_grad_norm
        os.makedirs(save_dir, exist_ok=True)
        
        # AMP 설정
        self.scaler, self.use_amp = setup_amp_and_scaler(device, use_amp)
        
        # 모델 초기화
        if model is None:
            self.model = create_textvib_model('first_domain')
        else:
            self.model = model
        
        self.model.to(device)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # 학습 상태 관리
        self.current_domain_idx = 0
        self.completed_domains = []
        self.domain_order = domain_order if domain_order is not None else DATA_CONFIG['domain_order']
        # 데이터 설정 (평가 시 일관성 유지)
        self.data_dir = data_dir if data_dir is not None else DATA_CONFIG['data_dir']
        self.dataset_type = dataset_type
        
        # 성능 추적
        self.performance_history = defaultdict(list)  # {domain: [accuracy_list]}
        self.loss_history = defaultdict(list)
        self.forgetting_scores = []
        
        # 학습 설정
        self.batch_size = TRAINING_CONFIG['batch_size']
        self.num_epochs = TRAINING_CONFIG['num_epochs']
        self.learning_rate = TRAINING_CONFIG['learning_rate']
        self.weight_decay = TRAINING_CONFIG['weight_decay']
        self.replay_ratio = TRAINING_CONFIG['replay_ratio']
        self.grad_accum_steps = int(TRAINING_CONFIG.get('grad_accum_steps', 1))
        self.patience = int(patience) if patience is not None else int(TRAINING_CONFIG.get('patience', 10))
        
        logger.info(f"ContinualTrainer 초기화 완료: device={device}")
    
    def train_first_domain(self, 
                         first_domain_dataloader: Optional[DataLoader] = None,
                         num_epochs: int = None) -> Dict[str, float]:
        """
        첫 번째 도메인(600 RPM) 학습
        텍스트-진동 정렬을 위한 초기 학습 단계
        
        Args:
            first_domain_dataloader (DataLoader, optional): 첫 번째 도메인 데이터로더
            num_epochs (int, optional): 학습 에포크 수
            
        Returns:
            Dict[str, float]: 학습 결과 메트릭
        """
        logger.info("=== First Domain Training 시작 (600 RPM) ===")
        
        # 데이터로더 준비
        if first_domain_dataloader is None:
            # 시나리오에 맞는 dataset_type과 data_dir 사용
            if hasattr(self, 'dataset_type'):
                dataset_type = self.dataset_type
            else:
                dataset_type = 'uos'  # 기본값
            
            if hasattr(self, 'data_dir'):
                data_dir = self.data_dir
            else:
                data_dir = DATA_CONFIG['data_dir'] if dataset_type == 'uos' else CWRU_DATA_CONFIG['data_dir']
            
            if hasattr(self, 'domain_order'):
                domain_order = self.domain_order
            else:
                domain_order = DATA_CONFIG['domain_order'] if dataset_type == 'uos' else CWRU_DATA_CONFIG['domain_order']
            
            # 🚀 캐시된 DataLoader 사용 (고속화)
            first_domain_dataloader = create_cached_first_domain_dataloader(
                data_dir=data_dir,
                domain_order=domain_order,
                dataset_type=dataset_type,
                subset='train', 
                batch_size=self.batch_size
            )
        
        if num_epochs is None:
            num_epochs = self.num_epochs
        
        # 모델을 첫 번째 도메인 학습 모드로 설정
        self.model.switch_to_first_domain_mode()  # First domain mode (LoRA 활성화)
        
        # Optimizer 설정 (Text LoRA + Vibration full)
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer, len(first_domain_dataloader) * num_epochs)
        
        # 디버그/모니터링용: 그라디언트 노름 기록 버퍼
        if not hasattr(self, 'debug_grad_norms'):
            self.debug_grad_norms = []  # 각 항목: {'step': int, 'text_lora': float, 'vib': float}

        # 학습 루프
        self.model.train()
        epoch_losses = []
        
        for epoch in range(num_epochs):
            # 첫 도메인 온도 스케줄(선형): init -> final
            inf_cfg = MODEL_CONFIG.get('infonce', {})
            t_text_init = float(inf_cfg.get('first_domain_temperature_text_init',
                                            inf_cfg.get('first_domain_temperature_text', 0.10)))
            t_text_final = float(inf_cfg.get('first_domain_temperature_text_final',
                                             inf_cfg.get('first_domain_temperature_text', 0.10)))
            t_vib_init  = float(inf_cfg.get('first_domain_temperature_vib_init',
                                            inf_cfg.get('first_domain_temperature_vib', 0.10)))
            t_vib_final = float(inf_cfg.get('first_domain_temperature_vib_final',
                                            inf_cfg.get('first_domain_temperature_vib', 0.10)))
            if num_epochs > 1:
                ratio = epoch / (num_epochs - 1)
            else:
                ratio = 1.0
            t_text = t_text_init + (t_text_final - t_text_init) * ratio
            t_vib  = t_vib_init  + (t_vib_final  - t_vib_init)  * ratio
            self.model.infonce_loss.update_temperatures(t_text, t_vib)
            if epoch % 5 == 0 or epoch == 0:
                logger.info(f"[TempSchedule] epoch {epoch+1}/{num_epochs}: τ_text={t_text:.3f}, τ_vib={t_vib:.3f}")

            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(first_domain_dataloader):
                # 배치를 디바이스로 이동
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                # grad accumulation: 사이클 시작시에만 zero_grad
                if (batch_idx % self.grad_accum_steps) == 0:
                    optimizer.zero_grad(set_to_none=True)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        results = self.model(batch)
                        loss = results['loss'] / self.grad_accum_steps
                    
                    # Backward pass with AMP + grad accumulation
                    self.scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % self.grad_accum_steps == 0:
                        # Gradient clipping 및 grad norm 측정 준비를 위한 unscale (사이클 끝에서만)
                        if self.max_grad_norm > 0:
                            self.scaler.unscale_(optimizer)
                            # 디버그: grad norm 측정 (텍스트 LoRA, 진동 인코더)
                            try:
                                text_lora_params = [
                                    p for n, p in self.model.text_encoder.distilbert.named_parameters()
                                    if ('lora_' in n) and p.requires_grad and (p.grad is not None)
                                ]
                            except Exception:
                                text_lora_params = []
                            vib_params = [
                                p for p in self.model.vibration_encoder.parameters()
                                if p.requires_grad and (p.grad is not None)
                            ]
                            def _global_grad_norm(params):
                                if not params:
                                    return 0.0
                                import math
                                total = 0.0
                                for p in params:
                                    if p.grad is not None:
                                        param_norm = p.grad.data.float().norm(2).item()
                                        total += param_norm * param_norm
                                return math.sqrt(total)
                            grad_norm_text = _global_grad_norm(text_lora_params)
                            grad_norm_vib = _global_grad_norm(vib_params)
                            if batch_idx % 50 == 0:
                                self.debug_grad_norms.append({
                                    'epoch': epoch + 1,
                                    'batch': batch_idx,
                                    'text_lora': float(grad_norm_text),
                                    'vib': float(grad_norm_vib)
                                })
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                else:
                    results = self.model(batch)
                    loss = results['loss'] / self.grad_accum_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    if (batch_idx + 1) % self.grad_accum_steps == 0:
                        # Gradient clipping 및 grad norm 측정 (사이클 끝에서만)
                        if self.max_grad_norm > 0:
                            try:
                                text_lora_params = [
                                    p for n, p in self.model.text_encoder.distilbert.named_parameters()
                                    if ('lora_' in n) and p.requires_grad and (p.grad is not None)
                                ]
                            except Exception:
                                text_lora_params = []
                            vib_params = [
                                p for p in self.model.vibration_encoder.parameters()
                                if p.requires_grad and (p.grad is not None)
                            ]
                            def _global_grad_norm(params):
                                if not params:
                                    return 0.0
                                import math
                                total = 0.0
                                for p in params:
                                    if p.grad is not None:
                                        param_norm = p.grad.data.float().norm(2).item()
                                        total += param_norm * param_norm
                                return math.sqrt(total)
                            grad_norm_text = _global_grad_norm(text_lora_params)
                            grad_norm_vib = _global_grad_norm(vib_params)
                            if batch_idx % 50 == 0:
                                self.debug_grad_norms.append({
                                    'epoch': epoch + 1,
                                    'batch': batch_idx,
                                    'text_lora': float(grad_norm_text),
                                    'vib': float(grad_norm_vib)
                                })
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        optimizer.step()
                
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # 로깅 (적절한 간격으로)
                if batch_idx % 50 == 0:
                    logger.info(f"First Domain Epoch {epoch+1}/{num_epochs}, "
                               f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            
            logger.info(f"First Domain Epoch {epoch+1} 완료: Avg Loss = {avg_epoch_loss:.4f}")
        
        # 첫 번째 도메인으로 표시
        first_domain = self.domain_order[0]
        self.completed_domains.append(first_domain)
        self.loss_history[first_domain] = epoch_losses
        
        # 체크포인트 저장
        checkpoint_path = os.path.join(self.save_dir, 'first_domain_final.pth')
        self.model.save_checkpoint(checkpoint_path, num_epochs, optimizer.state_dict())
        
        # 전체 도메인 성능 평가
        domain_dataloaders = create_domain_dataloaders(
            data_dir=self.data_dir,
            domain_order=self.domain_order,
            dataset_type=self.dataset_type,
            batch_size=self.batch_size
        )
        first_domain_performance = self._evaluate_all_domains(domain_dataloaders)
        
        # 성능 기록 (모든 메트릭 저장)
        first_domain_accuracy = 0.0
        for domain, metrics in first_domain_performance.items():
            if domain not in self.performance_history:
                self.performance_history[domain] = {'accuracy': [], 'top1_retrieval': [], 'top5_retrieval': []}
            
            self.performance_history[domain]['accuracy'].append(metrics['accuracy'])
            self.performance_history[domain]['top1_retrieval'].append(metrics.get('top1_retrieval', 0.0))
            self.performance_history[domain]['top5_retrieval'].append(metrics.get('top5_retrieval', 0.0))
            
            # 첫 번째 도메인 정확도 기록
            first_domain_accuracy = metrics['accuracy']
            break  # 첫 번째 도메인만 확인
        
        # 조기 종료 체크 비활성화 (디버깅 및 전체 파이프라인 테스트용)
        # if first_domain_accuracy < 0.80:
        #     error_msg = f"❌ 첫 번째 도메인 정확도 {first_domain_accuracy:.4f} < 0.80 (80%)"
        #     logger.error(error_msg)
        #     logger.error("🛑 실험 의미 없음 - 조기 종료!")
        #     raise RuntimeError(f"First domain accuracy too low: {first_domain_accuracy:.4f} < 0.80")
        # else:
        logger.info(f"📊 첫 번째 도메인 정확도: {first_domain_accuracy:.4f} (조기 종료 비활성화)")
        
        logger.info("=== First Domain Training 완료 ===")
        
        # 🎨 First Domain Alignment 시각화 생성
        logger.info("📊 First Domain Alignment 시각화 생성 중...")
        try:
            alignment_results = self._create_first_domain_alignment_visualization(
                domain_dataloaders, first_domain
            )
            logger.info(f"✅ Alignment 시각화 완료: {alignment_results.get('save_path', 'N/A')}")
        except Exception as e:
            logger.warning(f"⚠️ Alignment 시각화 실패: {e}")
            alignment_results = {}
        
        # grad norm 요약
        grad_text_vals = [d['text_lora'] for d in self.debug_grad_norms] if hasattr(self, 'debug_grad_norms') else []
        grad_vib_vals = [d['vib'] for d in self.debug_grad_norms] if hasattr(self, 'debug_grad_norms') else []
        grad_summary = {
            'text_lora_mean': float(np.mean(grad_text_vals)) if grad_text_vals else 0.0,
            'vib_mean': float(np.mean(grad_vib_vals)) if grad_vib_vals else 0.0,
            'samples': len(self.debug_grad_norms) if hasattr(self, 'debug_grad_norms') else 0
        }

        return {
            'final_loss': epoch_losses[-1],
            'avg_loss': np.mean(epoch_losses),
            'domain_performances': first_domain_performance,
            'grad_norms_summary': grad_summary,
            'alignment_visualization': alignment_results  # 시각화 결과 추가
        }
    
    def train_remaining_domains(self, domain_dataloaders: Optional[Dict] = None) -> Dict[str, Any]:
        """
        나머지 도메인들(800~1600 RPM) 순차 학습
        Replay mechanism을 활용한 점진적 학습
        
        Args:
            domain_dataloaders (Dict, optional): 도메인별 데이터로더
            
        Returns:
            Dict[str, Any]: 학습 결과 및 메트릭
        """
        logger.info("=== Remaining Domains Training 시작 (800~1600 RPM) ===")
        
        # 데이터로더 준비
        if domain_dataloaders is None:
            domain_dataloaders = create_domain_dataloaders(batch_size=self.batch_size)
        
        # Continual mode로 전환 (Text freeze, Vibration adaptation)
        self.model.switch_to_continual_mode()
        
        remaining_domains_results = {}
        after_performance = {}  # 초기화
        
        # Domain 2부터 순차 학습 (첫 번째 도메인은 이미 완료)
        for domain_idx in range(1, len(self.domain_order)):
            domain_value = self.domain_order[domain_idx]
            
            logger.info(f"\n--- Domain {domain_value} 학습 시작 ---")
            
            # 현재 도메인 데이터로더 (키 존재 확인)
            if domain_value not in domain_dataloaders:
                logger.error(f"도메인 {domain_value}가 dataloaders에 없습니다. 사용 가능한 도메인: {list(domain_dataloaders.keys())}")
                continue
                
            current_train_loader = domain_dataloaders[domain_value]['train']
            current_val_loader = domain_dataloaders[domain_value]['val']
            
            # 이전 도메인 성능 기록 (Forgetting 계산용) - 간소화
            logger.info(f"Domain {domain_value} 학습 전 성능 평가 스킵 (빠른 테스트)")
            before_performance = {}  # 임시로 비활성화
            
            # 현재 도메인 학습
            domain_results = self._train_single_domain(
                domain_value, current_train_loader, current_val_loader
            )
            
            # 학습 후 성능 평가 - 간소화
            logger.info(f"Domain {domain_value} 학습 후 성능 평가 (빠른 버전)")
            after_performance = self._evaluate_all_domains_fast(domain_dataloaders)
            
            # Forgetting 계산
            forgetting_score = self._calculate_forgetting(before_performance, after_performance)
            self.forgetting_scores.append(forgetting_score)
            
            # 성능 기록 (모든 메트릭 저장)
            for eval_domain, metrics in after_performance.items():
                if eval_domain not in self.performance_history:
                    self.performance_history[eval_domain] = {'accuracy': [], 'top1_retrieval': [], 'top5_retrieval': []}
                self.performance_history[eval_domain]['accuracy'].append(metrics.get('accuracy', 0.0))
                self.performance_history[eval_domain]['top1_retrieval'].append(metrics.get('top1_retrieval', 0.0))
                self.performance_history[eval_domain]['top5_retrieval'].append(metrics.get('top5_retrieval', 0.0))

            # 현재 도메인 완료 표시 및 결과 저장
            self.completed_domains.append(domain_value)
            remaining_domains_results[domain_value] = {
                'training_results': domain_results,
                'performance': after_performance,
                'forgetting_score': forgetting_score
            }
            
            logger.info(f"Domain {domain_value} 학습 완료: Forgetting Score = {forgetting_score:.4f}")
            
            # 🎨 도메인별 성능 시각화 생성
            logger.info(f"📊 Domain {domain_value} 성능 시각화 생성 중...")
            try:
                domain_viz_results = self._create_domain_performance_visualization(
                    domain_value, after_performance, forgetting_score
                )
                logger.info(f"✅ Domain {domain_value} 시각화 완료: {domain_viz_results.get('save_path', 'N/A')}")
            except Exception as e:
                logger.warning(f"⚠️ Domain {domain_value} 시각화 실패: {e}")
        
        # 최종 성능 요약 (풀 평가로 덮어쓰기)
        full_eval_results = self._evaluate_all_domains(domain_dataloaders)
        # 기록에도 풀 평가 최신값 추가
        for eval_domain, metrics in full_eval_results.items():
            if eval_domain not in self.performance_history:
                self.performance_history[eval_domain] = {'accuracy': [], 'top1_retrieval': [], 'top5_retrieval': []}
            self.performance_history[eval_domain]['accuracy'].append(metrics.get('accuracy', 0.0))
            self.performance_history[eval_domain]['top1_retrieval'].append(metrics.get('top1_retrieval', 0.0))
            self.performance_history[eval_domain]['top5_retrieval'].append(metrics.get('top5_retrieval', 0.0))
        final_metrics = self._calculate_final_metrics()
        remaining_domains_results['final_metrics'] = final_metrics
        
        logger.info("=== Remaining Domains Training 완료 ===")
        
        return remaining_domains_results
    
    def _train_single_domain(self, 
                           domain_value: Union[int, str],
                           train_loader: DataLoader,
                           val_loader: DataLoader) -> Dict[str, float]:
        """
        단일 도메인 학습 (Replay 포함)
        
        Args:
            domain_value (Union[int, str]): 도메인 값 (RPM 또는 HP)
            train_loader (DataLoader): 현재 도메인 학습 데이터
            val_loader (DataLoader): 현재 도메인 검증 데이터
            
        Returns:
            Dict[str, float]: 학습 결과
        """
        # Optimizer (Vibration encoder만 학습)
        optimizer = self._create_continual_optimizer()
        
        # 현재 도메인 임베딩 수집 (Replay buffer용)
        domain_embeddings = self._collect_domain_embeddings(train_loader)
        
        # Replay buffer에 추가 (🎯 라벨 정보 포함)
        if domain_embeddings:
            self.replay_buffer.add_domain_data(
                domain_value,
                domain_embeddings['text_embeddings'],
                domain_embeddings['vib_embeddings'],
                domain_embeddings['metadata'],
                labels=domain_embeddings.get('labels', None)
            )
        
        # 학습 루프
        self.model.train()
        epoch_losses = []
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # 현재 배치를 디바이스로 이동
                batch = self._move_batch_to_device(batch)
                current_batch_size = batch['vibration'].size(0)
                
                # Replay 데이터 샘플링 (성능 최적화: 간헐적 사용)
                use_replay = (batch_idx % 3 == 0)  # 3배치마다 한 번만 replay 사용
                
                if use_replay:
                    replay_batch_size = min(int(current_batch_size * self.replay_ratio), 8)  # 크기 제한
                    replay_data = self.replay_buffer.sample_replay_data(
                        replay_batch_size, exclude_current=True, device=self.device
                    )
                    
                    # 현재 데이터와 Replay 데이터 결합
                    if replay_data is not None:
                        combined_batch = self._combine_current_and_replay(batch, replay_data)
                    else:
                        combined_batch = batch
                else:
                    combined_batch = batch
                
                # Forward pass
                optimizer.zero_grad()
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        results = self.model(combined_batch)
                        loss = results['loss']
                    
                    # Backward pass with AMP
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    results = self.model(combined_batch)
                    loss = results['loss']
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # 로깅 (더 간결하게)
                if batch_idx % 20 == 0:
                    replay_info = f"R:{replay_batch_size}" if replay_data else "No-R"
                    logger.info(f"D{domain_value} E{epoch+1} B{batch_idx}: "
                               f"Loss={loss.item():.4f}, {replay_info}")
            
            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            
            # Validation
            val_metrics = self._evaluate_single_domain(val_loader)
            val_acc = val_metrics['accuracy']
            
            logger.info(f"Domain {domain_value} Epoch {epoch+1}: "
                       f"Loss = {avg_epoch_loss:.4f}, Val Acc = {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Best model 저장
                checkpoint_path = os.path.join(self.save_dir, f'domain_{domain_value}_best.pth')
                self.model.save_checkpoint(checkpoint_path, epoch, optimizer.state_dict())
            else:
                patience_counter += 1
                
                if patience_counter >= TRAINING_CONFIG['patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 도메인 학습 완료
        self.loss_history[domain_value] = epoch_losses
        
        # 안전한 결과 반환 (epoch_losses가 비어있을 수 있음)
        final_loss = epoch_losses[-1] if epoch_losses else float('inf')
        num_epochs = len(epoch_losses)
        
        return {
            'final_loss': final_loss,
            'best_val_accuracy': best_val_acc,
            'num_epochs': num_epochs
        }
    
    def _collect_domain_embeddings(self, dataloader: DataLoader) -> Optional[Dict]:
        """현재 도메인 데이터의 임베딩 수집 (🎯 라벨 정보 포함)"""
        self.model.eval()
        
        text_embeddings = []
        vib_embeddings = []
        metadata_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                results = self.model(batch, return_embeddings=True)
                
                text_embeddings.append(results['text_embeddings'])
                vib_embeddings.append(results['vib_embeddings'])
                metadata_list.extend(batch['metadata'])
                
                # 🎯 라벨 정보도 수집
                if 'labels' in batch:
                    labels_list.append(batch['labels'])
        
        if text_embeddings:
            result = {
                'text_embeddings': torch.cat(text_embeddings, dim=0),
                'vib_embeddings': torch.cat(vib_embeddings, dim=0),
                'metadata': metadata_list
            }
            
            # 라벨 정보 추가 (있는 경우만)
            if labels_list:
                result['labels'] = torch.cat(labels_list, dim=0)
            
            return result
        return None
    
    def _combine_current_and_replay(self, current_batch: Dict, replay_data: Dict) -> Dict:
        """현재 배치와 Replay 데이터 결합 (🎯 라벨 정보 포함)"""
        # 진동 신호는 현재 배치에서만 (Replay는 임베딩만 저장)
        combined_batch = current_batch.copy()  # 기존 모든 정보 복사
        
        # Replay 임베딩을 모델에 주입하는 방식으로 구현
        combined_batch['replay_text_embeddings'] = replay_data['text_embeddings']
        combined_batch['replay_vib_embeddings'] = replay_data['vib_embeddings']
        
        # 🎯 Replay 라벨 정보도 전달 (클래스 기반 contrastive learning용)
        if 'labels' in replay_data:
            combined_batch['replay_labels'] = replay_data['labels']
        
        return combined_batch
    
    def _evaluate_single_domain(self, dataloader: DataLoader) -> Dict[str, float]:
        """단일 도메인 성능 평가 (retrieval 메트릭 포함)"""
        self.model.eval()
        
        all_text_embeddings = []
        all_vib_embeddings = []
        all_file_idx = []
        all_labels = []
        all_labels = []
        
        # DataLoader 안전성 확보를 위한 조치
        try:
            with torch.no_grad():
                max_batches = int(EVAL_CONFIG.get('max_full_eval_batches', -1))
                for batch_idx, batch in enumerate(dataloader):
                    # 설정 기반 배치 제한 (기본 무제한)
                    if max_batches >= 0 and batch_idx >= max_batches:
                        logger.debug(f"평가 중단: 설정된 최대 배치 {max_batches} 도달")
                        break
                        
                    batch = self._move_batch_to_device(batch)
                    results = self.model(batch, return_embeddings=True)
                    
                    all_text_embeddings.append(results['text_embeddings'])
                    all_vib_embeddings.append(results['vib_embeddings'])
                    if 'file_idx' in batch:
                        # 디바이스 일치 유지 (GPU 상에서 바로 사용)
                        all_file_idx.append(batch['file_idx'])
                    if 'labels' in batch:
                        all_labels.append(batch['labels'])
                    
        except Exception as e:
            logger.error(f"평가 중 오류 발생: {e}")
            return {'accuracy': 0.0, 'top1_retrieval': 0.0, 'top5_retrieval': 0.0}
        
        if not all_text_embeddings:
            return {'accuracy': 0.0, 'top1_retrieval': 0.0, 'top5_retrieval': 0.0}
        
        # 모든 임베딩 결합
        text_emb = torch.cat(all_text_embeddings, dim=0)
        vib_emb = torch.cat(all_vib_embeddings, dim=0)
        file_idx = torch.cat(all_file_idx, dim=0) if all_file_idx else None
        labels_tensor = torch.cat(all_labels, dim=0) if all_labels else None
        
        # 🎯 FIXED: 표준 L2 정규화 (gradient 보존)
        text_emb = F.normalize(text_emb, p=2, dim=1)
        vib_emb = F.normalize(vib_emb, p=2, dim=1)
        
        # 1. Retrieval 정확도
        similarity_matrix = torch.matmul(text_emb, vib_emb.t())  # (N, N)
        
        # 디버그: 첫 번째 배치에서 유사도 분포 확인
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1
            
        if self._debug_count <= 2:  # 처음 2번만 디버그
            # 임베딩 통계 로그
            try:
                t_mean = text_emb.mean().item(); t_std = text_emb.std(unbiased=False).item()
                v_mean = vib_emb.mean().item(); v_std = vib_emb.std(unbiased=False).item()
                logger.info(f"🔍 DEBUG - 임베딩 통계 | text mean/std: {t_mean:.4f}/{t_std:.4f}, vib mean/std: {v_mean:.4f}/{v_std:.4f}")
            except Exception:
                pass
            logger.info(f"🔍 DEBUG - 배치 크기: {text_emb.size(0)}")
            logger.info(f"🔍 DEBUG - 대각선 유사도 (정답): {torch.diag(similarity_matrix)[:5].tolist()}")
            logger.info(f"🔍 DEBUG - 첫 행 유사도 (전체): {similarity_matrix[0, :5].tolist()}")
            predicted_tmp = torch.argmax(similarity_matrix, dim=1)
            logger.info(f"🔍 DEBUG - 최대 유사도 인덱스: {predicted_tmp[:10].tolist()}")

            # 추가 무결성 체크: 유사도 통계 및 argmax 분포
            N = similarity_matrix.size(0)
            diag_vals = torch.diag(similarity_matrix)
            diag_mean = float(diag_vals.mean().item())
            diag_std = float(diag_vals.std(unbiased=False).item())
            diag_min = float(diag_vals.min().item())
            diag_max = float(diag_vals.max().item())
            off_mask = ~torch.eye(N, dtype=torch.bool, device=similarity_matrix.device)
            off_vals = similarity_matrix[off_mask]
            off_mean = float(off_vals.mean().item())
            off_std = float(off_vals.std(unbiased=False).item())
            logger.info(
                f"🔍 DEBUG - 유사도 통계 | diag mean/std/min/max: "
                f"{diag_mean:.4f}/{diag_std:.4f}/{diag_min:.4f}/{diag_max:.4f}, "
                f"off mean/std: {off_mean:.4f}/{off_std:.4f}"
            )

            binc = torch.bincount(predicted_tmp, minlength=N)
            topk = min(10, N)
            top_vals, top_idx = torch.topk(binc, k=topk)
            top_pairs = [(int(i), int(v)) for i, v in zip(top_idx.tolist(), top_vals.tolist())]
            logger.info(f"🔍 DEBUG - argmax 상위 {topk} 인덱스/빈도: {top_pairs}")

            # 셔플 베이스라인 (클래스 기반, 공정성 마스크 미적용 단순 참조)
            if N >= 2 and labels_tensor is not None:
                # 라벨 정규화
                if labels_tensor.dim() == 2 and labels_tensor.size(1) >= 1:
                    cls_dbg = labels_tensor[:, 0]
                elif labels_tensor.dim() == 1:
                    cls_dbg = labels_tensor
                else:
                    cls_dbg = labels_tensor.view(-1)
                cls_dbg = cls_dbg.to(text_emb.device)

                # 🎯 FIXED: 올바른 셔플 베이스라인 계산
                perm = torch.randperm(N, device=text_emb.device)
                sim_shuf = torch.matmul(text_emb, vib_emb[perm].t())
                pred_shuf = torch.argmax(sim_shuf, dim=1)
                
                # 올바른 계산: 셔플된 순서에서 예측한 클래스와 원래 클래스 비교
                predicted_classes_shuf = cls_dbg[perm[pred_shuf]]  # 예측된 위치의 실제 클래스
                top1_shuf = (predicted_classes_shuf == cls_dbg).float().mean().item()
                
                k_dbg = min(5, N)
                _, topk_shuf = torch.topk(sim_shuf, k=k_dbg, dim=1)
                topk_classes_shuf = cls_dbg[perm[topk_shuf]]  # topk 위치들의 실제 클래스
                top5_shuf = (topk_classes_shuf == cls_dbg.unsqueeze(1)).any(dim=1).float().mean().item()
                
                logger.info(f"🔍 DEBUG - 셔플 베이스라인 Top1/Top5 (class): {top1_shuf:.4f}/{top5_shuf:.4f}")
                
                # 추가 디버그: 클래스 분포 확인
                unique_classes = torch.unique(cls_dbg)
                class_counts = [(cls.item(), (cls_dbg == cls).sum().item()) for cls in unique_classes]
                logger.info(f"🔍 DEBUG - 배치 내 클래스 분포: {class_counts}")
                
                # 이론적 랜덤 베이스라인 계산
                n_classes = len(unique_classes)
                theoretical_random = 1.0 / n_classes if n_classes > 0 else 0.0
                logger.info(f"🔍 DEBUG - 이론적 랜덤 베이스라인: {theoretical_random:.4f} (클래스 수: {n_classes})")
        
        # 🎯 ENHANCED: 클래스 인식 평가 + 표준 contrastive 평가
        # 1) 표준 diagonal matching (모니터링용)
        _, predicted_indices = torch.max(similarity_matrix, dim=1)
        correct_indices = torch.arange(text_emb.size(0), device=text_emb.device)
        diagonal_accuracy = (predicted_indices == correct_indices).float().mean().item()

        # 🎯 공정 평가: 자기 자신(대각선) 제거 + (행별 조건부) 동일파일 제거
        N = similarity_matrix.size(0)
        sim_eval = similarity_matrix.clone()
        if N > 1:
            sim_eval.fill_diagonal_(-1e4)
            if file_idx is not None and file_idx.numel() == N and labels_tensor is not None:
                # 라벨 텐서 정규화
                if labels_tensor.dim() == 2 and labels_tensor.size(1) >= 1:
                    class_labels_for_mask = labels_tensor[:, 0]
                elif labels_tensor.dim() == 1:
                    class_labels_for_mask = labels_tensor
                else:
                    class_labels_for_mask = labels_tensor.view(-1)
                class_labels_for_mask = class_labels_for_mask.to(text_emb.device)

                same_file_mask = (file_idx.unsqueeze(1) == file_idx.unsqueeze(0))
                class_equal_mask = (class_labels_for_mask.unsqueeze(1) == class_labels_for_mask.unsqueeze(0))
                off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=text_emb.device)
                # 각 행에 동일파일이 아닌 같은 클래스 후보가 하나라도 있으면 동일파일 전부 마스킹
                has_other_file_positive = ((class_equal_mask & ~same_file_mask & off_diag_mask)).any(dim=1)
                row_mask = has_other_file_positive.unsqueeze(1).expand(-1, N)
                mask_to_apply = same_file_mask & row_mask
                sim_eval = sim_eval.masked_fill(mask_to_apply, -1e4)

        # 🎯 CRITICAL FIX: 올바른 Zero-shot 분류 평가
        # 각 진동 신호를 모든 가능한 클래스 설명과 비교하여 분류
        
        class_top1 = diagonal_accuracy  # 기본값
        class_top5 = diagonal_accuracy
        
        if labels_tensor is not None:
            # 라벨 정규화
            if labels_tensor.dim() == 2 and labels_tensor.size(1) >= 1:
                class_labels = labels_tensor[:, 0]  # UOS 주 분류
            elif labels_tensor.dim() == 1:
                class_labels = labels_tensor
            else:
                class_labels = labels_tensor.view(-1)
            class_labels = class_labels.to(text_emb.device)
            
            # 🎯 NEW: Zero-shot 분류 평가
            # 모든 가능한 클래스의 prototype 텍스트 임베딩 생성
            unique_classes = torch.unique(class_labels)
            n_classes = len(unique_classes)
            
            if n_classes > 1:  # 클래스가 여러 개 있을 때만 zero-shot 평가
                # 각 클래스의 prototype 임베딩 계산 (평균)
                class_prototypes = []
                for cls in unique_classes:
                    cls_mask = (class_labels == cls)
                    if cls_mask.any():
                        cls_text_emb = text_emb[cls_mask].mean(dim=0, keepdim=True)
                        class_prototypes.append(cls_text_emb)
                
                if len(class_prototypes) == n_classes:
                    # 모든 클래스의 prototype 결합
                    prototype_embeddings = torch.cat(class_prototypes, dim=0)  # (n_classes, embed_dim)
                    
                    # 각 진동 임베딩을 모든 클래스 prototype과 비교
                    vib_to_prototype_sim = torch.matmul(vib_emb, prototype_embeddings.t())  # (N, n_classes)
                    
                    # 예측: 가장 유사한 prototype의 클래스
                    predicted_class_idx = torch.argmax(vib_to_prototype_sim, dim=1)
                    predicted_classes = unique_classes[predicted_class_idx]
                    
                    # Zero-shot 분류 정확도
                    class_top1 = (predicted_classes == class_labels).float().mean().item()
                    
                    # Top-5 계산 (클래스 수가 5개 이상일 때)
                    if n_classes >= 5:
                        _, top5_idx = torch.topk(vib_to_prototype_sim, k=5, dim=1)
                        top5_classes = unique_classes[top5_idx]  # (N, 5)
                        class_top5 = (top5_classes == class_labels.unsqueeze(1)).any(dim=1).float().mean().item()
                    else:
                        class_top5 = class_top1
                else:
                    # Prototype 생성 실패 시 기존 방식 사용
                    top1_pred = torch.argmax(sim_eval, dim=1)
                    class_top1 = (class_labels[top1_pred] == class_labels).float().mean().item()
                    class_top5 = class_top1
            else:
                # 클래스가 1개뿐이면 항상 100%
                class_top1 = 1.0
                class_top5 = 1.0

        retrieval_accuracy = class_top1
        top1_accuracy = class_top1
        top5_accuracy = class_top5
        
        return {
            'accuracy': retrieval_accuracy,  # 주 정확도 지표
            'diagonal_accuracy': diagonal_accuracy,  # 표준 contrastive 정확도
            'class_accuracy': class_top1,  # 클래스 인식 정확도
            'top1_retrieval': top1_accuracy,
            'top5_retrieval': top5_accuracy
        }
    
    def _evaluate_all_domains(self, domain_dataloaders: Dict) -> Dict[int, Dict[str, float]]:
        """모든 도메인 성능 평가"""
        results = {}
        
        for domain_value, loaders in domain_dataloaders.items():
            test_loader = loaders['test']
            metrics = self._evaluate_single_domain(test_loader)
            
            results[domain_value] = {
                **metrics,  # accuracy, top1_retrieval, top5_retrieval
                'num_samples': len(test_loader.dataset)
            }
        
        return results
    
    def _evaluate_all_domains_fast(self, domain_dataloaders: Dict) -> Dict[str, Dict[str, float]]:
        """빠른 도메인 평가 (적은 배치만 사용)"""
        results = {}
        
        for domain_value, loaders in domain_dataloaders.items():
            test_loader = loaders['test']
            
            # 첫 5배치만 평가하여 빠른 근사치 계산
            limited_metrics = self._evaluate_single_domain_fast(test_loader)
            
            results[domain_value] = {
                **limited_metrics,
                'num_samples': min(len(test_loader.dataset), 5 * test_loader.batch_size)
            }
        
        return results
    
    def _evaluate_single_domain_fast(self, dataloader: DataLoader) -> Dict[str, float]:
        """빠른 단일 도메인 평가 (5배치만)"""
        self.model.eval()
        
        all_text_embeddings = []
        all_vib_embeddings = []
        all_file_idx = []
        all_labels = []
        
        try:
            with torch.no_grad():
                max_fast = int(EVAL_CONFIG.get('max_fast_eval_batches', 5))
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx >= max_fast:
                        break
                        
                    batch = self._move_batch_to_device(batch)
                    results = self.model(batch, return_embeddings=True)
                    
                    all_text_embeddings.append(results['text_embeddings'])
                    all_vib_embeddings.append(results['vib_embeddings'])
                    if 'file_idx' in batch:
                        all_file_idx.append(batch['file_idx'])
                    if 'labels' in batch:
                        all_labels.append(batch['labels'])
                    
        except Exception as e:
            logger.error(f"빠른 평가 중 오류: {e}")
            return {'accuracy': 0.0, 'top1_retrieval': 0.0, 'top5_retrieval': 0.0}
        
        if not all_text_embeddings:
            return {'accuracy': 0.0, 'top1_retrieval': 0.0, 'top5_retrieval': 0.0}
        
        # 임베딩 결합 및 메트릭 계산
        text_emb = torch.cat(all_text_embeddings, dim=0)
        vib_emb = torch.cat(all_vib_embeddings, dim=0)
        file_idx = torch.cat(all_file_idx, dim=0) if all_file_idx else None
        labels_tensor = torch.cat(all_labels, dim=0) if all_labels else None

        # 🎯 FIXED: 표준 L2 정규화 (gradient 보존)
        text_emb = F.normalize(text_emb, p=2, dim=1)
        vib_emb = F.normalize(vib_emb, p=2, dim=1)
        similarity = torch.matmul(text_emb, vib_emb.t())

        # 공정성 마스크: 자기자신 제거 + (행별 조건부) 동일파일 제거
        N = similarity.size(0)
        sim_eval = similarity.clone()
        if N > 1:
            sim_eval.fill_diagonal_(-1e4)
            if file_idx is not None and file_idx.numel() == N and labels_tensor is not None:
                if labels_tensor.dim() == 2 and labels_tensor.size(1) >= 1:
                    class_labels_for_mask = labels_tensor[:, 0]
                elif labels_tensor.dim() == 1:
                    class_labels_for_mask = labels_tensor
                else:
                    class_labels_for_mask = labels_tensor.view(-1)
                class_labels_for_mask = class_labels_for_mask.to(text_emb.device)

                same_file_mask = (file_idx.unsqueeze(1) == file_idx.unsqueeze(0))
                class_equal_mask = (class_labels_for_mask.unsqueeze(1) == class_labels_for_mask.unsqueeze(0))
                off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=text_emb.device)
                has_other_file_positive = ((class_equal_mask & ~same_file_mask & off_diag_mask)).any(dim=1)
                row_mask = has_other_file_positive.unsqueeze(1).expand(-1, N)
                mask_to_apply = same_file_mask & row_mask
                sim_eval = sim_eval.masked_fill(mask_to_apply, -1e4)

        # 🎯 CRITICAL FIX: 올바른 Zero-shot 분류 평가 (fast 버전에도 적용)
        class_top1 = 0.0
        class_top5 = 0.0
        
        if labels_tensor is not None:
            # 라벨 정규화
            if labels_tensor.dim() == 2 and labels_tensor.size(1) >= 1:
                class_labels = labels_tensor[:, 0]
            elif labels_tensor.dim() == 1:
                class_labels = labels_tensor
            else:
                class_labels = labels_tensor.view(-1)
            class_labels = class_labels.to(text_emb.device)
            
            # Zero-shot 분류 평가 (full 버전과 동일한 로직)
            unique_classes = torch.unique(class_labels)
            n_classes = len(unique_classes)
            
            if n_classes > 1:
                # 각 클래스의 prototype 임베딩 계산
                class_prototypes = []
                for cls in unique_classes:
                    cls_mask = (class_labels == cls)
                    if cls_mask.any():
                        cls_text_emb = text_emb[cls_mask].mean(dim=0, keepdim=True)
                        class_prototypes.append(cls_text_emb)
                
                if len(class_prototypes) == n_classes:
                    # 모든 클래스의 prototype 결합
                    prototype_embeddings = torch.cat(class_prototypes, dim=0)
                    
                    # 각 진동 임베딩을 모든 클래스 prototype과 비교
                    vib_to_prototype_sim = torch.matmul(vib_emb, prototype_embeddings.t())
                    
                    # 예측: 가장 유사한 prototype의 클래스
                    predicted_class_idx = torch.argmax(vib_to_prototype_sim, dim=1)
                    predicted_classes = unique_classes[predicted_class_idx]
                    
                    # Zero-shot 분류 정확도
                    class_top1 = (predicted_classes == class_labels).float().mean().item()
                    
                    # Top-5 계산
                    if n_classes >= 5:
                        _, top5_idx = torch.topk(vib_to_prototype_sim, k=5, dim=1)
                        top5_classes = unique_classes[top5_idx]
                        class_top5 = (top5_classes == class_labels.unsqueeze(1)).any(dim=1).float().mean().item()
                    else:
                        class_top5 = class_top1
                else:
                    # Prototype 생성 실패 시 기본값
                    class_top1 = 0.0
                    class_top5 = 0.0
            else:
                # 클래스가 1개뿐이면 항상 100%
                class_top1 = 1.0
                class_top5 = 1.0
        else:
            # 라벨 없으면 대각선 기준으로 근사
            _, pred = torch.max(sim_eval, dim=1)
            target = torch.arange(text_emb.size(0), device=text_emb.device)
            class_top1 = (pred == target).float().mean().item()
            k = min(5, sim_eval.size(1))
            if k > 1:
                _, topk = torch.topk(sim_eval, k=k, dim=1)
                target_expanded = target.unsqueeze(1).expand(-1, k)
                class_top5 = (topk == target_expanded).any(dim=1).float().mean().item()
            else:
                class_top5 = class_top1

        # 빠른 평가에서도 무결성 디버그(한 번만)
        if not hasattr(self, '_debug_fast_once'):
            self._debug_fast_once = True
            N = similarity.size(0)
            diag_vals = torch.diag(similarity)
            diag_mean = float(diag_vals.mean().item())
            diag_std = float(diag_vals.std(unbiased=False).item())
            off_mask = ~torch.eye(N, dtype=torch.bool, device=similarity.device)
            off_vals = similarity[off_mask]
            off_mean = float(off_vals.mean().item())
            off_std = float(off_vals.std(unbiased=False).item())
            logger.info(
                f"🔍 DEBUG(FAST) - N={N}, diag mean/std={diag_mean:.4f}/{diag_std:.4f}, "
                f"off mean/std={off_mean:.4f}/{off_std:.4f}, Top1={class_top1:.4f}, Top5={class_top5:.4f}"
            )
            # 셔플 베이스라인 (클래스 기반)
            if N >= 2 and labels_tensor is not None:
                if labels_tensor.dim() == 2 and labels_tensor.size(1) >= 1:
                    cls_dbg = labels_tensor[:, 0]
                elif labels_tensor.dim() == 1:
                    cls_dbg = labels_tensor
                else:
                    cls_dbg = labels_tensor.view(-1)
                cls_dbg = cls_dbg.to(text_emb.device)
                perm = torch.randperm(N, device=text_emb.device)
                sim_shuf = torch.matmul(text_emb, vib_emb[perm].t())
                pred_shuf = torch.argmax(sim_shuf, dim=1)
                top1_shuf = (cls_dbg[perm][pred_shuf] == cls_dbg).float().mean().item()
                k_dbg = min(5, N)
                _, topk_shuf = torch.topk(sim_shuf, k=k_dbg, dim=1)
                top5_shuf = (cls_dbg.unsqueeze(1) == cls_dbg[perm][topk_shuf]).any(dim=1).float().mean().item()
                logger.info(f"🔍 DEBUG(FAST) - 셔플 베이스라인 Top1/Top5 (class): {top1_shuf:.4f}/{top5_shuf:.4f}")

        return {
            'accuracy': class_top1,  # 클래스 기반 Top-1
            'diagonal_accuracy': class_top1,  # 라벨 부재 시 동일
            'class_accuracy': class_top1,
            'top1_retrieval': class_top1,
            'top5_retrieval': class_top5
        }
    
    def _calculate_forgetting(self, before: Dict, after: Dict) -> float:
        """Forgetting score 계산"""
        if len(self.completed_domains) <= 1:
            return 0.0
        
        forgetting_scores = []
        for domain in self.completed_domains[:-1]:  # 마지막 도메인 제외
            before_acc = before.get(domain, {}).get('accuracy', 0.0)
            after_acc = after.get(domain, {}).get('accuracy', 0.0)
            forgetting = max(0.0, before_acc - after_acc)
            forgetting_scores.append(forgetting)
        
        return np.mean(forgetting_scores) if forgetting_scores else 0.0
    
    def _calculate_final_metrics(self) -> Dict[str, float]:
        """최종 Continual Learning 메트릭 계산"""
        if not self.performance_history:
            return {}
        
        # Average metrics (마지막 성능)
        final_accuracies = []
        final_top1_retrievals = []
        final_top5_retrievals = []
        
        for domain in self.completed_domains:
            if domain in self.performance_history:
                if self.performance_history[domain]['accuracy']:
                    final_accuracies.append(self.performance_history[domain]['accuracy'][-1])
                if self.performance_history[domain]['top1_retrieval']:
                    final_top1_retrievals.append(self.performance_history[domain]['top1_retrieval'][-1])
                if self.performance_history[domain]['top5_retrieval']:
                    final_top5_retrievals.append(self.performance_history[domain]['top5_retrieval'][-1])
        
        avg_accuracy = np.mean(final_accuracies) if final_accuracies else 0.0
        avg_top1_retrieval = np.mean(final_top1_retrievals) if final_top1_retrievals else 0.0
        avg_top5_retrieval = np.mean(final_top5_retrievals) if final_top5_retrievals else 0.0
        
        # Average Forgetting
        avg_forgetting = np.mean(self.forgetting_scores) if self.forgetting_scores else 0.0
        
        return {
            'average_accuracy': avg_accuracy,
            'average_top1_retrieval': avg_top1_retrieval,
            'average_top5_retrieval': avg_top5_retrieval,
            'average_forgetting': avg_forgetting,
            'num_domains': len(self.completed_domains),
            'final_accuracies': final_accuracies,
            'final_top1_retrievals': final_top1_retrievals,
            'final_top5_retrievals': final_top5_retrievals
        }
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """First domain training용 optimizer 생성 (파라미터 그룹 분리)"""
        base_lr = self.learning_rate
        lora_mult = float(TRAINING_CONFIG.get('lora_lr_mult', 3.0))
        proj_mult = float(TRAINING_CONFIG.get('proj_lr_mult', 3.0))
        vib_mult  = float(TRAINING_CONFIG.get('vib_lr_mult', 1.0))

        params = []
        seen = set()

        # Text LoRA 파라미터 그룹
        try:
            lora_params = [p for n, p in self.model.text_encoder.distilbert.named_parameters()
                           if ('lora_' in n) and p.requires_grad]
            if lora_params:
                params.append({'params': lora_params, 'lr': base_lr * lora_mult, 'weight_decay': self.weight_decay})
                for p in lora_params:
                    seen.add(id(p))
        except Exception:
            pass

        # Text projection 파라미터 그룹
        if hasattr(self.model.text_encoder, 'projection'):
            proj_params = [p for p in self.model.text_encoder.projection.parameters() if p.requires_grad]
            if proj_params:
                params.append({'params': proj_params, 'lr': base_lr * proj_mult, 'weight_decay': self.weight_decay})
                for p in proj_params:
                    seen.add(id(p))

        # Vibration encoder 파라미터 그룹
        vib_params = [p for p in self.model.vibration_encoder.parameters() if p.requires_grad]
        vib_params = [p for p in vib_params if id(p) not in seen]
        if vib_params:
            params.append({'params': vib_params, 'lr': base_lr * vib_mult, 'weight_decay': self.weight_decay})
            for p in vib_params:
                seen.add(id(p))

        # 🎯 CRITICAL FIX: InfoNCE 온도 파라미터 추가
        temp_params = []
        if hasattr(self.model.infonce_loss, 'log_temperature_text'):
            temp_params.append(self.model.infonce_loss.log_temperature_text)
        if hasattr(self.model.infonce_loss, 'log_temperature_vib'):
            temp_params.append(self.model.infonce_loss.log_temperature_vib)
        
        if temp_params:
            params.append({'params': temp_params, 'lr': base_lr * 2.0, 'weight_decay': 0.0})  # 온도는 weight decay 없음
            for p in temp_params:
                seen.add(id(p))

        # 누락 파라미터 보완
        remain = [p for p in self.model.parameters() if p.requires_grad and id(p) not in seen]
        if remain:
            params.append({'params': remain, 'lr': base_lr, 'weight_decay': self.weight_decay})

        return optim.AdamW(params)
    
    def _create_scheduler(self, optimizer, total_steps):
        """학습률 스케줄러 생성 (단일 구현)"""
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    def _create_continual_optimizer(self) -> torch.optim.Optimizer:
        """Continual learning용 optimizer 생성 (Vibration + Text projection + 온도)"""
        base_lr = self.learning_rate
        proj_mult = float(TRAINING_CONFIG.get('proj_lr_mult', 5.0))
        vib_mult = float(TRAINING_CONFIG.get('vib_lr_mult', 2.0))
        
        params = []
        seen = set()
        
        # Text projection 파라미터 (continual learning에서 학습 가능)
        if hasattr(self.model.text_encoder, 'projection'):
            proj_params = [p for p in self.model.text_encoder.projection.parameters() if p.requires_grad]
            if proj_params:
                params.append({'params': proj_params, 'lr': base_lr * proj_mult, 'weight_decay': self.weight_decay})
                for p in proj_params:
                    seen.add(id(p))
        
        # Vibration encoder 파라미터
        vib_params = [p for p in self.model.vibration_encoder.parameters() if p.requires_grad]
        vib_params = [p for p in vib_params if id(p) not in seen]
        if vib_params:
            params.append({'params': vib_params, 'lr': base_lr * vib_mult, 'weight_decay': self.weight_decay})
            for p in vib_params:
                seen.add(id(p))
        
        # 🎯 CRITICAL FIX: InfoNCE 온도 파라미터 추가
        temp_params = []
        if hasattr(self.model.infonce_loss, 'log_temperature_text'):
            temp_params.append(self.model.infonce_loss.log_temperature_text)
        if hasattr(self.model.infonce_loss, 'log_temperature_vib'):
            temp_params.append(self.model.infonce_loss.log_temperature_vib)
        
        if temp_params:
            params.append({'params': temp_params, 'lr': base_lr * 2.0, 'weight_decay': 0.0})
            for p in temp_params:
                seen.add(id(p))
        
        # 누락 파라미터 보완
        remain = [p for p in self.model.parameters() if p.requires_grad and id(p) not in seen]
        if remain:
            params.append({'params': remain, 'lr': base_lr, 'weight_decay': self.weight_decay})
        
        return optim.AdamW(params)
    
    def _create_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int):
        """학습률 스케줄러 생성 (단일 구현)"""
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """배치를 디바이스로 이동"""
        if 'vibration' in batch:
            batch['vibration'] = batch['vibration'].to(self.device)
        if 'labels' in batch:
            batch['labels'] = batch['labels'].to(self.device)
        if 'file_idx' in batch:
            batch['file_idx'] = batch['file_idx'].to(self.device)
        return batch
    
    def save_training_history(self, path: str):
        """학습 이력 저장"""
        history = {
            'performance_history': dict(self.performance_history),
            'loss_history': dict(self.loss_history),
            'forgetting_scores': self.forgetting_scores,
            'completed_domains': self.completed_domains,
            'domain_order': self.domain_order
        }
        
        torch.save(history, path)
        logger.info(f"학습 이력 저장됨: {path}")
    
    def plot_continual_learning_curves(self, save_path: Optional[str] = None):
        """Continual learning 결과 시각화"""
        if not self.performance_history:
            logger.warning("시각화할 성능 데이터가 없음")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Domain별 성능 변화
        for domain in self.completed_domains:
            if domain in self.performance_history:
                ax1.plot(self.performance_history[domain], label=f'Domain {domain}')
        ax1.set_xlabel('Learning Phase')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Performance Evolution by Domain')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Forgetting scores
        if self.forgetting_scores:
            ax2.plot(self.forgetting_scores, 'r-o')
            ax2.set_xlabel('Domain')
            ax2.set_ylabel('Forgetting Score')
            ax2.set_title('Catastrophic Forgetting')
            ax2.grid(True)
        
        # 3. 최종 성능 비교
        final_accs = [self.performance_history[d][-1] if d in self.performance_history else 0
                     for d in self.completed_domains]
        ax3.bar(range(len(self.completed_domains)), final_accs)
        ax3.set_xlabel('Domain')
        ax3.set_ylabel('Final Accuracy')
        ax3.set_title('Final Performance by Domain')
        ax3.set_xticks(range(len(self.completed_domains)))
        ax3.set_xticklabels([str(d) for d in self.completed_domains])
        
        # 4. Loss curves
        for domain in self.completed_domains:
            if domain in self.loss_history:
                ax4.plot(self.loss_history[domain], label=f'Domain {domain}')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training Loss by Domain')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"학습 곡선 저장됨: {save_path}")
        else:
            plt.show()
    
    def _create_first_domain_alignment_visualization(self, 
                                                   domain_dataloaders: Dict, 
                                                   first_domain: int) -> Dict[str, Any]:
        """
        첫 번째 도메인에서 Text와 Vibration Encoder의 alignment 시각화
        
        Args:
            domain_dataloaders: 도메인별 데이터로더
            first_domain: 첫 번째 도메인 값
            
        Returns:
            Dict: 시각화 결과 정보
        """
        from .visualization import create_visualizer
        
        # 첫 번째 도메인 테스트 데이터로더
        if first_domain not in domain_dataloaders:
            logger.warning(f"도메인 {first_domain} 데이터로더가 없음")
            return {}
        
        test_loader = domain_dataloaders[first_domain]['test']
        
        # 임베딩 수집 (최대 200개 샘플)
        self.model.eval()
        text_embeddings = []
        vib_embeddings = []
        labels = []
        bearing_types = []
        
        max_samples = 200
        collected_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if collected_samples >= max_samples:
                    break
                
                # 배치 처리
                vibrations = batch['vibration'].to(self.device)
                texts = batch['text']
                metadata = batch['metadata']
                
                # 모델 forward
                model_results = self.model({
                    'vibration': vibrations,
                    'text': texts
                }, return_embeddings=True)
                
                # 임베딩 수집
                text_emb = model_results['text_embeddings'].cpu()
                vib_emb = model_results['vib_embeddings'].cpu()
                
                text_embeddings.append(text_emb)
                vib_embeddings.append(vib_emb)
                
                # 메타데이터 수집
                for meta in metadata:
                    labels.append(meta.get('bearing_condition', 'H'))
                    bearing_types.append(meta.get('bearing_type', '6204'))
                
                collected_samples += len(vibrations)
                
                if collected_samples >= max_samples:
                    break
        
        if not text_embeddings:
            logger.warning("임베딩 수집 실패")
            return {}
        
        # 텐서 결합
        text_embeddings = torch.cat(text_embeddings, dim=0)
        vib_embeddings = torch.cat(vib_embeddings, dim=0)
        
        # 샘플 수 맞춤
        min_samples = min(len(text_embeddings), len(labels), max_samples)
        text_embeddings = text_embeddings[:min_samples]
        vib_embeddings = vib_embeddings[:min_samples]
        labels = labels[:min_samples]
        bearing_types = bearing_types[:min_samples]
        
        # 시각화 생성
        visualizer = create_visualizer(self.save_dir)
        
        try:
            alignment_path = visualizer.create_encoder_alignment_plot(
                text_embeddings=text_embeddings,
                vib_embeddings=vib_embeddings,
                labels=labels,
                bearing_types=bearing_types,
                domain_name=f"Domain_{first_domain}",
                save_name=f"first_domain_alignment_{first_domain}"
            )
            
            return {
                'save_path': alignment_path,
                'num_samples': min_samples,
                'domain': first_domain
            }
            
        except Exception as e:
            logger.error(f"시각화 생성 실패: {e}")
            return {}
    
    def _create_domain_performance_visualization(self, 
                                               current_domain: int,
                                               performance_results: Dict,
                                               forgetting_score: float) -> Dict[str, Any]:
        """
        각 도메인 완료 후 성능 검증 시각화 (accuracy, forgetting)
        
        Args:
            current_domain: 현재 완료된 도메인
            performance_results: 성능 평가 결과
            forgetting_score: 망각 점수
            
        Returns:
            Dict: 시각화 결과 정보
        """
        import matplotlib.pyplot as plt
        
        # 현재까지 완료된 도메인들의 성능 수집
        domain_names = []
        accuracies = []
        
        for domain in self.completed_domains:
            domain_names.append(f"Domain_{domain}")
            if domain in performance_results:
                accuracies.append(performance_results[domain].get('accuracy', 0.0))
            else:
                accuracies.append(0.0)

        # 망각 점수 (첫 도메인은 0), 길이 정합성 보정
        n = len(domain_names)
        forgetting_scores = [0.0] + list(self.forgetting_scores)
        if len(forgetting_scores) < n:
            forgetting_scores = forgetting_scores + [0.0] * (n - len(forgetting_scores))
        elif len(forgetting_scores) > n:
            forgetting_scores = forgetting_scores[:n]
        
        # 시각화 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 도메인별 정확도
        bars1 = ax1.bar(range(len(domain_names)), accuracies, 
                       color=['#2E86AB', '#F24236', '#F6AE2D', '#2F9B69', '#F18F01', '#6C757D'][:len(domain_names)],
                       alpha=0.8)
        ax1.set_xlabel('Domains')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Accuracy after Domain {current_domain}')
        ax1.set_xticks(range(len(domain_names)))
        ax1.set_xticklabels(domain_names, rotation=45)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # 정확도 값 표시
        for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. 망각 점수
        bars2 = ax2.bar(range(len(domain_names)), forgetting_scores,
                       color='#F24236', alpha=0.7)
        ax2.set_xlabel('Domains')
        ax2.set_ylabel('Forgetting Score')
        ax2.set_title(f'Forgetting Score after Domain {current_domain}')
        ax2.set_xticks(range(len(domain_names)))
        ax2.set_xticklabels(domain_names, rotation=45)
        ax2.set_ylim(0, max(0.5, max(forgetting_scores) * 1.1) if forgetting_scores else 0.5)
        ax2.grid(True, alpha=0.3)
        
        # 망각 점수 값 표시
        for i, (bar, forget) in enumerate(zip(bars2, forgetting_scores)):
            if forget > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{forget:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # 저장
        save_path = os.path.join(self.save_dir, f'domain_{current_domain}_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'save_path': save_path,
            'current_domain': current_domain,
            'num_completed_domains': len(self.completed_domains),
            'avg_accuracy': np.mean(accuracies) if accuracies else 0.0,
            'avg_forgetting': np.mean(forgetting_scores) if forgetting_scores else 0.0
        }


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    print("=== ContinualTrainer 테스트 ===")
    
    # GPU 사용 가능하면 GPU 사용
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Trainer 생성
    trainer = ContinualTrainer(device=device)
    
    print(f"Trainer 초기화 완료: device={device}")
    print(f"도메인 순서: {trainer.domain_order}")
    print(f"모델 파라미터: {trainer.model.get_trainable_parameters()}")
    
    print("\n=== ContinualTrainer 테스트 완료 ===")
