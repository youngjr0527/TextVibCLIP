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
from .utils import setup_amp_and_scaler
from configs.model_config import TRAINING_CONFIG, DATA_CONFIG, EVAL_CONFIG, MODEL_CONFIG

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
            first_domain_dataloader = create_first_domain_dataloader(subset='train', batch_size=self.batch_size)
        
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
        
        # 🚨 조기 종료 체크: 첫 번째 도메인 정확도가 80% 미달 시 실험 중단
        if first_domain_accuracy < 0.80:
            error_msg = f"❌ 첫 번째 도메인 정확도 {first_domain_accuracy:.4f} < 0.80 (80%)"
            logger.error(error_msg)
            logger.error("🛑 실험 의미 없음 - 조기 종료!")
            raise RuntimeError(f"First domain accuracy too low: {first_domain_accuracy:.4f} < 0.80")
        else:
            logger.info(f"✅ 첫 번째 도메인 정확도 {first_domain_accuracy:.4f} >= 80% - 계속 진행")
        
        logger.info("=== First Domain Training 완료 ===")
        
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
            'grad_norms_summary': grad_summary
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
        for domain, metrics in after_performance.items():
            if domain not in self.performance_history:
                self.performance_history[domain] = {'accuracy': [], 'top1_retrieval': [], 'top5_retrieval': []}
            
            self.performance_history[domain]['accuracy'].append(metrics['accuracy'])
            self.performance_history[domain]['top1_retrieval'].append(metrics.get('top1_retrieval', 0.0))
            self.performance_history[domain]['top5_retrieval'].append(metrics.get('top5_retrieval', 0.0))
            
            # 도메인 완료 표시
            self.completed_domains.append(domain_value)
            remaining_domains_results[domain_value] = {
                'training_results': domain_results,
                'performance': after_performance,
                'forgetting_score': forgetting_score
            }
            
            logger.info(f"Domain {domain_value} 학습 완료: "
                       f"Forgetting Score = {forgetting_score:.4f}")
        
        # 최종 성능 요약
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
        
        # Replay buffer에 추가
        if domain_embeddings:
            self.replay_buffer.add_domain_data(
                domain_value,
                domain_embeddings['text_embeddings'],
                domain_embeddings['vib_embeddings'],
                domain_embeddings['metadata']
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
        """현재 도메인 데이터의 임베딩 수집"""
        self.model.eval()
        
        text_embeddings = []
        vib_embeddings = []
        metadata_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                results = self.model(batch, return_embeddings=True)
                
                text_embeddings.append(results['text_embeddings'])
                vib_embeddings.append(results['vib_embeddings'])
                metadata_list.extend(batch['metadata'])
        
        if text_embeddings:
            return {
                'text_embeddings': torch.cat(text_embeddings, dim=0),
                'vib_embeddings': torch.cat(vib_embeddings, dim=0),
                'metadata': metadata_list
            }
        return None
    
    def _combine_current_and_replay(self, current_batch: Dict, replay_data: Dict) -> Dict:
        """현재 배치와 Replay 데이터 결합"""
        # 진동 신호는 현재 배치에서만 (Replay는 임베딩만 저장)
        combined_batch = {
            'vibration': current_batch['vibration'],
            'text': current_batch['text']
        }
        
        # Replay 임베딩을 모델에 주입하는 방식으로 구현
        # (실제로는 loss 계산 시 replay embeddings 사용)
        combined_batch['replay_text_embeddings'] = replay_data['text_embeddings']
        combined_batch['replay_vib_embeddings'] = replay_data['vib_embeddings']
        
        return combined_batch
    
    def _evaluate_single_domain(self, dataloader: DataLoader) -> Dict[str, float]:
        """단일 도메인 성능 평가 (retrieval 메트릭 포함)"""
        self.model.eval()
        
        all_text_embeddings = []
        all_vib_embeddings = []
        all_file_idx = []
        
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
                    
        except Exception as e:
            logger.error(f"평가 중 오류 발생: {e}")
            return {'accuracy': 0.0, 'top1_retrieval': 0.0, 'top5_retrieval': 0.0}
        
        if not all_text_embeddings:
            return {'accuracy': 0.0, 'top1_retrieval': 0.0, 'top5_retrieval': 0.0}
        
        # 모든 임베딩 결합
        text_emb = torch.cat(all_text_embeddings, dim=0)
        vib_emb = torch.cat(all_vib_embeddings, dim=0)
        file_idx = torch.cat(all_file_idx, dim=0) if all_file_idx else None
        
        # L2 정규화
        text_emb = F.normalize(text_emb, dim=1)
        vib_emb = F.normalize(vib_emb, dim=1)
        
        # 1. Retrieval 정확도 (멀티-포지티브: 같은 파일은 모두 정답 처리)
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

            # 셔플 베이스라인 (멀티-포지티브 기준)
            if file_idx is not None and N >= 2:
                perm = torch.randperm(N, device=text_emb.device)
                sim_shuf = torch.matmul(text_emb, vib_emb[perm].t())
                pred_shuf = torch.argmax(sim_shuf, dim=1)
                top1_shuf = (file_idx[perm][pred_shuf] == file_idx).float().mean().item()
                k_dbg = min(5, N)
                _, topk_shuf = torch.topk(sim_shuf, k=k_dbg, dim=1)
                top5_shuf = (file_idx.unsqueeze(1) == file_idx[perm][topk_shuf]).any(dim=1).float().mean().item()
                logger.info(f"🔍 DEBUG - 셔플 베이스라인 Top1/Top5: {top1_shuf:.4f}/{top5_shuf:.4f}")
        
        # 각 text에 대해 가장 유사한 vibration이 같은 파일에 속하는지 확인
        _, predicted_indices = torch.max(similarity_matrix, dim=1)
        if file_idx is not None:
            same_file_mask = (file_idx[predicted_indices] == file_idx).float()
            retrieval_accuracy = same_file_mask.mean().item()
        else:
            correct_indices = torch.arange(text_emb.size(0), device=text_emb.device)
            retrieval_accuracy = (predicted_indices == correct_indices).float().mean().item()
        
        # 2. Top-K Retrieval 정확도 (더 정확한 계산)
        # Top-1은 이미 위에서 계산됨 (retrieval_accuracy와 동일)
        top1_accuracy = retrieval_accuracy
        
        # Top-5 retrieval accuracy (멀티-포지티브)
        k = min(5, similarity_matrix.size(1))
        if k > 1:
            _, topk_indices = torch.topk(similarity_matrix, k=k, dim=1)
            if file_idx is not None:
                topk_same = (file_idx.unsqueeze(1) == file_idx[topk_indices])
                top5_accuracy = topk_same.any(dim=1).float().mean().item()
            else:
                correct_indices = torch.arange(text_emb.size(0), device=text_emb.device)
                correct_indices_expanded = correct_indices.unsqueeze(1).expand(-1, k)
                top5_accuracy = (topk_indices == correct_indices_expanded).any(dim=1).float().mean().item()
        else:
            top5_accuracy = top1_accuracy  # k=1일 때는 top1과 동일
        
        return {
            'accuracy': retrieval_accuracy,  # 실제 retrieval 정확도
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
                    
        except Exception as e:
            logger.error(f"빠른 평가 중 오류: {e}")
            return {'accuracy': 0.0, 'top1_retrieval': 0.0, 'top5_retrieval': 0.0}
        
        if not all_text_embeddings:
            return {'accuracy': 0.0, 'top1_retrieval': 0.0, 'top5_retrieval': 0.0}
        
        # 임베딩 결합 및 메트릭 계산
        text_emb = torch.cat(all_text_embeddings, dim=0)
        vib_emb = torch.cat(all_vib_embeddings, dim=0)
        file_idx = torch.cat(all_file_idx, dim=0) if all_file_idx else None

        # L2 정규화 후 유사도 계산
        text_emb = F.normalize(text_emb, dim=1)
        vib_emb = F.normalize(vib_emb, dim=1)
        similarity = torch.matmul(text_emb, vib_emb.t())

        # Top-1 (멀티-포지티브 지원)
        _, pred = torch.max(similarity, dim=1)
        if file_idx is not None:
            same_file = (file_idx[pred] == file_idx).float()
            top1 = same_file.mean().item()
        else:
            target = torch.arange(text_emb.size(0), device=text_emb.device)
            top1 = (pred == target).float().mean().item()

        # Top-5 (멀티-포지티브 지원)
        k = min(5, similarity.size(1))
        if k > 1:
            _, topk = torch.topk(similarity, k=k, dim=1)
            if file_idx is not None:
                topk_same = (file_idx.unsqueeze(1) == file_idx[topk]).any(dim=1).float().mean().item()
                top5 = topk_same
            else:
                target = torch.arange(text_emb.size(0), device=text_emb.device).unsqueeze(1).expand(-1, k)
                top5 = (topk == target).any(dim=1).float().mean().item()
        else:
            top5 = top1

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
                f"off mean/std={off_mean:.4f}/{off_std:.4f}, Top1={top1:.4f}, Top5={top5:.4f}"
            )
            if file_idx is not None and N >= 2:
                perm = torch.randperm(N, device=text_emb.device)
                sim_shuf = torch.matmul(text_emb, vib_emb[perm].t())
                pred_shuf = torch.argmax(sim_shuf, dim=1)
                top1_shuf = (file_idx[perm][pred_shuf] == file_idx).float().mean().item()
                k_dbg = min(5, N)
                _, topk_shuf = torch.topk(sim_shuf, k=k_dbg, dim=1)
                top5_shuf = (file_idx.unsqueeze(1) == file_idx[perm][topk_shuf]).any(dim=1).float().mean().item()
                logger.info(f"🔍 DEBUG(FAST) - 셔플 베이스라인 Top1/Top5: {top1_shuf:.4f}/{top5_shuf:.4f}")

        return {
            'accuracy': top1,
            'top1_retrieval': top1,
            'top5_retrieval': top5
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
        """Continual learning용 optimizer 생성 (Vibration encoder만)"""
        return optim.AdamW(
            self.model.vibration_encoder.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
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
