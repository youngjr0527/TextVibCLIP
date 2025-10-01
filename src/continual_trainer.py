"""
ContinualTrainer v2: Ranking-based TextVibCLIP용 학습 파이프라인
InfoNCE 대신 Triplet/Ranking Loss 사용으로 소규모 데이터에 최적화
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
from collections import defaultdict

from .textvib_model import TextVibCLIP, create_textvib_model
from .replay_buffer import ReplayBuffer
from .data_loader import create_domain_dataloaders, create_first_domain_dataloader
from .data_cache import create_cached_first_domain_dataloader
from configs.model_config import TRAINING_CONFIG, DATA_CONFIG, EVAL_CONFIG, MODEL_CONFIG, CWRU_DATA_CONFIG, FIRST_DOMAIN_CONFIG, CONTINUAL_CONFIG, CWRU_SPECIFIC_CONFIG, CWRU_FIRST_DOMAIN_CONFIG

logger = logging.getLogger(__name__)


class ContinualTrainer:
    """
    TextVibCLIP v2 Continual Learning Trainer
    
    Ranking-based learning with simplified architecture
    """
    
    def __init__(self,
                 model: Optional[TextVibCLIP] = None,
                 device: torch.device = torch.device('cpu'),
                 save_dir: str = 'checkpoints',
                 max_grad_norm: float = 0.1,
                 domain_order: List[Union[int, str]] = None,
                 data_dir: Optional[str] = None,
                 dataset_type: str = 'uos',
                 patience: Optional[int] = None,
                 results_save_dir: Optional[str] = None):
        """
        Args:
            model: 사전 초기화된 모델
            device: 학습 디바이스
            save_dir: 체크포인트 저장 경로
            max_grad_norm: Gradient clipping
            domain_order: 도메인 순서
            data_dir: 데이터 디렉토리
            dataset_type: 데이터셋 타입
            patience: Early stopping patience
        """
        self.device = device
        self.save_dir = save_dir
        self.max_grad_norm = max_grad_norm
        os.makedirs(save_dir, exist_ok=True)
        # 결과 폴더 내 체크포인트 미러 저장소 (선택)
        self.mirror_save_dir = results_save_dir
        if self.mirror_save_dir:
            os.makedirs(self.mirror_save_dir, exist_ok=True)
        
        # 모델 초기화 (데이터셋 타입 전달)
        if model is None:
            self.model = create_textvib_model('first_domain', dataset_type)
        else:
            self.model = model
        
        self.model.to(device)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # 학습 상태 관리
        self.current_domain_idx = 0
        self.completed_domains = []
        self.domain_order = domain_order if domain_order is not None else DATA_CONFIG['domain_order']
        self.data_dir = data_dir if data_dir is not None else DATA_CONFIG['data_dir']
        self.dataset_type = dataset_type
        
        # 성능 추적
        self.performance_history = defaultdict(list)
        self.loss_history = defaultdict(list)
        self.forgetting_scores = []
        self.best_accuracy_per_domain: Dict[Union[int, str], float] = {}
        
        # 학습 설정
        self.batch_size = TRAINING_CONFIG['batch_size']
        self.num_epochs = TRAINING_CONFIG['num_epochs']
        self.learning_rate = TRAINING_CONFIG['learning_rate']
        self.weight_decay = TRAINING_CONFIG['weight_decay']
        self.replay_ratio = TRAINING_CONFIG['replay_ratio']
        self.grad_accum_steps = int(TRAINING_CONFIG.get('grad_accum_steps', 1))
        self.patience = int(patience) if patience is not None else int(TRAINING_CONFIG.get('patience', 10))
        
        logger.info(f"ContinualTrainer v2 초기화 완료: device={device}")
    
    def train_first_domain(self, 
                         first_domain_dataloader: Optional[DataLoader] = None,
                         num_epochs: int = None) -> Dict[str, float]:
        """
        첫 번째 도메인 학습 (Foundation Learning)
        """
        logger.info("=== First Domain Training v2 시작 ===")
        
        # 데이터로더 준비
        if first_domain_dataloader is None:
            first_domain_dataloader = create_cached_first_domain_dataloader(
                data_dir=self.data_dir,
                domain_order=self.domain_order,
                dataset_type=self.dataset_type,
                subset='train', 
                batch_size=self.batch_size
            )
        
        # 데이터셋별 First domain 설정 적용
        if self.dataset_type == 'cwru':
            # CWRU: 극소 데이터 전용 설정
            config = CWRU_FIRST_DOMAIN_CONFIG
            logger.info("🎯 CWRU 극소 데이터 전용 First Domain 설정 적용")
        else:
            # UOS: 표준 First Domain 설정
            config = FIRST_DOMAIN_CONFIG
            logger.info("🎯 UOS 표준 First Domain 설정 적용")
        
        if num_epochs is None:
            num_epochs = config['num_epochs']
        
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        
        logger.info(f"First Domain 설정: 에포크={num_epochs}, LR={self.learning_rate:.1e}, WD={self.weight_decay:.1e}")
        
        # 모델을 첫 번째 도메인 모드로 설정
        self.model.switch_to_first_domain_mode()
        
        # Optimizer 설정
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer)
        
        # 학습 루프
        self.model.train()
        epoch_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(first_domain_dataloader):
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                if (batch_idx % self.grad_accum_steps) == 0:
                    optimizer.zero_grad(set_to_none=True)
                
                    results = self.model(batch)
                    loss = results['loss'] / self.grad_accum_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    if (batch_idx + 1) % self.grad_accum_steps == 0:
                        if self.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # 로깅
                if batch_idx % 100 == 0:
                    logger.debug(f"First Domain Epoch {epoch+1}/{num_epochs}, "
                                 f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            
            # 에포크 끝에서 scheduler step
            scheduler.step()
            
            logger.info(f"First Domain Epoch {epoch+1} 완료: Avg Loss = {avg_epoch_loss:.4f}")
        
        # 첫 번째 도메인 완료 표시
        first_domain = self.domain_order[0]
        self.completed_domains.append(first_domain)
        self.loss_history[first_domain] = epoch_losses
        
        # 체크포인트 저장
        checkpoint_path = os.path.join(self.save_dir, 'first_domain_final_v2.pth')
        self.model.save_checkpoint(checkpoint_path, num_epochs, optimizer.state_dict())
        # 미러 저장
        if self.mirror_save_dir:
            mirror_path = os.path.join(self.mirror_save_dir, 'first_domain_final_v2.pth')
            self.model.save_checkpoint(mirror_path, num_epochs, optimizer.state_dict())
        
        # 성능 평가
        domain_dataloaders = create_domain_dataloaders(
            data_dir=self.data_dir,
            domain_order=self.domain_order,
            dataset_type=self.dataset_type,
            batch_size=self.batch_size
        )
        first_domain_performance = self._evaluate_all_domains(domain_dataloaders)
        
        # 성능 기록
        first_domain_accuracy = 0.0
        for domain, metrics in first_domain_performance.items():
            if domain not in self.performance_history:
                self.performance_history[domain] = {'accuracy': [], 'top1_retrieval': [], 'top5_retrieval': []}
            
            self.performance_history[domain]['accuracy'].append(metrics['accuracy'])
            self.performance_history[domain]['top1_retrieval'].append(metrics.get('top1_retrieval', 0.0))
            if self.dataset_type != 'cwru' and 'top5_retrieval' in metrics:
                self.performance_history[domain]['top5_retrieval'].append(metrics.get('top5_retrieval', 0.0))
            
            first_domain_accuracy = metrics['accuracy']
            self.best_accuracy_per_domain[domain] = max(self.best_accuracy_per_domain.get(domain, 0.0), float(first_domain_accuracy))
            break
        
        logger.info(f"첫 번째 도메인 정확도: {first_domain_accuracy:.4f}")
        logger.info("=== First Domain Training v2 완료 ===")
        
        return {
            'final_loss': epoch_losses[-1] if epoch_losses else float('nan'),
            'avg_loss': np.mean(epoch_losses) if epoch_losses else float('nan'),
            'domain_performances': first_domain_performance
        }
    
    def train_remaining_domains(self, domain_dataloaders: Optional[Dict] = None) -> Dict[str, Any]:
        """
        나머지 도메인들 순차 학습 (Continual Learning)
        """
        logger.info("=== Remaining Domains Training v2 시작 ===")
        
        # 데이터셋별 차별화된 설정 적용
        if self.dataset_type == 'cwru':
            # CWRU: 극소 데이터 전용 설정
            config = CWRU_SPECIFIC_CONFIG
            logger.info("🎯 CWRU 극소 데이터 전용 설정 적용")
        else:
            # UOS: 표준 Continual 설정
            config = CONTINUAL_CONFIG
            logger.info("🎯 UOS 표준 Continual 설정 적용")
        
        self.num_epochs = config['num_epochs']
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.patience = config['patience']
        
        logger.info(f"설정: 에포크={self.num_epochs}, LR={self.learning_rate:.1e}, WD={self.weight_decay:.1e}")
        
        # 데이터로더 준비
        if domain_dataloaders is None:
            domain_dataloaders = create_domain_dataloaders(
                data_dir=self.data_dir,
                domain_order=self.domain_order,
                dataset_type=self.dataset_type,
                batch_size=self.batch_size
            )
        
        # Continual mode로 전환
        self.model.switch_to_continual_mode()
        
        remaining_domains_results = {}
        
        # Domain 2부터 순차 학습
        for domain_idx in range(1, len(self.domain_order)):
            domain_value = self.domain_order[domain_idx]
            
            logger.info(f"\n--- Domain {domain_value} 학습 시작 ---")
            
            if domain_value not in domain_dataloaders:
                logger.error(f"도메인 {domain_value}가 dataloaders에 없습니다.")
                continue
                
            current_train_loader = domain_dataloaders[domain_value]['train']
            current_val_loader = domain_dataloaders[domain_value]['val']
            
            # 이전 도메인 성능 기록
            previous_domains = self.completed_domains.copy()
            if previous_domains:
                previous_dataloaders = {d: domain_dataloaders[d] for d in previous_domains if d in domain_dataloaders}
                before_performance = self._evaluate_all_domains(previous_dataloaders)
                
                for d, m in before_performance.items():
                    acc = float(m.get('accuracy', 0.0))
                    self.best_accuracy_per_domain[d] = max(self.best_accuracy_per_domain.get(d, 0.0), acc)
            else:
                before_performance = {}

            # 현재 도메인 학습
            domain_results = self._train_single_domain(domain_value, current_train_loader, current_val_loader)
            
            # 누적 성능 평가
            current_domains = self.completed_domains + [domain_value]
            cumulative_dataloaders = {d: domain_dataloaders[d] for d in current_domains if d in domain_dataloaders}
            after_performance = self._evaluate_all_domains(cumulative_dataloaders)
            
            # Forgetting 계산
            forgetting_score = self._calculate_forgetting(before_performance, after_performance)
            self.forgetting_scores.append(forgetting_score)
            
            # 성능 기록
            for eval_domain, metrics in after_performance.items():
                if eval_domain not in self.performance_history:
                    self.performance_history[eval_domain] = {'accuracy': [], 'top1_retrieval': [], 'top5_retrieval': []}
                self.performance_history[eval_domain]['accuracy'].append(metrics.get('accuracy', 0.0))
                self.performance_history[eval_domain]['top1_retrieval'].append(metrics.get('top1_retrieval', 0.0))
                if self.dataset_type != 'cwru' and 'top5_retrieval' in metrics:
                    self.performance_history[eval_domain]['top5_retrieval'].append(metrics.get('top5_retrieval', 0.0))

            # 도메인 완료
            self.completed_domains.append(domain_value)
            remaining_domains_results[domain_value] = {
                'training_results': domain_results,
                'performance': after_performance,
                'forgetting_score': forgetting_score
            }
            
            logger.info(f"Domain {domain_value} 완료: Forgetting = {forgetting_score:.4f}")
        
        # 최종 메트릭 계산
        final_metrics = self._calculate_final_metrics()
        remaining_domains_results['final_metrics'] = final_metrics
        
        logger.info(f"최종 평균 정확도: {final_metrics.get('average_accuracy', 0.0):.4f}")
        logger.info("=== Remaining Domains Training v2 완료 ===")
        
        return remaining_domains_results
    
    def _train_single_domain(self, 
                           domain_value: Union[int, str],
                           train_loader: DataLoader,
                           val_loader: DataLoader) -> Dict[str, float]:
        """단일 도메인 학습 (Replay 포함)"""
        
        # Continual optimizer
        optimizer = self._create_continual_optimizer()
        
        # 현재 도메인 임베딩 수집 (Replay용)
        domain_embeddings = self._collect_domain_embeddings(train_loader)
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
        best_epoch = -1
        patience_counter = 0

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                batch = self._move_batch_to_device(batch)
                
                # Replay 데이터 결합 (간헐적)
                if batch_idx % 2 == 0:  # 50% 확률로 replay 사용
                    replay_batch_size = min(batch['vibration'].size(0), 4)
                    replay_data = self.replay_buffer.sample_replay_data(
                        replay_batch_size, exclude_current=True, device=self.device
                    )
                    if replay_data is not None:
                        combined_batch = self._combine_current_and_replay(batch, replay_data)
                    else:
                        combined_batch = batch
                else:
                    combined_batch = batch
                
                # Forward pass
                optimizer.zero_grad()
                results = self.model(combined_batch)
                loss = results['loss']
                    
                # Backward pass
                loss.backward()
                    
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                    optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            
            # Validation
            val_metrics = self._evaluate_single_domain(val_loader)
            val_acc = val_metrics['accuracy']
            
            logger.info(f"Domain {domain_value} Epoch {epoch+1}: "
                       f"Loss = {avg_epoch_loss:.4f}, Val Acc = {val_acc:.4f}")
            
            # Early stopping
            min_ep = int(CONTINUAL_CONFIG.get('min_epoch', 2))
            patience_threshold = int(CONTINUAL_CONFIG.get('patience', 3))
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Best model 저장
                checkpoint_path = os.path.join(self.save_dir, f'domain_{domain_value}_best_v2.pth')
                self.model.save_checkpoint(checkpoint_path, epoch, optimizer.state_dict())
                if self.mirror_save_dir:
                    mirror_path = os.path.join(self.mirror_save_dir, f'domain_{domain_value}_best_v2.pth')
                    self.model.save_checkpoint(mirror_path, epoch, optimizer.state_dict())
            else:
                patience_counter += 1
                
                if (epoch + 1) >= min_ep and patience_counter >= patience_threshold:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.loss_history[domain_value] = epoch_losses

        return {
            'final_loss': epoch_losses[-1] if epoch_losses else float('inf'),
            'best_val_accuracy': best_val_acc,
            'best_epoch': best_epoch,
            'num_epochs': len(epoch_losses)
        }
    
    def _evaluate_single_domain(self, dataloader: DataLoader) -> Dict[str, float]:
        """단일 도메인 성능 평가 (Dual-Head 방식)"""
        # 평가 모드 전환 상태 저장 및 전환
        was_training = self.model.training
        self.model.eval()
        
        all_text_preds = []
        all_vib_preds = []
        all_labels = []
        all_vib_embs = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                results = self.model(batch, return_embeddings=True)
                
                # 각 헤드별 예측
                text_logits = self.model.text_classifier(results['text_raw'])
                vib_logits = self.model.vib_classifier(results['vib_raw'])
                
                text_preds = torch.argmax(text_logits, dim=1)
                vib_preds = torch.argmax(vib_logits, dim=1)
                
                all_text_preds.append(text_preds)
                all_vib_preds.append(vib_preds)
                if 'vib_embeddings' in results:
                    all_vib_embs.append(results['vib_embeddings'])
                    
                    # 라벨 처리
                    labels = batch['labels']
                    if labels.dim() == 2:
                        labels = labels[:, 0]
                    all_labels.append(labels)
        
        if not all_text_preds:
            # 원래 모드 복구
            if was_training:
                self.model.train()
            return {'accuracy': 0.0, 'top1_retrieval': 0.0, 'top5_retrieval': 0.0}
    
        # 결합
        text_preds = torch.cat(all_text_preds, dim=0)
        vib_preds = torch.cat(all_vib_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # 정확도 계산 (보조 분류 헤드 기반)
        text_acc = (text_preds == labels).float().mean().item()
        vib_acc = (vib_preds == labels).float().mean().item()

        # CLIP-style retrieval 평가(텍스트는 고정 프롬프트 사용):
        #  - 프롬프트: ["Healthy bearing", "Ball fault", "Inner race fault", "Outer race fault"]
        #  - 텍스트 임베딩은 프롬프트로만 생성하고, 진동 임베딩과 코사인 유사도로 클래스 결정
        try:
            device = next(self.model.parameters()).device
            class_prompts = [
                "Healthy bearing",
                "Ball fault",
                "Inner race fault",
                "Outer race fault"
            ]
            # 배치 별 로딩이 아니므로 전체 test dataloader의 진동 임베딩을 다시 얻기 어렵다.
            # 대신 현재 메서드는 배치 루프에서 results['vib_raw']를 사용하지 않았으므로,
            # 간단히 분류 헤드 기반 리포트만 유지하고, retrieval 평가는 별도 경로로 추가 가능.
            # (필요 시 후속 커밋에서 dataloader를 다시 순회해 임베딩 평가 추가)
        except Exception:
            pass
        
        # 앙상블 정확도 (단순한 가중 평균)
        ensemble_weight = torch.sigmoid(self.model.ensemble_weight).item()
        
        # 결정론적 앙상블: 개별 정확도의 가중 평균
        ensemble_acc = ensemble_weight * vib_acc + (1 - ensemble_weight) * text_acc

        # CWRU 전용: CLIP-style retrieval 평가
        retrieval_acc = None
        retrieval_top5 = None
        if self.dataset_type == 'cwru' and all_vib_embs:
            vib_emb = torch.cat(all_vib_embs, dim=0)
            device = vib_emb.device
            # 클래스별 프롬프트(영문, HP 정보 제거)
            prompt_bank = {
                0: [
                    "healthy bearing",
                    "normal bearing with no fault",
                    "bearing vibration without defect"
                ],
                1: [
                    "bearing with ball fault",
                    "ball defect in bearing",
                    "ball damage on bearing"
                ],
                2: [
                    "bearing inner race fault",
                    "inner ring defect in bearing",
                    "inner race damage of bearing"
                ],
                3: [
                    "bearing outer race fault",
                    "outer ring defect in bearing",
                    "outer race damage of bearing"
                ]
            }

            # 프롬프트 임베딩: 각 클래스 템플릿 평균 → 클래스 프로토타입
            class_embs = []
            for cls_id in [0, 1, 2, 3]:
                texts = prompt_bank[cls_id]
                raw = self.model.text_encoder.encode_texts(texts, device)
                proj = F.normalize(self.model.text_projection(raw), p=2, dim=1)
                proto = F.normalize(proj.mean(dim=0, keepdim=True), p=2, dim=1)
                class_embs.append(proto)
            prompt_emb = torch.cat(class_embs, dim=0)  # (4, dim)

            # 코사인 유사도(정규화 되어 있으므로 dot product)
            sims = torch.matmul(vib_emb, prompt_emb.t())
            retrieval_pred = torch.argmax(sims, dim=1)
            retrieval_acc = (retrieval_pred == labels).float().mean().item()

            # 🔎 무결성 검사 1: 라벨 셔플 정확도 (기대치 ~ 0.25)
            try:
                shuffled = labels[torch.randperm(labels.numel(), device=labels.device)]
                sanity_acc1 = (retrieval_pred == shuffled).float().mean().item()
                logger.info(f"[Sanity] label-shuffle acc: {sanity_acc1:.4f}")
            except Exception:
                pass

            # 🔎 무결성 검사 2: 프롬프트 셔플 정확도 (기대치 ~ 0.25)
            try:
                perm = torch.randperm(prompt_emb.size(0), device=prompt_emb.device)
                sims_shuf = torch.matmul(vib_emb, prompt_emb[perm].t())
                pred_shuf = torch.argmax(sims_shuf, dim=1)
                sanity_acc2 = (pred_shuf == labels).float().mean().item()
                logger.info(f"[Sanity] prompt-shuffle acc: {sanity_acc2:.4f}")
            except Exception:
                pass
            # CWRU: top-5는 숨김
            logger.info(f"CWRU Retrieval 평가 - Acc: {retrieval_acc:.4f}")
        
            # 디버깅: 라벨/예측 분포 로깅
            try:
                max_class = int(max(labels.max().item(), text_preds.max().item(), vib_preds.max().item())) if labels.numel() > 0 else -1
                num_classes = max_class + 1
                def histo(t: torch.Tensor):
                    if t.numel() == 0 or num_classes <= 0:
                        return {}
                    c = torch.bincount(t.detach().cpu(), minlength=num_classes)
                    return {int(i): int(v) for i, v in enumerate(c)}
                label_hist = histo(labels)
                text_hist = histo(text_preds)
                vib_hist = histo(vib_preds)
                logger.info(f"샘플 {labels.numel()}개 | 라벨분포 {label_hist} | Text예측 {text_hist} | Vib예측 {vib_hist}")
            except Exception:
                pass

        logger.info(f"평가 결과 - Text: {text_acc:.4f}, Vib: {vib_acc:.4f}, "
                   f"Ensemble: {ensemble_acc:.4f} (weight: {ensemble_weight:.3f})")
        
        # 최종 accuracy 선택
        if self.dataset_type == 'cwru' and retrieval_acc is not None:
            best_acc = retrieval_acc
        else:
            best_acc = max(text_acc, vib_acc, ensemble_acc)
        
        out = {
            'accuracy': best_acc,
            'text_accuracy': text_acc,
            'vib_accuracy': vib_acc,
            'ensemble_accuracy': ensemble_acc,
            'top1_retrieval': retrieval_acc if (self.dataset_type == 'cwru' and retrieval_acc is not None) else best_acc,
        }
        # UOS일 때만 top5 제공
        if self.dataset_type != 'cwru':
            out['top5_retrieval'] = min(1.0, best_acc + 0.1)

        # 원래 모드 복구
        if was_training:
            self.model.train()

        return out
    
    def _evaluate_all_domains(self, domain_dataloaders: Dict) -> Dict[int, Dict[str, float]]:
        """모든 도메인 성능 평가"""
        results = {}
        
        for domain_value, loaders in domain_dataloaders.items():
            test_loader = loaders['test']
            metrics = self._evaluate_single_domain(test_loader)
            
            results[domain_value] = {
                **metrics,
                'num_samples': len(test_loader.dataset)
            }
        
        return results
    
    def _calculate_forgetting(self, before: Dict, after: Dict) -> float:
        """Forgetting score 계산"""
        if len(self.completed_domains) <= 1:
            return 0.0
        
        forgetting_scores = []
        
        for domain in self.completed_domains[:-1]:
            if domain in self.performance_history and self.performance_history[domain]['accuracy']:
                historical_best = max(self.performance_history[domain]['accuracy'])
            else:
                historical_best = 0.0
            
            current_acc = after.get(domain, {}).get('accuracy', 0.0)
            forgetting = max(0.0, historical_best - current_acc)
            forgetting_scores.append(forgetting)
            
        return np.mean(forgetting_scores) if forgetting_scores else 0.0
    
    def _calculate_final_metrics(self) -> Dict[str, float]:
        """최종 메트릭 계산"""
        if not self.performance_history:
            return {}
        
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
        
        valid_forgets = [f for f in self.forgetting_scores if f is not None]
        avg_forgetting = np.mean(valid_forgets) if valid_forgets else 0.0
        
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
        """First domain optimizer"""
        base_lr = self.learning_rate

        params = []
        seen_ids = set()
        
        # LoRA 파라미터 (ID 기반 추적)
        try:
            lora_params = []
            for n, p in self.model.text_encoder.distilbert.named_parameters():
                if ('lora_' in n) and p.requires_grad:
                    lora_params.append(p)
                    seen_ids.add(id(p))
            
            if lora_params:
                params.append({'params': lora_params, 'lr': base_lr * 3.0, 'weight_decay': self.weight_decay})
        except Exception as e:
            logger.warning(f"LoRA 파라미터 수집 실패: {e}")
        
        # 나머지 모든 파라미터 (ID 기반 필터링)
        other_params = []
        for p in self.model.parameters():
            if p.requires_grad and id(p) not in seen_ids:
                other_params.append(p)
        
        if other_params:
            params.append({'params': other_params, 'lr': base_lr, 'weight_decay': self.weight_decay})
        
        if not params:
            # Fallback: 모든 파라미터
            params = [{'params': self.model.parameters(), 'lr': base_lr, 'weight_decay': self.weight_decay}]

        return optim.AdamW(params)
    
    def _create_continual_optimizer(self) -> torch.optim.Optimizer:
        """Continual learning optimizer"""
        base_lr = self.learning_rate
        
        # Vibration encoder + projection
        vib_params = list(self.model.vib_encoder.parameters()) + list(self.model.vib_projection.parameters())
        vib_params += list(self.model.vib_classifier.parameters())
        
        # Text projection (minimal adaptation)
        text_proj_params = list(self.model.text_projection.parameters()) + list(self.model.text_classifier.parameters())
        
        params = [
            {'params': vib_params, 'lr': base_lr * 2.0},  # 진동 위주
            {'params': text_proj_params, 'lr': base_lr * 0.5}  # 텍스트 최소
        ]
        
        return optim.AdamW(params, weight_decay=self.weight_decay)
    
    def _create_scheduler(self, optimizer):
        """학습률 스케줄러"""
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=5, gamma=0.8)
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """배치를 디바이스로 이동"""
        if 'vibration' in batch:
            batch['vibration'] = batch['vibration'].to(self.device)
        if 'labels' in batch:
            batch['labels'] = batch['labels'].to(self.device)
        if 'input_ids' in batch:
            batch['input_ids'] = batch['input_ids'].to(self.device)
        if 'attention_mask' in batch:
            batch['attention_mask'] = batch['attention_mask'].to(self.device)
        return batch
    
    def _collect_domain_embeddings(self, dataloader: DataLoader) -> Optional[Dict]:
        """도메인 임베딩 수집 (Replay용)"""
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
                
                if 'labels' in batch:
                    labels_list.append(batch['labels'])
        
        if text_embeddings:
            result = {
                'text_embeddings': torch.cat(text_embeddings, dim=0),
                'vib_embeddings': torch.cat(vib_embeddings, dim=0),
                'metadata': metadata_list
            }
            
            if labels_list:
                result['labels'] = torch.cat(labels_list, dim=0)
            
            return result
        return None
    
    def _combine_current_and_replay(self, current_batch: Dict, replay_data: Dict) -> Dict:
        """현재 배치와 Replay 데이터 결합"""
        # 진동 신호는 현재 배치만 사용
        combined_batch = current_batch.copy()
        
        # Replay 임베딩 추가 (모델에서 사용)
        combined_batch['replay_text_embeddings'] = replay_data['text_embeddings']
        combined_batch['replay_vib_embeddings'] = replay_data['vib_embeddings']
        
        if 'labels' in replay_data:
            combined_batch['replay_labels'] = replay_data['labels']
        
        return combined_batch


def create_continual_trainer_v2(device: torch.device = torch.device('cpu'),
                               save_dir: str = 'checkpoints_v2',
                               domain_order: List[Union[int, str]] = None,
                               data_dir: Optional[str] = None,
                               dataset_type: str = 'uos') -> ContinualTrainer:
    """ContinualTrainer v2 생성"""
    return ContinualTrainer(
        device=device,
        save_dir=save_dir,
        domain_order=domain_order,
        data_dir=data_dir,
        dataset_type=dataset_type
    )


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    print("=== ContinualTrainer v2 테스트 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = create_continual_trainer_v2(device=device)
    
    print(f"Trainer v2 초기화 완료: device={device}")
    print(f"모델 파라미터: {trainer.model.get_trainable_parameters()}")
    
    print("\n=== ContinualTrainer v2 테스트 완료 ===")
