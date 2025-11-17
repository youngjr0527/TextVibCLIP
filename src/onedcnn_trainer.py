"""
OneDCNNTrainer: 1D-CNN 분류 모델용 Continual Learning Trainer
TextVibCLIP 비교군 실험용
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

from .onedcnn_model import OneDCNNClassifier, create_onedcnn_model
from .replay_buffer import ReplayBuffer
from .data_loader import create_domain_dataloaders, create_first_domain_dataloader
from .data_cache import create_cached_first_domain_dataloader, create_cached_domain_dataloaders
from configs.model_config import TRAINING_CONFIG, DATA_CONFIG, EVAL_CONFIG, FIRST_DOMAIN_CONFIG, CONTINUAL_CONFIG

logger = logging.getLogger(__name__)


class OneDCNNTrainer:
    """
    1D-CNN 분류 모델용 Continual Learning Trainer
    
    TextVibCLIP 비교군 실험용 단일 모달 학습 파이프라인
    """
    
    def __init__(self,
                 model: Optional[OneDCNNClassifier] = None,
                 device: torch.device = torch.device('cpu'),
                 save_dir: str = 'checkpoints',
                 max_grad_norm: float = 1.0,
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
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        
        self.mirror_save_dir = results_save_dir
        if self.mirror_save_dir:
            os.makedirs(self.mirror_save_dir, exist_ok=True)
        
        # 모델 초기화
        if model is None:
            self.model = create_onedcnn_model(dataset_type)
        else:
            self.model = model
        
        self.model.to(device)
        
        # Replay buffer (Vanilla 1D-CNN은 기본적으로 사용 안 함, buffer_size=0)
        self.replay_buffer = ReplayBuffer(buffer_size_per_domain=0)
        
        # 학습 상태 관리
        self.current_domain_idx = 0
        self.completed_domains = []
        self.domain_order = domain_order if domain_order is not None else DATA_CONFIG['domain_order']
        self.data_dir = data_dir if data_dir is not None else DATA_CONFIG['data_dir']
        self.dataset_type = dataset_type
        
        # 성능 추적
        self.performance_history = defaultdict(lambda: {'accuracy': []})
        self.loss_history = defaultdict(list)
        self.forgetting_scores = []
        self.best_accuracy_per_domain: Dict[Union[int, str], float] = {}
        
        # 학습 설정
        self.batch_size = TRAINING_CONFIG['batch_size']
        self.num_epochs = TRAINING_CONFIG['num_epochs']
        self.learning_rate = TRAINING_CONFIG['learning_rate']
        self.weight_decay = TRAINING_CONFIG['weight_decay']
        self.replay_ratio = TRAINING_CONFIG.get('replay_ratio', 0.3)
        self.grad_accum_steps = int(TRAINING_CONFIG.get('grad_accum_steps', 1))
        self.patience = int(patience) if patience is not None else int(TRAINING_CONFIG.get('patience', 10))
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"OneDCNNTrainer 초기화 완료: device={device}")
    
    def train_first_domain(self, 
                         first_domain_dataloader: Optional[DataLoader] = None,
                         num_epochs: int = None) -> Dict[str, float]:
        """
        첫 번째 도메인 학습
        """
        logger.info("=== First Domain Training 시작 ===")
        
        # 데이터로더 준비
        if first_domain_dataloader is None:
            first_domain_dataloader = create_cached_first_domain_dataloader(
                data_dir=self.data_dir,
                domain_order=self.domain_order,
                dataset_type=self.dataset_type,
                subset='train', 
                batch_size=self.batch_size
            )
        
        # First Domain 설정 적용
        config = FIRST_DOMAIN_CONFIG
        logger.info("🎯 First Domain 설정 적용")
        
        if num_epochs is None:
            num_epochs = config['num_epochs']
        
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        
        logger.info(f"First Domain 설정: 에포크={num_epochs}, LR={self.learning_rate:.1e}, WD={self.weight_decay:.1e}")
        
        # Optimizer 설정
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
        
        # 학습 루프
        self.model.train()
        epoch_losses = []
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        # Validation loader
        val_loader = create_cached_first_domain_dataloader(
            data_dir=self.data_dir,
            domain_order=self.domain_order,
            dataset_type=self.dataset_type,
            subset='val',
            batch_size=self.batch_size
        )
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(first_domain_dataloader):
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                if (batch_idx % self.grad_accum_steps) == 0:
                    optimizer.zero_grad(set_to_none=True)
                
                # 진동 신호와 레이블 추출
                vibration = batch['vibration']
                labels = batch['labels']
                if labels.dim() == 2:
                    class_labels = labels[:, 0]  # 주 분류 레이블
                else:
                    class_labels = labels
                
                # Forward
                logits = self.model(vibration)
                loss = self.criterion(logits, class_labels)
                
                # Backward
                loss = loss / self.grad_accum_steps
                loss.backward()
                
                if ((batch_idx + 1) % self.grad_accum_steps) == 0 or (batch_idx + 1) == len(first_domain_dataloader):
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    optimizer.step()
                
                epoch_loss += loss.item() * self.grad_accum_steps
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            epoch_losses.append(avg_epoch_loss)
            scheduler.step()
            
            # Validation
            val_metrics = self._evaluate_single_domain(val_loader)
            val_acc = val_metrics['accuracy']
            
            logger.info(f"First Domain Epoch {epoch+1}/{num_epochs}: "
                       f"Loss = {avg_epoch_loss:.4f}, Val Acc = {val_acc:.4f}")
            
            # Early stopping
            min_ep = int(config.get('min_epoch', 5))
            patience_threshold = int(config.get('patience', 10))
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Best model 저장
                if self.save_dir is not None:
                    checkpoint_path = os.path.join(self.save_dir, 'first_domain_best.pth')
                    self.model.save_checkpoint(checkpoint_path, epoch, optimizer.state_dict())
            else:
                patience_counter += 1
                
                if (epoch + 1) >= min_ep and patience_counter >= patience_threshold:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.loss_history['first_domain'] = epoch_losses
        self.completed_domains.append(self.domain_order[0])
        self.current_domain_idx = 1
        
        # 첫 번째 도메인 성능 평가 및 기록
        first_domain_value = self.domain_order[0]
        first_test_loader = create_cached_first_domain_dataloader(
            data_dir=self.data_dir,
            domain_order=self.domain_order,
            dataset_type=self.dataset_type,
            subset='test',
            batch_size=self.batch_size
        )
        first_domain_metrics = self._evaluate_single_domain(first_test_loader)
        first_domain_acc = float(first_domain_metrics.get('accuracy', 0.0))
        
        # performance_history 초기화 및 첫 번째 도메인 성능 기록
        if first_domain_value not in self.performance_history:
            self.performance_history[first_domain_value] = {'accuracy': []}
        self.performance_history[first_domain_value]['accuracy'].append(first_domain_acc)
        
        # Replay buffer에 첫 도메인 샘플 저장
        self._update_replay_buffer(first_domain_dataloader, first_domain_value)
        
        logger.info(f"=== First Domain Training 완료: Best Val Acc = {best_val_acc:.4f}, Test Acc = {first_domain_acc:.4f} (epoch {best_epoch}) ===")
        
        return {
            'final_loss': epoch_losses[-1] if epoch_losses else float('inf'),
            'best_val_accuracy': best_val_acc,
            'test_accuracy': first_domain_acc,
            'best_epoch': best_epoch,
            'num_epochs': len(epoch_losses)
        }
    
    def train_remaining_domains(self, domain_dataloaders: Optional[Dict] = None) -> Dict[str, Any]:
        """
        나머지 도메인 순차 학습
        """
        logger.info("=== Remaining Domains Training 시작 ===")
        
        if domain_dataloaders is None:
            domain_dataloaders = create_cached_domain_dataloaders(
                data_dir=self.data_dir,
                domain_order=self.domain_order,
                dataset_type=self.dataset_type,
                batch_size=self.batch_size
            )
        
        # Continual config 적용
        config = CONTINUAL_CONFIG
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        
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
                # performance_history 초기화 (없는 경우)
                if eval_domain not in self.performance_history:
                    self.performance_history[eval_domain] = {'accuracy': []}
                acc = float(metrics.get('accuracy', 0.0))
                self.performance_history[eval_domain]['accuracy'].append(acc)
            
            remaining_domains_results[domain_value] = domain_results
            self.completed_domains.append(domain_value)
            self.current_domain_idx = domain_idx + 1
            
            # Replay buffer 업데이트
            self._update_replay_buffer(current_train_loader, domain_value)
        
        # 최종 메트릭 계산
        final_metrics = self._compute_final_metrics()
        
        logger.info("=== Remaining Domains Training 완료 ===")
        
        return {
            'domain_results': remaining_domains_results,
            'final_metrics': final_metrics
        }
    
    def _train_single_domain(self, 
                            domain_value: Union[int, str],
                            train_loader: DataLoader,
                            val_loader: DataLoader) -> Dict[str, float]:
        """단일 도메인 학습"""
        self.model.train()
        
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONTINUAL_CONFIG['num_epochs'], eta_min=1e-6
        )
        
        epoch_losses = []
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(CONTINUAL_CONFIG['num_epochs']):
            epoch_loss = 0.0
            num_batches = 0
            
            # Replay buffer에서 샘플 샘플링
            replay_batch = self._sample_replay_batch()
            
            for batch_idx, batch in enumerate(train_loader):
                batch = self._move_batch_to_device(batch)
                
                # Replay 샘플과 현재 배치 결합
                if replay_batch is not None:
                    combined_batch = self._combine_batches(batch, replay_batch)
                else:
                    combined_batch = batch
                
                # Forward pass
                if (batch_idx % self.grad_accum_steps) == 0:
                    optimizer.zero_grad(set_to_none=True)
                
                vibration = combined_batch['vibration']
                labels = combined_batch['labels']
                if labels.dim() == 2:
                    class_labels = labels[:, 0]
                else:
                    class_labels = labels
                
                logits = self.model(vibration)
                loss = self.criterion(logits, class_labels)
                
                loss = loss / self.grad_accum_steps
                loss.backward()
                
                if ((batch_idx + 1) % self.grad_accum_steps) == 0 or (batch_idx + 1) == len(train_loader):
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    optimizer.step()
                
                epoch_loss += loss.item() * self.grad_accum_steps
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            epoch_losses.append(avg_epoch_loss)
            scheduler.step()
            
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
                
                if self.save_dir is not None:
                    checkpoint_path = os.path.join(self.save_dir, f'domain_{domain_value}_best.pth')
                    self.model.save_checkpoint(checkpoint_path, epoch, optimizer.state_dict())
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
        """단일 도메인 성능 평가"""
        was_training = self.model.training
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                
                vibration = batch['vibration']
                labels = batch['labels']
                if labels.dim() == 2:
                    class_labels = labels[:, 0]
                else:
                    class_labels = labels
                
                logits = self.model(vibration)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.append(preds)
                all_labels.append(class_labels)
        
        if not all_preds:
            if was_training:
                self.model.train()
            return {'accuracy': 0.0}
        
        preds = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        accuracy = (preds == labels).float().mean().item()
        
        if was_training:
            self.model.train()
        
        return {'accuracy': accuracy}
    
    def _evaluate_all_domains(self, domain_dataloaders: Dict) -> Dict[Union[int, str], Dict[str, float]]:
        """모든 도메인 성능 평가"""
        results = {}
        
        for domain_value, loaders in domain_dataloaders.items():
            if 'test' in loaders:
                test_loader = loaders['test']
            elif 'val' in loaders:
                test_loader = loaders['val']
            else:
                continue
            
            metrics = self._evaluate_single_domain(test_loader)
            results[domain_value] = metrics
        
        return results
    
    def _calculate_forgetting(self, 
                             before_performance: Dict,
                             after_performance: Dict) -> float:
        """Forgetting 점수 계산"""
        if not before_performance:
            return 0.0
        
        forgetting_scores = []
        for domain, before_metrics in before_performance.items():
            if domain in after_performance:
                before_acc = float(before_metrics.get('accuracy', 0.0))
                after_acc = float(after_performance[domain].get('accuracy', 0.0))
                forgetting = max(0.0, before_acc - after_acc)
                forgetting_scores.append(forgetting)
        
        return np.mean(forgetting_scores) if forgetting_scores else 0.0
    
    def _compute_final_metrics(self) -> Dict[str, Any]:
        """최종 메트릭 계산"""
        final_accuracies = []
        
        for domain_value in self.domain_order:
            if domain_value in self.performance_history:
                history = self.performance_history[domain_value]['accuracy']
                if history:
                    final_accuracies.append(history[-1])
        
        average_accuracy = np.mean(final_accuracies) if final_accuracies else 0.0
        average_forgetting = np.mean(self.forgetting_scores) if self.forgetting_scores else 0.0
        
        return {
            'final_accuracies': final_accuracies,
            'average_accuracy': average_accuracy,
            'average_forgetting': average_forgetting
        }
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """배치를 디바이스로 이동"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _sample_replay_batch(self) -> Optional[Dict]:
        """Replay buffer에서 배치 샘플링"""
        if self.replay_buffer.buffer_size_per_domain == 0:
            return None
        
        # 원시 신호를 저장한 경우 직접 사용
        if hasattr(self.replay_buffer, 'raw_vibrations'):
            # 원시 신호 버퍼에서 샘플링
            num_samples = int(self.batch_size * self.replay_ratio)
            all_vibrations = []
            all_labels = []
            
            for domain_id in self.replay_buffer.domains:
                if domain_id in self.replay_buffer.raw_vibrations:
                    domain_vibs = self.replay_buffer.raw_vibrations[domain_id]
                    domain_labels = self.replay_buffer.raw_labels[domain_id]
                    
                    # 랜덤 샘플링
                    num_domain_samples = min(num_samples // len(self.replay_buffer.domains), len(domain_vibs))
                    if num_domain_samples > 0:
                        indices = torch.randperm(len(domain_vibs))[:num_domain_samples]
                        all_vibrations.append(domain_vibs[indices])
                        all_labels.append(domain_labels[indices])
            
            if not all_vibrations:
                return None
            
            vibrations = torch.cat(all_vibrations, dim=0)
            labels = torch.cat(all_labels, dim=0)
            
            return {
                'vibration': vibrations.to(self.device),
                'labels': labels.to(self.device)
            }
        else:
            # 기존 임베딩 기반 버퍼 (사용 안 함)
            return None
    
    def _combine_batches(self, batch1: Dict, batch2: Dict) -> Dict:
        """두 배치 결합"""
        combined = {
            'vibration': torch.cat([batch1['vibration'], batch2['vibration']], dim=0),
            'labels': torch.cat([batch1['labels'], batch2['labels']], dim=0)
        }
        return combined
    
    def _update_replay_buffer(self, dataloader: DataLoader, domain_value: Union[int, str]):
        """Replay buffer 업데이트 (원시 신호 저장)"""
        if self.replay_buffer.buffer_size_per_domain == 0:
            return
        
        # 원시 신호 버퍼 초기화 (없는 경우)
        if not hasattr(self.replay_buffer, 'raw_vibrations'):
            self.replay_buffer.raw_vibrations = {}
            self.replay_buffer.raw_labels = {}
        
        # 도메인별로 균등하게 샘플 선택
        samples_per_class = self.replay_buffer.buffer_size_per_domain // 7  # UOS: 7 classes
        
        class_vibrations = defaultdict(list)
        class_labels_list = defaultdict(list)
        
        for batch in dataloader:
            batch = self._move_batch_to_device(batch)
            vibrations = batch['vibration']
            labels = batch['labels']
            if labels.dim() == 2:
                class_labels = labels[:, 0]
            else:
                class_labels = labels
            
            for i in range(len(vibrations)):
                cls = int(class_labels[i].item())
                if len(class_vibrations[cls]) < samples_per_class:
                    class_vibrations[cls].append(vibrations[i].cpu())
                    class_labels_list[cls].append(labels[i].cpu())
        
        # 도메인별로 저장
        all_vibrations = []
        all_labels = []
        for cls_samples in class_vibrations.values():
            all_vibrations.extend(cls_samples)
        for cls_labels in class_labels_list.values():
            all_labels.extend(cls_labels)
        
        if all_vibrations:
            self.replay_buffer.raw_vibrations[domain_value] = torch.stack(all_vibrations)
            self.replay_buffer.raw_labels[domain_value] = torch.stack(all_labels)
            
            # Replay buffer가 활성화된 경우에만 로그 출력
            if self.replay_buffer.buffer_size_per_domain > 0:
                logger.debug(f"Domain {domain_value}: {len(all_vibrations)}개 원시 신호를 replay buffer에 저장")
    
    def _create_optimizer(self):
        """Optimizer 생성"""
        return optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def _create_scheduler(self, optimizer):
        """Scheduler 생성"""
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs, eta_min=1e-6
        )

