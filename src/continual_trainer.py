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
from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from .textvib_model import TextVibCLIP, create_textvib_model
from .replay_buffer import ReplayBuffer
from .data_loader import create_domain_dataloaders, create_combined_dataloader, create_first_domain_dataloader
from .utils import setup_amp_and_scaler
from configs.model_config import TRAINING_CONFIG, DATA_CONFIG, EVAL_CONFIG

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
                 max_grad_norm: float = 1.0):
        """
        Args:
            model (TextVibCLIP, optional): 사전 초기화된 모델
            device (torch.device): 학습 디바이스
            save_dir (str): 체크포인트 저장 경로
            use_amp (bool): AMP 사용 여부
            max_grad_norm (float): Gradient clipping 최대 norm
        """
        self.device = device
        self.save_dir = save_dir
        self.max_grad_norm = max_grad_norm
        os.makedirs(save_dir, exist_ok=True)
        
        # AMP 설정
        self.scaler, self.use_amp = setup_amp_and_scaler(device, use_amp)
        
        # 모델 초기화
        if model is None:
            self.model = create_textvib_model('joint')
        else:
            self.model = model
        
        self.model.to(device)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # 학습 상태 관리
        self.current_domain_idx = 0
        self.completed_domains = []
        self.domain_order = DATA_CONFIG['domain_order']
        
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
        self.model.switch_to_joint_mode()  # Joint mode 유지 (LoRA 활성화)
        
        # Optimizer 설정 (Text LoRA + Vibration full)
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer, len(first_domain_dataloader) * num_epochs)
        
        # 학습 루프
        self.model.train()
        epoch_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(first_domain_dataloader):
                # 배치를 디바이스로 이동
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                optimizer.zero_grad()
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        results = self.model(batch)
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
                    results = self.model(batch)
                    loss = results['loss']
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    optimizer.step()
                
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
        domain_dataloaders = create_domain_dataloaders(batch_size=self.batch_size)
        first_domain_performance = self._evaluate_all_domains(domain_dataloaders)
        
        # 성능 기록 (모든 메트릭 저장)
        for domain, metrics in first_domain_performance.items():
            if domain not in self.performance_history:
                self.performance_history[domain] = {'accuracy': [], 'top1_retrieval': [], 'top5_retrieval': []}
            
            self.performance_history[domain]['accuracy'].append(metrics['accuracy'])
            self.performance_history[domain]['top1_retrieval'].append(metrics.get('top1_retrieval', 0.0))
            self.performance_history[domain]['top5_retrieval'].append(metrics.get('top5_retrieval', 0.0))
        
        logger.info("=== First Domain Training 완료 ===")
        
        return {
            'final_loss': epoch_losses[-1],
            'avg_loss': np.mean(epoch_losses),
            'domain_performances': first_domain_performance
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
        
        # Domain 2부터 순차 학습 (첫 번째 도메인은 이미 완료)
        for domain_idx in range(1, len(self.domain_order)):
            domain_rpm = self.domain_order[domain_idx]
            
            logger.info(f"\n--- Domain {domain_rpm} RPM 학습 시작 ---")
            
            # 현재 도메인 데이터로더
            current_train_loader = domain_dataloaders[domain_rpm]['train']
            current_val_loader = domain_dataloaders[domain_rpm]['val']
            
            # 이전 도메인 성능 기록 (Forgetting 계산용)
            before_performance = self._evaluate_all_domains(domain_dataloaders)
            
            # 현재 도메인 학습
            domain_results = self._train_single_domain(
                domain_rpm, current_train_loader, current_val_loader
            )
            
            # 학습 후 성능 평가
            after_performance = self._evaluate_all_domains(domain_dataloaders)
            
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
            self.completed_domains.append(domain_rpm)
            remaining_domains_results[domain_rpm] = {
                'training_results': domain_results,
                'performance': after_performance,
                'forgetting_score': forgetting_score
            }
            
            logger.info(f"Domain {domain_rpm} 학습 완료: "
                       f"Forgetting Score = {forgetting_score:.4f}")
        
        # 최종 성능 요약
        final_metrics = self._calculate_final_metrics()
        remaining_domains_results['final_metrics'] = final_metrics
        
        logger.info("=== Remaining Domains Training 완료 ===")
        
        return remaining_domains_results
    
    def _train_single_domain(self, 
                           domain_rpm: int,
                           train_loader: DataLoader,
                           val_loader: DataLoader) -> Dict[str, float]:
        """
        단일 도메인 학습 (Replay 포함)
        
        Args:
            domain_rpm (int): 도메인 RPM
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
                domain_rpm,
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
                
                # Replay 데이터 샘플링
                replay_batch_size = int(current_batch_size * self.replay_ratio)
                replay_data = self.replay_buffer.sample_replay_data(
                    replay_batch_size, exclude_current=True, device=self.device
                )
                
                # 현재 데이터와 Replay 데이터 결합
                if replay_data is not None:
                    combined_batch = self._combine_current_and_replay(batch, replay_data)
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
                    logger.info(f"D{domain_rpm} E{epoch+1} B{batch_idx}: "
                               f"Loss={loss.item():.4f}, {replay_info}")
            
            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            
            # Validation
            val_metrics = self._evaluate_single_domain(val_loader)
            val_acc = val_metrics['accuracy']
            
            logger.info(f"Domain {domain_rpm} Epoch {epoch+1}: "
                       f"Loss = {avg_epoch_loss:.4f}, Val Acc = {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Best model 저장
                checkpoint_path = os.path.join(self.save_dir, f'domain_{domain_rpm}_best.pth')
                self.model.save_checkpoint(checkpoint_path, epoch, optimizer.state_dict())
            else:
                patience_counter += 1
                
                if patience_counter >= TRAINING_CONFIG['patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 도메인 학습 완료
        self.loss_history[domain_rpm] = epoch_losses
        
        return {
            'final_loss': epoch_losses[-1],
            'best_val_accuracy': best_val_acc,
            'num_epochs': len(epoch_losses)
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
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                results = self.model(batch, return_embeddings=True)
                
                all_text_embeddings.append(results['text_embeddings'])
                all_vib_embeddings.append(results['vib_embeddings'])
        
        if not all_text_embeddings:
            return {'accuracy': 0.0, 'top1_retrieval': 0.0, 'top5_retrieval': 0.0}
        
        # 모든 임베딩 결합
        text_emb = torch.cat(all_text_embeddings, dim=0)
        vib_emb = torch.cat(all_vib_embeddings, dim=0)
        
        # L2 정규화
        text_emb = F.normalize(text_emb, dim=1)
        vib_emb = F.normalize(vib_emb, dim=1)
        
        # 1. 기존 threshold 기반 정확도
        similarity = torch.cosine_similarity(text_emb, vib_emb, dim=1)
        predicted = (similarity > 0.5).long()
        actual = torch.ones_like(predicted)
        threshold_accuracy = (predicted == actual).float().mean().item()
        
        # 2. Text-to-Vibration Retrieval
        similarity_matrix = torch.matmul(text_emb, vib_emb.t())  # (N, N)
        
        # Top-1 retrieval accuracy
        _, top1_indices = torch.topk(similarity_matrix, k=1, dim=1)
        correct_indices = torch.arange(text_emb.size(0), device=text_emb.device).unsqueeze(1)
        top1_accuracy = (top1_indices == correct_indices).float().mean().item()
        
        # Top-5 retrieval accuracy
        k = min(5, similarity_matrix.size(1))
        _, topk_indices = torch.topk(similarity_matrix, k=k, dim=1)
        correct_indices_expanded = correct_indices.expand(-1, k)
        top5_accuracy = (topk_indices == correct_indices_expanded).any(dim=1).float().mean().item()
        
        return {
            'accuracy': threshold_accuracy,
            'top1_retrieval': top1_accuracy,
            'top5_retrieval': top5_accuracy
        }
    
    def _evaluate_all_domains(self, domain_dataloaders: Dict) -> Dict[int, Dict[str, float]]:
        """모든 도메인 성능 평가"""
        results = {}
        
        for domain_rpm, loaders in domain_dataloaders.items():
            test_loader = loaders['test']
            metrics = self._evaluate_single_domain(test_loader)
            
            results[domain_rpm] = {
                **metrics,  # accuracy, top1_retrieval, top5_retrieval
                'num_samples': len(test_loader.dataset)
            }
        
        return results
    
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
        """Joint training용 optimizer 생성"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def _create_continual_optimizer(self) -> torch.optim.Optimizer:
        """Continual learning용 optimizer 생성 (Vibration encoder만)"""
        return optim.AdamW(
            self.model.vibration_encoder.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def _create_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int):
        """Learning rate scheduler 생성"""
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """배치를 디바이스로 이동"""
        if 'vibration' in batch:
            batch['vibration'] = batch['vibration'].to(self.device)
        if 'labels' in batch:
            batch['labels'] = batch['labels'].to(self.device)
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
