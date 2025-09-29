"""
CWRU 전용 단순 분류기
극소 데이터에서는 멀티모달 학습 대신 진동 인코더만 사용

핵심 아이디어:
1. 진동 인코더 + 직접 분류만 사용
2. 텍스트는 추론 시에만 활용 (후처리)
3. 과적합 방지에 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import logging

from .vibration_encoder import create_vibration_encoder
from configs.model_config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class SimpleCWRUClassifier(nn.Module):
    """
    CWRU 전용 단순 분류기
    
    멀티모달 학습 없이 진동 인코더만 사용
    텍스트는 추론 시 후처리로만 활용
    """
    
    def __init__(self, num_classes: int = 4):
        super().__init__()
        
        # 진동 인코더만 사용
        self.vib_encoder = create_vibration_encoder()
        
        # 단순한 분류 헤드 (과적합 방지)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # 강한 드롭아웃
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # 클래스별 텍스트 매핑 (추론용)
        self.class_to_text = {
            0: "Healthy bearing condition observed",
            1: "Ball element defect detected",
            2: "Inner race fault observed", 
            3: "Outer race defect detected"
        }
        
        logger.info(f"SimpleCWRUClassifier 초기화: {num_classes}개 클래스")
    
    def forward(self, batch: Dict) -> Dict:
        """
        Forward pass (학습용)
        
        Args:
            batch: 배치 데이터
            
        Returns:
            Dict: loss 정보
        """
        vibration = batch['vibration']
        labels = batch['labels']
        
        if labels.dim() == 2:
            labels = labels[:, 0]  # CWRU는 1차원 라벨
        
        # 진동 인코딩 + 분류
        vib_emb = self.vib_encoder(vibration)
        logits = self.classifier(vib_emb)
        
        # 단순한 CrossEntropy Loss
        loss = F.cross_entropy(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'embeddings': vib_emb
        }
    
    def predict_with_text(self, vibration_signal: torch.Tensor) -> tuple:
        """
        추론: 진동 신호 → 분류 → 텍스트 설명
        
        Args:
            vibration_signal: 진동 신호
            
        Returns:
            tuple: (predicted_class, text_description, confidence)
        """
        self.eval()
        
        with torch.no_grad():
            if vibration_signal.dim() == 1:
                vibration_signal = vibration_signal.unsqueeze(0)
            
            # 진동 분류
            vib_emb = self.vib_encoder(vibration_signal)
            logits = self.classifier(vib_emb)
            
            # 예측 결과
            probs = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probs.max().item()
            
            # 텍스트 설명 매핑
            text_description = self.class_to_text.get(predicted_class, "Unknown")
            
            return predicted_class, text_description, confidence
    
    def get_trainable_parameters(self) -> int:
        """학습 가능한 파라미터 수"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleCWRUTrainer:
    """CWRU 전용 단순 트레이너"""
    
    def __init__(self, device: torch.device = torch.device('cuda')):
        self.device = device
        self.model = SimpleCWRUClassifier().to(device)
        
        # 극소 데이터 전용 설정
        self.learning_rate = 1e-4
        self.weight_decay = 1e-3
        self.num_epochs = 5
        self.patience = 2
        
        logger.info(f"SimpleCWRUTrainer 초기화: {self.model.get_trainable_parameters():,} 파라미터")
    
    def train(self, train_loader, val_loader):
        """단순 학습"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # 학습
            self.model.train()
            epoch_loss = 0.0
            
            for batch in train_loader:
                batch = self._move_to_device(batch)
                
                optimizer.zero_grad()
                results = self.model(batch)
                loss = results['loss']
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # 검증
            val_acc = self.evaluate(val_loader)
            
            logger.info(f"Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.4f}, Val Acc = {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return best_val_acc
    
    def evaluate(self, test_loader):
        """평가"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = self._move_to_device(batch)
                results = self.model(batch)
                
                predictions = torch.argmax(results['logits'], dim=1)
                labels = batch['labels']
                if labels.dim() == 2:
                    labels = labels[:, 0]
                
                correct += (predictions == labels).sum().item()
                total += len(labels)
        
        return correct / total if total > 0 else 0.0
    
    def _move_to_device(self, batch):
        """배치를 디바이스로 이동"""
        batch['vibration'] = batch['vibration'].to(self.device)
        batch['labels'] = batch['labels'].to(self.device)
        return batch


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    print("=== SimpleCWRUClassifier 테스트 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = SimpleCWRUTrainer(device)
    
    # 테스트 데이터
    test_vibration = torch.randn(2048).to(device)
    pred_class, text_desc, confidence = trainer.model.predict_with_text(test_vibration)
    
    print(f"예측 클래스: {pred_class}")
    print(f"텍스트 설명: {text_desc}")
    print(f"신뢰도: {confidence:.4f}")
    
    print("\n=== SimpleCWRUClassifier 테스트 완료 ===")
