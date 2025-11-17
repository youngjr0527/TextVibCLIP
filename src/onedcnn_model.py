"""
1D-CNN Classification Model: TextVibCLIP 비교군
진동 신호만을 사용하는 단일 모달 분류 모델
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

# 스크립트로 실행 시(project root가 sys.path에 없을 때) 루트 경로 자동 추가
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from configs.model_config import MODEL_CONFIG
from .vibration_encoder import VibrationEncoder

logger = logging.getLogger(__name__)


class OneDCNNClassifier(nn.Module):
    """
    1D-CNN 기반 분류 모델 (TextVibCLIP 비교군)
    
    진동 신호만을 사용하여 베어링 결함을 분류하는 단일 모달 모델
    """
    
    def __init__(self,
                 input_length: int = MODEL_CONFIG['vibration_encoder']['input_length'],
                 num_classes: int = 7,  # UOS: 7 classes
                 dataset_type: str = 'uos'):
        """
        Args:
            input_length (int): 입력 신호 길이 (기본값: 2048)
            num_classes (int): 분류 클래스 수 (UOS: 7)
            dataset_type (str): 데이터셋 타입 ('uos' or 'cwru')
        """
        super(OneDCNNClassifier, self).__init__()
        
        self.input_length = input_length
        self.num_classes = num_classes
        self.dataset_type = dataset_type
        
        # 진동 인코더 (기존 VibrationEncoder 재사용)
        self.vib_encoder = VibrationEncoder(
            input_length=input_length,
            embedding_dim=MODEL_CONFIG['embedding_dim']
        )
        
        # Classification head
        embedding_dim = MODEL_CONFIG['embedding_dim']
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, num_classes)
        )
        
        # 파라미터 초기화
        self._init_classifier()
        
        logger.info(f"1D-CNN Classifier 초기화: input_length={input_length}, "
                   f"num_classes={num_classes}, dataset_type={dataset_type}")
        logger.info(f"   총 파라미터: {self.get_trainable_parameters():,}")
    
    def _init_classifier(self):
        """Classification head 초기화"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, vibration: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            vibration: 진동 신호 (batch_size, input_length)
            
        Returns:
            torch.Tensor: 분류 로짓 (batch_size, num_classes)
        """
        # 진동 인코더를 통해 특징 추출
        vib_features = self.vib_encoder(vibration)  # (batch_size, embedding_dim)
        
        # Classification
        logits = self.classifier(vib_features)  # (batch_size, num_classes)
        
        return logits
    
    def get_trainable_parameters(self) -> int:
        """학습 가능한 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Optional[dict] = None):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_length': self.input_length,
                'num_classes': self.num_classes,
                'dataset_type': self.dataset_type
            }
        }
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, path)
        logger.info(f"체크포인트 저장: {path}")
    
    def load_checkpoint(self, path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"체크포인트 로드: {path} (epoch {checkpoint.get('epoch', 'unknown')})")
        return checkpoint.get('epoch', 0)


def create_onedcnn_model(dataset_type: str = 'uos') -> OneDCNNClassifier:
    """
    1D-CNN 분류 모델 생성
    
    Args:
        dataset_type (str): 데이터셋 타입 ('uos' or 'cwru')
        
    Returns:
        OneDCNNClassifier: 1D-CNN 분류 모델
    """
    if dataset_type == 'uos':
        num_classes = 7
    elif dataset_type == 'cwru':
        num_classes = 4
    else:
        raise ValueError(f"지원하지 않는 데이터셋 타입: {dataset_type}")
    
    model = OneDCNNClassifier(
        input_length=MODEL_CONFIG['vibration_encoder']['input_length'],
        num_classes=num_classes,
        dataset_type=dataset_type
    )
    
    logger.info(f"1D-CNN Classifier 생성 완료: {model.get_trainable_parameters():,} 파라미터")
    
    return model


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    print("=== 1D-CNN Classifier 테스트 ===")
    
    # 모델 생성
    model = create_onedcnn_model('uos')
    
    # 테스트 데이터 생성
    batch_size = 32
    input_length = MODEL_CONFIG['vibration_encoder']['input_length']
    test_signals = torch.randn(batch_size, input_length)
    
    print(f"입력 신호 shape: {test_signals.shape}")
    
    # GPU 사용 가능하면 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_signals = test_signals.to(device)
    
    # Forward pass 테스트
    with torch.no_grad():
        logits = model(test_signals)
    
    print(f"출력 로짓 shape: {logits.shape}")
    print(f"예측 클래스: {torch.argmax(logits, dim=1)}")
    
    # 파라미터 정보
    print(f"총 파라미터 수: {model.get_trainable_parameters():,}")
    
    print("\n=== 1D-CNN Classifier 테스트 완료 ===")

