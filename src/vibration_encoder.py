"""
Vibration Encoder: 1D-CNN 기반 진동 신호 인코더
1D 진동 신호를 512차원 임베딩으로 변환
베어링 결함의 다양한 주파수 패턴을 효과적으로 감지

TST → 1D-CNN 교체 이유:
- 메모리 효율성: O(n) 복잡도로 안정적 처리
- 성능 우수성: 79.0% vs 66.7% (TST 대비 12.3%p 향상)
- 실용성: 일반적인 GPU에서도 원활한 작동 가능
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

logger = logging.getLogger(__name__)


class VibrationEncoder(nn.Module):
    """
    1D-CNN 기반 진동 신호 인코더
    
    베어링 결함의 다양한 주파수 패턴을 다중 스케일 커널로 효과적 감지:
    - Block 1: 고주파 충격 패턴 (베어링 결함 특유의 충격파)
    - Block 2: 중간 주파수 패턴 (회전 주기, 조화파)
    - Block 3: 저주파 구조적 진동 패턴
    - Block 4: 특징 집약
    
    1D 진동 신호 (4096 samples) → 512차원 임베딩 변환
    """
    
    def __init__(self,
                 input_length: int = MODEL_CONFIG['vibration_encoder']['input_length'],
                 embedding_dim: int = MODEL_CONFIG['embedding_dim']):
        """
        Args:
            input_length (int): 입력 신호 길이 (기본값: 4096)
            embedding_dim (int): 최종 출력 임베딩 차원 (기본값: 512)
        """
        super(VibrationEncoder, self).__init__()
        
        self.input_length = input_length
        self.embedding_dim = embedding_dim
        
        # 다중 스케일 1D Convolution Layers
        # 베어링 결함의 다양한 주파수 특성을 캡처하기 위해 서로 다른 커널 크기 사용
        self.conv_layers = nn.Sequential(
            # Block 1: 고주파 충격 패턴 감지 (베어링 결함 특유의 충격파)
            nn.Conv1d(1, 64, kernel_size=16, stride=2, padding=8),  # 4096 -> 2048
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Block 2: 중간 주파수 패턴 (회전 주기, 조화파)
            nn.Conv1d(64, 128, kernel_size=32, stride=2, padding=16),  # 2048 -> 1024
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Block 3: 저주파 구조적 진동 패턴
            nn.Conv1d(128, 256, kernel_size=64, stride=2, padding=32),  # 1024 -> 512
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Block 4: 특징 집약
            nn.Conv1d(256, 512, kernel_size=32, stride=2, padding=16),  # 512 -> 256
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Global Average Pooling (시간축 정보 집약)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection to embedding space
        self.projection = nn.Sequential(
            nn.Linear(512, MODEL_CONFIG['projection']['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['projection']['dropout']),
            nn.Linear(MODEL_CONFIG['projection']['hidden_dim'], embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        # Auxiliary classification head (bearing condition 4-class)
        aux_cfg = MODEL_CONFIG.get('aux_classification', {'enabled': False})
        self.use_aux_head = bool(aux_cfg.get('enabled', False))
        if self.use_aux_head:
            num_classes = int(aux_cfg.get('num_classes', 4))
            aux_dropout = float(aux_cfg.get('dropout', 0.1))
            self.aux_head = nn.Sequential(
                nn.Dropout(aux_dropout),
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(embedding_dim // 2, num_classes)
            )
        
        # 파라미터 초기화
        self._init_parameters()
        
        logger.info(f"1D-CNN VibrationEncoder 초기화: input_length={input_length}, "
                   f"embedding_dim={embedding_dim}")
        logger.info(f"   커널 크기: [16, 32, 64, 32] - 다중 주파수 대역 커버")
        logger.info(f"   총 파라미터: {self.get_trainable_parameters():,}")
    
    def _init_parameters(self):
        """파라미터 초기화 (Xavier uniform)"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: 진동 신호 (batch_size, input_length)
            mask: padding mask (사용되지 않음, TST 호환성을 위해 유지)
            
        Returns:
            torch.Tensor: 진동 임베딩 (batch_size, embedding_dim)
        """
        batch_size, seq_len = x.shape
        
        # Input validation
        if seq_len != self.input_length:
            raise ValueError(f"입력 길이 불일치: 예상 {self.input_length}, 실제 {seq_len}")
        
        # Reshape: (batch_size, input_length) -> (batch_size, 1, input_length)
        x = x.unsqueeze(1)
        
        # 1D Convolution layers
        x = self.conv_layers(x)  # (batch_size, 512, reduced_length)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch_size, 512, 1)
        x = x.squeeze(-1)  # (batch_size, 512)
        
        # Final projection
        output = self.projection(x)  # (batch_size, embedding_dim)
        
        return output
    
    def get_trainable_parameters(self) -> int:
        """학습 가능한 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_parameters(self):
        """모든 파라미터 freeze"""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("1D-CNN VibrationEncoder 파라미터 freeze 완료")
    
    def unfreeze_parameters(self):
        """모든 파라미터 unfreeze"""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("1D-CNN VibrationEncoder 파라미터 unfreeze 완료")


def create_vibration_encoder() -> VibrationEncoder:
    """
    설정에 따른 1D-CNN VibrationEncoder 생성
    
    Returns:
        VibrationEncoder: 설정된 1D-CNN 진동 인코더
    """
    encoder = VibrationEncoder()
    
    logger.info(f"1D-CNN VibrationEncoder 생성 완료: {encoder.get_trainable_parameters():,} 파라미터")
    
    return encoder


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    print("=== 1D-CNN VibrationEncoder 테스트 ===")
    
    # 인코더 생성
    encoder = create_vibration_encoder()
    
    # 테스트 데이터 생성
    batch_size = 32  # 1D-CNN은 큰 배치 크기 지원
    input_length = MODEL_CONFIG['vibration_encoder']['input_length']
    
    # 더미 진동 신호 (정규분포)
    test_signals = torch.randn(batch_size, input_length)
    
    print(f"입력 신호 shape: {test_signals.shape}")
    
    # GPU 사용 가능하면 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    test_signals = test_signals.to(device)
    
    # Forward pass 테스트
    with torch.no_grad():
        embeddings = encoder(test_signals)
    
    print(f"출력 임베딩 shape: {embeddings.shape}")
    print(f"임베딩 norm: {torch.norm(embeddings, dim=1)}")
    
    # 파라미터 정보
    print(f"총 파라미터 수: {encoder.get_trainable_parameters():,}")
    
    # 다양한 길이 신호 테스트 (오류 발생 예상)
    try:
        wrong_length_signal = torch.randn(batch_size, input_length // 2).to(device)
        encoder(wrong_length_signal)
    except ValueError as e:
        print(f"예상된 오류 발생: {e}")
    
    # Auxiliary head 테스트 (활성화된 경우)
    if encoder.use_aux_head:
        with torch.no_grad():
            aux_logits = encoder.aux_head(embeddings)
        print(f"Auxiliary logits shape: {aux_logits.shape}")
    
    print("\n=== 1D-CNN VibrationEncoder 테스트 완료 ===")