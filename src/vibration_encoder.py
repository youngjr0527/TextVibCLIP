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
        
        # 🎯 OPTIMIZED: 2048 입력에 최적화된 4-layer 1D-CNN
        # 자연스러운 차원 축소: 2048 → 1024 → 512 → 256 → 128
        kernel_sizes = MODEL_CONFIG['vibration_encoder']['kernel_sizes']
        channels = MODEL_CONFIG['vibration_encoder']['channels']
        dropout_rate = MODEL_CONFIG['vibration_encoder']['dropout']
        
        self.conv_layers = nn.Sequential(
            # Block 1: 고주파 충격 패턴 감지 (베어링 결함 특유의 충격파)
            nn.Conv1d(1, channels[0], kernel_size=kernel_sizes[0], stride=2, padding=kernel_sizes[0]//2),  # 2048 → 1024
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Block 2: 중간 주파수 패턴 (회전 주기, 조화파)
            nn.Conv1d(channels[0], channels[1], kernel_size=kernel_sizes[1], stride=2, padding=kernel_sizes[1]//2),  # 1024 → 512
            nn.BatchNorm1d(channels[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Block 3: 저주파 구조적 진동 패턴
            nn.Conv1d(channels[1], channels[2], kernel_size=kernel_sizes[2], stride=2, padding=kernel_sizes[2]//2),  # 512 → 256
            nn.BatchNorm1d(channels[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Block 4: 특징 집약 및 최종 표현
            nn.Conv1d(channels[2], channels[3], kernel_size=kernel_sizes[3], stride=2, padding=kernel_sizes[3]//2),  # 256 → 128
            nn.BatchNorm1d(channels[3]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Global Average Pooling (시간축 정보 집약)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection to embedding space (마지막 conv 채널 수에 맞춤)
        final_conv_channels = channels[-1]  # 512
        self.projection = nn.Sequential(
            nn.Linear(final_conv_channels, MODEL_CONFIG['projection']['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['projection']['dropout']),
            nn.Linear(MODEL_CONFIG['projection']['hidden_dim'], embedding_dim)
            # 🎯 FIXED: LayerNorm 제거 (gradient vanishing 방지)
        )
        
        # 🎯 FIXED: 안정적인 스케일링 팩터
        self.embedding_scaler = nn.Parameter(torch.tensor(3.0))  # 10.0 → 3.0 (안정화)
        
        # Projection layer 초기화 (CLIP-style)
        with torch.no_grad():
            # 첫 번째 projection layer: Xavier normal
            nn.init.xavier_normal_(self.projection[0].weight)
            nn.init.zeros_(self.projection[0].bias)
            
            # 마지막 projection layer: 표준 초기화
            nn.init.xavier_normal_(self.projection[3].weight)
            nn.init.zeros_(self.projection[3].bias)

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
        logger.info(f"   OPTIMIZED: 커널 크기: {kernel_sizes} - 4-layer 베어링 최적화")
        logger.info(f"   OPTIMIZED: 채널 수: {channels} - 자연스러운 64→512 증가")
        logger.info(f"   총 파라미터: {self.get_trainable_parameters():,}")
    
    def _init_parameters(self):
        """파라미터 초기화 (개선된 초기화)"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                # He initialization (ReLU에 적합)
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # 마지막 projection layer는 더 작은 초기화
                if module == self.projection[-1]:  # 마지막 Linear layer
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
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
        
        # 🎯 SIMPLIFIED: 통일된 입력 길이 (2048)
        # 입력 길이 검증
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
        
        # 🎯 CRITICAL FIX: 임베딩 크기 스케일링 (Text encoder와 균형 맞추기)
        output = output * self.embedding_scaler
        
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