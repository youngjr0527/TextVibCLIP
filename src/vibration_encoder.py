"""
Vibration Encoder: TST 기반 진동 신호 인코더
1D 진동 신호를 512차원 임베딩으로 변환
"""

import math
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

# 스크립트로 실행 시(project root가 sys.path에 없을 때) 루트 경로 자동 추가
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from configs.model_config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal 위치 인코딩
    TST 논문의 표준 구현
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch_size, d_model)
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TimeSeriesTransformerEncoderLayer(nn.Module):
    """
    단일 Transformer Encoder Layer (Time Series 특화)
    """
    
    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'gelu'):
        super(TimeSeriesTransformerEncoderLayer, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        
        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"지원하지 않는 activation: {activation}")
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: (seq_len, batch_size, d_model)
            src_mask: attention mask
        Returns:
            output: (seq_len, batch_size, d_model)
        """
        # Multi-head self-attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class VibrationEncoder(nn.Module):
    """
    TST 기반 진동 신호 인코더
    
    1D 진동 신호 -> 512차원 임베딩 변환
    """
    
    def __init__(self,
                 input_length: int = MODEL_CONFIG['vibration_encoder']['input_length'],
                 d_model: int = MODEL_CONFIG['vibration_encoder']['d_model'],
                 num_heads: int = MODEL_CONFIG['vibration_encoder']['num_heads'],
                 num_layers: int = MODEL_CONFIG['vibration_encoder']['num_layers'],
                 dim_feedforward: int = MODEL_CONFIG['vibration_encoder']['dim_feedforward'],
                 dropout: float = MODEL_CONFIG['vibration_encoder']['dropout'],
                 activation: str = MODEL_CONFIG['vibration_encoder']['activation'],
                 embedding_dim: int = MODEL_CONFIG['embedding_dim']):
        """
        Args:
            input_length (int): 입력 신호 길이
            d_model (int): Transformer hidden dimension
            num_heads (int): Attention heads 수
            num_layers (int): Transformer layers 수
            dim_feedforward (int): FFN hidden dimension
            dropout (float): Dropout 비율
            activation (str): Activation function
            embedding_dim (int): 최종 출력 임베딩 차원
        """
        super(VibrationEncoder, self).__init__()
        
        self.input_length = input_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        # Token downsampling (patching) 옵션
        self.use_token_downsampling = MODEL_CONFIG['vibration_encoder'].get('use_token_downsampling', False)
        self.patch_size = int(MODEL_CONFIG['vibration_encoder'].get('patch_size', 1))
        
        # Input projection (1D signal -> d_model dimension)
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=input_length)
        
        # Transformer encoder layers
        encoder_layers = []
        for _ in range(num_layers):
            layer = TimeSeriesTransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            )
            encoder_layers.append(layer)
        
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        
        # Global pooling strategies
        self.pooling_type = 'attention'  # 'mean', 'max', 'attention'
        
        if self.pooling_type == 'attention':
            # Attention-based pooling
            self.attention_pooling = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1)
            )
        
        # Output projection (d_model -> embedding_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, MODEL_CONFIG['projection']['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['projection']['dropout']),
            nn.Linear(MODEL_CONFIG['projection']['hidden_dim'], embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # 파라미터 초기화
        self._init_parameters()
        
        if self.use_token_downsampling and self.patch_size > 1:
            eff_tokens = input_length // self.patch_size
            logger.info(f"VibrationEncoder 초기화: L{input_length} -> tokens {eff_tokens} (patch={self.patch_size}), d{d_model}, "
                        f"layers={num_layers}, heads={num_heads}")
        else:
            logger.info(f"VibrationEncoder 초기화: L{input_length}, d{d_model}, "
                       f"layers={num_layers}, heads={num_heads}")
    
    def _init_parameters(self):
        """파라미터 초기화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: 진동 신호 (batch_size, input_length)
            mask: padding mask (optional)
            
        Returns:
            torch.Tensor: 진동 임베딩 (batch_size, embedding_dim)
        """
        batch_size, seq_len = x.shape
        
        # Input validation
        if seq_len != self.input_length:
            raise ValueError(f"입력 길이 불일치: 예상 {self.input_length}, 실제 {seq_len}")
        
        # Reshape for projection: (batch_size, seq_len) -> (batch_size, seq_len, 1)
        x = x.unsqueeze(-1)

        # Input projection: (batch_size, seq_len, 1) -> (batch_size, seq_len, d_model)
        x = self.input_projection(x)

        # Token downsampling (patching) to reduce attention tokens
        if self.use_token_downsampling and self.patch_size > 1:
            if seq_len % self.patch_size != 0:
                raise ValueError(
                    f"patch_size({self.patch_size})가 seq_len({seq_len})을 나누지 못합니다. 입력 길이를 조정하거나 patch_size를 변경하세요."
                )
            num_patches = seq_len // self.patch_size
            x = x.view(batch_size, num_patches, self.patch_size, self.d_model).mean(dim=2)
            # x: (batch_size, num_patches, d_model)

        # Transpose for transformer: (batch_size, tokens, d_model) -> (tokens, batch_size, d_model)
        x = x.transpose(0, 1)

        # Scale and positional encoding (TST-style scaling)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer encoder layers
        for layer in self.transformer_encoder:
            x = layer(x, src_mask=mask)
        
        # Transpose back: (seq_len, batch_size, d_model) -> (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)
        
        # Global pooling
        if self.pooling_type == 'mean':
            # Mean pooling
            pooled = torch.mean(x, dim=1)  # (batch_size, d_model)
        elif self.pooling_type == 'max':
            # Max pooling
            pooled, _ = torch.max(x, dim=1)  # (batch_size, d_model)
        elif self.pooling_type == 'attention':
            # Attention-based pooling
            attention_weights = self.attention_pooling(x)  # (batch_size, seq_len, 1)
            attention_weights = F.softmax(attention_weights, dim=1)
            pooled = torch.sum(x * attention_weights, dim=1)  # (batch_size, d_model)
        else:
            raise ValueError(f"지원하지 않는 pooling_type: {self.pooling_type}")
        
        # Output projection
        output = self.output_projection(pooled)  # (batch_size, embedding_dim)
        
        return output
    
    def get_trainable_parameters(self) -> int:
        """학습 가능한 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_parameters(self):
        """모든 파라미터 freeze"""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("VibrationEncoder 파라미터 freeze 완료")
    
    def unfreeze_parameters(self):
        """모든 파라미터 unfreeze"""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("VibrationEncoder 파라미터 unfreeze 완료")


def create_vibration_encoder() -> VibrationEncoder:
    """
    설정에 따른 VibrationEncoder 생성
    
    Returns:
        VibrationEncoder: 설정된 진동 인코더
    """
    encoder = VibrationEncoder()
    
    logger.info(f"VibrationEncoder 생성 완료: {encoder.get_trainable_parameters():,} 파라미터")
    
    return encoder


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    print("=== VibrationEncoder 테스트 ===")
    
    # 인코더 생성
    encoder = create_vibration_encoder()
    
    # 테스트 데이터 생성
    batch_size = 8
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
    
    print("\n=== VibrationEncoder 테스트 완료 ===")
