"""
TextVibCLIP: 진동 신호와 텍스트의 멀티모달 대조 학습 모델
CLIP-inspired architecture with asymmetric continual learning
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import logging

# 스크립트로 실행 시(project root가 sys.path에 없을 때) 루트 경로 자동 추가
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from .text_encoder import TextEncoder, create_text_encoder
from .vibration_encoder import VibrationEncoder, create_vibration_encoder
from configs.model_config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """
    Bidirectional InfoNCE Loss with asymmetric temperature
    
    - First domain training: τ_text = τ_vib (균등 학습)
    - Continual learning: τ_text > τ_vib (텍스트 안정화, 진동 적응 강화)
    """
    
    def __init__(self, 
                 temperature_text: float = 0.07,
                 temperature_vib: float = 0.07,
                 reduction: str = 'mean'):
        """
        Args:
            temperature_text (float): Text-to-vibration direction temperature
            temperature_vib (float): Vibration-to-text direction temperature  
            reduction (str): Loss reduction method
        """
        super(InfoNCELoss, self).__init__()
        
        # 🎯 CRITICAL FIX: 학습 가능한 온도 파라미터 (CLIP-style)
        # 초기값을 log space에서 설정하여 안정적인 학습
        self.log_temperature_text = nn.Parameter(torch.log(torch.tensor(temperature_text)))
        self.log_temperature_vib = nn.Parameter(torch.log(torch.tensor(temperature_vib)))
        
        self.reduction = reduction
        
        logger.info(f"InfoNCE Loss 초기화: τ_text={temperature_text:.3f}, τ_vib={temperature_vib:.3f} (학습 가능)")
    
    def _class_based_infonce_loss(self, logits: torch.Tensor, positive_mask: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        클래스 기반 InfoNCE loss
        
        Args:
            logits: 유사도 매트릭 (N, M)
            positive_mask: 같은 클래스는 True, 다른 클래스는 False (N, M)
            temperature: 온도 파라미터
            
        Returns:
            torch.Tensor: InfoNCE loss
        """
        # Temperature scaling
        logits_scaled = logits / temperature
        
        # 🎯 AMP 안전성: Half precision 범위 내에서 연산
        logits_f32 = logits_scaled.float()
        N = logits_f32.size(0)
        M = logits_f32.size(1)
        device = logits_f32.device
        
        # 분모: log-sum-exp(all)
        log_denominator = torch.logsumexp(logits_f32, dim=1)  # (N,)
        
        # 자기자신 제외 마스크(직사각형 대응)
        eye_mask = torch.zeros((N, M), dtype=torch.bool, device=device)
        diag_len = min(N, M)
        if diag_len > 0:
            idx = torch.arange(diag_len, device=device)
            eye_mask[idx, idx] = True
        pos_mask_no_self = positive_mask & (~eye_mask)
        
        # 기본: positive 위치만 남김
        masked_logits = logits_f32.masked_fill(~pos_mask_no_self, -1e4)
        log_numerator = torch.logsumexp(masked_logits, dim=1)  # (N,)
        
        # 🎯 FIXED: 행별로 positive가 전혀 없는 경우 안전한 fallback
        has_positive = pos_mask_no_self.any(dim=1)  # (N,)
        if not torch.all(has_positive):
            # positive가 없는 행들에 대해서만 대각 원소 사용
            no_positive_mask = ~has_positive  # (N,)
            
            if diag_len > 0:
                # 대각선 원소를 사용할 수 있는 행들 (row index < diag_len)
                row_idx = torch.arange(N, device=device)
                can_use_diag = (row_idx < diag_len) & no_positive_mask  # (N,)
                
                if can_use_diag.any():
                    # 해당 행들의 대각선 원소 값
                    diag_values = torch.zeros(N, device=device)  # (N,) 크기로 초기화
                    diag_indices = row_idx[can_use_diag]
                    diag_values[can_use_diag] = logits_f32[diag_indices, diag_indices]
                    
                    # positive가 없는 행들에 대해서만 대각선 값으로 대체
                    log_numerator = torch.where(can_use_diag, diag_values, log_numerator)
        
        # InfoNCE: -log(exp(pos_sum) / exp(all_sum))
        loss_per_sample = -(log_numerator - log_denominator)

        # 🎯 If some rows have at least one positive (excluding self), average only over them.
        #    If none have positives, fall back to averaging all (diagonal already injected above).
        if has_positive.any():
            return loss_per_sample[has_positive].mean()
        else:
            return loss_per_sample.mean()
    
    def forward(self, 
                text_embeddings: torch.Tensor, 
                vib_embeddings: torch.Tensor,
                batch_labels: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Bidirectional InfoNCE loss 계산
        
        Args:
            text_embeddings: (batch_size, embedding_dim)
            vib_embeddings: (batch_size, embedding_dim)
            
        Returns:
            Tuple[torch.Tensor, Dict]: (total_loss, loss_components)
        """
        batch_size = text_embeddings.size(0)
        
        # 🎯 CRITICAL FIX: 강제 임베딩 정렬 (CLIP-inspired)
        # L2 정규화
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        vib_embeddings = F.normalize(vib_embeddings, p=2, dim=1)
        
        # 학습 가능한 온도 파라미터 사용
        temp_text = torch.exp(self.log_temperature_text)
        temp_vib = torch.exp(self.log_temperature_vib)
        
        # 🎯 FIXED: 적절한 스케일링 (발산 방지)
        # 온도가 낮을 때 적당한 스케일링만 적용
        alignment_scale = torch.clamp(1.0 / temp_text, min=1.0, max=10.0)  # 100 → 10 (10배 감소)
        text_embeddings = text_embeddings * alignment_scale
        vib_embeddings = vib_embeddings * alignment_scale
        
        # 🎯 FIXED: 클래스 기반 Contrastive Learning
        # 같은 고장 유형끼리 positive pairs, 다른 고장 유형은 negative pairs
        
        # 배치에서 라벨 정보 추출 (매개변수로 전달받음)
        # batch_labels = batch.get('labels', None)  # 이제 매개변수로 받음
        if batch_labels is not None:
            if batch_labels.dim() == 2:
                if batch_labels.size(1) == 2:
                    # UOS: 첫 번째 차원이 주 분류 (7-클래스)
                    class_labels = batch_labels[:, 0]  # [0,1,2,3,4,5,6] for H/B/IR/OR/L/U/M
                elif batch_labels.size(1) == 1:
                    # CWRU: (batch_size, 1) 형태의 라벨
                    class_labels = batch_labels[:, 0]  # [0,1,2,3] for Normal/B/IR/OR
                else:
                    # 예상치 못한 형태
                    class_labels = batch_labels[:, 0]
            elif batch_labels.dim() == 1:
                # 1차원 라벨 (직접 사용)
                class_labels = batch_labels
            else:
                # Fallback: diagonal matching
                class_labels = torch.arange(batch_size).to(text_embeddings.device)
        else:
            # Fallback: diagonal matching (기존 방식)
            class_labels = torch.arange(batch_size).to(text_embeddings.device)
        
        # Cosine similarity matrix
        similarity_matrix = torch.matmul(text_embeddings, vib_embeddings.t())  # (N, N)
        
        # 클래스 기반 positive/negative mask 생성
        class_labels = class_labels.to(text_embeddings.device)
        positive_mask = (class_labels.unsqueeze(1) == class_labels.unsqueeze(0))  # (N, N)
        
        # 클래스 기반 InfoNCE loss 계산 (학습 가능한 온도 사용)
        temp_text = torch.exp(self.log_temperature_text)
        temp_vib = torch.exp(self.log_temperature_vib)
        
        loss_text_to_vib = self._class_based_infonce_loss(
            similarity_matrix, positive_mask, temp_text
        )
        loss_vib_to_text = self._class_based_infonce_loss(
            similarity_matrix.t(), positive_mask.t(), temp_vib
        )
        
        # Total bidirectional loss
        total_loss = (loss_text_to_vib + loss_vib_to_text) / 2.0
        
        # Loss components for monitoring
        loss_components = {
            'text_to_vib': loss_text_to_vib,
            'vib_to_text': loss_vib_to_text,
            'total': total_loss
        }
        
        return total_loss, loss_components
    
    def update_temperatures(self, temperature_text: float, temperature_vib: float):
        """온도 파라미터 업데이트 (Continual learning에서 사용)"""
        with torch.no_grad():
            self.log_temperature_text.copy_(torch.log(torch.tensor(temperature_text)))
            self.log_temperature_vib.copy_(torch.log(torch.tensor(temperature_vib)))
        logger.info(f"Temperature 업데이트: τ_text={temperature_text:.3f}, τ_vib={temperature_vib:.3f}")
    
    @property
    def temperature_text(self):
        """현재 text temperature 값 반환"""
        return torch.exp(self.log_temperature_text).item()
    
    @property  
    def temperature_vib(self):
        """현재 vibration temperature 값 반환"""
        return torch.exp(self.log_temperature_vib).item()


class TextVibCLIP(nn.Module):
    """
    TextVibCLIP 메인 모델
    
    Text Encoder + Vibration Encoder + InfoNCE Loss
    First domain training 및 Continual learning 지원
    """
    
    def __init__(self,
                 domain_stage: str = 'first_domain',
                 embedding_dim: int = MODEL_CONFIG['embedding_dim']):
        """
        Args:
            domain_stage (str): 'first_domain' (Domain 1) 또는 'continual' (Domain 2+)
            embedding_dim (int): 임베딩 차원
        """
        super(TextVibCLIP, self).__init__()
        
        self.domain_stage = domain_stage
        self.embedding_dim = embedding_dim
        
        # Text Encoder 생성
        self.text_encoder = create_text_encoder(domain_stage)
        
        # Vibration Encoder 생성
        self.vibration_encoder = create_vibration_encoder()

        # 🎯 Cross-Modal Projection Layer (Residual stack 옵션)
        self.text_projection = self._build_projection(embedding_dim)
        self.vibration_projection = self._build_projection(embedding_dim)

        # Prototypes (class anchors) for shared semantic space
        proto_cfg = MODEL_CONFIG.get('prototypes', {})
        self.use_prototypes = bool(proto_cfg.get('enabled', False))
        self.prototype_tau = float(proto_cfg.get('tau', 0.1))
        self.prototype_lambda = float(proto_cfg.get('lambda_proto', 0.5))
        self.prototype_ema_m = float(proto_cfg.get('ema_momentum', 0.99))
        self.prototype_init_from_text = bool(proto_cfg.get('init_from_text', True))

        # 클래스 수는 UOS의 7을 기본으로 사용 (CWRU 4는 하위 부분집합으로 처리)
        aux_cfg = MODEL_CONFIG.get('aux_classification', {})
        self.num_classes = int(aux_cfg.get('num_classes', 7))

        if self.use_prototypes:
            # 학습 가능한 프로토타입 테이블 (K, D)
            proto = torch.randn(self.num_classes, embedding_dim) * (1.0 / (embedding_dim ** 0.5))
            self.prototypes = nn.Parameter(proto)
            # 지연 초기화/EMA 플래그 및 버퍼
            self.register_buffer('prototypes_initialized', torch.tensor(0, dtype=torch.uint8))
            self.register_buffer('prototype_counts', torch.zeros(self.num_classes, dtype=torch.long))

        # Bilinear similarity head (optional)
        sim_cfg = MODEL_CONFIG.get('similarity', {})
        self.use_bilinear = bool(sim_cfg.get('bilinear_enabled', False))
        self.lambda_bilinear = float(sim_cfg.get('lambda_bilinear', 0.0))
        if self.use_bilinear:
            self.bilinear_W = nn.Parameter(torch.eye(embedding_dim))
        
        # InfoNCE Loss 설정
        if domain_stage == 'first_domain':
            temp_text = MODEL_CONFIG['infonce']['first_domain_temperature_text']
            temp_vib = MODEL_CONFIG['infonce']['first_domain_temperature_vib']
        else:  # continual
            temp_text = MODEL_CONFIG['infonce']['continual_temperature_text']
            temp_vib = MODEL_CONFIG['infonce']['continual_temperature_vib']
        
        self.infonce_loss = InfoNCELoss(temp_text, temp_vib)
        
        # Continual learning 상태 관리
        self.is_continual_mode = (domain_stage == 'continual')
        
        logger.info(f"TextVibCLIP 초기화 완료: {domain_stage} stage")

    def _build_projection(self, embedding_dim: int) -> nn.Module:
        rp_cfg = MODEL_CONFIG.get('residual_projection', {'enabled': False})
        if not bool(rp_cfg.get('enabled', False)):
            return nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 2),
                nn.ReLU(),
                nn.Dropout(MODEL_CONFIG['projection']['dropout']),
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
        # Residual MLP Block (Pre-LN)
        class ResidualMLP(nn.Module):
            def __init__(self, dim: int, ffn_mult: int = 4, dropout: float = 0.1):
                super().__init__()
                hidden = dim * ffn_mult
                self.norm = nn.LayerNorm(dim)
                self.fc1 = nn.Linear(dim, hidden)
                self.act = nn.GELU()
                self.dropout = nn.Dropout(dropout)
                self.fc2 = nn.Linear(hidden, dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.norm(x)
                h = self.fc1(h)
                h = self.act(h)
                h = self.dropout(h)
                h = self.fc2(h)
                return x + h

        layers = []
        num_layers = int(rp_cfg.get('num_layers', 3))
        ffn_mult = int(rp_cfg.get('ffn_mult', 4))
        dropout = float(rp_cfg.get('dropout', 0.1))
        # 입력 정규화 + 선형 매핑으로 진입
        layers.append(nn.LayerNorm(embedding_dim))
        layers.append(nn.Identity())  # 자리표시자(형태 유지)
        # Residual blocks
        for _ in range(num_layers):
            layers.append(ResidualMLP(embedding_dim, ffn_mult=ffn_mult, dropout=dropout))
        # 최종 정규화
        layers.append(nn.LayerNorm(embedding_dim))
        return nn.Sequential(*layers)
    
    def forward(self, 
                batch: Dict[str, Union[torch.Tensor, List[str]]],
                return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            batch: 배치 데이터 딕셔너리
                - 'vibration': 진동 신호 (batch_size, input_length)
                - 'text': 텍스트 리스트 (batch_size,)
            return_embeddings: 임베딩도 반환할지 여부
            
        Returns:
            Dict with loss and optionally embeddings
        """
        vibration_signals = batch['vibration']
        texts = batch['text']
        
        device = vibration_signals.device
        
        # Vibration encoding
        vib_embeddings = self.vibration_encoder(vibration_signals)
        
        # Text encoding
        if 'input_ids' in batch and 'attention_mask' in batch:
            # 이미 토크나이징된 데이터 사용 (배치 효율성)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            text_embeddings = self.text_encoder.encode_tokenized(input_ids, attention_mask)
        else:
            # 텍스트 리스트 직접 토크나이징 (하위 호환성)
            text_embeddings = self.text_encoder.encode_texts(texts, device, max_length=128)
        
        # 🎯 CRITICAL FIX: Cross-Modal Projection 적용
        text_embeddings = F.normalize(self.text_projection(text_embeddings), p=2, dim=1)
        vib_embeddings = F.normalize(self.vibration_projection(vib_embeddings), p=2, dim=1)
        
        # 🎯 FIXED: 표준 contrastive learning (diagonal pairs only)
        # 각 text-vibration 쌍은 배치 내에서 대각선 위치에서만 매칭
        # 멀티-포지티브 로직 제거 (연구 의도에 맞지 않음)
        multi_positive = None  # 항상 None으로 설정

        # Replay 임베딩이 제공되면 분모(negative pool)를 확장하여 InfoNCE 계산
        replay_text = batch.get('replay_text_embeddings', None)
        replay_vib = batch.get('replay_vib_embeddings', None)
        
        if replay_text is not None and replay_vib is not None and replay_text.numel() > 0 and replay_vib.numel() > 0:
            # 🎯 FIXED: 클래스 기반 Replay InfoNCE
            text_norm = F.normalize(text_embeddings, p=2, dim=1)
            vib_norm = F.normalize(vib_embeddings, p=2, dim=1)
            replay_text_norm = F.normalize(replay_text.detach(), p=2, dim=1)
            replay_vib_norm = F.normalize(replay_vib.detach(), p=2, dim=1)
            
            # 현재 배치 + Replay 배치 결합
            all_text = torch.cat([text_norm, replay_text_norm], dim=0)  # (N+R, d)
            all_vib = torch.cat([vib_norm, replay_vib_norm], dim=0)  # (N+R, d)
            
            # 클래스 라벨 결합 (현재 + replay)
            batch_labels = batch.get('labels', None)
            replay_labels = batch.get('replay_labels', None)
            
            if batch_labels is not None and replay_labels is not None:
                # 현재 배치 라벨 처리
                if batch_labels.dim() == 2:
                    if batch_labels.size(1) == 2:
                        current_classes = batch_labels[:, 0]  # UOS 주 분류
                    else:
                        current_classes = batch_labels[:, 0]  # CWRU (batch_size, 1)
                else:
                    current_classes = batch_labels  # 1차원 라벨
                
                # Replay 라벨 처리
                if replay_labels.dim() == 2:
                    if replay_labels.size(1) == 2:
                        replay_classes = replay_labels[:, 0]  # UOS
                    else:
                        replay_classes = replay_labels[:, 0]  # CWRU (batch_size, 1)
                else:
                    replay_classes = replay_labels  # 1차원 라벨
                
                all_classes = torch.cat([current_classes, replay_classes], dim=0)
                
                # 클래스 기반 positive mask
                positive_mask = (all_classes.unsqueeze(1) == all_classes.unsqueeze(0))
                
                # Text->Vib with replay
                sim_t2v = torch.matmul(text_norm, all_vib.t())
                loss_t2v = self.infonce_loss._class_based_infonce_loss(
                    sim_t2v, positive_mask[:text_norm.size(0)], self.infonce_loss.temperature_text
                )
                
                # Vib->Text with replay  
                sim_v2t = torch.matmul(vib_norm, all_text.t())
                loss_v2t = self.infonce_loss._class_based_infonce_loss(
                    sim_v2t, positive_mask[:vib_norm.size(0)], self.infonce_loss.temperature_vib
                )
                
                loss = (loss_t2v + loss_v2t) / 2.0
                loss_components = {
                    'text_to_vib': loss_t2v,
                    'vib_to_text': loss_v2t,
                    'total': loss
                }
            else:
                # 라벨 정보 없으면 기존 diagonal 방식
                batch_size = text_norm.size(0)
                labels = torch.arange(batch_size, device=device)
                all_vib = torch.cat([vib_norm, replay_vib_norm], dim=0)
                all_text = torch.cat([text_norm, replay_text_norm], dim=0)
                
                logits_t2v = torch.matmul(text_norm, all_vib.t()) / self.infonce_loss.temperature_text
                logits_v2t = torch.matmul(vib_norm, all_text.t()) / self.infonce_loss.temperature_vib
                
                loss_t2v = F.cross_entropy(logits_t2v, labels, reduction=self.infonce_loss.reduction)
                loss_v2t = F.cross_entropy(logits_v2t, labels, reduction=self.infonce_loss.reduction)
                
                loss = (loss_t2v + loss_v2t) / 2.0
                loss_components = {
                    'text_to_vib': loss_t2v,
                    'vib_to_text': loss_v2t,
                    'total': loss
                }
        else:
            # 표준 또는 멀티-포지티브 InfoNCE
            if multi_positive is None:
                # 🎯 라벨 정보를 InfoNCE에 전달
                batch_labels = batch.get('labels', None)
                loss, loss_components = self.infonce_loss(text_embeddings, vib_embeddings, batch_labels)
            else:
                # 🎯 FIXED: 표준 L2 정규화 (gradient 보존)
                text_norm = F.normalize(text_embeddings, p=2, dim=1)
                vib_norm = F.normalize(vib_embeddings, p=2, dim=1)
                logits_t2v = torch.matmul(text_norm, vib_norm.t()) / self.infonce_loss.temperature_text
                logits_v2t = torch.matmul(vib_norm, text_norm.t()) / self.infonce_loss.temperature_vib
                pos_t2v = multi_positive
                pos_v2t = multi_positive
                loss_t2v = self._multi_positive_infonce_loss(logits_t2v, pos_t2v)
                loss_v2t = self._multi_positive_infonce_loss(logits_v2t, pos_v2t)
                loss = (loss_t2v + loss_v2t) / 2.0
                loss_components = {
                    'text_to_vib': loss_t2v,
                    'vib_to_text': loss_v2t,
                    'total': loss
                }

        # Bilinear similarity auxiliary loss (optional)
        if self.use_bilinear and self.lambda_bilinear > 0.0:
            # s_ij = t_i^T W v_j, 멀티-포지티브 마스크 기반 InfoNCE와 동일한 형태로 사용
            batch_labels = batch.get('labels', None)
            t = F.normalize(text_embeddings, p=2, dim=1)
            v = F.normalize(vib_embeddings, p=2, dim=1)
            logits_bi = torch.matmul(torch.matmul(t, self.bilinear_W), v.t())
            if batch_labels is not None:
                if batch_labels.dim() == 2:
                    cls = batch_labels[:, 0]
                elif batch_labels.dim() == 1:
                    cls = batch_labels
                else:
                    cls = torch.arange(t.size(0), device=t.device)
                pos_mask = (cls.unsqueeze(1) == cls.unsqueeze(0))
            else:
                # 대각선만 포지티브
                pos_mask = torch.eye(t.size(0), dtype=torch.bool, device=t.device)
            # 온도는 text 방향 것을 사용
            bi_loss = self.infonce_loss._class_based_infonce_loss(logits_bi, pos_mask, self.infonce_loss.temperature_text)
            loss = loss + self.lambda_bilinear * bi_loss
            loss_components['bilinear'] = bi_loss

        # Prototype alignment loss (optional)
        if self.use_prototypes:
            proto_loss = None
            labels_for_proto = batch.get('labels', None)
            if labels_for_proto is not None:
                # 라벨 정규화: UOS (B,2) => 첫번째 열, CWRU (B,1) => 열 0, 1D => 그대로
                if labels_for_proto.dim() == 2:
                    class_labels = labels_for_proto[:, 0]
                elif labels_for_proto.dim() == 1:
                    class_labels = labels_for_proto
                else:
                    class_labels = None

                if class_labels is not None:
                    class_labels = class_labels.clamp(min=0, max=self.num_classes - 1)
                    # 프로토타입 정규화 후 로짓 계산
                    proto_norm = F.normalize(self.prototypes, p=2, dim=1)
                    logits_text_proto = torch.matmul(text_embeddings, proto_norm.t()) / self.prototype_tau
                    logits_vib_proto = torch.matmul(vib_embeddings, proto_norm.t()) / self.prototype_tau
                    ce_text = F.cross_entropy(logits_text_proto, class_labels)
                    ce_vib = F.cross_entropy(logits_vib_proto, class_labels)
                    proto_loss = (ce_text + ce_vib) * 0.5
                    loss = loss + self.prototype_lambda * proto_loss
                    loss_components['proto_ce'] = proto_loss

                    # 초기화(텍스트 기반 평균) 및 EMA 업데이트
                    with torch.no_grad():
                        # 배치 클래스별 평균 (텍스트/진동 평균을 함께 사용)
                        unique_classes = class_labels.unique()
                        for cls in unique_classes.tolist():
                            mask = (class_labels == cls)
                            if mask.any():
                                mean_t = text_embeddings[mask].mean(dim=0)
                                mean_v = vib_embeddings[mask].mean(dim=0)
                                mean_tv = F.normalize(0.5 * (mean_t + mean_v), p=2, dim=0)

                                if (self.prototypes_initialized.item() == 0) and self.prototype_init_from_text:
                                    # 최초 한 번 텍스트 기반으로 더 강하게 초기화
                                    self.prototypes.data[cls] = mean_tv
                                else:
                                    # EMA 업데이트
                                    current = F.normalize(self.prototypes.data[cls], p=2, dim=0)
                                    updated = self.prototype_ema_m * current + (1.0 - self.prototype_ema_m) * mean_tv
                                    self.prototypes.data[cls] = F.normalize(updated, p=2, dim=0)

                        if self.prototypes_initialized.item() == 0:
                            self.prototypes_initialized.fill_(1)

        # 🎯 CRITICAL FIX: CWRU 전용 강화된 Auxiliary Classification
        aux_cfg = MODEL_CONFIG.get('aux_classification', {'enabled': False})
        if aux_cfg.get('enabled', False):
            aux_labels = batch.get('labels', None)
            if aux_labels is not None:
                if aux_labels.dim() == 2 and aux_labels.size(1) >= 1:
                    # UOS: 첫 번째가 주 분류 (7-클래스)
                    main_class = aux_labels[:, 0]
                    if hasattr(self.vibration_encoder, 'use_aux_head') and self.vibration_encoder.use_aux_head:
                        logits_cls = self.vibration_encoder.aux_head(vib_embeddings)
                        ce_loss = F.cross_entropy(logits_cls, main_class)
                        loss = loss + float(aux_cfg.get('loss_weight', 1.0)) * ce_loss
                        loss_components['aux_ce'] = ce_loss
                elif aux_labels.dim() == 1:
                    # 🎯 CWRU: 강화된 직접 분류 (contrastive learning 보완)
                    if hasattr(self.vibration_encoder, 'use_aux_head') and self.vibration_encoder.use_aux_head:
                        logits_cls = self.vibration_encoder.aux_head(vib_embeddings)
                        ce_loss = F.cross_entropy(logits_cls, aux_labels)
                        
                        # CWRU에서는 auxiliary loss를 주요 loss로 강화
                        aux_weight = float(aux_cfg.get('loss_weight', 3.0))
                        if hasattr(batch, 'get') and 'metadata' in batch:
                            # CWRU 데이터인지 확인
                            metadata_sample = batch['metadata'][0] if batch['metadata'] else {}
                            if metadata_sample.get('dataset_type') == 'cwru':
                                aux_weight = 10.0  # CWRU에서는 10배 강화
                        
                        loss = loss + aux_weight * ce_loss
                        loss_components['aux_ce'] = ce_loss
                        loss_components['aux_weight'] = aux_weight
        
        # 결과 딕셔너리 구성
        results = {
            'loss': loss,
            'loss_components': loss_components
        }
        
        if return_embeddings:
            results.update({
                'text_embeddings': text_embeddings,
                'vib_embeddings': vib_embeddings
            })
        
        return results

    def _multi_positive_infonce_loss(self, logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        """멀티-포지티브 InfoNCE 손실
        Args:
            logits: (N, M) 유사도 로짓
            positive_mask: (N, M) bool 텐서, 양성 위치 True
        Returns:
            스칼라 손실 (mean)
        """
        # AMP(half)에서의 overflow 회피: 연산을 float32로 수행
        logits_f32 = logits.float()
        # 정규화를 위한 log-sum-exp (분모)
        log_denom = torch.logsumexp(logits_f32, dim=1)  # (N,)
        # 양성 로짓만 남기고 log-sum-exp (분자)
        # 모든 행에 최소 하나 이상의 양성이 존재한다고 가정 (자기 자신 포함)
        masked_logits = logits_f32.masked_fill(~positive_mask, -1e4)
        log_num = torch.logsumexp(masked_logits, dim=1)  # (N,)
        loss_vec = -(log_num - log_denom)
        return loss_vec.mean()
    
    def encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """텍스트만 인코딩 (추론용)"""
        return self.text_encoder.encode_texts(texts, device)
    
    def encode_vibration(self, vibration_signals: torch.Tensor) -> torch.Tensor:
        """진동 신호만 인코딩 (추론용)"""
        return self.vibration_encoder(vibration_signals)
    
    def switch_to_continual_mode(self):
        """Continual learning 모드로 전환"""
        self.is_continual_mode = True
        
        # 🎯 CRITICAL FIX: Text encoder 부분 freeze (완전 freeze 문제 해결)
        # LoRA는 freeze하되, projection layer는 학습 가능하게 유지
        self.text_encoder.disable_lora_training()
        
        # Projection layer는 학습 가능하게 유지 (최소한의 adaptation)
        if hasattr(self.text_encoder, 'projection'):
            for param in self.text_encoder.projection.parameters():
                param.requires_grad = True  # False → True (학습 가능)
        
        # InfoNCE temperature 업데이트
        temp_text = MODEL_CONFIG['infonce']['continual_temperature_text']
        temp_vib = MODEL_CONFIG['infonce']['continual_temperature_vib']
        self.infonce_loss.update_temperatures(temp_text, temp_vib)
        
        logger.info("Continual learning 모드로 전환 완료 (LoRA freeze, Projection 학습 가능)")
    
    def switch_to_first_domain_mode(self):
        """First domain training 모드로 전환"""
        self.is_continual_mode = False
        
        # Text encoder LoRA 학습 활성화
        self.text_encoder.enable_lora_training()
        # LoRA 파라미터 개수/학습 가능 상태 로깅
        try:
            lora_params_total = 0
            lora_params_trainable = 0
            for name, p in self.text_encoder.distilbert.named_parameters():
                if 'lora_' in name:
                    lora_params_total += p.numel()
                    if p.requires_grad:
                        lora_params_trainable += p.numel()
            logger.info(
                f"LoRA 파라미터 상태: total={lora_params_total:,}, trainable={lora_params_trainable:,}"
            )
        except Exception as e:
            logger.info(f"LoRA 파라미터 상태 확인 스킵: {e}")
        
        # Projection layer 재활성화
        if hasattr(self.text_encoder, 'projection'):
            for param in self.text_encoder.projection.parameters():
                param.requires_grad = True
        
        # InfoNCE temperature 업데이트
        temp_text = MODEL_CONFIG['infonce']['first_domain_temperature_text']
        temp_vib = MODEL_CONFIG['infonce']['first_domain_temperature_vib']
        self.infonce_loss.update_temperatures(temp_text, temp_vib)
        
        logger.info("First domain training 모드로 전환 완료 (TextEncoder 활성화)")
    
    # 하위 호환성을 위한 별칭
    def switch_to_joint_mode(self):
        """하위 호환성을 위한 별칭"""
        return self.switch_to_first_domain_mode()
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """각 컴포넌트별 학습 가능한 파라미터 수 반환"""
        text_params = self.text_encoder.get_trainable_parameters()
        text_lora_params = self.text_encoder.get_lora_parameters()
        vib_params = self.vibration_encoder.get_trainable_parameters()
        
        return {
            'text_total': text_params,
            'text_lora': text_lora_params,
            'vibration': vib_params,
            'total': text_params + vib_params
        }
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Optional[Dict] = None):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'domain_stage': self.domain_stage,
            'embedding_dim': self.embedding_dim,
            'is_continual_mode': self.is_continual_mode
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, path)
        logger.info(f"체크포인트 저장 완료: {path}")
    
    def load_checkpoint(self, path: str, device: torch.device) -> Dict:
        """체크포인트 로딩"""
        checkpoint = torch.load(path, map_location=device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # 모드 복원
        if checkpoint.get('is_continual_mode', False):
            self.switch_to_continual_mode()
        else:
            self.switch_to_first_domain_mode()
        
        logger.info(f"체크포인트 로딩 완료: {path}")
        
        return checkpoint


def create_textvib_model(domain_stage: str = 'first_domain') -> TextVibCLIP:
    """
    도메인 단계에 따른 TextVibCLIP 모델 생성
    
    Args:
        domain_stage (str): 'first_domain' (Domain 1) 또는 'continual' (Domain 2+)
        
    Returns:
        TextVibCLIP: 설정된 모델
    """
    model = TextVibCLIP(domain_stage=domain_stage)
    
    # 파라미터 정보 출력
    param_info = model.get_trainable_parameters()
    logger.info(f"TextVibCLIP 생성 완료: {domain_stage} stage")
    logger.info(f"Text encoder 파라미터: {param_info['text_total']:,} "
               f"(LoRA: {param_info['text_lora']:,})")
    logger.info(f"Vibration encoder 파라미터: {param_info['vibration']:,}")
    logger.info(f"총 학습 가능한 파라미터: {param_info['total']:,}")
    
    return model


def compute_similarity_scores(text_embeddings: torch.Tensor, 
                            vib_embeddings: torch.Tensor) -> torch.Tensor:
    """
    텍스트와 진동 임베딩 간 유사도 점수 계산 (추론용)
    
    Args:
        text_embeddings: (num_texts, embedding_dim)
        vib_embeddings: (num_vibs, embedding_dim)
        
    Returns:
        torch.Tensor: 유사도 행렬 (num_texts, num_vibs)
    """
    # 🎯 FIXED: 표준 L2 정규화 (gradient 보존)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    vib_embeddings = F.normalize(vib_embeddings, p=2, dim=1)
    
    # Cosine similarity
    similarity_matrix = torch.matmul(text_embeddings, vib_embeddings.t())
    
    return similarity_matrix


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    print("=== TextVibCLIP 테스트 ===")
    
    # First domain training 모델 테스트
    print("\n1. First Domain Training 모델")
    model_first = create_textvib_model('first_domain')
    
    # 테스트 데이터 생성
    batch_size = 4
    input_length = MODEL_CONFIG['vibration_encoder']['input_length']
    
    test_batch = {
        'vibration': torch.randn(batch_size, input_length),
        'text': [
            "A deep groove ball bearing operating at 600 rpm with healthy rotating component and ball fault.",
            "A tapered roller bearing operating at 800 rpm with healthy rotating component and healthy bearing.",
            "A cylindrical roller bearing operating at 1000 rpm with unbalanced rotating component and inner race fault.",
            "A deep groove ball bearing operating at 1200 rpm with healthy rotating component and outer race fault."
        ]
    }
    
    # GPU 사용 가능하면 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_first.to(device)
    test_batch['vibration'] = test_batch['vibration'].to(device)
    
    # Forward pass 테스트
    results = model_first(test_batch, return_embeddings=True)
    
    print(f"Loss: {results['loss'].item():.4f}")
    print(f"Text embeddings shape: {results['text_embeddings'].shape}")
    print(f"Vibration embeddings shape: {results['vib_embeddings'].shape}")
    
    # Continual learning 모드 전환 테스트
    print("\n2. Continual Learning 모드 전환")
    model_first.switch_to_continual_mode()
    
    results_continual = model_first(test_batch, return_embeddings=True)
    print(f"Continual mode loss: {results_continual['loss'].item():.4f}")
    
    # 유사도 계산 테스트
    print("\n3. 유사도 계산 테스트")
    similarity_matrix = compute_similarity_scores(
        results['text_embeddings'], 
        results['vib_embeddings']
    )
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Diagonal (positive pairs): {torch.diag(similarity_matrix)}")
    
    print("\n=== TextVibCLIP 테스트 완료 ===")
