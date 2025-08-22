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
        self.temperature_text = temperature_text
        self.temperature_vib = temperature_vib
        self.reduction = reduction
        
        logger.info(f"InfoNCE Loss 초기화: τ_text={temperature_text:.3f}, τ_vib={temperature_vib:.3f}")
    
    def forward(self, 
                text_embeddings: torch.Tensor, 
                vib_embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Bidirectional InfoNCE loss 계산
        
        Args:
            text_embeddings: (batch_size, embedding_dim)
            vib_embeddings: (batch_size, embedding_dim)
            
        Returns:
            Tuple[torch.Tensor, Dict]: (total_loss, loss_components)
        """
        batch_size = text_embeddings.size(0)
        
        # L2 normalization
        text_embeddings = F.normalize(text_embeddings, dim=1)
        vib_embeddings = F.normalize(vib_embeddings, dim=1)
        
        # Cosine similarity matrix
        similarity_matrix = torch.matmul(text_embeddings, vib_embeddings.t())  # (batch_size, batch_size)
        
        # Positive pairs are on the diagonal
        labels = torch.arange(batch_size).to(text_embeddings.device) 
 
        # Text-to-vibration direction
        # 의미: "텍스트가 쿼리, 진동이 답변"
        logits_text_to_vib = similarity_matrix / self.temperature_text
        loss_text_to_vib = F.cross_entropy(logits_text_to_vib, labels, reduction=self.reduction)
        
        # Vibration-to-text direction  
        # 의미: "진동이 쿼리, 텍스트가 답변"
        logits_vib_to_text = similarity_matrix.t() / self.temperature_vib
        loss_vib_to_text = F.cross_entropy(logits_vib_to_text, labels, reduction=self.reduction)
        
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
        self.temperature_text = temperature_text
        self.temperature_vib = temperature_vib
        logger.info(f"Temperature 업데이트: τ_text={temperature_text:.3f}, τ_vib={temperature_vib:.3f}")


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
        
        # Replay 임베딩이 제공되면 분모(negative pool)를 확장하여 InfoNCE 계산
        replay_text = batch.get('replay_text_embeddings', None)
        replay_vib = batch.get('replay_vib_embeddings', None)
        
        if replay_text is not None and replay_vib is not None and replay_text.numel() > 0 and replay_vib.numel() > 0:
            # 정규화
            text_norm = F.normalize(text_embeddings, dim=1)
            vib_norm = F.normalize(vib_embeddings, dim=1)
            replay_text_norm = F.normalize(replay_text.detach(), dim=1)
            replay_vib_norm = F.normalize(replay_vib.detach(), dim=1)
            
            batch_size = text_norm.size(0)
            labels = torch.arange(batch_size, device=device)
            
            # Text->Vib: 현재 텍스트를 쿼리로, 현재 진동(정답) + 리플레이 진동(모두 negative)으로 분모 확장
            all_vib = torch.cat([vib_norm, replay_vib_norm], dim=0)  # (N+R, d)
            logits_t2v = torch.matmul(text_norm, all_vib.t()) / self.infonce_loss.temperature_text  # (N, N+R)
            loss_t2v = F.cross_entropy(logits_t2v, labels, reduction=self.infonce_loss.reduction)
            
            # Vib->Text: 현재 진동을 쿼리로, 현재 텍스트(정답) + 리플레이 텍스트(모두 negative)
            all_text = torch.cat([text_norm, replay_text_norm], dim=0)  # (N+R, d)
            logits_v2t = torch.matmul(vib_norm, all_text.t()) / self.infonce_loss.temperature_vib  # (N, N+R)
            loss_v2t = F.cross_entropy(logits_v2t, labels, reduction=self.infonce_loss.reduction)
            
            total_loss = (loss_t2v + loss_v2t) / 2.0
            loss_components = {
                'text_to_vib': loss_t2v,
                'vib_to_text': loss_v2t,
                'total': total_loss
            }
        else:
            # 표준 InfoNCE
            loss, loss_components = self.infonce_loss(text_embeddings, vib_embeddings)
        
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
    
    def encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """텍스트만 인코딩 (추론용)"""
        return self.text_encoder.encode_texts(texts, device)
    
    def encode_vibration(self, vibration_signals: torch.Tensor) -> torch.Tensor:
        """진동 신호만 인코딩 (추론용)"""
        return self.vibration_encoder(vibration_signals)
    
    def switch_to_continual_mode(self):
        """Continual learning 모드로 전환"""
        self.is_continual_mode = True
        
        # Text encoder 완전 freeze (LoRA + projection 모두)
        self.text_encoder.disable_lora_training()
        
        # Projection layer도 완전히 freeze
        if hasattr(self.text_encoder, 'projection'):
            for param in self.text_encoder.projection.parameters():
                param.requires_grad = False
        
        # InfoNCE temperature 업데이트
        temp_text = MODEL_CONFIG['infonce']['continual_temperature_text']
        temp_vib = MODEL_CONFIG['infonce']['continual_temperature_vib']
        self.infonce_loss.update_temperatures(temp_text, temp_vib)
        
        logger.info("Continual learning 모드로 전환 완료 (TextEncoder 완전 freeze)")
    
    def switch_to_first_domain_mode(self):
        """First domain training 모드로 전환"""
        self.is_continual_mode = False
        
        # Text encoder LoRA 학습 활성화
        self.text_encoder.enable_lora_training()
        
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
    # L2 normalization
    text_embeddings = F.normalize(text_embeddings, dim=1)
    vib_embeddings = F.normalize(vib_embeddings, dim=1)
    
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
