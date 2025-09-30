"""
TextVibCLIP v2: Ranking-based 멀티모달 베어링 진단 모델
InfoNCE 대신 Triplet/Margin Ranking Loss 사용으로 소규모 데이터에 최적화

핵심 아이디어:
1. 각 인코더가 독립적으로 분류 학습 (안정적)
2. 간단한 정렬 학습 (MSE 기반)
3. 실제 사용: 진동 신호 → 후보 텍스트 중 최고 유사도 선택
4. Continual learning: 진동 위주 적응
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import logging

from .text_encoder import create_text_encoder
from .vibration_encoder import create_vibration_encoder
from configs.model_config import MODEL_CONFIG, FIRST_DOMAIN_CONFIG, CONTINUAL_CONFIG

logger = logging.getLogger(__name__)


class RankingLoss(nn.Module):
    """
    Ranking-based Loss for Text-Vibration Alignment
    
    핵심: 같은 클래스의 text-vib는 가깝게, 다른 클래스는 멀게
    InfoNCE보다 소규모 데이터에 적합
    """
    
    def __init__(self, margin: float = 0.2, loss_type: str = 'triplet'):
        """
        Args:
            margin: Ranking margin
            loss_type: 'triplet' 또는 'margin_ranking'
        """
        super().__init__()
        self.margin = margin
        self.loss_type = loss_type
        logger.info(f"RankingLoss 초기화: margin={margin}, type={loss_type}")
    
    def forward(self, text_embeddings: torch.Tensor, 
                vib_embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Ranking loss 계산
        
        Args:
            text_embeddings: (batch_size, embed_dim)
            vib_embeddings: (batch_size, embed_dim)  
            labels: (batch_size,) 클래스 라벨
            
        Returns:
            torch.Tensor: ranking loss
        """
        batch_size = text_embeddings.size(0)
        
        if self.loss_type == 'triplet':
            return self._triplet_loss(text_embeddings, vib_embeddings, labels)
        else:
            return self._margin_ranking_loss(text_embeddings, vib_embeddings, labels)
    
    def _triplet_loss(self, text_emb: torch.Tensor, vib_emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Triplet Loss: anchor-positive-negative"""
        # 모든 쌍의 유사도 계산
        similarities = torch.matmul(vib_emb, text_emb.t())  # (B, B)
        
        # 같은 클래스 마스크
        same_class = (labels.unsqueeze(1) == labels.unsqueeze(0))
        
        losses = []
        for i in range(similarities.size(0)):
            # Positive: 같은 클래스 (자기 제외)
            positive_mask = same_class[i] & (torch.arange(similarities.size(1), device=similarities.device) != i)
            # Negative: 다른 클래스
            negative_mask = ~same_class[i]
            
            if positive_mask.any() and negative_mask.any():
                # 가장 가까운 positive와 가장 가까운 negative
                pos_sim = similarities[i][positive_mask].max()
                neg_sim = similarities[i][negative_mask].max()
                
                # Triplet loss: positive가 negative보다 margin만큼 더 높아야 함
                loss = F.relu(self.margin - pos_sim + neg_sim)
                losses.append(loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=similarities.device)
    
    def _margin_ranking_loss(self, text_emb: torch.Tensor, vib_emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Margin Ranking Loss (더 간단한 버전)"""
        similarities = torch.matmul(vib_emb, text_emb.t())
        
        # 정답 위치 (대각선)
        correct_sim = torch.diag(similarities)
        
        # 오답들과의 차이
        batch_size = similarities.size(0)
        losses = []
        
        for i in range(batch_size):
            correct = correct_sim[i]
            incorrect = similarities[i]
            incorrect = torch.cat([incorrect[:i], incorrect[i+1:]])  # 자기 제외
            
            # 모든 오답보다 margin만큼 높아야 함
            margin_losses = F.relu(self.margin - correct + incorrect)
            losses.append(margin_losses.mean())
        
        return torch.stack(losses).mean()


class TextVibCLIP(nn.Module):
    """
    TextVibCLIP v2: Ranking-based 아키텍처
    
    InfoNCE 대신 Triplet/Ranking Loss 사용
    실제 사용: 진동 신호 → 후보 텍스트 중 최고 유사도 선택
    """
    
    def __init__(self,
                 domain_stage: str = 'first_domain',
                 embedding_dim: int = 256,
                 dataset_type: str = 'uos'):
        super().__init__()
        
        self.domain_stage = domain_stage
        self.embedding_dim = embedding_dim
        self.dataset_type = dataset_type.lower()
        self.is_continual_mode = (domain_stage == 'continual')
        
        # 기존 인코더들 재활용
        self.text_encoder = create_text_encoder(domain_stage)
        self.vib_encoder = create_vibration_encoder()
        
        # 🎯 핵심: 공통 임베딩 공간으로의 projection
        self.text_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        self.vib_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # 🎯 Ranking Loss (InfoNCE 대신)
        self.ranking_loss = RankingLoss(margin=0.3, loss_type='triplet')
        
        # 🎯 앙상블 가중치 (추론 시 사용)
        self.ensemble_weight = nn.Parameter(torch.tensor(0.7))  # 진동 위주
        
        # 🎯 보조 분류 헤드 (데이터셋별 클래스 수)
        aux_cfg = MODEL_CONFIG.get('aux_classification', {})
        self.use_aux = aux_cfg.get('enabled', True)
        if self.use_aux:
            # 데이터셋별 클래스 수 설정
            if self.dataset_type == 'cwru':
                num_classes = 4  # CWRU: H, B, IR, OR
            else:
                num_classes = 7  # UOS: H, B, IR, OR, L, U, M
            
            # 데이터셋별 차별화된 분류기 구조
            if self.dataset_type == 'cwru':
                # CWRU: 매우 강한 정규화
                dropout_rate = 0.7
                hidden_dim = embedding_dim // 4  # 더 작은 hidden
            else:
                # UOS: 표준 정규화
                dropout_rate = 0.2
                hidden_dim = embedding_dim // 2
            
            self.text_classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(hidden_dim, num_classes)
            )
            
            self.vib_classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(embedding_dim, hidden_dim), 
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(hidden_dim, num_classes)
            )
        
        logger.info(f"TextVibCLIP v2 초기화 완료: {domain_stage} stage")
    
    def forward(self, 
                batch: Dict[str, Union[torch.Tensor, List[str]]],
                return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            batch: 배치 데이터
            return_embeddings: 임베딩 반환 여부
            
        Returns:
            Dict with loss and optionally embeddings
        """
        device = batch['vibration'].device
        
        # 인코딩
        # CWRU는 평가 시 고정 프롬프트(HP 무관) 기반 retrieval을 사용하므로,
        # 학습도 동일한 클래스 프롬프트(HP 접미사 미포함)로 정렬한다.
        # 라벨 기반 프롬프트는 학습시에만 사용(평가 시 라벨 누수 방지)
        use_prompt_training = (self.dataset_type == 'cwru') and self.training and ('labels' in batch)
        if use_prompt_training:
            # 라벨에서 클래스 프롬프트 생성 (영문, 동의어 템플릿 중 대표 문구)
            # bearing_condition_map 역매핑
            # 0:H, 1:B, 2:IR, 3:OR
            label_tensor = batch['labels']
            if label_tensor.dim() == 2:
                label_tensor = label_tensor[:, 0]

            prompt_map = {
                0: "healthy bearing",
                1: "bearing with ball fault",
                2: "bearing inner race fault",
                3: "bearing outer race fault"
            }
            prompt_texts = [prompt_map.get(int(c.item()), "healthy bearing") for c in label_tensor]
            text_raw = self.text_encoder.encode_texts(prompt_texts, device)
        else:
            if 'input_ids' in batch and 'attention_mask' in batch:
                text_raw = self.text_encoder.encode_tokenized(
                    batch['input_ids'].to(device), 
                    batch['attention_mask'].to(device)
                )
            else:
                text_raw = self.text_encoder.encode_texts(batch['text'], device)
        
        vib_raw = self.vib_encoder(batch['vibration'])
        
        # 공통 임베딩 공간으로 projection
        text_emb = F.normalize(self.text_projection(text_raw), p=2, dim=1)
        vib_emb = F.normalize(self.vib_projection(vib_raw), p=2, dim=1)
        
        # 라벨 처리
        labels = batch.get('labels', None)
        if labels is not None:
            if labels.dim() == 2:
                class_labels = labels[:, 0]  # UOS 주 분류, CWRU 첫 번째
            else:
                class_labels = labels
        else:
            # 라벨 없으면 대각선 매칭 가정
            class_labels = torch.arange(text_emb.size(0), device=device)
        
        # 🎯 핵심: Ranking Loss (InfoNCE 대신)
        ranking_loss = self.ranking_loss(text_emb, vib_emb, class_labels)
        
        total_loss = ranking_loss
        loss_components = {'ranking': ranking_loss}
        
        # 🎯 보조 분류 손실 (간단한 버전)
        if self.use_aux and labels is not None:
            # 도메인별 차별화된 가중치
            if self.is_continual_mode:
                aux_weight = CONTINUAL_CONFIG.get('aux_weight', 2.0)
            else:
                aux_weight = FIRST_DOMAIN_CONFIG.get('aux_weight', 5.0)
            
            # 텍스트 분류 (가중치 없음 - 간단하게)
            text_logits = self.text_classifier(text_raw)
            text_ce = F.cross_entropy(text_logits, class_labels)
            
            # 진동 분류 (가중치 없음 - 간단하게)
            vib_logits = self.vib_classifier(vib_raw)
            vib_ce = F.cross_entropy(vib_logits, class_labels)
            
            aux_loss = (text_ce + vib_ce) / 2.0
            total_loss += aux_weight * aux_loss
            
            loss_components['aux_text'] = text_ce
            loss_components['aux_vib'] = vib_ce
            loss_components['aux_total'] = aux_loss
            loss_components['aux_weight'] = aux_weight
        
        # 결과 구성
        results = {
            'loss': total_loss,
            'loss_components': loss_components
        }
        
        if return_embeddings:
            results.update({
                'text_embeddings': text_emb,
                'vib_embeddings': vib_emb,
                'text_raw': text_raw,
                'vib_raw': vib_raw
            })
        
        return results

    def predict_best_match(self, 
                          vibration_signal: torch.Tensor,
                          candidate_texts: List[str],
                          device: torch.device) -> Tuple[int, float]:
        """
        실제 사용: 진동 신호에 가장 맞는 텍스트 찾기
        
        Args:
            vibration_signal: (1, signal_length) 또는 (signal_length,)
            candidate_texts: 후보 텍스트 리스트
            device: 디바이스
            
        Returns:
            Tuple[int, float]: (best_text_index, confidence_score)
        """
        self.eval()
        
        with torch.no_grad():
            # 진동 신호 처리
            if vibration_signal.dim() == 1:
                vibration_signal = vibration_signal.unsqueeze(0)
            vibration_signal = vibration_signal.to(device)
            
            # 임베딩 생성
            vib_raw = self.vib_encoder(vibration_signal)
            vib_emb = F.normalize(self.vib_projection(vib_raw), p=2, dim=1)
            
            text_raw = self.text_encoder.encode_texts(candidate_texts, device)
            text_emb = F.normalize(self.text_projection(text_raw), p=2, dim=1)
            
            # 유사도 계산
            similarities = torch.matmul(vib_emb, text_emb.t())  # (1, N)
            
            # 가장 높은 유사도 선택
            best_idx = torch.argmax(similarities, dim=1).item()
            confidence = similarities.max().item()
            
            return best_idx, confidence
    
    def encode_vibration(self, vibration_signals: torch.Tensor) -> torch.Tensor:
        """진동 신호만 인코딩 (추론용)"""
        vib_raw = self.vib_encoder(vibration_signals)
        return F.normalize(self.vib_projection(vib_raw), p=2, dim=1)
    
    def encode_texts(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """텍스트만 인코딩 (추론용)"""
        text_raw = self.text_encoder.encode_texts(texts, device)
        return F.normalize(self.text_projection(text_raw), p=2, dim=1)
    
    def switch_to_continual_mode(self):
        """Continual learning 모드로 전환"""
        self.is_continual_mode = True
        
        # 텍스트 인코더 freeze (LoRA만)
        self.text_encoder.disable_lora_training()
        
        # Projection layer는 학습 가능하게 유지
        if hasattr(self.text_encoder, 'projection'):
            for param in self.text_encoder.projection.parameters():
                param.requires_grad = True
        
        logger.info("Continual learning 모드로 전환 완료 (텍스트 안정화, 진동 적응)")
    
    def switch_to_first_domain_mode(self):
        """First domain training 모드로 전환"""
        self.is_continual_mode = False
        
        # 텍스트 인코더 LoRA 활성화
        self.text_encoder.enable_lora_training()
        
        logger.info("First domain training 모드로 전환 완료 (전체 학습)")
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """각 컴포넌트별 학습 가능한 파라미터 수 반환"""
        text_params = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        vib_params = sum(p.numel() for p in self.vib_encoder.parameters() if p.requires_grad)
        proj_params = sum(p.numel() for p in self.text_projection.parameters() if p.requires_grad)
        proj_params += sum(p.numel() for p in self.vib_projection.parameters() if p.requires_grad)
        
        if self.use_aux:
            aux_params = sum(p.numel() for p in self.text_classifier.parameters() if p.requires_grad)
            aux_params += sum(p.numel() for p in self.vib_classifier.parameters() if p.requires_grad)
        else:
            aux_params = 0
        
        return {
            'text_encoder': text_params,
            'vib_encoder': vib_params,
            'projections': proj_params,
            'classifiers': aux_params,
            'total': text_params + vib_params + proj_params + aux_params
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


def create_textvib_model(domain_stage: str = 'first_domain', dataset_type: str = 'uos') -> TextVibCLIP:
    """
    TextVibCLIP v2 모델 생성
    
    Args:
        domain_stage: 'first_domain' 또는 'continual'
        dataset_type: 'uos' 또는 'cwru'
        
    Returns:
        TextVibCLIP: 새로운 ranking-based 모델
    """
    model = TextVibCLIP(domain_stage=domain_stage, dataset_type=dataset_type)
    
    # 파라미터 정보 출력
    param_info = model.get_trainable_parameters()
    logger.info(f"TextVibCLIP v2 생성 완료: {domain_stage} stage")
    logger.info(f"Text encoder: {param_info['text_encoder']:,}")
    logger.info(f"Vibration encoder: {param_info['vib_encoder']:,}")
    logger.info(f"Projections: {param_info['projections']:,}")
    logger.info(f"Classifiers: {param_info['classifiers']:,}")
    logger.info(f"총 학습 가능한 파라미터: {param_info['total']:,}")
    
    return model


def compute_text_vib_similarities(text_embeddings: torch.Tensor, 
                            vib_embeddings: torch.Tensor) -> torch.Tensor:
    """
    텍스트-진동 유사도 계산 (추론용)
    
    Args:
        text_embeddings: (num_texts, embedding_dim)
        vib_embeddings: (num_vibs, embedding_dim)
        
    Returns:
        torch.Tensor: 유사도 행렬 (num_vibs, num_texts)
    """
    # L2 정규화
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    vib_embeddings = F.normalize(vib_embeddings, p=2, dim=1)
    
    # Cosine similarity
    similarities = torch.matmul(vib_embeddings, text_embeddings.t())
    
    return similarities


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    print("=== TextVibCLIP v2 테스트 ===")
    
    # 모델 생성
    model = create_textvib_model('first_domain')
    
    # 테스트 데이터
    batch_size = 4
    test_batch = {
        'vibration': torch.randn(batch_size, 2048),
        'text': [
            "Healthy bearing condition observed",
            "Ball element defect detected", 
            "Inner race fault observed",
            "Outer race defect detected"
        ],
        'labels': torch.tensor([0, 1, 2, 3])
    }
    
    # GPU 사용
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_batch['vibration'] = test_batch['vibration'].to(device)
    test_batch['labels'] = test_batch['labels'].to(device)
    
    # Forward pass 테스트
    results = model(test_batch, return_embeddings=True)
    
    print(f"Loss: {results['loss'].item():.4f}")
    print(f"Loss components: {results['loss_components']}")
    print(f"Text embeddings shape: {results['text_embeddings'].shape}")
    print(f"Vib embeddings shape: {results['vib_embeddings'].shape}")
    
    # 실제 사용 시나리오 테스트
    print("\n=== 실제 사용 시나리오 테스트 ===")
    new_vibration = torch.randn(2048).to(device)
    candidates = [
        "Healthy bearing condition observed",
        "Ball element defect detected",
        "Inner race fault observed", 
        "Outer race defect detected"
    ]
    
    best_idx, confidence = model.predict_best_match(new_vibration, candidates, device)
    print(f"Best match: {candidates[best_idx]}")
    print(f"Confidence: {confidence:.4f}")
    
    print("\n=== TextVibCLIP v2 테스트 완료 ===")
