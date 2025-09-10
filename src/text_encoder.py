"""
Text Encoder: DistilBERT + LoRA 구현
베어링 진단 도메인에 특화된 텍스트 인코더
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, List, Optional, Tuple
import logging

from configs.model_config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """
    DistilBERT + LoRA 기반 텍스트 인코더
    
    베어링 진단 텍스트 설명을 512차원 임베딩으로 변환
    """
    
    def __init__(self,
                 model_name: str = MODEL_CONFIG['text_encoder']['model_name'],
                 embedding_dim: int = MODEL_CONFIG['embedding_dim'],
                 enable_lora: bool = True,
                 freeze_base: bool = False):
        """
        Args:
            model_name (str): DistilBERT 모델 이름
            embedding_dim (int): 출력 임베딩 차원
            enable_lora (bool): LoRA 활성화 여부
            freeze_base (bool): Base model freeze 여부 (Domain 2+에서 사용)
        """
        super(TextEncoder, self).__init__()
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.enable_lora = enable_lora
        self.freeze_base = freeze_base
        
        # Tokenizer 로딩
        # 토크나이저: 텍스트를 토큰으로 변환하는 객체
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        # DistilBERT 모델 로딩
        # DistilBERT: 텍스트를 임베딩으로 변환하는 모델
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        
        # LoRA 적용
        if enable_lora:
            self._apply_lora()
        
        # Base model freeze (Domain 2+에서 사용)
        if freeze_base:
            self._freeze_base_model()
        
        # Projection layer (DistilBERT hidden_size -> embedding_dim)
        bert_hidden_size = self.distilbert.config.hidden_size  # 768 for DistilBERT
        self.projection = nn.Sequential(
            nn.Linear(bert_hidden_size, MODEL_CONFIG['projection']['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['projection']['dropout']),
            nn.Linear(MODEL_CONFIG['projection']['hidden_dim'], embedding_dim)
            # 🎯 FIXED: LayerNorm 제거 (gradient vanishing 방지)
        )
        
        logger.info(f"TextEncoder 초기화 완료: LoRA={enable_lora}, Freeze={freeze_base}")
    
    def _apply_lora(self):
        """DistilBERT에 LoRA 적용"""
        try:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=MODEL_CONFIG['text_encoder']['lora_config']['r'],
                lora_alpha=MODEL_CONFIG['text_encoder']['lora_config']['lora_alpha'],
                target_modules=MODEL_CONFIG['text_encoder']['lora_config']['target_modules'],
                lora_dropout=MODEL_CONFIG['text_encoder']['lora_config']['lora_dropout'],
                bias="none"
            )
            
            self.distilbert = get_peft_model(self.distilbert, lora_config)
            logger.info(f"LoRA 적용 완료: rank={lora_config.r}, alpha={lora_config.lora_alpha}")
            
        except Exception as e:
            logger.error(f"LoRA 적용 실패: {e}")
            raise e
    
    def _freeze_base_model(self):
        """Base DistilBERT 파라미터 freeze (LoRA adapter는 제외)"""
        for name, param in self.distilbert.named_parameters():
            if 'lora_' not in name:  # LoRA 파라미터가 아닌 경우만 freeze
                param.requires_grad = False
        
        frozen_params = sum(1 for name, param in self.distilbert.named_parameters() if not param.requires_grad)
        logger.info(f"Base DistilBERT 파라미터 freeze 완료: {frozen_params}개 파라미터")
    
    def enable_lora_training(self):
        """LoRA 학습 활성화 (Domain 1에서 사용)"""
        # PEFT adapter 상태 확인 및 활성화
        if hasattr(self.distilbert, 'peft_config') and self.distilbert.peft_config:
            # adapter가 존재하는 경우
            if hasattr(self.distilbert, 'enable_adapters'):
                try:
                    self.distilbert.enable_adapters()
                    logger.info("LoRA adapter 활성화됨")
                except Exception as e:
                    # 일부 백엔드에서 adapter registry 미구현으로 ValueError("No adapter loaded")가 발생할 수 있음
                    # 파라미터 레벨에서 LoRA 학습을 강제 활성화하므로 경고가 아니라 정보로만 남김
                    logger.info(f"LoRA adapter 활성화 시도 건너뜀: {e}")
        else:
            logger.info("PEFT adapter가 초기화되지 않음 - LoRA 파라미터 직접 활성화")
        
        # LoRA 파라미터를 학습 가능하게 설정 (adapter 상태와 무관하게 실행)
        lora_params_found = 0
        for name, param in self.distilbert.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                lora_params_found += 1
        
        logger.info(f"LoRA 학습 활성화: {lora_params_found}개 LoRA 파라미터")
    
    def disable_lora_training(self):
        """LoRA 학습 비활성화 (Domain 2+에서 사용)"""
        # PEFT adapter 비활성화 시도 (여러 조건 체크)
        adapter_disabled = False
        
        if hasattr(self.distilbert, 'disable_adapters'):
            try:
                # 조건 1: peft_config 확인
                if hasattr(self.distilbert, 'peft_config') and self.distilbert.peft_config:
                    self.distilbert.disable_adapters()
                    adapter_disabled = True
                    logger.info("PEFT adapter 비활성화됨 (peft_config 경로)")
                
                # 조건 2: active_adapters 확인
                elif hasattr(self.distilbert, 'active_adapters') and len(self.distilbert.active_adapters) > 0:
                    self.distilbert.disable_adapters()
                    adapter_disabled = True
                    logger.info("PEFT adapter 비활성화됨 (active_adapters 경로)")
                    
                # 조건 3: adapter_name이 있는지 확인  
                elif hasattr(self.distilbert, 'adapter_name'):
                    try:
                        self.distilbert.disable_adapters()
                        adapter_disabled = True
                        logger.info("PEFT adapter 비활성화됨 (adapter_name 경로)")
                    except:
                        pass
                        
            except ValueError as e:
                if "No adapter loaded" in str(e):
                    logger.info("비활성화할 adapter가 없습니다 - 이미 비활성화됨")
                else:
                    logger.warning(f"Adapter 비활성화 중 예상치 못한 오류: {e}")
            except Exception as e:
                logger.warning(f"Adapter 비활성화 시도 중 오류 (무시됨): {e}")
        
        if not adapter_disabled:
            logger.info("PEFT adapter 상태 불명 - 파라미터 레벨에서 직접 freeze")
        
        # 모든 DistilBERT 파라미터 강제 freeze (adapter 상태와 무관)
        frozen_count = 0
        for param in self.distilbert.parameters():
            param.requires_grad = False
            frozen_count += 1
            
        logger.info(f"LoRA 학습 비활성화 완료: {frozen_count}개 파라미터 freeze")
    
    def tokenize_text(self, 
                     texts: List[str], 
                     max_length: int = 128) -> Dict[str, torch.Tensor]:
        """
        텍스트 토크나이징
        
        Args:
            texts (List[str]): 텍스트 리스트
            max_length (int): 최대 길이
            
        Returns:
            Dict[str, torch.Tensor]: 토크나이징된 결과
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids (torch.Tensor): 토큰 ID (batch_size, seq_len)
            attention_mask (torch.Tensor): 어텐션 마스크 (batch_size, seq_len)
            
        Returns:
            torch.Tensor: 텍스트 임베딩 (batch_size, embedding_dim)
        """
        # DistilBERT 인코딩
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # [CLS] 토큰 임베딩 추출 (첫 번째 토큰)
        #  Self-Attention으로 모든 토큰 정보가 [CLS]로 모이게 학습되기 때문.
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # Projection layer를 통해 목표 차원으로 변환
        text_embedding = self.projection(cls_embedding)  # (batch_size, embedding_dim)
        
        return text_embedding
    
    def encode_texts(self, 
                    texts: List[str], 
                    device: torch.device,
                    max_length: int = 128) -> torch.Tensor:
        """
        텍스트 리스트를 임베딩으로 변환 (편의 함수)
        
        Args:
            texts (List[str]): 텍스트 리스트
            device (torch.device): 디바이스
            max_length (int): 최대 길이
            
        Returns:
            torch.Tensor: 텍스트 임베딩 (batch_size, embedding_dim)
        """
        # 토크나이징
        tokenized = self.tokenize_text(texts, max_length)
        
        # 디바이스로 이동
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        
        # 인코딩
        # 학습 모드에서는 gradient가 흐르도록 설정
        with torch.set_grad_enabled(self.training):
            embeddings = self.forward(input_ids, attention_mask)
        
        return embeddings
    
    def encode_tokenized(self, 
                        input_ids: torch.Tensor,
                        attention_mask: torch.Tensor) -> torch.Tensor:
        """
        이미 토크나이징된 데이터를 임베딩으로 변환 (배치 효율성)
        
        Args:
            input_ids (torch.Tensor): 토큰 ID (batch_size, seq_len)
            attention_mask (torch.Tensor): 어텐션 마스크 (batch_size, seq_len)
            
        Returns:
            torch.Tensor: 텍스트 임베딩 (batch_size, embedding_dim)
        """
        # 학습 모드에서는 gradient가 흐르도록 설정
        with torch.set_grad_enabled(self.training):
            embeddings = self.forward(input_ids, attention_mask)
        
        return embeddings
    
    def get_trainable_parameters(self) -> int:
        """학습 가능한 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_lora_parameters(self) -> int:
        """LoRA 파라미터 수 반환"""
        if not self.enable_lora:
            return 0
        
        lora_params = 0
        for name, param in self.distilbert.named_parameters():
            if 'lora_' in name and param.requires_grad:
                lora_params += param.numel()
        
        return lora_params


class TextEncoderConfig:
    """TextEncoder 설정 관리 클래스"""
    
    @staticmethod # 굳이 객체를 만들지 않고도 호출 가능하도록 @staticmethod로 선언
    def get_domain1_config() -> Dict:
        """Domain 1 (First Domain Training) 설정"""
        return {
            'enable_lora': True,
            'freeze_base': True,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5
        }
    
    @staticmethod  
    def get_continual_config() -> Dict:
        """Domain 2+ (Continual Learning) 설정"""
        return {
            'enable_lora': True,
            'freeze_base': True,
            'learning_rate': 0,  # 학습하지 않음
            'weight_decay': 0
        }


def create_text_encoder(domain_stage: str = 'first_domain') -> TextEncoder:
    """
    도메인 단계에 따른 TextEncoder 생성
    
    Args:
        domain_stage (str): 'first_domain' (Domain 1) 또는 'continual' (Domain 2+)
        
    Returns:
        TextEncoder: 설정된 텍스트 인코더
    """
    if domain_stage == 'first_domain':
        config = TextEncoderConfig.get_domain1_config()
    elif domain_stage == 'continual':
        config = TextEncoderConfig.get_continual_config()
    else:
        raise ValueError(f"알 수 없는 domain_stage: {domain_stage}")
    
    encoder = TextEncoder(
        enable_lora=config['enable_lora'],
        freeze_base=config['freeze_base']
    )
    
    logger.info(f"TextEncoder 생성 ({domain_stage}): "
               f"Total={encoder.get_trainable_parameters():,}, "
               f"LoRA={encoder.get_lora_parameters():,}")
    
    return encoder


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    print("=== TextEncoder 테스트 ===")
    
    # Domain 1 (First Domain Training) 인코더 테스트
    print("\n1. Domain 1 (First Domain Training) 인코더")
    encoder_domain1 = create_text_encoder('first_domain')
    
    # 테스트 텍스트
    test_texts = [
        "A deep groove ball bearing operating at 600 rpm with healthy rotating component and ball fault.",
        "A tapered roller bearing operating at 800 rpm with healthy rotating component and healthy bearing.",
        "A cylindrical roller bearing operating at 1000 rpm with unbalanced rotating component and inner race fault."
    ]
    
    # 인코딩 테스트
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder_domain1.to(device)
    
    embeddings = encoder_domain1.encode_texts(test_texts, device)
    print(f"임베딩 shape: {embeddings.shape}")
    print(f"임베딩 norm: {torch.norm(embeddings, dim=1)}")
    
    # Domain 2+ (Continual Learning) 인코더 테스트
    print("\n2. Domain 2+ (Continual Learning) 인코더")
    encoder_continual = create_text_encoder('continual')
    encoder_continual.to(device)
    
    embeddings_continual = encoder_continual.encode_texts(test_texts, device)
    print(f"임베딩 shape: {embeddings_continual.shape}")
    
    # 파라미터 비교
    print(f"\nDomain 1 학습 가능한 파라미터: {encoder_domain1.get_trainable_parameters():,}")
    print(f"Domain 2+ 학습 가능한 파라미터: {encoder_continual.get_trainable_parameters():,}")
    
    print("\n=== TextEncoder 테스트 완료 ===")
