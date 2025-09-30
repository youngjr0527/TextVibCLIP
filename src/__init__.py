"""
TextVibCLIP: Multimodal Continual Learning for Bearing Fault Diagnosis

진동 신호와 텍스트 메타데이터를 결합한 CLIP-inspired contrastive learning framework
"""

__version__ = "1.0.0"
__author__ = "TextVibCLIP Research Team"

# 주요 컴포넌트 import (v2 아키텍처)
from .textvib_model_v2 import TextVibCLIP_v2, create_textvib_model_v2
from .text_encoder import TextEncoder
from .vibration_encoder import VibrationEncoder
from .data_loader import UOSDataset, create_domain_dataloaders
from .continual_trainer_v2 import ContinualTrainer_v2

__all__ = [
    'TextVibCLIP_v2',
    'create_textvib_model_v2',
    'TextEncoder', 
    'VibrationEncoder',
    'UOSDataset',
    'create_domain_dataloaders',
    'ContinualTrainer_v2'
]
