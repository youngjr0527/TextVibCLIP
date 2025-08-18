"""
TextVibCLIP: Multimodal Continual Learning for Bearing Fault Diagnosis

진동 신호와 텍스트 메타데이터를 결합한 CLIP-inspired contrastive learning framework
"""

__version__ = "1.0.0"
__author__ = "TextVibCLIP Research Team"

# 주요 컴포넌트 import
from .textvib_model import TextVibCLIP
from .text_encoder import TextEncoder
from .vibration_encoder import VibrationEncoder
from .data_loader import UOSDataset, create_domain_dataloaders
from .continual_trainer import ContinualTrainer

__all__ = [
    'TextVibCLIP',
    'TextEncoder', 
    'VibrationEncoder',
    'UOSDataset',
    'create_domain_dataloaders',
    'ContinualTrainer'
]
