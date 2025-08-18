"""
First Domain Training 단독 실행 스크립트
첫 번째 도메인(600 RPM)에서만 초기 학습
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import logging
from datetime import datetime

from src.continual_trainer import ContinualTrainer
from src.data_loader import create_first_domain_dataloader
from configs.model_config import TRAINING_CONFIG

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """First Domain Training 실행"""
    logger.info("=== First Domain Training 단독 실행 ===")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"디바이스: {device}")
    
    # Trainer 생성
    trainer = ContinualTrainer(device=device, save_dir='checkpoints/first_domain_only')
    
    # 첫 번째 도메인 데이터로더 생성
    logger.info("첫 번째 도메인 데이터로더 생성 중...")
    train_loader = create_first_domain_dataloader(subset='train', batch_size=32)
    logger.info(f"첫 번째 도메인 학습 데이터: {len(train_loader.dataset)}개 샘플")
    
    # First domain training 실행
    logger.info("First Domain Training 시작...")
    results = trainer.train_first_domain(
        first_domain_dataloader=train_loader,
        num_epochs=50
    )
    
    # 결과 출력
    logger.info("=== First Domain Training 결과 ===")
    logger.info(f"최종 Loss: {results['final_loss']:.4f}")
    logger.info(f"평균 Loss: {results['avg_loss']:.4f}")
    
    logger.info("도메인별 성능:")
    for domain, metrics in results['domain_performances'].items():
        logger.info(f"  Domain {domain}: {metrics['accuracy']:.4f}")
    
    # 모델 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'checkpoints/first_domain_only/first_domain_training_{timestamp}.pth'
    trainer.model.save_checkpoint(save_path, 50)
    logger.info(f"모델 저장됨: {save_path}")
    
    logger.info("=== First Domain Training 완료 ===")


if __name__ == "__main__":
    main()
