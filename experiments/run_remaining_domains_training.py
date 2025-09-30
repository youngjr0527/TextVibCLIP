"""
Remaining Domains Training 단독 실행 스크립트
사전 학습된 First Domain model에서 시작하여 나머지 도메인들 순차 학습
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import logging
import argparse
from datetime import datetime

from src.continual_trainer import ContinualTrainer
from src.textvib_model_v2 import create_textvib_model_v2 as create_textvib_model
from src.data_loader import create_domain_dataloaders

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Remaining Domains Training 실행')
    parser.add_argument('--first_domain_checkpoint', type=str, required=True,
                       help='First domain training 체크포인트 경로')
    parser.add_argument('--epochs', type=int, default=30,
                       help='도메인별 학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--replay_buffer_size', type=int, default=500,
                       help='Replay buffer 크기')
    parser.add_argument('--replay_ratio', type=float, default=0.3,
                       help='Replay 데이터 비율')
    
    return parser.parse_args()


def main():
    """Remaining Domains Training 실행"""
    args = parse_args()
    
    logger.info("=== Remaining Domains Training 단독 실행 ===")
    logger.info(f"First domain checkpoint: {args.first_domain_checkpoint}")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"디바이스: {device}")
    
    # First domain training 모델 로딩
    logger.info("First domain training 모델 로딩 중...")
    model = create_textvib_model('first_domain')
    
    if os.path.exists(args.first_domain_checkpoint):
        checkpoint = torch.load(args.first_domain_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"체크포인트 로딩 완료: {args.first_domain_checkpoint}")
    else:
        logger.error(f"체크포인트 파일을 찾을 수 없음: {args.first_domain_checkpoint}")
        return
    
    # Trainer 생성
    trainer = ContinualTrainer(
        model=model, 
        device=device, 
        save_dir='checkpoints/remaining_domains_only'
    )
    
    # 설정 업데이트
    trainer.num_epochs = args.epochs
    trainer.batch_size = args.batch_size
    trainer.replay_buffer.buffer_size_per_domain = args.replay_buffer_size
    trainer.replay_ratio = args.replay_ratio
    
    # 도메인별 데이터로더 생성
    logger.info("도메인별 데이터로더 생성 중...")
    domain_loaders = create_domain_dataloaders(batch_size=args.batch_size)
    
    # 첫 번째 도메인을 완료된 것으로 표시 (First domain training에서 학습됨)
    trainer.completed_domains = [600]  # 첫 번째 도메인 RPM
    
    # Remaining domains training 실행
    logger.info("Remaining Domains Training 시작...")
    results = trainer.train_remaining_domains(domain_loaders)
    
    # 결과 출력
    logger.info("=== Remaining Domains Training 결과 ===")
    final_metrics = results['final_metrics']
    logger.info(f"평균 정확도: {final_metrics['average_accuracy']:.4f}")
    logger.info(f"평균 망각도: {final_metrics['average_forgetting']:.4f}")
    logger.info(f"학습 도메인 수: {final_metrics['num_domains']}")
    
    # 도메인별 결과
    logger.info("도메인별 학습 결과:")
    for domain_rpm, domain_result in results.items():
        if isinstance(domain_result, dict) and 'training_results' in domain_result:
            training_res = domain_result['training_results']
            forgetting = domain_result['forgetting_score']
            logger.info(f"  Domain {domain_rpm}: "
                       f"최적 Val Acc = {training_res['best_val_accuracy']:.4f}, "
                       f"Forgetting = {forgetting:.4f}")
    
    # 학습 이력 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    history_path = f'results/remaining_domains_history_{timestamp}.pth'
    trainer.save_training_history(history_path)
    logger.info(f"학습 이력 저장됨: {history_path}")
    
    # 최종 모델 저장
    final_model_path = f'checkpoints/remaining_domains_only/remaining_domains_final_{timestamp}.pth'
    trainer.model.save_checkpoint(final_model_path, args.epochs)
    logger.info(f"최종 모델 저장됨: {final_model_path}")
    
    # 학습 곡선 시각화
    plot_path = f'plots/remaining_domains_curves_{timestamp}.png'
    os.makedirs('plots', exist_ok=True)
    trainer.plot_continual_learning_curves(plot_path)
    logger.info(f"학습 곡선 저장됨: {plot_path}")
    
    logger.info("=== Remaining Domains Training 완료 ===")


if __name__ == "__main__":
    main()
