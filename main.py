"""
TextVibCLIP 메인 실험 실행 스크립트
Joint Training + Continual Learning 전체 파이프라인
"""

import argparse
import logging
import os
import time
import torch
from datetime import datetime

# 모듈 import
from src.continual_trainer import ContinualTrainer
from src.data_loader import create_domain_dataloaders, create_combined_dataloader, create_first_domain_dataloader
from src.textvib_model import create_textvib_model
from src.utils import set_seed
from configs.model_config import TRAINING_CONFIG, DATA_CONFIG

# 로깅 설정
def setup_logging(log_dir: str = 'logs'):
    """로깅 설정"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"textvibclip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"로깅 설정 완료: {log_path}")
    return logger


def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='TextVibCLIP Continual Learning Experiment')
    
    # 실험 설정
    parser.add_argument('--experiment_name', type=str, default='textvibclip_experiment',
                       help='실험 이름')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='결과 저장 디렉토리')
    
    # 모델 설정
    parser.add_argument('--embedding_dim', type=int, default=512,
                       help='임베딩 차원')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='학습률')
    
    # 학습 설정
    parser.add_argument('--first_domain_epochs', type=int, default=50,
                       help='First domain training 에포크 수')
    parser.add_argument('--remaining_domains_epochs', type=int, default=30,
                       help='Remaining domains training 에포크 수')
    parser.add_argument('--replay_buffer_size', type=int, default=500,
                       help='Replay buffer 크기')
    parser.add_argument('--replay_ratio', type=float, default=0.3,
                       help='Replay 데이터 비율')
    
    # 실험 모드
    parser.add_argument('--mode', type=str, choices=['full', 'first_domain_only', 'remaining_domains_only'],
                       default='full', help='실험 모드')
    parser.add_argument('--load_first_domain_checkpoint', type=str, default=None,
                       help='First domain training 체크포인트 경로 (remaining_domains_only 모드용)')
    
    # 하드웨어 설정
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'], help='학습 디바이스')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader 워커 수')
    
    # 평가 설정
    parser.add_argument('--eval_interval', type=int, default=5,
                       help='평가 간격 (에포크)')
    parser.add_argument('--save_plots', action='store_true',
                       help='결과 플롯 저장 여부')
    
    # 재현성 설정
    parser.add_argument('--seed', type=int, default=42,
                       help='재현성을 위한 시드값')
    parser.add_argument('--no_amp', action='store_true',
                       help='AMP 비활성화')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Gradient clipping 최대 norm')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """디바이스 설정"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    if device.type == 'cuda':
        print(f"GPU 사용: {torch.cuda.get_device_name()}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CPU 사용")
    
    return device


def create_experiment_directory(base_dir: str, experiment_name: str) -> str:
    """실험 디렉토리 생성"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    
    # 하위 디렉토리 생성
    subdirs = ['checkpoints', 'logs', 'plots', 'results']
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    
    return exp_dir


def run_first_domain_training(trainer: ContinualTrainer, 
                             args: argparse.Namespace,
                             logger: logging.Logger) -> dict:
    """First Domain Training 실행 (600 RPM)"""
    logger.info("🚀 First Domain Training 시작 (600 RPM)")
    
    # 첫 번째 도메인만의 데이터로더 생성
    first_domain_loader = create_first_domain_dataloader(
        subset='train', 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    first_domain_rpm = DATA_CONFIG['domain_order'][0]
    logger.info(f"First Domain Training 데이터: Domain {first_domain_rpm} RPM, "
               f"{len(first_domain_loader.dataset)}개 샘플")
    
    # First domain training 실행
    start_time = time.time()
    first_domain_results = trainer.train_first_domain(
        first_domain_dataloader=first_domain_loader,
        num_epochs=args.first_domain_epochs
    )
    elapsed_time = time.time() - start_time
    
    logger.info(f"✅ First Domain Training 완료 (소요시간: {elapsed_time/60:.1f}분)")
    logger.info(f"최종 Loss: {first_domain_results['final_loss']:.4f}")
    
    # 도메인별 성능 출력
    for domain, metrics in first_domain_results['domain_performances'].items():
        logger.info(f"Domain {domain} 성능: {metrics['accuracy']:.4f}")
    
    return first_domain_results


def run_remaining_domains_training(trainer: ContinualTrainer,
                                  args: argparse.Namespace, 
                                  logger: logging.Logger) -> dict:
    """Remaining Domains Training 실행 (800~1600 RPM)"""
    logger.info("🔄 Remaining Domains Training 시작 (800~1600 RPM)")
    
    # 도메인별 데이터로더 생성
    domain_loaders = create_domain_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    remaining_domains = DATA_CONFIG['domain_order'][1:]  # 첫 번째 제외
    logger.info(f"Remaining Domains: {remaining_domains}")
    
    # Replay buffer 설정 업데이트
    trainer.replay_buffer.buffer_size_per_domain = args.replay_buffer_size
    trainer.replay_ratio = args.replay_ratio
    
    # Remaining domains training 실행
    start_time = time.time()
    remaining_domains_results = trainer.train_remaining_domains(domain_loaders)
    elapsed_time = time.time() - start_time
    
    logger.info(f"✅ Remaining Domains Training 완료 (소요시간: {elapsed_time/60:.1f}분)")
    
    # 최종 메트릭 출력
    final_metrics = remaining_domains_results['final_metrics']
    logger.info(f"평균 정확도: {final_metrics['average_accuracy']:.4f}")
    logger.info(f"평균 망각도: {final_metrics['average_forgetting']:.4f}")
    
    return remaining_domains_results


def save_experiment_results(trainer: ContinualTrainer,
                          first_domain_results: dict,
                          remaining_domains_results: dict,
                          exp_dir: str,
                          args: argparse.Namespace,
                          logger: logging.Logger):
    """실험 결과 저장"""
    logger.info("💾 실험 결과 저장 중...")
    
    # 1. 학습 이력 저장
    history_path = os.path.join(exp_dir, 'results', 'training_history.pth')
    trainer.save_training_history(history_path)
    
    # 2. 최종 모델 저장
    final_model_path = os.path.join(exp_dir, 'checkpoints', 'final_model.pth')
    trainer.model.save_checkpoint(final_model_path, 0)
    
    # 3. Replay buffer 저장
    buffer_path = os.path.join(exp_dir, 'results', 'replay_buffer.pth')
    trainer.replay_buffer.save_buffer(buffer_path)
    
    # 4. 실험 설정 저장
    experiment_config = {
        'args': vars(args),
        'first_domain_results': first_domain_results,
        'remaining_domains_results': remaining_domains_results,
        'model_config': {
            'embedding_dim': args.embedding_dim,
            'trainable_params': trainer.model.get_trainable_parameters()
        },
        'data_config': {
            'domain_order': DATA_CONFIG['domain_order'],
            'window_size': DATA_CONFIG['window_size']
        }
    }
    
    config_path = os.path.join(exp_dir, 'results', 'experiment_config.pth')
    torch.save(experiment_config, config_path)
    
    # 5. 플롯 저장 (옵션)
    if args.save_plots:
        plot_path = os.path.join(exp_dir, 'plots', 'continual_learning_curves.png')
        trainer.plot_continual_learning_curves(plot_path)
    
    logger.info(f"✅ 실험 결과 저장 완료: {exp_dir}")


def print_experiment_summary(first_domain_results: dict, 
                           remaining_domains_results: dict,
                           logger: logging.Logger):
    """실험 결과 요약 출력"""
    logger.info("\n" + "="*50)
    logger.info("📊 실험 결과 요약")
    logger.info("="*50)
    
    # First Domain Training 요약
    first_domain_rpm = DATA_CONFIG['domain_order'][0]
    logger.info(f"🎯 First Domain Training (Domain {first_domain_rpm} RPM):")
    logger.info(f"  - 최종 Loss: {first_domain_results['final_loss']:.4f}")
    logger.info(f"  - 평균 Loss: {first_domain_results['avg_loss']:.4f}")
    
    # Remaining Domains Training 요약
    final_metrics = remaining_domains_results['final_metrics']
    logger.info(f"🔄 Remaining Domains Training:")
    logger.info(f"  - 평균 정확도: {final_metrics['average_accuracy']:.4f}")
    logger.info(f"  - 평균 망각도: {final_metrics['average_forgetting']:.4f}")
    logger.info(f"  - 학습 도메인 수: {final_metrics['num_domains']}")
    
    # 도메인별 최종 성능
    logger.info(f"📈 도메인별 최종 성능:")
    for i, acc in enumerate(final_metrics['final_accuracies']):
        domain = DATA_CONFIG['domain_order'][i]
        logger.info(f"  - Domain {domain} RPM: {acc:.4f}")
    
    logger.info("="*50)


def main():
    """메인 실험 실행 함수"""
    # 인수 파싱
    args = parse_arguments()
    
    # 재현성 설정 (가장 먼저)
    set_seed(args.seed)
    
    # 실험 디렉토리 생성
    exp_dir = create_experiment_directory(args.save_dir, args.experiment_name)
    
    # 로깅 설정
    logger = setup_logging(os.path.join(exp_dir, 'logs'))
    logger.info(f"🎯 실험 시작: {args.experiment_name}")
    logger.info(f"📁 실험 디렉토리: {exp_dir}")
    logger.info(f"🌱 시드: {args.seed}")
    
    # 디바이스 설정
    device = setup_device(args.device)
    logger.info(f"🔧 디바이스: {device}")
    
    # 모델 및 Trainer 생성
    logger.info("🏗️ 모델 및 Trainer 초기화...")
    
    if args.mode == 'remaining_domains_only' and args.load_first_domain_checkpoint:
        # First domain checkpoint에서 모델 로딩
        model = create_textvib_model('joint')
        checkpoint = torch.load(args.load_first_domain_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer = ContinualTrainer(
            model=model, 
            device=device, 
            save_dir=os.path.join(exp_dir, 'checkpoints'),
            use_amp=not args.no_amp,
            max_grad_norm=args.max_grad_norm
        )
    else:
        trainer = ContinualTrainer(
            device=device, 
            save_dir=os.path.join(exp_dir, 'checkpoints'),
            use_amp=not args.no_amp,
            max_grad_norm=args.max_grad_norm
        )
    
    # 하이퍼파라미터 업데이트
    trainer.batch_size = args.batch_size
    trainer.learning_rate = args.learning_rate
    trainer.num_epochs = args.first_domain_epochs  # First domain training 기본값
    
    param_info = trainer.model.get_trainable_parameters()
    logger.info(f"📊 모델 파라미터: Total={param_info['total']:,}, "
               f"Text={param_info['text_total']:,}, Vib={param_info['vibration']:,}")
    
    # 실험 실행
    first_domain_results = {}
    remaining_domains_results = {}
    
    try:
        if args.mode in ['full', 'first_domain_only']:
            # First Domain Training 실행
            first_domain_results = run_first_domain_training(trainer, args, logger)
        
        if args.mode in ['full', 'remaining_domains_only']:
            # Remaining Domains Training 실행
            trainer.num_epochs = args.remaining_domains_epochs  # Remaining domains용 에포크 설정
            remaining_domains_results = run_remaining_domains_training(trainer, args, logger)
        
        # 결과 저장
        save_experiment_results(trainer, first_domain_results, remaining_domains_results, exp_dir, args, logger)
        
        # 결과 요약 출력
        if first_domain_results and remaining_domains_results:
            print_experiment_summary(first_domain_results, remaining_domains_results, logger)
        
        logger.info("🎉 실험 성공적으로 완료!")
        
    except Exception as e:
        logger.error(f"❌ 실험 실행 중 오류 발생: {str(e)}")
        logger.exception("상세 오류 정보:")
        raise e
    
    finally:
        # 정리 작업
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"📁 실험 결과 경로: {exp_dir}")


if __name__ == "__main__":
    main()
