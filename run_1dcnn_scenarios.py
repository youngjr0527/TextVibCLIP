#!/usr/bin/env python3
"""
1D-CNN 비교군 실험 스크립트
TextVibCLIP의 텍스트 모달리티 기여도 분석을 위한 비교군 실험

Usage:
    python run_1dcnn_scenarios.py --quick_test --epochs 10
"""

import argparse
import logging
import os
import torch
import torch.nn.functional as F
import time
import json
import numpy as np
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# 🎯 재현성 보장을 위한 시드 고정
def set_random_seeds(seed: int = 42):
    """모든 랜덤 시드 고정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎯 Random seeds fixed to {seed} for reproducibility")

# 프로젝트 루트 경로 추가
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.onedcnn_trainer import OneDCNNTrainer
from src.data_loader import create_domain_dataloaders
from src.data_cache import create_cached_domain_dataloaders, create_cached_first_domain_dataloader, clear_all_caches
from src.utils import set_seed
from src.visualization import create_visualizer
from configs.model_config import TRAINING_CONFIG, DATA_CONFIG


def setup_logging(log_dir: str) -> Tuple[logging.Logger, str]:
    """로깅 설정"""
    experiment_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(log_dir, f"{experiment_timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    log_filename = f"onedcnn_{experiment_timestamp}.log"
    log_path = os.path.join(experiment_dir, log_filename)
    
    # 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"1D-CNN 비교군 실험 시작: {log_path}")
    logger.info(f"실험 결과 폴더: {experiment_dir}")
    
    return logger, experiment_dir


class ScenarioConfig:
    """시나리오별 설정"""
    
    UOS_CONFIG = {
        'name': 'UOS_1DCNN_Baseline',
        'data_dir': 'data_scenario1',
        'dataset_type': 'uos',
        'domain_order': [600, 800, 1000, 1200, 1400, 1600],
        'domain_names': ['600RPM', '800RPM', '1000RPM', '1200RPM', '1400RPM', '1600RPM'],
        'shift_type': 'Varying Speed',
        'first_domain_epochs': 20,
        'remaining_epochs': 8,
        'batch_size': 16,
        'replay_buffer_size': 0,  # Vanilla 1D-CNN: replay buffer 사용 안 함
        'patience': 10
    }


def run_single_scenario(config: Dict, logger: logging.Logger, device: torch.device, args, experiment_dir: str) -> Dict:
    """단일 시나리오 실행"""
    logger.info(f"🚀 {config['name']} 시작!")
    logger.info(f"   아키텍처: 1D-CNN Classifier (TextVibCLIP 비교군)")
    logger.info(f"   Domains: {' → '.join(config['domain_names'])}")
    
    start_time = time.time()
    
    try:
        # Trainer 생성
        checkpoint_dir = os.path.join(experiment_dir, 'checkpoints', config['name'])
        
        trainer = OneDCNNTrainer(
            model=None,
            device=device,
            save_dir=checkpoint_dir,
            domain_order=config['domain_order'],
            data_dir=config['data_dir'],
            dataset_type=config['dataset_type'],
            results_save_dir=None
        )
        
        # 하이퍼파라미터 설정
        trainer.batch_size = config['batch_size']
        trainer.replay_buffer.buffer_size_per_domain = config['replay_buffer_size']
        
        # First Domain Training
        logger.info("📚 First Domain Training...")
        
        first_loader = create_cached_first_domain_dataloader(
            data_dir=config['data_dir'],
            domain_order=config['domain_order'],
            dataset_type=config['dataset_type'],
            subset='train',
            batch_size=config['batch_size']
        )
        
        first_results = trainer.train_first_domain(
            first_domain_dataloader=first_loader,
            num_epochs=config['first_domain_epochs']
        )
        
        # Remaining Domains Training
        logger.info("🔄 Remaining Domains Training...")
        
        domain_loaders = create_cached_domain_dataloaders(
            data_dir=config['data_dir'],
            domain_order=config['domain_order'],
            dataset_type=config['dataset_type'],
            batch_size=config['batch_size']
        )
        
        # 남도메인 에폭/설정 강제 반영
        try:
            from configs.model_config import CONTINUAL_CONFIG
            CONTINUAL_CONFIG['num_epochs'] = max(1, int(config.get('remaining_epochs', 3)))
        except Exception:
            pass
        
        remaining_results = trainer.train_remaining_domains(domain_loaders)
        
        # 시각화 생성
        try:
            logger.info("📊 시각화 생성 중...")
            visualizer = create_visualizer(experiment_dir)
            
            # Continual Learning Performance Curve
            visualizer.create_continual_learning_curve(
                domain_names=config['domain_names'],
                accuracies=remaining_results['final_metrics']['final_accuracies'],
                scenario_name=config['name']
            )
            
            # Forgetting Analysis Heatmap
            n_domains = len(config['domain_names'])
            accuracy_matrix = np.full((n_domains, n_domains), np.nan)
            
            for i in range(n_domains):
                for j in range(n_domains):
                    if j <= i:
                        test_domain = config['domain_order'][j]
                        if test_domain in trainer.performance_history:
                            history = trainer.performance_history[test_domain]['accuracy']
                            history_idx = i - j
                            if len(history) > history_idx:
                                accuracy_matrix[i, j] = history[history_idx]
            
            visualizer.create_forgetting_heatmap(
                domain_names=config['domain_names'],
                accuracy_matrix=accuracy_matrix,
                scenario_name=config['name']
            )
            
            logger.info("✅ 시각화 생성 완료!")
        except Exception as viz_err:
            logger.warning(f"시각화 생성 실패: {viz_err}")
        
        # 실험 설정 저장
        config_path = save_experiment_config(config, trainer, experiment_dir, device)
        logger.info(f"📝 실험 설정 저장: {config_path}")
        
        # 결과 정리
        final_metrics = remaining_results['final_metrics']
        total_time = time.time() - start_time
        
        # Forgetting Heatmap 데이터 추출
        n_domains = len(config['domain_names'])
        heatmap_matrix = []
        stage_averages = []
        
        for i in range(n_domains):
            row = []
            for j in range(n_domains):
                if j <= i:
                    test_domain = config['domain_order'][j]
                    if test_domain in trainer.performance_history:
                        history = trainer.performance_history[test_domain]['accuracy']
                        history_idx = i - j
                        if len(history) > history_idx:
                            row.append(round(history[history_idx] * 100, 2))
                        else:
                            row.append(None)
                    else:
                        row.append(None)
                else:
                    row.append(None)
            
            valid_values = [v for v in row if v is not None]
            if valid_values:
                stage_avg = round(sum(valid_values) / len(valid_values), 2)
            else:
                stage_avg = None
            
            heatmap_matrix.append(row)
            stage_averages.append(stage_avg)
        
        results = {
            'domain_names': config['domain_names'],
            'shift_type': config['shift_type'],
            'stage_accuracies': stage_averages,
            'average_accuracy': final_metrics['average_accuracy'],
            'average_forgetting': final_metrics['average_forgetting'],
            'forgetting_matrix': heatmap_matrix,
            'total_time': total_time,
            'first_domain_epochs': config['first_domain_epochs'],
            'remaining_epochs': config['remaining_epochs'],
            'batch_size': config['batch_size']
        }
        
        logger.info(f"✅ {config['name']} 완료!")
        logger.info(f"   평균 정확도: {final_metrics['average_accuracy']:.4f}")
        logger.info(f"   평균 망각도: {final_metrics['average_forgetting']:.4f}")
        logger.info(f"   소요 시간: {total_time/60:.1f}분")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ {config['name']} 실행 중 오류: {str(e)}")
        logger.exception("상세 오류 정보:")
        return None


def save_experiment_config(config: Dict, trainer, output_dir: str, device: torch.device) -> str:
    """실험 설정을 txt 파일로 저장"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_path = os.path.join(output_dir, f'experiment_config_{timestamp}.txt')
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("1D-CNN Baseline Experiment Configuration\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Scenario: {config['name']}\n")
        f.write(f"Dataset: {config['dataset_type'].upper()}\n")
        f.write(f"Domain Order: {' → '.join(config['domain_names'])}\n")
        f.write(f"Shift Type: {config['shift_type']}\n")
        f.write(f"Device: {device}\n\n")
        
        # Scenario Configuration
        f.write("-" * 50 + "\n")
        f.write("Scenario Configuration\n")
        f.write("-" * 50 + "\n")
        for key, value in config.items():
            if key not in ['name', 'data_dir', 'dataset_type', 'domain_order', 'domain_names', 'shift_type']:
                f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Model Architecture
        f.write("-" * 50 + "\n")
        f.write("Model Architecture\n")
        f.write("-" * 50 + "\n")
        f.write("Architecture: 1D-CNN Classifier (TextVibCLIP 비교군)\n")
        f.write("Input: Vibration signal only (no text modality)\n")
        f.write("Output: Classification logits (7 classes for UOS)\n")
        try:
            from configs.model_config import MODEL_CONFIG
            f.write(f"embedding_dim: {MODEL_CONFIG['embedding_dim']}\n")
            f.write(f"vibration_input_length: {MODEL_CONFIG['vibration_encoder']['input_length']}\n")
            f.write(f"vibration_kernel_sizes: {MODEL_CONFIG['vibration_encoder']['kernel_sizes']}\n")
            f.write(f"vibration_channels: {MODEL_CONFIG['vibration_encoder']['channels']}\n")
        except Exception as e:
            f.write(f"Model config loading failed: {e}\n")
        f.write("\n")
        
        # Training Configuration
        f.write("-" * 50 + "\n")
        f.write("Training Configuration\n")
        f.write("-" * 50 + "\n")
        try:
            from configs.model_config import FIRST_DOMAIN_CONFIG, CONTINUAL_CONFIG
            f.write("First Domain:\n")
            f.write(f"  epochs: {FIRST_DOMAIN_CONFIG['num_epochs']}\n")
            f.write(f"  learning_rate: {FIRST_DOMAIN_CONFIG['learning_rate']}\n")
            f.write(f"  weight_decay: {FIRST_DOMAIN_CONFIG['weight_decay']}\n")
            f.write("Remaining Domains:\n")
            f.write(f"  epochs: {CONTINUAL_CONFIG['num_epochs']}\n")
            f.write(f"  learning_rate: {CONTINUAL_CONFIG['learning_rate']}\n")
            f.write(f"  weight_decay: {CONTINUAL_CONFIG['weight_decay']}\n")
        except Exception as e:
            f.write(f"Training config loading failed: {e}\n")
        f.write("\n")
        
        # Replay Buffer Configuration
        f.write("-" * 50 + "\n")
        f.write("Replay Buffer Configuration\n")
        f.write("-" * 50 + "\n")
        f.write(f"buffer_size_per_domain: {trainer.replay_buffer.buffer_size_per_domain}\n")
        f.write(f"sampling_strategy: {trainer.replay_buffer.sampling_strategy}\n")
        f.write("\n")
        
        # Reproducibility Configuration
        f.write("-" * 50 + "\n")
        f.write("Reproducibility Configuration\n")
        f.write("-" * 50 + "\n")
        f.write(f"pytorch_seed: {torch.initial_seed()}\n")
        f.write(f"cudnn_deterministic: {torch.backends.cudnn.deterministic}\n")
        f.write(f"cudnn_benchmark: {torch.backends.cudnn.benchmark}\n")
        f.write("\n")
        
        # System Information
        f.write("-" * 50 + "\n")
        f.write("System Information\n")
        f.write("-" * 50 + "\n")
        f.write(f"python_version: {sys.version}\n")
        f.write(f"pytorch_version: {torch.__version__}\n")
        f.write(f"cuda_available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"cuda_version: {torch.version.cuda}\n")
            f.write(f"gpu_name: {torch.cuda.get_device_name()}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("Configuration saved successfully\n")
        f.write("=" * 80 + "\n")
    
    return config_path


def save_results(results: Dict, output_dir: str) -> str:
    """결과 저장"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(output_dir, f'results_{timestamp}.json')
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results_path


def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='1D-CNN 비교군 실험')
    
    parser.add_argument('--output_dir', type=str, default='results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--quick_test', action='store_true',
                       help='빠른 테스트 모드')
    parser.add_argument('--epochs', type=int, default=None,
                       help='에포크 수')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clear_cache', action='store_true')
    
    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = parse_arguments()
    
    # 재현성 설정
    set_random_seeds(args.seed)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 로깅 설정
    logger, experiment_dir = setup_logging(args.output_dir)
    
    # 캐시 관리
    if args.clear_cache:
        logger.info("🗑️ 캐시 삭제 중...")
        clear_all_caches()
    
    logger.info("🎯 1D-CNN 비교군 실험 시작!")
    logger.info("   아키텍처: 1D-CNN Classifier (TextVibCLIP 비교군)")
    logger.info("   목적: 텍스트 모달리티 기여도 분석")
    
    # 디바이스 설정
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"🔧 디바이스: {device}")
    
    # 시나리오 설정
    scenarios = [ScenarioConfig.UOS_CONFIG]
    
    # 에포크 설정
    if args.quick_test:
        test_epochs = args.epochs if args.epochs else 10
        logger.info(f"⚡ 빠른 테스트 모드: 에포크 {test_epochs}")
        for scenario in scenarios:
            scenario['first_domain_epochs'] = test_epochs
            scenario['remaining_epochs'] = max(test_epochs // 2, 3)
    elif args.epochs:
        for scenario in scenarios:
            scenario['first_domain_epochs'] = args.epochs
            scenario['remaining_epochs'] = max(args.epochs // 2, 3)
    
    # 시나리오별 실행
    all_results = {}
    total_start_time = time.time()
    
    for i, scenario in enumerate(scenarios, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"시나리오 {i}/{len(scenarios)}: {scenario['name']}")
        logger.info(f"{'='*60}")
        
        scenario_result = run_single_scenario(scenario, logger, device, args, experiment_dir)
        
        if scenario_result:
            all_results[scenario['name']] = scenario_result
        else:
            logger.error(f"❌ {scenario['name']} 실행 실패!")
    
    # 결과 저장
    if all_results:
        results_path = save_results(all_results, experiment_dir)
        logger.info(f"✅ 결과 저장: {results_path}")
    
    # 최종 요약
    total_time = time.time() - total_start_time
    logger.info(f"\n⏱️ 전체 실험 소요 시간: {total_time/60:.1f}분")
    
    # 성능 요약
    logger.info(f"\n📊 1D-CNN 비교군 성능 요약:")
    for scenario_name, result in all_results.items():
        avg_acc = result.get('average_accuracy', 0.0)
        avg_forget = result.get('average_forgetting', 0.0)
        logger.info(f"   {scenario_name}: 평균 정확도 {avg_acc:.4f}, 망각도 {avg_forget:.4f}")
    
    logger.info("🎉 1D-CNN 비교군 실험 완료!")


if __name__ == "__main__":
    main()

