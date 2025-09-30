#!/usr/bin/env python3
"""
TextVibCLIP v2 실험 스크립트
Ranking-based 아키텍처로 소규모 데이터에 최적화

Usage:
    python run_scenarios_v2.py --quick_test --epochs 10
    python run_scenarios_v2.py --skip_uos  # CWRU만
    python run_scenarios_v2.py --skip_cwru # UOS만
"""

import argparse
import logging
import os
import torch
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# 프로젝트 루트 경로 추가
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.continual_trainer import ContinualTrainer
from src.data_loader import create_domain_dataloaders
from src.data_cache import create_cached_domain_dataloaders, create_cached_first_domain_dataloader, clear_all_caches
from src.utils import set_seed
from src.visualization import create_visualizer
from configs.model_config import TRAINING_CONFIG, DATA_CONFIG, CWRU_DATA_CONFIG


def setup_logging(log_dir: str) -> Tuple[logging.Logger, str]:
    """로깅 설정"""
    experiment_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(log_dir, f"v2_{experiment_timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    log_filename = f"textvibclip_v2_{experiment_timestamp}.log"
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
    logger.info(f"TextVibCLIP v2 실험 시작: {log_path}")
    logger.info(f"실험 결과 폴더: {experiment_dir}")
    
    return logger, experiment_dir


class ScenarioConfig_v2:
    """시나리오별 설정 (v2용)"""
    
    UOS_CONFIG = {
        'name': 'UOS_Scenario1_VaryingSpeed_v2',
        'data_dir': 'data_scenario1',
        'dataset_type': 'uos',
        'domain_order': [600, 800, 1000, 1200, 1400, 1600],
        'domain_names': ['600RPM', '800RPM', '1000RPM', '1200RPM', '1400RPM', '1600RPM'],
        'shift_type': 'Varying Speed',
        'first_domain_epochs': 15,  # FIRST_DOMAIN_CONFIG 사용
        'remaining_epochs': 6,      # CONTINUAL_CONFIG 사용
        'batch_size': 8,            # 안정적 배치 크기
        'replay_buffer_size': 500,
        'patience': 8
    }
    
    CWRU_CONFIG = {
        'name': 'CWRU_Scenario2_VaryingLoad_v2',
        'data_dir': 'data_scenario2',
        'dataset_type': 'cwru',
        'domain_order': [0, 1, 2, 3],
        'domain_names': ['0HP', '1HP', '2HP', '3HP'],
        'shift_type': 'Varying Load',
        'first_domain_epochs': 15,
        'remaining_epochs': 6,
        'batch_size': 4,            # 16 → 4 (극소 데이터 대응)
        'replay_buffer_size': 50,   # 200 → 50 (작은 버퍼)
        'patience': 5
    }


def run_single_scenario_v2(config: Dict, logger: logging.Logger, device: torch.device, args, experiment_dir: str) -> Dict:
    """단일 시나리오 실행 (v2)"""
    logger.info(f"🚀 {config['name']} 시작!")
    logger.info(f"   아키텍처: Ranking-based (InfoNCE 대신 Triplet Loss)")
    logger.info(f"   Domains: {' → '.join(config['domain_names'])}")
    
    start_time = time.time()
    
    try:
        # Trainer v2 생성
        trainer = ContinualTrainer(
            device=device,
            save_dir=f"checkpoints_v2/{config['name']}",
            domain_order=config['domain_order'],
            data_dir=config['data_dir'],
            dataset_type=config['dataset_type']
        )
        
        # 하이퍼파라미터 설정
        trainer.batch_size = config['batch_size']
        trainer.replay_buffer.buffer_size_per_domain = config['replay_buffer_size']
        
        # First Domain Training
        logger.info("📚 First Domain Training v2...")
        
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
        logger.info("🔄 Remaining Domains Training v2...")
        
        domain_loaders = create_cached_domain_dataloaders(
            data_dir=config['data_dir'],
            domain_order=config['domain_order'],
            dataset_type=config['dataset_type'],
            batch_size=config['batch_size']
        )
        
        remaining_results = trainer.train_remaining_domains(domain_loaders)
        
        # 결과 정리
        final_metrics = remaining_results['final_metrics']
        total_time = time.time() - start_time
        
        results = {
            'domain_names': config['domain_names'],
            'shift_type': config['shift_type'],
            'final_accuracies': final_metrics['final_accuracies'],
            'final_top1_retrievals': final_metrics.get('final_top1_retrievals', []),
            'final_top5_retrievals': final_metrics.get('final_top5_retrievals', []),
            'average_accuracy': final_metrics['average_accuracy'],
            'average_forgetting': final_metrics['average_forgetting'],
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


def save_results_v2(results: Dict, output_dir: str) -> str:
    """결과 저장 (v2)"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(output_dir, f'results_v2_{timestamp}.json')
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results_path


def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='TextVibCLIP v2 실험')
    
    parser.add_argument('--output_dir', type=str, default='results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--quick_test', action='store_true',
                       help='빠른 테스트 모드')
    parser.add_argument('--epochs', type=int, default=None,
                       help='에포크 수')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_uos', action='store_true')
    parser.add_argument('--skip_cwru', action='store_true')
    parser.add_argument('--clear_cache', action='store_true')
    
    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = parse_arguments()
    
    # 재현성 설정
    set_seed(args.seed)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 로깅 설정
    logger, experiment_dir = setup_logging(args.output_dir)
    
    # 캐시 관리
    if args.clear_cache:
        logger.info("🗑️ 캐시 삭제 중...")
        clear_all_caches()
    
    logger.info("🎯 TextVibCLIP v2 실험 시작!")
    logger.info("   아키텍처: Ranking-based (Triplet Loss)")
    logger.info("   특징: 소규모 데이터 최적화, 실제 사용 시나리오 지원")
    
    # 디바이스 설정
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"🔧 디바이스: {device}")
    
    # 시나리오 설정
    scenarios = []
    if not args.skip_uos:
        scenarios.append(ScenarioConfig_v2.UOS_CONFIG)
    if not args.skip_cwru:
        scenarios.append(ScenarioConfig_v2.CWRU_CONFIG)
    
    if not scenarios:
        logger.error("❌ 실행할 시나리오가 없습니다!")
        return
    
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
        
        scenario_result = run_single_scenario_v2(scenario, logger, device, args, experiment_dir)
        
        if scenario_result:
            all_results[scenario['name']] = scenario_result
        else:
            logger.error(f"❌ {scenario['name']} 실행 실패!")
    
    # 결과 저장
    if all_results:
        results_path = save_results_v2(all_results, experiment_dir)
        logger.info(f"✅ 결과 저장: {results_path}")
    
    # 최종 요약
    total_time = time.time() - total_start_time
    logger.info(f"\n⏱️ 전체 실험 소요 시간: {total_time/60:.1f}분")
    
    # 성능 요약
    logger.info(f"\n📊 TextVibCLIP v2 성능 요약:")
    for scenario_name, result in all_results.items():
        avg_acc = result.get('average_accuracy', 0.0)
        avg_forget = result.get('average_forgetting', 0.0)
        logger.info(f"   {scenario_name}: 평균 정확도 {avg_acc:.4f}, 망각도 {avg_forget:.4f}")
    
    logger.info("🎉 TextVibCLIP v2 실험 완료!")


if __name__ == "__main__":
    main()
