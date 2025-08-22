#!/usr/bin/env python3
"""
TextVibCLIP 전체 시나리오 통합 실험 실행 스크립트

이 스크립트는 다음 기능을 제공합니다:
1. 시나리오 1 (UOS) + 시나리오 2 (CWRU) 자동 실행
2. 시나리오별/도메인별 성능 지표 CSV 저장
3. 종합 결과 요약 및 비교 분석
4. 실험 진행 상황 실시간 모니터링

Usage:
    python run_all_scenarios.py --output_dir results_comparison
    python run_all_scenarios.py --quick_test --epochs 10
"""

import argparse
import logging
import os
import torch
import time
import json
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import warnings

# Torchvision beta warning 비활성화
try:
    import torchvision
    torchvision.disable_beta_transforms_warning()
except:
    pass

# 기타 warning 억제
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# pandas import (CSV 저장용)
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️ pandas가 설치되지 않았습니다. CSV 저장 기능이 제한됩니다.")
    print("   설치: pip install pandas")

# 프로젝트 루트 경로 추가
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.continual_trainer import ContinualTrainer
from src.data_loader import BearingDataset, create_domain_dataloaders
from src.textvib_model import create_textvib_model
from src.utils import set_seed
from src.visualization import create_visualizer
from configs.model_config import TRAINING_CONFIG, DATA_CONFIG, CWRU_DATA_CONFIG

# 로깅 설정
def setup_logging(log_dir: str) -> logging.Logger:
    """통합 실험용 로깅 설정"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"all_scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)
    
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
    logger.info(f"통합 실험 로깅 시작: {log_path}")
    return logger


class ScenarioConfig:
    """시나리오별 설정 관리"""
    
    UOS_CONFIG = {
        'name': 'UOS_Scenario1_VaryingSpeed',
        'data_dir': 'data_scenario1',
        'dataset_type': 'uos',
        'domain_order': [600, 800, 1000, 1200, 1400, 1600],
        'domain_names': ['600RPM', '800RPM', '1000RPM', '1200RPM', '1400RPM', '1600RPM'],
        'shift_type': 'Varying Speed',
        'first_domain_epochs': 30,  # 데이터 많음
        'remaining_epochs': 20,
        'batch_size': 32,
        'replay_buffer_size': 1000,
        'patience': 5
    }
    
    CWRU_CONFIG = {
        'name': 'CWRU_Scenario2_VaryingLoad',
        'data_dir': 'data_scenario2',
        'dataset_type': 'cwru',
        'domain_order': [0, 1, 2, 3],
        'domain_names': ['0HP', '1HP', '2HP', '3HP'],
        'shift_type': 'Varying Load',
        'first_domain_epochs': 80,  # 데이터 적음
        'remaining_epochs': 50,
        'batch_size': 8,
        'replay_buffer_size': 200,
        'patience': 15
    }


class ExperimentResults:
    """실험 결과 관리 클래스"""
    
    def __init__(self):
        self.scenario_results = {}
        self.detailed_results = []
        self.summary_results = []
    
    def add_scenario_result(self, scenario_name: str, results: Dict):
        """시나리오 결과 추가"""
        self.scenario_results[scenario_name] = results
        
        # 상세 결과 추가 (도메인별)
        for domain_idx, domain_name in enumerate(results['domain_names']):
            if domain_idx < len(results['final_accuracies']):
                self.detailed_results.append({
                    'Scenario': scenario_name,
                    'Domain_Index': domain_idx + 1,
                    'Domain_Name': domain_name,
                    'Shift_Type': results['shift_type'],
                    'Accuracy': results['final_accuracies'][domain_idx],
                    'Top1_Retrieval': results.get('final_top1_retrievals', [0] * len(results['domain_names']))[domain_idx],
                    'Top5_Retrieval': results.get('final_top5_retrievals', [0] * len(results['domain_names']))[domain_idx],
                    'Samples_Per_Domain': results.get('samples_per_domain', 0),
                    'Total_Training_Time': results.get('total_time', 0)
                })
        
        # 요약 결과 추가 (시나리오별)
        self.summary_results.append({
            'Scenario': scenario_name,
            'Shift_Type': results['shift_type'],
            'Num_Domains': len(results['domain_names']),
            'Avg_Accuracy': results.get('average_accuracy', 0),
            'Avg_Forgetting': results.get('average_forgetting', 0),
            'Avg_Top1_Retrieval': np.mean(results.get('final_top1_retrievals', [0])),
            'Avg_Top5_Retrieval': np.mean(results.get('final_top5_retrievals', [0])),
            'Total_Samples': results.get('total_samples', 0),
            'Total_Time_Minutes': results.get('total_time', 0) / 60,
            'First_Domain_Epochs': results.get('first_domain_epochs', 0),
            'Remaining_Epochs': results.get('remaining_epochs', 0),
            'Batch_Size': results.get('batch_size', 0)
        })
    
    def save_to_csv(self, output_dir: str):
        """결과를 CSV 파일로 저장"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if not PANDAS_AVAILABLE:
            # pandas가 없으면 JSON으로 저장
            detailed_path = os.path.join(output_dir, f'detailed_results_{timestamp}.json')
            summary_path = os.path.join(output_dir, f'summary_results_{timestamp}.json')
            
            with open(detailed_path, 'w') as f:
                json.dump(self.detailed_results, f, indent=2)
            with open(summary_path, 'w') as f:
                json.dump(self.summary_results, f, indent=2)
            
            return detailed_path, summary_path, None
        
        # 1. 상세 결과 (도메인별)
        detailed_df = pd.DataFrame(self.detailed_results)
        detailed_path = os.path.join(output_dir, f'detailed_results_{timestamp}.csv')
        detailed_df.to_csv(detailed_path, index=False)
        
        # 2. 요약 결과 (시나리오별)
        summary_df = pd.DataFrame(self.summary_results)
        summary_path = os.path.join(output_dir, f'summary_results_{timestamp}.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # 3. 비교 결과 (피벗 테이블)
        pivot_path = None
        if len(self.detailed_results) > 0:
            try:
                pivot_df = detailed_df.pivot_table(
                    index=['Domain_Index', 'Domain_Name'],
                    columns='Scenario',
                    values=['Accuracy', 'Top1_Retrieval', 'Top5_Retrieval'],
                    aggfunc='first'
                )
                pivot_path = os.path.join(output_dir, f'comparison_results_{timestamp}.csv')
                pivot_df.to_csv(pivot_path)
            except Exception as e:
                print(f"⚠️ 피벗 테이블 생성 실패: {e}")
        
        return detailed_path, summary_path, pivot_path


def run_single_scenario(config: Dict, logger: logging.Logger, device: torch.device) -> Dict:
    """단일 시나리오 실행"""
    logger.info(f"🚀 {config['name']} 시작!")
    logger.info(f"   Domain Shift: {config['shift_type']}")
    logger.info(f"   Domains: {' → '.join(config['domain_names'])}")
    
    start_time = time.time()
    
    try:
        # Trainer 생성
        trainer = ContinualTrainer(
            device=device,
            save_dir=f"checkpoints/{config['name']}",
            use_amp=True,
            max_grad_norm=1.0
        )
        
        # 하이퍼파라미터 설정
        trainer.batch_size = config['batch_size']
        trainer.learning_rate = 1e-4
        trainer.replay_buffer.buffer_size_per_domain = config['replay_buffer_size']
        
        # 데이터셋 정보 수집
        sample_dataset = BearingDataset(
            data_dir=config['data_dir'],
            dataset_type=config['dataset_type'],
            domain_value=config['domain_order'][0],
            subset='train'
        )
        samples_per_domain = len(sample_dataset)
        total_samples = samples_per_domain * len(config['domain_order'])
        
        logger.info(f"   데이터: {samples_per_domain:,} 샘플/도메인, 총 {total_samples:,} 샘플")
        
        # First Domain Training
        logger.info("📚 First Domain Training...")
        from src.data_loader import create_first_domain_dataloader
        
        first_loader = create_first_domain_dataloader(
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
        trainer.num_epochs = config['remaining_epochs']
        
        # 도메인별 데이터로더 생성
        from src.data_loader import create_domain_dataloaders
        domain_loaders = create_domain_dataloaders(
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
            'samples_per_domain': samples_per_domain,
            'total_samples': total_samples,
            'total_time': total_time,
            'first_domain_epochs': config['first_domain_epochs'],
            'remaining_epochs': config['remaining_epochs'],
            'batch_size': config['batch_size']
        }
        
        logger.info(f"✅ {config['name']} 완료!")
        logger.info(f"   평균 정확도: {final_metrics['average_accuracy']:.4f}")
        logger.info(f"   평균 망각도: {final_metrics['average_forgetting']:.4f}")
        logger.info(f"   소요 시간: {total_time/60:.1f}분")
        
        # 도메인별 임베딩 수집 (시각화용)
        logger.info("📊 시각화용 임베딩 수집 중...")
        domain_embeddings = {}
        
        for domain in config['domain_order']:
            test_dataset = BearingDataset(
                data_dir=config['data_dir'],
                dataset_type=config['dataset_type'],
                domain_value=domain,
                subset='test'
            )
            
            if len(test_dataset) > 0:
                # 샘플링 (시각화용으로 적당한 수만)
                max_viz_samples = min(100, len(test_dataset))
                indices = torch.randperm(len(test_dataset))[:max_viz_samples]
                
                text_embeddings = []
                vib_embeddings = []
                metadata_list = []
                
                trainer.model.eval()
                with torch.no_grad():
                    for idx in indices:
                        sample = test_dataset[idx]
                        batch = {
                            'vibration': sample['vibration'].unsqueeze(0).to(device),
                            'text': [sample['text']]
                        }
                        
                        model_results = trainer.model(batch, return_embeddings=True)
                        text_embeddings.append(model_results['text_embeddings'])
                        vib_embeddings.append(model_results['vib_embeddings'])
                        metadata_list.append(sample['metadata'])
                
                if text_embeddings:
                    domain_embeddings[domain] = {
                        'text_embeddings': torch.cat(text_embeddings, dim=0),
                        'vib_embeddings': torch.cat(vib_embeddings, dim=0),
                        'metadata': metadata_list
                    }
        
        results['domain_embeddings'] = domain_embeddings
        
        return results
        
    except Exception as e:
        logger.error(f"❌ {config['name']} 실행 중 오류: {str(e)}")
        logger.exception("상세 오류 정보:")
        return None


def print_final_summary(results: ExperimentResults, logger: logging.Logger):
    """최종 결과 요약 출력"""
    logger.info("\n" + "="*80)
    logger.info("🎉 전체 시나리오 실험 완료!")
    logger.info("="*80)
    
    # 시나리오별 요약
    for summary in results.summary_results:
        logger.info(f"\n📊 {summary['Scenario']}:")
        logger.info(f"   Shift Type: {summary['Shift_Type']}")
        logger.info(f"   Domains: {summary['Num_Domains']}개")
        logger.info(f"   Avg Accuracy: {summary['Avg_Accuracy']:.4f}")
        logger.info(f"   Avg Forgetting: {summary['Avg_Forgetting']:.4f}")
        logger.info(f"   Total Time: {summary['Total_Time_Minutes']:.1f}분")
        logger.info(f"   Total Samples: {summary['Total_Samples']:,}개")
    
    # 비교 분석
    if len(results.summary_results) >= 2:
        uos_result = results.summary_results[0]
        cwru_result = results.summary_results[1]
        
        logger.info(f"\n🔍 시나리오 비교:")
        logger.info(f"   정확도 차이: {abs(uos_result['Avg_Accuracy'] - cwru_result['Avg_Accuracy']):.4f}")
        logger.info(f"   망각도 차이: {abs(uos_result['Avg_Forgetting'] - cwru_result['Avg_Forgetting']):.4f}")
        logger.info(f"   데이터 규모 비율: {uos_result['Total_Samples'] / cwru_result['Total_Samples']:.1f}:1")
    
    logger.info("="*80)


def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='TextVibCLIP 전체 시나리오 통합 실험')
    
    parser.add_argument('--output_dir', type=str, default='results_comparison',
                       help='결과 저장 디렉토리')
    parser.add_argument('--quick_test', action='store_true',
                       help='빠른 테스트 모드 (에포크 수 감소)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='에포크 수 (quick_test 모드용)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'], help='학습 디바이스')
    parser.add_argument('--seed', type=int, default=42,
                       help='재현성을 위한 시드값')
    parser.add_argument('--skip_uos', action='store_true',
                       help='UOS 시나리오 건너뛰기')
    parser.add_argument('--skip_cwru', action='store_true',
                       help='CWRU 시나리오 건너뛰기')
    
    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = parse_arguments()
    
    # 재현성 설정
    set_seed(args.seed)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 로깅 설정
    logger = setup_logging(args.output_dir)
    logger.info("🎯 TextVibCLIP 전체 시나리오 통합 실험 시작!")
    logger.info(f"📁 결과 저장 경로: {args.output_dir}")
    
    # 디바이스 설정
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"🔧 디바이스: {device}")
    if device.type == 'cuda':
        logger.info(f"   GPU: {torch.cuda.get_device_name()}")
    
    # 실험 결과 관리자
    results = ExperimentResults()
    
    # 시나리오 설정
    scenarios = []
    if not args.skip_uos:
        scenarios.append(ScenarioConfig.UOS_CONFIG)
    if not args.skip_cwru:
        scenarios.append(ScenarioConfig.CWRU_CONFIG)
    
    if not scenarios:
        logger.error("❌ 실행할 시나리오가 없습니다!")
        return
    
    # Quick test 모드 설정
    if args.quick_test:
        test_epochs = args.epochs if args.epochs else 5
        logger.info(f"⚡ 빠른 테스트 모드: 에포크 수를 {test_epochs}로 축소")
        for scenario in scenarios:
            scenario['first_domain_epochs'] = test_epochs
            scenario['remaining_epochs'] = test_epochs // 2
    
    # 시나리오별 실행
    total_start_time = time.time()
    
    for i, scenario in enumerate(scenarios, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"시나리오 {i}/{len(scenarios)}: {scenario['name']}")
        logger.info(f"{'='*60}")
        
        scenario_result = run_single_scenario(scenario, logger, device)
        
        if scenario_result:
            results.add_scenario_result(scenario['name'], scenario_result)
        else:
            logger.error(f"❌ {scenario['name']} 실행 실패!")
    
    # 결과 저장
    logger.info("\n💾 결과 저장 중...")
    try:
        detailed_path, summary_path, comparison_path = results.save_to_csv(args.output_dir)
        logger.info(f"✅ 상세 결과: {detailed_path}")
        logger.info(f"✅ 요약 결과: {summary_path}")
        logger.info(f"✅ 비교 결과: {comparison_path}")
    except Exception as e:
        logger.error(f"❌ 결과 저장 실패: {str(e)}")
    
    # 고급 시각화 생성
    logger.info("\n🎨 논문용 시각화 생성 중...")
    try:
        visualizer = create_visualizer(args.output_dir)
        
        # 시나리오별 결과 정리
        scenario_summary = {}
        domain_embeddings = {}
        
        for scenario_result in results.scenario_results.values():
            if 'domain_embeddings' in scenario_result:
                scenario_name = scenario_result.get('shift_type', 'Unknown')
                scenario_summary[scenario_name] = scenario_result
                domain_embeddings[scenario_name] = scenario_result['domain_embeddings']
        
        # 논문용 Figure 생성
        if scenario_summary and domain_embeddings:
            figure_paths = visualizer.create_paper_figures(
                scenario_summary, 
                domain_embeddings,
                output_prefix="TextVibCLIP"
            )
            
            logger.info(f"✅ 논문용 Figure {len(figure_paths)}개 생성 완료!")
            for path in figure_paths:
                logger.info(f"   📊 {Path(path).name}")
        else:
            logger.warning("⚠️ 시각화용 데이터가 부족합니다.")
            
    except Exception as e:
        logger.error(f"❌ 시각화 생성 실패: {str(e)}")
        logger.exception("상세 오류:")
    
    # 최종 요약
    total_time = time.time() - total_start_time
    logger.info(f"\n⏱️ 전체 실험 소요 시간: {total_time/60:.1f}분")
    
    print_final_summary(results, logger)
    
    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("🎉 모든 실험이 성공적으로 완료되었습니다!")


if __name__ == "__main__":
    main()
