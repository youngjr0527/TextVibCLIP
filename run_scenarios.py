#!/usr/bin/env python3
"""
TextVibCLIP 실험 스크립트
Triplet ranking loss 기반 아키텍처로 소규모 데이터에 최적화

Usage:
    python run_scenarios.py --quick_test --epochs 10
    python run_scenarios.py --skip_uos  # CWRU만
    python run_scenarios.py --skip_cwru # UOS만
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

# 🎯 재현성 보장을 위한 시드 고정 (중복 제거)
def set_random_seeds(seed: int = 42):
    """모든 랜덤 시드 고정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 환경
    np.random.seed(seed)
    random.seed(seed)
    
    # 추가적인 재현성 보장
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎯 Random seeds fixed to {seed} for reproducibility")

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
    experiment_dir = os.path.join(log_dir, f"{experiment_timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    log_filename = f"textvibclip_{experiment_timestamp}.log"
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
    logger.info(f"TextVibCLIP 실험 시작: {log_path}")
    logger.info(f"실험 결과 폴더: {experiment_dir}")
    
    return logger, experiment_dir


class ScenarioConfig:
    """시나리오별 설정"""
    
    UOS_CONFIG = {
        'name': 'UOS_Scenario1_VaryingSpeed',
        'data_dir': 'data_scenario1',
        'dataset_type': 'uos',
        'domain_order': [600, 800, 1000, 1200, 1400, 1600],
        'domain_names': ['600RPM', '800RPM', '1000RPM', '1200RPM', '1400RPM', '1600RPM'],
        'shift_type': 'Varying Speed',
        'first_domain_epochs': 20,  # 15 → 20 (더 안정적인 기초 학습)
        'remaining_epochs': 8,      # 6 → 8 (균형잡힌 적응 학습)
        'batch_size': 16,           # 8 → 16 (더 안정적인 그래디언트)
        'replay_buffer_size': 800,  # 1000 → 800 (최적화된 메모리 사용)
        'patience': 10              # 8 → 10 (더 여유있는 조기 종료)
    }
    
    CWRU_CONFIG = {
        'name': 'CWRU_Scenario2_VaryingLoad',
        'data_dir': 'data_scenario2',
        'dataset_type': 'cwru',
        'domain_order': [0, 1, 2, 3],
        'domain_names': ['0HP', '1HP', '2HP', '3HP'],
        'shift_type': 'Varying Load',
        'first_domain_epochs': 15,
        'remaining_epochs': 6,
        'batch_size': 4,            #  (극소 데이터 대응)
        'replay_buffer_size': 100,   
        'patience': 5
    }


def run_single_scenario(config: Dict, logger: logging.Logger, device: torch.device, args, experiment_dir: str) -> Dict:
    """단일 시나리오 실행"""
    logger.info(f"🚀 {config['name']} 시작!")
    logger.info(f"   아키텍처: Ranking-based (Triplet Loss)")
    logger.info(f"   Domains: {' → '.join(config['domain_names'])}")
    
    start_time = time.time()
    
    try:
        # Trainer 생성 (실험별 독립 체크포인트 디렉토리)
        # 재현성 보장: 각 실험이 독립적인 체크포인트 사용
        checkpoint_dir = os.path.join(experiment_dir, 'checkpoints', config['name'])
        
        # Replay-free 실험인 경우 완전히 새로운 모델로 시작
        if 'ReplayFree' in config['name']:
            logger.info("🔄 Replay-free 실험: 모델 완전 초기화")
            # 기존 모델이 있다면 완전히 제거하고 새로 생성
            trainer = ContinualTrainer(
                model=None,  # None으로 설정하여 완전히 새로운 모델 생성
                device=device,
                save_dir=checkpoint_dir,
                domain_order=config['domain_order'],
                data_dir=config['data_dir'],
                dataset_type=config['dataset_type'],
                results_save_dir=None
            )
        else:
            trainer = ContinualTrainer(
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
        
        # 남도메인 에폭/설정 강제 반영 (시나리오 설정과 trainer 내부 특수설정 불일치 해소)
        try:
            from configs.model_config import CWRU_SPECIFIC_CONFIG, CONTINUAL_CONFIG
            if config['dataset_type'] == 'cwru':
                CWRU_SPECIFIC_CONFIG['num_epochs'] = max(1, int(config.get('remaining_epochs', 3)))
            else:
                CONTINUAL_CONFIG['num_epochs'] = max(1, int(config.get('remaining_epochs', 3)))
        except Exception:
            pass

        remaining_results = trainer.train_remaining_domains(domain_loaders)

        # 🔎 시각화: 각 도메인의 test 임베딩 수집 후 PNG 저장
        try:
            # Replay-free 실험인 경우 시각화도 replay_free 디렉토리에 저장
            viz_dir = experiment_dir
            if 'ReplayFree' in config['name']:
                replay_free_dir = os.path.join(experiment_dir, 'replay_free')
                os.makedirs(replay_free_dir, exist_ok=True)
                viz_dir = replay_free_dir
            
            visualizer = create_visualizer(viz_dir)
            for domain_value in config['domain_order']:
                if domain_value not in domain_loaders:
                    continue
                test_loader = domain_loaders[domain_value]['test']
                emb = trainer._collect_domain_embeddings(test_loader)
                if not emb:
                    continue

                text_emb = emb.get('text_embeddings')
                vib_emb = emb.get('vib_embeddings')
                metadata_list = emb.get('metadata', [])

                # 라벨/베어링타입 추출
                labels = [m.get('bearing_condition', 'H') for m in metadata_list]
                # CWRU는 베어링 타입 라벨을 고정 표시(혼동 방지)
                if config['dataset_type'] == 'cwru':
                    bearing_types = ['CWRU'] * len(metadata_list)
                else:
                    bearing_types = [m.get('bearing_type', '6204') for m in metadata_list]

                domain_name = f"{domain_value}HP" if config['dataset_type'] == 'cwru' else f"{domain_value}RPM"

                # 정합 시각화: UOS/CWRU 공통 - 평가 기준과 동일하게 텍스트 프로토타입 사용(UOS는 7클래스)
                if vib_emb is not None and labels:
                    try:
                        if config['dataset_type'] == 'cwru':
                            prompt_bank = {
                                0: ["healthy bearing","normal bearing with no fault","bearing vibration without defect"],
                                1: ["bearing with ball fault","ball defect in bearing","ball damage on bearing"],
                                2: ["bearing inner race fault","inner ring defect in bearing","inner race damage of bearing"],
                                3: ["bearing outer race fault","outer ring defect in bearing","outer race damage of bearing"]
                            }
                            class_ids = [0,1,2,3]
                            label_map = {'H': 0, 'B': 1, 'IR': 2, 'OR': 3}
                        else:
                            # UOS 7클래스 프롬프트 (간결 버전)
                            prompt_bank = {
                                0: ["healthy bearing"],          # H_H
                                1: ["bearing with ball fault"],  # H_B
                                2: ["inner race fault"],         # H_IR
                                3: ["outer race fault"],         # H_OR
                                4: ["mechanical looseness"],     # L_H
                                5: ["rotor unbalance"],          # U_H
                                6: ["shaft misalignment"]        # M_H
                            }
                            class_ids = [0,1,2,3,4,5,6]
                            label_map = {'H_H':0,'H_B':1,'H_IR':2,'H_OR':3,'L_H':4,'U_H':5,'M_H':6}

                        # 클래스 프로토타입 임베딩 계산
                        class_protos = []
                        for cls_id in class_ids:
                            texts = prompt_bank[cls_id]
                            raw = trainer.model.text_encoder.encode_texts(texts, device)
                            proj = F.normalize(trainer.model.text_projection(raw), p=2, dim=1)
                            proto = F.normalize(proj.mean(dim=0, keepdim=True), p=2, dim=1)
                            class_protos.append(proto)
                        proto_mat = torch.cat(class_protos, dim=0)

                        # UOS 라벨 문자열 구성
                        if config['dataset_type'] == 'uos':
                            # 기존 labels는 'bearing_condition'만일 수 있음 → metadata로 조합 라벨 생성
                            rc = [m.get('rotating_component','H') for m in metadata_list]
                            bc = [m.get('bearing_condition','H') for m in metadata_list]
                            labels = [f"{r}_{b}" for r,b in zip(rc, bc)]

                        idx = torch.tensor([label_map.get(l, 0) for l in labels], device=proto_mat.device)
                        # 각 샘플 라벨에 해당하는 프로토타입을 텍스트 임베딩으로 사용
                        text_emb = proto_mat.index_select(0, idx)
                    except Exception as _e:
                        logger.warning(f"CWRU 프로토타입 기반 텍스트 임베딩 생성 실패: {_e}")

                # Per-domain encoder alignment t-SNE
                visualizer.create_encoder_alignment_plot(
                    text_embeddings=text_emb,
                    vib_embeddings=vib_emb,
                    labels=labels,
                    bearing_types=bearing_types,
                    domain_name=domain_name,
                    save_name="encoder_alignment"
                )

                # Similarity diagnostics (per-domain)
                if config['dataset_type'] == 'cwru':
                    try:
                        # 평가 프롬프트 프로토타입을 재사용
                        prompt_bank = {
                            0: [
                                "healthy bearing",
                                "normal bearing with no fault",
                                "bearing vibration without defect"
                            ],
                            1: [
                                "bearing with ball fault",
                                "ball defect in bearing",
                                "ball damage on bearing"
                            ],
                            2: [
                                "bearing inner race fault",
                                "inner ring defect in bearing",
                                "inner race damage of bearing"
                            ],
                            3: [
                                "bearing outer race fault",
                                "outer ring defect in bearing",
                                "outer race damage of bearing"
                            ]
                        }
                        class_protos = []
                        for cls_id in [0, 1, 2, 3]:
                            texts = prompt_bank[cls_id]
                            raw = trainer.model.text_encoder.encode_texts(texts, device)
                            proj = F.normalize(trainer.model.text_projection(raw), p=2, dim=1)
                            proto = F.normalize(proj.mean(dim=0, keepdim=True), p=2, dim=1)
                            class_protos.append(proto)
                        proto_mat = torch.cat(class_protos, dim=0)

                        pass
                    except Exception as _e:
                        logger.warning(f"시각화 실패: {_e}")
        except Exception as viz_err:
            logger.warning(f"시각화 생성에 실패했습니다: {viz_err}")
        
        #  추가 시각화 생성
        try:
            logger.info("📊  시각화 생성 중...")
            
            # Continual Learning Performance Curve
            visualizer.create_continual_learning_curve(
                domain_names=config['domain_names'],
                accuracies=remaining_results['final_metrics']['final_accuracies'],
                scenario_name=config['name']
            )
            
            # Forgetting Analysis Heatmap (실제 performance_history 사용)
            # Heatmap[i, j] = i번째 학습 단계 후, j번째 test domain 정확도
            # 위쪽 삼각형만 값 있음 (j <= i, 이미 학습한 도메인만)
            n_domains = len(config['domain_names'])
            accuracy_matrix = np.full((n_domains, n_domains), np.nan)
            
            # trainer.performance_history에서 실제 데이터 추출
            for i in range(n_domains):
                # i번째 학습 단계 (0~i번째 도메인까지 학습 완료)
                for j in range(n_domains):
                    # j번째 test domain
                    if j <= i:  # 이미 학습한 도메인만 (위쪽 삼각형)
                        test_domain = config['domain_order'][j]
                        if test_domain in trainer.performance_history:
                            history = trainer.performance_history[test_domain]['accuracy']
                            # j번째 도메인은 j번째 단계부터 평가됨
                            # i번째 단계에서의 인덱스 = i - j
                            history_idx = i - j
                            if len(history) > history_idx:
                                accuracy_matrix[i, j] = history[history_idx]

            visualizer.create_forgetting_heatmap(
                domain_names=config['domain_names'],
                accuracy_matrix=accuracy_matrix,
                scenario_name=config['name']
            )
            
            logger.info("✅  시각화 생성 완료!")
        except Exception as paper_viz_err:
            logger.warning(f" 시각화 생성 실패: {paper_viz_err}")
        
        # Replay-free 실험인 경우 별도 디렉토리에 저장
        if 'ReplayFree' in config['name']:
            replay_free_dir = os.path.join(experiment_dir, 'replay_free')
            os.makedirs(replay_free_dir, exist_ok=True)
            
            # 실험 설정 저장 (replay_free 디렉토리)
            config_path = save_experiment_config(config, trainer, replay_free_dir, device)
            logger.info(f"📝 실험 설정 저장 (replay-free): {config_path}")
            
            # 결과를 replay_free 디렉토리에 저장하도록 설정
            experiment_dir = replay_free_dir
        else:
            # 실험 설정 저장 (기본 디렉토리)
            config_path = save_experiment_config(config, trainer, experiment_dir, device)
            logger.info(f"📝 실험 설정 저장: {config_path}")
        
        # 결과 정리 (Heatmap 데이터 포함)
        final_metrics = remaining_results['final_metrics']
        total_time = time.time() - start_time
        
        # 🎯 Forgetting Heatmap 데이터 추출 (JSON 저장용)
        n_domains = len(config['domain_names'])
        heatmap_matrix = []
        stage_averages = []
        
        for i in range(n_domains):
            row = []
            for j in range(n_domains):
                if j <= i:  # 학습한 도메인만
                    test_domain = config['domain_order'][j]
                    if test_domain in trainer.performance_history:
                        history = trainer.performance_history[test_domain]['accuracy']
                        history_idx = i - j
                        if len(history) > history_idx:
                            row.append(round(history[history_idx] * 100, 2))  # 퍼센트
                        else:
                            row.append(None)
                    else:
                        row.append(None)
                else:
                    row.append(None)  # 아직 학습 안함
            
            # 각 행의 평균 계산
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
            # 🎯 주요 결과: Stage별 평균 (Heatmap 각 행 평균)
            'stage_accuracies': stage_averages,  # 이게 핵심!
            'average_accuracy': final_metrics['average_accuracy'],
            'average_forgetting': final_metrics['average_forgetting'],
            # 🎯 Forgetting Heatmap 전체 데이터
            'forgetting_matrix': heatmap_matrix,
            # 참고용 (논문에는 사용 안함)
            'final_top1_retrievals': final_metrics.get('final_top1_retrievals', []),
            'final_top5_retrievals': final_metrics.get('final_top5_retrievals', []),
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
    """Save experiment configuration to txt file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_path = os.path.join(output_dir, f'experiment_config_{timestamp}.txt')
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TextVibCLIP Experiment Configuration\n")
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
        try:
            from configs.model_config import MODEL_CONFIG
            f.write(f"embedding_dim: {MODEL_CONFIG['embedding_dim']}\n")
            f.write(f"text_dim: {MODEL_CONFIG['text_dim']}\n")
            f.write(f"vibration_input_dim: {MODEL_CONFIG['vibration_input_dim']}\n")
            f.write(f"vibration_input_length: {MODEL_CONFIG['vibration_encoder']['input_length']}\n")
            f.write(f"vibration_kernel_sizes: {MODEL_CONFIG['vibration_encoder']['kernel_sizes']}\n")
            f.write(f"vibration_channels: {MODEL_CONFIG['vibration_encoder']['channels']}\n")
            f.write(f"vibration_stride: {MODEL_CONFIG['vibration_encoder']['stride']}\n")
            f.write(f"dropout: {MODEL_CONFIG['vibration_encoder']['dropout']}\n")
            f.write(f"activation: {MODEL_CONFIG['vibration_encoder']['activation']}\n")
            f.write(f"normalization: {MODEL_CONFIG['vibration_encoder']['normalization']}\n")
            f.write(f"pooling: {MODEL_CONFIG['vibration_encoder']['pooling']}\n")
            
            # Text Encoder Configuration
            f.write(f"\ntext_encoder_model: {MODEL_CONFIG['text_encoder']['model_name']}\n")
            f.write(f"lora_rank: {MODEL_CONFIG['text_encoder']['lora_config']['r']}\n")
            f.write(f"lora_alpha: {MODEL_CONFIG['text_encoder']['lora_config']['lora_alpha']}\n")
            f.write(f"lora_target_modules: {MODEL_CONFIG['text_encoder']['lora_config']['target_modules']}\n")
            f.write(f"lora_dropout: {MODEL_CONFIG['text_encoder']['lora_config']['lora_dropout']}\n")
            
            # Projection Layers
            f.write(f"\nprojection_hidden_dim: {MODEL_CONFIG['projection']['hidden_dim']}\n")
            f.write(f"projection_output_dim: {MODEL_CONFIG['projection']['output_dim']}\n")
            f.write(f"projection_dropout: {MODEL_CONFIG['projection']['dropout']}\n")
            
            # Ranking Loss
            f.write(f"\nranking_margin: {MODEL_CONFIG['ranking_loss']['margin']}\n")
            f.write(f"ranking_loss_type: {MODEL_CONFIG['ranking_loss']['loss_type']}\n")
            
            # Auxiliary Classification
            f.write(f"\naux_classification_enabled: {MODEL_CONFIG['aux_classification']['enabled']}\n")
            f.write(f"aux_num_classes: {MODEL_CONFIG['aux_classification']['num_classes']}\n")
            f.write(f"aux_loss_weight: {MODEL_CONFIG['aux_classification']['loss_weight']}\n")
            f.write(f"aux_dropout: {MODEL_CONFIG['aux_classification']['dropout']}\n")
        except Exception as e:
            f.write(f"Model config loading failed: {e}\n")
        f.write("\n")
        
        # Training Configuration
        f.write("-" * 50 + "\n")
        f.write("Training Configuration\n")
        f.write("-" * 50 + "\n")
        try:
            from configs.model_config import FIRST_DOMAIN_CONFIG, CONTINUAL_CONFIG, CWRU_SPECIFIC_CONFIG, CWRU_FIRST_DOMAIN_CONFIG
            
            if config['dataset_type'] == 'cwru':
                f.write("CWRU-specific configuration:\n")
                f.write(f"  first_domain_epochs: {CWRU_FIRST_DOMAIN_CONFIG['num_epochs']}\n")
                f.write(f"  first_domain_lr: {CWRU_FIRST_DOMAIN_CONFIG['learning_rate']}\n")
                f.write(f"  first_domain_weight_decay: {CWRU_FIRST_DOMAIN_CONFIG['weight_decay']}\n")
                f.write(f"  first_domain_aux_weight: {CWRU_FIRST_DOMAIN_CONFIG['aux_weight']}\n")
                f.write(f"  first_domain_patience: {CWRU_FIRST_DOMAIN_CONFIG['patience']}\n")
                f.write(f"  remaining_epochs: {CWRU_SPECIFIC_CONFIG['num_epochs']}\n")
                f.write(f"  remaining_lr: {CWRU_SPECIFIC_CONFIG['learning_rate']}\n")
                f.write(f"  remaining_weight_decay: {CWRU_SPECIFIC_CONFIG['weight_decay']}\n")
                f.write(f"  remaining_aux_weight: {CWRU_SPECIFIC_CONFIG['aux_weight']}\n")
                f.write(f"  remaining_patience: {CWRU_SPECIFIC_CONFIG['patience']}\n")
            else:
                f.write("UOS standard configuration:\n")
                f.write(f"  first_domain_epochs: {FIRST_DOMAIN_CONFIG['num_epochs']}\n")
                f.write(f"  first_domain_lr: {FIRST_DOMAIN_CONFIG['learning_rate']}\n")
                f.write(f"  first_domain_weight_decay: {FIRST_DOMAIN_CONFIG['weight_decay']}\n")
                f.write(f"  first_domain_aux_weight: {FIRST_DOMAIN_CONFIG['aux_weight']}\n")
                f.write(f"  first_domain_patience: {FIRST_DOMAIN_CONFIG['patience']}\n")
                f.write(f"  first_domain_min_epochs: {FIRST_DOMAIN_CONFIG['min_epoch']}\n")
                f.write(f"  remaining_epochs: {CONTINUAL_CONFIG['num_epochs']}\n")
                f.write(f"  remaining_lr: {CONTINUAL_CONFIG['learning_rate']}\n")
                f.write(f"  remaining_weight_decay: {CONTINUAL_CONFIG['weight_decay']}\n")
                f.write(f"  remaining_aux_weight: {CONTINUAL_CONFIG['aux_weight']}\n")
                f.write(f"  remaining_patience: {CONTINUAL_CONFIG['patience']}\n")
                f.write(f"  remaining_min_epochs: {CONTINUAL_CONFIG['min_epoch']}\n")
        except Exception as e:
            f.write(f"Training config loading failed: {e}\n")
        f.write("\n")
        
        # Replay Buffer Configuration
        f.write("-" * 50 + "\n")
        f.write("Replay Buffer Configuration\n")
        f.write("-" * 50 + "\n")
        f.write(f"buffer_size_per_domain: {trainer.replay_buffer.buffer_size_per_domain}\n")
        f.write(f"embedding_dim: {trainer.replay_buffer.embedding_dim}\n")
        f.write(f"sampling_strategy: {trainer.replay_buffer.sampling_strategy}\n")
        try:
            from configs.model_config import CONTINUAL_CONFIG
            f.write(f"replay_ratio: {CONTINUAL_CONFIG.get('replay_ratio', 'N/A')}\n")
            f.write(f"replay_every_n: {CONTINUAL_CONFIG.get('replay_every_n', 'N/A')}\n")
            f.write(f"replay_selection: {CONTINUAL_CONFIG.get('replay_selection', 'N/A')}\n")
        except Exception:
            pass
        f.write("\n")
        
        # Data Configuration
        f.write("-" * 50 + "\n")
        f.write("Data Configuration\n")
        f.write("-" * 50 + "\n")
        try:
            from configs.model_config import DATA_CONFIG, CWRU_DATA_CONFIG
            data_config = CWRU_DATA_CONFIG if config['dataset_type'] == 'cwru' else DATA_CONFIG
            f.write(f"window_size: {data_config['window_size']}\n")
            f.write(f"overlap_ratio: {data_config['overlap_ratio']}\n")
            f.write(f"signal_normalization: {data_config['signal_normalization']}\n")
            f.write(f"validation_split: {data_config['validation_split']}\n")
            f.write(f"test_split: {data_config['test_split']}\n")
            f.write(f"max_text_length: {data_config['max_text_length']}\n")
        except Exception as e:
            f.write(f"Data config loading failed: {e}\n")
        f.write("\n")
        
        # Reproducibility Configuration
        f.write("-" * 50 + "\n")
        f.write("Reproducibility Configuration\n")
        f.write("-" * 50 + "\n")
        f.write(f"pytorch_seed: {torch.initial_seed()}\n")
        f.write(f"numpy_seed: {np.random.get_state()[1][0] if hasattr(np.random, 'get_state') else 'N/A'}\n")
        f.write(f"random_seed: {random.getstate()[1][0] if hasattr(random, 'getstate') else 'N/A'}\n")
        f.write(f"cuda_seed: {torch.cuda.initial_seed() if torch.cuda.is_available() else 'N/A'}\n")
        f.write(f"cudnn_deterministic: {torch.backends.cudnn.deterministic}\n")
        f.write(f"cudnn_benchmark: {torch.backends.cudnn.benchmark}\n")
        f.write("\n")
        
        # Checkpoint Information
        f.write("-" * 50 + "\n")
        f.write("Checkpoint Information\n")
        f.write("-" * 50 + "\n")
        f.write(f"save_dir: {trainer.save_dir}\n")
        f.write(f"max_grad_norm: {trainer.max_grad_norm}\n")
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
            f.write(f"gpu_count: {torch.cuda.device_count()}\n")
            f.write(f"current_gpu: {torch.cuda.current_device()}\n")
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
    parser = argparse.ArgumentParser(description='TextVibCLIP 실험')
    
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
    
    # 재현성 설정 (중복 제거 - set_random_seeds 사용)
    set_random_seeds(args.seed)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 로깅 설정
    logger, experiment_dir = setup_logging(args.output_dir)
    
    # 캐시 관리
    if args.clear_cache:
        logger.info("🗑️ 캐시 삭제 중...")
        clear_all_caches()
    
    logger.info("🎯 TextVibCLIP 실험 시작!")
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
        scenarios.append(ScenarioConfig.UOS_CONFIG)
    if not args.skip_cwru:
        scenarios.append(ScenarioConfig.CWRU_CONFIG)
    
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
    
    # 시나리오별 실행 (기존 + replay-free ablation study)
    all_results = {}
    total_start_time = time.time()
    
    for i, scenario in enumerate(scenarios, 1):
        # 1. 기존 실험 (replay buffer 사용)
        logger.info(f"\n{'='*60}")
        logger.info(f"시나리오 {i*2-1}/{len(scenarios)*2}: {scenario['name']} (with replay buffer)")
        logger.info(f"{'='*60}")
        
        scenario_result = run_single_scenario(scenario, logger, device, args, experiment_dir)
        
        if scenario_result:
            all_results[scenario['name']] = scenario_result
        else:
            logger.error(f"❌ {scenario['name']} 실행 실패!")
        
        # 2. Replay-free ablation study
        logger.info(f"\n{'='*60}")
        logger.info(f"시나리오 {i*2}/{len(scenarios)*2}: {scenario['name']} (replay-free)")
        logger.info(f"{'='*60}")
        
        # Replay-free 설정으로 시나리오 복사 및 수정
        replay_free_scenario = scenario.copy()
        replay_free_scenario['name'] = scenario['name'] + '_ReplayFree'
        replay_free_scenario['replay_buffer_size'] = 0  # Replay buffer 비활성화
        
        replay_free_result = run_single_scenario(replay_free_scenario, logger, device, args, experiment_dir)
        
        if replay_free_result:
            all_results[replay_free_scenario['name']] = replay_free_result
        else:
            logger.error(f"❌ {replay_free_scenario['name']} 실행 실패!")
    
    # 결과 저장
    if all_results:
        results_path = save_results(all_results, experiment_dir)
        logger.info(f"✅ 결과 저장: {results_path}")
    
    # 최종 요약
    total_time = time.time() - total_start_time
    logger.info(f"\n⏱️ 전체 실험 소요 시간: {total_time/60:.1f}분")
    
    # 성능 요약
    logger.info(f"\n📊 TextVibCLIP 성능 요약:")
    for scenario_name, result in all_results.items():
        avg_acc = result.get('average_accuracy', 0.0)
        avg_forget = result.get('average_forgetting', 0.0)
        logger.info(f"   {scenario_name}: 평균 정확도 {avg_acc:.4f}, 망각도 {avg_forget:.4f}")
    
    logger.info("🎉 TextVibCLIP 실험 완료!")


if __name__ == "__main__":
    main()
