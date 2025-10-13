#!/usr/bin/env python3
"""
시각화 재생성 스크립트
특정 results 디렉토리를 지정하면, 학습을 다시 수행하지 않고 현 모델로 임베딩을 수집하여
encoder alignment 및(필요 시) similarity diagnostics 이미지를 동일 폴더에 다시 저장합니다.

Usage:
  python scripts/redraw_visualizations.py --results_dir results/v2_20251001_022253
  # 선택 옵션: --only_uos / --only_cwru
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn.functional as F

from pathlib import Path

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.continual_trainer import ContinualTrainer
from src.data_cache import create_cached_domain_dataloaders
from src.visualization import create_visualizer
from run_scenarios import ScenarioConfig_v2


def build_trainer(dataset_type: str, data_dir: str, device: torch.device) -> ContinualTrainer:
    trainer = ContinualTrainer(
        device=device,
        save_dir=f"checkpoints/REDRAW_{dataset_type}",
        domain_order=(ScenarioConfig_v2.UOS_CONFIG['domain_order'] if dataset_type == 'uos' else ScenarioConfig_v2.CWRU_CONFIG['domain_order']),
        data_dir=data_dir,
        dataset_type=dataset_type
    )
    # 배치 크기 최소값 사용 (빠른 임베딩 수집)
    trainer.batch_size = 8 if dataset_type == 'uos' else 4
    return trainer


def create_cwru_prompt_prototypes(trainer: ContinualTrainer, device: torch.device) -> torch.Tensor:
    prompt_bank = {
        0: ["healthy bearing", "normal bearing with no fault", "bearing vibration without defect"],
        1: ["bearing with ball fault", "ball defect in bearing", "ball damage on bearing"],
        2: ["bearing inner race fault", "inner ring defect in bearing", "inner race damage of bearing"],
        3: ["bearing outer race fault", "outer ring defect in bearing", "outer race damage of bearing"],
    }
    class_protos = []
    for cls_id in [0, 1, 2, 3]:
        texts = prompt_bank[cls_id]
        raw = trainer.model.text_encoder.encode_texts(texts, device)
        proj = F.normalize(trainer.model.text_projection(raw), p=2, dim=1)
        proto = F.normalize(proj.mean(dim=0, keepdim=True), p=2, dim=1)
        class_protos.append(proto)
    proto_mat = torch.cat(class_protos, dim=0)
    return proto_mat


def create_uos_prompt_prototypes(trainer: ContinualTrainer, device: torch.device) -> torch.Tensor:
    # 7-클래스 간단 프롬프트
    prompt_bank = {
        0: ["healthy bearing"],          # H_H
        1: ["bearing with ball fault"],  # H_B
        2: ["inner race fault"],         # H_IR
        3: ["outer race fault"],         # H_OR
        4: ["mechanical looseness"],     # L_H
        5: ["rotor unbalance"],          # U_H
        6: ["shaft misalignment"],       # M_H
    }
    class_protos = []
    for cls_id in [0, 1, 2, 3, 4, 5, 6]:
        texts = prompt_bank[cls_id]
        raw = trainer.model.text_encoder.encode_texts(texts, device)
        proj = F.normalize(trainer.model.text_projection(raw), p=2, dim=1)
        proto = F.normalize(proj.mean(dim=0, keepdim=True), p=2, dim=1)
        class_protos.append(proto)
    proto_mat = torch.cat(class_protos, dim=0)
    return proto_mat


def redraw_cwru(results_dir: str, device: torch.device):
    logger = logging.getLogger(__name__)
    scenario = ScenarioConfig_v2.CWRU_CONFIG
    trainer = build_trainer('cwru', scenario['data_dir'], device)
    domain_loaders = create_cached_domain_dataloaders(
        data_dir=scenario['data_dir'],
        domain_order=scenario['domain_order'],
        dataset_type='cwru',
        batch_size=trainer.batch_size,
    )
    visualizer = create_visualizer(results_dir)

    proto_mat = create_cwru_prompt_prototypes(trainer, device)
    label_map = {'H': 0, 'B': 1, 'IR': 2, 'OR': 3}

    for domain_value in scenario['domain_order']:
        if domain_value not in domain_loaders:
            continue
        test_loader = domain_loaders[domain_value]['test']
        emb = trainer._collect_domain_embeddings(test_loader)
        if not emb:
            continue

        text_emb = emb.get('text_embeddings')  # 여기서는 프로토타입 기반으로 대체
        vib_emb = emb.get('vib_embeddings')
        metadata_list = emb.get('metadata', [])
        labels = [m.get('bearing_condition', 'H') for m in metadata_list]
        bearing_types = ['CWRU'] * len(metadata_list)
        domain_name = f"{domain_value}HP"

        # 텍스트 임베딩을 클래스 프로토타입으로 대체 (라벨 매핑)
        try:
            idx = torch.tensor([label_map.get(l, 0) for l in labels], device=proto_mat.device)
            text_emb = proto_mat.index_select(0, idx)
        except Exception as _e:
            logger.warning(f"CWRU 텍스트 프로토타입 생성 실패: {_e}")

        # Alignment t-SNE
        visualizer.create_encoder_alignment_plot(
            text_embeddings=text_emb,
            vib_embeddings=vib_emb,
            labels=labels,
            bearing_types=bearing_types,
            domain_name=domain_name,
            save_name="encoder_alignment"
        )

        # Similarity diagnostics
        try:
            visualizer.create_similarity_diagnostics_plot(
                vib_embeddings=vib_emb,
                labels=labels,
                prompt_embeddings=proto_mat,
                domain_name=domain_name,
                save_name="similarity_diagnostics"
            )
        except Exception as _e:
            logger.warning(f"CWRU similarity diagnostics 실패: {_e}")


def redraw_uos(results_dir: str, device: torch.device):
    logger = logging.getLogger(__name__)
    scenario = ScenarioConfig_v2.UOS_CONFIG
    trainer = build_trainer('uos', scenario['data_dir'], device)
    domain_loaders = create_cached_domain_dataloaders(
        data_dir=scenario['data_dir'],
        domain_order=scenario['domain_order'],
        dataset_type='uos',
        batch_size=trainer.batch_size,
    )
    visualizer = create_visualizer(results_dir)

    proto_mat = create_uos_prompt_prototypes(trainer, device)
    label_map = {'H_H':0,'H_B':1,'H_IR':2,'H_OR':3,'L_H':4,'U_H':5,'M_H':6}

    for domain_value in scenario['domain_order']:
        if domain_value not in domain_loaders:
            continue
        test_loader = domain_loaders[domain_value]['test']
        emb = trainer._collect_domain_embeddings(test_loader)
        if not emb:
            continue

        text_emb = emb.get('text_embeddings')  # 여기서는 프로토타입 기반으로 대체
        vib_emb = emb.get('vib_embeddings')
        metadata_list = emb.get('metadata', [])

        # UOS: 조합 라벨 생성 (rotating_component + bearing_condition)
        rc = [m.get('rotating_component','H') for m in metadata_list]
        bc = [m.get('bearing_condition','H') for m in metadata_list]
        labels = [f"{r}_{b}" for r,b in zip(rc, bc)]
        bearing_types = [m.get('bearing_type','6204') for m in metadata_list]
        domain_name = f"{domain_value}RPM"

        # 텍스트 임베딩을 클래스 프로토타입으로 대체 (라벨 매핑)
        try:
            idx = torch.tensor([label_map.get(l, 0) for l in labels], device=proto_mat.device)
            text_emb = proto_mat.index_select(0, idx)
        except Exception as _e:
            logger.warning(f"UOS 텍스트 프로토타입 생성 실패: {_e}")

        # Alignment t-SNE (7색/7마커 적용됨)
        visualizer.create_encoder_alignment_plot(
            text_embeddings=text_emb,
            vib_embeddings=vib_emb,
            labels=labels,
            bearing_types=bearing_types,
            domain_name=domain_name,
            save_name="encoder_alignment"
        )


def main():
    parser = argparse.ArgumentParser(description='TextVibCLIP 시각화 재생성')
    parser.add_argument('--results_dir', type=str, required=True, help='결과 저장 디렉토리(v2_타임스탬프)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'])
    parser.add_argument('--only_uos', action='store_true')
    parser.add_argument('--only_cwru', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    if not args.only_cwru:
        redraw_uos(results_dir, device)
    if not args.only_uos:
        redraw_cwru(results_dir, device)

    print(f"✅ 시각화 재생성 완료: {results_dir}")


if __name__ == '__main__':
    main()


