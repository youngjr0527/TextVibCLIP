#!/usr/bin/env python3
"""
CWRU 데이터 검증 및 디버깅 스크립트 (개선 버전)

기능:
1. 랜덤 샘플 추론 검증
2. 데이터 분할 및 로딩 검증  
3. 클래스별 성능 분석
4. 파일명 및 메타데이터 검증
5. 데이터 누수 가능성 체크
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import numpy as np

import torch
import torch.nn.functional as F

# 프로젝트 루트 경로 추가
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.textvib_model import create_textvib_model
from src.data_cache import CachedBearingDataset
from src.data_loader import BearingDataset
from configs.model_config import CWRU_DATA_CONFIG

CWRU_CLASS_ID_TO_NAME = {0: "H", 1: "B", 2: "IR", 3: "OR"}
CWRU_CLASS_ID_TO_TEXT = {
    0: ["healthy bearing", "normal bearing with no fault", "bearing vibration without defect"],
    1: ["bearing with ball fault", "ball defect in bearing", "ball damage on bearing"],
    2: ["bearing inner race fault", "inner ring defect in bearing", "inner race damage of bearing"],
    3: ["bearing outer race fault", "outer ring defect in bearing", "outer race damage of bearing"],
}

def pick_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_latest_experiment_dir(results_dir: Path, scenario_name: str) -> Path:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {results_dir}")
    candidates = sorted([p for p in results_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        ckpt_dir = c / "checkpoints" / scenario_name
        if ckpt_dir.exists():
            return c
    raise FileNotFoundError(f"No experiment dir with checkpoints/{scenario_name} under {results_dir}")

def load_model_for_domain(ckpt_base: Path, domain_value: int, device: torch.device) -> torch.nn.Module:
    # 도메인별 best 체크포인트 우선, 없으면 first_domain_final.pth로 폴백(0HP)
    candidates = [
        ckpt_base / f"domain_{domain_value}_best.pth",
        ckpt_base / "first_domain_final.pth"
    ]
    ckpt_path = None
    for p in candidates:
        if p.exists():
            ckpt_path = p
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"Checkpoint not found under {ckpt_base}")
    model = create_textvib_model(domain_stage="continual", dataset_type="cwru").to(device)
    model.load_checkpoint(str(ckpt_path), device=device)
    model.eval()
    return model

def build_class_prototypes(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    # 각 클래스의 프롬프트 임베딩 평균 → 프로토타입 (4, dim)
    class_embs = []
    for cls_id in [0, 1, 2, 3]:
        texts = CWRU_CLASS_ID_TO_TEXT[cls_id]
        raw = model.text_encoder.encode_texts(texts, device)
        proj = F.normalize(model.text_projection(raw), p=2, dim=1)
        proto = F.normalize(proj.mean(dim=0, keepdim=True), p=2, dim=1)
        class_embs.append(proto)
    return torch.cat(class_embs, dim=0)  # (4, dim)

def sample_indices(n_total: int, k: int, seed: int = 42) -> List[int]:
    k = max(1, min(k, n_total))
    rng = random.Random(seed)
    return rng.sample(range(n_total), k)

def verify_data_splitting(data_dir: Path, domain_value: int) -> None:
    """데이터 분할 및 파일 구조 검증"""
    print(f"\n🔍 === 데이터 분할 검증: {domain_value}HP ===")
    
    domain_dir = data_dir / f"Load_{domain_value}hp"
    if not domain_dir.exists():
        print(f"❌ 도메인 디렉토리 없음: {domain_dir}")
        return
    
    # 파일 구조 확인
    classes = ['H', 'B', 'IR', 'OR']
    subsets = ['train', 'val', 'test']
    
    print("📁 파일 구조:")
    for cls in classes:
        print(f"  {cls}: ", end="")
        for subset in subsets:
            fname = f"{cls}_{domain_value}hp_{subset}_01.mat"
            fpath = domain_dir / fname
            if fpath.exists():
                print(f"{subset}✓ ", end="")
            else:
                print(f"{subset}✗ ", end="")
        print()
    
    # 원본 데이터셋으로 분할 검증
    try:
        print("\n📊 원본 데이터셋 분할 검증:")
        for subset in subsets:
            ds = BearingDataset(
                data_dir=str(data_dir),
                dataset_type="cwru", 
                domain_value=domain_value,
                subset=subset
            )
            
            if len(ds.file_paths) > 0:
                # 클래스 분포 확인
                labels = [ds.metadata_list[i]['bearing_condition'] for i in range(len(ds.metadata_list))]
                label_counts = Counter(labels)
                print(f"  {subset}: {len(ds.file_paths)}개 파일, 클래스 분포: {dict(label_counts)}")
                
                # 파일명 확인
                file_names = [os.path.basename(f) for f in ds.file_paths]
                print(f"    파일들: {file_names}")
            else:
                print(f"  {subset}: 파일 없음")
    except Exception as e:
        print(f"❌ 데이터셋 검증 실패: {e}")


def verify_cached_dataset(data_dir: Path, domain_value: int) -> None:
    """캐시된 데이터셋 검증"""
    print(f"\n💾 === 캐시 데이터셋 검증: {domain_value}HP ===")
    
    try:
        test_ds = CachedBearingDataset(
            data_dir=str(data_dir),
            dataset_type="cwru",
            domain_value=domain_value,
            subset="test"
        )
        
        print(f"📊 캐시된 테스트 데이터: {len(test_ds)}개 샘플")
        
        if len(test_ds) > 0:
            # 첫 몇 개 샘플의 메타데이터 확인
            print("🔍 샘플 메타데이터:")
            for i in range(min(5, len(test_ds))):
                sample = test_ds[i]
                meta = sample.get("metadata", {})
                lbl = sample["labels"]
                lbl_id = int(lbl[0].item()) if lbl.ndim > 0 else int(lbl.item())
                
                print(f"  샘플 {i}: 클래스={CWRU_CLASS_ID_TO_NAME[lbl_id]}, "
                      f"파일={os.path.basename(meta.get('filepath', 'unknown'))}, "
                      f"윈도우={sample.get('window_idx', -1)}")
        
        # 클래스별 샘플 수 계산
        class_counts = defaultdict(int)
        for i in range(len(test_ds)):
            sample = test_ds[i]
            lbl = sample["labels"]
            lbl_id = int(lbl[0].item()) if lbl.ndim > 0 else int(lbl.item())
            class_counts[lbl_id] += 1
        
        print(f"📈 클래스별 샘플 수: {dict(class_counts)}")
        
    except Exception as e:
        print(f"❌ 캐시 데이터셋 검증 실패: {e}")


def run_comprehensive_verify(experiment_dir: Path, samples_per_domain: int, device: torch.device, domains: List[int]) -> None:
    """종합 검증 실행"""
    scenario_name = "CWRU_Scenario2_VaryingLoad"
    ckpt_base = experiment_dir / "checkpoints" / scenario_name
    if not ckpt_base.exists():
        raise FileNotFoundError(f"Checkpoint base not found: {ckpt_base}")

    data_dir = Path(CWRU_DATA_CONFIG["data_dir"])
    
    # 1. 데이터 분할 검증
    for domain_value in domains:
        verify_data_splitting(data_dir, domain_value)
        verify_cached_dataset(data_dir, domain_value)
    
    # 2. 모델 추론 검증
    print(f"\n🤖 === 모델 추론 검증 ===")
    overall_correct = 0
    overall_total = 0
    domain_results = {}
    
    for domain_value in domains:
        print(f"\n--- {domain_value}HP 도메인 ---")
        
        # 모델 및 프로토타입
        model = load_model_for_domain(ckpt_base, domain_value, device)
        prototypes = build_class_prototypes(model, device)

        # 테스트 데이터셋
        test_ds = CachedBearingDataset(
            data_dir=str(data_dir),
            dataset_type="cwru",
            domain_value=domain_value,
            subset="test"
        )

        n = len(test_ds)
        if n == 0:
            print(f"❌ {domain_value}HP: test samples = 0 (skip)")
            continue

        picked = sample_indices(n, samples_per_domain, seed=42 + domain_value)
        correct = 0
        class_results = defaultdict(lambda: {'correct': 0, 'total': 0})

        print(f"🎯 {len(picked)}/{n} 랜덤 샘플 검증:")
        for idx in picked:
            sample = test_ds[idx]
            vib = sample["vibration"].unsqueeze(0).to(device)
            lbl = sample["labels"]
            lbl_id = int(lbl[0].item()) if lbl.ndim > 0 else int(lbl.item())
            meta = sample.get("metadata", {})
            fpath = meta.get("filepath", "unknown")
            widx = int(sample.get("window_idx", -1))

            with torch.no_grad():
                vib_raw = model.vib_encoder(vib)
                vib_emb = F.normalize(model.vib_projection(vib_raw), p=2, dim=1)
                sims = torch.matmul(vib_emb, prototypes.t()).squeeze(0)
                pred_id = int(torch.argmax(sims).item())
                topk_vals, topk_ids = torch.topk(sims, k=3)

            is_ok = (pred_id == lbl_id)
            correct += int(is_ok)
            overall_correct += int(is_ok)
            overall_total += 1
            
            # 클래스별 결과 추적
            class_results[lbl_id]['total'] += 1
            if is_ok:
                class_results[lbl_id]['correct'] += 1

            status = "✓" if is_ok else "✗"
            topk_str = ", ".join([f"{CWRU_CLASS_ID_TO_NAME[int(cid)]}:{float(v):.3f}" for v, cid in zip(topk_vals.tolist(), topk_ids.tolist())])
            print(f"  {status} {CWRU_CLASS_ID_TO_NAME[lbl_id]}→{CWRU_CLASS_ID_TO_NAME[pred_id]} | {topk_str}")

        acc = 100.0 * correct / max(1, len(picked))
        domain_results[domain_value] = {
            'accuracy': acc,
            'correct': correct,
            'total': len(picked),
            'class_results': dict(class_results)
        }
        
        print(f"📊 {domain_value}HP 결과: {correct}/{len(picked)} ({acc:.2f}%)")
        
        # 클래스별 상세 결과
        for cls_id, results in class_results.items():
            cls_acc = 100.0 * results['correct'] / max(1, results['total'])
            print(f"  {CWRU_CLASS_ID_TO_NAME[cls_id]}: {results['correct']}/{results['total']} ({cls_acc:.1f}%)")

    # 3. 전체 결과 요약
    print(f"\n📈 === 전체 결과 요약 ===")
    if overall_total > 0:
        overall_acc = 100.0 * overall_correct / overall_total
        print(f"🎯 전체 샘플 정확도: {overall_correct}/{overall_total} ({overall_acc:.2f}%)")
        
        print(f"\n📊 도메인별 결과:")
        for domain_value, results in domain_results.items():
            print(f"  {domain_value}HP: {results['correct']}/{results['total']} ({results['accuracy']:.2f}%)")
        
        # 의심스러운 결과 감지
        if overall_acc > 95.0:
            print(f"\n⚠️  경고: 전체 정확도가 95%를 초과합니다 ({overall_acc:.2f}%)")
            print("   이는 다음 중 하나일 수 있습니다:")
            print("   1. 데이터 누수 (train/val/test 간 중복)")
            print("   2. 과적합 (데이터가 너무 적음)")
            print("   3. 클래스 분리가 너무 명확함")
            print("   → 데이터 분할 로직을 다시 검토해보세요!")

def check_data_leakage(data_dir: Path, domains: List[int]) -> None:
    """데이터 누수 가능성 체크"""
    print(f"\n🔍 === 데이터 누수 체크 ===")
    
    all_files = set()
    domain_files = {}
    
    for domain_value in domains:
        domain_dir = data_dir / f"Load_{domain_value}hp"
        if not domain_dir.exists():
            continue
            
        domain_files[domain_value] = set()
        
        # 모든 .mat 파일 수집
        for fpath in domain_dir.glob("*.mat"):
            filename = fpath.name
            domain_files[domain_value].add(filename)
            all_files.add(filename)
    
    # 파일명 중복 체크
    print("📁 도메인별 파일 중복 체크:")
    for d1 in domains:
        for d2 in domains:
            if d1 < d2:  # 중복 체크 방지
                common = domain_files.get(d1, set()) & domain_files.get(d2, set())
                if common:
                    print(f"  ⚠️ {d1}HP ↔ {d2}HP: {len(common)}개 중복 파일")
                    for f in sorted(common):
                        print(f"    - {f}")
                else:
                    print(f"  ✓ {d1}HP ↔ {d2}HP: 중복 없음")
    
    # 전체 파일명 패턴 분석
    print(f"\n📊 파일명 패턴 분석:")
    train_files = [f for f in all_files if '_train_' in f]
    val_files = [f for f in all_files if '_val_' in f]
    test_files = [f for f in all_files if '_test_' in f]
    
    print(f"  Train 파일: {len(train_files)}개")
    print(f"  Val 파일: {len(val_files)}개")
    print(f"  Test 파일: {len(test_files)}개")
    
    # 클래스별 파일 분포
    classes = ['H', 'B', 'IR', 'OR']
    for cls in classes:
        cls_train = [f for f in train_files if f.startswith(f'{cls}_')]
        cls_val = [f for f in val_files if f.startswith(f'{cls}_')]
        cls_test = [f for f in test_files if f.startswith(f'{cls}_')]
        print(f"  {cls}: Train({len(cls_train)}) Val({len(cls_val)}) Test({len(cls_test)})")


def main():
    parser = argparse.ArgumentParser(description="CWRU 데이터 종합 검증 및 디버깅")
    parser.add_argument("--experiment_dir", type=str, default="", help="results/{timestamp} 디렉토리 경로")
    parser.add_argument("--samples_per_domain", type=int, default=20, help="도메인당 검증할 샘플 수")
    parser.add_argument("--domains", type=str, default="0,1,2,3", help="검증할 도메인 리스트. 예: 0,1,2,3")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data_only", action="store_true", help="데이터 검증만 실행 (모델 추론 생략)")
    parser.add_argument("--leakage_check", action="store_true", help="데이터 누수 체크만 실행")
    args = parser.parse_args()

    device = pick_device(args.device)
    results_dir = ROOT / "results"
    data_dir = Path(CWRU_DATA_CONFIG["data_dir"])
    
    if args.experiment_dir:
        experiment_dir = Path(args.experiment_dir)
    else:
        experiment_dir = find_latest_experiment_dir(results_dir, "CWRU_Scenario2_VaryingLoad")

    domains = [int(x) for x in args.domains.split(",") if x.strip() != ""]
    
    print("🔍 CWRU 데이터 종합 검증 시작")
    print(f"📁 데이터 디렉토리: {data_dir}")
    print(f"📁 실험 디렉토리: {experiment_dir}")
    print(f"🔧 디바이스: {device}")
    print(f"🎯 도메인: {domains}, 샘플/도메인: {args.samples_per_domain}")
    
    # 데이터 누수 체크
    if args.leakage_check:
        check_data_leakage(data_dir, domains)
        return
    
    # 데이터 분할 검증
    print(f"\n{'='*60}")
    print("1단계: 데이터 분할 및 구조 검증")
    print(f"{'='*60}")
    for domain_value in domains:
        verify_data_splitting(data_dir, domain_value)
        verify_cached_dataset(data_dir, domain_value)
    
    # 데이터 누수 체크
    check_data_leakage(data_dir, domains)
    
    if args.data_only:
        print(f"\n✅ 데이터 검증 완료 (모델 추론 생략)")
        return
    
    # 모델 추론 검증
    print(f"\n{'='*60}")
    print("2단계: 모델 추론 검증")
    print(f"{'='*60}")
    run_comprehensive_verify(experiment_dir, args.samples_per_domain, device, domains)
    
    print(f"\n✅ CWRU 종합 검증 완료!")

if __name__ == "__main__":
    main()