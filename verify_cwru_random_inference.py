#!/usr/bin/env python3
import os
import sys
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F

# 프로젝트 루트 경로 추가
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.textvib_model import create_textvib_model
from src.data_cache import CachedBearingDataset
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

def run_verify(experiment_dir: Path, samples_per_domain: int, device: torch.device, domains: List[int]) -> None:
    scenario_name = "CWRU_Scenario2_VaryingLoad"
    ckpt_base = experiment_dir / "checkpoints" / scenario_name
    if not ckpt_base.exists():
        raise FileNotFoundError(f"Checkpoint base not found: {ckpt_base}")

    data_dir = Path(CWRU_DATA_CONFIG["data_dir"])
    overall_correct = 0
    overall_total = 0

    for domain_value in domains:
        # 모델 및 프로토타입
        model = load_model_for_domain(ckpt_base, domain_value, device)
        prototypes = build_class_prototypes(model, device)  # (4, dim)

        # 테스트 데이터셋(캐시 기반, collate 불필요)
        test_ds = CachedBearingDataset(
            data_dir=str(data_dir),
            dataset_type="cwru",
            domain_value=domain_value,
            subset="test"
        )

        n = len(test_ds)
        if n == 0:
            print(f"[{domain_value}HP] test samples = 0 (skip)")
            continue

        picked = sample_indices(n, samples_per_domain, seed=42 + domain_value)
        correct = 0

        print(f"\n=== Verify CWRU {domain_value}HP: {len(picked)}/{n} random samples ===")
        for idx in picked:
            sample = test_ds[idx]
            vib = sample["vibration"].unsqueeze(0).to(device)        # (1, 2048)
            lbl = sample["labels"]
            lbl_id = int(lbl[0].item()) if lbl.ndim > 0 else int(lbl.item())
            meta = sample.get("metadata", {})
            fpath = meta.get("filepath", "unknown")
            widx = int(sample.get("window_idx", -1))

            with torch.no_grad():
                vib_raw = model.vib_encoder(vib)
                vib_emb = F.normalize(model.vib_projection(vib_raw), p=2, dim=1)  # (1, dim)
                sims = torch.matmul(vib_emb, prototypes.t()).squeeze(0)          # (4,)
                pred_id = int(torch.argmax(sims).item())
                topk_vals, topk_ids = torch.topk(sims, k=3)

            is_ok = (pred_id == lbl_id)
            correct += int(is_ok)
            overall_correct += int(is_ok)
            overall_total += 1

            topk_str = ", ".join([f"{CWRU_CLASS_ID_TO_NAME[int(cid)]}:{float(v):.3f}" for v, cid in zip(topk_vals.tolist(), topk_ids.tolist())])
            print(f"- file={os.path.basename(fpath)} win={widx:>4} | true={CWRU_CLASS_ID_TO_NAME[lbl_id]} pred={CWRU_CLASS_ID_TO_NAME[pred_id]} | top3[{topk_str}]")

        acc = 100.0 * correct / max(1, len(picked))
        print(f"=== Domain {domain_value}HP summary: {correct}/{len(picked)} ({acc:.2f}%) ===")

    if overall_total > 0:
        print(f"\n>>> Overall sampled accuracy: {overall_correct}/{overall_total} ({100.0*overall_correct/overall_total:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="Verify random CWRU test samples via retrieval")
    parser.add_argument("--experiment_dir", type=str, default="", help="results/{timestamp} 디렉토리 경로")
    parser.add_argument("--samples_per_domain", type=int, default=10)
    parser.add_argument("--domains", type=str, default="0,1,2,3", help="검증할 도메인 리스트. 예: 0,1,2,3")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = pick_device(args.device)
    results_dir = ROOT / "results"
    if args.experiment_dir:
        experiment_dir = Path(args.experiment_dir)
    else:
        experiment_dir = find_latest_experiment_dir(results_dir, "CWRU_Scenario2_VaryingLoad")

    domains = [int(x) for x in args.domains.split(",") if x.strip() != ""]
    print(f"[Info] experiment_dir={experiment_dir}")
    print(f"[Info] device={device}")
    print(f"[Info] domains={domains}, samples_per_domain={args.samples_per_domain}")

    run_verify(experiment_dir, args.samples_per_domain, device, domains)

if __name__ == "__main__":
    main()