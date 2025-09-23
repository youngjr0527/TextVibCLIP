import os
import math
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Union
from torch.utils.data import DataLoader
import torch.nn.functional as F


def compute_subset_overlap_ratio(dataset, window_split_range: Tuple[float, float], other_split_range: Tuple[float, float]) -> float:
    """두 분할 구간 사이의 윈도우 중복 가능성 지표(보수적 추정).
    - 단일 파일 내에서 오버랩 윈도우 슬라이싱을 사용한다면, 인접 구간 경계의 중첩을 근사.
    - 반환값: 0.0(겹침없음)~1.0(완전중복) 사이 추정치.
    """
    try:
        total = dataset._raw_total_windows_per_file if hasattr(dataset, '_raw_total_windows_per_file') else dataset.windows_per_file
        s1, e1 = window_split_range
        s2, e2 = other_split_range
        a1, b1 = int(total * s1), int(total * e1)
        a2, b2 = int(total * s2), int(total * e2)
        inter = max(0, min(b1, b2) - max(a1, a2))
        union = max(1, (b1 - a1) + (b2 - a2) - inter)
        return float(inter) / float(union)
    except Exception:
        return 0.0


def evaluate_linear_probe(embeddings: torch.Tensor, labels: torch.Tensor, l2_reg: float = 1e-4, max_iter: int = 200) -> float:
    """고정 임베딩에 선형 분류기 정확도(로지스틱 회귀 식) 근사.
    Gradient descent로 간단히 학습하여 분리 가능성 점검.
    """
    device = embeddings.device
    num_classes = int(labels.max().item()) + 1
    num_features = embeddings.size(1)
    W = torch.zeros(num_features, num_classes, device=device, requires_grad=True)
    b = torch.zeros(num_classes, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-2, weight_decay=l2_reg)
    y = labels.long()
    x = F.normalize(embeddings, p=2, dim=1)
    for _ in range(max_iter):
        logits = x @ W + b
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = (x @ W + b).argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return float(acc)


def collect_embeddings(model, dataloader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_v = []
    all_y = []
    with torch.no_grad():
        for batch in dataloader:
            vib = batch['vibration'].to(device)
            out = model({'vibration': vib, 'text': batch.get('text', [''] * vib.size(0))}, return_embeddings=True)
            all_v.append(out['vib_embeddings'])
            y = batch.get('labels', None)
            if y is not None:
                if y.dim() == 2:
                    y = y[:, 0]
                all_y.append(y.to(device))
    if all_v and all_y:
        return torch.cat(all_v, 0), torch.cat(all_y, 0)
    return torch.empty(0, device=device), torch.empty(0, device=device)


def save_diagnostics_report(output_dir: str, scenario_name: str, rows: List[Dict[str, Any]]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'data_diagnostics_{scenario_name}.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

