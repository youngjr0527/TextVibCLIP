#!/usr/bin/env python3
"""Heatmap 시각화 테스트 (기존 실험 결과 재활용)"""

import json
import numpy as np
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization import create_visualizer

# 테스트할 실험 결과 디렉토리
RESULTS_DIR = 'results/20251014_184128'

print("🎨 Heatmap 시각화 테스트")
print("="*60)

# JSON 결과 로드
with open(f'{RESULTS_DIR}/results_20251014_185815.json', 'r') as f:
    results = json.load(f)

# Visualizer 생성
visualizer = create_visualizer(RESULTS_DIR)

# UOS 시나리오 테스트
print("\n1️⃣ UOS Scenario Heatmap 재생성...")
uos = results['UOS_Scenario1_VaryingSpeed']

# Forgetting matrix 재구성
n_domains = len(uos['domain_names'])
accuracy_matrix = np.full((n_domains, n_domains), np.nan)

if 'forgetting_matrix' in uos:
    # JSON에서 로드
    for i, row in enumerate(uos['forgetting_matrix']):
        for j, val in enumerate(row):
            if val is not None:
                accuracy_matrix[i, j] = val / 100.0  # 퍼센트 → 0-1 범위
else:
    # 없으면 예시 데이터
    example_data = [
        [99.74, None, None, None, None, None],
        [92.86, 92.77, None, None, None, None],
        [79.73, 65.10, 100.0, None, None, None],
        [51.97, 54.28, 73.57, 100.0, None, None],
        [64.76, 59.15, 69.16, 73.01, 100.0, None],
        [57.91, 65.91, 70.87, 42.34, 87.85, 100.0]
    ]
    for i, row in enumerate(example_data):
        for j, val in enumerate(row):
            if val is not None:
                accuracy_matrix[i, j] = val / 100.0

visualizer.create_forgetting_heatmap(
    domain_names=uos['domain_names'],
    accuracy_matrix=accuracy_matrix,
    scenario_name='UOS_Scenario1_VaryingSpeed'
)
print(f"   ✅ 저장: {RESULTS_DIR}/forgetting_heatmap_UOS_Scenario1_VaryingSpeed.png")

# CWRU 시나리오 테스트
print("\n2️⃣ CWRU Scenario Heatmap 재생성...")
cwru = results['CWRU_Scenario2_VaryingLoad']

n_domains_cwru = len(cwru['domain_names'])
accuracy_matrix_cwru = np.full((n_domains_cwru, n_domains_cwru), np.nan)

if 'forgetting_matrix' in cwru:
    for i, row in enumerate(cwru['forgetting_matrix']):
        for j, val in enumerate(row):
            if val is not None:
                accuracy_matrix_cwru[i, j] = val / 100.0
else:
    # CWRU 예시 (모두 100%)
    for i in range(n_domains_cwru):
        for j in range(i + 1):
            accuracy_matrix_cwru[i, j] = 1.0

visualizer.create_forgetting_heatmap(
    domain_names=cwru['domain_names'],
    accuracy_matrix=accuracy_matrix_cwru,
    scenario_name='CWRU_Scenario2_VaryingLoad'
)
print(f"   ✅ 저장: {RESULTS_DIR}/forgetting_heatmap_CWRU_Scenario2_VaryingLoad.png")

print("\n" + "="*60)
print("🎉 시각화 재생성 완료!")
print(f"\n📁 확인할 파일:")
print(f"  - {RESULTS_DIR}/forgetting_heatmap_UOS_Scenario1_VaryingSpeed.png")
print(f"  - {RESULTS_DIR}/forgetting_heatmap_CWRU_Scenario2_VaryingLoad.png")
print("\n💡 변경사항:")
print("  ✅ 모든 텍스트: 검은색 + 볼드체")
print("  ✅ Stage Avg: 별도 분리된 열 (수평 라벨)")
print("  ✅ 깔끔하고 통일된 디자인")

