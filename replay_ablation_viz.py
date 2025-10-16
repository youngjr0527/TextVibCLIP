#!/usr/bin/env python3
"""
Replay Buffer Ablation Study 시각화 도구
기존 실험 결과를 활용하여 replay vs replay-free 비교 그래프 생성

Usage:
    python replay_ablation_viz.py --results_dir results/20251014_233348
    python replay_ablation_viz.py --results_dir results/20251014_233348 --scenario UOS
"""

import argparse
import json
import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Optional

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization import create_visualizer

def find_latest_results(results_dir: str) -> Optional[str]:
    """가장 최근 실험 결과 디렉토리 찾기"""
    if not os.path.exists(results_dir):
        print(f"❌ 결과 디렉토리가 존재하지 않습니다: {results_dir}")
        return None
    
    # results 디렉토리 내의 모든 타임스탬프 디렉토리 찾기
    timestamp_dirs = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and len(item) == 15:  # YYYYMMDD_HHMMSS 형식
            timestamp_dirs.append(item_path)
    
    if not timestamp_dirs:
        print(f"❌ 결과 디렉토리를 찾을 수 없습니다: {results_dir}")
        return None
    
    # 가장 최근 디렉토리 반환
    latest_dir = max(timestamp_dirs)
    print(f"📁 가장 최근 실험 결과: {latest_dir}")
    return latest_dir

def load_results(experiment_dir: str) -> Dict:
    """실험 결과 JSON 파일 로드"""
    results_files = glob.glob(os.path.join(experiment_dir, "results_*.json"))
    
    if not results_files:
        print(f"❌ 결과 JSON 파일을 찾을 수 없습니다: {experiment_dir}")
        return {}
    
    # 가장 최근 결과 파일 사용
    latest_results_file = max(results_files)
    print(f"📄 결과 파일 로드: {os.path.basename(latest_results_file)}")
    
    with open(latest_results_file, 'r') as f:
        results = json.load(f)
    
    return results

def get_scenario_names(results: Dict) -> List[str]:
    """결과에서 시나리오 이름 추출 (ReplayFree 제외)"""
    scenarios = []
    for key in results.keys():
        if 'ReplayFree' not in key:
            scenarios.append(key)
    return scenarios

def create_comparison_plots(results: Dict, experiment_dir: str, target_scenarios: List[str] = None):
    """Replay vs Replay-Free 비교 그래프 생성"""
    
    # 시나리오 목록 결정
    if target_scenarios is None:
        target_scenarios = get_scenario_names(results)
    
    if not target_scenarios:
        print("❌ 비교할 시나리오가 없습니다.")
        return
    
    # Visualizer 생성
    visualizer = create_visualizer(experiment_dir)
    
    created_plots = []
    
    for scenario_name in target_scenarios:
        replay_free_name = scenario_name + '_ReplayFree'
        
        if scenario_name not in results:
            print(f"⚠️ Replay 결과를 찾을 수 없습니다: {scenario_name}")
            continue
            
        if replay_free_name not in results:
            print(f"⚠️ Replay-Free 결과를 찾을 수 없습니다: {replay_free_name}")
            continue
        
        print(f"📊 {scenario_name} 비교 그래프 생성 중...")
        
        try:
            save_path = visualizer.create_replay_comparison_plot(
                replay_results=results[scenario_name],
                replay_free_results=results[replay_free_name],
                scenario_name=scenario_name
            )
            
            if save_path:
                created_plots.append(save_path)
                print(f"   ✅ 저장 완료: {os.path.basename(save_path)}")
            else:
                print(f"   ❌ 생성 실패: {scenario_name}")
                
        except Exception as e:
            print(f"   ❌ 오류 발생: {scenario_name} - {str(e)}")
    
    return created_plots

def print_summary(results: Dict, created_plots: List[str]):
    """결과 요약 출력"""
    print("\n" + "="*60)
    print("🎉 Replay Buffer Ablation Study 시각화 완료!")
    print("="*60)
    
    print(f"\n📊 생성된 비교 그래프 ({len(created_plots)}개):")
    for plot_path in created_plots:
        print(f"   - {os.path.basename(plot_path)}")
    
    print(f"\n📈 성능 요약:")
    scenarios = get_scenario_names(results)
    
    for scenario_name in scenarios:
        replay_free_name = scenario_name + '_ReplayFree'
        
        if scenario_name in results and replay_free_name in results:
            replay_acc = results[scenario_name].get('average_accuracy', 0) * 100
            replay_forget = results[scenario_name].get('average_forgetting', 0) * 100
            replay_free_acc = results[replay_free_name].get('average_accuracy', 0) * 100
            replay_free_forget = results[replay_free_name].get('average_forgetting', 0) * 100
            
            acc_diff = replay_acc - replay_free_acc
            forget_diff = replay_forget - replay_free_forget
            
            print(f"\n   {scenario_name}:")
            print(f"     Accuracy: Replay {replay_acc:.1f}% vs Replay-Free {replay_free_acc:.1f}% (Δ{acc_diff:+.1f}%)")
            print(f"     Forgetting: Replay {replay_forget:.1f}% vs Replay-Free {replay_free_forget:.1f}% (Δ{forget_diff:+.1f}%)")
            
            if acc_diff > 0:
                print(f"     ✅ Replay 방식이 정확도에서 {acc_diff:.1f}%p 우수")
            elif acc_diff < 0:
                print(f"     ⚠️ Replay-Free 방식이 정확도에서 {abs(acc_diff):.1f}%p 우수")
            
            if forget_diff < 0:
                print(f"     ✅ Replay 방식이 망각도에서 {abs(forget_diff):.1f}%p 우수 (낮은 망각)")
            elif forget_diff > 0:
                print(f"     ⚠️ Replay-Free 방식이 망각도에서 {forget_diff:.1f}%p 우수 (낮은 망각)")

def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='Replay Buffer Ablation Study 시각화')
    
    parser.add_argument('--results_dir', type=str, default='results',
                       help='결과 디렉토리 (기본값: results)')
    parser.add_argument('--experiment_dir', type=str, default=None,
                       help='특정 실험 디렉토리 (예: results/20251014_233348)')
    parser.add_argument('--scenario', type=str, nargs='+', default=None,
                       help='생성할 시나리오 (예: UOS CWRU)')
    parser.add_argument('--latest', action='store_true',
                       help='가장 최근 실험 결과 사용')
    
    return parser.parse_args()

def main():
    """메인 실행 함수"""
    args = parse_arguments()
    
    print("🎨 Replay Buffer Ablation Study 시각화 도구")
    print("="*60)
    
    # 실험 디렉토리 결정
    if args.experiment_dir:
        experiment_dir = args.experiment_dir
        if not os.path.exists(experiment_dir):
            print(f"❌ 지정된 실험 디렉토리가 존재하지 않습니다: {experiment_dir}")
            return
    elif args.latest:
        experiment_dir = find_latest_results(args.results_dir)
        if not experiment_dir:
            return
    else:
        # results 디렉토리에서 가장 최근 실험 찾기
        experiment_dir = find_latest_results(args.results_dir)
        if not experiment_dir:
            return
    
    print(f"📂 실험 디렉토리: {experiment_dir}")
    
    # 결과 로드
    results = load_results(experiment_dir)
    if not results:
        return
    
    print(f"📋 로드된 시나리오: {list(results.keys())}")
    
    # 시나리오 필터링
    target_scenarios = None
    if args.scenario:
        target_scenarios = []
        for scenario in args.scenario:
            # 정확한 시나리오 이름 찾기
            matching_scenarios = [s for s in get_scenario_names(results) if scenario.upper() in s.upper()]
            if matching_scenarios:
                target_scenarios.extend(matching_scenarios)
            else:
                print(f"⚠️ 시나리오를 찾을 수 없습니다: {scenario}")
        
        if not target_scenarios:
            print("❌ 지정된 시나리오를 찾을 수 없습니다.")
            return
        
        print(f"🎯 대상 시나리오: {target_scenarios}")
    
    # 비교 그래프 생성
    created_plots = create_comparison_plots(results, experiment_dir, target_scenarios)
    
    # 결과 요약
    print_summary(results, created_plots)
    
    print(f"\n💡 생성된 그래프는 다음 위치에 저장되었습니다:")
    print(f"   {experiment_dir}")
    print(f"\n📊 그래프 특징:")
    print(f"   - 상단: 단계별 정확도 비교 (Stage-wise Accuracy)")
    print(f"   - 하단: 최종 도메인별 성능 (Final Domain Performance)")
    print(f"   - 파란색 선: With Replay Buffer")
    print(f"   - 빨간색 선: Replay-Free")

if __name__ == "__main__":
    main()
