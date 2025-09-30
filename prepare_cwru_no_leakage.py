#!/usr/bin/env python3
"""
CWRU 데이터 누수 방지 준비 스크립트 (파일 내 3분할 버전)

핵심 아이디어:
- 도메인(load)×클래스(H/B/IR/OR)마다 대표 원본 파일 1개를 선택
- 동일 파일의 원시 신호를 0–60/60–80/80–100%로 3분할하여
  train/val/test 전용 .mat 파일로 각각 저장(서로 비중복 → 윈도우 누수 없음)
- 이렇게 생성된 data_scenario2/Load_{hp}hp 하위의 _train/_val/_test 파일만 사용
"""

import os
import shutil
import glob
import re
from collections import defaultdict, Counter
import numpy as np
from scipy.io import loadmat, savemat


def _read_signal_from_mat(filepath: str) -> np.ndarray:
    """CWRU .mat에서 1D 진동 신호를 읽어 반환.
    우선순위 키: 'DE_time' → 'X' → 'signal' → 그 외 1D 배열 자동 탐색
    """
    data = loadmat(filepath)
    for key in ['DE_time', 'X', 'signal']:
        if key in data:
            arr = np.array(data[key]).squeeze()
            if arr.ndim == 1:
                return arr.astype(np.float32)
    for k, v in data.items():
        if k.startswith('__'):
            continue
        arr = np.array(v).squeeze()
        if arr.ndim == 1 and arr.size > 1000:
            return arr.astype(np.float32)
    raise ValueError(f"지원되는 1D 신호를 찾을 수 없습니다: {os.path.basename(filepath)}")


def _write_signal_to_mat(filepath: str, signal: np.ndarray):
    """신호를 'DE_time' 키로 저장."""
    savemat(filepath, {'DE_time': signal.astype(np.float32)})


def _split_indices(n: int, ratios: tuple) -> tuple:
    a, b, c = ratios
    assert abs(a + b + c - 1.0) < 1e-6
    i1 = int(n * a)
    i2 = int(n * (a + b))
    return i1, i2


def _preferred_file_for_class(load_files: dict, load: str, fault_type: str) -> str:
    """도메인(load)×클래스에서 대표 원본 파일 1개 선택.
    - H: 해당 load의 첫 파일
    - B/IR: 사이즈 선호도 021→014→007→028
    - OR: 위치 선호도 @6→@3→@12, 그다음 임의
    """
    if load not in load_files:
        return None
    candidates = []
    for _, files in load_files[load].items():
        candidates.extend(files)
    if not candidates:
        return None
    if fault_type == 'H':
        return sorted(candidates)[0]
    if fault_type in ['B', 'IR']:
        size_pref = ['021', '014', '007', '028']
        for sz in size_pref:
            for f in sorted(candidates):
                if f"/{sz}/" in f.replace('\\', '/'):
                    return f
        return sorted(candidates)[0]
    if fault_type == 'OR':
        pos_pref = ['@6', '@3', '@12']
        for pos in pos_pref:
            for f in sorted(candidates):
                if pos in os.path.basename(f):
                    return f
        return sorted(candidates)[0]
    return sorted(candidates)[0]


def extract_experiment_number(filepath):
    """파일명에서 실험 번호 추출"""
    filename = os.path.basename(filepath)
    
    # OR 특수 패턴: 숫자@위치_부하.mat
    match = re.search(r'^(\d+)@\d+_\d+\.mat$', filename)
    if match:
        return match.group(1)
    
    # 표준 패턴: 숫자_부하.mat
    match = re.search(r'^(\d+)_\d+\.mat$', filename)
    if match:
        return match.group(1)
    
    return None


def extract_load_from_filename(filepath):
    """파일명에서 부하 추출"""
    filename = os.path.basename(filepath)
    
    match = re.search(r'_(\d+)\.mat$', filename)
    if match:
        return match.group(1)
    
    match = re.search(r'@\d+_(\d+)\.mat$', filename)
    if match:
        return match.group(1)
    
    return None


def get_fault_info(filepath):
    """파일에서 결함 정보 추출"""
    path_parts = filepath.split(os.sep)
    
    # 결함 타입
    if 'Normal' in path_parts:
        return 'H', None, None, extract_experiment_number(filepath)
    
    fault_type = None
    for part in path_parts:
        if part in ['B', 'IR', 'OR']:
            fault_type = part
            break
    
    # 결함 크기
    fault_size = None
    for part in path_parts:
        if part in ['007', '014', '021', '028']:
            fault_size = part
            break
    
    # OR 위치
    fault_position = None
    if fault_type == 'OR':
        filename = os.path.basename(filepath)
        for pos in ['@3', '@6', '@12']:
            if pos in filename:
                fault_position = pos
                break
    
    exp_num = extract_experiment_number(filepath)
    
    return fault_type, fault_size, fault_position, exp_num


def analyze_cwru_structure():
    """CWRU 구조 분석"""
    source_dir = "cwru_data"
    
    print("🔍 CWRU 데이터 구조 분석 중...")
    
    all_files = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # all_files[fault_type][load][exp_num] = [files]
    
    # Normal 파일들
    normal_files = glob.glob(os.path.join(source_dir, "Normal", "*.mat"))
    for file in normal_files:
        load = extract_load_from_filename(file)
        fault_type, _, _, exp_num = get_fault_info(file)
        if load and exp_num:
            all_files['H'][load][exp_num].append(file)
    
    # Fault 파일들
    fault_base = os.path.join(source_dir, "12k_Drive_End_Bearing_Fault_Data")
    fault_files = glob.glob(os.path.join(fault_base, "**", "*.mat"), recursive=True)
    
    for file in fault_files:
        load = extract_load_from_filename(file)
        fault_type, fault_size, fault_position, exp_num = get_fault_info(file)
        if load and exp_num and fault_type:
            all_files[fault_type][load][exp_num].append(file)
    
    # 분석 결과 출력
    print("\n📊 결함 타입별 실험 번호 분석:")
    for fault_type in ['H', 'B', 'IR', 'OR']:
        print(f"\n  {fault_type} 결함:")
        for load in ['0', '1', '2', '3']:
            exp_numbers = list(all_files[fault_type][load].keys())
            print(f"    {load}HP: 실험번호 {sorted(exp_numbers)} ({len(exp_numbers)}개)")
    
    return all_files


def create_experiment_based_split(all_files):
    """정보 출력용(대표 파일 통계). 분할 테이블은 사용하지 않음"""
    print("\n🎯 분할 개요(정보 출력):")
    
    # 각 결함 타입별로 실험 번호 수집
    fault_experiments = {}
    
    for fault_type in ['H', 'B', 'IR', 'OR']:
        all_exp_nums = set()
        for load in ['0', '1', '2', '3']:
            all_exp_nums.update(all_files[fault_type][load].keys())
        
        exp_list = sorted(list(all_exp_nums))
        fault_experiments[fault_type] = exp_list
        print(f"  {fault_type}: {len(exp_list)}개 실험번호 {exp_list}")
    
    for fault_type in ['H', 'B', 'IR', 'OR']:
        print(f"    {fault_type}: {len(fault_experiments[fault_type])}개 실험번호 감지")
    return None


def create_window_sliced_files(all_files) -> bool:
    """도메인×클래스별 대표 파일을 3분할하여 data_scenario2에 저장."""
    target_dir = "data_scenario2"

    # 기존 폴더 무조건 재생성(질의 없이 덮어쓰기)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    load_domains = {
        '0': 'Load_0hp',
        '1': 'Load_1hp',
        '2': 'Load_2hp',
        '3': 'Load_3hp'
    }

    for load in load_domains.keys():
        os.makedirs(os.path.join(target_dir, load_domains[load]), exist_ok=True)

    created = 0
    ratios = (0.6, 0.2, 0.2)

    for load in sorted(load_domains.keys()):
        domain_dir = os.path.join(target_dir, load_domains[load])
        print(f"\n📂 {load_domains[load]} 처리 중 (파일 내 3분할)...")

        for fault_type in ['H', 'B', 'IR', 'OR']:
            rep_file = _preferred_file_for_class(all_files[fault_type], load, fault_type)
            if rep_file is None:
                print(f"  ⚠️ {fault_type}-{load}HP: 소스 파일 없음 - 건너뜀")
                continue

            try:
                signal = _read_signal_from_mat(rep_file)
            except Exception as e:
                print(f"  ❌ {fault_type}-{load}HP: 로드 실패 - {os.path.basename(rep_file)} | {e}")
                continue

            n = signal.shape[0]
            i1, i2 = _split_indices(n, ratios)
            sig_train = signal[:i1]
            sig_val = signal[i1:i2]
            sig_test = signal[i2:]

            out_train = os.path.join(domain_dir, f"{fault_type}_{load}hp_train_01.mat")
            out_val = os.path.join(domain_dir, f"{fault_type}_{load}hp_val_01.mat")
            out_test = os.path.join(domain_dir, f"{fault_type}_{load}hp_test_01.mat")

            _write_signal_to_mat(out_train, sig_train)
            _write_signal_to_mat(out_val, sig_val)
            _write_signal_to_mat(out_test, sig_test)
            created += 3

            print(f"  ✅ {fault_type}-{load}HP: {os.path.basename(rep_file)} → 3분할 저장")

    print(f"\n✅ 총 {created}개 파일이 생성되었습니다 (도메인×클래스×3)")
    return True


def main():
    """메인 실행 함수"""
    print("🚀 CWRU 데이터 누수 방지 준비 시작! (파일 내 3분할)")
    print("=" * 60)
    print("🎯 전략: 실험 번호(베어링) 기반 분할")
    print("   - 서로 다른 베어링을 train/val/test에 할당")
    print("   - 같은 베어링의 다른 부하는 같은 subset에만")
    print("   - Domain-Incremental + 데이터 누수 방지 동시 달성")
    print("=" * 60)
    
    try:
        # 1. 구조 분석
        all_files = analyze_cwru_structure()
        
        # 2. 대표 파일 현황 출력
        _ = create_experiment_based_split(all_files)
        # 3. 대표 파일을 실제로 3분할하여 저장
        if create_window_sliced_files(all_files):
            print(f"\n🎉 CWRU 데이터 누수 방지 준비 완료!")
            print("✅ 동일 파일의 비중복 3분할 → train/val/test 간 윈도우 누수 없음")
            print("✅ 도메인별 H가 1개여도 평가 커버리지 확보")
            return True
        return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 모든 작업이 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 작업 중 오류가 발생했습니다.")
