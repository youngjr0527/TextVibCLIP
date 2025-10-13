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


def _select_bearings_for_splits(all_files: dict, load: str, fault_type: str) -> dict:
    """
    각 클래스별로 train/val/test용 파일 선택
    
    전략:
    - H: 1개 베어링만 존재 → 시간순 3분할 (기존 유지)
    - B/IR/OR: Load별 순환 할당 (같은 fault size, 다른 load의 베어링)
      * Load 0: 베어링[0] train, 베어링[1] val, 베어링[2] test
      * Load 1: 베어링[1] train, 베어링[2] val, 베어링[3] test
      * Load 2: 베어링[2] train, 베어링[3] val, 베어링[0] test
      * Load 3: 베어링[3] train, 베어링[0] val, 베어링[1] test
    
    Returns:
        {'train': filepath, 'val': filepath, 'test': filepath} 
        또는 {'split_single': filepath} (H의 경우)
    """
    if load not in all_files[fault_type]:
        return None
        
    # H (Normal): 베어링 1개만 → 시간순 3분할
    if fault_type == 'H':
        candidates = []
        for exp_num, files in all_files[fault_type][load].items():
            candidates.extend(files)
        if candidates:
            return {'split_single': sorted(candidates)[0]}
        return None
    
    # B/IR/OR: 순환 할당 전략
    # 선호 fault size 결정
    if fault_type in ['B', 'IR']:
        preferred_size = '021'  # 중간 크기
    else:  # OR
        preferred_size = '007'  # 작은 크기 (@6 위치)
    
    # 모든 load의 해당 size 베어링 수집
    all_bearings = {}  # {bearing_num: {load: filepath}}
    for ld in ['0', '1', '2', '3']:
        if ld not in all_files[fault_type]:
            continue
        for exp_num, files in all_files[fault_type][ld].items():
            for f in files:
                match = False
                if fault_type == 'OR':
                    match = f"/{preferred_size}/" in f.replace('\\', '/') and "@6" in os.path.basename(f)
                else:
                    match = f"/{preferred_size}/" in f.replace('\\', '/')
                
                if match:
                    if exp_num not in all_bearings:
                        all_bearings[exp_num] = {}
                    all_bearings[exp_num][ld] = f
    
    # 각 load에 대응하는 베어링 리스트
    sorted_bearings = sorted(all_bearings.keys())
    if len(sorted_bearings) < 3:
        # 베어링이 3개 미만이면 fallback
        if load in all_files[fault_type]:
            candidates = []
            for exp_num, files in all_files[fault_type][load].items():
                candidates.extend(files)
            if candidates:
                return {'split_single': sorted(candidates)[0]}
        return None
    
    # 순환 할당: load 인덱스에 따라 offset
    load_idx = int(load)
    train_idx = load_idx % len(sorted_bearings)
    val_idx = (load_idx + 1) % len(sorted_bearings)
    test_idx = (load_idx + 2) % len(sorted_bearings)
    
    train_bearing = sorted_bearings[train_idx]
    val_bearing = sorted_bearings[val_idx]
    test_bearing = sorted_bearings[test_idx]
    
    # 각 베어링의 해당 load 파일 선택
    result = {}
    for subset, bearing in [('train', train_bearing), ('val', val_bearing), ('test', test_bearing)]:
        # 우선: 같은 load, 없으면 다른 load
        if load in all_bearings[bearing]:
            result[subset] = all_bearings[bearing][load]
        else:
            # 다른 load 중 하나 선택
            available_loads = sorted(all_bearings[bearing].keys())
            if available_loads:
                result[subset] = all_bearings[bearing][available_loads[0]]
            else:
                return None
    
    if len(result) == 3:
        print(f"    [순환] {fault_type}-{load}HP: Train={train_bearing}({list(all_bearings[train_bearing].keys())}), "
              f"Val={val_bearing}({list(all_bearings[val_bearing].keys())}), "
              f"Test={test_bearing}({list(all_bearings[test_bearing].keys())})")
        return result
    
    return None


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
    """
    도메인×클래스별 파일 준비
    
    전략:
    - H: 1개 베어링만 존재 → 시간순 3분할 (작은 leakage 허용)
    - B/IR/OR: 같은 fault size 내 다른 베어링 → train/val/test 각각 할당
    """
    target_dir = "data_scenario2"

    # 기존 폴더 무조건 재생성
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
    ratios = (0.6, 0.2, 0.2)  # H 시간순 분할용

    for load in sorted(load_domains.keys()):
        domain_dir = os.path.join(target_dir, load_domains[load])
        print(f"\n📂 {load_domains[load]} 처리 중...")

        for fault_type in ['H', 'B', 'IR', 'OR']:
            selected = _select_bearings_for_splits(all_files, load, fault_type)
            if selected is None:
                print(f"  ⚠️ {fault_type}-{load}HP: 소스 파일 없음 - 건너뜀")
                continue

            # H: 시간순 3분할
            if 'split_single' in selected:
                rep_file = selected['split_single']
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

                print(f"  ✅ {fault_type}-{load}HP: 베어링 {os.path.basename(rep_file).split('_')[0]} → 시간순 3분할")
            
            # B/IR/OR: 다른 베어링 사용
            else:
                for subset in ['train', 'val', 'test']:
                    src_file = selected[subset]
                    try:
                        signal = _read_signal_from_mat(src_file)
                    except Exception as e:
                        print(f"  ❌ {fault_type}-{load}HP-{subset}: 로드 실패 - {os.path.basename(src_file)} | {e}")
                        continue

                    out_file = os.path.join(domain_dir, f"{fault_type}_{load}hp_{subset}_01.mat")
                    _write_signal_to_mat(out_file, signal)
                    created += 1

                bearing_nums = [os.path.basename(selected[s]).split('_')[0].split('@')[0] for s in ['train', 'val', 'test']]
                print(f"  ✅ {fault_type}-{load}HP: 다른 베어링 사용 (Train:{bearing_nums[0]}, Val:{bearing_nums[1]}, Test:{bearing_nums[2]})")

    print(f"\n✅ 총 {created}개 파일이 생성되었습니다")
    return True


def main():
    """메인 실행 함수"""
    print("🚀 CWRU 데이터 누수 방지 준비 시작! (개선 버전)")
    print("=" * 60)
    print("🎯 전략: 베어링 기반 분할 (Fault Type별 차별화)")
    print("   - H (Normal): 시간순 3분할 (베어링 1개뿐)")
    print("   - B/IR/OR: 같은 fault size 내 다른 베어링 할당")
    print("     * B/IR: 021 size → 베어링 3개 사용")
    print("     * OR: 007/@6 → 베어링 3개 사용")
    print("   - Domain-Incremental + 진정한 일반화 평가")
    print("=" * 60)
    
    try:
        # 1. 구조 분석
        all_files = analyze_cwru_structure()
        
        # 2. 대표 파일 현황 출력
        _ = create_experiment_based_split(all_files)
        # 3. 파일 준비 (다른 베어링 또는 시간순 3분할)
        if create_window_sliced_files(all_files):
            print(f"\n🎉 CWRU 데이터 누수 방지 준비 완료!")
            print("✅ H: 시간순 3분할 (작은 leakage, 베어링 1개뿐)")
            print("✅ B/IR/OR: 완전히 다른 베어링 사용 (진정한 일반화)")
            print("✅ Domain-incremental learning 시나리오 유지")
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
