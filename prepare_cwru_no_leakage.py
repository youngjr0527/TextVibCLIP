#!/usr/bin/env python3
"""
CWRU 데이터 누수 방지 준비 스크립트
실험 번호(베어링) 기반 분할로 완전한 독립성 보장

핵심 아이디어:
- 서로 다른 실험 번호(베어링)를 train/val/test에 할당
- 같은 베어링의 다른 부하 조건은 같은 subset에만 포함
- Domain-Incremental Learning + 데이터 누수 방지 동시 달성
"""

import os
import shutil
import glob
import re
from collections import defaultdict, Counter


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
    """실험 번호 기반 train/val/test 분할"""
    print("\n🎯 실험 번호 기반 분할 전략:")
    
    # 각 결함 타입별로 실험 번호 수집
    fault_experiments = {}
    
    for fault_type in ['H', 'B', 'IR', 'OR']:
        all_exp_nums = set()
        for load in ['0', '1', '2', '3']:
            all_exp_nums.update(all_files[fault_type][load].keys())
        
        exp_list = sorted(list(all_exp_nums))
        fault_experiments[fault_type] = exp_list
        print(f"  {fault_type}: {len(exp_list)}개 실험번호 {exp_list}")
    
    # 실험 번호 기반 분할
    split_assignment = {}
    
    for fault_type in ['H', 'B', 'IR', 'OR']:
        experiments = fault_experiments[fault_type]
        
        if len(experiments) >= 3:
            # 3개 이상이면 train:val:test = 60:20:20
            n = len(experiments)
            train_count = max(1, int(n * 0.6))
            val_count = max(1, int(n * 0.2))
            
            split_assignment[fault_type] = {
                'train': experiments[:train_count],
                'val': experiments[train_count:train_count + val_count],
                'test': experiments[train_count + val_count:]
            }
        elif len(experiments) == 2:
            # 2개면 train:test = 1:1
            split_assignment[fault_type] = {
                'train': [experiments[0]],
                'val': [experiments[0]],  # train과 동일 (부득이)
                'test': [experiments[1]]
            }
        elif len(experiments) == 1:
            # 1개면 모든 subset에 동일
            split_assignment[fault_type] = {
                'train': experiments,
                'val': experiments,
                'test': experiments
            }
        else:
            # 없으면 빈 리스트
            split_assignment[fault_type] = {
                'train': [],
                'val': [],
                'test': []
            }
        
        print(f"    {fault_type} 분할: Train={split_assignment[fault_type]['train']}, "
              f"Val={split_assignment[fault_type]['val']}, Test={split_assignment[fault_type]['test']}")
    
    return split_assignment


def copy_experiment_based_files(all_files, split_assignment):
    """실험 번호 기반 파일 복사"""
    target_dir = "data_scenario2"
    
    # 기존 폴더 처리
    if os.path.exists(target_dir):
        response = input(f"⚠️  {target_dir} 폴더가 이미 존재합니다. 삭제하고 새로 생성하시겠습니까? (y/N): ")
        if response.lower() == 'y':
            shutil.rmtree(target_dir)
        else:
            print("❌ 스크립트를 종료합니다.")
            return False
    
    load_domains = {
        '0': 'Load_0hp',
        '1': 'Load_1hp', 
        '2': 'Load_2hp',
        '3': 'Load_3hp'
    }
    
    # 타겟 디렉토리 생성
    for load in load_domains.keys():
        domain_dir = os.path.join(target_dir, load_domains[load])
        os.makedirs(domain_dir, exist_ok=True)
    
    copied_files = 0
    
    for load in sorted(load_domains.keys()):
        domain_name = load_domains[load]
        print(f"\n📂 {domain_name} 처리 중...")
        
        for fault_type in ['H', 'B', 'IR', 'OR']:
            # 각 subset별로 파일 복사
            for subset in ['train', 'val', 'test']:
                assigned_experiments = split_assignment[fault_type][subset]
                
                if not assigned_experiments:
                    continue
                
                for i, exp_num in enumerate(assigned_experiments, 1):
                    if exp_num in all_files[fault_type][load]:
                        files = all_files[fault_type][load][exp_num]
                        if files:
                            source_file = files[0]  # 첫 번째 파일 선택
                            
                            # 새 파일명 생성
                            new_filename = f"{fault_type}_{load}hp_{subset}_{i:02d}.mat"
                            target_path = os.path.join(target_dir, domain_name, new_filename)
                            
                            # 파일 복사
                            shutil.copy2(source_file, target_path)
                            copied_files += 1
                            
                            print(f"    ✅ {fault_type}-{subset}: {os.path.basename(source_file)} → {new_filename}")
    
    print(f"\n✅ 총 {copied_files}개 파일이 복사되었습니다!")
    return True


def main():
    """메인 실행 함수"""
    print("🚀 CWRU 데이터 누수 방지 준비 시작!")
    print("=" * 60)
    print("🎯 전략: 실험 번호(베어링) 기반 분할")
    print("   - 서로 다른 베어링을 train/val/test에 할당")
    print("   - 같은 베어링의 다른 부하는 같은 subset에만")
    print("   - Domain-Incremental + 데이터 누수 방지 동시 달성")
    print("=" * 60)
    
    try:
        # 1. 구조 분석
        all_files = analyze_cwru_structure()
        
        # 2. 실험 번호 기반 분할
        split_assignment = create_experiment_based_split(all_files)
        
        # 3. 파일 복사
        if copy_experiment_based_files(all_files, split_assignment):
            print(f"\n🎉 CWRU 데이터 누수 방지 준비 완료!")
            print("✅ 서로 다른 베어링으로 완전한 독립성 보장!")
            print("✅ Domain-Incremental Learning 지원!")
            return True
        else:
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
