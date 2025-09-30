#!/usr/bin/env python3
"""
간단한 CWRU 데이터셋 준비 스크립트
Domain-Incremental Learning을 위한 최적화

목표:
- 각 도메인(0HP, 1HP, 2HP, 3HP)당 4개 파일 (H, B, IR, OR 각 1개)
- 021 크기 우선 사용 (일관성)
- @6 위치 우선 사용 (OR 결함)
- 윈도우 레벨 분할 사용 (모든 클래스를 모든 subset에 포함)
"""

import os
import shutil
import glob
import re
from collections import defaultdict


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
        return 'H', None, None
    
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
            if pos in filename or pos in filepath:
                fault_position = pos
                break
    
    return fault_type, fault_size, fault_position


def select_best_file(candidates, fault_type):
    """최적 파일 선택 (021 크기, @6 위치 우선)"""
    if not candidates:
        return None
    
    if fault_type == 'H':
        return candidates[0]  # Normal은 유일
    
    # 021 크기 우선
    for filepath in candidates:
        _, size, position = get_fault_info(filepath)
        if size == '021':
            if fault_type == 'OR':
                if position == '@6':
                    return filepath  # OR: 021 + @6 최우선
            else:
                return filepath  # B, IR: 021 크기 최우선
    
    # @6 위치 우선 (OR만)
    if fault_type == 'OR':
        for filepath in candidates:
            _, size, position = get_fault_info(filepath)
            if position == '@6':
                return filepath
    
    # 기본: 첫 번째 파일
    return candidates[0]


def main():
    """메인 실행 함수"""
    print("🚀 간단한 CWRU 데이터 준비 시작!")
    print("=" * 50)
    print("🎯 Domain-Incremental Learning 최적화:")
    print("   - 각 도메인당 4개 파일 (H, B, IR, OR)")
    print("   - 021 크기 우선, @6 위치 우선")
    print("   - 윈도우 레벨 분할로 모든 클래스 포함")
    print("=" * 50)
    
    source_dir = "cwru_data"
    target_dir = "data_scenario2"
    
    load_domains = {
        '0': 'Load_0hp',
        '1': 'Load_1hp', 
        '2': 'Load_2hp',
        '3': 'Load_3hp'
    }
    
    fault_types = ['H', 'B', 'IR', 'OR']
    
    # 소스 확인
    if not os.path.exists(source_dir):
        print(f"❌ 오류: {source_dir} 폴더를 찾을 수 없습니다.")
        return False
    
    # 타겟 폴더 처리
    if os.path.exists(target_dir):
        response = input(f"⚠️  {target_dir} 폴더가 이미 존재합니다. 삭제하고 새로 생성하시겠습니까? (y/N): ")
        if response.lower() == 'y':
            shutil.rmtree(target_dir)
        else:
            print("❌ 스크립트를 종료합니다.")
            return False
    
    # 파일 수집
    print("\n📊 파일 수집 중...")
    
    all_files = defaultdict(lambda: defaultdict(list))
    
    # Normal 파일들
    normal_files = glob.glob(os.path.join(source_dir, "Normal", "*.mat"))
    for file in normal_files:
        load = extract_load_from_filename(file)
        if load in load_domains:
            all_files[load]['H'].append(file)
    
    # Fault 파일들
    fault_base = os.path.join(source_dir, "12k_Drive_End_Bearing_Fault_Data")
    fault_files = glob.glob(os.path.join(fault_base, "**", "*.mat"), recursive=True)
    
    for file in fault_files:
        load = extract_load_from_filename(file)
        fault_type, _, _ = get_fault_info(file)
        
        if load in load_domains and fault_type in ['B', 'IR', 'OR']:
            all_files[load][fault_type].append(file)
    
    # 타겟 디렉토리 생성
    for load in load_domains.keys():
        domain_dir = os.path.join(target_dir, load_domains[load])
        os.makedirs(domain_dir, exist_ok=True)
    
    # 파일 선택 및 복사
    copied_files = 0
    
    for load in sorted(load_domains.keys()):
        domain_name = load_domains[load]
        print(f"\n📂 {domain_name} 처리 중...")
        
        for fault_type in fault_types:
            candidates = all_files[load][fault_type]
            
            if not candidates:
                print(f"    ❌ {fault_type}: 파일 없음")
                continue
            
            # 최적 파일 선택
            best_file = select_best_file(candidates, fault_type)
            
            if best_file:
                # 새 파일명
                new_filename = f"{fault_type}_{load}hp.mat"
                target_path = os.path.join(target_dir, domain_name, new_filename)
                
                # 파일 복사
                shutil.copy2(best_file, target_path)
                copied_files += 1
                
                # 선택 정보
                _, size, position = get_fault_info(best_file)
                info_parts = []
                if size:
                    info_parts.append(f"크기:{size}")
                if position:
                    info_parts.append(f"위치:{position}")
                
                info_str = f"({', '.join(info_parts)})" if info_parts else ""
                print(f"    ✅ {fault_type}: {os.path.basename(best_file)} {info_str}")
    
    print(f"\n✅ 총 {copied_files}개 파일이 복사되었습니다!")
    print(f"📁 생성된 폴더: {target_dir}/")
    
    # 구조 확인
    print(f"\n📊 생성된 구조:")
    for load in sorted(load_domains.keys()):
        domain_dir = os.path.join(target_dir, load_domains[load])
        if os.path.exists(domain_dir):
            files = [f for f in os.listdir(domain_dir) if f.endswith('.mat')]
            print(f"   {load_domains[load]}: {len(files)}개 파일")
            for file in sorted(files):
                print(f"     - {file}")
    
    print(f"\n🎯 Domain-Incremental Learning 준비 완료!")
    print("   - 각 도메인: 4개 클래스 (H, B, IR, OR)")
    print("   - 윈도우 분할: 모든 subset에 모든 클래스 포함")
    print("   - 데이터 누수: 랜덤 윈도우 분할로 최소화")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 모든 작업이 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 작업 중 오류가 발생했습니다.")
