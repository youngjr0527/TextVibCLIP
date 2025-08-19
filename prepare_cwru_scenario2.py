#!/usr/bin/env python3
"""
CWRU 데이터셋에서 Continual Learning Scenario 2용 데이터 준비 스크립트

이 스크립트는 CWRU 원본 데이터셋에서 다음 조건에 맞는 데이터만 선별하여 
data_scenario2 폴더로 복사합니다:

1. Drive End 12kHz 데이터만 사용 (Fan End 제외)
2. Normal + 3가지 고장 유형 (B, IR, OR) 사용
3. Fault size는 무시하고 라벨 균형 맞춤
4. Load별 도메인 구성: Domain 1(0 load) → Domain 2(1 load) → Domain 3(2 load) → Domain 4(3 load)
5. 깔끔한 디렉토리 구조로 재구성

Usage:
    python prepare_cwru_scenario2.py
"""

import os
import shutil
import glob
import random
from pathlib import Path
from collections import defaultdict


def analyze_cwru_data(source_dir):
    """CWRU 데이터 분포 분석"""
    print("📊 CWRU 데이터 분석 중...")
    
    # Normal 데이터 분석
    normal_files = glob.glob(os.path.join(source_dir, "Normal", "*.mat"))
    print(f"Normal 데이터: {len(normal_files)}개")
    for file in sorted(normal_files):
        filename = os.path.basename(file)
        load = filename.split('_')[1].split('.')[0]
        print(f"  - {filename}: Load {load}")
    
    # Fault 데이터 분석
    fault_data = defaultdict(lambda: defaultdict(list))
    fault_files = glob.glob(os.path.join(source_dir, "12k_Drive_End_Bearing_Fault_Data", "**", "*.mat"), recursive=True)
    
    for file in fault_files:
        path_parts = Path(file).parts
        
        # 경로에서 fault type 추출 (B, IR, OR)
        if len(path_parts) >= 4:
            fault_type = path_parts[-4]  # 12k_Drive_End_Bearing_Fault_Data 다음 폴더
            if fault_type == '12k_Drive_End_Bearing_Fault_Data':
                fault_type = path_parts[-3]  # B, IR, OR
        else:
            continue
            
        # 경로에서 fault size 추출
        fault_size = path_parts[-2] if len(path_parts) >= 2 else 'unknown'
        
        filename = os.path.basename(file)
        
        # Load 정보 추출
        if '_' in filename:
            load = filename.split('_')[1].split('.')[0]
        elif '@' in filename:
            # OR 데이터의 특수 케이스 (@3, @6, @12)
            if '@3' in file:
                load = '0'
            elif '@6' in file:
                load = '1' 
            elif '@12' in file:
                load = '2'
            else:
                load = '0'
        else:
            load = '0'
        
        fault_data[fault_type][fault_size].append((file, load))
    
    print("\nFault 데이터 분포:")
    for fault_type, sizes in fault_data.items():
        print(f"  {fault_type}:")
        for size, files in sizes.items():
            load_dist = defaultdict(int)
            for _, load in files:
                load_dist[load] += 1
            print(f"    {size}: {len(files)}개 - {dict(load_dist)}")
    
    return normal_files, fault_data


def create_balanced_dataset(normal_files, fault_data, target_dir):
    """라벨 균형을 맞춘 데이터셋 생성"""
    print("\n🎯 라벨 균형 맞춤 데이터셋 생성...")
    
    # 타겟 디렉토리 생성
    if os.path.exists(target_dir):
        print(f"⚠️  {target_dir} 폴더가 이미 존재합니다.")
        response = input("   기존 폴더를 삭제하고 새로 생성하시겠습니까? (y/N): ")
        if response.lower() == 'y':
            shutil.rmtree(target_dir)
            print(f"   기존 {target_dir} 폴더를 삭제했습니다.")
        else:
            print("   스크립트를 종료합니다.")
            return False
    
    os.makedirs(target_dir, exist_ok=True)
    
    # Load별 도메인 구성
    load_domains = {
        '0': 'Load_0hp',
        '1': 'Load_1hp', 
        '2': 'Load_2hp',
        '3': 'Load_3hp'
    }
    
    # 각 도메인별 디렉토리 생성
    for load, domain_name in load_domains.items():
        os.makedirs(os.path.join(target_dir, domain_name), exist_ok=True)
    
    copied_files = 0
    file_stats = defaultdict(lambda: defaultdict(int))
    
    # Normal 데이터 처리
    print("\n📁 Normal 데이터 처리 중...")
    for file in normal_files:
        filename = os.path.basename(file)
        load = filename.split('_')[1].split('.')[0]
        
        if load in load_domains:
            domain_name = load_domains[load]
            new_filename = f"Normal_{load}hp.mat"
            target_path = os.path.join(target_dir, domain_name, new_filename)
            
            shutil.copy2(file, target_path)
            copied_files += 1
            file_stats[domain_name]['Normal'] += 1
            print(f"  복사: {filename} → {domain_name}/{new_filename}")
    
    # Fault 데이터 처리 (라벨 균형 맞춤)
    print("\n📁 Fault 데이터 처리 중 (라벨 균형 맞춤)...")
    
    # 각 fault type별로 load당 동일한 수의 파일 선택
    target_files_per_fault_per_load = 1  # 각 load별로 fault type당 1개씩
    
    for fault_type, sizes in fault_data.items():
        print(f"\n  {fault_type} 처리 중...")
        
        # 모든 fault size의 파일을 load별로 그룹화
        load_grouped_files = defaultdict(list)
        for size, files in sizes.items():
            for file_path, load in files:
                if load in load_domains:
                    load_grouped_files[load].append((file_path, size))
        
        # 각 load별로 균등하게 파일 선택
        for load, files in load_grouped_files.items():
            if len(files) >= target_files_per_fault_per_load:
                # 랜덤하게 선택 (fault size 무시)
                selected_files = random.sample(files, target_files_per_fault_per_load)
            else:
                # 모든 파일 사용
                selected_files = files
            
            domain_name = load_domains[load]
            
            for i, (file_path, size) in enumerate(selected_files):
                filename = os.path.basename(file_path)
                # 새 파일명: {fault_type}_{load}hp_{index}.mat
                new_filename = f"{fault_type}_{load}hp_{i+1}.mat"
                target_path = os.path.join(target_dir, domain_name, new_filename)
                
                shutil.copy2(file_path, target_path)
                copied_files += 1
                file_stats[domain_name][fault_type] += 1
                print(f"    복사: {filename} → {domain_name}/{new_filename}")
    
    # 통계 출력
    print(f"\n✅ 총 {copied_files}개 파일이 성공적으로 복사되었습니다!")
    print("\n📊 복사된 파일 통계:")
    
    total_per_domain = {}
    for domain_name in sorted(file_stats.keys()):
        print(f"\n  {domain_name}:")
        domain_total = 0
        for condition, count in sorted(file_stats[domain_name].items()):
            print(f"    {condition}: {count}개")
            domain_total += count
        print(f"    총계: {domain_total}개")
        total_per_domain[domain_name] = domain_total
    
    print(f"\n📈 도메인별 총 파일 수: {dict(total_per_domain)}")
    
    # 라벨 균형 검증
    all_conditions = set()
    for domain_stats in file_stats.values():
        all_conditions.update(domain_stats.keys())
    
    print(f"\n🎯 라벨 균형 검증:")
    for condition in sorted(all_conditions):
        counts = [file_stats[domain].get(condition, 0) for domain in sorted(file_stats.keys())]
        print(f"  {condition}: {counts} (총 {sum(counts)}개)")
    
    return True


def main():
    # 설정
    source_dir = "cwru_data"
    target_dir = "data_scenario2"
    
    # 소스 디렉토리 확인
    if not os.path.exists(source_dir):
        print(f"❌ 오류: {source_dir} 폴더를 찾을 수 없습니다.")
        print("   CWRU 데이터셋이 현재 디렉토리에 있는지 확인해주세요.")
        return False
    
    print("🚀 CWRU Scenario 2 데이터 준비 시작!")
    print(f"   소스: {source_dir}")
    print(f"   타겟: {target_dir}")
    print("   시나리오: Varying Load (0hp → 1hp → 2hp → 3hp)")
    
    # 시드 설정 (재현 가능한 랜덤 선택)
    random.seed(42)
    
    # 데이터 분석
    normal_files, fault_data = analyze_cwru_data(source_dir)
    
    # 균형 맞춘 데이터셋 생성
    success = create_balanced_dataset(normal_files, fault_data, target_dir)
    
    if success:
        print(f"\n🎉 CWRU Scenario 2 데이터 준비 완료!")
        print(f"📁 생성된 폴더: {target_dir}/")
        print("📋 도메인 구성:")
        print("   - Domain 1: Load_0hp (0 horsepower)")
        print("   - Domain 2: Load_1hp (1 horsepower)")  
        print("   - Domain 3: Load_2hp (2 horsepower)")
        print("   - Domain 4: Load_3hp (3 horsepower)")
        print("🏷️  라벨: Normal, B(Ball), IR(Inner Race), OR(Outer Race)")
    else:
        print("❌ 데이터 준비 실패")
    
    return success


if __name__ == "__main__":
    main()
