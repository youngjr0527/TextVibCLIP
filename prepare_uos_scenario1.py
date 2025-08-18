#!/usr/bin/env python3
"""
UOS 데이터셋에서 Continual Learning Scenario 1용 데이터 준비 스크립트

이 스크립트는 UOS 원본 데이터셋에서 다음 조건에 맞는 데이터만 선별하여 
data_scenario1 폴더로 복사합니다:

1. 16kHz 샘플링 레이트만 사용 (8kHz 제외)
2. 단일 결함만 포함 (복합결함 제외)
3. U3->U, M3->M으로 relabel (U1,U2,M1,M2 제외)
4. 깔끔한 디렉토리 구조로 재구성

Usage:
    python prepare_uos_scenario1.py
"""

import os
import shutil
import glob
from pathlib import Path


def main():
    # 설정
    source_dir = "uos_data"
    target_dir = "data_scenario1"
    
    # 소스 디렉토리 확인
    if not os.path.exists(source_dir):
        print(f" 오류: {source_dir} 폴더를 찾을 수 없습니다.")
        print("   UOS 데이터셋이 현재 디렉토리에 있는지 확인해주세요.")
        return False
    
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
    print(f"✅ {target_dir} 폴더를 생성했습니다.")
    
    # 16kHz 파일들 중에서 단일 결함만 선별
    # 포함할 패턴: H_H, H_B, H_IR, H_OR, L_H, U3_H, M3_H
    include_patterns = ['H_H', 'H_B', 'H_IR', 'H_OR', 'L_H', 'U3_H', 'M3_H']
    
    print("\n📁 파일 검색 및 복사 중...")
    
    copied_files = 0
    file_stats = {}
    
    # 모든 16kHz .mat 파일 찾기
    pattern = os.path.join(source_dir, "**", "SamplingRate_16000", "**", "*.mat")
    all_files = glob.glob(pattern, recursive=True)
    
    print(f"   16kHz 파일 총 {len(all_files)}개 발견")
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        # 파일명에서 회전체상태_베어링상태 추출
        parts = filename.split('_')
        if len(parts) < 5:
            continue
            
        condition = f"{parts[0]}_{parts[1]}"
        
        # 포함할 패턴인지 확인
        if condition not in include_patterns:
            continue
        
        # 경로 정보 추출
        path_parts = Path(file_path).parts
        bearing_type = path_parts[-4]  # BearingType_xxx
        rotating_speed = path_parts[-2]  # RotatingSpeed_xxx
        
        # 새 디렉토리 구조 생성
        new_dir = os.path.join(target_dir, bearing_type, rotating_speed)
        os.makedirs(new_dir, exist_ok=True)
        
        # 새 파일명 생성 (U3->U, M3->M, 샘플링레이트 제거)
        new_filename = filename
        new_filename = new_filename.replace('U3_', 'U_')
        new_filename = new_filename.replace('M3_', 'M_')
        new_filename = new_filename.replace('_16_', '_')
        
        # 파일 복사
        new_file_path = os.path.join(new_dir, new_filename)
        shutil.copy2(file_path, new_file_path)
        
        copied_files += 1
        
        # 통계 정보 수집
        condition_key = new_filename.split('_')[0] + '_' + new_filename.split('_')[1]
        if condition_key not in file_stats:
            file_stats[condition_key] = 0
        file_stats[condition_key] += 1
        
        if copied_files % 10 == 0:
            print(f"   진행률: {copied_files}개 복사됨...")
    
    print(f"\n✅ 총 {copied_files}개 파일이 성공적으로 복사되었습니다!")
    
    # 통계 정보 출력
    print("\n📊 복사된 파일 통계:")
    print("   조건별 파일 개수:")
    for condition, count in sorted(file_stats.items()):
        condition_desc = get_condition_description(condition)
        print(f"     {condition}: {count}개 - {condition_desc}")
    
    # 디렉토리 구조 확인
    print(f"\n📂 생성된 디렉토리 구조:")
    print_directory_tree(target_dir)
    
    print(f"\n🎉 data_scenario1 폴더 준비가 완료되었습니다!")
    print(f"   이제 {target_dir} 폴더를 사용하여 Continual Learning 실험을 진행할 수 있습니다.")
    
    return True


def get_condition_description(condition):
    """조건 코드를 설명으로 변환"""
    descriptions = {
        'H_H': '정상 회전체 + 정상 베어링',
        'H_B': '정상 회전체 + 볼 결함',
        'H_IR': '정상 회전체 + 내륜 결함',
        'H_OR': '정상 회전체 + 외륜 결함',
        'L_H': '느슨함 회전체 + 정상 베어링',
        'U_H': '불균형 회전체 + 정상 베어링',
        'M_H': '정렬불량 회전체 + 정상 베어링'
    }
    return descriptions.get(condition, '알 수 없는 조건')


def print_directory_tree(directory, prefix="", max_depth=3, current_depth=0):
    """디렉토리 구조를 트리 형태로 출력"""
    if current_depth >= max_depth:
        return
    
    try:
        items = sorted(os.listdir(directory))
        for i, item in enumerate(items):
            if item.startswith('.'):
                continue
            
            item_path = os.path.join(directory, item)
            is_last = i == len(items) - 1
            
            if os.path.isdir(item_path):
                print(f"{prefix}{'└── ' if is_last else '├── '}{item}/")
                extension = "    " if is_last else "│   "
                print_directory_tree(item_path, prefix + extension, max_depth, current_depth + 1)
            else:
                # 파일 개수만 표시 (너무 많은 파일명 출력 방지)
                if current_depth == max_depth - 1:
                    file_count = len([f for f in items if f.endswith('.mat')])
                    if file_count > 0:
                        print(f"{prefix}{'└── ' if is_last else '├── '}({file_count}개 .mat 파일)")
                    break
    except PermissionError:
        print(f"{prefix}[권한 없음]")


if __name__ == "__main__":
    print("🔧 UOS 데이터셋 Scenario 1 준비 스크립트")
    print("=" * 50)
    
    success = main()
    
    if success:
        print("\n💡 다음 단계:")
        print("   1. data_scenario1 폴더 구조 확인")
        print("   2. TextVibCLIP 모델에서 데이터 로더 설정")
        print("   3. Continual Learning 실험 시작")
    else:
        print("\n❌ 스크립트 실행이 실패했습니다. 오류를 확인해주세요.")
