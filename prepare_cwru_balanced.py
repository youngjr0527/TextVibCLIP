#!/usr/bin/env python3
"""
균형 잡힌 CWRU 데이터셋 준비 스크립트
각 클래스당 동일한 수의 파일로 균형 맞춤

목표: H:1, B:1, IR:1, OR:1 (각 도메인당)
"""

import os
import shutil
import glob
import random
import re
from pathlib import Path
from collections import defaultdict


class BalancedCWRUProcessor:
    """균형 잡힌 CWRU 데이터 처리 클래스"""
    
    def __init__(self, source_dir="cwru_data", target_dir="data_scenario2"):
        self.source_dir = source_dir
        self.target_dir = target_dir
        
        self.load_domains = {
            '0': 'Load_0hp',
            '1': 'Load_1hp', 
            '2': 'Load_2hp',
            '3': 'Load_3hp'
        }
        
        self.fault_types = ['H', 'B', 'IR', 'OR']
        self.file_stats = defaultdict(lambda: defaultdict(int))
        self.copied_files = 0
        
    def extract_load_from_filename(self, filepath):
        """파일명에서 부하 추출"""
        filename = os.path.basename(filepath)
        
        match = re.search(r'_(\d+)\.mat$', filename)
        if match:
            return match.group(1)
        
        match = re.search(r'@\d+_(\d+)\.mat$', filename)
        if match:
            return match.group(1)
        
        return None
    
    def get_fault_type_from_path(self, filepath):
        """경로에서 결함 타입 추출"""
        path_parts = filepath.split(os.sep)
        
        if 'Normal' in path_parts:
            return 'H'
        
        for part in path_parts:
            if part in ['B', 'IR', 'OR']:
                return part
        
        return None
    
    def collect_balanced_files(self):
        """균형 잡힌 파일 수집 (각 클래스당 1개씩)"""
        print("📊 균형 잡힌 파일 수집 중...")
        
        load_files = defaultdict(lambda: defaultdict(list))
        
        # Normal 파일들
        normal_files = glob.glob(os.path.join(self.source_dir, "Normal", "*.mat"))
        for file in normal_files:
            load = self.extract_load_from_filename(file)
            if load in self.load_domains:
                load_files[load]['H'].append(file)
        
        # Fault 파일들
        fault_base = os.path.join(self.source_dir, "12k_Drive_End_Bearing_Fault_Data")
        fault_files = glob.glob(os.path.join(fault_base, "**", "*.mat"), recursive=True)
        
        for file in fault_files:
            load = self.extract_load_from_filename(file)
            fault_type = self.get_fault_type_from_path(file)
            
            if load in self.load_domains and fault_type in self.fault_types:
                load_files[load][fault_type].append(file)
        
        # 🎯 균형 맞춤: 각 클래스당 1개씩만 선택
        balanced_files = defaultdict(lambda: defaultdict(list))
        
        for load in self.load_domains.keys():
            print(f"\n  Load {load}HP 균형 맞춤:")
            for fault_type in self.fault_types:
                available_files = load_files[load][fault_type]
                if available_files:
                    # 랜덤하게 1개 선택
                    selected_file = random.choice(available_files)
                    balanced_files[load][fault_type] = [selected_file]
                    print(f"    {fault_type}: {len(available_files)}개 중 1개 선택")
                else:
                    print(f"    {fault_type}: 파일 없음 ⚠️")
        
        return balanced_files
    
    def copy_balanced_files(self, balanced_files):
        """균형 잡힌 파일들 복사"""
        print(f"\n📁 균형 잡힌 파일 복사: {self.target_dir}")
        
        # 기존 폴더 처리
        if os.path.exists(self.target_dir):
            response = input(f"⚠️  {self.target_dir} 폴더가 이미 존재합니다. 삭제하고 새로 생성하시겠습니까? (y/N): ")
            if response.lower() == 'y':
                shutil.rmtree(self.target_dir)
            else:
                print("❌ 스크립트를 종료합니다.")
                return False
        
        # 타겟 디렉토리 생성
        for load in self.load_domains.keys():
            domain_dir = os.path.join(self.target_dir, self.load_domains[load])
            os.makedirs(domain_dir, exist_ok=True)
        
        # 파일 복사
        for load in sorted(self.load_domains.keys()):
            domain_name = self.load_domains[load]
            print(f"\n📂 {domain_name} 처리 중...")
            
            for fault_type in self.fault_types:
                files = balanced_files[load][fault_type]
                
                if not files:
                    print(f"    ⚠️  {fault_type}: 파일 없음")
                    continue
                
                filepath = files[0]  # 1개만 있음
                new_filename = f"{fault_type}_{load}hp.mat"
                target_path = os.path.join(self.target_dir, domain_name, new_filename)
                
                # 파일 복사
                shutil.copy2(filepath, target_path)
                self.copied_files += 1
                self.file_stats[domain_name][fault_type] += 1
                
                print(f"    ✅ {os.path.basename(filepath)} → {new_filename}")
        
        return True
    
    def print_statistics(self):
        """복사 결과 통계"""
        print(f"\n✅ 총 {self.copied_files}개 파일이 복사되었습니다!")
        print("\n📊 균형 잡힌 파일 통계:")
        
        for domain_name in sorted(self.file_stats.keys()):
            print(f"\n  📁 {domain_name}:")
            for fault_type in self.fault_types:
                count = self.file_stats[domain_name].get(fault_type, 0)
                print(f"    {fault_type}: {count}개")
        
        # 균형 검증
        all_counts = []
        for domain_name in self.file_stats.keys():
            for fault_type in self.fault_types:
                count = self.file_stats[domain_name].get(fault_type, 0)
                if count > 0:
                    all_counts.append(count)
        
        if all_counts and len(set(all_counts)) == 1:
            print(f"\n✅ 완벽한 균형 달성! (각 클래스당 {all_counts[0]}개)")
        else:
            print(f"\n⚠️  불균형 존재: {set(all_counts)}")


def main():
    """메인 실행 함수"""
    print("🚀 균형 잡힌 CWRU 데이터 준비 시작!")
    print("=" * 50)
    print("🎯 목표: 각 클래스당 1개 파일 (H:1, B:1, IR:1, OR:1)")
    print("🎯 효과: 클래스 균형으로 안정적 학습")
    print("=" * 50)
    
    # 시드 설정
    random.seed(42)
    
    # 프로세서 초기화
    processor = BalancedCWRUProcessor()
    
    # 소스 디렉토리 확인
    if not os.path.exists(processor.source_dir):
        print(f"❌ 오류: {processor.source_dir} 폴더를 찾을 수 없습니다.")
        return False
    
    try:
        # 1. 균형 잡힌 파일 수집
        balanced_files = processor.collect_balanced_files()
        
        # 2. 파일 복사
        if processor.copy_balanced_files(balanced_files):
            # 3. 통계 출력
            processor.print_statistics()
            
            print(f"\n🎉 균형 잡힌 CWRU 데이터 준비 완료!")
            print(f"📁 생성된 폴더: {processor.target_dir}/")
            print("✅ 클래스 균형으로 안정적 학습 가능!")
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
