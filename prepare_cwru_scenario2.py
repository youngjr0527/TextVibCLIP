#!/usr/bin/env python3
"""
최종 CWRU 데이터셋 준비 스크립트 - 원본 파일 구조 활용

핵심 전략:
1. 모든 원본 파일을 그대로 복사 (결함 크기, 위치, 파일번호 다양성 확보)
2. 파일명만 통일: {결함타입}_{부하}hp_{순번}.mat
3. 데이터로더에서 파일 레벨 랜덤 분할 (시간적 누수 방지)

장점:
- 모든 원본 데이터 활용 (데이터 손실 없음)
- 결함 속성 다양성 확보 (크기, 위치, 번호)
- 파일 레벨 분할로 데이터 누수 방지
- 간단하고 안정적인 구현

Usage:
    python prepare_cwru_scenario2_final.py
"""

import os
import shutil
import glob
import random
import re
from pathlib import Path
from collections import defaultdict, Counter


class FinalCWRUProcessor:
    """최종 CWRU 데이터 처리 클래스"""
    
    def __init__(self, source_dir="cwru_data", target_dir="data_scenario2"):
        self.source_dir = source_dir
        self.target_dir = target_dir
        
        # Load별 도메인 매핑
        self.load_domains = {
            '0': 'Load_0hp',
            '1': 'Load_1hp', 
            '2': 'Load_2hp',
            '3': 'Load_3hp'
        }
        
        # 결함 타입 매핑
        self.fault_types = ['H', 'B', 'IR', 'OR']
        
        # 통계 저장
        self.file_stats = defaultdict(lambda: defaultdict(int))
        self.copied_files = 0
        
    def extract_load_from_filename(self, filepath):
        """파일명에서 부하(Load) 정보 정확히 추출"""
        filename = os.path.basename(filepath)
        
        # 표준 패턴: 숫자_부하.mat
        match = re.search(r'_(\d+)\.mat$', filename)
        if match:
            return match.group(1)
        
        # OR 특수 패턴들 처리
        match = re.search(r'@\d+_(\d+)\.mat$', filename)
        if match:
            return match.group(1)
        
        print(f"⚠️  부하 추출 실패: {filename}")
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
    
    def collect_all_files_by_load(self):
        """Load별로 모든 파일 수집"""
        print("📊 Load별 파일 수집 중...")
        
        load_files = defaultdict(lambda: defaultdict(list))  # {load: {fault_type: [files]}}
        
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
        
        # 통계 출력
        print("\n📊 수집된 파일 통계:")
        for load in sorted(self.load_domains.keys()):
            print(f"  Load {load}HP:")
            for fault_type in self.fault_types:
                count = len(load_files[load][fault_type])
                print(f"    {fault_type}: {count}개")
        
        return load_files
    
    def copy_all_files(self, load_files):
        """모든 파일을 새로운 구조로 복사"""
        print(f"\n📁 파일 복사 시작: {self.target_dir}")
        
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
        
        # 파일 복사 실행
        for load in sorted(self.load_domains.keys()):
            domain_name = self.load_domains[load]
            print(f"\n📂 {domain_name} 처리 중...")
            
            for fault_type in self.fault_types:
                files = load_files[load][fault_type]
                
                if not files:
                    print(f"    ⚠️  {fault_type}: 파일 없음")
                    continue
                
                print(f"    📁 {fault_type}: {len(files)}개 파일 복사 중...")
                
                # 파일들을 랜덤하게 섞기 (다양성 확보)
                shuffled_files = files.copy()
                random.shuffle(shuffled_files)
                
                # 모든 파일 복사
                for i, filepath in enumerate(shuffled_files, 1):
                    # 새 파일명 생성 (순번 포함)
                    new_filename = f"{fault_type}_{load}hp_{i:03d}.mat"
                    
                    target_path = os.path.join(self.target_dir, domain_name, new_filename)
                    
                    # 파일 복사
                    shutil.copy2(filepath, target_path)
                    self.copied_files += 1
                    self.file_stats[domain_name][fault_type] += 1
                    
                    # 원본 파일 정보 로깅 (디버깅용)
                    original_info = self._extract_original_info(filepath)
                    print(f"      ✅ {os.path.basename(filepath)} → {new_filename} {original_info}")
        
        return True
    
    def _extract_original_info(self, filepath):
        """원본 파일의 속성 정보 추출 (로깅용)"""
        path_parts = filepath.split(os.sep)
        info_parts = []
        
        for part in path_parts:
            if part in ['007', '014', '021', '028']:
                info_parts.append(f"size:{part}")
            elif part.startswith('@'):
                info_parts.append(f"pos:{part}")
        
        return f"({', '.join(info_parts)})" if info_parts else ""
    
    def print_statistics(self):
        """복사 결과 통계 출력"""
        print(f"\n✅ 총 {self.copied_files}개 파일이 성공적으로 복사되었습니다!")
        print("\n📊 복사된 파일 통계:")
        
        # 도메인별 통계
        for domain_name in sorted(self.file_stats.keys()):
            print(f"\n  📁 {domain_name}:")
            domain_total = 0
            for fault_type in self.fault_types:
                count = self.file_stats[domain_name].get(fault_type, 0)
                print(f"    {fault_type}: {count}개")
                domain_total += count
            print(f"    총계: {domain_total}개")
        
        # 전체 통계
        print(f"\n🎯 전체 통계:")
        for fault_type in self.fault_types:
            total_count = sum(self.file_stats[domain][fault_type] 
                            for domain in self.file_stats.keys())
            print(f"  {fault_type}: {total_count}개")
    
    def verify_dataset(self):
        """생성된 데이터셋 검증"""
        print(f"\n🔍 생성된 데이터셋 검증: {self.target_dir}")
        
        if not os.path.exists(self.target_dir):
            print("❌ 타겟 디렉토리가 존재하지 않습니다!")
            return False
        
        success = True
        
        for load, domain_name in self.load_domains.items():
            domain_path = os.path.join(self.target_dir, domain_name)
            
            if not os.path.exists(domain_path):
                print(f"❌ {domain_name} 폴더가 존재하지 않습니다!")
                success = False
                continue
            
            files = os.listdir(domain_path)
            print(f"  📁 {domain_name}: {len(files)}개 파일")
            
            # 결함 타입별 개수 확인
            type_counts = defaultdict(int)
            for file in files:
                if file.endswith('.mat'):
                    fault_type = file.split('_')[0]
                    type_counts[fault_type] += 1
            
            print(f"    타입별: {dict(type_counts)}")
            
            # 모든 결함 타입이 있는지 확인
            missing_types = set(self.fault_types) - set(type_counts.keys())
            if missing_types:
                print(f"    ⚠️  누락된 타입: {missing_types}")
                success = False
        
        return success


def main():
    """메인 실행 함수"""
    print("🚀 최종 CWRU 데이터 준비 시작!")
    print("=" * 60)
    print("🎯 전략: 모든 원본 파일 활용 + 파일명 통일")
    print("🔄 다양성: 결함 크기, 위치, 파일번호 모두 포함")
    print("🛡️  안전성: 파일 레벨 분할로 데이터 누수 방지")
    print("=" * 60)
    
    # 시드 설정
    random.seed(42)
    
    # 프로세서 초기화
    processor = FinalCWRUProcessor()
    
    # 소스 디렉토리 확인
    if not os.path.exists(processor.source_dir):
        print(f"❌ 오류: {processor.source_dir} 폴더를 찾을 수 없습니다.")
        return False
    
    try:
        # 1. 파일 수집
        load_files = processor.collect_all_files_by_load()
        
        # 2. 파일 복사
        if processor.copy_all_files(load_files):
            # 3. 통계 및 검증
            processor.print_statistics()
            
            if processor.verify_dataset():
                print(f"\n🎉 최종 CWRU 데이터 준비 완료!")
                print(f"📁 생성된 폴더: {processor.target_dir}/")
                print("📋 도메인 구성:")
                for load, domain in processor.load_domains.items():
                    print(f"   - Domain {int(load)+1}: {domain}")
                print("\n✅ 모든 원본 파일 활용으로 데이터 다양성 확보!")
                print("✅ 파일명 통일로 데이터로더 호환성 확보!")
                return True
            else:
                print("❌ 데이터셋 검증 실패")
                return False
        else:
            print("❌ 데이터셋 생성 실패")
            return False
            
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 모든 작업이 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 작업 중 오류가 발생했습니다.")
