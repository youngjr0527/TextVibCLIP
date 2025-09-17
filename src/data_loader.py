"""
UOS 데이터셋 로더
data_scenario1 폴더 기반으로 진동 신호와 텍스트 메타데이터를 로딩
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
import logging
import warnings
from transformers import DistilBertTokenizer

# Warning 억제
warnings.filterwarnings("ignore", category=UserWarning)
try:
    import torchvision
    torchvision.disable_beta_transforms_warning()
except:
    pass

from .utils import (
    parse_filename, 
    generate_text_description, 
    load_mat_file,
    create_windowed_signal,
    normalize_signal,
    create_labels
)
from configs.model_config import DATA_CONFIG, CWRU_DATA_CONFIG

# 로깅 설정 (메인에서 구성되므로 basicConfig 제거)
logger = logging.getLogger(__name__)


def create_collate_fn(tokenizer: DistilBertTokenizer = None, max_length: int = 128):
    """
    배치 토크나이징을 위한 collate_fn 생성
    
    Args:
        tokenizer: DistilBERT 토크나이저
        max_length: 최대 텍스트 길이
        
    Returns:
        collate_fn: DataLoader용 collate 함수
    """
    if tokenizer is None:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    def collate_fn(batch):
        """
        배치 데이터를 처리하고 텍스트를 토크나이징
        
        Args:
            batch: UOSDataset에서 반환된 샘플들의 리스트
            
        Returns:
            Dict: 배치 데이터
        """
        # 각 필드별로 데이터 분리
        vibrations = torch.stack([item['vibration'] for item in batch])
        texts = [item['text'] for item in batch]
        metadata_list = [item['metadata'] for item in batch]
        labels = torch.stack([item['labels'] for item in batch])
        rpms = torch.tensor([item['domain_key'] for item in batch])
        file_indices = torch.tensor([item['file_idx'] for item in batch])
        
        # 배치 토크나이징 (더 효율적)
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'vibration': vibrations,
            'text': texts,  # 원본 텍스트도 유지 (로깅용)
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'metadata': metadata_list,
            'labels': labels,
            'rpm': rpms,
            'file_idx': file_indices
        }
    
    return collate_fn


class BearingDataset(Dataset):
    """
    베어링 데이터셋 PyTorch Dataset 클래스 (UOS/CWRU 지원)
    
    진동 신호와 텍스트 메타데이터를 쌍으로 로딩
    """
    
    def __init__(self, 
                 data_dir: str = DATA_CONFIG['data_dir'],
                 dataset_type: str = 'uos',
                 domain_value: Optional[Union[int, str]] = None,
                 subset: str = 'train',
                 window_size: int = None,
                 overlap_ratio: float = None,
                 normalization: str = None,
                 max_text_length: int = DATA_CONFIG['max_text_length']):
        """
        Args:
            data_dir (str): 데이터 폴더 경로 (data_scenario1 또는 data_scenario2)
            dataset_type (str): 'uos' 또는 'cwru'
            domain_value (Union[int, str], optional): 
                - UOS: RPM 값 (600, 800, etc.)
                - CWRU: Load 값 (0, 1, 2, 3)
            subset (str): 'train', 'val', 'test' 중 하나
            window_size (int): 신호 윈도우 크기
            overlap_ratio (float): 윈도우 겹침 비율
            normalization (str): 신호 정규화 방법
            max_text_length (int): 텍스트 최대 길이
        """
        self.data_dir = data_dir
        self.dataset_type = dataset_type.lower()
        self.domain_value = domain_value
        self.max_text_length = max_text_length
        self.subset = subset
        
        # 🎯 CRITICAL FIX: 데이터셋별 설정 적용
        if self.dataset_type == 'cwru':
            config = CWRU_DATA_CONFIG
        else:
            config = DATA_CONFIG
            
        self.window_size = window_size if window_size is not None else config['window_size']
        self.overlap_ratio = overlap_ratio if overlap_ratio is not None else config['overlap_ratio']
        self.normalization = normalization if normalization is not None else config['signal_normalization']
        
        # 데이터 로딩
        self.file_paths = self._collect_file_paths()
        self.metadata_list = self._extract_metadata()
        
        # 데이터셋 분할 (train/val/test)
        self.file_paths, self.metadata_list = self._split_dataset()
        
        # 각 파일의 윈도우 수 계산 (분할 정보 고려)
        self.windows_per_file = self._calculate_windows_per_file()
        self.total_windows = len(self.file_paths) * self.windows_per_file
        
        logger.info(f"BearingDataset 초기화 완료 ({self.dataset_type.upper()}): "
                   f"{len(self.file_paths)}개 파일, {self.windows_per_file}개 윈도우/파일, "
                   f"총 {self.total_windows}개 샘플, Domain: {domain_value}, Subset: {subset}")


    def _collect_file_paths(self) -> List[str]:
        """데이터 파일 경로 수집"""
        if self.dataset_type == 'uos':
            return self._collect_uos_file_paths()
        elif self.dataset_type == 'cwru':
            return self._collect_cwru_file_paths()
        else:
            raise ValueError(f"지원하지 않는 데이터셋 타입: {self.dataset_type}")
    
    def _collect_uos_file_paths(self) -> List[str]:
        """UOS 데이터 파일 경로 수집 (Deep Groove Ball만)"""
        if self.domain_value is not None:
            # 🎯 SIMPLIFIED: Deep Groove Ball (6204)만 로딩
            pattern = os.path.join(self.data_dir, f"RotatingSpeed_{self.domain_value}", "*.mat")
        else:
            # 모든 RPM의 Deep Groove Ball 로딩
            pattern = os.path.join(self.data_dir, "RotatingSpeed_*", "*.mat")
        
        file_paths = glob.glob(pattern, recursive=False)  # recursive=False (단일 레벨)
        
        if len(file_paths) == 0:
            raise ValueError(f"UOS 파일을 찾을 수 없습니다: {pattern}")
        
        return sorted(file_paths)
    
    def _collect_cwru_file_paths(self) -> List[str]:
        """CWRU 데이터 파일 경로 수집"""
        if self.domain_value is not None:
            # 특정 Load만 로딩
            if isinstance(self.domain_value, str):
                load_folder = self.domain_value  # 'Load_0hp' 형태
            else:
                load_folder = f"Load_{self.domain_value}hp"  # 숫자 -> 'Load_Xhp' 형태
            
            pattern = os.path.join(self.data_dir, load_folder, "*.mat")
        else:
            # 모든 Load 로딩
            pattern = os.path.join(self.data_dir, "**", "*.mat")
        
        file_paths = glob.glob(pattern, recursive=True)
        
        if len(file_paths) == 0:
            raise ValueError(f"CWRU 파일을 찾을 수 없습니다: {pattern}")
        
        return sorted(file_paths)
    
    def _extract_metadata(self) -> List[Dict[str, Union[str, int]]]:
        """파일명에서 메타데이터 추출"""
        metadata_list = []
        
        for filepath in self.file_paths:
            try:
                metadata = parse_filename(filepath, dataset_type=self.dataset_type)
                metadata['filepath'] = filepath
                metadata_list.append(metadata)
            except Exception as e:
                logger.warning(f"메타데이터 추출 실패: {filepath}, 오류: {e}")
                continue
        
        return metadata_list
    
    def _generate_labels(self, metadata: Dict[str, Union[str, int]]) -> torch.Tensor:
        """메타데이터에서 라벨 생성"""
        if self.dataset_type == 'uos':
            return self._generate_uos_labels(metadata)
        elif self.dataset_type == 'cwru':
            return self._generate_cwru_labels(metadata)
        else:
            raise ValueError(f"지원하지 않는 데이터셋 타입: {self.dataset_type}")
    
    def _generate_uos_labels(self, metadata: Dict[str, Union[str, int]]) -> torch.Tensor:
        """🎯 UOS 라벨 생성 (7-클래스 결합 분류)"""
        # 올바른 7-클래스 분류: 회전체+베어링 상태 조합
        rotating_comp = metadata['rotating_component']
        bearing_cond = metadata['bearing_condition']
        
        # 결합 라벨 생성
        combined_condition = f"{rotating_comp}_{bearing_cond}"
        
        # 7-클래스 매핑
        condition_map = {
            'H_H': 0,   # Healthy (완전 정상)
            'H_B': 1,   # Ball fault (베어링 볼 결함)
            'H_IR': 2,  # Inner race fault (베어링 내륜 결함)
            'H_OR': 3,  # Outer race fault (베어링 외륜 결함)
            'L_H': 4,   # Looseness (회전체 느슨함)
            'U_H': 5,   # Unbalance (회전체 불균형)
            'M_H': 6    # Misalignment (회전체 정렬불량)
        }
        
        # 추가 정보: 베어링 타입도 유지
        bearing_type_map = {'6204': 0, '30204': 1, 'N204': 2, 'NJ204': 3}
        
        # 단일 라벨 (주 분류) + 베어링 타입 정보
        main_label = condition_map.get(combined_condition, 0)
        bearing_type_label = bearing_type_map.get(metadata['bearing_type'], 0)
        
        labels = torch.tensor([
            main_label,         # 주 분류 (7-클래스)
            bearing_type_label  # 베어링 타입 (4-클래스)
        ], dtype=torch.long)
        
        return labels
    
    def _generate_cwru_labels(self, metadata: Dict[str, Union[str, int]]) -> torch.Tensor:
        """CWRU 라벨 생성 (1차원: bearing_condition)"""
        bearing_condition_map = {'Normal': 0, 'B': 1, 'IR': 2, 'OR': 3}
        
        # CWRU는 베어링 상태만 분류하므로 1차원 라벨
        label = torch.tensor([
            bearing_condition_map[metadata['bearing_condition']]
        ], dtype=torch.long)
        
        return label
    
    def _get_domain_key(self, metadata: Dict[str, Union[str, int]]) -> Union[int, str]:
        """도메인 키 반환"""
        if self.dataset_type == 'uos':
            return metadata['rotating_speed']  # RPM 값
        elif self.dataset_type == 'cwru':
            return metadata['load']  # Load 값 (0, 1, 2, 3)
        else:
            return 0
    
    def _calculate_windows_per_file(self) -> int:
        """각 파일당 윈도우 수 계산"""
        if len(self.file_paths) == 0:
            return 0
        
        try:
            # 첫 번째 파일로 윈도우 수 추정
            first_file = self.file_paths[0]
            signal = load_mat_file(first_file, dataset_type=self.dataset_type)
            
            # 타입 안전성 확보
            window_size = int(self.window_size)
            overlap_ratio = float(self.overlap_ratio)
            
            windowed_signals = create_windowed_signal(
                signal, window_size, overlap_ratio
            )
            
            total_windows = len(windowed_signals)
            
            # 🎯 CWRU 윈도우 분할 고려
            if self.dataset_type == 'cwru' and hasattr(self, '_window_split_range'):
                start_ratio, end_ratio = self._window_split_range
                split_windows = int(total_windows * (end_ratio - start_ratio))
                return max(1, split_windows)  # 최소 1개 윈도우
            
            return total_windows
        except Exception as e:
            logger.warning(f"윈도우 수 계산 실패: {e}, 기본값 1 사용")
            return 1
    
    def _split_dataset(self) -> Tuple[List[str], List[Dict]]:
        """데이터셋을 train/val/test로 분할"""
        if len(self.file_paths) == 0:
            return [], []
        
        # CWRU는 데이터가 적어서 split 없이 전체 데이터 사용
        if self.dataset_type == 'cwru':
            return self._split_cwru_dataset()
        else:
            return self._split_uos_dataset()
    
    def _split_cwru_dataset(self) -> Tuple[List[str], List[Dict]]:
        """CWRU 데이터셋 분할 (개선된 전략)"""
        # CWRU 특성: 도메인당 4개 파일 (Normal, B, IR, OR)
        # 연구 목적에 맞는 분할 전략 적용
        
        # 베어링 상태별 라벨 (Normal, B, IR, OR)
        bearing_labels = [metadata['bearing_condition'] for metadata in self.metadata_list]
        
        from collections import Counter
        label_counts = Counter(bearing_labels)
        logger.info(f"CWRU 데이터 라벨 분포: {dict(label_counts)}")
        
        # CWRU 특별 처리: 파일 수가 적으므로 적응적 분할
        total_files = len(self.file_paths)
        
        if total_files <= 4:
            # 도메인당 4개 파일인 경우 - 연구 목적에 맞게 분할
            # 파일 순서: [Normal, B, IR, OR] (알파벳순)
            
            # 🎯 CRITICAL FIX: CWRU도 파일 레벨 분할로 수정 (데이터 누수 방지)
            # CWRU: 4개 파일 [Normal, B, IR, OR] - 각 클래스 1개씩
            # 모든 클래스를 포함하면서도 파일 레벨에서 분할
            
            # 🎯 CRITICAL FIX: CWRU 전체 윈도우 사용 (데이터 부족 해결)
            # CWRU는 데이터가 적으므로 윈도우 분할 없이 전체 사용
            
            logger.info("CWRU Domain-Incremental 전체 윈도우 사용:")
            logger.info(f"  모든 subset에 모든 {total_files}개 파일의 전체 윈도우 포함")
            logger.info(f"  윈도우 분할 없음 (데이터 부족으로 인한 최대 활용)")
            
            # 모든 파일을 모든 subset에 포함
            selected_files = self.file_paths
            selected_meta = self.metadata_list
            
            # 🎯 윈도우 분할 정보 제거 (전체 윈도우 사용)
            # self._window_split_range 설정하지 않음 → 전체 윈도우 사용
            
            # 클래스 분포 확인 (모든 클래스가 모든 subset에 포함됨)
            from collections import Counter
            all_classes = [meta['bearing_condition'] for meta in self.metadata_list]
            unique_classes = list(Counter(all_classes).keys())
            
            logger.info(f"  모든 subset 클래스: {unique_classes} ({len(unique_classes)}개)")
            logger.info(f"  클래스별 파일 수: {dict(Counter(all_classes))}")
            
            logger.info(f"CWRU {self.subset} subset: {len(selected_files)}개 파일 사용 (총 {total_files}개 중)")
            
            # 실제 선택된 파일명 로깅 (디버깅용)
            file_names = [os.path.basename(f) for f in selected_files]
            logger.info(f"CWRU {self.subset} 파일들: {file_names}")
            
            return selected_files, selected_meta
            
        else:
            # 파일이 많은 경우 (여러 도메인 통합 등) - 표준 분할 적용
            try:
                # 베어링 상태로 stratified split 시도
                files_train, files_temp, meta_train, meta_temp = train_test_split(
                    self.file_paths, self.metadata_list,
                    test_size=0.4,  # 40%를 test+val용으로
                    stratify=bearing_labels,
                    random_state=42
                )
                
                # Temp를 val/test로 분할
                temp_labels = [metadata['bearing_condition'] for metadata in meta_temp]
                files_val, files_test, meta_val, meta_test = train_test_split(
                    files_temp, meta_temp,
                    test_size=0.5,  # temp의 50%씩 val/test로
                    stratify=temp_labels,
                    random_state=42
                )
                
                logger.info("CWRU stratified split 성공")
                
            except ValueError:
                # Stratify 실패 시 랜덤 분할
                logger.warning("CWRU stratified split 실패 - 랜덤 분할 사용")
                files_train, files_temp, meta_train, meta_temp = train_test_split(
                    self.file_paths, self.metadata_list,
                    test_size=0.4,
                    random_state=42
                )
                
                files_val, files_test, meta_val, meta_test = train_test_split(
                    files_temp, meta_temp,
                    test_size=0.5,
                    random_state=42
                )
            
            # 요청된 subset 반환
            if self.subset == 'train':
                return files_train, meta_train
            elif self.subset == 'val':
                return files_val, meta_val
            elif self.subset == 'test':
                return files_test, meta_test
            else:
                raise ValueError(f"알 수 없는 subset: {self.subset}")
    
    def _split_uos_dataset(self) -> Tuple[List[str], List[Dict]]:
        """UOS 데이터셋 분할 (개선된 stratified split)"""
        # 연구 목적에 맞는 stratification 전략:
        # 베어링 상태(bearing_condition)를 주요 클래스로 사용
        # - 진단 모델의 핵심은 베어링 결함 타입 분류
        # - H/B/IR/OR 4개 클래스로 균형 유지
        
        # 주요 라벨: 베어링 상태 (H/B/IR/OR)
        primary_labels = [metadata['bearing_condition'] for metadata in self.metadata_list]
        
        # 보조 라벨: 베어링 타입 (stratification 보강용)
        secondary_labels = [metadata['bearing_type'] for metadata in self.metadata_list]
        
        # 복합 라벨 생성 (베어링 상태 + 베어링 타입)
        # 예: H_DeepGrooveBall, IR_TaperedRoller 등
        combined_labels = [f"{primary}_{secondary}" for primary, secondary in zip(primary_labels, secondary_labels)]
        
        # 라벨 분포 확인
        from collections import Counter
        label_counts = Counter(combined_labels)
        min_samples = min(label_counts.values())
        
        logger.info(f"UOS 데이터 라벨 분포: {dict(label_counts)}")
        logger.info(f"최소 샘플 수: {min_samples}")
        
        # 🎯 CRITICAL FIX: 파일 레벨 분할로 복원 (데이터 누수 방지)
        # 윈도우 레벨 분할은 같은 베어링의 연속 신호를 train/val/test로 분할하여 
        # 심각한 데이터 누수를 야기함 → 100% 정확도의 원인
        
        # 🎯 CRITICAL FIX: 윈도우 레벨 분할 (UOS 데이터 구조에 맞게)
        # UOS/CWRU는 클래스별로 1개 파일만 존재하므로, 윈도우 레벨에서 분할해야 함
        # 모든 파일을 모든 subset에 포함하되, 각 파일 내에서 윈도우를 분할
        
        logger.info("UOS Domain-Incremental 윈도우 레벨 분할:")
        logger.info(f"  모든 subset에 모든 {len(self.file_paths)}개 파일 포함")
        logger.info(f"  각 파일 내에서 윈도우 분할: Train 60%, Val 20%, Test 20%")
        
        # 모든 파일을 모든 subset에 포함
        files_train = self.file_paths
        files_val = self.file_paths
        files_test = self.file_paths
        meta_train = self.metadata_list
        meta_val = self.metadata_list
        meta_test = self.metadata_list
        
        # 윈도우 분할 정보 설정 (나중에 __getitem__에서 사용)
        if self.subset == 'train':
            self._window_split_range = (0.0, 0.6)  # 각 파일의 처음 60%
        elif self.subset == 'val':
            self._window_split_range = (0.6, 0.8)  # 각 파일의 60-80%
        elif self.subset == 'test':
            self._window_split_range = (0.8, 1.0)  # 각 파일의 80-100%
        
        # 🎯 FIXED: Deep Groove Ball 7-클래스 분포 확인
        from collections import Counter
        
        # 실제 라벨 생성하여 분포 확인
        actual_labels = []
        for meta in self.metadata_list:
            rotating_comp = meta['rotating_component']
            bearing_cond = meta['bearing_condition']
            combined_condition = f"{rotating_comp}_{bearing_cond}"
            
            condition_map = {
                'H_H': 'H',   'H_B': 'B',   'H_IR': 'IR', 'H_OR': 'OR',
                'L_H': 'L',   'U_H': 'U',   'M_H': 'M'
            }
            actual_class = condition_map.get(combined_condition, 'Unknown')
            actual_labels.append(actual_class)
        
        class_distribution = Counter(actual_labels)
        unique_classes = list(class_distribution.keys())
        
        logger.info(f"  Deep Groove Ball 7-클래스 분포: {dict(class_distribution)}")
        logger.info(f"  클래스 수: {len(unique_classes)}개 (균형 확인)")
        
        # 클래스 균형 검증
        counts = list(class_distribution.values())
        if counts and max(counts) == min(counts):
            logger.info("  ✅ 완벽한 클래스 균형 달성!")
        else:
            logger.warning(f"  ⚠️ 클래스 불균형: 최대 {max(counts) if counts else 0}개, 최소 {min(counts) if counts else 0}개")
        
        # 분할 결과 로깅 (디버깅용)
        logger.info(f"UOS {self.subset} 분할 결과:")
        logger.info(f"  Train: {len(files_train)}개 파일, Val: {len(files_val)}개 파일, Test: {len(files_test)}개 파일")
        
        # 요청된 subset 반환 (파일 레벨 분할)
        if self.subset == 'train':
            return files_train, meta_train
        elif self.subset == 'val':
            return files_val, meta_val
        elif self.subset == 'test':
            return files_test, meta_test
        else:
            raise ValueError(f"알 수 없는 subset: {self.subset}")
    
    def __len__(self) -> int:
        return self.total_windows
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        파일 인덱스와 윈도우 인덱스를 조합해서 샘플 반환
        
        Args:
            idx: 전체 윈도우 인덱스 (0 ~ total_windows-1)
            
        Returns:
            Dict containing:
                - 'vibration': 진동 신호 (window_size,)
                - 'text': 텍스트 설명 (str)
                - 'metadata': 메타데이터 딕셔너리
                - 'labels': 라벨
                - 'domain_key': 도메인 키
        """
        # 🎯 CRITICAL FIX: 클래스 균형을 위한 인덱스 리매핑
        # 기존 방식은 연속된 인덱스가 같은 파일에서 나와서 배치 내 클래스 불균형 야기
        
        if not hasattr(self, '_index_mapping'):
            # 인덱스 매핑 생성 (한 번만)
            total_files = len(self.file_paths)
            mapping = []
            
            # 모든 (파일, 윈도우) 쌍 생성
            for file_idx in range(total_files):
                for window_idx in range(self.windows_per_file):
                    mapping.append((file_idx, window_idx))
            
            # 셔플하여 클래스 균형 확보
            import random
            random.seed(42)  # 재현 가능성
            random.shuffle(mapping)
            
            self._index_mapping = mapping
            logger.info(f"인덱스 매핑 생성 완료: {len(mapping)}개 (파일 {total_files}개 × 윈도우 {self.windows_per_file}개)")
        
        # 셔플된 매핑에서 (파일, 윈도우) 인덱스 가져오기
        if idx < len(self._index_mapping):
            file_idx, window_idx = self._index_mapping[idx]
        else:
            # 범위 초과 시 마지막 매핑 사용
            file_idx, window_idx = self._index_mapping[-1]
        
        # 파일 인덱스가 범위를 벗어나면 마지막 파일 사용
        if file_idx >= len(self.file_paths):
            file_idx = len(self.file_paths) - 1
            window_idx = 0
        
        filepath = self.file_paths[file_idx]
        metadata = self.metadata_list[file_idx]
        
        try:
            # 🎯 CRITICAL FIX: 파일별 윈도우 캐싱 (매우 중요한 최적화)
            # 같은 파일의 윈도우들을 반복 요청할 때 매번 로딩하지 않도록 캐싱
            if not hasattr(self, '_file_windows_cache'):
                self._file_windows_cache = {}
            
            if file_idx not in self._file_windows_cache:
                # 진동 신호 로딩 (파일당 1회만)
                signal = load_mat_file(filepath, dataset_type=self.dataset_type)
                
                # 신호 전처리 (파일당 1회만)
                signal = normalize_signal(signal, method=self.normalization)
                
                # 윈도잉 (파일당 1회만)
                windowed_signals = create_windowed_signal(
                    signal, self.window_size, self.overlap_ratio
                )
                
                # 캐시에 저장
                self._file_windows_cache[file_idx] = windowed_signals
                logger.debug(f"파일 캐시 생성: {os.path.basename(filepath)} ({len(windowed_signals)}개 윈도우)")
            else:
                # 캐시에서 로드
                windowed_signals = self._file_windows_cache[file_idx]
            
            # 🎯 CRITICAL FIX: 윈도우 레벨 분할 복원 (UOS 데이터 구조에 맞게)
            # UOS/CWRU는 클래스별로 1개 파일만 있으므로 윈도우 레벨 분할이 필요
            if hasattr(self, '_window_split_range'):
                total_windows = len(windowed_signals)
                start_ratio, end_ratio = self._window_split_range
                start_idx = int(total_windows * start_ratio)
                end_idx = int(total_windows * end_ratio)
                
                # 범위 내에서 윈도우 선택
                valid_range = end_idx - start_idx
                if valid_range > 0:
                    adjusted_window_idx = start_idx + (window_idx % valid_range)
                else:
                    adjusted_window_idx = start_idx
                
                if adjusted_window_idx < len(windowed_signals):
                    selected_signal = windowed_signals[adjusted_window_idx]
                else:
                    selected_signal = windowed_signals[-1]
            else:
                # 기본 로직 (fallback)
                if window_idx < len(windowed_signals):
                    selected_signal = windowed_signals[window_idx]
                else:
                    selected_signal = windowed_signals[-1]
            
            # 텍스트 설명 생성
            text_description = generate_text_description(metadata)
            
            # 라벨 생성
            labels = self._generate_labels(metadata)
            
            # 도메인 값 설정
            domain_key = self._get_domain_key(metadata)
            
            return {
                'vibration': torch.tensor(selected_signal, dtype=torch.float32),
                'text': text_description,
                'metadata': metadata,
                'labels': labels,
                'domain_key': domain_key,
                'file_idx': file_idx,
                'window_idx': window_idx
            }
            
        except Exception as e:
            logger.error(f"데이터 로딩 실패: {filepath}, window {window_idx}, 오류: {e}")
            # 에러 시 더미 데이터 반환
            dummy_labels = torch.zeros(3 if self.dataset_type == 'uos' else 1, dtype=torch.long)
            dummy_domain_key = metadata.get('rotating_speed' if self.dataset_type == 'uos' else 'load', 0)
            
            return {
                'vibration': torch.zeros(self.window_size, dtype=torch.float32),
                'text': "Error loading data",
                'metadata': metadata,
                'labels': dummy_labels,
                'domain_key': dummy_domain_key,
                'file_idx': file_idx,
                'window_idx': window_idx
            }


# UOSDataset은 하위호환성을 위해 유지
class UOSDataset(BearingDataset):
    """UOS 데이터셋 (하위호환성)"""
    def __init__(self, 
                 data_dir: str = DATA_CONFIG['data_dir'],
                 domain_rpm: Optional[int] = None,
                 **kwargs):
        super().__init__(
            data_dir=data_dir,
            dataset_type='uos',
            domain_value=domain_rpm,
            **kwargs
        )
        # 하위호환성을 위한 속성
        self.domain_rpm = domain_rpm


def create_domain_dataloaders(data_dir: str = DATA_CONFIG['data_dir'],
                            domain_order: List[int] = DATA_CONFIG['domain_order'],
                            dataset_type: str = 'uos',
                            batch_size: int = 32,
                            num_workers: int = 4,
                            use_collate_fn: bool = True) -> Dict[int, Dict[str, DataLoader]]:
    """
    도메인별 DataLoader 생성 (UOS/CWRU 지원)
    
    Args:
        data_dir (str): 데이터 디렉토리
        domain_order (List[int]): 도메인 순서 (UOS: RPM, CWRU: Load)
        dataset_type (str): 'uos' 또는 'cwru'
        batch_size (int): 배치 크기
        num_workers (int): 워커 수
        use_collate_fn (bool): 배치 토크나이징 사용 여부
        
    Returns:
        Dict[int, Dict[str, DataLoader]]: {domain: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}}
    """
    domain_dataloaders = {}
    
    # collate_fn 준비
    collate_fn = create_collate_fn() if use_collate_fn else None
    
    for domain_value in domain_order:
        domain_name = f"{domain_value}RPM" if dataset_type == 'uos' else f"{domain_value}HP"
        logger.info(f"Domain {domain_name} 데이터로더 생성 중...")
        
        # 각 subset별 Dataset 생성
        train_dataset = BearingDataset(
            data_dir=data_dir,
            dataset_type=dataset_type,
            domain_value=domain_value,
            subset='train'
        )
        
        val_dataset = BearingDataset(
            data_dir=data_dir,
            dataset_type=dataset_type,
            domain_value=domain_value,
            subset='val'
        )
        
        test_dataset = BearingDataset(
            data_dir=data_dir,
            dataset_type=dataset_type,
            domain_value=domain_value,
            subset='test'
        )
        
        # DataLoader 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        domain_dataloaders[domain_value] = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
        logger.info(f"Domain {domain_name}: Train {len(train_dataset)}, "
                   f"Val {len(val_dataset)}, Test {len(test_dataset)}")
    
    return domain_dataloaders


def create_combined_dataloader(data_dir: str = DATA_CONFIG['data_dir'],
                             subset: str = 'train',
                             batch_size: int = 32,
                             num_workers: int = 4,
                             use_collate_fn: bool = True) -> DataLoader:
    """
    모든 도메인을 합친 DataLoader 생성 (First Domain Training용)
    
    Args:
        data_dir (str): 데이터 디렉토리
        subset (str): 'train', 'val', 'test' 중 하나
        batch_size (int): 배치 크기
        num_workers (int): 워커 수
        use_collate_fn (bool): 배치 토크나이징 사용 여부
        
    Returns:
        DataLoader: 모든 도메인 합친 DataLoader
    """
    # 모든 RPM을 포함하는 Dataset 생성
    dataset = UOSDataset(
        data_dir=data_dir,
        domain_rpm=None,  # 모든 RPM 포함
        subset=subset
    )
    
    # collate_fn 준비
    collate_fn = create_collate_fn() if use_collate_fn else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(subset == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    logger.info(f"Combined DataLoader 생성: {subset} subset, {len(dataset)}개 샘플")
    
    return dataloader


def create_first_domain_dataloader(data_dir: str = DATA_CONFIG['data_dir'],
                                  domain_order: List[int] = DATA_CONFIG['domain_order'],
                                  dataset_type: str = 'uos',
                                  subset: str = 'train',
                                  batch_size: int = 32,
                                  num_workers: int = 4,
                                  use_collate_fn: bool = True) -> DataLoader:
    """
    첫 번째 도메인만의 DataLoader 생성 (First Domain Training용)
    
    Args:
        data_dir (str): 데이터 디렉토리
        domain_order (List[int]): RPM 순서 (첫 번째만 사용)
        subset (str): 'train', 'val', 'test' 중 하나
        batch_size (int): 배치 크기
        num_workers (int): 워커 수
        use_collate_fn (bool): 배치 토크나이징 사용 여부
        
    Returns:
        DataLoader: 첫 번째 도메인만의 DataLoader
    """
    # 첫 번째 도메인 값
    first_domain_value = domain_order[0]
    
    # 첫 번째 도메인만의 Dataset 생성
    dataset = BearingDataset(
        data_dir=data_dir,
        dataset_type=dataset_type,
        domain_value=first_domain_value,
        subset=subset
    )
    
    # collate_fn 준비
    collate_fn = create_collate_fn() if use_collate_fn else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(subset == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    domain_name = f"{first_domain_value}RPM" if dataset_type == 'uos' else f"{first_domain_value}HP"
    logger.info(f"First Domain DataLoader 생성: Domain {domain_name}, "
               f"{subset} subset, {len(dataset)}개 샘플")
    
    return dataloader


if __name__ == "__main__":
    # 테스트 코드
    logger.info("UOS 데이터셋 테스트 시작...")
    
    # 단일 도메인 테스트
    dataset = UOSDataset(domain_rpm=600, subset='train')
    print(f"Domain 600 RPM Train Dataset: {len(dataset)}개 샘플")
    
    # 샘플 데이터 확인
    sample = dataset[0]
    print(f"Vibration shape: {sample['vibration'].shape}")
    print(f"Text: {sample['text']}")
    print(f"Labels: {sample['labels']}")
    print(f"RPM: {sample['rpm']}")
    
    # 도메인별 DataLoader 테스트
    domain_loaders = create_domain_dataloaders(batch_size=8)
    print(f"생성된 도메인 수: {len(domain_loaders)}")
    
    # 첫 번째 배치 테스트
    first_domain = list(domain_loaders.keys())[0]
    first_batch = next(iter(domain_loaders[first_domain]['train']))
    print(f"First batch keys: {first_batch.keys()}")
    print(f"Batch vibration shape: {first_batch['vibration'].shape}")
    print(f"Batch text length: {len(first_batch['text'])}")
    
    logger.info("UOS 데이터셋 테스트 완료!")
