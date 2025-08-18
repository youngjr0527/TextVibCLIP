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
from transformers import DistilBertTokenizer

from .utils import (
    parse_filename, 
    generate_text_description, 
    load_mat_file,
    create_windowed_signal,
    normalize_signal,
    create_labels
)
from configs.model_config import DATA_CONFIG

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
        rpms = torch.tensor([item['rpm'] for item in batch])
        
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
            'rpm': rpms
        }
    
    return collate_fn


class UOSDataset(Dataset):
    """
    UOS 베어링 데이터셋 PyTorch Dataset 클래스
    
    진동 신호와 텍스트 메타데이터를 쌍으로 로딩
    """
    
    def __init__(self, 
                 data_dir: str = DATA_CONFIG['data_dir'],
                 domain_rpm: Optional[int] = None,
                 window_size: int = DATA_CONFIG['window_size'],
                 overlap_ratio: float = DATA_CONFIG['overlap_ratio'],
                 normalization: str = DATA_CONFIG['signal_normalization'],
                 max_text_length: int = DATA_CONFIG['max_text_length'],
                 subset: str = 'train'):
        """
        Args:
            data_dir (str): data_scenario1 폴더 경로
            domain_rpm (int, optional): 특정 RPM만 로딩 (None이면 모든 RPM)
            window_size (int): 신호 윈도우 크기
            overlap_ratio (float): 윈도우 겹침 비율
            normalization (str): 신호 정규화 방법
            max_text_length (int): 텍스트 최대 길이
            subset (str): 'train', 'val', 'test' 중 하나
        """
        self.data_dir = data_dir
        self.domain_rpm = domain_rpm
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.normalization = normalization
        self.max_text_length = max_text_length
        self.subset = subset
        
        # 데이터 로딩
        self.file_paths = self._collect_file_paths()
        self.metadata_list = self._extract_metadata()
        
        # 데이터셋 분할 (train/val/test)
        self.file_paths, self.metadata_list = self._split_dataset()
        
        logger.info(f"UOSDataset 초기화 완료: {len(self.file_paths)}개 파일, "
                   f"Domain RPM: {domain_rpm}, Subset: {subset}")
    
    def _collect_file_paths(self) -> List[str]:
        """data_scenario1에서 .mat 파일 경로 수집"""
        if self.domain_rpm is not None:
            # 특정 RPM만 로딩
            pattern = os.path.join(self.data_dir, "**", f"RotatingSpeed_{self.domain_rpm}", "*.mat")
        else:
            # 모든 RPM 로딩
            pattern = os.path.join(self.data_dir, "**", "*.mat")
        
        file_paths = glob.glob(pattern, recursive=True)
        
        if len(file_paths) == 0:
            raise ValueError(f"파일을 찾을 수 없습니다: {pattern}")
        
        return sorted(file_paths)
    
    def _extract_metadata(self) -> List[Dict[str, Union[str, int]]]:
        """파일명에서 메타데이터 추출"""
        metadata_list = []
        
        for filepath in self.file_paths:
            try:
                metadata = parse_filename(filepath)
                metadata['filepath'] = filepath
                metadata_list.append(metadata)
            except Exception as e:
                logger.warning(f"메타데이터 추출 실패: {filepath}, 오류: {e}")
                continue
        
        return metadata_list
    
    def _split_dataset(self) -> Tuple[List[str], List[Dict]]:
        """데이터셋을 train/val/test로 분할"""
        if len(self.file_paths) == 0:
            return [], []
        
        # 클래스별 stratified split을 위한 라벨 생성
        labels = []
        for metadata in self.metadata_list:
            # 3가지 속성을 결합한 복합 라벨 생성
            label = f"{metadata['rotating_component']}_{metadata['bearing_condition']}_{metadata['bearing_type']}"
            labels.append(label)
        
        # Train/Test 분할 (80/20)
        files_train, files_test, meta_train, meta_test = train_test_split(
            self.file_paths, self.metadata_list, 
            test_size=DATA_CONFIG['test_split'],
            stratify=labels, 
            random_state=42
        )
        
        # Train에서 Validation 분할
        if len(files_train) > 1:
            labels_train = [f"{m['rotating_component']}_{m['bearing_condition']}_{m['bearing_type']}" 
                           for m in meta_train]
            
            files_train_final, files_val, meta_train_final, meta_val = train_test_split(
                files_train, meta_train,
                test_size=DATA_CONFIG['validation_split'] / (1 - DATA_CONFIG['test_split']),
                stratify=labels_train,
                random_state=42
            )
        else:
            files_train_final, files_val = files_train, []
            meta_train_final, meta_val = meta_train, []
        
        # 요청된 subset 반환
        if self.subset == 'train':
            return files_train_final, meta_train_final
        elif self.subset == 'val':
            return files_val, meta_val
        elif self.subset == 'test':
            return files_test, meta_test
        else:
            raise ValueError(f"알 수 없는 subset: {self.subset}")
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict containing:
                - 'vibration': 진동 신호 (window_size,)
                - 'text': 텍스트 설명 (str)
                - 'metadata': 메타데이터 딕셔너리
                - 'labels': 라벨 (3차원: rotating_component, bearing_condition, bearing_type)
        """
        filepath = self.file_paths[idx]
        metadata = self.metadata_list[idx]
        
        try:
            # 진동 신호 로딩
            signal = load_mat_file(filepath)
            
            # 신호 전처리
            signal = normalize_signal(signal, method=self.normalization)
            
            # 윈도잉 (여러 윈도우 중 하나를 랜덤하게 선택)
            windowed_signals = create_windowed_signal(
                signal, self.window_size, self.overlap_ratio
            )
            
            # 랜덤하게 하나의 윈도우 선택
            if len(windowed_signals) > 1:
                window_idx = np.random.randint(0, len(windowed_signals))
                selected_signal = windowed_signals[window_idx]
            else:
                selected_signal = windowed_signals[0]
            
            # 텍스트 설명 생성
            text_description = generate_text_description(metadata)
            
            # 라벨 생성 (3가지 분류)
            rotating_component_map = {'H': 0, 'L': 1, 'U': 2, 'M': 3}
            bearing_condition_map = {'H': 0, 'B': 1, 'IR': 2, 'OR': 3}
            bearing_type_map = {'6204': 0, '30204': 1, 'N204': 2, 'NJ204': 3}
            
            labels = torch.tensor([
                rotating_component_map[metadata['rotating_component']],
                bearing_condition_map[metadata['bearing_condition']], 
                bearing_type_map[metadata['bearing_type']]
            ], dtype=torch.long)
            
            return {
                'vibration': torch.tensor(selected_signal, dtype=torch.float32),
                'text': text_description,
                'metadata': metadata,
                'labels': labels,
                'rpm': metadata['rotating_speed']
            }
            
        except Exception as e:
            logger.error(f"데이터 로딩 실패: {filepath}, 오류: {e}")
            # 에러 시 더미 데이터 반환
            return {
                'vibration': torch.zeros(self.window_size, dtype=torch.float32),
                'text': "Error loading data",
                'metadata': metadata,
                'labels': torch.zeros(3, dtype=torch.long),
                'rpm': metadata.get('rotating_speed', 0)
            }


def create_domain_dataloaders(data_dir: str = DATA_CONFIG['data_dir'],
                            domain_order: List[int] = DATA_CONFIG['domain_order'],
                            batch_size: int = 32,
                            num_workers: int = 4,
                            use_collate_fn: bool = True) -> Dict[int, Dict[str, DataLoader]]:
    """
    도메인별(RPM별) DataLoader 생성
    
    Args:
        data_dir (str): 데이터 디렉토리
        domain_order (List[int]): RPM 순서
        batch_size (int): 배치 크기
        num_workers (int): 워커 수
        use_collate_fn (bool): 배치 토크나이징 사용 여부
        
    Returns:
        Dict[int, Dict[str, DataLoader]]: {RPM: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}}
    """
    domain_dataloaders = {}
    
    # collate_fn 준비
    collate_fn = create_collate_fn() if use_collate_fn else None
    
    for rpm in domain_order:
        logger.info(f"Domain {rpm} RPM 데이터로더 생성 중...")
        
        # 각 subset별 Dataset 생성
        train_dataset = UOSDataset(
            data_dir=data_dir, 
            domain_rpm=rpm, 
            subset='train'
        )
        
        val_dataset = UOSDataset(
            data_dir=data_dir,
            domain_rpm=rpm, 
            subset='val'
        )
        
        test_dataset = UOSDataset(
            data_dir=data_dir,
            domain_rpm=rpm,
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
        
        domain_dataloaders[rpm] = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
        logger.info(f"Domain {rpm} RPM: Train {len(train_dataset)}, "
                   f"Val {len(val_dataset)}, Test {len(test_dataset)}")
    
    return domain_dataloaders


def create_combined_dataloader(data_dir: str = DATA_CONFIG['data_dir'],
                             subset: str = 'train',
                             batch_size: int = 32,
                             num_workers: int = 4,
                             use_collate_fn: bool = True) -> DataLoader:
    """
    모든 도메인을 합친 DataLoader 생성 (Joint Training용)
    
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
                                  subset: str = 'train',
                                  batch_size: int = 32,
                                  num_workers: int = 4,
                                  use_collate_fn: bool = True) -> DataLoader:
    """
    첫 번째 도메인만의 DataLoader 생성 (올바른 Joint Training용)
    
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
    # 첫 번째 도메인 RPM
    first_domain_rpm = domain_order[0]
    
    # 첫 번째 도메인만의 Dataset 생성
    dataset = UOSDataset(
        data_dir=data_dir,
        domain_rpm=first_domain_rpm,
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
    
    logger.info(f"First Domain DataLoader 생성: Domain {first_domain_rpm} RPM, "
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
