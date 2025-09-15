"""
데이터 로딩 캐싱 시스템
반복 실험 시 데이터 로딩 시간을 대폭 단축
"""

import os
import pickle
import hashlib
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataCache:
    """데이터 로딩 캐싱 시스템"""
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Args:
            cache_dir: 캐시 파일 저장 디렉토리
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 캐시 통계
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"DataCache 초기화: {self.cache_dir}")
    
    def _get_cache_key(self, **kwargs) -> str:
        """캐시 키 생성 (파라미터 기반 해싱)"""
        # 중요한 파라미터들만 사용하여 해시 생성
        key_params = {
            'data_dir': kwargs.get('data_dir', ''),
            'dataset_type': kwargs.get('dataset_type', ''),
            'domain_value': kwargs.get('domain_value', ''),
            'subset': kwargs.get('subset', ''),
            'window_size': kwargs.get('window_size', 4096),
            'overlap_ratio': kwargs.get('overlap_ratio', 0.5),
            'normalization': kwargs.get('normalization', 'standardize')
        }
        
        # 해시 생성
        key_str = str(sorted(key_params.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """캐시 파일 경로 생성"""
        return self.cache_dir / f"data_{cache_key}.pkl"
    
    def _get_metadata_path(self, cache_key: str) -> Path:
        """캐시 메타데이터 파일 경로 생성"""
        return self.cache_dir / f"meta_{cache_key}.pkl"
    
    def get_cached_data(self, **kwargs) -> Optional[Dict[str, Any]]:
        """캐시된 데이터 로드"""
        cache_key = self._get_cache_key(**kwargs)
        cache_path = self._get_cache_path(cache_key)
        meta_path = self._get_metadata_path(cache_key)
        
        if not cache_path.exists() or not meta_path.exists():
            self.cache_misses += 1
            return None
        
        try:
            # 메타데이터 확인
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # 데이터 디렉토리 변경 시간 확인
            data_dir = kwargs.get('data_dir', '')
            if os.path.exists(data_dir):
                dir_mtime = os.path.getmtime(data_dir)
                if dir_mtime > metadata.get('cache_time', 0):
                    logger.info(f"데이터 디렉토리가 변경됨, 캐시 무효화: {cache_key[:8]}")
                    self.cache_misses += 1
                    return None
            
            # 캐시된 데이터 로드
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            self.cache_hits += 1
            logger.info(f"✅ 캐시 적중: {cache_key[:8]} (적중률: {self.cache_hits/(self.cache_hits+self.cache_misses)*100:.1f}%)")
            
            return cached_data
            
        except Exception as e:
            logger.warning(f"캐시 로드 실패: {e}")
            self.cache_misses += 1
            return None
    
    def save_cached_data(self, data: Dict[str, Any], **kwargs):
        """데이터를 캐시에 저장"""
        cache_key = self._get_cache_key(**kwargs)
        cache_path = self._get_cache_path(cache_key)
        meta_path = self._get_metadata_path(cache_key)
        
        try:
            # 데이터 저장
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 메타데이터 저장
            metadata = {
                'cache_time': time.time(),
                'cache_key': cache_key,
                'params': kwargs,
                'data_size': len(data.get('samples', []))
            }
            
            with open(meta_path, 'wb') as f:
                pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"💾 캐시 저장: {cache_key[:8]} ({metadata['data_size']}개 샘플)")
            
        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")
    
    def clear_cache(self):
        """캐시 전체 삭제"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir()
            logger.info("🗑️ 캐시 전체 삭제 완료")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        cache_files = list(self.cache_dir.glob("data_*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses) * 100,
            'cached_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024)
        }


# 전역 캐시 인스턴스
_global_cache = None

def get_global_cache() -> DataCache:
    """전역 캐시 인스턴스 반환"""
    global _global_cache
    if _global_cache is None:
        _global_cache = DataCache()
    return _global_cache


def preprocess_and_cache_dataset(data_dir: str, 
                                dataset_type: str,
                                domain_value: Optional[int] = None,
                                subset: str = 'train',
                                window_size: int = 4096,
                                overlap_ratio: float = 0.5,
                                normalization: str = 'standardize') -> Dict[str, Any]:
    """
    데이터셋 전처리 및 캐싱
    
    Returns:
        Dict containing:
            - 'samples': List of preprocessed samples
            - 'file_paths': List of file paths
            - 'metadata': List of metadata
    """
    cache = get_global_cache()
    
    # 캐시 확인
    cache_params = {
        'data_dir': data_dir,
        'dataset_type': dataset_type,
        'domain_value': domain_value,
        'subset': subset,
        'window_size': window_size,
        'overlap_ratio': overlap_ratio,
        'normalization': normalization
    }
    
    cached_data = cache.get_cached_data(**cache_params)
    if cached_data is not None:
        return cached_data
    
    # 캐시 미스 - 실제 데이터 로딩
    logger.info(f"🔄 데이터 전처리 시작: {dataset_type} {domain_value} {subset}")
    start_time = time.time()
    
    # 실제 데이터 로딩 로직 (기존 BearingDataset 로직 사용)
    from .data_loader import BearingDataset
    
    # 임시 데이터셋 생성하여 전처리된 데이터 추출
    temp_dataset = BearingDataset(
        data_dir=data_dir,
        dataset_type=dataset_type,
        domain_value=domain_value,
        subset=subset,
        window_size=window_size,
        overlap_ratio=overlap_ratio,
        normalization=normalization
    )
    
    # 모든 샘플 전처리
    samples = []
    for idx in range(len(temp_dataset)):
        sample = temp_dataset[idx]
        # 텐서를 numpy로 변환 (pickle 호환성)
        processed_sample = {
            'vibration': sample['vibration'].numpy(),
            'text': sample['text'],
            'metadata': sample['metadata'],
            'labels': sample['labels'].numpy(),
            'domain_key': sample['domain_key'],
            'file_idx': sample['file_idx'],
            'window_idx': sample['window_idx']
        }
        samples.append(processed_sample)
    
    # 캐시할 데이터 구성
    cached_data = {
        'samples': samples,
        'file_paths': temp_dataset.file_paths,
        'metadata_list': temp_dataset.metadata_list,
        'windows_per_file': temp_dataset.windows_per_file
    }
    
    # 캐시 저장
    cache.save_cached_data(cached_data, **cache_params)
    
    elapsed_time = time.time() - start_time
    logger.info(f"✅ 데이터 전처리 완료: {len(samples)}개 샘플, {elapsed_time:.1f}초")
    
    return cached_data


class CachedBearingDataset(torch.utils.data.Dataset):
    """캐시된 데이터를 사용하는 고속 Dataset"""
    
    def __init__(self, 
                 data_dir: str,
                 dataset_type: str,
                 domain_value: Optional[int] = None,
                 subset: str = 'train',
                 window_size: int = 4096,
                 overlap_ratio: float = 0.5,
                 normalization: str = 'standardize'):
        """
        Args:
            data_dir: 데이터 디렉토리
            dataset_type: 'uos' 또는 'cwru'
            domain_value: 도메인 값 (RPM 또는 Load)
            subset: 'train', 'val', 'test'
            window_size: 윈도우 크기
            overlap_ratio: 윈도우 겹침 비율
            normalization: 정규화 방법
        """
        # 캐시된 데이터 로드
        self.cached_data = preprocess_and_cache_dataset(
            data_dir=data_dir,
            dataset_type=dataset_type,
            domain_value=domain_value,
            subset=subset,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            normalization=normalization
        )
        
        self.samples = self.cached_data['samples']
        self.file_paths = self.cached_data['file_paths']
        self.metadata_list = self.cached_data['metadata_list']
        
        # 속성 설정
        self.dataset_type = dataset_type
        self.domain_value = domain_value
        self.subset = subset
        
        logger.info(f"CachedBearingDataset 초기화: {len(self.samples)}개 샘플 (캐시 사용)")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """캐시된 샘플 반환"""
        if idx >= len(self.samples):
            idx = len(self.samples) - 1
        
        sample = self.samples[idx]
        
        # numpy를 다시 텐서로 변환
        return {
            'vibration': torch.from_numpy(sample['vibration']).float(),
            'text': sample['text'],
            'metadata': sample['metadata'],
            'labels': torch.from_numpy(sample['labels']).long(),
            'domain_key': sample['domain_key'],
            'file_idx': sample['file_idx'],
            'window_idx': sample['window_idx']
        }


def create_cached_domain_dataloaders(data_dir: str,
                                   domain_order: List[int],
                                   dataset_type: str = 'uos',
                                   batch_size: int = 32,
                                   num_workers: int = 0,  # 캐시 사용 시 멀티프로세싱 불필요
                                   use_collate_fn: bool = True) -> Dict[int, Dict[str, torch.utils.data.DataLoader]]:
    """
    캐시된 데이터를 사용하는 고속 DataLoader 생성
    
    Args:
        data_dir: 데이터 디렉토리
        domain_order: 도메인 순서
        dataset_type: 'uos' 또는 'cwru'
        batch_size: 배치 크기
        num_workers: 워커 수 (캐시 사용 시 0 권장)
        use_collate_fn: collate_fn 사용 여부
        
    Returns:
        도메인별 DataLoader 딕셔너리
    """
    from .data_loader import create_collate_fn
    
    domain_dataloaders = {}
    collate_fn = create_collate_fn() if use_collate_fn else None
    
    logger.info(f"🚀 캐시된 DataLoader 생성 시작: {dataset_type} ({len(domain_order)}개 도메인)")
    
    for domain_value in domain_order:
        domain_name = f"{domain_value}RPM" if dataset_type == 'uos' else f"{domain_value}HP"
        
        # 각 subset별 캐시된 Dataset 생성
        train_dataset = CachedBearingDataset(
            data_dir=data_dir,
            dataset_type=dataset_type,
            domain_value=domain_value,
            subset='train'
        )
        
        val_dataset = CachedBearingDataset(
            data_dir=data_dir,
            dataset_type=dataset_type,
            domain_value=domain_value,
            subset='val'
        )
        
        test_dataset = CachedBearingDataset(
            data_dir=data_dir,
            dataset_type=dataset_type,
            domain_value=domain_value,
            subset='test'
        )
        
        # DataLoader 생성 (num_workers=0으로 고속화)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        test_loader = torch.utils.data.DataLoader(
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
    
    # 캐시 통계 출력
    cache = get_global_cache()
    stats = cache.get_cache_stats()
    logger.info(f"📊 캐시 통계: 적중률 {stats['hit_rate']:.1f}%, "
               f"캐시 파일 {stats['cached_files']}개, "
               f"크기 {stats['total_size_mb']:.1f}MB")
    
    return domain_dataloaders


def create_cached_first_domain_dataloader(data_dir: str,
                                        domain_order: List[int],
                                        dataset_type: str = 'uos',
                                        subset: str = 'train',
                                        batch_size: int = 32,
                                        num_workers: int = 0,
                                        use_collate_fn: bool = True) -> torch.utils.data.DataLoader:
    """
    캐시된 첫 번째 도메인 DataLoader 생성
    
    Args:
        data_dir: 데이터 디렉토리
        domain_order: 도메인 순서 (첫 번째만 사용)
        dataset_type: 'uos' 또는 'cwru'
        subset: 'train', 'val', 'test'
        batch_size: 배치 크기
        num_workers: 워커 수
        use_collate_fn: collate_fn 사용 여부
        
    Returns:
        첫 번째 도메인 DataLoader
    """
    from .data_loader import create_collate_fn
    
    first_domain_value = domain_order[0]
    
    # 캐시된 Dataset 생성
    dataset = CachedBearingDataset(
        data_dir=data_dir,
        dataset_type=dataset_type,
        domain_value=first_domain_value,
        subset=subset
    )
    
    # collate_fn 준비
    collate_fn = create_collate_fn() if use_collate_fn else None
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(subset == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    domain_name = f"{first_domain_value}RPM" if dataset_type == 'uos' else f"{first_domain_value}HP"
    logger.info(f"First Domain DataLoader 생성 (캐시): Domain {domain_name}, "
               f"{subset} subset, {len(dataset)}개 샘플")
    
    return dataloader


def clear_all_caches():
    """모든 캐시 삭제 (디버깅용)"""
    cache = get_global_cache()
    cache.clear_cache()
    logger.info("🗑️ 모든 캐시가 삭제되었습니다")


if __name__ == "__main__":
    # 캐시 시스템 테스트
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 캐시 시스템 테스트")
    
    # 첫 번째 로드 (캐시 미스)
    start_time = time.time()
    data1 = preprocess_and_cache_dataset(
        data_dir='data_scenario1',
        dataset_type='uos',
        domain_value=600,
        subset='train'
    )
    first_load_time = time.time() - start_time
    print(f"첫 번째 로드: {first_load_time:.2f}초")
    
    # 두 번째 로드 (캐시 적중)
    start_time = time.time()
    data2 = preprocess_and_cache_dataset(
        data_dir='data_scenario1',
        dataset_type='uos',
        domain_value=600,
        subset='train'
    )
    second_load_time = time.time() - start_time
    print(f"두 번째 로드: {second_load_time:.2f}초")
    
    # 속도 향상 계산
    speedup = first_load_time / max(0.001, second_load_time)
    print(f"🚀 속도 향상: {speedup:.1f}배")
    
    # 캐시 통계
    cache = get_global_cache()
    stats = cache.get_cache_stats()
    print(f"📊 캐시 통계: {stats}")
