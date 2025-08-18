"""
유틸리티 함수들
파일명 파싱, 텍스트 생성, 데이터 전처리 등
"""

import os
import re
import numpy as np
import scipy.io as sio
from typing import Dict, Tuple, Optional
import torch

def parse_filename(filename: str) -> Dict[str, str]:
    """
    UOS 데이터셋 파일명에서 메타데이터 추출
    
    파일명 형식: {회전체상태}_{베어링상태}_{베어링타입}_{회전속도}.mat
    예시: H_B_6204_600.mat
    
    Args:
        filename (str): .mat 파일명
        
    Returns:
        Dict[str, str]: 추출된 메타데이터
    """
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    parts = name_without_ext.split('_')
    
    if len(parts) != 4:
        raise ValueError(f"예상된 파일명 형식이 아님: {basename}")
    
    return {
        'rotating_component': parts[0],  # H, L, U, M
        'bearing_condition': parts[1],   # H, IR, OR, B  
        'bearing_type': parts[2],        # 6204, 30204, N204, NJ204
        'rotating_speed': int(parts[3])  # 600, 800, 1000, 1200, 1400, 1600
    }


def generate_text_description(metadata: Dict[str, str]) -> str:
    """
    메타데이터에서 자연어 텍스트 설명 생성
    
    Args:
        metadata (Dict[str, str]): parse_filename 출력
        
    Returns:
        str: 생성된 텍스트 설명
    """
    # 회전체 상태 매핑
    rotating_component_map = {
        'H': 'healthy rotating component',
        'L': 'looseness in rotating component', 
        'U': 'unbalanced rotating component',
        'M': 'misaligned rotating component'
    }
    
    # 베어링 상태 매핑
    bearing_condition_map = {
        'H': 'healthy bearing',
        'B': 'ball fault',
        'IR': 'inner race fault', 
        'OR': 'outer race fault'
    }
    
    # 베어링 타입 매핑
    bearing_type_map = {
        '6204': 'deep groove ball bearing',
        '30204': 'tapered roller bearing',
        'N204': 'cylindrical roller bearing',
        'NJ204': 'cylindrical roller bearing'
    }
    
    rotating_desc = rotating_component_map.get(
        metadata['rotating_component'], 
        f"unknown rotating component ({metadata['rotating_component']})"
    )
    
    bearing_desc = bearing_condition_map.get(
        metadata['bearing_condition'],
        f"unknown bearing condition ({metadata['bearing_condition']})" 
    )
    
    bearing_type_desc = bearing_type_map.get(
        metadata['bearing_type'],
        f"unknown bearing type ({metadata['bearing_type']})"
    )
    
    # 자연어 문장 생성
    text = (f"A {bearing_type_desc} operating at {metadata['rotating_speed']} rpm "
            f"with {rotating_desc} and {bearing_desc}.")
    
    return text


def load_mat_file(filepath: str, signal_key: str = None) -> np.ndarray:
    """
    .mat 파일에서 진동 신호 로딩
    
    Args:
        filepath (str): .mat 파일 경로
        signal_key (str, optional): 신호 키 이름 (자동 탐지 시 None)
        
    Returns:
        np.ndarray: 진동 신호 (1D array)
    """
    try:
        mat_data = sio.loadmat(filepath)
        
        # __header__, __version__, __globals__ 제외하고 실제 데이터 키 찾기
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        
        if signal_key is not None:
            if signal_key not in mat_data:
                raise KeyError(f"키 '{signal_key}'를 찾을 수 없습니다: {filepath}")
            signal = mat_data[signal_key]
        else:
            if len(data_keys) == 0:
                raise ValueError(f"데이터 키를 찾을 수 없습니다: {filepath}")
            elif len(data_keys) == 1:
                signal = mat_data[data_keys[0]]
            else:
                # 가장 큰 배열을 신호로 가정
                signal = max([mat_data[k] for k in data_keys], key=lambda x: x.size)
        
        # 1D 배열로 변환
        signal = np.squeeze(signal)
        if signal.ndim != 1:
            raise ValueError(f"1D 신호가 아닙니다. Shape: {signal.shape}")
            
        return signal.astype(np.float32)
        
    except Exception as e:
        raise RuntimeError(f"파일 로딩 실패 {filepath}: {str(e)}")


def create_windowed_signal(signal: np.ndarray, 
                         window_size: int = 4096, 
                         overlap_ratio: float = 0.5) -> np.ndarray:
    """
    긴 신호를 고정 길이 윈도우로 분할
    
    Args:
        signal (np.ndarray): 원본 신호
        window_size (int): 윈도우 크기
        overlap_ratio (float): 겹침 비율 (0~1)
        
    Returns:
        np.ndarray: 윈도우된 신호들 (num_windows, window_size)
    """
    if len(signal) < window_size:
        # 신호가 윈도우보다 짧으면 zero-padding
        padded = np.zeros(window_size, dtype=signal.dtype)
        padded[:len(signal)] = signal
        return padded.reshape(1, -1)
    
    step_size = int(window_size * (1 - overlap_ratio))
    num_windows = (len(signal) - window_size) // step_size + 1
    
    windows = []
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        windows.append(signal[start:end])
    
    return np.array(windows)


def normalize_signal(signal: np.ndarray, method: str = 'standardize') -> np.ndarray:
    """
    신호 정규화
    
    Args:
        signal (np.ndarray): 입력 신호
        method (str): 정규화 방법 ('standardize', 'minmax', 'none')
        
    Returns:
        np.ndarray: 정규화된 신호
    """
    if method == 'standardize':
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    elif method == 'minmax':
        min_val, max_val = np.min(signal), np.max(signal)
        return (signal - min_val) / (max_val - min_val + 1e-8)
    elif method == 'none':
        return signal
    else:
        raise ValueError(f"알 수 없는 정규화 방법: {method}")


def create_labels(metadata_list: list, 
                 label_type: str = 'multi') -> Tuple[np.ndarray, Dict]:
    """
    메타데이터에서 라벨 생성
    
    Args:
        metadata_list (list): 메타데이터 딕셔너리들의 리스트
        label_type (str): 라벨 종류
            - 'multi': 3가지 라벨 (rotating_component, bearing_condition, bearing_type)
            - 'bearing_only': 베어링 상태만
            - 'component_only': 회전체 상태만
            
    Returns:
        Tuple[np.ndarray, Dict]: (라벨 배열, 라벨 인코더 딕셔너리)
    """
    if label_type == 'multi':
        # 3가지 라벨 생성
        rotating_components = [m['rotating_component'] for m in metadata_list]
        bearing_conditions = [m['bearing_condition'] for m in metadata_list] 
        bearing_types = [m['bearing_type'] for m in metadata_list]
        
        from sklearn.preprocessing import LabelEncoder
        
        rc_encoder = LabelEncoder()
        bc_encoder = LabelEncoder()
        bt_encoder = LabelEncoder()
        
        rc_labels = rc_encoder.fit_transform(rotating_components)
        bc_labels = bc_encoder.fit_transform(bearing_conditions)
        bt_labels = bt_encoder.fit_transform(bearing_types)
        
        # 3개 라벨을 하나의 배열로 합침
        labels = np.column_stack([rc_labels, bc_labels, bt_labels])
        
        label_encoders = {
            'rotating_component': rc_encoder,
            'bearing_condition': bc_encoder, 
            'bearing_type': bt_encoder
        }
        
        return labels, label_encoders
        
    else:
        raise NotImplementedError(f"라벨 타입 '{label_type}'은 아직 구현되지 않았습니다.")


def get_device() -> torch.device:
    """적절한 디바이스 반환"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def set_seed(seed: int = 42):
    """재현성을 위한 전역 시드 설정"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations (속도 저하 가능성)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✅ 전역 시드 설정 완료: {seed}")


def setup_amp_and_scaler(device: torch.device, use_amp: bool = True):
    """AMP와 GradScaler 설정"""
    if use_amp and device.type == 'cuda':
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print("✅ AMP (Automatic Mixed Precision) 활성화")
        return scaler, True
    else:
        print("ℹ️  AMP 비활성화 (CPU 또는 사용자 설정)")
        return None, False
