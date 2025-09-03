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

def parse_filename(filename: str, dataset_type: str = 'uos') -> Dict[str, str]:
    """
    데이터셋 파일명에서 메타데이터 추출
    
    UOS 파일명 형식: {회전체상태}_{베어링상태}_{베어링타입}_{회전속도}.mat
    예시: H_B_6204_600.mat
    
    CWRU 파일명 형식: {베어링상태}_{부하}.mat
    예시: B_0hp_1.mat, Normal_2hp.mat
    
    Args:
        filename (str): .mat 파일명
        dataset_type (str): 'uos' 또는 'cwru'
        
    Returns:
        Dict[str, str]: 추출된 메타데이터
    """
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    if dataset_type.lower() == 'uos':
        return _parse_uos_filename(name_without_ext)
    elif dataset_type.lower() == 'cwru':
        return _parse_cwru_filename(name_without_ext)
    else:
        raise ValueError(f"지원하지 않는 데이터셋 타입: {dataset_type}")


def _parse_uos_filename(name_without_ext: str) -> Dict[str, str]:
    """UOS 파일명 파싱"""
    parts = name_without_ext.split('_')
    
    if len(parts) != 4:
        raise ValueError(f"예상된 UOS 파일명 형식이 아님: {name_without_ext}")
    
    return {
        'dataset_type': 'uos',
        'rotating_component': parts[0],  # H, L, U, M
        'bearing_condition': parts[1],   # H, IR, OR, B  
        'bearing_type': parts[2],        # 6204, 30204, N204, NJ204
        'rotating_speed': int(parts[3])  # 600, 800, 1000, 1200, 1400, 1600
    }


def _parse_cwru_filename(name_without_ext: str) -> Dict[str, str]:
    """CWRU 파일명 파싱"""
    parts = name_without_ext.split('_')
    
    if len(parts) < 2:
        raise ValueError(f"예상된 CWRU 파일명 형식이 아님: {name_without_ext}")
    
    bearing_condition = parts[0]  # Normal, B, IR, OR
    load_part = parts[1]  # 0hp, 1hp, 2hp, 3hp
    
    # Load에서 숫자 추출
    if 'hp' in load_part:
        load = int(load_part.replace('hp', ''))
    else:
        load = 0
    
    return {
        'dataset_type': 'cwru',
        'bearing_condition': bearing_condition,  # Normal, B, IR, OR
        'load': load,  # 0, 1, 2, 3 (horsepower)
        'rotating_component': 'H',  # CWRU는 회전체 상태가 항상 정상
        'bearing_type': 'deep_groove_ball'  # CWRU는 Deep Groove Ball Bearing 사용
    }


def generate_text_description(metadata: Dict[str, str]) -> str:
    """
    메타데이터에서 자연어 텍스트 설명 생성
    
    Args:
        metadata (Dict[str, str]): parse_filename 출력
        
    Returns:
        str: 생성된 텍스트 설명
    """
    dataset_type = metadata.get('dataset_type', 'uos')
    
    if dataset_type == 'uos':
        return _generate_uos_text_description(metadata)
    elif dataset_type == 'cwru':
        return _generate_cwru_text_description(metadata)
    else:
        raise ValueError(f"지원하지 않는 데이터셋 타입: {dataset_type}")


def _generate_uos_text_description(metadata: Dict[str, str]) -> str:
    """UOS 데이터셋용 다양한 텍스트 설명 생성 (CRITICAL FIX: 텍스트 다양성 대폭 개선)"""
    import random
    
    # 회전체 상태 매핑 (다양한 표현)
    rotating_component_variations = {
        'H': ['healthy rotating machinery', 'normal rotating component', 'well-balanced rotating system', 'properly aligned rotating shaft'],
        'L': ['loose rotating component', 'looseness in the rotating machinery', 'mechanical looseness in shaft', 'loose coupling in rotating system'], 
        'U': ['unbalanced rotating component', 'mass imbalance in rotor', 'unbalanced rotating machinery', 'rotor imbalance condition'],
        'M': ['misaligned rotating component', 'shaft misalignment', 'angular misalignment in rotating system', 'parallel misalignment condition']
    }
    
    # 베어링 상태 매핑 (다양한 표현)
    bearing_condition_variations = {
        'H': ['healthy bearing condition', 'normal bearing', 'fault-free bearing', 'bearing in good condition', 'undamaged bearing'],
        'B': ['ball defect', 'rolling element fault', 'ball bearing damage', 'defective ball element', 'ball surface fault'],
        'IR': ['inner race defect', 'inner ring fault', 'inner raceway damage', 'inner race surface defect', 'inner ring wear'], 
        'OR': ['outer race defect', 'outer ring fault', 'outer raceway damage', 'outer race surface defect', 'outer ring wear']
    }
    
    # 베어링 타입 매핑 (더 상세한 설명)
    bearing_type_variations = {
        '6204': ['deep groove ball bearing model 6204', 'single-row deep groove ball bearing', '6204 series ball bearing', 'radial ball bearing 6204'],
        '30204': ['tapered roller bearing model 30204', 'single-row tapered roller bearing', '30204 series tapered bearing', 'conical roller bearing 30204'],
        'N204': ['cylindrical roller bearing model N204', 'single-row cylindrical roller bearing', 'N204 series roller bearing', 'radial roller bearing N204'],
        'NJ204': ['cylindrical roller bearing model NJ204', 'NJ204 series roller bearing with flanges', 'flanged cylindrical roller bearing', 'NJ-type roller bearing']
    }
    
    # 속도 관련 다양한 표현
    speed_variations = [
        f"operating at {metadata['rotating_speed']} RPM",
        f"running at {metadata['rotating_speed']} revolutions per minute",
        f"rotating at {metadata['rotating_speed']} rpm speed",
        f"with rotational speed of {metadata['rotating_speed']} rpm"
    ]
    
    # 문장 구조 템플릿들
    templates = [
        "A {bearing_type} {speed} with {rotating_desc} and {bearing_desc}.",
        "Industrial bearing system: {bearing_type} {speed}, showing {bearing_desc} and {rotating_desc}.",
        "Rotating machinery with {bearing_type} {speed}, characterized by {bearing_desc} and {rotating_desc}.",
        "Mechanical system featuring {bearing_type} {speed}, exhibiting {rotating_desc} with {bearing_desc}.",
        "Bearing fault diagnosis case: {bearing_type} {speed}, presenting {bearing_desc} in combination with {rotating_desc}."
    ]
    
    # 랜덤 선택으로 다양성 확보
    rotating_desc = random.choice(rotating_component_variations.get(
        metadata['rotating_component'], 
        [f"unknown rotating component ({metadata['rotating_component']})"]
    ))
    
    bearing_desc = random.choice(bearing_condition_variations.get(
        metadata['bearing_condition'],
        [f"unknown bearing condition ({metadata['bearing_condition']})"]
    ))
    
    bearing_type = random.choice(bearing_type_variations.get(
        metadata['bearing_type'],
        [f"unknown bearing type ({metadata['bearing_type']})"]
    ))
    
    speed = random.choice(speed_variations)
    template = random.choice(templates)
    
    # 문장 생성
    text = template.format(
        bearing_type=bearing_type,
        speed=speed,
        rotating_desc=rotating_desc,
        bearing_desc=bearing_desc
    )
    
    return text


def _generate_cwru_text_description(metadata: Dict[str, str]) -> str:
    """CWRU 데이터셋용 다양한 텍스트 설명 생성 (CRITICAL FIX: 텍스트 다양성 대폭 개선)"""
    import random
    
    # 베어링 상태 매핑 (다양한 표현)
    bearing_condition_variations = {
        'Normal': ['healthy bearing condition', 'normal bearing operation', 'fault-free bearing', 'bearing in perfect condition', 'undamaged bearing state'],
        'B': ['ball defect', 'rolling element fault', 'ball bearing damage', 'defective ball element', 'ball surface deterioration'],
        'IR': ['inner race defect', 'inner ring fault', 'inner raceway damage', 'inner race surface wear', 'inner ring deterioration'], 
        'OR': ['outer race defect', 'outer ring fault', 'outer raceway damage', 'outer race surface wear', 'outer ring deterioration']
    }
    
    # 부하 관련 다양한 표현
    load = metadata.get('load', 0)
    load_variations = [
        f"operating under {load} horsepower load",
        f"subjected to {load} HP mechanical load",
        f"running with {load} horsepower loading condition",
        f"under {load} HP operational load",
        f"experiencing {load} horsepower load stress"
    ]
    
    # 베어링 타입 다양한 표현
    bearing_type_variations = [
        'deep groove ball bearing',
        'single-row deep groove ball bearing',
        'radial ball bearing',
        'deep groove radial ball bearing',
        'standard deep groove bearing'
    ]
    
    # 문장 구조 템플릿들
    templates = [
        "A {bearing_type} {load} with {bearing_desc}.",
        "Industrial bearing system: {bearing_type} {load}, showing {bearing_desc}.",
        "Mechanical test setup with {bearing_type} {load}, characterized by {bearing_desc}.",
        "Laboratory bearing specimen: {bearing_type} {load}, exhibiting {bearing_desc}.",
        "Bearing fault diagnosis case: {bearing_type} {load}, presenting {bearing_desc}.",
        "Motor drive end bearing: {bearing_type} {load}, demonstrating {bearing_desc}."
    ]
    
    # 랜덤 선택으로 다양성 확보
    bearing_desc = random.choice(bearing_condition_variations.get(
        metadata['bearing_condition'],
        [f"unknown bearing condition ({metadata['bearing_condition']})"]
    ))
    
    bearing_type = random.choice(bearing_type_variations)
    load_desc = random.choice(load_variations)
    template = random.choice(templates)
    
    # 문장 생성
    text = template.format(
        bearing_type=bearing_type,
        load=load_desc,
        bearing_desc=bearing_desc
    )
    
    return text


def load_mat_file(filepath: str, signal_key: str = None, dataset_type: str = None) -> np.ndarray:
    """
    .mat 파일에서 진동 신호 로딩
    
    Args:
        filepath (str): .mat 파일 경로
        signal_key (str, optional): 신호 키 이름 (자동 탐지 시 None)
        dataset_type (str, optional): 데이터셋 타입 ('uos', 'cwru')
        
    Returns:
        np.ndarray: 진동 신호 (1D array)
    """
    try:
        mat_data = sio.loadmat(filepath)
        
        # __header__, __version__, __globals__ 제외하고 실제 데이터 키 찾기
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        
        if signal_key is not None:
            # 명시적으로 지정된 키 사용
            if signal_key not in mat_data:
                raise KeyError(f"키 '{signal_key}'를 찾을 수 없습니다: {filepath}")
            signal = mat_data[signal_key]
        else:
            # 데이터셋 타입에 따른 자동 선택
            if dataset_type == 'cwru':
                # CWRU: Drive End 채널만 사용
                de_key = None
                for key in data_keys:
                    if '_DE_time' in key:  # Drive End 채널 찾기
                        de_key = key
                        break
                
                if de_key is None:
                    raise ValueError(f"CWRU Drive End 채널을 찾을 수 없습니다: {filepath}, 키들: {data_keys}")
                
                signal = mat_data[de_key]
                
            elif dataset_type == 'uos':
                # UOS: 'Data' 키 우선 선택
                if 'Data' in data_keys:
                    signal = mat_data['Data']
                elif len(data_keys) == 1:
                    signal = mat_data[data_keys[0]]
                else:
                    # 가장 큰 배열을 신호로 가정
                    signal = max([mat_data[k] for k in data_keys], key=lambda x: x.size)
            else:
                # 기본 동작 (하위 호환성)
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
