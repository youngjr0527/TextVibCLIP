"""
TextVibCLIP 모델 설정 파일
모든 하이퍼파라미터와 모델 구성을 중앙 집중식으로 관리
"""

# 기본 모델 파라미터
MODEL_CONFIG = {
    # 임베딩 차원
    'embedding_dim': 512,
    'text_dim': 768,  # DistilBERT hidden size
    'vibration_input_dim': 1,  # UOS: 단일 채널 진동 신호
    
    # Text Encoder (DistilBERT + LoRA)
    'text_encoder': {
        'model_name': 'distilbert-base-uncased',
        'freeze_base': True,  # Domain 2+에서 base model freeze
        'lora_config': {
            'r': 32,  # LoRA rank
            'lora_alpha': 64,
            'target_modules': ["q_lin", "v_lin"],  # DistilBERT attention layers
            'lora_dropout': 0.1
        }
    },
    
    # Vibration Encoder (TST)
    'vibration_encoder': {
        'input_length': 4096,  # 진동 신호 길이 (windowing)
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'pos_encoding': 'sincos',
        'activation': 'gelu',
        'normalization_layer': 'LayerNorm',
        # 메모리 절감을 위한 토큰 다운샘플링(패칭)
        # 입력 길이는 유지(4096)하되, 내부 Transformer에 투입되는 토큰 수를
        # patch_size 배로 줄여 OOM 위험을 낮춤. 예: 4096 -> 512 (patch_size=8)
        'use_token_downsampling': True,
        'patch_size': 8
    },
    
    # InfoNCE Loss
    'infonce': {
        # Joint Training (Domain 1)
        'joint_temperature_text': 0.07,
        'joint_temperature_vib': 0.07,
        
        # Continual Learning (Domain 2+) - 최적화된 비대칭 설정  
        'continual_temperature_text': 0.12,  # 높음: text 변화 억제 (안정성 확보)
        'continual_temperature_vib': 0.04,   # 낮음: vib 적응 강화 (학습 촉진)
    },
    
    # Projection layers
    'projection': {
        'hidden_dim': 1024,
        'output_dim': 512,
        'dropout': 0.1
    }
}

# 학습 파라미터
TRAINING_CONFIG = {
    # 기본 학습 설정
    'batch_size': 8,  # 메모리 안전성을 위해 축소 (Quadro RTX 5000 16GB 고려)
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'warmup_steps': 1000,
    
    # Continual Learning 설정
    'replay_buffer_size': 500,  # 도메인당 저장할 embedding 수
    'replay_ratio': 0.3,  # 새 데이터 vs replay 데이터 비율
    
    # Early stopping
    'patience': 10,
    'min_delta': 1e-4,
    
    # 체크포인트
    'save_interval': 5,
    'checkpoint_dir': 'checkpoints',
}

# 데이터 설정
DATA_CONFIG = {
    # 기본 설정 (UOS)
    'data_dir': 'data_scenario1',
    'dataset_type': 'uos',
    'domain_order': [600, 800, 1000, 1200, 1400, 1600],  # UOS RPM 순서
    'validation_split': 0.2,
    'test_split': 0.2,
    
    # 데이터 전처리
    'signal_normalization': 'standardize',  # 'standardize', 'minmax', 'none'
    'window_size': 4096,
    'overlap_ratio': 0.5,
    
    # 텍스트 생성
    'max_text_length': 128,
}

# CWRU 데이터 설정
CWRU_DATA_CONFIG = {
    'data_dir': 'data_scenario2',
    'dataset_type': 'cwru',
    'domain_order': [0, 1, 2, 3],  # CWRU Load 순서 (0hp, 1hp, 2hp, 3hp)
    'validation_split': 0.2,
    'test_split': 0.2,
    
    # 데이터 전처리
    'signal_normalization': 'standardize',
    'window_size': 4096,
    'overlap_ratio': 0.5,
    
    # 텍스트 생성
    'max_text_length': 128,
}

# 평가 메트릭
EVAL_CONFIG = {
    'metrics': [
        'accuracy',
        'precision',
        'recall', 
        'f1_score',
        'confusion_matrix'
    ],
    
    # Continual Learning 메트릭
    'continual_metrics': [
        'average_accuracy',      # 모든 도메인 평균
        'forgetting_rate',       # 이전 도메인 성능 저하
        'forward_transfer',      # 새 도메인 학습 효율
        'backward_transfer'      # 이전 도메인 성능 향상
    ]
}

# 디바이스 설정
DEVICE_CONFIG = {
    'use_cuda': True,
    'gpu_id': 2,
    'mixed_precision': True,  # AMP 사용
}
