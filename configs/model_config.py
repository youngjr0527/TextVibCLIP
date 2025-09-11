"""
TextVibCLIP 모델 설정 파일
모든 하이퍼파라미터와 모델 구성을 중앙 집중식으로 관리
"""

# 기본 모델 파라미터
MODEL_CONFIG = {
    # 임베딩 차원
    'embedding_dim': 256,  # 512 → 256로 축소
    'text_dim': 768,  # DistilBERT hidden size
    'vibration_input_dim': 1,  # UOS/CWRU: 단일 채널 진동 신호 (CWRU는 Drive End만)
    
    # Text Encoder (DistilBERT + LoRA)
    'text_encoder': {
        'model_name': 'distilbert-base-uncased',
        'freeze_base': True,  # Domain 2+에서 base model freeze
        'lora_config': {
            'r': 32,  # LoRA rank
            'lora_alpha': 64,
            'target_modules': ["q_lin", "v_lin"], 
            'lora_dropout': 0.1
        }
    },
    
    # Vibration Encoder (1D-CNN) - UPGRADED ARCHITECTURE
    'vibration_encoder': {
        'input_length': 4096,  # 진동 신호 길이 (windowing)
        'architecture': '1D-CNN',  # 아키텍처 타입
        'kernel_sizes': [16, 32, 64, 32, 16],  # 더 깊은 5-layer 구조
        'channels': [128, 256, 512, 1024, 512],   # 더 큰 채널 수 (64→128 시작, 최대 1024)
        'stride': 2,  # 모든 Conv1d의 stride
        'dropout': 0.15,  # 0.1 → 0.15: 큰 모델에 맞춰 드롭아웃 증가
        'activation': 'relu',  # Activation function
        'normalization': 'batch_norm',  # Normalization type
        'pooling': 'adaptive_avg',  # Global pooling type
        # 업그레이드된 아키텍처: 더 깊고 넓은 네트워크로 표현력 증가
        # 5-layer deep CNN으로 복잡한 진동 패턴 학습 능력 향상
    },
    
    # InfoNCE Loss - FIXED: 정상 온도 범위로 복원
    'infonce': {
        # First Domain Training (Domain 1) - CLIP 표준 온도
        'first_domain_temperature_text': 0.07,   # 표준 contrastive learning 온도
        'first_domain_temperature_vib': 0.07,    # 균등 학습
        
        # Continual Learning (Domain 2+) - 검증된 비대칭 설정  
        'continual_temperature_text': 0.10,  # text 안정성 (freeze되므로 높은 온도)
        'continual_temperature_vib': 0.05,   # vibration 적극 학습 (낮은 온도)

        # First domain 온도 스케줄(선형): init → final (없으면 고정 온도 사용)
        'first_domain_temperature_text_init': 0.10,
        'first_domain_temperature_text_final': 0.07,
        'first_domain_temperature_vib_init': 0.10,
        'first_domain_temperature_vib_final': 0.07,
    },
    
    # Projection layers
    'projection': {
        'hidden_dim': 512,  # 1024 → 512: 임베딩 차원 축소에 맞춤
        'output_dim': 256,  # 512 → 256: 최종 임베딩 차원
        'dropout': 0.1
    },
    # Auxiliary classification for first domain bootstrapping
    'aux_classification': {
        'enabled': False,  # 🎯 FIXED: Auxiliary loss 비활성화 (contrastive learning 집중)
        'num_classes': 7,  # UOS 7-클래스 지원 (H/B/IR/OR/L/U/M)
        'loss_weight': 0.5,  # 가중치 감소
        'dropout': 0.1
    }
}

# 학습 파라미터
TRAINING_CONFIG = {
    # 기본 학습 설정
    'batch_size': 32,  # 실용적 배치 크기 (31개 negative samples)
    'num_epochs': 100,  # 50 → 100: 더 충분한 학습
    'learning_rate': 3e-4,  # 표준 contrastive learning 학습률
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
    # Gradient accumulation to reduce memory footprint
    'grad_accum_steps': 1,

    # 파라미터 그룹 LR 멀티플라이어 (텍스트 LoRA/프로젝션 가속)
    'lora_lr_mult': 3.0,
    'proj_lr_mult': 3.0,
    'vib_lr_mult': 1.0,
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
    ],

    # 평가 배치 제한 (메모리 안전성 및 정확한 평가)
    'max_full_eval_batches': 20,  # 최대 20배치로 제한 (메모리 안전)
    # 빠른 평가 배치 제한 (FAST 경로)
    'max_fast_eval_batches': 5
}

# 디바이스 설정
DEVICE_CONFIG = {
    'use_cuda': True,
    'gpu_id': 2,
    'mixed_precision': False,  # 🎯 AMP 비활성화 (수치 안정성)
}
