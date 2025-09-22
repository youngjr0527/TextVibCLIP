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
    
    # Prototypes (class anchors) for shared semantic space
    'prototypes': {
        'enabled': True,           # turn on prototype alignment loss
        'tau': 0.12,               # temperature for prototype logits (stabilize CE)
        'lambda_proto': 0.25,       # loss weight for prototype CE
        'ema_momentum': 0.995,      # EMA momentum for prototype updates
        'init_from_text': True,     # initialize prototypes from text class means
        'continual_lambda': 0.1,    # lower prototype weight for continual phase
        # Continual 단계 프로토타입 업데이트 모드: 'both' | 'text_only' | 'frozen'
        'update_mode_continual': 'text_only'
    },
    
    # Vibration Encoder (1D-CNN) - 2048 최적화 아키텍처
    'vibration_encoder': {
        'input_length': 2048,  # 4096 → 2048 (베어링 회전 주기 최적화)
        'architecture': '1D-CNN',  # 아키텍처 타입
        'kernel_sizes': [16, 32, 64, 32],  # 4-layer 구조 (2048에 최적화)
        'channels': [64, 128, 256, 512],   # 자연스러운 채널 증가 (64→512)
        'stride': 2,  # 모든 Conv1d의 stride
        'dropout': 0.1,  # 0.15 → 0.1 (적절한 정규화)
        'activation': 'relu',  # Activation function
        'normalization': 'batch_norm',  # Normalization type
        'pooling': 'adaptive_avg',  # Global pooling type
        # 2048 입력에 최적화된 아키텍처: 자연스러운 차원 축소
        # 2048 → 1024 → 512 → 256 → 128 → Global Pool → 256 embedding
    },
    
    # InfoNCE Loss - OPTIMIZED: Cross-Modal Projection 기반 최적화
    'infonce': {
        # First Domain Training (Domain 1) - 높은 온도로 안정화
        'first_domain_temperature_text': 0.2,   # 0.05 → 0.2 (Cross-Modal 최적화)
        'first_domain_temperature_vib': 0.2,    # 0.05 → 0.2 (Cross-Modal 최적화)
        
        # Continual Learning (Domain 2+) - 비대칭 설정
        'continual_temperature_text': 0.07,  # 0.05 → 0.07 (text 안정성)
        'continual_temperature_vib': 0.04,   # 0.03 → 0.04 (약 +30%)

        # First domain 온도 스케줄(선형): init → final
        'first_domain_temperature_text_init': 0.07,  # 0.05 → 0.07
        'first_domain_temperature_text_final': 0.05, # 0.01 → 0.05
        'first_domain_temperature_vib_init': 0.07,   # 0.05 → 0.07
        'first_domain_temperature_vib_final': 0.05,  # 0.01 → 0.05
    },
    
    # Projection layers
    'projection': {
        'hidden_dim': 512,  # 1024 → 512: 임베딩 차원 축소에 맞춤
        'output_dim': 256,  # 512 → 256: 최종 임베딩 차원
        'dropout': 0.1
    },
    # Residual projection stack (optional)
    'residual_projection': {
        'enabled': True,
        'num_layers': 3,
        'ffn_mult': 4,
        'dropout': 0.1
    },

    # Similarity head options
    'similarity': {
        'bilinear_enabled': True,
        'lambda_bilinear': 0.15
    },

    # 도메인별 동적 하이퍼파라미터 오버라이드(continual 구간)
    'domain_overrides': {
        1000: {
            'continual_temperature_vib': 0.045,
            'continual_lambda_proto': 0.08
        },
        1200: {
            'continual_temperature_vib': 0.05,
            'continual_lambda_proto': 0.05
        }
    },

    # Regularizers for continual learning
    'regularizers': {
        'rkd_enabled': True,
        'lambda_rkd': 0.2,
        'lwf_enabled': False,
        'lambda_lwf': 0.0
    },
    # Auxiliary classification for first domain bootstrapping
    'aux_classification': {
        'enabled': True,   # 🎯 CRITICAL FIX: Auxiliary loss 활성화 (supervised signal 강화)
        'num_classes': 7,  # UOS 7-클래스 지원 (H/B/IR/OR/L/U/M)
        'loss_weight': 2.0,  # 5.0 → 2.0 (overfitting 방지)
        'dropout': 0.2     # 0.05 → 0.2 (정규화 강화)
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
    'replay_ratio': 0.6,  # 새 데이터 vs replay 데이터 비율
    'replay_every_n': 1,  # 몇 배치마다 replay를 섞을지 (작은 에포크에서는 1 권장)
    'replay_selection': 'balanced',  # replay 샘플 선택 전략: 'random' | 'balanced' | 'representative'
    
    # Early stopping
    'patience': 12,
    'min_epoch_per_domain': 5,
    'min_delta': 1e-4,
    
    # 체크포인트
    'save_interval': 5,
    'checkpoint_dir': 'checkpoints',
    # Gradient accumulation to reduce memory footprint
    'grad_accum_steps': 1,

    # 파라미터 그룹 LR 멀티플라이어 (텍스트 LoRA/프로젝션 가속)
    'lora_lr_mult': 3.0,
    'proj_lr_mult': 5.0,  # 3.0 → 5.0 (continual learning에서 projection 학습 강화)
    'vib_lr_mult': 2.0,   # 1.0 → 2.0 (vibration encoder 학습 강화)

    # First-domain two-stage schedule
    'first_domain_stage1_epochs': 8,  # Stage-1: encoders freeze, projection/prototypes only
    # 도메인별 리플레이 부스팅(중간 도메인 안정화)
    'replay_boost_domains': [1000, 1200],
    'replay_boost_ratio': 0.7
}

# 데이터 설정
DATA_CONFIG = {
    # 기본 설정 (UOS)
    'data_dir': 'data_scenario1',
    'dataset_type': 'uos',
    'domain_order': [600, 800, 1000, 1200, 1400, 1600],  # UOS RPM 순서
    'validation_split': 0.2,
    'test_split': 0.2,
    
    # 데이터 전처리 (베어링 회전 주기 최적화)
    'signal_normalization': 'standardize',  # 'standardize', 'minmax', 'none'
    'window_size': 2048,  # 4096 → 2048 (1-5 회전 포함, 효율적 결함 감지)
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
    
    # 데이터 전처리 (UOS와 통일)
    'signal_normalization': 'standardize',
    'window_size': 2048,  # UOS와 동일 (통일된 아키텍처)
    'overlap_ratio': 0.5,  # UOS와 동일 (일관성)
    
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
    'max_full_eval_batches': -1,  # 🎯 FIXED: 전체 평가 (제한 없음)
    # 빠른 평가 배치 제한 (FAST 경로)
    'max_fast_eval_batches': 10
}

# 디바이스 설정
DEVICE_CONFIG = {
    'use_cuda': True,
    'gpu_id': 2,
    'mixed_precision': False,  # 🎯 AMP 비활성화 (수치 안정성)
}
