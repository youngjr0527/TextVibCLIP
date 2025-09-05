"""
TextVibCLIP 모델 설정 파일
모든 하이퍼파라미터와 모델 구성을 중앙 집중식으로 관리
"""

# 기본 모델 파라미터
MODEL_CONFIG = {
    # 임베딩 차원
    'embedding_dim': 512,
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
    
    # Vibration Encoder (1D-CNN) 
    'vibration_encoder': {
        'input_length': 4096,  # 진동 신호 길이 (windowing)
        'architecture': '1D-CNN',  # 아키텍처 타입
        'kernel_sizes': [16, 32, 64, 32],  # 다중 스케일 커널 크기
        'channels': [64, 128, 256, 512],   # 각 블록의 채널 수
        'stride': 2,  # 모든 Conv1d의 stride
        'dropout': 0.1,  # Dropout 비율
        'activation': 'relu',  # Activation function
        'normalization': 'batch_norm',  # Normalization type
        'pooling': 'adaptive_avg',  # Global pooling type
        # 메모리 효율성: O(n) 복잡도로 TST 대비 안정적 처리
        # 성능 우수성: 79.0% (베어링 75.2% + 회전체 82.8%)
        # 실용성: 일반적인 GPU에서도 원활한 작동
    },
    
    # InfoNCE Loss
    'infonce': {
        # First Domain Training (Domain 1) - CRITICAL FIX: 온도 파라미터 급격히 감소
        'first_domain_temperature_text': 0.01,   # 0.05 → 0.01로 급격히 감소 (collapse 현상 해결)
        'first_domain_temperature_vib': 0.01,    # 0.05 → 0.01로 급격히 감소 (sharp contrastive learning)
        
        # Continual Learning (Domain 2+) - 최적화된 비대칭 설정  
        'continual_temperature_text': 0.15,  # 0.12 → 0.15: text 안정성 더 강화
        'continual_temperature_vib': 0.05,   # 0.04 → 0.05: vib 학습을 조금 완화

        # First domain 온도 스케줄(선형): init → final (없으면 고정 온도 사용)
        'first_domain_temperature_text_init': 0.02,
        'first_domain_temperature_text_final': 0.01,
        'first_domain_temperature_vib_init': 0.02,
        'first_domain_temperature_vib_final': 0.01,
    },
    
    # Projection layers
    'projection': {
        'hidden_dim': 1024,
        'output_dim': 512,
        'dropout': 0.1
    },
    # Auxiliary classification for first domain bootstrapping
    'aux_classification': {
        'enabled': True,
        'num_classes': 4,
        'loss_weight': 2.0,  # CRITICAL FIX: Auxiliary loss 가중치 증가 (1.0→2.0)
        'dropout': 0.1
    }
}

# 학습 파라미터
TRAINING_CONFIG = {
    # 기본 학습 설정
    'batch_size': 64,  # CRITICAL FIX: collapse 방지를 위해 더 큰 배치 (32→64, 63개 negative samples)
    'num_epochs': 100,  # 50 → 100: 더 충분한 학습
    'learning_rate': 1e-4,  # CRITICAL FIX: collapse 방지를 위해 조정 (3e-4→1e-4, 더 안정적)
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

    # 평가 배치 제한 (전체 평가에서 하드 캡 제거; -1이면 무제한)
    'max_full_eval_batches': -1,
    # 빠른 평가 배치 제한 (FAST 경로)
    'max_fast_eval_batches': 5
}

# 디바이스 설정
DEVICE_CONFIG = {
    'use_cuda': True,
    'gpu_id': 2,
    'mixed_precision': True,  # AMP 사용
}
