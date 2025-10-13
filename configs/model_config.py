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
    
    # Ranking Loss - Triplet margin 설정
    'ranking_loss': {
        # Triplet Loss margin (같은 클래스 vs 다른 클래스 분리)
        'margin': 0.3,  # 고정된 마진으로 안정적 학습
        'loss_type': 'triplet',  # 'triplet' 또는 'margin_ranking'
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

    # Auxiliary classification for first domain bootstrapping
    'aux_classification': {
        'enabled': True,   # 🎯 CRITICAL FIX: Auxiliary loss 활성화 (supervised signal 강화)
        'num_classes': 7,  # UOS 7-클래스 지원 (H/B/IR/OR/L/U/M)
        'loss_weight': 1.0,  # 5.0 → 1.0 (과적합 방지, Ranking loss와 균형)
        'dropout': 0.2     # 0.1 → 0.2 (과적합 방지 강화)
    }
}

    # 첫 번째 도메인 전용 학습 설정 (Foundation Learning)
FIRST_DOMAIN_CONFIG = {
    # 🎯 Foundation Learning: Ranking Loss와 Auxiliary Head 균형
    'num_epochs': 15,           # 충분한 기초 학습
    'learning_rate': 2e-4,      # 1e-4 → 2e-4 (적당한 학습률)
    'weight_decay': 5e-5,       # 1e-4 → 5e-5 (정규화 완화)
    'aux_weight': 2.0,          # 10.0 → 2.0 (Ranking loss와 균형)
    'patience': 8,              # 5 → 8 (충분한 학습 기회)
    'min_epoch': 5,             # 3 → 5 (최소 기초 학습 보장)
    
    # 파라미터 그룹 LR 멀티플라이어 (적극적 학습)
    'lora_lr_mult': 3.0,
    'proj_lr_mult': 5.0,        # 높은 projection 학습률
    'vib_lr_mult': 2.0,         # 높은 vibration 학습률
    
    # 스케줄러 설정
    'scheduler_type': 'cosine', # Cosine annealing (안정적 감소)
    'eta_min': 1e-6,
    
    # Two-stage 학습
    'stage1_epochs': 8,         # Projection/prototypes 먼저 학습
}

# CWRU 첫 번째 도메인 전용 설정 (극소 데이터 대응)
CWRU_FIRST_DOMAIN_CONFIG = {
    # 🎯 CWRU Foundation: 극도로 보수적 학습
    'num_epochs': 20,           # 3 → 20 (기초 학습 강화)
    'learning_rate': 5e-5,      # 1e-4 → 5e-5 (매우 낮은 학습률)
    'weight_decay': 5e-3,       # 1e-3 → 5e-3 (매우 강한 정규화)
    'aux_weight': 0.1,          # 0.5 → 0.1 (ranking 거의 전용)
    'patience': 1,              # 3 → 1 (즉시 조기 종료)
    'min_epoch': 1,             # 2 → 1 (최소 1 에포크)
    
    # 매우 보존적 학습
    'lora_lr_mult': 0.5,
    'proj_lr_mult': 1.0,
    'vib_lr_mult': 1.0,
    
    'scheduler_type': 'constant',  # 스케줄러 없음
    'stage1_epochs': 1
}

# Continual Learning 전용 설정 (Adaptation Learning) 
CONTINUAL_CONFIG = {
    # 🎯 Adaptation Learning: Auxiliary Head 중심 빠른 적응
    'num_epochs': 4,            # 6 → 4 (적정 적응)
    'learning_rate': 5e-5,      # 1e-4 → 5e-5 (더 보존적)
    'weight_decay': 2e-4,       # 1e-4 → 2e-4 (과적합 방지 강화)
    'aux_weight': 5.0,          # 0.5 → 5.0 (Auxiliary Head 중심)
    'patience': 2,              # 3 → 2 (더 엄격한 조기 종료)
    'min_epoch': 2,             # 최소 적응 학습 유지
    
    # 파라미터 그룹 LR 멀티플라이어 (보존적 학습)
    'lora_lr_mult': 1.0,        # 텍스트 안정화
    'proj_lr_mult': 2.0,        # 적당한 projection 적응
    'vib_lr_mult': 3.0,         # 진동 위주 적응
    
    # 스케줄러 설정
    'scheduler_type': 'step',   # Step LR (안정적)
    'step_size': 3,
    'gamma': 0.8,
    
    # Replay 설정
    'replay_buffer_size': 500,
    'replay_ratio': 0.6,
    'replay_every_n': 1,
    'replay_selection': 'balanced',
    'replay_boost_domains': [1000, 1200],
    'replay_boost_ratio': 0.7
}

# CWRU 전용 설정 (극소 데이터 대응)
CWRU_SPECIFIC_CONFIG = {
    # 🎯 극소 데이터 대응: 과적합 방지 극대화
    'num_epochs': 2,            # 3 → 2 (극도로 짧은 학습)
    'learning_rate': 1e-6,      # 1e-5 → 1e-6 (극도로 낮은 학습률)
    'weight_decay': 1e-2,       # 1e-3 → 1e-2 (극도로 강한 정규화)
    'aux_weight': 0.05,         # 1.0 → 0.05 (ranking 거의 전용)
    'patience': 1,              # 즉시 조기 종료
    'min_epoch': 1,             # 최소 1 에포크
    
    # 극도로 보존적 학습
    'lora_lr_mult': 0.1,
    'proj_lr_mult': 0.5,
    'vib_lr_mult': 1.0,
    
    # 간단한 스케줄러
    'scheduler_type': 'constant',
    
    # 최소 Replay
    'replay_buffer_size': 20,   # 100 → 20 (극소 버퍼)
    'replay_ratio': 0.1,       # 0.3 → 0.1 (최소 replay)
    'replay_every_n': 5        # 3 → 5 (가끔만 replay)
}

# 기존 TRAINING_CONFIG (하위 호환성)
TRAINING_CONFIG = {
    # 기본 설정 (FIRST_DOMAIN_CONFIG 기반)
    'batch_size': 32,
    'num_epochs': 25,
    'learning_rate': 3e-4,
    'weight_decay': 1e-5,
    'warmup_steps': 1000,
    
    # Continual Learning 설정
    'replay_buffer_size': 500,
    'replay_ratio': 0.6,
    'replay_every_n': 1,
    'replay_selection': 'balanced',
    
    # Early stopping
    'patience': 8,
    'min_epoch_per_domain': 5,
    'min_delta': 1e-4,
    
    # 체크포인트
    'save_interval': 5,
    'checkpoint_dir': 'checkpoints',
    'grad_accum_steps': 1,

    # 파라미터 그룹 LR 멀티플라이어
    'lora_lr_mult': 3.0,
    'proj_lr_mult': 5.0,
    'vib_lr_mult': 2.0,

    # First-domain two-stage schedule
    'first_domain_stage1_epochs': 8,
    # 도메인별 리플레이 부스팅
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

# CWRU 데이터 설정 (데이터 증강 강화)
CWRU_DATA_CONFIG = {
    'data_dir': 'data_scenario2',
    'dataset_type': 'cwru',
    'domain_order': [0, 1, 2, 3],  # CWRU Load 순서 (0hp, 1hp, 2hp, 3hp)
    'validation_split': 0.2,
    'test_split': 0.2,
    
    # 데이터 전처리 (UOS와 통일 유지)
    'signal_normalization': 'standardize',
    'window_size': 2048,  # UOS와 동일 (아키텍처 일관성)
    'overlap_ratio': 0.6,  # 0.8 → 0.6 (중복 감소로 일반화 유도)
    
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

