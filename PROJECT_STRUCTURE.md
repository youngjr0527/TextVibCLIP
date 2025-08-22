# TextVibCLIP 프로젝트 구조 설계

## 🎯 연구 목표 재확인
- **Multimodal Continual Learning**: 진동 신호 + 텍스트 메타데이터
- **Domain Shift**: 회전 속도 변화 (600→800→1000→1200→1400→1600 RPM)
- **Text Encoder**: DistilBERT + LoRA (Domain 1에서만 학습, 이후 freeze)
- **Vibration Encoder**: TST + Full training + Replay
- **Loss**: Bidirectional InfoNCE with asymmetric temperature

## 📁 제안하는 모듈 구조

```
TextVibCLIP/
├── 📁 src/                          # 소스 코드 모듈들
│   ├── 📄 __init__.py               
│   ├── 📄 data_loader.py            # UOS data_scenario1 로더
│   ├── 📄 text_encoder.py           # DistilBERT + LoRA 구현
│   ├── 📄 vibration_encoder.py      # TST 기반 인코더
│   ├── 📄 textvib_model.py          # 메인 TextVibCLIP 모델
│   ├── 📄 continual_trainer.py      # Continual Learning 파이프라인
│   ├── 📄 replay_buffer.py          # Replay mechanism
│   └── 📄 utils.py                  # 유틸리티 함수들
├── 📁 configs/                      # 설정 파일들
│   ├── 📄 model_config.py           # 모델 하이퍼파라미터
│   └── 📄 experiment_config.py      # 실험 설정
├── 📁 experiments/                  # 실험 실행 스크립트들
│   ├── 📄 run_first_domain_training.py     # Domain 1 First Domain Training
│   ├── 📄 run_continual_learning.py # Domain 2+ Continual Learning
│   └── 📄 evaluate_model.py         # 모델 평가
├── 📁 notebooks/                    # 분석용 노트북
│   └── 📄 analysis.ipynb            # 결과 분석
└── 📄 main.py                       # 전체 실험 실행
```

## 🔧 각 모듈의 역할

### 1. `src/data_loader.py`
- UOS data_scenario1 폴더에서 데이터 로딩
- 파일명 파싱 → 텍스트 메타데이터 생성
- Domain별 데이터 분할 (RPM 기준)
- PyTorch Dataset/DataLoader 구현

### 2. `src/text_encoder.py`
- DistilBERT + LoRA 구현
- 텍스트 임베딩 생성 (512차원)
- Domain 1: LoRA fine-tuning
- Domain 2+: Freeze

### 3. `src/vibration_encoder.py`
- TST (Time Series Transformer) 기반
- 진동 신호 → 임베딩 (512차원)
- 모든 도메인에서 Full training

### 4. `src/textvib_model.py`
- Bidirectional InfoNCE loss
- Asymmetric temperature (τ_text, τ_vib)
- First Domain/Continual 모드 지원

### 5. `src/continual_trainer.py`
- Domain별 순차 학습 파이프라인
- Replay buffer 관리
- 성능 평가 및 로깅

### 6. `src/replay_buffer.py`
- 이전 도메인 embedding 저장
- Memory-efficient storage
- Sampling strategy

## 🚀 구현 순서

1. **기본 구조 생성** ✅
2. **데이터 로더 구현**
3. **Text Encoder 구현**  
4. **Vibration Encoder 구현**
5. **메인 모델 구현**
6. **Continual Learning 파이프라인**
7. **실험 스크립트 작성**
8. **테스트 및 디버깅**

## 📊 실험 시나리오

### Domain 순서:
1. **Domain 1 (600 RPM)**: First Domain Training
2. **Domain 2 (800 RPM)**: Continual Learning 시작
3. **Domain 3-6 (1000-1600 RPM)**: 순차 학습

### 평가 메트릭:
- **Average Accuracy**: 모든 도메인 평균 성능
- **Forgetting Rate**: 이전 도메인 성능 저하
- **Forward Transfer**: 새 도메인 학습 효율성
- **Memory Efficiency**: Replay buffer 크기 vs 성능

## 🎯 최종 목표
- **논문용 실험 결과** 생성
- **Ablation studies** 지원
- **재현 가능한 코드** 구조
