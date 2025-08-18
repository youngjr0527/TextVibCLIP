# TextVibCLIP 사용 가이드

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 필요한 패키지 설치
pip install torch transformers peft scikit-learn matplotlib seaborn tqdm
```

### 2. 데이터 준비

UOS 데이터셋을 다운로드한 후, 실험용 데이터 준비:

```bash
python prepare_uos_scenario1.py
```

### 3. 전체 실험 실행

```bash
# 기본 실험 (Joint Training + Continual Learning)
python main.py --experiment_name "my_experiment" --batch_size 32 --joint_epochs 50 --continual_epochs 30

# GPU 사용
python main.py --device cuda --experiment_name "gpu_experiment"

# 결과 플롯 저장
python main.py --experiment_name "experiment_with_plots" --save_plots
```

## 📋 실험 모드

### 1. 전체 파이프라인 (추천)

```bash
python main.py --mode full --experiment_name "full_pipeline"
```

- Joint Training (Domain 1)
- Continual Learning (Domain 2-6)
- 자동 결과 저장 및 분석

### 2. Joint Training만

```bash
python main.py --mode joint_only --experiment_name "joint_only"
# 또는
python experiments/run_joint_training.py
```

### 3. Continual Learning만

```bash
python main.py --mode continual_only --load_joint_checkpoint checkpoints/joint_model.pth
# 또는
python experiments/run_continual_learning.py --joint_checkpoint checkpoints/joint_model.pth
```

## ⚙️ 주요 하이퍼파라미터

### 모델 설정

```bash
--embedding_dim 512           # 임베딩 차원
--batch_size 32              # 배치 크기
--learning_rate 1e-4         # 학습률
```

### Continual Learning 설정

```bash
--replay_buffer_size 500     # 도메인당 저장할 샘플 수
--replay_ratio 0.3           # Replay 데이터 비율
--joint_epochs 50            # Joint training 에포크
--continual_epochs 30        # Domain별 학습 에포크
```

## 📊 모델 평가

### 학습된 모델 평가

```bash
python experiments/evaluate_model.py \
    --model_checkpoint results/my_experiment_20250101_120000/checkpoints/final_model.pth \
    --subset test \
    --save_results \
    --output_dir evaluation_results
```

### 평가 메트릭

- **Classification Accuracy**: 텍스트-진동 쌍 분류 정확도
- **Cross-domain Retrieval**: Top-k retrieval 성능
- **Embedding Similarity**: 평균 cosine similarity
- **Continual Learning**: Average accuracy, Forgetting rate

## 📁 출력 구조

실험 완료 후 다음과 같은 구조로 결과가 저장됩니다:

```
results/my_experiment_20250101_120000/
├── checkpoints/                 # 모델 체크포인트
│   ├── joint_training_final.pth
│   ├── domain_600_best.pth
│   ├── domain_800_best.pth
│   └── final_model.pth
├── logs/                        # 학습 로그
│   └── textvibclip_20250101_120000.log
├── plots/                       # 시각화 결과
│   └── continual_learning_curves.png
└── results/                     # 실험 결과
    ├── training_history.pth
    ├── replay_buffer.pth
    └── experiment_config.pth
```

## 🔧 고급 사용법

### 1. 커스텀 설정

`configs/model_config.py`에서 모델 구성 수정:

```python
MODEL_CONFIG = {
    'embedding_dim': 1024,  # 더 큰 임베딩
    'vibration_encoder': {
        'num_layers': 8,    # 더 깊은 TST
        # ...
    }
}
```

### 2. 새로운 데이터셋 추가

```python
# src/data_loader.py 확장
class CustomDataset(Dataset):
    # 새로운 데이터셋 구현
    pass
```

### 3. 실험 스크립트 커스터마이징

```python
# experiments/ 폴더에 새로운 실험 스크립트 추가
# 기존 스크립트를 참고하여 작성
```

## 🐛 문제 해결

### 1. GPU 메모리 부족

```bash
# 배치 크기 줄이기
python main.py --batch_size 16

# Mixed precision 사용 (model_config.py에서 설정)
DEVICE_CONFIG['mixed_precision'] = True
```

### 2. 데이터 로딩 오류

```bash
# 데이터 경로 확인
ls data_scenario1/

# 데이터 재준비
python prepare_uos_scenario1.py
```

### 3. 의존성 오류

```bash
# 필수 패키지 재설치
pip install --upgrade torch transformers peft

# CUDA 버전 확인 (GPU 사용 시)
python -c "import torch; print(torch.cuda.is_available())"
```

## 📈 성능 최적화

### 1. 학습 속도 향상

- GPU 사용: `--device cuda`
- 더 많은 워커: `--num_workers 8`
- Mixed precision 활성화

### 2. 메모리 최적화

- 배치 크기 조정: `--batch_size 16`
- Replay buffer 크기 조정: `--replay_buffer_size 300`

### 3. 수렴 개선

- 학습률 조정: `--learning_rate 5e-5`
- 에포크 수 증가: `--joint_epochs 100`

## 📚 추가 리소스

- **연구 논문**: `연구paper.txt` 참조
- **모델 구조**: `PROJECT_STRUCTURE.md` 참조
- **API 문서**: 각 모듈의 docstring 참조

## 🤝 기여 방법

1. 새로운 기능이나 버그 수정
2. 문서 개선
3. 실험 결과 공유
4. 코드 리뷰 및 최적화 제안

---

## 💡 실험 예시

### 기본 실험

```bash
# 1. 데이터 준비
python prepare_uos_scenario1.py

# 2. 전체 실험 실행
python main.py \
    --experiment_name "baseline_experiment" \
    --batch_size 32 \
    --joint_epochs 50 \
    --continual_epochs 30 \
    --replay_buffer_size 500 \
    --save_plots

# 3. 결과 평가
python experiments/evaluate_model.py \
    --model_checkpoint results/baseline_experiment_*/checkpoints/final_model.pth \
    --save_results
```

### Ablation Study

```bash
# LoRA rank 비교
python main.py --experiment_name "lora_rank_16" # r=16 (기본값)
python main.py --experiment_name "lora_rank_64" # configs에서 r=64로 수정

# Replay buffer 크기 비교
python main.py --experiment_name "replay_100" --replay_buffer_size 100
python main.py --experiment_name "replay_1000" --replay_buffer_size 1000

# Temperature 설정 비교 (configs에서 수정)
```

이 가이드를 따라하면 TextVibCLIP을 성공적으로 사용할 수 있습니다! 🚀
