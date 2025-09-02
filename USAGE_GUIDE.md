# TextVibCLIP 사용 가이드

## 🚀 빠른 시작

### 1. 데이터 준비

#### 시나리오 1: UOS 데이터셋 (Varying Speed)
```bash
# UOS 원본 데이터를 uos_data/ 폴더에 배치 후
python prepare_uos_scenario1.py
```
- **목표**: RPM 변화에 따른 continual learning
- **도메인**: 600 → 800 → 1000 → 1200 → 1400 → 1600 RPM
- **출력**: `data_scenario1/` 폴더 생성 (126개 파일, 완벽한 라벨 균형)

#### 시나리오 2: CWRU 데이터셋 (Varying Load)
```bash
# CWRU 원본 데이터를 cwru_data/ 폴더에 배치 후  
python prepare_cwru_scenario2.py
```
- **목표**: 부하 변화에 따른 continual learning
- **도메인**: 0hp → 1hp → 2hp → 3hp
- **출력**: `data_scenario2/` 폴더 생성 (16개 파일, 완벽한 라벨 균형)

### 2. 실험 실행

#### 🚀 **통합 실험 (권장)** - 두 시나리오 한번에 실행
```bash
# 전체 실험 (UOS + CWRU)
python run_all_scenarios.py --output_dir results

# 빠른 테스트
python run_all_scenarios.py --quick_test --epochs 5 --output_dir test_results

# UOS만 실행
python run_all_scenarios.py --skip_cwru --output_dir uos_only

# CWRU만 실행  
python run_all_scenarios.py --skip_uos --output_dir cwru_only
```

#### 개별 시나리오 실행
```bash
# UOS 시나리오 (기본)
python main.py --experiment_name uos_varying_speed --save_visualizations

# CWRU 시나리오
python main.py \
    --experiment_name cwru_varying_load \
    --dataset_type cwru \
    --save_visualizations

# 시각화 없이 실행 (빠른 테스트)
python main.py --experiment_name quick_test --dataset_type uos
```

#### 주요 옵션
- `--first_domain_epochs 50`: First domain training 에포크 수
- `--remaining_domains_epochs 30`: Remaining domains training 에포크 수  
- `--batch_size 8`: 배치 크기
- `--learning_rate 1e-4`: 학습률
- `--replay_buffer_size 500`: Replay buffer 크기
- `--save_plots`: 기본 학습 곡선 저장
- `--save_visualizations`: **고급 시각화 저장 (t-SNE, confusion matrix 등)**
- `--dataset_type uos/cwru`: 데이터셋 타입 선택

### 3. 실험 모드

#### 전체 파이프라인
```bash
python main.py --mode full
```

#### First Domain만 학습
```bash
python main.py --mode first_domain_only
```

#### Remaining Domains만 학습 (체크포인트 로딩)
```bash
python main.py \
    --mode remaining_domains_only \
    --load_first_domain_checkpoint results/experiment_xxx/checkpoints/first_domain_final.pth
```

## 📊 데이터셋 비교

| 항목 | UOS (시나리오 1) | CWRU (시나리오 2) |
|------|------------------|-------------------|
| **Domain Shift** | Varying Speed | Varying Load |
| **Domain 순서** | 600→800→1000→1200→1400→1600 RPM | 0→1→2→3 HP |
| **베어링 타입** | 3종 (Deep Groove Ball, Tapered Roller, Cylindrical Roller) | 1종 (Deep Groove Ball) |
| **회전체 상태** | 4종 (H/L/U/M) | 1종 (H, 정상) |
| **베어링 상태** | 4종 (H/B/IR/OR) | 4종 (Normal/B/IR/OR) |
| **파일 수** | 126개 (7 × 18) | 16개 (4 × 4) |
| **윈도우/파일** | ~58개 | ~58개 |
| **총 샘플 수** | ~7,308개 | ~928개 |

## 🔧 모델 구성

### TextVibCLIP 아키텍처
- **Text Encoder**: DistilBERT + LoRA (parameter-efficient adaptation)
- **Vibration Encoder**: Time Series Transformer (TST) 기반
- **Loss Function**: Bidirectional InfoNCE with asymmetric temperature

### Continual Learning 전략
1. **First Domain**: First domain training (Text LoRA + Vibration full training)
2. **Remaining Domains**: Text freeze + Vibration adaptation + Replay buffer

### 온도 파라미터
- **First Domain Training**: τ_text = τ_vib = 0.07 (균등 학습)
- **Continual Learning**: τ_text = 0.12, τ_vib = 0.04 (비대칭 적응)

## 📈 결과 분석

### 통합 실험 결과 (CSV + 시각화)
```
results/
├── detailed_results_YYYYMMDD_HHMMSS.csv              # 도메인별 상세 성능
├── summary_results_YYYYMMDD_HHMMSS.csv               # 시나리오별 요약 성능
├── comparison_results_YYYYMMDD_HHMMSS.csv            # 시나리오 간 비교표
├── all_scenarios_YYYYMMDD_HHMMSS.log                 # 전체 실행 로그
└── 📊 논문용 시각화 결과들:
    ├── TextVibCLIP_1_UOS_Scenario1_tsne.png          # UOS t-SNE (라벨별 구분)
    ├── TextVibCLIP_1_CWRU_Scenario2_tsne.png         # CWRU t-SNE (라벨별 구분)
    ├── TextVibCLIP_2_continual_summary.png           # Continual Learning 종합 비교
    ├── TextVibCLIP_3_UOS_Scenario1_domain_shift.png  # UOS Domain Shift 분석
    ├── TextVibCLIP_3_CWRU_Scenario2_domain_shift.png # CWRU Domain Shift 분석
    ├── TextVibCLIP_4_UOS_Scenario1_confusion.png     # UOS Confusion Matrix
    └── TextVibCLIP_4_CWRU_Scenario2_confusion.png    # CWRU Confusion Matrix
```

### CSV 파일 구조

#### 1. `detailed_results.csv` (도메인별 상세)
| Scenario | Domain_Index | Domain_Name | Shift_Type | Accuracy | Top1_Retrieval | Top5_Retrieval | Samples_Per_Domain |
|----------|--------------|-------------|------------|----------|----------------|----------------|-------------------|
| UOS_Scenario1 | 1 | 600RPM | Varying Speed | 0.xxxx | 0.xxxx | 0.xxxx | 7488 |
| UOS_Scenario1 | 2 | 800RPM | Varying Speed | 0.xxxx | 0.xxxx | 0.xxxx | 7488 |
| CWRU_Scenario2 | 1 | 0HP | Varying Load | 0.xxxx | 0.xxxx | 0.xxxx | 232 |

#### 2. `summary_results.csv` (시나리오별 요약)
| Scenario | Shift_Type | Num_Domains | Avg_Accuracy | Avg_Forgetting | Total_Samples | Total_Time_Minutes |
|----------|------------|-------------|--------------|----------------|---------------|-------------------|
| UOS_Scenario1 | Varying Speed | 6 | 0.xxxx | 0.xxxx | 44928 | xxx.x |
| CWRU_Scenario2 | Varying Load | 4 | 0.xxxx | 0.xxxx | 928 | xxx.x |

### 개별 실험 결과
```
results/experiment_name_timestamp/
├── checkpoints/          # 모델 체크포인트
├── logs/                # 학습 로그
├── plots/               # 학습 곡선 플롯
└── results/             # 성능 메트릭
    ├── training_history.pth
    ├── experiment_config.pth
    └── replay_buffer.pth
```

### 주요 메트릭
- **Average Accuracy**: 모든 도메인 평균 정확도
- **Average Forgetting**: 이전 도메인 성능 저하 정도
- **Top-1/Top-5 Retrieval**: 검색 성능
- **Domain별 성능**: 각 도메인별 상세 성능

### 📊 논문용 시각화 Figure 설명

#### **Figure 1: Advanced t-SNE Embedding Space**
- **목적**: Multimodal alignment 품질 시각적 증명
- **내용**: 
  - 좌측: 모달리티별 구분 (Text=원형, Vibration=삼각형)
  - 우측: 고장 유형별 구분 (Normal/B/IR/OR)
- **논문 활용**: "텍스트와 진동이 동일한 임베딩 공간에 잘 정렬됨"

#### **Figure 2: Continual Learning Performance Summary**
- **목적**: 두 시나리오 종합 비교 분석
- **내용**:
  - (a) 도메인별 정확도 진화
  - (b) Catastrophic forgetting 비교
  - (c) Retrieval 성능 비교  
  - (d) 데이터 규모 vs 성능 관계
- **논문 활용**: "제안 방법의 continual learning 효과성 증명"

#### **Figure 3: Domain Shift Analysis**
- **목적**: Domain shift 정도 정량화
- **내용**:
  - 도메인별 임베딩 중심간 거리 히트맵
  - 순차적 domain transition 크기 분석
- **논문 활용**: "RPM/Load 변화가 임베딩 공간에 미치는 영향 분석"

#### **Figure 4: Confusion Matrices**
- **목적**: 분류 성능 상세 분석
- **내용**: 각 도메인별 혼동 행렬
- **논문 활용**: "어떤 고장 유형이 분류하기 어려운지 분석"

## 🛠️ 트러블슈팅

### 메모리 부족 시
```bash
python main.py --batch_size 4 --num_workers 2
```

### GPU 사용 불가 시
```bash
python main.py --device cpu --no_amp
```

### 데이터 로딩 오류 시
- `data_scenario1/` 또는 `data_scenario2/` 폴더 존재 확인
- 데이터 전처리 스크립트 재실행

## 📚 참고 자료

- **README_about_TextVibCLIP.md**: 상세 기술 문서
- **연구제안서.txt**: 연구 배경 및 방법론
- **src/**: 소스 코드 상세 구현