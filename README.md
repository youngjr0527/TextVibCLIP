# TextVibCLIP: Multimodal Continual Learning for Bearing Fault Diagnosis

## 🎯 연구 개요

**TextVibCLIP**은 베어링 고장 진단을 위한 **Domain-Incremental Continual Learning** 프레임워크입니다. 진동 신호와 텍스트 메타데이터를 결합하여 CLIP-inspired contrastive learning을 적용하고, 운전 조건 변화(domain shift)에 robust한 진단 모델을 구현합니다.

### 🔑 핵심 특징
- **Domain-Incremental Learning**: 클래스는 고정, 도메인만 순차적 변화
- **Multimodal Contrastive Learning**: 진동 신호 + 텍스트 메타데이터
- **Asymmetric Continual Adaptation**: Text encoder freeze + Vibration encoder adaptation
- **Similarity-based Retrieval**: 실제 사용 시 후보 텍스트 중 최고 유사도 선택
- **두 가지 시나리오**: UOS (Varying Speed), CWRU (Varying Load)

### 📊 성능 평가 방식
- **주 평가 지표**: **Retrieval Accuracy** (진동-텍스트 코사인 유사도 기반)
- **이유**: 실제 모델 배포 시 사용 방식과 완전히 일치
- **보조 지표**: Text/Vib/Ensemble accuracy (각 인코더 성능 분석용)

---

## 📊 실험 시나리오

### **시나리오 1: UOS - Varying Speed (Deep Groove Ball)**
- **Domain 순서**: 600 → 800 → 1000 → 1200 → 1400 → 1600 RPM
- **베어링 타입**: Deep Groove Ball (6204) 단일 타입
- **클래스**: {H, B, IR, OR, L, U, M} (7개) - **모든 도메인에서 동일**
- **Domain Shift**: 회전 속도 변화 (환경 변화)

### **시나리오 2: CWRU - Varying Load**  
- **Domain 순서**: 0 → 1 → 2 → 3 HP
- **클래스**: {Normal, B, IR, OR} (4개) - **모든 도메인에서 동일**
- **Domain Shift**: 부하 변화 (환경 변화)

### ⚠️ **중요**: Domain-Incremental Learning
- **600→800 RPM 변화**: 새로운 클래스가 아님, 환경 변화임
- **모델은 동일한 클래스를 분류**하되, **분포 변화에 적응**해야 함
- **Train/Val/Test**: 모든 subset에서 **동일한 클래스 집합** 유지

---

## 🏗️ 시스템 아키텍처 (논문 작성자용)

### **전체 구조**

```
Input:
┌─────────────────┐    ┌─────────────────┐
│   진동 신호       │    │   텍스트 설명     │
│ [2048 samples]  │    │ "A ball bearing │
└─────────────────┘    │  with fault..." │
         │              └─────────────────┘
         ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│ VibrationEncoder│    │   TextEncoder   │
│  (1D-CNN)       │    │  (DistilBERT    │
│  7M params      │    │   + LoRA)       │
└─────────────────┘    └─────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│  Projection     │    │  Projection     │
│  (2-layer MLP)  │    │  (2-layer MLP)  │
└─────────────────┘    └─────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│   [256-dim]     │    │   [256-dim]     │
│  진동 임베딩      │    │  텍스트 임베딩    │
└─────────────────┘    └─────────────────┘
         │                        │
         └────────────┬─────────────┘
                    ▼
         ┌─────────────────┐
         │  Ranking Loss   │
         │   (Triplet)     │
         └─────────────────┘
                    +
         ┌─────────────────┐
         │ Auxiliary Loss  │
         │ (Classification)│
         └─────────────────┘
```

### **모델 컴포넌트 상세**

#### **1. Vibration Encoder (1D-CNN)**
```python
Architecture:
- Input: [batch, 2048] (raw vibration signal)
- Conv1D layers: [64, 128, 256, 512] channels
- Kernel sizes: [7, 5, 3, 3]
- MaxPooling after each conv
- Global Average Pooling
- Output: [batch, 256] (vibration features)

Total params: ~7M
Learning: Full training (모든 도메인)
```

#### **2. Text Encoder (DistilBERT + LoRA)**
```python
Architecture:
- Base: DistilBERT (66M params, frozen)
- LoRA adapters: rank=8, alpha=16
  - Query/Value matrices에만 적용
  - Trainable params: ~0.3M
- Output: [batch, 256] (text features)

Total params: 66M (base) + 0.3M (LoRA)
Learning: 
  - Domain 1: LoRA fine-tuning
  - Domain 2+: Freeze (LoRA 비활성화)
```

#### **3. Projection Layers**
```python
Both Text & Vib:
- Linear(256, 256) + ReLU + Dropout(0.1)
- Linear(256, 256) + LayerNorm
- Output: [batch, 256] (normalized embeddings)

Purpose: 공통 임베딩 공간으로 매핑
Learning: 항상 학습 가능
```

#### **4. Auxiliary Classification Heads**
```python
Text Classifier:
- Linear(256, num_classes)
- Purpose: 텍스트 인코더 안정화

Vib Classifier:
- Linear(256, num_classes)
- Purpose: 진동 인코더 안정화

num_classes: 4 (CWRU), 7 (UOS)
```

### **Continual Learning 전략**
- **Domain 1**: Text LoRA + Vibration 동시 학습
- **Domain 2+**: Text freeze + Vibration adaptation + Replay buffer

---

## 🎯 **실제 산업 사용 시나리오**

TextVibCLIP의 핵심 가치는 **실제 산업 현장에서의 활용 방식**에 있습니다. 모델이 배포된 후의 추론 과정은 다음과 같습니다:

### **추론 프로세스**:

```python
# 산업 현장에서의 실제 사용
def diagnose_bearing_fault(new_vibration_signal):
    # 1. 새로운 진동 신호 입력
    vibration_embedding = vibration_encoder(new_vibration_signal)
    
    # 2. 가능한 모든 진단 텍스트 후보군 준비
    candidate_texts = [
        "Healthy bearing condition observed",
        "Ball element defect detected", 
        "Inner race fault observed",
        "Outer race defect detected",
        "Mechanical looseness detected",
        "Rotor unbalance detected",
        "Shaft misalignment detected"
    ]
    
    # 3. 각 후보 텍스트를 임베딩 공간에 매핑
    text_embeddings = text_encoder(candidate_texts)
    
    # 4. 유사도 계산 및 최고 매칭 선택
    similarities = cosine_similarity(vibration_embedding, text_embeddings)
    best_match_index = argmax(similarities)
    
    # 5. 최종 진단 결과
    diagnosis = candidate_texts[best_match_index]
    confidence = similarities[best_match_index]
    
    return diagnosis, confidence
```

### **핵심 장점**:
- **유연성**: 새로운 설명 방식이나 고장 유형 추가 시 쉬운 확장
- **해석 가능성**: 자연어 설명으로 직관적인 진단 결과
- **신뢰도 제공**: 유사도 점수로 진단 신뢰도 정량화
- **Zero-shot 확장**: 학습하지 않은 새로운 텍스트 설명도 활용 가능

이러한 **similarity-based retrieval 방식**은 기존의 고정된 분류 체계와 달리, 실제 산업 현장의 다양하고 동적인 요구사항에 유연하게 대응할 수 있습니다.

---

## 🚀 실행 방법

### 1. 빠른 테스트
```bash
python run_all_scenarios.py --quick_test --epochs 5
```

### 2. 전체 실험
```bash
python run_all_scenarios.py --epochs 50
```

### 3. 개별 시나리오
```bash
# UOS만 실행
python run_all_scenarios.py --skip_cwru --epochs 30

# CWRU만 실행  
python run_all_scenarios.py --skip_uos --epochs 30
```

---

## 📁 프로젝트 구조

```
TextVibCLIP/
├── 📄 run_all_scenarios.py              # 통합 실험 스크립트
├── 📄 prepare_uos_scenario1.py          # UOS 데이터 전처리 (Deep Groove Ball만)
├── 📄 prepare_cwru_scenario2.py         # CWRU 데이터 전처리
├── 📁 src/                              # 소스 코드
│   ├── textvib_model.py                 # TextVibCLIP 메인 모델
│   ├── continual_trainer.py            # Continual Learning 트레이너
│   ├── data_loader.py                   # 데이터 로더 (UOS/CWRU)
│   ├── data_cache.py                    # 데이터 캐싱 시스템
│   ├── text_encoder.py                  # DistilBERT + LoRA
│   ├── vibration_encoder.py             # 1D-CNN
│   └── utils.py                         # 유틸리티 함수들
├── 📁 configs/                          # 설정 파일
│   └── model_config.py                  # 모델 및 실험 설정
├── 📁 uos_data/                         # UOS 원본 데이터
├── 📁 cwru_data/                        # CWRU 원본 데이터
├── 📁 data_scenario1/                   # UOS 전처리 데이터 (Deep Groove Ball)
├── 📁 data_scenario2/                   # CWRU 전처리 데이터
├── 📁 cache/                            # 데이터 캐시
└── 📁 results/                          # 실험 결과
```

---

## 📊 결과 해석

### 성능 메트릭

#### **주 평가 지표: Retrieval Accuracy**
- **정의**: 진동 임베딩과 텍스트 프로토타입 간 코사인 유사도 기반 분류 정확도
- **평가 방식**: 
  1. 각 클래스의 텍스트 프로토타입 생성 (여러 텍스트 설명의 평균 임베딩)
  2. 진동 신호를 임베딩 공간에 매핑
  3. 모든 클래스 프로토타입과의 코사인 유사도 계산
  4. 가장 높은 유사도를 가진 클래스로 분류
- **중요성**: **실제 모델 사용 방식과 완전히 일치**하는 평가 (SCI 논문 권장)

#### **보조 평가 지표**
- **Text Accuracy**: 텍스트 분류 헤드 기반 정확도 (텍스트 인코더 성능)
- **Vib Accuracy**: 진동 분류 헤드 기반 정확도 (진동 인코더 성능)
- **Ensemble Accuracy**: Text + Vib 가중 평균 정확도
- **Top5 Retrieval**: 상위 5개 후보 중 정답 포함 비율 (UOS만 제공)
- **Forgetting Score**: 이전 도메인 성능 저하 정도

### 평가 방식 선택 근거

**왜 Retrieval Accuracy인가?**

실제 산업 현장 배포 시 모델은 다음과 같이 작동합니다:
```python
# 실제 사용 시나리오
vibration_embedding = model.encode_vibration(new_signal)
text_embeddings = model.encode_texts(candidate_descriptions)
similarities = cosine_similarity(vibration_embedding, text_embeddings)
diagnosis = candidate_descriptions[argmax(similarities)]
```

따라서 **Retrieval Accuracy**가 실제 성능을 가장 정확하게 반영합니다.

### 기대 결과
- **UOS**: 더 어려운 태스크 (7개 클래스) → 낮은 정확도 예상
- **CWRU**: 상대적으로 쉬운 태스크 (4개 클래스) → 높은 정확도 예상

---

## 🔧 기술적 세부사항 (논문 Methodology용)

### **손실 함수 수식**

#### **Triplet Ranking Loss**
```
L_triplet = 1/N * Σ_{i=1}^N max(0, m - sim(z_vib^i, z_text^+) + sim(z_vib^i, z_text^-))

where:
- z_vib^i: anchor (진동 임베딩)
- z_text^+: positive (같은 클래스 텍스트 임베딩)
- z_text^-: negative (다른 클래스 텍스트 임베딩)
- m: margin (0.3)
- sim(a, b) = cosine_similarity(a, b) = (a · b) / (||a|| ||b||)
```

#### **Auxiliary Classification Loss**
```
L_aux = L_text + L_vib

L_text = CrossEntropy(f_text(z_text), y)
L_vib = CrossEntropy(f_vib(z_vib), y)

where:
- f_text, f_vib: 분류 헤드 (Linear layers)
- y: ground truth labels
```

#### **Total Loss**
```
L_total = L_triplet + λ_aux * L_aux

λ_aux = {
    2.0  (First domain: 균형잡힌 학습)
    5.0  (Remaining domains: 빠른 적응)
}
```

### **Continual Learning Algorithm**

```
Algorithm: TextVibCLIP Continual Learning

Input: 
  - Domain sequence D = {D1, D2, ..., D6}
  - Each Di = {(x_vib, x_text, y)}
  
Output:
  - Model θ adapted to all domains
  - Replay buffer R

1. First Domain Training (D1):
   θ_text ← LoRA fine-tune on D1
   θ_vib ← Full training on D1
   R ← ∅
   Save: θ_D1

2. For each remaining domain Di (i = 2, ..., 6):
   a. Freeze θ_text (disable LoRA)
   b. Sample R_replay from R
   c. Train θ_vib on Di ∪ R_replay:
      - Batch = 50% Di + 50% R_replay
      - Loss = L_triplet + 5.0 * L_aux
   d. Update R with samples from Di:
      - Select top-k diverse samples
      - R ← R ∪ samples_Di (max: 500 for UOS, 50 for CWRU)
   e. Evaluate on all {D1, ..., Di}
   f. Save: θ_Di

3. Return θ, R
```

### **Retrieval Evaluation Algorithm**

```
Algorithm: Retrieval-based Classification

Input:
  - Test set T = {(x_vib, y)}
  - Class prompts P = {P_0, P_1, ..., P_C}
  - Each P_c = ["template1", "template2", "template3"]

Output:
  - Retrieval accuracy

1. Generate text prototypes:
   For each class c:
     embeddings_c = [encode_text(p) for p in P_c]
     prototype_c = mean(embeddings_c)
     prototype_c = normalize(prototype_c)

2. Classify test samples:
   For each (x_vib, y) in T:
     z_vib = encode_vibration(x_vib)
     z_vib = normalize(z_vib)
     
     similarities = [cosine_sim(z_vib, prototype_c) for all c]
     prediction = argmax(similarities)
     
     if prediction == y:
       correct += 1

3. Return accuracy = correct / |T|
```

### **데이터 분할 전략**

#### **CWRU 파일 레벨 분할**
```
For each fault type (B, IR, OR):
  - 3 bearings available
  - Assign: bearing_1 → train, bearing_2 → val, bearing_3 → test
  - Prevents same bearing in multiple subsets

For Normal (H):
  - Single bearing, multiple time segments
  - Split by time: early → train, middle → val, late → test
```

#### **UOS 윈도우 레벨 분할**
```
For each class:
  - Single file per class
  - Extract overlapping windows
  - Shuffle windows (seed=42)
  - Split: 70% train, 15% val, 15% test
  - Prevents temporal ordering memorization
```

---

## 🎯 연구 기여

1. **Domain-Incremental Continual Learning**: 산업 현장의 실제 요구사항 반영
2. **Multimodal Contrastive Learning**: 진동 + 텍스트 정보 활용
3. **Asymmetric Adaptation**: 모달리티별 차별적 학습 전략
4. **실용적 검증**: 실제 베어링 데이터셋으로 검증

---

## 📈 실험 결과 예시

### UOS 시나리오 (Varying Speed, Deep Groove Ball)
- **Retrieval Accuracy**: ~60-85% (7개 클래스, 실제 사용 방식 기준)
- **Top5 성능**: ~80-90% (검색 관점에서 우수)
- **망각도**: 0.0% (Replay buffer 효과)
- **참고**: Vib/Text/Ensemble accuracy는 보조 지표로 제공

### CWRU 시나리오 (Varying Load)
- **Retrieval Accuracy**: ~90-100% (4개 클래스, 실제 사용 방식 기준)
- **일관성**: 모든 도메인에서 안정적 성능
- **망각도**: 0.0% (효과적인 지식 보존)
- **참고**: CWRU는 상대적으로 쉬운 태스크

---

## 🔬 **실험 파이프라인 상세 (논문 작성자용)**

### **전체 실험 흐름**

```
python run_scenarios.py 실행
    ↓
1️⃣ First Domain Training (예: 600RPM 또는 0HP)
    ├─ Text Encoder: LoRA fine-tuning (parameter-efficient)
    ├─ Vibration Encoder: Full training
    ├─ Loss: Triplet Ranking Loss + Auxiliary Classification Loss
    ├─ 에포크: 15 epochs (UOS), 15 epochs (CWRU)
    └─ 결과: first_domain_final.pth 저장
    ↓
2️⃣ Remaining Domains Training (예: 800~1600RPM 또는 1~3HP)
    ├─ Text Encoder: Freeze (LoRA 비활성화)
    ├─ Vibration Encoder: Full adaptation
    ├─ Replay Buffer: 이전 도메인 샘플 500개 (UOS) / 50개 (CWRU) 저장
    ├─ Loss: Triplet Ranking Loss + Auxiliary Loss (weight=5.0, 빠른 적응)
    ├─ 에포크: 6 epochs per domain
    └─ 결과: domain_{value}_best.pth 각각 저장
    ↓
3️⃣ Evaluation (모든 도메인)
    ├─ 주 평가: Retrieval Accuracy (진동-텍스트 코사인 유사도)
    ├─ 보조 평가: Text/Vib/Ensemble Accuracy
    └─ Forgetting Score: 이전 도메인 성능 저하 측정
    ↓
4️⃣ 결과 저장
    └─ results/{timestamp}/results_{timestamp}.json
```

### **1. First Domain Training (Foundation Learning)**

**목적**: 진동-텍스트 멀티모달 임베딩 공간 구축

**학습 대상**:
- **Text Encoder**: DistilBERT + LoRA (rank=8)
  - Base model: 66M params (freeze)
  - LoRA adapters: ~0.3M params (trainable)
  - 역할: 고장 유형의 의미론적 표현 학습
  
- **Vibration Encoder**: 1D-CNN
  - 7M params (all trainable)
  - 역할: 진동 신호의 특징 패턴 학습

**손실 함수**:
```python
L_total = L_triplet + λ_aux * L_aux

L_triplet = 1/N * Σ max(0, margin - sim(vib_i, text_same) + sim(vib_i, text_diff))
L_aux = CrossEntropy(text_logits) + CrossEntropy(vib_logits)

λ_aux = 2.0  # First domain에서는 균형잡힌 학습
margin = 0.3  # Triplet margin
```

**최적화**:
- Optimizer: AdamW
- Learning rate: 1e-4 (UOS), 5e-5 (CWRU)
- Weight decay: 1e-4
- Gradient clipping: max_norm=0.1

**데이터**:
- Batch size: 8 (UOS), 4 (CWRU)
- Epochs: 15
- Early stopping: patience=8 (UOS), 5 (CWRU)

### **2. Remaining Domains Training (Continual Adaptation)**

**목적**: 새로운 운전 조건에 적응하면서 이전 지식 보존

**Asymmetric Learning Strategy**:
- **Text Encoder**: **완전 Freeze**
  - LoRA 비활성화
  - Projection layer만 최소 적응
  - 이유: 고장 유형의 의미는 RPM/Load와 무관하게 일정
  
- **Vibration Encoder**: **Full Adaptation**
  - 모든 파라미터 학습
  - 이유: 진동 패턴은 운전 조건에 민감하게 변화

**Replay Buffer Mechanism**:
```python
# 각 도메인 학습 후
replay_buffer.add_samples(
    embeddings,      # 진동 임베딩 저장
    texts,           # 원본 텍스트 저장
    labels,          # 라벨 저장
    max_samples=500  # UOS: 500, CWRU: 50
)

# 다음 도메인 학습 시
new_batch + replay_samples → model
```

**샘플 선택 전략**:
- **다양성 기반**: 클래스별 균등 샘플링
- **신뢰도 기반**: 높은 confidence 샘플 우선
- **최신성 고려**: 최근 도메인 샘플 포함

**손실 함수**:
```python
L_total = L_triplet + λ_aux * L_aux

λ_aux = 5.0  # Continual에서는 빠른 적응을 위해 증가
```

**최적화**:
- Learning rate: 5e-5 (UOS), 2e-5 (CWRU)
- Epochs per domain: 6
- Batch composition: 50% new domain + 50% replay samples

### **3. Evaluation Protocol**

**3.1 Retrieval Accuracy (주 평가 지표)**

**평가 과정**:
```python
# Step 1: 텍스트 프로토타입 생성
for each class:
    texts = ["healthy bearing", "normal bearing", ...]  # 3개 템플릿
    text_embeddings = text_encoder(texts)
    prototype = mean(text_embeddings)  # 평균 임베딩

# Step 2: 진동 신호 분류
for each test_sample:
    vib_embedding = vib_encoder(vibration_signal)
    similarities = cosine_similarity(vib_embedding, all_prototypes)
    prediction = argmax(similarities)

# Step 3: 정확도 계산
retrieval_accuracy = (predictions == ground_truth).mean()
```

**프롬프트 템플릿**:
- **CWRU (4-클래스)**:
  - Class 0: "healthy bearing", "normal bearing with no fault", "bearing vibration without defect"
  - Class 1: "bearing with ball fault", "ball defect in bearing", "ball damage on bearing"
  - Class 2: "bearing inner race fault", "inner ring defect in bearing", "inner race damage of bearing"
  - Class 3: "bearing outer race fault", "outer ring defect in bearing", "outer race damage of bearing"

- **UOS (7-클래스)**:
  - Class 0: "healthy bearing", "normal bearing with no fault", "bearing vibration without defect"
  - Class 1: "bearing with ball fault", "ball defect in bearing", "ball damage on bearing"
  - Class 2: "bearing inner race fault", "inner ring defect in bearing", "inner race damage of bearing"
  - Class 3: "bearing outer race fault", "outer ring defect in bearing", "outer race damage of bearing"
  - Class 4: "mechanical looseness detected", "mechanical looseness fault", "looseness in mechanical system"
  - Class 5: "rotor unbalance detected", "rotor imbalance fault", "unbalanced rotor condition"
  - Class 6: "shaft misalignment detected", "shaft misalignment fault", "misaligned shaft condition"

**중요**: 이 평가 방식이 `model.predict_best_match()` 함수의 실제 동작과 **완전히 일치**합니다.

**3.2 보조 평가 지표**

- **Text Accuracy**: 텍스트 분류 헤드 기반 (텍스트 인코더 성능 측정)
- **Vib Accuracy**: 진동 분류 헤드 기반 (진동 인코더 성능 측정)
- **Ensemble Accuracy**: `w * vib_acc + (1-w) * text_acc` (w는 학습된 가중치)

**3.3 Forgetting Score**

```python
# 각 이전 도메인에 대해
forgetting_i = max_accuracy_i - current_accuracy_i

# 평균 forgetting
average_forgetting = mean(forgetting_i for all previous domains)
```

### **4. 데이터 분할 전략 (Data Leakage 방지)**

**CWRU 데이터셋**:
- **파일 레벨 분할**: 같은 베어링의 다른 파일을 train/val/test로 분리
- **전략**: B/IR/OR 결함은 서로 다른 베어링 할당, H 결함은 시간 기반 분할
- **목적**: 같은 베어링의 연속 신호가 여러 subset에 들어가는 것 방지

**UOS 데이터셋**:
- **윈도우 레벨 랜덤 분할**: 각 클래스당 1개 파일이므로 윈도우를 랜덤 분할
- **Shuffle**: 파일 순서 + 윈도우 순서 모두 랜덤화 (seed=42)
- **목적**: 파일 순서나 윈도우 연속성을 모델이 암기하는 것 방지

**공통**:
- **Stratified split**: 클래스 균형 유지 (70% train, 15% val, 15% test)
- **Window overlap**: 0.25 (CWRU), 0.25 (UOS) - 낮은 overlap으로 독립성 확보

### **5. 하이퍼파라미터 요약 (논문 Table용)**

| 항목 | UOS | CWRU | 설명 |
|------|-----|------|------|
| **First Domain Training** |
| Epochs | 15 | 15 | Foundation learning |
| Learning rate | 1e-4 | 5e-5 | CWRU는 작은 데이터로 낮은 LR |
| Batch size | 8 | 4 | CWRU는 극소 데이터 대응 |
| Aux loss weight (λ_aux) | 2.0 | 2.0 | 균형잡힌 학습 |
| **Remaining Domains Training** |
| Epochs per domain | 6 | 6 | 빠른 적응 |
| Learning rate | 5e-5 | 2e-5 | Continual에서 더 낮은 LR |
| Batch size | 8 | 4 | First domain과 동일 |
| Aux loss weight (λ_aux) | 5.0 | 5.0 | 빠른 적응을 위해 증가 |
| Replay buffer size | 500 | 50 | UOS는 더 많은 샘플 |
| **공통 설정** |
| Embedding dimension | 256 | 256 | 임베딩 공간 차원 |
| Triplet margin | 0.3 | 0.3 | Ranking loss margin |
| Weight decay | 1e-4 | 1e-4 | L2 regularization |
| Gradient clipping | 0.1 | 0.1 | 안정적 학습 |
| LoRA rank | 8 | 8 | Low-rank adaptation |
| LoRA alpha | 16 | 16 | Scaling factor |

### **6. 데이터셋 통계 (논문 Table용)**

#### **UOS Dataset (Scenario 1: Varying Speed)**

| Domain | RPM | Train | Val | Test | Total |
|--------|-----|-------|-----|------|-------|
| D1 | 600 | 1225 | 262 | 262 | 1749 |
| D2 | 800 | 1225 | 262 | 262 | 1749 |
| D3 | 1000 | 1225 | 262 | 262 | 1749 |
| D4 | 1200 | 1225 | 262 | 262 | 1749 |
| D5 | 1400 | 1225 | 262 | 262 | 1749 |
| D6 | 1600 | 1225 | 262 | 262 | 1749 |
| **Total** | - | **7350** | **1572** | **1572** | **10494** |

- **Classes**: 7 (H, B, IR, OR, L, U, M)
- **Bearing type**: Deep Groove Ball (6204) only
- **Signal length**: 2048 samples
- **Window overlap**: 0.25

#### **CWRU Dataset (Scenario 2: Varying Load)**

| Domain | Load | Train | Val | Test | Total |
|--------|------|-------|-----|------|-------|
| D1 | 0HP | 218 | 47 | 47 | 312 |
| D2 | 1HP | 221 | 47 | 48 | 316 |
| D3 | 2HP | 221 | 47 | 48 | 316 |
| D4 | 3HP | 218 | 47 | 47 | 312 |
| **Total** | - | **878** | **188** | **190** | **1256** |

- **Classes**: 4 (Normal, B, IR, OR)
- **Signal length**: 2048 samples
- **Window overlap**: 0.25

---

## 🔬 **연구 방법론 상세**

### **1. Domain-Incremental Learning의 핵심**

#### **기존 연구의 한계**:
- **Class-Incremental**: 새로운 클래스가 순차적으로 등장 (예: 고장유형 A → B → C)
- **Task-Incremental**: 완전히 다른 태스크 (예: 분류 → 회귀 → 검출)

#### **본 연구의 접근 (Domain-Incremental)**:
- **클래스 집합 고정**: {H, B, IR, OR, L, U, M} 항상 동일
- **환경 조건 변화**: 600→800→1000 RPM (운전 조건만 변화)
- **실제 산업 현장 반영**: 새로운 고장이 아닌 운전 조건 변화

**핵심 아이디어**: "같은 고장을 다른 환경에서도 정확히 진단할 수 있는가?"

### **2. Multimodal Contrastive Learning의 혁신**

#### **기존 베어링 진단의 한계**:
- **단일 모달**: 진동 신호만 사용
- **고정 환경**: 특정 조건에서만 학습
- **일반화 부족**: 새로운 환경에서 성능 저하

#### **TextVibCLIP의 혁신**:
- **진동 + 텍스트**: "A deep groove ball bearing with inner race fault at 600 rpm"
- **의미론적 앵커**: 텍스트가 도메인 변화에 robust한 기준점 제공
- **CLIP 패러다임**: 이미지-텍스트 → 진동-텍스트로 확장

**핵심 통찰**: "텍스트 설명이 진동 패턴 학습을 안내한다"

### **3. Asymmetric Continual Adaptation**

#### **기존 Continual Learning의 문제**:
- **Symmetric 학습**: 모든 컴포넌트를 동일하게 업데이트
- **Catastrophic Forgetting**: 새 도메인 학습 시 이전 지식 손실

#### **본 연구의 Asymmetric 전략**:

**Text Encoder (의미론적 지식)**:
- **Domain 1**: LoRA fine-tuning으로 베어링 도메인 지식 학습
- **Domain 2+**: LoRA freeze, Projection만 최소 적응
- **논리**: 고장 유형 의미는 RPM과 무관하게 일정

**Vibration Encoder (신호 패턴)**:
- **모든 Domain**: Full parameter training
- **Replay buffer**: 이전 도메인 패턴 보존
- **논리**: 진동 패턴은 RPM 변화에 민감하므로 적극적 적응 필요

### **4. 실험 설계의 엄밀성**

#### **Domain-Incremental 평가 프로토콜**:

**표준 Continual Learning 평가**:
```
Domain 1 학습 후: 평가 범위 [Domain 1]
Domain 2 학습 후: 평가 범위 [Domain 1, Domain 2]
Domain 3 학습 후: 평가 범위 [Domain 1, Domain 2, Domain 3]
...
```

**Forgetting 측정**:
- **각 이전 도메인**: 역대 최고 성능 vs 현재 성능
- **평균 Forgetting**: 모든 이전 도메인의 망각도 평균
- **0에 가까울수록 좋음**: 이전 지식 잘 보존

#### **Retrieval-based 평가 (주 평가 지표)**:
- **각 클래스의 텍스트 prototype 생성**: 같은 클래스 텍스트들의 평균 임베딩
- **진동 신호 분류**: 모든 prototype과 코사인 유사도 비교하여 가장 유사한 클래스 선택
- **CLIP 방식**: 실제 배포 시 사용 방식과 동일한 평가
- **핵심**: 이 평가 방식이 `predict_best_match()` 함수의 실제 동작과 일치

### **5. 데이터 구조와 전처리**

#### **UOS 데이터셋 특성**:
- **베어링 타입**: Deep Groove Ball (6204) 단일화로 복잡도 감소
- **클래스 정의**: 회전체 상태 + 베어링 상태 조합 (7가지)
- **시간적 분할**: 같은 베어링의 시간 순서를 고려한 train/val/test 분할

#### **CWRU 데이터셋 특성**:
- **부하 변화**: 0→1→2→3 HP (기계적 스트레스 증가)
- **데이터 증강**: 윈도우 크기 축소 + 겹침 증가로 샘플 수 확보
- **클래스 균형**: 모든 고장 유형이 모든 부하 조건에서 동일하게 포함

### **6. 연구의 실용적 가치**

#### **산업 현장 적용성**:
- **동적 운전 조건**: 실제 공장에서 RPM/부하는 지속적으로 변화
- **점진적 적응**: 새로운 운전 조건에 대해 전체 재학습 없이 적응
- **지식 보존**: 기존 고장 진단 능력을 잃지 않으면서 새 조건 학습

#### **기존 방법 대비 장점**:
- **Transfer Learning**: 고정된 source→target, 연속적 변화 대응 불가
- **Domain Adaptation**: 단일 도메인 쌍, 다중 도메인 순차 학습 불가
- **Fine-tuning**: Catastrophic forgetting 문제

**TextVibCLIP**: 연속적 도메인 변화 + 지식 보존 + 실시간 적응

### **7. 실험 결과의 해석**

#### **성능 지표의 의미**:
- **Retrieval Accuracy 60-85%**: 7개 클래스 분류에서 우수 (랜덤 14.3% 대비, 실제 사용 방식 기준)
- **Top-5 성능 80-90%**: 실제 진단에서 후보군 제시 관점에서 실용적
- **Forgetting 0%**: Replay mechanism의 효과적 작동
- **보조 지표**: Text/Vib/Ensemble accuracy는 각 인코더의 개별 성능 분석용

#### **Domain별 성능 변화 패턴**:
- **초기 도메인**: 기본 성능 확립
- **중간 도메인**: 일시적 성능 저하 (새로운 패턴 학습)
- **후기 도메인**: 누적 학습 효과로 성능 향상

**이는 실제 산업 현장에서 기계가 다양한 조건에 노출되면서 점진적으로 robust해지는 과정을 모사**

### **8. 연구의 이론적 기여**

#### **Multimodal Continual Learning 이론**:
- **Cross-modal Anchoring**: 텍스트가 도메인 변화에 불변인 의미론적 앵커 역할
- **Asymmetric Adaptation**: 모달리티별 도메인 민감도에 따른 차별적 학습
- **Contrastive Stability**: 대조 학습이 제공하는 안정적 특징 공간

#### **산업 AI의 새로운 패러다임**:
- **설명 가능한 진단**: 텍스트 설명을 통한 투명한 진단 과정
- **적응적 시스템**: 환경 변화에 자동 적응하는 지능형 시스템
- **지식 누적**: 경험이 쌓일수록 더 정확해지는 학습 시스템

---

## 🔍 주요 개념

### Domain vs Class
- **Domain**: 운전 조건 (RPM, Load) - 모델이 적응해야 하는 환경 변화
- **Class**: 고장 유형 (H/B/IR/OR/L/U/M) - 모델이 분류하는 대상 (고정)

### Continual Learning 목표
- **같은 클래스를 계속 분류**하되, **변화하는 환경에 적응**
- **이전 도메인 지식 보존**하면서 **새 도메인 학습**

---

