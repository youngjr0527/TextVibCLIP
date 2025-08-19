# TextVibCLIP: When Vibration Meets Text - Multimodal Continual Learning for Bearing Fault Diagnosis

---

## 🎯 프로젝트 개요

**TextVibCLIP**은 베어링 고장 진단을 위한 혁신적인 멀티모달 continual learning framework이다. 진동 신호와 텍스트 metadata를 결합하여 CLIP-inspired contrastive learning을 적용, domain shift 문제를 해결한다. Joint training으로 text encoder와 vibration encoder를 InfoNCE loss로 학습하며, asymmetric adaptation 전략을 통해 continual domain learning을 구현한다.

### 🔑 핵심 아이디어
- **진동 신호** + **텍스트 metadata** → **joint multimodal training** via bidirectional InfoNCE
- **Contrastive learning**으로 unified embedding space 형성
- **Continual learning** with asymmetric temperature (text minimal update, vibration focused adaptation)
- **Vibration encoder**: Time Series Transformer (TST) 사용 – vibration signals의 temporal patterns capture

### 🎯 실험 시나리오
- **시나리오 1 (UOS)**: Varying Speed - RPM 변화에 따른 continual learning (600→800→1000→1200→1400→1600 RPM)
- **시나리오 2 (CWRU)**: Varying Load - 부하 변화에 따른 continual learning (0→1→2→3 HP)

```
진동 신호: [복잡한 파형 데이터] 
    ↓ VibrationEncoder (TST)
[512차원 벡터]
    ↓ TextVibCLIP (bidirectional InfoNCE)
[공통 임베딩 공간] ← 같은 공간!
    ↑ TextVibCLIP (bidirectional InfoNCE)  
[512차원 벡터]
    ↑ TextEncoder
텍스트: "A deep groove ball bearing operating at 600 rpm with ball fault."
```

---

## 🔬 연구 배경

### 문제 상황
1. **베어링 고장 진단**의 중요성
   - 산업 설비의 핵심 부품
   - 조기 발견 시 막대한 비용 절감 가능

2. **기존 방법의 한계**
   - 단일 도메인에 특화 (특정 베어링 타입, 고정 회전체 상태)
   - Domain shift로 인한한 성능 저하
   - 텍스트 metadata 활용 부족, unimodal approaches의 generalization 부족

### 해결 방안
- **Multimodal contrastive learning**: Vibration + text joint training
- **Asymmetric continual adaptation**: Temperature scheduling으로 modality별 gradient control
- **Replay mechanism**: Catastrophic forgetting mitigation

---

## 🏗️ 시스템 아키텍처


### 모델 구조도
```
Input:
┌─────────────────┐    ┌─────────────────┐
│   진동 신호       │    │   텍스트 설명     │
│ [1600 samples]  │    │ "A ball bearing │
└─────────────────┘    │  with fault..." │
         │              └─────────────────┘
         ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│ VibrationEncoder│    │   TextEncoder   │
│   (TST-based)    │    │  (DistilBERT)   │
└─────────────────┘    └─────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│   [512-dim]     │    │   [512-dim]     │
│  진동 임베딩      │    │  텍스트 임베딩    │
└─────────────────┘    └─────────────────┘
         │                        │
         └────────────┬─────────────┘
                    ▼
         ┌─────────────────┐
         │   TextVibCLIP   │
         │ (InfoNCE joint) │
         └─────────────────┘
                    │
                    ▼
         ┌─────────────────┐
         │  공통 임베딩 공간  │
         │   [512-dim]     │
         └─────────────────┘
```

---

## 📁 파일 구조

### 핵심 파일들
```
TextVibCLIP/
├── 📄 main.py                           # 메인 실험 실행 스크립트
├── 📁 src/                              # 소스 코드
│   ├── textvib_model.py                 # TextVibCLIP 메인 모델
│   ├── continual_trainer.py            # Continual Learning 트레이너
│   ├── data_loader.py                   # 데이터 로더 (UOS/CWRU 지원)
│   ├── text_encoder.py                  # DistilBERT + LoRA 텍스트 인코더
│   ├── vibration_encoder.py             # TST 기반 진동 인코더
│   └── utils.py                         # 유틸리티 함수들
├── 📁 configs/                          # 설정 파일들
│   └── model_config.py                  # 모델 및 데이터 설정
├── 📁 uos_data/                         # UOS 원본 데이터셋
├── 📁 cwru_data/                        # CWRU 원본 데이터셋  
├── 📁 data_scenario1/                   # 시나리오 1 데이터 (UOS, Varying Speed)
├── 📁 data_scenario2/                   # 시나리오 2 데이터 (CWRU, Varying Load)
├── 📄 prepare_uos_scenario1.py          # UOS 데이터 전처리 스크립트
├── 📄 prepare_cwru_scenario2.py         # CWRU 데이터 전처리 스크립트
└── 📁 checkpoints/                      # 체크포인트 저장소
```

#### 주요 컴포넌트 ⭐
- **TextVibCLIP Model**: Text와 vibration encoders를 InfoNCE로 joint 학습, multimodal contrastive & continual adaptation 구현
- **Continual Trainer**: Domain별 순차 학습 및 성능 평가 관리  
- **BearingDataset**: UOS/CWRU 통합 지원, 윈도잉 기반 샘플 생성

---

## 🔧 기술적 세부사항

### InfoNCE 손실 함수
Bidirectional form 구현:

```python
# InfoNCE
InfoNCE = 1/(2N) * Σ[i=1 to N] [
    -log(exp(<z_text^i, z_vib^i>/τ_text) / Σ_j exp(<z_text^i, z_vib^j>/τ_text)) +
    -log(exp(<z_vib^i, z_text^i>/τ_vib) / Σ_j exp(<z_vib^i, z_text^j>/τ_vib))
]
```

### Temperature 매개변수의 역할
- **Domain 1**: τ_text = τ_vib = 0.07 (balanced alignment).
- **Domain 2~N**: τ_text = 0.1 (text minimal adapt), τ_vib = 0.05 (vibration focused).

### 데이터 구조

#### 시나리오 1: UOS 데이터셋 (Varying Speed)
**파일명 규칙:**
```
예시: H_B_30204_600.mat
├── H: 회전체 상태 (H=Healthy, L=Looseness, U=Unbalance, M=Misalignment)
├── B: 베어링 상태 (H=Healthy, B=Ball fault, IR=Inner race, OR=Outer race)  
├── 30204: 베어링 타입 (6204=Deep Groove Ball, 30204=Tapered Roller, N204/NJ204=Cylindrical Roller)
└── 600: 회전 속도 (600 RPM)
```

**텍스트 생성 예시:**
```python
# 입력: H_B_30204_600.mat
# 출력: "A tapered roller bearing operating at 600 rpm with healthy rotating component and ball fault."
```

**데이터 전처리:**
```bash
python prepare_uos_scenario1.py
```
1. UOS 데이터셋에서 16kHz 데이터만 필터링
2. 단일 결함만 선별 (복합결함 제외)  
3. U3→U, M3→M으로 relabel하여 라벨 균형 확보

#### 시나리오 2: CWRU 데이터셋 (Varying Load)
**파일명 규칙:**
```
예시: B_0hp_1.mat
├── B: 베어링 상태 (Normal=정상, B=Ball fault, IR=Inner race, OR=Outer race)
├── 0hp: 부하 (0, 1, 2, 3 horsepower)
└── 1: 파일 순번
```

**텍스트 생성 예시:**
```python
# 입력: B_0hp_1.mat
# 출력: "A deep groove ball bearing operating under 0 horsepower load with ball fault."
```

**데이터 전처리:**
```bash
python prepare_cwru_scenario2.py
```
1. Drive End 12kHz 데이터만 사용 (Fan End 제외)
2. Fault size 무시하고 라벨 균형 맞춤
3. Load별 도메인 구성: 0hp → 1hp → 2hp → 3hp

---
### LoRA 전략
### **Text Encoder: LoRA Only**

```python
text_encoder:
  - Domain 1 (600 RPM): LoRA fine-tuning (베어링 knowledge 학습)
  - Domain 2+ (800~1600 RPM): FREEZE (semantic knowledge는 도메인간 불변)
  - 이유: 베어링 fault types (H/B/IR/OR)는 RPM 바뀌어도 동일
```

### **Vibration Encoder: Full Training + Replay**

```python
vibration_encoder:
  - Domain 1 (600 RPM): Full training (scratch부터)
  - Domain 2+ (800~1600 RPM): Full fine-tuning + Replay
  - 이유: 진동 패턴 학습이 핵심, 최대 표현력 필요
```
| Component | 역할 | 방법 | 이유 |
| --- | --- | --- | --- |
| **Text Encoder** | Semantic guidance | LoRA only | 한번 학습하면 충분 |
| **Vibration Encoder** | Domain adaptation | Full + Replay | Continual learning 핵심 |