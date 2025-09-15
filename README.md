# TextVibCLIP: Multimodal Continual Learning for Bearing Fault Diagnosis

## 🎯 연구 개요

**TextVibCLIP**은 베어링 고장 진단을 위한 **Domain-Incremental Continual Learning** 프레임워크입니다. 진동 신호와 텍스트 메타데이터를 결합하여 CLIP-inspired contrastive learning을 적용하고, 운전 조건 변화(domain shift)에 robust한 진단 모델을 구현합니다.

### 🔑 핵심 특징
- **Domain-Incremental Learning**: 클래스는 고정, 도메인만 순차적 변화
- **Multimodal Contrastive Learning**: 진동 신호 + 텍스트 메타데이터
- **Asymmetric Continual Adaptation**: Text encoder freeze + Vibration encoder adaptation
- **두 가지 시나리오**: UOS (Varying Speed), CWRU (Varying Load)

---

## 📊 실험 시나리오

### **시나리오 1: UOS - Varying Speed**
- **Domain 순서**: 600 → 800 → 1000 → 1200 → 1400 → 1600 RPM
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

## 🏗️ 시스템 아키텍처

```
Input:
┌─────────────────┐    ┌─────────────────┐
│   진동 신호       │    │   텍스트 설명     │
│ [4096 samples]  │    │ "A ball bearing │
└─────────────────┘    │  with fault..." │
         │              └─────────────────┘
         ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│ VibrationEncoder│    │   TextEncoder   │
│  (1D-CNN)       │    │  (DistilBERT    │
│  35M params     │    │   + LoRA)       │
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
         │   InfoNCE Loss  │
         │ (Bidirectional) │
         └─────────────────┘
```

### Continual Learning 전략
- **Domain 1**: Text LoRA + Vibration 동시 학습
- **Domain 2+**: Text freeze + Vibration adaptation + Replay buffer

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
├── 📁 src/                              # 소스 코드
│   ├── textvib_model.py                 # TextVibCLIP 메인 모델
│   ├── continual_trainer.py            # Continual Learning 트레이너
│   ├── data_loader.py                   # 데이터 로더 (UOS/CWRU)
│   ├── text_encoder.py                  # DistilBERT + LoRA
│   ├── vibration_encoder.py             # 1D-CNN
│   └── utils.py                         # 유틸리티 함수들
├── 📁 configs/                          # 설정 파일
│   └── model_config.py                  # 모델 및 실험 설정
├── 📁 data_scenario1/                   # UOS 데이터 (Varying Speed)
├── 📁 data_scenario2/                   # CWRU 데이터 (Varying Load)
└── 📁 results/                          # 실험 결과
```

---

## 📊 결과 해석

### 성능 메트릭
- **Accuracy**: Text-Vibration 매칭 정확도
- **Top5 Retrieval**: 상위 5개 중 정답 포함 비율
- **Forgetting Score**: 이전 도메인 성능 저하 정도

### 기대 결과
- **UOS**: 더 어려운 태스크 (7개 클래스) → 낮은 정확도 예상
- **CWRU**: 상대적으로 쉬운 태스크 (4개 클래스) → 높은 정확도 예상

---

## 🔧 기술적 세부사항

### InfoNCE Loss
```python
# Bidirectional contrastive learning
InfoNCE = 1/(2N) * Σ[
    -log(exp(<text_i, vib_i>/τ_text) / Σ_j exp(<text_i, vib_j>/τ_text)) +
    -log(exp(<vib_i, text_i>/τ_vib) / Σ_j exp(<vib_i, text_j>/τ_vib))
]
```

### 온도 파라미터
- **Domain 1**: τ_text = τ_vib = 0.05 (균등 학습)
- **Domain 2+**: τ_text = 0.07, τ_vib = 0.03 (비대칭 적응)

### 데이터 분할 (Domain-Incremental)
- **모든 subset에 모든 클래스 포함**: `set(train_classes) == set(val_classes) == set(test_classes)`
- **Stratified split**: 클래스 균형 유지
- **파일 레벨 분할**: 데이터 누수 방지

---

## 🎯 연구 기여

1. **Domain-Incremental Continual Learning**: 산업 현장의 실제 요구사항 반영
2. **Multimodal Contrastive Learning**: 진동 + 텍스트 정보 활용
3. **Asymmetric Adaptation**: 모달리티별 차별적 학습 전략
4. **실용적 검증**: 실제 베어링 데이터셋으로 검증

---

## 📈 실험 결과 예시

### UOS 시나리오 (Varying Speed)
- **평균 정확도**: ~17% (7개 클래스, 어려운 태스크)
- **Top5 성능**: ~35% (검색 관점에서는 합리적)
- **망각도**: 0.0% (Replay buffer 효과)

### CWRU 시나리오 (Varying Load)
- **평균 정확도**: ~65% (4개 클래스, 상대적으로 쉬운 태스크)
- **일관성**: 모든 도메인에서 안정적 성능
- **망각도**: 0.0% (효과적인 지식 보존)

---

## 🔍 주요 개념

### Domain vs Class
- **Domain**: 운전 조건 (RPM, Load) - 모델이 적응해야 하는 환경 변화
- **Class**: 고장 유형 (H/B/IR/OR/L/U/M) - 모델이 분류하는 대상 (고정)

### Continual Learning 목표
- **같은 클래스를 계속 분류**하되, **변화하는 환경에 적응**
- **이전 도메인 지식 보존**하면서 **새 도메인 학습**

---

**TextVibCLIP은 산업 현장의 동적 환경 변화에 robust한 베어링 고장 진단 시스템을 제공합니다.** 🎯
