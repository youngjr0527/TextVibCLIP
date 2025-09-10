# TextVibCLIP 연구 가이드 (최신 업데이트)

## 🎯 **최신 분류 체계 (2025-09-10 업데이트)**

### **UOS 데이터셋: 7-클래스 시스템**
```
🎯 완전한 고장 진단 분류:

H (Healthy):      H_H  - 회전체 정상 + 베어링 정상 (완전 건강)
B (Ball Fault):   H_B  - 회전체 정상 + 베어링 볼 결함  
IR (Inner Race):  H_IR - 회전체 정상 + 베어링 내륜 결함
OR (Outer Race):  H_OR - 회전체 정상 + 베어링 외륜 결함
L (Looseness):    L_H  - 회전체 느슨함 + 베어링 정상
U (Unbalance):    U_H  - 회전체 불균형 + 베어링 정상  
M (Misalignment): M_H  - 회전체 정렬불량 + 베어링 정상
```

**연구 의의**: 베어링 결함과 회전체 결함을 분리하여 더 정밀한 진단 가능

### **CWRU 데이터셋: 4-클래스 시스템**
```
Normal: 완전 정상
B:      볼 결함
IR:     내륜 결함  
OR:     외륜 결함
```

## 🔧 **데이터 분할 전략 (윈도우 레벨)**

### **UOS**: 시간적 분할
```
각 파일 (예: H_H_6204_600.mat → 624개 윈도우):
├── Train: 윈도우 0-374 (60%)     - 모든 7클래스 포함
├── Val:   윈도우 375-499 (20%)   - 모든 7클래스 포함  
└── Test:  윈도우 500-623 (20%)   - 모든 7클래스 포함
```

### **CWRU**: 시간적 분할
```
각 파일 (예: Normal_0hp.mat → 58개 윈도우):
├── Train: 윈도우 0-40 (70%)      - 모든 4클래스 포함
├── Val:   윈도우 41-49 (15%)     - 모든 4클래스 포함
└── Test:  윈도우 50-57 (15%)     - 모든 4클래스 포함  
```

**장점**: 
- ✅ Data leakage 방지
- ✅ 모든 클래스가 모든 subset에 포함
- ✅ 시계열 특성 고려한 자연스러운 분할

## 🚀 **개선된 아키텍처**

### **모델 구성**
```python
TextVibCLIP:
├── Text Encoder: DistilBERT + LoRA (589K 파라미터)
├── Vibration Encoder: 5-layer 1D-CNN (35M 파라미터)
├── Embedding Dim: 256 (효율성 개선)
├── InfoNCE Loss: τ_text=0.07, τ_vib=0.07
└── No LayerNorm (gradient flow 보존)
```

### **Continual Learning 전략**
```python
Domain 1 (600 RPM / 0 HP):
├── Text LoRA + Vibration 동시 학습
├── 7-클래스 또는 4-클래스 분류 학습
└── Alignment 시각화 생성

Domain 2+ (800+ RPM / 1+ HP):  
├── Text freeze + Vibration adaptation
├── Replay buffer 활용
├── 도메인별 성능 시각화 생성
└── Forgetting score 추적
```

## 📊 **성능 평가 시스템**

### **표준 Contrastive Learning 평가**
```python
# 각 텍스트가 자신의 진동 쌍을 정확히 찾는지 평가
similarity_matrix = text_emb @ vib_emb.T
predicted = torch.argmax(similarity_matrix, dim=1)  
target = torch.arange(batch_size)
accuracy = (predicted == target).float().mean()
```

**의미**: 텍스트 설명과 진동 신호의 정확한 매칭 능력 측정

### **Continual Learning 메트릭**
- **Average Accuracy**: 모든 도메인 평균 성능
- **Forgetting Score**: 이전 도메인 성능 저하 정도
- **Top-1/Top-5 Retrieval**: 검색 성능

## 🎨 **실시간 시각화**

### **First Domain 완료 시**
- **Alignment Plot**: Text-Vibration t-SNE 시각화
- **성능 검증**: 80% 기준 없이 연속 실행

### **각 Domain 완료 시**  
- **Performance Plot**: 도메인별 정확도 변화
- **Forgetting Plot**: 망각 점수 추적
- **실시간 모니터링**: 학습 상태 즉시 확인

## 🚀 **실행 명령어**

```bash
# 전체 실험 (권장)
python run_all_scenarios.py --output_dir results

# 빠른 테스트
python run_all_scenarios.py --quick_test --epochs 3 --output_dir test

# UOS만 실행  
python run_all_scenarios.py --skip_cwru --output_dir uos_only

# CWRU만 실행
python run_all_scenarios.py --skip_uos --output_dir cwru_only
```

## 🎯 **연구 기여도**

### **기술적 혁신**
1. **7-클래스 UOS**: 베어링 + 회전체 결합 진단
2. **윈도우 레벨 분할**: 시계열 특성 고려한 올바른 분할
3. **실시간 시각화**: 학습 과정 투명성 확보
4. **표준 평가**: 정확한 contrastive learning 성능 측정

### **실용적 가치**  
- **더 정밀한 진단**: 7가지 고장 모드 구분
- **실제 산업 적용**: 다양한 운전 조건 대응
- **연구 재현성**: 명확한 실험 프로토콜

---

> **"이제 진정한 의미의 multimodal continual learning 실험이 가능합니다!"** 🚀
