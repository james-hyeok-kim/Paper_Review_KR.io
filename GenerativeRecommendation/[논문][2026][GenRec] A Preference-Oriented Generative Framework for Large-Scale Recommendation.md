# GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation

저자: Yanyan Zou, Junbo Qi, Lunsong Huang, Yu Li, Kewei Xu, Jiabao Gao, Binglei Zhao, Xuanhua Yang, Sulong Xu, Shengjie Li (JD.com)

출간: SIGIR 2026

논문: [arXiv 2604.14878](https://arxiv.org/abs/2604.14878)

---

## 0. Abstract

* **JD App 실서비스**에 배포된 대규모 생성형 추천 프레임워크 GenRec 제안
* 기존 생성형 추천 시스템의 세 가지 실용적 한계를 동시에 해결:
  1. 페이지네이션 메커니즘으로 인한 **출력 불일치(inconsistency)** 문제
  2. 긴 사용자 행동 시퀀스와 다중 토큰 Semantic ID 인코딩에 따른 **계산 비용** 문제
  3. 사용자 만족 신호와 생성 결과 간의 **선호도 정렬(preference alignment)** 문제
* 한 달간 A/B 테스트에서 클릭 수 **+9.5%**, 거래 수 **+8.7%** 달성

---

## 1. Introduction

**생성형 추천의 산업 배포 문제**

TIGER 등 생성형 추천 연구는 학술 벤치마크에서 뛰어난 성능을 보이지만, 수억 명의 사용자를 서빙하는 실서비스에 적용하면 다음 문제들이 발생:

1. **출력 불일치**: 실서비스는 페이지 단위(pagination)로 추천을 요청 — 동일한 모델 입력에서도 요청 시점마다 다른 결과가 생성될 수 있어 사용자 경험 저하
2. **계산 비용**: 대규모 아이템 카탈로그에서 긴 사용자 히스토리를 다중 토큰 Semantic ID로 인코딩하면 Prefilling 비용 폭증
3. **선호도 불일치**: 자기회귀 생성은 다음 토큰 예측 손실로만 학습되어, 실제 사용자 클릭/구매 등 선호 신호와 괴리 발생

**GenRec의 접근**: 세 가지 독립적 모듈(Page-wise NTP, Asymmetric Token Merger, GRPO-SR)로 각 문제를 해결

---

## 2. Related Work

* **생성형 검색**: TIGER, P5, UniSRec — Semantic ID 기반 seq2seq 추천
* **RLHF / 선호도 정렬**: InstructGPT, DPO — LLM의 인간 피드백 정렬을 추천에 응용
* **지식 증류 및 토큰 압축**: 대형 모델 추론 최적화 기법을 추천 시스템 입력 압축에 응용
* **산업용 추천 시스템**: DIN, BST 등 — 대규모 실시간 추천, GenRec은 이들과 달리 생성형 접근

---

## 3. GenRec 방법론

### 3.1 Page-wise Next-Token Prediction (Page-wise NTP)

**문제**: 기존 아이템 단위 NTP는 **One-to-Many 모호성** 발생
- 동일 사용자 히스토리에서 여러 아이템이 "정답"일 수 있음
- 페이지 단위 추천 시스템에서는 요청마다 다른 아이템이 노출되어 일관성 저하

**해결**: 아이템 단위가 아닌 **페이지(노출 묶음) 단위**로 학습 목표 설정

$$\mathcal{L}_{\text{NTP}} = -\sum_{p} \sum_{i \in \text{page}_p} \log P(\text{Semantic ID}(i) \mid \text{history}, \text{page context})$$

* 더 밀도 높은 기울기 신호(denser gradient signal) 제공
* 같은 페이지 내 아이템 간 관계를 모델이 파악하여 추천 일관성 향상

### 3.2 Asymmetric Linear Token Merger (ALTM)

**문제**: 다중 토큰 Semantic ID (예: 아이템당 3~6개 토큰)로 인한 긴 입력 시퀀스 → Prefilling 비용 ↑

**해결**: Prefilling 단계에서만 다중 토큰을 단일 토큰으로 압축 (Decoding 단계는 원래 해상도 유지)

$$\tilde{e}_i = W_{\text{merge}} \cdot [e_i^{(1)}; e_i^{(2)}; \ldots; e_i^{(D)}]$$

* **비대칭(Asymmetric)**: 입력 압축 ↔ 출력 전개 비율이 다름
* 약 **2×** 입력 길이 감소 → Prefilling 속도 대폭 향상
* 추천 품질 저하 없이 처리량(throughput) 개선

### 3.3 GRPO-SR (Preference Alignment via RL)

**문제**: MLE 기반 학습은 사용자 클릭/구매 선호 신호를 직접 최적화하지 못함

**해결**: Group Relative Policy Optimization (GRPO)를 추천에 맞게 적용

**학습 목표**:
$$\mathcal{L}_{\text{GRPO-SR}} = \mathcal{L}_{\text{RL}} + \lambda \mathcal{L}_{\text{NLL}}$$

* $\mathcal{L}_{\text{RL}}$: GRPO 기반 강화학습 (그룹 상대적 보상)
* $\mathcal{L}_{\text{NLL}}$: NLL 정규화로 학습 안정성 확보 (정책 붕괴 방지)

**보상 설계 (Hybrid Reward)**:
- **Dense Reward Model**: 생성된 추천의 예측 만족도 점수 제공
- **Relevance Gate**: 관련 없는 아이템에 보상 주는 "보상 해킹(reward hacking)" 방지

$$r(i) = r_{\text{dense}}(i) \cdot \mathbf{1}[\text{relevant}(i)]$$

---

## 4. Experimental Setup

### 배포 환경
* **플랫폼**: JD App (징둥닷컴 모바일 앱)
* **규모**: 수억 명의 일일 활성 사용자 (DAU), 수억 개 아이템 카탈로그
* **평가**: 한 달간 온라인 A/B 테스트 (실제 트래픽 분할)

### 평가 지표 (온라인)
* 클릭 수 (Click Count)
* 거래 수 (Transaction Count)

### 오프라인 평가
* 내부 JD 데이터셋 사용, Recall@K / NDCG@K
* 비교: 기존 생성형 추천 베이스라인(TIGER 류), 기존 JD 서비스 모델

---

## 5. Experiments

### 온라인 A/B 테스트 결과

| 지표 | 개선율 |
|---|---|
| 클릭 수 (Click Count) | **+9.5%** |
| 거래 수 (Transaction Count) | **+8.7%** |

* 통계적으로 유의미한 개선 (p < 0.05)

### 각 컴포넌트 기여도 (Ablation)

| 설정 | 성능 |
|---|---|
| Baseline (기존 생성형 추천) | 기준 |
| + Page-wise NTP | 일관성 개선, 클릭 ↑ |
| + ALTM | 처리량 ~2× 향상, 품질 유지 |
| + GRPO-SR | 선호도 정렬 개선, 추가 성능 ↑ |
| 전체 GenRec | 최고 성능 |

### 계산 효율성

* ALTM 적용 시 Prefilling 지연(latency) 약 50% 감소
* 동일 서버 자원으로 2배 처리량 달성

---

## 의의

#### 1. 생성형 추천의 실서비스 배포 가능성 실증
* 수억 명 규모의 실제 사용자에게 생성형 추천이 작동함을 A/B 테스트로 증명
* 학술 벤치마크를 넘어 실산업 배포 가능성을 보인 첫 대규모 사례

#### 2. 산업적 세 가지 핵심 문제 동시 해결
* 출력 일관성, 계산 효율, 선호도 정렬을 각각 전용 모듈로 해결
* 각 컴포넌트가 독립적으로 기존 시스템에 플러그인 가능

#### 3. RLHF의 추천 시스템 적용
* LLM 분야의 GRPO를 추천 특성(페이지 단위 보상, 관련성 게이트)에 맞게 재설계
* 보상 해킹 방지 메커니즘을 추천 도메인에 최초 적용
