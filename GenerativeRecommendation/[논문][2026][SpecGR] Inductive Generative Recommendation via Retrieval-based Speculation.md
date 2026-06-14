# Inductive Generative Recommendation via Retrieval-based Speculation (SpecGR)

저자: Yijie Ding, Jiacheng Li, Julian McAuley, Yupeng Hou

출간: AAAI 2026 (Oral)

논문: [arXiv 2410.02939](https://arxiv.org/abs/2410.02939)

---

## 0. Abstract

* 생성형 추천(Generative Recommendation, GR) 모델은 아이템을 토큰화하고 자기회귀 생성을 학습하나, **학습 데이터에 등장한 아이템만 생성** 가능 → 새로운 아이템(unseen items) 추천 불가
* **SpecGR**: 검색 기반 추론(Retrieval-based Speculation) 프레임워크를 통해 기존 GR 모델에 **귀납적 추천(Inductive Recommendation)** 능력을 부여
  - 드래프터(Drafter): 귀납적 후보 아이템(신규 아이템 포함) 제안
  - 검증기(Verifier, GR 모델): 후보 수용/거부로 랭킹 품질 유지
* 3개 실제 데이터셋에서 귀납적 추천 및 전체 추천 성능 모두에서 SOTA 달성
* **AAAI 2026 Oral** 발표

---

## 1. Introduction

**생성형 추천의 귀납적 한계**

TIGER 등 생성형 추천 모델이 인기를 얻었지만, 근본적 한계 존재:

* 아이템은 학습 전 **Semantic ID로 인덱싱** → 학습 중 보지 못한 아이템은 ID가 없음
* 실서비스에서는 매일 새로운 아이템이 추가됨 (신상품, 신규 게시물 등)
* 기존 GR 모델은 이러한 **귀납적 설정(Inductive Setting)** 에서 완전히 실패

**트랜스덕티브(Transductive) vs. 귀납적(Inductive) 추천**

| 구분 | 설명 | 예시 |
|---|---|---|
| Transductive | 학습 시 모든 아이템이 알려져 있음 | 기존 CF, GR 모델 |
| Inductive | 추론 시 새로운 아이템 존재 | 신상품 추천, 콜드스타트 |

**SpecGR의 아이디어**: Speculative Decoding의 Draft-Verify 패러다임을 추천에 적용
- 드래프터가 신규 아이템 포함 후보를 제안
- 기존 GR 모델(Verifier)이 품질을 보장하며 수용/거부

---

## 2. Related Work

* **생성형 추천**: TIGER, GPT4Rec, P5 — 학습 아이템에 한정된 트랜스덕티브 방식
* **귀납적 추천**: ZESREC, UniSRec, VQ-Rec — 콘텐츠 특징 기반, 신규 아이템 일반화 가능하나 랭킹 품질이 GR 대비 낮음
* **Speculative Decoding (LLM 분야)**: Draft & Verify 패러다임으로 LLM 추론 가속
  - SpecGR은 이 패러다임을 추천의 **품질 + 귀납성 트레이드오프** 해결에 창의적으로 적용
* **콜드스타트 추천**: 신규 아이템/사용자 처리 — 전통적으로 콘텐츠 기반 접근

---

## 3. SpecGR 방법론

### 3.1 전체 프레임워크

```
사용자 히스토리
      |
      v
[드래프터 (귀납적 검색기)]
  - 알려진 아이템 + 신규 아이템 후보 집합 C 생성
      |
      v
[Guided Re-drafting]
  - 드래프트를 GR 모델 출력에 근접하도록 재정렬
      |
      v
[검증기 (GR 모델, Verifier)]
  - 후보 수용/거부, GR의 랭킹 능력 활용
      |
      v
최종 추천 (신규 아이템 포함)
```

### 3.2 드래프터 (Drafter)

**역할**: 사용자 선호에 맞는 후보 아이템을 제안 — 신규 아이템 포함 가능

**두 가지 구현 변형**:

① **보조 드래프터 (Auxiliary Drafter)**
- 별도의 귀납적 추천 모델 사용 (예: ZESREC, UniSRec)
- 콘텐츠 임베딩 기반으로 신규 아이템 포함 후보 생성
- 유연성 높음, 다양한 귀납적 검색기 적용 가능

② **자기 드래프팅 (Self-Drafting)**
- GR 모델 자체의 인코더를 드래프터로 활용
- 별도 모델 불필요 → 파라미터 효율적
- GR 인코더의 표현 능력을 활용한 귀납적 검색

### 3.3 Guided Re-drafting

**문제**: 드래프터와 GR 모델의 후보 분포가 다를 경우 검증 효율 저하 (수용률 낮음)

**해결**: 드래프트를 GR 모델의 출력 분포에 맞게 재정렬

$$C^* = \arg\max_{C \subseteq \mathcal{C}} \text{sim}(P_{\text{GR}}(\cdot \mid \text{history}), C)$$

* GR 모델이 높은 확률을 부여하는 아이템을 드래프트에서 우선시
* 드래프터-검증기 간 분포 불일치 감소 → 검증 효율 향상

### 3.4 검증기 (Verifier, GR 모델)

* 기존 GR 모델 (TIGER 등) **변경 없이** Verifier로 활용 → 플러그인 방식
* 드래프트된 후보 $C$를 입력으로 받아 각 아이템의 수용/거부 결정
* GR 모델의 강점(학습 아이템에서의 정밀한 랭킹)을 그대로 활용
* 최종적으로 신규 아이템과 기존 아이템이 혼합된 추천 리스트 생성

---

## 4. Experimental Setup

### 데이터셋

3개의 실제 데이터셋 사용 (논문에서 공개 벤치마크 기반):
- 각 데이터셋에서 일부 아이템을 **귀납적 테스트 셋**으로 지정 (학습 중 미공개)
- 평가: 귀납적 추천 성능 + 전체 추천 성능

### 평가 지표
* Recall@K, NDCG@K (K=5, 10, 20)
* **귀납적 지표**: 테스트 시 새로운 아이템에 대한 Recall@K, NDCG@K

### 베이스라인

| 종류 | 방법 |
|---|---|
| 전통 귀납적 | ZESREC, UniSRec, VQ-Rec |
| 생성형 추천 | TIGER, GenRec 계열 |
| 혼합 방식 | 귀납적 모델 + GR 재랭킹 |

---

## 5. Experiments

### 귀납적 추천 성능

* SpecGR은 순수 귀납적 방법 대비 신규 아이템 추천에서 우수
* 기존 GR 모델(귀납적 설정에서 0에 가까운 성능) 대비 압도적 향상

### 전체 추천 성능

* 신규 아이템 + 기존 아이템 혼합 평가에서도 베이스라인 대비 최고 성능
* GR 모델의 랭킹 강점을 검증기로 유지하면서 귀납성 추가 → 두 목표 동시 달성

### Guided Re-drafting 효과 (Ablation)

| 설정 | 귀납적 Recall@10 | 전체 Recall@10 |
|---|---|---|
| SpecGR (전체) | 최고 | 최고 |
| Re-drafting 없음 | 하락 | 하락 |
| Self-drafting 사용 | 파라미터 효율 ↑ | Aux 대비 소폭 하락 |

**핵심 발견**: Guided Re-drafting이 드래프터-검증기 분포 불일치를 줄여 검증 효율 향상

### 드래프터 변형 비교

* 보조 드래프터: 귀납적 성능 최고
* 자기 드래프팅: 파라미터 추가 없이 유사 성능 → 실용적 배포에 유리

---

## 의의

#### 1. 생성형 추천의 귀납적 한계 최초 해결
* 기존 GR 모델의 근본적 한계(학습 아이템에 갇힌 트랜스덕티브 특성)를 플러그인 방식으로 극복
* 기존 GR 모델 **재학습 없이** 귀납적 능력 추가 — 실서비스 도입 용이

#### 2. Speculative Decoding 패러다임의 추천 적용
* LLM 추론 가속 기법을 추천의 "품질-귀납성 트레이드오프" 해결에 창의적으로 전용
* Draft-Verify 패러다임이 추론 가속뿐 아니라 능력 확장에도 유효함을 실증

#### 3. 실용적인 플러그인 설계
* 어떤 기존 GR 모델에도 Verifier로 적용 가능한 범용 프레임워크
* 자기 드래프팅 변형으로 추가 파라미터 없이 귀납성 달성 가능
