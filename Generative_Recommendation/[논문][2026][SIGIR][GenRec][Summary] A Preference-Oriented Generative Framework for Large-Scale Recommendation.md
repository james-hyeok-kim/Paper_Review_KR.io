# A Preference-Oriented Generative Framework for Large-Scale Recommendation

저자 :

Junyi Chen, Chao Bao, Jiajie Xu, Hanbo Li, Rui Liu, Hao Wang

Alibaba Group

발표 : SIGIR 2026

논문 : [PDF](https://arxiv.org/pdf/2604.14878)

출처 : [https://arxiv.org/abs/2604.14878](https://arxiv.org/abs/2604.14878)

---

## 0. Summary

### 0.1. 문제 (Problem)

* 대규모 산업 추천 시스템에서 생성형 추천(Generative Recommendation)을 배포하려면 세 가지 근본적인 도전을 동시에 해결해야 한다.

**① One-to-many 모호성 (One-to-many Ambiguity)**
- 기존 생성형 추천은 자기회귀적(autoregressive) Next Token Prediction(NTP) 방식으로 훈련하며, 학습 데이터에서 사용자가 클릭한 아이템을 "정답 토큰" 시퀀스로 예측한다.
- 그런데 추천에서는 동일한 사용자·맥락에 대해 여러 아이템이 동등하게 정답일 수 있다(one-to-many 문제). 개별 아이템 단위로 NTP 손실을 계산하면, 사용자가 선호할 다른 아이템들의 신호가 상쇄되어 학습 신호가 모호해진다.

**② 시퀀스 길이 비효율 (Sequence Length Inefficiency)**
- 아이템 ID를 여러 토큰(sub-token)으로 표현하는 방식(예: RQ-VAE의 Semantic ID)은 하나의 아이템을 3~4개 토큰으로 인코딩하므로, 사용자 이력 길이 L인 경우 입력 시퀀스 길이가 3L~4L로 팽창한다.
- Transformer의 self-attention 연산량은 시퀀스 길이의 제곱에 비례하므로, 산업 환경의 긴 이력(L>100)에서는 계산 비용이 폭증한다.

**③ 사용자 선호 정렬 부재 (Preference Alignment Gap)**
- 자기회귀 NTP 훈련은 클릭된 아이템을 모방하도록 모델을 최적화하지만, 클릭 데이터에는 노이즈(광고 클릭, 우연한 클릭)와 암묵적 편향이 포함된다.
- 실제 사용자 만족도(선호도)와 NTP 손실 최저화가 반드시 일치하지 않아, 훈련된 모델이 진짜 사용자 선호를 반영하지 못하는 정렬 격차가 발생한다.

### 0.2. 핵심 아이디어 (Core Idea)

GenRec는 세 가지 독립적이고 호환 가능한 모듈을 통해 위의 문제를 각각 해결한다.

**① Page-wise NTP — 페이지 단위 차세대 아이템 예측**

기존 NTP가 아이템 단위로 다음 토큰을 예측하는 반면, Page-wise NTP는 페이지(page) 개념을 도입한다. 추천 시스템은 사용자에게 아이템을 한 번에 하나씩이 아닌 페이지(K개 묶음)로 제공한다. 따라서 "다음에 클릭할 아이템" 대신 "다음 페이지에서 선호할 아이템 집합"을 예측 단위로 삼는다.

```
[User History: i1, i2, ..., iN] → [Next Page: {j1, j2, j3, ..., jK}]
```

페이지 내 아이템 집합을 레이블로 사용하므로 one-to-many 모호성이 줄어들고, 같은 시퀀스에서 K배 더 많은 긍정 신호를 얻어 학습 효율이 높아진다. 또한 인상(impression) 데이터를 활용해 부정 샘플(클릭되지 않은 아이템)도 명시적으로 반영할 수 있다.

**② ALTM (Asymmetric Linear Token Merger) — 비대칭 선형 토큰 병합**

ALTM은 Transformer의 입력단에서 토큰을 동적으로 병합하여 시퀀스 길이를 줄이는 모듈이다.

```
Original: [t1_1, t1_2, t1_3, t2_1, t2_2, t2_3, ...]  (3 tokens per item)
ALTM:     [m1,              m2,              ...]       (1 token per item)
```

구체적으로:
- **입력(이력) 측**: 연속된 K개의 서브토큰을 선형 가중치 합(weighted sum)으로 하나의 병합 토큰 $m_i = \sum_{k=1}^{K} w_k t_{i,k}$로 압축한다. 가중치 $w_k$는 학습 가능하다.
- **출력(생성) 측**: 생성 시에는 토큰을 병합하지 않고 원래의 서브토큰 단위 자기회귀 생성을 유지한다(비대칭 구조).
- 결과적으로 시퀀스 길이가 K배(≈2–3×) 단축되어 self-attention 비용이 $K^2$배 감소한다.

이 비대칭 설계의 핵심: 이력 이해(이해 측)에서는 압축해도 의미가 보존되지만, 생성(출력 측)에서는 정확한 토큰 시퀀스를 생성해야 하므로 압축하지 않는다.

**③ GRPO-SR (Group Relative Policy Optimization for Sequential Recommendation) — 선호 정렬 강화학습**

GRPO-SR은 대형 언어 모델의 RLHF(Reinforcement Learning from Human Feedback) 패러다임을 추천에 적용한 모듈이다.

**보상 함수 설계**:
$$r(a, y) = r_{rel}(a, y) \cdot \mathbf{1}[\text{valid}(a)] + \lambda \cdot r_{format}(a)$$

- $r_{rel}$: 생성된 아이템이 사용자의 실제 선호(next-page 레이블)와 얼마나 관련되는지를 나타내는 관련성 보상. 클릭된 아이템이면 높은 점수, 인상만 된 아이템이면 낮은 점수, 무관 아이템이면 0.
- **Relevance Gate**: $\mathbf{1}[\text{valid}(a)]$는 생성된 아이템이 실제 카탈로그에 존재하는지 여부를 확인하는 게이트. 존재하지 않는 아이템(환각)에는 보상 0.
- $r_{format}$: 생성 포맷 준수 여부(올바른 Semantic ID 형식인지).

**GRPO 업데이트**:
$$\mathcal{L}_{GRPO} = -\mathbb{E}\left[\sum_t \min\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{ref}(a_t|s_t)} A_t, \text{clip}\left(\frac{\pi_\theta}{\pi_{ref}}, 1-\epsilon, 1+\epsilon\right) A_t\right)\right] + \beta \cdot KL(\pi_\theta \| \pi_{ref})$$

- 그룹 내 여러 샘플의 보상을 정규화하여 advantage $A_t$를 계산하므로, 별도의 보상 모델 없이 상대적 선호를 학습한다(Group Relative).
- KL divergence 항($\beta \cdot KL$)이 NTP 사전훈련 모델($\pi_{ref}$)에서 너무 멀어지지 않도록 정규화(NLL 정규화 역할).
- 하이브리드 보상(관련성 + 포맷)이 노이즈가 있는 클릭 데이터보다 실제 사용자 선호를 더 잘 반영한다.

### 0.3. 효과 (Effects)

* **Page-wise NTP**: one-to-many 모호성 해소, 페이지당 K배 풍부한 훈련 신호, 인상 기반 부정 샘플 활용 가능
* **ALTM**: 시퀀스 길이 약 2–3× 단축, self-attention FLOPs 4–9× 감소, 긴 이력 처리 가능
* **GRPO-SR**: 노이즈가 있는 클릭 신호를 넘어 실제 사용자 선호 정렬, 환각(hallucination) 억제(Relevance Gate), 별도 보상 모델 불필요

### 0.4. 결과 (Results)

* **오프라인 평가**: Amazon, Alibaba 내부 데이터셋에서 TIGER, SASRec 등 기존 SOTA 대비 Recall@K, NDCG@K에서 일관된 향상.
* **온라인 A/B 테스트** (Alibaba 대규모 서비스 배포):
  - ALTM 적용으로 추론 레이턴시 약 40% 감소 (시퀀스 길이 압축 효과)
  - GRPO-SR 적용으로 CTR(Click-Through Rate) +2.3%, GMV(Gross Merchandise Value) +1.8% 향상
  - Page-wise NTP + GRPO-SR 조합이 단독 모듈 대비 추가 향상 확인
* 세 모듈은 독립적으로 또는 조합하여 적용 가능하며, 기존 생성형 추천 프레임워크에 플러그인 방식으로 통합 가능

### 0.5. 상세 동작 방식 (How It Works)

**[사전훈련] Page-wise NTP**

```
입력: 사용자 이력 시퀀스 [i1_tokens, i2_tokens, ..., iN_tokens]
페이지 레이블: {j1, j2, ..., jK} (다음 페이지에서 상호작용한 아이템 집합)

손실: L_page = -1/K * Σ_{k=1}^{K} log P(j_k | history, j1,...,j_{k-1})
```

집합이므로 순서가 없다 → 소팅(빈도순, 랜덤 셔플)으로 다양한 순서를 학습하여 순서 편향 제거.

**[인퍼런스] ALTM 압축**

```
for each item i with sub-tokens [t_{i,1}, ..., t_{i,K}]:
    merged_token[i] = sum(w_k * t_{i,k} for k in range(K))
    
compressed_sequence = [merged_token[1], merged_token[2], ..., merged_token[N]]
# 길이: N (원래 N*K 대비 K배 압축)
```

**[파인튜닝] GRPO-SR**

```
1. 현재 정책 π_θ로 G개 샘플 생성: {a1, a2, ..., aG}
2. 각 샘플에 보상 계산: r_i = reward(a_i, next_page_label)
3. Advantage 계산: A_i = (r_i - mean(r)) / std(r)  ← Group Relative
4. PPO-clip 스타일 정책 업데이트 + KL 페널티
5. 반복
```

---

## 1. Introduction

### 기존 생성형 추천의 산업 배포 격차

생성형 추천 모델(TIGER, LC-Rec 등)은 학술 벤치마크에서 우수한 성능을 보이지만, 실제 수억 명의 사용자와 수천만 개의 아이템이 존재하는 산업 환경에 배포하기 위해서는 다음 세 가지 추가 요구사항을 충족해야 한다:

1. **신호 품질**: 산업 데이터는 학술 데이터보다 훨씬 노이즈가 많고 one-to-many 관계가 복잡하다.
2. **계산 효율**: 실시간 추천은 수십 밀리초 이내에 결과를 반환해야 하므로, 긴 이력을 효율적으로 처리하는 것이 필수다.
3. **선호 정렬**: 클릭률(CTR) 최적화만으로는 사용자의 장기적 만족(GMV, 재방문율)을 달성할 수 없다.

GenRec는 이 세 요구사항을 세 개의 독립 모듈로 분리 해결하는 모듈형 프레임워크를 제안한다.

### 기존 관련 연구의 한계

| 방법 | 한계 |
|------|------|
| TIGER, P5 등 기존 Generative Rec | One-to-many 미해결, 학술 데이터 위주 |
| RLHF for LLM | 별도 보상 모델 필요, 추천 특화 보상 미정의 |
| Token Merging (ToMe 등) | 이미지/텍스트 도메인, 추천 Semantic ID 구조 미고려 |

---

## 2. Method

### 2.1. Page-wise NTP 세부

추천 시스템의 사용자 세션:
$$U = [i_1, i_2, ..., i_N] \xrightarrow{\text{split}} \text{pages: } P_1 = \{i_1,...,i_K\},\ P_2 = \{i_{K+1},...,i_{2K}\},\ ...$$

각 페이지 $P_t$를 레이블로, 직전 페이지들의 이력을 입력으로 하는 조건부 확률을 최대화한다:
$$\mathcal{L}_{page} = -\frac{1}{|P_t|} \sum_{j \in P_t} \log P_\theta(j \mid i_1, ..., i_{t \cdot K - 1})$$

집합 $P_t$ 내 아이템의 순서는 등가이므로, 훈련 시 여러 순열(permutation)을 샘플링하여 순서 불변성을 높인다.

**인상 데이터 활용**:
$$r_{rel}(j) = \begin{cases} 1 & j \in \text{clicked items} \\ 0.2 & j \in \text{impressed but not clicked} \\ 0 & \text{otherwise} \end{cases}$$

### 2.2. ALTM 세부

아이템 $i$가 K개 서브토큰 $[e_{i,1}, ..., e_{i,K}] \in \mathbb{R}^{K \times d}$로 표현된다고 할 때:

$$m_i = \text{LayerNorm}\left(\sum_{k=1}^K w_k \cdot e_{i,k}\right), \quad w_k = \text{softmax}(\alpha_k),\ \alpha_k \in \mathbb{R}$$

$\alpha_k$는 글로벌 학습 파라미터로 모든 아이템에 동일하게 적용된다(경량). 병합 후 시퀀스:
$$[m_1, m_2, ..., m_N] \in \mathbb{R}^{N \times d} \quad \text{(vs 원래 } \mathbb{R}^{NK \times d}\text{)}$$

**비대칭성**: 디코더(생성 단계)에서는 병합하지 않고 서브토큰 단위로 자기회귀 생성을 수행한다. 이는 생성 정확도를 보장하기 위함이다.

### 2.3. GRPO-SR 세부

**단계별 학습**:
1. **Stage 1**: Page-wise NTP로 기반 모델 $\pi_{ref}$ 훈련
2. **Stage 2**: $\pi_{ref}$를 초기화로 GRPO-SR 파인튜닝

**보상 함수**:
$$r(a, y^+) = \text{Relevance}(a, y^+) \cdot \text{Gate}_{valid}(a)$$

- $y^+$: next-page 레이블 아이템 집합
- $\text{Gate}_{valid}(a)$: 생성된 Semantic ID가 실제 아이템 카탈로그에 존재하면 1, 아니면 0 (환각 억제)

**KL 정규화**:
- NLL 손실(Page-wise NTP)을 KL 항 대신 직접 추가하는 변형도 실험:
$$\mathcal{L}_{hybrid} = \mathcal{L}_{GRPO} + \gamma \cdot \mathcal{L}_{NLL}$$

---

## 3. Experiments

### 데이터셋

| 데이터셋 | 규모 | 출처 |
|---------|------|------|
| Amazon Product Reviews | 수백만 사용자·아이템 | 학술 벤치마크 |
| Alibaba Internal Dataset | 수억 사용자 (산업 규모) | Alibaba 내부 로그 |

### 비교 모델

| 모델 | 특징 |
|------|------|
| SASRec | Self-Attentive Sequential Recommendation |
| TIGER | RQ-VAE Semantic ID + T5 생성형 추천 |
| LC-Rec | 협업 필터링 VQ + LLaMA 정렬 튜닝 |
| GenRec (ours) | Page-wise NTP + ALTM + GRPO-SR |

### 주요 결과

**오프라인 (Recall@10, NDCG@10)**:

| 모델 | Amazon Recall@10 | Amazon NDCG@10 |
|------|-----------------|-----------------|
| SASRec | - | 기준선 |
| TIGER | +X% | +Y% |
| GenRec (full) | 최고 | 최고 |

**온라인 A/B (Alibaba)**:

| 모듈 | CTR 변화 | GMV 변화 | 레이턴시 |
|------|----------|----------|---------|
| ALTM only | 유지 | 유지 | -40% |
| GRPO-SR only | +1.5% | +1.2% | - |
| Page-NTP + GRPO-SR | +2.3% | +1.8% | - |
| Full GenRec | +2.5% | +2.1% | -38% |

### Ablation

| 변형 | 성능 |
|------|------|
| w/o Page-wise NTP (item-wise NTP) | ↓ (one-to-many 모호성) |
| w/o ALTM | ↓ (레이턴시 증가, 긴 이력 처리 불가) |
| w/o GRPO-SR (NTP만) | ↓ (선호 정렬 부재) |
| GRPO 보상 w/o Relevance Gate | ↓ (환각 증가) |
| Full GenRec | 최고 |

---

## 4. Conclusion

GenRec는 산업 규모 생성형 추천의 세 가지 핵심 병목(one-to-many 모호성, 시퀀스 길이 비효율, 선호 정렬 부재)을 세 개의 독립 모듈로 분리 해결하는 모듈형 프레임워크다.

**핵심 기여**:
1. **Page-wise NTP**: 아이템이 아닌 페이지 단위 예측으로 one-to-many 모호성 해소 및 학습 신호 밀도 향상
2. **ALTM**: 비대칭 선형 병합으로 입력 시퀀스 길이를 K배 단축, 산업 레이턴시 요구 충족
3. **GRPO-SR**: 보상 모델 없는 그룹 상대적 선호 정렬, Relevance Gate로 환각 억제

**한계 및 향후 과제**:
- 페이지 경계 정의(K 값)가 데이터 특성에 민감하여 하이퍼파라미터 튜닝 필요
- GRPO-SR의 그룹 샘플 생성 비용이 학습 시간을 증가시킴
- 다국어·멀티모달 아이템으로의 확장성 미검증

---

## Appendix

### A.1. 핵심 사전 개념

**① 자기회귀 언어 모델 (Autoregressive LM)**
토큰 시퀀스 $[t_1, t_2, ..., t_n]$을 학습할 때, 각 토큰을 이전 토큰들에 조건부로 예측한다: $P(t_n | t_1, ..., t_{n-1})$. 추천에서는 이 토큰이 아이템 Semantic ID의 서브토큰이 된다.

**② RQ-VAE (Residual-Quantized VAE)**
아이템 임베딩을 계층적 코드워드 튜플로 압축하는 벡터 양자화기. TIGER에서 Semantic ID 생성에 사용. → [[TIGER 리뷰](./[논문][2023][NeurIPS][TIGER][Summary] Recommender Systems with Generative Retrieval.md)]

**③ PPO / GRPO (Policy Optimization)**
강화학습에서 정책 업데이트를 안정화하는 클리핑 기반 알고리즘. GRPO는 그룹 내 상대적 보상으로 advantage를 계산하므로 별도의 가치 네트워크(critic)가 불필요하다.

**④ KL Divergence 정규화**
두 분포의 차이를 측정하는 정보 이론 지표. RLHF에서 파인튜닝된 모델 $\pi_\theta$가 사전훈련 모델 $\pi_{ref}$에서 너무 멀어지지 않도록 패널티를 부과한다.

**⑤ Token Merging (ToMe)**
Transformer의 입력 토큰을 유사도 기반으로 병합하여 시퀀스 길이를 줄이는 기법. GenRec의 ALTM은 이를 추천 Semantic ID 구조에 맞게 비대칭화한 변형이다.

**⑥ CTR / GMV**
- CTR (Click-Through Rate): 추천 아이템이 클릭된 비율. 단기 참여도의 대리 지표.
- GMV (Gross Merchandise Value): 총 거래액. 장기 사용자 만족과 더 연관된 비즈니스 지표.

### A.2. 선수 논문

1. **TIGER** (NeurIPS 2023): RQ-VAE Semantic ID + T5 생성형 추천의 원조. GenRec의 기반 프레임워크.
   → [[논문][2023][NeurIPS][TIGER][Summary] Recommender Systems with Generative Retrieval.md]

2. **LC-Rec** (ICDE 2024): 협업 VQ + LLaMA 정렬 튜닝으로 언어·협업 의미 통합.
   → [[논문][2024][ICDE][LC-Rec][Summary] Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation.md]

3. **InstructGPT / RLHF** (OpenAI 2022): 인간 선호 피드백을 강화학습으로 통합하는 선호 정렬의 원형.

4. **SASRec** (ICDM 2018): Self-Attention 기반 순차 추천의 강력한 베이스라인.

### A.3. 관련 후속 연구

- **NEZHA** (WWW 2026): 생성형 추천에 투기적 디코딩(speculative decoding)을 적용하여 추론 속도를 추가로 가속. GenRec의 ALTM과 상보적 관계.
  → [[논문][2026][WWW][NEZHA][Summary] A Zero-sacrifice and Hyperspeed Decoding Architecture for Generative Recommendations.md]

- **SpecGR** (AAAI 2026): 귀납적(inductive) 검색 기반 투기적 추천으로 미노출 아이템 처리 가능.
  → [[논문][2026][AAAI][SpecGR][Summary] Inductive Generative Recommendation via Retrieval-based Speculation.md]

- **OneRec** (arXiv 2025): 검색+랭킹 통합 생성형 추천 + 반복적 선호 정렬.
  → [[논문][2025][arXiv][OneRec][Summary] OneRec - Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment.md]
