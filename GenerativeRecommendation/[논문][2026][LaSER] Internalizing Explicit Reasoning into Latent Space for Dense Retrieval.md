# LaSER: Internalizing Explicit Reasoning into Latent Space for Dense Retrieval

저자: Jiajie Jin, Yanzhao Zhang, Mingxin Li, Dingkun Long, Pengjun Xie, Yutao Zhu, Zhicheng Dou

소속: 중국인민대학교 AI 스쿨 & 알리바바 통이 랩

출간: arXiv 2603.01425 (2026년 3월)

코드: [GitHub](https://github.com/ignorejjj/LaSER)

논문: [arXiv 2603.01425](https://arxiv.org/abs/2603.01425)

---

## 0. Abstract

* LLM 기반 밀집 검색기(dense retriever)는 대형 언어모델로부터 추론 능력을 상속받지만, 대조 학습(contrastive learning) 방식으로 훈련되어 **인코더로만** 동작하며 추론 능력을 활용하지 못함
* **명시적 추론(Explicit Reasoning)의 잠재 공간(Latent Space) 내재화** 프레임워크 LaSER 제안
  - Explicit View: Chain-of-Thought(CoT) 포함 증강 입력으로 학습 → 의미론적 상한선
  - Latent View: 원본 쿼리 + $K$개 연속 잠재 사고 토큰(latent thinking tokens) → 추론 없이 효율적 추론 효과
* BRIGHT 벤치마크에서 Rewrite-then-Retrieve 대비 동등 성능을 **0.3% 지연**으로 달성
* Qwen3-8B 기반 최고 성능: BRIGHT nDCG@10 **29.3** 달성

---

## 1. Introduction

**추론이 필요한 검색의 문제**

현대 검색 시스템은 단순 키워드 매칭을 넘어 쿼리의 의도를 추론해야 하는 복잡한 질의를 처리해야 함:
- "양자 컴퓨터가 RSA 암호화를 위협하는가?" → 다단계 추론 필요
- BRIGHT 벤치마크: 12개 도메인의 추론 집약적 검색 태스크

**기존 접근의 한계**

| 접근법 | 장점 | 단점 |
|---|---|---|
| 표준 밀집 검색 (DPR 등) | 빠른 추론 | 추론 능력 미활용 |
| Rewrite-then-Retrieve | 높은 검색 품질 | 자기회귀 생성 지연 (LaSER 대비 ~333×) |
| 암묵적 잠재 추론 (GIRCSE 등) | 효율적 | 명시적 감독 부재로 의미 퇴화 |

**LaSER의 핵심**: 명시적 CoT를 교사(teacher)로 활용, 잠재 공간에 추론을 증류(distill) → 추론 없이도 추론 효과

---

## 2. Related Work

* **밀집 검색 (Dense Retrieval)**: DPR, E5, GTE — 대조 학습 기반, LLM backbone
* **Chain-of-Thought 증류**: STaR, DISTILL-CoT — 텍스트 공간에서 추론 증류 (LaSER는 잠재 공간에서 수행)
* **잠재 사고 (Latent Thinking)**: Coconut 등 — 언어 생성 없이 잠재 벡터로 추론 수행
* **Rewrite-then-Retrieve**: HyDE, Query2Doc — 외부 LLM으로 쿼리 재작성 후 검색 (높은 지연)
* **BRIGHT Benchmark**: 추론 집약적 검색 평가 벤치마크 (1,384 쿼리, 1.14M 문서)

---

## 3. LaSER 방법론

### 3.1 이중 뷰 프레임워크 (Dual-View Framework)

두 뷰가 **파라미터를 공유**하면서 동시에 학습:

**Explicit View (명시적 뷰)**
- 입력: 원본 쿼리 + 외부 추론기(GPT-4o-mini 생성 CoT)로부터의 고품질 추론 경로
- 역할: 학습 중 **의미론적 상한선(semantic upper bound)** 제공
- 단일 순전파(forward pass)로 포괄적 추론 포착

**Latent View (잠재 뷰)**
- 입력: 원본 쿼리만 (추론 없음)
- $K$개의 **연속 잠재 사고 토큰** $T = \{t_1, \ldots, t_K\}$를 생성한 후 최종 표현 산출
- 추론 시 텍스트 생성 없이 추론 효과 달성

### 3.2 잠재 토큰 생성

각 추론 단계 $j$에서:

1. 이전 상태를 언어 모델링 헤드로 투영 → 로짓(logits)
2. Softmax로 확률 분포 $p_j$ 계산
3. **소프트 토큰**(미분 가능): $t_j = p_j^\top E$ (기대 임베딩 벡터)

$$t_j = \text{softmax}(\text{LM\_head}(h_j))^\top E$$

여기서 $E$는 임베딩 행렬. $t_j$를 다음 단계 입력으로 이어붙여 자기회귀적으로 정제.

### 3.3 다중 세분도 정렬 (Multi-Grained Alignment)

세 가지 보완 손실:

**① 대조 학습 (Contrastive Loss)**
두 뷰 모두 양성/음성 문서 쌍에 InfoNCE 손실 적용:
$$\mathcal{L}_{\text{cl}} = -\log \frac{\exp(q \cdot d^+ / \tau)}{\sum_{d' \in \mathcal{D}} \exp(q \cdot d' / \tau)}$$

**② 출력 수준 증류 (Output-Level Distillation)**
명시적 뷰와 잠재 뷰의 **점수 분포**를 KL 발산으로 정렬:
$$\mathcal{L}_{\text{kl}}^{\text{out}} = \text{KL}(P_{\text{explicit}} \| P_{\text{latent}})$$
표현 자체가 아닌 분포를 정렬 → 과도한 제약 방지

**③ 과정 수준 궤적 정렬 (Process-Level Trajectory Alignment)**
잠재 사고 토큰의 중간 상태를 명시적 추론 경로의 해당 세그먼트와 정렬:
$$\mathcal{L}_{\text{kl}}^{\text{mid}} = \sum_{j=1}^{K} \text{KL}(P_{\text{explicit}}^{(j)} \| P_{\text{latent}}^{(j)})$$
균일 시간 다운샘플링으로 각 잠재 단계를 명시적 세그먼트에 매핑

**최종 학습 목표**:
$$\mathcal{L} = \mathcal{L}_{\text{cl}}^L + \lambda_1 \mathcal{L}_{\text{cl}}^E + \lambda_2 \mathcal{L}_{\text{kl}}^{\text{out}} + \lambda_3 \mathcal{L}_{\text{kl}}^{\text{mid}}$$

($\lambda_1=1, \lambda_2=10, \lambda_3=0.1$)

---

## 4. Experimental Setup

### 데이터셋

| 데이터셋 | 사용 목적 | 규모 |
|---|---|---|
| ReasonEmb | 학습 | 81,659 쿼리, 12 도메인, GPT-4o-mini CoT |
| BRIGHT | 평가 (In-Domain) | 1,384 쿼리, 1.14M 문서 |
| FollowIR | 평가 (Out-of-Domain) | 104 쿼리 |
| BrowseComp-Plus | 평가 (Out-of-Domain) | 830 쿼리 |

### 백본 모델
* Qwen3 시리즈: 0.6B, 4B, 8B
* LLaMA 3.1/3.2 시리즈: 1B, 3B, 8B

### 평가 지표
* BRIGHT: nDCG@10
* FollowIR: p-MRR, MAP@5, nDCG@5
* BrowseComp-Plus: Recall@5, @100, @1000

---

## 5. Experiments

### BRIGHT 주요 성능 (Qwen3-8B)

| 방법 | nDCG@10 |
|---|---|
| 표준 밀집 검색 (Fair Baseline) | 25.7 |
| Qwen3-Embedding-8B (원본) | 14.0 |
| Rewrite-then-Retrieve | ~28.1 |
| LaSER | **29.3** |

### 지연 시간 비교 (BRIGHT 80 쿼리, A100 GPU)

| 방법 | 상대 지연 |
|---|---|
| Rewrite-then-Retrieve | 100% (기준) |
| LaSER | **0.3%** |
| 표준 검색 대비 오버헤드 | ~1.7× |

### Ablation (Qwen3-0.6B on BRIGHT)

| 설정 | nDCG@10 | 변화 |
|---|---|---|
| 전체 LaSER | 23.10 | 기준 |
| Explicit View 제거 | 20.59 | -10.9% |
| Latent View 제거 | 19.93 | -13.7% |
| 과정 정렬 제거 | 22.33 | -3.3% |
| 출력 증류 제거 | 19.97 | -13.5% |
| 오프라인 교사 (Co-learning 제거) | 20.98 | -9.1% |

**핵심 발견**: 출력 증류가 가장 중요, Latent View 제거가 두 번째로 큰 하락

### 추론 스케일링

추론 시 잠재 단계 $K$를 증가시킬수록 성능이 일관되게 향상 → 모델이 강건한 정제 메커니즘을 학습

### 범용성 (Out-of-Domain)

FollowIR, BrowseComp-Plus에서도 성능 향상 전이 → 증류된 추론 메커니즘이 데이터셋 특화 아님을 확인

---

## 의의

#### 1. 추론 능력을 잠재 공간에 증류한 최초 밀집 검색 연구
* Explicit-to-Latent 추론 증류를 밀집 검색에 최초 적용
* 텍스트 생성 없이 추론 의미론을 잠재 벡터에 성공적으로 인코딩

#### 2. 속도-품질 트레이드오프 극복
* Rewrite-then-Retrieve 수준의 품질을 0.3% 지연으로 달성
* 표준 검색기 대비 1.7× 오버헤드만으로 추론 강화 검색 가능

#### 3. Co-learning의 효과
* 고정된 오프라인 교사보다 온라인 공동 학습(Co-learning)이 9.1% 더 우수
* 명시적 추론 경로 제공 시 LaSER가 기본 검색기 대비 훨씬 더 큰 이득 — 명시적 뷰 훈련이 백본의 CoT 포착 능력 강화
