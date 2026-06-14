# Internalizing Explicit Reasoning into Latent Space for Dense Retrieval

저자 :

Yidan Liu, Haoran Xin, Minghan Li, Zheng Liu, Defu Lian, Enhong Chen

University of Science and Technology of China (USTC)

Microsoft Research Asia

발표 : arXiv 2026

논문 : [PDF](https://arxiv.org/pdf/2603.01425)

출처 : [https://arxiv.org/abs/2603.01425](https://arxiv.org/abs/2603.01425)

---

## 0. Summary

### 0.1. 문제 (Problem)

고난도 정보 검색(Dense Retrieval)에서 LLM 기반 임베딩 모델이 충분한 추론 능력을 발휘하지 못하는 문제를 다룬다.

**기존 접근 방식과 한계**:

*방법 1: 명시적 CoT 추론 후 인코딩*
- 외부 LLM(예: GPT-4o-mini)이 쿼리에 대해 Chain-of-Thought 추론 텍스트를 생성한 뒤, 이 텍스트와 원본 쿼리를 결합하여 임베딩한다.
- **한계**: 추론 단계가 고정(frozen)된 텍스트로 변환되므로, 임베딩 학습 과정에서 추론 품질이 개선되지 않는다. 또한 추론 텍스트의 길이가 입력 시퀀스를 늘려 인코딩 비용이 증가한다.

*방법 2: 순수 내재적 추론 (CoT-Distillation)*
- 임베딩 모델 자체가 내부적으로 추론하도록 KD(Knowledge Distillation)를 수행한다.
- **한계**: 추론 과정이 모델 내부에 완전히 은닉(black-box)되어 해석이 불가능하고, 복잡한 다단계 추론을 단일 벡터로 압축하는 데 한계가 있다.

*핵심 문제*: 명시적 추론(해석 가능, 단계별)과 내재적 추론(효율적, 학습 가능)의 장점을 동시에 얻지 못하고 있다.

### 0.2. 핵심 아이디어 (Core Idea)

LaSER(Latent Space Explicit Reasoning)는 **Dual-View 프레임워크**를 통해 명시적 추론의 해석 가능성과 내재적 추론의 효율성을 결합한다.

**① Explicit View — 텍스트 CoT 추론**

외부 LLM(GPT-4o-mini)이 쿼리 $q$에 대해 단계별 추론 텍스트 $R = [r_1, r_2, ..., r_m]$을 생성한다. 이 추론 체인은 관련 문서를 찾기 위해 어떤 추론 단계가 필요한지를 명시적으로 서술한다.

```
Query: "Quantum computing의 약점에 대한 논문"
Reasoning: "1. 먼저 quantum computing의 핵심 개념(큐비트, 결어긋남)을 파악한다.
            2. 현재 기술적 한계(오류율, 결맞음 시간)를 식별한다.
            3. 이 한계를 다루는 논문들을 탐색한다."
```

이 추론 텍스트는 Explicit View의 임베딩 입력으로 사용된다.

**② Latent View — K개 연속 잠재 사고 토큰**

모델이 명시적 텍스트 없이 $K$개의 소프트 토큰(soft token) $Z = [z_1, z_2, ..., z_K]$을 잠재 공간에서 자기회귀적으로 생성한다. 이 잠재 토큰들은 "기대 임베딩(expected embedding)"으로, 각 추론 단계에 대응하는 연속 벡터다.

$$z_k = f_\theta(q, z_1, ..., z_{k-1}), \quad k = 1, ..., K$$

잠재 토큰 $z_K$ (마지막 토큰)가 최종 쿼리 임베딩으로 사용된다. 이 방식은 텍스트 생성 없이 임베딩 공간 내에서 추론을 완결한다.

**③ Multi-grained Alignment — 두 뷰의 다중 단계 정렬**

Explicit View와 Latent View가 같은 추론 경로를 학습하도록 세 가지 정렬 손실을 부과한다:

| 정렬 단계 | 손실 함수 | 목적 |
|----------|-----------|------|
| **출력 정렬** | Contrastive Loss (InfoNCE) | 최종 임베딩이 같은 관련 문서를 가리키도록 |
| **출력 KL 정렬** | KL Divergence | Latent View 분포 ≈ Explicit View 분포 |
| **프로세스 정렬** | Trajectory Alignment | 중간 잠재 토큰 $z_k$ ≈ Explicit CoT $r_k$의 임베딩 |

프로세스 정렬은 각 단계의 잠재 토큰이 대응하는 명시적 추론 단계 임베딩에 가까워지도록 강제한다:
$$\mathcal{L}_{process} = \sum_{k=1}^K \| z_k - \text{Enc}(r_k) \|_2^2$$

### 0.3. 효과 (Effects)

* **추론 능력 내재화**: 텍스트 CoT를 생성하지 않고도 동등한 추론 깊이를 잠재 공간에서 수행
* **효율성**: Latent View는 K개의 토큰만 생성하므로, 긴 CoT 텍스트 없이 빠른 인퍼런스
* **해석 가능성**: Explicit View가 여전히 텍스트 추론을 제공하므로 학습 과정이 투명
* **단계별 정렬**: 프로세스 정렬 덕분에 모델이 중간 추론 단계를 건너뛰지 않고 체계적으로 학습

### 0.4. 결과 (Results)

* **BRIGHT 벤치마크**: 복잡한 추론이 필요한 고난도 검색 벤치마크에서 SOTA 달성
  - BRIGHT는 수학·코딩·과학·법률 등 다양한 도메인의 복잡한 쿼리로 구성
  - E5-mistral-7B, GTE-Qwen2-7B 등 강력한 베이스라인 대비 일관된 향상
* **일반 검색 벤치마크 (MTEB)**: BEIR 데이터셋에서 경쟁력 있는 성능 유지 (일반화 능력)
* Latent View 단독(K=8 토큰)도 Explicit View와 유사한 성능을 달성하여 추론이 성공적으로 내재화됨을 검증
* 프로세스 정렬이 없으면 성능이 크게 하락 → 단계별 정렬이 핵심

### 0.5. 상세 동작 방식 (How It Works)

**[학습 데이터 구축] ReasonEmb**

LaSER는 ReasonEmb라는 새로운 훈련 데이터셋을 구축한다:
```
1. BRIGHT 등 기존 검색 데이터셋에서 (query, relevant_doc, hard_negative) 트리플렛 수집
2. GPT-4o-mini로 각 쿼리에 대해 단계별 CoT 추론 텍스트 R 생성
3. 추론 텍스트를 K개 단계로 분할: [r1, r2, ..., rK]
4. 각 r_k를 임베딩하여 프로세스 정렬 학습에 사용
```

**[훈련 파이프라인]**

```
Stage 1: 베이스 LLM (e.g., Mistral-7B)으로 Explicit View 임베딩 모델 초기화
         → 일반 검색 데이터로 Contrastive Loss 사전훈련

Stage 2: Latent View 헤드 추가
         → Multi-grained Alignment로 공동 훈련:
            L_total = L_contrastive + α·L_KL + β·L_process
```

**[인퍼런스]**

```
입력: 쿼리 q
1. Latent View: z1 = f(q), z2 = f(q, z1), ..., zK = f(q, z_{K-1})
2. 최종 임베딩: e_q = normalize(zK)
3. ANN 검색: top-k 문서 = argmax_{d} e_q · e_d
# CoT 텍스트 생성 없음 → 빠른 인퍼런스
```

---

## 1. Introduction

### 추론 기반 검색의 필요성

BRIGHT 벤치마크(Su et al., 2024)는 기존의 어휘적·의미적 매칭으로는 해결할 수 없고, 다단계 추론이 필요한 검색 태스크를 평가한다. 예를 들어:

- *쿼리*: "이 알고리즘의 시간복잡도가 O(n log n)인 이유를 설명하는 논문"
- *필요한 추론*: 알고리즘 → 분할정복 패턴 인식 → 재귀 관계 분석 → 마스터 정리 적용 → 관련 논문 탐색

기존 밀집 검색기(Dense Retriever, e.g., BERT, E5)는 이런 다단계 추론을 단일 포워드 패스로 수행하기 어렵다.

### 기존 해결책의 트레이드오프

| 방법 | 추론 가능 | 학습 가능 | 추론 속도 |
|------|---------|----------|---------|
| Explicit CoT + 인코딩 | O | X (고정 추론) | 느림 |
| 순수 내재적 추론 | X (불투명) | O | 빠름 |
| **LaSER (제안)** | O | O | 빠름 |

---

## 2. Method

### 2.1. Dual-View 임베딩 아키텍처

```
쿼리 q
├── Explicit View (학습 시에만)
│   └── [q + R] → LLM 인코더 → e_explicit
└── Latent View (학습 + 인퍼런스)
    └── [q] → Latent Decoder → [z1, ..., zK] → e_latent = zK
```

백본: LLM 기반 임베딩 모델(e.g., Mistral-7B, Qwen2-7B)

Latent Decoder는 경량 Transformer 디코더 블록으로, 쿼리 임베딩을 조건으로 K개의 잠재 토큰을 자기회귀적으로 생성한다.

### 2.2. Multi-grained Alignment 손실

**출력 대조 손실 (Contrastive)**:
$$\mathcal{L}_{con} = -\log \frac{\exp(e_{latent} \cdot e_{d^+} / \tau)}{\sum_{d' \in \mathcal{B}} \exp(e_{latent} \cdot e_{d'} / \tau)}$$

여기서 $e_{d^+}$는 관련 문서 임베딩, $\mathcal{B}$는 배치 내 모든 문서(하드 네거티브 포함), $\tau$는 온도 파라미터.

**출력 KL 정렬**:
$$\mathcal{L}_{KL} = D_{KL}\left(P_{explicit}(\cdot \mid q) \| P_{latent}(\cdot \mid q)\right)$$

두 뷰의 소프트맥스 분포(배치 내 문서들에 대한)가 일치하도록 정규화.

**프로세스 정렬**:
$$\mathcal{L}_{proc} = \frac{1}{K} \sum_{k=1}^K \| z_k - \text{sg}[\text{Enc}(r_k)] \|_2^2$$

$\text{sg}[\cdot]$은 stop-gradient 연산으로 Explicit View 임베딩은 고정하고 Latent View만 업데이트.

**전체 손실**:
$$\mathcal{L} = \mathcal{L}_{con} + \alpha \mathcal{L}_{KL} + \beta \mathcal{L}_{proc}$$

### 2.3. ReasonEmb 데이터셋

| 속성 | 값 |
|------|-----|
| 출처 | BRIGHT + 일반 검색 데이터셋 |
| CoT 생성 모델 | GPT-4o-mini |
| 추론 단계 수 K | 4–8 |
| 총 규모 | ~100K 트리플렛 |
| 도메인 | 수학, 코딩, 과학, 법률, 의학, 금융 |

---

## 3. Experiments

### 데이터셋

| 벤치마크 | 특징 | 용도 |
|---------|------|------|
| BRIGHT | 복잡한 추론 필요 검색 (11개 도메인) | 주요 평가 |
| BEIR | 18개 도메인 다양한 검색 태스크 | 일반화 평가 |

### 비교 모델

| 모델 | 파라미터 | 특징 |
|------|---------|------|
| E5-mistral-7B | 7B | 강력한 LLM 임베딩 베이스라인 |
| GTE-Qwen2-7B | 7B | 최신 LLM 임베딩 |
| Reasoning + Embed (명시적 CoT) | 7B | 텍스트 CoT 후 인코딩 |
| LaSER (ours) | 7B+ | Dual-View 정렬 |

### BRIGHT 결과 (nDCG@10)

| 모델 | 평균 | 수학 | 코딩 | 과학 |
|------|------|------|------|------|
| E5-mistral-7B | ~22 | ~18 | ~15 | ~25 |
| 명시적 CoT | ~26 | ~23 | ~19 | ~28 |
| LaSER | **~30** | **~27** | **~22** | **~32** |

### Ablation (BRIGHT 기준)

| 변형 | nDCG@10 |
|------|---------|
| Latent View only (w/o Explicit) | 하락 |
| Explicit View only (w/o Latent) | 하락 |
| w/o Process Alignment | 유의미한 하락 |
| w/o KL Alignment | 소폭 하락 |
| **Full LaSER** | **최고** |

---

## 4. Conclusion

LaSER는 추론 기반 고난도 검색에서 명시적 CoT와 잠재 공간 추론을 Dual-View 프레임워크로 결합한다. 학습 시 두 뷰를 다중 단계 정렬로 공동 훈련하고, 인퍼런스 시에는 Latent View만 사용하여 속도를 유지하면서 추론 능력을 내재화한다.

**핵심 기여**:
1. **Dual-View 프레임워크**: 명시적 텍스트 추론과 잠재 공간 추론의 동시 학습
2. **Multi-grained Alignment**: 출력 레벨(Contrastive + KL) + 프로세스 레벨(궤적) 정렬
3. **ReasonEmb 데이터셋**: 추론 기반 검색 훈련을 위한 새로운 다단계 CoT 데이터셋
4. **효율적 인퍼런스**: CoT 텍스트 생성 없이 K개 잠재 토큰으로 빠른 추론

**한계**:
- ReasonEmb 구축에 GPT-4o-mini API 비용 필요
- K(잠재 토큰 수) 설정이 태스크 복잡도에 민감
- 추론 단계 분할(CoT를 K개로 나누는 방법)이 자동화 되어 있지 않음

---

## Appendix

### A.1. 핵심 사전 개념

**① Dense Retrieval (밀집 검색)**
쿼리와 문서를 각각 고차원 벡터로 임베딩하고, 내적(dot product) 또는 코사인 유사도로 관련 문서를 검색하는 방식. BERT 기반 Bi-Encoder(DPR 등)가 대표적. ANN 인덱스(FAISS)로 수백만 문서 중 빠르게 검색.

**② Chain-of-Thought (CoT) 추론**
복잡한 추론 문제를 단계별 중간 추론 과정을 명시하면서 풀어나가는 프롬프팅 기법. LLM이 "먼저 A를 파악하고, 그 다음 B를 분석하고..."처럼 추론 체인을 명시적으로 생성한다.

**③ Soft Token / Continuous Token**
일반 텍스트 토큰이 아닌 연속 벡터 공간의 토큰. Prompt Tuning, Prefix Tuning, Latent Chain-of-Thought 등에서 입력 또는 중간 상태로 사용. 이산 텍스트로 표현할 필요 없이 모델의 내부 표현 공간에서 직접 최적화 가능.

**④ Knowledge Distillation (KD)**
더 크고 강력한 교사 모델(teacher)의 예측을 소프트 레이블로 활용하여 작은 학생 모델(student)을 훈련하는 기법. LaSER에서 Explicit View(교사)가 Latent View(학생)를 지도.

**⑤ BRIGHT 벤치마크**
(Su et al., 2024) 단순 어휘·의미 매칭이 아닌 복잡한 추론이 필요한 정보 검색을 평가하는 벤치마크. 수학, 코딩, 법률, 의학, 금융 등 11개 도메인으로 구성되며, 기존 검색 모델들이 크게 고전하는 어려운 벤치마크.

**⑥ InfoNCE Loss (Contrastive Loss)**
자기지도 학습에서 사용하는 대조 손실. 쿼리 임베딩이 관련 문서(positive)와 가깝고 비관련 문서(negative)와 멀어지도록 훈련. 분모에 전체 배치의 문서를 넣어 계산하므로 배치 크기가 클수록 학습 신호가 강해진다.

### A.2. 선수 논문

1. **DPR** (EMNLP 2020): Dense Passage Retrieval. 이중 인코더(Bi-Encoder)로 질의응답 검색을 위한 밀집 검색의 기초.

2. **E5-mistral-7B** (arXiv 2023): Mistral-7B 기반 텍스트 임베딩 모델. 지시 따르기(instruction-following) 방식으로 다양한 검색 태스크를 단일 모델로 처리.

3. **BRIGHT** (Su et al., 2024): 복잡한 추론 기반 검색 벤치마크. LaSER의 핵심 평가 대상.

4. **Chain-of-Thought Prompting** (Wei et al., NeurIPS 2022): 단계별 추론 체인을 명시적으로 생성하는 프롬프팅 기법의 원조.

### A.3. 관련 후속 연구

- **추론 강화 임베딩 (Reasoning Embeddings)**: LLM의 추론 능력을 검색에 활용하는 방향은 활발히 연구 중. LaSER의 잠재 공간 추론 내재화 방식은 향후 멀티모달 검색, 코드 검색 등으로 확장 가능.

- **추천 시스템에서의 추론**: LaSER는 검색(Retrieval)에 특화되어 있으나, 사용자 이력에서 복잡한 추론이 필요한 추천 시나리오에도 유사한 Dual-View 아이디어 적용 가능.
