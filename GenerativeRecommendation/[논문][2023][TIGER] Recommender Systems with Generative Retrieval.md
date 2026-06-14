# Recommender Systems with Generative Retrieval (TIGER)

저자: Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Maheswaran Sathiamoorthy (Google)

출간: NeurIPS 2023

논문: [arXiv 2305.05065](https://arxiv.org/abs/2305.05065)

---

## 0. Abstract

* 추천 시스템에서 기존의 **임베딩 기반 최근접 이웃 탐색(ANN)** 대신, **생성형 검색(Generative Retrieval)** 패러다임을 도입
* 각 아이템에 **Semantic ID**라는 의미론적 코드워드 튜플을 부여하고, 사전학습된 언어모델(T5)이 사용자 히스토리를 입력받아 다음에 상호작용할 아이템의 Semantic ID를 자기회귀적으로 예측
* Amazon Beauty, Sports, Toys, Yelp, ML-1M 등 5개 벤치마크에서 기존 SOTA 대비 일관된 성능 향상 달성
* 콜드스타트(Cold-start) 아이템 추천 성능이 특히 향상됨

---

## 1. Introduction

**기존 추천 시스템의 한계**

* Two-Tower 등 임베딩 기반 방법은 아이템을 연속 벡터 공간에 임베딩하고 ANN 검색으로 후보를 뽑음
* 이 구조는 아이템 간의 의미론적 관계를 벡터 유사도로만 표현하며, **새로운 아이템(콜드스타트)** 에 대한 일반화가 어려움
* 또한 사용자-아이템 상호작용을 학습하는 과정에서 **모델과 검색 인덱스가 분리**되어 있어 end-to-end 최적화가 불가

**생성형 검색의 핵심 아이디어**

* 아이템 추천을 "다음 아이템 ID 시퀀스를 생성하는 문제"로 재정의
* 자연어 처리의 seq2seq 모델처럼, 사용자 히스토리 → 추천 아이템 ID를 직접 생성
* 핵심 과제: **아이템을 어떻게 의미있는 이산(discrete) ID로 표현할 것인가?**

---

## 2. Related Work

* **협업 필터링(Collaborative Filtering)**: MF, NCF 등 — 아이템을 latent vector로 표현, 콜드스타트 취약
* **순차 추천(Sequential Recommendation)**: SASRec, BERT4Rec 등 — 사용자 히스토리의 순서를 모델링하나 여전히 임베딩 기반 검색
* **언어모델 기반 추천**: P5 등 — 아이템을 자연어로 표현하나 텍스트 생성으로 인한 환각(hallucination) 문제
* **생성형 검색(DSI, NCI 등)**: 문서 검색 분야에서 문서 ID를 직접 생성하는 연구 — 추천 시스템에 적용한 첫 사례가 TIGER

---

## 3. TIGER: 방법론

### 3.1 Semantic ID 생성 (RQ-VAE)

**목표**: 각 아이템을 의미론적으로 유사한 아이템끼리 유사한 코드를 갖도록 이산 ID 튜플로 표현

**Residual Quantized Variational AutoEncoder (RQ-VAE)**

아이템의 콘텐츠 임베딩(예: 텍스트/이미지 특징을 통해 사전학습된 임베딩)을 입력으로, 계층적 코드북을 통해 압축:

1. 아이템 임베딩 $e_i$를 첫 번째 코드북에서 가장 가까운 코드워드로 양자화 → 코드 $c_i^{(1)}$
2. 잔차(residual) $r_i^{(1)} = e_i - q^{(1)}(e_i)$를 두 번째 코드북으로 양자화 → 코드 $c_i^{(2)}$
3. 이를 $D$단계 반복 → 최종 Semantic ID = $(c_i^{(1)}, c_i^{(2)}, \ldots, c_i^{(D)})$

**결과**: 의미론적으로 유사한 아이템(같은 카테고리, 유사한 속성)은 앞쪽 코드($c^{(1)}$)를 공유하는 트리 구조 형성

$$\text{Semantic ID}(i) = \text{RQ-VAE}(e_i) = (c_i^{(1)}, c_i^{(2)}, \ldots, c_i^{(D)})$$

### 3.2 TIGER 모델 아키텍처

**백본**: 사전학습된 T5 (encoder-decoder Transformer)

**입력**: 사용자 히스토리 내 아이템들의 Semantic ID 시퀀스

$$\text{Input} = [c_{i_1}^{(1)}, c_{i_1}^{(2)}, \ldots, c_{i_1}^{(D)}, \;\; \text{[SEP]} \;\; c_{i_2}^{(1)}, \ldots, c_{i_{t-1}}^{(D)}]$$

**출력**: 다음 아이템 $i_t$의 Semantic ID 자기회귀 생성

$$P(i_t \mid \text{history}) = \prod_{d=1}^{D} P(c_{i_t}^{(d)} \mid c_{i_t}^{(1)}, \ldots, c_{i_t}^{(d-1)}, \text{history})$$

**추론 (Beam Search)**
- Beam search로 상위 $K$개의 Semantic ID 후보를 생성
- Semantic ID → 실제 아이템 매핑을 통해 최종 추천 리스트 반환
- 유효하지 않은 ID 생성을 방지하기 위해 Constrained Beam Search 사용 가능

### 3.3 학습

* **손실 함수**: 표준 교차 엔트로피 (각 코드 위치에서)
* **사전학습 T5 파인튜닝**: 아이템 도메인 특화 데이터로 파인튜닝
* **콘텐츠 임베딩**: SentenceT5를 사용해 아이템 텍스트(제목, 카테고리, 설명)를 인코딩 후 RQ-VAE 입력으로 사용

---

## 4. Experimental Setup

### 데이터셋

| 데이터셋 | 도메인 | 아이템 수 | 상호작용 수 |
|---|---|---|---|
| Beauty | 아마존 뷰티 | ~12K | ~198K |
| Sports | 아마존 스포츠 | ~18K | ~296K |
| Toys | 아마존 완구 | ~12K | ~167K |
| Yelp | 레스토랑 | ~20K | ~1.3M |
| ML-1M | 영화 | ~3.7K | ~1M |

### 평가 지표
* Recall@K, NDCG@K (K=5, 10)
* Leave-one-out 방식: 각 사용자의 마지막 상호작용을 테스트로 사용

### 비교 베이스라인
* 협업 필터링: BPR-MF, LightGCN
* 순차 추천: SASRec, BERT4Rec, FDSA
* 언어모델 기반: P5, UniSRec

---

## 5. Experiments

### 주요 결과

* TIGER는 5개 데이터셋 모두에서 기존 SOTA 대비 일관된 성능 향상 달성
* 특히 **콜드스타트 아이템** (상호작용 없는 신규 아이템) 추천에서 두드러진 우위: Semantic ID가 아이템 콘텐츠 정보를 인코딩하므로 상호작용 이력 없이도 의미론적 위치를 파악 가능

### Ablation: Semantic ID vs. Random ID

* Random ID (랜덤 할당) 대비 Semantic ID가 일관되게 우수
* RQ-VAE 계층 수 $D$: 적을수록 구분력 부족, 많을수록 생성 어려움 → 실험상 $D=3$이 최적

### Ablation: 코드북 크기

* 코드북 크기가 너무 작으면 서로 다른 아이템이 동일 ID를 공유 (충돌)
* 너무 크면 희소성 증가로 학습 어려움
* 각 데이터셋 아이템 수에 따라 최적값이 다름 (일반적으로 256 또는 512)

---

## 의의

#### 1. 추천 시스템에 생성형 검색 패러다임 도입
* 임베딩 벡터 + ANN 검색의 two-stage 파이프라인을 단일 seq2seq 모델로 통합
* 모델 학습과 아이템 검색이 end-to-end로 최적화됨

#### 2. Semantic ID를 통한 콘텐츠-행동 정보 통합
* RQ-VAE로 아이템 콘텐츠의 의미 구조를 계층적 코드로 압축
* 신규 아이템에 대한 일반화 능력(콜드스타트) 대폭 향상

#### 3. 생성형 추천 연구의 기초
* TIGER 이후 GenRec, SpecGR, NEZHA 등 생성형 추천 연구의 흐름을 정의한 선구적 연구
* Semantic ID 개념은 후속 연구들의 핵심 구성요소로 채택됨
