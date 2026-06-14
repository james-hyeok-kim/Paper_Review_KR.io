# Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation (LC-Rec)

저자: Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ming Chen, Ji-Rong Wen (RUCAIBox, 중국인민대학교)

출간: ICDE 2024 (IEEE 40th International Conference on Data Engineering)

논문: [arXiv 2311.09049](https://arxiv.org/abs/2311.09049) | [GitHub](https://github.com/RUCAIBox/LC-Rec)

---

## 0. Abstract

* LLM 기반 추천 시스템에서 **언어 의미론(Language Semantics)**과 **협업 의미론(Collaborative Semantics)** 사이의 큰 간극을 해결
* 두 가지 핵심 기여:
  1. **협업 ID 인덱싱**: 연습 기반 벡터 양자화 + 균일 의미 매핑으로 원환하도록 LLM 어휘에 통합되는 아이템 인덱스(Collaborative ID) 생성
  2. **정렬 튜닝(Alignment Tuning)**: 여러 전용 튜닝 태스크로 LLM에 협업 의미론을 내재화
* Amazon 페버리원 3개 데이터셋에서 전통 추천 모델 및 기존 LLM 기반 추천 모델들 대비 일관된 성능 향상
* 백본: LLaMA-7B

---

## 1. Introduction

**LLM 기반 추천의 두 가지 접근법과 한계**

주요 관련 연구들이 쓰는 접근법:

| 접근법 | 예시 | 장점 | 단점 |
|---|---|---|---|
| 아이템 텍스트 기반 | P5, VQ-Rec 등 | 언어 의미론 활용 | **협업 신호 미활용** |
| 임의 정수 ID | BIGRec 등 | 단순함 | LLM 어휘 외 토큰, **의미론적 링크 없음** |

**근본 문제**: 사용자-아이템 상호작용에서 얻는 **협업 신호(누가 어떤 아이템을 함께 선호하는가)**가 LLM에 직접 졸재하지 않음

**LC-Rec의 핵심 아이디어**: 협업 필터링에서 학습된 아이템의 잔차 벡터를 이산 코드로 양자화하여 LLM 어휘에 컴팩하고, 이 **Collaborative ID**를 LLM에 의미론적으로 포함시키는 다양한 튜닝 태스크를 설계

---

## 2. Related Work

* **협업 필터링**: MF, LightGCN, SASRec — 협업 신호를 임베딩으로 표현하나 자연어 능력 미활용
* **LLM 기반 추천**: P5, LLaRA, LLMRank — 자연어 이해력을 사용하나 협업 신호 간과
* **생성형 추천**: TIGER — Semantic ID로 언어 의미론은 인코딩하나 협업 신호는 인코딩하지 못함
* **벡터 양자화**: VQ-VAE, RQ-VAE — LC-Rec은 협업 증류를 위한 특화된 균일 의미 매핑 추가

---

## 3. LC-Rec 방법론

### 3.1 협업 ID 인덱싱 (Collaborative Item Indexing)

**목표**: 협업 신호를 담은 딕스크릿 코드를 LLM이 이해할 수 있는 어휘로 표현

**Step 1: 연습 임베딩 학습**

SASRec 등 순차 추천 모델로 사용자-아이템 상호작용을 학습하여 아이템의 협업 표현 $h_i$를 확보:

$$h_i = \text{CollaborativeEncoder}(\text{interaction\_history involving } i)$$

**Step 2: 균일 의미 매핑을 통한 벡터 양자화 (VQ with Uniform Semantic Mapping)**

일반 VQ-VAE는 코드북 활용륙이 불균일해지는 **Codebook Collapse** 문제 발생

해결: **균일 의미 매핑(Uniform Semantic Mapping)** 구성

$$c_i = \arg\min_{k \in [K]} \| h_i - e_k \|^2 \quad \text{s.t. } |\{i : c_i = k\}| \approx N/K$$

* 성능: 모든 코드워드가 고르게 사용됨 → 충돌 없는 아이템 코드 포트폴리오 확보
* 다단계 계층 코드: 다수 코드워드의 시퀀스 $(c_i^{(1)}, c_i^{(2)}, \ldots)$로 아이템 $i$를 표현

**Collaborative ID**: 이 코드 시퀀스를 LLM 토큰 ID로 사용

$$\text{Collaborative ID}(i) = [\text{``[COL''}+c_i^{(1)}\text{``]''}, \ldots, \text{``[COL''}+c_i^{(L)}\text{``]''}]$$

이 토큰들은 LLaMA-7B의 어휘에 새로 추가되어 학습됨

### 3.2 정렬 튜닝 (Alignment Tuning)

**목표**: LLM이 Collaborative ID의 의미를 언어 수준에서 이해하도록 6개 전용 지시보 (instruction) 튜닝태스크 설계:

| 태스크명 | 입력 | 콨력 | 목적 |
|---|---|---|---|
| `seqrec` | 사용자 히스토리 (Collaborative ID 시퀀스) | 다음 아이템 Collaborative ID | 기본 추천 태스크 |
| `item2index` | 아이템 텍스트 설명 | Collaborative ID | 텍스트 → 협업 ID 매핑 |
| `index2item` | Collaborative ID | 아이템 텍스트 설명 | 협업 ID → 텍스트 매핑 |
| `fusionseqrec` | 사용자 히스토리 + 아이템 텍스트 혼합 | Collaborative ID | 언어+협업 정보 융합 |
| `itemsearch` | 사용자 선호 쿼리 | Collaborative ID 목록 | 선호 기반 아이템 검색 |
| `preferenceobtain` | 사용자 히스토리 | 선호 텍스트 요약 | 사용자 선호 추출 |

**학습 전략**: Instruction-following 형식으로 모든 태스크를 동시에 태스크 샘플링

$$\mathcal{L} = -\sum_{\tau} \sum_{t} \log P(y_t \mid x_{\tau}, y_{<t}; \theta)$$

($\tau$: 태스크, $x_\tau$: 입력 지시보, $y_t$: 타겟 토큰)

### 3.3 전체 학습 파이프라인

```
Step 1. 협업 모델 학습 (SASRec 등)
         |
         v
Step 2. Uniform VQ로 Collaborative ID 생성
         |
         v
Step 3. LLaMA-7B + 6개 지시보 태스크로 Alignment Tuning
         |
         v
최종 LC-Rec 모델
```

---

## 4. Experimental Setup

### 데이터셋

Amazon 페버리월 데이터셋 3개 사용:

| 데이터셋 | 도메인 | 특징 |
|---|---|---|
| Instruments | 악기/음악 장비 | 중간 규모 |
| Arts | 예술 용품 | 중간 규모 |
| Games | 비디오게임 | 중간 규모 |

### 학습 환경
* **백본 LLM**: LLaMA-7B
* **GPU**: 8 GPU 분산 학습 (DeepSpeed)
* **학습률**: 5e-5
* **에폭**: 4
* **정밀도**: bfloat16
* **파라미터 효율화**: LoRA 유사 가중치 델타 표현 방식 제공

### 평가 지표
* HR@K, NDCG@K (K=5, 10, 20)
* Leave-one-out 평가

### 비교 베이스라인

| 찬류 | 방법 |
|---|---|
| 전통 협업 필터링 | BPR-MF, SASRec, BERT4Rec |
| LLM 기반 (텍스트) | P5, PALR, LLaRA |
| LLM 기반 (ID) | BIGRec |
| 혼합 | VQ-Rec, VIP5 |

---

## 5. Experiments

### 주요 결과

* LC-Rec는 3개 데이터셋 모두에서 SOTA 달성
* 특히 기존 LLM 기반 추천 (텍스트 ID 기반)에 비해 대폭 향상을 보임
* BIGRec(임의 정수 ID) 대비: 협업 ID를 사용하는 LC-Rec이 일관되게 우수 → 협업 연동의 중요성 실증

### Ablation: 핵심 컴포넌트 기여도

| 설정 | HR@10 | 변화 |
|---|---|---|
| 전체 LC-Rec | 최고 | 기준 |
| Collaborative ID 제거 (임의 ID) | 하락 | 협업 신호 직접 활용의 효과 |
| Uniform Mapping 제거 | 하락 | 코드북 충돌 다양성 저하 |
| `item2index` 태스크 제거 | 하락 | ID-텍스트 연결 중요성 |
| `index2item` 태스크 제거 | 하락 | 양방향 정렬의 중요성 |
| `fusionseqrec` 제거 | 하락 | 언어+협업 융합 중요성 |

### Uniform Semantic Mapping 효과

* 일반 VQ는 코드북 collapse로 일부 코드워드만 사용 → 아이템 ID 충돌 발생
* Uniform Mapping 적용 시: 모든 코드워드가 고르게 사용됨 → 충돌 방지 + 형승된 성능

### 다양한 태스크 상시 학습 (Multi-task) 효과

* 6개 태스크 모두를 함께 학습할 때 최고 성능
* `item2index` + `index2item` 태스크가 양방향 ID-텍스트 연결을 강화하여 특히 중요

---

## 의의

#### 1. 언어-협업 의미론 통합의 체계적 접근
* 단순히 텍스트를 사용하거나 임의 ID를 쓰는 구독에서 보다 원칙적인 두 의미론 통합 방법론 제시
* **협업 신호를 어휘화(vocabularization)**하여 LLM이 협업 패턴을 직접 이해하도록 함

#### 2. Uniform Semantic Mapping으로 Codebook Collapse 해결
* VQ 기반 협업 ID 학습의 코드북 충돌 문제에 대한 명시적 해결책 제시
* 아이템 ID 충돌 방지 → LLM이 Collaborative ID에서 고유한 협업 패턴을 학습 가능

#### 3. 다방향 정렬 태스크로 양방향 언어-협업 정렬
* ID → 텍스트, 텍스트 → ID 양방향 태스크로 LLM이 Collaborative ID의 의미를 완전히 내면화
* 이 접근은 이후 GenRec, NEZHA, SpecGR 등의 ID 설계에도 영향을 줌
