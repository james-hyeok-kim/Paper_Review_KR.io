# ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs

저자 :

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, Maosong Sun

Tsinghua University, ModelBest Inc., Renmin University of China, Yale University, WeChat AI (Tencent)

발표 : ICLR 2024

논문 : [PDF](https://arxiv.org/pdf/2307.16789)

출처 : [https://arxiv.org/abs/2307.16789](https://arxiv.org/abs/2307.16789)

---

## 0. Summary

<p align='center'>
<img src="figs/ToolLLM/fig_01.png" alt="Figure 1: ToolBench 구축 3단계 및 API Retriever + ToolLLaMA 학습/추론 파이프라인" width="800"/>
</p>

### 0.1. 문제 (Problem)

* 오픈소스 LLM(LLaMA 등)은 외부 도구(API)를 사용하는 **tool-use 능력이 현저히 부족**하다.
* 기존 instruction tuning 데이터셋은 언어 능력 위주로 구성되어, 실제 REST API 호출을 포함하지 않거나 소수의 단순 API만 다룬다.
* 기존 도구 학습 데이터셋의 세 가지 한계:
  1. **제한된 API 수**: 실제 RESTful API를 다루지 않거나, 다양성이 없는 소수의 API만 포함 (예: APIBench는 1,645개, API-Bank는 53개).
  2. **단일 도구 시나리오**: 하나의 API만 사용하는 상황만 커버하고, 실세계에서 필요한 여러 API의 복합 활용을 다루지 않음.
  3. **추론 능력 부족**: CoT나 ReACT 방식이 복잡한 지시를 처리하기엔 탐색 공간이 너무 좁고, 오류 전파(error propagation) 문제가 있음.

### 0.2. 핵심 아이디어 (Core Idea)

**핵심 한 줄**: ChatGPT를 활용해 16,000개 이상의 실제 API 데이터셋(ToolBench)을 자동 구축하고, 트리 기반 탐색 알고리즘(DFSDT)으로 LLaMA를 학습시켜 ChatGPT 수준의 도구 사용 능력을 오픈소스 LLM에 부여한다.

**(1) ToolBench — 대규모 실제 API 데이터셋**
* 정의: RapidAPI Hub에서 수집한 **3,451개 도구, 16,464개 실제 RESTful API**를 포함하는 instruction-tuning 데이터셋. 단일 도구(I1), 동일 카테고리 다중 도구(I2), 동일 컬렉션 다중 도구(I3) 세 유형의 지시를 포함. 총 126,486개의 (지시, 솔루션 경로) 쌍, 469,585회의 실제 API 호출 수행.
* 왜 필요한가: 기존 데이터셋 대비 규모와 다양성이 압도적으로 크고(API 수 기준 30x 이상), 실제 API 호출·응답까지 포함하여 현실적인 훈련 환경을 제공.
* 비유: 교과서 문제 풀이(기존 방식)가 아니라 실제 직장에서 다양한 업무 시스템을 써보며 익히는 방식.

**(2) DFSDT (Depth-First Search-based Decision Tree) — 향상된 추론 전략**

<p align='center'>
<img src="figs/ToolLLM/fig_02.jpeg" alt="Figure 2: ToolBench의 다양한 방법의 Pass Rate / Win Rate 비교" width="500"/>
</p>

* 정의: 모델이 여러 추론 경로를 탐색하고, 잘못된 경로는 포기(backtrack)하며 새로운 경로를 탐색할 수 있도록 하는 결정 트리 알고리즘. 각 노드가 "Thought + API Name + Parameters"를 포함하며, 오류 발생 시 "Finish by Giving Up"을 통해 새 노드를 확장한다.
* 왜 필요한가: 기존 CoT/ReACT는 단일 경로만 탐색하여 초기 오류가 전체 추론을 망치는 오류 전파 문제가 있고, 탐색 공간이 제한적. DFSDT는 여러 경로를 평가하여 복잡한 지시도 해결 가능.
* 비유: 미로에서 막다른 길을 만나면 되돌아가서 다른 경로를 시도하는 DFS 미로 탐색 전략.

**(3) ToolEval — 자동 평가 시스템**
* 정의: ChatGPT 기반의 두 가지 지표: (1) **Pass Rate**: 제한된 예산 내에 지시를 성공적으로 수행한 비율, (2) **Win Rate**: 두 솔루션 경로 중 어느 것이 더 좋은지 ChatGPT가 판단.
* 왜 필요한가: RapidAPI의 API가 시간에 따라 변하고, 유효한 솔루션 경로가 무한히 다양하기 때문에 고정된 정답 레이블을 만들 수 없음. 인간 평가자와 **87.1%(pass rate) / 80.3%(win rate)**의 높은 일치도를 달성.

**(4) Neural API Retriever — 대규모 API 풀에서 자동 API 추천**
* 정의: Sentence-BERT 기반의 dense retriever로, 사용자 지시를 입력받아 16,000+ API 풀에서 관련 API를 추천. 대조학습(contrastive learning)으로 훈련.
* 왜 필요한가: 사용자가 16,000개 API 중 적합한 것을 수동으로 선택하는 것은 비현실적. 자동 API 추천으로 실제 배포 가능한 pipeline 완성.

### 0.3. 효과 (Effects)

* **오픈소스 LLM의 도구 사용 능력 대폭 향상**: LLaMA-2 7B를 ToolBench로 파인튜닝한 ToolLLaMA가 ChatGPT와 comparable한 성능 달성. Text-Davinci-003, Claude-2를 능가.
* **DFSDT의 일반적 우월성**: 모든 LLM(GPT-4, ChatGPT, Claude-2 등)에서 DFSDT가 ReACT를 크게 능가. ChatGPT+DFSDT는 pass rate 기준으로 GPT-4+ReACT를 상회.
* **OOD 일반화**: ToolBench와 완전히 다른 도메인의 APIBench에서도 Gorilla(APIBench 전용 훈련 모델)와 comparable한 성능 달성.
* **API Retriever의 효과**: 오라클 API 대신 검색된 API를 사용했을 때 오히려 성능이 향상 (더 적합한 API를 찾아내기 때문).

### 0.4. 결과 (Results)

**메인 실험 (Table 4, ToolBench 평균)**:

| 모델 | 방법 | 평균 Pass Rate | 평균 Win Rate |
|------|------|:------------:|:-----------:|
| GPT-4 | DFSDT | **70.4** | **71.1** |
| ChatGPT | DFSDT | 64.3 | 64.8 |
| **ToolLLaMA** | **DFSDT** | **64.4** | **57.2** |
| **ToolLLaMA** | **DFSDT+Retriever** | **67.3** | **63.1** |
| Claude-2 | DFSDT | 43.5 | 22.6 |
| Text-Davinci-003 | DFSDT | 34.4 | 6.8 |
| ChatGPT | ReACT | 40.2 | — (baseline) |
| Vicuna/Alpaca | ReACT/DFSDT | 0.0 | 0.0 |

핵심: ToolLLaMA-DFSDT (pass rate 64.4%)는 ChatGPT-ReACT (40.2%)를 크게 능가하고, ChatGPT-DFSDT (64.3%)와 거의 동등하다. Retriever 사용 시 오히려 67.3%로 향상.

**API Retriever 성능 (Table 2, NDCG@5)**:

| 방법 | I1 | I2 | I3 |
|------|:--:|:--:|:--:|
| BM25 | 18.4 | 57.5 | 84.2 |
| Ada (OpenAI) | 49.6 | 78.0 | — |
| **Ours** | **89.7** | **87.1** | **68.2** |

**DFSDT vs ReACT (Table 3, ChatGPT 기준 Pass Rate 평균)**:

| 방법 | 평균 Pass Rate |
|------|:-----------:|
| ReACT | 35.3 |
| ReACT@N (동일 비용) | 44.5 |
| **DFSDT (ours)** | **63.8** |

**OOD 일반화 — HuggingFace (Table 5)**:

| 모델+Retriever | Hallucination (↓) | AST Accuracy (↑) |
|--------------|:----------------:|:---------------:|
| ToolLLaMA + Our Retriever | 10.60 | 88.80 |
| Gorilla-ZS + BM25 | 46.90 | 44.36 |
| Gorilla-RS + BM25 | 6.42 | 89.27 |

ToolLLaMA는 ToolBench와 완전히 다른 도메인임에도 Gorilla-RS와 거의 동등한 AST accuracy 달성.

### 0.5. 상세 동작 방식 (How It Works)

**[전체 파이프라인 흐름]**: 데이터 구축 → 모델 학습 → 추론

```
[RapidAPI Hub]
    │
    ▼ (필터링: 53,190 APIs → 16,464 APIs / 10,853 tools → 3,451 tools)
[API Collection: 3,451 도구, 16,464 APIs]
    │
    ▼ (ChatGPT로 지시 생성, I1/I2/I3 세 유형)
[Instruction Generation: ~200k (지시, 관련API) 쌍]
    ├── I1 (단일 도구): 87,413개
    ├── I2 (동일 카테고리 다중 도구): 84,815개
    └── I3 (동일 컬렉션 다중 도구): 25,251개
    │
    ▼ (DFSDT로 솔루션 경로 탐색, 469,585회 실제 API 호출)
[Solution Path Annotation: 126,486 (지시, 솔루션경로) 쌍]
    │
    ├── [LLaMA-2 7B SFT] → ToolLLaMA (컨텍스트 8192 토큰)
    └── [Sentence-BERT 대조학습] → API Retriever
         │
         ▼ (추론 시)
[사용자 지시] → [API Retriever: Top-5 API 추천] → [ToolLLaMA DFSDT] → [최종 답변]
                                                              ↑
                                              [ToolEval: Pass Rate / Win Rate 자동 평가]
```

**Step 1. API 수집 및 필터링**:
- RapidAPI Hub에서 49개 카테고리, 500+ 컬렉션의 API를 수집.
- 기능 테스트, 응답 품질 평가로 필터링 → 3,451개 도구, 16,464개 고품질 API 확보.
- 각 API 문서: 이름, 설명, HTTP 메서드, 파라미터, 코드 스니펫, 예시 응답 포함.

**Step 2. 지시 생성 (세 유형)**:
- I1 (단일 도구): 각 도구마다 해당 도구의 API만으로 완결되는 지시 생성.
- I2 (동일 카테고리 다중 도구): 같은 카테고리의 2-5개 도구를 조합. 카테고리 내 기능 유사성 활용.
- I3 (동일 컬렉션 다중 도구): 같은 컬렉션 내 도구 조합으로 더 세밀한 주제 기반 조합.
- ChatGPT에게 12(단일)/36(다중)개의 전문가 작성 seed 예시를 in-context로 제공.

**Step 3. DFSDT 솔루션 경로 탐색**:
- 각 스텝: "Thought: ... → API Name: ... → Parameters: {...} → Observation: {응답}"
- 오류/실패 발생 시: "Finish by Giving Up" 함수 호출 → 새 노드 확장 (DFS 백트래킹).
- 성공 시: "Finish with Final Answer" 함수 호출 → 솔루션 경로 저장.
- ReACT@N (동일 비용) 대비 DFSDT가 현저히 높은 pass rate 달성.

**Step 4. ToolLLaMA 학습**:
- LLaMA-2 7B에 SFT 적용. 126,486개 (지시, 솔루션 경로) 쌍으로 학습.
- Positional Interpolation으로 컨텍스트 길이 4096 → 8192 토큰 확장.
- API 응답 압축: 긴 응답은 ChatGPT로 핵심 정보만 추출하여 길이 단축.

**Step 5. 추론 (Inference)**:
- 사용자 지시 입력 → API Retriever가 16,464개 API 풀에서 Top-5 추천.
- ToolLLaMA가 DFSDT 전략으로 다중 라운드 API 호출 후 최종 답변 도출.
- ToolEval이 Pass Rate / Win Rate 두 지표로 자동 평가.

---

## 1. Introduction

현재 오픈소스 LLM들은 ChatGPT, GPT-4 같은 상용 모델과 비교할 때 도구 사용(tool use) 능력에서 큰 차이를 보인다. Instruction tuning이 언어 능력 중심으로만 이루어져, 실제 API를 호출하고 복잡한 멀티스텝 작업을 수행하는 능력이 크게 부족하다.

이 논문은 다음 세 가지 핵심 문제를 해결한다:
1. **어떻게 다양하고 현실적인 tool-use 데이터를 자동 구축할 수 있는가?** → ToolBench (3단계 자동 구축 파이프라인)
2. **복잡한 멀티스텝 추론을 어떻게 개선할 수 있는가?** → DFSDT (DFS 기반 결정 트리)
3. **실제 환경에서 대규모 API 풀을 어떻게 효과적으로 활용할 수 있는가?** → Neural API Retriever

---

## 2. Dataset Construction

### 2.1. API Collection

RapidAPI Hub에서 초기 53,190개 API를 수집 후, 기능 테스트와 응답 품질 평가를 통해 16,464개의 고품질 API만 유지. 각 API는 풍부한 메타데이터(기능 설명, 파라미터, 코드 예시, 응답 예시)를 포함하여 zero-shot 방식으로도 활용 가능하도록 설계.

RapidAPI 계층 구조:
- **카테고리 (Category)**: 49개 coarse-grained 카테고리 (sports, finance, weather 등)
- **컬렉션 (Collection)**: 500+ fine-grained 컬렉션 (Chinese APIs, database APIs 등)
- **도구 (Tool)**: 여러 API 엔드포인트를 포함하는 서비스 단위 (3,451개)
- **API**: 개별 엔드포인트 (16,464개)

### 2.2. Instruction Generation

<p align='center'>
<img src="figs/ToolLLM/fig_03.jpeg" alt="Figure 3: RapidAPI 계층 구조(좌)와 지시 생성 과정(우)" width="700"/>
</p>

세 유형의 지시를 생성하여 단일/다중 도구 시나리오를 모두 커버:

| 유형 | 설명 | 인스턴스 수 |
|------|------|:---------:|
| I1 (단일 도구) | 하나의 도구 API만 사용 | 87,413 |
| I2 (동일 카테고리 다중 도구) | 같은 카테고리 2-5개 도구 조합 | 84,815 |
| I3 (동일 컬렉션 다중 도구) | 같은 컬렉션 2-5개 도구 조합 | 25,251 |

프롬프트 구성: (1) 태스크 설명, (2) 각 API 문서, (3) 3개 seed 예시 (12/36개 전문가 작성 예시에서 랜덤 샘플링).

### 2.3. Solution Path Annotation with DFSDT

<p align='center'>
<img src="figs/ToolLLM/fig_04.png" alt="Figure 4: DFSDT vs CoT/ReACT 비교(좌), 솔루션 경로 탐색 과정(우)" width="700"/>
</p>

기존 ReACT의 두 가지 한계를 DFSDT로 극복:
- **오류 전파 (Error Propagation)**: 초기 실수가 이후 추론 전체를 망침 → DFSDT는 잘못된 경로를 포기하고 새 경로 탐색
- **제한된 탐색 (Limited Exploration)**: 단일 경로만 탐색 → DFSDT는 결정 트리로 다양한 경로 평가

DFSDT 구현 세부:
- 각 노드: "Thought → API Name → Parameters → Observation" 형식
- 실패 노드: "Finish by Giving Up" 함수로 백트래킹, 새 노드는 이전 노드들의 정보를 참고해 다양하게 생성
- 성공 노드: "Finish with Final Answer" 함수로 솔루션 경로 완성
- DFS 선택 이유: 유효 경로 하나만 찾으면 되므로 BFS보다 비용 효율적

결과: 126,486개 솔루션 경로 확보 (ChatGPT 기준 DFSDT pass rate 63.8% vs ReACT 35.3%)

---

## 3. Experiments

### 3.1. ToolEval — 자동 평가

ChatGPT 기반 자동 평가 시스템으로 두 가지 지표 제공:
- **Pass Rate**: 제한된 API 호출 예산 내 지시 성공 완수 비율
- **Win Rate**: ChatGPT 평가자가 두 솔루션 중 더 나은 것을 선택 (50% 이상이면 ChatGPT-ReACT보다 우수)

인간 평가와의 일치도: Pass Rate 87.1%, Win Rate 80.3% → 신뢰할 수 있는 자동 평가 시스템으로 검증됨.

### 3.2. API Retriever 성능

Sentence-BERT dense retriever가 BM25, OpenAI Ada 임베딩 대비 NDCG@1, NDCG@5 모두에서 일관되게 우수. 특히 I1(단일 도구)에서 NDCG@5 = 89.7로 매우 높은 성능.

### 3.3. Main Experiments

**핵심 발견**:
1. Vicuna/Alpaca는 pass rate = 0% — 기존 instruction tuning의 tool-use 도메인 한계 확인
2. 모든 모델에서 DFSDT가 ReACT를 크게 능가 (ChatGPT+DFSDT가 GPT-4+ReACT를 pass rate 기준 상회)
3. ToolLLaMA-DFSDT는 ChatGPT-DFSDT와 동등한 성능 (7B 오픈소스 모델로 달성)
4. ToolLLaMA+Retriever는 오라클 API 사용보다 오히려 성능 향상 (더 적합한 API 선택)

### 3.4. OOD Generalization (APIBench)

ToolBench와 완전히 다른 도메인(TorchHub, TensorHub, HuggingFace)에서 테스트:
- ToolLLaMA는 해당 도메인에서 전혀 훈련하지 않았음에도 Gorilla(APIBench 전용 훈련)와 comparable한 성능
- 특히 Gorilla-ZS+BM25 대비 큰 폭 우위 (HuggingFace: 88.80 vs 44.36 AST accuracy)
- Hallucination rate: ToolLLaMA+Retriever 10.60% vs Gorilla-ZS+BM25 46.90%

---

## 4. Related Work

| 분야 | 주요 연구 | ToolLLM과의 관계 |
|------|----------|----------------|
| Tool Learning | Toolformer, HuggingGPT, RestGPT | 실제 API 다양성·복잡도에서 크게 확장 |
| Instruction Tuning | Alpaca, Vicuna, Self-Instruct | 언어 능력 위주 → tool-use 도메인으로 확장 |
| Decision Making | ReACT, Reflexion, ToT | DFSDT는 실제 무한 API 환경에 특화된 탐색 |
| API Datasets | Gorilla/APIBench, API-Bank | 규모·다양성·실제 API 여부에서 ToolBench가 압도 |

DFSDT와 Tree-of-Thought(ToT)는 유사한 아이디어이나, ToT는 Game of 24, Crosswords 같은 단순 태스크를 대상으로 하는 반면 DFSDT는 무한한 결정 공간을 가진 실제 API 호출 환경에 특화되어 구현 방식이 크게 다르다.

---

## 5. Conclusion

ToolLLM은 세 가지 핵심 기여를 제시한다:
1. **ToolBench**: 16,464개 실제 RESTful API, 세 유형의 지시(단일/다중 도구), 126,486개 솔루션 경로를 포함하는 대규모 tool-use 데이터셋.
2. **DFSDT**: 복잡한 멀티스텝 추론을 위한 DFS 기반 결정 트리 알고리즘. ReACT 대비 현저한 성능 향상.
3. **ToolLLaMA + Neural API Retriever**: ChatGPT와 comparable한 성능의 오픈소스 LLM 및 자동 API 추천 시스템.

코드, 모델, 데모: [https://github.com/OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench)

---

## 6. 사전 지식 (Prerequisites)

### 필수
- **LLM Instruction Tuning**: Alpaca/Vicuna 방식의 SFT 원리
- **ReACT**: Thought-Action-Observation 루프
- **REST API 기초**: HTTP 메서드, JSON 응답, 엔드포인트 개념
- **Transformer/LLaMA**: 기본 아키텍처 이해

### 권장
- **Chain-of-Thought (CoT)**: 중간 추론 단계 생성 기법
- **Dense Retrieval**: Sentence-BERT, 대조학습 원리
- **NDCG**: 검색 평가 지표 이해
- **ChatGPT Function Calling**: API를 함수로 등록하는 기능

### 심화
- **Tree of Thoughts (ToT)**: DFSDT와의 비교 이해를 위해
- **Gorilla/APIBench**: OOD 평가 비교 대상 이해
- **Reflexion**: DFSDT가 확장하는 기법
- **Positional Interpolation**: 컨텍스트 길이 확장 기법 (Chen et al., 2023)
