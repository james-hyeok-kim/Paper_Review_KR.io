# VISRAG: VISION-BASED RETRIEVAL-AUGMENTED GENERATION ON MULTI-MODALITY DOCUMENTS

저자 : 

Shi Yu1∗, Chaoyue Tang2∗, Bokai Xu2∗, Junbo Cui2∗, Junhao Ran3, Yukun Yan1†,

Zhenghao Liu4, Shuo Wang1, Xu Han1, Zhiyuan Liu1†, Maosong Sun1

1Department of Computer Science and Technology, Tsinghua University

2ModelBest Inc. 3Rice University 4Northeastern University

yus21@mails.tsinghua.edu.cn

발표 : ICLR 2025

논문 : [PDF](https://arxiv.org/pdf/2410.10594)

---

## 0. Summary

<p align = 'center'>
<img width="460" height="554" alt="image" src="https://github.com/user-attachments/assets/6c9297ec-86a1-4128-9a88-5649ed9913d2" />
</p>

<p algin = 'center'>
<img width="1009" height="587" alt="image" src="https://github.com/user-attachments/assets/75be0ab0-7f61-4b29-80bc-88427a11051f" />
</p>


### 0.1. 문제 (Problem)

* 기존의 RAG(TextRAG) 시스템의 한계.
    * 정보 손실: 문서에서 텍스트를 추출하기 위해 레이아웃 인식, OCR(광학 문자 인식) 등의 복잡한 파싱 과정을 거치는데, 이 과정에서 이미지, 도표, 레이아웃 등 중요한 시각적 정보가 유실되거나 오류가 발생합니다.
        * 레이아웃 인식(Layout Recognition), 광학 문자 인식(OCR), 텍스트 결합(Text Joining)과 같은 여러 단계가 얽힌 복잡한 과정
    * 멀티모달 대응 불가: 텍스트만 활용하기 때문에 이미지와 텍스트가 섞인 실제 환경의 복합 문서를 충분히 활용하지 못합니다.

### 0.2. 핵심 아이디어 (Core Idea)

* VisRAG는 문서에서 텍스트를 파싱하는 대신, 문서 페이지 자체를 이미지로 보고 직접 처리하는 VLM(Vision-Language Model) 기반의 파이프라인을 제안합니다.
* VisRAG-Ret (검색기):
    * 텍스트 쿼리와 문서 이미지를 동일한 임베딩 공간으로 매핑합니다.
    * 가중 평균 풀링(Weighted Mean Pooling)을 사용하여 임베딩 벡터를 생성합니다.
    * (여기서 $w_{i}$는 토큰 가중치, $h_{i}$는 은닉 상태)

$$v=\sum_{i=1}^{S}w_{i}h_{i}$$ 

* VisRAG-Gen (생성기):
    * 검색된 상위 k개의 문서 이미지를 VLM에 직접 입력하여 답변을 생성합니다.
    * 여러 이미지를 처리하기 위해 '페이지 이어붙이기(Concatenation)'나 '가중치 기반 선택(Weighted Selection)' 기법 등을 사용합니다.

### 0.3. 효과 (Effects)

* 데이터 보존: 파싱 과정을 생략함으로써 원본 문서의 모든 정보(레이아웃, 이미지, 글꼴 등)를 손실 없이 활용합니다.
* 효율적인 학습: 텍스트 기반 모델에 비해 적은 양의 데이터로도 더 높은 성능을 달성하는 데이터 효율성을 보여줍니다.
* 메모리 효율성: 기존 시각 검색 모델(예: ColPali) 대비 단일 벡터를 사용하여 훨씬 적은 메모리(4.5KB vs 256KB)를 차지하면서도 우수한 성능을 유지합니다.

### 0.4. 결과 (Results)

* 성능 향상: 기존 TextRAG 대비 20~40%의 상대적 성능 향상을 기록했습니다.
* 검색 능력: VisRAG-Ret은 최신 텍스트 및 시각 기반 검색기들을 뛰어넘는 성능을 보였으며, 특히 데이터가 적은 상황(Out-of-domain)에서 더 강력한 일반화 능력을 입증했습니다.
* 사례 연구: 텍스트 추출이 어려운 화려한 폰트나 레이아웃 정보가 중요한 문항에서 VisRAG가 훨씬 더 정확한 답변을 생성함을 확인했습니다.   

### 0.5. 검색 엔진의 메커니즘 (VisRAG-Ret)

* 가중 평균 풀링(Weighted Mean Pooling):
    * 생성형 VLM의 특성을 고려해, 마지막 레이어의 히든 스테이트(Hidden states) 중 뒤쪽 토큰에 더 높은 가중치를 주는 방식을 채택했습니다.
    * 인과적 어텐션(Causal Attention) 구조
    * 뒤쪽 토큰일수록 앞서 나온 모든 정보에 대한 요약본을 담고 있을 확률이 높기 때문에, 뒤쪽 은닉 상태(Hidden States)에 더 높은 가중치를 부여
$$v=\sum_{i=1}^{S}w_{i}h_{i}, \quad w_{i}=\frac{i}{\sum_{j=1}^{S}j}$$

* 가중치 분모 계산: $1 + 2 + 3 = 6$
* 각 토큰의 가중치:
    * 첫 번째 토큰 ( $w_1$ ): $1/6 \approx 16.7\%$
    * 두 번째 토큰 ( $w_2$ ): $2/6 \approx 33.3\%$
    * 세 번째 토큰 ( $w_3$ ): $3/6 \approx 50.0\%$  

* 단일 벡터 임베딩: 페이지당 수천 개의 벡터를 사용하는 다른 시각 검색 모델(예: ColPali)과 달리, 페이지당 하나의 벡터(4.5KB)만 사용하여 대규모 시스템 구축에 훨씬 유리합니다.

### 0.6. 멀티 페이지 생성 전략 (VisRAG-Gen)

* 페이지 이어붙이기 (Page Concatenation): 여러 이미지를 하나로 합쳐 단일 이미지 모델이 읽게 만듭니다.
    * 이 방식은 생성 단계에서 사용되며, 검색된 여러 개의 문서를 모델에 한 번에 입력하기 위한 전략
* 가중치 기반 선택 (Weighted Selection): 각 페이지별 답변의 당첨 확률(Perplexity)과 검색 점수를 조합해 최적의 답변을 고릅니다.
    * 단순히 점수가 높은 것을 고르는 것이 아니라, 검색 모델의 확신과 생성 모델의 확신을 곱해서 최종 승자를 정하는 방식입니다.
* 멀티 이미지 입력: 최신 VLM(MiniCPM-V 2.6 등)의 기능을 활용해 여러 장의 이미지를 동시에 넣고 교차 추론을 수행합니다.

### 0.74. 학습 및 운영 효율성

* 데이터 효율성: TextRAG 모델이 75,000개의 예제로 도달하는 성능을 VisRAG는 단 20,000개의 예제만으로 달성할 만큼 학습 효율이 좋습니다.
* 파이프라인 단순화: 레이아웃 분석, OCR, 텍스트 정제 등의 복잡한 단계(Cascade of processes)를 생략하므로 시스템 구조가 훨씬 간결해집니다.
* 속도: 오프라인 문서 처리 단계에서 파싱 과정을 건너뛰기 때문에, 전체 처리 시간을 약 58% 절감할 수 있습니다.  

---
