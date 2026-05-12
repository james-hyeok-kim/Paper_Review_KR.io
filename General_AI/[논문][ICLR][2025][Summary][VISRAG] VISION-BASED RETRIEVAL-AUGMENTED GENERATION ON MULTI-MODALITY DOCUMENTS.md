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

---
