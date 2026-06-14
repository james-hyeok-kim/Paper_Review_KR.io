# NEZHA: A Zero-sacrifice and Hyperspeed Decoding Architecture for Generative Recommendations

저자: Yejing Wang, Shengyu Zhou, Jinyu Lu, Ziwei Liu, Langming Liu, Maolin Wang, Wenlin Zhang, Feng Li, Wenbo Su, Pengjie Wang, Jian Xu, Xiangyu Zhao

출간: WWW 2026 (The Web Conference 2026)

논문: [arXiv 2511.18793](https://arxiv.org/abs/2511.18793)

코드: [GitHub](https://github.com/Applied-Machine-Learning-Lab/WWW2026_NEZHA)

---

## 0. Abstract

* LLM 기반 생성형 추천 시스템은 학술 벤치마크에서 뛰어난 성능을 보이지만, **높은 추론 지연**으로 실서비스 배포에 심각한 장애가 됨
* **NEZHA**: 생성형 추천 전용 **Zero-sacrifice + Hyperspeed** 디코딩 아키텍처 제안
  - **Zero-sacrifice**: 기존 추천 품질 지표를 전혀 희생하지 않음
  - **Hyperspeed**: 추론 처리량을 대폭 향상
* 핵심: 기본 모델에 경량 자기회귀 Draft Head를 통합하고, 해시셋 기반 환각 제거 Verifier 사용
* **2025년 10월부터 타오바오(Taobao)에 실배포**, 일일 활성 사용자 수억 명 서빙

---

## 1. Introduction

**생성형 추천의 추론 병목**

LLM 기반 생성형 추천(TIGER 계열)의 자기회귀 디코딩은 본질적으로 순차적:
- 토큰 $t$를 생성하려면 $t-1$번째까지 완료되어야 함
- 아이템 Semantic ID가 $D$개 토큰이면, 아이템 하나를 생성하는 데 $D$번의 순차 연산
- 수백만 QPS(초당 쿼리 수)를 처리하는 실서비스에서 치명적

**기존 추론 가속 방법의 한계**

* **Speculative Decoding (일반 LLM용)**: 별도 Draft 모델 + Verifier → 추가 메모리/지연
  - EAGLE, Medusa 등: 추천 특화 구조 부재, 환각(hallucination) 아이템 생성 처리 미흡
* **병렬 디코딩**: 품질 저하 없이 병렬화하기 어려운 자기회귀 특성

**NEZHA의 핵심 관찰**: 추천 시스템의 유효 아이템 공간은 미리 알 수 있음 (카탈로그 고정) → 이를 활용한 효율적 검증 가능

---

## 2. Related Work

* **생성형 추천**: TIGER, GenRec, SpecGR — Semantic ID 기반 seq2seq 추천
* **Speculative Decoding**: EAGLE, Medusa, Draft & Verify — 일반 LLM 추론 가속
* **Beam Search 기반 추천**: 기존 생성형 추천에서 Top-K 후보 탐색에 Beam Search 사용
* **추천 시스템 서빙**: 실시간 추천 서빙의 지연 제약 (수십 ms 이내)

---

## 3. NEZHA 방법론

### 3.1 전체 아키텍처

NEZHA는 두 구성요소로 이루어짐:

```
[기본 생성 모델 (Primary Model)]
        |
  [통합 Draft Head]
        |
  [Hash-Set Verifier]
        |
  [최종 추천 리스트]
```

### 3.2 통합 Draft Head (Integrated Autoregressive Draft Head)

**기본 아이디어**: 외부 Draft 모델 대신, **기본 모델 내부에 경량 Draft Head를 직접 삽입**

* Draft Head는 기본 모델의 중간 레이어 은닉 상태(hidden state)를 입력으로 받아 후속 토큰을 예측
* 기본 모델의 1회 Forward pass에서 동시에 여러 후보 토큰 시퀀스를 Draft로 생성

**장점**:
- 별도 Draft 모델 불필요 → 추가 메모리 사용 없음
- 기본 모델과 파라미터 공유 → 의미론적 정합성 유지
- 기본 모델의 마지막 레이어에 Draft Head 추가로 최소 수정

### 3.3 입력 프롬프트 구조

추천 특화 프롬프트 설계로 seq2seq 생성 무결성 유지:
- 사용자 히스토리의 Semantic ID 시퀀스를 특수 포맷으로 구성
- Draft Head가 유효한 Semantic ID 구조를 생성하도록 유도

### 3.4 Hash-Set Verifier (환각 제거)

**문제**: Draft Head가 카탈로그에 존재하지 않는 Semantic ID를 생성할 수 있음 (환각)

**해결**: 유효한 Semantic ID 전체를 해시셋(Hash-Set)으로 사전 구축

$$\text{verify}(\text{draft\_id}) = \text{draft\_id} \in \mathcal{H}_{\text{catalog}}$$

* 검증 복잡도: $O(1)$ (해시셋 조회)
* 모델 프리 (Model-free): 추가 신경망 Verifier 불필요 → 지연 최소화
* 환각 아이템 즉시 폐기 → 품질 저하 방지

### 3.5 Zero-sacrifice 보장

Draft Head + Hash-Set Verifier 조합이 기존 품질을 보존하는 이유:
1. Draft Head는 기본 모델의 representations를 직접 활용 → 의미론적 동등성
2. Hash-Set Verifier는 유효 아이템만 통과시켜 추천 정확도 유지
3. 기본 모델의 최종 출력을 Ground-truth로 검증에 사용

---

## 4. Experimental Setup

### 평가 환경
* **온라인**: 타오바오 실서비스 (2025년 10월 이후 상시 배포)
* **오프라인**: 공개 데이터셋 + 내부 Taobao 데이터셋

### 비교 베이스라인
* 일반 LLM 가속: EAGLE, Medusa
* 표준 Speculative Decoding
* 기본 생성형 추천 (가속 없음)

### 평가 지표
* 추론 처리량 (Throughput): 초당 생성 아이템 수
* 추론 지연 (Latency): 요청당 응답 시간
* 추천 품질: HR@K, NDCG@K (오프라인) / CTR, GMV (온라인)

---

## 5. Experiments

### 핵심 결과

**추론 가속**
* NEZHA는 기존 생성형 추천 대비 대폭적인 처리량 향상 달성
* EAGLE, Medusa 등 일반 LLM 가속 기법 대비 추천 특화 최적화로 추가 이득

**Zero-sacrifice 검증**
* 오프라인 Recall@K, NDCG@K: 기본 모델과 동일 수준 유지
* 온라인 A/B: 기존 서비스 대비 품질 저하 없이 추론 속도 향상

### 타오바오 실배포 성과

* **배포 기간**: 2025년 10월부터 (6개월 이상 운영 중)
* **규모**: 수억 명 일일 활성 사용자(DAU) 서빙
* **비즈니스 임팩트**: 광고 매출 수십억(Billion) 위안 규모 기여

### Ablation: 각 컴포넌트 기여

| 설정 | 처리량 | 품질 |
|---|---|---|
| 기본 생성형 추천 | 1× | 기준 |
| + Draft Head (외부 모델) | 향상 | 미세 하락 |
| + 통합 Draft Head | 더 향상 | 유지 |
| + Hash-Set Verifier | 유지 | 환각 제거로 상승 |
| 전체 NEZHA | **최고 처리량** | **기준과 동일** |

---

## 의의

#### 1. 생성형 추천의 실서비스 추론 병목 해결
* 이론적으로 우수한 생성형 추천이 실서비스에서 겪는 가장 큰 장벽(지연)을 제거
* 타오바오 수억 명 규모에서 실증된 유일한 생성형 추천 가속 아키텍처

#### 2. 추천 도메인 특화 Speculative Decoding
* 일반 LLM의 Speculative Decoding(EAGLE, Medusa)을 추천 특성에 맞게 재설계
* 유한한 카탈로그 공간을 활용한 Hash-Set Verifier로 모델 프리 검증 달성

#### 3. Zero-sacrifice 원칙
* 추론 가속에서 일반적으로 발생하는 품질-속도 트레이드오프를 극복
* 추천 품질 지표를 전혀 희생하지 않는 엄격한 원칙 수립
