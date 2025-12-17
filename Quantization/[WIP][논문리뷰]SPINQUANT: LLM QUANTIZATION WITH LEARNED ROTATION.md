# SPINQUANT: LLM QUANTIZATION WITH LEARNED ROTATION

저자 : Zechun Liu∗ / Changsheng Zhao∗ / Igor Fedorov / Bilge Soran / Dhruv Choudhary / Raghuraman Krishnamoorthi / Vikas Chandra / Yuandong Tian / Tijmen Blankevoort

Meta

출간 : arXiv preprint arXiv:2405.16406, 2024

논문 : [PDF](https://arxiv.org/pdf/2405.16406)

---

## 1. Introduction

### 1. 핵심 문제: 양자화와 이상치(Outliers)

* 이상치(Outliers)의 존재: 가중치나 활성화 행렬에 존재하는 이상치 값들은 양자화 범위를 넓혀버려, 대부분의 일반적인 값들이 사용할 수 있는 유효 비트 수를 줄어들게 하고 큰 양자화 오차를 유발합니다.

* 기존 연구들은 가중치와 활성화값 간의 양자화 난이도를 조절하거나 혼합 정밀도(mixed-precision)를 사용하는 방식으로 이를 완화하려 했습니다.

### 2. 해결책: 학습된 회전 (Learned Rotations)

* 회전 불변성(Rotational Invariance): LLM의 가중치 행렬에 회전 행렬을 곱해도 전체 네트워크의 출력값은 변하지 않는 특성을 활용합니다.

* 회전의 효과: 무작위 회전(Random rotation)을 적용하면 이상치(outliers)가 줄어들고 분포가 평탄해져 양자화가 쉬워집니다.

* 무작위 회전 vs. 학습된 회전: 기존 연구(예: QuaRot)처럼 무작위 회전을 사용할 경우, 어떤 회전 행렬을 쓰느냐에 따라 성능 편차가 매우 큽니다 (제로샷 추론 과제에서 최대 13점 차이).

* 최적화: SpinQuant는 운에 맡기는 무작위 회전 대신, 양자화 손실을 최소화하는 회전 행렬을 직접 학습하고 최적화(Cayley SGD 사용)하여 일관된 고성능을 보장합니다.


### 3. 두 가지 SpinQuant 전략

* SpinQuant$_{no_had}$:
    * 학습된 회전 행렬을 기존 가중치에 흡수(merge)시키는 방식입니다.
    * 장점: 추론(Inference) 시 모델 구조를 변경하거나 추가 연산을 할 필요가 없습니다.

* SpinQuant$_{had}$:
    * 4-bit와 같은 극도로 낮은 비트의 활성화값(Activation)이나 KV 캐시(KV cache) 양자화가 필요한 경우 사용합니다.
    * 장점: 온라인 하다마드(Hadamard) 회전을 추가하여 내부 이상치를 더욱 효과적으로 제어합니다.

### 4. 주요 성과

* LLaMA-2 7B 모델: 가중치, 활성화값, KV 캐시를 모두 4-bit로 양자화했을 때, 전체 정밀도(Full precision) 모델과의 점수 차이를 불과 2.9점으로 좁혔습니다. 이는 LLM-QAT보다 19.1점, SmoothQuant보다 25.0점 더 높은 성능입니다.

* LLaMA-3 8B 모델: 양자화가 어려운 이 모델에서도 경쟁 기술인 QuaRot 대비 성능 격차를 최대 45.1% 줄였습니다


---

---

---
