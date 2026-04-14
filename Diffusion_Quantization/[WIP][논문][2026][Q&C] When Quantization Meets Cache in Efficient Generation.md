# Q&C: When Quantization Meets Cache in Efficient Generation

저자 : Xin Ding1 Xin Li1∗ Haotong Qin2 Zhibo Chen13∗

1University of Science and Technology of China 

2 ETH Zürich

3 Zhongguancun Academy

발표 : ICLR 2026 (International Conference on Learning Representations)

논문 : [PDF](https://openreview.net/pdf?id=AH7hbA7Zkk)

---

## 0. Summary

* 양자화와 캐시를 동시에 사용할 경우 성능이 급격히 저하되는 두 가지 핵심 원인을 찾아내고, 이를 해결하기 위한 하이브리드 가속 방법을 제안

### 0.1. Challenges

1) 캘리브레이션 데이터 효율성 저하 (Data Inefficiency)
    1) 사후 훈련 양자화(PTQ) 캘리브레이션 셋이 필요
    2) 캐시 메커니즘은 이전 단계의 연산 결과를 재사용
    3) 수집된 데이터들이 서로 너무 비슷해지는 '중복성' 문제
2) 노출 편향(Exposure Bias) 증폭
    1) 추론(Inference) 단계에서 모델이 이전에 생성한 '오차가 섞인 결과물'을 바탕으로 다음 단계를 예측하면서 오차가 눈덩이처럼 불어나는 현상
    2) 양자화와 캐시를 각각 쓸 때는 심하지 않던 이 편향이, 두 기술을 함께 쓸 때 급격히 심해짐
    3) 데이터의 분산(Variance)이 왜곡


### 0.2. Main Idea

1) TAP (Temporal-Aware Parallel Clustering)
    1) 가장 유니크하고 중요한 데이터만 골라내는 스마트 필터
    2) 데이터의 특징(공간적)뿐만 아니라 어느 타임스텝에서 왔는지(시간적)를 모두 고려하여 데이터를 그룹화(Clustering)
2) VC (Variance Compensation)
    1) 왜곡된 데이터의 분산을 실시간으로 교정해주는 볼륨 조절기
    2) 각 타임스텝마다 현재 데이터의 분산이 원래 의도된 분포에서 얼마나 벗어났는지 계산
    3) 생성 단계마다 발생하는 미세한 오차들을 즉각 보정해줌으로써 노출 편향에 의한 '에러 누적'을 막고 안정적인 생성


### 0.3. 결과 Table

* ImageNet $256 \times 256$ (W8A8) 

| 방법 (Method) | 타임스텝 (Steps) | 가속 배율 (Speedup) | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| DDPM (Full Precision) | 250 | $1\times$ | 4.53 | 17.93 | 278.50 | 0.8231 |
| PTQ4DIT | 250 | $2\times$ | 4.63 | 17.72 | 274.86 | 0.8299 |
| **Q&C (Ours)** | 250 | **$3.12\times$** | 4.68 | 17.84 | 268.65 | 0.8195 |
| DDPM (Full Precision) | 50 | $5\times$ | 5.22 | 17.63 | 237.8 | 0.8056 |
| PTQ4DIT | 50 | $10\times$ | 5.45 | 19.50 | 250.68 | 0.7882 |
| Baseline (PTQ + Cache) | 50 | $11.5\times$ | 13.67 | 25.86 | 189.65 | 0.7124 |
| **Q&C (Ours)** | 50 | **$12.7\times$** | **5.43** | **19.52** | **250.68** | **0.7895** |


---


---


---

---


---
