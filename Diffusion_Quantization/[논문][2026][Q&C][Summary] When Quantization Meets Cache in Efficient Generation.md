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

* 캘리브레이션 데이터 열화 — 캐시 사용 시 PTQ 캘리브레이션 데이터의 샘플 간 코사인 유사도가 급격히 상승해 다양성이 감소
* Exposure Bias 증폭 — 양자화와 캐시를 함께 쓸 때 분산(variance) 드리프트가 발생해 오차가 누적됨 (각각 단독 사용 시에는 미미)

### 0.2. Main Idea

* TAP (Temporal-Aware Parallel Clustering) — 공간적 유사도와 시간적 유사도를 결합한 병렬 클러스터링으로 캘리브레이션 샘플을 효율적으로 선정. 복잡도를 O(n³)에서 O(rn)으로 감소
    * 유사도는 cosine similarity로 계산
    * K개로 clusting해서 Uniform Sampling
* VC (Variance Compensation) — 타임스텝별 보정 스케일 팩터 K를 분석적 해(closed-form)로 계산해 분산 드리프트를 추가 학습 없이 보정
    * 각 타임스탬프에서 중간 샘플의 분산을 Adpative 수정
    * rQNSR과 MSE를 결합하여 구성
    * variance를 scale( $K_t$ ), sample 자체에 곱함
    * $\tilde{x}_t = \mu_t + K_t \cdot (\hat{x}_t - \mu_t)$

#### 양자화(Quantization) 설정
* W8A8 설정을 중심
* 추가 설정: ImageNet $512\times512$ 해상도 실험에서는 가중치 4비트, 활성화 8비트(W4A8) 조합을 사용
* 방법론의 견고함을 보여주기 위해 4비트 및 6비트(W4A8, W6A8) 환경에서도 실험을 수행했습니다.
* 최신 모델인 FLUX.1이나 PixArt-$\Sigma$에 대해서는 INT W4A4 설정을 적용
* 양자화 방식: 모든 가중치는 채널별(Channel-wise), 활성화 함수는 텐서별(Tensor-wise)로 균일 양자화(Uniform quantizer)를 적용했습니다. 

#### 캐시(Cache) 메커니즘

* 건너뛰는 부분: 주로 Self-Attention(자기 주의 집중) 레이어와 MLP(Multi-Layer Perceptron) 레이어의 출력을 저장하고 재사용함으로써, 해당 레이어들의 중복 계산을 건너뜁니다.
* 동작 방식:캐시 간격 $N$을 설정하여, $mod(t, N) = 0$인 단계에서만 전체 순전파(Full forward pass)를 수행하고 특징을 업데이트(Caching)합니다.
    * 그 사이의 $N-1$ 단계 동안은 저장된 특징을 그대로 재사용하여 전체 모델 계산 과정을 생략(Bypassing)합니다.
* 핵심 원리: 고수준 특징(High-level features)은 단계별 변화가 적다는 점을 활용해 이를 고정하고, 저수준의 세부 사항만 업데이트하는 식으로 계산 효율을 높입니다.  

* 디퓨전 단계 축소(Step Reduction): 250단계에서 50단계로 줄여 5배 향상.
* 양자화(Quantization): W8A8 설정을 통해 1.96배 향상.
* 캐시 재사용(Cache Reuse): 위에서 언급한 1.28배 향상.
* 계산식: $5 \times 1.96 \times 1.28 \approx 12.54 \text{x}$ (측정 오차 포함 약 12.7x).  

### 0.3. 효과

* 최대 12.7× 가속 (스텝 감소 + 양자화 + 캐시 결합)
* FID 기준 단순 양자화+캐시 스택(13.67) 대비 Q&C(5.43)로 대폭 개선
* 추가 학습 불필요, 800개 캘리브레이션 샘플만으로 동작


### 0.4. 결과 Table

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

### 0.5. 모델

* DiT-XL/2 (256×256, 512×512)
* LDM (Latent Diffusion Model)
* FLUX.1, PixArt-Σ (부록 일반화 실험)
* Open-Sora (비디오 생성)
* Stable Diffusion (텍스트 조건부 생성)


### 0.6. 비교 대상

* PTQ4DM, Q-Diffusion, PTQD, RepQ, PTQ4DiT: 양자화 방법
* DeepCache, Learn-to-Cache, FORA: 캐시 방법
* QuantCache, CacheQuant: 동시기 양자화+캐시 통합 방법
* QAT + DeepCache: 양자화 인식 학습 방법

### 0.7. 데이터셋

* ImageNet (256×256, 512×512) — 메인 실험
* LSUN-Bedroom, LSUN-Church — 일반화 실험
* MJHQ-30K — FLUX.1, PixArt-Σ 평가
* MS-COCO, PartiPrompts — 텍스트 조건부 생성
* VBench — 비디오 생성 평가


### 0.8. 평가 지표

* FID (Fréchet Inception Distance) ↓ — 생성 품질
* sFID (Spatial FID) ↓ — 공간적 품질
* IS (Inception Score) ↑ — 다양성 및 품질
* Precision ↑ — 생성 정밀도
* Speed (×배 가속) ↑ — 추론 속도


---


---


---

---


---
