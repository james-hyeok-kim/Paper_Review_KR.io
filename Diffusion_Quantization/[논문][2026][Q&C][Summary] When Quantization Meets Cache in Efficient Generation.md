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
* VC (Variance Compensation) — 타임스텝별 보정 스케일 팩터 K를 분석적 해(closed-form)로 계산해 분산 드리프트를 추가 학습 없이 보정

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
