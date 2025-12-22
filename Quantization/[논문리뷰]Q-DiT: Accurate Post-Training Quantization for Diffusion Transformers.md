# Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers

저자 : Lei Chen1 Yuan Meng1 Chen Tang1,2 Xinzhu Ma2 Jingyan Jiang3 Xin Wang1 Zhi Wang1 Wenwu Zhu1

1Tsinghua University 2MMLab, CUHK 3Shenzhen Technology University

출간 : Proceedings of the Computer Vision and Pattern Recognition(CVPR), 2025

논문 : [PDF](https://arxiv.org/pdf/2406.17343)

---

## 1. Introduction

<p align = 'center'>
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/3e894174-98e2-4dd1-9a33-1f653477cd6a" />
</p>

### 1. DiT 모델의 부상과 한계

* Q-DiT는 기존 UNet 기반 확산 모델에서 Diffusion Transformer (DiT) 구조로 변화함에 따라 발생하는 막대한 계산 비용과 성능 저하 문제를 해결하기 위해 제안된 새로운 사후 훈련 양자화(Post-Training Quantization, PTQ) 방법

### 2. 기존 양자화 방식의 문제점

* 특성 차이: DiT는 가중치와 활성화 값 모두에서 **공간적 분산(spatial variance)**이 크고, 활성화 값의 경우 시간적 분산(temporal variance) 또한 뚜렷하게 나타나는 독특한 특성을 보입니다.

### 3. Q-DiT의 핵심 솔루션

* 자동 양자화 세밀도 할당 (Automatic Quantization Granularity Allocation): 최적의 그룹 크기를 진화 알고리즘(evolutionary search)
* 샘플별 동적 활성화 양자화 (Sample-wise Dynamic Activation Quantization): 타임스텝(timestep)과 샘플에 따라 변하는 활성화 값의 분포에 맞춰 양자화 파라미터를 실시간(on-the-fly)으로 조정합니다.

---

## 2. Related Work

### 1. 모델 양자화 (Model Quantization)
* QAT
* PTQ

### 2. 트랜스포머 양자화 (Quantization of Transformers)

* 기존 기술의 한계
    * 활성화 값의 이상치(outlier)를 처리하기 위한 다양한 기법(LLM.int8(), Outlier Suppression 등)이 제안되었으나, 이러한 방식들은 확산 모델(Diffusion Model) 특유의 성질 때문에 DiT에 직접 적용하기 어렵습니다.

### 3. 확산 모델 양자화 (Quantization of Diffusion Models)

* 기존 연구들의 특징
    * PTQ4DM, Q-diffusion: 타임스텝별 활성화 값의 분산을 발견하고 재구성 기반 방식을 사용합니다.
    * PTQD, TDQ: 양자화 노이즈 보정이나 MLP 레이어를 통한 파라미터 예측 방식을 제안했습니다.
    * PTQ4DiT: DiT 전용 PTQ 방법으로, 채널별 강도 균형(CSB) 등을 통해 W4A8 양자화를 시도했습니다.

* Q-DiT의 차별점: 기존 방식들은 트랜스포머 구조의 특성과 디노이징 과정에서의 동적인 활성화 변화를 동시에 처리하지 못해 성능 저하가 발생하며, Q-DiT는 이를 통합적으로 해결하고자 합니다.


---

## 3. Observations of DiT Quantization

* 기존의 UNet 기반 양자화 방법론을 DiT에 적용했을 때 왜 성능이 급격히 떨어지는지, 그 근본적인 원인 두 가지를 분석

<p align = 'center'>
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/c3b49b2d-c741-4057-bbce-0f67f8a9b7f5" />
</p>

### 관찰 1: 입력 채널 전반에 걸친 가중치와 활성화 값의 심각한 분산

* 입력 채널별 편차: DiT는 출력 채널(Output channel)보다 입력 채널(Input channel) 간의 분산이 훨씬 더 큽니다
* 활성화 값의 이상치(Outliers): 특정 채널에 매우 큰 활성화 값이 집중되어 있는 현상이 발견
    * 텐서 전체를 한꺼번에 양자화(Tensor-wise)할 경우, 이러한 소수의 이상치 때문에 나머지 일반적인 값들의 양자화 정밀도가 크게 훼손되는 문제가 발생

### 관찰 2: 타임스텝에 따른 활성화 값의 심각한 분포 변화(Distribution Shift)

* 시간적 변동성: 활성화 값의 분포는 디노이징의 각 타임스텝(Timestep)마다 크게 달라짐
* 샘플별 변동성: 입력되는 샘플(데이터)에 따라서도 상당한 차이를 보인다는 점을 추가로 발견


---

## 4. Preliminary

* Q-DiT의 기반이 되는 균등 양자화(Uniform Quantization)의 정의와 수식적 정의

<p align = 'center'>
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/c7688424-82f7-4f86-8167-4d1f21510071" />
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/d637f24b-4ea7-4310-9c0c-83d8b624e61d" />
</p>

$$\hat{x} = Q(x; b) = s \cdot \left( \text{clip}\left( \left\lfloor \frac{x}{s} \right\rfloor + Z, 0, 2^b - 1 \right) - Z \right) \quad (1)$$


---

## 5. Method: Q-DiT

### 5.1. 자동 양자화 세밀도 할당 (Automatic Quantization Granularity Allocation)
* 그룹 양자화(Group Quantization)를 도입
    * 비단조성(Non-monotonicity) 발견: 일반적인 생각과 달리, 그룹 크기를 단순히 작게 한다고(세밀하게 나눈다고) 항상 성능이 좋아지지는 않는다는 사실을 발견했습니다.
        * 예를 들어, 그룹 크기를 128에서 96으로 줄였을 때 오히려 FID가 11.8% 악화되기도 했습니다.
    * FID/FVD를 최소화하는 레이어별 최적 그룹 크기 조합을 찾아냅니다.

$$L(g) = \text{FID}(R, G_g) \quad (4)$$
$$L(g) = \text{FVD}(R, G_g) \quad (5)$$

* 연산량(BitOps) 제약 조건 내에서 FID/FVD를 최소화하는 레이어별 그룹 크기 구성 $g$를 찾는 것이 목표

$$g^* = \arg \min L(g), \quad \text{s.t. } B(g) \le N_{bitops} \quad (6)$$

### 5.2. 샘플별 동적 활성화 양자화 (Sample-wise Dynamic Activation Quantization)

* 기존 방식의 한계: TDQ처럼 MLP로 파라미터를 예측하거나 모든 타임스텝의 파라미터를 미리 저장하는 방식은 메모리 오버헤드가 너무 큽니다(모델 크기의 약 39% 증가).
* 실시간(On-the-fly) 계산: 추론 과정에서 현재 들어온 샘플 $i$와 타임스텝 $t$의 활성화 값( $x_{i,t}$ )에서 즉석으로 최솟값과 최댓값을 구해 양자화 파라미터를 계산
* 연산자 융합(Operator Fusion): 이 계산 과정을 이전 연산 단계와 통합하여, 행렬 곱셈 연산량에 비해 무시할 수 있는 수준의 적은 비용으로 동적 양자화를 수행

* 동적 스케일
 
$$s_{i,t} = \frac{\max(x_{i,t}) - \min(x_{i,t})}{2^b - 1} \quad (7)$$

* 동적 제로 포인트

$$Z_{i,t} = \left\lfloor -\frac{\min(x_{i,t})}{s_{i,t}} \right\rfloor \quad (8)$$

<p align = 'center'>
<img width="400" height="800" alt="image" src="https://github.com/user-attachments/assets/16f869bc-a874-4551-b038-77a9d1a31aad" />
</p>

#### 알고리즘 1: 자동 양자화 세밀도 할당 과정
1. 임의의 그룹 크기 구성을 가진 인구(Population)를 초기화합니다
2. 각 구성에 대해 실제 샘플과 생성된 샘플 사이의 FID(이미지) 또는 FVD(비디오)를 계산하여 점수를 매깁니다
3. 성적이 좋은 상위 $K$개의 구성을 선택합니다
4. 교차(Crossover) 및 변이(Mutation) 연산을 통해 비트 연산량 제약 조건을 만족하는 새로운 구성을 생성하고 이 과정을 반복합니다.
5. 최종적으로 가장 성능이 좋은 최적의 그룹 크기( $g^{best}$ )를 모델에 적용합니다.

---

## 6. Experiments

### 6.1. 실험 설정 (Experimental Setup)

<p align = 'center'>
<img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/ddc57728-5115-46af-bd77-fd0eb6a96485" />
<img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/5a4dc818-e54c-4db2-bff9-22473dab2f40" />
</p>

* 이미지 생성: ImageNet $256\times256$ 및 $512\times512$ 해상도에서 DiT-XL/2 모델을 사용했습니다.
    * DDIM 샘플러(50, 100단계)와 분류기 없는 가이드(Classifier-free guidance)를 적용하여 평가했습니다.
* 비디오 생성: Open-Sora 프로젝트의 STDiT3 모델을 사용했으며, VBench 벤치마크의 16개 차원에서 성능을 측정했습니다.
* 비교 대상: PTQ4DM, RepQ-ViT, TFMQ-DM, PTQ4DiT 등 최신 양자화 기법들과 비교했습니다.

### 6.2. 주요 결과 (Main Results)

<p align = 'center'>
<img width="800" height="300" alt="image" src="https://github.com/user-attachments/assets/0aa86c65-fa1d-4e82-b11c-8f71ec2bcf53" />
</p>

* 이미지 생성 (Table 2)
    * W6A8 설정: Q-DiT는 FID 12.21, IS 117.75를 기록하며 전정밀도(FP) 모델에 근접한 손실 없는 압축(near-lossless compression)을 달성했습니다.
    * W4A8 설정: 다른 베이스라인들이 심각한 성능 저하를 보인 것과 달리, Q-DiT는 FID 15.76을 유지하며 압도적인 성능을 보였습니다

* 비디오 생성 (Table 3)
    * W4A8 설정에서 16개 지표 중 15개에서 베이스라인(G4W+P4A)을 능가했으며, 원본 모델 대비 성능 저하를 최소화했습니다. 이는 비디오 품질과 일관성을 잘 유지함을 보여줍니다.

### 6.3. 절제 연구 (Ablation Studies)

<p align = 'center'>
<img width="400" height="250" alt="image" src="https://github.com/user-attachments/assets/4d09a525-bb7e-489b-a2c7-ede65791a905" />
</p>

* Q-DiT의 각 구성 요소가 성능 향상에 기여하는 바를 분석했습니다 (Table 4)
    * RTN baseline: W4A8에서 FID 225.50으로 매우 낮은 성능을 보였습니다.
    * 그룹 양자화 (Group size 128): FID가 13.77로 대폭 개선되었습니다.
    * 샘플별 동적 활성화 양자화: FID 6.64로 추가적인 성능 향상을 이끌어냈습니다.
    * 자동 그룹 크기 할당: 최종적으로 FID 6.40을 달성하여 전정밀도 모델(5.31)에 매우 근접한 결과를 얻었습니다.

---

## 7. Conclusion

### 2. 성과 및 의의

* 탁월한 성능: 광범위한 실험을 통해 기존의 모든 베이스라인보다 뛰어난 성능을 입증했습니다.
* 저비트 고효율: 특히 ImageNet 256x256 데이터셋에서 모델을 W4A8로 양자화했음에도 불구하고, FID 점수 상승(성능 저하)이 단 1.09에 불과할 정도로 원본의 품질을 잘 보존했습니다.


### 3. 한계점 및 향후 연구 (Limitations and Future Work)

* 계산 비용 문제: 최적의 레이어별 그룹 크기 구성을 찾기 위해 사용되는 진화 알고리즘(Evolutionary Algorithm)이 계산적으로 비싸고 시간이 많이 소요된다는 점을 한계로 꼽았습니다.
* 최적화 계획: 향후 연구에서는 이 검색 과정을 더욱 효율적으로 최적화하여 전체적인 시스템 구축 비용을 낮출 계획입니다.


---

## Appendix

### 1. PTQD: Accurate Post-Training Quantization for Diffusion Models (NeurIPS 2023)
* 핵심 문제: 저비트 양자화 시 발생하는 '양자화 노이즈'가 확산 모델의 원래 노이즈 일정(variance schedule)과 충돌하여 생성 품질이 급격히 저하되는 문제를 해결하고자 했습니다.
* 주요 방법
    * 양자화 노이즈를 모델 출력과 상관관계가 있는 부분과 그렇지 않은 부분으로 분리합니다.
    * 분산 일정 교정(Variance Schedule Calibration)을 통해 상관관계가 없는 노이즈 부분을 보정하여 성능을 높였습니다.
* 특징: 기존 PTQ 방식보다 높은 품질의 샘플을 생성하면서도 계산량을 크게 줄였습니다.


### 2. TDQ: Temporal Dynamic Quantization for Diffusion Models (NeurIPS 2023)

* 핵심 문제: 확산 모델의 활성화(activation) 분포가 타임스텝에 따라 계속 변화하므로, 고정된 양자화 파라미터를 사용하면 성능이 떨어진다는 점에 주목했습니다.
* 주요 방법
    * MLP 레이어 활용: 아주 작은 크기의 다층 퍼셉트론(MLP)을 사용하여 각 타임스텝마다 최적의 활성화 양자화 파라미터를 동적으로 추정합니다.
    * 작동 원리
        * 타임스텝 기반 추정입력 데이터: 현재 디노이징 단계인 타임스텝( $t$ ) 정보를 입력으로 받습니다.
        * MLP의 역할: 아주 작은 크기의 MLP가 학습을 통해 각 타임스텝 $t$에 가장 적합한 양자화 파라미터(예: Scaling factor)를 결과값으로 내놓습니다.
        * 동적 적용: 고정된 값을 쓰는 대신, 매 스텝마다 MLP가 계산해준 파라미터를 활성화 값에 적용하여 양자화 오류를 줄입니다.
* 특징: 추론 시 추가적인 계산 오버헤드 없이 타임스텝별로 최적화된 양자화 간격을 제공합니다.


### 3. PTQ4DiT: Post-training Quantization for Diffusion Transformers (NeurIPS 2024)
* 핵심 문제: 기존 UNet 기반 방식이 아닌 **트랜스포머 기반 확산 모델(DiT)**에 특화된 최초의 PTQ 연구입니다. DiT에서 나타나는 극단적인 채널 값(outliers)과 시간적 변화 문제를 다룹니다.
* 주요 방법
    * 채널별 중요성 균형 조정(CSB, Channel-wise Salience Balancing): 가중치와 활성화 사이의 극단적인 값들을 재분배하여 양자화 오류를 줄입니다.
    * 스피어만 $\rho$ 기반 중요성 보정(SSC, Spearman's $\rho$-guided Salience Calibration): 시간적 변화를 반영하여 중요 채널을 보정합니다.
* 특징: DiT 모델에서 W4A8(가중치 4비트, 활성화 8비트) 양자화를 최초로 효과적으로 구현했습니다.

