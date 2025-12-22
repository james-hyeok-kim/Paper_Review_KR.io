# Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers

저자 : Lei Chen1 Yuan Meng1 Chen Tang1,2 Xinzhu Ma2 Jingyan Jiang3 Xin Wang1 Zhi Wang1 Wenwu Zhu1

1Tsinghua University 2MMLab, CUHK 3Shenzhen Technology University

출간 : Proceedings of the Computer Vision and Pattern Recognition(CVPR), 2025

논문 : [PDF](https://arxiv.org/pdf/2406.17343)

---

## 1. Introduction

<p align = 'center'>
<img width="1127" height="575" alt="image" src="https://github.com/user-attachments/assets/3e894174-98e2-4dd1-9a33-1f653477cd6a" />
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

<p algin = 'center'>
<img width="564" height="502" alt="image" src="https://github.com/user-attachments/assets/c3b49b2d-c741-4057-bbce-0f67f8a9b7f5" />
</p>

### 관찰 1: 입력 채널 전반에 걸친 가중치와 활성화 값의 심각한 분산

* 입력 채널별 편차: DiT는 출력 채널(Output channel)보다 입력 채널(Input channel) 간의 분산이 훨씬 더 큽니다
* 활성화 값의 이상치(Outliers): 특정 채널에 매우 큰 활성화 값이 집중되어 있는 현상이 발견
    * 텐서 전체를 한꺼번에 양자화(Tensor-wise)할 경우, 이러한 소수의 이상치 때문에 나머지 일반적인 값들의 양자화 정밀도가 크게 훼손되는 문제가 발생

### 관찰 2: 타임스텝에 따른 활성화 값의 심각한 분포 변화(Distribution Shift)

* 시간적 변동성: 활성화 값의 분포는 디노이징의 각 타임스텝(Timestep)마다 크게 달라짐
* 샘플별 변동성: 입력되는 샘플(데이터)에 따라서도 상당한 차이를 보인다는 점을 추가로 발견


---

---

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

