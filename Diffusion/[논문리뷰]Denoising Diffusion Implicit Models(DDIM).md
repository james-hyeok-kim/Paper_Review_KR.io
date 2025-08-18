# Denoising Diffusion Implicit Models

저자 : Jiaming Song, Chenlin Meng, Stefano Ermon

논문 : [PDF](https://arxiv.org/pdf/2010.02502)

일자 : Submitted on 6 Oct 2020  (CVPR, Computer Vision and Pattern Recognition)

Published as a conference paper at ICLR 2021

## Summary

* DDIM (Denoising Diffusion Implicit Models)은 Denoising Diffusion Probabilistic Models (DDPMs)의 샘플링 속도를 개선한 모델
* DDIM은 DDPM과 동일 훈련 방식을 사용
* DDIM은 비마르코프(non-Markovian) 사용, 샘플링에 필요한 단계 감소

<p align="center">
<img width="653" height="139" alt="image" src="https://github.com/user-attachments/assets/bd4fc49f-f068-4d6c-b6d6-316d3b6c5a31" />
</p> 



### DDPM 한계
* DDPM은 샘플 하나를 만들기 위해 수많은 마르코프 체인(Markov chain) 시뮬레이션

* 예를 들어, Nvidia 2080 Ti GPU를 기준으로 32x32 크기 이미지 5만 개를 생성하는 데 DDPM은 약 20시간이 걸리지만, GAN은 1분도 채 걸리지 않습니다.

### DDIM의 핵심 아이디어 (The Core Idea of DDIM)

* 훈련과 샘플링의 분리 (Decoupling Training and Sampling): DDIM의 핵심은 훈련에 사용되는 목표 함수가 마르코프 확산 과정뿐만 아니라 다양한 비마르코프 확산 과정에도 동일하게 적용될 수 있다는 점

* DDPM으로 이미 학습된 모델을 그대로 DDIM의 생성 과정을 구현할 수 있으며, 추가적인 재훈련이 필요 없습니다.

* 가속화된 샘플링 (Accelerated Sampling): DDIM은 생성 과정을 짧은 단계(short generative Markov chains)로 시뮬레이션할 수 있도록 설계되었습니다. 

* DDPM보다 10~50배 더 빠르게

* 샘플링에 필요한 단계 수(S)를 조절함으로써 계산량과 샘플 품질 사이의 균형을 맞출 수 있습니다.


### DDIM의 주요 특징 및 장점 (Key Features and Benefits of DDIM)
1. 샘플 품질 및 효율성 (Sample Quality and Efficiency):

* DDIM은 DDPM보다 적은 샘플링 단계(S)에서 더 우수한 샘플 품질

2. 샘플 일관성 (Sample Consistency):

* DDIM의 생성 과정은 결정론적(deterministic)이므로, 동일한 초기 잠재 변수(x_T)에서 이미지의 고수준 특징(high-level features)이 유사하게 유지됩니다.

3. 의미 있는 이미지 보간 (Semantically Meaningful Image Interpolation):

* DDIM의 일관성 덕분에, 잠재 공간($x_T$ 공간)에서 직접 보간(interpolation)을 수행하여 의미 있는 이미지 변환을 만들어낼 수 있습니다.

* GAN과 유사한 이 특성은 잠재 변수를 조작하여 생성되는 이미지의 고수준 특징을 직접 제어할 수 있게 해줍니다. DDPM은 확률적(stochastic) 생성 과정 때문에 이러한 보간이 어렵습니다.

4. 잠재 공간으로부터의 재구성 (Reconstruction from Latent Space):

* DDIM은 이미지($x_0$)를 잠재 변수($x_T$)로 인코딩한 후, 다시 $x_0$로 재구성하는 작업에 활용될 수 있습니다.

* DDIM은 DDPM보다 낮은 재구성 오류를 보이며, 이는 DDPM의 확률적 특성 때문에 불가능한 기능입니다.

### 1. 서론(Introduction):

DDPM의 배경과 장점(적대적 학습 불필요)을 소개하고, 느린 샘플링이라는 치명적인 단점을 다시 강조하며 DDIM을 제안하는 동기를 설명합니다. 

### 2. 배경(Background):

DDPM의 순방향 확산 과정과 역방향 생성 과정에 대한 수학적 정의를 설명합니다.

#### Forward Process (Diffusion Process) $q$

* $q(x_{1:T}|x_0) := \prod^T_{t=1}q(x_t|x_{t−1}), where q(xt|xt−1) := \mathcal{N} \left( \sqrt{\frac{\alpha_t}{\alpha_{t-1}}x_{t-1}}, \left(1 - \frac{α_t}{α_{t−1}} \right)I \right)$


#### Reverse Process $p_{\theta}$
* $q(x_t|x_0) := \int q(x_{1:t}|x_0)dx_{1:(t−1)} = \mathcal{N} (x_t;\sqrt{α_t}x_0,(1 − α_t)I)$
* $x_t =\sqrt{α_t}x_0 + \sqrt{1 − α_t}\epsilon, \\ where \\ \epsilon \sim \mathcal{N} (0, I) \\ (4)$

#### Loss

* $L_γ(\epsilon_θ) := \sum^T_{t=1}γ_t \mathcal{E}_{x0∼q(x_0),\epsilon_t \sim \mathcal{N}(0,I)} [\parallel \epsilon^{(t)}_θ(\sqrt{α_t}x_0 + \sqrt{1 − α_t} \epsilon_t) − \epsilon_t \parallel^2_2] \\ (5)$

* DDPM $\gamma = \frac{β_t^2}{2σ_t^2α_t(1−\bar{α}_t)}$
* $γ = 1$도 가능함을 알게됨(다른논문에서)

#### DDPM vs DDIM
* $DDPM \\ \bar{\alpha_t} = DDIM \\ \alpha_t$

### DDIM 핵심 아이디어 Non Markovian Process

#### 새로운 Forward조건부 분포

* $x_t =\sqrt{α_t}x_0 + \sqrt{1 − α_t}\epsilon, \\ where \\ \epsilon \sim \mathcal{N} (0, I) \\ (4)$

* $q(x_t|x_0) := \mathcal{N}(\sqrt{α_t}x_0,(1 − α_t)I)$ 의 분포를 따를 때,

(4)를 바탕으로 $x_{t-1}$ 예측하기

* $q_σ(x_{t−1}∣x_t,x_0)=\mathcal{N}(\sqrt{α_{t−1}}x_0 +  \sqrt{1−α_{t−1}−σ_t^2} \cdot \frac{x_t− \sqrt{α_t} x_0}{\sqrt{1−α_t}},σ_t^2I) \\ (7)$


* $\sigma_t : 확률을 조절하는 새로운 파라미터$


#### Reverse Process

예측된 $x_0 (Eq. 9)$

$$f_θ^{(t)}(x_t):=(x_t-\sqrt{1−α_t} \cdot ϵ_θ^{(t)}(x_t))/\sqrt{α_t}$$
​
이는 $x_t=\sqrt{α_t}x_0+\sqrt{1−α_t}ϵ$ 식을 $x_0$ 에 대해 정리한 것

모델이 예측한 노이즈 $\epsilon_\theta^{(t)}(x_t)$를 실제 노이즈 $ϵ_t$대신 사용


https://www.youtube.com/watch?v=n2P6EMbN0pc
