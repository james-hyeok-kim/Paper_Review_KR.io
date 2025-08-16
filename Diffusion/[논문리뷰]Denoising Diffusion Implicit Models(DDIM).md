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
* $q(x_t|x_{t-1}) := N(x_t;\sqrt{1- \beta_{t}}x_{t-1},\beta_{t}I)$

#### Reverse Process $p_{\theta}$
* $p_{\theta}(x_{t-1}|x_t) :=  N (x_{t-1};\mu_{\theta}(x_t,t),\sum_{\theta}(x_t,t))$



### 3.1 DDIM의 순방향 과정 일반화 (Generalization of the Forward Process)
반면, DDIM은 비마르코프(Non-Markovian) 과정을 도입합니다. 이 새로운 과정은 다음과 같은 두 가지 확률 분포로 정의됩니다.

1. $q_σ(x_t∣x_0)$: 원본 데이터 $x_0$로부터 임의의 단계 t의 노이즈 이미지 $x_t$를 샘플링하는 분포, 이는 DDPM과 동일하게 가우시안 분포로 정의

```math
q_σ(x_t∣x_0)=N(x_t;\sqrt{α_t}x_0,(1−α_t)I)
```

2. $q_σ(x_{t−1}∣x_t,x_0)$: 현재 상태 $x_t$와 원본 데이터 $x_0$가 주어졌을 때, 이전 단계 $x_{t-1}$을 추론하는 분포입니다. 이 분포는 다음과 같이 정의됩니다:

```math
q_σ(x_{t−1}∣x_t,x_0)=N(\sqrt{α_{t−1}}x_0 + \sqrt{1−α_{t−1}−σ_t^2}⋅\frac{x_t−\sqrt{α_t}x_0}{\sqrt{1−α_{t}}},σ_t^2I)
```

$σ_t$는 새로운 하이퍼파라미터로, 생성 과정의 확률성(stochasticity)을 조절하는 역할을 합니다.

이 수식은 $x_t$와 $x_0$가 주어지면 $x_{t-1}$이 결정되므로, $x_{t-1}$은 $x_t$뿐만 아니라 $x_0$에도 의존하는 비마르코프적인 성격을 갖습니다.


### 3.2 동일한 학습 목표 (Same Training Objective)
논문의 핵심은 DDIM이 도입한 이 새로운 비마르코프 순방향 과정이, DDPM에서 사용된 노이즈 예측 손실 함수와 동일한 학습 목표를 공유한다는 점을 수학적으로 증명한 것입니다.

DDIM의 변분 하한(variational lower bound) 목적 함수 $J_\sigma(\epsilon_\theta)$는 DDPM의 단순화된 노이즈 예측 목적 함수 $L_\gamma(\epsilon_\theta)$와 상수 C만큼 차이가 있을 뿐, 본질적으로 동일하다는 것을 증명했습니다 (Theorem 1)
  
