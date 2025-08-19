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

* $q(x_{1:T}|x_0) := \prod^T_{t=1}q(x_t|x_{t−1}),\\ where \\ q(xt|xt−1) := \mathcal{N} \left( \sqrt{\frac{\alpha_t}{\alpha_{t-1}}x_{t-1}}, \left(1 - \frac{α_t}{α_{t−1}} \right)I \right)$


#### Reverse Process $p_{\theta}$
* $q(x_t|x_0) := \int q(x_{1:t}|x_0)dx_{1:(t−1)} = \mathcal{N} (x_t;\sqrt{α_t}x_0,(1 − α_t)I)$
* $x_t =\sqrt{α_t}x_0 + \sqrt{1 − α_t}\epsilon, \\ where \\ \epsilon \sim \mathcal{N} (0, I) \\ (4)$

#### Loss

* $L_γ(\epsilon_θ) := \sum^T_{t=1}γ_t \mathcal{E}_{x0∼q(x_0),\epsilon_t \sim \mathcal{N}(0,I)} [\parallel \epsilon^{(t)}_θ(\sqrt{α_t}x_0 + \sqrt{1 − α_t} \epsilon_t) − \epsilon_t \parallel^2_2] \\ (5)$

* DDPM $\gamma = \frac{β_t^2}{2σ_t^2α_t(1−\bar{α}_t)}$
* $γ = 1$도 가능함을 알게됨(다른논문에서)
* DDIM에서는  $\gamma = 1$을 사용

#### DDPM vs DDIM
* $DDPM \\ \bar{\alpha_t} = DDIM \\ \alpha_t$

### DDIM 핵심 아이디어 Non Markovian Process

#### 새로운 Forward 조건부 분포 $q$

[Youtube](https://www.youtube.com/watch?v=n2P6EMbN0pc)

* $x_t =\sqrt{α_t}x_0 + \sqrt{1 − α_t}\epsilon, \\ where \\ \epsilon \sim \mathcal{N} (0, I) \\ (4)$

* $q_\sigma(x_t|x_0) := \mathcal{N}(\sqrt{α_t}x_0,(1 − α_t)I)$ 의 분포를 따를 때,

(4)를 바탕으로 $x_{t-1}$ 예측하기

$$q_σ(x_{t−1}∣x_t,x_0)=\mathcal{N}(\sqrt{α_{t−1}}x_0 +  \sqrt{1−α_{t−1}−σ_t^2} \cdot \frac{x_t− \sqrt{α_t} x_0}{\sqrt{1−α_t}},σ_t^2I) \\ (7)$$


* $\sigma_t$ : 확률을 조절하는 새로운 파라미터


##### (7) 유도과정

* $q_\sigma(x_t|x_0) = \mathcal{N}(\sqrt{α_t}x_0,(1 − α_t)I)$
* $q_\sigma(x_{t-1}|x_0) = \mathcal{N}(\sqrt{\alpha_{t-1}}x_0, (1-\alpha_{t-1}I)$
  * $p(x) = \mathcal{N}(x|\mu,\Lambda^{-1})$
  * $\Lambda : Lambda$
* $p(y|x) = \mathcal{N}(y|Ax + b, L^{-1})$
  * y가 x에 대한 선형 변환에 가우시안 노이즈가 더해진 형태 $y=Ax+b+\epsilon$
  * 이때 노이즈 $\epsilon$은 평균 0, 공분산 $L^{-1}$인 가우시안 분포 $\epsilon \sim \mathcal{N}(0, L^{-1})$
* $p(y) = \mathcal{N}(y|A\mu + b, L^{-1}+A\Lambda^{-1}A^{T})$

$p(y|x) = \mathcal{N}(y|Ax + b, L^{-1})$ 유도(공분산의 성질을 이용하여 y의 공분산을 계산)

```math
\begin{align}
Cov(X+c)&=Cov(X) (상수 벡터를 더해도 공분산은 변하지 않음) \\\\
Cov(X+Y)&=Cov(X)+Cov(Y) (X와 Y가 독립일 경우) \\\\
Cov(AX)&=A⋅Cov(X)⋅A^T \\\\
\end{align}
```
위 성질 사용
```math
\begin{align}
Cov(y)&=Cov(Ax+b+ϵ) \\\\
Cov(y)&=Cov(Ax+ϵ)  (상수 b는 공분산에 영향을 주지 않음) \\\\
Cov(y)&=Cov(Ax)+Cov(ϵ)  \\\\
Cov(y)&=A⋅Cov(x)⋅A^T  +Cov(ϵ) \\\\
Cov(x)&=Λ^{−1}(x의 공분산) \\\\
Cov(ϵ)&=L^{-1} (노이즈의 공분산) \\\\
Cov(y)&=AΛ^{−1}A^{T}+L^{−1} \\\\
\end{align}
```
 
```math
\begin{align}
p(y)=\mathcal{N}(y∣\underbrace{Aμ+b}_{평균}, \underbrace{L^{−1}+AΛ^{−1} A^{T}}_ {공분산})
\end{align}
```

* $p(y) \leftarrow q_\sigma(x_{t-1}|x_0)$
* $p(x) = \mathcal{N}(x|\mu, \Lambda^{-1})$
* $p(x) \leftarrow q_\sigma(x_t|x_0) = \mathcal{N}(\sqrt{\alpha_t}x_0, (1-\alpha_t)I)$
* $p(y|x) = \mathcal{N}(y|Ax+b,L^{-1})$
* $p(y|x) \leftarrow q_\sigma(x_{t-1}|x_t,x_0) = \mathcal{N} \left(\sqrt{a_{t-1}}x_0  + \sqrt{1-\alpha_{t-1}-\sigma^2_t} \cdot \frac{x_t - \sqrt{\alpha_t}x_0}{\sqrt{1-\alpha_t}} , \sigma_t^2 I \right)$
* $q_\sigma(x_{t-1}|x_0) = \mathcal{N}(y|A\mu + b, L^{-1}+A\Lambda^{-1}A^T)$


```math
\begin{align}
\mu &= \sqrt{\alpha_t}x_0 \\\\
\Lambda^{-1} &= (1-\alpha_t)I \\\\
A &= \sqrt{1-\alpha_{t-1}-\sigma^2_t} \cdot \frac{1}{\sqrt{1-\alpha_t}} \\\\
b &= \sqrt{\alpha_{t-1}}x_0 - \sqrt{1-\alpha_{t-1}-\sigma_t^2} \cdot \frac{\sqrt{\alpha_t}x_0}{\sqrt{1-\alpha_t}} \\\\
L^{-1} &= \sigma^2_tI
\end{align}
```

```math
\begin{align}
A\mu+b &= \sqrt{1-\alpha_{t-1}-\sigma^2_t} \cdot \frac{1}{\sqrt{1-\alpha_t}}\sqrt{\alpha_t}x_0 + \sqrt{\alpha_{t-1}}x_0 - \sqrt{1-alpha_{t-1}-\sigma_t^2} \cdot \frac{\sqrt{\alpha_t}x_0}{\sqrt{1-\alpha_t}} \\\
&= \sqrt{\alpha_t}x_0 
\end{align}
```

```math
\begin{align}
L^{-1} + A\Lambda^{-1}A^T &= \sigma_t^2I + \sqrt{1-\alpha_{t-1}-\sigma^2_t} \cdot \frac{1}{\sqrt{1-\alpha_t}} (1-\alpha_t)I \left(\sqrt{1-\alpha_{t-1}-\sigma^2_t} \cdot \frac{1}{\sqrt{1-\alpha_t}} \right)^T \\\\
&= \sigma_t^2I + (1-\alpha_{t-1}-\sigma_t^2)I \\\\
&= (1-\alpha_{t-1})I
\end{align}
```

* 최종식 유도
```math
\begin{align}
q_\sigma(x_t|x_0) &= \mathcal{N}(\sqrt{\alpha_t}x0, (1-\alpha_t)I) \\\\
q_\sigma(x_{t-1}|x_0) &= \mathcal{N}(\sqrt{\alpha_{t-1}}x_0,(1-\alpha_{t-1})I) \\\\
& 위에서 정의된 식 (DDPM) \\\\
q_\sigma(x_{t-1}|x_t,x_0) &= \mathcal{N}(x_{t-1}; \mu_q = A_tx_t+B_tx_0,\sigma^2_{*t}I) \;\; 로 정의 \\\\
p(y) &= \mathcal{N}(y|A\mu + b, L^{-1}+A\Lambda^{-1}A^{T}) \;\; 해당식을 활용하면 \\\\
\mu &= \sqrt{\alpha_t}x_0 \\\\
\Lambda^{-1} &= (1-\alpha_t)I \\\\
A &= \sqrt{1-\alpha_{t-1}-\sigma^2_t} \cdot \frac{1}{\sqrt{1-\alpha_t}} \\\\
&= A_t \\\\
b &= \sqrt{\alpha_{t-1}}x_0 - \sqrt{1-\alpha_{t-1}-\sigma_t^2} \cdot \frac{\sqrt{\alpha_t}x_0}{\sqrt{1-\alpha_t}} \\\\
b &= B_tx_0 \\\\
L^{-1} &= \sigma^2_tI \\\\
q_\sigma(x_{t-1}|x_0) &= \mathcal{N}(x_{t-1};\mu_q = \frac{\sqrt{1-\alpha_{t-1}-\sigma^2_t}}{\sqrt{1-\alpha_t}}\sqrt{\alpha_t}x_t + (\sqrt{\alpha_{t-1}} - \sqrt{1-\alpha_{t-1}-\sigma_t^2} \cdot \frac{\sqrt{\alpha_t}}{\sqrt{1-\alpha_t}})x_0, \sigma_{*t}^2I) \\\\
&= \mathcal{N}(x_{t-1};\mu_q = \sqrt{α_{t−1}}x_0 +  \sqrt{1−α_{t−1}−σ_t^2} \cdot \frac{x_t− \sqrt{α_t} x_0}{\sqrt{1−α_t}},σ_{*t}^2I) \;\; (7)
\end{align}
```

* Trained DDPM을 DDIM non-markovian에서 사용 가능 (retrain x )

$q_\sigma(x_t|x_{t-1},x_0) \neq q_\sigma(x_t|x_{t-1})$

#### Sampling

1. Goal: $x_t$에서 $x_{t-1}$을 만들고 싶다.

2. Problem: 이상적인 방법(q)은 우리가 모르는 원본 이미지($x_0$)나 실제 노이즈(ε)를 필요로 해서 못 쓴다.

3. Solution: q의 수식 형태를 본떠서, 실제 노이즈 ε를 신경망이 예측한 노이즈 $ε_θ$로 대체한 근사 모델 $p_θ$를 만든다.

4. Sampling: $p_θ$라는 정규분포에서 샘플 $x_{t-1}$을 뽑기 위해 리파라미터라이제이션 트릭 (결과 = 평균 + 표준편차 × 랜덤값)을 사용한다.


```math
\begin{align}
q_\sigma(x_{t-1}|x_0) &= \mathcal{N}(\sqrt{α_{t−1}}x_0 +  \sqrt{1−α_{t−1}−σ_t^2} \cdot \frac{x_t− \sqrt{α_t} x_0}{\sqrt{1−α_t}},σ_{*t}^2I) \;\; (7) \\\\
q_\sigma(x_t|x_0) &= \mathcal{N}(\sqrt{\alpha_t}x_0, (1-\alpha_t)I) \\\\
x_t &= \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\epsilon \\\\
x_0 &= \frac{x_t - \sqrt{1-\alpha_t}\epsilon}{\sqrt{\alpha_t}} \\\\
q_\sigma(x_{t-1}|x_t,x_0) &= \mathcal{N} \left(\sqrt{\alpha_{t-1}}\frac{x_t-\sqrt{1-\alpha_t}\epsilon}{\sqrt{\alpha_t}} + \sqrt{1-\alpha_{t-1}-\sigma^2_t}\epsilon,\sigma_t^2I \right) \\\\
p_\theta(x_{t-1}|x_t) &= \mathcal{N} \left(\sqrt{\alpha_{t-1}}\frac{x_t-\sqrt{1-\alpha_t}\epsilon_\theta}{\sqrt{\alpha_t}} + \sqrt{1-\alpha_{t-1}-\sigma^2_t}\epsilon_\theta,\sigma_t^2I \right) \\\\
x_{t-1} &= \sqrt{\alpha_{t-1}} \left(\frac{x_t-\sqrt{1-\alpha_t}\epsilon_\theta^{(t)}(x_t)}{\sqrt{\alpha_t}} \right) + \sqrt{1-\alpha_{t-1}-\sigma_t^2}\cdot \epsilon_\theta^{(t)}(x_t)+\sigma_t\epsilon_t \;\; (Reparameterization - DDIM) \\\\
x_{t-1} &= \frac{x_t}{\sqrt{\alpha_t}} - \frac{1-\alpha_t}{(\sqrt{1-\alpha_t})\sqrt{\alpha_t}}\epsilon_\theta^{(t)}(x_t) + \sigma_t\epsilon_t \;\; DDPM \; Sampling \; step \; (Comparison) \\\\
\end{align}
```

#### Impact of Variance in DDIM
DDIM에서 정의하는 Variance (12) 
$\sigma_t^2 = \frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$

```math
\begin{align}
& DDIM x_{t-1} 일반화 \\\\
x_{t-1} &= \sqrt{\alpha_{t-1}} \left( \underbrace{\frac{x_t-\sqrt{1-\alpha_t}\epsilon_\theta^t(x_t)}{\sqrt{\alpha_t}}}_{predicted \; x_0} \right) + \underbrace{\sqrt{1-\alpha_{t-1}-\sigma^2_t} \cdot \epsilon_\theta^{(t)}(x_t)}_{direction pointing \; to \; x_t} + \underbrace{\sigma_t\epsilon_t}_{random \; noise} \;\; (12) \\\\
&DDPM 정의 \\\\
\tilde{\beta}_t &= \frac{1-\alpha_{t-1}}{1-\alpha_t}\beta_t = \frac{1-\alpha_{t-1}}{1-\alpha_t} \left(1-\frac{\alpha_t}{\alpha_{t-1}} \right) \\\\
& 분산(Variance) = \sigma_t^2 \\\\
& DDPM = DDIM, \sigma_t^2 = \tilde{\beta_t} \\\\
\bar{\alpha_t} &= \alpha_t \cdot \bar{\alpha_{t-1}} \\\\
\sigma_t^2 &= \frac{1-\bar{\alpha_{t-1}}}{1-\bar{\alpha_t}} \left( 1- \frac{\bar{\alpha_t}}{\bar{\alpha_{t-1}}} \right) \\\\
&=\frac{1-\bar{\alpha_{t-1}}}{1-\bar{\alpha_t}} \left( 1- \frac{\alpha_t \cdot \bar{\alpha_{t-1}}}{\bar{\alpha_{t-1}}} \right)\\\\
&=\frac{1-\bar{\alpha_{t-1}}}{1-\bar{\alpha_t}} \left( 1- \alpha_t \right)\\\\
&=\frac{(1-\alpha_t)(1-\bar{\alpha_{t-1}})}{1-\bar{\alpha_t}} \\\\
\end{align}
```

* (12)식 정리
```math
\begin{align}
\sigma_t^2 &= \frac{(1-\alpha_t)(1-\bar{\alpha_{t-1}})}{1-\bar{\alpha_t}} \\\\
\end{align}
```

```math
\begin{align}
x_{t-1} &= \sqrt{\alpha_{t-1}} \left( \frac{x_t-\sqrt{1-\alpha_t}\epsilon_\theta^t(x_t)}{\sqrt{\alpha_t}} \right) + \sqrt{1-\alpha_{t-1}-\sigma^2_t} \cdot \epsilon_\theta^{(t)}(x_t) + \sigma_t\epsilon_t \;\; (12) \\\\
&= \frac{x_t}{\sqrt{\alpha_t}}-\epsilon_\theta^{(t)}(x_t) \left( \sqrt{\frac{1-\bar{\alpha_t}}{\alpha_t}} - \sqrt{(1-\bar{\alpha}_{t-1}) - \frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}} \right) +\sigma_t\epsilon_t \\\\
&= \frac{x_t}{\sqrt{\alpha_t}}-\epsilon_\theta^{(t)}(x_t) \left( \sqrt{\frac{1-\bar{\alpha_t}}{\alpha_t}} -\sqrt{\frac{(1-\bar{\alpha}_{t-1})(1-\bar{\alpha_t}) - (1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}} \right) + \sigma_t\epsilon_t \\\\
&= \frac{x_t}{\sqrt{\alpha_t}}-\epsilon_\theta^{(t)}(x_t) \left( \sqrt{\frac{1-\bar{\alpha_t}}{\alpha_t}} -\sqrt{\frac{(1-\bar{\alpha}_{t-1})((1-\bar{\alpha}_t) - (1-\alpha_t))}{1-\bar{\alpha}_t}}\right) + \sigma_t\epsilon_t \\\\
&= \frac{x_t}{\sqrt{\alpha_t}}-\epsilon_\theta^{(t)}(x_t) \left( \sqrt{\frac{1-\bar{\alpha_t}}{\alpha_t}} -\sqrt{\frac{(1-\bar{\alpha}_{t-1})(-\alpha_t\bar{\alpha}_{t-1} + \alpha_t)}{1-\bar{\alpha}_t}}\right) + \sigma_t\epsilon_t \\\\
\end{align}
```

```math
\begin{align}
x_{t-1} &= \frac{x_t}{\sqrt{\alpha_t}}-\epsilon_\theta^{(t)}(x_t) \left( \sqrt{\frac{1-\bar{\alpha_t}}{\alpha_t}} - \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{\sqrt{1-\bar{\alpha}_t}} \right) +\sigma_t\epsilon_t \\\\
&= \frac{x_t}{\sqrt{\alpha_t}}-\epsilon_\theta^{(t)}(x_t) \left( \frac{1-\bar{\alpha}_t-\alpha_t+\bar{\alpha}_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t}} \right) + \sigma_t\epsilon_t \\\\
&= \frac{x_t}{\sqrt{\alpha_t}}-\epsilon_\theta^{(t)}(x_t) \left( \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t}} \right) + \sigma_t\epsilon_t \\\\
\end{align}
```

```math
\begin{align}

& \text{DDPM  Sampling  Step} \\\\
x_{t-1} &= \frac{x_t}{\sqrt{\alpha_t}} - \frac{(1-\alpha_t)}{(\sqrt{1-\bar{\alpha}_t})\sqrt{\alpha_t}}\epsilon_\theta^{(t)}(x_t)+\sigma_t\epsilon_t \\\\
\end{align}
```

*  $\sigma_t^2$, DDPM = DDIM

$$\therefore q_\sigma(x_t|x_{t-1},x_0) = q_\sigma(x_t|x_{t-1})$$

DDPM Posterior와 같은 분산으로 Non-Markovian process(DDIM)이 Markovian forward prcess(DDPM)로 일반화, DDPM과 같아진다

#### Accelerated Sampling in DDIM

```math
\begin{align}
q_{\sigma}(x_{t-1}|x_0) &= \mathcal{N}(\sqrt{\alpha_{t-1}}x_0 + \sqrt{1-\alpha_{t-1}-\sigma_t^2} \cdot \frac{x_t- \sqrt{\alpha_t} x_0}{\sqrt{1-\alpha_t}}, \sigma_{t}^2 I) \;\; \text{(7) DDIM} \\\\
q(x_{t-1}|x_0) &= \mathcal{N}(\sqrt{\alpha_{t-1}}x_0, (1-\alpha_{t-1})I) \;\; \text{DDPM} \\\\
q_{\sigma,\tau} (x_{1:T}|x_0) &= q_{\sigma,\tau}(x_{\tau_S}|x_0) \prod^S_{i=1}q_{\sigma,\tau} (x_{\tau_{i-1}}|x_{\tau_i}, x_0) \prod_{t\in\bar{\tau}}q_{\sigma,\tau} (x_t|x_0) \;\; \text{(52)} \\\\
& \tau \text{ is a sub-sequence of } [1, \dots, T] \text{ of length S with } \tau_S = T, \{x_{\tau_1}, \dots, x_{\tau_s}\} \\\\
& \bar{\tau} := \{1, \dots, T\} \setminus \tau \\\\
q_{\sigma,\tau}(x_t|x_0) &= \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I) \quad \forall t \in \bar{\tau} \cup \{T\} \\\\
q_{\sigma,\tau}(x_{\tau_i}|x_0) &= \mathcal{N}(\sqrt{\alpha_{\tau_i}}x_0, (1-\alpha_{\tau_i})I) \quad \forall i \in [S] \;\;(54) \\\\
q_{\sigma,\tau}(x_{\tau_{i-1}}|x_{\tau_i}, x_0) &= \mathcal{N}\left(\sqrt{\alpha_{\tau_{i-1}}}x_0 + \sqrt{1-\alpha_{\tau_{i-1}}-\sigma_{\tau_i}^2} \cdot \frac{x_{\tau_i} - \sqrt{\alpha_{\tau_i}} x_0}{\sqrt{1-\alpha_{\tau_i}}}, \sigma_{\tau_i}^2 I\right) \quad \forall i \in [S] \;\; \text{(7) DDIM} \\\\
p_{\theta}(x_{0:T}) &:= \underbrace{p_{\theta}(x_T)\prod^S_{i=1}p^{(\tau_i)}_{\theta}(x_{\tau_{i-1}}|x_{\tau_i})}_{\text{"use to produce samples"}} \times \underbrace{\prod_{t \in \bar{\tau}}p^{(t)}_{\theta}(x_0|x_t)}_{\text{"in variational objective"}} \;\; (55)
\end{align}
```

#### Reverse Process

예측된 $x_0 (Eq. 9)$

$$f_θ^{(t)}(x_t):=(x_t-\sqrt{1−α_t} \cdot ϵ_θ^{(t)}(x_t))/\sqrt{α_t}$$
​
이는 $x_t=\sqrt{α_t}x_0+\sqrt{1−α_t}ϵ$ 식을 $x_0$ 에 대해 정리한 것

모델이 예측한 노이즈 $\epsilon_\theta^{(t)}(x_t)$를 실제 노이즈 $ϵ_t$대신 사용


https://www.youtube.com/watch?v=n2P6EMbN0pc
