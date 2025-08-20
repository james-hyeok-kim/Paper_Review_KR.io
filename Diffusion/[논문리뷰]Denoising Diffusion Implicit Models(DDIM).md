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
\end{align}
```

```math
\begin{align}
& p_{\theta}(x_{0:T}) \text{"는 가속 샘플링을 위한 생성과정(Generative Process)를 수학적으로 정의"} \\\\
p_{\theta}(x_{0:T}) &:= \underbrace{p_{\theta}(x_T)\prod^S_{i=1}p^{(\tau_i)}_{\theta}(x_{\tau_{i-1}}|x_{\tau_i})}_{\text{"use to produce samples"}} \times \underbrace{\prod_{t \in \bar{\tau}}p^{(t)}_{\theta}(x_0|x_t)}_{\text{"in variational objective"}} \;\; (55) \\\\
&= p_\theta(x_T) \text{"사전 분포 (Prior Distribution)"} x \text{"Sampling Path"} x \text{"변분 목적 함수 (Variational Objective Term)"} \\\\
\end{align}
```

We consider two types of selection procedure for τ given the desired dim($τ$) < T:

* Linear: we select the timesteps such that $τ_i$ = $\[ci\]$ for some c;
* Quadratic: we select the timesteps such that $τ_i$ = $\[ci^2\]$ for some c.


### Result
* Lower is Better
<img width="1103" height="387" alt="image" src="https://github.com/user-attachments/assets/d26245a4-fc48-4ea5-8407-b63c3bf30d5e" />


## Diffusion Models as Score Based Models vs Stochastical Differential Equation(SDE)

### Score Based Model
* Given Probability p(x)가 주어졌을때, Score는 $\nabla_xlog(p(x))$ (Gradient and Log density function)

* 이 Gradient Log Density function은 Direction을 나타낸다 (Noise $\rightarrow$ Image)

<img width="500" height="465" alt="image" src="https://github.com/user-attachments/assets/d3c5cece-1530-4868-ba96-5db420d7edae" />

<img width="934" height="545" alt="image" src="https://github.com/user-attachments/assets/6514dcfb-364a-42a2-a5a1-ae3bcd3f9deb" />

* Gradient 방향으로 움직여주면, High Likelyhood로 모인다.

<img width="907" height="543" alt="image" src="https://github.com/user-attachments/assets/9e313d2b-b9b8-4812-a1c9-3c4cdc976541" />

* Local optimal 에 모일수도 있다.

```math
\begin{align}
s_\theta(x) \approx \nabla_xlog(p(x)) \\\\
\frac{1}{2}E_{x\sim p_data} \parallel \nabla_x log(p_{data}(x)) - s_\theta(x) \parallel^2_2 \\\\
\text{Minimizing Euclidean Distance between Data Score x and Estimated Score x} \\\\
E_{x\sim p_data} \left[ \frac{1}{2} \parallel s_\theta(x) \parallel^2_2 + tr(\nabla_xs_\theta(x)) \right] \\\\
tr = Trace \text{대각합} 
\end{align}
```

* 대각합으로 변환한 이유 ($∇ₓlog(p_{data}(x))$ 계산할 방법이 없어)

  * 문제의 근원: $p_{data}(x)$의 정체를 모른다는 것
  * 우리가 학습하려는 데이터의 실제 확률 분포 $p_{data}(x)$ 의 정확한 함수식을 모른다는 점
  * 우리가 가진 것은 $p_{data}(x)$ 라는 함수 자체가 아니라, 그 분포에서 추출된 샘플(sample)들의 집합 뿐
  * 함수식을 모르기 때문에, 당연히 $log(p_{data}(x))$를 계산할 수 없고, 그것을 미분한 $∇ₓlog(p_{data}(x))$ (스코어) 역시 절대 직접 계산할 수 없습니다.
  * 따라서, 원래의 목적 함수 $||∇ₓlog(p_{data}(x)) - s_θ(x)||_2²$는 이론적으로는 완벽하지만 실제로는 **계산이 불가능한 '그림의 떡'**인 셈입니다.

* 수학적 돌파구: '부분적분'을 이용한 미분 옮기기
  * 부분적분 공식은 다음과 같습니다.
  * $\int u(x)v′(x)dx=u(x)v(x)−\int u′(x)v(x)dx$
  * $E_{x∼p_{data}}[복잡한 항]≈ \int p_{data}(x)⋅(복잡한 항)dx$
  * $p_{data}(x)$를 u역할로 봅니다. 모델과 관련된 나머지 부분을 v' 역할로 봅니다.


```math
\begin{align}
L(\theta) &= \frac{1}{2}E_{x\sim p_data} \parallel \nabla_x log(p_{data}(x)) - s_\theta(x) \parallel^2_2 \\\\
& \text{적분형태} \\\\
&= \frac{1}{2} \int p_{data}(x) \parallel \nabla_x log(p_{data}(x)) - s_\theta(x) \parallel^2_2dx \\\\
& \text{제곱 항 전개} \\\\
&= \frac{1}{2} \int p_{data}(x) \text{[} \parallel \nabla_x log(p_{data}(x)) \parallel^2 -2(\nabla_xlogp_{data})^Ts_\theta + \parallel s_\theta(x) \parallel^2 \text{]}dx \\\\
&= \underbrace{\frac{1}{2} \int p_{data}(x) \parallel \nabla_x log(p_{data}(x)) \parallel^2 dx}_{\text{1 항}} - \underbrace{\int p_{data}(\nabla_xlogp_{data})^Ts_\theta dx}_{\text{2 항}} + \underbrace{\frac{1}{2}\int p_{data} \parallel s_\theta(x) \parallel^2dx}_{\text{3 항}} \\\\
& \text{1 항} \theta \text{ 와 무관하므로 상수} \\\\
& \text{2 항은 계산 불가의 } \nabla_x \log p_{data} \text{ 를 포함, 우리가 부분적분할 대상} \\\\
& \text{3 항 } s_\theta \text{ 에 대한 항, 계산 가능} \\\\
& \text{2 항} \\\\
& - \int p_{data}(x)(\nabla_x log p_{data}(x))^T s_\theta(x)dx \\\\
\nabla_x log(f(x)) &= (\nabla_x f(x)) / f(x) \text{미분 트릭 적용} \\\\
& - \int p_{data}(x) \left(\frac{\nabla_x log p_{data}(x)}{p_{data}(x)} \right)^T s_\theta(x)dx \\\\
& - \int (\nabla_x p_{data}(x))^Ts_\theta(x)dx \\\\
& \text{다차원 부분적분} \int (\nabla f)^T g dx = - \int f (\nabla \cdot g) dx \\\\
& + \int p_{data}(x)tr(\nabla_x s_\theta (x))dx \\\\
L(\theta) &= (상수) + \int  p_{data}(x)tr(\nabla_x s_\theta (x))dx  + \frac{1}{2}\int p_{data} \parallel s_\theta(x) \parallel^2dx \\\\
&= E_{x\sim p_{data}} \left[tr(\nabla_x s_\theta(x)) + \frac{1}{2}\parallel s_\theta (x) \parallel^2 \right] + 상수 \\\\
\end{align}
```


### $tr(\nabla_x s_\theta(x))$ 계산이 computatively expensive

* "모델이 깨끗한 데이터의 스코어를 배우게 하는 대신, 약간의 노이즈를 섞은 데이터의 스코어를 배우게 하면 더 안정적이고 효과적이지 않을까?"

* Original Score Matching $\rightarrow$ Denoising Score Matching

* 이제 모델 $s_θ$는 깨끗한 데이터 x가 아닌, 노이즈 낀 데이터 $x̃$ 를 입력받습니다.

* 목표 스코어도 $∇ₓlog(p_{data}(x))$ 가 아닌, 노이즈 낀 데이터의 분포 $q_σ(x̃)$ 의 스코어인 $∇_{x̃}log(q_σ(x̃))$ 로 바뀌었습니다.

```math
\begin{align}
L(\theta) &= E_{x\sim p_{data}} \left[\frac{1}{2}\parallel s_\theta (x) \parallel_2^2  +  \text{tr}(\nabla_x s_\theta(x)) \right]\\\\
& q_\sigma(\tilde{x}) \rightarrow \text{Noise가 추가된 x의 q 확률 밀도함수} \\\\
&q_\sigma(\tilde{x}) = \int q_\sigma(\tilde{x}|x) p_{data}(x) dx \\\\
&= \frac{1}{2}E_{\tilde{x} \sim q_{\sigma}} \left[ \parallel \nabla_{\tilde{x}}\log q_\sigma(\tilde{x}) - s_\theta(\tilde{x}) \parallel_2^2 \right] \\\\
\end{align}
```


###  $∇_{x̃}log(q_σ(x̃))$를 계산하려면 여전히 $p_{data}$를 알아야 함. 즉, 또다시 계산 불가능한 문제
* 핵심은 목표 스코어를 $∇_{\tilde{x}}log(q_σ(x̃))$에서 $∇_{\tilde{x}}log(q_σ(\tilde{x}|x))$로 바꾼 것입니다.

* 계산 가능한 목표 (✅): $q_σ(\tilde{x}|x)$는 "깨끗한 데이터 x가 주어졌을 때, 노이즈 낀 데이터 $\tilde{x}$가 나올 확률"입니다. 이것은 우리가 직접 정의하는 간단한 가우시안 분포 $N(\tilde{x}|x,σ²I)$입니다.

* 우리는 이 함수의 정확한 식을 알고 있기 때문에, 스코어 $∇_{\tilde{x}}log(q_σ(\tilde{x}|x))$를 쉽게 계산할 수 있습니다. 그 결과가 바로 $-(\tilde{x}-x)/σ²$ 입니다.

###  $∇_{x̃}log(q_σ(x̃)) \rightarrow ∇_{x̃}log(q_σ(x̃|x))$

$$∇_\tilde{x} log q_σ(\tilde{x})=E_{x∼q(x∣\tilde{x})}[∇_\tilde{x} log q_σ(\tilde{x}∣x)] $$

*  손실 함수 L(θ)를 파라미터 θ로 미분한 값, 즉 **기울기 $∇_θ L(θ)$**를 사용합니다. 만약 두 손실 함수의 기울기가 같다면 $(∇_θ L_{hard} = ∇_θ L_{easy})$, 두 함수를 최적화하는 것은 완벽하게 동일한 과정이 됩니다.
```math
\begin{align}
∇_θL_{hard} &= E_\tilde{x}[−2(∇_\tilde{x}log q_σ(\tilde{x}) − s_θ(\tilde{x}))∇_θs_θ(\tilde{x})] \\\\
&=E_\tilde{x}[−2(E_{x\sim q(x|\tilde{x})} [∇_\tilde{x}log q_σ(\tilde{x}|x)] − s_θ(\tilde{x}))∇_θs_θ(\tilde{x})] \\\\
&=E_\tilde{x}[−2(∇_\tilde{x}log q_σ(\tilde{x}|x) − s_θ(\tilde{x}))∇_θs_θ(\tilde{x})] = \nabla_\theta L_{easy} \\\\ 
\end{align}
```

### $\tilde{x} = x+\epsilon (noise)$

```math
\begin{align}
L_\theta &= \frac{1}{2}E_{\tilde{x} \sim q_{\sigma}} \left[ \parallel \nabla_{\tilde{x}}\log q_\sigma(\tilde{x}) - s_\theta(\tilde{x}) \parallel_2^2 \right] \\\\
&= \frac{1}{2}E_{x\sim P_{data}, \\ \tilde{x} \sim q_{\sigma}(\tilde{x}|x)} \left[ \parallel \nabla_{\tilde{x}}\log q_\sigma(\tilde{x}|x) - s_\theta(\tilde{x}) \parallel_2^2 \right] \\\\
&= \frac{1}{2}E_{x\sim P_{data}, \\ \tilde{x} \sim q_{\sigma}(\tilde{x}|x)} \left[ \parallel -\frac{\epsilon}{\sigma^2} - s_\theta(\tilde{x}) \parallel_2^2 \right] \\\\
\end{align}
```

### Score Based Model

<img width="1105" height="660" alt="image" src="https://github.com/user-attachments/assets/5af61fda-8ba3-4c4b-8789-f8657c7a6026" />

* Noise conditional score Sampling is very similar to Diffusion reverse process

<img width="1119" height="447" alt="image" src="https://github.com/user-attachments/assets/b98051d7-1005-4a6d-bb97-11284db77dbc" />

---

### Stochastic Differential Equation
$$dx = f(x, t)dt + g(t)dw$$

시간이 아주 미세하게 ($dt$) 변할때, 데이터 x가 얼마나 미세하게 ($dx$) 변하는지 설명

* $f(x,t)dt$ : Drift 항
  * 데이터 x와 시간 t에 따라 데이터가 어떤 정해진 방향으로 미세하기 움직이도록
* $g(t)dw$ : Diffusion 항
  * 무작위적인 부분 담당

```math
\begin{align}
q(x_t|x_{t−1}) &:= N (x_t;\sqrt{1 − β_t}x_{t−1}, β_tI \;\; \text{(DDPM Transition, Forward)} \\\\
& x_t\text{정의} \\\\
x_{t} &= \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}z_t \\\\
& \sqrt{1-\beta_t} \approx 1- \frac{\beta_t}{2}  \;\; \text{(approximate 사용)} \\\\
dx &= -\frac{1}{2}\beta(t)xdt + \sqrt{\beta(t)}dw \;\; \text{(DDPM SDE)} \\\\
dx &= [f(x,t) - g(t)^2\nabla_xlogp_t(x)]dt + g(t)d\bar{w} \;\; \text{(General Form of Reverse SDE)}  \\\\
dx &= [-\frac{1}{2} \beta(t)x - \beta(t) \nabla_xlogp_t(x)]dt + \sqrt{\beta(t)}dw  \text{(General to DDPM SDE)} \\\\
& \frac{1}{\sqrt{1-\beta_t}} \approx 1+\frac{\beta_t}{2} \;\; \text{(approximate 사용하여 } x_{t-1} \text{ 표현)} \\\\
x_{t-1} &= \frac{1}{\sqrt{\alpha_t}}\left(x_{t-1} -\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)\right) + \sigma_tz \;\; \text{(DDPM Sampling algorithm 2)}\\\\
&= \frac{1}{\sqrt{1-\beta_t}}(x_t+\beta_t s_\theta(x_t)) + \sqrt{\beta_t}z_t \\\\
& x_{t-1} 유도 \\\\

% Definitions
\alpha_t &= 1 - \beta_t \\\\
s_{\theta}(x_t) &= -\frac{1}{\sqrt{1-\bar{\alpha}_t}} \epsilon_{\theta}(x_t, t) \\\\
\epsilon_{\theta}(x_t, t) &= -\sqrt{1-\bar{\alpha}_t} \cdot s_{\theta}(x_t) \\\\
% Derivation
x_{t-1} &= \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_{\theta}(x_t, t)\right) + \sigma_t z \\\\
&= \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \left(-\sqrt{1-\bar{\alpha}_t} \cdot s_{\theta}(x_t)\right)\right) + \sigma_t z \\\\
&= \frac{1}{\sqrt{\alpha_t}}\left(x_t + (1-\alpha_t)s_{\theta}(x_t)\right) + \sigma_t z \\\\
&= \frac{1}{\sqrt{1-\beta_t}}\left(x_t + \beta_t s_{\theta}(x_t)\right) + \sigma_t z \\\\
&= \frac{1}{\sqrt{1-\beta_t}}\left(x_t + \beta_t s_{\theta}(x_t)\right) + \sqrt{\beta_t} z_t
\end{align}
```

#### $x_{t-1} - x_t$ 수학적 근사 $\rightarrow$ Reverse SDE 근사

```math
\begin{align*}
d\mathbf{x} &= \mathbf{f}(\mathbf{x}, t)dt + g(t)d\mathbf{w} \text{  (Forward SDE)}\\\\
d\mathbf{x} &= \left[\mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt \text{  (Reverse SDE)} \\\\
dx &= -\frac{1}{2}\beta(t)x \, dt + \sqrt{\beta(t)} \, dw \quad \text{  (DDPM Forward SDE)} \\\\
& \frac{1}{\sqrt{1-\beta_t}} \approx 1 + \frac{\beta_t}{2} \quad \text{as } \beta_t \to 0  \\\\
x_{t-1} &= \sqrt{\bar{\alpha}_{t-1}}\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta^{(t)}(x_t)}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \epsilon_\theta^{(t)}(x_t) + \sigma_t \epsilon_t \text{  (DDIM 의 일반적 Sampling 알고리즘)} \\\\
x_{t-1} &= \frac{x_t}{\sqrt{1-\beta_t}} - \left(\sqrt{\frac{1-\bar{\alpha}_t}{1-\beta_t}} - \sqrt{1-\bar{\alpha}_{t-1}}\right)\epsilon_\theta(x_t)  \;\; (\alpha_t = 1-\beta_t \text{  를 활용하여 정리)} \\\\
x_{t-1} &= \frac{x_t}{\sqrt{1-\beta_t}} - \left(\sqrt{\frac{1-\bar{\alpha}_t}{1-\beta_t}} - \sqrt{\frac{1-\bar{\alpha}_t}{1-\beta_t}}\right)\epsilon_\theta(x_t) \\\\
x_{t-1} &= \left(1+\frac{\beta_t}{2}\right)x_t - \frac{\beta_t}{2\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t) \\\\
x_{t-1}-x_t &= \left(\frac{\beta_t}{2}\right)x_t - \frac{\beta_t}{2\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t)
\end{align*}
```

* $x_{t-1} - x_t  ↔  dx$ (미세한 변화량)

* 알고리즘의 $(β_t/2)x_t$ 항  ↔  역방향 SDE의 드리프트 항

* 알고리즘의 $ε_θ$ 항  ↔  역방향 SDE의 스코어 $(∇ₓlog p_t(x))$ 항

* SDE를 푸는 더 발전된 수치해석 기법을 도입하여 샘플링 속도나 품질을 개선하는 연구가 가능

```math
\begin{align}
dx &= [f(x,t) - g(t)^2\nabla_xlogp_t(x)]dt + g(t)d\bar{w} \;\; \text{(General Form of Reverse SDE)}  \\\\
d\mathbf{x} &= \left[\mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt \text{  (Reverse SDE)} \\\\
\end{align}
```
* 1/2이 있는 식은 확률 흐름 ODE(Probability Flow ODE)이고, 1/2이 없는 식은 일반적인 역방향 SDE(Reverse SDE)

순방향 SDE dx = f dt + g dw를 거꾸로 되돌리는 방법에는 크게 두 가지가 있습니다. (noise 유무)

1. 역방향 SDE (Reverse SDE)

수식: $dx = [f(x,t) − g(t)²∇ₓlog p_t(x)]dt + g(t)dw̄$

2. 확률 흐름 ODE (Probability Flow ODE)

수식: $dx = [f(x, t) - (1/2)g(t)²∇ₓlog p_t(x)]dt$



---


### Appendix
#### L2 Norm
$\parallel ⋅ \parallel_2$ : L2 노름 (Euclidean Norm)
이 기호는 **L2 노름(norm)**을 나타내며, 벡터의 '크기' 또는 '길이'를 측정하는 가장 일반적인 방법입니다. 우리가 보통 생각하는 두 점 사이의 직선 거리를 계산하는 것과 같습니다.

예를 들어, 2차원 벡터 v = (x, y)가 있다면, L2 노름은 다음과 같이 계산됩니다.

$$\parallel v \parallel_2 = \sqrt{x^2+y^2}$$

$$\parallel v \parallel_2^2 = x^2 + y^2$$
 

​
#### $∇_{\tilde{x}}log(q_σ(\tilde{x}|x)) \rightarrow -(\tilde{x}-x)/σ²$

* 확률밀도 함수
```math
\begin{align}
q_{\sigma}(\tilde{x}|x) &= \frac{1}{\sqrt{(2\pi)^D |\sigma^2 I|}} \exp\left(-\frac{1}{2}(\tilde{x}-x)^T(\sigma^2 I)^{-1}(\tilde{x}-x)\right) \\\\
\end{align}
```
* Log
  * $(σ²I)⁻¹$는 역행렬이므로 $(1/σ²)I$가 됩니다.
  * $(x̃-x)ᵀ I (x̃-x)$는 벡터 $(x̃-x)$의 내적(dot product)이므로, 제곱 L2 노름 ||x̃-x||²와 같습니다.
```math
\begin{align}
\log q_{\sigma}(\tilde{x}|x) &= \text{상수} - \frac{1}{2}(\tilde{x}-x)^T(\sigma^2 I)^{-1}(\tilde{x}-x) \\\\
&= \text{상수} - \frac{1}{2\sigma^2} \parallel \tilde{x} - x \parallel_2^2 \\\\
\end{align}
```
* $∇_{x̃}$ 미분 계산
  * ∑ 안쪽의 $(x̃ᵢ - xᵢ)²$ 항을 $x̃ᵢ$에 대해 편미분하면, 체인룰(chain rule)에 의해 $2(x̃ᵢ - xᵢ)$가 됩니다.
  * x̃와 x는 단순히 하나의 숫자가 아니라, 여러 개의 숫자로 이루어진 벡터 $\rightarrow \\ \sum$
```math
\begin{align}
\nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x}|x) &= \nabla_{\tilde{x}} \left[ \text{상수} - \frac{1}{2\sigma^2} \sum_i (\tilde{x}_i - x_i)^2 \right] \\\\
&= - \frac{1}{2\sigma^2} \cdot \left[ 2(\tilde{x}_1 - x_1), 2(\tilde{x}_2 - x_2), \dots \right] \\\\
&= - \frac{1}{\sigma^2} \left[ \tilde{x}_1 - x_1, \tilde{x}_2 - x_2, \dots \right] \\\\
&= - \frac{\tilde{x} - x}{\sigma^2}
\end{align}
```
