# Denoising Diffusion Probabilistic Models (DDPM)
저자(소속) : Jonathan Ho (UC Berkeley), Ajay Jain(UC Berkeley), Pieter Abbeel(UC Berkeley)

논문 : [PDF](https://arxiv.org/pdf/2006.11239)

일자 : 16 Dec 2020


---

## 핵심 아이디어

### 순방향 확산(forward process): 실제 데이터 
* $𝑥_0$​에 Gaussian 노이즈를 점진적으로 추가하여 단계별로 $𝑥_1,𝑥_2, ... , 𝑥_𝑇$ 를 생성.
* $𝑇$가 충분히 크면 $𝑥_𝑇$는 거의 정규분포 $𝑁(0,𝐼)$와 유사해짐 

### 역방향 과정(reverse process)
* 노이즈 이미지에서 점차 노이즈를 제거하여 원본 데이터를 복원하는 확률적 경로를 학습.
* 이 과정이 데이터 생성의 핵심 

#### 기존 생성 모델들(GAN, VAE 등)의 단점을 보완하며, 모드 커버리지와 샘플 품질 모두 우수함.

#### GAN은 mode collapse 문제가 있었고, VAE는 샘플 품질이 낮았음.


## 초록
Implementation : [Git](https://github.com/hojonathanho/diffusion)

## 도입
<p align="center">
<img src = "https://github.com/user-attachments/assets/1351575a-8638-446c-9a9b-d5d9dc8db15c" width="60%" height="60%">
</p> 

Markov chain forwarding 방식으로 noise를 더하고, reverse방식으로 noise에서 이미지를 생성

---

## 배경

#### 목표
$$p_θ(x)=∫p_θ(x∣z)p(z)dz$$

다음과 같은 형태의 확률 생성 모델을 다룹니다

* $𝑧$는 latent variable (잠재 변수)
* $𝑝(𝑧)$: 간단한 prior 분포 $(ex: \mathcal{N}(0,𝐼))$
* $𝑝_𝜃(𝑥∣𝑧)$: decoder (복원 모델)
* 이 모델에서 $log𝑝_𝜃(𝑥)$ 를 직접 계산하는 건 어렵다. → 추정을 통해 근사.


#### Variational Inference and ELBO
🔹 아이디어:

* 복잡한 $𝑝_𝜃(𝑥)$ 를 직접 계산하는 대신, ELBO (Evidence Lower Bound)를 최대화해서 근사한다.

* $𝑞_𝜙(𝑧∣𝑥)$ : encoder 또는 approximate posterior

🔹 ELBO 정의

$$log\ 𝑝_𝜃(𝑥)≥𝐸_{𝑞_𝜙(𝑧∣𝑥)}[log⁡𝑝_𝜃(𝑥∣𝑧)]−𝐷_{KL}(𝑞_𝜙(𝑧∣𝑥) \parallel 𝑝(𝑧))$$

* 이 식은 다음 두 항의 합으로 해석됨:
  * 복원항 (likelihood term): $𝐸_𝑞[log⁡𝑝(𝑥∣𝑧)]$
  * 정규화항 (KL term): posterior가 prior와 얼마나 다른지를 나타냄

#### Variational Inference in DDPM
🔹 Diffusion에서는 어떻게 사용되는가?

* DDPM에서는 latent variable $𝑧$대신, 노이즈가 점진적으로 추가된 상태 $𝑥_𝑡$ 들이 잠재 변수처럼 사용됨.

* forward process는 known Gaussian noise process인 반면,

* reverse process $𝑝_𝜃(𝑥_{𝑡−1}∣𝑥_𝑡)$ 는 learnable한 분포로써 학습됨.

이 구조가 VAE와 유사한 변분 추정(Variational Inference) 구조를 가짐 → 따라서 ELBO 기반 loss로 학습 가능함

---

## Forward Process (Diffusion Process) $q$
* $q(x_{1:T}|x_0) := \displaystyle\prod_{t=1}^{T}q(x_t|x_{t-1})$
* $q(x_t|x_{t-1}) := \mathcal{N}(x_t;\sqrt{1- \beta_{t}}x_{t-1},\beta_{t}I)$
* 작은 가우시안 노이즈를 T단계에 걸쳐 점차 추가
* Variance(Noise) Schedule $\beta_1, ... , \beta_T:$
  * 미리 정해둔 노이즈값 (예: 0.0001 ~ 0.02)
* $\sqrt{1- \beta_{t}}$ 로 scaling하는 이유는 variance가 발산하는 것을 막기 위해서
  * $\sqrt{1-\beta_t}^2 + \beta_t^2 = 1 = \sigma^2(variance)$ $\rightarrow$ 가우시안 분포
* 여기서
  *  $𝛼_𝑡:=1−𝛽_𝑡$
  * $\bar{α_t}:=\displaystyle\prod_{𝑠=1}^{𝑡}𝛼_𝑠$

👉 즉, 한번에 $𝑥_0$ 에서 $𝑥_𝑡$를 샘플링할 수 있음.

---

## Reverse Process $p_{\theta}$
* $p_{\theta}(x_{0:T}) \rightarrow reverse \ process$
* Markov chain with learned Gaussian transitions, $p(x_T) = \mathcal{N}(x_T;0,I):$ (Normal distribution)
* 보통 Normal Distribution의 표현 $X \sim N(\mu, \sigma^2)$ 평균 $(\mu)$ , 분산 $(\sigma)^2$ 로 표현
* $p_{\theta}(x_{0:T}) := p(x_{T})\displaystyle\prod_{t=1}^{T}p_{\theta}(x_{t-1}|x_{t})$
* $p_{\theta}(x_{t-1}|x_t) :=  \mathcal{N} (x_{t-1};\mu_{\theta}(x_t,t),\sum_{\theta}(x_t,t))$

---

## Training (학습)
* 훈련 목표 : Variational Upper Bound인 L을 최소화

$$L=E_q​[ −logp_θ​(x_0)\]≤E_{q}​[−log\frac{p_θ(x_{0:T})​}{q(x_{1:T​}∣x_0)}\]$$

#### Variational Upper Bound 유도
정확한 유도 과정:

1. 로그-우도(Log-Likelihood)

```math
\log \; p_θ(x_0)=\log\int{p_θ}(x_{0:T}))dx_{1:T}
```
​
여기서 $p_\theta(x_{0:T})$는 모든 시점의 데이터를 포함하는 결합 확률 분포입니다.

2. $q(x_{1:T}|x_0)$로 확장

```math
\log\;p_θ(x_0)= \log \int{p_θ(x_{0:T}) \frac{q(x_{1:T}∣x_0)}{q(x_{1:T}∣x_0)} dx_{1:T}} = \log \;E_{q(x_{1:T}∣x_0)} \left[\frac{p_θ(x_{0:T})}{q(x_{1:T}∣x_0)} \right]
```

여기서 $q(x_{1:T}|x_0)$는 우리가 학습하는 모델인 인코더(encoder)에 해당하는 분포입니다.

확률 변수 X가 분포 $p(X)$를 따를 때, 함수 $f(X)$의 기댓값은 $E_{p(X)}[f(X)]=∫f(X)p(X)dX$ 입니다.

이제 우리의 식을 이 기댓값의 형태로 바꿔봅시다.

```math
\log p_θ(x_0)=\log \int \left(\frac{p_θ(x_{0:T})}{q(x_{1:T}∣x_0)} \right)⋅q(x_{1:T}∣x_0)dx_{1:T}
```

* $f(X)$에 해당하는 부분은 $\left(\frac{p_θ(x_{0:T})}{q(x_{1:T}∣x_0)} \right)$ 입니다.

* 분포 $p(X)$에 해당하는 부분은 $q(x_{1:T}∣x_0)$입니다.

* 적분(integral) 변수는 $x_{1:T}$입니다.​


3. 젠센 부등식(Jensen's inequality) 적용
로그 함수는 오목 함수(concave function)이므로 젠센 부등식이 적용됩니다.

$$logE[X]≥E[logX]$$

```math
\log \; p_θ(x_0)≥E_{q(x_{1:T}∣x_0)} \left[\log \frac{p_θ(x_{0:T})}{q(x_{1:T}∣x_0)} \right]=ELBO
```

4. L (Variational Upper Bound)의 등장:
위에서 유도된 $\log p_\theta(x_0) \ge \text{ELBO}$를 재정렬하면 다음과 같습니다.

```math
−log\;p_θ(x_0)≤−ELBO=−E_{q(x_{1:T}∣x_0)} \left[log\frac{p_θ(x_{0:T})}{q(x_{1:T}∣x_0)} \right]
```

이때, 우변의 항 $- \text{ELBO}$를 우리는 **L (Variational Upper Bound)**라고 부릅니다.

```math
L=E_{q(x_{1:T}∣x_0)} \left[−\log \;p_θ(x_{0:T})+ \log \; q(x_{1:T}∣x_0) \right]
```


### Loss 유도
$$L=E_q[D_{KL}​(q(x_T|x_0​)\parallel p(x_T))+\displaystyle\sum_{t>1}D_{KL}​(q(x_{t−1}​|x_t​,x_0)\parallel p_θ(x_{t−1}|x_t))−\log p_θ(x_0|x_1)] \\ (5) $$
* 유도 (Loss 수식 이해) [Youtube](https://www.youtube.com/watch?v=ybvJbvllgJk)
* Bayesian Rule $p(x|y) = \frac{p(x,y)}{p(y)}$
* Markov Chain $q(x_t|x_{t-1},x_{t-2},x_0) = q(x_t|x_{t-1})$

```math
\begin{align}
L & = E_q \left[ − \log \frac{p_θ(x_{0:T})}{q(x_{1:T} |x_0)} \right] \;\; (17) \\\\
&= E_q \left[ − \log p(x_T) − \displaystyle\sum_{t≥1} \log \frac{p_θ(x_{t−1}|x_t)}{q(x_t|x_{t−1})} \right] \;\;(18) \\\\
* & t\geq1 \rightarrow t\gt1 \\\\
&= E_q \left[− \log \ p(x_T) − \displaystyle\sum_{t>1} \log \frac{p_θ(x_{t−1}|x_t)}{q(x_t|x_{t−1})} − \log \frac{p_θ(x_0|x_1)}{q(x_1|x_0)} \right] \;\;(19) \\\\
* &\frac{1}{q(x_t|x_{t-1})} = \frac{1}{q(x_{t-1}|x_t,x_0)} \cdot \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)} \\\\
* & q(x_t|x_{t-1}) = q(x_t|x_{t-1}, x_0) = \frac{q(x_t,x_{t-1},x_0)}{q(x_{t-1},x_0)} \cdot \frac{q(x_t,x_0)}{q(x_t,x_0)} = q(x_{t-1}|x_t,x_0) \cdot \frac{q(x_t,x_0)}{q(x_{t-1},x_0)} \\\\
&= E_q \left[− \log \ p(x_T) − \displaystyle\sum_{t>1} \log \frac{p_θ(x_{t−1}|x_t)}{q(x_{t−1}|x_t, x_0)} · \frac{q(x_{t−1}|x_0)}{q(x_t|x_0)} − \log \frac{p_θ(x_0|x_1)}{q(x_1|x_0)} \right] \;\;(20) \\\\
&= E_q \left[− \log \ p(x_T) − \displaystyle\sum_{t>1} \log \frac{p_θ(x_{t−1}|x_t)}{q(x_{t−1}|x_t, x_0)} -\displaystyle\sum_{t>1} log \frac{q(x_{t−1}|x_0)}{q(x_t|x_0)} − \log \frac{p_θ(x_0|x_1)}{q(x_1|x_0)} \right]  \\\\
* &-\displaystyle\sum_{t>1} log \frac{q(x_{t−1}|x_0)}{q(x_t|x_0)} = -log \frac{q(x_1|x_0)}{q(x_2|x_0)} -log \frac{q(x_2|x_0)}{q(x_3|x_0)} -log \frac{q(x_3|x_0)}{q(x_4|x_0)}  \cdots = -log\frac{q(x_1|x_0)}{q(x_T|x_0)} \\\\
&= E_q \left[ − \log \ \frac{p(x_T)}{q(x_T |x_0)} − \displaystyle\sum_{t>1} \log \frac{p_θ(x_{t−1}|x_t)}{q(x_{t−1}|x_t, x_0)} − \log \ p_θ(x_0|x_1) \right] \;\;(21) \\\\
&= E_q \left[ D_{KL}(q(x_T|x_0) \parallel p(x_T)) + \displaystyle\sum_{t>1} D_{KL}(q(x_{t−1}|x_t, x_0) \parallel p_θ(x_{t−1}|x_t)) − \log \ p_θ(x_0|x_1) \right] \;\;(22) \\\\
\end{align}
```

<img width="1117" height="141" alt="image" src="https://github.com/user-attachments/assets/56681e35-47f1-4217-bb4b-a12d7e5a03be" />

Loss를 통해 P를 어떻게 구하는지는 알았는데, q는 어떻게 구할건지

---

### $p$가 닮아야할 확률분포 $q$에 대해서 이해하기

$$q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t,x_0), \tilde{\beta}_tI) \\ (6)$$

```math
\begin{align}
{\tilde{\mu}}_{t}(x_t,x_0) = \frac{\sqrt{{\bar{\alpha}}_{t-1}} \beta_t}{1-\bar{\alpha}_t}x_0 + \frac {\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-{\bar{\alpha}}_t} x_{t} \;\; (7) \\\\
\tilde{\beta_t}:=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t  \;\; (7)
\end{align}
```



* $평균 : \tilde{\mu}_t, 분산 : \tilde{\beta}_t$의 가우시안분포를 따른다는 뜻


$$q(x_{t−1}∣x_t,x_0)=q(x_t∣x_{t−1})\frac{q(x_{t−1}∣x_0)}{q(x_t∣x_0)}$$
* 유도
```math
\begin{align}
&* Bayesian Rule \\\\
q(x_{t−1}∣x_t,x_0)&=\frac{q(x_{t−1}, x_t, x_0)}{q(x_t,x_0)} \cdot \frac{q(x_{t-1}, x_0)}{q(x_{t-1}, x_0)} \\\
q(x_{t−1}∣x_t,x_0)&=\frac{q(x_{t−1}| x_t, x_0)}{q(x_t,x_0)} \cdot q(x_{t-1}, x_0) \\\\
q(x_{t−1}∣x_t,x_0)&=\frac{q(x_{t−1}| x_t, x_0)}{q(x_t,x_0)} \cdot q(x_{t-1}, x_0) \cdot \frac{q(x_0)}{q(x_0)}  \\\\
q(x_{t−1}∣x_t,x_0)&=q(x_{t−1}| x_t, x_0) \cdot \frac{q(x_{t-1}| x_0)}{q(x_t|x_0)} \\\\
&* Markov Chain \\\\
q(x_{t−1}∣x_t,x_0)&=q(x_t∣x_{t−1})\frac{q(x_{t−1}∣x_0)}{q(x_t∣x_0)}
\end{align}
```

* 위의 베이즈 정리 식의 각 항을 지수 부분만으로 나타내면 다음과 같습니다.

  * 정규분포의 확률밀도 함수는 $f(x) \propto exp(-\frac{(x-\mu)^2}{2\sigma^2})$의 형태를 가진다

  * [유도공식](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
​

```math
\begin{align}
1. & q(x_t∣x_{t−1})=\mathcal{N}(x_t;\sqrt{1−β_t}x_{t−1},β_tI) \;\; (2) \\\\
1-1. & 지수 부분: − \frac{(x_t − \sqrt{1−β_t}x_{t−1})^2}{2β_t} \\\\
2. & q(x_{t−1}∣x_0) = \mathcal{N}(x_{t−1};\sqrt{\bar{\alpha}_{t-1}}x_0, (1- \bar{\alpha}_{t-1}I) \;\; (4) \\\\
2-1. & 지수 부분: -\frac{(x_{t−1}-\sqrt{\bar{α}_{t−1}}x_0)^2}{2(1−\bar{α}_{t−1})} \\\\
3. & q(x_t∣x_0)=\mathcal{N}(x_t;\sqrt{\bar{α}_t}x_0,(1-\bar{α}_t)I) \;\; (4) \\\\
3-1. & 지수 부분: −\frac{(x_t − \sqrt{\bar{α}_t}x_0)^2}{2(1−\bar{α}_t)} \\\\
q(x_{t−1}∣x_t,x_0) & \propto exp(지수_1+지수_2−지수_3) \\\\
q(x_{t−1}∣x_t,x_0) & \propto exp(− \frac{(x_t − \sqrt{1−β_t}x_{t−1})^2}{2β_t} -\frac{(x_{t−1}−\sqrt{\bar{α}_{t−1}}x_0)^2}{2(1−\bar{α}_{t−1})}−\frac{(x_t − \sqrt{\bar{α}_t}x_0)^2}{2(1−\bar{α}_t)})
\end{align}
```


<img width="855" height="855" alt="image" src="https://github.com/user-attachments/assets/0e7faed0-3c02-4e53-8099-ea2993eee963" />


---
## Loss에서 확률분포 p가 닮아야할 q 이해하기 ( $(5)\rightarrow(8)$ )
```math
\begin{align}
L_{t-1} &= \displaystyle\sum_{t>1}D_{KL}​(q(x_{t−1}​|x_t​,x_0)\parallel p_θ(x_{t−1}|x_t)) \;\; (5) \\\\
&= E_q \left[ \frac{1}{2\sigma_t^2}\parallel \tilde{\mu}_t(x_t,x_0) - \mu_\theta(x_t,t)\parallel^2 \right] + C \;\;(8) \\\\
\end{align}
```



### 다변량 정규화 분포 [Blog](https://blog.naver.com/mj93m/221097578389)

* 다변량 정규분포(multivariate normal distribution) 란 말 그대로 복수개의 확률 변수가 존재하고 그것을 한번에 모형화 한 것을 말한다.

<img width="842" height="637" alt="image" src="https://github.com/user-attachments/assets/ad7fd158-351d-4deb-9033-593106d57022" />


#### 정규분포 평균: $\mu$ , 표준편차: $\sigma$ 

$$X \sim \mathcal{N}(\mu,\sigma^2)$$

PDF(확률밀도함수)

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma}}$$

#### 다변량 정규분포 (Multivariate Normal Distribution)

$$X \sim MVN_p(\mu, \Sigma)$$

$$
\mu = 
\begin{pmatrix}
\mu_1 \\
\vdots \\
\mu_p 
\end{pmatrix}
,\Sigma = 
\begin{pmatrix}
\sigma^2_{11} & \dots & \sigma^2_{1p} \\
\vdots & & \vdots \\
\sigma^2_{p1} & \dots & \sigma^2_{pp}
\end{pmatrix}
$$

PDF(확률밀도함수)

$$f(X) = \frac{1}{\sqrt{2\pi\Sigma}} \exp ^{- \left( \frac{(X-\mu)^T(X-\mu)}{2\Sigma} \right)}$$



#### KL Divergence 간소화 (핵심 아이디어)
두 정규분포 $P_1 = \mathcal{N}(\mu_1, \Sigma_1)$과 $P_2 = \mathcal{N}(\mu_2, \Sigma_2)$ 사이의 KL Divergence 공식은 다음과 같습니다.

```math
\begin{align}
D_{KL}(P_1 \parallel P_2) &= \frac{1}{2} \left( \log\frac{∣Σ_2|}{∣Σ_1|}−d+tr(Σ_2^{-1}Σ_1)+(μ_2−μ_1)^TΣ_2^{-1}(μ_2−μ_1) \right)
\end{align}
```

#### 두 정규분포 $D_{KL}$ 유도
```math
\begin{align}
D_{KL}(P_1 \parallel P_2) &= \int_{−∞}^{∞}p_1(x) \log \left(\frac{p_1(x)}{p_2(x)} \right)dx \\\\
&= \int p_1(x)log(p_1(x))dx - \int p_1(x)log(p_2(x))dx \\\\
&= E_{x∼P_1}[log(p_1(x))]−E_{x∼P_1}[log(p_2(x))] \\\\
\end{align}
```
#### 1. $log(p(x))$
```math
\begin{align}
log(p(x))&= \log \left( \frac{1}{\sqrt{2πσ^2}}e^{−\frac{(x−μ)^2}{2σ^2}} \right) \\\\
&=log \left(\frac{1}{\sqrt{2πσ^2}} \right) − \frac{(x−μ)^2}{2σ^2} \\\\
&=−\frac{1}{2}log(2πσ^2) − \frac{(x−μ)^2}{2σ^2}  \\\\
&=−\frac{1}{2}log(2π)−log(σ) − \frac{(x−μ)^2}{2σ^2}  \\\\
\end{align}
```
#### 2. $E[log(p_2(x))]$
```math
\begin{align}
E_{x∼P_1}[log(p_2(x))]&=E_{x∼P_1} \left[−\frac{1}{2}log(2π)−log(σ_2)−\frac{(x−μ_2)^2}{2σ_2^2} \right] \\\\
&=−\frac{1}{2}log(2π)−log(σ_2)−\frac{1}{2σ_2^2}E_{x∼P_1}[(x−μ_2)^2] \\\\
\end{align}
```
- $E_{x \sim P_1}[(x-\mu_2)^2]를 계산, 괄호 안에 μ_1을 +-$
```math
\begin{align}
E_{x∼P_1}[(x−μ_1+μ_1−μ_2)^2] &= E_{x∼P_1}[((x−μ_1)+(μ_1−μ_2))^2] \\\\
&=E_{x∼P_1}[(x−μ_1)^2+2(x−μ_1)(μ_1−μ_2)+(μ_1−μ_2)^2] \\\\
&=E_{x∼P_1}[(x−μ_1)^2]+2(μ_1−μ_2) E_{x∼P_1}[x−μ_1]+(μ_1−μ_2)^2 \\\\
\end{align}
```
- $E_{x \sim P_1}[(x - \mu_1)^2]는 P_1 분포의 정의에 따라 분산 σ_1^2 $
- $E_{x \sim P_1}[x - \mu_1]는 E[x] - μ_1 = μ_1 - μ_1 = 0$
- $(\mu_1 - \mu_2)^2는 상수$
```math
\begin{align}
E_{x∼P_1}[log(p_2(x))]&= −\frac{1}{2}log(2π)−log(σ_2)−\frac{σ_1^2+(μ_1−μ_2)^2}{2σ_2^2} \\\\
\end{align}
```
#### 3. $E[log(p_1(x))]$
```math
\begin{align}
E_{x∼P_1}[log(p_1(x))]&=E_{x∼P_1} \left[−\frac{1}{2}log(2π)−log(σ_1)−\frac{(x−μ_1)^2}{2σ_1^2} \right] \\\\
&=−\frac{1}{2}log(2π)−log(σ_1)−\frac{1}{2σ_1^2}E_{x∼P_1}[(x−μ_1)^2] \\\\
& E_{x \sim P_1}[(x-\mu_1)^2] = \sigma_1^2 \\\\
&=−\frac{1}{2}log(2π)−log(σ_1)−\frac{\sigma_1^2}{2σ_1^2} \\\\
&= −\frac{1}{2}log(2π)−log(σ_1)−\frac{1}{2} \\\\
\end{align}
```

#### 4. 최종 $E[log(p_2(x))] + E[log(p_1(x))]$
```math
\begin{align}
D_{KL}(P_1 \parallel P_2) &= \left( −\frac{1}{2}log(2π)−log(σ_1)−\frac{1}{2} \right) - \left( −\frac{1}{2}log(2π)−log(σ_2)−\frac{σ_1^2+(μ_1−μ_2)^2}{2σ_2^2} \right) \\\\
&= −log(σ_1)−\frac{1}{2} −log(σ_2)−\frac{σ_1^2+(μ_1−μ_2)^2}{2σ_2^2} \\\\
&= \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
\end{align}
```


## Loss에서 확률분포 p가 닮아야할 q 정리 ( $(5)\rightarrow(8)$ )

여기서 DDPM의 핵심적인 설계 선택이 들어갑니다. $p_\theta$의 분산을 q의 분산과 동일하게 고정합니다.

즉, $\Sigma_\theta(x_t, t) = \tilde{\beta}_t I$로 설정합니다. 보통 $\tilde{\beta}_t$를 $\sigma_t^2$로 표기하기도 합니다. 따라서 두 분포의 분산은 $\Sigma_1 = \Sigma_2 = \sigma_t^2 I$로 같아집니다.

이 가정을 위 KL Divergence 공식에 대입하면 수식이 극적으로 간소화됩니다.

```math
\begin{align}
로그 항: & \log\frac{|\sigma_t^2 I|}{|\sigma_t^2 I|} = \log(1) = 0 \\\\
Trace 항: & \text{tr}((\sigma_t^2 I)^{-1}(\sigma_t^2 I)) = \text{tr}(I) = d
\end{align}
```

* Trace항은 행렬의 대각 합

여기서 d는 데이터의 차원(dimension)입니다. 이제 남은 항들을 정리하면 다음과 같습니다.

```math
\begin{align}
D_{KL}(q \parallel p_θ) & =\frac{1}{2} \left( 0−d+d+(μ_θ − \tilde{μ})^T(σ_t^2I)^{−1}(μ_θ−\tilde{μ})\right) \\\\
& = \frac{1}{2}(μ_θ−\tilde{μ})^T \left(\frac{1}{σ_t^2}I \right)(μ_θ−\tilde{μ}) \\\\
& =  \frac{1}{2σ_t^2}(μ_θ−\tilde{μ})^T(μ_θ−\tilde{μ})
\end{align}
```

벡터 내적 $v^T v$는 L2-norm의 제곱 $\|v\|^2$과 같으므로, 최종적으로 KL Divergence는 두 평균 벡터 간의 **제곱 거리(Squared Distance)**에 비례하는 형태로 정리됩니다.

```math
\begin{align}
D_{KL}(q(x_{t−1}∣x_t,x_0) \parallel p_θ(x_{t−1}∣x_t))= \frac{1}{2σ_t^2} \parallel \tilde{μ}_t(x_t,x_0)−μ_θ (x_t,t) \parallel^2 \;\;(8)
\end{align}
```

### (8)수식의 의미
* $x_0$에 t step noise 더한 이미지 $x_t$를 Neural Net에 줬을때, $q(x_{t-1})$예측, $=q(x_{t-1}|x_t,x_0)$

$$q(x_{t-1}|x_t,x_0) = N(x_{t-1}; \tilde{\mu}_t(x_t,x_0), \tilde{\beta}_tI) \\ (6)$$

```math
\begin{align}
L_{t−1} − C &=E_{x_0,\epsilon}\left[\frac{1}{2σ^2_t}\parallel\tilde{µ}_t \left(x_t(x_0,\epsilon),\frac{1}{\sqrt{\bar{α}_t}}(x_t(x_0,\epsilon)− \sqrt{1 − \bar{α}_t}\epsilon) \right) − µ_θ(x_t(x_0,\epsilon), t) \parallel^2 \right] \;\; (9) \\\\
&= E_{x_0,\epsilon} \left[\frac{1}{2σ^2_t} \parallel \frac{1}{\sqrt{α_t}} \left(x_t(x_0,\epsilon) − \frac{β_t}{\sqrt{1 − \bar{α}_t}}\epsilon\right)− µ_θ(x_t(x_0,\epsilon),t) \parallel^2 \right] \;\; (10) \\\\
\mu_\theta(x_t, t) &= \tilde{\mu}_t\left(x_t, \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_\theta(x_t)) \right) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t) \right) \;\; (11) \\\\
&E_{x_0, \epsilon}\left[ \frac{\beta^2_t}{2\sigma^2_t\alpha_t(1-\bar{\alpha}_t)} \left\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t) \right\|^2 \right] \;\; (12)
\end{align}
```

### (9),(10),(11),(12)는 (8)수식과 같은 의미

* $x_t$를 $x_0$와 noise $\epsilon$으로 치환

* $x_{t-1}$은 $x_t$에서 noise$\epsilon$을 1 step 걷어낸다 $\rightarrow$ 걷어낼 noise를 예측한다
 
* (8) $\rightarrow$ (12)는 평균 $\tilde{\mu}$에서 (이미지) $\epsilon$ (노이즈)의 식으로 변환

#### (8) $\rightarrow$ (9)
* 기댓값(E)의 대상을 변경 (재매개변수화)
  * 식 (8)의 $E_q$는 $q(x_t|x_0)$ 분포, 즉 $x_0$에서 $x_t$를 만드는 과정 전체에 대한 기댓값을 의미합니다.
  * 식 (9)의 $E_{x_0, ε}$는 이 과정을 더 구체적으로 풀어쓴 것입니다

* $x_t$를 $x_t(x_0, ε)$로 명시
  * 위의 변경에 따라, 수식에 있던 모든 $x_t$를 $x_t(x_0, ε)$로 바꾸어 $x_t$가 $x_0$와 $ε$에 의해 결정된다는 것을 명확하게 보여줍니다.

* $(4)$활용

$$q(x_t∣x_0)=\mathcal{N}(x_t;\sqrt{\bar{α}_t}x_0,(1-\bar{α}_t)I) \\ (4)$$

$$x_t(x_0∣\epsilon)=\sqrt{\bar{α}_t}x_0,(1-\bar{α}_t)\epsilon), \ \epsilon \sim N(0,I)$$



---

---

---

---

### 간소화

* 이를 간소화하여 훈련을 하는데, 역방향 과정 평균 $(\mu_\theta)$ 재매개변수화하여 간소화 한다.
  * 이미지 $x_t$로부터 이전 시점의 이미지 $x_{t-1}$ 을 예측하는 역방향 과정(reverse process)을 학습
  * 이 역방향 과정은 다음과 같은 조건부 가우시안 분포로 정의

$$p_θ(x_{t−1}∣x_t)=N(x_{t−1};μ_θ(x_t,t),\sum_θ(x_t,t))$$

* 여기서 $\mu_{\theta}(x_t,t)$는 모델이 학습해야 할 평균 함수
* $\Sigma_{\theta}(x_t,t)$는 공분산 행렬
* 논문에서는 공분산 행렬 $\Sigma_{\theta}(x_t,t)$를 고정된 상수로 설정하므로, 결국 모델이 학습해야 할 핵심은 평균 함수 $μ_θ(x_t,t)$ 입니다.

#### $\mu_{\theta}(x_t,t)$에측의 문제점
* 복잡한 관계: $x_t$ 와 $x_{t−1}$ 사이의 관계는 노이즈가 추가되는 정도 $(β_t)$ 에 따라 복잡하게 변함
* 학습의 어려움: 신경망이 이 복잡한 평균 값을 직접 예측하도록 훈련시키는 것은 어려울 수 있음

#### $\epsilon_{\theta}$를 예측하도록 재매개변수화

* 논문은 이 문제를 해결하기 위해 $\mu_{\theta}(x_t,t)$를 직접 예측하는 대신, $x_t$에 추가된 노이즈 ϵ을 예측하는 신경망 $\epsilon_{\theta}(x_t,t)$를 학습하도록 재매개변수화합니다.
* 이 아이디어는 정방향 과정의 핵심 관계에서 출발합니다
  * 정방향 과정에서 $x_t$는 원본 이미지 $x_0$에 노이즈 ϵ이 추가된 형태로 표현될 수 있습니다

$$x_t=\sqrt{\bar{α}_t}x_0 + \sqrt{1−\bar{α}_t}ϵ$$

* 여기서 $\epsilon \sim \mathcal{N}(0,I)$는 표준 정규 분포에서 샘플링된 노이즈입니다.
* 이 식을 ϵ에 대해 정리하면 다음과 같습니다

$$ϵ=\frac{x_t-\sqrt{\bar{α_t}}x_0}{\sqrt{1-\bar{α_t}}}$$

* 역방향 과정의 평균 $\mu_{\theta}(x_t,t)$는 사실상 정방향 과정의 사후 분포 평균 $\tilde{\mu}_t(x_t,x_0)$를 근사하는 것을 목표로 합니다.
* 이 $\tilde{\mu}_t(x_t,x_0)$는 다음과 같이 표현됩니다:

```math
{\tilde{\mu}}_{t}(x_t,x_0) = \frac{\sqrt{{\bar{\alpha}}_{t-1}} \beta_t}{1-\bar{\alpha}_t}x_0 + \frac {\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-{\bar{\alpha}}_t} x_{t}
```

```math
\tilde{\beta_t}:=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
```



* 이 $\tilde{μ}_t(x_t,x_0)$ 식에 위에서 정리한 ϵ의 표현식을 대입하고 정리하면, $\tilde{\mu}_t(x_t,x_0)$가 ϵ에 대한 함수로 표현될 수 있음을 알 수 있습니다.

* 논문에서는 이 관계를 활용하여 $\mu_{\theta}(x_t,t)$를 다음과 같이 새로운 형태로 정의합니다:

$$μ_θ(x_t,t)=\frac{1}{\sqrt{α_t}}(x_t − \frac{β_t}{\sqrt{1−\bar{α}t}}ϵ_θ(x_t,t))$$

여기서 $\epsilon_{\theta}(x_t,t)$는 신경망이 학습할 노이즈 예측 함수입니다.

* 단순한 학습 목표: 이 재매개변수화를 통해, 원래 복잡했던 $L_{t−1}$항(KL 발산)이 다음과 같이 매우 간단한 평균 제곱 오차(MSE) 형태로 변환됩니다:

$$E_{x_0,ϵ} \left[\frac{β_{t}^{2}}{2σ_{t}^{2}α_{t}(1−\bar{α_t})} \parallel ϵ−ϵ_θ(\sqrt{\bar{α_t}}x_0 + \sqrt{1−\bar{α_t}}ϵ,t) \parallel^2 \right]$$


기존 초기 식

$$L_{t-1} = E_q \left[\frac{1}{2\sigma^2_t} \parallel \tilde{\mu_t}(X_t,X_0)-\mu_\theta(X_t, t)\parallel^2 \right] + C$$

* $\tilde{𝜇_𝑡}(𝑥_𝑡, 𝑥_0)$: forward process에서의 closed-form posterior 평균
* $𝜇_𝜃(𝑥_𝑡,𝑡)$: 학습된 reverse process 평균
* $𝜎_𝑡^2$: reverse process의 고정된 분산
* $𝐶$: $θ$에 영향을 주지 않는 상수 (따라서 최적화에는 무관)

*  이 $D_{KL}$은 가우시안 분포 간의 비교이므로, 평균 제곱 오차 형태로 나타낼 수 있습니다.

* $L_{t−1}$ 항은 $x_{t-1}$의 실제 사후 분포 $q(x_{t-1}|x_t,x_0)$와 모델이 예측하는 역방향 과정 $p_θ(x_{t−1}∣x_t)$ 간의 $D_{KL}$로 정의됩니다.

* 요소
  * 분산 스케줄 $(β_t)$
  * 알파 $(α_t)$ 정의: $α_t:=1−β_t$로 정의됩니다
  * 바 알파 $(\bar{α_{t}})$ = $\prod_{s=1}^{t} α_{s}$
  * 노이즈 예측 신경망 $(ϵ_θ(x_t,t))$

<p align="center">
<img width="804" height="206" alt="image" src="https://github.com/user-attachments/assets/ba2539a9-93a5-417b-bb4f-fda5aa30bdb4" />
</p> 

---


### Parameterization (Noise Prediction)
역방향 분포의 평균은 다음과 같이 계산 가능:

$$𝜇_𝜃(𝑥_𝑡,𝑡)=\frac{1}{𝛼_𝑡}(𝑥_𝑡−\frac{𝛽_𝑡}{\sqrt{1−\bar{𝛼_t}}}𝜖_𝜃(𝑥_𝑡,𝑡))$$

여기서 $𝜖_𝜃(𝑥_𝑡,𝑡)$은 neural network가 예측하는 노이즈

즉, 모델은 실제로 노이즈를 예측하는 방식으로 학습됨 (MSE loss)

### 추가적인 조건
* sampling $x_t$ at arbitrary timestep t
* $\alpha_t := 1-\beta_t$
* $\bar{\alpha_t} = \prod^t_{s=1}\alpha_s$
* $q(x_t|x_0) = N(x_t;\sqrt{\bar{a_t}}x_0, (1-\bar{\alpha_t})I)$


DDPM의 역방향 과정이 score-based model 및 Langevin dynamics와 어떻게 연결되는지를 설명합니다. 

약간 수학적으로 더 깊이 들어가는 파트

### Langevin Dynamics란?

Langevin dynamics는 확률분포 

$𝑝(𝑥)$에서 샘플링할 때 자주 쓰이는 stochastic differential equation 기반 샘플링 방법입니다.

💡 정의
$$𝑥_{𝑡+1}=𝑥_𝑡+\frac{𝜂}{2}∇_𝑥log𝑝(𝑥_𝑡)+\sqrt{𝜂}⋅𝑁(0,𝐼)$$

* $∇_x logp(x_t)$는 score function이라 부릅니다.
* 즉, 확률 밀도함수의 gradient 방향으로 이동 + 약간의 노이즈 추가
→ 반복적으로 이 업데이트를 적용하면 $𝑝(𝑥)$에서 샘플링 가능
* $𝜂$은 step size or learning rate를 의미
* $𝑁(0,𝐼)$는 평균 0, 분산 1인 정규분포, python에서는 numpy.random.normal(loc=0.0, scale=1.0, size= ...)으로 구현가능

🔹 Score Matching과 연결
DDPM의 denoising 모델은 사실상 score function을 예측하고 있습니다.

* DDPM은 각 시점 𝑡마다, $𝑥_𝑡∼𝑞(𝑥_𝑡∣𝑥_0)$에서 샘플링하고

* 네트워크는 노이즈 $𝜖_𝜃(𝑥_𝑡,𝑡)$를 예측 → 이건 $∇_{𝑥_𝑡}log𝑞(𝑥_𝑡)$ 의 방향과 같은 역할

따라서 DDPM은 score function을 간접적으로 학습한다고 볼 수 있음

🔹 결론: DDPM ≈ Score-based Model
논문은 다음과 같이 요약합니다:

```
In the limit of small $𝛽_𝑡$ the reverse DDPM process becomes equivalent to Langevin dynamics with a learned score function.
```

즉:

* $𝛽_𝑡$ → 0 : 매우 작은 노이즈 단계에서

* 역방향 DDPM 과정은 Langevin dynamics와 동일해짐

* 따라서 DDPM은 score-based 생성 모델의 특수한 형태로 해석 가능

---

### Progressive Coding

#### 문제
* 확산 모델이 이전에는 보여주지 못했던 고품질 샘플을 생성할 수 있음을 입증했습니다.
* 샘플 품질의 비약적인 발전에도 불구하고, DDPM은 다른 가능도 기반 모델(likelihood-based models)에 비해 경쟁력 있는 NLL (손실 없는 코딩 길이)를 달성하지 못했습니다.
* 즉, 모델이 데이터를 얼마나 효율적으로 압축하여 표현하는지를 나타내는 지표인 로그 가능도 값은 상대적으로 낮았습니다.
* 모델이 데이터를 압축하는 효율성 측면에서는 다른 모델보다 떨어진다는 문제점
* 이러한 모순적인 결과, 즉 "샘플 품질은 좋은데 로그 가능도는 왜 낮지?" 라는 의문에서 '프로그레시브 코딩'의 개념이 출발
* 논문의 저자들은 "지각 불가능한 이미지 디테일(imperceptible image details)"을 설명하는 데 대부분의 손실 없는 코딩 길이를 소모한다는 것을 발견
  * 인간의 눈으로는 구별하기 어려운 아주 작은 차이까지도 인코딩하려고 노력하기 때문에 발생하는 현상


#### Progressive Coding의 등장
* 전송률-왜곡(rate-distortion)행동을 분석하기 위해 **점진적 손실 코딩(progressive lossy code)**의 개념이 도입
  * 이는 변동 바운드(variational bound)의 각 항을 '전송률'과 '왜곡'으로 해석하여, 시간이 지남에 따라 이미지 정보가 어떻게 점진적으로 압축되고 복원되는지를 보여줌
* '프로그레시브 코딩'은 단순히 이미지를 생성하는 것을 넘어, 정보가 어떻게 계층적으로 인코딩되고 디코딩되는지에 대한 통찰을 제공합니다.
  * 먼저 큰 스케일의 특징이 전송되고, 점차 미세한 디테일이 추가되는 방식은 효율적인 데이터 전송 및 점진적인 이미지 복원 시나리오에서 유용하게 활용될 수 있습니다

#### Progressive Coding 수식적 해석

* 훈련 목표 : Variational Upper Bound인 L을 최소화

$$L=E_q​[−logp_θ​(x_0)]≤E_q​[−log\frac{p_θ(x_{0:T})​}{q(x_{1:T​}∣x_0)}]$$

* 이를 아래와 같이 재 정립

$$L=E_q[D_{KL}​(q(x_T|x_0​)\parallel p(x_T))+\displaystyle\sum_{t>1}D_{KL}​(q(x_{t−1}​|x_t​,x_0)\parallel p_θ(x_{t−1}|x_t))−\log p_θ(x_0|x_1)]$$

* 전송률 (Rate)
  * $L_T=D_{KL}(q(x_T∣x_0 )∣∣p(x_T))$: 이는 초기 잠재 변수 $x_T$ 를 전송하는 데 필요한 비트 수를 나타냅니다.
    * $x_0$에서 확산 과정을 거쳐 얻은 $x_T$의 분포이고, $p(x_T)$는 사전에 정의된 (보통 표준 정규) 분포입니다.
    * 이 값이 낮을수록 모델이 $x_T$를 사전 분포에 가깝게 만들 수 있어 효율적인 전송이 가능합니다.
    * DDPM에서는 $L_T≈0$이 되도록 $β_t$ 스케줄을 설정하여, $x_T$가 $x_0$와 거의 상호 정보가 없도록 만듭니다.
  * $\displaystyle\sum_{t>1}D_{KL}(q(x_{t−1}|x_t,x_0)∣∣p_θ(x_{t−1} ∣x_t))$
    * 이는 역방향 과정에서 각 스텝 t마다 $x_{t-1}$을 전송하는 데 필요한 추가적인 비트 수를 나타냅니다.
    * $q(x_{t-1}|x_t,x_0)$는 정방향 과정의 사후 분포(posterior)이며,
    * $p_{\theta}(x_{t-1}|x_t)$는 모델이 학습한 역방향 과정의 전이 분포입니다.
    * 이 KL 발산은 모델이 실제 전이 분포를 얼마나 잘 근사하는지를 측정하며, 이 값이 낮을수록 더 효율적인 디코딩이 가능합니다.


* 왜곡 (Distortion)
  * $L_0$를 왜곡으로 간주
    * $−logp_θ(x_0∣x_1)$: 이는 $x_1$로부터 최종 데이터 $x_0$를 복원하는 과정의 손실을 나타냅니다.
    * DDPM에서는 $x_0$가 이산적인 이미지 픽셀이므로, $p_{\theta}(x_0|x_1)$는 $x_1$에 조건화된 $x_0$의 이산 디코더로 정의
    * 이 값이 낮을수록 복원된 $x_0$가 원본에 가깝다는 것을 의미

* 왜곡 - 전송률 플롯 (Rate-Distortion Plot)
  * Distortion (RMSE): 왜곡은 평균 제곱근 오차(Root Mean Squared Error, RMSE)로 측정
    * $x_0$는 원본 이미지,$\hat{x}_0$는 $x_t$로부터 추정된 이미지이며, 왜곡은 $\sqrt{\parallel x_0-\hat{x}_0\parallel ^2/D}$로 계산됩니다.
    * 여기서 D는 데이터 차원입니다. 이는 [0,255] 스케일에서 측정됩니다.
  * Rate (bits/dim): 전송률은 특정 시간 t까지 수신된 누적 비트 수(cumulative number of bits received so far)로 계산됩니다.
    * 플롯의 X축은 역방향 과정 스텝 $(T-t)$를 나타내고, Y축은 각각 왜곡과 전송률을 나타냅니다.
    * 이 플롯에서 중요한 관찰은: 낮은 전송률 영역에서 왜곡이 급격히 감소한다는 것입니다.
    * 이는 모델이 처음에 적은 비트를 사용하여 이미지의 큰 스케일 특징(low-frequency information)과 같은 중요한 정보를 먼저 복원하고, 이후 더 많은 비트를 사용하여 미세한 디테일(high-frequency information)을 복원한다는 것을 의미합니다.
    * 대부분의 비트가 지각 불가능한 왜곡에 할당된다는 것

* 추정된 원본 이미지 $(\hat{x}_0)$
  * 프로그레시브 코딩 시나리오에서, 수신기는 임의의 타임스텝 t에서 $x_t$라는 정보를 가질 때, 다음 식을 통해 원본 이미지 $x_0$를 추정할 수 있습니다
  * $\hat{x}_0=(x_t−\sqrt{1−\bar{α}_t}ϵ_θ(x_t))/\sqrt{\bar{α}_t}$
  * 여기서 $\epsilon_{\theta}(x_t)$는 모델이 예측한 노이즈
  * 이 식은 $x_t=\sqrt{\bar{α}_t}x_0 + \sqrt{1−\bar{α}t}ϵ$ 에서 $x_0$에 대해 재정렬하여 얻어진 것으로, 주어진 $x_t$와 모델이 예측한 노이즈를 사용하여 원본 이미지를 역추정하는 과정입니다.


#### Likelihood Based Model

VAEs는 잠재 변수 모델로, 인코더와 디코더 네트워크를 학습합니다. 

인코더는 데이터를 잠재 공간으로 매핑하고, 디코더는 잠재 공간의 샘플을 다시 데이터 공간으로 매핑합니다. 

가능도의 하한(Evidence Lower Bound, ELBO)을 최적화하여 간접적으로 가능도를 다룹니다.

* 변분 오토인코더 (Variational Autoencoders, VAEs)
  * Conditional VAE: 조건부 정보를 이용하여 VAE를 확장한 모델입니다.
  * Beta-VAE: ELBO의 KL 발산 항에 가중치를 주어 잠재 공간의 disentanglement를 촉진합니다.


#### Log-likelihood Based Models

* 가능도 기반 모델을 실제로 학습하고 평가할 때 로그 가능도를 목적 함수로 사용하거나 평가 지표로 사용하는 경우를 지칭

#### NLL

NLL은 모델이 주어진 데이터를 얼마나 잘 설명하는지 또는 예측하는지를 나타내는 지표

음의 로그 가능도(Negative Log-Likelihood, NLL): 로그 가능도에 음수 부호를 붙인 것입니다. 

* 대부분의 최적화 알고리즘은 손실 함수를 최소화하는 방향으로 작동하기 때문에, 최대화 목표인 가능도/로그 가능도 대신 음의 로그 가능도를 최소화하는 방식을 사용합니다.
* NLL 값이 낮을수록 모델의 성능이 좋다는 것을 의미합니다.

$$NLL(θ)=−\displaystyle\prod_{i=1}^{N}logP(x_i∣θ)$$

---
### Interpolation
원본 이미지(source images) 사이의 **잠재 공간(latent space)**에서 부드러운 전환을 생성하여 새로운 이미지를 합성하는 기술

DDPM 논문에서 내부보간법은 다음과 같은 단계를 통해 이루어집니다:

1. 원본 이미지 인코딩 (확산 과정): 먼저 두 개의 원본 이미지 $x_0$​와 $x_0^{\prime}$를 가져옵니다.
    1. 이 이미지들은 확산 과정(forward process)을 통해 잠재 변수 공간으로 인코딩됩니다. 즉, 각 이미지에 점진적으로 가우시안 노이즈가 추가되어 $x_t$와 $x_t^{\prime}$라는 노이즈가 추가된 버전으로 변환됩니다.
    2. 논문에서는 "q를 확률적 인코더로 사용하여"라고 표현합니다.
    3. $x_t∼q(x_t∣x_0)$
    4. $x^{\prime}_t∼q(x_t∣x_0^{\prime})$
    5. 이 과정에서 다양한 노이즈(different values of λ)에 대해 노이즈가 고정되어 $x_t$와 $x_t^{\prime}$가 동일하게 유지될 수 있습니다.
2. 잠재 공간(latent space) 보간: 인코딩된 두 잠재 변수 $x_t$와 $x_t^{\prime}$사이를 선형적으로 보간하여 중간 잠재 변수 $\bar{x_t}$를 생성
    1. $x_t=(1−λ)x_t+λx_t^{\prime}$여기서 λ는 0과 1 사이의 값으로, 보간의 정도를 조절합니다. λ=0이면 $x_t$에 가까워지고, λ=1이면 $x_t^{\prime}$에 가까워집니다.
3. 이미지 공간 디코딩 (역방향 과정): 보간된 잠재 변수 $\bar{x_t}$ 를 역방향 과정(reverse process)을 통해 이미지 공간으로 디코딩하여 새로운 이미지 $\bar{x_0}$를 생성합니다.
    1. $\bar{x_0}∼p(x_0∣\bar{x_t})$
    2. 이 역방향 과정은 노이즈 제거된 버전의 원본 이미지를 선형 보간하는 것에서 발생하는 아티팩트(artifacts)를 제거하는 데 사용됩니다.

#### Interpolation의 의미
1. 품질과 다양성: DDPM은 이 과정을 통해 원본 이미지의 특징을 부드럽게 혼합한 그럴듯한(plausible) 이미지를 생성
2. 세분화된 제어: 확산 스텝 수(T−t)를 조절함으로써 보간의 세분화된 정도(fine granularities)와 거친 정도(coarse granularities)를 제어
    1. T−t가 작을수록 (확산 스텝이 적을수록): 원본 이미지의 구조가 많이 보존되어 픽셀 공간에서의 보간에 가깝게 됨
    2. T−t가 클수록 (확산 스텝이 많을수록): 원본 이미지의 정보가 더 많이 파괴되어 잠재 공간에서 완전히 새로운 샘플이 생성
3. 잠재 공간의 의미: 내부보간법은 DDPM의 잠재 공간이 의미 있는 정보를 인코딩하고 있음을 시사 



---

## 부록
### Markov Chain
#### 마르코프 성질 + 이산시간 확률 과정
마르코프 체인은 '마르코프 성질'을 가진 '이산시간 확률과정' 입니다.
마르코프 성질 - 과거와 현재 상태가 주어졌을 때의 미래 상태의 조건부 확률 분포가 과거 상태와는 독립적으로 현재 상태에 의해서만 결정됨
이산시간 확률과정 - 이산적인 시간의 변화에 따라 확률이 변화하는 과정
<p align="center">
<img src = "https://github.com/user-attachments/assets/7ae5afbc-7884-4e35-a570-cb87513daaf7" width="40%" height="40%">
</p> 

#### 결합확률분포(Joint Probability Distribution)
예를 들어 확률 변수 $X_1,X_2, ... , X_n$ 이 있다고 가정하면,
일반적으로 이 확률변수들의 결합확률분포는 다음과 같이 계산할 수 있다.

$$ P(X_1,X_2, ... , X_n) = P(X_1) \times P(X_2|X_1) \times P(X_3|X_2,X_1)\times  ...  \times P(X_n|X_{n-1}, X_{n_2} , ... , X_1) $$
 
하지만 마르코프 성질을 이용하면 위 보다 더 단순한 계산을 통해 결합확률분포를 구할 수 있다.

$$ P(X_n|X_{n-1}, X_{n_2} , ... , X_1) = P(X_{t+1}|X_t) $$
 

만약 어떠한 상태의 시점이고, 확률분포가 마르코프 성질을 따른다면 

$$ P(X_1,X_2, ... , X_n) = P(X_1) \times P(X_2|X_1) \times P(X_3|X_2)\times  ...  \times P(X_n|X_{n-1}) $$

단순화 할 수 있고 일반화를 적용하면 이전에 결합확률분포의 계산을 다음과 같이 단순화 가능하다.

### variational bound
* VAE(Variational Auto-Encoder)에서 쓰이는 개념으로 실제 분포와 추정 분포 두 분포 사이의 거리인 KL divergence를 최소화 시키기 위해 도입되는 개념입니다.
#### Variational Inference
* Variational Method에서 유래
* 복잡한 문제를 간다한 문제로 변화시켜 근사
* Variational Parameter변수를 추가로 도입
* 추정 문제를 최적화

##### Example $\lambda$ 도입하여 log x를 직선으로 근사
* $g(x) = log(x) \rightarrow f(x) = \lambda x - b(\lambda)$
* $f^*(\lambda) = \displaystyle\min_x \lbrace \lambda x - f(x) \rbrace$
* 확률 분포로 확장 $q(X) = A(X, \lambda_{0}), (where \ \lambda_{0} = arg\displaystyle\max_{\lambda} \lbrace A(X_0,\lambda)\rbrace)$


##### Variational Inference 식 유도
https://modulabs.co.kr/blog/variational-inference-intro

* $p(x)$ 확률분포, 은닉변수 $Z$, 양변에 log를 씌우면 Jensen 부등식을 통해 Lower Bound 표현
* $q(Z|\lambda)$ 에서 $\lambda$ 는 Variational Parameter, $\lambda$가 $q$에 작동한다는 표현
* $KL(p \parallel q) = \sum_Z p(Z) log p(Z) / q(Z)$ 로 정의, 두 확률분포의 차이를 계산하는 함수

$logp(X) = log(\displaystyle\sum_Z p(X,Z))$

$\ \ \ \ \ \ \ \ \ \ \ \ \  = log(\displaystyle\sum_Z p(X,Z)\frac{q(Z|\lambda)}{q(Z|\lambda)})$

$\ \ \ \ \ \ \ \ \ \ \ \ \  = log(\displaystyle\sum_Z q(Z|\lambda)\frac{p(X,Z)}{q(Z|\lambda)})$

$\ \ \ \ \ \ \ \ \ \ \ \ \  \ge \displaystyle\sum_Z q(Z|\lambda)log\frac{p(X,Z)}{p(Z|\lambda)}$
   



### 

