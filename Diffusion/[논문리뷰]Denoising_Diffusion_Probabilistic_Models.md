# Denoising Diffusion Probabilistic Models - 1
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
* $𝑝(𝑧): 간단한 prior 분포 (ex: $𝑁(0,𝐼)$)
* $𝑝_𝜃(𝑥∣𝑧)$: decoder (복원 모델)
* 이 모델에서 $log𝑝_𝜃(𝑥)$ 를 직접 계산하는 건 어렵다. → 추정을 통해 근사.


#### Variational Inference and ELBO
🔹 아이디어:

* 복잡한 $𝑝_𝜃(𝑥)$ 를 직접 계산하는 대신, **ELBO (Evidence Lower Bound)**를 최대화해서 근사한다.

* $𝑞_𝜙(𝑧∣𝑥)$ : encoder 또는 approximate posterior

🔹 ELBO 정의

$$log𝑝_𝜃(𝑥)≥𝐸_{𝑞_𝜙(𝑧∣𝑥)}[log⁡𝑝_𝜃(𝑥∣𝑧)]−𝐷_{KL}(𝑞_𝜙(𝑧∣𝑥)∥𝑝(𝑧))$$

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

### Forward Process (Diffusion Process) $q$
* $q(x_{1:T}|x_0) := \displaystyle\prod_{t=1}^{T}q(x_t|x_{t-1}), \ \ \ q(x_t|x_{t-1}) := N(x_t;\sqrt{1- \beta_{t}}x_{t-1},\beta_{t}I)$
* 작은 가우시안 노이즈를 T단계에 걸쳐 점차 추가
* Variance(Noise) Schedule $\beta_1, ... , \beta_T:$ - 미리 정해둔 노이즈값 (예: 0.0001 ~ 0.02)
* $\sqrt{1- \beta_{t}}$ 로 scaling하는 이유는 variance가 발산하는 것을 막기 위해서
* 여기서 $𝛼_𝑡:=1−𝛽_𝑡$, $\bar{α_t}:=\displaystyle\prod_{𝑠=1}^{𝑡}𝛼_𝑠$

👉 즉, 한번에 $𝑥_0$ 에서 $𝑥_𝑡$를 샘플링할 수 있음.

### Reverse Process $p_{\theta}$
* $p_{\theta}(x_{0:T}) \rightarrow reverse \ process$
* Markov chain with learned Gaussian transitions, $p(x_T) = N(x_T;0,I):$ (Normal distribution)
* 보통 Normal Distribution의 표현 $X \sim N(\mu, \sigma^2)$ 평균 $(\mu)$ , 분산 $(\sigma)$ 로 표현
* $p_{\theta}(x_{0:T}) := p(x_{T})\displaystyle\prod_{t=1}^{T}p_{\theta}(x_{t-1}|x_{t}),  \ \ \ p_{\theta}(x_{t-1}|x_t) :=  N (x_{t-1};\mu_{\theta}(x_t,t),\sum_{\theta}(x_t,t))$


### Training (학습)
* Variational Bound를 최적화 하는 형태로 진행
* Negative log likelihood
* $E\left[ -log p_\theta(x_0) \right] \le E_q \left[ -log \frac{p_\theta (x_{0:t})}{q(x_{1:T}|x_0)} \right] = E_q \left[ -log p(x_T) - \displaystyle\sum_{t \ge 1} log \frac{p_\theta (x_{t-1})}{q(x_{t}|x_{t-1})} \right] =: L$
  * $$L=E_q [log\frac{p_θ(x_0,…,x_T)}{q(x_1,…,x_T ∣x_0)}]$$
이 loss는 다음처럼 분해 가능:
  * $$𝐿=𝐸_𝑞[\displaystyle\prod_{𝑡=1}^{𝑇}𝐷_{𝐾𝐿}(𝑥𝑡−1∣𝑥𝑡,𝑥0)∥𝑝𝜃(𝑥𝑡−1∣𝑥𝑡))−log⁡𝑝𝜃(𝑥0∣𝑥1)]$$
* $E$는 기대값, 변수 x의 기대값으로 반복실험의 평균적인 값을 의미

### Parameterization (Noise Prediction)
역방향 분포의 평균은 다음과 같이 계산 가능:

$$𝜇_𝜃(𝑥_𝑡,𝑡)=\frac{1}{𝛼_𝑡}(𝑥_𝑡−\frac{𝛽_𝑡}{\sqrt{1−\bar{𝛼_t}}}𝜖_𝜃(𝑥_𝑡,𝑡))$$

여기서 $𝜖_𝜃(𝑥_𝑡,𝑡)$은 neural network가 예측하는 노이즈

즉, 모델은 실제로 노이즈를 예측하는 방식으로 학습됨 (MSE loss)

### 추가적인 조건
* sampling $x_t$ at arbitrary timestep t
* $\alpha_t := 1-\beta_t$
* $\bar{\alpha_t}=\prod^t_{s=1}\alpha_s$
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

🔹 4.2 Score Matching과 연결
DDPM의 denoising 모델은 사실상 score function을 예측하고 있습니다.

* DDPM은 각 시점 𝑡마다, $𝑥_𝑡∼𝑞(𝑥_𝑡∣𝑥_0)에서 샘플링하고

* 네트워크는 노이즈 $𝜖_𝜃(𝑥_𝑡,𝑡)$를 예측 → 이건$∇_{𝑥_𝑡}log𝑞(𝑥_𝑡)$의 방향과 같은 역할

따라서 DDPM은 score function을 간접적으로 학습한다고 볼 수 있음

🔹 4.3 결론: DDPM ≈ Score-based Model
논문은 다음과 같이 요약합니다:

```
In the limit of small $𝛽_𝑡$ the reverse DDPM process becomes equivalent to Langevin dynamics with a learned score function.
```

즉:

* $𝛽_𝑡$ → 0 : 매우 작은 노이즈 단계에서

* 역방향 DDPM 과정은 Langevin dynamics와 동일해짐

* 따라서 DDPM은 score-based 생성 모델의 특수한 형태로 해석 가능

---

## 실험

---

## 결과

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
