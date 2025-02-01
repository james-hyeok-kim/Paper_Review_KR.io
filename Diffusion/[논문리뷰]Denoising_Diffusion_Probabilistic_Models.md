# Denoising Diffusion Probabilistic Models - 1
저자(소속) : Jonathan Ho (UC Berkeley), Ajay Jain(UC Berkeley), Pieter Abbeel(UC Berkeley)

논문 : [PDF](https://arxiv.org/pdf/2006.11239)

일자 : 16 Dec 2020

## 초록
Implementation : [Git](https://github.com/hojonathanho/diffusion)

## 도입
<p align="center">
<img src = "https://github.com/user-attachments/assets/1351575a-8638-446c-9a9b-d5d9dc8db15c" width="60%" height="60%">
</p> 

Markov chain forwarding 방식으로 noise를 더하고, reverse방식으로 noise에서 이미지를 생성

## 배경
### Reverse Process $p_{\theta}$
* $p_{\theta}(x_{0:T}) \rightarrow reverse \ process$
* Markov chain with learned Gaussian transitions, $p(x_T) = N(x_T;0,I):$ (Normal distribution)
* 보통 Normal Distribution의 표현 $N(\mu, \sigma^2)$ 평균$(\mu)$, 분산$(\sigma)$로 표현
* $p_{\theta}(x_{0:T}) := p(x_{T})\displaystyle\prod_{t=1}^{T}p_{\theta}(x_{t-1}|x_{t}), \ \ \ p_{\theta}(x_{t-1}|x_t) :=  N (x_{t-1};\mu_{\theta}(x_t,t),\sum_{\theta}(x_t,t))$

### Forward Process (Diffusion Process) $q$
* $q(x_{1:T}|x_0) := \displaystyle\prod_{t=1}^{T}q(x_t|x_{t-1}), \ \ \ q(x_t|x_{t-1}) := N(x_t;\sqrt{1- \beta_{t}}x_{t-1},\beta_{t}I)$
* Variance(Noise) Schedule $\beta_1, ... , \beta_T:$

### Training (학습)
* Variational Bound를 최적화 하는 형태로 진행
* Negative log likelihood
* $E\left[ -log p_\theta(x_0) \right] \le E_q \left[ -log \frac{p_\theta (x_{0:t})}{q(x_{1:T}|x_0)} \right] = E_q \left[ -log p(x_T) - \displaystyle\sum_{t \ge 1} log \frac{p_\theta (x_{t-1})}{q(x_{t}|x_{t-1})} \right] =: L$

### 추가적인 조건
* sampling $x_t$ at arbitrary timestep t
* $\alpha_t := 1-\beta_t$
* $\bar{\alpha_t}=\prod^t_{s=1}\alpha_s$
* $q(x_t|x_0) = N(x_t;\sqrt{\bar{a_t}}x_0, (1-\bar{\alpha_t})I)$


## 실험

## 결과

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
* $KL(p||q) = \sum_Z p(Z) log p(Z) / q(Z)$ 로 정의, 분포간 얼마나 떨어져 있는지 표현

$logp(X) = log(\displaystyle\sum_Z p(X,Z))$

$\ \ \ \ \ \ \ \ \ \ \ \ \  = log(\displaystyle\sum_Z p(X,Z)\frac{q(Z|\lambda)}{q(Z|\lambda)})$

$\ \ \ \ \ \ \ \ \ \ \ \ \  = log(\displaystyle\sum_Z q(Z|\lambda)\frac{p(X,Z)}{q(Z|\lambda)})$

$\ \ \ \ \ \ \ \ \ \ \ \ \  \ge \displaystyle\sum_Z q(Z|\lambda)log\frac{p(X,Z)}{p(Z|\lambda)}$
   

### 
