# Score-Based Generative Modeling through Stochastic Differential Equations
저자 : Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole

논문 : [PDF](https://arxiv.org/pdf/2011.13456)

일자 : Submitted on 26 Nov 2020 (v1), last revised 10 Feb 2021 (this version, v2), ICLR

<p align="center">
<img width="666" height="307" alt="image" src="https://github.com/user-attachments/assets/1ba90f03-e42b-4107-bbde-947fc08329fa" /></p>

## 핵심 아이디어
* Forward Process를 연속적인 시간의 흐름으로 보고 이 과정을 수학적으로 완벽히 되돌리면 노이즈에서 실제 이미지가 발생
* 이모든 과정을 확률적 미분 방정식(SDE)를 통해 구현

---

## Background

### SMLD
* Denoising Score Matching with Langevin Dynamics, SMLD
* SMLD는 데이터에 노이즈를 점진적으로 주입하고, 그 과정을 역전시켜 데이터를 생성하는 것을 학습하는 방식

#### SMLD 핵심
* Noise 주입
1. 데이터 분포 $p_{data}(x)$ 에 가우시안 노이즈 주입, 교란된 데이터 생성 $p_{\sigma}(\tilde{x})$, $p_{\sigma}(\tilde{x}|x) := \mathcal{N}(\tilde{x};x,\sigma^2I)$ 정규분포 따라는 교란 커널 사용(Perturbation Kernel)
2. $\sigma_{min} \sim \sigma_{max}$ 모두 양수 노이즈 시퀀스 사용, $\sigma_1 < \sigma_2 < ... < \sigma_N$
3. $\sigma_{min}$은 $p_{data}(x)$와 거의 동일하게 작고, $\sigma_{max}$는 $N(x; 0, \sigma_{max}^2I)$ 의 가우시안 분포를 따른다

* 스코어 함수 추정
1. 스코어 확률 밀도 그래디언트 $\nabla \log p_\sigma(x)$를 추정하기 위해 노이즈 조건부 스코어 네트워크 (NCSN, Noise Conditional Score Network) $s_\theta(x,\sigma)$ 라는 신경망을 훈련

$$
\theta^{*} = \arg \min_{\theta} \displaystyle\sum^{N}_{i=1} \sigma_i^2 \mathcal{E}_{p_{data}(x)}\mathcal{E}_{p_{\sigma_i}}(\tilde{x}|x) [\parallel s_{\theta}(\tilde{x}, \sigma_i) - \nabla_{\tilde{x}} \log p_{\sigma_i}(\tilde{x}|x) \parallel^2_2] \quad \quad (1)
$$

2. 충분한 데이터와 모델 용량이 주어지면, 최적의 스코어 기반 모델 $s_\theta(x, \sigma)$는 $\nabla_x \log p_\sigma(x)$와 거의 일치하게 된다.

* 샘플 생성 (Langevin Dynamics)

1. 학습된 스코어 네트워크를 사용하여 새로운 데이터를 생성할 때는 랑주뱅 MCMC(Markov Chain Monte Carlo) 동역학
2. 샘플링은 $\sigma_{max}$에 해당하는 단순한 가우시안 노이즈 $N(x | 0, \sigma_{max}^2I)$에서 $x_N$을 샘플링하는 것으로 시작
3. 이후 노이즈 스케일을 N부터 1까지 점진적으로 줄여나가면서 일련의 랑주뱅 MCMC 단계를 순차적으로 실행하여 $x_1$을 얻습니다.
4. 이 과정은 식 (2)에 설명되어 있습니다.
* M이 무한대로 가고 스텝 크기 $ε_i$가 0으로 갈 때, $x_M^1$은 $p_{σ_{min}}(x)$ 또는 $p_{data}(x)$ 에서 정확한 샘플이 됩니다.

$$
x_i^m = x_i^{m-1}+\epsilon_i s_{\theta}^{*}(x_i^{m-1},\sigma_i) + \sqrt{2\epsilon_i}z_i^m, \quad \quad m=1, 2, \cdots, M, \quad \quad (2)
$$

$$
M \to \infty, \quad \epsilon_i \to 0, \quad x_1^M \to p_{\sigma_{min}}(x) \approx p_{data}(x)
$$

---

랑주뱅 MCMC동역학
랑주뱅 동역학은 물리 현상을 설명하기 위한 모델이고, 랑주뱅 MCMC 동역학은 그 모델을 통계적 샘플링을 위한 알고리즘으로 응용한 것

* 랑주뱅 다이내믹스의 업데이트 규칙을 반복적으로 적용하여 마르코프 연쇄를 구성하는 샘플링 방법론 또는 알고리즘
* 랑주뱅 MCMC는 SDE(Stochastic Differential Equation) 솔버가 다음 타임스텝의 샘플에 대한 초기 추정치를 제공하는 "예측기(predictor)" 단계 이후에, 추정된 샘플의 주변 분포를 "교정(corrector)"하는 역할을 합니다.
* 예를 들어, SMLD(Score Matching with Langevin Dynamics)에서는 각 노이즈 스케일에서 스코어를 추정한 다음, 생성 과정 동안 점진적으로 노이즈 스케일을 줄여가며 랑주뱅 다이내믹스를 사용하여 샘플링합니다.

---
 
### 스코어 함수 (Score)

* 스코어 함수 $(\nabla x \log p_t(x))$
  * 특정 시간 t에서의 노이즈 낀 데이터 $x(t)$의 확률 분포 $(p_t(x))$ 에 로그를 씌운 뒤, 데이터 x에 대해 미분(gradient)한 값입니다.
  * 직관적으로 이 '스코어'는 노이즈 낀 데이터 상태에서 어느 방향으로 가야 더 진짜 데이터에 가까워지는지 알려주는 역할
  * 즉, 데이터의 밀도가 높은 방향을 가리키는 벡터입니다.
 
### SDE(Stochastic Differential Equation, 확률적 미분 방정식)
* 어떤 시스템의 시간에 따른 변화에 예측 불가능한 '무작위성(randomness)'이 포함될 때, 그 움직임을 설명하는 수학적 도구입니다.
* 일반적인 미분 방정식이 예측 가능한 움직임만을 다룬다면, SDE는 여기에 '랜덤한 충격'이 계속해서 더해지는 상황을 모델링합니다.

$$
dx=f(x,t)dt+g(t)dw \quad \quad (5)
$$

1. Drift Term (드리프트 항): $f(x,t)dt$

* 시간이 dt만큼 흘렀을 때, 현재 상태 x와 시간 t에 따라 시스템이 평균적으로 어느 방향으로 얼마만큼 움직이는지를 결정합니다.
* 비유: 강물에 떠 있는 나뭇잎이 있을 때, 강의 주된 흐름이 바로 드리프트입니다. 이 흐름은 나뭇잎을 대체로 하류로 밀어냅니다.

2. Diffusion Term (확산 항): $g(t)dw$

* 이 부분은 시스템의 예측 불가능한, 무작위적(stochastic)인 움직임을 나타냅니다.
* dw는 '위너 과정(Wiener Process)' 또는 '브라운 운동(Brownian Motion)'이라고 불리는 순수한 랜덤 노이즈를 의미합니다.
* 매 순간 아주 작은 랜덤한 충격을 주는 것이라고 생각할 수 있습니다.
* g(t)는 **확산 계수(diffusion coefficient)**로, 이 랜덤한 충격의 **세기(강도)**를 조절합니다.
* 시간이 지남에 따라 랜덤성의 영향이 커지거나 작아지게 할 수 있습니다.
* 비유: 강물 위의 나뭇잎이 물의 소용돌이나 바람 때문에 예측 불가능하게 이리저리 흔들리는 움직임이 바로 확산입니다.


### 역방향 SDE
* 역방향 SDE (Reverse SDE): 노이즈 분포에서 시작하여 노이즈를 천천히 제거하여 원래의 데이터 분포를 생성하는 과정입니다. 
* 앤더슨(Anderson, 1982)의 연구에 따르면, 확산 과정의 역과정 또한 확산 과정이며, 다음 역방향 SDE로 주어집니다

$$
dx = [f(x, t) - g(t)^2 \nabla x log p_t(x)]dt + g(t)d \bar{W}  \quad \quad (6)
$$

여기서 $p_t(x)$ 는 시간 t에서의 교란된 데이터 분포의 확률 밀도이며, $\nabla x log p_t(x)$가 바로 스코어 함수입니다

### DDPM

$$
p_{\alpha_i}(\mathbf{x}_i | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_i; \sqrt{\alpha_i}\mathbf{x}_0, (1 - \alpha_i)\mathbf{I}), \text{ where } \alpha_i := \prod_{j=1}^i (1 - \beta_j), \quad \beta_j = \text{positive noise scale}
$$

(2)식에 위 내용을 적용하면 아래와 같다

$$
\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \sum_{i=1}^N (1 - \alpha_i) \mathbb{E}_{p_{\text{data}}(\mathbf{x})} \mathbb{E}_{p_{\alpha_i}(\tilde{\mathbf{x}}|\mathbf{x})} \left[ \left\| \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, i) - \nabla_{\tilde{\mathbf{x}}} \log p_{\alpha_i}(\tilde{\mathbf{x}} | \mathbf{x}) \right\|_2^2 \right]. \quad \quad (3)
$$

식 (3)을 통해 $s_{\theta^{*}}(x, i)$를 학습한 후, 노이즈로 부터 새로운 데이터를 생성(역방향 프로세스)

$$x_{i-1} = \frac{1}{\sqrt{1-\beta_i}}(x_i - \beta_i s_{\theta^{*}}(x_i, i)) + \sqrt{\beta_i}z_i \quad (4)$$

(4)는 DDPM에서 샘플을 생성하는 한 단계를 나타냅니다.

* $x_i$: 현재 단계의 노이즈가 섞인 이미지입니다.

* $x_{i−1}$ : 우리가 계산하고자 하는, 노이즈가 한 단계 제거된 이전 이미지입니다.

* $s_{\theta^∗}(x_i,i)$: 현재 이미지 $x_i$ 와 현재 단계 $i$ 를 입력받아, 이미지에 추가된 노이즈를 예측하는 최적으로 학습된 신경망 모델입니다.

* $\frac{1}{\sqrt{1−β_i}}(x_i − \beta_{i}s_{\theta^∗}(x_i, i))$: 이 부분은 예측된 노이즈를 현재 이미지에서 제거하여 이미지를 더 깨끗하게 만드는 과정입니다. $\beta_i$는 해당 단계에서 추가되었던 노이즈의 양(variance)을 나타내는 작은 상수입니다.

* $z_i$: 평균이 0이고 분산이 1인 표준 정규분포에서 추출한 새로운 노이즈입니다.

* $\sqrt{\beta_i}z_i$: 생성 과정의 무작위성을 위해 약간의 노이즈를 다시 추가하는 항입니다. 이 과정은 생성된 샘플의 다양성을 보장합니다.

---

### 3.3 ESTIMATING SCORES FOR THE SDE

* x와 연속적인 시간 변수 t를 모두 입력으로 받는 시간 의존적인 스코어 기반 모델 $s_\theta(x,t)$ 를 훈련시킬 것을 제안
* 이 모델은 스코어 매칭(score matching)이라는 기법


$$
\theta^{*} = \arg\min_{\theta} \mathbb{E}_{t} \{ \lambda(t) \mathbb{E}_{x(0)} \mathbb{E}_{x(t)|x(0)} [\| s_{\theta}(x(t), t) - \nabla_{x(t)}\log p_{0t}(x(t)|x(0)) \|_{2}^{2}] \} \quad (7)
$$

1. 데이터 준비

* 무작위로 시간 t를 하나 뽑습니다 ($\mathbb{E}_{t}$).

* 원본 데이터셋에서 깨끗한 데이터 $x(0)$를 하나 샘플링합니다 ($\mathbb{E}_{x(0)}$).

* $x(0)$에 순방향 SDE를 따라 시간 t만큼 노이즈를 추가하여 손상된 데이터 $x(t)$를 만듭니다 ($\mathbb{E}_{x(t)|x(0)}$).

2. 손실 계산

* 모델의 예측값: 신경망 모델에 손상된 데이터 $x(t)$와 시간 t를 입력하여 스코어 $s_{\theta}(x(t), t)$를 예측합니다.

* 실제 정답: 실제 정답 스코어인 $\nabla_{x(t)}\log p_{0t}(x(t)|x(0))$를 계산합니다.

* 오차 계산: 모델의 예측값과 실제 정답 사이의 평균 제곱 오차(Mean Squared Error)를 계산합니다. 이 값이 바로 모델이 최소화해야 할 손실(loss)입니다.

3. 최적화

* 모든 가능한 시간 t와 데이터에 대해 이 오차의 기댓값을 최소화하는 모델 파라미터 $\theta^{*}$를 찾습니다 ($\arg\min_{\theta}$).

* $\lambda(t)$: 이 항은 가중치 함수로, 각기 다른 시간 t(즉, 다른 노이즈 수준)에서의 학습 중요도를 조절하여 훈련을 안정시키는 역할을 합니다.

결론적으로 수식 (7)은 "모든 시간과 데이터에 대해, 모델이 예측하는 스코어가 실제 스코어와 같아지도록 모델을 훈련시킨다"는 의미



### 3.4 EXAMPLES: VE, VP SDES AND BEYOND
* SMLD는 DDPM과 별개가 아니라, SDE의 서로다른 이산화(Discretization) 버전

* Variance Exploding(VE) : 시간이 지남에 따라 분산이 폭발하는 특징
* Variance Preserving (VP) : 드리프트 항과 확산형이 균형을 이루어 보존되는 특징

### 1. Variance Exploding (VE) SDE: SMLD의 연속적 일반화

#### 기존 SMLD 이산화 과정

$$x_{i}=x_{i-1}+\sqrt{\sigma_{i}^{2}-\sigma_{i-1}^{2}}z_{i-1}, \quad i=1,\cdot\cdot\cdot,N \quad (8)$$

* $x_{i-1}$은 이전 단계의 노이즈가 섞인 이미지입니다.

* $z_{i-1}$은 표준 정규 분포(평균 0, 분산 1)에서 샘플링된 노이즈입니다.

* $\sqrt{\sigma_{i}^{2}-\sigma_{i-1}^{2}}$는 노이즈의 강도를 조절하는 항입니다. i 단계에서 추가되는 노이즈의 분산(variance)이 $(\sigma_{i}^{2}-\sigma_{i-1}^{2})$가 되도록 하여, 최종적으로 i 단계까지 진행했을 때 추가된 총 노이즈의 분산이 $\sigma_i^2$가 되도록 만듭니다.

#### 연속적 SDE로 확장

* $N \rightarrow \infty$, 연속적 변화라 가정

$$dx = \sqrt{\frac{d}{dt}[\sigma^2(t)]} \, dw \quad (9)$$

* $dx$: 아주 짧은 시간 동안의 이미지 x의 변화량입니다.

* $dw$: 아주 작은 가우시안 노이즈 조각(Wiener process)입니다.

* $\sqrt{\frac{d[\sigma^2(t)]}{dt}}$ : 확산 계수(diffusion coefficient)라 불리며, 시간 t에 얼마나 강한 노이즈를 추가할지를 결정합니다. 이는 이산적인 과정에서의 $\sqrt{\sigma_{i}^{2}-\sigma_{i-1}^{2}}$에 해당합니다. 즉, 시간 $t$에 따른 분산( $\sigma^2(t)$ )의 변화율에 따라 노이즈의 크기가 결정됩니다. 이 식에는 $dt$ 항이 없습니다. 이는 데이터를 특정 방향으로 이끄는 드리프트(drift)가 없고, 순수하게 노이즈만 추가되는 과정임을 의미합니다.

### 2. Variance Preserving (VP) SDE: DDPM의 연속적 일반화

* DDPM도 데이터에 노이즈를 추가하지만 SMLD와는 약간 다른 방식

#### 기존 DDPM 이산화 과정

* DDPM Markov Chain

$$x_i = \sqrt{1 - \beta_i} x_{i-1} + \sqrt{\beta_i} z_{i-1} \quad (10)$$


* SMLD와 달리, DDPM은 두 가지 작용을 동시에 합니다.
* 축소: 이전 이미지 $x_{i-1}$를 $\sqrt{1-\beta_{i}}$만큼 약간 축소시킵니다. 이는 이미지를 원점(0) 방향으로 살짝 당기는 효과를 줍니다.
* 노이즈 추가: $\sqrt{\beta_{i}}$만큼의 강도로 노이즈 $z_{i-1}$를 추가합니다.


#### 연속적 SDE로의 확장
* $N \rightarrow \infty$, 연속적 변화라 가정, SDE로 수렴

$$dx = -\frac{1}{2}\beta(t)x \, dt + \sqrt{\beta(t)} \, dw \quad(11)$$

* $-\frac{1}{2}\beta(t)x dt$: 이 항은 드리프트 항(drift term)입니다. 현재 데이터 x를 원점(0) 방향으로(−x 때문에) 끌어당기는 역할을 합니다. 이는 이산적 과정의 축소 단계( $\sqrt{1−\beta_i}x_{i−1}$ )에 해당합니다.

* $\sqrt{\beta(t)}dw$ : 이 항은 확산 항(diffusion term)으로, 노이즈를 추가하는 역할을 합니다.


### 3. Sub-VP SDE: 새로운 제안 ("BEYOND")
* 논문은 기존 모델들을 SDE로 통합하는 것에서 더 나아가, 이 프레임워크를 기반으로 새로운 SDE를 제안합니다. 이것이 바로 sub-VP SDE

* Sub-VP SDE는 VP SDE와 유사하지만, 확산 계수가 다릅니다.

$$dx = -\frac{1}{2}\beta(t)x \, dt + \sqrt{\beta(t)\left(1 - e^{-2 \int_{0}^{t} \beta(s) ds}\right)} \, dw \quad (12) $$

* 드리프트 항: $-\frac{1}{2}\beta(t)x \, dt$ 로 VP SDE와 동일합니다.

* 확산 항: $\sqrt{\beta(t)\left(1 - e^{-2 \int_{0}^{t} \beta(s) ds}\right)} \, dw$. VP SDE와 비교하면, 노이즈 강도에 $(1-e^{-2\int_{0}^{t}\beta(s)ds})$라는 추가적인 항이 곱해져 있습니다. 이 값은 항상 1보다 작거나 같기 때문에, sub-VP SDE는 VP SDE보다 더 적은 양의 노이즈를 추가하게 됩니다.

* SDE가 만드는 확률 과정의 분산은 항상 같은 조건의 VP SDE가 만드는 분산보다 작거나 같습니다(bounded by). 이러한 특징 때문에 sub-VP SDE라는 이름이 붙었으며, 특히 이 모델은 가능도(likelihood) 계산에서 좋은 성능을 보입니다.

요약하자면, 3.4절은 SMLD는 VE SDE, DDPM은 VP SDE라는 연속적인 확률 과정의 특수한 이산화 사례임을 밝히고, 이 통합된 관점을 바탕으로 더 나은 성능을 보이는 sub-VP SDE라는 새로운 모델을 제안하는 중요한 부분입니다.

---

### 4. SOLVING THE REVERSE SDE

#### 4.1 General-Purpose Numerical SDE Solvers
* 오일러-마루야마(Euler-Maruyama)와 같은 기존의 수치 해석 기법을 역방향 SDE에 적용해 샘플을 생성하는 것
* Ancestral Sampling: DDPM에서 사용된 샘플링 방법으로, 이 논문에서는 이 역시 특정 SDE 솔버 중 하나임을 보여줍니다.
* Reverse Diffusion Sampler: 논문에서 새롭게 제안하는 솔버로, Table 1에서 볼 수 있듯이 기존의 Ancestral Sampling보다 약간 더 나은 성능을 보입니다

#### 4.2 PREDICTOR-CORRECTOR SAMPLERS
* 논문의 핵심 제안 중 하나로, 두 단계를 결합하여 샘플의 품질을 높이는 방식입니다.
* 예측(Predictor): 1번의 SDE 솔버를 사용해 다음 시간 단계의 샘플을 대략적으로 예측합니다.
* 수정(Corrector): 예측된 샘플을 score 기반 MCMC 방법(예: Langevin MCMC)을 사용해 한 번 더 정교하게 다듬어 오류를 수정합니다.
* 방식은 기존 SMLD(수정 단계 위주)와 DDPM(예측 단계 위주)의 샘플링 방식을 통합하고 개선하는 효과가 있습니다.

<img width="1251" height="236" alt="image" src="https://github.com/user-attachments/assets/2d68f664-4f38-4c0d-ab04-e7696583b3ee" />

이 표는 CIFAR-10 데이터셋에서 위 방법들의 이미지 생성 품질을 FID 점수(낮을수록 좋음)로 비교한 결과입니다.


* SDE 종류
  * Variance Exploding (VE) SDE: SMLD 방식에 해당합니다.
  * Variance Preserving (VP) SDE: DDPM 방식에 해당합니다.

* 샘플러 종류
  * P1000/P2000: 예측(Predictor) 단계만 1000번 또는 2000번 수행합니다.
  * C2000: 수정(Corrector) 단계만 2000번 수행합니다.
  * PC1000: 예측 1000번 + 수정 1000번을 수행하며, P2000, C2000과 계산량이 동일합니다.

#### 핵심 결론
* 수정 단계(Corrector) 단독 사용은 비효율적입니다.
  * C2000의 FID 점수(20.43, 19.06)는 다른 방법에 비해 매우 높게 나타나 성능이 좋지 않습니다.
* 예측-수정(PC) 샘플러는 성능을 크게 향상시킵니다.
  * 동일한 예측기(예: reverse diffusion)를 사용했을 때, 예측만 사용한 P1000보다 예측-수정 방식을 사용한다
  * PC1000의 FID 점수가 항상 더 낮습니다 (더 좋습니다).
* 단순히 예측 단계 수를 늘리는 것보다 예측-수정 방식이 더 효과적입니다.
  * 동일 계산량에서 예측 단계만 2000번 수행한 P2000보다, 예측과 수정을 1000번씩 결합한다.
  * PC1000이 대부분 더 좋은 성능을 보입니다.
* Probability Flow는 수정 단계와 결합될 때 가장 강력한 성능을 보입니다.
  * 예측기로만 사용하면 성능이 다소 떨어지지만(특히 VE SDE), 수정 단계와 결합된 PC1000에서는
  * 가장 낮은 FID 점수인 3.06을 기록하며 최고의 성능을 달성했습니다


#### 4.3 PROBABILITY FLOW AND CONNECTION TO NEURAL ODES

* SDE와 동일한 확률 분포를 따르지만, 무작위성이 없는 결정론적(deterministic)인 상미분 방정식(ODE)을 사용하는 방법입니다.

* 이 접근법은 샘플링 과정이 결정론적이므로 정확한 가능도(likelihood) 계산이 가능해지고, 잠재 공간(latent space)을 다루기 용이하며, 더 효율적인 샘플링이 가능하다는 장점이 있습니다

* 쉽게 비유하자면, 강물에 잉크를 떨어뜨렸을 때(데이터), 잉크 입자들이 물 분자와 무작위로 충돌하며 퍼져나가는 과정(SDE)이 있습니다. 반면, 각 지점에서 물의 평균적인 흐름 방향과 속도만을 따라 잉크 입자가 이동하는 경로를 생각할 수 있는데, 이것이 바로 확률 흐름 ODE에 해당합니다

* ODE equation

$$dx = \left[ f(x,t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x) \right] \quad (13) dt$$

* $dx$: x의 미소(infinitesimal) 변화량

* $f(x,t)$: 순방향 SDE의 드리프트 계수(drift coefficient). x가 평균적으로 어떻게 변하는지를 나타내는 항. 예를 들어, VP SDE의 경우  $f(x,t)=−\frac{1}{2}\beta(t)x$ 인데, 이는 시간이 흐를수록 데이터가 원점(0) 방향으로 점차 수축하는 경향을 보인다는 의미입니다.

* $g(t)$: 순방향 SDE의 확산 계수(diffusion coefficient). 주입되는 노이즈의 강도를 조절하는 함수

* $\nabla_x \log p_t(x)$: 점수 함수(score function). 특정 시간 t에서의 데이터 분포 $p_t(x)$에 대해 로그를 취한 뒤, 데이터 x에 대해 그래디언트(gradient)를 계산한 것. 이 값은 확률 밀도가 높은 방향, 즉 데이터가 존재할 가능성이 높은 방향을 가리키는 벡터입니다. 모델은 이 점수 함수를 신경망으로 근사하여 학습합니다.

* $- \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)$ : 이 항은 SDE의 무작위 확산(diffusion) 항을 결정론적인 흐름으로 변환하는 역할을 합니다. 점수 함수가 가리키는 방향(확률이 높아지는 방향)으로 x를 이동시켜, 노이즈를 제거하고 원본 데이터의 형태를 복원하는 핵심적인 부분입니다.

* 결론적으로, 방정식 (13)은 순방향 SDE의 평균적인 움직임(f(x,t))과 점수 함수를 통해 추정된 데이터 생성 방향$- \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)$을 결합하여, 노이즈( $x(T)$ )에서 시작해 원본 데이터( $x(0)$ )로 변환되는 결정론적인 경로를 제시합니다.

##### Neural ODEs

* 점수 함수 $\nabla_x \log p_t(x)$는 실제로 알 수 없기 때문에, 이를 신경망 모델 $s_\theta(x, t)$로 근사

$$\frac{dx}{dt} = f(x,t) - \frac{1}{2} g(t)^2 s_\theta(x,t)$$

* 정확한 가능도 계산 (Exact Likelihood Computation): 신경망 ODE의 특징인 '순간 변수 변환 공식(instantaneous change of variables formula)'을 사용하여 어떤 데이터에 대한 정확한 로그 가능도(log-likelihood)를 계산할 수 있습니다.

* 잠재 표현 조작 (Manipulating Latent Representations): 이 ODE를 순방향으로 풀면 데이터(x(0))를 잠재 공간의 표현(x(T))으로 인코딩할 수 있고, 역방향으로 풀면 다시 디코딩할 수 있습니다. 이는 이미지 보간(interpolation)과 같은 다양한 편집 작업을 가능하게 합니다.

* 고유하게 식별 가능한 인코딩 (Uniquely Identifiable Encoding): 순방향 과정(SDE)은 학습 가능한 파라미터가 없으므로, 완벽하게 학습된 점수 모델이 주어지면 어떤 데이터에 대한 잠재 표현은 유일하게 결정됩니다.

* 효율적인 샘플링 (Efficient Sampling): 범용 ODE 솔버(black-box ODE solver)를 사용하면 샘플링 속도와 정확도 사이의 균형을 유연하게 조절할 수 있습니다. 예를 들어, 오차 허용치를 높이면 계산량을 90% 이상 줄이면서도 시각적 품질 저하 없이 샘플을 생성할 수 있습니다

### 4.4 ARCHITECTURE IMPROVEMENTS

저자들은 주로 Ho et al. (2020)의 DDPM 아키텍처를 기반으로 다음과 같은 요소들을 추가하고 실험했습니다.


1. FIR을 이용한 안티앨리어싱 (Anti-aliasing with FIR): 이미지를 업샘플링하거나 다운샘플링할 때 발생하는 계단 현상이나 깨짐(aliasing)을 줄이기 위해 FIR(Finite Impulse Response) 필터를 사용했습니다. 이는 StyleGAN-2에서 효과가 입증된 기법입니다.

2. 스킵 연결 리스케일링 (Skip Connection Rescaling): 모델 내의 모든 스킵 연결(skip connection)의 가중치를 $1/\sqrt{2}$로 조정했습니다. 이는 ProgressiveGAN, StyleGAN 등 성공적인 GAN 모델들에서 학습 안정성과 성능을 높이기 위해 사용된 기법입니다.

3. BigGAN 잔차 블록 (BigGAN Residual Blocks): 기존 DDPM의 잔차 블록(residual block)을 더 성능이 좋은 것으로 알려진 BigGAN의 잔차 블록으로 교체했습니다.

4. 잔차 블록 수 증가 (Increased Residual Blocks): 각 해상도 레벨마다 사용되는 잔차 블록의 수를 2개에서 4개로 늘려 모델의 깊이와 표현력을 향상시켰습니다.

5. 점진적 성장 아키텍처 (Progressive Growing): 저해상도부터 고해상도까지 점진적으로 이미지를 생성하는 아키텍처를 도입했습니다. StyleGAN-2에서 영감을 받아 입력과 출력에 "input skip", "residual" 등 다양한 방식을 적용했습니다.


* 최적의 아키텍처: NCSN++와 DDPM++

* VE SDE (SMLD 계열): 위에서 언급된 모든 개선 사항이 평균적으로 성능 향상에 기여했습니다. 이 요소들을 조합하여 최적화된 아키텍처를 NCSN++라고 명명했습니다.

* VP SDE (DDPM 계열): VE SDE와는 약간 다르게, FIR 필터나 점진적 성장 아키텍처를 사용하지 않은 구성이 가장 좋은 성능을 보였습니다. 이 최적의 아키텍처는 DDPM++로 명명되었습니다


### 5 CONTROLLABLE GENERATION

* 핵심 아이디어: 조건부 역방향 SDE
* 역방향 과정에 '가이드'를 추가하는 개념
* 기존의 역방향 SDE 수식(Eq. 6)을 다음과 같이 수정합니다 (Eq. 14)

$$
dx = \left\{ f(x,t) - g(t)^2 \left[ \nabla_x \log p_t(x) + \nabla_x \log p_t(y \mid x) \right] \right\} dt + g(t) d\bar{w} \quad (14)
$$

* $\nabla_x \log p_t(x)$: 기존의 (무조건부) 점수 함수입니다. 데이터가 존재할 확률이 높은 방향을 알려줍니다.
* $\nabla_x \log p_t(y|x)$: 조건부 점수 함수입니다. 현재의 노이즈 섞인 이미지 $x(t)$가 우리가 원하는 조건 y를 만족할 확률이 가장 높아지는 방향을 알려주는 '가이드' 역할을 합니다.


#### 주요 응용 분야

1. 클래스 조건부 생성 (Class-Conditional Generation)
* "말 이미지를 만들어줘"와 같이 특정 클래스의 이미지를 생성하는 기술
* 작동 방식: 먼저, 다양한 노이즈 레벨의 이미지(x(t))를 보고 해당 이미지의 클래스(y)를 맞추는 시간 의존적인 분류기(time-dependent classifier)를 별도로 학습, 분류기를 '가이드'로 사용

2. 이미지 인페인팅 (Image Inpainting)
* 빈 공간을 자연스럽게 채워 넣는 기술
* 작동 방식: 여기서 조건(y)은 이미지의 '알려진 부분(known pixels)'이 됩니다. 모델은 알려진 픽셀 정보는 유지하면서, 알려지지 않은 '빈 공간(unknown dimensions)'에 대해서만 생성 과정을 진행

3. 이미지 컬러화 (Colorization) 
* 조건(y)은 '흑백 정보'가 됩니다. 모델은 주어진 흑백 정보를 바탕으로 비어있는 '색상 정보' 채널을 채워 넣는, 즉 인페인팅을 수행합

### 6. CONCLUSION

#### 성과 

* 기존 연구 통합 (SMLD + DDPM)
* 새로운 샘플링 알고리즘: 예측기-교정기(Predictor-Corrector)
* 정확한 가능도 계산: 확률 흐름 ODE를 통해 모델의 가능도(likelihood)를 정확하게 계산
* 잠재 공간 활용: 데이터를 잠재 코드로 고유하게 인코딩하고, 이를 조작하여 이미지를 편집하는 등 다양한 활용이 가능해졌습니다.
* 제어 가능한 생성: 클래스 조건부 생성, 인페인팅, 컬러화 등 사용자가 원하는 조건에 맞는 이미지를 생성

#### 한계점 및 향후 연구 방향
* 샘플링 속도: 제안된 샘플링 방법들이 성능을 개선했지만, 여전히 GAN과 같은 모델에 비해 샘플링 속도가 느립니다. 따라서 점수 기반 모델의 안정적인 학습과 GAN의 빠른 샘플링 능력을 결합하는 연구가 중요한 과제로 남아있습니다.

* 하이퍼파라미터: 다양한 샘플러를 사용할 수 있게 되면서, 조정해야 할 하이퍼파라미터의 수가 늘어났습니다. 앞으로 이러한 하이퍼파라미터를 자동으로 선택하고 조정하는 방법을 개선하고, 각 샘플러의 장단점을 더 깊이 연구할 필요가 있습니다










