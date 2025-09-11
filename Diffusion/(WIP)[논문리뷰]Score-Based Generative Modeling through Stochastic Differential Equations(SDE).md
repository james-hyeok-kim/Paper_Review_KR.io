# Score-Based Generative Modeling through Stochastic Differential Equations
저자 : Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole

논문 : [PDF](https://arxiv.org/pdf/2011.13456)

일자 : Submitted on 26 Nov 2020 (v1), last revised 10 Feb 2021 (this version, v2), ICLR


<img width="666" height="307" alt="image" src="https://github.com/user-attachments/assets/1ba90f03-e42b-4107-bbde-947fc08329fa" />

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
    ◦ M이 무한대로 가고 스텝 크기 εi가 0으로 갈 때, xM1은 pσmin(x) 또는 p_data(x)에서 정확한 샘플이 됩니다.

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
dx=f(x,t)dt+g(t)dw
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

#### SDE(Stochastic Differential Equation) 프레임워크와의 연결

* DDPM에서 사용되는 노이즈 교란은 분산 보존(Variance Preserving, VP) SDE $(dx = -1/2 \beta(t)x dt + \sqrt{\beta}(t) dw)$ 의 이산화로 간주될 수 있습니다.
* 즉, 이산적인 노이즈 스케일 시퀀스 $\beta_i$는 연속적인 시간 함수 $\beta(t)$ 로 일반화됩니다.


    ◦ DDPM 모델에서 βi는 일반적으로 **등차수열(arithmetic sequence)**을 따르며, βi = β̄min/N + (i-1)/(N(N-1))(β̄max - β̄min)와 같이 정의됩니다. N이 무한대로 갈 때, β(t) = β̄min + t(β̄max - β̄min)로 수렴합니다. 실험에서는 β̄min = 0.1, β̄max = 20과 같은 특정 값이 사용됩니다.
• 수치적 고려사항: DDPM에 해당하는 VP SDE는 t=0에서 불연속성 문제는 없지만, t→0일 때 x(t)의 분산이 소멸되어 훈련 및 샘플링 시 수치적 불안정성 문제가 발생할 수 있습니다. 따라서 실제 계산에서는 t를 [ε, 1] 범위로 제한하며, 샘플링에는 ε = 10^-3, 훈련 및 가능도 계산에는 ε = 10^-5와 같은 작은 상수가 사용됩니다.
• 일반화 및 개선: 새로운 SDE 기반 프레임워크는 DDPM의 원래 샘플링 방법인 선조 샘플링을 **예측기(predictor)**로 사용하고, 항등 함수(identity function)를 **교정기(corrector)**로 사용하는 예측기-교정기(Predictor-Corrector, PC) 샘플러로 일반화 및 개선합니다.
요약하자면, β1에서 βN까지의 노이즈 스케일 시퀀스는 DDPM의 핵심적인 부분으로, 데이터를 점진적으로 노이즈로 손상시키고 이를 역전시키는 확률적 과정을 정의하며, 연속적인 SDE 프레임워크에서는 β(t) 함수로 일반화되어 모델의 훈련과 샘플링에 중요한 역할을 합니다.
