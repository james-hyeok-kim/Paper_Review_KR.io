## Deep Unsupervised Learning using Nonequilibrium Thermodynamics
---

* 저자(소속) :
  * Jascha Sohl-Dickstein (Stanford University)
  * Eric A. Weiss(University of California, Berkeley)
  * Niru Maheswaranathan (Stanford University)
  * Surya Ganguli(Stanford University)

* 논문 : [PDF](https://arxiv.org/pdf/1503.03585)

* 일자 : 18 Nov 2015, ICML(International conference on machine learning)

---

### 1. Introduction

 #### 1.1 문제 설정: Tractability vs Flexibility

* 배경 문제: 머신러닝에서는 복잡한 데이터 분포를 잘 모델링하는 것이 목표입니다.
  * 그러나 대부분의 확률 모델은 다음 두 가지 중 하나를 만족하기는 쉽지만, 동시에 만족하기는 어렵습니다:

* Tractable: 수학적으로 계산이 가능해야 함 (예: 정규분포처럼 확률 계산이 쉬운 모델)

* Flexible: 복잡하고 다양한 분포를 표현할 수 있어야 함
(예: 어떤 함수 $φ(x)$든지 가능한 분포 $𝑝(𝑥) = \frac{𝜙(𝑥)}{𝑍}$)

* 문제점: 유연한 모델은 정규화 상수 
$Z=∫ϕ(x)dx$를 계산하기가 거의 불가능합니다.

* 샘플링이나 학습도 Monte Carlo 방식으로 매우 비쌈.

##### 1.1.1 기존 해결 시도들 → 이런 방법들은 트레이드오프를 완전히 해결하지는 못합니다.
* Variational Bayes
* Contrastive Divergence
* Score Matching
* Pseudolikelihood
* Loopy Belief Propagation
* Mean Field Theory
* Minimum Probability Flow

#### 1.2 논문의 핵심 아이디어
이 논문에서는 물리학(특히 비평형 통계물리학 nonequilibrium thermodynamics)에서 아이디어를 차용하여 다음과 같은 새로운 방법을 제안합니다:

##### 1.2.1 핵심 개념: 확산 프로세스 기반 생성 모델
* 데이터를 점점 노이즈화하는 Forward Diffusion Process를 정의하고,
* 그것을 역으로 되돌리는 Reverse Diffusion Process를 학습하여,
* 데이터 분포를 생성할 수 있는 모델로 삼습니다.

##### 1.2.2 모델의 장점:
* 유연한 구조: 수천 개의 레이어(타임스텝)도 사용 가능.
* 샘플링이 정확: 각 단계가 tractable한 확률분포라 전체 샘플링도 tractable.
* 확률 계산이 쉬움: Likelihood와 Posterior 계산이 효율적.
* 다른 분포와의 곱셈이 쉬움: 예를 들어 Posterior 계산시 조건부 분포와 곱하기가 가능.

---

### 2. Algorithm

#### 목표
* Forward diffusion process를 통해 복잡한 데이터 분포를 **단순한 분포(예: 정규분포)**로 변환

* Reverse diffusion process를 학습하여, 단순한 분포로부터 데이터를 복원하는 생성 모델을 정의

* 이 과정은 물리학에서 말하는 **비평형 확산(nonequilibrium diffusion)**에 기반하며, 확률모델 자체를 Markov chain의 종단 상태로 정의합니다.

---

#### 2.1 Forward Trajectory (Inference Process)

##### 2.1.1 핵심 개념:
데이터 분포 $q(x^(0))$로부터 시작해서, 확산 커널을 반복 적용하여 점점 구조를 무너뜨려 단순한 분포 $π(x^(T))$로 만듭니다.
$q(x^{(t)}∣x^{(t−1)})=T_{π}(x^{(t)}∣x^{(t−1)};β_t)$

즉, 노이즈를 점점 추가하는 Markov chain입니다.
* $T_{π}$: 확산 커널 (Gaussian, Binomial 등)
* $β_{t}$: 확산 강도(시간마다 다르게 설정 가능)
* 𝑇: 전체 타임스텝 수

전체 확률 경로:

$$q(x^{(0:T)})=q(x^{(0)}) \displaystyle\prod^{t=1}_{t=1}q(x^{(t)}∣x^{(t−1)})$$

---

### 2.2 Reverse Trajectory (Generative Process)
#### 2.2.1 핵심 개념
* 정규분포 $𝜋(𝑥^{(𝑇)})$ 에서 시작해서, 학습된 reverse Markov chain을 거쳐 원래 데이터 분포로 되돌아가는 생성 모델.
$$p(x^{(0:T)})=p(x^{(T)}) \displaystyle\prod_{t=1}^{T} p(x^{(t−1)} ∣x ^{(t)})$$

##### 핵심:
* 각 스텝 $p(x^{(t−1)}|x^{(t)})$는 forward의 역과 같은 구조를 갖되, mean, covariance 또는 flip 확률만 학습하면 됩니다.

* Gaussian: mean $f_{μ}(x^{(t)} ,t)$, covariance $f_Σ (x^{(t)} ,t)$

* Binomial: bit flip rate $f_b(x^{(t)},t)$

---

### 2.3 Model Probability (Log-Likelihood)
직접 $p(x^{(0)})$ 를 계산하기는 어렵지만, forward/reverse 경로의 확률비를 계산해 근사합니다:

$$p(x^{(0)})=E_{q(x^{(1:T)}∣x^{(0)})}\left[\frac{p(x^{(0:T)})}{q(x^{(1:T)}|x^{(0)})} \right]$$

이건 Annealed Importance Sampling과 Jarzynski Equality와 유사한 방식입니다.

#### 개념정리 Log-Likelihood란?
Likelihood
* 어떤 데이터 $x$가 관측되었을때, 모델이 그 데이터를 낼 확률
* $p_{\theta}(x)$:파라미터 $\theta$를 가진 모델이 $x$를 생성할 확률
* 여러 데이터가 있을 경우 전체 likelihodd는 곲

  $$L(\theta)=\displaystyle\prod_{i=1}^Np_{\theta}(x_i)$$

* Log-Likelihodd는 위 확률의 로그를 취한 것
* 곱이 너무 작아지는 것을 방지, 수학적으로 미분이 쉬워서 최적화에 유리

---

### 2.4 Training (Log-likelihood Bound Maximization)
로그 가능도(Log likelihodd): $𝐿=𝐸_{𝑞(𝑥^{(0)})}[\log ⁡𝑝(𝑥^{(0)})]$

이는 직접 계산이 어려워서 Jensen's inequality로 lower bound 𝐾를 도입:


$$𝐿≥𝐾=−\displaystyle\sum_{𝑡=2}^{𝑇}𝐸_{𝑞(𝑥^{(0)},𝑥^{(𝑡)})}[𝐷_{𝐾𝐿}(𝑞(𝑥^{(𝑡−1)}∣𝑥^{(𝑡)},𝑥^{(0)})∥𝑝(𝑥^{(𝑡−1)}∣𝑥^{(𝑡)}))]+entropy terms$$

즉, reverse transition과 posterior 간의 KL divergence를 최소화하는 것이 학습의 핵심입니다.

* DKL (Kullback-Leibler Divergence)
* $D_{KL}(P∣∣Q)=∑_x P(x) \log \frac{P(x)}{Q(x)}$

|제목|내용|
|:---|:---|
|$q(x^{(t)}∣x^{(0)})$ | forward diffusion 과정: $x^{(0)}→x^{(T)}$|
|$p(x^{(t−1)}∣x^{(t)})$ | 학습하고자 하는 reverse process (생성 모델)|
|D_{KL}(q∣∣p) |	두 분포 사이의 차이 측정|
|$E_{q(x^{(0)},x^{(t)})}[⋅]$ |	forward process에서 샘플링된 샘플로부터 기대값을 계산|
|entropy terms| 일부는 정규화상수 또는 모델에 상관없는 항으로 무시 가능|

✅ 학습 대상: 각 스텝의 reverse kernel (mean, covariance, flip rate 등)

#### 2.4.1 Diffusion Schedule $𝛽_𝑡$
* Gaussian: $𝛽_1$은 고정, 나머지는 gradient로 학습 가능

* Binomial: $𝛽_𝑡 = \frac{1}{𝑇−𝑡+1}$등 일정한 노이즈 감소 스케줄 사용

---

### 2.5 Posterior 계산 및 분포 곱셈

문제 : 이미지 복원, inpainting, denoising 등에서는 $p(x^{(0)}∣known data)$ 계산이 필요합니다.

해결: Diffusion model은 임의의 함수 $r(x^{(0)})$를 곱해서 새로운 분포 $\tilde{p}(x^{(0)})∝p(x^{(0)})r(x{(0)})$를 구성할 수 있습니다.

이는 reverse kernel에 perturbation을 주는 방식으로 처리됨

Gaussian인 경우엔 closed-form으로 처리 가능

---

### Posterior란? (사후확률)

#### 정의 

어떤 관측값 y가 주어졌을 때, 잠재 변수 𝑥가 실제로 어떤 값일지를 예측하는 확률분포예요.

$$p(x|y) = \frac{p(x) \cdot p(y|x)}{p(y)}$$

* $p(x)$: 사전 확률 (prior)

* $p(y∣x)$: likelihood (관측값의 조건부 확률)

* $p(x∣y)$: posterior (우리가 알고 싶은 것)

* $p(y)$: 정규화 상수

* “이런 결과가 관측되었는데, 원인은 무엇일까?” →  원인을 추정하는 역추론적 사고가 posterior입니다.

#### 논문 맥락에서의 Posterior

$$ posterior \propto p(x^{(0)}) \cdot r(x^{(0)})$$

* $p(x^{(0)})$ : 모델 분포
* $r(x^{(0)})$ : 조건, 관측된 부분분
  
---

### Appendix
#### 1. Monte Carlo
  - 정확한 계산이 어렵거나 불가능할 때, 많은 수의 랜덤 샘플을 뽑아서 평균을 내면 근사값이 된다는 아이디어입니다.

##### 1-1. 예시
* 정적분 근사 어떤 함수 $f(x)$의 정적분을 계산하고 싶을 때, $I = \int\limits_a^b f(x)dx$ 이걸 직접 계산하기 어렵다면,
* 구간 [𝑎, 𝑏]에서 무작위로 N개의 샘플 $X_1, ... X_n$을 뽑아서
  
$I \approx \frac{𝑏−𝑎}{N} \displaystyle\sum_{i=1}^{N} f(x_i) $
처럼 샘플 평균 근사 가능


##### 1-2. 단점
* 계산량이 많아요. (샘플 수가 커야 정확해짐)
* 분산이 클 수 있음 → 샘플 수가 부족하면 근사값이 매우 부정확
* 좋은 샘플링 분포를 잘 선택해야 함 (안 그러면 "희소 영역"은 놓침)

---

#### 2. Variational Bayes (VB)
##### 2-1. 목적:복잡한 **posterior $p(z∣x)$**를 직접 계산하기 어려울 때, tractable한 분포 $q(z)$ 로 근사해서 **Evidence Lower Bound (ELBO)**를 최적화함.
##### 2-2. 아이디어
* $\log p(x) ≥ E_{q(z)} [\log p(x,z) − \log q(z)]$

##### 2-3. 한계:
* 선택한 $q(z)$가 실제 posterior를 잘 못 따라가면 부정확
* 모델과 inference 분포 사이 비대칭성 → 학습 어려움


#### 3. Contrastive Divergence (CD)
##### 3-1. 목적
* Boltzmann machine 같이 정규화 상수 Z가 없는 에너지 기반 모델의 파라미터 학습

##### 3-2. 아이디어:
* Gibbs Sampling으로 한두 step만 진행하여 실제 분포와 모델 분포 간 차이를 줄임
* 학습 대상: 데이터 분포와 모델 분포 간의 차이 (score function)

$Δθ∝E_{data}[f(x)]−E_{model}[f(x)]$

##### 3-3. 한계:
* 근사 샘플링이기 때문에 이론적으로 보장 안 됨
* 많게는 수천 스텝 필요 → 비효율적


#### 4. Score Matching
##### 4-1. 목적:
* 정규화 상수 없는 확률분포에서도 학습 가능하게 함

##### 4-2. 아이디어:
score function $∇_x \log p(x)$을 이용해 다음을 최소화:

$J(θ)=E_{p_{data}} [∥∇_{x} \log p_{θ}(x)−∇_{x} \log p_{data}(x)∥^{2}]$

##### 4-3. 한계:
* 2차 도함수 계산 필요 → 고차원에서는 느림
* 샘플링 자체는 불가능

#### 5. Pseudolikelihood
##### 5-1. 목적
* Markov Random Field 같은 모델에서 복잡한 joint likelihood 대신 조건부 확률만 사용

$PL(x)= \displaystyle\prod_{i}p(x_i|x_{i-1})$

##### 5-2. 한계
* 조건부 확률만 최대화 → global 구조 학습에는 한계
* Likelihood를 직접 최적화하는 것보다 정확도 낮을 수 있음

#### 6. Loopy Belief Propagation (LBP)

##### 6-1. 목적
* 그래프 모델 (특히 MRF, CRF)에서 근사적인 marginal inference

##### 6-2. 아이디어
* 메시지 전달 알고리즘을 사이클이 있는 그래프에도 적용

* 반복적 메시지 전달을 통해 근사분포 계산

##### 6-3. 한계
* 수렴이 보장되지 않음

* 근사 정확도 떨어질 수 있음

#### 7. Mean Field Theory
##### 7-1. 목적
* 복잡한 분포를 독립된 단일 변수의 곱으로 근사

$p(x)≈ \displaystyle\prod_{i} q_{i}(x_{i})$

##### 7-2. 한계
변수 간 의존성 무시 → 복잡한 구조 표현 불가능

간단한 구조에서는 작동하지만 일반화에 약함

#### 8. Minimum Probability Flow (MPF)
##### 8-1. 목적
* Energy-based model에서 정규화 상수 없이도 학습 가능하게 함

##### 8-2. 아이디어
* 데이터 분포 $p_{data}$에서 가까운 이웃 상태로의 확률 흐름을 줄이도록 학습
* 즉, 데이터와 비데이터 사이의 flow를 줄이기

$min_{𝜃} \displaystyle\sum_{𝑥∈data} \displaystyle\sum_{𝑥'}𝑇(𝑥′∣𝑥) \log \frac{𝑝_{𝜃}(𝑥′)}{𝑝_{𝜃}(𝑥)}$

##### 8-3. 한계
* 이웃 상태 정의에 따라 결과가 민감
* 복잡한 분포에는 한계

---

##### 9. Annealed Importance Sampling (AIS)
* 복잡한 분포에서 샘플링
* 중간 분포열 설정: AIS는 쉬운 시작 분포 $(π_0)$와 샘플링하고자 하는 목표 분포 $(π_N)$ 사이에 일련의 중간 분포들 $(π_1,π_2,…,π_{N−1})$ 을 정의
* 분포가 부드럽게 전환되도록 설계, $π_k(x)∝(π_0(x))^{1−β_k}(π_N(x))^{β_k}$와 같은 형태로 정의할 수 있으며, $β_k$는 0에서 1까지 점진적으로 증가
* 중요도 가중치 경로 $(x_0, \ldots, x_N)$에 대한 가중치 w는 다음과 같이 정의됩니다.
$$w=\frac{π_1(x_0)π_2(x_1)}{π_0(x_0)π_1(x_1)} ⋯ \frac{π_N(x_{N−1})}{π_{N−1}(x_{N−1})} \frac{전이확률 역과정}{전이확률 정과정}$$  (정확한 형태는 사용된 MCMC 전이 커널에 따라 달라진다)

##### 10. Jarzynski Equality
* Jarzynski Equality는 비평형 과정을 통해 두 평형 상태 간의 자유 에너지 차이(ΔF)를 통계적으로 연결하는 중요한 등식

* 수식: $ΔF=−k_BTln⟨e^{−W/k_BT}⟩$
    - $ΔF$: 초기 평형 상태와 최종 평형 상태 사이의 헬름홀츠 자유 에너지 변화
    - $k_B$ : 볼츠만 상수
    - $T$: 온도
    - $W$: 비평형 과정 동안 시스템에 가해진 일 (work)
    - $⟨⋅⟩$: 비평형 과정의 앙상블 평균 (즉, 동일한 비평형 과정을 여러 번 반복했을 때 얻어지는 일 W 값들의 지수 평균)


Jarzynski Equality는 고전 열역학의 제2법칙(ΔF≤W)을 통계역학적으로 확장한 것으로 해석될 수 있습니다.
$\left< e^{-W/k_B T} \right>$ 는 항상 $e^{-\left< W \right>/k_B T}$보다 크거나 같으므로, $\Delta F \le \left< W \right>$가 성립합니다. 이는 평균적으로는 비가역 과정의 일이 항상 자유 에너지 변화보다 크거나 같다는 열역학 제2법칙을 만족합니다.
