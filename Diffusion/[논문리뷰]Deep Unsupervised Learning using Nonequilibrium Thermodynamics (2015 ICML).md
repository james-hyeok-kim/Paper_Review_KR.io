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

그러나 대부분의 확률 모델은 다음 두 가지 중 하나를 만족하기는 쉽지만, 동시에 만족하기는 어렵습니다:

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


🔍 3. Score Matching
✅ 목적:
정규화 상수 없는 확률분포에서도 학습 가능하게 함

⚙️ 아이디어:
score function 
∇
𝑥
log
⁡
𝑝
(
𝑥
)
∇ 
x
​
 logp(x)을 이용해 다음을 최소화:

𝐽
(
𝜃
)
=
𝐸
𝑝
𝑑
𝑎
𝑡
𝑎
[
∥
∇
𝑥
log
⁡
𝑝
𝜃
(
𝑥
)
−
∇
𝑥
log
⁡
𝑝
𝑑
𝑎
𝑡
𝑎
(
𝑥
)
∥
2
]
J(θ)=E 
p 
data
​
 
​
 [∥∇ 
x
​
 logp 
θ
​
 (x)−∇ 
x
​
 logp 
data
​
 (x)∥ 
2
 ]
⚠️ 한계:
2차 도함수 계산 필요 → 고차원에서는 느림

샘플링 자체는 불가능

🔍 4. Pseudolikelihood
✅ 목적:
Markov Random Field 같은 모델에서 복잡한 joint likelihood 대신 조건부 확률만 사용

𝑃
𝐿
(
𝑥
)
=
∏
𝑖
𝑝
(
𝑥
𝑖
∣
𝑥
∖
𝑖
)
PL(x)= 
i
∏
​
 p(x 
i
​
 ∣x 
∖i
​
 )
⚠️ 한계:
조건부 확률만 최대화 → global 구조 학습에는 한계

Likelihood를 직접 최적화하는 것보다 정확도 낮을 수 있음

🔍 5. Loopy Belief Propagation (LBP)
✅ 목적:
그래프 모델 (특히 MRF, CRF)에서 근사적인 marginal inference

⚙️ 아이디어:
메시지 전달 알고리즘을 사이클이 있는 그래프에도 적용

반복적 메시지 전달을 통해 근사분포 계산

⚠️ 한계:
수렴이 보장되지 않음

근사 정확도 떨어질 수 있음

🔍 6. Mean Field Theory
✅ 목적:
복잡한 분포를 독립된 단일 변수의 곱으로 근사

𝑝
(
𝑥
)
≈
∏
𝑖
𝑞
𝑖
(
𝑥
𝑖
)
p(x)≈ 
i
∏
​
 q 
i
​
 (x 
i
​
 )
⚠️ 한계:
변수 간 의존성 무시 → 복잡한 구조 표현 불가능

간단한 구조에서는 작동하지만 일반화에 약함

🔍 7. Minimum Probability Flow (MPF)
✅ 목적:
Energy-based model에서 정규화 상수 없이도 학습 가능하게 함

⚙️ 아이디어:
데이터 분포 
𝑝
𝑑
𝑎
𝑡
𝑎
p 
data
​
 에서 가까운 이웃 상태로의 확률 흐름을 줄이도록 학습

즉, 데이터와 비데이터 사이의 flow를 줄이기

min
⁡
𝜃
∑
𝑥
∈
data
∑
𝑥
′
𝑇
(
𝑥
′
∣
𝑥
)
log
⁡
𝑝
𝜃
(
𝑥
′
)
𝑝
𝜃
(
𝑥
)
θ
min
​
  
x∈data
∑
​
  
x 
′
 
∑
​
 T(x 
′
 ∣x)log 
p 
θ
​
 (x)
p 
θ
​
 (x 
′
 )
​
 
⚠️ 한계:
이웃 상태 정의에 따라 결과가 민감

복잡한 분포에는 한계
