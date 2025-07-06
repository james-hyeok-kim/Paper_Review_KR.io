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

 1.1 문제 설정: Tractability vs Flexibility

배경 문제: 머신러닝에서는 복잡한 데이터 분포를 잘 모델링하는 것이 목표입니다.
그러나 대부분의 확률 모델은 다음 두 가지 중 하나를 만족하기는 쉽지만, 동시에 만족하기는 어렵습니다:

Tractable: 수학적으로 계산이 가능해야 함 (예: 정규분포처럼 확률 계산이 쉬운 모델)

Flexible: 복잡하고 다양한 분포를 표현할 수 있어야 함
(예: 어떤 함수 $φ(x)$든지 가능한 분포 $𝑝(𝑥) = \frac{𝜙(𝑥)}{𝑍}$)

문제점: 유연한 모델은 정규화 상수 
$Z=∫ϕ(x)dx$를 계산하기가 거의 불가능합니다.

샘플링이나 학습도 Monte Carlo 방식으로 매우 비쌈.

1.1.1 기존 해결 시도들 → 이런 방법들은 트레이드오프를 완전히 해결하지는 못합니다.
* Variational Bayes
* Contrastive Divergence
* Score Matching
* Pseudolikelihood
* Loopy Belief Propagation
* Mean Field Theory
* Minimum Probability Flow


---

### Appendix
#### 1. Monte Carlo
  - 정확한 계산이 어렵거나 불가능할 때, 많은 수의 랜덤 샘플을 뽑아서 평균을 내면 근사값이 된다는 아이디어입니다.

1-1. 예시
* 정적분 근사 어떤 함수 $f(x)$의 정적분을 계산하고 싶을 때, $I = \int\limits_a^b f(x)dx$ 이걸 직접 계산하기 어렵다면,
* 구간 [𝑎, 𝑏]에서 무작위로 N개의 샘플 $X_1, ... X_n$을 뽑아서
  
$I \approx \frac{𝑏−𝑎}{N} \displaystyle\sum_{i=1}^{N} f(x_i) $
처럼 샘플 평균 근사 가능


1-2. 단점
* 계산량이 많아요. (샘플 수가 커야 정확해짐)
* 분산이 클 수 있음 → 샘플 수가 부족하면 근사값이 매우 부정확
* 좋은 샘플링 분포를 잘 선택해야 함 (안 그러면 "희소 영역"은 놓침)

