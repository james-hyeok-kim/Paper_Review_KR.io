# CLASSIFIER-FREE DIFFUSION GUIDANCE
저자 : Jonathan Ho & Tim Salimans Google Research, Brain team

출간 : NeurIPS, 2022.

논문 : [PDF](https://arxiv.org/pdf/2207.12598)

---

### Abstract

* Classifier 없이 Diffusion Model 품질 향상

#### 배경

* Classifier Guidance는 Classifier를 따로 훈련 시켜야 한다는 단점
* Classifier 없이 Guidance 수행할 수 있는가?
* Conditional & Unconditional Diffusion Model을 Jointly Train

---

### Introduction

#### 기존 문제점

* 필요성: 생성 모델(예: BigGAN, Glow)에서는 샘플의 다양성을 조금 희생하더라도 개별 샘플의 품질을 높이기 위해 '(low temperature sampling)'이나 '(truncation)' 기법을 사용
* Diffusion Model 한계: 점수 벡터를 스케일링하거나 노이즈를 줄이는 단순한 방식으로는 이러한 효과를 얻을 수 없음

#### Classifier Guidance 해결책

* 단점 및 의문점
  * 복잡성 : 훈련 복잡성 (Classifier)
  * Adversarial Attack 의혹 : 이미 분류기를 속이려는 적대적 공격과 유사
    * IS / FID가 이미지가 좋아저서 좋아진건지 단순 분류기 지표를 속여서 좋아진 것인지 에 대한 의문

---

### Background
#### Forward Process Noise

$$
x \sim p(x) \quad \text{and} \quad z = \{z_\lambda \mid \lambda \in [\lambda_{\min}, \lambda_{\max}]\}
$$

$$
q(z_\lambda | x) = \mathcal{N}(\alpha_\lambda x, \sigma^2_\lambda I), \quad \text{where } \alpha^2_\lambda = \frac{1}{1 + e^{-\lambda}}, \quad \sigma^2_\lambda = 1 - \alpha^2_\lambda \quad (1)
$$

$$
q(z_\lambda | z_{\lambda'}) = \mathcal{N}\left(\frac{\alpha_\lambda}{\alpha_{\lambda'}} z_{\lambda'}, \sigma^2_{\lambda|\lambda'} I\right), \quad \text{where } \lambda < \lambda', \quad \sigma^2_{\lambda|\lambda'} = (1 - e^{\lambda - \lambda'}) \sigma^2_\lambda \quad (2)
$$

$$q(z_\lambda|x)=\mathcal{N}(\alpha_\lambda x,\sigma_\lambda^2 I)$$

* $z_\lambda$는 원본 $x$를 $\alpha_\lambda$만큼 축소하고, 분산 $\sigma_\lambda^2$만큼의 가우시안 노이즈를 더한 형태
* $\alpha_\lambda^2 = \frac{1}{1+e^{-\lambda}}$: 신호(Signal)의 비율
* $\sigma_\lambda^2 = 1 - \alpha_\lambda^2$: 잡음(Noise)의 비율
* 이 둘의 제곱 합은 1 ($\alpha^2 + \sigma^2 = 1$)이 되도록 설계되어 있어, 노이즈가 추가되어도 전체 데이터의 분산(Variance)은 일정하게 유지됩니다 (Variance Preserving)
* $\lambda$는 로그 신호 대 잡음비(log signal-to-noise ratio)로 해석됩니다. $\lambda$가 클수록 신호가 강하고(깨끗한 이미지), 작을수록 잡음이 강함
$$
\lambda = \log \left( \frac{\alpha^2_\lambda}{\sigma^2_\lambda} \right)
$$

$$q(z_\lambda|z_{\lambda'})=\mathcal{N}\left(\frac{\alpha_\lambda}{\alpha_{\lambda'}}z_{\lambda'},\sigma_{\lambda|\lambda'}^2 I\right)$$

* 조건: $\lambda < \lambda'$ (즉, $\lambda'$가 더 깨끗한 상태이고 $\lambda$가 더 노이즈가 많은 상태입니다)
* 의미: 이 수식은 마르코프 체인(Markov Chain)의 성질
* 전 단계($z_{\lambda'}$)의 상태를 알면, 다음 단계($z_\lambda$)의 분포를 계산할 수 있음을 보여줌
* 새로운 분산 $\sigma_{\lambda|\lambda'}^2$은 두 시점 사이의 노이즈 차이를 나타냄


#### 이상적인 역방향 전이 (정답을 알고 있을때)

$$
q(z_{\lambda_0} | z_\lambda, x) = \mathcal{N}(\tilde{\mu}_{\lambda_0|\lambda}(z_\lambda, x), \tilde{\sigma}^2_{\lambda_0|\lambda} I) \quad \text{where} 
$$


$$
\tilde{\boldsymbol{\mu}}_{\lambda'|\lambda}(\mathbf{z}_\lambda, \mathbf{x}) = e^{\lambda-\lambda'} \frac{\alpha_{\lambda'}}{\alpha_\lambda} \mathbf{z}_\lambda + (1 - e^{\lambda-\lambda'}) \alpha_{\lambda'} \mathbf{x} \quad (3)
$$

$$
 \quad \tilde{\sigma}^2_{\lambda'|\lambda} = (1 - e^{\lambda-\lambda'}) \sigma^2_{\lambda'} \quad(3)
$$

* 원본 데이터 $x$를 알고있다면, 이즈가 더 많은 상태($z_\lambda$)에서 노이즈가 덜한 상태($z_{\lambda'}$)로 가는 확률 분포를 정확히 계산 가능
* 의미: 이것은 모델이 학습해야 할 '정답지(Ground Truth)' 역할을 하는 사후 확률(Posterior) 분포

#### 실제 모델의 역방향 전이 (정답을 모를 때)

$$
p_\theta(\mathbf{z}_{\lambda_0} | \mathbf{z}_\lambda) = \mathcal{N} \left( \tilde{\boldsymbol{\mu}}_{\lambda_0|\lambda}(\mathbf{z}_\lambda, \mathbf{x}_\theta(\mathbf{z}_\lambda)), \left( \tilde{\sigma}^2_{\lambda_0|\lambda} \right)^{1-v} \left( \sigma^2_{\lambda|\lambda_0} \right)^v \mathbf{I} \right) \quad (4)
$$

* 원본 데이터 $x$를 모르고 노이즈($z_\lambda$)만 주어집니다. 따라서 모델은 $x$ 대신 **모델이 추정한 값 $x_\theta(z_\lambda)$**를 사용하여 위의 분포를 흉내
* 식 (3)의 $\tilde{\mu}$ 공식에 $x$ 대신 모델의 예측값 $x_\theta(z_\lambda)$를 대입하여 사용
* 분산은 두 가지 값($\tilde{\sigma}^2$와 $\sigma^2$) 사이를 로그 공간에서 보간(interpolation)하여 결정
* 보간 비율은 하이퍼파라미터 $v$로 조절
* 저자들은 $v$를 학습시키기보다 상수(constant)로 고정하는 것이 효과적임을 발견

#### 파라미터화 + 학습목표
* 모델 $x_\theta$는 원본 이미지를 직접 예측하는 대신, 현재 이미지에 섞인 노이즈($\epsilon$)를 예측하도록 설계
* 예측된 노이즈를 이용해 원본 이미지 $x$를 역산합니다: $x_\theta(z_\lambda) = (z_\lambda - \sigma_\lambda\epsilon_\theta(z_\lambda))/\alpha_\lambda$.

$$
\mathbb{E}_{\epsilon, \lambda} \left[ \| \epsilon_\theta(\mathbf{z}_\lambda) - \epsilon \|_2^2 \right] \quad(5)
$$

* Denoising Score Matching
* $\lambda$의 분포 $p(\lambda)$가 Uniform하다면, 이는 변분 하한(Variational Lower Bound, VLB)을 최적화하는 것과 같습니다. 균일하지 않다면 가중치가 적용된 VLB로 해석


#### Noise Schedule
* 학습 및 샘플링 시 $\lambda$(로그 신호 대 잡음비)를 어떻게 선택
* $u$를 0과 1 사이에서 균등하게 뽑은 뒤, $\lambda = -2 \log \tan(au + b)$ 공식을 통해 변환하여 사용

$$
\lambda = -2 \log \tan(a u + b)
$$

$$
\text{where} \quad b = \arctan(e^{-\lambda_{\max}/2}), \quad a = \arctan(e^{-\lambda_{\min}/2}) - b
$$

#### 스코어 매칭
$$
\epsilon_\theta(z_\lambda) \approx -\sigma_\lambda \nabla_{z_\lambda} \log p(z_\lambda)
$$

* 학습된 모델 $\epsilon_\theta(z_\lambda)$는 데이터 분포의 로그 밀도 기울기(gradient of log-density), 즉 스코어(Score)를 추정하는 것과 같다
* $\epsilon_\theta(z_\lambda) \approx -\sigma_\lambda \nabla_{z_\lambda} \log p(z_\lambda)$즉, 모델은 "데이터가 밀집해 있는 방향(더 그럴듯한 이미지 쪽)"을 가리키는 나침반 역할
* 이를 따라가는 과정은 랑주뱅 역학(Langevin diffusion)과 유사
* 조건부 모델링: 클래스 조건(예: '강아지' 사진 생성)을 줄 때는, 모델 입력에 클래스 정보 $c$만 추가하면 됩니다 ( $\epsilon_\theta(z_\lambda, c)$ ).


