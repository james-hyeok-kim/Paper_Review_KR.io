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

### 1. Introduction

#### 기존 문제점

* 필요성: 생성 모델(예: BigGAN, Glow)에서는 샘플의 다양성을 조금 희생하더라도 개별 샘플의 품질을 높이기 위해 '(low temperature sampling)'이나 '(truncation)' 기법을 사용
* Diffusion Model 한계: 점수 벡터를 스케일링하거나 노이즈를 줄이는 단순한 방식으로는 이러한 효과를 얻을 수 없음

#### Classifier Guidance 해결책

* 단점 및 의문점
  * 복잡성 : 훈련 복잡성 (Classifier)
  * Adversarial Attack 의혹 : 이미 분류기를 속이려는 적대적 공격과 유사
    * IS / FID가 이미지가 좋아저서 좋아진건지 단순 분류기 지표를 속여서 좋아진 것인지 에 대한 의문

---

### 2. Background
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


---

### 3. GUIDANCE

* GAN이나 Flow 모델 같은 경우, '트렁케이션(truncation)'이나 '저온 샘플링(low temperature sampling)'을 통해 입력 노이즈의 분산을 줄임으로써 이미지 품질을 높임
* 확산 모델의 문제: 하지만 확산 모델에서는 단순히 노이즈의 분산을 줄이거나 점수(Score)를 스케일링하면, 이미지가 선명해지는 것이 아니라 흐릿하고 품질이 낮은 결과
* 가이던스(Guidance)라는 특별한 기법이 필요

#### 3.1 Classifier Guidance (기존 방식)

$$
\hat{\epsilon}_\theta(\mathbf{z}_\lambda, c) = \epsilon_\theta(\mathbf{z}_\lambda, c) - w \sigma_\lambda \nabla_{\mathbf{z}_\lambda} \log p_\theta(c|\mathbf{z}_\lambda) \approx -\sigma_\lambda \nabla_{\mathbf{z}_\lambda} [\log p(\mathbf{z}_\lambda|c) + w \log p_\theta(c|\mathbf{z}_\lambda)]
$$

* $\epsilon_{\theta}(z_{\lambda},c)$: 확산 모델이 예측한 조건부 스코어 (즉, 데이터가 있어야 할 방향)입니다
* $\nabla_{z_{\lambda}}\log~p_{\theta}(c|z_{\lambda})$: 보조 분류기가 예측한 클래스 $c$에 대한 로그 우도의 그래디언트입니다. 이 항은 모델이 "분류기가 선호하는 방향"으로 움직이도록 만듭니다
* $w$는 가이던스 강도(strength)를 조절하는 파라미터


$$
\tilde{p}_\theta(\mathbf{z}_\lambda|\mathbf{c}) \propto p_\theta(\mathbf{z}_\lambda|\mathbf{c}) \cdot p_\theta(\mathbf{c}|\mathbf{z}_\lambda)^w
$$

$$
\epsilon_\theta(\mathbf{z}_\lambda, c) \rightarrow \tilde{\epsilon}_\theta(\mathbf{z}_\lambda, c)
$$

* 샘플링 시 $\tilde{\epsilon}$ 사용하면 위에서 언급된 방식의 샘플을 생성하는 것과 동일합니다.

<img width="813" height="293" alt="image" src="https://github.com/user-attachments/assets/a356691e-eff5-46ed-960e-b3826073fc31" />

* 가이던스 강도($w$)를 높일수록 데이터 분포가 넓게 퍼져 있는 형태에서 특정 지점으로 좁게 모이는(집중되는) 현상. 이는 다양성이 줄어들고 품질(확실성)이 높아짐을 시각적으로 보여줍


$$
\begin{align}
&\epsilon_\theta(\mathbf{z}_\lambda) - (w+1)\sigma_\lambda \nabla_{\mathbf{z}_\lambda} \log p_\theta(c | \mathbf{z}_\lambda) \approx - \sigma_\lambda \nabla_{\mathbf{z}_\lambda} \left[ \log p(\mathbf{z}_\lambda) + (w+1) \log p_\theta(c | \mathbf{z}_\lambda) \right] \\
&\qquad = - \sigma_\lambda \nabla_{\mathbf{z}_\lambda} \left[ \log p(\mathbf{z}_\lambda)p_\theta(c | \mathbf{z}_\lambda) + w \log p_\theta(c | \mathbf{z}_\lambda) \right] \\
&\qquad = - \sigma_\lambda \nabla_{\mathbf{z}_\lambda} \left[ \log p(\mathbf{z}_\lambda | c) + w \log p_\theta(c | \mathbf{z}_\lambda) \right]
\end{align}
$$

* Unconditional model에 가중치 w+1로 classifier guidance를 적용하면 이론적으로 가중치 w 로 conditional model에 적용하는 것과 동일한 결과가 나타난다.



#### 3.2 Classifier-Free Guidance

* $\epsilon_\theta(\mathbf{z}_\lambda, c)$ 를 수정하여 Classifier 없이 Classifier Guidance와 같은 효과를 얻고자 함

##### Algorithm 1

<img width="808" height="257" alt="image" src="https://github.com/user-attachments/assets/f1f4c17c-d656-4cf7-99a9-1eedd5bef339" />

* 별도의 분류기를 훈련하는 대신 단일 모델 사용, 조건부 확산 모델( $\epsilon_{\theta}(z_{\lambda},c)$ )과 비조건부 확산 모델( $\epsilon_{\theta}(z_{\lambda})$)을 단일 네트워크로 통합
* 비조건부 훈련: 훈련 중 일정 확률 $p_{uncond}$로 조건 정보 $c$(클래스 라벨)를 널 토큰(null token, $\emptyset$)으로 설정하여 제거
  * $c=\emptyset$일 때 모델은 비조건부 점수 추정치( $\epsilon_{\theta}(z_{\lambda})$ )를 학습
  * 장점: 이 방식은 훈련 파이프라인을 복잡하게 만들지 않고, 총 모델 파라미터 수를 늘리지 않아 매우 간단합니다

# Here!!

##### Sampling, 점수 결합

$$\tilde{\epsilon}_{\theta}(z_{\lambda},c)=(1+w)\epsilon_{\theta}(z_{\lambda},c)-w\epsilon_{\theta}(z_{\lambda}) \quad \text{[cite: 120]} \quad (6)$$

* $\epsilon_{\theta}(z_{\lambda},c)$: 클래스 $c$를 조건으로 한 조건부 점수 (클래스 $c$ 쪽으로 이동)
* $\epsilon_{\theta}(z_{\lambda})$: 조건이 없는 비조건부 점수 (모든 데이터 쪽으로 이동).$w$: 가이던스 강도를 조절하는 매개변수입니다. $w$가 클수록 가이던스가 강해집니다
* 효과 해석: 이 수식은 **"조건부 확률($\epsilon_{\theta}(z_{\lambda},c)$)의 방향을 비조건부 확률($\epsilon_{\theta}(z_{\lambda})$)의 방향보다 $w$만큼 더 강조"**하라는 의미로, 클래스 $c$에 해당하는 특징을 증폭시키는 효과를 가져옵니다.
