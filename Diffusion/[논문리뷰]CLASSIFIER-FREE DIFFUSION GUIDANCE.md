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

* $\lambda$ 식 유도
$$\lambda = \log(\text{SNR}) = \log\left(\frac{\text{신호 분산}}{\text{노이즈 분산}}\right) = \log\left(\frac{\alpha_\lambda^2}{\sigma_\lambda^2}\right)$$

1. $\lambda$의 정의에서 로그를 지수로 변환

$$e^\lambda = \frac{\alpha_\lambda^2}{\sigma_\lambda^2}$$

2. Variance Preserving (VP) 조건($\sigma_\lambda^2 = 1 - \alpha_\lambda^2$)을 대입

$$e^\lambda = \frac{\alpha_\lambda^2}{1 - \alpha_\lambda^2}$$

3. $\alpha_\lambda^2$에 대해 식을 정리

$$e^\lambda (1 - \alpha_\lambda^2) = \alpha_\lambda^2$$

$$e^\lambda - e^\lambda \alpha_\lambda^2 = \alpha_\lambda^2$$

$$e^\lambda = \alpha_\lambda^2 (1 + e^\lambda)$$

$$\alpha_\lambda^2 = \frac{e^\lambda}{1 + e^\lambda}$$

4. 분모와 분자를 $e^\lambda$로 나누면 논문의 최종 식

$$\alpha_\lambda^2 = \frac{1}{e^{-\lambda} + 1}$$

#### Backward Process
##### 이상적인 역방향 전이 (정답을 알고 있을때)

$$
q(z_{\lambda_0} | z_\lambda, x) = \mathcal{N}(\tilde{\mu}_{\lambda_0|\lambda}(z_\lambda, x), \tilde{\sigma}^2_{\lambda_0|\lambda} I) \quad \text{where} 
$$


$$
\tilde{\boldsymbol{\mu}}_{\lambda'|\lambda}(\mathbf{z}_\lambda, \mathbf{x}) = e^{\lambda-\lambda'} \frac{\alpha_{\lambda'}}{\alpha_\lambda} \mathbf{z}_\lambda + (1 - e^{\lambda-\lambda'}) \alpha_{\lambda'} \mathbf{x} \quad (3)
$$

$$
 \quad \tilde{\sigma}^2_{\lambda'|\lambda} = (1 - e^{\lambda-\lambda'}) \sigma^2_{\lambda'} \quad(3)
$$

* 원본 데이터 $x$를 알고있다면, 노이즈가 더 많은 상태($z_\lambda$)에서 노이즈가 덜한 상태($z_{\lambda'}$)로 가는 확률 분포를 정확히 계산 가능
* 의미: 이것은 모델이 학습해야 할 '정답지(Ground Truth)' 역할을 하는 사후 확률(Posterior) 분포

##### 실제 모델의 역방향 전이 (정답을 모를 때)

$$
p_\theta(\mathbf{z}_{\lambda'} | \mathbf{z}_\lambda) = \mathcal{N} \left( \tilde{\boldsymbol{\mu}}_{\lambda'|\lambda}(\mathbf{z}_\lambda, \mathbf{x}_\theta(\mathbf{z}_\lambda)), \left( \tilde{\sigma}^2_{\lambda'|\lambda} \right)^{1-v} \left( \sigma^2_{\lambda|\lambda'} \right)^v  \right) \quad (4)
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
* 언제 얼마나 노이즈를 섞을지(or 뺄지)를 결정하는 시간적 지표
* 학습 및 샘플링 시 $\lambda$(로그 신호 대 잡음비)를 어떻게 선택
  * 완전한 노이즈에서 어떤 속도로 노이즈를 걷어낼지 결정
* $u$를 0과 1 사이에서 균등하게 뽑은 뒤, $\lambda = -2 \log \tan(au + b)$ 공식을 통해 변환하여 사용
  * 왜 저 복잡한 탄젠트($\tan$) 공식을 쓰는가? 학습에 **가장 중요한 구간(Signal과 Noise가 적절히 섞인 구간)** 을 더 집중적으로 학습하기 위해서입니다.

$$
\lambda = -2 \log \tan(a u + b)
$$

$$
\text{where} \quad b = \arctan(e^{-\lambda_{\max}/2}), \quad a = \arctan(e^{-\lambda_{\min}/2}) - b
$$

#### 스코어 매칭
* 데이터가 있는 진짜 방향을 알려주는 나침반 학습법

$$
\epsilon_\theta(z_\lambda) \approx -\sigma_\lambda \nabla_{z_\lambda} \log p(z_\lambda)
$$

* 모델 예측 노이즈와 $\epsilon_\theta$와 실제 노이즈 $\epsilon$의 차이를 줄이는 것이 스코어 매칭
* 조건부 모델링: 클래스 조건(예: '강아지' 사진 생성)을 줄 때는, 모델 입력에 클래스 정보 $c$만 추가하면 됩니다 ( $\epsilon_\theta(z_\lambda, c)$ ).
* 이를 따라가는 과정은 랑주뱅 역학(Langevin diffusion)과 유사
  * 나침반(방향) + 발걸음 + 비틀거림(노이즈)
* 수학적으로 랑주뱅 역학은 다음 두 항의 합으로 정의됩니다
 
$$x_{new} = x_{old} + \underbrace{\epsilon \nabla \log p(x)}_{\text{기울기 방향 이동}} + \underbrace{\sqrt{2\epsilon} z}_{\text{무작위 노이즈 주입}}$$



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

<p align='center'>
<img width="813" height="293" alt="image" src="https://github.com/user-attachments/assets/a356691e-eff5-46ed-960e-b3826073fc31" />
</p>

* 가이던스 강도($w$)를 높일수록 데이터 분포가 넓게 퍼져 있는 형태에서 특정 지점으로 좁게 모이는(집중되는) 현상. 이는 다양성이 줄어들고 품질(확실성)이 높아짐을 시각적으로 보여줍


$$
\begin{align}
&\epsilon_\theta(\mathbf{z}_\lambda) - (w+1)\sigma_\lambda \nabla_{\mathbf{z}_\lambda} \log p_\theta(c | \mathbf{z}_\lambda) \approx - \sigma_\lambda \nabla_{\mathbf{z}_\lambda} \left[ \log p(\mathbf{z}_\lambda) + (w+1) \log p_\theta(c | \mathbf{z}_\lambda) \right] \\
&\qquad = - \sigma_\lambda \nabla_{\mathbf{z}_\lambda} \left[ \log p(\mathbf{z}_\lambda)p_\theta(c | \mathbf{z}_\lambda) + w \log p_\theta(c | \mathbf{z}_\lambda) \right] \\
&\qquad = - \sigma_\lambda \nabla_{\mathbf{z}_\lambda} \left[ \log p(\mathbf{z}_\lambda | c) + w \log p_\theta(c | \mathbf{z}_\lambda) \right]
\end{align}
$$

* Unconditional model에 가중치 w+1로 classifier guidance를 적용하면 이론적으로 가중치 w 로 conditional model에 적용하는 것과 동일한 결과가 나타난다.

##### Equation 유도

$$\epsilon_\theta(z_\lambda, c) \approx -\sigma_\lambda \nabla_{z_\lambda} \log p(z_\lambda|c)$$

모델이 예측한 값($\epsilon$)은 "데이터 분포의 기울기($\nabla \log p$)에 $-\sigma_\lambda$를 곱한 것"과 같다

원래 식 (좌변)

$$\epsilon_\theta(z_\lambda, c) - w\sigma_\lambda \nabla_{z_\lambda} \log p_\theta(c|z_\lambda)$$

단계 1: $\epsilon_\theta$ 자리에 정의 대입

$$\underbrace{-\sigma_\lambda \nabla_{z_\lambda} \log p(z_\lambda|c)}_{\epsilon_\theta \text{의 정의}} - w\sigma_\lambda \nabla_{z_\lambda} \log p_\theta(c|z_\lambda)$$

단계 2: 공통 인수 $-\sigma_\lambda$로 묶기두 항 모두 $-\sigma_\lambda$를 가지고 있으므로 밖으로 묶어냅니다. (주의: 두 번째 항은 마이너스가 밖으로 나갔으므로 안에는 플러스가 남습니다.)

$$-\sigma_\lambda \left[ \nabla_{z_\lambda} \log p(z_\lambda|c) + w \nabla_{z_\lambda} \log p_\theta(c|z_\lambda) \right]$$

단계 3: 미분 연산자( $\nabla$ ) 묶기 (Linearity)기울기(gradient, $\nabla$ )는 덧셈에 대해 분배 법칙이 성립합니다 ( $\nabla A + \nabla B = \nabla(A+B)$ ). 또한 상수( $w$ )는 미분 기호 안으로 들어갈 수 있습니다 ( $w \nabla B = \nabla (wB)$ ).따라서 두 로그 확률을 하나의 미분 기호 안으로 합칠 수 있습니다.

$$-\sigma_\lambda \nabla_{z_\lambda} \left[ \log p(z_\lambda|c) + w \log p_\theta(c|z_\lambda) \right]$$

이것이 질문하신 식의 우변입니다.



#### 3.2 Classifier-Free Guidance

* $\epsilon_\theta(\mathbf{z}_\lambda, c)$ 를 수정하여 Classifier 없이 Classifier Guidance와 같은 효과를 얻고자 함
* 3.1 Classifier 에서 $\nabla_{z_{\lambda}}\log~p_{\theta}(c|z_{\lambda})$ 을 대체하고 싶음


##### Sampling, 점수 결합

$$\tilde{\epsilon}_{\theta}(z_{\lambda},c)=(1+w)\epsilon_{\theta}(z_{\lambda},c)-w\epsilon_{\theta}(z_{\lambda}) \quad \text{[cite: 120]} \quad (6)$$

* $\epsilon_{\theta}(z_{\lambda},c)$: 클래스 $c$를 조건으로 한 조건부 점수 (클래스 $c$ 쪽으로 이동)
* $\epsilon_{\theta}(z_{\lambda})$: 조건이 없는 비조건부 점수 (모든 데이터 쪽으로 이동)
* $w$: 가이던스 강도를 조절하는 매개변수입니다. $w$가 클수록 가이던스가 강해집니다
* 효과 해석: 이 수식은 "조건부 확률( $\epsilon_{\theta}(z_{\lambda},c)$ )의 방향을 비조건부 확률( $\epsilon_{\theta}(z_{\lambda})$ )의 방향보다 $w$만큼 더 강조"하라는 의미로, 클래스 $c$ 에 해당하는 특징을 증폭시키는 효과를 가져옵니다.

##### 의의
* $\nabla_{z_{\lambda}}\log~p_{\theta}(c|z_{\lambda})$ 별도의 Classifier 없음으로 $\tilde{\epsilon}_{\theta}$ 으로 가는 것이 Gradient-based Adversarial Attack이라고 볼수 없다

---
##### Algorithm 1

<p align='center'>
<img width="808" height="257" alt="image" src="https://github.com/user-attachments/assets/f1f4c17c-d656-4cf7-99a9-1eedd5bef339" />
</p>

* 별도의 분류기를 훈련하는 대신 단일 모델 사용, 조건부 확산 모델( $\epsilon_{\theta}(z_{\lambda},c)$ )과 비조건부 확산 모델( $\epsilon_{\theta}(z_{\lambda})$ )을 단일 네트워크로 통합

0. 입력 (Input) $p_{uncond}$ : '비조건부 학습 확률'입니다. (예: 0.1).

2. 데이터 샘플링 (Repeat Loop)
* $(x, c) \sim p(x, c)$
* 데이터셋에서 이미지( $x$ )와 그에 해당하는 레이블( $c$, 예: "강아지")을 꺼냅니다

3. 조건 드롭아웃 (Condition Dropout) - 가장 중요한 부분 $c \leftarrow \emptyset$ with probability $p_{uncond}$

* 주사위를 던져서 $p_{uncond}$ 확률에 당첨되면, 레이블 $c$ 를 버리고 빈 값(Null token, $\emptyset$ )으로 바꿔치기합니다
* 경우 A (90%): 모델에게 "이건 강아지야"라고 알려주고 학습시킵니다. $\rightarrow$ 조건부 모델($\epsilon(z, c)$) 학습
* 경우 B (10%): 모델에게 "이게 뭔지 안 알려줄 거야(Null)"라고 하고 학습시킵니다. $\rightarrow$ 비조건부 모델($\epsilon(z)$)학습

* 이 과정을 통해 모델은 텍스트가 있을 때 그리는 법과, 텍스트가 없을 때(Null) 그리는 법을 동시에 배웁니다

노이즈 추가 및 손실 계산 (Diffusion Process)

4. $\lambda \sim p(\lambda)$

5. $\epsilon \sim \mathcal{N}(0, I)$

* $노이즈 강도( $\lambda$ )와 실제 노이즈( $\epsilon$ )를 무작위로 뽑습니다

6. $z_\lambda = \alpha_\lambda x + \sigma_\lambda \epsilon$

* 이미지 $x$에 노이즈를 섞어 $z_\lambda$를 만듭니다

7. Gradient Step on $\nabla_{\theta}||\epsilon_{\theta}(z_{\lambda},c)-\epsilon||^{2}$

* 모델에게 $z_\lambda$ 와 조건 $c$ (혹은 $\emptyset$ )를 주고 "어떤 노이즈가 섞였게?"라고 물어봅니다
* 모델이 예측한 노이즈와 실제 노이즈($\epsilon$)의 차이(오차)를 줄이도록 학습합니다.
 
##### Algorithm 2
<p align='center'>
<img width="909" height="375" alt="image" src="https://github.com/user-attachments/assets/9577ea5f-4002-4b3f-802e-0be69e4dbe99" />
</p>

* 이 알고리즘의 가장 중요한 특징은 매 스텝마다 모델을 두 번 실행해야 한다는 점
  * 기존 방식(Classifier Guidance): 생성 모델 1번 + 분류기 모델 1번 실행
  * 이 방식(Classifier-Free Guidance): 조건부 생성($\epsilon(z, c)$) 1번 + 비조건부 생성($\epsilon(z)$) 1번 실행.

1. 1단계: 노이즈 생성 (Initialization) $z_1 \sim \mathcal{N}(0, I)$ 완전한 무작위 노이즈 이미지 $z_1$을 하나 만듭니다.

2. 2단계: 반복적인 노이즈 제거 (Loop)총 $T$ 번의 단계(Timesteps)를 거치며 조금씩 이미지를 선명하게 만듭니다. 각 단계( $t$ )마다 다음 과정을 수행합니다.

3. 가이던스 적용 (Core Step)앞서 배운 수식(Eq. 6)을 여기서 적용합니다.

$$\tilde{\epsilon}_t = (1+w)\epsilon_\theta(z_t, c) - w\epsilon_\theta(z_t)$$

이 계산을 통해 "일반적인 이미지 방향보다는, 고양이 특징이 더 강하게 나타나는 방향"의 수정된 노이즈 $\tilde{\epsilon}_t$ 를 구합니다.

두 가지 노이즈 예측 (Two Forward Passes)현재 상태의 이미지($z_t$)를 모델에 두 번 넣습니다

  3-1. 조건부 예측: $\epsilon_\theta(z_t, c)$ $\rightarrow$ "고양이를 그리려면 어떤 노이즈를 빼야 하니?"
  
  3-2. 비조건부 예측: $\epsilon_\theta(z_t)$ $\rightarrow$ "그냥 일반적인 이미지를 그리려면 어떤 노이즈를 빼야 하니?"
  
(참고: 논문에서는 비조건부 모델을 따로 만들지 않고, 조건 $c$ 를 비우는 방식으로 하나의 모델을 공유합니다.)

 
4. 이미지 업데이트 (Sampling Step)계산된 $\tilde{\epsilon}_t$를 사용하여 이미지를 업데이트합니다.

$$\tilde{x}_t = (z_t - \sigma_{\lambda_t}\tilde{\epsilon}_t) / \alpha_{\lambda_t}$$

현재 이미지( $z_t$ )에서 수정된 노이즈( $\tilde{\epsilon}_t$ )를 빼서, 더 깨끗한 이미지( $z_{t+1}$ )로 나아갑니다

5. 정규분포(Gaussian Distribution)에서 값을 하나 뽑는다

$$z_{t+1} \sim \mathcal{N}(\underbrace{\tilde{\mu}_{\lambda_{t+1}|\lambda_t}(z_t, \tilde{x}_t)}_{\text{평균}}, \underbrace{(\tilde{\sigma}^2_{\lambda_{t+1}|\lambda_t})^{1-v}(\sigma^2_{\lambda_t|\lambda_{t+1}})^v}_{\text{분산}})$$

* 조건문의 의미 (if t < T else ...): 샘플링(랜덤 추출)


---

### 4. Experiment

<p align = 'center'>
<img width="697" height="492" alt="image" src="https://github.com/user-attachments/assets/f406c594-9480-4ff6-b956-5cae20c8b37f" />
</p>


1. 가이던스 강도 ($w$)에 따른 변화
  1-1. 낮은 $w$ ($0.1 \sim 0.3$): 최고의 FID 점수를 기록하여 가장 자연스러운 이미지를 생성
  1-2. 높은 $w$ ($4.0$ 이상): 최고의 IS 점수
* 가이던스 강도가 높아질수록 샘플의 다양성은 줄어들지만, 개별 샘플의 선명도가 높아지고 색상이 진해지는(saturated) 경향

2. 비조건부 학습 확률 ($p_{uncond}$)의 영향
  2-1. 결과: $p_{uncond} = 0.1$ 또는 $0.2$일 때가 $0.5$일 때보다 전반적으로 더 좋은 성능
  2-2. 아주 적은 부분만 비조건부 생성 능력에 할애해도 효과적인 가이던스가 가능하다는 것

<p align = 'center'>
<img width="808" height="563" alt="image" src="https://github.com/user-attachments/assets/d44ca235-6185-444c-9a81-d04f8959b2b3" />
</p>

* vs. Classifier Guidance (ADM): $w=0.3$일 때, $128 \times 128$ ImageNet에서 분류기를 사용하는 ADM 모델보다 더 좋은(낮은) FID 점수를 기록
* vs. BigGAN-deep: $w=4.0$일 때, BigGAN-deep의 최고 성능 설정보다 더 나은 FID 및 IS 점수를 동시에 달성

---
### 5. Discussion

1. 실용적 장점: 극도의 단순성 (Simplicity)
* 학습이 쉽다

2. 이론적 의의: 순수 생성 모델의 가능성
* 비(非) 적대적 방식

3. 한계점 및 고려사항
* 샘플링 속도 (Sampling Speed)
  * Classifier-Free Guidance는 매 스텝마다 조건부 및 비조건부 스코어를 계산하기 위해 **두 번의 포워드 패스(forward pass)**를 실행
* 다양성 감소 (Decreased Diversity)
  * 샘플의 품질(Fidelity)을 높이기 위해 다양성(Diversity)을 희생하는 방식

---
