# Diffusion Models Beat GANs on Image Synthesis
저자 : Prafulla Dhariwal∗ OpenAI, Alex Nichol∗ OpenAI

출간 : NeurIPS, 2021.

논문 : [PDF](https://arxiv.org/abs/2105.05233)

---

### Abstract
* Classifier Guidance라는 새롭고 효율적인 방법 도입

* ImageNet
  * 128x128 : FID 2.97
  * 256x256 : FID 4.59
  * 512x512 : FID 7.72

* Classifier Guidance + upsampling
  * 256x256 : FID 3.94
  * 512x512 : FID 3.85

---
 
### Introduction

#### GAN 한계
* 학습이 불안정
* 다양성 부족

#### Likelihood 모델
* VAE or Diffusion (Likelihood-based) Model 다양성 우수
* 학습 안정적
* 샘플 Quality는 GAN보다 뒤처짐

#### 연구제안
* 모델 아키텍처 개선 - unconditional 이미지 생성
* Classifier Guidance  도입 - 다양성 조절 및 품질 극대화


### Background
#### Forward (Noising)
* 가우시안 노이즈 추가
#### Backward (Denoising)
* 노이즈 제거, 이미지 생성
#### Training
* 손실 함수: MSE, $||\epsilon - \epsilon_{\theta}(x_{t},t)||^2$ 를 최소화

#### Sampling
* 평균 $\mu_{\theta}$
* 노이즈 $\epsilon_{\theta}$
* 분산 $\Sigma_{\theta}$

#### Main Improvement

* 아키텍처 개선
    * Layer 수 보다, Channel를 늘리는 것이 성능에 효과적
    * 어텐션 헤드의 수를 늘리거나 헤드 당 채널 수를 줄이는 것이 FID를 개선
    * 적응형 그룹 정규화 (AdaGN)
* 분류기 가이던스 (Classifier Guidance)
    * 분류기(Classifier) $p_{\phi}(y|x_t)$를 사용
    * 분류기의 그라디언트 $\nabla_{x} \log p_{\phi}(y|x_t)$를 사용

* Learned Variance
  * DDPM에서는 분산이 상수로 고정했지만 이 논문은 분산을 학습 하도록 했음
  * 50단계 미만의 샘플링은 DDIM 방식을 채택

$$\Sigma_\theta(x_t, t) = \exp(v \log \beta_t + (1 - v) \log \tilde{\beta}_t)$$

* $\beta_t, \tilde{\beta}_t$ 사이 interpolation
* $\Sigma_\theta(x_t, t), \epsilon_\theta(x_t, t)$ 둘다 학습

$$L_{\text{simple}} + \lambda L_{\text{vlb}}$$

* DDPM은 $L_{simple}$만 사용하여 노이즈 $(\epsilon_\theta)$ 를 훈련
  * 단순 평균제곱오차(MSE)
* 현재 논문에서는 $L_{vlb}$를 추가
  * VLB : Variational Lower Bound
  * $\mu, \Sigma$ 둘다 고려한 수식
  * $\lambda$ 는 가중치 (0.0001)

<p align ='center'>
<img width="1285" height="435" alt="image" src="https://github.com/user-attachments/assets/93a23319-7e68-4e6f-97a6-5e104d46d6cd" />
</p>

<p align ='center'>
<img width="1315" height="716" alt="image" src="https://github.com/user-attachments/assets/b01cbf09-7430-4895-857f-d1e152f8aa6a" />
</p>

<p align ='center'>
<img width="1295" height="285" alt="image" src="https://github.com/user-attachments/assets/e5bf837e-49d8-4c02-8384-98be62696549" />
</p>


#### 성능 평가 지표
* FID(Frechet Inception Distance) : 이미지 품질과 다양성 판단
* IS(Inception Score) : 특정 클래스 분류 되는지 품질과 다양성 측정
* Precision and Recall : 품질과 다양성 분리하여 측정

---

### Architecture Improvements
* UNet 아키텍처 개선 부분

1. 모델 크기를 일정하게 유지하면서 깊이/너비 증가
2. Attention head의 수 증가
3. Attention을 16×16 뿐만 아니라 32×32와 8×8 에서도 사용
4. BigGAN의 residual block을 upsampling과 downsampling에 사용
5. Residual connection을 $\frac{1}{\sqrt{2}}$로 rescale

<p align="center">
<img width="736" height="274" alt="image" src="https://github.com/user-attachments/assets/7dc5bc62-f001-4eac-87f9-d2976f11f53a" />
</p>

#### 주요 발견 및 결정사항
* 너비 > 깊이
  * 모델을 더 깊게 만들면 FID가 향상되긴 했지만, 훈련 시간이 너무 오래 걸림
  * 반면 모델을 더 넓게 만들면 더 빠른 훈련 시간 내에 비슷한 성능에 도달했습니다.
* 어텐션 설정 64
  * 어텐션 헤드를 늘리거나 헤드당 채널 수를 줄이는 것이 FID 향상에 도움
  * 저자들은 최종적으로 헤드당 64 채널을 사용하는 것이 훈련 속도와 성능 면에서 가장 균형 잡힌 선택이라고 판단
* 효과적인 조합
  * 'Residual connection 리스케일링'을 제외한 대부분의 변경 사항(다중 해상도 어텐션, BigGAN 블록 등)이 성능 향상에 기여
  * 이 효과들은 함께 적용했을 때 누적되어(compounding effect) FID를 크게 낮췄습니다

#### 핵심 개선: AdaGN (Adaptive Group Normalization)
* 이 레이어는 각 residual block에서 시간(timestep) 임베딩과 클래스(class) 임베딩을 주입(inject)하는 역할
* 방식
  * 임베딩($y$)에서 예측된 스케일($y_s$)과 바이어스($y_b$)를 Group Normalization이 적용된 활성화 함수($h$)에 적용
  * GroupNorm에서 Linear Projection $\rightarrow$ Scale, Bias 

$$AdaGN(h, y) = y_s \cdot GroupNorm(h) + y_b$$

* 성능 저자들은 이 AdaGN 방식이 기존의 단순 덧셈 방식(Ho et al. [25])보다 FID를 확실하게 향상시키는 것을 Table 3에서 확인

<p align="center">
<img width="796" height="170" alt="image" src="https://github.com/user-attachments/assets/4274fae2-4575-40e6-9987-fed11f368e6f" />
</p>

 ### Classifier Guidance
 * GAN은 클래스 정보를 적극적으로 활용하여(예: 클래스 조건부 정규화) 품질을 높임
 * 기존 Diffusion Model 은 단순히 클래스 정보를 AdaGN(3.1절) 같은 곳에 주입할 뿐, 샘플링 품질을 높이는 데 적극적으로 사용하지 못함
 * 목표: 이미 훈련된 확산 모델 $p(x)$ 에 별도의 분류기 $p(y|x)$ 를 붙여서, 샘플링 과정을 원하는 클래스 $y$ 방향으로 "안내"하여 품질을 극대화하는 것

#### 작동원리
* Bayes' theorem을 샘플링 과정에 적용
* 우리가 원하는 것은 $p(x_t | x_{t+1}, y)$, 즉 (이전 단계 이미지 $x_{t+1}$과 클래스 $y$가 주어졌을 때 다음 단계 $x_t$가 나올 확률
* 이 확률은 $p(x_t | x_{t+1}) \cdot p(y | x_t)$에 비례
  * $p(x_t | x_{t+1})$: 기존 확산 모델의 "다음 단계" 예측
  * $p(y | x_t)$: 분류기가 "현재 이미지 $x_t$가 클래스 $y$일 확률
  * 핵심: 확산 모델의 다음 단계( $\mu$ )에 분류기의 그래디언트( $\nabla_x \log p_{\phi}(y|x_t)$ ) 를 더해줌

#### 핵심 기술
* 그래디언트 스케일링 (s)단순히 분류기 그래디언트를 사용( $s=1$ )했더니, 품질이 크게 나아지지 않음
  * 해결책: 이 그래디언트의 영향력을 조절하는 스케일(scale) 값 $s$를 도입
    * $s=0$: 안내 없음 (기존 확산 모델)
    * $s > 1$: 그래디언트의 영향력을 $s$배 증폭시켜 클래스 방향으로 더 강하게 반영
  * 결과: 이 스케일 $s$ 값이 바로 품질과 다양성을 조절하는 "손잡이(knob)" 역할
    * $s$ 값을 높이면 ($s \uparrow$): 모델이 분류기의 판단에 더 집중하여 품질(Precision)과 IS가 크게 향상
    * $s$ 값을 낮추면 ($s \downarrow$): 다양성(Recall)이 향상됩니다

### Conditional Reverse Noising Process (2~10)

<p align="center">
<img width="787" height="230" alt="image" src="https://github.com/user-attachments/assets/e8cdcd9c-5b47-4bdb-ad2e-ce7cb9d212ba" />
</p>


#### 조건부 확률 정의
$$p_{\theta,\phi}(x_{t}|x_{t+1},y) = Z p_{\theta}(x_{t}|x_{t+1})p_{\phi}(y|x_{t}) \quad \quad (2)$$

* 비조건부 전이 확률 $p_{\theta}$
* 분류기 확률 $p_{\phi}$

#### 비조건부 모델의 가우시안 정의

$$p_{\theta}(x_{t}|x_{t+1}) = \mathcal{N}(\mu, \Sigma) \quad \quad (3)$$

$$\log p_{\theta}(x_{t}|x_{t+1}) = -\frac{1}{2}(x_{t}-\mu)^{T}\Sigma^{-1}(x_{t}-\mu) + C \quad \quad (4)$$

#### 분류기의 테일러 급수 근사

* $x_t = \mu$ 근처에서 테일러 급수(Taylor expansion)를 사용해 선형 함수로 근사

$$\log p_{\phi}(y|x_{t}) \approx \log p_{\phi}(y|x_{t})|_{x_{t}=\mu} + (x_{t}-\mu)\nabla_{x_{t}}\log p_{\phi}(y|x_{t})|_{x_{t}=\mu} \quad \quad (5)$$

$$= (x_{t}-\mu)g + C_{1} \quad \quad (6)$$

$g = \nabla_{x_{t}}\log p_{\phi}(y|x_{t})|{x{t}=\mu}$

#### 가우시안 분포 재구성 (Completing the Square)

(2)의 Log Likelihood

$$\log(p_{\theta}(x_{t}|x_{t+1})p_{\phi}(y|x_{t})) = \log p_{\theta}(x_{t}|x_{t+1}) + \log p_{\phi}(y|x_{t})$$

$$\log p_{\phi}(y|x_{t}) \approx (x_{t}-\mu)g + C_{1} \quad \quad (6)$$

$$
\begin{aligned}
\log(p_\theta(x_t|x_{t+1})p_\phi(y|x_t)) &\approx -\frac{1}{2}(x_t - \mu)^T \Sigma^{-1} (x_t - \mu) + (x_t - \mu)^T g + C_2 \quad \quad (7) \\
&= -\frac{1}{2}(x_t - \mu - \Sigma g)^T \Sigma^{-1} (x_t - \mu - \Sigma g) + \frac{1}{2}g^T \Sigma g + C_2  \quad \quad (8) \\
&= -\frac{1}{2}(x_t - \mu - \Sigma g)^T \Sigma^{-1} (x_t - \mu - \Sigma g) + C_3  \quad \quad (9) \\
&= \log p(z) + C_4, \quad z \sim \mathcal{N}(\mu + \Sigma g, \Sigma)  \quad \quad (10) 
\end{aligned}
$$

##### (7)
1. 가우시안 분포(정규분포)의 원래 공식

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)$$

2. Log

$$\log p(x) = \log\left( \frac{1}{\sqrt{2\pi\sigma^2}} \right) + \log\left( \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right) \right)$$

$$\log p(x) = \underbrace{\log(\text{상수})}_{C} - \frac{1}{2\sigma^2}(x - \mu)^2$$

$$\log p(x) = -\frac{1}{2\sigma^2}(x - \mu)^2 + C$$

* 스칼라(숫자)일 때: $\frac{1}{\sigma^2}(x - \mu)^2$
* 벡터일 때: $(x - \mu)^T \Sigma^{-1} (x - \mu)$

$$\log p_{\theta}(x_{t}|x_{t+1}) = -\frac{1}{2}(x_{t}-\mu)^{T}\Sigma^{-1}(x_{t}-\mu) + C \quad \quad(7)$$

#### (8)

$$\text{식} = \underbrace{-\frac{1}{2}(x_t - \mu)^T \Sigma^{-1} (x_t - \mu)}_{\text{확산 모델 (2차 함수)}} + \underbrace{(x_t - \mu)^T g}_{\text{분류기 (1차 함수)}} + C$$

* $z = x_t - \mu$

$$\text{식} = -\frac{1}{2} z^T \Sigma^{-1} z + z^T g + C$$

* target

$$
-\frac{1}{2}(z - A)^T \Sigma^{-1} (z - A)
$$

$$
= -\frac{1}{2} \left( z^T \Sigma^{-1} z - z^T \Sigma^{-1} A - A^T \Sigma^{-1} z + A^T \Sigma^{-1} A \right)
$$

$z^T \Sigma^{-1} A = A^T \Sigma^{-1} z$

$$= -\frac{1}{2} z^T \Sigma^{-1} z + z^T \Sigma^{-1} A - \frac{1}{2} A^T \Sigma^{-1} A$$

* 가진 식: $-\frac{1}{2} z^T \Sigma^{-1} z + \mathbf{z^T g}$

* 전개 식: $-\frac{1}{2} z^T \Sigma^{-1} z + \mathbf{z^T \Sigma^{-1} A} - \frac{1}{2} A^T \Sigma^{-1} A$

$g = \Sigma^{-1} A$, $A = \Sigma g$

불필요한 상수항( $-\frac{1}{2} A^T \Sigma^{-1} A$ ) : $-\frac{1}{2} (\Sigma g)^T \Sigma^{-1} (\Sigma g) = -\frac{1}{2} g^T \Sigma \Sigma^{-1} \Sigma g = \mathbf{-\frac{1}{2} g^T \Sigma g}$

$$\text{결과} = \underbrace{-\frac{1}{2}(z - \Sigma g)^T \Sigma^{-1} (z - \Sigma g)}_{\text{완전 제곱식 부분}} + \underbrace{\frac{1}{2} g^T \Sigma g}_{\text{상수 보정 부분}} + C_2$$


### Conditional Sampling for DDIM

<p align="center">
<img width="787" height="237" alt="image" src="https://github.com/user-attachments/assets/0d205809-49b1-41d8-b9fe-350a2fc517df" />
</p>

$$\nabla_{x_{t}}\log p_{\theta}(x_{t}) = -\frac{1}{\sqrt{1-\overline{\alpha}_{t}}}\epsilon_{\theta}(x_{t}) \quad \quad (11)$$

* 좌변 (LHS): Score Function (점수 함수)
   * 어느 방향으로 이동해야 진짜 데이터($x_0$)처럼 보일 확률($p_{\theta}$)이 높아지는가?
* 우변 (RHS): Scaled Negative Noise (스케일된 음의 노이즈)
   * 노이즈를 걷어내는 방향
* Score(데이터로 가는 방향) $\approx$ -Noise(노이즈의 반대 방향)

#### 우리가 원하는 것은 조건부 분포 $p(x_t)p(y|x_t)$에서 샘플링
$$\nabla_{x_{t}}\log(p_{\theta}(x_{t})p_{\phi}(y|x_{t})) = \nabla_{x_{t}}\log p_{\theta}(x_{t}) + \nabla_{x_{t}}\log p_{\phi}(y|x_{t}) \quad \quad (12)$$

* 원래 이미지의 Score + 분류기의 Score

* (12)에 (11) 반영하면 (13)

$$= -\frac{1}{\sqrt{1-\overline{\alpha}_{t}}}\epsilon_{\theta}(x_{t}) + \nabla_{x_{t}}\log p_{\phi}(y|x_{t}) \quad \quad (13)$$


* 전체 Score(식 13)를 하나의 새로운 노이즈 예측값 $\hat{\epsilon}(x_t)$

$$-\frac{1}{\sqrt{1-\overline{\alpha}_{t}}} \hat{\epsilon}(x_t) = \text{Total Score}$$

$$\hat{\epsilon}(x_{t}) := \epsilon_{\theta}(x_{t}) - \sqrt{1-\overline{\alpha}_{t}}\nabla_{x_{t}}\log p_{\phi}(y|x_{t}) \quad \quad (14)$$


### Scaling Classifier Gradients

* 그라디언트에 1보다 큰 상수를 곱해주는 단순한 조작으로 클래스 이미지 생성에 도움
* 이론적으로 정확한 값인 스케일 $1.0$, 생성된 이미지는 해당 클래스처럼 보이지 않았음
* 1 보다 큰 상수 $s$ (예: 10.0), 클래스 확률이 거의 100%로 올라갔음

$$s \cdot \nabla_x \log p(y|x) = \nabla_x \log (p(y|x)^s)$$

<p align = 'center'>
<img width="1070" height="302" alt="image" src="https://github.com/user-attachments/assets/5b1ad5ad-b493-474c-8486-3c7de182bf77" />
</p>


* $s \approx 1$ 분류기 개입이 적고, 다양성 높고, 이미지 특징이 약하거나 흐릿
* $s > 1$ 분류기 강하게 개입, 품질/충실도 높음, 다양성 감소

* Metrics (지표)
  * FID ($\downarrow$): 낮을수록 좋음 (이미지 품질 + 다양성 종합 점수)
  * IS ($\uparrow$): 높을수록 좋음 (이미지의 선명도 및 클래스 명확성)
  * Precision ($\uparrow$): 높을수록 좋음 (이미지 품질/충실도)
  * Recall ($\uparrow$): 높을수록 좋음 (이미지 다양성)

---

## Result

<p align="center">
<img width="598" height="502" alt="image" src="https://github.com/user-attachments/assets/3ca1d688-65a6-4aaa-9b37-bfc705c02160" />
</p>

#### SOTA 달성

* ImageNet에서, Diffusion 모델이 당시 최고 성능의 모델들을 제치고 1등을

* ADM (Ablated Diffusion Model) 저자들이 만든 모델

* 경쟁 상대: BigGAN-deep (당시 최강의 GAN), VQ-VAE-2 (Likelihood 기반 모델)

* 결과 (ImageNet 256x256)
  * ADM-G (Guidance 적용): FID 3.97을 기록하며 BigGAN-deep(FID 6.95)을 큰 차이로 이겼습니다.
  * ADM (Guidance 미적용): 가이드 없이도 FID 10.94를 기록하여 Diffusion 자체의 강력함을 보여주었습니다.

* 의미: "Diffusion 모델은 생성 속도가 느리고 화질이 흐릿하다"는 편견을 깨고, "GAN보다 더 진짜 같은 이미지를 만들 수 있다"는 것을 최초로 입증했습니다.

#### Fidelity(품질) vs Diversity(다양성) 트레이드오프 분석

* Precision(품질) - Recall(다양성) 곡선

* GAN : 품질은 좋지만 다양성이 떨어지는(Recall이 낮은) 영역
* Diffusion : Diffusion은 Scale $s$를 조절함으로써, "다양성을 중시하는 모드(High Recall)"와 "고화질을 중시하는 모드(High Precision)" 사이를 자유롭게 오갈 수 있음

#### 고해상도 (ImageNet 512x512)

* 방식: 512x512 이미지를 한 번에 생성하는 것은 계산 비용이 너무 큽니다. 따라서 Upsampling(업샘플링) 방식을 사용했습니다.

* 먼저 저해상도 모델로 이미지를 생성합니다.
* 별도로 훈련된 Upsampler Diffusion Model을 사용해 이를 512x512로 키웁니다.
* 결과: 이 방식(Guidance 포함)으로 FID 3.85를 기록하며, 역시 BigGAN-deep을 능가

#### 정성적 평가 (Qualitative Results)
* GAN: 종종 기괴한 텍스처나 무너진 구조(Artifacts)가 나타나는 경향이 있습니다.
* Diffusion (ADM-G): 전체적인 구조(Global Structure)가 매우 안정적이며, 털 질감이나 그림자 같은 세부 묘사가 훨씬 자연스럽습니다. 특히 Guidance를 강하게 줄수록(Scale이 높을수록) 물체가 더 명확해지는 현상을 시각적으로 확인

---

## Limitations and Future work

1. 샘플링 속도의 한계 (Sampling Speed)
   1-1. GAN에 비해 샘플링 시간이 훨씬 오래
2. 레이블 데이터 의존성 (Labeled Datasets)
   2-1. 분류기 가이던스(Classifier Guidance)' 기법은 레이블(정답지)이 있는 데이터셋에만 적용
3. CLIP 모델의 노이즈 버전을 사용하여 텍스트 캡션으로 이미지 생성을 가이드하는(Text-to-Image) 방식에도 적용

---
