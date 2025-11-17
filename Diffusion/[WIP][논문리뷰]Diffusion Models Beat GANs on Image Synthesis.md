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
* 손실함수 MSE
* $\left\| \epsilon_\theta(x_t, t) - \epsilon \right\|^2$
#### Sampling
* Backward process에서 진행
* $x_t$ 에서 노이즈 $\epsilon_\theta$를 예측하며 $x_{t-1}, \mu_\theta$ 를 계산
#### Main Improvement
* Learned Variance
  * DDPM에서는 분산이 상수로 고정했지만 이 논문은 분산을 학습 하도록 했음
  * 50단계 미만의 샘플링은 DDIM 방식을 채택

$$\Sigma_\theta(x_t, t) = \exp(v \log \beta_t + (1 - v) \log \tilde{\beta}_t)$$

* $\beta_t, \tilde{\beta}_t$ 사이 interpolation
* $\Sigma_\theta(x_t, t), \epsilon_\theta(x_t, t)$ 둘다 학습

$$L_{\text{simple}} + \lambda L_{\text{vlb}}$$

* DDPM은 $L_{simple}$만 사용하여 노이즈$(\epsilon_\theta)$를 훈련
  * 단순 평균제곱오차(MSE)
* 현재 논문에서는 $L_{vlb}$를 추가
  * VLB : Variational Lower Bound
  * $\mu, \Sigma$ 둘다 고려한 수식
  * $\lambda$ 는 가중치 (0.0001)


#### 성능 평가 지표
* FID(Frechet Inception Distance) : 이미지 품질과 다양성 판단
* IS(Inception Score) : 특정 클래스 분류 되는지 품질과 다양성 측정
* Precision and Recall : 품질과 다양성 분리하여 측정


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

### Conditional Reverse Noising Process

<p align="center">
<img width="787" height="230" alt="image" src="https://github.com/user-attachments/assets/e8cdcd9c-5b47-4bdb-ad2e-ce7cb9d212ba" />
</p>

$$p_{\theta,\phi}(x_t | x_{t+1}, y) \propto \underbrace{p_{\theta}(x_t | x_{t+1})}_{\text{A. 확산 모델}} \cdot \underbrace{p_{\phi}(y | x_t)}_{\text{B. 분류기}}$$

* 이 두 확률을 곱한 새로운 분포 $p_{\theta,\phi}$에서 직접 샘플링하는 것은 계산이 매우 어렵

#### 근사

* 확산 모델( $p_{\theta}$ )은 원래 가우시안 분포 $\mathcal{N}(\mu, \Sigma)$ (평균 $\mu$, 분산 $\Sigma$ )
* 분류기( $p_{\phi}$ )의 $\log p_{\phi}(y|x_t)$를 1차 함수(직선)로 근사 (테일러 전개)
  * $\log p_{\phi}(y|x_t) \approx \text{상수} + (x_t - \mu) \cdot \mathbf{g}$
  * $\mathbf{g}$가 바로 분류기의 그래디언트 $\nabla_{x_t} \log p_{\phi}(y|x_t)$

* 기존 분포: $\mathcal{N}(\mu, \Sigma)$
* 안내된 분포 (근사): $\mathcal{N}(\mathbf{\mu + \Sigma g}, \Sigma)$

* Algorithm 1 에서 샘플링 시 평균에 $s\Sigma \nabla_{x_t} \log p_{\phi}(y|x_t)$ (즉, $s\Sigma \mathbf{g}$)를 더해주는 이유

$$
\begin{aligned}
\log p_\phi(y|x_t) &\approx \log p_\phi(y|x_t)|_{x_t=\mu} + (x_t - \mu) \nabla_{x_t} \log p_\phi(y|x_t)|_{x_t=\mu} \quad &(5) \\
&= (x_t - \mu)g + C_1 \quad &(6)
\end{aligned}
$$

*  $\log p_\phi(y|x_t)$를 평균 $\mu$ 에서 테일러 급수를 사용한 1차 항 근사 (식 5~6)

$g = \nabla_{x_t} \log p_\phi(y|x_t)|_{x_t=\mu}$, and $C_1$ is a constant

$$
\begin{aligned}
\log(p_\theta(x_t|x_{t+1})p_\phi(y|x_t)) &\approx -\frac{1}{2}(x_t - \mu)^T \Sigma^{-1} (x_t - \mu) + (x_t - \mu)g + C_2 \quad &(7) \\
&= -\frac{1}{2}(x_t - \mu - \Sigma g)^T \Sigma^{-1} (x_t - \mu - \Sigma g) + \frac{1}{2}g^T \Sigma g + C_2 \quad &(8) \\
&= -\frac{1}{2}(x_t - \mu - \Sigma g)^T \Sigma^{-1} (x_t - \mu - \Sigma g) + C_3 \quad &(9) \\
&= \log p(z) + C_4, \quad z \sim \mathcal{N}(\mu + \Sigma g, \Sigma) \quad &(10)
\end{aligned}
$$

* 식 7
  * 좌변, Diffusion의 역과정(Reverse process) 확률 $p_\theta$와 분류기 확률 $p_\phi$를 곱한 것의 로그
  * 우변 첫째항, Diffusion 모델의 역과정 $p_\theta(x_t|x_{t+1})$은 정규분포 $\mathcal{N}(\mu, \Sigma)$를 따릅니다. 정규분포의 로그 취하면 2차 형식(Quadratic form)인 $-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)$
  * 우변의 두 번째 항 ($(x_t - \mu)g$): 식 (6)에서 구한 분류기의 근사값

* 식 8~9 완전 제곱식 만들기
  * $-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu) + (x-\mu)g$ 형태를 정리하면, 평균이 이동된 새로운 2차 형식 $-\frac{1}{2}(x - (\mu + \Sigma g))^T \Sigma^{-1} (x - (\mu + \Sigma g))$ 꼴로 바꿀 수
  * 남는 찌꺼기 항들($\frac{1}{2}g^T \Sigma g$ 등)은 $x_t$와 무관하므로 상수로 처리되어 $C_3$로 흡수

* 식 10
  * 이는 평균이 $\mu + \Sigma g$이고 공분산이 $\Sigma$인 정규분포의 로그 확률 밀도 함수(log-pdf)와 모양이 같음
  * 분류기 정보( $y$ )를 반영하여 샘플링할 분포는 원래의 평균 $\mu$에서 $\Sigma g$만큼 이동한 정규분포

* Key Takeaway
  * $\mu' = \mu + \Sigma \cdot \nabla \log p(y|x_t)$

### Conditional Sampling for DDIM

<p align="center">
<img width="787" height="237" alt="image" src="https://github.com/user-attachments/assets/0d205809-49b1-41d8-b9fe-350a2fc517df" />
</p>

* DDPM 방식에서 DDIM 샘플링 과정 적용 관련 내용
  * 위에 조건부 샘플링, Stochastic한 샘플링에서만 유효
  * DDIM 같은 Deterministic한 샘플링 에는 적용 불가

* DDPM과 DDIM의 공통적 관계
* 모델이 예측한 노이즈 $\epsilon_\theta(x_t)$는 데이터 분포의 로그 기울기(Score)와 비례

$$\nabla_{x_t} \log p(x_t) \propto - \frac{\epsilon_\theta(x_t)}{\sqrt{1-\bar{\alpha}_t}}$$

$$\nabla_{x_t} \log p(x_t|y) = \underbrace{\nabla_{x_t} \log p(x_t)}_{\text{Unconditional Score}} + \underbrace{\nabla_{x_t} \log p(y|x_t)}_{\text{Classifier Gradient}}$$

* 이를 이용해 조건부 노이즈 예측값 정의



$$
\begin{aligned}
\nabla_{x_t} \log(p_\theta(x_t)p_\phi(y|x_t)) &= \nabla_{x_t} \log p_\theta(x_t) + \nabla_{x_t} \log p_\phi(y|x_t) \quad &(12) \\
&= -\frac{1}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t) + \nabla_{x_t} \log p_\phi(y|x_t) \quad &(13) \\
\hat{\epsilon}(x_t) &= \epsilon_\theta(x_t) - \sqrt{1 - \bar{\alpha}_t} \nabla_{x_t} \log p_\phi(y|x_t) \quad (14)
\end{aligned}
$$


### Scaling Classifier Gradients

$$\hat{\epsilon}(x_t) := \epsilon_\theta(x_t) - \sqrt{1-\bar{\alpha}_t} \cdot \red{s} \cdot \nabla_{x_t} \log p_\phi(y|x_t)$$

* 단순히 $s=1$을 사용하면 생성된 이미지가 클래스 조건($y$)을 따르기는 하지만, 여전히 모호하거나 해당 클래스의 가장 전형적인 특징을 보여주지 못하는 경우가 많음

* 강아지일 수도 있는 이미지를 누가봐도 확실한 강아지 이미지를 생성하려고 노력

* $s \approx 1$ 분류기 개입이 적고, 다양성 높고, 이미지 특징이 약하거나 흐릿
* $s > 1$ 분류기 강하게 개입, 품질/충실도 높음, 다양성 감소

<p align="center">
<img width="667" height="205" alt="image" src="https://github.com/user-attachments/assets/c9afcf09-e104-4f7d-b2f2-b31dafbc4d1b" />
</p>

* Metrics (지표)
  * FID ($\downarrow$): 낮을수록 좋음 (이미지 품질 + 다양성 종합 점수)
  * IS ($\uparrow$): 높을수록 좋음 (이미지의 선명도 및 클래스 명확성)
  * Precision ($\uparrow$): 높을수록 좋음 (이미지 품질/충실도)
  * Recall ($\uparrow$): 높을수록 좋음 (이미지 다양성)


### Result

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
