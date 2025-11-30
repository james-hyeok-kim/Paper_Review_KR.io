# Diffusion Models Beat GANs on Image Synthesis
저자 : Prafulla Dhariwal, Alex Nichol, OpenAI

출간 : NeurIPS, 2021.

논문 : [PDF](https://arxiv.org/pdf/2105.05233)

---
## Introduction

#### 문제

* GAN은 훈련이 어렵고, 하이퍼파라미터 선정에 민감
* 확산 모델(Diffusion Models) 모델
    * GAN보다 훈련이 쉽고 데이터의 다양성을 더 잘 포착한다는 장점이 있음
    * 시각적인 샘플 품질 면에서는 여전히 GAN에 미치지 못함

---

## Background

#### Training
* 손실 함수: MSE, $||\epsilon - \epsilon_{\theta}(x_{t},t)||^2$ 를 최소화

#### Sampling
* 평균 $\mu_{\theta}$
* 노이즈 $\epsilon_{\theta}$
* 분산 $\Sigma_{\theta}$

---
#### Improvement

* 아키텍처 개선
    * Layer 수 보다, Channel를 늘리는 것이 성능에 효과적
    * 어텐션 헤드의 수를 늘리거나 헤드 당 채널 수를 줄이는 것이 FID를 개선
    * 적응형 그룹 정규화 (AdaGN)
* 분류기 가이던스 (Classifier Guidance)
    * 분류기(Classifier) $p_{\phi}(y|x_t)$를 사용
    * 분류기의 그라디언트 $\nabla_{x} \log p_{\phi}(y|x_t)$를 사용

<p align ='center'>
<img width="1285" height="435" alt="image" src="https://github.com/user-attachments/assets/93a23319-7e68-4e6f-97a6-5e104d46d6cd" />
</p>

<p align ='center'>
<img width="1315" height="716" alt="image" src="https://github.com/user-attachments/assets/b01cbf09-7430-4895-857f-d1e152f8aa6a" />
</p>

<p align ='center'>
<img width="1295" height="285" alt="image" src="https://github.com/user-attachments/assets/e5bf837e-49d8-4c02-8384-98be62696549" />
</p>


$$AdaGN(h,y) = y_{s} \cdot \text{GroupNorm}(h) + y_{b}$$

* AdaGN
    * 타임스텝(timestep)과 클래스(class) 정보를 모델에 더 효과적으로 주입하기 위해 도입된 기법
    * $y = [y_s, y_b]$: 타임스텝 임베딩과 클래스 임베딩을 선형 투영(linear projection)하여 얻은 값


* 분산 학습 (Learned Variances): $L_{simple}$과 변분 하한(variational lower bound) $L_{vlb}$를 결합한 하이브리드 목적함수를 사용
* DDIM 샘플링

$$\Sigma_{\theta}(x_{t},t) = \exp(v \log \beta_{t} + (1-v) \log \tilde{\beta}_{t}) \quad \quad (1)$$

* $\beta_t$ (상한선, Upper Bound)
    * 확산 과정(forward process)에서 사용된 고정된 분산 값
* $\tilde{\beta}_t$ (하한선, Lower Bound)
    * 원본 데이터 $x_0$를 알고 있을 때 계산할 수 있는 사후 확률(posterior)의 분산입니다
* $v$ (학습 파라미터, Interpolation Factor)
    * 신경망이 출력하는 값으로, 0과 1 사이의 값을 가집니다
    * 이 값은 모델이 현재 타임스텝에서 분산을 $\beta_t$에 가깝게 할지, $\tilde{\beta}_t$에 가깝게 할지를 결정

---

## Classifier Guidance


* GAN은 조건부 이미지 생성 시 클래스 레이블 정보를 적극적으로 활용하여 성공을 거두었음
* 확률적 샘플링(Stochastic Sampling)을 위한 유도 (식 2~10): 가우시안 분포의 평균을 이동시키는 방식
* 결정론적 샘플링(DDIM)을 위한 유도 (식 11~14): 노이즈 예측값( $\epsilon$ )을 수정하는 방식

### Conditional Reverse Noising Process (2~10)

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


---

## Result
<p align = 'center'>
<img width="1062" height="888" alt="image" src="https://github.com/user-attachments/assets/77533e9e-c67b-4245-929a-cc6e478dbd38" />
</p>

* 정량적 평가: GAN을 능가하는 SOTA 달성
* 모든 태스크에서 최고 성능 (LSUN 침실/말/고양이, ImageNet 등)
* 고해상도 ImageNet에서의 압승, ImageNet $256\times256$ 및 $512\times512$ 해상도
* 빠른 샘플링 가능성, 25 단계(Steps)


<p align = 'center'>
<img width="1058" height="589" alt="image" src="https://github.com/user-attachments/assets/95d08759-651d-414c-8033-f6fcb9ee969d" />
</p>

* 정성적 평가: 다양성과 품질의 균형 (Figure 6)
* 가이던스(Guidance): 다양성(Recall)을 희생하여 정밀도(Precision/충실도)를 높이는 데 매우 효과적입니다.
* 업샘플링(Upsampling): 다양성(Recall)을 높게 유지하면서 정밀도(Precision)를 개선합니다.
* 최적의 조합: 저해상도 모델에 가이던스를 적용한 뒤, 이를 업샘플링하는 방식

---

## Limitations and Future work

1. 샘플링 속도의 한계 (Sampling Speed)
   1-1. GAN에 비해 샘플링 시간이 훨씬 오래
2. 레이블 데이터 의존성 (Labeled Datasets)
   2-1. 분류기 가이던스(Classifier Guidance)' 기법은 레이블(정답지)이 있는 데이터셋에만 적용
3. CLIP 모델의 노이즈 버전을 사용하여 텍스트 캡션으로 이미지 생성을 가이드하는(Text-to-Image) 방식에도 적용

---
