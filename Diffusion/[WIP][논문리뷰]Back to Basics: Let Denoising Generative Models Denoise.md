# Back to Basics: Let Denoising Generative Models Denoise (JiT - Just Image Transformers)
저자 : Tianhong Li, Kaiming He, MIT

출간 : arXiv 2025. (CVPR 2026, ICLR 2026 등??)

논문 : [PDF](https://arxiv.org/pdf/2511.13720)

<p align = 'center'>
<img width="681" height="639" alt="image" src="https://github.com/user-attachments/assets/38b2f741-47a4-4368-8c04-55b23e3883d6" />
</p>

---


## Abstract

* 신경망이 노이즈( $\epsilon$ ) 자체를 예측하거나, 데이터와 노이즈가 섞인 속도($v$, flow velocity)를 예측하도록 훈련

* Manifold Assumption (매니폴드)
    * 자연 데이터( $x$ ): 고차원 픽셀 공간 안에 존재하지만, 실제로는 저차원 매니폴드(manifold) 위에 분포합니다 (즉, 데이터의 구조가 질서 정연함)
    * 노이즈( $\epsilon$ ): 고차원 공간 전체에 무질서

* JiT (Just Image Transformers)
    * 단순함: 토크나이저(tokenizer), 사전 훈련(pre-training), 추가적인 손실 함수(extra loss) 없이, 픽셀 단위에서 작동하는 단순한 트랜스포머 모델을 사용
    * 작동 방식: 이미지를 큰 패치(16 또는 32 픽셀)로 나누어 처리하며, 신경망은 깨끗한 이미지( $x$ )를 직접 예측
 
---

## 1. Introduction

#### 기존 해결책의 한계와 새로운 제안

* 기존의 한계: '잠재 디퓨전 모델(Latent Diffusion Models, 예: Stable Diffusion)'은 이미지를 미리 압축하여 차원 문제를 숨겼을 뿐 해결한 것은 아님
    * 픽셀 공간에서 작동하는 모델들은 복잡한 설계(Dense convolution 등)에 의존해야 했음
* 새로운 제안 (Back to Basics): 신경망이 깨끗한 이미지를 직접 예측( $x$-prediction )하도록 함
* JiT (Just image Transformers): 이 방식을 사용하면, 토크나이저나 복잡한 사전 훈련 없이 단순한 비전 트랜스포머(ViT)만으로도 고해상도 이미지를 효과적으로 생성

#### 연구의 의의
* ImageNet 데이터셋 실험(256, 512 해상도)에서, 기존 방식($\epsilon$ 또는 $v$ 예측)이 처참하게 실패하는 고차원 패치 환경에서도 $x$-prediction을 사용한 모델(JiT)은 훌륭한 성능
* 이 연구는 "Self-contained(자립형)" 모델을 지향하며, 토크나이저를 만들기 어려운 과학 데이터(단백질, 날씨 등) 영역에서도 디퓨전 트랜스포머를 쉽게 적용할 수 있는 길

---

## 2. Related Work

#### Diffusion Models and Their Predictions
* 초기 및 표준: 초기 디퓨전 모델은 확률 분포의 매개변수를 예측했으나, DDPM이 등장하면서 노이즈( $\epsilon$ 를 예측( $\epsilon$ -prediction ) 하는 것이 표준으로 자리 잡음
* 속도 예측 ( $v$-prediction): 이후 연구들은 데이터와 노이즈가 결합된 '속도( $v$ )'를 예측하는 방식을 도입했습니다. 이는 Flow Matching 모델들과도 연결
* EDM의 한계: 획기적인 연구였던 EDM조차도 '사전 조건화(pre-conditioning)' 방식을 사용하여, 네트워크가 순수한 노이즈 제거 이미지보다는 데이터와 노이즈가 섞인 값을 출력하도록 유도
    * EDM은 2022년 NVIDIA의 Karras 등이 발표한 논문 "Elucidating the Design Space of Diffusion-Based Generative Models"를 줄여서 부르는 말
    * EDM이 디퓨전 모델의 설계를 체계적으로 정리한 '교과서' 같은 역할

#### Denoising Models & Manifold Learning

* Denoising Autoencoders (DAEs): 과거의 DAE는 매니폴드 가정을 기반으로 데이터의 저차원 구조를 학습하기 위해 깨끗한 데이터를 예측하도록 훈련
* Score Matching과의 차이: 반면, 현대 디퓨전 모델의 기반이 된 'Denoising Score Matching'은 수식적으로 노이즈( $\epsilon$ )를 예측하는 것과 동일
    * 이 논문은 DAE의 철학(데이터 예측)으로 돌아가고자 함
* 매니폴드 학습: 병목(bottleneck) 구조를 통해 고차원 데이터 속의 유용한 저차원 정보를 걸러내는 고전적인 학습 방법론
    * Latent Diffusion Models (LDM)은 이를 오토인코더 단계에서 수행하지만, 이 논문은 디퓨전 과정 자체에서 이를 수행

#### Pixel-space Diffusion
* 픽셀 공간에서 직접 작동하는 모델들의 한계
* CNN 기반: 초기 픽셀 디퓨전은 U-Net 같은 무거운 CNN을 사용하여 계산 비용이 높음
* ViT 기반의 난관: 비전 트랜스포머(ViT)를 픽셀에 직접 적용하면 패치 하나의 차원(dimensionality)이 너무 높아져서 성능이 급격히 떨어지는 문제
* 기존의 복잡한 해결책: 이를 해결하기 위해 기존 연구들은 계층적 구조를 쓰거나(SiD2, PixelFlow), NeRF 헤드를 붙이거나(PixNerd), 복잡한 사전 훈련(Pre-training)을 도입

#### x-prediction
* 새로운 것이 아님: 깨끗한 데이터를 예측하는 $x$-prediction은 사실 초기 DDPM 코드에도 있었던 자연스러운 방식
    * 당시에는 노이즈 예측($\epsilon$-pred) 성능이 더 좋아서 잊혀졌음
* 이미지 복원 분야: 이미지 복원(Restoration) 분야에서는 깨끗한 이미지를 예측하는 것이 당연한 목표
* 이 연구의 차별점: 저자들은 $x$-prediction이라는 개념을 새로 만든 것이 아니라, 고차원 데이터 공간에서 저차원 매니폴드를 학습할 때 이 방식이 필수적임을 규명

---

## 3. On Prediction Outputs of Diffusion Models

* $x$: 깨끗한 원본 데이터
* $\epsilon$: 순수한 노이즈
* $v$: 속도 (Velocity, 데이터와 노이즈가 섞인 변화율)

<p align = 'center'>
<img width="1368" height="303" alt="image" src="https://github.com/user-attachments/assets/a6cdd957-690a-46c5-beb3-72aa007a7821" />
</p>

* Prediction Space : 신경망이 $x, \epsilon, v$ 중 무엇을 직접 출력할 것인가?
* Loss Space : 정답과의 차이를 $x, \epsilon, v$ 중 어떤 공간에서 계산할 것인가?

<p align = 'center'>
<img width="690" height="778" alt="image" src="https://github.com/user-attachments/assets/17db4679-b7d2-46cf-94bd-a70ae682cae8" />
</p>

* 실험 설정
    * 진짜 데이터: 2차원 평면 위의 나선형(Spiral) 구조 (저차원 매니폴드)
    * 관찰 공간: 이를 512차원이라는 거대한 고차원 공간에 묻어둠 ($d=2, D=512$)
    * 네트워크: 용량이 제한된 작은 신경망(MLP) 사용
* 실험 결과
    * $\epsilon$-pred, $v$-pred: 차원( $D$ )이 높아질수록 노이즈 정보를 감당하지 못해 처참하게 실패(Catastrophic failure)하고 이미지가 뭉개짐
    * $x$-pred: 네트워크 용량이 부족해도 선명한 나선형 구조를 완벽하게 복원합니다

#### 수식 해석

$$z_{t} = t x + (1-t) \epsilon \quad \quad (1)$$ 

* 학습을 위해 **깨끗한 이미지($x$)**와 **노이즈($\epsilon$)**를 섞어서 **손상된 이미지($z_t$)**를 만드는 과정

$$v = x - \epsilon \quad \quad (2)$$

* 어떤 방향과 속도로 변해야 하는지를 나타내는 속도( $v$ )
* 유도 과정: 수식 (1)을 시간 $t$에 대해 미분

$$\frac{d}{dt} z_t = \frac{d}{dt}(tx + (1-t)\epsilon) = x - \epsilon$$

$$\mathcal{L} = \mathbb{E}_{t,x,\epsilon} ||v_{\theta}(z_{t}, t) - v||^2 \quad \quad (3)$$

* 손실 함수 (Loss Function)

$$dz_{t}/dt = v_{\theta}(z_{t}, t) \quad \quad (4)$$

* 샘플링 과정 (Sampling ODE)
    * 실제로 이미지를 생성하는 과정
* 작동
    * 완전한 노이즈( $z_0$ )에서 시작
    * 신경망이 알려주는 속도( $v_{\theta}$ )를 따라 조금씩 이동 (미분방정식 풀이)
    * $t=0$에서 $t=1$까지 이동하면 깨끗한 이미지( $z_1$ )가 완성

$$
\begin{cases}
\boldsymbol{x}_\theta = \text{net}_\theta \\
\boldsymbol{z}_t = t \boldsymbol{x}_\theta + (1 - t) \boldsymbol{\epsilon}_\theta \\
\boldsymbol{v}_\theta = \boldsymbol{x}_\theta - \boldsymbol{\epsilon}_\theta
\end{cases} \tag{5}
$$

* System of Equations
    * 신경망의 출력이 무엇이든($x, \epsilon, v$), 수식 (1)과 (2)의 관계를 이용하면 나머지 값들을 모두 계산해낼 수 있음


---

## 4. “Just Image Transformers” for Diffusion




---

## 5. Comparisons

---

## 6. Discussion and Conclusion

---

## Appendix

### Manifold
* 매니폴드(Manifold)'는 복잡한 수학적 정의보다는 "데이터가 존재하는 숨겨진 규칙이나 구조"를 설명하기 위한 개념으로 이해
* 비유: 아주 거대한 3차원 방(고차원 공간)이 있다고 상상해 보세요. 이 방 안의 공기 분자처럼 무수히 많은 점을 찍을 수 있음
* 데이터의 위치: 하지만 '진짜 강아지 사진'이나 '진짜 풍경 사진'은 이 방 아무 데나 무작위로 존재하지 않습니다
    * 진짜 데이터들은 방 한구석에 얇은 종이 한 장(저차원 매니폴드) 위에만 모여 있음

---
