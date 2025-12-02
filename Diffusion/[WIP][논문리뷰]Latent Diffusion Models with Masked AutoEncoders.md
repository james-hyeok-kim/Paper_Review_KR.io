# Latent Diffusion Models with Masked AutoEncoders

저자 : Junho Lee1* Jeongwoo Shin1* Hyungwook Choi1 Joonseok Lee1† 1Seoul National University, Seoul, Korea

출간 : International Conference on Computer Vision (ICCV), 2025

논문 : [PDF](https://arxiv.org/pdf/2507.09984)

---


## 1. Introduction

<p align = 'center'>
<img width="538" height="462" alt="image" src="https://github.com/user-attachments/assets/9837fe3e-1386-497e-85d3-d519699dab53" />
</p>

#### 1. 연구 배경 및 문제 제기

* 기존 모델의 한계
    * 최근 확산 모델(Diffusion Models, DMs)이 이미지 생성 분야에서 뛰어난 성능을 보이지만, 픽셀 공간에서 직접 연산하기 때문에 계산 비용이 매우 높다는 단점이 있습니다.
* LDM의 등장
    * LDMs는 오토인코더(Autoencoder)를 사용해 이미지를 압축된 '잠재 공간(Latent space)'으로 변환하여 연산 효율을 높입니다.
* 핵심 문제
    * 일반적으로 사용되는 VAE(Variational Autoencoder)는 잠재 공간을 확률 분포에 맞추려다 보니 원본 이미지 복원 능력이 떨어지는 경향이 있습니다.
    * Stable Diffusion과 같은 최신 모델들은 이를 보완하기 위해 다양한 손실 함수(Loss)를 추가했지만, 정작 **"LDM을 위해 오토인코더가 갖춰야 할 최적의 속성이 무엇인가?"**에 대한 근본적인 연구는 부족했습니다.


#### 2. 오토인코더의 3가지 핵심 조건 제안

1. 잠재 공간의 부드러움 (Latent Smoothness)
    * 잠재 공간 내에서 작은 변화가 이미지의 급격한 변화로 이어지지 않아야 합니다.
    * 공간이 부드러워야 확산 모델이 노이즈가 섞인 상태에서도 안정적으로 이미지를 생성할 수 있습니다.
    * 분석 결과, 일반적인 AE(AutoEncoder)는 이 속성이 부족하며, 확률적 인코딩을 하는 VAE 계열이 훨씬 유리합니다.

2. 지각적 압축 품질 (Perceptual Compression Quality)
    * 이미지를 압축할 때 시각적 디테일은 줄이되 의미(Semantics)는 유지해야 합니다.
    * 기존 모델(SD-VAE)은 너무 과하게 압축하여(객체 수준), 미세한 디테일을 잃어버리는 경향이 있습니다
    * 저자들은 **계층적 압축(Hierarchical compression)** 을 제안합니다.
        * '객체(큰 틀) -> 부분(이목구비 등) -> 미세 디테일' 순서로 계층적으로 정보를 저장하여 압축과 복원 품질의 균형을 맞춥니다.

3. 복원 품질 (Reconstruction Quality)
    * 압축된 정보를 다시 이미지로 만들 때, 사람 눈에 자연스러워 보이는 것(Perceptual level)뿐만 아니라 픽셀 단위의 정확도(Pixel-level)도 높아야 합니다.
    * 기존 SD-VAE는 눈에 보이는 품질은 좋지만 픽셀 정확도는 떨어지는 것으로 나타났습니다.

#### 3. 해결책: VMAE (Variational Masked AutoEncoders)

* Masked AutoEncoder (MAE)의 장점 활용: MAE가 이미지의 특징을 계층적으로 잘 학습한다는 점에 착안했습니다.
* 성능 및 효율성
    * VMAE는 SD-VAE 대비 파라미터 수는 13.4%
    * 연산량(GFLOPs)은 4.1% 수준으로 매우 가볍지만
    * 생성 품질은 더 뛰어납니다.
    * 또한 학습 속도도 2.7배 더 빠릅니다.
 
---

## 2. Related Work

#### 이미지 생성을 위한 오토인코더 (Autoencoders for Image Generation)


* 두 가지 주요 접근법
    * MAE 기반 (MAEs-based): MaskGIT이나 MAGE 같은 모델들은 마스킹된 인코딩 방식을 사용하며, 주로 이산적(discrete) 토큰 공간을 활용합니다.
    * VAE 기반 (VAEs-based): VQ-VAE나 VQGAN은 코드북(Codebook)을 활용한 이산적 잠재 공간을 사용하여 생성 성능을 높였습니다.

* LDM에서의 오토인코더 문제점
    * LDM은 오토인코더를 통해 압축된 잠재 표현을 생성하고 이를 다시 이미지로 복원합니다.
    * 이 과정에서 생성 속도는 빨라지지만, 복원 품질(Reconstruction quality)이 저하되는 문제가 자주 발생합니다.

* 최신 연구들의 시도
    * Stable Diffusion 3, Emu, FLUX 등 최신 모델들은 복원 품질을 높이기 위해 잠재 차원(Latent dimensionality)의 크기를 키우는 방식을 택했습니다.
    * 차원이 커지면 세부 묘사는 좋아지지만 연산량은 늘어납니다.
    * 이외에도 DC-AE 등은 공간적 압축률을 높이면서도 품질을 잃지 않기 위해 노력해 왔습니다.

---

## 3. Preliminary

### 3.1. Autoencoders

#### 1. 기본 오토인코더 (Deterministic & Probabilistic)

* Vanilla AutoEncoders (AEs)
    * 가장 기본적인 형태로, 오직 복원 손실(Reconstruction loss)만을 최소화하도록 학습됩니다.
    * 입력을 압축했다가 다시 그대로 복구하는 것이 목표
    * 잠재 공간이 고정된 벡터(Deterministic)로 형성됩니다.

* Denoising AutoEncoders (DAEs)
    * AE의 확장형으로, 입력 데이터에 노이즈를 섞은 후 원본(노이즈 없는 상태)을 복구하도록 학습합니다.
    * 이 과정을 통해 더 견고한(Robust) 특징을 학습하고 일반화 성능을 높입니다.

* Variational AutoEncoders (VAEs)
    * 확률적(Probabilistic) 프레임워크를 도입한 모델입니다.
    * 입력을 고정된 벡터가 아닌 잠재 분포(Distribution)로 인코딩합니다.
    * KL-divergence를 사용하여 잠재 공간이 가우시안 분포를 따르도록 규제(Regularize)함으로써, 새로운 데이터를 샘플링(생성)하기 유리하게 만듭니다

#### 2. 최신 LDM 표준 모델

* StableDiffusion VAEs (SD-VAEs)
    * 기존 LDM에서 사용되던 VQGAN을 기반으로 하지만, 양자화(Quantization) 층을 없애고 연속적인 특징(Continuous features)을 사용합니다.
    * 단순 복원 손실뿐만 아니라, 적대적 손실(Adversarial loss)과 지각적 손실(Perceptual loss)을 추가로 사용하여 사람 눈에 자연스러운 고품질 이미지를 만듭니다.
    * 이 논문에서는 최신 모델인 Stable Diffusion 3의 설정을 따릅니다.
 
#### 3. 제안 모델의 기반

* Masked AutoEncoders (MAEs)
    * Vision Transformer (ViT)를 기반으로 한 자기지도 학습(Self-supervised learning) 방법입니다.
    * 이미지의 일부 패치를 무작위로 가리고(Masking), 인코더는 보이는 부분만 처리하여 잠재 벡터로 만듭니다.
    * 디코더는 이 잠재 벡터와 마스크 토큰을 이용해 가려진 부분을 예측하여 복원합니다.
    * 저자들은 이 MAE가 가진 계층적 특징 학습 능력에 주목하여 연구를 시작했습니다.

### 3.2. Latent Diffusion Models

#### 1. 기존 확산 모델 (Standard Diffusion Models)
* 픽셀 공간(Pixel Space)에서 작동
$$L_{DM} = \mathbb{E}_{x, \epsilon \sim N(0,1), t} [||\epsilon - \epsilon_{\theta}(x_{t}, t)||_{2}^{2}]$$

#### 2. 잠재 확산 모델 (Latent Diffusion Models, LDMs)

* 잠재 공간(Latent Space)에서 작동
* 핵심 변경점: 픽셀 이미지 $x$를 그대로 쓰는 대신, 사전 학습된 인코더 $\mathcal{E}$를 통해 압축된 잠재 표현 $z \sim \mathcal{E}(x)$를 사용

$$L_{LDM} = \mathbb{E}_{\mathcal{E}(x), y, \epsilon \sim N(0,1), t} [||\epsilon - \epsilon_{\theta}(z_{t}, t, \tau_{\theta}(y))||_{2}^{2}]$$

* $z_t$: 노이즈가 섞인 잠재 벡터
* $\tau_{\theta}(y)$: 텍스트나 클래스 레이블 같은 조건부 입력(Conditioning inputs) $y$를 잠재 공간에 투영하는 함수로, 이를 통해 다양한 멀티모달 생성 작업이 가능해집니다

---

## 4. Variational Masked AutoEncoders
### 4.1. Support of Latent Space
* LDM(Latent Diffusion Model)이 성공적으로 학습되려면, 데이터가 압축되는 잠재 공간이 연속적이고 부드러운(Smooth) 형태
    * 만약 잠재 공간이 너무 희소(Sparse)하거나 차원이 낮으면, 확산 모델이 학습하는 과정에서 '스코어 매칭(Score matching)'이 불안정해져 성능이 저하


#### Deterministic vs Probabilistic

<p align = 'center'>
<img width="540" height="435" alt="image" src="https://github.com/user-attachments/assets/564ac2e2-f4f6-409b-bb8f-13b3528870e1" />
</p>


* 결정론적 인코딩 (예: AE, DAE)
    * 방식: 입력 이미지를 잠재 공간의 고정된 하나의 점(Fixed vector)으로 변환합니다.
    * 문제점 (Sparse Latent Space): 디코더는 오직 그 특정 점들만 다시 이미지로 복구할 수 있습니다.
    * 확산 모델이 노이즈를 제거하는 과정에서 예측값이 아주 조금만 빗나가도(오차 발생), 디코더가 이를 해석하지 못해 이미지가 망가집니다.
    * 즉, "정답이 아니면 오답"인 상황이 됩니다.

* 확률적 인코딩 (예: VAE, VMAE)
    * 방식: 입력 이미지를 하나의 점이 아닌 확률 분포(Distribution)로 변환합니다.
    * 장점 (Smooth Latent Space): 잠재 공간이 점들이 아닌 '영역'으로 채워집니다.
    * 확산 모델의 예측값이 정답 근처(Vicinity)에 떨어지기만 하면, 디코더가 이를 안정적으로 원본 이미지로 복원해 냅니다.
    * 즉, "정답 근처면 정답으로 인정"하는 유연성을 가집니다.

#### VMAE의 설계 방식 (VMAE는 확률적 접근을 채택)

* 입력 데이터 $x$를 고정된 벡터가 아닌 확률 분포 $q_{\phi}(z|x)$로 인코딩하여 잠재 공간을 부드럽게 만듭니다
* 또한, KL 발산(KL Divergence)을 사용하여 잠재 분포가 가우시안 분포(Prior, $p(z)$)를 따르도록 규제(Regularize)합니다
* 이 규제는 잠재 공간의 분산을 단위(Unit) 분산으로 유지하게 하여 확산 모델의 학습 조건(VP condition)을 만족

$$L_{reg} = \mathbb{E}_{p_{data}(x)} [D_{KL}(q_{\phi}(z|x) || p(z))] \quad \quad (1)$$


### 4.2. Perceptual Compression - 계층적 압축(Hierarchical Compression)

<p align = 'center'>
<img width="542" height="426" alt="image" src="https://github.com/user-attachments/assets/56f99231-3f67-4a6a-b85a-5fa14b150a0e" />
</p>

#### 1. 지각적 압축의 정의와 딜레마
* 정의: 지각적 압축이란 원본 데이터를 작은 잠재 공간으로 인코딩할 때, 미세한 디테일은 버리되 전체적인 의미(Semantics)는 보존하는 것을 말합니다.
* 딜레마 (연속적 스펙트럼): '보존해야 할 의미'와 '버려도 되는 디테일'의 경계는 명확하지 않습니다.
    * 이는 픽셀 수준(압축 없음)에서 객체 수준(최대 압축)까지 이어지는 연속적인 스펙트럼상에 존재합니다.
* 중간 단계: 부위(Parts, 예: 눈, 코) 또는 패턴(Patterns, 예: 질감, 색상).

#### 2. 기존 모델(SD-VAE)의 한계: 너무 과한 압축
* SD-VAE가 '객체 수준'에 치우친 과도한 압축을 수행

#### 3. VMAE의 해결책: 계층적 압축 (Hierarchical Compression)

* 구조: 잠재 공간을 다음과 같이 계층적으로 구성합니다.
    * 객체 수준 (Object-level): 큰 의미 덩어리 (예: 기린 전체)
    * 부위 수준 (Part-level): 구성 요소 (예: 기린의 얼굴, 다리)
    * 패턴 수준 (Pattern-level): 미세 디테일 (예: 털 무늬, 질감)

* 장점
    * 확산 모델(Diffusion Model) 입장에서는 상위 레벨의 단순화된 정보가 있어 학습하기 쉽습니다.
    * 동시에 하위 레벨의 디테일 정보가 살아있어, 디코더가 고품질로 복원할 수 있습니다

#### 4. 구현 방법: 마스킹된 부분 예측 (Masked-part Prediction)

* 작동 방식
    * 인코더는 이미지의 보이는 부분( $x_v$ )만 잠재 변수( $z$ )로 만듭니다.
    * 디코더는 이 $z$를 보고 가려진 부분($x_m$)이 무엇일지 예측해야 합니다
    * 효과
        * 가려진 부분을 맞추려면 모델은 '이것이 기린이다(객체)'라는 정보뿐만 아니라 '이 부분에는 이런 무늬가 와야 한다(패턴)'는 정보까지 모두 학습
        * 자연스럽게 계층적 특징을 배우게 됩니다.

#### 마스킹 손실함수 ($\mathcal{L}_M$)

$$\mathcal{L}_M = \mathbb{E}_{p_{data}(x), p(x_v|x), p(x_m|x), q_{\phi}(z|x_v)} [-\log p_{\theta}(x_m|z)]$$

* $\mathbb{E}$ (Expectation): 평균
* $p_{data}(x)$: 원본 이미지 데이터의 분포
* $p(x_v|x)$, $p(x_m|x)$: 원본 이미지 $x$를 보이는 패치( $x_v$ )와 가려진 패치( $x_m$ )로 나누는 과정
    * (예: 이미지의 60%를 가림)
* $q_{\phi}(z|x_v)$: 인코더(Encoder)
    * 전체 이미지가 아닌 보이는 부분( $x_v$ )만을 입력받아 잠재 변수 $z$의 분포를 예측
* $-\log p_{\theta}(x_m|z)$: 디코더(Decoder)의 예측 오차
    * 잠재 변수 $z$를 보고 가려진 부분 $x_m$을 예측했을 때의 음의 로그 우도(Negative Log-Likelihood)입니다.
    * 이 값이 작을수록 예측을 잘한 것
* 핵심 의미: "보이는 조각( $x_v$ )만으로 압축된 정보( $z$ )를 줄 테니, 나머지 가려진 조각( $x_m$ )이 무엇일지 예측해 봐라.

#### 잠재 공간 규제 함수 ($\mathcal{L}_{reg}$)

$$\mathcal{L}_{reg} = \mathbb{E}_{p_{data}(x), p(x_v|x)} [D_{KL}(q_{\phi}(z|x_v) || p(z))]$$

* 일반적인 VAE는 전체 이미지 $x$를 보고 $q(z|x)$를 규제하지만
* VMAE는 마스킹된 예측(Masked prediction)을 수행하므로 보이는 부분 $x_v$를 조건으로 하는 분포 $q(z|x_v)$를 규제하도록 식이 수정



---


