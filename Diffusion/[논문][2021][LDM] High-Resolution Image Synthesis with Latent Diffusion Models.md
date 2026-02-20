# High-Resolution Image Synthesis with Latent Diffusion Models
저자 : Robin Rombach1 * Andreas Blattmann1 ∗ Dominik Lorenz1 Patrick Esser Bjorn Ommer

출간 : CVPR 2022

논문 : [PDF](https://arxiv.org/pdf/2112.10752)

---
<p align = 'center'>
<img width="541" height="343" alt="image" src="https://github.com/user-attachments/assets/ad8d7d8e-0853-4090-923f-8b865b76701a" />
</p>


### 0. Abstract
1. 핵심 문제: 높은 계산 비용
  1-1. 학습 비용: 강력한 DM을 최적화하는 데 수백 GPU 일(days)이 소요됩니다.
  1-2. 추론 비용: 순차적인 평가 과정 때문에 추론(인퍼런스) 비용이 매우 비쌈

2. 제안된 해결책: 잠재 확산 모델 (Latent Diffusion Models, LDMs) 
   2-1. 이미 학습된 강력한 오토인코더(autoencoders)의 **잠재 공간(latent space)**에서 확산 모델을 적용하는 방법을 제안

3. 주요 기술: 크로스 어텐션 (Cross-Attention)

---

### 1. Introduction

1. 컴퓨팅 비용의 문제
2. GAN의 한계: GAN은 좋은 결과를 보여주지만, 데이터의 변동성이 제한적이며, 적대적 학습 과정이 복잡한 멀티모달(multi-modal) 분포로 확장하기 어렵다는 한계
3. 잠재 공간(Latent Space)으로의 전환

    3-1. 지각적 압축(Perceptual Compression)

        3-1-1. 오토인코더 학습
   
    3-2. 의미적 압축(Semantic Compression)
   
        3-2-1. 잠재 공간에서 확산 모델(DM)을 학습


---
### 2. Related Work

**1. 이미지 합성을 위한 생성 모델 (Generative Models for Image Synthesis)**

| 모델 유형 | 장점 | 단점 (한계) |
| :--- | :--- | :--- |
| **GANs**<br>(Generative Adversarial Networks) | • 고해상도 이미지의 효율적 샘플링 가능<br>• 높은 지각적(perceptual) 품질 | • 학습이 불안정하고 최적화가 어려움<br>• 데이터의 전체 분포를 포착하지 못하는 모드 붕괴(Mode-collapse) 발생 |
| **VAEs & Flow-based Models**<br>(Variational Autoencoders) | • 효율적인 고해상도 합성 가능<br>• 학습 과정이 안정적이고 데이터 밀도 추정에 강점 | • 샘플의 품질이 GAN에 비해 낮음 (흐릿한 결과 등) |
| **ARMs**<br>(Autoregressive Models) | • 강력한 데이터 밀도 추정 성능 | • 계산 비용이 매우 높음<br>• 순차적 샘플링 과정 때문에 저해상도 이미지에 국한됨 |
| **Diffusion Models**<br>(DMs) | • 데이터 밀도 추정과 샘플 품질 모두에서 SOTA(최고 성능) 달성<br>• 모드 붕괴가 없고 학습이 안정적임 | • 픽셀 공간에서 작동하므로 감지 불가능한 세부 사항까지 모델링하느라 비용 낭비<br>• 추론 속도가 느리고 학습 비용이 매우 높음 (높은 GPU 소모) |

**2. 2단계 이미지 합성 (Two-Stage Image Synthesis)**

* VQ-VAEs & VQGANs
    * 1단계로 이미지를 이산적(discrete)인 잠재 공간으로 압축하는 오토인코더를 학습하고
    * 2단계로 그 잠재 공간을 자기회귀(Autoregressive) 모델(트랜스포머 등)로 학습합니다
    * 한계: 자기회귀 모델을 학습시키기 위해 과도한 압축이 필요합니다.
    * 압축률이 높으면 정보 손실이 발생하고, 압축률을 낮추면 연산 비용이 급증하는 딜레마가 있습니다.


**3. LDMs의 차별점**

* 트랜스포머 대신 컨볼루션(Convolution) 사용
    * 기존 VQGAN 등은 2단계에서 트랜스포머(Attention 기반)를 사용하여 잠재 공간을 학습
    * 이는 메모리를 많이 차지하여 잠재 공간을 크게 줄여야 함
    * LDMs는 2단계에서 컨볼루션 기반의 UNet을 사용
    * 이는 고차원 데이터에 훨씬 더 유연하게 확장되므로, 과도한 압축을 할 필요가 없습니다.


* 최적의 균형점 도달
    * 압축을 덜 해도 되기 때문에 이미지의 디테일을 잃지 않으면서도(High Fidelity), 픽셀 공간보다는 훨씬 효율적인 연산이 가능
    * 즉, 압축률과 생성 능력 사이의 최적점(Sweet spot)을 찾을 수 있습니다.


**4. VQ-VAE / VQGAN**

* VQ-VAE나 VQGAN은 이미지를 압축하여 이산적인 토큰(discrete tokens)의 배열로 변환합니다. 
* 자연어 처리(NLP)에서 강력한 성능을 입증한 트랜스포머(Transformer)를 사용하여 이미지 토큰의 순서를 예측(Auto-regressive)하도록 학습

---

### 3. Method

1. 지각적 이미지 압축(Perceptual Image Compression)
2. 잠재 확산 모델(Latent Diffusion Models)
3. 컨디셔닝 메커니즘(Conditioning Mechanisms)

#### 1. 지각적 이미지 압축(Perceptual Image Compression)

* 픽셀 공간(pixel space)과 지각적으로 동등하지만 차원이 낮은 잠재 공간(latent space)을 학습하기 위해 오토인코더(autoencoder)를 훈련

<p align = 'center'>
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/db1526c7-3951-47bb-8935-dda80c5fa73f" />
</p>

* 구조: 인코더 $\mathcal{E}$는 이미지 $x$를 잠재 표현 $z = \mathcal{E}(x)$로 인코딩하고, 디코더 $\mathcal{D}$는 잠재 표현에서 이미지를 복원
* 다운샘플링 인자 (Downsampling Factor $f$): 인코더는 이미지를 $f$ 배만큼 다운샘플링합니다 ($f = H/h = W/w$). 연구진은 $f=2^m$ (예: 4, 8, 16 등)에 대해 실험

* 정규화 (Regularization): 잠재 공간의 분산이 너무 커지는 것을 막기 위해 두 가지 정규화 방식을 실험했습니다.
    * KL-reg: VAE처럼 표준 정규 분포에 가깝게 만드는 방식.
    * VQ-reg: VQGAN처럼 디코더 내에 벡터 양자화(vector quantization) 레이어를 사용하는 방식.

#### 2. 잠재 확산 모델 (Latent Diffusion Models)
* 확산 모델(Diffusion Model)

* 효율성: 의미론적으로 중요한 정보에 집중할 수 있고 학습 및 계산 효율성이 크게 향상
* 신경망 구조: 모델의 백본(backbone) $\epsilon_\theta$는 시간 조건부 UNet (time-conditional UNet)으로 구현
* 목적 함수: 모델은 노이즈가 섞인 잠재 변수 $z_t$ 로부터 노이즈를 예측하도록 학습

$$L_{LDM} := \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0,1), t} \left[ \| \epsilon - \epsilon_\theta(z_t, t) \|_2^2 \right]$$

#### 3. 컨디셔닝 메커니즘 (Conditioning Mechanisms)
* 크로스 어텐션(Cross-Attention)

<p align = 'center'>
<img width="300" height="400" alt="image" src="https://github.com/user-attachments/assets/4bc21fb5-c828-40f4-906b-94c60244dbb9" />
</p>

* 도메인 특화 인코더 ($\tau_\theta$): 텍스트와 같은 조건 입력 $y$를 중간 표현 $\tau_\theta(y)$로 변환
* 크로스 어텐션 (Cross-Attention): UNet의 중간 레이어에 어텐션 메커니즘을 적용하여 조건 정보를 주입합니다.

$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d}}) \cdot V$$

* 여기서 $Q$(Query)는 UNet의 중간 표현에서 오고, $K$(Key)와 $V$(Value)는 조건 입력 $y$에서 옵니다

* 결과: 이를 통해 이미지 인페인팅, 클래스 조건부 합성, 텍스트-이미지 생성 등 다양한 생성 작업을 수행할 수 있는 유연한 생성기

* 도메인 특화 인코더(Domain Specific Encoder, $\tau_\theta$)는 텍스트나 레이아웃 같은 비시각적(non-visual) 데이터를 확산 모델(UNet)이 이해할 수 있는 숫자 형태(vector/embedding)로 번역해 주는 '통역사' 역할


---
### 4. Experiments

#### 4.1 On Perceptual Compression Tradeoffs

1. 다운샘플링 인자 $f \in {1, 2, 4, 8, 16, 32}$를 다르게 설정, 압축률이 모델 성능에 미치는 영향을 분석
2. $f=1$ (픽셀 기반): 픽셀 공간에서 직접 학습하는 경우로, 훈련 속도가 매우 느림
3. $f=32$ (과도한 압축): 너무 많이 압축하면 정보 손실이 커져서, 일정 훈련 단계 이후에는 품질(Fidelity)이 정체
4. 결과 ($f=4 \sim 16$): 효율성과 결과물의 품질 사이에서 가장 좋은 균형
5. 특히 LDM-4와 LDM-8이 가장 우수한 결과
6. 이들은 픽셀 기반 모델보다 훨씬 낮은 FID(낮을수록 좋음) 점수를 기록하면서도 샘플 생성 속도는 훨씬 빠름


#### 4.2. Image Generation with Latent Diffusion

** LDM의 성능을 평가**

* 데이터셋: CelebA-HQ(얼굴), FFHQ, LSUN-Churches/Bedrooms.
* 성능: CelebA-HQ에서 FID 5.11을 기록하며 기존의 Likelihood 기반 모델 및 GAN 모델들을 제치고 새로운 SOTA(최고 성능)를 달성
* 특징: GAN 기반 방식보다 정밀도(Precision)와 재현율(Recall)이 전반적으로 우수하여, 데이터의 분포를 더 폭넓게 커버함을 입증

<p align = 'center'>
<img width="500" height="175" alt="image" src="https://github.com/user-attachments/assets/2b1500ec-4b84-42bd-b61e-7af9bc4c7d00" />
</p>

<p align = 'center'>
<img width="500" height="175" alt="image" src="https://github.com/user-attachments/assets/0a7eb99e-502d-47ea-9b9d-f1639c7bb902" />
</p>

<p align = 'center'>
<img width="500" height="250" alt="image" src="https://github.com/user-attachments/assets/3222a28f-8b10-4ed7-8527-ed3f75bc638e" />
</p>



#### 4.3. Conditional Latent Diffusion

* 텍스트-투-이미지 (Text-to-Image)
    * LAION-400M 데이터셋으로 학습했으며, BERT 토크나이저를 사용
    * 파라미터 수가 훨씬 적음에도 불구하고, 최신 AR(자기회귀) 모델이나 다른 확산 모델(GLIDE 등)과 대등한 성능

* 레이아웃-투-이미지 (Layout-to-Image)
    * COCO 및 OpenImages 데이터셋을 사용해 바운딩 박스 레이아웃을 기반으로 이미지를 생성
    * 이전의 최고 성능 모델들을 능가하는 결과

<p align = 'center'>
<img width="500" height="175" alt="image" src="https://github.com/user-attachments/assets/9b54f36e-de87-494c-bae5-e94a747c5af4" />
</p>

#### 4.4. Super-Resolution with Latent Diffusion

* 저해상도 이미지를 조건(Conditioning)으로 하여 LDM을 학습
* SR3(기존 SOTA 확산 모델)와 비교했을 때, FID 점수에서 더 우수한 성능
* SR3가 미세한 구조(fine structure) 생성에는 다소 유리
* LDM은 텍스처 렌더링에 강점이 있으며 추론 속도가 더 빠름

<p align = 'center'>
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/7a11b092-c3f4-4748-9857-c3874b3ee6be" />
</p>

#### 4.5. Inpainting with Latent Diffusion

* 이미지의 지워진 부분을 자연스럽게 채우는 작업
* 효율성: 픽셀 기반 모델보다 학습 및 샘플링 속도가 2.7배 이상 빠르면서도 FID 점수는 1.6배 이상 향상
* 다양성: LaMa와 같은 기존 특화 모델과 달리, LDM은 하나의 입력에 대해 다양하고 자연스러운 여러 결과물을 생성할 수 있음

---

### 5. Limitations & Societal Impact

1. 기술적 한계 (Limitations)

* 속도 문제: LDM은 픽셀 기반 확산 모델보다는 계산 요구량이 적지만, 샘플링 과정이 순차적(sequential)이기 때문에 GAN(Generative Adversarial Networks)보다는 여전히 속도가 느림
* 정밀도 한계: LDM은 오토인코더를 사용하여 이미지를 압축($f=4$ 등)한 상태에서 작동
* 비록 화질 저하는 매우 적지만, 픽셀 공간에서의 미세한 정확도(fine-grained accuracy)가 요구되는 작업에서는 이 압축 과정이 병목(bottleneck)이 될 수 있음
* 특히 연구진은 자신들의 초해상도(Super-resolution) 모델이 이러한 재구성 능력의 한계로 인해 어느 정도 제약을 받을 수 있다고 가정

2. 사회적 영향 (Societal Impact)

* 긍정적 측면 (민주화)
    * LDM은 학습 및 추론 비용을 크게 줄였기 때문에, 거대 기업뿐만 아니라 더 많은 사람들이 이 기술에 접근하고 창의적인 응용 프로그램을 만들 수 있게 돕습니다(기술의 민주화).

* 부정적 측면 (오남용)
    * 조작된 데이터나 가짜 뉴스(misinformation), 스팸을 생성하고 유포
    * 딥페이크(Deep fakes)와 같은 이미지 조작은 심각한 문제

* 데이터 프라이버시
    * 만약 동의 없이 수집된 민감한 개인 정보가 데이터에 포함되어 있다면 이는 큰 문제

* 편향(Bias) 문제
    * 딥러닝 모델은 학습 데이터에 존재하는 편향을 재생산하거나 악화시키는 경향

* 데이터를 얼마나 왜곡하는지는 여전히 중요한 연구 과제

---
### 의의

1. LDM(Latent Diffusion Models)은 계산 복잡도를 획기적으로 줄이면서도 시각적 충실도(Visual Fidelity)를 유지하여, 제한된 컴퓨팅 자원으로도 고해상도 이미지를 생성할 수 있게 만들었음
2. 크로스 어텐션(Cross-Attention) 레이어를 UNet 백본에 도입하여 텍스트, 레이아웃, 바운딩 박스 등 다양한 형태의 조건(Conditioning)을 유연하게 처리할 수 있게
3. 범용적인 오토인코더의 재사용성

---


### Appendix

#### Latent Space

압축된 데이터의 표현을 의미

예를 들어 Fully Convolutional Networks(FCN)를 사용하여 이미지를 분류하도록 모델을 훈련시키려고 한다. 아래의 모델은 인코더 부분에서 이미지의 차원을 축소한다. 우리는 이를 손실 압축(Lossy Compression)의 한 형태로 간주

<p align = 'center'>
<img width="700" height="400" alt="image" src="https://github.com/user-attachments/assets/15afa185-7edf-487d-be6f-0173d110e4dd" />
</p>

**Autoencoders 및 생성 모델(Generative Models)**
latent space에서 데이터의 '근접성'을 조작하는 딥 러닝 모델의 일반적인 유형은 identity function으로 작동하는 신경망인 오토인코더이다. 오토인코더는 입력된 내용을 출력하는 방법을 학습한다.

<p align = 'center'>
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/cf86f328-4baa-4b2c-9eec-cd2bd09c7c5b" />
</p>


#### 1. KL-reg (KL 정규화)
**VAE (Variational Autoencoder)의 원리를 차용**

* 목표: 학습된 잠재 변수 $z$의 분포가 표준 정규 분포(Standard Normal Distribution, $\mathcal{N}(0,1)$ )와 유사해지도록
* 작동 원리: 손실 함수(Loss function)에 KL-발산(Kullback-Leibler Divergence) 페널티를 추가
    * 이는 모델이 생성한 잠재 분포와 표준 정규 분포 사이의 차이를 줄이는 역할
* 특징: 이미지의 미세한 디테일을 잘 복원(Reconstruction)하기 위해, 정규화의 강도(가중치)를 매우 작게(약 $10^{-6}$) 설정
    * 결과적으로 잠재 공간은 연속적(continuous)이고 부드러운 형태를 띠게 됩니다.

<img width="800" height="350" alt="image" src="https://github.com/user-attachments/assets/558d5c74-02ca-4596-adb6-3cf86db22265" />

#### 2. VQ-reg (벡터 양자화 정규화)

**VQGAN (Vector Quantized GAN)의 구조를 활용**
* 양자화(Quantization) for Latent Space (인코더가 만드는 잠재 공간은 때로는 무질서, 범위를 넘어선다)

* 목표: 잠재 공간을 이산적(discrete)인 벡터들의 집합인 코드북(Codebook)을 사용하여 정제

* LDM에서의 독특한 적용: 이 논문에서는 양자화 레이어를 "디코더에 흡수(absorbed by the decoder)"시켰다고 설명

* 확산 모델(DM) 학습 시에는 양자화 레이어를 거치기 전의 잠재 벡터 $z$를 추출하여 사용

---

