# PixelDiT: Pixel Diffusion Transformers for Image Generation

저자 : Yongsheng Yu1,2 * Wei Xiong1 † Weili Nie1 Yichen Sheng1 Shiqiu Liu1 Jiebo Luo2 / 1 NVIDIA, 2 University of Rochester, † Project Lead and Main Advising

출간 : CVPR(	Computer Vision and Pattern Recognition), 2025

논문 : [PDF](https://arxiv.org/pdf/2511.20645)

---

## 1. Introduction

#### 1. 연구 배경: 잠재 확산 모델(LDMs)의 구조적 한계

* 목표 불일치: 확산 모델이 별도로 학습된 오토인코더(Autoencoder)에 의존하는데, 오토인코더의 '재구성 목표'와 생성 모델의 '생성 목표'가 완전히 일치하지 않습니다.
* 손실 재구성(Lossy Reconstruction): 오토인코더의 압축 과정에서 고주파 디테일(예: 텍스트, 미세한 질감)이 손실됩니다. 이로 인해 확산 모델 성능이 아무리 좋아도 최종 이미지 품질에 한계가 생깁니다


#### 2. 문제 제기: 픽셀 공간 확산의 어려움

* 오토인코더 없는 픽셀 공간 확산을 다시 도입하려 했으나
* 픽셀 모델링(Pixel Modeling)이라는 핵심 난관에 봉착

* 트레이드오프(Trade-off)
    * 큰 패치를 사용하면 연산은 줄지만 디테일 표현이 약해집니다.
    * 작은 패치를 사용하면 디테일은 살지만, 연산 비용이 제곱(Quadratic)으로 증가하여 학습이 너무 느려집니다.
* 즉, 글로벌한 의미(Semantics)와 픽셀 단위의 세밀한 업데이트를 동시에 효율적으로 처리할 메커니즘이 부족했습니다


#### 3. 해결책: PixelDiT (Dual-level Architecture)

* 이중 레벨 설계 (Dual-level Design): 이미지의 의미 학습과 픽셀 학습을 분리했습니다.
    * Patch-level DiT: 큰 패치 단위를 사용하여 이미지의 전반적인 구조와 내용을 학습합니다.
    * Pixel-level DiT: 픽셀 단위의 미세한 텍스처를 정교하게 다듬습니다.
* 핵심 기술
    * 이를 효율적으로 구현하기 위해 픽셀 단위 AdaLN 변조(Pixel-wise AdaLN modulation)와
    * 픽셀 토큰 압축(Pixel token compaction) 기술을 도입했습니다.

#### 4. 주요 성과 및 기여

* 압도적 성능: ImageNet 256x256에서 FID 1.61을 기록하며 기존 픽셀 생성 모델들을 크게 앞섰습니다
* 확장성: 텍스트-이미지 생성(Text-to-Image)으로 확장하여 $1024^2$ 해상도(메가픽셀 단위) 학습에 성공했습니다
* 편집 용이성: VAE의 압축 손실이 없기 때문에, 이미지 편집 시 배경의 작은 텍스트나 디테일이 뭉개지지 않고 잘 보존됩니다

---

## 2. Related Work

<p align = 'center'>
<img width="813" height="330" alt="image" src="https://github.com/user-attachments/assets/b73d4b6e-ec33-42ea-bfd2-092780a4f356" />
</p>

### 2.1. Latent Diffusion Models with Autoencoders

#### 1. LDM의 표준 방식과 장점
* 작동 원리: LDM은 오토인코더(Autoencoder)를 사용해 이미지를 압축된 표현 공간(Latent space)으로 변환한 뒤, 그 공간에서 노이즈 제거(Denoising)를 수행합니다.
* 이점: 이 방식은 연산량과 메모리 사용량을 크게 절약해 주며, 한정된 자원 내에서도 더 큰 백본 모델을 사용해 고해상도 학습을 가능하게 합니다.
* 일반적 구조: 대부분의 LDM은 재구성 정확도와 압축률 사이의 균형을 맞추기 위해 변분 오토인코더(VAE)를 사용합니다.


#### 2. 왜 다시 픽셀 공간인가? (LDM의 한계)

* 효율성에도 불구하고 LDM은 오토인코더에 의존함으로 인해 발생하는 명확한 병목 현상(Bottleneck)을 가지고 있습니다.


* 재구성 병목 (Reconstruction Bottleneck)
    * 생성되는 샘플의 충실도(Fidelity)가 오토인코더의 성능에 의해 제한됩니다.
    * 과감한 압축 과정에서 고주파 디테일이나 미세한 구조가 제거되는 경향이 있습니다.


* 학습 비용 증가
    * 대규모 오토인코더를 사전에 학습하거나 함께 학습시키는 것은 픽셀 공간 학습에 비해 추가적인 데이터와 연산 비용을 발생시킵니다.


* 목표 불일치 (Objective Misalignment)
    * 오토인코더의 '재구성 목표'와 확산 모델의 '생성 목표'가 완전히 일치하지 않습니다.
    * 이로 인해 잠재 공간 내에서 분포 변화(예: 텍스처가 뭉개지거나 색상이 변함)가 발생하고, 확산 모델이 이를 보정해야 하는 부담을 안게 됩니다.


* 추론 지연
    * 샘플링 단계에서 잠재 표현을 다시 이미지로 변환(Decoding)하는 과정이 추가적인 시간 비용을 발생시킵니다.


### 2.2. Pixel-Space Diffusion Models

#### 1. 초기 역사와 핵심 난관초기 연구

* 픽셀 공간 확산 모델(Diffusion)은 LDM보다 먼저 연구되었으며, 픽셀 공간에서 직접 노이즈를 제거하여 고화질 이미지를 생성할 수 있음을 입증했습니다

* 핵심 난관 (계산 비용): 하지만 이미지 해상도가 높아질수록 연산량과 메모리 비용이 제곱(Quadratic)으로 증가하는 문제가 있어, 메가픽셀(Megapixel) 단위의 고해상도 이미지를 엔드투엔드(End-to-end)로 학습하는 것은 비용적으로 거의 불가능했습니다
    * Quadratic 이유: Self-Attention


#### 2. 최근의 시도들 (PixelDiT 이전의 접근법)

##### 효율성을 위한 최근의 접근 방식
* JetFormer: 원본 픽셀과 텍스트에 대해 자기회귀(Autoregressive) 방식을 적용했습니다
* Simple Diffusion: 스킵 연결(Skip connections)이 있는 효율적인 합성곱 신경망(CNN) 구조를 제안했습니다
* Fractal Generative Models: 장거리 구조 학습을 위해 프랙탈 설계를 도입했습니다
* PixelFlow & PixNerd: 계층적 흐름(Hierarchical flow) 모델이나 경량화된 신경 필드(Neural field) 레이어를 사용하여 효율성을 높였습니다

##### 동시대 연구 (Concurrent Works) - 고차원 픽셀 데이터를 다루기 위해

* EPG: 자기 지도 사전 학습(Self-supervised pre-training)과 생성 미세 조정(Generative fine-tuning)을 연결하는 2단계 프레임워크를 채택했습니다
* FARMER: 고차원 픽셀 처리를 위해 자기회귀 모델링과 정규화 흐름(Normalizing flows)을 결합했습니다
* JiT: 일반적인 트랜스포머(Plain Transformers)가 고차원 데이터를 효율적으로 모델링할 수 있음을 보여주었습니다


##### PixelDiT의 차별점

* 순수 트랜스포머 기반
* 고해상도 학습: 별도의 단계나 우회적인 방법 없이, $1024^2$ 해상도에서 직접 엔드투엔드 학습이 가능한 픽셀 공간 확산 모델을 제시

---

## 3. Method 

### 1. 이중 레벨 DiT 아키텍처 (Dual-level DiT Architecture)

#### 패치 레벨 경로 (Patch-level Pathway)

* 이미지를 겹치지 않는 $p \times p$ (예: $16 \times 16$) 패치로 나누어 처리
* 긴 범위의 어텐션(Attention)을 수행하여 이미지의 전반적인 레이아웃과 내용을 캡처
* 생성된 시맨틱 토큰(Semantic Tokens)은 픽셀 레벨 경로에 컨텍스트 정보를 제공하는 역할

#### 픽셀 토큰 압축 (Pixel Token Compaction)

<p align = 'center'>
<img width="200" height="100" alt="image" src="https://github.com/user-attachments/assets/fd083910-267e-4192-9d3c-9e4b0bd22a50" />
</p>

* 문제점: 모든 픽셀( $H \times W$ )에 대해 직접 어텐션을 수행하면 계산 비용이 너무 높습니다(메모리 및 연산량 폭증)
* 해결책: 어텐션 연산을 수행하기 직전에 $p^2$개의 픽셀 토큰을 1개의 토큰으로 선형 압축(Linear Compress)
    * 하나의 패치 안에는 $p \times p$개(예: 256개)의 픽셀 토큰이 존재
    *  $p^2$개의 픽셀 토큰들을 단순히 평균 내는 것이 아니라, **학습 가능한 선형 변환(Linear Projection $\mathcal{C}$)**을 통해 하나의 '패치 토큰'으로 압축
* 효과: 어텐션 시퀀스 길이를 $p^2$배(예: 256배) 줄여 계산을 효율적으로 만든 뒤, 어텐션 후 다시 원래 픽셀 수로 확장(Expand)합니다
* 이는 VAE의 손실 압축과는 달리, 어텐션 연산 효율성만을 위해 일시적으로 압축하는 것


##### Patch Embedding

$$s_0 = W_{patch} x_{patch} \quad \quad (1)$$

* 설명
    * 입력 이미지를 겹치지 않는 $p \times p$ 크기의 패치로 나눈 뒤($x_{patch}$)
    * 이를 선형 투영(Linear Projection, $W_{patch}$)하여 차원 $D$를 가진 초기 시맨틱 토큰(Semantic Token) $s_0$를 생성

##### Global Conditioning Vector

* 무엇을(Class)' 그리고 '어느 시점에(Timestep)' 생성해야 하는지 알려주는 제어 신호

$$c = \text{SiLU}(W_t t + W_y y + b) \quad \quad (2)$$ 

* 간 단계(timestep) 임베딩 $t$와 클래스 레이블 $y$를 각각 선형 변환한 후 더하고, 활성화 함수 SiLU를 통과시켜 글로벌 컨디셔닝 벡터 $c$ 를 만듦

* 벡터 $c$는 이후 모든 블록에서 AdaLN(Adaptive Layer Normalization)을 통해 이미지 생성 과정을 제어하는 데 사용

##### DiT 블록 내부 연산 (AdaLN 메커니즘)

$$\bar{s}_i = \text{RMSNorm}(s_i) \quad \quad (3)$$ 

##### 어텐션 (Attention with AdaLN)

$$\bar{s}_i = s_i + \alpha_1(c) \cdot \text{Attn}(\gamma_1(c) \cdot \bar{s}_i + \beta_1(c); \text{ROPE}) \quad \quad (4)$$

* 정규화된 토큰에 대해 글로벌 벡터 $c$에서 유도된 스케일( $\gamma_1$ )과 시프트( $\beta_1$ ) 파라미터를 적용한 후, 자기 어텐션(Self-Attention)을 수행
*  2D ROPE(Rotary Positional Embedding)가 사용
*  마지막으로 게이팅 파라미터( $\alpha_1$ )를 곱해 잔차 연결(Residual Connection)을 수행

##### MLP (Feed-Forward Network with AdaLN)

$$s_{i+1} = \hat{s}_i + \alpha_2(c) \odot \text{MLP}(\gamma_2(c) \odot \text{RMSNorm}(\hat{s}_i) + \beta_2(c)) \quad \quad (5)$$

* 어텐션을 거친 결과( $\hat{s}_i$ )를 다시 정규화하고, 마찬가지로 벡터 $c$에서 유도된 파라미터( $\gamma_2, \beta_2$ )로 변조(Modulation)한 뒤 MLP를 통과시킵니다.
* 최종적으로 게이트( $\alpha_2$ )를 통해 다음 블록의 입력( $s_{i+1}$ )을 만듦

#### 픽셀 레벨 경로 (Pixel-level Pathway)

* 개별 픽셀 단위의 데이터를 처리하여 텍스처와 디테일을 정교
* 계산 효율성을 위해 패치 레벨보다 훨씬 작은 은닉 차원($D_{pix} \ll D$, 예: 16)을 사용
* 이 경로는 PiT(Pixel Transformer) 블록


### 2. 픽셀 트랜스포머 블록 (Pixel Transformer Block)

#### 픽셀별 AdaLN 변조 (Pixel-wise AdaLN Modulation)

<p align = 'center'>
<img width="400" height="500" alt="image" src="https://github.com/user-attachments/assets/8048547a-0db9-4482-bb9b-774b7723c0af" />
</p>

* 문제점: 기존 방식(Patch-wise)은 패치 내의 모든 픽셀( $p^2$ 개)에 동일한 조건(파라미터)을 적용하므로, 픽셀마다 다른 세밀한 변화를 표현하기 어렵습니다.

* 해결책: 패치 레벨에서 넘어온 시맨틱 토큰을 MLP를 통해 확장하여, 모든 개별 픽셀마다 서로 다른 변조 파라미터(scale, shift 등)를 생성해 적용합니다. 이를 통해 글로벌 맥락에 맞으면서도 픽셀별로 특화된 업데이트가 가능해집니다

$$X \in \mathbb{R}^{B \times C \times H \times W} \xrightarrow{\text{reshape + linear}} \mathbb{R}^{B \times H \times W \times D_{\text{pix}}} \quad \quad (6)$$

* 설명: 배치 크기($B$), 채널($C$), 높이($H$), 너비($W$)를 가진 입력 이미지 $X$를 받아, 각 픽셀 하나하나를 독립적인 토큰으로 변환합니다
    * 이를 위해 선형 레이어(Linear layer)를 통과시켜, 각 픽셀을 $D_{pix}$라는 작은 은닉 차원(예: 16)을 가진 벡터로 만듭니다
* 의미: 패치 단위로 뭉뚱그려 처리하는 것이 아니라, 모든 픽셀을 개별적으로 처리할 준비를 하는 단계입니다3.

$$\Theta = \Phi(s_{cond}) \in \mathbb{R}^{(B \cdot L) \times p^2 \times 6D_{pix}} \quad \quad (7)$$


### 3. 텍스트-이미지 생성을 위한 확장 (Text-to-Image Generation)

* MM-DiT 블록 사용: 텍스트 조건을 처리하기 위해 패치 레벨 경로에 MM-DiT(Multi-Modal DiT) 블록을 적용하여 이미지와 텍스트 토큰을 융합합니다.
* 구조적 특징: 텍스트 토큰은 패치 레벨에서만 처리되며, 픽셀 레벨 경로는 텍스트를 직접 받지 않고 패치 레벨에서 처리된 시맨틱 토큰을 통해서만 정보를 전달받습니다.

#### 텍스트 인코딩은 Gemma-2 모델을 사용

* PixelDiT는 텍스트를 처리하기 위해 별도의 텍스트 인코더를 처음부터 학습시키지 않고, 사전 학습된 Gemma-2 인코더를 고정(Frozen)된 상태로 가져와 사용
* 인코딩된 텍스트 정보는 패치 레벨 경로(Patch-level Pathway)에만 주입
* MM-DiT 블록 사용: 패치 레벨 경로의 블록들을 MM-DiT(Multi-Modal DiT) 블록으로 교체하여 사용
* 이중 스트림: 이 블록 내부에서 이미지 토큰과 텍스트 토큰은 서로 다른 스트림(QKV 프로젝션)을 통해 처리되며 상호작용


### 4. 학습 목표 (Training Objectives)

* Rectified Flow: 픽셀 공간에서 Rectified Flow 방식을 채택하여 속도 매칭(velocity-matching) 손실 함수를 사용해 학습합니다.
* 표현 정렬 (Representation Alignment): 학습 안정화를 위해 패치 레벨 토큰이 사전에 학습된 DINOv2 인코더의 특징(feature)과 유사해지도록 정렬하는 보조 손실 함수를 추가했습니다.

$$\mathcal{L}_{diff} = \mathbb{E}_{t,x,\epsilon} [ \parallel f_{\theta}(x_t, t, y) - v_t \parallel_2^2 ] \quad \quad (8)$$

* $x_t$: 시간 $t$에서의 노이즈가 섞인 이미지
* $y$: 조건 (텍스트나 클래스 레이블)
* $f_{\theta}$: 학습되는 모델 (PixelDiT)
* $v_t$: 목표 속도 벡터 (노이즈 이미지에서 깨끗한 이미지로 가는 방향)

---

## 4. Experiments

<p align = 'center'>
<img width="350" height="200" alt="image" src="https://github.com/user-attachments/assets/5c29ea9b-63e7-4617-b5e9-9308047cd166" />
</p>

### 1. 실험 설정 및 모델 변형 (Model Variants)

* 모델 크기: 확장성을 입증하기 위해 Base(B), Large(L), Extra Large(XL) 세 가지 크기의 모델을 실험했습니다
    * 기본 실험 모델은 PixelDiT-XL이며, 파라미터 수는 약 797M입니다
* 데이터셋: 클래스 조건부 생성은 ImageNet ($256 \times 256$), 텍스트-이미지 생성은 내부 수집 데이터($1024^2$)를 사용했습니다.

### 2. ImageNet 클래스 조건부 이미지 생성 (Class-conditioned Image Generation)

* 성능: PixelDiT-XL은 gFID 1.61을 달성
    * 최근 픽셀 공간 모델인 PixelFlow-XL(1.98), PixNerd-XL(1.93)보다 훨씬 뛰어난 결과
* 빠른 수렴: 학습 80 에포크 만에 이미 gFID 2.36을 기록하여, 400 에포크를 학습한 기존 ADM-U 모델(FID 4.59)보다 훨씬 빠르게 고품질 이미지를 생성
* 잠재 모델과 경쟁: 별도의 사전 학습된 오토인코더(VAE) 없이도, REPA나 RAE 같은 최신 잠재 확산 모델들의 성능에 근접


### 3. 텍스트-이미지 생성 (Text-to-Image Generation)

* 고해상도 확장성: $1024 \times 1024$ 해상도에서 직접 학습 및 생성이 가능
* 벤치마크 점수
    * GenEval: 0.74 점을 기록하여 많은 잠재 확산 모델들을 능가했습니다
    * DPG-bench: 83.5 점을 기록하여 경쟁력 있는 성능을 보였습니다
 * 효율성: $1024^2$ 해상도에서 초당 0.33장의 이미지를 생성할 수 있어, 픽셀 공간에서 직접 연산함에도 불구하고 잠재 모델들과 비슷한 수준의 처리량을 보였습니다

### 4. 이미지 편집에서의 디테일 보존 (Detail Preservation in Image Editing)

### 5. 어블레이션 연구 (Ablation Study)

<p align = 'center'>
<img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/dfe3d6ec-2729-4c42-84f6-16b276d114bc" />
</p>

* 구성 요소별 기여도 (Table 4)
    * Vanilla DiT (기본): 픽셀 모델링 없이 패치만 사용하면 FID 9.84로 성능이 매우 낮습니다
    * 토큰 압축 (Compaction): 이 기능이 없으면 메모리 초과(OOM)로 학습이 불가능합니다. 압축을 도입하면 학습이 가능해지고 FID 3.50을 달성합니다
    * Pixel-wise AdaLN: 픽셀별로 다른 제어 신호를 주입하면 FID가 2.36(80 에포크)으로 대폭 향상됩니다
* 패치 크기 및 모델 크기
    * 작은 모델(Base)에서는 패치 크기를 줄이는 것( $p=4$ )이 성능에 큰 도움이 되지만, 큰 모델(XL)에서는 $p=16$만으로도 충분히 좋은 성능과 효율성을 냅니다
    * 모델 크기를 키울수록(B $\rightarrow$ L $\rightarrow$ XL) 수렴 속도가 빨라지고 최종 품질이 좋아집니다

---

## Appendix

### 자기회귀 생성 모델(Autoregressive Generative Model)

* 데이터의 순서(Sequence)에 의존하여, "이전의 모든 데이터를 바탕으로 바로 다음에 올 데이터 하나를 예측"하는 방식

### 프랙탈 설계(Fractal Design)

* 생성형 AI 모델에서 복잡한 데이터의 전체 구조와 세부 구조가 서로 유사한 형태(자기 유사성, Self-similarity)를 띠도록 모델을 설계하는 방식을 말합니다.

* 사용자가 언급한 "장거리 구조 학습(Long-range structure learning)"의 맥락에서 이 설계가 중요한 이유는, 데이터의 아주 먼 과거와 현재, 혹은 전체와 부분 사이의 관계를 효율적으로 파악하기 위해서입니다.

---
