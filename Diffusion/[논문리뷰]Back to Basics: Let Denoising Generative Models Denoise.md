# Back to Basics: Let Denoising Generative Models Denoise (JiT - Just Image Transformers)
저자 : Tianhong Li, Kaiming He, MIT

출간 : arXiv 2025. (CVPR 2026, ICLR 2026 등??)

논문 : [PDF](https://arxiv.org/pdf/2511.13720)

<p align = 'center'>
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/38b2f741-47a4-4368-8c04-55b23e3883d6" />
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
<img width="800" height="200" alt="image" src="https://github.com/user-attachments/assets/a6cdd957-690a-46c5-beb3-72aa007a7821" />
</p>

* Prediction Space : 신경망이 $x, \epsilon, v$ 중 무엇을 직접 출력할 것인가?
* Loss Space : 정답과의 차이를 $x, \epsilon, v$ 중 어떤 공간에서 계산할 것인가?

<p align = 'center'>
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/17db4679-b7d2-46cf-94bd-a70ae682cae8" />
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

<p align = 'center'>
<img width="684" height="582" alt="image" src="https://github.com/user-attachments/assets/519aa81b-eefc-491b-a32e-2465c8f58230" />
</p>


#### Just Image Transformers

* 구조: 기본적인 Vision Transformer (ViT) 구조를 그대로 따릅니다
    * 이미지를 $p \times p$ 크기의 패치로 자릅니다
        * (예: $16 \times 16$ 픽셀)
    * 이 패치들을 1차원 벡터로 펼쳐서(Linear Embedding) 트랜스포머에 넣습니다
    * 트랜스포머 블록을 통과한 후, 다시 선형 층(Linear Predictor)을 통해 깨끗한 이미지 패치($x$)를 예측합니다
* 특징: 토크나이저(VAE), 사전 훈련, 추가적인 손실 함수 등이 전혀 없는 자립형(Self-contained) 모델입니다

#### What to Predict by the Network?

<p align = 'center'>
<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/3f122669-9974-4fa6-b57f-cb8ec48fc8c2" />
</p>

* ImageNet 256x256 해상도에서 패치 크기를 16으로 설정(패치 차원 = 768)하고 실험을 진행했
* 결과 ( $x$ vs $\epsilon$ vs $v$ )
    * $x$-prediction (데이터 예측): 매우 잘 작동
        * 손실 함수를 어떤 기준(Loss Space)으로 설정하든 상관없이 안정적
     * $\epsilon$ / $v$-prediction (노이즈/속도 예측): 처참하게 실패(Catastrophic failure)
         * 네트워크가 고차원 패치 속에 섞인 노이즈 정보를 감당하지 못하기 때문
* 비교 (저차원일 때)
    * 해상도를 낮추어 패치 차원이 작을 때($4 \times 4$ 패치, 48차원)는 $\epsilon$이나 $v$를 예측해도 잘 작동
    * 즉, 기존 연구들이 노이즈 예측의 문제를 몰랐던 건, 주로 저차원 잠재 공간(Latent Space)이나 작은 이미지에서만 실험했기 때문

#### Counter-intuitive Findings
<p align = 'center'>
<img width="600" height="550" alt="image" src="https://github.com/user-attachments/assets/6bd0050b-a623-4fca-a94f-6a74a260e819" />
</p>

* 통념을 깨는 두 가지 중요한 발견
* 네트워크 너비(Hidden Size)를 키울 필요가 없다
    * 일반적으로 입력 데이터의 차원이 크면 네트워크도 커야 한다고 생각
    * 하지만 $x$-prediction을 사용하면, 패치 차원(예: 3072, 12288)이 네트워크의 히든 사이즈(예: 768)보다 훨씬 커도 문제없이 작동
    * 신경망이 노이즈를 다 외우는 게 아니라, 저차원 매니폴드 구조만 학습하면 되기 때문
 * 병목(Bottleneck)이 오히려 도움이 된다
     * 입력 패치를 트랜스포머에 넣을 때 차원을 줄이는 병목 층(Bottleneck Linear Layer)을 사용했더니 성능이 더 좋아졌다
     * 병목 구조가 노이즈를 걸러내고 유용한 정보만 통과시키는 역할을 하여 매니폴드 학습을 돕기 때문

#### Algorithm
<p align = 'center'>
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/0b1db9a8-7c76-48aa-9349-74f5ad2895c1" />
</p>


#### “Just Advanced” Transformers
<p align = 'center'>
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/fc35eb2e-6ea7-4065-aced-e3c254c8b7a9" />
</p>

* 최신 기술 적용 (Just Advanced): 언어 모델(LLM) 등에서 검증된 최신 기법들(SwiGLU, RMSNorm, ROPE 등)을 적용하고, In-context conditioning(클래스 토큰을 여러 개 붙이는 방식)을 도입하여 성능을 극대화

---

## 5. Comparisons

#### High-resolution generation on pixels
* 실험: 저자들은 ImageNet 512x512 및 1024x1024 해상도에서 실험을 진행했습니다
    * 패치 크기 전략: 해상도가 커질 때 시퀀스 길이를 늘리는 대신, 패치 크기를 키우는 전략을 썼습니다
    * 1024 해상도에서는 패치 크기가 $64 \times 64$가 되며, 패치 하나의 차원이 무려 12,288차원이 됩니다
* 결과: 기존 모델이라면 히든 사이즈(Hidden units)가 입력 차원보다 작아 정보 손실로 실패했겠지만, $x$-prediction을 사용하는 JiT는 이 고차원 입력을 무리 없이 처리하며 성공적으로 이미지를 생성

#### Scalability
* 모델 크기: Base(B), Large(L), Huge(H), Giga(G) 사이즈로 모델을 키워가며 실험했습니다.
* 성능 향상: 모델 사이즈가 커질수록 FID(이미지 품질 지표, 낮을수록 좋음)가 꾸준히 개선되었습니다. 특히 JiT-G 모델은 512 해상도에서 1.78이라는 매우 낮은 FID를 달성했습니다.
* 특이점: 모델이 커질수록 256 해상도와 512 해상도 간의 성능 격차가 줄어들었습니다. 이는 큰 모델일수록 고해상도 생성이라는 어려운 작업을 더 잘 학습한다는 것을 의미합니다.


#### Reference results from previous works

<p align = 'center'>
<img width="400" height="600" alt="image" src="https://github.com/user-attachments/assets/991cb119-92aa-47be-942e-714f0bfb9463" />
</p>

<p align = 'center'>
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/3d3e246f-8fee-4045-9a58-4f58d3835911" />
</p>

* 비교 대상
    * Latent Diffusion (예: DiT, SiT): 이미지를 압축하는 VAE(토크나이저)를 사용하고, 추가적인 학습 데이터나 손실 함수를 사용하는 모델들 .
    * Pixel Diffusion (예: SiD, PixelFlow): 픽셀 공간에서 작동하지만 복잡한 구조를 가진 모델들
* JiT의 성과
    * 토크나이저(VAE), 사전 훈련(Pre-training), 지각 손실(Perceptual loss, VGG 사용), 적대적 손실(Adversarial loss) 등 외부의 도움을 전혀 받지 않고도(Self-contained) 경쟁력 있는 결과를 냈습니다.
    * ImageNet 512x512에서 FID 1.78을 기록하며, VAE를 쓰는 모델들이나 복잡한 픽셀 모델들과 대등한 성능을 보였습니다.


---

## 6. Discussion and Conclusion

* 미래 전망: 과학 분야로의 확장 가능성
    * 토크나이저의 부재: 단백질 구조, 분자 배열, 기상 데이터(Weather)와 같은 과학 데이터들은 이미지와 달리 효과적인 토크나이저(VAE 등)를 설계하기가 매우 어렵습니다.
    * 새로운 패러다임: 이 논문이 제안한 방식(JiT)은 복잡한 토크나이저 없이 '원시 데이터(Raw Data)'에 직접 트랜스포머를 적용할 수 있음을 증명했습니다.
    * 비전: 저자들은 이 범용적인 "Diffusion + Transformer" 패러다임이 고차원 자연 데이터를 다루는 다양한 과학 분야의 기초 모델(Foundation)이 되기를 기대합니다.

---

## 의의
1. "화질의 유리천장"을 깨뜨림 (Information Bottleneck 제거)
* 기존 LDM(Stable Diffusion 등)은 아무리 Diffusion 모델을 잘 학습시켜도, 마지막에 이미지를 복원하는 VAE(Decoder)의 성능 이상으로 화질을 높일 수 없었습니다.

2. "압축"에서 "이해"로의 전환 (Semantic Alignment)
* VAE의 Latent Space는 단순히 "이미지를 작게 줄이는 것"에 최적화되어 있을 뿐, 그 이미지가 "무엇인지(강아지인지 고양이인지)"를 이해하도록 설계되지는 않았습니다.

3. 학습 파이프라인의 단순화 (End-to-End Training)
* 기존의 문제
    * VAE를 먼저 학습시킨다. (이미지 압축기)
    * 그 다음 Diffusion 모델을 학습시킨다. (생성기)

---

## Appendix

### Manifold
* 매니폴드(Manifold)'는 복잡한 수학적 정의보다는 "데이터가 존재하는 숨겨진 규칙이나 구조"를 설명하기 위한 개념으로 이해
* 비유: 아주 거대한 3차원 방(고차원 공간)이 있다고 상상해 보세요. 이 방 안의 공기 분자처럼 무수히 많은 점을 찍을 수 있음
* 데이터의 위치: 하지만 '진짜 강아지 사진'이나 '진짜 풍경 사진'은 이 방 아무 데나 무작위로 존재하지 않습니다
    * 진짜 데이터들은 방 한구석에 얇은 종이 한 장(저차원 매니폴드) 위에만 모여 있음

### In-Context Class tokens

* 고급 클래스 조건화(Conditioning) 기법 - 정보를 시퀀스 안에 직접 넣자
* 보통 트랜스포머 기반 디퓨전(DiT) 모델은 클래스 정보를 adaLN-Zero라는 방식을 통해 정규화(Normalization) 레이어에 주입합니다. 즉, 레이어의 통계치를 살짝 조절하는 간접적인 방식
* "In-context conditioning"은 원래의 비전 트랜스포머(ViT)처럼 클래스 정보를 하나의 '토큰(단어)'으로 만들어서 이미지 패치들의 시퀀스 앞단에 직접 붙여버리는 방식
* 이 논문의 독창적 접근: "하나로는 부족하다, 32개를 넣자"
    * 이 32개의 토큰은 모두 같은 클래스 정보를 담고 있지만, 서로 다른 **위치 임베딩(Positional Embedding)**을 가집니다. 즉, 모델 입장에서는 "강아지"라는 힌트를 32번 반복해서 아주 강력하게 받는 셈
    * 중간 투입 (Late Start Block): 이 토큰들을 맨 처음 입력단(Input)부터 넣지 않고, 트랜스포머의 중간 블록부터 끼워 넣습니다.

<p align = 'center'>
<img width="450" height="500" alt="image" src="https://github.com/user-attachments/assets/158292c1-3d92-4620-8ce2-3868d40cbe0f" />
</p>

* 예를 들어 JiT-H 모델의 경우 32개의 블록 중 10번째 블록부터 이 토큰들을 추가합니다
* 병행 사용: 이 논문의 모델은 기존의 adaLN-Zero 방식과 이 In-context Class Tokens 방식을 동시에 사용

---
