# Scalable Diffusion Models with Transformers
저자 : William Peebles* UC Berkeley, Saining Xie New York University

출간 : 2023 CVPR

논문 : [PDF](https://arxiv.org/pdf/2212.09748)

<p align = 'center'>
<img width="1417" height="732" alt="image" src="https://github.com/user-attachments/assets/1a5e44da-1df9-4f1a-84c4-04bf89e6de11" />
</p>

---
## 0. Abstarct

* U-Net을 트랜스포머(Transformer)로 대체하였으며, 이 트랜스포머는 이미지의 잠재 패치(latent patches)를 처리하도록 설계

* 트랜스포머의 깊이(depth)나 너비(width)를 늘리거나 입력 토큰의 수를 증가시켜 Gflops를 높일수록, 이미지 품질을 나타내는 FID(Fréchet Inception Distance) 수치가 일관되게 낮아지는(좋아지는) 것을 확인

* 최고 수준의 성능 달성 (State-of-the-Art)
    * DiT-XL/2
    * 클래스 조건부(class-conditional) ImageNet $512\times512$ 및 $256\times256$ 벤치마크에서 기존의 모든 확산 모델을 능가

---

## 1. Introduction

**연구의 목표: U-Net에서 트랜스포머로**

* DiT (Diffusion Transformers): 연구진은 이를 증명하기 위해 비전 트랜스포머(ViT)의 설계를 따르는 새로운 모델 클래스인 DiT를 제안
* 픽셀 공간이 아닌, 이미지가 압축된 표현(Latent space) 내에서 작동하는 잠재 확산 모델(LDM) 프레임워크를 사용하여 효율성을 높였음
* 확산 모델을 트랜스포머로 전환함으로써, 다른 도메인(언어, 비전 등)의 확장성(scalability)과 효율성 등의 장점을 그대로 가져올 수있음


---

## 2. Related Work

* 존 확산 모델(DDPM) 내에서 트랜스포머는 주로 CLIP 임베딩 생성 등 비공간적(non-spatial) 데이터 처리에 제한적으로 사용 (예: DALL-E 2)
* 하지만 이 논문은 트랜스포머를 이미지 확산 모델의 핵심 백본(backbone)으로 사용하여 그 확장성을 연구한다는 점에서 차별화
* 아키텍처 복잡도 (Architecture Complexity)
    * Gflops(연산량)를 복잡도 분석의 기준 
    * 이미지 생성 모델에서 단순히 파라미터(parameter) 수만 비교하는 것은 이미지 해상도 등의 요소를 반영하지 못해 정확한 복잡도 지표가 될 수 없음

---

## 3. Diffusion Transformers

* 연구진은 비전 트랜스포머(ViT)의 설계를 최대한 따르면서도, 확산 모델의 특성(노이즈 및 조건 정보 처리)을 반영하기 위해 몇 가지 중요한 설계를 도입

### 3.2. Diffusion Transformer Design Space
#### 패치화 (Patchify): 입력을 토큰으로 변환

* 입력: VAE를 거친 잠재 표현 $z$ (예: $32 \times 32 \times 4$ )가 입력
* 패치 분할: 입력을 $p \times p$ 크기의 패치로 자르고, 각 패치를 선형 임베딩(embedding)하여 차원 $d$를 가진 토큰으로 만듦
* 위치 임베딩: 그 후 표준적인 ViT의 주파수 기반 위치 임베딩(sine-cosine)을 적용

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$

$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$ 

* 패치 크기( $p$ )의 중요성: 패치 크기 $p$ 를 반으로 줄이면 토큰의 수 $T$ 는 4배가 되며, 이는 전체 Gflops(연산량)를 최소 4배 증가
* 연구진은 $p=2, 4, 8$을 설계 변수로 설정했습니다

<p align = 'center'>
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/2c35c1e1-014a-4031-b429-1243e58f0ecb" />
</p>

#### DiT block design

* 타임스텝 $t$와 클래스 레이블 $c$ 같은 조건부 정보(Conditioning)를 처리

1. 인-컨텍스트 (In-context Conditioning): $t$와 $c$의 임베딩을 이미지 토큰 시퀀스에 단순히 2개의 추가 토큰으로 이어 붙임
    * ViT의 cls 토큰과 유사한 방식이며 연산량 증가가 거의 없음

2. 크로스 어텐션 (Cross-attention Block): $t$와 $c$를 이미지 토큰과 별도의 시퀀스로 처리하며, 멀티 헤드 크로스 어텐션 레이어를 추가하여 정보를
    * 연산량(Gflops)이 약 15% 증가

3. 적응형 레이어 정규화 (Adaptive Layer Norm, adaLN): GAN에서 자주 쓰이는 방식
    * 표준 레이어 정규화(Layer Norm)를 대체하여, $t$와 $c$의 임베딩 합으로부터 정규화에 필요한 스케일(scale)과 시프트(shift) 파라미터를 직접 회귀(regression)
    * 연산 효율이 가장 좋습니다

4. adaLN-Zero Block (최종 선택): adaLN 방식을 개선한 것
    * ResNet의 초기화 팁을 적용하여, 잔차 연결(residual connection) 직전에 적용되는 스케일 파라미터( $\alpha$ )도 추가로 회귀
    * $\alpha$를 0으로 초기화하면, 학습 초기에는 전체 블록이 항등 함수(identity function)처럼 동작하여 학습 안정성을 크게 높임
    * 적응형 레이어 정규화(Adaptive Layer Norm)와 제로 초기화(Zero-Initialization)를 결합

4-1. 조건부 파라미터 생성 (Modulation Parameters)
$$\text{emb} = \text{SiLU}(t_{emb} + c_{emb})$$

$$(\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2) = \text{Linear}(\text{emb})$$

4-2. 블록 내부 연산 (Block Operations)

4-2-1. 멀티 헤드 셀프 어텐션 (Multi-Head Self-Attention)

* 첫 번째 정규화와 어텐션 연산에 $\gamma_1, \beta_1, \alpha_1$을 적용

$$u = \text{LayerNorm}(x)$$

$$\hat{u} = u \cdot (1 + \gamma_1) + \beta_1$$

$$h_{attn} = \text{MHSA}(\hat{u})$$

$$x^{(1)} = x + \alpha_1 \cdot h_{attn}$$

* 참고: $(1 + \gamma_1)$을 사용하는 이유는 제로 초기화( $\gamma_1=0$ ) 상황에서 입력 신호를 보존

4-2-2. (Pointwise Feed-Forward)

* 두 번째 정규화와 MLP 연산에 $\gamma_2, \beta_2, \alpha_2$를 적용

$$v = \text{LayerNorm}(x^{(1)})$$

$$\hat{v} = v \cdot (1 + \gamma_2) + \beta_2$$

$$h_{mlp} = \text{MLP}(\hat{v})$$

$$y = x^{(1)} + \alpha_2 \cdot h_{mlp}$$

* 최종적으로 $y$가 이 블록의 출력값

* MLP = FC + NonLinear (GELU, ReLU etc) + FC
    * FC → GELU → FC

<p align = 'center'>
<img width="300" height="250" alt="image" src="https://github.com/user-attachments/assets/591109e7-b024-46dd-8068-a0f99208c712" />
</p>

#### Model size

* 표준 비전 트랜스포머(ViT)의 구성을 그대로 따르는 방식을 택함

* 레이어 수 ( $N$ ): 트랜스포머 블록을 얼마나 깊게 쌓을 것인가
* 히든 사이즈 ( $d$ ): 각 토큰이 가지는 벡터의 차원 크기
* 어텐션 헤드 수: 멀티 헤드 어텐션에서 몇 개의 헤드를 사용할 것인가

| 모델 (Model) | 레이어 수 (N) | 히든 사이즈 (d) | 헤드 수 (Heads) | 연산량 (Gflops) |
| :--- | :--- | :--- | :--- | :--- |
| DiT-S (Small) | 12 | 384 | 6 | 1.4 |
| DiT-B (Base) | 12 | 768 | 12 | 5.6 |
| DiT-L (Large) | 24 | 1024 | 16 | 19.7 |
| DiT-XL (XLarge) | 28 | 1152 | 16 | 29.1 |

#### Transformer decoder

* DiT 블록을 통과한 토큰들은 여전히 1차원 시퀀스 형태
* 이를 확산 모델의 학습 목표인 노이즈 예측(noise prediction)과 공분산 예측(diagonal covariance prediction)을 위해 원래 입력되었던 공간적 형태(spatial input)와 동일한 모양으로 복원

$$Layer Norm $\rightarrow$ Linear Layer $\rightarrow$ Reshape$$

* 표준 선형 디코더(standard linear decoder)를 사용

1. 마지막 토큰 시퀀스에 레이어 정규화(Layer Norm)를 적용
2. 각 토큰을 선형 레이어(Linear Layer)에 통과시켜 차원을 확장 (FC)
    * 각 토큰은 $p \times p \times 2C$ 크기의 텐서로 변환
    * $p$: 패치 크기 (Patch size)
    * $C$: 입력 잠재 표현(Latent)의 채널 수
    * $2C$인 이유: 모델이 노이즈와 공분산($\Sigma$) 두 가지를 동시에 예측하기 때문에 채널 수가 2배
3. 재배열 (Rearrangement / Unpatchify)
    * 디코딩된 토큰들을 원래 이미지의 공간적 위치에 맞춰 재배열
    * 예: $32 \times 32 \times 4$

---

### 4. Experimental Setup

#### Training
1. ImageNet 데이터셋을 사용하여 클래스 조건부(class-conditional) 모델을 학습
2. 해상도: $256 \times 256$ 및 $512 \times 512$ 두 가지 해상도에서 실험
3. 최적화 도구: AdamW 옵티마이저를 사용했
4. 학습률 (Learning Rate): $1 \times 10^{-4}$의 고정된 학습률을 사용
    * 특이점: 일반적인 비전 트랜스포머(ViT) 학습과 달리, 학습률 웜업(warmup)이나 복잡한 정규화 기법 없이도 매우 안정적으로 학습
5. 데이터 증강: 수평 뒤집기(horizontal flips)만 유일하게 사용
6. EMA: 생성 모델링의 관례에 따라 가중치에 대한 지수 이동 평균(Exponential Moving Average, 0.9999 decay)을 유지하여 최종 평가에 사용

#### Diffusion
* 이 모델은 잠재 확산 모델(LDM)이므로, 이미지를 압축하는 과정이 포함

1. VAE: Stable Diffusion에서 사용된 **사전 학습된 VAE(Variational Autoencoder)**를 그대로 가져와 사용
2. 압축률: 인코더는 이미지를 $1/8$ 크기로 다운샘플링합니다
    * 예: $256 \times 256 \times 3 \rightarrow 32 \times 32 \times 4$
3. 확산 파라미터: 기존 연구인 ADM(Guided Diffusion)과의 공정한 비교를 위해, $t_{max}=1000$ 선형 스케줄 등 ADM의 하이퍼파라미터를 그대로 유지

#### Evaluation Metrics

1. 기준: FID-50K를 사용하며, 250단계의 DDPM 샘플링을 거친 이미지를 평가합니다
2. 정확성: FID는 구현 방식에 따라 값이 달라질 수 있으므로, 정확한 비교를 위해 ADM의 TensorFlow 평가 스위트를 사용하여 측정
3. 보조 지표: Inception Score, sFID, Precision/Recall 등도 함께

#### Compute
* 프레임워크: JAX
* 하드웨어: TPU-v3 Pods를 사용하여 학습


---

### 5. Experiments

#### DiT Block Design
* adaLN-Zero가 가장 뛰어난 성능

#### Scaling model size and patch size
당연한 얘기

* 확장 법칙 (Scaling Laws):모델 크기 증가: 모델을 더 깊고 넓게(Depth/Width 증가) 만들수록 성능이 좋아짐
* 패치 크기 감소: 패치를 작게 만들면 토큰 수( $T$ )가 늘어나 연산량이 증가하며, 성능이 좋아짐
* Gflops의 중요성: 파라미터 수가 비슷하더라도 Gflops(연산량)가 더 높은 모델(예: 패치 사이즈를 줄인 경우)이 성능이 더 좋아짐
* Gflops와 FID 사이에는 강력한 음의 상관관계

<p align = 'center'>
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/5ce132a1-044b-469c-9039-65b3c9a0b382" />
</p>

* 패치 크기를 8에서 2로 줄인다는 것은, 이미지를 더 잘게 쪼개서 토큰의 개수( $T$ )를 늘린다는 뜻 (연산량 증가)
* 모델 크기가 고정되어 있더라도 패치 크기를 줄여서 토큰 수를 늘리면 성능이 대폭 향상

#### DiT Gflops are critical to improving performance & Larger DiT models are more compute-efficient
* 결과: 큰 모델이 더 효율적입니다.

<p align = 'center'>
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/1568b59b-48a7-440d-a28e-64e0f1530930" />
</p>


#### SOTA

<p align = 'center'>
<img width="300" height="500" alt="image" src="https://github.com/user-attachments/assets/ce90f8ae-ab52-4492-a122-615642816d71" />
</p>

1. Table 2: ImageNet 256x256 해상도 성능 비교

    * DiT-XL/2의 성과 - 최고 기록 달성: cfg=1.50 (분류기 없는 가이던스 스케일 1.5)을 사용했을 때, FID 2.27을 기록
    * 확산 모델 중 가장 성능이 좋았던 LDM-4-G의 3.60을 크게 앞서는 수치이며, 당시 최강의 GAN이었던 StyleGAN-XL(2.30)보다도 낮은(좋은) 수치

2. Table 3: ImageNet 512x512 해상도 성능 비교

    * DiT-XL/2의 성과 - cfg=1.50에서 FID 3.04를 기록
    * 픽셀 기반 확산 모델의 최고봉이었던 ADM-G, ADM-U의 3.85보다 훨씬 좋은 성능
    * ADM이 1983 Gflops 이상의 연산량을 사용하는 데 비해, DiT-XL/2는 524.6 Gflops만으로 더 좋은 성능


---
### 의의

#### 1. 확산 모델의 아키텍처 패러다임 전환 (U-Net $\rightarrow$ Transformer)기존의 한계 탈피

#### 2. 확산 모델의 확장 법칙(Scaling Laws) 발견

#### 3. SOTA

#### 4. 아키텍처 통합 및 미래 연구 가속화

* 확산 모델이 트랜스포머 아키텍처를 채택함으로써, 자연어 처리나 다른 비전 분야에서 개발된 최신 학습 기법, 최적화 기술, 하드웨어 가속 등의 이점
* 단순함: 복잡한 다운샘플링/업샘플링 구조 없이 표준적인 블록 반복 구조를 사용함

---
