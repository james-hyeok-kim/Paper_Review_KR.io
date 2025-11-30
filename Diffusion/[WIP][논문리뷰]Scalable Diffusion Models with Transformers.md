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
<img width="672" height="662" alt="image" src="https://github.com/user-attachments/assets/2c35c1e1-014a-4031-b429-1243e58f0ecb" />
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





---
