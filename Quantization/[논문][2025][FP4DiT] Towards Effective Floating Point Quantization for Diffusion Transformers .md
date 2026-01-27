# FP4DiT: Towards Effective Floating Point Quantization for Diffusion Transformers

저자 : 

Ruichen Chen∗

ruichen1@ualberta.ca

Department of Electrical and Computer Engineering

University of Alberta

Keith G. Mills∗

kgmills@ualberta.ca

Department of Electrical and Computer Engineering

University of Alberta

Di Niu dniu@ualberta.ca

Department of Electrical and Computer Engineering

University of Alberta

발표 : arXiv preprint arXiv:2503.15465, 2025 

논문 : [PDF](https://arxiv.org/pdf/2503.15465)

---

## 1. Introduction

### 1. 기존 양자화 기술의 한계:

* 대부분의 기존 확산 모델용 사후 양자화(PTQ) 기법은 U-Net 구조에 최적화되어 있어 최신 DiT 아키텍처에는 잘 맞지 않을 수 있습니다.
* 현재 주류인 정수(INT) 양자화는 신경망의 가중치와 활성화 함수(activation)의 비균일한 분포를 제대로 반영하지 못합니다.

### 2. FP4DiT의 핵심 제안

* FPQ 활용: 가중치와 활성화 분포를 더 잘 정렬하기 위해 FPQ를 사용하여 W4A6(가중치 4비트, 활성화 6비트) 양자화를 달성합니다.
    * E3M0 / E2M1 / E1M2

* 기술적 개선
    * Scale-aware AdaRound: 기존 정수 기반의 적응형 반올림(AdaRound) 기술을 FPQ에 맞게 확장 및 일반화하여 가중치 보정의 성능을 높였습니다.
    * 온라인 활성화 양자화: DiT의 활성화 함수가 입력 패치 데이터에 따라 달라진다는 점을 발견하고, 이를 수용할 수 있는 강력한 온라인 양자화 기법을 적용했습니다.
    * 비용 절감: 가중치 보정(Calibration) 비용을 기존 대비 8배 이상 대폭 줄였습니다.

### 3. 주요 기여 및 실험 결과

* 분포 분석: U-Net은 단계가 진행됨에 따라 활성화 범위가 줄어드는 반면, DiT는 범위가 시간에 따라 이동(shift)한다는 차이점을 밝혀냈습니다.
    * U-Net: Shirinking (Quantization: Temporally-Aware 방식 채택)
    * DiT : Shifting
        * Activation range자체는 일정, 중심이나 분포값이 시간에 따라 이동(Shift)
* 실험적 우위: PixArt-α, PixArt-Σ, Hunyuan 모델을 대상으로 한 실험에서 FP4DiT는 W4A8 및 W4A6 정밀도 모두에서 기존의 Q-Diffusion, ViDiT-Q, Q-DiT 등의 방식을 능가했습니다.
* 성능 지표: HPSv2 및 CLIP 점수 등 여러 텍스트-이미지(T2I) 지표에서 설득력 있는 시각적 콘텐츠를 생성함을 입증했습니다.


---

## 2. Related Work

### 1. 확산 모델의 PTQ (DM PTQ)
* 타임스텝 의존성: 확산 모델은 디노이징 타임스텝에 따라 활성화 범위가 크게 변한다는 독특한 과제가 있습니다.

* 기존 연구: Q-Diffusion, TFMQ-DM 등은 이러한 타임스텝 특성을 보정하기 위해 제안되었으나, 주로 U-Net 아키텍처에 특화되어 있습니다.

### 2. LLM 양자화와의 비교
* 이상치(Outlier) 문제: 수십억 개의 파라미터를 가진 트랜스포머 모델은 양자화하기 어려운 이상치 숨은 상태(hidden states)를 생성하는 경향이 있습니다.

* 구조적 차이: LLM에서 사용되는 이상치 억제 기법은 레이어 정규화(LayerNorm)의 아핀 이동(Affine Shift $\gamma$(스케일)와 $\beta$(이동, Shift) 파라미터)을 활용하지만, DiT는 타임스텝 임베딩에 결합된 적응형 레이어 정규화(AdaLN)를 사용하기 때문에 동일한 기법을 적용하기 어렵습니다.

### 3. 기존 DiT PTQ 연구와의 차별점
* 선행 연구: HQ-DiT(ImageNet 모델 중심)나 ViDiT-Q(채널 밸런싱, 혼합 정밀도 사용)가 존재합니다.

* FP4DiT의 접근: 이전의 DiT PTQ 연구들이 간과했던 가중치 재구성(Weight Reconstruction) 기법 등 기존 확산 모델 PTQ의 강점들을 통합하고 개선하여 텍스트-이미지(T2I) 모델에 최적화했습니다.

---

## 3. Methodology

### 3.1 Uniform vs. Non-Uniform Quantization

<p align = 'center'>
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/e694d5f4-fbfd-4efb-9ffa-2b5332c4d301" />
</p>

$$X^{(int)} = \text{clip}(\lfloor \frac{X}{s} \rfloor + z, I_{min}, I_{max}) \quad(1)$$

* Uniform Quantization: INT

$$f = (-1)^{d_s} 2^{p-b} (1 + \frac{d_1}{2} + \frac{d_2}{2^2} + \dots + \frac{d_m}{2^m}) \quad(2)$$

* Non-Uniform Quantization: FPQ
    * 구성 요소: 부호($d_s$), 지수($p$), 가수($m$) 비트로 구성

* 왜 FPQ(비균일)가 더 유리한가?
    * 풍부한 표현력: 비트 배분(지수 vs 가수)을 어떻게 하느냐에 따라 다양한 값 분포를 만들 수 있어, 낮은 비트(Low-bit)에서도 더 세밀한 표현이 가능합니다.
    * 데이터 분포 일치: 딥러닝 모델의 가중치와 활성화 값은 대개 비균일한 분포를 띠는데, FPQ는 이러한 특성에 더 잘 부합합니다. 이는 기존 IEEE 754 'Float16'보다 'BFloat16'이 특정 알고리즘에서 우수한 성능을 보이는 것과 유사한 원리입니다.
    * 유연성: FP4 형식에서도 E1M2, E2M1, E3M0 등 다양한 변형을 통해 각 층의 특성에 최적화된 형식을 선택할 수 있습니다.

#### 3.1.1 Optimized FP Formats in DiT Blocks.

<p align = 'center'>
<img width="350" height="450" alt="image" src="https://github.com/user-attachments/assets/4caf05b1-ffc5-4945-8591-e1b1a0ee2cc4" />
</p>

<p align = 'center'>
<img width="350" height="300" alt="image" src="https://github.com/user-attachments/assets/df69fe1b-1659-4df1-827e-fd0a4b261fc7" />
</p>

##### 1. 민감한 구간 보존을 위한 전략

* FP4DiT는 이 통찰을 양자화에 적용하여, 첫 번째 피드포워드 선형 층(GELU 직전)에 E3M0 형식을 적용합니다.
* E3M0(지수 3비트, 가수 0비트) 포맷은 0 근처에 더 많은 양자화 지점(discrete values)을 배치할 수 있어, GELU의 민감한 구간을 훨씬 정밀하게 표현할 수 있게 해줍니다.


##### 2. DiT 모델별 혼합 형식(Mixed-format) 적용

<div align="center">
   
|모델|첫 번째 선형 층 (FFN)|나머지 모든 가중치|
|:---:|:---:|:---:|
|PixArt-α|E3M0|E2M1| 
|Hunyuan|E3M0|E2M1| 
|PixArt-Σ|E3M0|E1M2|
   
</div>


### 3.2 AdaRound for FP

#### 1. AdaRound의 기본 개념
* 최적의 반올림(Rounding): 일반적인 양자화는 가장 가까운 값으로 반올림(Rounding-to-nearest)을 수행하지만, 이것이 항상 최적은 아닙니다.
* 손실 최소화: AdaRound는 가중치 변화( $\Delta w$ )가 전체 손실에 미치는 영향을 2차 테일러 확장(Second-order Taylor Expansion)으로 분석하여, 반올림 오차를 최소화하는 방향을 찾습니다.
* 학습 가능한 변수: 가중치를 올림할지 내림할지를 결정하는 이진 게이트(Binary Gate) 역할을 하는 변수 $V$를 도입하여 최적화를 수행합니다.


$$\min_{\Delta V} \| Wx - \tilde{W}x \|^2_F + \lambda f_{reg}(\Delta V) \quad(4)$$

* $Wx$: 양자화 전 레이어의 출력 (원본)
* $\tilde{W}x$: 양자화 후 레이어의 출력
* $f_{reg}(\Delta V)$: 가중치가 0(내림) 또는 1(올림)로 수렴하도록 유도하는 정규화 항(Regularization)

#### 2. FP 양자화에서의 기존 AdaRound의 한계

* 일정한 스케일 가정: 기존 AdaRound는 모든 양자화 구간에서 스케일( $s$ )이 일정하다고 가정하는 정수(INT) 양자화에 맞춰 설계되었습니다.
* FP의 다중 스케일 문제: 부동소수점(FP) 양자화는 지수(Exponent) 비트에 따라 $2^E$개의 서로 다른 스케일을 가집니다.
* 불안정한 업데이트: 스케일이 달라지면 최적화 과정에서 기울기(Gradient)의 크기가 스케일에 의존하게 되어, 업데이트가 불균형하고 불안정해지는 문제가 발생합니다.

#### 3. Scale-aware AdaRound (제안 방법)

* 논문은 FP의 다중 스케일 특성을 고려한 Scale-aware AdaRound를 제안합니다. 핵심은 스케일에 따라 변하는 기울기를 정규화(Normalize)하는 것입니다.
* 수식 수정: 기존의 정류된 시그모이드 함수(Rectified Sigmoid Function)를 스케일( $s$ )로 나누어 보정합니다.

$$h^{\prime}(V^{\prime})=clip(\sigma(\frac{V^{\prime}}{s})(\zeta-\gamma)+\gamma,0,1) \quad (8)$$

* 기울기 독립성: 이 수정을 통해 최적화 과정의 감산 항( $\nabla F(V'_n)$ )이 스케일 $s$값에 관계없이 일정해짐을 수학적으로 증명하였습니다.
    * 발생하는 현상: 스케일 $s$가 크면 기울기도 커져서 가중치가 크게 변하고, $s$가 작으면 가중치가 거의 변하지 않습니다
    * 부동소수점(FP)에서의 문제: FP 양자화는 구간마다 스케일($s$)이 모두 다릅니다 ( $2^E$개 ). 이로 인해 어떤 가중치는 너무 빨리 변하고 어떤 가중치는 너무 느리게 변하는 불균형한 학습(Imbalance)이 발생합니다.
    * 논문에서 제안한 핵심은 가중치 결정 변수 $V'$을 정의할 때 스케일 $s$로 나누어 주는 것
    * 변수 $V'$에 대해 미분을 수행하면, 함수 안에 있던 $\frac{V'}{s}$ 항 때문에 미분값 외부로 $\frac{1}{s}$이 튀어나오게 됩니다.

$$\frac{\partial h'(V')}{\partial V'} = \frac{1}{s} \cdot (\text{시그모이드 미분 항})$$

* 안정화된 최적화: 결과적으로 모든 스케일에서 가중치 재구성(Weight Reconstruction)이 안정적으로 이루어지며 양자화 성능이 향상됩니다.

#### 4. 성능 및 효율성 결과

* 수렴 속도 향상: Scale-aware AdaRound는 기존 INT 기반 AdaRound보다 8배 적은 반복 횟수(2.5k vs 20k steps)만으로도 최적의 성능에 도달할 수 있습니다.
* 정확도 개선: 다중 스케일 환경을 정확히 반영함으로써 저비트(W4) 설정에서도 더 높은 CLIP 점수와 이미지 생성 품질을 보여줍니다.


#### 5. DiT 활성화 값 분석

<p align = 'center'>
<img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/f6954ee8-143f-44ac-9a1a-2971bf702655" />
</p>

* 특정 레이어에서 나오는 전체 데이터의 최댓값이나 평균적인 범위를 추적

* Diffusion Transformer (DiT) 모델이 기존의 U-Net 기반 확산 모델과 어떤 데이터 분포 차이를 보이는지 통계적으로 설명합니다.

* (a) DiT Input Value Box Plot: 타임스텝(Timestep)별 입력 값의 분포를 보여줍니다. U-Net 기반 모델은 타임스텝이 진행됨에 따라 입력 값의 범위가 줄어드는 경향이 있지만, DiT는 범위가 비교적 일정하게 유지됨을 보여줍니다.

* (b) Output Scale (7th DiT Block): 특정 레이어(예: 7번째 DiT 블록)의 출력 스케일을 시계열로 나타냅니다. 시간이 지나도 스케일 변화가 거의 없이 일정(constant)하다는 것을 증명하며, 이는 특정 양자화 전략이 왜 효과적인지에 대한 근거가 됩니다.

* (c) 7th DiT Block Output Box Plot: 실제 출력값들의 분포를 박스 플롯으로 나타내어, 데이터의 범위가 안정적임을 재확인시켜 줍니다.

### 3.3 Token-wise Activation Quantization

* Token-wise Activation Quantization은 레이어 전체가 아닌 개별 토큰(Token 또는 Patch) 단위로 서로 다른 스케일링 파라미터를 적용하는 기법입니다.

#### Token-wise 방식이 필요성

<p align = 'center'>
<img width="338" height="342" alt="image" src="https://github.com/user-attachments/assets/69d4c3b8-bac6-41b9-9ab8-1024001a5c64" />
</p>

* 하나의 타임스텝($t$)을 고정해놓고, 그 안에 들어있는 수많은 토큰(패치)들 각각의 스케일을 비교

* 동적 범위의 불일치: 특정 패치(토큰)는 값이 매우 크고(Outlier), 다른 패치는 값이 매우 작을 수 있습니다. 레이어 전체에 하나의 스케일을 적용하면, 작은 값을 가진 토큰들은 정보가 뭉개져 버립니다.

* 타임스텝별 변화: Diffusion 과정에서 노이즈 수준이 변함에 따라 활성화 값의 분포가 계속 이동합니다. 정적인(Static) 양자화로는 이를 따라가기 어렵습니다.


#### 작동 원리 (Mechanism)

* Token-wise 양자화는 모델이 실행되는 시점(Inference time)에 각 토큰의 통계치를 계산하여 즉석에서 양자화를 수행합니다.
* 통계값 추출: 현재 처리 중인 토큰 $i$의 활성화 벡터 $x_i$에 대해 최댓값(Max)이나 절대 최댓값(Absmax)을 구합니다.
* 개별 스케일링: 각 토큰 벡터를 해당 토큰의 최댓값으로 나누어 $[0, 1]$ 또는 $[-1, 1]$ 범위로 정규화합니다.
* FP 양자화 적용: 정규화된 값을 저정밀도 부동소수점(예: FP6, FP8) 포맷으로 변환합니다. 이때 각 토큰은 자신만의 '스케일' 정보를 별도로 보유하게 됩니다.


#### FP4DiT에서의 핵심 장점

* 이상치(Outlier) 저항성: 특정 토큰에 튀는 값이 있어도 해당 토큰의 스케일만 커질 뿐, 다른 토큰의 양자화 정밀도에는 영향을 주지 않습니다.

* FP(부동소수점) 포맷과의 시너지: FP 포맷은 0 근처의 작은 값들을 아주 세밀하게 표현할 수 있습니다.
    * Token-wise로 범위를 좁혀주면, FP의 비균등한 그리드를 각 토큰의 실제 데이터 분포에 가장 효율적으로 배치할 수 있게 됩니다.

* 정밀도 향상: 특히 W4A6(가중치 4비트, 활성화 6비트)과 같은 극한의 저비트 환경에서도 이미지 생성 품질(CLIP score, FID 등)을 유지하는 핵심 동력이 됩니다.

---

## 4. Results

### 1. 실험 환경 (Experimental Setup)
* 대상 모델: PixArt- $\alpha$ , PixArt- $\Sigma$ , Hunyuan-DiT 등 최신 Diffusion Transformer(DiT) 모델.
* 비트 정밀도
    * W4A8: 가중치 4-bit, 활성화 8-bit (거의 손실 없는 성능 타겟).
    * W4A6: 가중치 4-bit, 활성화 6-bit (초저정밀도 극한 테스트).
* 비교 대상: 기존의 정수 기반 양자화 방식(Q-Diffusion, TFMQ-DM, ViDiT-Q 등).


### 2. 정량적 성능 (Quantitative Results)
<p align = 'center'>
<img width="823" height="757" alt="image" src="https://github.com/user-attachments/assets/ec58f19d-c3b1-4079-b830-e9cbc6169d9f" />
</p>


### 3. 정성적 결과: Figure (7)의 시각적 증거

* 시각적 일관성: 기존 INT 기반 양자화 모델들은 이미지의 세부 디테일이 뭉개지거나 색감이 왜곡되는 현상이 발생합니다.

* 결과: FP4DiT는 원본(Full-precision)과 시각적으로 거의 구분이 불가능한 수준의 결과물을 생성함을 보여줍니다. 이는 Token-wise 양자화가 이상치(Outlier)를 효과적으로 방어했기 때문입니다.

### 4. 효율성 및 속도 (Efficiency)

* Calibration 속도: 전체 재학습 없이 소량의 데이터로 최적화하는 PTQ 방식을 사용하여, 기존 방식 대비 가중치 보정(Calibration) 시간을 8배 이상 단축시켰습니다.

* 하드웨어 가속: NVIDIA Blackwell GPU와 같은 최신 하드웨어에서 FP4/FP6 연산 지원을 활용할 경우, 레이어 수준에서 약 1.5배의 속도 향상과 메모리 절감 효과를 거둘 수 있음을 보였습니다.


---

## 5. Appendix

### 가중치 (Weight - W4)
* W4A8과 W4A6 모두 가중치는 FP4를 사용합니다.
* 일반적으로 E2M1 포맷(지수 2비트, 가수 1비트)을 사용하여, 아주 적은 비트로도 가중치의 큰 값과 작은 값을 효과적으로 표현합니다.

### 활성화 값 (Activation - A8 / A6)

* W4A8: 활성화 값에 FP8 (주로 E4M3 또는 E5M2)을 사용합니다.
* W4A6: 활성화 값에 FP6 (주로 E3M2)을 사용합니다.


---
  
