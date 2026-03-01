# SVDQUANT: ABSORBING OUTLIERS BY LOW-RANK COMPONENTS FOR 4-BIT DIFFUSION MODELS

저자: 

Muyang Li1∗ ‡ Yujun Lin1∗ Zhekai Zhang1† Tianle Cai4 Xiuyu Li5‡

Junxian Guo1,6 Enze Xie2 Chenlin Meng7 Jun-Yan Zhu3 Song Han1,2

1MIT 2NVIDIA 3CMU 4Princeton 5UC Berkeley 6SJTU 7Pika Labs

출간: ICLR 2025, arXiv 버전(v4)은 2025년 11월 8일에 업데이트

논문: [PDF](https://arxiv.org/pdf/2411.05007)

---

## 1. Introduction


### 1. 배경 및 당면 과제: 디퓨전 모델의 대형화

<p align = 'center'>
<img width="250" height="350" alt="image" src="https://github.com/user-attachments/assets/3281bea2-736e-409a-9b8e-baa4de6ddc80" />
</p>

* 모델 규모의 확장: 최근 디퓨전 모델은 시각적 품질과 텍스트 정렬 능력을 높이기 위해 모델 크기를 급격히 키우고 있습니다.
    * 예를 들어, Stable Diffusion 1.4(800M)에서 시작해 SDXL(2.6B), AuraFlow(6B), 그리고 최신 FLUX.1(12B)에 이르기까지 매개변수 수가 크게 증가
* 높은 연산 집약도: 디퓨전 모델은 대규모 언어 모델(LLM)보다 연산량이 훨씬 많으며, 모델 크기가 커짐에 따라 연산 비용이 더 빠르게 증가
* 연산 병목 현상(Compute-bounded): LLM의 지연 시간이 주로 가중치를 로드하는 메모리 대역폭에 의해 결정되는 것과 달리, 디퓨전 모델은 단일 배치에서도 연산 자체가 병목이 되는 특성을 가집니다. 따라서 속도를 높이려면 가중치뿐만 아니라 활성화(activation)까지 모두 낮은 비트로 양자화

### 2. 기존 방식의 한계

* 품질 저하 문제: 가중치와 활성화를 모두 4비트로 양자화(W4A4)하는 공격적인 방식은 심각한 이미지 품질 저하를 초래하기 쉽습니다.

* Smoothing 방식의 부족함: 가중치와 활성화 사이의 이상치(outlier)를 재분배하는 기존의 '스무딩(smoothing)' 기법은 양쪽 모두가 이상치에 민감한 상황에서는 충분한 효과를 거두지 못합니다.


### 3. 제안 방법: SVDQuant 패러다임

<p align = 'center'>
<img width="772" height="273" alt="image" src="https://github.com/user-attachments/assets/a2bb174b-01e9-400e-90c1-7f08117fbf8b" />
</p>

* 이상치 통합 및 이동: 먼저 스무딩 기법을 통해 활성화의 이상치를 가중치로 이동시켜 통합합니다.
* SVD를 이용한 분해: 업데이트된 가중치를 특잇값 분해(SVD)하여, 큰 이상치를 포함하는 16비트 고정밀 저차원 분기(low-rank branch)와 나머지 잔차를 처리하는 4비트 양자화 분기로 나눕니다.
* Nunchaku 추론 엔진: 독립적인 저차원 분기 실행으로 인한 메모리 오버헤드를 막기 위해, 두 분기의 커널을 하나로 합쳐 데이터 이동을 최소화하는 전용 엔진 'Nunchaku'를 공동 설계했습니다.

### 4. 주요 성과

* 메모리 절감: 12B 규모의 FLUX.1 모델의 메모리 사용량을 3.5배 줄였습니다.
* 속도 향상: 16GB VRAM을 가진 노트북용 RTX 4090 GPU에서 CPU 오프로딩 없이 실행 가능하게 하여, 16비트 모델 대비 8.7배, 기존 4비트 가중치 전용 양자화(W4A16) 대비 3배의 속도 향상을 달성했습니다.
* 범용성: UNet 및 DiT 구조 모두에서 시각적 품질을 잘 유지하며, 재양자화 없이 기존 LoRA와도 원활하게 통합됩니다.


---

## 2. Related Work

### 1. 디퓨전 모델 및 가속화 (Diffusion Models & Acceleration)

* 모델의 진화: 초기 컨볼루션 기반의 UNet 구조에서 최근에는 트랜스포머 기반의 DiT 아키텍처로 전환되고 있으며, 모델의 규모가 계속 커지고 있습니다.
* 속도 개선 노력: 느린 추론 속도를 해결하기 위해 적은 단계로 이미지를 생성하는 샘플러(Few-step samplers) 연구나 모델을 압축하는 증류(Distillation) 기법이 활발히 연구되어 왔습니다.
* 기타 최적화: 효율적인 아키텍처 설계, 희소 추론(Sparse inference), 분산 추론 등 다양한 하드웨어 가속 방식이 제안되었습니다.

### 2. 양자화 기법 (Quantization)

* LLM에서의 활용: 모델 크기를 줄이고 추론을 가속화하기 위해 LLM 분야에서 양자화가 널리 사용되어 왔습니다.
* 디퓨전 모델 양자화: 초기에는 8비트 양자화(Q-Diffusion, PTQ4DM 등)가 주를 이루었으며, 이후 타임스텝을 고려한 양자화나 비디오 생성 모델용 양자화 등으로 발전했습니다.
* SVDQuant의 진보: 기존 연구들이 주로 8비트 수준에 머물거나 실제 속도 향상을 보고하지 못한 것과 달리, SVDQuant는 4비트 양자화(W4A4)를 성공적으로 적용하고 실제 GPU에서의 속도 향상을 입증한 최초의 사례 중 하나입니다.

### 3. 저차원 분해 (Low-rank Decomposition)

* 압축 및 튜닝: 저차원 분해는 연산 효율을 높이거나 LoRA와 같이 효율적인 모델 파인튜닝을 위해 널리 쓰여왔습니다.
* 기존 저차원 양자화와의 차이: 이전 연구(예: LoRC)는 양자화 오류를 보상하기 위해 저차원 분기를 사용했지만, 주로 가중치에만 집중하여 실제 추론 속도 향상이 미미했습니다.
* 공동 설계(Co-design): SVDQuant는 가중치와 활성화를 함께 양자화하며, 별도 분기로 인한 오버헤드를 줄이기 위해 Nunchaku 추론 엔진을 통해 커널을 융합했다는 점이 핵심적인 차별점입니다.

---

## 3. Quantization Preliminary

### 1. 양자화의 기본 정의

* 양자화 공식: 텐서 $X$의 양자화 표현 $Q_X$는 다음과 같이 정의됩니다.

$$Q_X = \text{round}\left(\frac{X}{s_X}\right)$$

* 스케일링 인자 ($s_X$): 데이터의 범위를 양자화 범위에 맞추기 위해 사용됩니다.

$$s_X = \frac{\max(X)}{q_{max}}$$

* $q_{max}$ 값: $k$-비트 부호 있는 정수(INT)의 경우 $q_{max} = 2^{k-1} - 1$이며, 4비트 부동소수점(FP4)의 경우 6으로 설정됩니다.


### 2. 선형 레이어 연산의 근사

$$XW \approx Q(X)Q(W) = s_X s_W \cdot Q_X Q_W$$

### 3. 하드웨어 가속을 위한 제약 조건

* 동일 비트 폭 요구: 현대의 상업용 GPU에서 연산 속도를 높이려면 입력( $Q_X$ )과 가중치( $Q_W$ )가 동일한 비트 폭을 사용해야 합니다.
* 업캐스팅(Upcast)의 문제: 만약 비트 폭이 다르면 연산 중에 낮은 정밀도의 데이터를 높은 정밀도로 변환해야 하므로, 양자화로 얻을 수 있는 성능 이점이 사라지게 됩니다.
* 용어 정의: 논문에서는 $z$-비트 가중치와 $y$-비트 활성화를 $WzAy$로 표기하며, 본 연구의 목표는 W4A4(4비트 가중치 및 활성화) 가속입니다.

### 4. 기존 기법의 한계

* W4A4와 같은 공격적인 양자화에서 발생하는 이상치(Outlier) 문제를 해결하기 위해 기존에 사용되던 방법들의 한계점은 다음과 같습니다.

#### 양자화 인식 훈련 (QAT)
* FLUX.1과 같이 10B 이상의 매개변수를 가진 대형 모델을 튜닝하기에는 막대한 컴퓨팅 자원이 필요함.

#### 회전 (Rotation)

* "디퓨전 모델의 적응형 정규화(Adaptive Normalization) 레이어 때문에 오프라인 적용이 불가능하며, 온라인 적용은 실행 오버헤드가 큼."


---

## 4. Method

### 1. 문제 정의 (Problem Formulation)
양자화의 목표는 실제 연산 결과와 양자화된 연산 결과 사이의 오차( $E$ )를 최소화하는 것입니다. 

$$E(X, W) = \|XW - Q(X)Q(W)\|_F$$

논문은 명제 4.1을 통해 이 오차가 다음 네 가지 요소에 의해 결정됨을 수학적으로 증명합니다. 

* 가중치( $W$ )
* 입력( $X$ )

### 2. SVDQuant: 이상치 흡수 전략

<p align = 'center'>
<img width="852" height="318" alt="image" src="https://github.com/user-attachments/assets/a2d6a77e-6da9-4fa8-8020-8d673918f50d" />
</p>

$$
XW = \hat{X}\hat{W} \approx \underbrace{\hat{X}L_1 L_2}_{\text{16-bit low-rank branch}} + \underbrace{Q(\hat{X})Q(R)}_{\text{4-bit residual}} \qquad (5)
$$

* Step 1: 활성화에서 가중치로 이상치 이동 (Smoothing)
    * 활성화( $X$ )에 있는 이상치는 양자화를 어렵게 만드는 주범입니다.
    * 스무딩 인자( $\lambda$ )를 사용하여 활성화의 이상치를 가중치( $W$ ) 쪽으로 옮깁니다.
    * 결과적으로 활성화는 양자화하기 쉬워지지만, 반대로 가중치는 이상치가 더 많아져 양자화하기 훨씬 어려워집니다(Figure 4 참고). 

* Step 2: 가중치 이상치를 저차원 분기로 흡수 (SVD)
    * 어려워진 가중치를 해결하기 위해, 가중치( $\hat{W}$ )를 16비트 고정밀 저차원 분기( $L_1 L_2$ )와 4비트 잔차( $R$ )로 분해합니다.
    * SVD(특잇값 분해)를 통해 가장 지배적인 값(이상치)들을 저차원 분기가 가져가게 함으로써, 남은 잔차( $R$ )는 범위가 크게 압축되고 이상치가 거의 없는 상태가 되어 4비트로도 정확하게 양자화할 수 있습니다. 

### 3. Nunchaku: 추론 엔진 최적화

#### 1. 근본적인 문제 - 메모리 접근 병목 (Memory Bottleneck)

* 이론적으로 LoRA(Rank 16 또는 32)는 전체 연산량에서 차지하는 비중이 매우 적습니다. 그러나 이를 단순히 별도의 분기로 실행하면 다음과 같은 문제가 발생합니다.

* 중복 데이터 이동: LoRA와 4비트 분기가 각각 동일한 입력 데이터를 메모리에서 읽고, 각자의 결과를 다시 메모리에 쓰는 과정에서 불필요한 데이터 이동이 발생합니다.

* 캐시 효율 저하: 특히 활성화 데이터가 GPU의 L2 캐시 용량을 초과할 경우, 느린 DRAM에 반복적으로 접근하게 되어 4비트 양자화로 얻은 속도 이득의 약 50%가 상쇄될 수 있습니다.



#### 2. Nunchaku의 핵심 솔루션 - 커널 융합 (Kernel Fusion)

<p align = 'center'>
<img width="829" height="316" alt="image" src="https://github.com/user-attachments/assets/a2930fc9-e90d-4c1d-a4e4-bc8aea71e3d9" />
</p>


* Nunchaku는 서로 다른 분기의 연산을 하나로 묶어 데이터를 한 번만 읽고 쓰도록 최적화합니다.

* 통합 커널 1 (Down Proj + Quantize)
    * LoRA의 Down Projection( $L_1$ )과 저비트 분기의 활성화 양자화(Quantize) 과정을 하나로 합칩니다.
    * 두 연산이 입력 데이터를 공유하므로 메모리 읽기 횟수가 절반으로 줄어듭니다.

* 통합 커널 2 (Up Proj + 4-bit Compute)
   * LoRA의 Up Projection( $L_2$ )과 4비트 연산(Compute) 커널을 합칩니다.
   * 두 연산의 결과값을 내부적으로 합산한 뒤 출력 데이터를 한 번만 저장합니다.


#### Down Proj와 Quantize를 어떻게 합쳤는가?

* 데이터 로드 (1회): GPU가 메모리에서 입력 활성화 값( $\hat{X}$ ) 조각을 한 번만 읽어와서 GPU 내부의 빠른 연산 공간(레지스터 또는 공유 메모리)에 둡니다.
* 동시 연산: 읽어온 그 데이터를 사용해 두 가지 일을 즉시 수행합니다
* Down Proj: 저차원 행렬 $L_1$과 곱셈 연산을 수행하여 중간값( $\hat{X}L_1$ )을 만듭니다
* Quantize: 동일한 데이터에 대해 스케일링을 적용하고 4비트로 변환하여 $Q_{\hat{X}}$를 만듭니다
* 결과 저장: 연산이 끝난 두 결과값( $\hat{X}L_1$과 $Q_{\hat{X}}, s_X$ )만 다음 단계로 전달합니다.


#### 3. Nunchaku 도입의 효과

* 실질적 무비용 연산: 지연 시간을 기존 50%에서 5~10% 수준으로 대폭 낮췄습니다.

* 극적인 속도 향상
    * RTX 4090 (Laptop): 메모리 부족으로 인한 CPU 오프로딩을 제거하여 16비트 모델 대비 10.1배의 속도 향상을 달성했습니다.
    * RTX 5090 (Desktop): Blackwell 아키텍처의 NVFP4 정밀도를 활용해 기존 가중치 전용 양자화(W4A16)보다 3.1배 빠르게 작동합니다.

* Nunchaku는 LoRA 분기를 기존 SVD 저차원 분기에 통합하여 효율적으로 실행할 수 있도록 지원합니다.

## 4. Euqation

$$\|\hat{X}\hat{W}-(\hat{X}L_1L_2+Q(\hat{X})Q(R))\|_F = \|\hat{X}R-Q(\hat{X})Q(R)\|_F = E(\hat{X},R) \quad (6)$$

* 전체 Error는 Low Rank Branch의 quantization error로 축소

$$\mathbb{E}[\max(|R|)]\le c\cdot\mathbb{E}[\|R\|_F] \quad(7)$$

* 최대값은 전체 크기에 비례
* $\max(|R|)$: 잔차 행렬 $R$에서 가장 큰 값(이상치)입니다.
* $\|R\|_F$: 행렬의 모든 요소의 제곱합의 루트값으로, 행렬의 '전체 에너지' 또는 '전체 크기'를 뜻합니다.
* "행렬의 이상치(최대값)는 행렬 전체의 크기보다 갑자기 터무니없이 커질 수 없으며, 일정한 비율($c$) 안에서 움직인다"는 통계적 가정입니다. 특히 데이터가 정규 분포를 따를 경우 이 관계가 성립함이 수학적으로 증명되어 있습니다. 

* 양자화 오차의 상한선 (명제 4.2)잔차 $R$의 양자화 오차가 수학적으로 어떻게 제한되는지 보여주는 최종 결론입니다.

$$\mathbb{E}[\|R-Q(R)\|_F]\le\frac{c\sqrt{\text{size}(R)}}{q_{max}}\cdot\mathbb{E}[\|R\|_F] \quad (8)$$

* 핵심 결론: 양자화 오차는 잔차 행렬의 크기( $\|R\|_F$ )에 비례합니다.
* 따라서 SVD를 통해 $\|R\|_F = \|\hat{W}-L_1L_2\|_F$를 최소화하는 $L_1, L_2$를 찾는 것이 양자화 오차를 줄이는 최적의 방법임을 수학적으로 정당화합니다.


1) 1단계 (스케일 인자의 역할): 양자화에서 스케일 인자 $s_R$은 보통 $\frac{\max(|R|)}{q_{max}}$로 결정됩니다. 즉, 가장 큰 값( $\max(|R|)$ )이 양자화의 한 칸(Step) 크기를 결정합니다.
2) 2단계 (오차 정의): 양자화 오차 $\|R - Q(R)\|_F$는 각 요소가 반올림되면서 발생하는 오차들의 합입니다. 수학적으로 이 오차의 기댓값은 다음과 같이 정리됩니다.

$$\mathbb{E}[\|R - Q(R)\|_F] \le \mathbb{E}[s_R] \cdot \sqrt{\text{size}(R)}$$

(여기서 $\sqrt{\text{size}(R)}$은 행렬의 전체 요소 개수에 따른 가중치입니다.) 

3) 3단계 ($s_R$ 대입): 위 식에 $s_R = \frac{\max(|R|)}{q_{max}}$를 대입하면 다음 식이 나옵니다.

$$\frac{\sqrt{\text{size}(R)}}{q_{max}} \cdot \mathbb{E}[\max(|R|)]$$

4) 4단계 (수식 (7) 적용): 마지막으로 여기에 앞서 정의한 $\mathbb{E}[\max(|R|)] \le c \cdot \mathbb{E}[|R|_F]$를 대입하면 최종적으로 수식 (8)이 완성됩니다.

$$\mathbb{E}[\|R - Q(R)\|_F] \le \frac{c \sqrt{\text{size}(R)}}{q_{max}} \cdot \mathbb{E}[\|R\|_F]$$


<p align = 'center'>
<img width="200" height="300" alt="image" src="https://github.com/user-attachments/assets/97f71348-1a0c-45af-a505-09855781824a" />
</p>

* $W$ (검은색 실선): 원래 모델의 가중치 행렬입니다
* $\hat{W}$ (빨간색 실선): 활성화(activation)에 있던 이상치를 가중치 쪽으로 옮겨오는 스무딩(Smoothing) 과정을 거친 후의 가중치
* $R$ (회색 실선): $\hat{W}$에서 상위 특잇값들을 저차원 분기($L_{1}L_{2}$)로 추출하고 남은 잔차(Residual) 행렬
* x 축 : 인덱스(순서)
* y 축 : 특잇값(Singular Value)의 크기 (Magnitude)

* 그래프의 핵심 통찰급격한 하강 (Steep Drop)
    * $\hat{W}$의 초기 32개 특잇값은 매우 급격하게 감소합니다. SVDQuant는 이 지배적인(dominant) 값들을 16비트 고정밀 분기가 담당하게 하여 양자화 오차를 원천 차단합니다.
* 잔차의 안정화
    * 상위 값들을 제거하고 남은 잔차 $R$의 곡선은 매우 낮고 평탄한 형태를 띱니다. 이는 가중치의 값 범위가 크게 압축되고 이상치가 사라졌음을 의미하며, 결과적으로 4비트(INT4/FP4)로 양자화하더라도 정보 손실이 거의 발생하지 않는 상태가 됩니다.

---

## 5. Experiments


### 1. 실험 환경 (Setups)

* 대상 모델: FLUX.1 (12B), PixArt-Σ (600M), SANA (1.6B), SDXL (2.6B) 등 UNet과 DiT 구조를 모두 포함합니다.
* 데이터셋: Midjourney 스타일의 MJHQ-30K와 사실적인 이미지 중심의 sDCI (Densely Captioned Images)를 사용해 범용성을 평가했습니다.
* 비교 대상: 가중치 전용 양자화인 NF4, 8비트 양자화 기법인 ViDiT-Q, MixDQ, 그리고 산업 표준인 TensorRT 등과 비교했습니다.
* 주요 지표: 이미지 품질(FID, Image Reward), 모델 간 유사도(LPIPS, PSNR, SSIM) 등을 측정했습니다.

### 2. 주요 결과 (Results)

<p align = 'center'>
<img width="717" height="774" alt="image" src="https://github.com/user-attachments/assets/146c30b9-6071-43e4-9a18-bdeb98269456" />
</p>

#### 시각적 품질 (Visual Quality)

* 8비트 결과: 16비트 모델과 거의 차이가 없는 무손실 품질을 보여주며, 기존 8비트 베이스라인들을 능가했습니다.
* 4비트 결과: W4A4 환경에서도 NF4(W4A16)보다 높은 품질을 유지했습니다. 특히 FLUX.1-dev에서는 인간의 선호도를 반영하는 Image Reward 점수가 원본 16비트 모델을 뛰어넘기도 했습니다.
* 텍스트 정렬: 다른 4비트 방식들이 텍스트 프롬프트의 세부 사항(예: 흔들 의자 등)을 놓치는 것과 달리, SVDQuant는 텍스트 정보를 정확히 이미지에 반영했습니다.

#### 메모리 및 속도 향상 (Memory & Speedup)

* 모델 크기: 12B 파라미터의 FLUX.1 모델 크기를 22.2 GiB에서 6.1 GiB로 약 3.6배 줄였습니다.
* 노트북 GPU (RTX 4090 16GB): 메모리 부족으로 인한 CPU 오프로딩을 없애고 전체 모델을 GPU에 올림으로써 10.1배의 속도 향상을 기록했습니다.
* 최신 GPU (RTX 5090): Blackwell 아키텍처의 NVFP4 정밀도를 활용해 16비트 모델 및 기존 4비트 가중치 전용 모델보다 3.1배 빠른 추론 속도를 달성했습니다.

#### LoRA 통합 (Integrate with LoRA)
* 재양자화 과정 없이 기존의 다양한 스타일 LoRA(Realism, Anime 등)를 즉시 적용할 수 있습니다.
* Nunchaku 엔진은 LoRA 분기를 SVD 저차원 분기에 융합하여 실행 오버헤드를 최소화합니다.

---
