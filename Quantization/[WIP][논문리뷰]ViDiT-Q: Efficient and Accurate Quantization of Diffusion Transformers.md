# ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers

저자

Tianchen Zhao12, Tongcheng Fang12, Haofeng Huang1, Rui Wan1, Widyadewi Soedarmadji1, Enshu Liu1

Shiyao Li1, Zinan Lin3, Guohao Dai24, Shengen Yan2, Huazhong Yang1, Xuefei Ning1∗, Yu Wang1*

1 Tsinghua University, 2 Infinigence AI, 3 Microsoft, 4 Shanghai Jiaotong Universit

출간 : arXiv preprint arXiv:2406.02540, 2024

논문 : [PDF](https://arxiv.org/pdf/2406.02540)

---

## 1. Introduction

### 1. 연구 배경: 고성능 DiT 모델의 비용 문제

* 막대한 자원 소모: 엣지 디바이스(Edge devices) 배포를 어렵게
* 구체적 예시: OpenSORA 모델의 경우, 16프레임의 $512 \times 512$ 비디오 하나를 생성하는 데 10GB 이상의 GPU 메모리와 Nvidia A100 기준 약 50초가 소요

### 2. 문제점: 기존 양자화 방식의 한계

* CNN이나 거대 언어 모델(LLM)을 위해 설계된 기존 양자화 방법들은 텍스트 기반 이미지 및 비디오 생성 작업에서 성능 저하
* 높은 데이터 변동성 (High Data Variation): DiT 모델은 토큰(Token), 타임스텝(Timestep), 조건(Condition) 등 여러 차원에서 데이터 변동이 심해, 양자화 범위를 설정하기 어렵고 오차가 커집니다
* 복합적인 품질 지표: 시각적 생성 작업은 단순한 수치 오차 최소화만으로는 텍스트 정렬(alignment), 시각적 충실도(fidelity), 시간적 일관성(temporal consistency) 등 다면적인 품질을 보장하기 어렵

### 3. 해결책: ViDiT-Q (Video & Image Diffusion Transformer Quantization)

* 세밀한(Fine-grained) & 동적(Dynamic) 양자화: 동적으로 변하는 정밀한 양자화 파라미터를 사용
* 정적-동적(Static-Dynamic) 채널 밸런싱: 시간에 따라 변하는 채널 불균형(channel imbalance) 문제를 해결하기 위해, 스케일링(Scaling)과 회전(Rotation) 기반 방식을 결합한 새로운 기술을 설계
* 지표 분리 혼합 정밀도(Metric-Decoupled Mixed Precision): 양자화가 텍스트 정렬, 화질 등 생성 품질의 각 측면에 미치는 영향을 분리하여 분석

### 4. 주요 기여 및 성과

* 성능 유지: ViDiT-Q는 W8A8(8비트) 및 W4A8(4비트 가중치/8비트 활성화) 양자화에서 시각적 품질 저하가 거의 없는 결과를 달성
* 효율성 증대: 효율적인 GPU 커널 구현을 통해 2-2.5배의 메모리 절약과 1.4-1.7배의 지연 시간(Latency) 단축

---

## 2. Related Works

### 2.1 DIFFUSION TRANSFORMERS FOR IMAGE AND VIDEO GENERATION

#### 1. CNN에서 Transformer로의 전환
* CNN 대신 트랜스포머(Transformer)구조를 확산 모델의 백본으로 채택하여 "Diffusion Transformers (DiTs)"라는 개념을 도입

#### 2. 이미지 생성 모델 (Image Generation)

* DiT & UViT
* PixArt- $\alpha$ : 텍스트를 이미지로 변환(Text-to-Image)하는 작업에서 DiT 구조를 탐구한 모델

#### 3. 비디오 생성 모델 (Video Generation)
* 초기 모델: CNN 백본
* Latte: 텍스트를 비디오로 변환(Text-to-Video)하는 작업에 트랜스포머를 처음으로 도입한 선구적인 모델
* SORA & OpenSORA: OpenAI의 SORA 성능 $\rightarrow$ 비디오 확산 트랜스포머(Video Diffusion Transformers) 


### 2.2 IMAGE AND VIDEO GENERATION EVALUATION METRICS

#### 1. 이미지 평가 지표 (Image Metrics)

* FID (Fréchet Inception Distance) & IS (Inception Score)
    * 생성된 이미지와 참조 이미지 간의 특징 차이(Inception network feature difference)를 측정하여 이미지의 품질(Quality)과 충실도(Fidelity)를 평가하는 데 가장 널리 사용되는 지표

* ClipScore
    * 생성된 이미지가 주어진 텍스트 프롬프트(지시사항)를 얼마나 잘 따르는지, 즉 텍스트-이미지 정렬(Text-Image Alignment)을 평가

* ImageReward & HPS (Human Preference Score)
    * 실제 사용자 데이터를 기반으로 학습된 보상 모델을 사용하여, 인간의 선호도(Human Preference)를 평가에 반영

#### 2. 비디오 평가 지표 (Video Metrics)

* FVD (Fréchet Video Distance)
    * 이미지의 FID를 비디오 영역으로 확장한 것, 비디오의 전반적인 특징 분포 차이를 측정
* CLIPSIM
    * 비디오 내용과 텍스트 지시사항 간의 유사도(Similarity)를 추정
* CLIP-temp
    * 비디오 프레임들 간의 의미적(Semantic) 유사도를 측정하여 일관성을 평가
* Flow-score
    * EvalCrafter 벤치마크의 일부로 제안되었으며, 비디오의 움직임 품질(Motion Quality)을 평가
* DOVER
    * 비디오 자체의 품질(Video Quality Assessment)을 평가하는 데 사용


#### 3. 결론: 다각적 평가의 중요성

* 다양한 측면의 지표들을 종합적으로 고려해야 양자화(Quantization)가 생성 품질에 미치는 영향을 정확히 파악


### 2.3 MODEL QUANTIZATION

#### 1. 사후 학습 양자화 (Post-Training Quantization, PTQ)

* PTQ는 모델 학습이 끝난 후, 가중치와 활성화 값을 더 낮은 비트의 정수(Integer)로 변환하여 모델을 압축하는 효율적인 방법

#### 2. 기존 연구의 흐름

* 확산 모델 (Diffusion Models)
    * 확산 모델의 핵심인 타임스텝(Timestep) 차원에 집중
    * Q-Diffusion 및 PTQ4DM과 같은 연구는 각 타임스텝별로 활성화 데이터를 수집하여 양자화 파라미터를 결정하는 방식을 사용
* 트랜스포머 (Transformers)
    * 주로 거대 언어 모델(LLM)이나 비전 트랜스포머(ViT)에서 발생하는 채널 불균형(Channel Imbalance) 문제를 해결하는 데 주력
    * SmoothQuant: 채널별 스케일링(Scaling)을 통해 가중치와 활성화 값의 양자화 난이도를 균형 있게 맞춥니다.
    * Quarot: 직교 행렬 회전(Rotation)을 사용하여 채널 간의 값 분포를 더 고르게 만듭니다.
* 확산 트랜스포머 (DiTs)
    * 최근 연구들은 DiT 구조에 특화된 채널 불균형 해결책을 모색
    * Q-DiT: 채널마다 서로 다른 양자화 파라미터를 할당하여 불균형을 해결
    * PTQ4DiT: 모든 타임스텝에 적용되는 고정된(Fixed) 채널 밸런스 마스크를 설계하여, 시간에 따라 변하는 채널 불균형 문제에 대응

#### 3. 기존 방법의 한계점

* 고난이도 DiT 작업에 직접 적용할 경우, 눈에 띄는 성능 저하
* 이유: 기존 방법들은 DiT 모델이 가진 복잡한 데이터 변동성(토큰, 타임스텝, 조건 등)을 완벽하게 처리하지 못하기 때문

---

## 3 PRELIMINARY ANALYSIS

### 3.1 QUANTIZATION ERROR ANALYSIS

#### 1. 양자화 문제의 정의 (Problem Formulation)

* "원본 FP(Floating-Point) 모델과 양자화된 모델 간의 차이를 최소화하는 최적의 전략을 찾는 것"

$$min_{W_{q},X_{q}}\sum_{l}^{L}(||W^{(l)}-Q(W^{(l)})||_{2}^{2}+||X^{(l)}-Q(X^{(l)})||_{2}^{2}) \quad \quad (1)$$

* 양자화 연산자 $Q$

$$x_{int}=Q(x;s,z,b)=clamp(\lfloor\frac{x}{s}\rfloor+z,0,2^{b}-1) \quad \quad (2)$$

* $s$는 스케일링 팩터
* z$는 제로 포인트
* $b$는 비트 수

#### 2. 오차의 구성: 클램핑 오차 vs. 반올림 오차

* 클램핑 오차 (Clamping Error): 표현 범위를 벗어난 값을 잘라내면서 발생합니다.
    * 스케일링 팩터 $s$를 키우면 표현 범위가 넓어져 클램핑 오차는 줄어듭니다
* 반올림 오차 (Rounding Error): 실수 값을 정수로 변환하면서 발생합니다.
    * $s$가 커지면 각 정수 사이의 간격이 넓어져 반올림 오차($[-\frac{s}{2}, \frac{s}{2}]$ 범위)가 증가합니다.
* 즉, 이 둘은 Trade-off 관계에

#### 3. 주요 오차 원인: 그룹 내 큰 데이터 변동성 (High Data Variation)

* 대부분의 최신 연구 및 배포 도구는 MinMax 양자화 방식을 채택하고 있습니다.
* 이 방식은 데이터의 최대값( $max(x)$ )과 최소값( $min(x)$ )을 모두 포함하도록 $s$를 설정하여 클램핑 오차를 제거합니다.
* 남은 주요 오차 원인은 반올림 오차(Rounding Error)가 됩니다.
* 양자화 그룹(group) 내에 데이터의 변동성이 클 경우(즉, outlier가 존재할 경우), 이 큰 값을 포함하기 위해 스케일링 팩터 $s$가 불필요하게 커집니다.
    * 텐서 전체를 하나의 그룹으로 묶는(tensor-wise) 경우, 소수의 매우 큰 값 때문에 $s$가 커지게 되고, 이는 대부분의 작은 값들에 대해 큰 반올림 오차를 유발합니다

#### 4. 비간섭성(Incoherence) 개념

* 논문은 이를 설명하기 위해 "비간섭성(Incoherence)" 개념을 인용
* 데이터 그룹 내에서 최대값이 평균적인 크기보다 훨씬 클 때(outlier가 있을 때), 해당 그룹은 "incoherent"하다고 하며 양자화하기 어렵습니다.
* 따라서 오차를 줄이기 위해서는 그룹 내 데이터 분포가 균일하도록 만드는 추가적인 처리가 필수적입니다.


### 3.2 UNIQUE CHALLENGES FOR DITS AND VISUAL GENERATION

#### 1. DiT 모델의 구조적 챌린지: 다차원적 데이터 편차

* 토큰별 편차 (Token-wise Variation): 이미지나 비디오를 구성하는 비주얼 토큰들 사이에 상당한 값의 차이가 존재합니다. 특히 비디오 DiT의 경우 공간적(spatial) 차원뿐만 아니라 시간적(temporal) 차원에서도 편차가 발생합니다.

* 조건별 편차 (Condition-wise Variation): 생성 품질을 높이는 CFG(Classifier-Free Guidance) 기법은 조건이 있는 경우(Cond)와 없는 경우(Uncond)를 각각 계산합니다. 이 두 경우 사이의 활성화 값 차이가 매우 커서, 이를 하나의 그룹으로 묶어 양자화하면 오차가 커집니다.

* 타임스텝별 편차 (Timestep-wise Variation): Diffusion 모델은 노이즈를 제거하기 위해 여러 단계(Timestep)를 반복합니다. 동일한 레이어라 하더라도 타임스텝이 달라지면 활성화 값의 분포가 크게 변합니다.

* 채널별 편차 (Channel-wise Variation): 가중치와 활성화 값 모두 채널 간의 값 차이가 큽니다. 특히 중요한 발견은 활성화 값의 채널 불균형이 시간에 따라 변한다(Time-varying)는 점입니다. 즉, 특정 채널이 항상 큰 값을 가지는 것이 아니라 타임스텝마다 달라집니다

#### 2. 시각적 생성 작업의 챌린지: 평가 지표의 불일치

* 일반적인 양자화 방법은 단순히 수치적 오차(예: MSE)를 최소화하는 것을 목표로 합니다. 하지만 시각적 생성 작업에서는 이것만으로 충분하지 않습니다.

* 다각적인 품질 평가 필요: 생성된 이미지나 비디오는 심미성(Aesthetic), 텍스트 정합성(Alignment), 시간적 일관성(Temporal Consistency) 등 다양한 관점에서 평가되어야 합니다.

* 단순 오차 최소화의 한계: 단순히 절대적인 수치 오차를 줄이는 것만으로는 위에서 언급한 다면적인 생성 품질을 보존하기 어렵습니다. 즉, 수치적으로는 오차가 작아도 눈으로 보기엔 품질이 크게 떨어질 수 있습니다.


---

## 4 VIDIT-Q: QUANTIZATION SCHEME TAILORED FOR DITS


### 4.1 FINE-GRAINED GROUPING AND DYNAMIC QUANTIZATION

### 4.2 STATIC-DYNAMIC CHANNEL BALANCING

### 4.3 METRIC DECOUPLED MIXED PRECISION DESIGN

---

## 5 EXPERIMENTS

### 5.1 IMPLEMENTATION DETAILS AND EXPERIMENTAL SETTINGS

### 5.2 MAIN RESULTS

### 5.3 HARDWARE RESOURCE SAVINGS

---

## 6 CONCLUSION AND LIMITATIONS

---


