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

### 3.2 UNIQUE CHALLENGES FOR DITS AND VISUAL GENERATION

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


