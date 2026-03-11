# Ditto: Accelerating Diffusion Model via Temporal Value Similarity

저자 : Sungbin Kim∗, Hyunwuk Lee∗†, Wonho Cho, Mincheol Park, and Won Woo Ro

School of Electrical and Electronic Engineering, Yonsei University, Seoul, Republic of Korea

발표 : 2025 IEEE International Symposium on High-Performance Computer Architecture (HPCA)

논문 : [PDF](https://arxiv.org/pdf/2501.11211)

---

## 1. Introduction

### 1. 핵심 관찰: 인접한 타임스텝 간의 높은 유사성

* 높은 유사도: 연구팀의 분석 결과, 인접한 타임스텝 사이의 활성화 값(Activation)은 0.98에 달하는 매우 높은 유사도를 보였습니다.
* 좁은 값 범위: 이러한 유사성 덕분에 인접 타임스텝 간의 차이값(Temporal Difference)은 원래 활성화 값보다 훨씬 좁은 범위(최대 8.96배 좁음)를 가집니다.
* 양자화 효율성: 양자화된 모델에서 이 차이값의 96.01%는 절반의 비트 너비(4비트 이하)로 표현 가능하며, 그중 44.48%는 0인 것으로 나타났습니다.

### 2. 제안 솔루션

* Ditto 알고리즘: 첫 타임스텝만 전체 비트 너비로 수행하고, 이후 단계는 시간적 차이값만을 이용해 연산합니다. 이를 통해 0 값은 건너뛰고(Zero skipping) 낮은 비트 너비로 연산하여 효율을 높입니다.
* Defo (Ditto Execution Flow Optimization): 차이값 연산 시 발생하는 메모리 오버헤드를 줄이기 위해, 각 레이어별로 차이값 연산이 유리한지 동적으로 판단하여 최적의 실행 흐름을 결정합니다.
* Ditto 하드웨어: 단일 연산기(PE) 설계 내에서 동적 희소성(Sparsity)과 가변 비트 너비를 동시에 지원하는 전용 가속기입니다.

### 3. 연구 성과

* 기존 가속기 대비 최대 1.5배의 속도 향상과 17.74%의 에너지 절감을 달성했습니다.이미지 품질(FID, IS 등)을 떨어뜨리지 않으면서도 연산 효율성을 극대화했습니다.

---

## 2. Diffusion Model

### A. 디퓨전 모델의 기초 (Preliminaries) 

* 순방향 확산: 원본 이미지에 노이즈를 반복적으로 주입하여 완전한 가우시안 노이즈로 만드는 과정입니다. 
* 역방향 확산: 노이즈에서 시작하여 이를 반복적으로 제거함으로써 이미지를 복원 및 생성하는 과정입니다. 
* 노이즈 제거 모델 (Denoising Model): 역방향 프로세스에서 노이즈를 줄이기 위해 순차적으로 연결된 블록(Block)들로 구성된 신경망을 사용합니다. 일반적으로 ResNet 블록과 Attention 블록이 사용되지만, 최근에는 Transformer 블록만을 사용하는 DiT나 Latte 같은 모델도 등장하고 있습니다. 
* 재귀적 구조: 이전 타임스텝( $T_t$ )의 출력이 현재 타임스텝( $T_{t-1}$ )의 입력으로 다시 들어가는 재귀적 피드백 메커니즘을 가집니다. 
* 문제점: 이러한 반복적 특성과 높은 계산 요구량 때문에 병렬화가 불가능하며, 실행 시간이 매우 길고 연산 강도가 높습니다. 

### B. 디퓨전 모델의 값 유사성 (Value Similarity) 

<p align = 'center'>
<img width="1196" height="617" alt="image" src="https://github.com/user-attachments/assets/79da1da8-4ed9-4e74-a8e5-0fbe67a88efa" />
</p>


* 시간적 유사성(Temporal Similarity): 인접한 타임스텝 간의 코사인 유사도를 측정한 결과, 여러 모델에서 평균 0.983이라는 매우 높은 수치를 기록했습니다. 

* 레이어별 일관성: 특정 레이어뿐만 아니라 다양한 레이어와 타임스텝 전체에서 유사도가 0.947 이상으로 유지됨을 확인했습니다. 

* 공간적 유사성과의 비교: 기존 이미지 처리 신경망에서 주로 활용하던 공간적 유사성(Spatial Similarity)은 평균 0.31 수준으로 나타났습니다. 이는 시간적 유사성이 공간적 유사성보다 훨씬 강력하며, 디퓨전 모델 가속화를 위한 더 큰 기회를 제공함을 의미합니다. 


---

## 3. Motivation

<p align = 'center'>
<img width="589" height="296" alt="image" src="https://github.com/user-attachments/assets/50ed1e37-acd4-4f40-86e0-d1b96ee2e957" />
</p>

### 3.A. 인접 타임스텝 간의 값 차이 (Value Differences)

* 실험 데이터: 특정 레이어(conv-in) 분석 결과, 원래 활성화 값의 범위는 평균 4.73이었으나, 타임스텝 간 차이값의 범위는 0.23에 불과했습니다. 
* 평균 수치: 다양한 디퓨전 모델 전체를 분석했을 때, 시간적 차이값의 범위는 원래 활성화 값보다 평균 8.96배 좁았습니다. 
* 일관성: 이러한 현상은 특정 시점이 아니라 모든 타임스텝에서 일관되게 나타났습니다. 

### 3.B. 좁은 값 범위의 이점 (Advantages)

* 값의 범위가 좁아지면 양자화(Quantization) 과정에서 데이터를 표현하는 데 필요한 비트(Bit) 수를 획기적으로 줄일 수 있습니다. 

#### 1. 비트 너비(Bit-width) 요구량 감소

* 8비트 양자화 모델을 기준으로 분석한 결과는 다음과 같습니다.
* Zero 비율: 시간적 차이값 중 44.48%가 0으로 나타났습니다. 이는 유사한 값들이 양자화 과정에서 동일한 값으로 묶이기 때문입니다. 
* 저정밀도 연산: 0을 포함하여 4비트 이하로 표현 가능한 데이터가 전체의 96.01%를 차지했습니다. 
* 대조군 비교: 원래 활성화 값이나 공간적 차이(Spatial Difference) 방식에서는 4비트를 초과하는 데이터 비중이 훨씬 높았습니다.
    * 공간적 차이(Spatial Difference)
        * 이미지 처리 신경망에서 공간적 유사성을 활용하는 가속기인 Diffy의 메커니즘을 기본 모델로 사용
        * Diffy는 컨볼루션(Convolution) 레이어의 슬라이딩 윈도우 간 유사성만을 다루지만, 본 논문에서는 디퓨전 모델의 특성을 고려하여 완전 연결(Fully Connected) 레이어와 어텐션(Attention) 레이어까지 확장
        * 공간적 차이는 현재 타임스텝의 하나의 텐서 내부에서 인접한 데이터 시퀀스 간의 차이를 계산하는 방식
        * 원본 데이터: $[120, 122, 121, 125, 124]$
            * 공간적 차이 데이터: $[120, 2, -1, 4, -1]$



#### 2. 실제 연산량(BOPs) 절감 효과

<p align = 'center'>
<img width="583" height="162" alt="image" src="https://github.com/user-attachments/assets/41196776-5a36-4e06-ba72-5a46e1b6a77b" />
</p>

* 비트 연산량(BOPs, Bit Operations)을 분석했을 때, 시간적 차이값을 활용하는 방식의 효율성이 입증되었습니다. 
* 연산량 감소: 기존 양자화 모델 대비 평균 53.3%, 공간적 차이 방식 대비 23.1% 더 적은 BOPs로 연산이 가능합니다. 
* 모델별 차이: 특히 DDPM이나 CHUR 같은 모델에서는 0의 비율이 더 높아 각각 68.8%, 71.5%의 높은 연산량 절감 효과를 보였습니다. 
* 지속성: 이미지 생성의 마지막 단계에서 노이즈 제거량이 많아지며 절감 폭이 다소 줄어들긴 하지만, 전체 과정을 통틀어 일관된 성능 향상을 유지합니다. 


