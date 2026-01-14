# Diffusion Model Quantization: A Review

저자 : Qian Zeng, Chenggong Hu, Mingli Song, Jie Song

Zhejiang University

출간 : arXiv preprint arXiv:2505.05215, 2025 

논문 : [PDF](https://arxiv.org/pdf/2505.05215)

---
## 1. Introduction

### 기존의 해결 노력

* 샘플러 최적화: 역시간 SDE나 ODE를 수치적으로 풀어 샘플링 단계 자체를 줄이려는 시도입니다.
* 모델 압축: 가지치기(Pruning), 지식 증류(Distillation), 그리고 양자화(Quantization) 기법이 적용되고 있습니다.
    * 가지치기: 가중치 구조를 파괴하여 처음부터 재학습해야 하는 경우가 많습니다.
    * 지식 증류: 성능은 좋으나 막대한 데이터와 계산 자원(예: 199 A100 GPU days)이 필요합니다.
    * 양자화: 표현의 충실도와 계산 효율성 사이에서 효과적인 균형을 맞추며, 특히 엣지 디바이스 배포를 위한 가속화 솔루션으로 주목받고 있습니다

### 양자화 연구의 흐름과 한계

* 초기 연구: 가우시안 분포 기반 타임스텝 샘플링을 통한 보정(Calibration) 기법인 PTQ4DM이 기반을 닦았습니다.
* 발전된 기법: 타임스텝별 동적 양자화(TDQ), 미분 가능한 타임스텝 그룹화, 비디오 생성을 위한 시간적 정렬 등이 제안되었습니다.
* 구조별 대응: U-Net뿐만 아니라 확산 트랜스포머(DiT)의 특성에 맞춘 그룹 양자화 및 채널 평활화 기법들이 등장했습니다.


### 4. 주요 양자화 기법 (Key Quantization Methods)

<img width="830" height="707" alt="image" src="https://github.com/user-attachments/assets/cd7e7a59-e1cf-41eb-82ba-29d47cb12d45" />



* 초기 및 기반 기술
    * 가우시안 분포 기반 타임스텝 샘플링(PTQ4DM) [40]
    * 분포 정렬 보정(Liu et al.) [50]
* 시간적 동적 양자화 (Temporal/Dynamic)
    * 단계별 활성화 양자화(TDQ) [45]
    * 타임스텝 그룹화(Wang et al.) [51]
    * 시간적 특징 유지(TFMQ-DM) [52]
    * 비디오 생성을 위한 시간 정렬(Tian et al.) [53]
* 학습 기반 및 미세 조정 (QAT/LoRA)
    * 양자화 인식 학습(Q-DM [54]
    * QuEST [55], MEFT-QDM [56])
    * LoRA 기반의 4비트 양자화(QaLoRA [57], IntLoRA [58])
* 극단적 양자화 (Binary/Mixed-Precision)
    * 1~2비트 이진화 및 초저비트 연구(BLD [59], BinaryDM [60], BiDM [61])
    * 혼합 정밀도 전략(BitsFusion [62], BDM [63])
* 오류 보정 및 트랜스포머 대응
    * 오류 보정 메커니즘(PTQD [44], $D^2$-DPM [46], Tac-QDM [64])
    * 확산 트랜스포머(DiT) 양자화 기법(A-QDiT [66], Q-DiT [67], PTQ4DiT [68], DIT-AS [69], ViDiT-Q [70], HQ-DiT [71])

---

## 2. Background and Preliminary

### 2.1 Diffusion Models

<p align = 'center'>
<img width="777" height="325" alt="image" src="https://github.com/user-attachments/assets/1ac36d01-bd63-4c7b-8373-6526c8343f7d" />
</p>

* (a) 시간 단계에 따른 양자화 노이즈 누적 (Error Accumulation)
* (b) 시간 단계별 활성화 범위의 변화 (Activation Distribution Shift)
* (c) 시간적 특징 불일치 현상 (Temporal Feature Mismatch)
    * 양자화된 시간 임베딩( $\hat{emb}_t$ )이 원래의 시간 단계 $t$보다 다른 단계( $t+\delta_t$ )의 임베딩과 더 높은 유사성을 보이는 현상을 설명합니다.
    * 그래프의 파란색 곡선에 나타난 변곡점들은 양자화로 인해 시간 정보가 혼동되어, 모델이 현재 단계를 부정확하게 인식하게 함으로써 샘플링 경로가 꼬이거나 역행하는 문제를 유발함을 시사합니다.

### 2.2 Model Quantization

#### 균일 양자화(Uniform quantization)

$$x_{int}=clamp(\lfloor\frac{x}{s}\rfloor+z, 0, 2^{b})\quad(9)$$

#### Post-Training Quantization, PTQ

* 추가적은 훈련이 필요없다.
* 추가 데이터가 전혀 없거나, 아주 적은양의 캘리브레이션 데이터셋(calibration dataset)으로 수행 가능

#### Quantization-Aware Training, QAT

* 훈련 과정 중에 양자화 오류를 모델링하기 위해 모의 양자화(simulated quantization)를 도입
* 4비트 이하의 저비트 양자화에서도 우수한 성능
* 역전파(backpropagation) 시 미분이 불가능한 라운딩 연산을 처리하기 위해 Straight-Through Estimator (STE)를 사용하여 기울기(gradient)를 근사화

### 2.3 Challenges in Quantizing Diffusion Models

#### 2.3.1 Challenges from the Diffusion Mechanism

* C#1: 타임스텝별 활성화 분포의 변화 (Activation distributions vary across time steps)
    * 디퓨전 과정의 각 타임스텝 $t$마다 입력 데이터의 분포가 계속해서 변합니다.
    * 이로 인해 층별(layer-wise) 활성화 값들이 시간에 따라 크게 달라져 양자화 난이도를 높입니다.
* C#2: 타임스텝 간 양자화 오차의 누적 (Quantization errors accumulate across time steps)
    * 단일 단계의 오차가 여러 층을 거치며 전파될 뿐만 아니라, 디퓨전의 반복적인 특성상 샘플링 타임스텝을 따라 오차가 점진적으로 쌓입니다.
* C#3: 시간 정보 혼란 현상 (Temporal confusion phenomenon)
    * 타임 임베딩(TimeEmbedding) 레이어를 양자화하면 실제 시간과 양자화된 시간 표현 사이에 불일치가 생겨 샘플링 궤적이 꼬이거나 역행하는 등의 혼란이 발생할 수 있습니다.

#### 2.3.2 Challenges from Model Architectures

* C#4: Concatenate 연산으로 인한 이봉 분포 (Bimodal data distribution from concatenate operations)
    * U-Net의 스킵 커넥션(shortcut layers)에서 얕은 층의 특징과 깊은 층의 특징을 직접 결합(concatenate)합니다.
    * 채널 간 데이터 범위 차이가 크기 때문에 가중치와 활성화 값들이 두 개의 정점을 가진 이봉 분포를 보이게 되며, 이는 불균형한 압축과 특징 표현의 불일치를 초래합니다.

* C#5: Cross Attention Quantization으로 텍스트-시각적 특징 정렬 불일치 (Misalignment of textual and visual features)
    * U-Net 기반 텍스트-이미지 모델은 크로스 어텐션(cross-attention) 모듈을 통해 텍스트 정보를 주입합니다.
    * 양자화 과정에서 이 정렬이 깨지면 모델이 멀티모달 정보를 정확하게 통합하지 못해 생성 품질이 저하됩니다.

* C#6: 다층적 데이터 변동성 (Data Variance Across Multiple Levels)
    * DiT는 토큰 기반 Vision Feature을 사용하는데, 이는 시간적·공간적으로 변동성이 매우 큽니다.
    * 글로벌 어텐션, 텍스트 조건부 어텐션 등 다양한 모듈 간의 가중치 격차와 타임스텝 조건에 따른 활성화 패턴 변화가 복합적으로 작용하여 양자화의 정밀도를 떨어뜨립니다.

---

##

---

##

---
