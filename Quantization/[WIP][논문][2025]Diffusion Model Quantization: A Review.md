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
    * Calibration dataset: 활성화 값(Activation) 분포 파악

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

<p align = 'center'>
<img width="675" height="831" alt="image" src="https://github.com/user-attachments/assets/c7086117-1706-4298-bd72-660d28e854d0" />
</p>

---

## 3. A Taxonomy of Quantization Algorithms for Diffusion Models

### 3.1 Unet-Based Diffusion Model Quantization

### 3.1.1 PTQ-Based Approaches

* BRECQ를 베이스라인으로 삼아 블록 단위 재구성(block-wise reconstruction) 메커니즘을 적용

#### 1. 캘리브레이션 전략 최적화 (Calibration Strategy)

* PTQ4DM: 역방향 샘플링 궤적을 활용하여 캘리브레이션 세트를 구성하는 것이 효과적임을 확인했습니다.
    * 특히 타임스텝이 $t_0$에 가까울수록 효과가 좋다는 점에 착안하여, $t$를 정규 분포로 샘플링하는 NDTC(Normally Distributed Time-step Calibration) 방식을 제안했습니다.
* EDA-DM (TDAC): 각 타임스텝 샘플의 '대체 불가능성'을 밀도(Density)와 다양성(Variety) 점수로 평가하여 샘플링 밀도를 결정함으로써 분포 정렬을 개선했습니다.

#### 2. 이봉 분포 제거 및 모델 구조 대응

* Q-Diffusion: U-Net의 스킵 커넥션에서 발생하는 이봉 분포(Bimodal distribution) 문제를 해결하기 위해, 결합(Concatenate) 연산 직전에 양자화를 먼저 수행하는 'Split' 양자화 전략을 제안

#### 3. 시간적 동적 양자화 (Temporal Dynamic Quantization)

* 모든 타임스텝이 동일한 양자화 파라미터를 공유할 때 발생하는 성능 저하(C#1)를 해결하기 위한 기술

* TDQ: 각 타임스텝마다 별도의 활성화 양자화 파라미터를 학습하는 방식입니다.
    * 얕은 다층 퍼셉트론(MLP)을 사용하여 입력 $t$에 따른 스케일 팩터 $s_a$를 예측하며, 추론 전 미리 계산해둘 수 있어 추가 오버헤드가 없습니다.
* APQ-DM: 타임 스케줄을 여러 그룹으로 나누고, 미분 가능한 그룹 할당 전략을 통해 최적의 양자화 함수를 각 그룹에 할당합니다

#### 4. 시간 정보 유지 (Time Information Maintenance)

* 양자화로 인해 시간 임베딩 정보가 왜곡되어 샘플링 궤적이 이탈하는 문제(C#3)를 다룹니다

* TFMQ-DM (TIAR): 시간 임베딩 레이어들을 '시간 정보 블록'으로 통합하고, 양자화 전후의 특징 perturbation을 최소화하는 정렬 손실 함수를 도입했습니다.
* QVD: 비디오 생성 모델에서 시간적 변별력을 평가하는 TDScore를 도입하여, 시간 정보의 고유성을 유지하며 양자화 파라미터를 최적화합니다.

#### 5. 양자화 오차 보정 (Quantization Error Correction)

* 반복적인 노이즈 제거 과정에서 누적되는 양자화 노이즈(C#2)를 수학적으로 보정
* PTQD: 양자화 노이즈를 원래 출력과 상관관계가 있는 성분과 없는 성분으로 분해하여 각각 보정합니다.
* $D^2$-DPM: 양자화 노이즈를 가우시안 분포로 모델링하고, 샘플링 공식에서 평균 편향(Mean bias)을 빼고 분산 편향(Variance bias)을 조절하는 이중 노이즈 제거(Dual Denoising) 메커니즘을 제안.
* TaC-QDM: 노이즈 추정치의 오차와 입력 편향(Input bias)을 분리하여 각각 보정함으로써 오차 전파를 차단합니다.

### 3.1.2 QAT-Based Approaches

#### 1. 통합적 QAT 최적화 (Holistic QAT Optimization)

* 낮은 비트의 양자화에서 발생하는 활성화 분포의 흔들림(C#1)과 오차 누적(C#2)을 해결하기 위한 포괄적인 최적화 방법
* Q-DM: 활성화 분포의 진동을 완화하기 위해 타임스텝 인식 양자화(TaQ)를 제안하고, 노이즈 추정 모방(NeM) 목적 함수를 설계하여 양자화 모델이 풀프레시전(FP) 모델의 출력을 따르도록 정렬합니다.
* QuEST: 저비트에서 테일러 급수 근사가 깨지는 문제를 해결하기 위해 점진적 선택 미세 조정(Progressive Selective Fine-Tuning)을 사용하여 양자화에 민감한 모듈을 먼저 학습한 후 전체 모듈을 튜닝합니다.
* TuneQDM: 채널별 가중치 패턴 변화를 조정하기 위해 다중 채널 단위 스케일 업데이트(Multi-Channel-Wise Scale Update)를 도입한 경량 QAT 프레임워크입니다.

#### 2. 초저비트 디퓨전 모델 (Ultra-Low-Bit DMs)

* BinaryDM: 가중치 이진화로 인한 표현력 붕괴를 막기 위해 진화 가능한 베이스 이진화기(EBB)와 저랭크 표현 모방(LRM) 기술을 사용하여 W1A4 수준을 달성했습니다.
    * 진화 가능한 베이스 이진화기 (EBB, Evolvable-Basis Binarizer)
        * 이진화 과정에서 발생하는 극심한 오차 누적을 방지하기 위해 제안되었습니다.
        * 수학적 공식: 다음과 같이 두 개의 학습 가능한 스칼라( $\sigma_{I}, \sigma_{II}$ )를 사용하여 가중치를 근사합니다
        * $$w_{EBB}^{bi}=\sigma_{I}sign(w)+\sigma_{II}sign(w-\sigma_{I}sign(w))$$
    * 저랭크 표현 모방 (LRM, Low-Rank Representation Mimicking)
        * 주성분 분석(PCA) 활용: 이진화된 모델의 표현 한계를 고려하여, 중간 레이어의 특징들을 먼저 PCA를 통해 분해합니다.
        * 저랭크 공간 정렬: 풀프레시전(FP) 모델의 구성 요소와 양자화된 모델의 대응 구성 요소를 저랭크 공간(low-rank space) 내에서 미세하게 정렬합니다.  
* BiDM: W1A1이라는 이론적 한계에 도전하며, 타임스텝 친화적 이진 구조(TBS)와 공간 패치 증류(SPD)를 통해 이진 특징과 풀프레시전 특징을 정렬합니다.
    * 타임스텝 친화적 이진 구조 (TBS, Timestep-friendly Binary Structure)
        * 학습 가능한 스케일링 매칭: 기존 XNOR-Net 방식의 컨볼루션 연산에서 고정되어 있던 행렬 $K$를 학습 가능하도록 설계하여, 이진화된 가중치와 입력 사이의 근사 정밀도를 높입니다.
        * 보상 표현력 강화: DeepCache매커니즘 내에 학습 가능한 모멘텀 업데이트(learnable momentum updates)를 통합하여, 이진 모델의 부족한 표현 능력을 보완하고 특징 연결성을 강화합니다
    * 공간 패치 증류 (SPD, Space Patched Distillation)
        * 이진 모델(Binary)과 풀프레시전(FP) 모델 사이의 특징을 일치시키기 어려운 문제를 해결하기 위해 제안된 증류(Distillation) 기법입니다.
        * 패치 분할(Patching): 풀프레시전 블록과 양자화 블록에서 출력된 중간 특징들을 $p^2$개의 패치로 나눕니다.
        * 어텐션 기반 손실 계산: 각 패치별로 어텐션 가이드 손실(attention-guided loss)을 개별적으로 계산합니다. 이는 특징 맵 전체를 한 번에 비교하는 것보다 더 세밀한 정렬을 가능하게 합니다.
        * 손실 누적 및 정규화: 각 패치에서 얻은 오차를 정규화하고 합산하여 최종 증류 손실을 도출하며, 이를 통해 이진화된 특징이 고정밀 특징을 최대한 모방하도록 유도합니다.
* BitsFusion: 대규모 텍스트-이미지 모델을 위해 혼합 정밀도 최적화를 적용한 1.99비트 가중치 양자화 프레임워크를 제안했습니다.
* BI-DiffSR: 이미지 초해상도(SR)를 위해 타임스텝 인식 재분배(TaR)를 적용한 이진화 디퓨전 모델입니다.
    * (TaR: Timestep-aware Redistribution)
    * TaA와의 결합: 타임스텝 인식 활성화 함수(TaA)와 결합하여, 각 시간 단계에 가장 적합한 활성화 패턴과 데이터 배치를 형성함으로써 생성 이미지의 품질을 개선

### 3.2 Diffusion Transformer Quantization

### 3.2.1 PTQ-Based Approaches

#### 1. 그룹별 양자화 (Group-wise Quantization)
* Yang 등 [66]: 1단계 캘리브레이션(1-step calibration) + 가중치 그룹 양자화(Weight Group Quantization)
    * 역확산의 시작점인 노이즈 단계를 1단 Calibration data로 사용
* Chen 등 [67] (Q-DiT): 가중치를 입력 채널을 따라 양자화할 것을 제안, FID 점수를 최소화하는 최적의 그룹 크기를 결정하는 자동 그룹 할당(Automatic Group Allocation)

#### 2. 채널 평활화 (Channel Equalization)
* Wu 등 [68] (PTQ4DiT): 채널별 중요도 균형화(CSB)와 Spearman의 $\rho$ 기반 중요도 캘리브레이션(SSC)을 제안했습니다
    * 채널별 중요도: 최대 절대값을 중요도로 정의
    * Spearman’s $\rho$: 모든 타임스텝의 활성화 중요도를 동일하게 취급하지 않고, 가중치 중요도와의 상관관계(Spearman’s $\rho$)를 분석하여 타임스텝별 가중치( $\eta_t$ )를 부여
    * 캘리브레이션 1단 사용
* Dong 등 [69] (DITAS): 여러 타임스텝에 걸친 이상치 정보를 집계하여 채널 재균형을 달성하는 시간 집계 평활화(TAS)를 제안했습니다
    * TAS의 주요 목표는 활성화(activation)에 집중된 양자화 난이도를 가중치(weight)로 분산시켜 전체적인 양자화 오차를 줄이는 것
* Zhao 등 [70] (ViDiT-Q): 활성화에는 토큰별 양자화를, 가중치에는 채널별 양자화를 적용했습니다. 초기 단계는 스케일 기반 방식을, 후기 단계는 회전 기반 방식을 사용하는 전용 솔루션을 개발했습니다.
* Liu 등 [71] (HQ-DiT): DiT 추론에서 가중치와 활성화 모두에 4비트 부동 소수점(FP4) 정밀도를 적용한 최초의 사례입니다.
    * Hadamard 변환을 적용하여 데이터 이상치를 크게 줄이고, 지수 비트와 가수 비트의 할당을 동적으로 결정합니다.
* Li 등 [141] (SVDQuant): 이상치를 재분배하는 대신, 저계수(low-rank) 분기를 통해 이상치를 흡수하는 4비트 양자화 패러다임입니다. 오버헤드를 줄이기 위해 Nunchaku 추론 엔진을 공동 설계하여 메모리 액세스를 최적화했습니다

### 3.2.2 QAT-Based Approaches

* DiT 모델은 파라미터 사이즈가 매우 크기 때문에, 훈련 과정에서 막대한 계산 능력과 리소스가 요구

#### 향후 연구 방향 및 전망

* QAT수행 시 메모리 부족 (DiT)
* 메모리 압축 기술과의 통합: 향후 연구는 고급 메모리 압축 기술을 통해 QAT의 효율성을 최적화하는 데 집중될 것으로 보입니다.
* 실용적인 통합: 메모리 활용도를 높임으로써 QAT 수행 시 발생하는 리소스 부담을 줄이고, DiT 모델에 더 실용적이고 효율적으로 QAT를 통합할 수 있는 경로를 모색해야 합니다.



---

## 4 Benchmarking Experiments

---
