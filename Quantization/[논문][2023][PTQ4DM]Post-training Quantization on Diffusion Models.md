# Post-training Quantization on Diffusion Models

저자 : Yuzhang Shang1,4*, Zhihang Yuan2*, Bin Xie1, Bingzhe Wu3, Yan Yan1†

1 Illinois Institute of Technology, 2Houmo AI, 3Tencent AI Lab, 4Cisco Research

출간 : CVPR(	Computer Vision and Pattern Recognition), 2023

논문 : [PDF](https://arxiv.org/pdf/2211.15736)

---

## 1. Introduction

<p align = 'center'>
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/7ad236d9-8092-4571-88f2-cc9ae7fe77d3" />
</p>

### 배경 및 문제점

* 느린 속도의 원인: 논문은 속도 저하의 원인을 두 가지로 분석합니다.
    * 노이즈를 제거하는 반복 과정(iteration)이 너무 깁니다.
    * 각 반복 단계에서 노이즈를 추정하는 신경망 연산이 매우 무겁습니다

### 기존 연구의 한계

* 반복 횟수(step)를 줄이는 데에만 집중
* 반복마다 수행되는 무거운 신경망의 연산 비용 자체를 줄이는 문제는 간과

### 제안하는 해결책: 학습 후 양자화 (PTQ)

* PTQ 선택 이유: 재학습(Retraining)이나 미세 조정(Fine-tuning)이 필요한 압축 방식 대신 학습 후 양자화(PTQ)를 선택
    * 데이터 접근성: Dall-E2와 같은 거대 모델의 학습 데이터는 개인정보 보호나 상업적 이유로 공개되지 않는 경우가 많습니다.
    * 비용: 모델을 재학습하는 데 막대한 GPU 자원과 시간이 소모


### 기술적 난관과 혁신 (PTQ4DM)

* 기존 PTQ의 문제점
    * 일반적인 PTQ 방식은 단일 시간 단계(single-time-step)를 가정하고 설계
    * 확산 모델은 시간 단계(time-step)에 따라 노이즈 추정 네트워크의 출력 분포가 계속 변함
    * 따라서 기존 방식을 그대로 적용하면 성능이 크게 떨어집니다.
* 해결책 (NDTC: Normally Distributed Time-step Calibration)
    * 새로운 보정(calibration) 방법을 개발
    * 확산 모델의 다중 시간 단계 구조에 맞춰 최적화된 방식

### 주요 기여 및 성과

* 최초의 시도
    * 확산 모델 가속화를 위해 재학습 없는(training-free) 네트워크 압축을 시도한 최초의 연구입니다.
* 성능
    * 제안된 방식(PTQ4DM)은 전체 정밀도(full-precision) 모델을 8-bit 모델로 양자화하면서도
    * 성능 저하가 거의 없거나 오히려 향상된 결과를 보여줍니다.
* 호환성
    * DDIM과 같은 기존의 고속 샘플링 방식과 함께 사용할 수 있는 플러그 앤 플레이(plug-and-play) 모듈로 작동합니다.

---
## 2. Related Work

### 2.1. Diffusion Model Acceleration

#### 1. 기존 접근 방식(더 짧은 샘플링 경로 탐색)

* 기존의 확산 모델 가속화 연구들은 주로 성능을 유지하면서 더 짧은 샘플링 경로(shorter sampling trajectories)
* 즉 반복 횟수(time-steps)를 줄이는 방법을 찾는 데 집중


##### 주요 연구 사례
* Chen et al.: 그리드 서치(grid search)를 통해 6단계만의 짧은 경로를 찾았지만, 시간이 기하급수적으로 늘어나 긴 경로에는 적용하기 어렵습니다. (탐색시간이 너무 길다)
* Watson et al.: 경로 탐색을 동적 프로그래밍(dynamic programming) 문제로 모델링했습니다.
* Song et al. (DDIM): 학습 목표는 같지만 역방향 프로세스(reverse process)에서 샘플링 속도가 훨씬 빠른 비마르코프(non-Markovian) 프로세스를 구성했습니다.
* ODE/SDE Solver 활용: 확산 모델을 미분방정식(ODE/SDE) 형태로 정의하고, 더 빠른 솔버(solver)를 사용하여 효율성을 높였습니다.
* 근사법 및 기타: 몬테카를로 방법을 이용한 근사법 이나, 여러 단계를 한 단계로 압축하는 지식 증류(Knowledge Distillation) 방식 등이 제안되었습니다.

##### 기존 방식의 한계점
대부분의 기존 방식들(예: 지식 증류 등)은 사전 학습된 모델을 얻은 후에도 **추가적인 학습(additional training)**이 필요합니다. 이는 많은 자원을 소모하며, 훈련 데이터 접근이 어려운 상황에서는 사용하기 어렵습니다.

##### 본 연구의 차별점: 네트워크 압축 (Network Compression)

* 이 논문은 기존 연구들과 달리 반복 횟수를 줄이는 것이 아니라, 각 반복 단계에서 실행되는 네트워크 자체를 압축하는 데 초점을 맞춥니다.
* 따라서 기존의 고속 샘플링 방법(예: DDIM) 위에 플러그 앤 플레이(plug-and-play) 모듈처럼 추가하여 함께 사용할 수 있습니다.
* 최초의 시도: 저자들의 지식에 따르면, 확산 모델에 대해 학습 후 양자화(Post-Training Quantization)를 적용한 첫 번째 연구입니다.


### 2.2. Post-training Quantization

$$x_{int}=clamp(\lfloor\frac{x}{s}\rceil-z,p_{min},p_{max}) \quad (1)$$


#### 양자화 오차 최소화
* PTQ의 목표는 양자화 전의 텐서( $X_{fp}$ )와 양자화 후의 텐서( $X_{sim}$ ) 사이의 오차($L_{quant}$)를 최소화하는 파라미터($s, z$)를 선택하는 것
* 가장 효과적이고 널리 쓰이는 방법은 두 텐서 간의 **평균 제곱 오차(MSE)**를 최소화하는 것
* 기타 지표: L1 거리, 코사인 거리(Cosine distance), KL 발산(KL divergence) 등도 사용될 수 있습니다

#### 보정(Calibration)의 중요성
* 가중치(Weights)는 바로 양자화할 수 있지만, 네트워크의 활성화(Activations) 값을 계산하기 위해서는 입력 데이터가 필요합니다.
* 보정 데이터셋 (Calibration Dataset): PTQ는 소수의 라벨이 없는 입력 샘플(예: 128개의 이미지)을 사용하여 네트워크의 활성화 텐서를 수집
* 목적: 실제 데이터의 분포와 유사한 샘플을 사용해야만 적절한 양자화 파라미터를 선택하여 과적합(Overfitting)을 방지

#### 제로샷 양자화 (Zero-shot Quantization, ZSQ)

* 섹션 마지막 부분에서는 실제 데이터에 접근할 수 없는 경우를 위한 제로샷 양자화를 언급
* ZSQ는 실제 이미지 대신, 네트워크 내의 배치 정규화(Batch Normalization) 정보(평균 및 분산 등)를 활용하여 가상의 데이터를 생성
* 확산 모델에서의 한계: 저자들은 확산 모델의 경우 노이즈로부터 이미지를 생성하는 방식이므로, 기존의 ZSQ 방식을 적용하기 어렵다고 지적


---
## 3. PTQ on Diffusion Models

### 3.1. Preliminaries

(Skip)

<p align = 'center'>
<img width="1104" height="362" alt="image" src="https://github.com/user-attachments/assets/2cfb6d2b-fbe3-4c27-9dfa-2aa57bf76140" />
</p>

### 3.2. Exploration on Operation Selection

#### 1. 기본 양자화 대상 선정

* 양자화 대상: U-Net 구조 내의 합성곱(Convolution) 레이어와 완전 연결(Fully-connected) 레이어.
* 참고: 배치 정규화(Batch Normalization)는 합성곱 레이어에 통합(fold)하여 처리합니다.
* 양자화 제외 (Full-precision 유지): Softmax나 SiLU(Sigmoid Linear Unit)와 같은 특수 함수들은 양자화 시 오차가 크게 발생할 수 있고 연산 집약적이지 않으므로 32비트(Full-precision)로 유지합니다.

#### 2. 확산 모델 특화 질문 및 실험
* 질문 1: 네트워크가 출력하는 평균( $\mu$ )과 분산( $\Sigma$ )을 양자화해도 되는가?
* 질문 2: 샘플링된 이미지 $x_{t-1}$ 을 양자화해도 되는가?

<img width="1102" height="742" alt="image" src="https://github.com/user-attachments/assets/aba81f2c-2764-46aa-aa65-2047768afc4b" />


#### 3. 실험 결과 (Table 1 분석)

<img width="544" height="298" alt="image" src="https://github.com/user-attachments/assets/bb485e1c-f6ab-4bc6-8099-cba0f856bc9a" />


* 이 질문에 답하기 위해 저자들은 $\mu$, $\Sigma$, $x_{t-1}$을 생성하는 연산만 개별적으로 양자화하여 성능 변화(FID, IS 점수 등)를 측정했습니다
* 결과: 32비트 모델(FP)과 비교했을 때, $\mu$, $\Sigma$, $x_{t-1}$을 각각 혹은 모두 양자화하더라도 성능 저하가 거의 없었습니다
* 예: FP 모델의 FID는 21.63인데, 세 가지를 모두 양자화했을 때의 FID는 21.99로 큰 차이가 없음7.


#### 4. 결론

* 실험 결과, 확산 모델의 출력값인 $\mu$, $\Sigma$ 및 샘플링된 $x_{t-1}$은 양자화에 민감하지 않음이 확인
* 따라서 저자들은 이 부분들을 생성하는 연산까지 모두 양자화하기로 결정


### 3.3. Exploration on Calibration Dataset

#### 1. 핵심 관찰 결과 (3.3.1 Analysis)

* 관찰 0: 활성화 분포는 타임스텝(Time-step)에 따라 변한다.
    * 노이즈 제거 과정의 시점( $t$ )이 달라지면 신경망 내부의 활성화 값 분포도 크게 달라집니다
    * 따라서 단일 시점만을 고려한 기존의 PTQ 방식은 확산 모델에 적용할 수 없습니다
* 관찰 1: '디노이징 과정(Denoising Process)'에서 생성된 샘플이 더 효과적이다.
    * 실제 이미지(Raw images)나 확산 과정(Diffusion process)을 흉내 낸 샘플보다, 노이즈( $x_T$ )에서 출발하여 디노이징 과정을 거쳐 생성된 샘플( $x_t$ )을 보정 데이터로 썼을 때 성능이 가장 좋았습니다
    * 결국 양자화 파라미터(스케일, 영점 등)를 맞출 때, 모델이 한 번도 볼 일 없는 '완벽한 노이즈 이미지(이미지 + 노이즈)'를 기준으로 맞추는 것보다, 모델이 실제로 생성하면서 마주치게 될 '약간은 불완전한 노이즈 이미지(디노이징 과정)'를 기준으로 맞추는 것이 훨씬 정확
    * 모델이 실제로 생성하면서 마주치게 될 약간은 불완전한 노이즈 이미지를 기준으로 맞추는게 훨씬 정확하다
* 관찰 2: 실제 이미지( $x_0$ )에 가까운 샘플일수록 보정에 유리하다.
    * 타임스텝 $t$가 작을수록(즉, 노이즈가 많이 제거되어 실제 이미지에 가까울수록) 해당 시점의 데이터가 양자화 파라미터 보정에 더 중요한 역할을 합니다
* 관찰 3: 단일 시점보다는 '다양한 타임스텝'의 샘플이 필요하다.
    * 모든 샘플을 같은 타임스텝에서 뽑는 것보다, 여러 타임스텝에 걸쳐 다양하게 뽑는 것이 분포의 변화를 반영할 수 있어 성능이 좋습니다


#### 2. 제안 방법: NDTC (3.3.2 Normally Distributed Time-step Calibration)

<p align = 'center'>
<img width="538" height="322" alt="image" src="https://github.com/user-attachments/assets/868b1300-b39b-4258-ad03-5fc8627d06e5" />
</p>

이 방법은 서로 상충할 수 있는 조건들(실제 이미지에 가까워야 함 vs 다양한 시점을 커버해야 함)의 균형을 맞춥니다

* 작동 방식
    * 타임스텝 샘플링: 타임스텝 $t$를 균등하게 뽑지 않고, 정규 분포(Normal Distribution)를 따르도록 샘플링
        * 이때 평균( $\mu$ )을 조절하여 실제 이미지에 가까운(작은 $t$ ) 쪽의 샘플이 더 많이 뽑히도록 하되, 전체 범위를 아우르도록 합니다
    * 데이터 생성: 선택된 타임스텝 $t$에 맞춰, 전체 정밀도(Full-precision) 모델을 이용해 노이즈로부터 $x_t$를 생성하여 보정 데이터로 사용합니다

* 타임스텝 샘플링("중요한 단원을 집중적으로 공부하자")
    * 논문에서 타임스텝 $t$를 균등하게(Uniformly) 뽑지 않고, 정규 분포(Normal Distribution)를 써서 한쪽으로 치우치게 뽑는다
        * 이유는 "효율성" 때문
    * 상황: 확산 모델의 디노이징 과정은 $x_T$(완전 노이즈)에서 시작해서 $x_0$(완전한 이미지)로 가는 긴 여정(예: 1000단계)
    * 앞선 관찰(Observation 2)에서, "이미지가 거의 완성되어 가는 후반부($t$가 0에 가까울 때)"의 데이터가 양자화 품질에 훨씬 더 중요하다는 것을 발견했
    * 완전 노이즈 상태($t$가 클 때)는 정보가 별로 없어서 모델 보정에 큰 도움이 안 되기 때문입니다.
    
#### 3. 결론

이 NDTC 방식을 사용하여 보정 데이터셋을 구성한 결과, 8비트로 양자화된 확산 모델이 전체 정밀도(32비트) 모델과 거의 대등하거나 심지어 더 우수한 성능(FID, IS 점수 기준)을 보여주었습니다

### Algorithm
<p align = 'center'>
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/f2f154ef-81fc-4709-96c6-63e595ad676f" />
</p>

* 입력(Input)
    * $N$: 수집할 보정 데이터의 개수 (예: 1024개)
    * $\mu$: 정규 분포의 평균 (어느 타임스텝을 중점적으로 뽑을지 결정)
    * $p_{\theta}(x_{t-1}|x_t)$: 학습이 완료된 32비트(Full-precision) 확산 모델

1. 타임스텝($t_i$) 뽑기 (Lines 3-5)
* 먼저, $N$개의 데이터를 만들기 위해 각각 어느 시점( $t$ )의 데이터를 만들지 결정
* 샘플링: 정규 분포 $\mathcal{N}(\mu, \frac{T}{2})$를 따르는 난수 $t_i$를 하나 뽑습니다
    * 균등 분포(Uniform)가 아니라 정규 분포를 쓰는 이유는, 실제 이미지에 가까운 중요한 구간을 더 자주 뽑기 위해서입니다.
 * 정수 변환: 뽑힌 $t_i$는 실수이므로, floor 함수를 써서 정수로 내림합니다
 * 범위 제한(Clamp): 값이 너무 크거나 작으면 안 되므로, $0$과 $T$(최대 타임스텝) 사이의 값으로 고정

2. 해당 시점까지 데이터 생성하기 (Lines 7-10)
* 결정된 타임스텝 $t_i$에 해당하는 데이터를 실제로 만듭니다. 여기서 '실제 디노이징 과정'을 사용하는 것이 핵심입니다.
* 초기화: 완전한 가우시안 노이즈 $x_T$를 생성합니다
* 디노이징 루프: $T$(끝)부터 시작해서 우리가 목표한 $t_i$가 될 때까지 모델을 돌립니다
    * $x_T \rightarrow x_{T-1} \rightarrow \dots \rightarrow x_{t_i}$
    * 이 과정에서 학습된 모델 $p_{\theta}$를 사용하여 한 단계씩 노이즈를 제거합니다
* 스냅샷: 루프가 $t_i$에 도달하면 멈추고, 그때의 결과물 $x_{t_i}$를 가져옵니다.

3. 수집 완료 (Line 13)
* 위 과정을 $N$번 반복하여 모은 데이터들의 집합 $\mathcal{C}={x_{t_i}}_{i=1}^{N}$을 최종 보정 데이터셋으로 출력

### Exploration on Parameter Calibration
<p align = 'center'>
<img width="549" height="231" alt="image" src="https://github.com/user-attachments/assets/27130db1-2411-4ef7-8766-112bc77cdd88" />
</p>

* MSE를 양자화 보정 지표로 채택

---

## 4. More Experiments

* (NDTC 보정 + MSE 지표)을 사용하여, "이 방법이 정말 다양한 상황에서도 잘 작동하는가?"를 증명하는 확장 실험

<p align = 'center'>
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/9d102b87-288d-4c0f-8f24-1a04da246ea0" />
</p>

* 흥미로운 발견 (DDPM 성능 향상)
    * DDPM(4000 steps)의 경우, 8비트 양자화 모델이 원본 32비트 모델보다 오히려 더 좋은 성능
    * CIFAR10 DDPM: FP(FID 9.28) vs PTQ4DM(FID 9.55 -> 오타가 아니라면 표에서는 PTQ4DM의 FID가 7.10으로 더 낮게(좋게) 나옴).
    * 양자화가 불필요한 정보를 제거하면서 오히려 노이즈 제거 능력이 효율화되었을 가능성을 시사


<p align = 'center'>
<img width="400" height="180" alt="image" src="https://github.com/user-attachments/assets/912b2b1d-9024-4132-94f6-ec2fe6972d78" />
</p>

* 속도 향상 (Appendix Table 7 참고)
    * 8비트 양자화를 통해 NVIDIA GPU 기준 약 2배의 추론 속도 향상


---


## Appendix

#### BRECQ(Block Reconstruction Quantization)

* BRECQ와 같은 기존 방법들은 이미지 분류(CNN)처럼 입력이 한 번 들어가서 결과가 나오는 '단일 타임스텝(Single-time-step)' 모델을 위해 설계

* 이 방식들은 데이터의 분포가 시간 흐름에 따라 변하지 않는다고 가정

* 2021년 ICLR 학회에서 발표된 기술로, 신경망을 블록 단위로 재구성하여 양자화 오차를 최소화하는 고성능 PTQ 기법

* 신경망은 보통 여러 레이어가 묶인 '블록(예: ResNet의 Residual Block)' 구조로 되어 있다는 점에 착안했습니다.

* "블록 하나를 통과한 결과물(Output)이, 양자화 전과 후가 최대한 똑같아지도록 맞추자"는 것이 핵심입니다.

##### 작동원리

1. 원본 블록 실행: 데이터를 전체 정밀도(FP32) 블록에 통과시켜 이상적인 출력값을 얻습니다.
2. 양자화 블록 실행: 동일한 데이터를 양자화된(INT8) 블록에 통과시킵니다.
3. 재구성 오차 최소화: 두 출력값 사이의 차이(Reconstruction Error)를 줄이기 위해, 블록 내의 가중치(Weight) 파라미터를 미세하게 조정합니다.
* 이때 '헤시안(Hessian)' 행렬이라는 2차 미분 정보를 활용하거나, 좌표 하강법(Coordinate Descent) 같은 최적화 기법을 사용하여 정교하게 맞춥니다.


#### AdaRound (Adaptive Rounding)

1. 핵심 아이디어: "반올림(Rounding)을 융통성 있게 하자"

* "2.4를 굳이 2로 보내야 할까? 전체 결과에 도움이 된다면 3으로 보내는 게(올림) 더 낫지 않을까?"
* 즉, 단순히 가까운 값이 아니라, 최종 결과물의 오차를 줄이는 방향으로 올림할지 내림할지를 '학습'해서 결정합니다.

* AdaRound (2020년): "가중치를 반올림할 때, 레이어 단위로 최적의 조합을 찾아보자." (PTQ의 성능을 획기적으로 올림)

---
