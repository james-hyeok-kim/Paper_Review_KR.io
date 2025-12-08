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
* 관찰 2: 실제 이미지( $x_0$ )에 가까운 샘플일수록 보정에 유리하다.
    * 타임스텝 $t$가 작을수록(즉, 노이즈가 많이 제거되어 실제 이미지에 가까울수록) 해당 시점의 데이터가 양자화 파라미터 보정에 더 중요한 역할을 합니다
* 관찰 3: 단일 시점보다는 '다양한 타임스텝'의 샘플이 필요하다.
    * 모든 샘플을 같은 타임스텝에서 뽑는 것보다, 여러 타임스텝에 걸쳐 다양하게 뽑는 것이 분포의 변화를 반영할 수 있어 성능이 좋습니다


#### 2. 제안 방법: NDTC (3.3.2 Normally Distributed Time-step Calibration)

위의 관찰 결과들을 종합하여 저자들은 NDTC라는 데이터 수집 알고리즘을 제안합니다. 이 방법은 서로 상충할 수 있는 조건들(실제 이미지에 가까워야 함 vs 다양한 시점을 커버해야 함)의 균형을 맞춥니다7.작동 방식:타임스텝 샘플링: 타임스텝 $t$를 균등하게 뽑지 않고, **정규 분포(Normal Distribution)**를 따르도록 샘플링합니다8.이때 평균($\mu$)을 조절하여 실제 이미지에 가까운(작은 $t$) 쪽의 샘플이 더 많이 뽑히도록 하되, 전체 범위를 아우르도록 합니다9.데이터 생성: 선택된 타임스텝 $t$에 맞춰, 전체 정밀도(Full-precision) 모델을 이용해 노이즈로부터 $x_t$를 생성하여 보정 데이터로 사용합니다10.3. 결론이 NDTC 방식을 사용하여 보정 데이터셋을 구성한 결과, 8비트로 양자화된 확산 모델이 전체 정밀도(32비트) 모델과 거의 대등하거나 심지어 더 우수한 성능(FID, IS 점수 기준)을 보여주었습니다11.


---


## Appendix

BRECQ나 AdaRound

---
