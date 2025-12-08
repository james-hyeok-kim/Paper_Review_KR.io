# TCAQ-DM: Timestep-Channel Adaptive Quantization for Diffusion Models

저자 : Haocheng Huang1,2,*, Jiaxin Chen1,2,*,Jinyang Guo3, Ruiyi Zhan1,2, Yunhong Wang1,2,†

1 State Key Laboratory of Virtual Reality Technology and Systems, Beihang University, Beijing, China

2 School of Computer Science and Engineering, Beihang University, Beijing, China

3 School of Artificial Intelligence, Beihang University, Beijing, China

출간 : AAAI(Association for the Advancement of Artificial Intelligence), 2025

논문 : [PDF](https://arxiv.org/pdf/2412.16700)

---

## 1. Introduction

### 기존 PTQ 방식이 실패하는 이유 (핵심 과제)

<p align = 'center'>
<img width="400" height="600" alt="image" src="https://github.com/user-attachments/assets/adfa3c3b-6813-4f87-b306-9a0e201fac91" />
</p>


1. 활성화 값의 급격한 변동: 합성곱(Convolution) 레이어의 활성화 값 범위가 채널과 타임스텝에 따라 극심하게 변동하여 큰 양자화 오차를 유발합니다.
2. 동적인 분포 변화: Softmax 이후 레이어의 활성화 분포가 타임스텝이 진행됨에 따라 동적으로 변하며, 특히 거듭제곱 법칙(power-law)과 유사한 형태를 띠게 되어 고정된 양자화기로는 처리가 어렵습니다.
3. 데이터 불일치 (Input Mismatch): 양자화 모델을 재구성(reconstruction)할 때 사용하는 입력 데이터와 실제 추론 시 반복적 샘플링을 통해 얻는 데이터 간에 불일치가 발생하여 편향(bias)이 생깁니다.

* 훈련은 정답지를 보고 공부하고, 시험은 오답이 누적된 문제를 품


### 제안하는 해결책: TCAQ-DM

TCAQ-DM (Timestep-Channel Adaptive Quantization for Diffusion Models)

1. 타임스텝-채널 결합 재파라미터화 (TCR Module):
* 활성화 값의 변동이 심한 합성곱 레이어를 위해 고안되었습니다. 타임스텝과 채널 전반에 걸쳐 활성화 범위를 균형 있게 맞춰주는 모듈입니다.

2. 동적 적응형 양자화기 (DAQ Module):
* 타임스텝에 따라 변하는 분포를 처리하기 위해 설계되었습니다. Softmax 이후 레이어의 분포가 거듭제곱 법칙을 따를 가능성을 추정
* 상황에 따라 Log2 양자화기와 균일(Uniform) 양자화기 중 최적의 것을 동적으로 선택합니다.


3. 점진적 정렬 재구성 (PAR Strategy):
* 데이터 불일치 문제를 해결하기 위한 전략
* 양자화 과정의 재구성 단계에서 양자화된 입력을 사용하여, 실제 추론 과정의 데이터 흐름과 일치하도록 조정합니다

---

## 2. Related Work

### 효율적인 디퓨전 모델 (Efficient Diffusion Model)

* 샘플링 단계(Timesteps)의 수를 줄이는 데 초점

#### 훈련 기반 방법 (Training-based)

* 모델 증류(Distillation)나 샘플 궤적 학습 등을 통해 적은 단계로도 이미지를 생성할 수 있도록 모델을 재학습시킵니다.

#### 비훈련 기반 방법 (Training-free):

* 사전 학습된 모델을 그대로 두면서, 더 효율적인 샘플러(예: DDIM, 맞춤형 SDE/ODE solver)를 설계하여 샘플링 과정을 가속화합니다.

#### 한계점

* 추론 시간은 줄여주지만, 모델의 크기 자체는 줄어들지 않습니다. 여전히 높은 메모리 사용량과 연산 복잡도가 문제로 남습니다.


### 모델 양자화 (Model Quantization)

#### 일반적인 양자화 방법 (General Quantization Methods)
* QAT (Quantization-Aware Training)
* PTQ (Post-Training Quantization):

#### 디퓨전 모델을 위한 PTQ (PTQ for Diffusion Models)
일반적인 PTQ 방식을 디퓨전 모델에 바로 적용하면 성능 저하가 심각하기 때문에, 이를 해결하기 위한 여러 연구가 진행되었습니다.

##### 기존 연구들의 시도

* PTQ4DM: 다양한 타임스텝에서 데이터를 수집하여 보정.
* Q-Diffusion: 스킵 연결(Skip connection) 레이어를 분할하여 성능 향상.
* PTQD: 샘플러를 수정하여 누적 오차 제거.
* 기타 (APQ-DM, TFMQ-DM 등): 동적 그룹화 전략이나 타임스텝 간 정보 편향 완화 등을 시도함.

### 기존 연구의 한계점 (Research Gap)

* 활성화 값의 변동성 미고려: 합성곱 레이어에서 채널과 타임스텝에 따라 활성화 값의 범위가 급격히 변동하는 문제를 공동으로(jointly) 해결하지 못했습니다.

* 동적 분포 변화 미고려: Softmax 이후 레이어의 분포가 타임스텝마다 변하는데, 기존 방식은 고정된 양자화기를 사용하여 오차가 큽니다.

* 데이터 불일치 (Input Mismatch) 무시: 양자화 모델 재구성(Reconstruction) 단계의 입력 데이터와 실제 추론 과정의 데이터 흐름이 일치하지 않는 문제를 간과했습니다.

---

## 3. Methodology

<p align = 'center'>
<img width="884" height="590" alt="image" src="https://github.com/user-attachments/assets/dbd8ac80-1d05-4d40-b544-2a0b61952c37" />
</p>

### 1. 타임스텝-채널 결합 재파라미터화 (TCR: Timestep-Channel Joint Reparameterization)
* 이 모듈은 초기화 단계에서 합성곱(Convolution) 레이어의 활성화 값(Activation)이 타임스텝과 채널에 따라 극심하게 변동하는 문제를 해결

#### 문제점
* 디퓨전 모델의 활성화 값은 특정 채널이나 타임스텝에서 갑자기 값이 튀는(outlier) 현상이 발생합니다.
* 이러한 변동성은 양자화 범위를 설정하기 어렵게 만들어 큰 오차를 유발합니다

#### 작동 원리
* 재파라미터화(Reparameterization): 수학적으로 선형 변환을 통해 활성화 값의 큰 범위를 가중치(Weight) 쪽으로 일부 이동시켜 균형을 맞춥니다. 이렇게 하면 출력값은 유지하면서 양자화하기 쉬운 형태로 데이터 분포가 바뀝니다.
* 스케일링 벡터 계산 ( $r$ ): 단순히 한 시점만 보는 것이 아니라, 모든 타임스텝의 데이터를 통합하여 스케일링 벡터를 계산합니다.
* 가중 평균 적용: 활성화 값이 큰 타임스텝에 더 많은 가중치를 두어, 중요한 정보가 손실되지 않도록 전체적인 범위를 조정합니다.
* 결과: 활성화 값의 범위가 채널과 타임스텝 전반에 걸쳐 평탄화(balance)되어 양자화 오차가 줄어듭니다.

##### 선형 변환

$$Y = W \times X$$

$$Y = (W \cdot s) \times (X / s)$$

1. 변환 전 (Before)

* 활성화( $X$ ): 값이 0에서 100까지 널뛰기합니다. (범위가 너무 넓음 → 양자화 시 오차 급증)
* *가중치( $W$ ): 값이 -1에서 1 사이로 안정적입니다. (범위가 좁음 → 여유가 있음)

2. 변환 후 (After)

* $s=10$ 이라고 가정
* 활성화($X/10$): 값이 0에서 10으로 줄어듭니다. (양자화하기 딱 좋은 크기가 됨)
* 가중치($W \cdot 10$): 값이 -10에서 10으로 늘어납니다. (여전히 양자화하기에 무리 없는 범위)

A. 채널별 적용 (Channel-wise)
* 단순히 하나의 숫자 $s$로 전체를 나누는 것이 아니라, 각 채널(Channel)마다 다른 스케일링 값( $r$ )을 적용합니다.
* 4어떤 채널은 값이 아주 크고, 어떤 채널은 작기 때문입니다.수식으로는 벡터 $r$을 사용하여 각 채널에 맞는 맞춤형 '나눗셈'을 수행합니다.

B. 타임스텝 통합 (Timestep-aware Aggregation)
* 디퓨전 모델은 $T=1000$에서 $T=0$까지 반복해서 같은 가중치( $W$ )를 사용합니다.
* 하지만 활성화 값( $X$ )은 타임스텝마다 다릅니다.
* 문제: $T=900$일 때는 $s=5$가 필요한데, $T=100$일 때는 $s=2$가 필요할 수 있습니다.
    * 가중치( $W$ )는 하나뿐이라서 $s$를 하나로 정해야 합니다.
* 해결: 저자들은 모든 타임스텝의 활성화 값을 조사한 뒤, 가중 평균(Weighted Average) 방식을 사용해 '모든 타임스텝을 아우르는 최적의 $s$ (논문에서는 $r$ 벡터)'를 계산해 냈습니다.

(Here)

### 2. 동적 적응형 양자화기 (DAQ: Dynamically Adaptive Quantizer)
이 모듈은 초기화 단계에서 Softmax 이후(Post-Softmax) 레이어의 활성화 값 분포가 타임스텝마다 달라지는 문제를 해결합니다.


문제점:

Softmax를 통과한 데이터는 타임스텝에 따라 분포 모양이 변합니다.

어떤 때는 좁은 범위의 정규 분포를 띠지만, 어떤 때는 긴 꼬리를 가진 거듭제곱 법칙(Power-law) 분포를 띱니다.


기존의 고정된 양자화기(Uniform 또는 Log2) 하나만으로는 이 두 가지 상황을 모두 커버할 수 없습니다.

작동 원리:


분포 감지 (MLE): 최대 우도 추정(Maximum Likelihood Estimation)을 사용하여 현재 타임스텝의 데이터가 '거듭제곱 법칙' 분포에 얼마나 가까운지 계산합니다.

양자화기 자동 선택:


거듭제곱 법칙 분포일 경우 (Ratio > 0): 작은 값과 큰 값을 동시에 잘 표현하는 Log2 양자화기를 사용합니다.



그렇지 않을 경우 (Ratio ≤ 0): 일반적인 Uniform(균일) 양자화기를 사용합니다.


효율성: 이 선택 과정은 오프라인으로 미리 계산할 수 있어 실제 추론 속도에는 거의 영향을 주지 않습니다.

### 3. 점진적 정렬 재구성 (PAR: Progressively Aligned Reconstruction)

이 전략은 재구성 단계에서 양자화 모델 튜닝 시 발생하는 데이터 불일치(Input Mismatch) 문제를 해결합니다.

문제점:

앞서 설명해 드린 대로, 양자화 파라미터를 튜닝할 때 깨끗한 데이터(FP 모델 출력)를 쓰면, 실제 추론 시 노이즈가 섞인 데이터(양자화 모델 출력)가 들어왔을 때 성능이 떨어집니다.

작동 원리:


1단계 (기본 재구성): 먼저 일반적인 방식(BRECQ 등)으로 모델을 1차 튜닝합니다.

2단계 (데이터 샘플링): 1차 튜닝된 양자화 모델을 사용하여 새로운 보정 데이터셋(Calibration Set)을 생성합니다. 이 데이터에는 양자화 오차가 포함되어 있습니다.

3단계 (점진적 튜닝): 생성된 '오차가 포함된 데이터'를 입력으로 사용하여 다시 모델을 미세 조정(Reconstruction)합니다.


반복: 이 과정을 몇 차례 반복하여, 모델이 실제 추론 시 겪게 될 데이터 분포에 점진적으로 적응하도록 만듭니다.


---



---
