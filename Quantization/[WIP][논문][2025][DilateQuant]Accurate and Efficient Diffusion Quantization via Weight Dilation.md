# DilateQuant: Accurate and Efficient Diffusion Quantization via Weight Dilation

저자 :
Xuewen Liu1, 2 Zhikai Li1, 2 Qingyi Gu1, ∗
1 Institute of Automation, Chinese Academy of Sciences

2 School of Artificial Intelligence, University of Chinese Academy of Sciences

발표 : Under review as a conference paper at ICLR 2025

논문 : [PDF](https://arxiv.org/pdf/2409.14307v1)

---

## 1. Introduction

<img width="964" height="509" alt="image" src="https://github.com/user-attachments/assets/e86fda9e-8ada-4cc6-a179-58e670fa5e68" />


### DilateQuant의 3가지 핵심 솔루션

1) 가중치 팽창 (Weight Dilation, WD)

* 가중치 중 포화되지 않은(Unsaturated) 채널을 찾아내어, 이를 수학적으로 동일한 스케일링을 통해 최대한 확장(Dilate)합니다.


2) 시간 병렬 양자화기 (Temporal Parallel Quantizer, TPQ)

* 각 타임스텝(Time-step)에 맞는 개별 양자화 파라미터를 설정합니다.
* 단순히 양자화기를 여러 개 쌓는 것이 아니라, 인덱싱 방식을 통해 여러 타임스텝을 병렬로 처리할 수 있게 설계하여 학습 시간과 데이터 비용을 획기적으로 줄였습니다.


3) 블록 단위 지식 증류 (Block-wise Knowledge Distillation, BKD)

* 모델 전체를 다시 학습시키는 대신, 원래 모델(Full-precision)과 양자화된 모델을 블록 단위로 정렬하여 성능을 보정합니다.
    * 양자화된 블록이 원본과 최대한 유사한 결과값을 내도록 가중치( $w$ )와 양자화 파라미터( $\Delta, z$ )를 동시에 미세 조정(Fine-tuning)

$$\mathcal{L} = MSE(B_k(out_{k-1}) - \hat{B}_k(\hat{out}_{k-1}))$$

* 이 방식은 원래의 방대한 학습 데이터가 필요 없으며, 역전파(Backpropagation) 경로가 짧아 메모리 사용량을 최소화합니다.



### 결과 및 요약

* DilateQuant는 기존의 QAT(양자화 인식 학습) 수준의 정확도를 유지하면서도, PTQ(사후 양자화) 수준의 효율성을 달성했습니다.
* 효율성: 기존 최신 기술(EfficientDM) 대비 타임스텝 보정 시간 160배 단축, 학습 시간 2배 단축이라는 놀라운 성과를 거두었습니다.
* 정확도: 4-bit 및 6-bit 양자화 환경에서 DDPM, LDM, Stable-Diffusion 등 다양한 모델과 데이터셋에 걸쳐 기존 방식보다 훨씬 뛰어난 이미지 생성 품질(FID)을 입증했습니다.



---

## 2. Related Work

### 2.1 Diffusion Model Acceleration

#### 1. 기존의 가속화 방식: 샘플링 경로 단축 

* 대부분의 기존 연구는 수천 번에 달하는 노이즈 제거(denoising) 과정을 줄이기 위해 '더 짧은 샘플링 경로'를 찾는 데 집중해 왔습니다.
    * 분산 스케줄 조정 (Nichol & Dhariwal, 2021): 분산 스케줄을 조정하여 노이즈 제거 단계를 단축합니다.
    * 비마르코프(Non-Markovian) 프로세스 (Song et al., 2020): 노이즈 제거 방정식을 수정하여 확산 과정을 비마르코프 프로세스로 일반화함으로써 단계를 줄입니다.
    * 고차 솔버(High-order Solvers) (Lu et al., 2022): 고차 솔버를 사용하여 확산 생성 과정을 근사화합니다.
* 이러한 방법들은 원래 단계의 약 10%만으로도 유사한 성능을 낼 정도로 성공적이었으나, 비싼 재학습 비용이 발생하거나 복잡한 계산이 수반된다는 단점이 있습니다.

#### 2. DilateQuant의 차별점: 네트워크 자체의 가속화 

* DilateQuant는 단순히 단계 수를 줄이는 것이 아니라, 각 노이즈 제거 단계에서 실행되는 복잡한 네트워크 자체를 가속화하는 데 초점을 맞춥니다.
* 양자화(Quantization) 적용: 각 단계의 네트워크 연산을 저비트 양자화로 처리하여 계산 비용을 낮춥니다.
* 모델 압축: 연산 가속뿐만 아니라 전체 모델의 크기를 압축하는 효과를 동시에 거둡니다.
    * 양자화 후 압축 (packing)

### 2.2 Model Optimization

#### 사후 양자화 (Post-Training Quantization, PTQ)

* 특징: 소량의 데이터로 양자화 파라미터를 보정(Calibration)하며, 가중치 미세 조정이 필요 없습니다.
* 장점: 데이터 및 시간 측면에서 매우 효율적입니다.
* 단점: 확산 모델의 독특한 시간적 네트워크 구조 때문에 표준적인 PTQ 방식(예: BRECQ)은 성능 유지에 실패하는 경우가 많습니다. 기존의 확산 모델용 PTQ 방식들도 아직 6비트 양자화의 벽을 넘지 못하고 있습니다.

#### 양자화 인식 학습 (Quantization-Aware Training, QAT)

* 특징: 양자화 연산을 적용한 상태에서 전체 모델을 다시 학습시킵니다.
* 장점: 낮은 비트(4-bit 등)에서도 높은 성능을 유지할 수 있습니다.
* 단점: 원본 데이터셋, 긴 학습 시간, 막대한 GPU 자원이 필요하여 확산 모델처럼 복잡한 네트워크에는 실용성이 떨어집니다. 예를 들어, TDQ 같은 연구는 5만 개의 데이터로 20만 번의 학습 반복이 필요합니다.


#### 2. 확산 모델 양자화의 현재 주소

* 효율적 대안들: EfficientDM은 LoRA 모듈을 사용해 가중치를 미세 조정하고, QuEST는 민감한 레이어만 선택적으로 학습시키는 방식을 사용합니다.
* 표준화 문제: 하지만 이러한 방식들은 특정 레이어를 양자화에서 제외하거나 하드웨어에서 지원하기 어려운 방식을 사용하는 등 비표준(Non-standard) 설정인 경우가 많습니다.
* 남겨진 과제: 따라서 높은 정확도와 효율성을 동시에 갖춘 표준 저비트 양자화 방법을 찾는 것이 여전히 중요한 과제로 남아 있습니다.

---

## 3. Preliminaries

### 3.1 Quantization

1) 양자화 및 역양자화 과정

* 양자화(Quant): 부동소수점 값 $x$를 정수 값 $x_{int}$로 변환합니다.

$$x_{int} = clip(\lfloor \frac{x}{\Delta} \rfloor + z, 0, 2^b - 1) \quad (1)$$

* 역양자화(DeQuant): 양자화된 정수 값을 다시 부동소수점에 가까운 값 $\hat{x}$로 복원합니다.

$$\hat{x} = \Delta \cdot (x_{int} - z) \approx x \quad (2)$$

* 여기서 각 변수의 의미는 다음과 같습니다.
    * $b$ (비트 너비, bit-width): 클리핑 함수가 허용하는 값의 범위를 결정합니다.
    * $\Delta$ (스케일 인자, scale factor): 부동소수점과 정수 사이의 간격을 조절합니다.
    * $z$ (제로 포인트, zero-point): 부동소수점의 0이 양자화된 영역에서 어디에 위치할지 결정합니다.


2) 양자화 오차의 종류

* 반올림 오차 ( $E_{round}$ ): 부동소수점을 정수로 반올림하는 과정에서 발생합니다.
* 클리핑 오차 ( $E_{clip}$ ): 특정 범위를 벗어나는 값을 강제로 최대/최소값으로 자르는 과정에서 발생합니다.



3) 파라미터 설정 방법 (Calibration)

* 양자화 파라미터( $\Delta, z$ )를 설정하기 위해 주로 두 가지 보정 방법을 사용합니다.
* Max-Min 방식: 데이터의 실제 최댓값과 최솟값을 기준으로 파라미터를 설정합니다. 클리핑 오차( $E_{clip}$ )를 없앨 수 있지만, 데이터 범위가 넓을 경우 스케일 인자( $\Delta$ )가 커져 반올림 오차가 증가할 수 있습니다.
* MSE (Mean Square Error) 방식: 전체적인 오차를 최소화할 수 있는 적절한 값으로 파라미터를 설정합니다. 이 방식은 반올림 오차를 줄일 수 있지만 일부 데이터가 잘려 나가는 클리핑 오차( $E_{clip}$ )가 발생할 수 있습니다.


$$\Delta = \frac{\max(x) - \min(x)}{2^b - 1}, \quad z = \left\lfloor \frac{-\min(x)}{\Delta} \right\rceil \quad (3)$$

$$Q(x) = \Delta \cdot \left( \text{clip} \left( \left\lfloor \frac{x}{\Delta} \right\rceil + z, 0, 2^b - 1 \right) - z \right)\quad(4)$$

### 3.2 Equivalent Scaling


<img width="1001" height="455" alt="image" src="https://github.com/user-attachments/assets/83a8f2f5-7722-4c1f-97f3-eff115d570a9" />

#### 1. 등가 스케일링의 정의

* 수학적 등가성: 채널별로 스케일링 인자( $s$ )를 적용하여 변환하되, 최종 결과값( $Y$ )은 변하지 않도록 유지하는 기법입니다.
* 난이도 이전: 활성화 값의 양자화가 어려울 때, 그 어려움을 상대적으로 양자화가 쉬운 가중치 쪽으로 넘겨주는 역할을 합니다.

#### 2. 수학적 원리

* 선형 레이어(Linear layer) $Y = XW$를 예로 들면, 활성화 값( $X$ )을 스케일링 인자( $s$ )로 나누고, 가중치( $W$ )에는 동일한 인자를 곱해줍니다.

$$Y = (X/s)(s \cdot W) \quad (5)$$

* 이 공식은 컨볼루션(Conv) 레이어에도 동일하게 적용됩니다.
* $s > 1$일 경우: 활성화 값의 범위는 좁아지고(나누기), 가중치의 범위는 넓어집니다(곱하기). 이를 통해 활성화 값의 양자화 오차를 줄일 수 있습니다.


#### 3. 주요 특징 및 이점

* 추론 오버헤드 없음: 스케일링 인자( $s$ )는 오프라인에서 이전 레이어의 파라미터에 미리 통합(Fuse)될 수 있습니다. 따라서 실제 모델이 구동될 때 추가적인 계산 비용이 발생하지 않습니다.
* LLM에서의 활용: 주로 거대언어모델(LLM)에서 특정 채널에 발생하는 튀는 값(Outliers)을 부드럽게 만들어 양자화 성능을 높이는 데 사용되어 왔습니다.

#### 4. 확산 모델에서의 도전 과제

* 기존 LLM 방식(SmoothQuant 등)은 특정 채널에만 이상치가 존재한다고 가정하지만, 확산 모델은 모든 채널에 이상치가 존재하는 특성이 있습니다.
* 단순히 모든 채널을 제약 없이 스케일링하면 가중치의 범위가 너무 커져서 모델 학습 시 수렴이 어려워지는 문제가 발생합니다.


---

## 4. Method

### 4.1 Weight Dilation


### 4.2 Temporal Parallel Quantizer

### 4.3 Block-wise Knowledge Distilation



---

## 5. Experiment

### 5.1 Experimental Setup

### 5.2 Main Result



---

## 6. Conclusion


---


