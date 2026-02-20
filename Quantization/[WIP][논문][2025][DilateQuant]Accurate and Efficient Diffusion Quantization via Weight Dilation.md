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

1) Weight 팽창 (Weight Dilation, WD)

* Weight 중 포화되지 않은(Unsaturated) 채널을 찾아내어, 이를 수학적으로 동일한 스케일링을 통해 최대한 확장(Dilate)합니다.


2) 시간 병렬 Quantizer (Temporal Parallel Quantizer, TPQ)

* 각 타임스텝(Time-step)에 맞는 개별 양자화 파라미터를 설정합니다.
* 단순히 Quantizer를 여러 개 쌓는 것이 아니라, 인덱싱 방식을 통해 여러 타임스텝을 병렬로 처리할 수 있게 설계하여 학습 시간과 데이터 비용을 획기적으로 줄였습니다.


3) 블록 단위 지식 증류 (Block-wise Knowledge Distillation, BKD)

* 모델 전체를 다시 학습시키는 대신, 원래 모델(Full-precision)과 양자화된 모델을 블록 단위로 정렬하여 성능을 보정합니다.
    * 양자화된 블록이 원본과 최대한 유사한 결과값을 내도록 Weight( $w$ )와 양자화 파라미터( $\Delta, z$ )를 동시에 미세 조정(Fine-tuning)

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

* 특징: 소량의 데이터로 양자화 파라미터를 보정(Calibration)하며, Weight 미세 조정이 필요 없습니다.
* 장점: 데이터 및 시간 측면에서 매우 효율적입니다.
* 단점: 확산 모델의 독특한 시간적 네트워크 구조 때문에 표준적인 PTQ 방식(예: BRECQ)은 성능 유지에 실패하는 경우가 많습니다. 기존의 확산 모델용 PTQ 방식들도 아직 6비트 양자화의 벽을 넘지 못하고 있습니다.

#### 양자화 인식 학습 (Quantization-Aware Training, QAT)

* 특징: 양자화 연산을 적용한 상태에서 전체 모델을 다시 학습시킵니다.
* 장점: 낮은 비트(4-bit 등)에서도 높은 성능을 유지할 수 있습니다.
* 단점: 원본 데이터셋, 긴 학습 시간, 막대한 GPU 자원이 필요하여 확산 모델처럼 복잡한 네트워크에는 실용성이 떨어집니다. 예를 들어, TDQ 같은 연구는 5만 개의 데이터로 20만 번의 학습 반복이 필요합니다.


#### 2. 확산 모델 양자화의 현재 주소

* 효율적 대안들: EfficientDM은 LoRA 모듈을 사용해 Weight를 미세 조정하고, QuEST는 민감한 레이어만 선택적으로 학습시키는 방식을 사용합니다.
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



#### 1. 등가 스케일링의 정의

* 수학적 등가성: 채널별로 스케일링 인자( $s$ )를 적용하여 변환하되, 최종 결과값( $Y$ )은 변하지 않도록 유지하는 기법입니다.
* 난이도 이전: Activation 값의 양자화가 어려울 때, 그 어려움을 상대적으로 양자화가 쉬운 Weight 쪽으로 넘겨주는 역할을 합니다.

#### 2. 수학적 원리

* 선형 레이어(Linear layer) $Y = XW$를 예로 들면, Activation 값( $X$ )을 스케일링 인자( $s$ )로 나누고, Weight( $W$ )에는 동일한 인자를 곱해줍니다.

$$Y = (X/s)(s \cdot W) \quad (5)$$

* 이 공식은 컨볼루션(Conv) 레이어에도 동일하게 적용됩니다.
* $s > 1$일 경우: Activation 값의 범위는 좁아지고(나누기), Weight의 범위는 넓어집니다(곱하기). 이를 통해 Activation 값의 양자화 오차를 줄일 수 있습니다.


#### 3. 주요 특징 및 이점

* 추론 오버헤드 없음: 스케일링 인자( $s$ )는 오프라인에서 이전 레이어의 파라미터에 미리 통합(Fuse)될 수 있습니다. 따라서 실제 모델이 구동될 때 추가적인 계산 비용이 발생하지 않습니다.
* LLM에서의 활용: 주로 거대언어모델(LLM)에서 특정 채널에 발생하는 튀는 값(Outliers)을 부드럽게 만들어 양자화 성능을 높이는 데 사용되어 왔습니다.

#### 4. 확산 모델에서의 도전 과제

* 기존 LLM 방식(SmoothQuant 등)은 특정 채널에만 이상치가 존재한다고 가정하지만, 확산 모델은 모든 채널에 이상치가 존재하는 특성이 있습니다.
* 단순히 모든 채널을 제약 없이 스케일링하면 Weight의 범위가 너무 커져서 모델 학습 시 수렴이 어려워지는 문제가 발생합니다.


---

## 4. Method

<p align = 'center'>
<img width="1001" height="455" alt="image" src="https://github.com/user-attachments/assets/83a8f2f5-7722-4c1f-97f3-eff115d570a9" />
</p>

### 4.1 Weight Dilation

#### 1. 양자화 오차 분석 (Mathematical Error Analysis)

* 표준 양자화 함수

$$Q(X) = \Delta_x \cdot clip(\lfloor \frac{X}{\Delta_x} \rceil)$$

$$Q(W) = \Delta_w \cdot clip(\lfloor \frac{W}{\Delta_w} \rceil) \ quad (6)$$

* 양자화 오차 ( $E$ )

$$E_x = \Delta_x \cdot (E_{round} + E_{clip})$$

$$E_w = \Delta_w \cdot E_{round}$$

* 여기서 $E_{round}$는 반올림 오차, $E_{clip}$은 클리핑 오차를 의미합니다.
* 스케일링 인자 $s > 1$을 도입하여 $X' = X/s$ 와 $W' = sW$ 로 변환하면 Activation 스케일 인자 $\Delta'_x$ 가 감소하여 Activation 오차( $E'_x$ )는 줄어듭니다.
* 하지만 Weight 스케일 인자 $\Delta'_w$가 커져( $\Delta'_w / \Delta_w > 1$ ) Weight 오차가 증가하고, 이는 학습 시 모델 수렴을 방해합니다.
* 따라서 WD의 목표는 $s > 1$을 확보하면서도 $\Delta'_w \approx \Delta_w$를 유지하는 완벽한 스케일링을 찾는 것입니다.

#### 2. WD의 채널 선택 및 스케일링 메커니즘

<p align = 'center'>
<img width="961" height="559" alt="image" src="https://github.com/user-attachments/assets/05cf9b28-c89a-45a9-8432-6dae82226910" />
</p>


* WD는 Weight 양자화가 출력 채널(Out-channel) 단위로 이루어지고, 스케일링은 입력 채널(In-channel) 단위로 이루어진다는 점을 교묘하게 이용합니다.
    * 제한 집합 ( $A$ ) 설정
         * 각 출력 채널 Weight의 최댓값( $W_{max}$ )과 최솟값( $W_{min}$ )을 결정하는 데 기여하는 입력 채널 인덱스들을 기록하여 집합 $A$를 형성합니다.
    * 포화되지 않은 채널(Unsaturated Channels) 탐색
        * 인덱스 $k$가 집합 $A$에 포함되면 ( $k \in A$ ), Weight 범위를 보존하기 위해 스케일링을 하지 않습니다 ( $s_k = 1$ ).
        * 인덱스 $k$가 $A$에 없으면 ( $k \notin A$ ), 이 채널은 '포화되지 않은' 상태이므로 Weight 경계까지 값을 늘릴(Dilate) 여유가 있습니다.

#### 3. 스케일링 인자 계산 수식

* 포화되지 않은 채널( $k \notin A$ )에 대해, Weight 범위( $W_{max}, W_{min}$ )를 넘지 않는 선에서 Weight $W_k$를 최대한 팽창시키는 $s_k$를 다음과 같이 계산합니다.
* 최대 팽창 인자 계산

$$s_{k1} = \min(W_{max} / W_k.clamp(min=0)) \quad (10)$$

$$s_{k2} = \min(W_{min} / W_k.clamp(max=0)) \quad (11)$$

$$s_k = \min(s_{k1}, s_{k2}) \quad (12)$$

* 이 과정을 통해 Activation 값의 범위는 효과적으로 좁히면서도( $s > 1$ ), Weight의 양자화 파라미터( $W_{max}, W_{min}$ )는 변하지 않게 유지할 수 있습니다. 결과적으로 Activation Quantization는 쉬워지고 모델은 안정적으로 수렴하게 됩니다.


### 4.2 Temporal Parallel Quantizer

#### 1. 배경 및 목적

* 기존 방식의 한계: 이전 방법들(EfficientDM, QuEST 등)은 하나의 레이어에 대해 여러 개의 Activation Quantizer를 두고, 각 타임스텝별로 개별적인 보정 데이터셋을 사용하여 하나씩 최적화했습니다. 이는 데이터 사용이 비효율적이고 학습 시간이 매우 오래 걸리는 단점이 있었습니다.

* TPQ의 핵심 아이디어: 단순히 Quantizer를 여러 개 쌓는 대신, 하나의 Quantizer 내에서 타임스텝별 양자화 파라미터 세트를 설정합니다. 인덱싱(Indexing) 방식을 통해 각 샘플의 타임스텝에 맞는 파라미터를 호출하여 사용함으로써, 서로 다른 타임스텝의 양자화 파라미터들을 병렬로 학습할 수 있게 합니다.

#### 2. 수식 설명

1) 타임스텝별 양자화 파라미터 정의 (수식 13)모델이 총 $T$개의 타임스텝을 가질 때, TPQ는 각 단계에 대응하는 스케일 인자 ( $\Delta_x$ )와 제로 포인트( $z_x$ )의 집합을 가집니다.

$$\Delta_x = \{\Delta_x^1, \Delta_x^2, \Delta_x^3, \dots, \Delta_x^T\}, \quad z_x = \{z_x^1, z_x^2, z_x^3, \dots, z_x^T\} \quad (13)$$

2) 양자화 연산 (수식 14) 특정 타임스텝 인덱스 집합 $\mathbb{T}$에 속하는 입력 샘플 $x$에 대한 양자화 연산은 다음과 같이 정의됩니다.

$$Q(x) = \Delta_x^\mathbb{T} \cdot \left( clip \left( \lfloor \frac{x}{\Delta_x^\mathbb{T}} \rfloor + z_x^\mathbb{T}, 0, 2^b - 1 \right) - z_x^\mathbb{T} \right) \quad (14)$$

* $\Delta_x^\mathbb{T}, z_x^\mathbb{T}$: 현재 입력된 데이터의 타임스텝 $\mathbb{T}$에 해당하는 양자화 파라미터입니다.이 구조 덕분에 서로 다른 타임스텝의 데이터를 한 번에 처리(Batch processing)하면서도 각각 올바른 파라미터를 적용받을 수 있어 학습 효율이 극대화됩니다.

#### 3. 레이어별 세부 설계

* Conv 및 Linear 레이어: 입력 데이터의 타임스텝 인덱스에 맞춰 파라미터를 호출합니다.
* Attention 레이어: 어텐션 메커니즘은 여러 개의 헤드( $H$ )를 가지므로, 타임스텝뿐만 아니라 헤드 정보까지 고려하여 $\Delta_x^{\mathbb{T}*H}$ 및 $z_x^{\mathbb{T}*H}$ 파라미터를 사용합니다.


#### 4. 기대 효과
* 학습 효율성: 기존 SOTA 방식(EfficientDM) 대비 보정 데이터 사용량을 160배, 학습 시간을 2배 단축하는 성과를 거두었습니다.
* 성능 향상: 시간에 따라 급격히 변하는 Activation 값 분포에 유연하게 대응함으로써 저비트 양자화에서의 정확도를 크게 개선했습니다.

### 4.3 Block-wise Knowledge Distillation

#### 1. 도입 배경

* 기존 QAT의 한계표준적인 QAT는 낮은 비트 양자화에서 성능을 잘 유지하지만, 확산 모델에 적용할 때는 다음과 같은 문제가 있습니다.
* 데이터 확보의 어려움: 저작권이나 개인정보 문제로 원본 훈련 데이터를 얻기 어려울 수 있습니다.
    * Distillation에서는 Calibration data set만 사용
* 자원 소모 및 불안정성: 전체 네트워크를 한꺼번에 재학습시키는 것은 시간이 너무 오래 걸리고 학습 과정이 불안정합니다.


#### 2. BKD의 작동 원리 및 수식

* BKD는 모델을 여러 개의 블록( $B_1, \dots, B_K$ ) 단위로 나누고, 각 블록별로 원본 모델(Full-precision)과 양자화된 모델의 출력을 정렬(Align)시킵니다.
* 양자화하려는 블록을 $\hat{B}_k$라고 할 때, 가중치( $w$ )와 양자화 파라미터( $\Delta_x^\mathbb{T}, z_x^\mathbb{T}, \Delta_w$ )를 업데이트하기 위해 다음과 같은 평균 제곱 오차(MSE) 손실 함수를 사용합니다.

$$\mathcal{L}_{\Delta_x^\mathbb{T}, z_x^\mathbb{T}, \Delta_w, w} = MSE(B_k \cdot B_{k-1} \cdot \dots \cdot B_1(x) - \hat{B}_k \cdot \hat{B}_{k-1} \cdot \dots \cdot \hat{B}_1(x)) \quad (15)$$

* $B_k$: 원본(Full-precision) 모델의 $k$번째 블록입니다.
* $\hat{B}_k$: 양자화가 적용된 $k$번째 블록입니다.
* $x$: 입력 샘플로, 원본 모델에 의해 생성된 데이터를 사용하여 데이터 없이(Data-free) 수행할 수 있습니다.

#### 3. BKD의 주요 장점

* 데이터 효율성: 원본 훈련 데이터에 의존하지 않고도 보정이 가능합니다.
* 학습 안정성 및 메모리 절약: 모델 전체가 아닌 블록 단위로 최적화하므로 역전파(Backpropagation) 경로가 짧아집니다. 이는 학습 안정성을 높이고 GPU 메모리 사용량을 획기적으로 줄여줍니다.
* 병렬 최적화: 양자화 파라미터와 가중치를 동시에 병렬로 학습시켜 시간을 절약하며, 가중치가 각 타임스텝의 특성에 더 잘 적응하도록 만듭니다. 


---

## 5. Experiment

### 5.1 Experimental Setup

#### 1. 평가 모델 및 데이터셋

* 사용 모델: DDPM, LDM(LDM-4, LDM-8), 그리고 Stable-Diffusion을 포함합니다.
* 데이터셋: CIFAR-10, LSUN-Bedroom, LSUN-Church, ImageNet, MS-COCO 등 총 5개의 데이터셋에서 성능을 평가했습니다.

#### 2. 평가 지표 (Evaluation Metrics)

* 이미지 품질: FID(Fréchet Inception Distance), sFID, IS(Inception Score)를 기본으로 사용합니다.
* 텍스트-이미지 정렬: 텍스트 가이드 생성 작업의 경우, 이미지와 텍스트의 의미적 유사성을 측정하는 CLIP score를 추가했습니다.
* 효율성: 모델의 가속 및 압축 효과를 시각화하기 위해 비트 연산량(Bit Operations)과 모델 크기(Size)를 계산했습니다.

#### 3. 양자화 및 학습 세부 설정

* 양자화 방식: 가중치에는 표준적인 채널별(Channel-wise) 양자화를, 활성화 값에는 레이어별(Layer-wise) 양자화를 적용했습니다.
* 데이터 및 반복 횟수: 보정을 위해 5,120개의 샘플을 선택했으며, 배치 크기 32로 5,000회(5K) 반복 학습을 수행했습니다.
* 최적화 도구(Optimizer): Adam 최적화 도구를 사용했습니다.
* 학습률(Learning Rate): 양자화 파라미터에는 $1 \times 10^{-4}$, 가중치 미세 조정에는 $1 \times 10^{-2}$의 학습률을 설정했습니다.


#### 4. 하드웨어 환경

* 모든 실험은 단일 RTX A6000 GPU에서 수행되었습니다.


### 5.2 Main Result

#### 1. 무조건부 이미지 생성 (Unconditional Generation)

<p align = 'center'>
<img width="956" height="889" alt="image" src="https://github.com/user-attachments/assets/773cac0e-663c-4907-9c4c-8eccc1b143e1" />
</p>

* 실험은 CIFAR-10, LSUN-Bedroom, LSUN-Church 데이터셋을 대상으로 진행되었습니다.
* 4비트 양자화의 한계 돌파: 기존 방식(EDA-DM, QuEST, EfficientDM)은 4비트 설정에서 성능이 크게 저하되거나 실사용이 불가능한 수준의 FID 점수를 보였습니다. 반면, DilateQuant는 LSUN 데이터셋에서 EfficientDM 대비 FID를 각각 6.28 및 4.98 개선하며 독보적인 품질을 보여주었습니다.
* 6비트 양자화의 정밀도: 6비트 설정에서 DilateQuant는 원본 모델(Full-precision)과 거의 차이가 없는 수준의 이미지 정밀도를 달성했습니다.
* 성능 요약 (Table 1)
    * DDPM (CIFAR-10): 4비트에서 DilateQuant는 9.13 FID를 기록하여, EDA-DM(120.24)이나 EfficientDM(81.27)보다 압도적으로 우수합니다.



#### 2. 조건부 이미지 생성 (Conditional Generation)

<p align = 'center'>
<img width="941" height="595" alt="image" src="https://github.com/user-attachments/assets/75b98853-8052-433f-988c-1487e65260d0" />
</p>

* 텍스트 가이드 생성(Stable-Diffusion)과 클래스 가이드 생성(ImageNet)에 대한 결과입니다.

* Stable-Diffusion 가속 및 압축: 6비트 정밀도에서 DilateQuant는 FID를 24.69로 개선하면서도, 모델 크기를 5.3배 압축하고 비트 연산량(Bit Operations)을 27.9배 감소시켰습니다. 이는 실제 환경에서 저지연(low-latency) 애플리케이션 적용 가능성을 높입니다.

* 클래스 가이드 생성: ImageNet 데이터셋의 LDM-4 모델 실험에서도 모든 비트 너비 설정에서 기존 방식보다 유의미한 성능 향상을 확인했습니다.

#### 3. 효율성 및 정확도 균형 (Trade-off)

* DilateQuant는 QAT(양자화 인식 학습) 수준의 정확도를 내면서도 PTQ(사후 양자화) 수준의 효율성을 유지한다는 점이 가장 큰 특징입니다.
* 시간 및 자원 절감: CIFAR-10 기준, QAT 방식이 13.9시간의 학습 시간과 9.9GB의 메모리를 소모할 때, DilateQuant는 단 1.08시간과 3.4GB의 메모리만으로도 유사한 정확도(FID 9.13 vs 7.30)를 확보했습니다.
* 데이터 효율성: 수만 장의 데이터를 사용하는 QAT와 달리, 별도의 원본 훈련 데이터 없이 소량의 보정 샘플만으로 이 성과를 냈습니다.


### 5.3 Ablation Study

<p align = 'center'>
<img width="877" height="250" alt="image" src="https://github.com/user-attachments/assets/73f11443-c98f-4442-b42a-f43af836f28a" />
</p>

#### 1. 각 구성 요소의 효과 (Table 3 분석)

* TPQ의 중요성: 시간에 따라 변하는 활성화 값 분포가 성능 저하의 주범이므로, TPQ를 알고리즘의 뼈대(Backbone)로 사용합니다.
* WD의 기여: WD를 도입하면 활성화 양자화 오차를 효과적으로 줄여 FID 점수를 대폭 낮춥니다.
* BKD의 효과: BKD는 PTQ 수준의 시간과 메모리 비용을 유지하면서도 성능을 추가로 끌어올립니다.

#### 2. 프레임워크 간 효율성 비교 (Table 4 분석)

<p align = 'center'>
<img width="893" height="221" alt="image" src="https://github.com/user-attachments/assets/1b3b0fa0-4bee-4ed7-bc45-dbc71ce597f7" />
</p>

* DilateQuant가 기존의 PTQ 및 QAT 방식들과 비교했을 때 자원 소모량 대비 얼마나 효율적인지 보여줍니다.
* 정확도(Accuracy): DilateQuant는 9.13 FID를 기록하여, 막대한 자원을 쓰는 **QAT 방식(7.30 FID)**에 근접하는 높은 성능을 보였습니다.
* 시간 비용(Time Cost): QAT 방식이 13.89시간 걸릴 때, DilateQuant는 단 1.08시간 만에 학습을 끝내 PTQ(0.97시간)와 비슷한 효율성을 보였습니다.
* 메모리 점유(GPU Memory): QAT(9,974MB)의 약 1/3 수준인 3,439MB의 메모리만 사용하여 매우 가볍습니다.


#### 3. 기타 분석 (부록 참조)

<p align = 'center'>
<img width="896" height="927" alt="image" src="https://github.com/user-attachments/assets/177c8f42-4947-47ac-85f7-e176e9bf53e9" />
</p>

* 강건성(Robustness): 다양한 샘플러(DDIM, PLMS, DPM-Solver)와 타임스텝(20, 100) 설정에서도 성능이 안정적임을 확인했습니다.

<p align = 'center'>
<img width="881" height="213" alt="image" src="https://github.com/user-attachments/assets/04bdc8e7-71a2-472e-acb7-a1a59c75da81" />
</p>

* 미적 평가(Aesthetic): 인간의 선호도를 모방한 미적 점수 평가에서도 기존 방식(EfficientDM)보다 높은 점수를 받았습니다.

---

## 6. Conclusion

* 새로운 양자화 프레임워크 제안: 확산 모델을 위한 혁신적인 양자화 프레임워크인 DilateQuant를 제안하였으며, 이는 기존 방식들과 비교하여 높은 정확도와 뛰어난 효율성을 동시에 제공합니다.
* 활성화 값 범위 문제 해결 (WD): 입력 채널 가중치의 불포화(unsaturation) 특성을 발견하고 이를 이용해 활성화 값의 넓은 범위를 완화했습니다. 가중치를 제약된 범위 내에서 팽창(dilation)시킴으로써 활성화 양자화 오차를 추가 비용 없이 가중치 양자화로 흡수시켰습니다.
* 시간에 따른 변화 대응 (TPQ): 시간에 따라 변하는 활성화 값에 맞춤형 타임스텝 양자화 파라미터를 설정하는 유연한 양자화기를 설계했습니다. 이 양자화기는 병렬 학습을 지원하여 성능을 크게 향상시키는 동시에 훈련 시간을 단축했습니다.
* 효율적인 성능 강화 (BKD): 양자화된 모델을 블록 단위로 정밀 모델(Full-precision)과 정렬시키는 새로운 지식 증류 전략을 도입했습니다. 파라미터의 동시 훈련과 짧은 역전파 경로를 통해 시간과 메모리 사용량을 최소화하며 성능을 높였습니다.
* 실험적 입증: 광범위한 실험을 통해 DilateQuant가 특히 저비트(4-bit, 6-bit) 양자화 환경에서 기존의 최신 방법들을 유의미하게 능가함을 증명했습니다.

---


