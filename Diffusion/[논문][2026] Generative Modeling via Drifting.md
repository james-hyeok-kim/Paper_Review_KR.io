# Generative Modeling via Drifting

저자: Mingyang Deng 1 He Li 1 Tianhong Li 1 Yilun Du 2 Kaiming He 1

1 MIT 2 Harvard University.

발표: 2026년 2월 4일 (arXiv:2602.04770)

논문: [PDF](https://arxiv.org/pdf/2602.04770)

---

<p align = 'center'>
<img width="717" height="285" alt="image" src="https://github.com/user-attachments/assets/58021a55-4c01-49ab-9e3f-6bcb15880ec4" />
</p>

* Drifting Models는 생성 모델링(Generative Modeling)을 위한 새로운 패러다임
* 기존의 확산(Diffusion) 또는 플로우 매칭(Flow Matching) 모델들이 추론 시에 여러 번의 반복적인 단계를 거쳐 노이즈를 데이터로 변환하는 것
* Drifting 모델은 학습 시간 동안의 반복적인 최적화 과정을 분포의 진화로 활용하여 추론 시에는 단 한 번의 단계(one-step inference)만으로 고품질의 데이터를 생성합니다.
    * 반복의 주체 변경: 복잡한 분포를 단순한 단계로 쪼개는 철학은 유지하되, 그 단계를 추론 시점이 아닌 딥러닝 최적화 단계(SGD 등)로 옮겨왔습니다.


---
## 1. Introduction

### 1. 핵심 개념: 학습 시간 동안의 분포 진화

* 푸시포워드(Pushforward) 모델링
    * 사전 분포 $p_{prior}$를 데이터 분포 $p_{data}$로 매핑하는 함수 $f$를 학습하는 과정으로 정의
    * 학습 과정의 활용: 딥러닝 최적화(예: SGD)는 본질적으로 반복
    * Drifting 모델은 이 학습 과정을 통해 푸시포워드 분포 $q = f_{\sharp} p_{prior}$ 가 점진적으로 데이터 분포에 맞춰지도록 합니다.
    * 드리프팅 필드(Drifting Field): 샘플의 이동을 제어하는 '드리프팅 필드'를 도입합니다. 이 필드는 생성된 분포 $q$와 실제 데이터 분포 $p$ 사이의 상호작용에 기반하며, 두 분포가 일치할 때 0이 되어 평형 상태에 도달합니다.
 

### 2. 주요 장점 및 성능

* 효율적인 추론: 별도의 반복적인 추론 절차가 필요 없는 단일 패스(single-pass) 네트워크로 구성되어 있어 효율적입니다.
* 상태 최첨단(SOTA) 결과: ImageNet $256\times256$ 데이터셋에서 단일 단계 생성 모델 중 가장 뛰어난 성능을 보였습니다.
    * 잠재 공간(Latent Space): FID 1.54 달성.
    * 픽셀 공간(Pixel Space): FID 1.61 달성.
* 모드 붕괴(Mode Collapse) 방지: 실제 데이터 분포의 여러 모드가 샘플을 끌어당기기 때문에 특정 모드로 뭉치는 현상에 강한 내성을 가집니다.

### 3. 학습 메커니즘

* 손실 함수: 드리프팅 필드의 제곱 노름(squared norm)을 최소화하는 간단한 학습 목표를 사용합니다.
* 특징 공간 활용: 고차원 데이터 생성을 위해 원본 데이터 공간 대신 사전 학습된 자기 자기주도 학습(SSL) 모델의 특징 공간(Feature Space)에서 드리프팅 손실을 계산하여 풍부한 그래디언트 정보를 얻습니다.

$$\|v(x, t)\|^2 = \sum_{i=1}^{d} |v_i(x, t)|^2$$

* $v(x, t)$가 주어졌을 때, 특정 위치 $x$에서의 제곱 노름은 해당 벡터의 길이를 제곱한 값입니다. 유클리드 공간에서는 보통 $L^2$ 노름을 사용
* 오직 크기의 제곱만을 나타낸다

---

## 2. Related Works

### 1. 관련 연구 섹션의 목적

* 지식의 지도 작성: 현재 분야가 어디까지 발전했는지 보여줍니다.
* 차별성 강조: 기존 연구들의 한계점(예: 계산 복잡도, 경로의 불안정성 등)을 지적하며 내 연구의 필요성을 부각합니다.
* 정당성 부여: 내가 사용하는 방법론(제곱 노름 규제 등)이 이전의 어떤 이론적 토대 위에 있는지 설명합니다.

### 2. '드리프팅 필드' 관련 논문의 전형적인 구조

1) 생성 모델의 발전 (Diffusion & Flow Matching)
    1) Score-based Models & DDPM: 노이즈를 제거하며 데이터를 생성하는 초기 모델들을 언급합니다.
        1) 스코어 함수(Score Function)를 학습하는 방식
        2) 드리프트(Drift)의 초기 개념이 등장
    2) Continuous Normalizing Flows (CNF): 데이터를 연속적인 흐름(ODE/SDE)으로 변환하는 방식의 기초를 설명합니다. 여기서 드리프트 필드 $v(x, t)$의 개념이 본격적으로 등장합니다.
        1) 분포를 직접 추정하는 대신, 노이즈와 데이터를 잇는 벡터 필드(Vector Field) $v_t(x)$를 직접 학습하는 방식(Flow Matching)을 다룹니다.
        2) 우리는 이제 스코어가 아닌, 입자가 이동하는 '속도(Velocity)' 자체에 집중하기 시작 

2) 최적 운송 이론 (Optimal Transport, OT)
    1) Monge-Kantorovich 문제: 데이터를 이동시킬 때 '최소 비용'으로 이동시키는 고전적 이론을 설명합니다.
    2) Benamou-Brenier Formulation: 앞서 설명한 제곱 노름( $\|v\|^2$ )을 시간과 공간에 대해 적분하여 최소화하는 방식이 어떻게 '최적 경로'를 보장하는지 학술적 근거를 제시합니다.
        1) 기존 Flow Matching은 경로가 구불구불, 제곱 노름 규제를 쓰면 경로가 직선이 되어 효율적    

3) 경로 평활화 및 규제 기법 (Path Smoothing & Regularization)
    1) 기존 모델들이 생성 경로가 구불구불하여 계산 효율이 떨어졌던 점을 지적합니다.
    2) 이를 해결하기 위해 Velocity 기반 규제나 Optimal Transport Flow Matching 등을 도입한 최신 연구들을 소개하며, 본인의 연구와 비교합니다.


---

## 3. Drifting Models for Generation


1. 일반적인 확산 모델의 SDE 구성확산 모델은 일반적으로 다음과 같은 형태의 SDE를 통해 데이터를 노이즈로 바꾸거나, 노이즈에서 데이터를 생성합니다.

$$dx = \mathbf{f}(x, t)dt + g(t)dw$$

* 드리프트 항 ( $\mathbf{f}(x, t)dt$ ): 결정론적인 부분으로, 샘플이 시간에 따라 이동하는 평균적인 방향을 결정합니다.
* 확산 항 ( $g(t)dw$ ): 확률론적인 부분으로, 브라운 운동( $dw$ )에 의한 무작위 노이즈를 추가하여 분포를 퍼뜨리는 역할을 합니다.

### 3.1. Pushforward at Training Time

* 모델이 학습 과정에서 수학적으로 어떻게 데이터를 변형(Transformation)시키는지 그 '메커니즘'을 설명하는 핵심 파트

#### 1. Pushforward( $\sharp$ )의 개념

* 수학에서 Pushforward는 하나의 확률 분포를 함수(또는 흐름)를 통해 다른 공간으로 이동시키는 것을 의미합니다.
* 만약 우리가 학습시킨 드리프팅 필드 $v(x, t)$에 의해 정의된 흐름(Flow)을 $\phi_t$라고 한다면, 시간 $t$에서의 분포 $p_t$는 다음과 같이 표현됩니다.

$$p_t = [\phi_t]_\sharp p_0$$

* 직관적 해석: "시작점의 모래알( $p_0$ )들을 벡터 필드( $v$ )를 따라 $t$ 시간 동안 밀어냈을 때 형성된 모래 더미의 모양이 바로 $p_t$다."

#### 2. Training Time에서의 핵심 프로세스

* 학습 시에는 전체 경로를 다 계산할 수 없으므로, 보통 조건부(Conditional) 방식을 사용하여 효율적으로 'Push'를 수행합니다.
1) 데이터 쌍 샘플링 (Sampling)먼저 노이즈 데이터 $x_0 \sim p_0$와 실제 데이터 $x_1 \sim p_1$을 샘플링합니다.
2) 경로 정의 (Probability Path) $x_0$ 에서 $x_1$으로 가는 가장 단순한 경로(보통 직선)를 정의합니다. 이를 보간(Interpolation)이라고 합니다.

$$x_t = (1-t)x_0 + tx_1$$

3) 타겟 드리프트 필드 설정이 경로 $x_t$ 위에서 입자가 움직여야 할 '이상적인 속도'를 계산합니다.

$$v_t(x_t) = \frac{dx_t}{dt} = x_1 - x_0$$

이 $x_1 - x_0$가 바로 모델이 배워야 할 타겟(Ground Truth)이 됩니다.

#### 3. 제곱 노름(Squared Norm)과의 연결고리

* 모델 $v_\theta(x, t)$가 실제 타겟 $v_t$를 얼마나 잘 따라가는지 측정할 때, 우리는 두 벡터 사이의 차이의 제곱 노름을 최소화합니다.

$$\mathcal{L} = \mathbb{E}_{t, x_0, x_1} [ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 ]$$

* 왜 제곱인가? 앞서 언급했듯 미분이 용이하고, $L^2$ 거리를 최소화하는 것이 확률 분포 간의 Optimal Transport(최적 운송) 경로를 찾는 것과 수학적으로 동일하기 때문입니다.

* 학습 결과: 이 손실 함수를 최소화하면, 모델은 $p_0$에서 $p_1$으로 데이터를 보낼 때 가장 에너지를 적게 쓰는(즉, 가장 직선에 가까운) 방식으로 Pushforward하는 법을 배우게 됩니다.


### 3.2. Drifting Field for Training

#### 1. 드리프트 필드 (Drifting Field)의 정의

* 드리프트 필드는 훈련 중인 샘플 $x$가 다음 단계에서 얼마나 이동해야 하는지( $\Delta x$ )를 결정하는 함수 $V_{p,q}$입니다.
* 공식: 다음 반복(Iteration)에서의 샘플 위치는 다음과 같이 결정됩니다

$$x_{i+1}=x_i+V_{p,q_i}(x_i)$$

* 의존성: 이 필드는 데이터 분포( $p$ )와 현재 모델이 생성하고 있는 분포( $q$ )에 따라 달라집니다.

#### 2. 평형 상태와 반대칭성 (Equilibrium and Anti-symmetry)

* 모델의 궁극적인 목표는 생성 분포 $q$가 데이터 분포 $p$와 일치하게 만드는 것입니다. 이 상태를 평형(Equilibrium)이라고 하며, 이때 샘플의 이동은 멈춰야 합니다( $V=0$ ).
* 반대칭 조건: 논문은 드리프트 필드가 반대칭성 ( $V_{p,q}(x) = -V_{q,p}(x)$ )을 가져야 함을 제안합니다.
* 이 조건이 만족되면, 두 분포가 일치( $q=p$ )할 때 드리프트 필드는 자연스럽게 $0$이 되어 샘플들이 더 이상 움직이지 않는 평형 상태에 도달합니다.
* 훈련 목적함수 (Training Objective)평형 상태 조건을 바탕으로, 네트워크 $f$를 최적화하기 위한 손실 함수( $\mathcal{L}$ )를 다음과 같이 정의합니다

$$\mathcal{L}=\mathbb{E}_{\epsilon}[||f_{\theta}(\epsilon) - stopgrad(f_{\theta}(\epsilon)+V_{p,q_{\theta}}(f_{\theta}(\epsilon)))||^{2}]$$

* 작동 방식: 현재 네트워크의 예측값( $f_{\theta}(\epsilon)$ )을 '이동이 적용된 목표 상태'( $x + \Delta x$ )로 밀어붙이는 방식입니다.
* Stop-gradient: 목표 상태를 계산할 때 stopgrad를 사용하여 이전 상태를 고정함으로써, 복잡한 분포 간의 역전파 문제를 단순화합니다.
* 이 메커니즘을 통해 딥러닝 최적화 도구(optimizer)는 반복적인 훈련 과정을 거치며 생성 분포를 데이터 분포 쪽으로 자연스럽게 진화시키게 됩니다.


### 3.3. Designing the Drifting Field

<p align = 'center'>
<img width="354" height="387" alt="image" src="https://github.com/user-attachments/assets/a4b869ab-f33f-4d1c-bc46-dbee2c9a0a47" />
</p>


* 핵심 아이디어는 생성된 샘플이 실제 데이터 쪽으로는 끌려가고(인력), 현재 생성된 다른 샘플들로부터는 밀려나게(척력) 만드는 것

#### 1. 기본 개념: 인력과 척력 (Attraction and Repulsion)

* 드리프트 필드는 두 가지 힘의 조합으로 정의

$$V_{p,q}(x) := V_{p}^{+}(x) - V_{q}^{-}(x)$$

* 인력 항 ( $V_{p}^{+}$ ): 데이터 분포 $p$에서 추출된 양성 샘플(Positive samples, $y^{+}$ )들이 $x$를 끌어당기는 힘입니다.
* 척력 항 ( $V_{q}^{-}$ ): 현재 모델이 생성한 분포 $q$에서 추출된 음성 샘플(Negative samples, $y^{-}$)들이 $x$를 밀어내는 힘입니다.
* 이 필드는 데이터 분포 $p$와 생성 분포 $q$가 일치할 때 서로 상쇄되어 $0$이 되며, 이를 통해 반대칭성(Anti-symmetry)과 평형 상태를 유지합니다.

#### 2. 커널(Kernel) 디자인

* 샘플 간의 유사도를 측정하기 위해 커널 함수 $k(x, y)$를 사용합니다. 논문에서는 지수 함수 형태의 커널을 채택했습니다

$$k(x, y) = \exp\left(-\frac{1}{\tau}||x - y||\right)$$

* 온도 매개변수 ( $\tau$ ): 거리에 따른 영향력을 조절합니다.
* 특징: 이 커널은 가까운 샘플일수록 더 강한 드리프트를 유도하여 훈련 신호를 풍부하게 제공합니다.
* 정규화: 실제 구현 시에는 Softmax 연산을 통해 가중치를 정규화하여 사용합니다.

#### 3. 최종 계산 공식

위의 요소들을 결합하면, 샘플 $x$의 최종 이동 벡터는 다음과 같이 통합된 기대치 형태로 표현됩니다:

$$V_{p,q}(x) = \frac{1}{Z_{p}Z_{q}}\mathbb{E}_{p,q}[k(x, y^{+})k(x, y^{-})(y^{+} - y^{-})]$$

* 이 공식은 양성 샘플과 음성 샘플의 차이( $y^{+} - y^{-}$ )에 커널 가중치를 곱하여 계산됩니다.
* 확률적 훈련 (Stochastic Training): 매 훈련 단계마다 미니배치(mini-batch)를 사용하여 이 기댓값을 근사치로 계산합니다.
* 음성 샘플의 재활용: 효율성을 위해 같은 배치 내에서 생성된 샘플 $x$를 음성 샘플 $y^{-}$로 재사용합니다.


#### 4. 핵심 특징 요약

* 모드 붕괴 방지: 만약 모델이 특정 지점으로 뭉치면(모드 붕괴), 데이터의 다른 부분들이 샘플을 강력하게 끌어당겨 다시 퍼지게 만듭니다.
* 유연성: 인력과 척력이 상쇄되는 평형 조건만 만족한다면 다양한 커널이나 함수 형태를 적용할 수 있습니다.
* 강건함: 정규화(Normalization) 과정을 통해 특징값의 크기나 차원에 상관없이 안정적으로 작동합니다.

### 3.4. Drifting in Feature Space

* 드리프트 손실(Drifting Loss)을 가공되지 않은 원본 데이터 공간(예: 픽셀)이 아닌, 특징 추출기(Feature Extractor)를 통해 변환된 고차원 특징 공간에서 계산하는 방법을 설명

#### 1. 특징 공간에서의 손실 함수 정의

* 고차원 데이터(이미지 등)를 효과적으로 생성하기 위해, 원본 데이터 대신 특징 추출기 $\phi$를 통과한 특징값들 사이의 드리프트를 최소화합니다.
* 손실 함수 공식

$$\mathbb{E}[||\phi(x)-stopgrad(\phi(x)+V(\phi(x)))||^{2}]$$

* 여기서 $x = f_{\theta}(\epsilon)$는 생성기의 출력물이며, $V$는 특징 공간에서 정의된 드리프트 필드입니다.

#### 2. 멀티 스케일 및 위치 활용 (Multi-scale Features)

* 하나의 특징 벡터만 사용하는 대신, 인코더의 여러 계층(Scale)과 공간적 위치(Location)에서 추출된 다양한 특징들을 결합하여 사용합니다.
* ResNet 스타일 인코더: 여러 단계의 피처 맵을 사용하여 훈련에 필요한 더 풍부한 그래디언트 정보를 제공합니다.
* 공식 확장: 각 특징 $j$에 대한 손실을 모두 합산하여 최종 손실을 구합니다.

$$\sum_{j}\mathbb{E}[||\phi_{j}(x)-stopgrad(\phi_{j}(x)+V(\phi_{j}(x)))||^{2}]$$

#### 3. 왜 특징 공간인가?

* 의미적 유사성 (Semantic Similarity): 드리프트 모델의 핵심인 커널 함수 $k(\cdot, \cdot)$가 잘 작동하려면, 의미적으로 유사한 샘플들이 특징 공간상에서 가깝게 위치해야 합니다.
* 자기주도 학습 (Self-supervised Learning): 이를 위해 MoCo, SimCLR 또는 본 논문에서 제안하는 latent-MAE와 같은 사전 훈련된 자기주도 학습 모델을 특징 추출기로 사용합니다.
* 훈련 전용: 특징 인코더는 훈련 시에만 사용되며, 추론(Inference) 시에는 전혀 필요하지 않습니다. 따라서 생성기 자체는 단일 패스로 가볍게 작동합니다.

#### 4. 기존 방식(지각 손실)과의 차이점

| 구분 | 구분지각 손실 (Perceptual Loss) | 특징 공간 드리프트 (Feature-space Drifting) |
| :--- | :--- | :--- |
| 목표 | 생성된 샘플과 특정 타겟 샘플 사이의 거리 최소화 | 생성된 분포 ($\phi_{\sharp} q$)와 데이터 분포 ($\phi_{\sharp} p$)를 일치시킴 |
| 쌍(Pairing) | 생성물 $x$와 타겟 $x_{target}$의 쌍이 필요함 (Supervised) | 샘플 간의 쌍이 필요 없으며 분포 관점에서 접근함 (Unsupervised) |

### 3.5. Classifier-Free Guidance

* 확산 모델에서 생성 품질을 높이기 위해 널리 사용되는 CFG 기법을 드리프트 모델(Drifting Models)의 프레임워크에 어떻게 통합했는지 설명

#### 1. 기본 원리

* 분류기 없는 가이드(CFG)는 클래스 조건부 분포와 무조건부 분포 사이를 외삽(extrapolate)하여 생성 품질을 향상시키는 기법입니다.
* 드리프트 모델에서 클래스 레이블 $c$가 주어지면, 목표 데이터 분포 $p$는 $p_{data}(\cdot|c)$가 되며 여기서 양성 샘플($y^+$)을 추출합니다.

#### 2. 가이드 구현 방식: 음성 샘플의 혼합

* 가이드를 달성하기 위해, 음성 샘플(Negative samples)의 분포를 다음과 같이 혼합하여 정의합니다

$$\tilde{q}(\cdot|c) \triangleq (1-\gamma)q_{\theta}(\cdot|c) + \gamma p_{data}(\cdot|\emptyset)$$

* 여기서 $\gamma \in [0, 1)$는 혼합 비율(mixing rate)이며, $p_{data}(\cdot|\emptyset)$는 무조건부 데이터 분포를 나타냅니다.
* 이 학습 목표를 통해 최종적으로 모델이 근사하고자 하는 분포 $q_\theta$는 다음과 같은 선형 결합 형태가 됩니다

$$q_{\theta}(\cdot|c) = \alpha p_{data}(\cdot|c) - (\alpha - 1) p_{data}(\cdot|\emptyset)$$

* 이때 $\alpha = \frac{1}{1-\gamma} \ge 1$이며, 이는 기존 CFG의 정신과 일치합니다.

#### 3. 실무적 적용 및 장점

* 실제 구현: 훈련 시에 생성된 데이터 외에도 무조건부 데이터( $p_{data}(\cdot|\emptyset)$ )에서 추가적인 음성 샘플을 추출하여 사용합니다.
* 훈련 시 동작: 기존 확산 모델은 추론 시에 조건부/무조건부 모델을 모두 실행해야 하므로 연산량이 늘어나지만, 드리프트 모델에서 CFG는 훈련 시의 행동으로 설계되었습니다.
* 원스텝 유지: 결과적으로 추론(Inference) 시에는 별도의 추가 연산 없이 단 한 번의 신경망 실행(1-NFE)으로 가이드가 적용된 고품질 이미지를 생성할 수 있는 속도 측면의 장점을 그대로 유지합니다.


---

## 4. Implementation for Image Generation

### 1. 토크나이저 (Tokenizer)
* 기본적으로 모델은 잠재 공간(Latent Space)에서 생성을 수행합니다. 
* 이를 위해 표준적인 SD-VAE 토크나이저를 채택하여, 이미지를 $32\times32\times4$ 차원의 잠재 공간으로 압축하여 처리합니다.

### 2. 네트워크 아키텍처 (Architecture)

* 구조: 생성기( $f_\theta$ )는 DiT(Diffusion Transformer)와 유사한 구조를 가집니다.
* 입출력: $32\times32\times4$ 차원의 가우시안 노이즈를 입력받아 동일한 차원의 잠재 벡터를 출력합니다.
* 설계: 패치 크기는 2를 사용하며(DiT/2), 클래스 조건부 및 기타 추가 조건을 처리하기 위해 adaLN-zero 구조를 사용합니다.

### 3. 분류기 없는 가이드 (Classifier-Free Guidance, CFG)

* Drifting 모델은 훈련 단계에서 CFG를 통합하도록 설계되었습니다.
* 훈련 시: CFG 스케일 $\alpha$ 를 무작위로 샘플링하여 이에 따른 음성 샘플(Negative samples)을 준비하고, 네트워크를 이 값에 조건화합니다.
* 추론 시: 추가적인 훈련 없이도 가이드 스케일 $\alpha$를 자유롭게 지정하여 조절할 수 있으며, 이 과정에서도 단 한 번의 신경망 실행(1-NFE)으로 생성이 완료됩니다.

### 4. 배치 구성 (Batching) 및 훈련

* 실제 훈련 시에는 여러 개의 클래스 레이블( $N_c$ )을 샘플링하고, 각 레이블에 대해 독립적으로 알고리즘을 수행합니다.
* 훈련 에포크: 생성된 샘플 $x$의 수를 기준으로 정의되며, 전체 데이터셋 크기를 배치 크기로 나눈 반복 횟수를 한 에포크로 간주합니다.

### 5. 특징 추출기 (Feature Extractor)

* 모델은 특징 공간에서의 Drifting 손실(Drifting Loss)을 사용하여 훈련됩니다.
* 모델 종류: 주로 self-supervised 방식으로 사전 훈련된 ResNet 스타일의 인코더(MoCo, SimCLR 등) 또는 본 연구에서 직접 개발한 latent-MAE를 사용합니다.
* 다중 스케일: ResNet의 여러 단계에서 특징 맵을 추출하여 다중 스케일 및 다중 위치에서의 손실을 합산함으로써 더 풍부한 그래디언트 정보를 얻습니다.

### 6. 픽셀 공간 생성 (Pixel-space Generation)

* 잠재 공간뿐만 아니라 픽셀 공간에서의 직접 생성도 지원합니다.
* 이 경우 입출력은 모두 $256\times256\times3$ 차원이 되며, 패치 크기는 16을 사용합니다(DiT/16).
* 특징 추출기는 픽셀 공간에 직접 적용되어 Drifting 손실을 계산합니다.

---

## 5. Experiments

### 5.1. Toy Experiments

<p align = 'center'>
<img width="521" height="586" alt="image" src="https://github.com/user-attachments/assets/f177f3bd-f23e-444b-8374-ad33ce0ff590" />
</p>


#### 1. 생성 분포의 진화 (Evolution of the generated distribution)
* 2D 환경에서 생성된 분포 $q$(주황색)가 훈련 시간에 따라 두 개의 모드를 가진 타겟 분포 $p$(파란색)로 진화하는 과정을 시각화하여 보여줍니다.
* 실험에서는 세 가지 서로 다른 초기화 상태를 가정했습니다
* Case 1: 두 모드 사이에서 시작한 경우.
* Case 2: 두 모드 모두에서 멀리 떨어진 곳에서 시작한 경우.
* Case 3: 하나의 모드에만 붕괴(collapse)된 상태로 시작한 경우.


#### 2. 모드 붕괴에 대한 견고성 (Robustness to mode collapse)

* 실험 결과, 세 가지 초기화 상태 모두에서 모델은 모드 붕괴 현상 없이 타겟 분포를 정확하게 근사해냈습니다.
* 특히 한 모드에만 뭉쳐서 시작한 경우(Case 3)에도 성공적으로 다른 모드를 찾아냈습니다.
* 원리: 만약 생성 분포 $q$가 하나의 모드에만 쏠리더라도, 타겟 분포 $p$의 다른 모드들이 샘플들을 끌어당기는 힘(attraction)을 작용시키기 때문에 샘플들이 계속 이동하게 되고, 결국 $q$가 전체 분포를 덮을 때까지 진화하게 됩니다.

#### 3. 샘플의 진화 및 손실 값의 변화 (Evolution of the samples)

<p align = 'center'>
<img width="525" height="391" alt="image" src="https://github.com/user-attachments/assets/e9008526-3a5e-4856-bff0-47a39ea37433" />
</p>

* 작은 MLP(다층 퍼셉트론) 생성기를 사용하여 2D 평면상의 점(샘플)들이 훈련 반복(Iteration)에 따라 어떻게 이동하는지를 보여줍니다.
* 손실 함수(Loss): 훈련이 진행됨에 따라 손실 값( $||V||^2$, 즉 표류장(drifting field)의 제곱 노름 )이 점진적으로 감소합니다.
* 수렴: 손실 값이 줄어들어 표류(drift)가 최소화되는 평형 상태(equilibrium)에 도달하면, 생성된 분포 $q$가 데이터 분포 $p$와 거의 일치하게 됩니다 ( $p \approx q$ ).


### 5.2. ImageNet Experiments


#### 1. 실험 설정 및 기본 조건
* 데이터셋: ImageNet $256\times256$ 해상도 이미지를 사용합니다.
* 기본 모델: 절제 연구(Ablation)를 위해 SD-VAE 잠재 공간에서 100 epoch 동안 훈련된 B/2 모델을 사용했습니다.
* 평가 지표: 5만 장의 생성된 이미지에 대해 FID(Fréchet Inception Distance) 점수를 측정했습니다.

#### 2. 주요 절제 연구 (Ablation Studies)

<p align = 'center'>
<img width="528" height="304" alt="image" src="https://github.com/user-attachments/assets/db151941-7859-411f-895c-e1f773765b80" />
</p>

* 반대칭성(Anti-symmetry)의 중요성모델의 평형 상태를 유지하기 위해서는 표류장( $V$ )이 반드시 반대칭($V_{p,q} = -V_{q,p}$)이어야 합니다. 이를 의도적으로 깨뜨렸을 때 성능이 급격히 저하됨을 확인했습니다.


<p align = 'center'>
<img width="529" height="290" alt="image" src="https://github.com/user-attachments/assets/d30f5048-d634-4db2-a2c2-b45fa17aae01" />
</p>

* 긍정 및 부정 샘플 할당 $V$ 를 추정하기 위해 사용되는 긍정 샘플( $N_{pos}$ )과 부정 샘플( $N_{neg}$ )의 수를 늘릴수록 생성 품질이 향상되었습니다.
* 이는 더 큰 샘플 세트가 표류장을 더 정확하게 추정하여 표현 학습을 돕기 때문입니다.

#### 3. 특징 공간(Feature Space)의 역할

* Drifting 모델은 픽셀 자체가 아닌 특징 공간에서의 손실을 계산합니다. 어떤 특징 인코더를 사용하느냐가 성능에 결정적인 역할을 합니다.
* 인코더 비교: SimCLR나 MoCo 같은 표준 인코더도 효과가 있지만, 잠재 공간에서 직접 훈련된 latent-MAE가 가장 뛰어난 성능을 보였습니다.
* 미세 조정: 분류기(Classifier)로 미세 조정한 latent-MAE를 사용할 경우 FID가 3.36까지 개선되었습니다.
* 필수성: 특징 인코더 없이 훈련할 경우 커널이 유사성을 효과적으로 설명하지 못해 학습에 실패했습니다.

#### 4. 시스템 레벨 성능 비교 (SOTA)

* Drifting 모델은 단 한 번의 실행(1-NFE)만으로도 기존의 다단계 확산(Diffusion) 및 흐름(Flow) 모델과 경쟁하거나 이를 압도하는 결과를 보여주었습니다.

* 잠재 공간 생성 (Latent-space)

<p align = 'center'>
<img width="534" height="485" alt="image" src="https://github.com/user-attachments/assets/8784153a-d630-4592-9885-723fb0d69277" />
</p>

* 픽셀 공간 생성 (Pixel-space)

* VAE 없이 픽셀을 직접 생성하는 방식에서도 1.61 FID를 기록하며, 기존의 GAN(StyleGAN-XL: 2.30 FID)보다 훨씬 적은 연산량(87G FLOPs)으로 더 높은 품질을 달성했습니다.

<p align = 'center'>
<img width="538" height="547" alt="image" src="https://github.com/user-attachments/assets/f45b41e9-e85e-4349-9448-b4abffce9cec" />
</p>

### 5.3. Experiments on Robotic Control


#### 1. 실험 설계 및 프로토콜

* 벤치마크: 실험 설계와 프로토콜은 기존의 Diffusion Policy(Chi et al., 2023)를 따릅니다.
* 모델 교체: Diffusion Policy의 핵심인 다단계(multi-step) 확산 기반 생성기를 본 논문의 단일 단계(one-step) Drifting 모델로 교체하여 성능을 비교했습니다.
* 특징 공간 미사용: 이미지 생성과 달리, 로봇 제어에서는 특징 공간(feature space)을 사용하지 않고 제어를 위한 원시 표현(raw representations) 위에서 직접 Drifting 손실을 계산했습니다.

#### 2. 주요 실험 결과 (Table 7 참조)

* Drifting 모델은 단 1회의 연산(1-NFE)만으로 100회의 연산(100-NFE)을 수행하는 최첨단 Diffusion Policy와 대등하거나 이를 능가하는 성공률을 기록했습니다.

<p align = 'center'>
<img width="533" height="555" alt="image" src="https://github.com/user-attachments/assets/bc342dc2-ac07-4a9e-a9dd-e4630961acac" />
</p>

#### 3. 실험의 의의

* 효율성: 100단계가 필요했던 기존 확산 모델의 성능을 단 1단계 만에 달성함으로써 제어 시스템의 추론 속도를 획기적으로 높일 수 있음을 보여줍니다.
* 범용성: 이미지뿐만 아니라 로봇의 상태값이나 시각 관측 데이터를 활용한 행동 생성에서도 Drifting 모델이 유망한 패러다임임을 시사합니다.

---

## 6. Discussion and Conclusion


### 1. 새로운 생성 패러다임: Drifting 모델

* 훈련 시 분포 진화: Drifting 모델은 훈련 과정 중에 푸시포워드(pushforward) 분포가 진화하는 과정을 모델링하는 새로운 패러다임을 제시합니다.
* 업데이트 규칙 중심: 딥러닝의 반복적인 최적화 과정에서 발생하는 업데이트 규칙( $x_{i+1} = x_i + \Delta x_i$ )에 집중합니다.
* 추론의 효율성: 추론 시에 반복적인 업데이트를 수행하는 확산(Diffusion) 또는 흐름(Flow) 기반 모델과 대조적으로, Drifting 모델은 자연스럽게 단일 단계(one-step) 추론을 수행합니다.

### 2. 이론적 및 실제적 한계점

* 이론적 역방향 증명: 연구를 통해 $q=p$일 때 표류장 $V=0$이 된다는 것은 보여주었으나, 그 역인 $V \to 0$이 $q \to p$로 이어지는지에 대한 이론적 조건은 여전히 미해결 과제로 남아 있습니다.
* 디자인 최적화: 현재 제시된 표류장의 형태, 커널의 종류, 특징 인코더의 구조 및 생성기 아키텍처 등은 여전히 최적화될 여지가 많은 하위 선택지들입니다.

### 3. 미래 전망 및 결론

* 훈련의 재해석: 본 연구는 신경망의 반복 훈련을 확산/흐름 모델의 기초가 되는 미분 방정식과는 다른 관점에서의 분포 진화 메커니즘으로 재프레임화했습니다.
* 연구 영감: 저자들은 이러한 관점이 향후 다른 형태의 분포 진화 메커니즘을 탐구하는 데 영감을 주기를 희망하며 글을 맺습니다.


---

## Drifting 훈련 과정 정리

### 1. 동력원: 표류장($V$) 설계

* 가장 중요한 도구는 샘플이 어디로 이동해야 할지 알려주는 표류장(Drifting Field, $V$ )입니다. 이 장(field)은 생성된 샘플을 다음 두 가지 힘으로 움직입니다
* 인력(Attraction): 실제 데이터 분포( $p_{data}$ )가 있는 방향으로 샘플을 끌어당깁니다.
* 척력(Repulsion): 현재 생성된 샘플들( $q$ )끼리는 서로 밀어내어 모드 붕괴(특정 형태만 생성되는 현상)를 방지합니다.
* 이 두 힘이 균형을 이루어 $V=0$이 되면, 생성된 분포가 실제 데이터와 일치하는 평형 상태에 도달한 것으로 간주합니다.

### 2. 훈련 목표: 고정점(Fixed-point) 최적화
* 네트워크가 샘플 $x$를 만들면, 훈련 루프는 이 샘플을 표류장만큼 이동시킨 지점( $x + V$ )을 새로운 목표 지점으로 설정합니다.
* 손실 함수(Loss)

$$\mathcal{L} = \mathbb{E}_{\epsilon} [ \| f_{\theta}(\epsilon) - \text{stopgrad}(f_{\theta}(\epsilon) + V) \|^2 ]$$

* 여기서 중요한 점은 stop-gradient를 사용하여 목표 지점을 고정해 두고, 네트워크가 그 지점을 향해 결과물을 내뱉도록 학습시킨다는 것입니다.
    * stop-gradient는 특정 변수를 계산 과정에서는 사용하되, 역전파(back-propagation) 시에는 해당 변수를 상수로 취급하여 미분값이 흐르지 않게 막는 연산

### 3. 실제 훈련 루프 (Step-by-Step)

* 샘플 생성: 노이즈( $\epsilon$ )를 네트워크( $f_\theta$ )에 통과시켜 가짜 샘플( $x$ )을 만듭니다.
* 특징 추출: 가짜 샘플과 진짜 샘플을 특징 인코더(MAE 등)에 통과시켜 고차원 특징 값을 얻습니다.
    * 픽셀 단위보다 특징 단위에서 유사도를 계산하는 것이 더 정확하기 때문입니다.
* 표류량 계산: 가짜 샘플과 진짜 샘플 사이의 거리를 계산하여 각 샘플이 움직여야 할 방향과 거리( $V$ )를 구합니다.
* 업데이트: 네트워크가 처음부터 $x + V$라는 '더 나은' 결과물을 바로 출력할 수 있도록 가중치를 업데이트합니다.

## SDE/ODE Traditional 방법

### SDE 기반 모델 (확률미분방정식, 주로 Diffusion)
* 데이터에 노이즈를 섞는 과정(Forward)과 이를 복원하는 과정(Reverse)을 연속적인 시간축에서 정의합니다.
* 핵심 수식 (Forward SDE)데이터를 점진적으로 가우시안 노이즈로 만드는 과정입니다.

$$dx = f(x, t)dt + g(t)dw$$

* $f(x, t)$: Drift coefficient (경향성)
* $g(t)$: Diffusion coefficient (확산성)
* $w$: 위너 프로세스 (Brownian motion)

### 훈련 방식: Score Matching

* SDE 기반 모델(예: Score-based Generative Models)의 핵심은 Score function( $\nabla_x \log p_t(x)$ )을 학습하는 것입니다.
* 앤더슨(Anderson)의 정리에 따르면, 위 SDE의 역과정(Reverse SDE)은 다음과 같이 정의됩니다.

$$dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{w}$$

* 따라서 모델 $s_\theta(x, t)$가 실제 데이터의 점수 함수인 $\nabla_x \log p_t(x)$를 모사하도록 학습합니다.
### 목적 함수 (Denoising Score Matching)

$$\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, x_t} \left[ \lambda(t) \| s_\theta(x_t, t) - \nabla_{x_t} \log p(x_t | x_0) \|^2 \right]$$

* 실제 구현에서는 노이즈 $\epsilon$을 예측하는 Diffusion Loss와 수학적으로 연결됩니다.

* 이 방식은 데이터와 노이즈 사이의 '이동 경로(Trajectory)'를 정의하고, 모델이 그 길을 잘 따라가도록 학습시키는 것이 핵심입니다.

---



