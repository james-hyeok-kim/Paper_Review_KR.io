# Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

저자 :

Cheng Chi∗1, Zhenjia Xu∗1, Siyuan Feng2, Eric Cousineau2, Yilun Du3, Benjamin Burchfiel2, Russ Tedrake 2,3, Shuran Song1,4

https://diffusion-policy.cs.columbia.edu/

발표 : 2023년 3월 14일 arXiv, RSS(Robotics: Science and Systems) 2023

논문 : [PDF](https://arxiv.org/pdf/2303.04137)

---

## 1. Introduction

* 로봇의 시각 정보와 동작을 연결하는 비주오모터(Visuomotor) 정책을 생성하는 새로운 방법인 Diffusion Policy를 제안

### 1. 기존 방식의 한계

* 로봇이 인간의 시연으로부터 학습(Imitation Learning)할 때, 기존의 지도 학습 방식(Supervised Regression)은 다음과 같은 문제에 직면합니다.
* 다중 모드 분포(Multimodal distributions): 동일한 상황에서 사람이 왼쪽이나 오른쪽으로 가는 등 여러 가지 정답이 존재할 때, 이를 하나로 평균 내어 출력하려다 보니 부정확한 동작이 발생합니다.
    * Jitter 현상: 왼쪽, 오른쪽 다른 모드에서 출력하면 갈팡질팡 떠는 현상  
* 고차원 동작 공간: 이미지 생성 모델이 고해상도 이미지를 만드는 것처럼 로봇도 미래의 일련의 동작(Sequence)을 한꺼번에 예측해야 하는데, 기존 방식은 이를 안정적으로 처리하기 어렵습니다.
    *  에너지 기반 모델(IBC) : 동작을 지정하는 에너지 모양이 불균형, non-smooth해서 하나 선택하기 어려움
    *  혼합 가우시안 모델(GMM) : 동작 모드가 차원에 따라 복잡도 상승 
* 학습의 불안정성: 에너지 기반 모델(EBM, Energy-Based Models) 등은 학습 과정이 매우 까다롭고 하이퍼파라미터에 민감한 경우가 많습니다.
    * 복잡한 손실 함수를 사용하면서 문제 발생
    * 에너지 기반 모델은 음수 샘플링이 필요한데, 차원이 높을수록 추정치가 부정확
    * 동작 예측 설정했을때, 오히려 성능이 떨어지거나 학습이 제대로 안되는 경우 발생
* 시간 인관성의 부재로 장기 계획이 없음, Step-by-step으로만 일을 처리 (복잡하고 긴 업무에서 일관성 없음)

---
#### 1.1. Appendix

1) 왜 낮은 에너지가 높은 확률인가?
    1) 에너지 기반 모델(EBM)은 통계 물리학의 볼츠만 분포(Boltzmann Distribution) 개념을 차용
    2) 반비례 관계: 확률 밀도 함수 $p(a|o)$를 정의할 때, 에너지 함수 $E(o, a)$에 마이너스(-) 부호를 붙인 지수 함수 형태( $e^{-E}$ )를 사용

2) 음수 샘플링
    1) EBM은 특정 상태(관측값 $o$와 동작 $a$ )에 대해 에너지 값 $E(o, a)$를 할당합니다.
    2) 이를 확률 $p(a|o)$로 변환하려면 모든 가능한 동작에 대한 에너지 합으로 나누어 전체 확률의 합이 1이 되도록 만들어야 합니다.
    3) 공식: $p_{\theta}(a|o) = \frac{e^{-E_{\theta}(o,a)}}{Z(o,\theta)}$ 여기서 $Z(o, \theta)$가 바로 정규화 상수
    4) 모든 가능성을 계산할 수 없어서, 안가본 길 (데이터 셋에 없는 가짜 샘플)을 뽑아서 음수로 지정, 높은 에너지 할당
    5) 문제는 샘플링이 실제 분포를 제대로 반영하지 못하면 에너지 왜곡을 가지고 온다 (Spike 같이)

3) IBC (Implicit Behavioral Cloning)
    1) 동작을 직접 예측하는 대신, 관측값과 동작의 쌍에 대해 에너지를 할당
    2) 작동 원리: 관측값 $o$와 동작 $a$를 입력받아 하나의 스칼라 값인 에너지 $E(o, a)$를 출력합니다. 추론 시에는 이 에너지 값을 최소화하는 동작 $a$를 최적화 과정을 통해 찾아냅니다.
    3) 장점: 에너지가 낮은 지점이 여러 곳일 수 있기 때문에, 전문가의 다중 모드(Multimodal) 동작 분포를 자연스럽게 표현
    4) 단점:
        1) 학습 불안정성(음수 샘플링)
        2) 고차원 샘플링 어려움 (긴 동작 시퀀스 어려움)

5) GMM (Gaussian Mixture Model) 기반 정책
    1) LSTM-GMM 또는 BC-RNN으로 언급되는 방식
    2) 동작 분포를 여러 개의 가우시안 분포의 합으로 표현
    3) 작동 원리: 신경망(주로 RNN이나 LSTM)이 관측값을 입력받아 여러 가우시안 분포의 파라미터(평균, 분산, 가중치)를 출력
    4) 장점: 명시적(Explicit)인 정책 형태를 가지므로 추론 속도가 매우 빠르고 구현이 단순
    5) 단점:
        1) 모드 개수 설정: 사전에 가우시안 모드 개수를 고정해야 한다
             1) (장애물을 피해 물건 옮길때, 왼쪽으로 돌아가는것(모드1), 오른쪽으로 돌아가는것(모드2))
        2) 시간적 일관성 부족: 미래 시퀀서 예측시 동작이 끊기거나 떨리는 현상 발생
        3) 고정된 분포 형태: 비정형적인 데이터를 가우시안 합으로 정확히 근사하기 어려움

---

### 2. Diffusion Policy의 핵심 아이디어

* Diffusion Policy는 동작을 직접 출력하는 대신, 조건부 노이즈 제거 확산 프로세스(Conditional Denoising Diffusion Process)를 사용합니다.
    * CNN에서는 FiLM(Feature-wise Linear Modulation)방식을 사용
    * Transformer에서는 Cross-Attention을 이용
    * 시각관측 정보(카메라) + 로봇 상태 + 관측 수평선(과거 데이터)
* 그라디언트 필드 학습: 모델은 동작 분포의 점수 함수(Score function) 그라디언트를 학습합니다.
* 반복적 최적화: 추론 시점에 가우시안 노이즈에서 시작하여, 학습된 그라디언트 필드를 따라 동작을 반복적으로 정제(Refine)하여 최종 동작을 생성합니다.

### 3. 주요 기술적 기여

* 논문은 물리 로봇에서 확산 모델의 잠재력을 끌어내기 위해 세 가지 주요 요소를 도입했습니다.
* 후퇴 수평선 제어 (Receding Horizon Control): 미래의 동작 시퀀스를 예측하고 실행함으로써 동작의 일관성을 유지하면서도 변화에 유연하게 대응합니다.
* 시각적 조건화 (Visual Conditioning): 시각 관측 정보를 확산 프로세스의 조건으로 입력하여 실시간 추론이 가능하게 설계했습니다.
* 시계열 확산 트랜스포머 (Time-series Diffusion Transformer): 기존 CNN 기반 모델의 과도한 평활화(Over-smoothing) 문제를 해결하기 위해 새로운 트랜스포머 아키텍처를 제안했습니다.

### 4. 연구 결과 요약

* 시뮬레이션과 실제 환경을 포함한 총 15개 작업에서 벤치마크를 수행한 결과, 기존 최첨단(SOTA) 방식들보다 평균 46.9%의 성능 향상을 보이며 그 효과를 입증했습니다.

---

## 2. Diffusion Policy Formulation

### 2.1 DDPM (Denoising Diffusion Probabilistic Models) 기초

* Diffusion Policy는 동작 생성 과정을 Stochastic Langevin Dynamics라고 불리는 노이즈 제거 과정으로 모델링합니다.
* 프로세스: 가우시안 노이즈에서 추출된 $x^{K}$에서 시작하여, $K$번의 반복적인 노이즈 제거 단계를 거쳐 노이즈가 없는 깨끗한 출력 $x^{0}$을 생성합니다.
* 핵심 수식

$$x^{k-1} = \alpha(x^{k} - \gamma\epsilon_{\theta}(x^{k}, k) + \mathcal{N}(0, \sigma^{2}I)) \quad (1)$$

* 여기서 $\epsilon_{\theta}$는 학습을 통해 최적화되는 노이즈 예측 네트워크입니다.
* 해석: 이 과정은 에너지 함수 $E(x)$의 그라디언트 필드를 따라가는 노이즈 섞인 그라디언트 하강법(Noisy Gradient Descent) 단계로 해석될 수 있습니다.

### 2.2 비주오모터 학습을 위한 수정 사항

* 기존의 이미지 생성용 DDPM을 로봇 정책 학습에 적용하기 위해 두 가지 중요한 변화를 주었습니다.

#### 1) 폐루프 동작 시퀀스 예측 (Closed-loop Action-sequence Prediction)

* 단일 시점의 동작이 아니라, 미래의 일련의 동작 궤적을 한꺼번에 예측합니다.
* 입력 ( $O_{t}$ ): 현재 시점 $t$를 기준으로 최근 $T_{o}$ 단계의 관측 데이터 시퀀스.
* 출력 ( $A_{t}$ ): 미래 $T_{p}$ 단계의 동작 시퀀스를 예측하고, 그중 $T_{a}$ 단계만큼 실제로 실행한 뒤 다시 계획을 세웁니다.
* 장점: 시간적 일관성(Temporal Consistency)을 유지하면서도 외부 변화에 반응하는 유연성을 확보합니다.

#### 2) 시각적 관측 조건화 (Visual Observation Conditioning)

* 동작 $A_{t}$를 생성할 때 시각 정보 $O_{t}$를 조건(Condition)으로 입력받아 $p(A_{t}|O_{t})$ 분포를 학습합니다.
* 수정된 노이즈 예측: $\epsilon_{\theta}(O_{t}, A_{t}^{k}, k)$.
* 학습 손실 함수: 실제 노이즈와 예측된 노이즈 사이의 평균 제곱 오차(MSE)를 최소화합니다.

$$L = \text{MSE}(\epsilon^{k}, \epsilon_{\theta}(O_{t}, A_{t}^{k}, k)) \quad(3)$$

### 2.3 Diffusion Policy의 강점

* 다중 모드 표현: 복잡한 인간의 시연 데이터에 존재하는 여러 방식의 해결책(Multimodal)을 자연스럽고 정확하게 표현할 수 있습니다.
* 안정적인 학습: 에너지 기반 모델(EBM)에서 문제가 되는 정규화 상수( $Z$ ) 계산을 피함으로써 학습 과정이 매우 안정적입니다.
    * 기울기 함수 (Score function)를 학습
* 고차원 확장성: 이미지 생성 모델이 증명했듯, 고차원의 동작 시퀀스 출력도 효과적으로 처리합니다.

#### 2.3.1 전통방식

* 특정 관측 $o$에 대한 동작 $a$의 확률 분포 $p(a|o)$는 다음과 같이 정의

$$p_{\theta}(a|o) = \frac{e^{-E_{\theta}(o,a)}}{Z(o, \theta)}$$

* 정규화 상수 $Z(o, \theta)$는 확률의 총합을 1로 만들기 위해 모든 가능한 동작 $a$에 대한 에너지를 적분한 값

$$Z(o, \theta) = \int e^{-E_{\theta}(o, a)} \, da$$

#### 2.3.2 Diffusion의 해법: "기울기"만 보는 Score Matching

* Diffusion Policy는 확률의 '절대적인 높이'를 맞추는 대신, 로그 확률의 기울기(Score function)를 학습하는 방식

* 수학적으로 로그 확률에 대한 미분을 수행

$$\nabla_a \log p_{\theta}(a|o) = \nabla_a \log \left( \frac{e^{-E_{\theta}(o,a)}}{Z(o, \theta)} \right)$$

$$\nabla_a \log p_{\theta}(a|o) = \nabla_a \left( -E_{\theta}(o,a) - \log Z(o, \theta) \right)$$

* 여기서 $\log Z(o, \theta)$는 동작 $a$에 대한 함수가 아니므로, $a$로 미분하면 $0$이 되어 사라짐

$$\nabla_a \log p_{\theta}(a|o) = -\nabla_a E_{\theta}(o,a) \approx -\epsilon_{\theta}(o, a, k)$$

* $Z$로부터의 자유: 이제 모델( $\epsilon_{\theta}$ )은 $Z$를 평가하거나 추정할 필요가 전혀 없습니다.
* 방향성만 학습: 확률의 절대적인 값(높이)이 아니라, 어느 방향으로 가야 에너지가 낮아지는지(기울기)만 배우면 됩니다.


---


## 3. Key Design Decisions

### 1. 네트워크 아키텍처 선택 (CNN vs. Transformer)

* 모델의 중추가 되는 $\epsilon_{\theta}$(노이즈 예측 네트워크)를 구현하기 위해 두 가지 아키텍처를 검토했습니다.

* CNN 기반 Diffusion Policy
    * 1D Temporal CNN을 활용하며, 시각 관측 정보는 FiLM 방식을 통해 모든 레이어에 주입됩니다.
    * 장점: 하이퍼파라미터 튜닝 없이 대부분의 작업에서 안정적으로 작동합니다.
    * 단점: 속도 명령처럼 신호가 급격하게 변하는 고주파 동작에서는 성능이 떨어지는 경향이 있습니다.
* 시계열 확산 트랜스포머 (Time-series Diffusion Transformer)
    * CNN의 과도한 평활화(Over-smoothing) 효과를 줄이기 위해 제안된 새로운 구조입니다.
        * 동작이 급격하게 변해야 하는 구간에서 뭉뜽그려 부드럽게 예측 해버리는 현상
        * 1D Convolution은 완만한 흐름을 선호하는 유도 편향(Inductive Bias)를 가집니다.
    * 장점: 작업의 복잡도가 높고 동작의 변화가 빠른 작업에서 최첨단 성능을 보여줍니다.
        * Diffusion Transformer에서는 전역적으로 파악하기 때문에 강제로 부드럽게 연결하는 경향이 적음
        * Input Token Embedding + Cross-Attention + Causal Attention
    * 단점: 하이퍼파라미터에 매우 민감하며 튜닝이 어렵습니다.
* 권장 사항: 새로운 작업에는 먼저 CNN 기반으로 시작하고, 성능이 부족할 경우 트랜스포머로 전환하는 것을 추천합니다.

### 2. 시각 인코더 (Visual Encoder)

* 가공되지 않은 이미지 시퀀스(Raw Images)를 저차원 특징인 $O_t$로 변환하는 역할을 하며, 정책망(Policy Network)과 엔드투엔드(End-to-End)로 함께 학습됩니다.
    * Policy Network
        * CNN: 1D CNN (시계열)
        * Transformer: Time + Condition input
    * 기본 구조: ResNet-18을 기반으로 하되 로봇 제어에 특화된 두 가지 수정을 가했습니다.
    * 공간 소프트맥스(Spatial Softmax): 전역 평균 풀링(Global Average Pooling) 대신 사용하여 물체의 위치와 같은 공간 정보를 더 잘 유지하도록 했습니다.
        * 각 채널의 feature map에서 softmax
    * 그룹 정규화(GroupNorm): 확산 모델에서 흔히 쓰이는 EMA(지수 이동 평균)와 결합했을 때 학습을 안정화하기 위해 BatchNorm 대신 사용되었습니다.

### 3. 노이즈 스케줄 (Noise Schedule)

* 확산 과정에서 노이즈가 추가되고 제거되는 속도를 제어하며, 이는 모델이 동작의 고주파/저주파 특성을 포착하는 방식에 큰 영향을 미칩니다.
* 최적의 선택: 다양한 실험 결과, 로봇 제어 작업에는 Square Cosine Schedule이 가장 우수한 성능을 보였습니다.
* Square Cosine Schedule
    * 기존의 선형(Linear) 방식 대신 이 방식을 채택
    * 반복 단계 $k$에 따라 변하는 $\sigma, \alpha, \gamma$ 값과 추가되는 가우시안 노이즈 $\epsilon^k$의 함수로 정의

$$f(k) = \cos^2\left(\frac{k/K + s}{1 + s} \cdot \frac{\pi}{2}\right)$$

$$\bar{\alpha}_k = \frac{f(k)}{f(0)}$$

* $k$: 현재 확산 단계 ( $0 \sim K$ )
* $K$: 전체 확산 단계 수 (이 논문에서는 주로 100)
* $s$: 아주 작은 오프셋(주로 0.008). $k=0$일 때 노이즈가 너무 작아지는 것을 방지합니다.

* 정보 손실의 선형성 (Linearity of Information Loss)
    * Linear Schedule: 연산 초기에는 정보가 너무 천천히 사라지다가, 마지막 단계에서 갑자기 모든 정보가 파괴됩니다. 이로 인해 모델이 노이즈가 가득한 마지막 단계에서 유의미한 학습을 하기 어렵습니다.
    * Square Cosine Schedule: 수식에 포함된 $\cos^2$ 덕분에 연산 전체 과정에서 정보(Signal)가 선형적으로 매끄럽게 감소합니다. 즉, 모든 단계 $k$에서 모델이 골고루 학습할 수 있는 환경을 제공합니다.

* 노이즈 비율( $\beta_k$ )의 연산
    * 실제 코드 구현 시에는 각 단계에서 추가할 노이즈 양인 $\beta_k$를 다음 연산을 통해 구하게 됩니다
    * 코사인 스케줄을 사용하면 $\beta_k$ 값이 초반에 급격히 커지지 않고 완만하게 증가하므로, 로봇 동작 시퀀스의 미세한 디테일(고주파 성분)을 더 잘 보존하며 연산

$$\beta_k = 1 - \frac{\bar{\alpha}_k}{\bar{\alpha}_{k-1}}$$

---



## 4.

---


## 5. 

---



## 6. 

---
