# Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

저자 :

Cheng Chi∗1, Zhenjia Xu∗1, Siyuan Feng2, Eric Cousineau2, Yilun Du3, Benjamin Burchfiel2, Russ Tedrake 2,3, Shuran Song1,4

https://diffusion-policy.cs.columbia.edu/

발표 : 2023년 3월 14일 arXiv, RSS(Robotics: Science and Systems) 2023

논문 : [PDF](https://arxiv.org/pdf/2303.04137)

---

## 1. Introduction

* 로봇의 시각 정보와 동작을 연결하는 비주오모터(Visuomotor) 정책을 생성하는 새로운 방법인 Diffusion Policy를 제안

<p align = 'center'>
<img width="1091" height="498" alt="image" src="https://github.com/user-attachments/assets/1bc17542-b862-4990-a8ce-33be5424e08a" />
</p>


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

4) GMM (Gaussian Mixture Model) 기반 정책
    1) LSTM-GMM 또는 BC-RNN으로 언급되는 방식
    2) 동작 분포를 여러 개의 가우시안 분포의 합으로 표현
    3) 작동 원리: 신경망(주로 RNN이나 LSTM)이 관측값을 입력받아 여러 가우시안 분포의 파라미터(평균, 분산, 가중치)를 출력
    4) 장점: 명시적(Explicit)인 정책 형태를 가지므로 추론 속도가 매우 빠르고 구현이 단순
    5) 단점:
        1) 모드 개수 설정: 사전에 가우시안 모드 개수를 고정해야 한다
             1) (장애물을 피해 물건 옮길때, 왼쪽으로 돌아가는것(모드1), 오른쪽으로 돌아가는것(모드2))
        2) 시간적 일관성 부족: 미래 시퀀서 예측시 동작이 끊기거나 떨리는 현상 발생
        3) 고정된 분포 형태: 비정형적인 데이터를 가우시안 합으로 정확히 근사하기 어려움

5) BET(Behavior Transformer)
    1) 아키텍처: 내부적으로는 minGPT와 같은 트랜스포머 디코더 구조를 사용하여 시각 정보로부터 동작을 예측
    2) k-모드 학습: "하나의 모델로 k개의 모드를 학습한다"는 목표 아래, 클러스터링 기법을 사용하여 전문가의 다양한 행동 양식을 포착하려 시도
    3) 동작의 이산화(Quantization): 연속적인 로봇의 동작 공간을 그대로 사용하는 대신, 이를 여러 개의 격자(Bin)로 나누어 분류(Classification) 문제로 변환하여 처리

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

### 5. 수식적 차이

#### 5.1. 기본 수식

* Implicit BC: 정책 $\pi_{\theta}$를 관측치 $o$와 동작 $a$에 대한 연속 **에너지 함수 $E_{\theta}(o, a)$의 최소화 문제(argmin)**로 정의합니다.
   * $\pi_{\theta}$는 정책을 표기하는 기호
   * $o \to a$로 가는 지도

$$\pi_{\theta}(o) = \text{argmin}_a E_{\theta}(o, a)$$


* Diffusion Policy: 정책을 조건부 노이즈 제거 확산 프로세스(Conditional Denoising Diffusion Process)로 모델링합니다. 직접적인 에너지 값을 출력하는 대신, 동작 분포의 점수 함수(Score function)의 기울기를 학습합니다.
    * $x^K \to x^0$로 가는 계단의 한 칸

$$x^{k-1} = \alpha(x^k - \gamma \epsilon_{\theta}(O_t, A_t^k, k) + \mathcal{N}(0, \sigma^2 I))$$


#### 5.2. 손실 함수 (Loss Function)

* Implicit BC: 정답 동작의 에너지는 낮추고 샘플링된 가짜 동작의 에너지는 높이도록 하는 InfoNCE 손실 함수를 사용합니다.

$$\mathcal{L}_{InfoNCE} = \sum_{i=1}^N -\log(\tilde{p}_{\theta}(y_i | x, \{\tilde{y}_i^j\}))$$

* Diffusion Policy: 실제 노이즈와 모델이 예측한 노이즈 사이의 차이를 줄이는 평균 제곱 오차(MSE) 손실 함수를 사용합니다.

$$\mathcal{L} = \text{MSE}(\epsilon^k, \epsilon_{\theta}(O_t, A_t^k, k))$$


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

$$L = \text{MSE}(\epsilon^{k}, \epsilon_{\theta}(O_{t}, A_{t}^{k}, k)) \quad(5)$$

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

## 4. Intriguing Properties of Diffusion Policy

<p align = 'center'>
<img width="553" height="399" alt="image" src="https://github.com/user-attachments/assets/b07d8dcf-2841-4e65-9238-bed5f77d374a" />
</p>

### 4.1 다중 모드 동작 분포 모델링 (Multi-Modal Action Distributions)

* Diffusion Policy의 가장 큰 장점 중 하나는 인간의 시연 데이터에 포함된 다양한 해결 방식(Multi-modality)을 자연스럽고 정교하게 학습한다는 점입니다. 
* 동적 수렴: 초기 샘플( $A_t^K$ )은 서로 다른 수렴 분지(Convergence basins)를 지정하는 데 도움을 주며, 이후 반복적인 최적화 과정을 통해 개별 동작 샘플이 서로 다른 모드 사이를 이동하거나 특정 모드로 수렴할 수 있게 합니다.
    * 왼쪽 돌기, 오른쪽 돌기와 같이 두개의 solution이 있을 때, 초기 위치에 따라 왼쪽, 오른쪽이 결정되나 K번 iteration이후 하나로 수렴, Jitter 없이 동작
* 일관된 모드 선택: 예를 들어 T자형 블록을 밀 때 왼쪽이나 오른쪽 중 한 방향을 선택해야 한다면, Diffusion Policy는 한 번의 실행(Rollout) 동안 하나의 모드를 확실히 선택하여 유지합니다. 반면, BET 같은 모델은 시간적 일관성이 부족해 두 모드 사이에서 갈팡질팡하며 떨리는 현상이 발생합니다. 

### 4.2 위치 제어와의 시너지 (Synergy with Position Control)

* 이 연구에서 발견한 놀라운 사실 중 하나는 Diffusion Policy가 위치 제어(Position Control) 모드에서 동작할 때 성능이 극대화된다는 점입니다.
* 기존 관행과의 차이: 대부분의 기존 모방 학습 연구는 속도 제어(Velocity Control)에 의존해 왔으나, Diffusion Policy는 위치 제어에서 훨씬 더 우수한 성능을 보였습니다.
* 이유: 위치 제어 모드에서는 동작의 다중 모드 특성이 더 뚜렷하게 나타나는데, Diffusion Policy는 이를 매우 잘 표현할 수 있기 때문입니다. 또한 위치 제어는 속도 제어보다 오차 누적 효과가 적어 시퀀스 예측에 더 적합합니다.
* 유휴 동작(Idle Actions)에 대한 강인함: 텔레오퍼레이션 중 발생하는 일시 정지나 느린 움직임(Idle actions)에 대해 기존 모델(BC-RNN, IBC)은 멈춰버리는 등 오버피팅되기 쉽지만, Diffusion Policy는 이러한 유휴 동작이 포함된 데이터에서도 안정적으로 작동합니다. 

### 4.3 동작 시퀀스 예측의 이점 (Action-Sequence Prediction)

* Diffusion Policy는 단일 단계가 아니라 미래의 일련의 동작들(High-dimensional action sequence)을 한꺼번에 예측합니다.
* 고차원 확장성: 이미지 생성 모델이 고해상도 이미지를 잘 만드는 것처럼, DDPM은 출력 차원이 높아져도 모델의 표현력을 잃지 않고 잘 확장됩니다.
* 비교 우위: IBC는 고차원 공간에서 샘플링하는 데 어려움을 겪고, GMM 방식은 모드의 개수를 미리 정해야 하는 한계가 있지만, Diffusion Policy는 이러한 제약 없이 고차원 시퀀스를 효과적으로 생성합니다.
* 최적의 예측 길이: 실험 결과, 약 8단계의 미래 동작을 예측하는 것이 시간적 일관성과 반응성 사이의 가장 좋은 균형을 보여주었습니다. 

### 4.4 학습 안정성 (Training Stability)
* 에너지 기반 모델(EBM)인 IBC와 비교했을 때, Diffusion Policy는 학습 과정이 매우 안정적입니다.
* 정규화 상수의 문제: IBC는 확률 분포를 정의하기 위해 계산이 불가능한 정규화 상수( $Z$ )를 음수 샘플링으로 추정해야 하며, 이 과정이 학습 불안정성을 유발합니다.
* 점수 함수 학습: Diffusion Policy는 $Z$와 무관한 점수 함수(Score function)를 직접 모델링하므로, 학습 및 추론 과정에서 $Z$를 평가할 필요가 전혀 없어 매우 안정적인 수렴을 보여줍니다. 

### 4.5 제어 이론과의 연결 (Connections to Control Theory)
* 단순한 선형 시스템의 경우, Diffusion Policy의 최적 노이즈 제거기(Optimal denoiser)는 결과적으로 해당 작업을 수행하기 위한 시스템 역학 모델(Dynamics model)을 암시적으로 학습하게 된다는 것을 수학적으로 증명했습니다. 이는 딥러닝 기반의 정책이 고전적인 제어 이론과도 일맥상통하는 지점이 있음을 시사합니다.
    *  훈련 과정에서 모델은 물리 법칙($A, B$)을 억지로라도 배울 수밖에 없다

* 시스템 상태 ( $s$ ): 현재 로봇의 위치나 속도입니다.
* 물리 법칙 ( $A, B$ ): 다음 상태가 어떻게 변할지 결정하는 행렬입니다. $s_{t+1} = As_t + Ba_t$라는 식으로 표현되죠.
* 전문가의 정답 ( $a = -Ks$ ): 숙련된 조종사는 현재 상태 $s$를 보고 최적의 동작 $a$를 결정합니다 (예: LQR 제어기).
    * 제어 이론(Control Theory)에서 $a = -Ks$와 같이 마이너스( $-$ ) 부호를 사용하는 가장 근본적인 이유는 현재 상태가 목표에서 벗어난 방향의 반대 방향으로 힘을 가해야 다시 제자리로 돌아올 수 있기 때문

$$s_{t+1} = As_t + B(-Ks_t) = (A - BK)s_t$$


---


## 5. Evaluation

### 1. 평가 환경 및 방법론

* 벤치마크 구성: RoboMimic(5개 작업), Push-T, Multi-modal Block Pushing, Franka Kitchen 등 4개 벤치마크를 사용했습니다.

* 비교 모델 (Baselines): 기존의 주요 모델인 LSTM-GMM(RoboMimic의 BC-RNN), IBC, BET를 베이스라인으로 설정했습니다.

* 실험 규모: 3개의 트레이닝 시드와 50개의 서로 다른 초기 환경 조건에서 총 1,500회 이상의 실험을 수행하여 평균 성공률을 도출했습니다.

### 2. 주요 실험 결과 및 통찰 (Key Findings)

* 실험 결과, Diffusion Policy는 모든 테스트에서 기존 모델들을 압도하며 평균 46.9%의 성능 향상을 기록했습니다.
* 핵심 발견 사항, 다중 모드(Multimodal) 해결 능력
    * 단기(Short-horizon): Push-T 작업에서 왼쪽/오른쪽 중 한 방향을 확실히 선택하여 부드럽게 작업했습니다.
    * 장기(Long-horizon): 여러 단계가 필요한 Kitchen 작업에서 하위 목표들을 유연하게 완수하며 베이스라인 대비 213% 이상의 성능 향상을 보였습니다.
* 위치 제어(Position Control)의 우위: 기존 방식들이 속도 제어를 선호했던 것과 달리, Diffusion Policy는 위치 제어 환경에서 훨씬 더 높은 성능을 보였습니다.
* 액션 호라이즌(Action Horizon)의 균형: 미래의 동작을 너무 짧게(1단계) 혹은 너무 길게(64단계 이상) 예측하는 것보다, 8단계 정도를 예측하는 것이 일관성과 반응성 측면에서 가장 효과적이었습니다.
* 지연 시간(Latency) 강인함: 이미지 처리 및 추론에 의한 지연이 발생하더라도, 미래 동작 시퀀스를 미리 예측하는 특성 덕분에 약 4단계의 지연까지는 성능 저하 없이 견뎌냈습니다.
* 학습 안정성: IBC가 학습 중 성능이 요동치는 것과 달리, Diffusion Policy는 하이퍼파라미터 변화에도 매우 안정적인 성능을 유지했습니다.

### 3. 비전 인코더 절제 연구 (Ablation Study)

<p align = 'center'>
<img width="536" height="283" alt="image" src="https://github.com/user-attachments/assets/ab6e3dac-58a4-45c8-86b0-5c59ecfbecd0" />
</p>

* 시각 정보를 어떻게 처리하는 것이 가장 좋은지 확인하기 위한 추가 실험도 진행되었습니다.
* 최적의 전략: 이미 학습된(Pre-trained) 비전 인코더를 그대로 쓰거나(Frozen), 처음부터 새로 학습시키는(Scratch) 것보다, 기존 인코더를 가져와 작은 학습률로 미세 조정(Fine-tuning)하는 것이 가장 좋은 성능을 보였습니다.
* 트랜스포머의 성능: 특히 CLIP으로 학습된 ViT-B/16 모델을 미세 조정했을 때 단 50에포크 만에 98%의 성공률을 달성했습니다.


---


## 6. Realworld Evaluation

### 1. 하드웨어 설정 및 작업 개요

* UR5 로봇 스테이션: 실제 환경의 Push-T 작업을 수행하는 데 사용되었습니다.
* Franka 로봇 스테이션: 머그컵 뒤집기(Mug Flipping), 소스 붓기(Sauce Pouring), 소스 펴바르기(Sauce Spreading) 작업을 수행했습니다.
* 제어 주기: 정책은 10Hz 주기로 명령을 내리며, 이는 로봇 실행을 위해 더 높은 주파수(125Hz~1kHz)로 보간(interpolation)되어 전달됩니다.

### 2. 주요 실제 작업 및 결과

* Diffusion Policy는 모든 작업에서 기존의 베이스라인(IBC, LSTM-GMM)을 크게 앞지르는 성능을 보여주었습니다.

<div align = 'center'>
   
| 작업 명칭 | 주요 특징 및 난이도 | 성공률 |
| :--- | :--- | :---: |
| **Realworld Push-T** | 다단계 작업(밀기 후 엔드이펙터 치우기), 미세한 위치 조정 필요 | 95% |
| **Mug Flipping** | 복잡한 3D 회전(6자유도), 파지법의 다중 모드성(잡기 vs 밀기) | 90% |
| **Sauce Pouring** | 비강체(액체) 다루기, 국자를 채우기 위한 유휴(Idle) 동작 포함 | 79% |
| **Sauce Spreading** | 주기적 동작(나선형 패턴), 소스의 불규칙한 흐름 대응 | 100% |

</div>


### 3. 핵심 분석 및 강점

* 엔드투엔드(End-to-End) 학습의 중요성
    * 비교 결과: ImageNet이나 R3M 같은 사전 학습된 비전 인코더를 사용하는 것보다, 정책망과 비전 인코더를 함께 학습시키는 방식이 가장 좋은 성능을 보였습니다.
    * 효과: R3M을 사용한 모델은 동작이 떨리거나 멈추는 경향이 있었지만, 엔드투엔드 모델은 훨씬 매끄럽고 정확하게 움직였습니다.
* 외부 방해에 대한 강인함 (Robustness)
    * 시각적 차단: 카메라 앞을 손으로 가려도(3초간) 정책은 궤도를 유지하며 작업을 완수했습니다.
    * 물체 위치 변경: 작업 도중 T자형 블록을 옮기면, 로봇은 즉시 계획을 수정하여 반대 방향에서 밀어내는 등 새로운 행동을 합성해 냈습니다.

---

## 7. Realworld Bimanual Tasks

### 1. 양손 작업을 위한 시스템 확장

* 관측 및 동작 공간: 두 로봇 팔의 엔드이펙터 포즈와 그리퍼 너비 정보를 포함하도록 확장되었습니다. 
* 시각 정보: 두 대의 장면 카메라와 각 팔에 부착된 손목 카메라(총 4대)의 이미지를 결합하여 사용합니다. 
* 원격 조작(Teleoperation): 양손의 조화로운 움직임을 기록하기 위해 Meta Quest Pro VR 기기나 햅틱 장치(Haption Virtuose)를 도입했습니다.

### 2. 주요 양손 협업 작업

<p align = 'center'>
<img width="540" height="389" alt="image" src="https://github.com/user-attachments/assets/831d8da8-f0a9-45d8-b7f4-5e314717d53c" />
</p>

1) 양손 달걀 거품기 (Bimanual Egg Beater)
    1) 작업 내용: 한 팔로 거품기를 잡고 볼(Bowl) 안에 넣은 뒤, 다른 팔로 손잡이를 3회 이상 돌리는 복잡한 도구 사용 작업입니다.
    2) 특이사항: 이 작업은 힘 조절이 중요하여 햅틱 피드백 없이는 전문가조차 시연을 완료하기 어려웠으나, 햅틱으로 수집된 데이터를 통해 Diffusion Policy는 55%의 성공률을 기록했습니다.

<p align = 'center'>
<img width="537" height="296" alt="image" src="https://github.com/user-attachments/assets/362c6382-b8f2-49f3-b01d-932801461700" />
</p>

2) 양손 매트 펼치기 (Bimanual Mat Unrolling)
    1) 작업 내용: 말려 있는 매트의 한쪽을 잡고 들어 올려 테이블 위에 평평하게 펼치는 작업입니다.
    2) 결과: 초기 상태에 따라 왼쪽이나 오른쪽 중 어느 쪽으로도 펼칠 수 있는 능숙함을 보였으며, 75%의 성공률을 달성했습니다. 

<p align = 'center'>
<img width="544" height="633" alt="image" src="https://github.com/user-attachments/assets/cd30f9fb-209d-4b5f-bf4a-62396c302407" />
</p>

3) 양손 셔츠 접기 (Bimanual Shirt Folding)
    1) 작업 내용: 소매를 접고, 셔츠를 끌어당겨 정렬한 뒤 반으로 접는 등 최대 9단계에 이르는 매우 긴 호라이즌(Long-horizon) 작업입니다.
    2) 결과: 두 그리퍼가 매우 가깝게 접근해야 하는 충돌 위험 상황에서도 75%의 성공률을 보이며 안정적으로 작업을 완수했습니다.

### 3. 주요 성과 및 한계

* 확장성 입증: 별도의 알고리즘 수정 없이도 양손의 협동이 필요한 고난도 작업을 수행할 수 있음을 입증했습니다.
* 실패 원인: 주로 초기 파지(Grasp) 실패 시 이를 스스로 교정하지 못하고 같은 행동을 반복하는 '고착' 현상이 주요 실패 원인으로 지목되었습니다. 


---

## 8. Related Work

### 1. 명시적 정책 (Explicit Policy)
* 관측 정보를 동작으로 직접 매핑하는 가장 일반적인 형태의 정책입니다.
    * 이미지 입력 -> CNN/Transformer -> MLP -> 동작(Action) 출력
    * 전통적인 모방 학습(Behavior Cloning, BC) 모델
    * Vanilla BC
    * BC-RNN / LSTM-Policy
    * ResNet-18 + MLP
* 특징: 직접적인 회귀(Regression) 손실 함수를 통해 학습하며, 신경망의 한 번의 순전파(Forward pass)로 동작을 생성하므로 추론이 매우 빠릅니다.
    * 정답과 똑같은 동작을 출력하자 (MSE = 0)
    * $MSE(action, pred\_action)$
        * Diffusion: $MSE(noise, pred\_noise)$
* 한계: 전문가 시연에 포함된 다중 모드(Multimodal) 행동을 모델링하기 어렵고, 고정밀 작업에서 성능이 떨어지는 경향이 있습니다.
* 보완 시도: 동작 공간을 이산화(Discretization)하거나 혼합 밀도 네트워크(MDN) 등을 사용하기도 하지만, 차원의 저주나 모드 붕괴(Mode collapse) 문제, 하이퍼파라미터 민감도 문제가 여전히 존재합니다.
    * Mixture Density Network: 일반적인 회귀 모델은 하나의 숫자(평균)만 출력하지만, MDN은 여러 개의 가우시안 분포(정규분포)를 섞어서 출력
    * 일반 회귀: "왼쪽(-10)과 오른쪽(+10)의 평균인 0(정면 충돌)으로 가라
    * MDN: "40% 확률로 -10(왼쪽)에 수렴하는 분포가 있고, 60% 확률로 +10(오른쪽)에 수렴하는 분포

### 2. 암시적 정책 (Implicit Policy)

* 에너지 기반 모델(EBM)을 사용하여 동작의 분포를 정의하는 방식입니다.
    * 관측값( $s$ )과 함께 '가상의 동작( $a$ )'을 같이 입력합니다. 모델은 이 동작이 얼마나 좋은지 에너지(점수)만 매겨
    * 에너지 함수 $E(s, a)$ 안에 숨겨져(Implicitly defined) 있기 때문에 암시적이라고 말한다.
* 특징: 각 동작에 에너지 값을 할당하고 에너지가 최소인 동작을 찾는 최적화 문제를 풉니다. 여러 동작에 낮은 에너지를 할당할 수 있어 다중 모드 분포를 자연스럽게 표현합니다.
* 한계: 학습을 위한 손실 함수 계산 시 '음수 샘플(Negative samples)'을 추출해야 하는데, 이 과정이 매우 불안정하여 실제 적용이 어렵습니다.


### 3. 확산 모델 (Diffusion Models)

* 무작위 노이즈를 데이터 분포로 반복해서 정제하는 최신 생성 모델입니다.
* 핵심 원리: 암시적 동작 점수(Score)의 그라디언트 필드를 학습하고 추론 시 이를 최적화하는 것으로 이해할 수 있습니다.
* 기존 연구와의 차이:기존 연구들은 주로 오프라인 강화 학습이나 경로 계획(Planning)에 확산 모델을 사용했습니다.
* 본 논문의 기여: 확산 모델을 비주오모터 제어 정책으로 효과적으로 결합하기 위해 폐루프 제어(Closed-loop control), 새로운 트랜스포머 아키텍처, 시각적 정보 통합 방법을 제안했습니다.특히 실 로봇 시스템에서 중요한 후퇴 수평선(Receding-horizon) 계획, 위치 제어의 중요성, 실시간 추론 최적화 등을 중점적으로 다루어 차별화했습니다.

---

## 9. Limitations and Future Work

### 한계점 (Limitations)

* 모방 학습의 한계 계승: 기본적으로 행동 복제(Behavior Cloning) 방식을 따르기 때문에, 전문가의 시연 데이터가 부족할 경우 성능이 최적화되지 못하는 문제점을 그대로 가지고 있습니다.
    * Diffusion은 행동복제(BC: Behavior Cloning 방식을 채택), 데이터가 부족하면 training 이 잘 안됨
    * 로봇의 시각-운동 정책(visuomotor policy)를 학습할때, 지도학습 장식의 일종인 행동복제 사용
* 높은 계산 비용 및 지연 시간: LSTM-GMM과 같은 기존의 단순한 방식들에 비해 계산 비용이 더 많이 들고 추론 시 지연 시간(Inference Latency)이 깁니다.
    * Diffusion: iteration
    * 16~100배 정도 연산량 차이   
* 고주파 제어의 어려움: 행동 시퀀스 예측 방식이 지연 문제를 일부 완화해주기는 하지만, 매우 빠른 반응 속도가 필요한 고주파 제어 작업에는 충분하지 않을 수 있습니다.
    * 고주파 제어 : 날아오는 공 잡기, 수술 로봇, 흔들리는 물체 균형
    * 저주파 제어 : 물건 옮기기
    * 고주파 제어에서는 속도가 느려서 잘 안될 수 있다.

### 향후 과제 (Future Work)

* 다양한 학습 패러다임 적용: 최적이 아니거나 부정적인 데이터를 활용할 수 있도록 강화 학습(Reinforcement Learning)과 같은 다른 패러다임에 확산 정책을 적용하는 연구가 필요합니다.
    * 현재는 지도학습   
* 추론 속도 가속화: 추론에 필요한 단계를 줄여 지연 시간을 단축하기 위해 새로운 노이즈 스케줄, 추론 솔버(Inference Solvers), 일관성 모델(Consistency Models) 등 최신 확산 모델 가속 기술들을 활용할 수 있습니다.

---

## 10. Conclusion

### 1. 압도적인 성능과 안정성

* 범용적 성능 검증: 시뮬레이션과 실제 환경을 포함한 총 15개의 다양한 작업을 통해 확산 기반 정책의 타당성을 평가했습니다.
* 기존 모델 압도: Diffusion Policy는 기존의 로봇 학습 방법들을 일관되고 결정적으로 압도하는 성능을 보여주었습니다.
* 학습 용이성: 성능뿐만 아니라 학습 과정이 안정적(Stable)이며 훈련하기 쉽다는 점이 입증되었습니다.

### 2. 성능 극대화를 위한 3대 설계 요소

* 후퇴 수평선 제어 (Receding-horizon control): 미래의 행동 시퀀스를 예측하고 실행하는 방식.
* 말단 장치 위치 제어 (End-effector position control): 속도 제어보다 위치 제어가 확산 모델과 더 큰 시너지를 냄.
* 효율적인 시각 조건화 (Visual conditioning): 실시간 제어를 위해 시각 정보를 효율적으로 처리하는 방식.

### 3. 새로운 통찰: "정책 구조가 병목이다"

* 구조의 중요성: 행동 복제(Behavior Cloning)의 품질에는 데이터의 양이나 로봇의 물리적 능력 등 많은 요소가 영향을 미치지만, 이 연구 결과는 '정책 모델의 구조' 자체가 성능의 심각한 병목 현상이었음을 시사합니다.
* 미래 방향: 단순히 데이터를 늘리는 것을 넘어, 확산 기반의 정책 구조를 탐구하는 것이 로봇 학습의 발전에 매우 중요하다는 점을 역설하며 마무리됩니다.


---


---
