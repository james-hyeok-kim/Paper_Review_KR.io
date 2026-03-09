# Implicit Behavioral Cloning

저자 : 
Pete Florence, Corey Lynch, Andy Zeng, Oscar Ramirez, Ayzaan Wahid,

Laura Downs, Adrian Wong, Johnny Lee, Igor Mordatch, Jonathan Tompson

Robotics at Google

발표 : CoRL(Conference on Robot Learning) 2021

논문 : [PDF](https://arxiv.org/pdf/2109.00137)


---

## 1. Introduction

<p align = 'center'>
<img width="885" height="269" alt="image" src="https://github.com/user-attachments/assets/a67e3894-b5eb-49a5-98cb-804359e65328" />
</p>

### 1. 기존 방식의 한계: 명시적 정책 (Explicit Policies)

* 정의: 대부분의 기존 BC 정책은 입력 관측치 $o$를 출력 동작 $a$로 직접 매핑하는 $\hat{a}=F_{\theta}(o)$ 형태의 명시적 연속 피드포워드 모델(예: 딥 네트워크)을 사용합니다.
* 문제점
    * 이러한 명시적 모델은 복잡하고 잠재적으로 불연속적이거나 다가(multi-valued, set-valued) 함수를 근사하는 데 어려움이 있습니다.
    * 데이터가 불연속적인 경우, 명시적 모델은 훈련 샘플 사이의 모든 중간 값을 취하며 보간(interpolation)하는 특성이 있어 아티팩트(artifacts)가 발생할 수 있습니다.

### 2. 제안된 방식: 암시적 정책 (Implicit Policies)

* 개념: 정책을 직접적인 매핑이 아닌, 관측치와 동작을 모두 입력으로 받는 연속 에너지 함수 $E_{\theta}(o, a)$의 최소화 문제(argmin)로 재구성합니다.
* 공식: $\pi_{\theta}(o) = \text{argmin}_a E_{\theta}(o, a)$
* 작동 방식: 추론 시점에 주어진 관측치 $o$에 대해 샘플링 또는 경사 하강법을 통한 최적화를 수행하여 최적의 동작을 찾아내는 암시적 회귀(Implicit Regression)를 수행합니다.
* EBM 활용: 이를 위해 조건부 에너지 기반 모델(Energy-Based Modeling, EBM) 문제를 해결하는 방식을 채택합니다.

### 3. 연구의 핵심 결과 및 의의

* 성능 향상: 암시적 모델을 사용한 간단한 변경만으로도 정밀한 블록 삽입(1mm 오차 허용), 물체 분류, 바이매뉴얼(bi-manual) 조작 등 접촉이 많은(contact-rich) 작업에서 성능이 크게 향상되었습니다.
    * 접촉이 많은 작업: 블록을 제자리에 넣기 위해 살짝씩 미세하기 밀거나 조정하는 동작
    * Bi-manual: 두개의 로봇 팔이 협업하는 작업(물건 쓸어담기)
* 데이터 효율성 및 범용성: 보상 정보 없이도 표준 시뮬레이션 벤치마크(D4RL)에서 최신 오프라인 강화학습(RL) 알고리즘과 대등하거나 더 나은 성능을 보여주었습니다.
    * D4RL(Datasets for Deep Data-driven Reinforcement Learning): 오프라인 강화학습 연구를 위한 표준 데이터 셋
        * Reward(보상) 없이, 인간 시연 데이터만을 활용하여 학습
        * Adroit(고난도 손 조작)
            * 펜돌리기, 망치질, 문열기, 물체이동
        * Franka Kitchen(주방 보조 작업)
            * Task: 전자레인지 열기, 가스레인지 켜기, 전등 스위치 누리기, 캐비닛 열기 등
            * 데이터 구성 : 'Complete'(모든 작업 수행), 'Partial'(일부 작업 수행), 'Mixed'(다양한 시도 혼합) 등으로 나뉘어 모델의 일반화 능력을 테스트
        * Gym-Mujoco (기초 이동 제어)
            * 종류: HalfCheetah(달리기), Hopper(뜀뛰기), Walker2d(걷기) 등이 포함.
            * 특이사항: 이 환경들은 주로 단일 모드의 연속적인 동작을 다루기 때문에, 암시적(Implicit) 모델보다 기존의 명시적(Explicit) 모델이 더 유리할 수 있는 영역 
* 이론적 뒷받침: 암시적 모델은 다중 모드(multi-modal) 분포뿐만 아니라 불연속 함수를 표현할 수 있는 능력이 있음을 이론적으로 정당화합니다.



---

## 2. Background: Implicit Model Training and Inference


### 1. 암시적 모델의 정의

* 구성: 범용 함수 근사기 $E: \mathbb{R}^{m+n} \to \mathbb{R}^1$를 사용하여 에너지 함수 $E_\theta(x, y)$를 정의합니다. 
* 추론 원리: 입력 $x$가 주어졌을 때, 에너지 $E_\theta(x, y)$를 최소화하는 $y$값을 찾아내어 이를 예측값 $\hat{y}$으로 정의합니다.

$$\hat{y} = \text{argmin}_y E_\theta(x, y)$$

### 2. 학습 방법 (Training)

* 암시적 모델을 학습시키기 위해 에너지 기반 모델(EBM)의 기법과 InfoNCE 스타일의 손실 함수를 사용합니다. 
* 데이터 구성: 데이터셋 ${x_i, y_i}$와 함께 회귀 범위(bounds)인 $y_{min}, y_{max}$를 사용합니다. 
* 음적 샘플 생성 (Negative Counter-examples): 각 정답 샘플 $y_i$에 대해, 에너지가 높아야 할(정답이 아닌) 반대 사례인 ${\tilde{y}i^j}{j=1}^{N_{neg}}$를 생성합니다. 
* 손실 함수 (Loss Function): 정답 샘플의 에너지는 낮추고, 음적 샘플의 에너지는 높이도록 설계된 Negative Log Likelihood 손실 함수를 적용합니다. 이는 주어진 $x$에 대해 정답 $y$가 나타날 확률 $p_\theta(y|x) = \frac{\exp(-E_\theta(x, y))}{Z(x, \theta)}$을 최대화하는 과정입니다.


#### 1. $\tilde{p}_\theta$ (예측 확률 함수) 설명

* 이 부분은 주어진 관측치( $x_i$ )에서 특정 동작( $y_i$ )이 선택될 확률을 상대적인 에너지 값으로 계산합니다.

$$\tilde{p}_\theta(y_i | x, \{\tilde{y}_i^j\}_{j=1}^{N_{neg}}) = \frac{e^{-E_\theta(x_i, y_i)}}{e^{-E_\theta(x_i, y_i)} + \sum_{j=1}^{N_{neg}} e^{-E_\theta(x_i, \tilde{y}_i^j)}}$$

* 분자 ( $e^{-E_\theta(x_i, y_i)}$ ): 실제 전문가의 동작( $y_i$ )에 대한 에너지 기반 지수 값입니다. 에너지가 낮을수록(즉, $E_\theta$ 값이 작을수록) 확률값은 커집니다.
* 분모: 전문가의 동작 에너지와 음적 샘플(Negative samples, $\{\tilde{y}_i^j\}$ ) 에너지들의 합입니다.여기서 $\tilde{y}_i^j$는 모델을 속이기 위해 생성된 가짜 동작(Counter-examples)입니다.
* 의미: 수많은 선택지(정답 + 오답 샘플들) 중에서 정답 동작이 선택될 확률을 소프트맥스(Softmax) 형태로 정규화한 것입니다.

#### 2. $\mathcal{L}_{InfoNCE}$ (전체 손실 함수) 설명

* 이 함수는 모델의 전체 학습 목표를 정의하며, 정보 이론에서의 대비 학습(Contrastive Learning) 방식을 따릅니다.
    * 대비 학습 : 정답과 오답이 어떻게 다른지 학습

$$\mathcal{L}_{InfoNCE} = \sum_{i=1}^{N} -\log(\tilde{p}_\theta(y_i | x, \{\tilde{y}_i^j\}_{j=1}^{N_{neg}}))$$

* $-\log$ (Negative Log): 확률( $\tilde{p}_\theta$ )이 1에 가까워질수록(정답일 확률이 높을수록) 손실 값은 0에 수렴하고, 확률이 낮을수록 손실 값은 무한대로 커집니다.
* 학습 결과: 이 손실 함수를 최소화하면 모델은 자연스럽게 다음 두 가지를 수행하게 됩니다.
* 정답 에너지 낮추기: 실제 전문가가 한 동작 $y_i$의 에너지 $E_\theta(x_i, y_i)$를 최소화합니다.
* 오답 에너지 높이기
    * 무작위로 생성되거나 샘플링된 가짜 동작 ( $\tilde{y}_i^j$ )
    * 해당 가짜 동작의 에너지 ( $E_\theta(x_i, \tilde{y}_i^j)$ )를 상대적으로 높입니다.


### 3. 추론 방법 (Inference)

* 학습된 에너지 모델 $E_\theta(x, y)$가 있다면, 새로운 입력 $x$에 대해 에너지를 최소화하는 $y$를 찾기 위해 확률적 최적화(Stochastic Optimization)를 수행합니다. 

### 3.1. 미분 미사용(Derivative-free) 최적화

<p align = 'center'>
<img width="876" height="378" alt="image" src="https://github.com/user-attachments/assets/948158b2-cef4-49b5-b0f4-605abc686843" />
</p>

* 경사 값을 사용하지 않고 샘플링을 통해 최솟값을 찾는 방식입니다.
* 반복적 리샘플링 (Iterative Resampling)
    * 초기화: 동작 범위( $y_{min}, y_{max}$ ) 내에서 균등 분포( $\mathcal{U}$ )를 통해 다수의 후보 샘플( $N_{samples}$ )을 무작위로 뿌립니다.
    * 에너지 계산 및 확률 변환: 각 샘플에 대해 모델($E_\theta$)을 통과시켜 에너지를 계산하고, 이를 소프트맥스(Softmax)를 통해 확률값( $\tilde{p}_i$ )으로 변환합니다. 에너지가 낮을수록 선택될 확률이 높습니다.
    * 리샘플링 (Resampling): 계산된 확률에 따라 샘플들을 다시 뽑습니다 (복원 추출). 이때 확률이 높았던(에너지가 낮았던) 샘플들은 여러 번 뽑히고, 확률이 낮았던 샘플들은 제거됩니다.
    * 노이즈 추가 및 범위 축소: 새로 뽑힌 샘플들에 약간의 가우시안 노이즈( $\sigma$ )를 더해 주변을 탐색하게 하고, 다음 반복 회차에서는 노이즈의 크기를 줄여( $K\sigma$ ) 탐색 범위를 좁힙니다.
    * 최종 결정: 정해진 횟수( $N_{iters}$ )만큼 반복한 후, 가장 확률이 높은(에너지가 가장 낮은) 샘플을 최종 동작( $\hat{y}$ )으로 채택합니다.
* 장점 
    * 비볼록(Non-convex) 최적화에 강함: 에너지 지형이 복잡하거나 여러 개의 골짜기(Multi-modal)가 있어도 경사에 갇히지 않고 전역 최솟값을 찾을 가능성이 높습니다.
    * 구현의 단순성: 모델의 미분 가능 여부와 상관없이 에너지 함수값만 계산할 수 있으면 바로 적용 가능합니다.
    * 현실적인 성능: 논문에서는 보통 3회의 반복( $N_{iters}=3$ )과 약 16,384개의 샘플( $N_{samples}$ )을 사용하여 실시간 로봇 제어(약 7.22ms 소요)가 가능함을 보여주었습니다.
* 한계점
    * 차원의 저주: 동작 공간의 차원이 높아지면(예: 5차원 초과) 무작위 샘플링으로 유망한 지역을 찾아낼 확률이 급격히 낮아져 성능이 떨어집니다. 이 경우 논문에서는 자기회귀(Autoregressive) 방식이나 랑제빈(Langevin) 방식을 대안으로 제시합니다.

### 3.2 자기회귀적(Auto-regressive) 최적화

<p align = 'center'>
<img width="877" height="429" alt="image" src="https://github.com/user-attachments/assets/96cbb346-18c4-4964-ac36-2ecce0011322" />
</p>

#### 3.2.1. 핵심 메커니즘: 차원별 순차 결정
* 단계적 모델링: 첫 번째 차원 $y^1$을 먼저 결정한 뒤, 그 결과값을 입력으로 넣어 두 번째 차원 $y^2$를 결정합니다. 이 과정을 $m$번째 차원까지 반복합니다.
* 조건부 입력: $j$번째 차원의 에너지를 계산하는 모델 $E_\theta^j(x, y^{:j})$는 관측치 $x$와 이전에 결정된 $1$부터 $j$까지의 동작 차원값들을 모두 입력으로 받습니다.
* 샘플링 효율화: 한 번에 모든 차원의 조합을 찾는 대신, 한 번에 1차원씩만 샘플링하면 되므로 훨씬 적은 수의 샘플로도 정확한 최솟값을 찾아낼 수 있습니다.

#### 3.2.2. 알고리즘 과정 (Algorithm 2)
* 초기화: 전체 동작 차원에 대해 후보 샘플들을 무작위로 생성합니다.
* 순차 최적화 (For $j$ in $1 \dots m$ )
    * 현재 차원 $j$에 대해 에너지를 계산합니다.
        * 한번에 모든 차원을 결정하는 것이 아니라 순차적으로 좁혀나가는 방식
        * 이후 나머지 모든 차원(1 ~ m) softmax하여 값 결정  
    * 소프트맥스를 통해 확률을 계산하고, 해당 차원의 값을 리샘플링합니다.
    * 노이즈를 추가하고 범위를 축소하며 해당 차원의 최적값을 확정합니다.
* 다음 차원으로 이동: 확정된 $y^j$ 값을 고정하고 $y^{j+1}$ 차원에 대해 위 과정을 반복합니다.

#### 3.3.3. 장점과 단점

* 장점 (고차원 대응): 동작 공간이 커져도(예: 논문에서의 12~32차원 작업) 안정적으로 최적화가 가능합니다.
* 단점 (메모리 및 계산량): $m$개의 차원에 대해 별도의 모델이 필요하거나 순차적인 계산이 필요하므로 메모리 사용량이 많고 추론 속도가 단일 샘플링 방식보다 느려질 수 있습니다.


### 3.3. 랑제빈 샘플링(Langevin Sampling)

* 에너지 함수의 경사(Gradient) 정보를 활용하여 점진적으로 최솟값에 접근하는 방식입니다.

#### 3.3.3. 장점과 한계

* 장점 (고차원 확장성): 동작 차원이 30차원에 달하는 D4RL 과제나 N-D Particle 환경에서도 단 하나의 모델로 효과적인 추론이 가능합니다.
* 장점 (이론적 우수성): 단순히 값을 찾는 것을 넘어 동작의 확률 분포 자체를 샘플링할 수 있습니다.
* 단점 (구현 복잡성): 경사 기반 최적화 과정에서 하이퍼파라미터 튜닝이 까다롭고, 경사 안정화를 위한 추가적인 손실 함수 처리가 필요합니다.

#### 3.3.4. 실제 적용 결과

* D4RL Human-Experts: 동작 차원 최대 30D.
* N-D Particle Environment: 동작 차원 최대 32D까지 테스트.


<div align = 'center'>

| 구분 | 명시적(Explicit) 모델 | 암시적(Implicit) 모델 |
| :--- | :--- | :--- |
| **수식** | $\hat{y} = F_\theta(x)$ | $\hat{y} = \text{argmin}_y E_\theta(x, y)$ |
| **추론 방식** | 단일 네트워크 통과 (Feed-forward) | 최적화 과정을 통한 반복적 탐색 |
| **특징** | 계산이 빠르나 불연속성 표현에 취약함 | 계산량은 많으나 복잡한 다중 모드 분포 표현에 강함 |

</div>

---

## 3. Intriguing Properties of Implicit vs. Explicit Models

### 3.1. 불연속성 (Discontinuities) 처리

<p align = 'center'>
<img width="827" height="486" alt="image" src="https://github.com/user-attachments/assets/bc1389b4-9c16-4842-a2f9-4749fa9b51de" />
</p>

* 암시적 모델(Implicit Model): 중간 아티팩트 없이 불연속 지점을 날카롭게(sharply) 근사할 수 있습니다. 지점 간의 연결을 억지로 이어붙이지 않고, 에너지 지형의 골짜기가 끊어지는 형태로 표현하기 때문입니다. 
* 명시적 모델(Explicit Model): 기본적으로 연속 함수를 데이터에 맞추려(fitting) 하기 때문에, 훈련 샘플 사이의 모든 중간 값을 통과하며 보간(interpolation)하게 됩니다. 이로 인해 불연속 지점에서 원치 않는 중간값들이 예측되는 문제가 발생합니다.
* 데이터에 여러 가지 정답이 공존할 때, 명시적 모델의 보간 특성은 치명적인 오류를 범함
    * 로봇 앞에 장애물이 있어 왼쪽으로 피하거나 오른쪽으로 피해야 하는 상황이 있다고 가정
    * 훈련 데이터 범위를 벗어난 외삽(Extrapolation) 상황에서 예측값이 급격히 불안정해질 수 있다

### 3.2. 다중성 (Multi-valued functions) 표현

* 하나의 입력에 대해 여러 개의 유효한 정답이 존재하는 경우에 대한 비교입니다.
* 암시적 모델(Implicit Model): 에너지 함수 $E_\theta(x, y)$를 통해 여러 개의 국소 최솟값(Local Minima)을 동시에 가질 수 있습니다. 따라서 argmin 연산은 상황에 따라 여러 값의 집합(set of values)을 반환할 수 있어 다중 모드(Multi-modal) 분포를 자연스럽게 표현합니다.
* 명시적 모델(Explicit Model): MDN(Mixture Density Networks) 같은 특수한 구조를 쓰더라도 복잡한 다중 모드 분포를 맞추는 데 한계가 있으며, 일반적인 MSE 모델은 여러 정답의 평균값을 내버려 둘 다 틀리는 결과를 초래하기 쉽습니다.


### 3.3. 외삽 (Extrapolation) 능력

<img width="929" height="349" alt="image" src="https://github.com/user-attachments/assets/35d20822-ab1b-4a57-97d1-36f375d6d4ec" />


* 훈련 데이터 범위를 벗어난 지역( $x \in [0, 1]$ 외부)에서의 동작을 다룹니다. 
* 명시적 모델 (Explicit): 훈련 데이터 전체를 하나의 연속된 함수로 연결하려다 보니, 데이터가 없는 영역으로 나가면 단순한 선형 확장을 시도하거나 혹은 학습된 곡선의 기울기에 따라 엉뚱한 방향으로 값이 튀어버리는 경향이 있습니다.
* 암시적 모델 (Implicit): 데이터 경계 근처에 있는 가장 가까운 조각별 선형(Piecewise linear) 부분의 기울기를 유지하며 확장합니다. 이는 마치 "가장 최근에 본 경향성"을 그대로 이어가는 것과 같아서, 불연속적인 함수에서도 훨씬 합리적인 예측을 내놓습니다.
* (c), training data 이외에서 잘되는것 파랑, 안되는것 빨강
      


---

## 4. Policy Learning Results

### 1. D4RL 표준 벤치마크 결과

<p align = 'center'>
<img width="918" height="334" alt="image" src="https://github.com/user-attachments/assets/a4a4e8fd-b8a1-48d2-a295-67a0754db497" />
<img width="923" height="389" alt="image" src="https://github.com/user-attachments/assets/ba3b8500-3688-4e51-ba69-7962536555ce" />
</p>


* 오프라인 강화학습의 표준인 D4RL 데이터셋 중 인간 전문가의 시연 데이터가 포함된 가장 어려운 작업들을 대상으로 실험했습니다.
* 놀라운 발견: 암시적(EBM) 및 명시적(MSE) 정책 모두 벤치마크에 보고된 기존 BC 베이스라인을 크게 상회했습니다.
    * Nearest-Neighbor/BC/CQL/S4RL가 Baseline
* 최신 알고리즘과 경쟁: 보상(Reward) 정보를 전혀 사용하지 않았음에도 불구하고, CQL이나 S4RL과 같은 최신 오프라인 강화학습 알고리즘과 대등하거나 더 나은 결과를 보여주었습니다.
   * CQL(Conservative Q-Learning): 학습 데이터에 없는 동작을 수행할때, 위험한 행동을 하지 않도록 하는 것
   * S4RL(Surprisingly Simple Self-Supervision for Offline RL): 기존 데이터 셋에 노이즈를 더해 데이터 증강 하여 더 다양한 상황에 대비할 수 있게
* 데이터 품질 민감도: 상위 50%의 우수한 시연 데이터만 사용했을 때(RWR 방식), 암시적 모델은 명시적 모델보다 성능 향상 폭이 훨씬 커 데이터 품질을 더 잘 활용하는 것으로 나타났습니다.

### 2. N-D Particle 환경 (불연속성 및 고차원 테스트)

* 단순한 환경에서 불연속성과 차원의 변화가 성능에 미치는 영향을 고립시켜 분석했습니다.
* 불연속성 학습: 목표 지점에 도달하면 다음 목표로 즉시 전환해야 하는 불연속적인 정책을 학습해야 합니다.
* 차원의 저주 극복: 차원을 1에서 32까지 높였을 때, 명시적(MSE) 정책은 8차원까지만 성공했지만, 암시적 정책은 16차원까지 95%의 성공률을 유지했습니다.

### 3. 시뮬레이션 기반 조작 작업

* Simulated Pushing: 로봇 팔로 블록을 밀어 목표 지점에 넣는 작업으로, 단일 목표뿐 아니라 다단계(Multi-stage) 작업에서도 암시적 모델이 우수했습니다.
* Planar Sweeping: 수많은 작은 입자들을 휩쓸어 모으는 작업으로, 이미지 기반 암시적 모델이 최상의 명시적 모델보다 7% 높은 성능을 보였습니다.
* Bi-Manual Sweeping: 두 개의 로봇 팔을 정밀하게 협응시켜야 하는 12차원 동작 공간 작업에서 암시적 모델이 명시적 모델을 14% 앞섰습니다.

### 4. 실물 로봇 조작 결과 (Real Robot Manipulation)

<p align = 'center'>
<img width="958" height="267" alt="image" src="https://github.com/user-attachments/assets/1899763f-915e-4289-8f29-3984646b53cb" />
</p>

* raw RGB 이미지만을 입력으로 사용하여 4가지 실제 작업을 수행했습니다.
* Push-Red-then-Green: 두 개의 블록을 순서대로 지정된 위치로 이동시키는 작업입니다.
* Insert-Blue (정밀 삽입): 1mm 오차의 좁은 틈에 블록을 넣는 작업으로, 불연속적인 미세 조정이 필수적입니다. 이 작업에서 암시적 모델은 명시적 모델보다 10배 높은 성공률을 기록했습니다.
* Sort-Blue-from-Yellow (분류): 8개의 블록을 색상별로 분류하는 복잡한 작업으로, 암시적 모델이 2.4배 더 높은 성능을 보였습니다.

---

---

---
