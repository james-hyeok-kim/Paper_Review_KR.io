# Mastering Diverse Domains through World Models

저자 :

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap

Google DeepMind

University of Toronto

발표 : arXiv 2023

논문 : [PDF](https://arxiv.org/pdf/2301.04104)

출처 : [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)

---

## 0. Summary

<p align='center'>
<img src="figs/DreamerV3/fig_01.png" alt="Figure 01" width="800"/>
</p>

본 논문은 **DreamerV3**, 즉 Dreamer 계열의 3세대 강화학습(RL) 알고리즘을 제안한다. 핵심 메시지는 단 하나의 고정된 하이퍼파라미터 설정(single fixed configuration)으로 8개 도메인, 150개 이상의 과제에서 각 분야에 맞춰 튜닝된 전문 알고리즘(tuned experts)을 능가한다는 것이다. 특히 사람 데이터나 커리큘럼 없이 Minecraft에서 처음부터(from scratch) 다이아몬드를 채굴한 최초의 알고리즘이다.

### 0.1. 문제 (Problem)

* 기존 RL 알고리즘은 개발된 과제와 비슷한 환경에는 잘 적용되지만, **새로운 도메인으로 옮기면(예: 비디오 게임 → 로봇 제어)** 하이퍼파라미터 튜닝에 막대한 전문 지식·실험·연산이 필요하다.
* 이런 취약성(brittleness) 때문에 PPO 같은 범용 알고리즘조차 전문 알고리즘보다 성능이 낮고, 튜닝 비용이 큰 과제에는 RL 적용 자체가 어렵다.
* **월드 모델(world model) 기반 접근**은 직관적으로 매력적이지만, 보상의 크기(scale)와 빈도(frequency)가 도메인마다 수십~수만 배씩 다르기 때문에 손실 항들의 균형을 맞추며 안정적으로 학습하기가 오랫동안 미해결 문제였다.

### 0.2. 핵심 아이디어 (Core Idea)

* **월드 모델(World Model)** — (a) 환경의 동작 규칙을 신경망으로 학습한 "머릿속 시뮬레이터"다. (b) 실제 환경에서 매번 행동을 시험하면 느리고 위험하므로, 모델 안에서 미래를 미리 그려보고 학습한다. (c) 비유: 체스 선수가 손을 대기 전에 머릿속으로 몇 수 앞을 두어보는 것과 같다. Dreamer는 관측을 **이산 표현(discrete representation, 작은 숫자 코드들의 집합)** $z_t$ 로 압축하고, 순환 상태 $h_t$ 로 다음 표현을 예측한다.
* **상상 학습(Imagination)** — (a) 행동(actor)과 가치(critic)를 실제 데이터가 아니라 월드 모델이 만들어낸 가상 궤적(trajectory) 위에서 학습한다. (b) 실제 상호작용은 비싸므로, 한 번 본 경험으로부터 머릿속에서 수많은 미래를 굴려 정책을 개선한다. (c) 비유: 행동하기 전에 머릿속으로 결과를 시뮬레이션해보는 것.
* **견고성 기법(Robustness Techniques) — 이 논문의 진짜 contribution.** 하나의 설정으로 모든 도메인을 다루기 위한 5가지 장치다.
  * **symlog 변환** — (a) 큰 값과 작은 값을 같은 "자(scale)"로 압축하는 함수 $\mathrm{symlog}(x)=\mathrm{sign}(x)\ln(|x|+1)$. (b) 보상·관측의 크기가 도메인마다 달라도 한 설정으로 학습되게 한다. (c) 비유: 로그 자처럼 큰 수도 작은 수도 한눈에 보이게 눌러주되, 부호는 보존.
  * **symexp twohot 손실** — (a) 보상/가치를 하나의 실수로 회귀하지 않고, 지수 간격으로 배치된 칸(bin)들에 대한 **분포**로 예측한다. (b) 그래디언트 크기를 예측 목표값의 크기로부터 분리해 발산을 막는다. (c) 비유: 정확한 값을 콕 찍는 대신 인접한 두 칸에 확률을 나눠 거는 것.
  * **백분위 수익 정규화(percentile return normalization)** — 수익(return)을 5~95 백분위 범위로 나눠 대략 $[0,1]$ 에 맞추어, 고정된 엔트로피 스케일 $\eta=3\times10^{-4}$ 로 희소 보상에서도 탐험을 유지한다.
  * **KL 균형 + free bits**, **1% unimix** — KL 손실의 두 방향을 분리해 가중치를 다르게 주고(가중치 $\beta_{dyn}=1$, $\beta_{rep}=0.1$), 1나트(nat) 이하에서는 손실을 끄며, 분포를 1% 균등분포와 섞어 결정론적으로 붕괴하지 않게 한다.

### 0.3. 효과 (Effects)

* **하나의 설정으로 모든 도메인**: Atari, ProcGen, DMLab, Minecraft, Atari100k, 자기수용 제어(Proprio Control), 시각 제어(Visual Control), BSuite 등 8개 도메인에서 동일한 하이퍼파라미터로 동작.
* **예측 가능한 스케일링**: 모델 크기(12M→400M)와 리플레이 비율(replay ratio)을 키울수록 성능이 단조 증가하고, 큰 모델은 더 적은 상호작용으로 과제를 해결한다.
* **재현성**: 각 에이전트가 단 1개의 Nvidia A100 GPU로 학습되어 많은 연구실에서 재현 가능.
* **비지도 신호 의존**: 성능이 대부분 월드 모델의 비지도 복원(reconstruction) 손실에 의존 → 향후 비지도 사전학습 활용 가능성.

### 0.4. 결과 (Results)

* **Minecraft Diamond**: Dreamer가 100M 스텝 내에 학습한 모든 에이전트가 다이아몬드를 발견(학습 전체 기준 100%), baseline은 0%. 100M 스텝 평균 수익 Dreamer 9.1 vs IMPALA 7.1 / Rainbow 6.3 / PPO 5.1.
* **Atari (57개, 200M)**: gamer-median 830% 로 MuZero(693%), PPO(180%)를 능가.
* **DMLab (30개, 100M)**: 1B 스텝까지 학습한 IMPALA/R2D2+ baseline을 100M 스텝만으로 능가 → 1000% 이상의 데이터 효율 향상.
* **Proprio / Visual Control, BSuite**: 각각 D4PG·DMPO, DrQ-v2·CURL, Boot DQN 등 전문 알고리즘 대비 새로운 SOTA. 모든 도메인에서 PPO를 큰 격차로 능가.

### 0.5. 상세 동작 방식 (How It Works)

Dreamer는 **(1) 월드 모델 학습**과 **(2) Actor-Critic 학습** 두 루프가 리플레이된 경험으로부터 동시에(concurrently) 돌아가는 구조다.

**Step 1. 인코딩(Encoding)** — 입력: 환경 관측 $x_t$(이미지는 CNN, 벡터는 symlog 후 MLP). 처리: 인코더가 관측을 **이산 표현** $z_t$(소프트맥스 분포에서 샘플, straight-through 그래디언트)로 압축. 출력: 작은 코드 벡터 $z_t$.

**Step 2. 시퀀스 모델(RSSM)** — 입력: 직전 상태 $h_{t-1}$, 직전 표현 $z_{t-1}$, 직전 행동 $a_{t-1}$. 처리: GRU 기반 순환 모델 $h_t=f_\phi(h_{t-1},z_{t-1},a_{t-1})$ 가 다음 표현을 예측($\hat z_t$). 출력: 순환 상태 $h_t$. 이때 model state $s_t=\{h_t,z_t\}$ 가 정의된다.

**Step 3. 복원/예측(Heads)** — model state로부터 디코더는 관측 $\hat x_t$, 보상 예측기는 $\hat r_t$, 종료 예측기는 $\hat c_t$ 를 예측. 복원 손실로 표현이 정보를 담도록 만든다.

**Step 4. 상상(Imagination)** — 입력: 리플레이된 관측의 표현. 처리: 월드 모델 + actor가 실제 환경 없이 가상 궤적 $s_{1:T},a_{1:T},r_{1:T}$ 를 생성(예측 지평 $T=16$). 출력: 미래 궤적.

**Step 5. Critic & Actor 학습** — Critic은 부트스트랩 $\lambda$-수익 $R^\lambda_t$ 를 twohot 분포로 예측하고, Actor는 정규화된 수익으로 Reinforce 그래디언트 + 엔트로피 정규화로 행동을 개선. 학습된 Actor를 환경에 적용해 새 경험을 모으고 다시 Step 1로.

```
[관측 x_t] → [Encoder] → [이산 표현 z_t] ─┐
                                          ├─→ [model state s_t=(h_t,z_t)] → [Decoder/Reward/Continue 복원]
[a_{t-1}] → [RSSM seq model] → [h_t] ─────┘
                  (1) 월드 모델 학습 루프
─────────────────────────────────────────────────────────
[replay s 시작] → [월드 모델로 미래 상상] → [Critic: λ-return] → [Actor: 행동 선택] → [환경 상호작용] ↺
                  (2) Actor-Critic 학습 루프
```

---

## 1. Introduction

강화학습은 Go·Dota에서 인간을 능가하고, 대규모 언어모델을 사전학습 데이터 너머로 개선(RLHF)하는 핵심 요소가 될 만큼 발전했다. 그러나 현장에서는 PPO 같은 표준 알고리즘보다 **각 도메인 특성(연속 제어, 이산 행동, 희소 보상, 이미지 입력, 공간 환경, 보드게임 등)에 맞춰 설계·튜닝된 전문 알고리즘**이 더 높은 성능을 위해 흔히 사용된다. 문제는 이런 알고리즘을 충분히 새로운 과제로 옮길 때(비디오 게임 → 로봇) 하이퍼파라미터 조정에 많은 노력·전문성·연산이 든다는 점이다. 이 취약성은 RL을 새 문제에 적용하는 데 병목이 되며, 튜닝 비용이 큰 모델·과제에는 적용 자체를 막는다.

저자들은 **재구성 없이도 새 도메인을 정복하는 범용 알고리즘**을 인공지능의 중심 과제로 보고, 고정 하이퍼파라미터로 전문 알고리즘을 능가하는 Dreamer를 제안한다. 핵심은 **월드 모델을 학습**해 에이전트에게 풍부한 지각과 미래 상상 능력을 부여하는 것이다. 월드 모델은 잠재 행동의 결과를 예측하고, critic 신경망이 각 결과의 가치를 판단하며, actor 신경망이 최선의 결과로 이끄는 행동을 고른다. 직관적으로는 단순하지만 월드 모델을 견고하게 학습·활용하는 일은 미해결 문제였고, Dreamer는 정규화·균형·변환에 기반한 일련의 견고성 기법으로 이를 극복한다.

<p align='center'>
<img src="figs/DreamerV3/fig_02.png" alt="Figure 02" width="800"/>
</p>

RL의 한계를 더 밀어붙이기 위해, 저자들은 최근 연구의 초점이 된 비디오 게임 **Minecraft**를 다룬다. 무작위로 생성되는 무한한 3D 오픈월드에서, 희소 보상·어려운 탐험·긴 시간 지평·절차적 다양성 때문에 다이아몬드 채굴은 인공지능의 큰 도전으로 여겨져 왔다. 기존 접근은 사람 전문가 데이터와 도메인 특화 커리큘럼에 의존했지만, Dreamer는 기본 설정 그대로 **사람 데이터 없이 처음부터 다이아몬드를 채굴한 최초의 알고리즘**이다.

## 2. Method

Dreamer(DreamerV3)는 세 신경망으로 구성된다: 잠재 행동의 결과를 예측하는 **월드 모델**, 각 결과의 가치를 판단하는 **critic**, 가장 가치 있는 결과로 이끄는 행동을 고르는 **actor**. 세 요소는 리플레이된 경험으로부터 동시에 학습된다.

<p align='center'>
<img src="figs/DreamerV3/fig_03.png" alt="Figure 03" width="800"/>
</p>

### 2.1. 월드 모델 학습 (World Model Learning)

월드 모델은 **Recurrent State-Space Model(RSSM)** 로 구현된다. 인코더가 관측 $x_t$ 를 확률적 표현 $z_t$ 로 매핑하고, 순환 상태 $h_t$ 를 가진 시퀀스 모델이 과거 행동을 조건으로 표현 시퀀스를 예측한다. $h_t$ 와 $z_t$ 의 결합이 model state를 이루며, 여기서 보상 $r_t$, 에피소드 지속 플래그 $c_t\in\{0,1\}$ 를 예측하고 입력을 복원한다.

$$h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1}),\quad z_t \sim q_\phi(z_t \mid h_t, x_t),\quad \hat z_t \sim p_\phi(\hat z_t \mid h_t)$$

여기서 $f_\phi$ 는 시퀀스 모델(블록 대각 GRU), $q_\phi$ 는 인코더(posterior), $p_\phi$ 는 dynamics 예측기(prior)다. 표현은 소프트맥스 분포 벡터에서 샘플링되고 straight-through 그래디언트를 사용한다.

월드 모델 파라미터 $\phi$ 는 예측 손실 $L_{pred}$, dynamics 손실 $L_{dyn}$, 표현 손실 $L_{rep}$ 의 가중합으로 학습된다($\beta_{pred}=1$, $\beta_{dyn}=1$, $\beta_{rep}=0.1$).

$$L(\phi) = \mathbb{E}_{q_\phi}\Big[\textstyle\sum_{t=1}^{T}\big(\beta_{pred}L_{pred} + \beta_{dyn}L_{dyn} + \beta_{rep}L_{rep}\big)\Big]$$

dynamics 손실과 표현 손실은 stop-gradient $\mathrm{sg}(\cdot)$ 의 위치와 스케일만 다른 KL 발산이며, **free bits** 로 두 손실을 1나트($\approx 1.44$비트) 아래로 클리핑해 이미 충분히 작아진 경우 학습을 끈다.

$$L_{dyn}(\phi) = \max\big(1,\ \mathrm{KL}[\,\mathrm{sg}(q_\phi(z_t\mid h_t,x_t))\ \|\ p_\phi(z_t\mid h_t)\,]\big)$$

$$L_{rep}(\phi) = \max\big(1,\ \mathrm{KL}[\,q_\phi(z_t\mid h_t,x_t)\ \|\ \mathrm{sg}(p_\phi(z_t\mid h_t))\,]\big)$$

여기서 KL의 첫 인자가 사후분포, 둘째 인자가 사전분포다. 또한 인코더·dynamics 예측기의 범주형 분포를 99% 신경망 출력 + 1% 균등분포(**unimix**)로 두어 KL 폭주를 막는다.

<p align='center'>
<img src="figs/DreamerV3/fig_04.png" alt="Figure 04" width="800"/>
</p>

### 2.2. Critic 학습

Actor·critic은 월드 모델이 예측한 추상 궤적만으로 행동을 학습하며, model state $s_t=\{h_t,z_t\}$ 에서 동작한다. 할인율 $\gamma=0.997$ 로 수익 $R_t$ 를 정의하고, 예측 지평 $T=16$ 너머의 보상을 고려하기 위해 critic은 수익 분포를 근사한다. 부트스트랩 $\lambda$-수익은 다음과 같다.

$$R^\lambda_t = r_t + \gamma c_t\big((1-\lambda)v_t + \lambda R^\lambda_{t+1}\big),\quad R^\lambda_T = v_T$$

critic은 $-\sum_t \ln p_\psi(R^\lambda_t\mid s_t)$ 의 최대우도 손실로 학습되며, 수익이 여러 모드를 갖고 자릿수가 크게 변하므로 정규분포 대신 **지수 간격 칸(bin)에 대한 범주형 분포**로 매개화한다. 상상 궤적(가중치 $\beta_{val}=1$)과 리플레이 궤적(가중치 $\beta_{repval}=0.3$) 모두에 손실을 적용하고, 자기 파라미터의 지수이동평균(EMA)으로 critic을 정규화하며, 보상·critic 출력 가중치를 0으로 초기화해 초기 학습을 가속한다.

### 2.3. Actor 학습

actor는 수익을 최대화하되 엔트로피 정규화로 탐험한다. 고정 엔트로피 스케일 $\eta=3\times10^{-4}$ 를 모든 도메인에 쓰기 위해, 수익을 대략 $[0,1]$ 로 정규화한다. 단, 함수근사 잡음 증폭을 막기 위해 $L=1$ 이하의 작은 수익은 건드리지 않고 큰 수익만 줄인다. 이산·연속 행동 모두에 Reinforce 추정기를 사용한다.

$$L(\theta) = -\sum_{t=1}^{T}\mathrm{sg}\!\left(\frac{R^\lambda_t - v_\psi(s_t)}{\max(1, S)}\right)\log \pi_\theta(a_t\mid s_t)\ +\ \eta\, H[\pi_\theta(a_t\mid s_t)]$$

여기서 $S$ 는 수익 범위, $H$ 는 정책 엔트로피다. 이상치에 강건하도록 $S$ 는 5~95 백분위 범위를 EMA(0.99)로 평활화한다.

$$S = \mathrm{EMA}\big(\mathrm{Per}(R^\lambda_t, 95) - \mathrm{Per}(R^\lambda_t, 5),\ 0.99\big)$$

### 2.4. 견고한 예측 (Robust Predictions)

도메인마다 보상·수익의 크기가 다르므로, 제곱 손실은 발산하고 절대·Huber 손실은 학습이 정체된다. 이를 해결하기 위해 **symlog 변환**과 그 역함수 symexp를 쓴다.

$$\mathrm{symlog}(x) = \mathrm{sign}(x)\ln(|x|+1),\quad \mathrm{symexp}(x) = \mathrm{sign}(x)\big(\exp(|x|)-1\big)$$

symlog는 큰 양수·음수 크기를 모두 압축하되 원점 근처에서는 항등함수에 가까워, 부호를 보존하면서 스케일 문제를 흡수한다. 확률적 목표(보상·수익)에는 **symexp twohot 손실**을 쓴다. 신경망이 지수 간격 칸 $B=\mathrm{symexp}([-20,\dots,+20])$ 에 대한 로짓을 출력하고, 예측값은 칸 위치의 확률 가중 평균으로 읽는다. 목표는 가까운 두 칸에 선형 가중치를 나눠 거는 twohot 인코딩이며, 교차 엔트로피로 학습해 **그래디언트 크기를 목표값 크기로부터 분리**한다.

$$L(\theta) = -\,\mathrm{twohot}(y)^\top \log \mathrm{softmax}(f(x,\theta))$$

여기서 $y$ 는 회귀할 스칼라 목표, $f(x,\theta)$ 는 칸별 로짓이다.

## 3. Experiments

**설정(Setup)**: 8개 도메인(연속·이산 행동, 시각·저차원 입력, 밀집·희소 보상, 2D·3D, 절차 생성 포함) 150개 이상 과제를 **고정 하이퍼파라미터**로 평가한다. baseline은 각 벤치마크에 맞춰 튜닝된 전문 알고리즘과, 도메인 전반 성능을 최대화하도록 튜닝한 고품질 PPO(Acme 구현)다. 모든 Dreamer 에이전트는 **단일 A100 GPU** 로 학습된다. 기본 모델 크기는 200M(제어 도메인은 12M)이다.

<p align='center'>
<img src="figs/DreamerV3/fig_05.png" alt="Figure 05" width="800"/>
</p>

**주요 결과(Results)**:

* **Atari (57개, 200M 프레임)**: MuZero를 일부 연산만으로 능가, Rainbow·IQN도 능가. gamer-median 830%(MuZero 693%, PPO 180%), gamer-mean 3381%.
* **ProcGen (16개, 50M)**: 정규화 평균 66.01로 튜닝된 PPG(64.89)와 동급, Rainbow 능가. 고정 설정 PPO가 공개된 고도 튜닝 PPO(41.16)와 동급(42.80)임을 검증.
* **DMLab (30개, 100M)**: 1B 스텝의 IMPALA·R2D2+ 를 100M 스텝만으로 능가 → 1000%+ 데이터 효율.
* **Atari100k (26개, 400K)**: 트랜스포머 기반 IRIS·TWM, model-free SPR, SimPLe 등을 능가.
* **Proprio Control (18개) / Visual Control (20개)**: D4PG·DMPO·MPO, DrQ-v2·CURL 대비 새로운 SOTA.
* **BSuite (23개)**: Boot DQN 등 능가, 특히 보상 스케일 견고성(scale) 범주에서 큰 향상.

**Minecraft**: 무작위 생성 무한 3D 월드에서 12개 마일스톤(통나무→…→다이아몬드)을 희소 보상으로 달성해야 한다. 사람 데이터·커리큘럼 없이 기본 설정으로 적용한 Dreamer는 **학습한 모든 에이전트가 100M 스텝 내 다이아몬드를 발견(전체 기준 100%)**, baseline은 철 곡괭이(iron pickaxe)까지 진행하지만 다이아몬드는 0%. 100M 스텝 평균 수익 Dreamer 9.1 > IMPALA 7.1 > Rainbow 6.3 > PPO 5.1. 에피소드 단위로는 100M 스텝에서 0.4%의 에피소드에서 다이아몬드를 얻어 향후 연구 과제를 남긴다. VPT(720 GPU·9일·사람 데이터)와 달리 Dreamer는 1 GPU·9일·사람 데이터 없이 달성한다.

<p align='center'>
<img src="figs/DreamerV3/fig_06.png" alt="Figure 06" width="800"/>
</p>

**Ablation·스케일링**: 14개 과제에서 견고성 기법을 제거하면 모두 평균 성능에 기여하며, 특히 월드 모델의 KL 목표, 다음으로 수익 정규화와 symexp twohot 회귀가 중요하다(개별 기법은 일부 과제에서만 결정적). 학습 신호 분석 결과 Dreamer 성능은 **비지도 복원 손실**에 주로 의존하고, 보상·가치 그래디언트는 일부 과제만 추가 개선한다. 모델 크기(12M→400M)와 리플레이 비율을 키우면 성능이 단조 증가하고 필요한 상호작용은 감소한다.

## 4. Conclusion

저자들은 Dreamer 계열 3세대(DreamerV3)를 제시했다. 이는 **고정 하이퍼파라미터로 광범위한 도메인을 정복하는 범용 RL 알고리즘**으로, 150개 이상 과제뿐 아니라 데이터·연산 예산 변화에서도 견고하게 학습한다. 기본 설정 그대로 Minecraft에서 처음부터 다이아몬드를 채굴한 최초의 알고리즘이라는 이정표를 세웠으며, 학습된 월드 모델 기반이라는 점에서 인터넷 영상으로부터 세계 지식 학습, 도메인을 가로지르는 단일 월드 모델 같은 향후 연구의 길을 연다.

**Commentary**: DreamerV3의 가치는 새로운 한 가지 큰 아이디어보다, symlog·twohot·백분위 수익 정규화·KL 균형·unimix 같은 **"스케일 불변성을 만드는 작은 공학적 장치들의 조합"** 이 모여 "하나의 설정으로 모든 것"이라는 RL의 오랜 꿈을 실증했다는 데 있다. 성능 대부분이 비지도 복원 신호에서 나온다는 ablation은, 향후 대규모 비지도 사전학습과 RL의 결합 가능성을 시사하는 가장 흥미로운 지점이다.

---

## 부록: 사전 지식 (Prerequisites)

### A.1. 알아야 할 핵심 개념

- **강화학습 (Reinforcement Learning, RL)** — 에이전트가 환경과 상호작용하며 보상 신호로 행동 정책을 학습하는 패러다임. 상태(state), 행동(action), 보상(reward), 정책(policy), 가치함수(value function)의 개념을 이해해야 한다.
  - 본문 위치: 전반 (§2.2 Critic, §2.3 Actor)

- **Actor-Critic 알고리즘** — 정책을 직접 파라미터화하는 Actor와, 상태(또는 상태-행동)의 가치를 추정하는 Critic을 동시에 학습하는 RL 구조. REINFORCE 그래디언트 추정기를 사용해 Actor를 업데이트하고, Critic의 가치 추정으로 분산을 줄인다.
  - 본문 위치: §2.2 (Critic 학습), §2.3 (Actor 학습)

- **λ-수익 (Lambda Return / TD(λ))** — $n$-step 부트스트랩을 λ 가중 혼합으로 결합한 수익 추정기. $n$ 이 작으면 편향(bias)이 높고, 크면 분산(variance)이 높은 트레이드오프를 조정하는 데 사용된다.
  - 본문 위치: §2.2, 식 $R^\lambda_t = r_t + \gamma c_t((1-\lambda)v_t + \lambda R^\lambda_{t+1})$

- **VAE (Variational Autoencoder) / ELBO / KL 발산** — 잠재 변수 모델을 변분 추론으로 학습하는 프레임워크. 인코더(posterior $q$)와 디코더를 동시에 학습하며, ELBO(Evidence Lower Bound) = 재구성 손실 + KL 페널티로 정의된다. DreamerV3의 월드 모델이 이 구조 위에 세워진다.
  - 본문 위치: §2.1, $L_{dyn}$/$L_{rep}$ KL 손실 항

- **순환 신경망 / GRU (Gated Recurrent Unit)** — 시퀀스 데이터를 처리하는 순환 신경망 구조. GRU는 게이팅 메커니즘으로 장기 의존성을 포착하면서 LSTM보다 단순하다. DreamerV3의 RSSM이 GRU를 시퀀스 모델 $h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1})$ 로 사용한다.
  - 본문 위치: §2.1 (RSSM 시퀀스 모델)

- **RSSM (Recurrent State-Space Model)** — 결정론적(deterministic) 순환 상태 $h_t$ 와 확률적(stochastic) 잠재 표현 $z_t$ 를 결합한 잠재 동역학 모델. PlaNet에서 제안되어 Dreamer 계열 전반에 사용된다. Prior $p(z_t \mid h_t)$ 와 Posterior $q(z_t \mid h_t, x_t)$ 를 분리해 예측과 인식을 구분한다.
  - 본문 위치: §2.1, 식 $h_t$/$z_t$/$\hat{z}_t$

- **이산 잠재 표현 / 범주형 재매개변수화 (Categorical Reparameterization / Straight-Through Gradient)** — 범주형(categorical) 분포에서 샘플링한 이산 벡터를 역전파 가능하게 만드는 기법. 순전파에서는 argmax(또는 샘플)를 사용하고, 역전파에서는 소프트맥스 로짓의 그래디언트를 그대로 통과시킨다(straight-through estimator).
  - 본문 위치: §2.1 (표현 $z_t$ 샘플링)

- **모델 기반 RL (Model-Based RL)** — 환경의 전이 동역학 모델을 학습해 실제 환경 없이도 가상(상상) 궤적으로 정책을 개선하는 RL 접근. 샘플 효율이 높지만 모델 오차가 축적될 수 있다. DreamerV3는 월드 모델 내부 상상으로만 Actor-Critic을 학습한다.
  - 본문 위치: §0.1–§0.2, §2 전반

- **twohot 인코딩 / 분포형 수익 예측 (Distributional RL)** — 스칼라 보상이나 수익을 단일 실수가 아닌 이산 칸(bin)에 대한 확률 분포로 예측하는 기법. C51, QR-DQN 등에서 유래했으며, DreamerV3는 지수 간격(symexp) 칸에 twohot 인코딩으로 목표를 표현해 그래디언트 크기가 목표값 크기에 의존하지 않게 한다.
  - 본문 위치: §2.4, §2.2 (Critic의 범주형 분포 예측)

- **CNN (Convolutional Neural Network) / MLP (Multi-Layer Perceptron)** — 이미지 입력에는 CNN 인코더, 저차원 벡터 입력에는 MLP 인코더를 사용하는 기본 신경망 구조. DreamerV3는 두 가지 입력 모드를 모두 지원한다.
  - 본문 위치: §0.5 Step 1 (인코딩), §3 (설정)

---

### A.2. 먼저 읽으면 좋은 논문

1. **[2020][DreamerV2] Mastering Atari with Discrete World Models** ([arxiv:2010.02193](https://arxiv.org/abs/2010.02193)) — DreamerV3의 직전 세대. 이산 범주형 잠재 표현, KL 균형(KL balancing), 이산 world model을 Atari에 적용한 최초 성공 사례.
   - **왜?** DreamerV3의 RSSM 구조, KL 손실 분리($\beta_{dyn}$/$\beta_{rep}$), 이산 표현 모두가 V2를 직접 계승·개선한다. V3의 novelty를 파악하려면 V2와의 차이를 이해해야 한다.
   - **Repo 내 정리**: 없음

2. **[2019][DreamerV1] Dream to Control: Learning Behaviors by Latent Imagination** ([arxiv:1912.01603](https://arxiv.org/abs/1912.01603)) — Dreamer 계열의 시초. 잠재 상상(latent imagination)만으로 Actor-Critic을 학습하는 아이디어를 처음 제안.
   - **왜?** "상상 학습"의 원개념을 확립한 논문이며, DreamerV3가 계승하는 핵심 패러다임(월드 모델 내부 policy 학습)을 이해하는 데 필수다.
   - **Repo 내 정리**: 없음

3. **[2018][PlaNet] Learning Latent Dynamics for Planning from Pixels** ([arxiv:1811.04551](https://arxiv.org/abs/1811.04551)) — RSSM(결정론적 $h_t$ + 확률적 $z_t$ 이중 구조)을 처음 제안한 논문. 픽셀 입력에서 잠재 공간 플래닝을 실현.
   - **왜?** DreamerV3의 월드 모델 핵심 구성요소인 RSSM이 이 논문에서 도입됐다. §2.1의 수식 구조를 이해하려면 PlaNet 읽기를 권장한다.
   - **Repo 내 정리**: 없음

4. **[2019][MuZero] Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model** ([arxiv:1911.08265](https://arxiv.org/abs/1911.08265)) — 환경 모델 없이 MCTS와 잠재 모델만으로 다양한 게임을 정복한 DeepMind의 대표 model-based RL 알고리즘.
   - **왜?** DreamerV3의 Atari 실험에서 직접 비교되는 baseline이다(gamer-median: MuZero 693% vs DreamerV3 830%). model-based RL의 대안 패러다임으로 DreamerV3와의 접근 차이를 이해하는 데 유용하다.
   - **Repo 내 정리**: 없음

5. **[2023][IRIS] Transformers are Sample-Efficient World Models** ([arxiv:2209.00588](https://arxiv.org/abs/2209.00588)) — VQ-VAE로 이미지를 토큰화하고 Transformer로 autoregressive 월드 모델을 구성한 연구. Atari100k benchmark에서 DreamerV3와 비교되는 경쟁 방법.
   - **왜?** DreamerV3의 Atari100k 결과(§3)에서 직접 대비되는 방법으로, DreamerV3가 GRU 기반 RSSM을 택한 이유(Transformer 대비 설계 선택)를 이해하는 데 참고가 된다.
   - **Repo 내 정리**: 없음

---

### A.3. 관련/후속 논문

- **[2023][Safe DreamerV3] Safe Reinforcement Learning with World Models** ([arxiv:2307.07176](https://arxiv.org/abs/2307.07176)) — DreamerV3에 안전 제약(safety constraint)을 통합한 확장. 제약 위반 없이 복잡한 환경에서 학습.

- **[2023][STORM] Efficient Stochastic Transformer based World Models for Reinforcement Learning** ([arxiv:2310.09615](https://arxiv.org/abs/2310.09615)) — DreamerV3의 GRU 시퀀스 모델을 Transformer로 대체해 Atari100k에서 경쟁력 있는 효율을 달성.

- **[2024][DIAMOND] Diffusion for World Modeling: Visual Details Matter in Atari** ([arxiv:2405.12399](https://arxiv.org/abs/2405.12399)) — 이산 잠재 표현 대신 diffusion 모델로 월드 모델을 구성해 시각적 디테일을 보존하며 Atari100k SOTA를 경신.

- **[2025][DreamerV3 Nature] Mastering diverse control tasks through world models** ([Nature 640, 647–653](https://www.nature.com/articles/s41586-025-08744-2)) — 본 arXiv 논문의 동료 심사 게재 버전. block GRU, RMSNorm, SiLU 활성화, adaptive gradient clipping 등 추가 견고성 기법이 포함된 최종판.
