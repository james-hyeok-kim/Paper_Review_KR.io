# HybridVLA: Collaborative Diffusion and Autoregression in a Unified VLA Model

저자 : 

Jiaming Liu1∗, Hao Chen3∗, Pengju An1,2†, Zhuoyang Liu1†, Renrui Zhang3‡,

Chenyang Gu1,2, Xiaoqi Li1, Ziyu Guo3, Sixiang Chen1,2, Mengzhen Liu1,2, Chengkai Hou1,2,

Mengdi Zhao2, KC alex Zhou1, Pheng-Ann Heng3, Shanghang Zhang1,2 

1State Key Laboratory of Multimedia Information Processing, School of Computer Science,
Peking University;

2Beijing Academy of Artificial Intelligence (BAAI); 3CUHK

∗ Equal contribution, † Equal technical contribution, ‡ Project lead,  Corresponding author

Project web page: hybrid-vla.github.io

발표 : ICCV 2025 (International Conference on Computer Vision) 3월

논문 : [PDF](https://arxiv.org/pdf/2503.10631)

---

## 0. Summary 

<p align = 'center'>
<img width="742" height="360" alt="Image" src="https://github.com/user-attachments/assets/326ab57f-0299-47c5-b3e0-892cb078589f" />
</p>

### 1. 논문 요약 (Summary)

* HybridVLA는 로봇 조작(Manipulation)을 위해 Diffusion + Autoregressive를 하나의 LLM에 통합한 비전-언어-행동(VLA) 모델
    * Image(Vision Encoder) + Text(Tokenizer) + Robot state(MLP) + Diffusion(MLP, with Random data for Robot actions) $\rightarrow$ LLM prefix $\rightarrow$ Iteration 4 times $\rightarrow$ MLP $\rightarrow$ 로봇 행동의 물리적 수치
    * 예측 하는 대상이 행동이 아니라 노이즈(제거 될 노이즈)
        * 단일 팔: 7차원 벡터 $(\Delta x, \Delta y, \Delta z, Roll, Pitch, Yaw, Gripper)$.
        * [0.12, -0.85, 2.1, 0.05, -1.2, 0.4, 0.9]
    * prefix length가 너무 크면, DINOv2 or SigLIP같은 인코더를 통해 Visual Tokens으로 변환
        * HybridVLA (7B): DINOv2와 SigLIP
        * HybridVLA (2.7B): 가벼운 CLIP 모델만
* 핵심 문제: 기존 VLA 모델들은 행동 토큰 이산적 양자화로 제어의 정밀도가 떨어짐, LLM의 강력한 추론 능력을 충분히 활용하지 못하고 별도의 확산 헤드만 추가하는 한계가 있었습니다.

#### 주요 기술적 특징
* 통합 아키텍처: 별도의 헤드를 붙이는 대신, 단일 LLM 백본 내에 확산 노이즈 제거(Denoising) 과정과 차기 토큰 예측 과정을 공존시켰습니다.
    * $L_{dif}$ (Diffusion Loss): 입력된 노이즈가 실제 정답 노이즈와 얼마나 다른지 계산하는 평균 제곱 오차(MSE)입니다. 이는 로봇의 연속적인 물리 제어를 담당합니다.
    * $L_{ce}$ (AR Loss): 다음 단어(토큰)를 얼마나 정확히 맞혔는지 계산하는 교차 엔트로피(Cross-Entropy) 손실입니다. 이는 로봇의 상황 판단과 논리 추론을 담당합니다.
    * 최종 결합: $L_{hybrid} = L_{dif} + L_{ce}$.


$$L_{\text{dif}} = \mathbb{E}_{a,i,c} \left[ \left\| \epsilon - \epsilon_\pi(a_t^i, i, c) \right\|^2 \right]$$

* $\epsilon_\pi(a_t^i, i, c)$ : 현재의 노이즈 섞인 행동($a_t^i$), 디노이징 단계($i$), 그리고 입력 조건( $c$, 시각/언어 등)을 바탕으로 VLA 모델(LLM)이 예측한 노이즈


$${L}_{\text{ce}} = -\sum_{j=1}^{N} \log P(x_{j} \mid x_{<j}, c)$$

* $P(x_j \mid x_{<j}, c)$: 이전까지 생성된 토큰들( $x_{<j}$ )과 입력 조건($c$)이 주어졌을 때, 모델이 예측한 $x_j$ 토큰의 확률입니다. 


* 협력적 학습 레시피(CTR): 확산 토큰과 자기회귀 토큰이 서로 간섭하지 않고 상호 보완하도록 특수 마커 토큰(<BOD>, <EOD>)을 도입한 새로운 토큰 시퀀스 설계를 제안했습니다.
    * BOD: Beginning of Diffusion, EOD: End of Diffusion
    * 학습 혼선 방지: 마커가 없으면 Diffusion 토큰이 뒤에 올 AR 토큰(정답)을 직접 예측하려고 시도하는 등 연산 과정에서 간섭이 발생
* 협력적 행동 앙상블(CAE): 작업의 성격에 따라 두 방식의 예측값을 적응적으로 융합합니다. 자기회귀 토큰의 신뢰도가 높으면 두 값을 평균 내어 사용하고, 낮으면 확산 모델의 예측에 의존합니다.
    * 신뢰도 0.96 기준 (Threshold, $\theta$)
    * $c_{t+1}^{ar} = \frac{1}{n} \sum_{i=1}^{n} P(token_i)$
    * $(0.98 + 0.95 + 0.99 + 0.97 + 0.96 + 0.94 + 0.93) \div 7 = 6.72 \div 7 = \mathbf{0.96}$
    * 0.96 넘으면 확산 모델의 행동( $a_{t+1}^{d}$ )과 AR 모델의 행동( $a_{t+1}^{ar}$ )을 산술 평균 : $a_{t+1} = (a_{t+1}^{d} + a_{t+1}^{ar}) / 2$
    * 0.96 미만일 때 확산 모델 기반의 행동( $a_{t+1}^{d}$ )에만 100% 의존하여 로봇을 조작: $a_{t+1} = a_{t+1}^{d}$
* 성과: 시뮬레이션(RLBench) 및 실제 환경 테스트에서 기존 최고 수준(SOTA) 모델 대비 성공률을 각각 14%, 19% 향상시켰으며, 단일 팔 및 양팔 로봇 모두에서 강력한 성능을 입증했습니다.

### 2. 논문의 의의 (Significance)

1) 생성 패러다임의 진정한 통합: 과거 연구들이 '이어 붙이는' 방식, HybridVLA는 LLM의 임베딩 공간 내에서 두 생성 패러다임을 직접 결합. LLM이 가진 인터넷 규모의 지식과 추론 능력을 연속적인 행동 제어에 직접적으로 투사.
2) 상호 보완적 성능 극대화 (Mutual Reinforcement): 확산 모델은 노트북 덮개 닫기 같은 정밀한 조작에 강하고, 자기회귀 모델은 식물에 물 주기처럼 장면의 문맥 이해가 필요한 작업에 더 강점이 있다는 것입니다. HybridVLA는 이 두 장점을 유기적으로 결합하여 제어의 안정성을 확보했습니다.

3) 실용적인 효율성 및 범용성
    1) 추론 속도 최적화: KV 캐시 기술과 DDIM 샘플링 단계 축소(4단계)를 통해 7B 규모의 모델임에도 9.4 Hz의 실시간 제어 속도를 달성했습니다.
    2) 강력한 일반화 능력: 학습 시 보지 못한 물체, 배경, 조명 조건에서도 안정적으로 동작하며, 특히 양팔 로봇(Dual-arm)의 복잡한 협업 작업에서도 뛰어난 성능을 보였습니다.

4) 오픈 소스 생태계 기여: 760K개의 궤적을 포함한 대규모 로봇 데이터셋(Open X-Embodiment 등)을 활용한 학습 방법론을 상세히 공개하여, 향후 파운데이션 로봇 모델(Foundation Robot Models) 연구를 위한 새로운 이정표를 제시했습니다.

---

## 1. Introduction

<p align = 'center'>
<img width="742" height="360" alt="Image" src="https://github.com/user-attachments/assets/326ab57f-0299-47c5-b3e0-892cb078589f" />
</p>

### 2. 기존 VLA 모델들의 딜레마

* 자기회귀(AR) 방식의 한계
    * VLM의 강력한 상식 추론 능력을 그대로 물려받지만
    * 행동(Action)을 이산적으로 예측하기 때문에 정밀한 제어 불가
* 확산(Diffusion) 방식의 한계
    * 정밀하고 연속적인 제어가 가능하지만, 대개 VLM 뒤에 별도의 '확산 헤드'를 붙이는 방식
    * 이 경우 헤드는 VLM이 뽑아준 특징만 참고, VLM의 추론 능력을 충분히 활용하지 못함

### 3. HybridVLA의 핵심 제안: "우아한 통합"

* 단일 LLM 백본: 확산 기반의 연속적인 행동과 자기회귀 기반의 맥락적 추론을 하나의 대규모 언어 모델(LLM) 안으로 흡수.
* 협력적 학습 및 앙상블: 확산 모델의 노이즈 제거 과정을 다음 토큰 예측 프로세스에 매끄럽게 녹여냈으며, 작업 성격에 따라 두 예측을 적응적으로 융합하는 메커니즘을 설계했습니다.

### 4. 주요 성과 및 기여

* SOTA 달성: 시뮬레이션에서 14%, 실제 환경에서 19%의 평균 성공률 향상을 기록했습니다.
* 강력한 일반화: 보지 못한 물체, 배경, 위치, 조명 조건에서도 안정적인 조작 성능을 보여주었습니다.
* 유연한 모델 크기: 7B 모델뿐만 아니라 2.7B 모델(Phi-2 기반)에서도 효과를 입증하여 모델 크기에 구애받지 않는 범용성을 보여줍니다.


---

## 2. Related Work

### 1. 전통적 로봇 조작 및 VLM의 통합

* 강화학습(Reinforcement Learning) $\rightarrow$ 모방 학습(Imitation Learning).
* 최근 연구들은 비전-언어 모델(VLM)의 강력한 추론 능력을 로봇 조작에 이식하려는 시도.

### 2. 비전-언어-행동(VLA) 모델

* VLA 모델은 VLM의 능력을 활용하여 저수준의 $SE(3)$ 포즈(Pose)를 직접 예측.
* RT-2: 7-자유도(7-DoF) 행동을 이산적인 바구니(Bins)로 양자화하여 자기회귀적으로 예측.
* OpenVLA: Open X-Embodiment 데이터셋을 활용해 대규모 사전 학습을 수행.
* 한계: 이러한 자기회귀 방식은 행동을 이산화하는 과정에서 행동의 연속성을 파괴하여 정밀한 제어를 방해한다는 단점이 있습니다.

### 3. 로봇 공학에서의 확산 모델 (Diffusion Models)

* Octo 및 RDT-1B: 트랜스포머 구조에 확산 헤드를 추가하여 유연한 행동 예측을 가능하게 했습니다.

### 4. 확산 기반 VLA 모델과 HybridVLA의 차별점

* 기존의 확산 기반 VLA 모델( $\pi_0$, CogACT, DiVLA 등)은 VLM 뒤에 별도의 확산 헤드를 붙이는 방식을 취했습니다.
* '이중 시스템' 설계는 VLM을 단순한 특징 추출기로만 사용하여, VLM이 가진 사전 학습된 추론 능력을 충분히 활용하지 못합니다.
* HybridVLA의 독창성: 기존 연구들과 달리, 단일 LLM 내에서 확산 행동 생성과 차기 토큰 예측을 통합하여 두 패러다임이 서로를 강화하도록 설계되었습니다.

---

## 3. HybridVLA Method

### 3.1 HybridVLA Architecture (모델 구조)

* Pretrained VLM Base: 7B 모델은 Llama-2를, 2.7B 모델은 Phi-2를 LLM 백본으로 사용.
* Vision Encoders: 7B 모델은 DINOv2와 SigLIP을 조합하여 강력한 시각 특징을 추출하고, 2.7B 모델은 CLIP을 사용.
* Output Processing: LLM의 출력 토큰은 두 갈래로 처리
    * 확산 기반 행동( $a_{t+1}^d$ )은 MLP를 통해 연속적인 좌표로 변환
    * 자기회귀 기반 행동( $a_{t+1}^{ar}$ )은 디토크나이저(Detokenizer)를 통해 이산 토큰에서 복원.

### 3.2 Collaborative Training Recipe (협력적 학습 레시피)

* 토큰 시퀀스 설계
    * 로봇 상태( $f_r$ ), 확산 노이즈, 자기회귀 토큰을 하나의 시퀀스로 구성합니다.
    * 확산 토큰을 <BOD>와 <EOD> 마커로 감싸서 경계를 명확히 하고, 자기회귀 토큰보다 앞에 배치하여 정답 유출(Leakage)을 방지.
* 하이브리드 목적 함수
    * $L_{dif}$: 예측 노이즈와 실제 노이즈 간의 MSE 손실.
    * $L_{ce}$: 이산 액션 토큰에 대한 교차 엔트로피 손실.
    * 공식: $L_{hybrid} = L_{dif} + L_{ce}$.
* 단계별 학습
    * Open X-Embodiment, DROID 등 대규모 로봇 데이터셋에서 사전 학습을 거친 후
    * 특정 작업 데이터로 파인튜닝을 진행합니다.

### 3.3 Collaborative Action Ensemble (협력적 행동 앙상블)

* Diffusion Inference: DDIM 샘플링을 사용하여 4단계의 반복적인 노이즈 제거 과정을 거칩니다. 이때 KV 캐시(KV Cache)를 활용해 중복 계산을 줄이고 속도를 높입니다.
* Autoregressive Inference: 확산 토큰의 연속적 표현을 조건(Condition)으로 삼아 더 정밀한 토큰 예측을 수행합니다.
* 적응적 융합 로직: 자기회귀 토큰의 평균 신뢰도( $c_{t+1}^{ar}$ )를 기준으로 판단합니다.
   * $c_{t+1}^{ar} > 0.96$: 두 모델의 예측값을 평균 내어 실행합니다 ( $a_{t+1} = (a_{t+1}^d + a_{t+1}^{ar}) / 2$ ).
   * $c_{t+1}^{ar} < 0.96$: 확산 모델의 예측값만 사용합니다 ( $a_{t+1} = a_{t+1}^d$ ).
 

---

## 4. Experiment

<p align = 'center'>
<img width="750" height="262" alt="Image" src="https://github.com/user-attachments/assets/93366892-1222-451d-a1b5-74a6b6e3ae75" />
</p>

### 4.1 시뮬레이션 실험 (Simulation Experiment)

* 벤치마크: 10가지 테이블탑 작업을 포함하는 RLBench 시뮬레이터를 사용했습니다.
    * 예: 상자 닫기, 노트북 닫기, 변기 시트 내리기, 식물에 물 주기 등
* 주요 결과
    * HybridVLA(7B)는 평균 성공률 74%를 기록하며, 기존 SOTA 모델인 OpenVLA(자기회귀 방식, 41%)와 CogACT(확산 방식, 60%)를 크게 앞질렀습니다.
    * HybridVLA-dif(7B) 변체는 추론 시 확산 모델만 사용함에도 불구하고 66%의 높은 성공률을 보였으며, 9.4 Hz의 빠른 제어 속도를 달성했습니다.
    * 이 결과는 단일 LLM 백본에서 두 방식을 통합하는 것이 별도의 확산 헤드를 붙이는 것보다 훨씬 효과적임을 입증합니다.

### 4.2 소거법 연구 (Ablation Study)

* 협력적 학습 레시피(CTR): 확산과 자기회귀를 함께 학습시켰을 때, 각각 따로 학습시킨 것보다 성능이 향상되어 상호 강화(Mutual Reinforcement) 효과가 있음을 확인했습니다.
* 대규모 사전 학습(LSP): 로봇 데이터셋을 통한 사전 학습이 없을 경우 성공률이 74%에서 22%로 급락하여, 안정적인 제어를 위해 필수적임을 입증했습니다.
* 로봇 상태 임베딩(RSE): 현재 로봇의 상태 정보를 주입하는 것이 동작의 시간적 일관성을 높여 성능을 개선했습니다.
* 임계값(Threshold): 신뢰도 임계값을 0.96으로 설정했을 때 앙상블 효과가 가장 극대화되었습니다.

### 4.3 실제 세계 실험 (Real-World Experiment)

* 단일 팔: '충전기 뽑기(95%)', '물 붓기(80%)' 등 정밀한 회전과 위치 예측이 필요한 작업에서 뛰어난 성능을 보였습니다.
* 양팔: '공 들어서 옮기기(80%)', '반바지 접기(70%)' 등 두 팔의 정교한 협업이 필요한 복잡한 작업을 성공적으로 수행했습니다. 특히 기존 확산 기반 모델인 $\pi_0$보다 훨씬 높은 성공률을 기록했습니다.

### 4.4 일반화 실험 (Generalization Experiment)

* 실험 조건: 새로운 물체, 어지러운 배경, 새로운 높이(공간 위치), 변화하는 조명 조건.
* 결과: HybridVLA는 모든 시나리오에서 기존 모델 대비 가장 적은 성능 하락폭을 보였습니다. 이는 LLM의 사전 학습된 지능이 행동 제어와 결합되어 강력한 환경 적응력을 갖추게 되었음을 의미합니다.


---

## 5. Conclusion and Limitation

### 1. 핵심 결론 (Conclusion)

* 통합 프레임워크 구축
* 협력적 학습의 성과
* 강력한 성능: 이 모델은 시뮬레이션과 실제 세계 작업 모두에서 뛰어난 조작 견고성과 강력한 일반화 능력을 입증했습니다.

### 2. 한계점 및 해결책 (Limitation & Solution)

* 추론 속도의 제약: 기존의 자기회귀 VLA 모델들과 마찬가지로, 상대적으로 느린 자기회귀 생성 과정이 전체적인 추론 속도를 제한하는 요소가 됩니다.
* 속도 최적화 제안: 하지만 본 연구의 협력적 학습 덕분에 HybridVLA-dif 변체를 사용할 경우, 자기회귀 없이 확산 프로세스만으로도 추론이 가능하며 이를 통해 9.4 Hz의 실용적인 제어 속도를 확보할 수 있습니다.

---

