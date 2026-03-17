# UNIFIED DIFFUSION VLA: VISION-LANGUAGEACTION MODEL VIA JOINT DISCRETE DENOISING DIFFUSION PROCESS

저자 : 

Jiayi Chen1,∗ Wenxuan Song1,∗,‡ Pengxiang Ding2,3 Ziyang Zhou1 Han Zhao2,3, Feilong Tang4 Donglin Wang2 Haoang Li 1,

1HKUST(GZ) 2Westlake University 3Zhejiang University 4Monash University

∗Equal contribution ‡Project lead: songwenxuan0115@gmail.com  Corresponding author

발표 : ICLR 2025((International Conference on Learning Representations 2025), 5월

논문 : [PDF](https://arxiv.org/pdf/2511.01718)

---

## 0. Summary

### 0.1. UD-VLA 모델 요약
* Model Architecture
    * Unified Token Space(Text + Image + Robot action) $\rightarrow$ Hybrid Attention $\rightarrow$ JD3P(DiT이나, Discrete Tokens output)
* 핵심 메커니즘 (JD3P): 'Joint Discrete Denoising Diffusion Process'라는 독자적인 확산 공정을 제안했습니다.
    * 미래 이미지와 액션 토큰을 하나의 동기화된 궤적 내에서 동시에 정제(Denoising)
    * 시각적 예측과 물리적 행동이 서로 시너지를 내도록 설계.
* 하이브리드 어텐션 (Hybrid Attention)
    * 이미지 전역적 일관성 + 액션의 상관관계 파악, 양방향 어텐션
    * 액션이 미래 이미지를 미리 참조하여 '역운동학(Inverse Kinematics)'적으로 동작할 수 있도록 인과적(Causal) 구조를 결합했습니다.
    * Block 내에서는 양방향 어텐션: 텍스트, 이미지, 미래 이미지, 액션 블록
    * Casual Mask Attention : 미래 이미지 $\rightarrow$ 액션 사이, 미래 이미지 $\rightarrow$ 현재 이미지 사이
    * 금지사항: 액션 $\rightarrow$ 미래 이미지, 미래 정보 $\rightarrow$ 현재 입력
* 학습 방식: 사전 학습된 시각-언어 모델(VLM)을 기반으로 하며, 비디오 데이터를 통한 '미래 상태 모델링(1단계)'과 로봇 데이터를 통한 '공동 최적화(2단계)'의 파이프라인을 거쳐 학습됩니다.
* 성능: CALVIN, LIBERO, SimplerEnv 등 주요 로봇 벤치마크에서 기존 최고 수준(SOTA)을 경신했으며, 기존 자기회귀(Autoregressive) 방식보다 4배 빠른 추론 속도를 구현했습니다.

### 0.2. 모델의 주요 의의

#### 1. 시각적 예견과 행동의 진정한 통합

* UD-VLA는 이미지 생성과 액션 결정을 하나의 확산 과정으로 묶어 두 과정이 유기적으로 서로를 강화하게 만들었습니다.

#### 2. '시각적 사고의 사슬(Visual CoT)' 구현

* 추론 과정에서 액션 토큰이 반복적으로 미래 이미지 정보를 참조
* 인간이 행동하기 전 머릿속으로 미래를 그려보는 것과 유사
* 복잡한 장기 과제(Long-horizon tasks)에서 훨씬 정교한 계획

#### 3. 효율성과 정밀도의 동시 달성

* 이산 확산(Discrete Diffusion) 방식을 채택하여 고해상도 연속 이미지 생성의 부담을 줄임
* 병렬 디코딩과 KV-캐시 등의 기법을 통해 로봇 제어에 필수적인 실시간성(Low Latency)을 확보했습니다.

#### 4. 강력한 일반화 성능

* 실제 로봇 실험 결과, 학습 과정에서 보지 못한 새로운 물체나 배경(Unseen targets/backgrounds)에 대해서도 미래 이미지를 올바르게 생성
* 정확한 동작을 수행하는 높은 제로샷(Zero-shot) 능력을 증명했습니다.

---

## 1. Introduction

### 1. 배경 (통합 VLA 모델의 등장)

*  최근 로봇 연구는 단순히 동작만 예측하는 것을 넘어, 미래의 시각적 상태(Future images)를 함께 예측하는 통합 VLA(Unified VLA) 모델로 진화하고 있습니다.
* 미래 이미지를 먼저 예측하면 추상적인 동작 예측 문제가 더 다루기 쉬운 역운동학(Inverse Kinematics) 문제로 전환되어 계획 능력이 향상되기 때문입니다.
    * 순 운동학 (FK): "어깨를 30도, 팔꿈치를 45도 굽히면 내 손끝은 어디에 가 있을까?" (각도 → 위치)
    * 역 운동학 (IK): "내 손끝을 좌표 (x, y, z)로 보내려면 어깨와 팔꿈치는 각각 몇 도가 되어야 할까?" (위치 → 각도)

### 2. 기존 모델의 한계점

* 외부 전문가 의존 (Extrinsic Experts): 별도의 인코더나 확산 모델을 외부 전문가로 사용함으로써 모델 구조가 복잡해지고, 시각 생성과 동작 예측 사이의 결합이 약해집니다.
    * Vision Encoder(CLIP) + Action/Image Diffusion Models
* 별도의 디코딩 과정: 입력과 출력 공간은 통합했더라도, 실제 이미지 생성과 동작 예측은 별개의 프로세스로 작동합니다. 이로 인해 미래의 시각 정보가 동작 예측에 실시간으로 충분한 가이드를 주지 못하는 한계가 있습니다.

### 3. UD-VLA의 핵심 철학 (동기화된 시너지)

* UD-VLA의 핵심은 생성(Generation)과 행동(Acting)을 동기화된 확산 공정(Synchronous Denoising Process)을 통해 최적화하는 것입니다.
* 반복적 정제: 추론 시, Action Token이 미래 이미지 토큰을 참조(Causal Attention)하며 함께 진화합니다.
* 조밀한 가이드: 동작 토큰은 이미지 생성 과정에서 나오는 풍부한 시각적 단서를 반복적으로 학습하며, 결과적으로 초기화 상태에서 정밀한 동작으로 수렴하게 됩니다.

### 4. 주요 기여 및 성과

* JD3P 제안: 여러 모달리티를 단일 확산 궤적에 통합하는 핵심 메커니즘인 'Joint Discrete Denoising Diffusion Process'를 수립했습니다.
* 하이브리드 어텐션: 모달리티 내 상호작용은 최대화하되, 인과 관계를 보존하는 효율적인 구조를 설계했습니다.
* 성능 및 효율성: CALVIN, LIBERO 등에서 SOTA를 달성함과 동시에, 자기회귀 방식보다 4배 빠른 추론 속도를 입증했습니다.


---

## 2. Related Works

### 1. 통합 VLA (Unified VLAs)의 발전 단계

* 외부 전문가 기반 모델: GR-1, SEER, DreamVLA 등은 시각 표현 학습을 위해 재구성 손실(Reconstruction loss)을 사용하고, 확산 기반의 대조 학습을 통해 동작을 학습합니다. 이들은 외부 인코더와 디코더를 별도로 사용합니다.
* 공통 토큰 공간 모델: CoT-VLA, WorldVLA, UniVLA 등은 모든 모달리티를 공유된 토큰 공간에서 처리하여 외부 인코더/디코더의 필요성을 없앴습니다.
* 기존 모델의 한계: 대부분의 기존 통합 VLA는 이미지 생성과 동작 디코딩을 별개의 프로세스로 처리하며, 이는 높은 추론 지연 시간(Latency)의 원인이 됩니다.

### 2. 이산 확산 VLA (Discrete Diffusion VLA)

* 초기 모델: PD-VLA(Pre-trained Diffusion VLA)는 BART 스타일의 노이즈 제거 전략을 사용하여 동작 토큰을 무작위로 교체하고 정제합니다.
    * 무작위 교체 (Noise): (예: 팔을 10cm 올림) 중 일부를 무작위 값이나 틀린 값으로 바꿈
    * 정제 (Refinement/Denoising): 모델은 주변의 시각 정보(이미지)와 언어 명령(텍스트)을 보고, "아, 이 무작위 값은 틀렸어. 원래는 팔을 10cm 올리는 게 맞아"라고 판단하여 값을 올바르게 수정합
* 마스크 기반 모델: LLADA-VLA와 Discrete Diffusion VLA는 BERT 스타일의 마스크 예측 전략을 따릅니다.
    * 기존 방식 (Autoregressive): "나는", "사과를", "..." 다음에 올 말은? (순차적 예측)
    * 마스크 방식 (BERT 스타일): "나는 [MASK]를 먹었다." 여기서 [MASK]는 뭐야? (전후 맥락을 보고 예측)
* 효율성 개선: CEED-VLA는 일관성 증류(Consistency distillation)를 통해 성능 저하 없이 4배 이상의 속도 향상을 달성했습니다.
    * 일관성 증류: Step을 여러번 거치지 않고 한번에 가는 방법
    * 스승이 ODE 궤적을 따라 힘들게 걸어간 길을 데이터화하고, 제자는 그 궤적 어디서든 종착점까지 한 번에 점프하는 법을 증류(학습)받아 속도를 혁신하는 것
* UD-VLA와의 차점: 기존의 이산 확산 모델들은 오직 동작 예측에만 집중하며, 시각 토큰과 동작 토큰 사이의 상호작용을 통한 시너지는 거의 고려하지 않았습니다.

### 3. 기존 연구와의 비교 (Table 1 요약)

<p align = 'center'>
<img width="741" height="393" alt="image" src="https://github.com/user-attachments/assets/ecab2cda-ee5e-4ef5-8fe2-da87c6165858" />
</p>


<div align = 'center'>

| 특징 | 기존 통합 VLA (예: CoT-VLA) | UD-VLA (본 연구) |
| :--- | :--- | :--- |
| **입출력 공간** | 통합됨 | 통합됨 |
| **어텐션 구조** | 인과적(Causal) 또는 비대칭적 | 하이브리드(Hybrid) |
| **디코딩 방식** | 별도의 프로세스 (AR 또는 별개 Diff) | 공동 확산 공정 (JD3P) |

</div>

---

## 3. Method

<p align = 'center'>
<img width="1127" height="433" alt="image" src="https://github.com/user-attachments/assets/49a66cb8-048c-4c32-9d91-fd22a5d09afc" />
</p>

### 3.1 Unified Diffusion VLA

#### 1. 통합 토큰화 (Unified Tokenization)

<p align = 'center'>
<img width="200" height="230" alt="image" src="https://github.com/user-attachments/assets/f9b982d5-bc69-449d-b72f-b42889773095" />
</p>

* 언어: Emu3 모델의 설계를 따라 토큰화됩니다.
* 시각 (이미지): VQ 토크나이저를 사용하여 이산적인 시각 토큰으로 변환합니다.
* 행동 (액션): FAST 액션 토크나이저를 사용하여 동작 데이터를 토큰화합니다.
* 특수 토큰 사용: 각 모달리티의 경계를 명확히 하기 위해 이미지 앞뒤에는 <BOI>, <EOL>을, 액션 앞뒤에는 <BOA>, <EOA>라는 특수 토큰을 붙여 구조화합니다.
* 시퀀스 구성: [텍스트 토큰; 현재 이미지 토큰; 미래 이미지 토큰; 액션 토큰] 순서로 하나의 긴 문장처럼 결합됩니다. 여기서 앞의 두 요소는 입력(Input), 뒤의 두 요소는 모델이 만들어내야 할 출력(Output)입니다.

#### 2. 하이브리드 어텐션 메커니즘 (Hybrid Attention Mechanism)

* 입력부 처리: 텍스트 인과적(Causal) 어텐션, 이미지 양방향(Bidirectional) 어텐션을 통해 처리됩니다.
* 출력부의 블록화: 출력을 생성(미래 이미지) 블록과 행동(액션) 블록으로 나눕니다.
* 블록 내부 (양방향): 이미지 토큰끼리, 혹은 액션 토큰끼리는 서로를 자유롭게 참조하여 전체적인 일관성을 높입니다. 특히 액션 간의 양방향 소통은 시간 순서에만 의존하는 잘못된 학습(Shortcut learning)을 방지합니다.
* 블록 사이 (인과적): 행동 블록은 미래 이미지 블록을 참조할 수 있지만, 반대로 이미지가 액션 정보를 미리 알 수는 없습니다.

#### 3. JD3P(Joint Discrete Denoising Diffusion Process)

1. 통합 시퀀스 구성 (Joint Sequence)

* 가장 먼저 고정된 길이의 미래 이미지 토큰( $v_0$ )과 가변 길이의 액션 토큰( $a_0$ )을 하나의 벡터로 결합합니다.

$$v_0, a_0 = (v_{0,1}, \dots, v_{0,L_v}, a_{0,1}, \dots, a_{0,L_a}) \quad (2)$$

* 이때, 이미지( $V_v$ )와 액션( $V_a$ ) 어휘 사전(Vocabulary)에 특수 마스크 토큰 <MASK>를 추가하여 전체 기호 집합( $V_m$ )을 구성합니다.
    * $a_0$는 데이터 자체, $V_a$는 어휘사전
    * $V_m = V_v + V_a + 1$
        * $1$: 특수 토큰인 <MASK> (마스크 토큰)   

2. 순방향 노이즈 공정 (Noising Process)

* 데이터를 오염시키는 과정으로, 각 토큰을 확률 $\beta_t$에 따라 마스크 토큰(M)으로 교체합니다.

$$Q_t e_{t,r} = (1 - \beta_t) e_{t,r} + \beta_t e_M \quad (3)$$

* $e_{t,r}$: 특정 위치 $r$에 있는 토큰의 원-핫 벡터입니다.
* 이 과정을 통해 $t=0$(깨끗한 데이터)에서 $t=T$(완전한 마스크 상태)까지 점진적으로 정보를 가립니다.
* $e_M$: 특수 마스크 토큰(<MASK>)의 원-핫 벡터(One-hot vector)

3. 역방향 노이즈 제거 공정 (Denoising Process)

모델($p_\theta$)이 마스크된 상태에서 원래의 이미지와 액션을 찾아가는 과정입니다.

$$p_\theta(v_{t-1}, a_{t-1} | v_t, a_t, c) = p_\theta(v_{t-1} | v_t, c) p_\theta(a_{t-1} | v_t, a_t, c) \quad (5)$$

* 조건부 확률: 이미지 예측( $p_\theta(v_{t-1} | v_t, c)$ )을 먼저 수행하고, 그 결과를 바탕으로 액션( $p_\theta(a_{t-1} | v_t, a_t, c)$ )을 예측합니다.
* 하이브리드 어텐션 활용: 액션 예측 시 현재 마스크된 이미지( $v_t$ )의 정보를 참조함으로써 시각적 가이드를 받습니다.

4. 손실 함수 (Loss Function)

복잡한 확산 체인 대신, 한 번에 마스크를 예측하는 효율적인 목적 함수를 사용합니다.

<p align = 'center'>
<img width="606" height="68" alt="image" src="https://github.com/user-attachments/assets/a7e0ca5f-b38a-4e55-94a1-de0df537da3e" />
</p>

* $\beta$ (Down-weighting): 이미지 토큰의 수가 액션보다 훨씬 많기 때문에, 시각 정보가 학습을 지배하지 않도록 가중치를 낮춥니다.
* Selective Recovery: 마스크된 위치( $1\{=M\}$ )에 대해서만 교차 엔트로피(Cross-Entropy) 손실을 계산하여 원래 토큰을 복구하도록 학습합니다.
* 좌측항 미래 이미지, 우측항 액션 예측 복구 파트
    * 좌측항
        * $\log p_{\theta}^{(v)}(v_{0,j} | \mathbf{v}_t, \mathbf{c})$
            * 마스크된 이미지( $\mathbf{v}_t$ ) 와 입력조건 ( $\mathbf{c}$ )을 보고 원래 정답 이미지 토큰 ( $v{0,j}$ )을 맞출 확률
        * $1 \{v_{t,j} = M\}$ (지시 함수)
            * 현재 위치( $j$ )가 마스크( $M$ )인 경우에만 손실을 계산하도록 하는 필터
    * 우측항
        * $\log p_{\theta}^{(a)}(a_{0,i} | \mathbf{v}_t, \mathbf{a}_t, \mathbf{c})$
            * 모델이 마스크된 이미지( $\mathbf{v}_t$ )와 마스크된 액션( $\mathbf{a}_t$ ) 을 모두 참고하여 정답 액션 토큰( $a{0,i}$ )을 맞출 확률
        * $1 \{a_{t,i} = M\}$: 이미지와 마찬가지로, 마스크된 액션 위치에 대해서만 학습


---

## 4. Experiments

### 1. 사용된 벤치마크 (4.1 BENCHMARKS)

* CALVIN: 장기적인 언어 조건부 로봇 조작을 평가하며, 5개의 연속적인 하위 작업을 얼마나 성공적으로 수행하는지 측정합니다.
* LIBERO: 지식 전이와 평생 학습을 위한 벤치마크로, 공간 추론, 물체 일반화, 장기적 구성 기술 등을 테스트합니다.
* SimplerEnv: 실제 환경에서 수집된 데이터로 훈련된 정책이 시뮬레이션 내에서 얼마나 잘 일반화되는지 확인하는 'Real-to-Sim' 평가 도구입니다.

### 2. 시뮬레이션 결과 (4.2 MAIN RESULTS)

<p align = 'center'>
<img width="658" height="310" alt="image" src="https://github.com/user-attachments/assets/be9b2945-2d0c-4a6f-9fa0-5873cade8960" />
</p>

* CALVIN: 평균 성공 길이 4.64를 기록하며 모든 베이스라인을 압도했습니다. 이는 명시적인 시각 생성(Visual CoT)이 동작 예측에 효과적인 가이드를 제공함을 증명합니다.

<p align = 'center'>
<img width="646" height="490" alt="image" src="https://github.com/user-attachments/assets/819d1bbb-ff84-466c-bf04-20bd70f91214" />
</p>
  
* LIBERO: 평균 성공률 92.7%를 달성했습니다. 특히 장기 과제(Long suite)에서 89.6%를 기록하며 강력한 시간적 추론 능력을 보여주었습니다.
* SimplerEnv: 평균 성공률 59.4%로 타 모델들을 크게 앞섰습니다. 정밀한 조작이 필요한 블록 쌓기(Stack block) 작업에서 3D 지각 능력을 가진 모델보다도 24.9% 더 높은 성공률을 보였습니다.

### 3. 심층 분석 (4.3 IN-DEPTH ANALYSIS)

<p align = 'center'>
<img width="211" height="253" alt="image" src="https://github.com/user-attachments/assets/53605fa0-c1fa-4b42-8bc0-f26d7f11b51a" />
</p>

* 하이브리드 어텐션의 효과: 인과적(Causal) 또는 완전 양방향(Bidirectional) 어텐션보다 하이브리드 방식이 가장 높은 성능(4.64)을 기록했습니다. 이는 모달리티 간 정보 누출을 막으면서도 내부 일관성을 유지하는 설계가 유효했음을 의미합니다.

<p align = 'center'>
<img width="661" height="220" alt="image" src="https://github.com/user-attachments/assets/a24cd07c-3023-4ef1-ac0e-8a3b0fba76a3" />
</p>

* 미래 이미지 생성의 중요성: 시각 정보를 생성하지 않거나 현재 이미지만 재구성할 때보다, 미래 이미지를 예측할 때 가장 좋은 성능을 보였습니다.
* JD3P의 효율성: 공동 확산 공정(JD3P)은 독립적인 확산 방식보다 높은 성공률을 기록했을 뿐만 아니라, 디코딩 속도 면에서도 4.3배 빠른 효율성을 입증했습니다.

### 4. 실제 로봇 및 시각화 (4.4 & 4.5)

<p align = 'center'>
<img width="611" height="253" alt="image" src="https://github.com/user-attachments/assets/87cfc79f-d7b1-4f45-966c-a792fda41442" />
</p>

* 실제 환경 평가: 6자유도 로봇 팔과 손을 사용한 실험에서 80% 이상의 성공률을 기록했습니다. 특히 학습하지 않은 새로운 환경(Unseen tasks)에서도 시각적 일반화를 통해 정확한 동작을 수행했습니다.
* 시각화 분석: 생성된 미래 이미지는 실제 이미지와 매우 유사하게 과업의 진행 상황을 포착했습니다. 비록 세부적인 시각적 충실도는 낮을 수 있으나, 동작 계획을 위한 정보로서는 충분히 유효함을 확인했습니다.


---

## 5. Concolusion

### UD-VLA의 기술적 완성

* 통합 시스템 제안: 이해(Understanding), 생성(Generation), 그리고 행동(Acting)을 단일 시스템 내에서 통합한 Unified Diffusion VLA를 제안했습니다.
* JD3P 메커니즘: 핵심 동력인 JD3P(Joint Discrete Denoising Diffusion Process)를 통해 이미지와 액션 토큰을 하나의 동기화된 확산 궤적 내에서 동시에 정제(Refinement)합니다.
* 아키텍처 기초: 모든 모달리티를 아우르는 통합 멀티모달 공간(Unified Multimodal Space)과 정보의 흐름을 정교하게 제어하는 하이브리드 어텐션(Hybrid Attention) 메커니즘을 기반으로 구축되었습니다.

## 효율성 및 학습 전략

* 2단계 학습: 이미지 생성 능력을 먼저 학습시킨 후 로봇 동작을 공동 최적화하는 2단계 훈련 파이프라인을 설계했습니다.
* 추론 최적화: 성능과 효율성 사이의 균형을 맞추기 위해 테스트 타임(Test-time) 기술들을 도입하여 실시간 제어에 적합한 속도를 확보했습니다.

## 최종 평가 결과

* 성능 입증: 시뮬레이션 환경과 실제 환경 모두에서 기존 모델들을 뛰어넘는 최고 수준(State-of-the-art)의 결과를 달성했습니다.

---


