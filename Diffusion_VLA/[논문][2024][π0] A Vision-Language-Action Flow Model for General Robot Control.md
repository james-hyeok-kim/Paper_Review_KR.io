# π0: A Vision-Language-Action Flow Model for General Robot Control

저자 : 

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai,
Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine,
Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong,
Anna Walling, Haohuan Wang, Ury Zhilinsky

https://physicalintelligence.company/blog/pi0

발표 : 2024년 10월 31일에 arXiv

논문 : [PDF](https://arxiv.org/pdf/2410.24164v1)


---

**범용성(versatility)을 로봇 공학에 적용하기 위한 도전 과제와 그 해결책**

## 1. Introduction

<p align = 'center'>
<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/1264d987-de7d-485f-879e-4450709c2c5a" />
</p>

### 1. 로봇 학습의 핵심 과제: 범용성 (Versatility)

* 인간 지능의 강점: 인간은 기저귀 갈기부터 컴퓨터 프로그래밍까지, 물리적 환경의 제약과 예기치 못한 변화에 대응하며 다양한 과업을 해결하는 능력을 갖추고 있습니다.
* 현재 AI의 한계: 단백질 구조 예측처럼 특정 분야에서 인간을 능가하는 시스템은 존재하지만, 다양한 물리적 환경에서 범용적으로 작동하는 능력은 아직 부족합니다.
* 상황 인지의 필요성: 기존의 대규모 언어 모델(LLM)은 웹상의 텍스트와 이미지로 학습되어 추상적인 이해는 높으나, 실제 물리적 세계에 체화(situated)되지 않았다는 한계가 있습니다.

### 2. 로봇 파운데이션 모델의 필요성

* 데이터 희소성 해결: 특정 작업(예: 새 인식)에만 집중하기보다, 다양한 데이터를 미리 학습(Pre-training)한 후 특정 작업에 맞게 미세 조정(Fine-tuning)하는 방식이 더 효율적입니다.
* 강건함과 일반화: 다양한 로봇, 환경, 작업 데이터를 통합 학습함으로써, 모델은 실수에서 회복하거나 새로운 상황에 대응하는 능력을 갖추게 됩니다.
* 성공적인 전례: 자연어 처리(NLP)와 컴퓨터 비전 분야의 발전은 이미 이러한 대규모 사전 학습 방식의 효용성을 증명했습니다.

### 3. $\pi_0$가 제안하는 세 가지 돌파구

* 서론에서는 대규모 로봇 파운데이션 모델 개발을 위한 세 가지 주요 병목 지점을 정의하고 $\pi_0$가 이를 어떻게 해결하는지 설명합니다.
* 규모 (Scale): 약 10,000시간에 달하는 방대한 로봇 데이터셋을 활용하여 대규모 사전 학습의 이점을 극대화합니다.
* 아키텍처 (Architecture)
    * VLM 기반: 인터넷 규모의 지식을 상속받기 위해 사전 학습된 시각-언어 모델(VLM)인 PaliGemma를 백본으로 사용합니다.
    * Flow Matching: 연속적인 로봇 행동을 정밀하게 생성하기 위해 디퓨전(Diffusion)의 변형인 플로우 매칭 기법을 적용한 Action Expert를 추가했습니다.
* 학습 레시피 (Training Recipe)
    * 사전 학습(Pre-training): 다양한 로봇 기종(Cross-embodiment)과 작업 데이터를 섞어 기초 지식을 습득합니다.
    * 사후 학습(Post-training): 고품질의 큐레이션된 데이터를 통해 정교함과 효율성을 부여하여 실제 복잡한 작업(빨래 접기 등)을 수행할 수 있도록 합니다.


---

## 2. Related Work

### 1. 시각-언어-행동 (VLA) 모델과의 차이

* 기존 연구: RT-2와 같은 기존 VLA 모델들은 사전 학습된 VLM을 로봇 제어에 맞게 미세 조정하며, 행동(Action)을 텍스트 토큰과 유사하게 이산화(Discretization)하여 자기회귀(Autoregressive) 방식으로 생성합니다.
* $\pi_0$의 혁신: 행동을 토큰으로 바꾸는 대신, 플로우 매칭(Flow Matching)(디퓨전의 변형)을 사용하여 연속적인 행동 분포를 직접 모델링합니다.
* 이점: 이를 통해 기존 VLA가 어려움을 겪었던 고주파수(최대 50Hz) 제어와 고도의 숙련도가 필요한 정교한 작업(Action Chunking)이 가능해졌습니다.
    * 이산적 $\rightarrow$ 연속적
    * One Action $\rightarrow$ Action Chunking, 한 번의 추론으로 50단계의 미래 행동( $H=50$ )을 생성
    * 혼합 전문가(MoE): Action Expert
    * KV 캐싱 (Key-Value Caching): Action Expert의 출력부만 반복 계산

#### Flow matching

* Objective Function
    * $A_t$: 실제 로봇의 행동 데이터(Ground Truth action chunk)입니다.
    * $o_t$: 현재 관측값(이미지, 언어 명령, 로봇 상태)입니다.
    * $\tau$: 플로우 매칭 타임스텝으로, $[0, 1]$ 사이의 값을 가집니다.
    * $v_{\theta}$: 우리가 학습시키고자 하는 신경망(Action Expert)이 예측한 벡터 필드입니다.
    * $u$: 목표가 되는 정답 벡터 필드입니다

$$L^{\tau}(\theta) = \mathbb{E}_{p(A_t|o_t), q(A_t^{\tau}|A_t)} \|v_{\theta}(A_t^{\tau}, o_t) - u(A_t^{\tau}|A_t)\|^2$$

* Probability Path
    * 선형 가우시안(Linear-Gaussian) 경로
    * 정답 벡터 필드 $u$, 타임스텝 $\tau$에 따른 변화율(속도)

$$A_t^{\tau} = \tau A_t + (1 - \tau)\epsilon$$

$$u(A_t^{\tau}|A_t) = \epsilon - A_t$$

* Inference
    * $\delta$: 적분 간격(Step size)
    * $\tau=0$(완전한 노이즈)에서 시작하여 $\tau=1$(최종 행동)까지 총 10단계의 추론 과정

$$A_t^{\tau+\delta} = A_t^{\tau} + \delta v_{\theta}(A_t^{\tau}, o_t)$$
 

### 2. 디퓨전 및 생성 모델의 결합

하이브리드 구조: $\pi_0$는 이미지 생성 분야에서 시도되었던 디퓨전과 자기회귀 모델의 결합 방식(예: Transfusion)에서 영감을 얻었습니다.

차별점: 단순히 이미지를 만드는 것이 아니라, 개별 시퀀스 요소에 플로우 매칭 손실 함수를 적용하여 로봇의 행동 생성에 최적화했습니다.

Action Expert: 로봇 특화 토큰(상태 및 행동)을 처리하기 위해 별도의 가중치 세트인 '액션 전문가(Action Expert)'를 도입하여 성능을 높였습니다.

### 3. 대규모 로봇 학습 데이터

데이터 규모: 기존 연구들이 수십~백 시간 정도의 데이터를 사용하거나 특정 작업(물체 이동 등)에 국한되었던 것과 달리, $\pi_0$는 약 10,000시간의 자체 데모 데이터와 오픈 소스인 OXE(Open X-Embodiment) 데이터셋을 결합했습니다.

학습 레시피: LLM의 학습 방식과 유사하게, 방대한 데이터로 사전 학습(Pre-training)하여 기본 지식을 쌓고, 큐레이션된 데이터로 사후 학습(Post-training)하여 숙련도를 높이는 전략을 로봇 분야에 본격적으로 도입했습니다.

---

## 3. Overview

### 1. 데이터 혼합 (Pre-training Mixture)

* 자체 데이터셋: 7가지 로봇 구성과 68개의 과업에서 수집된 고도의 숙련도가 필요한 조작 데이터입니다.
* 오픈 소스 데이터: 22종의 로봇 데이터를 포함하는 전체 OXE(Open X-Embodiment) 데이터셋을 활용합니다.
* 언어 라벨링: 단순한 과업 이름뿐만 아니라, 약 2초 단위의 세밀한 하위 궤적(sub-trajectories) 주석을 포함하여 모델이 언어 명령을 깊이 있게 이해하도록 돕습니다.

### 2. 2단계 학습 레시피 (Two-stage Recipe)

* 사전 학습 (Pre-training): 광범위한 데이터로 일반적인 물리적 능력과 일반화 성능을 갖춘 '베이스 모델'을 만듭니다. 이 상태에서도 기초적인 수준의 언어 명령 수행이 가능합니다.
* 사후 학습 (Post-training): 복잡하고 정교한 작업(예: 빨래 접기)을 위해 고품질의 큐레이션된 데이터를 사용하여 특정 하류 과업(downstream tasks)에 맞게 모델을 적응시킵니다.

### 3. 모델 아키텍처 핵심

* VLM 기반: 구글의 PaliGemma를 백본으로 사용하며, 이를 통해 인터넷 규모의 시각-언어 지식을 상속받습니다.
* 연속적 행동 생성: 베이스 VLM에 플로우 매칭(Flow Matching) 기술을 추가하여 연속적인 행동 분포를 생성할 수 있도록 개조했습니다.
    * Action model에 Diffusion 활용


---

## 4. The $\pi_0$ Model

### 1. 아키텍처 기본 구조 (Transformer Backbone)

<img width="998" height="426" alt="image" src="https://github.com/user-attachments/assets/275661fb-389f-4450-a275-3f96e40ac277" />


* VLM 기반: $\pi_0$는 기본적으로 언어 모델 트랜스포머 백본을 사용합니다.
* PaliGemma 활용: 약 30억 개의 파라미터를 가진 PaliGemma를 베이스 모델로 사용하는데, 이는 크기가 비교적 작아 실시간 제어에 유리하기 때문입니다.
    * SigLIP + Gemma
* 입력 통합 (Late Fusion): 이미지 엔코더가 로봇의 시각 관측 데이터를 언어 토큰과 동일한 임베딩 공간으로 투영하여 결합합니다.

### 2. 로봇 특화 요소의 추가

* Proprioceptive State ( $q_t$ ): 로봇의 현재 관절 각도와 같은 상태 정보를 입력으로 받습니다.
* Action Expert (액션 전문가): 로봇의 상태( $q_t$ )와 노이즈 섞인 행동( $A_t^{\tau}$ ) 토큰만을 처리하기 위해 별도로 학습된 3억 개 규모의 가중치 세트를 추가했습니다.
* 혼합 전문가(MoE) 방식: 이미지와 텍스트는 기존 VLM 백본이 처리하고, 로봇 특화 데이터는 액션 전문가가 처리하는 일종의 MoE 구조를 가집니다.
    * 입력에 따라 MoE결정, VLM vs Action
    * 라우팅(Routing): 모델에 토큰이 들어오면, 그 토큰이 '이미지/텍스트'인지 아니면 '로봇 상태/행동'인지에 따라 해당 전문가 가중치로 보내집니다

<div align = 'center'>
   
| 전문가 구분 | 담당 데이터 (Token Routing) | 설명 |
| :--- | :--- | :--- |
| **VLM 백본 (Expert 1)** | 이미지($I_t^n$) 및 언어 명령($l_t$) | 기존 PaliGemma가 학습한 인터넷 스케일의 시각·언어 지식을 활용합니다. |
| **액션 전문가 (Expert 2)** | 로봇 상태($q_t$) 및 행동($A_t^\tau$) | VLM 사전 학습에서 보지 못한 로봇 특화 데이터를 처리하기 위해 추가된 300M 규모의 가중치입니다. |

</div>

### 3. 입력 토큰 구성 및 처리

* 이미지 ( $I_t^i$ ): 로봇당 2~3개의 RGB 이미지.
* 언어 명령 ( $l_t$ ): 수행해야 할 작업에 대한 텍스트.
* 로봇 상태 ( $q_t$ ): 조인트 각도 벡터.
* 행동 묶음 ( $A_t^{\tau}$ ): 플로우 매칭을 위해 노이즈가 섞인 미래 행동 시퀀스( $H=50$ ).

### 4. 어텐션 마스크 (Attention Mask) 설계

* 성능 최적화와 사전 학습 지식 보존을 위해 블록 단위 인과적 어텐션(Blockwise Causal Attention)을 사용합니다:
* 1블록 (이미지/언어): 기존 VLM 지식을 유지하기 위해 이후 블록(상태/행동)을 참조하지 못하게 차단합니다.
* 2블록 (로봇 상태 $q_t$ ): 이전 정보는 참조하지만, 매번 바뀌는 행동 토큰을 참조하지 않으므로 KV 캐싱이 가능해져 추론 속도가 빨라집니다.
* 3블록 (행동 토큰 $A_t^{\tau}$ ): 모든 입력 시퀀스를 참조하여 정밀한 움직임을 생성합니다.

---

## 5. Data Collection and Training Recipe

### 1. 학습 레시피: 사전 학습 및 사후 학습 (Pre-training & Post-training)

#### 사전 학습 (Pre-training)

<p align = 'center'>
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/acfe41b6-17fc-47ed-8cce-47468113d0cf" />
</p>

* 목적: 모델을 광범위한 작업에 노출시켜 범용적인 물리적 능력과 일반화 성능을 습득하게 합니다.
* 특징: 데이터의 질이 조금 낮더라도 최대한 다양한 상황(실수 및 복구 행동 포함)을 경험하게 하여 모델의 강건함(Robustness)을 키웁니다.

#### 사후 학습 (Post-training)

* 목적: 특정 하류 과업(downstream tasks)을 숙련되고 유연하게 수행할 수 있도록 적응시킵니다. 
* 특징: 일관되고 유연한 전략을 보여주는 고품질의 큐레이션된 데이터를 사용하여 효율성과 정밀도를 높입니다. 

### 2. 데이터셋 구성 및 규모

* 총 규모: 약 10,000시간의 로봇 시연 데이터를 사용하였으며, 이는 로봇 조작 모델 학습용으로는 역대 최대 규모입니다. 
* 자체 데이터 (π dataset): 7가지 로봇 구성과 68개 작업에서 수집된 9억 3백만 타임스텝의 데이터입니다. 
* 오픈 소스 데이터: OXE(Open X-Embodiment), Bridge v2, DROID 등을 포함하며 전체 혼합물의 약 9.1%를 차지합니다. 
* 데이터 균형: 특정 작업(예: 세탁물 접기)이 과도하게 대표되는 것을 막기 위해, 각 작업-로봇 조합에 $n^{0.43}$의 가중치를 두어 샘플링 비중을 조절했습니다. 

### 3. 로봇 시스템 사양 요약

<div align = 'center'>

| 로봇 플랫폼 (Robot Platform) | 구성 (Configuration) | 자유도 (DoF) | 카메라 수 | 특징 및 주요 용도 |
| :--- | :--- | :---: | :---: | :--- |
| UR5e | 단일 팔 (Single-arm) | 7 | 2 | 손목 및 어깨 카메라 장착, 기본 물체 이동 |
| Bimanual UR5e | 양팔 (Dual-arm) | 14 | 3 | 두 개의 UR5e 설정 결합, 양손 협업 작업 |
| Franka | 단일 팔 (Single-arm) | 8 | 2 | 정밀한 조작 작업 수행 |
| Bimanual Trossen | 양팔 (Dual-arm) | 14 | 3 | ALOHA 설정 기반, 고도의 숙련도 작업 |
| Bimanual ARX / AgileX | 양팔 (Dual-arm) | 14 | 3 | 사전 학습 데이터의 51%를 차지하는 주력 플랫폼 |
| Mobile Trossen / ARX | 양팔 + 이동 기지 | 16 | 3 | Mobile ALOHA 기반, 비홀로노믹 베이스 이동 조작 |
| Mobile Fibocom | 양팔 + 이동 기지 | 17 | 3 | 홀로노믹 베이스 장착, 복잡한 실내 이동 조작 |

</div>


---


## 6. Experimental Evaluation

### 1. 베이스 모델 평가 (Evaluating the Base Model)

사후 학습 없이 사전 학습만 완료된 상태에서 언어 명령을 통해 5가지 작업을 수행하며 모델의 기초 역량을 측정했습니다.

#### 평가 과업 및 로봇
* 셔츠 접기 (Shirt Folding): 평평하게 놓인 티셔츠의 소매를 접고 절반으로 접기 (B1-ARX).
* 테이블 정리 (Bussing Easy/Hard): 쓰레기와 식기를 각각 올바른 통에 담기 (UR5e).
* 식료품 담기 (Grocery Bagging): 과자 봉지 등을 종이백에 담기 (UR5e).
* 토스트 꺼내기 (Toast out of toaster): 토스터에서 빵을 꺼내 접시에 담기 (B1-Trossen).

#### 비교 결과

<p align = 'center'>
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/6fdad7d7-01e8-4bca-b77d-cfcae09f4e00" />
</p>

* 비교 대상인 기존 모델들(OpenVLA, Octo)을 시간 제약상 70만 스텝까지 학습시키지 못했기 때문
    * 700K step training( $\pi_0$ )
    * 160K step training( $\pi_0(Parity)$ )


* $\pi_0$는 OpenVLA 및 Octo와 같은 기존 로봇 파운데이션 모델보다 모든 과업에서 훨씬 높은 성공률을 기록했습니다.
* 특히 행동 묶음(Action Chunking)을 지원하지 않는 OpenVLA는 이러한 정교한 제어 과업에서 어려움을 겪었습니다.
* 모델 규모가 작고 VLM 사전 학습이 없는 $\pi_0$-small보다도 뛰어난 성능을 보여, VLM 사전 학습의 중요성을 입증했습니다.

### 2. 언어 명령 추종 (Following Language Commands)

* VLM 초기화가 로봇의 언어 지시 이해도에 미치는 영향을 확인하기 위해 미세 조정한 모델로 실험을 진행했습니다.
* 핵심 결과: $\pi_0$는 VLM 초기화가 없는 $\pi_0$-small보다 훨씬 더 정확하게 언어 명령을 따랐습니다.
* 인간 전문가의 세부 지시(Human)나 고수준 VLM 정책(HL)이 내리는 명령을 수행할 때 자율 주행 성능이 크게 향상되었습니다.
* 이는 대규모 VLM 지식이 로봇의 실제 행동과 긴밀하게 연결되어 있음을 시사합니다.

### 3. 새로운 정교한 작업 학습 (Learning New Dexterous Tasks)

* 사전 학습에 포함되지 않은 완전히 새로운 작업을 학습할 때의 효율성을 측정했습니다.


<div align = 'center'>
   
| 작업 분류 (Tier) | 과업 명칭 | 로봇 플랫폼 | 특징 |
| :--- | :--- | :--- | :--- |
| Easy (쉬움) | 볼 쌓기 (Stack Bowls) | UR5e | 사전 학습의 식기 조작과 유사함 |
| Easy (쉬움) | 수건 접기 (Towel Folding) | B1-ARX | 셔츠 접기와 유사한 동작 |
| Moderate (보통) | 전자레인지 조작 | B1-ARX | 미지(unseen)의 가전제품 포함 |
| Hard (어려움) | 키친타월 교체 | B1-UR5e | 새로운 물체와 생소한 동작 |
| Hard (어려움) | 서랍에 물건 담기 | Franka | 해당 로봇으로 학습한 적 없는 과업 |

</div>

---


## 7. Discussion, Limitations, and Future Work

### 1. 주요 논의 및 시사점 (Discussion)

* 로봇 파운데이션 모델의 실현: $\pi_0$는 대규모 데이터 사전 학습과 사후 학습 레시피를 통해 정교함, 일반화, 다단계 행동을 결합한 로봇 제어가 가능함을 입증했습니다. 
* LLM 패러다임의 이식: 인터넷 규모의 VLM 사전 학습과 로봇 특화 데이터 학습을 결합하는 방식은 대규모 언어 모델(LLM)이 지식을 습득하고 정렬(Alignment)하는 과정과 유사한 효과를 로봇 분야에서도 거둘 수 있음을 보여줍니다.
* 지식과 숙련도의 분리: 연구진은 모델의 근본적인 '지식'은 사전 학습 단계에서 습득되며, 사후 학습은 그 지식을 특정 과업에 맞게 '활용'하는 방법을 가르치는 역할을 한다고 분석합니다.

### 2. 한계점 및 향후 과제 (Limitations & Future Work)

* 연구진은 완벽한 범용 로봇 모델로 가기 위해 해결해야 할 과제들을 다음과 같이 제시했습니다. 

<div align = 'center'>

| 구분 | 내용 설명 |
| :--- | :--- |
| **데이터 구성의 이해** | 어떤 종류의 데이터가 학습에 가장 도움이 되는지, 데이터 간의 최적의 가중치 비율은 무엇인지에 대한 포괄적인 이해가 아직 부족합니다. |
| **성능 예측의 어려움** | 특정 과업을 완벽하게 수행하기 위해 정확히 어느 정도의 데이터가 필요한지 예측하는 것이 현재로서는 어렵습니다. |
| **범용성의 확장성** | 현재의 성공이 자율 주행, 네비게이션, 보행 로봇(legged locomotion) 등 완전히 다른 도메인으로도 확장(Positive Transfer)될 수 있는지 검증이 필요합니다. |
| **신뢰성 문제** | 모든 평가 과업이 완벽하게 신뢰할 수 있는 수준으로 작동하는 것은 아니며, 성능 편차를 줄이는 연구가 지속되어야 합니다. |

</div>
