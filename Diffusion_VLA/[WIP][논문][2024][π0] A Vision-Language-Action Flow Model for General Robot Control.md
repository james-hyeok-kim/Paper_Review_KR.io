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


---

## 4. The $\pi_0$ Model


---


---


---
