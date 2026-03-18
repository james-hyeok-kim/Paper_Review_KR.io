# Diffusion-VLA: Generalizable and Interpretable Robot Foundation Model via Self-Generated Reasoning

저자 : 

Junjie Wen 1 2 * Yichen Zhu 1 * † Minjie Zhu 1 2 * Zhibin Tang 1 Jinming Li 3

Zhongyi Zhou 2 Xiaoyu Liu 3 Chaomin Shen 2 Yaxin Peng 3 Feifei Feng 1

* Equal contribution

work done during Junjie Wen and Minjie Zhu’s internship at Midea Group. 

1 Midea Group, Shanghai,China 

2 East China Normal University, Shanghai, China 

3 Shanghai University, Shanghai, China. 

Correspondence to: Yichen Zhu <zhuyc25@midea.com>.

발표 : ICML 2025, arXiv 2025년 6월 4일

논문 : [PDF](https://arxiv.org/pdf/2412.03293)


---

<p align = 'center'>
<img width="716" height="558" alt="image" src="https://github.com/user-attachments/assets/3d50dde2-7c11-4762-83b9-744bfed2835c" />
</p>

## 0. Summary

* 정밀한 로봇 행동 생성에 한계가 있어 Diffusion을 도입했지만, Diffusion은 추론 능력이 부족
* 추론 주입 모듈 도입: VLM이 스스로 생성한 추론 문구를 Diffusion 학습 과정에 직접 삽입하여 로봇 행동의 정밀도 상승

### Architecture

$$[\text{이미지 + 텍스트} \rightarrow Vision \quad Encoder(SigLIP + Transformer) \rightarrow LLM(Qwen2-VL) \rightarrow $$

$$both( MLP \quad and \quad FiLM) \rightarrow Diffusion(1D \quad U-Net \quad or \quad Transformer, \quad no \quad DiT)]$$


### 의의

* 뛰어난 시각적 일반화 및 제로샷(Zero-shot) 성능
* 로봇 행동의 해석 가능성(Interpretability) 확보
    * 내부생각을 자연어로 보여줌
* 실시간 제어가 가능한 압도적인 추론 속도
    * DiVLA-2B: 82Hz (초당 82번)
    * DiVLA-7B: 42Hz
* 다양한 환경 적응 및 확장성(Scalability)
* 대화 및 시각적 질의응답(VQA) 능력 유지  


---

## 1. Introduction

### 1. 기존 모델들의 딜레마

* RT-2나 OpenVLA처럼 대형 언어 모델(LLM)의 '다음 토큰 예측(Next-Token Prediction)' 방식, 연속적인 로봇의 움직임을 불연속적인 토큰으로 쪼개다 보니 정밀도가 떨어지는 문제
* Diffusion Policy는 정밀도가 뛰어나지만 복잡한 문제를 푸는 추론능력이 결여

### 2. 핵심 아이디어: DiVLA의 탄생

* 두 모델을 단순 이어 붙이지 말고, 추론 주입 모듈을 개발, VLM이 스스로 생각한 논리적 추론 결과를 확산 모델의 정책(Policy) 학습 과정에 직접 꽂아 넣어서 로봇의 행동을 더 똑똑하고 유연하게

### 3. DiVLA가 증명한 7가지 장점

1. 시각적 일반화 (Visual generalization): 처음 보는 물체도 스스로 추론하여 올바르게 분류할 수 있음.
2. 강력한 행동 해석 가능성 (Strong action interpretability): 로봇이 왜 그런 행동을 했는지, 실패했다면 원인이 무엇인지 내부 의사결정 과정을 시각화하여 알 수 있음.
3. 새로운 지시 및 대화 능력 (Adaptability to novel instructions and conversational capability): 새로운 명령을 수행하면서도 유창한 대화 능력을 유지함.
4. 새로운 로봇 형태에 빠른 적응 (Fast adaptation to other embodiment): 양팔 로봇 등 전혀 다른 하드웨어 구조에도 적은 조정만으로 빠르게 적용 가능.
5. 매우 빠른 추론 속도 (Fast inference speed): 단일 A6000 GPU에서 가장 작은 2B 모델은 82Hz, 7B 모델은 42Hz로 실시간 반응이 가능함.
6. 향상된 시각적 강건함 (Enhanced visual generalization): 시각적 방해물이 등장하거나 배경이 바뀌어도 성능이 흔들리지 않음.
7. 확장성 (Scalability): 모델의 파라미터 크기(2B, 7B, 72B)를 키울수록 일반화 능력과 성능이 함께 향상됨.

---

## 2. Related Works

### 1. 자기회귀 모델 (Autoregression models)

* 기존 연구: LLM 다음 토큰 예측, 로봇의 물리적 제어도 해당 토큰으로 해결하려다 보니 한계
* DiVLA: LLM은 추론에만 집중

### 2. 확산 모델 (Diffusion Models)

* 기존 연구: Diffusion, 로봇의 다중 모달 행동(Multimodal action) 학습에 적용
* DiVLA: Diffusion에 언어 모델의 추론능력을 새롭게 도입

### 3. 로봇 파운데이션 모델 (Robot foundation models)

* 기존 연구: VLM 사전 학습 후, 로봇 데이터를 통해 Fine-Tuning
* DiVLA: LLM(AR) + Diffusion을 통합하여 추론과 로봇 조작 두가지를 동시에 수행하는 모델

### 4. 자기회귀 모델과 이미지 생성의 통합 (Unified auto-regressive model and image generation)

* 기존 연구: 멀티모달 이해(자기회귀)와 이미지 생성(확산)을 하나의 모델로 통합하려는 시도들(Show-O, Transfusion, Vila-U 등)이 활발하게 이루어지고 있음
* DiVLA: 기존 통합 모델들이 주로 '텍스트나 이미지 생성' 자체에 초점을 맞추었다면, DiVLA는 이 통합 구조를 '로봇 모델의 추론 능력 향상'을 위해 사용 (더 좋은 일반화)



---

## 3. Methodology

### 1. DiVLA 모델 아키텍처 (3.1 Architecture)

* 시각 인코더 (Vision Encoder): 다중 카메라 뷰에서 들어오는 이미지를 처리하기 위해 SigLIP 모델을 사용하여 시각적 특징을 추출하고 토큰화해.
* 시각-언어 모델 백본 (VLM Backbone): Qwen2-VL 모델(2B, 8B, 72B).
* 프로젝션 레이어 (Projection Layer): 두 개의 MLP 레이어.
* 확산 모델 (Diffusion Model): 최종적인 로봇의 관절 제어 신호를 생성하는 역할.
    * 특히, 완전히 새로운 형태의 로봇(예: 단일 팔 로봇 $\rightarrow$ 양팔 로봇)에 적용할 때는 처음부터 다 다시 만드는 게 아니라 맨 아래에 있는 MLP 레이어 하나만 새로 초기화해.
* 🌟 추론 주입 모듈 (Reasoning Injection Module)
    * 기존 모델들은 반복적인(Recursive) 과정을 거쳐야 해서 복잡하고 느려.
    * DiVLA는 언어 모델이 생성한 추론 토큰의 최종 임베딩을 FiLM(Feature-wise Linear Modulation)이라는 기법을 통해 확산 모델의 정책 네트워크 내부 레이어에 직접 주입(Injection).
    * 연산의 부담을 확 줄이면서도 로봇의 행동 결정 과정에 '스스로 한 생각'을 강하게 반영할 수 있게 만든 거지.

### 2. 모델 설계 및 학습 전략 (3.2 Model Design Choices)

* 학습 목표 (Training Objectives): 모델을 학습시킬 때 두 가지 목표, 확산 모델의 손실( $L_{diff}$ )과 언어 모델의 다음 토큰 예측 손실( $L_{ntp}$ ).
    * 전체 손실 함수는 $L = L_{diff} + \alpha L_{ntp}$ 형태
        * $L_{diff} \approx \quad 10x \quad L{ntp}$.
        * 균형을 맞추기 위해 가중치 $\alpha$를 10으로 설정.
* 사전 학습 데이터 (Pretraining Data)
    * 대규모 로봇 조작 데이터셋인 OXE와 Droid를 사전 학습에 사용.
    * 이 데이터셋들에는 로봇의 '행동'과 '명령어' 정도만 있고, 중간의 '추론 과정'이 들어있지 않음.
    * 연구진은 모델이 논리적으로 생각하는 법을 배울 수 있도록 GPT-4o를 활용해 데이터에 자동으로 '추론 문구'를 덧붙여 데이터를 변환해서 학습.

$$L_{diff} = \mathbb{E}_{x_0, \epsilon, t} \left[ ||\epsilon - \epsilon_\theta(x_t, t, c)||^2 \right]$$

* $\epsilon$: 실제 섞여 들어간 정규 분포 노이즈
* $\epsilon_\theta$: 확산 모델이 예측해 낸 노이즈
* $c$: 조건(Condition). DiVLA에서는 VLM이 '추론 주입 모듈(FiLM)'을 통해 쏴주는 추론 결과와 시각적 특징이 바로 이 조건으로 작용하여 행동을 유도합니다.

$$L_{ntp} = -\sum_{i} \log P(x_i \mid x_{\lt i}, c)$$

* $x_i$: 모델이 맞춰야 할 정답 토큰 (예: "육각렌치를", "오른쪽에", "놓는다" 등)
* $x_{<i}$: 이전에 이미 생성된 토큰들
* $c$: 로봇 카메라를 통해 입력된 이미지와 사용자의 명령어

---

## 4. Experiments

<p align = 'center'>
<img width="564" height="222" alt="image" src="https://github.com/user-attachments/assets/cd6e77ae-1639-4e00-99db-1fefd36dedc0" />
</p>

### 1. 다중 작업 및 시각적 일반화 (Multi-Task Learning & Visual Generalization, 4.2)

* 로봇이 사용자의 다양한 명령을 얼마나 잘 수행하는지, 그리고 주변 환경이 변해도 헷갈리지 않는지.

* 다중 작업 (Multi-task): 물건 선택하기, 세워진 냄비 뒤집기, 큐브를 상자에 넣기 등 5가지 기본 작업. 
* 시각적 방해 극복 (Visual Generalization)
    * 로봇 주변에 엉뚱한 물건(방해꾼)을 막 늘어놓거나, 바닥 배경을 바꾸거나, 화려한 조명을 켜서 로봇을 방해.
    * 다른 모델들은 성능이 뚝 떨어졌지만, DiVLA는 흔들림 없이 가장 높은 평균 성공률을 유지(Robustness).

### 2. 공장 분류 작업 (End-to-End Sorting, 4.3)

* 실제 산업 현장처럼, 테이블 위에 어지럽게 널려 있는 물건들을 4가지 카테고리(장난감 자동차, 털장갑, 인형, 육각렌치)로 분류해서 상자에 넣는 복잡한 작업.
* 학습 때 본 물건(Seen)과 처음 보는 물건(Unseen)을 섞어 놓고, 물건들이 서로 겹쳐 있는 아주 어려운(Cluttered) 상황.
* 결과
    * DiVLA는 평균 66.2%의 성공률, 2등인 OpenVLA(45.3%)를 큰 격차.
    * 특히 방해물이 아주 많은 극한 상황에서 Diffusion Policy(DP) 모델은 성공률이 9.2%로 폭락했지만, DiVLA는 60%.

### 3. 로봇의 속마음 엿보기: 행동 분석 (Behavior Analysis, 4.4)

* DiVLA의 '추론 주입 모듈' 효과.
    * 상황을 유연하게 '해석'
* 유추 능력: 로봇에게 처음 보는 '드라이버'를 줬더니, 모델이 스스로 "이건 육각렌치랑 비슷하게 생겼네?"라고 판단해서 육각렌치 상자에 분류해 넣음.
* 자기 수정 (Self-correction): 로봇이 "파란색 장난감 차를 잡는다"라고 생각하고 손을 뻗고 있을 때, 연구진이 중간에 몰래 '육각렌치'로 물건을 바꿔치기. 로봇이 즉각적으로 자신의 생각을 "육각렌치를 잡는다"로 바꾸고 그에 맞게 행동을 수정.

### 4. 제로샷 빈 피킹 (Zero-Shot Bin Picking, 4.5)

<p align = 'center'>
<img width="283" height="186" alt="image" src="https://github.com/user-attachments/assets/19e61e81-50be-4bbb-8f1d-fb540f8ae6e1" />
</p>

* 로봇이 학습 데이터에서 단 한 번도 본 적 없는 완전히 새로운 물체 102개를 집어 옮기는 테스트. 
* 물체의 크기, 색상, 질감, 말랑말랑한 정도가 다 달라서 어려움.
* 결과: DiVLA는 63.7%의 성공률. OpenVLA(28.4%), Octo(19.6%) 등 다른 모델들이 처음 보는 물체의 특징을 파악하지 못해 헤맨 것과 비교하면 엄청난 일반화(Generalization) 능력.

### 5. 양팔 로봇에 빠른 적응 (Adapt to Bimanual Robot, 4.6)

<p align = 'center'>
<img width="275" height="110" alt="image" src="https://github.com/user-attachments/assets/eb4e01de-22c6-4d98-a1f0-e4391b3f04df" />
</p>

* 외팔 로봇(Franka) $\rightarrow$ 두 팔을 쓰는 로봇(AgileX)에 모델을 적용.
* 테이블 위에 있는 식기류는 왼쪽으로, 쓰레기는 오른쪽 휴지통으로 치우는 '테이블 정리(Table bussing)' 작업.
* 결과: 본 적 있는 물건에선 72.9%, 처음 보는 물건이 섞여 있어도 70.8%의 높은 성공률.


---

## 5. Conclusion

1. DiVLA의 탄생과 최고 수준(SOTA)의 성능 증명
2. '추론'과 '행동'의 완벽한 결합
3. 압도적인 일반화(Generalization) 능력
4. VLA 모델 설계의 새로운 패러다임 제시


---
