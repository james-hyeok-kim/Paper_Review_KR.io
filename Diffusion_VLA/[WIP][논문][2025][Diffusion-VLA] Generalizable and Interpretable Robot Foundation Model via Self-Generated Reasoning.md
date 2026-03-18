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

---

