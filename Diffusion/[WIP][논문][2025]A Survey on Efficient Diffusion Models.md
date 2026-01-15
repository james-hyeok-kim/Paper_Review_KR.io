# Efficient Diffusion Models: A Survey

저자 
Hui Shen, 1, †, shen.1780@osu.edu 

Jingxuan Zhang, 2,†jz97@iu.edu

Boning Xiong, 3,†bnxiong24@m.fudan.edu.cn

Rui Hu, 4, †, marshallrui@gmail.com

Shoufa Chen, 5, shoufachen66@gmail.com

Zhongwei Wan, 1, wan.512@osu.edu

Xin Wang, 1, wang.15980@osu.edu

Yu Zhang, 6, zhangyu.ansel@gmail.com

Zixuan Gong, 6, gongzx@tongji.edu.cn

Guangyin Bao, 6, baogy@tongji.edu.cn

Chaofan Tao, 5, tcftrees@gmail.com

Yongfeng Huang, 7, 1155187959@link.cuhk.edu.hk

Ye Yuan, 8, yuanye_pku@pku.edu.cn

Mi Zhang, 1, mizhang.1@osu.edu∗

1 The Ohio State University

2 Indiana University

3 Fudan University

4 Hangzhou City University

5 The University of Hong Kong

6 Tongji University

7 The Chinese University of Hong Kong

8 Peking University


출간 : preprint arXiv:2502.06805, 2025

논문 : [PDF](https://arxiv.org/pdf/2502.06805)

코드 : [Git](https://github.com/AIoT-MLSys-Lab/Efficient-Diffusion-Model-Survey)

---

## 1. Introduction

<p align = 'center'>
<img width="810" height="559" alt="image" src="https://github.com/user-attachments/assets/11cb3441-59b8-4c3c-b01d-5de704233314" />
</p>

### 1. 확산 모델의 부상과 한계

* 높은 자원 요구
* 다단계 노이즈 제거
* 효율성 연구의 필요성:

### 2. 본 서베이 논문의 목적

* 포괄적 분류 체계 제공
* 최신 연구 통합
* 지속적인 자원 공유

### 3. 효율성 최적화를 위한 세 가지 핵심 관점

1) 알고리즘 수준 (Algorithm-Level)
    * 효율적 학습 및 미세 조정(Fine-tuning
    * 효율적 샘플링 및 모델 압축 
2) 시스템 수준 (System-Level)
    * 하드웨어-소프트웨어 공동 설계(Co-design)
    * 병렬 컴퓨팅 및 캐싱 기술
3) 프레임워크 (Frameworks)
    * 확산 모델 전용 최적화 기능 

---

## 2. Background and Applications

### 2.1 Basic Formulas for Diffusion Models

#### 1. DDPM (Denoising Diffusion Probabilistic Models)

* 순방향 과정 (Forward Process

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

* 역방향 과정 (Reverse Process)

$$p_\theta(x_{t-1}|x_t)$$

* 최적화 목적 함수 (Loss Function)

$$L_{simple}(\theta) := \mathbb{E}_{t, x_0, \epsilon} [||\epsilon - \epsilon_\theta(x_t, t)||^2]$$


#### 2. Score Matching

* 로그 밀도 함수 기울기 ( $s_d(x) = \nabla_x \log p_d(x)$ )
* 계산이 까다로운 파티션 함수(Partition function) $Z_\theta$를 직접 구할 필요가 없다는 장점
* 실제 데이터의 점수 함수를 모르더라도 헤시안(Hessian) 행렬의 대각합(Trace)을 이용해 학습이 가능

$$\hat{J}(\theta) = \frac{1}{N} \sum_{i=1}^{N} [tr(\nabla_x s_m(x_i; \theta)) + \frac{1}{2} ||s_m(x_i; \theta)||^2]$$

#### 3. Solvers (SDE 및 ODE)

* 이산적인 단계를 연속적인 시간 흐름으로 확장하여 물리적인 미분 방정식으로 해석
* SDE Solver (확률 미분 방정식)
    * 확산 과정을 확률적 과정으로 취급하며, 시간에 따라 무작위적인 변화를 포함
    * 순방향 SDE: $dx = f(x,t)dt + g(t)dw$
    * 역방향 SDE: $dx = [f(x,t) - g(t)^2 \nabla_x \log q_t(x)]dt + g(t)d\bar{w}$
    * 장점: 무작위성 덕분에 누적된 오차를 수축(Contract)시키는 효과가 있으며, 사전 분포 불일치(Prior mismatch)에 대해 더 강한 복원력을 가집니다.
    * 단점: 고품질 샘플을 얻기 위해 많은 수의 기능 평가(NFE)가 필요하며, 샘플링 속도가 상대적으로 느립니다.
* ODE Solver (상미분 방정식)
    * SDE와 동일한 주변 확률 밀도(Marginal probability densities)를 공유하지만, 무작위 변동 항이 제거된 형태입니다
        * 주변 확률 밀도(Marginal Probability Density)란, 특정 시점 $t$에서 데이터(샘플)들이 가질 수 있는 전체적인 분포의 모양을 의미
    * 수식: $dx = [f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log q_t(x)]dt$로 단순화


<div align="center">

|구분|SDE Solver|ODE Solver|
|:---:|:---:|:---:|
|특성|확률적 (Stochastic)|결정론적 (Deterministic)|
|궤적|무작위 변동 포함 (Brownian motion)|고정된 경로 (Smooth trajectory)|
|샘플링 속도|상대적으로 느림 (많은 NFE 필요)|매우 빠름 (적은 NFE로 수렴)|
|강점|오차 복구 능력 및 데이터 다양성|추론 효율성 및 빠른 생성|
|주요 모델|"NCSN, Score-based SDE "|"DDIM, DPM-Solver, DEIS "|

</div>


#### 4. Flow Matching (흐름 매칭)

* 연속 정규화 흐름(Continuous Normalizing Flows, CNFs)을 기반으로 하며, 분포 사이의 변환을 설명하는 벡터장(Vector field) $v_t$를 직접 학습
* 확산 모델보다 직관적이고 효율적인 학습이 가능하며, 노이즈에서 데이터로 향하는 직선 경로(Straight trajectory)를 생성할 수 있습니다.
* CFM (Conditional Flow Matching): 복잡한 적분 계산을 피하기 위해 각 샘플에 대한 조건부 확률 경로를 구성하여 학습 효율을 높입니다:

$$\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t, q(x_1), p_t(x|x_1)} ||v_t(x) - u_t(x|x_1)||^2$$



### 2.2 Applications

#### 2.2.1 Image Generation

1) 잠재 공간 활용 (Latent Space)
    * Stable Diffusion (Rombach et al., 2022)
2) 샘플링 단계의 획기적 단축 (Few-step Generation)
    * Latent Consistency Models (LCM): 일관성 손실(Consistency loss)과 증류(Distillation) 과정을 설계하여 단 4단계 만에 고품질 이미지를 생성
    * Progressive Distillation (Salimans & Ho, 2022): 스승-제자(Student-Teacher) 프레임워크를 통해 기존 50단계가 필요하던 샘플링을 품질 저하 없이 2~8단계로 단축할 수 있음을 입증
3) 구조적 제어와 효율성 (Spatial Control)
    * ControlNet (Zhang et al., 2023d): '제로 초기화 컨볼루션(Zero-initialized convolutions)'을 사용하여 사전 학습된 모델에 공간적 조건 제어 기능을 추가하는 효율적인 아키텍처를 제안
        * 제로 초기화 컨볼루션: 가중치(Weights)와 편향(Biases)을 모두 0으로 설정하여 학습을 시작하는 특수한 컨볼루션 레이어
        * 초기 상태의 중립성: 학습 시작 직후, 이 레이어의 출력값은 항상 0입니다. 따라서 추가된 제어 경로가 기존의 사전 학습된 모델(예: Stable Diffusion)에 아무런 영향을 주지 않으며, 모델은 원래 알고 있던 지식대로 이미지를 생성
        * 점진적인 학습: 학습이 진행됨에 따라 가중치가 0에서 서서히 변하면서, 모델이 추가된 조건(스케치, 깊이 지도 등)을 어떻게 반영할지 단계적으로 배워 나갑니다
4) 통합 프레임워크의 발전
    * efficient Diffusion (EDM) (Karras et al., 2022): 학습 안정성과 추론 속도를 모두 개선한 포괄적인 프레임워크를 제시하며 최첨단(SOTA) 품질을 유지

<div align="center">

| 기법 | 주요 특징 | 효율성 효과 |
| :--- | :--- | :--- |
| Stable Diffusion | 잠재 공간 연산 | 메모리 및 계산량 대폭 감소 |
| LCM | 일관성 증류 | 4단계 초고속 샘플링 |
| ControlNet | 제로 컨볼루션 | 추가 연산 최소화하며 제어력 확보 |
| EDM | 학습/샘플링 최적화 | 학습 안정성 및 추론 속도 향상 |

</div>

---

## 3. Algorithm-Level Efficiency Optimization



---

## 4. System-Level Efficiency Optimization

---

## 5. Frameworks

---

## 6. Future Work

---

