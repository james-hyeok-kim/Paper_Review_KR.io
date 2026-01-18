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


#### 2.2.2 Video Generation

1. 효율적인 학습 프레임워크 (Efficient Training)

* 모델 동결 및 모듈 추가: ED-T2V는 기존의 사전 학습된 이미지 생성 모델(Rombach et al., 2022)을 동결(Freeze)한 상태에서 추가적인 시간적(Temporal) 모듈만을 학습시켜 학습 비용을 절감합니다.
    * Efficient Diffusion for Text-to-Video)
    * 이미 잘 학습된 이미지 생성 모델(예: Stable Diffusion)의 지식을 재사용
    * 시간적 어텐션 (Temporal Attention)
    * 파라미터 효율적 미세 조정 (PEFT): LoRA(Low-Rank Adaptation)와 같은 기법을 비디오 영역에 적용

* 어댑터 활용: Xing et al. (2024)은 SimDA (Simple Diffusion Adapter), 공간적/시간적 어댑터를 사용하여 원본 T2I 모델은 그대로 두고 새로 추가된 어댑터 모듈만 업데이트하는 방식을 제안했습니다
    * T2I(Text-to-Image) 모델

#### 2.2.3 Text Generation

1) 텍스트 생성에서의 주요 도전 과제
    * 확산 모델은 본래 이미지와 같은 연속적인 데이터를 위해 설계되었기 때문에 텍스트에 적용할 때 몇 가지 어려움이 따릅니다.
    * 이산적 데이터의 특성: 이미지와 달리 단어는 불연속적인 토큰 형태이므로, 일반적인 가우시안 노이즈 주입 방식이 효과적이지 않을 수 있습니다.
    * 최적화 불안정성: 연속 공간을 위해 설계된 목적 함수들이 텍스트 확산 과정(특히 고차원 공간)에서는 불안정해지는 경향이 있습니다.

2) 효율적인 텍스트 생성 모델 및 기법
    * Masked-Diffuse LM (Chen et al., 2023a)
        * 확산 과정의 각 단계에서 교차 엔트로피(Cross-entropy) 손실 함수를 사용하여 모델 내부의 연속적인 표현과 최종적인 이산 텍스트 출력 사이의 간극을 효율적으로 메웁니다.
    * SeqDiffuSeq (Yuan et al., 2024)
        * 인코더-디코더 트랜스포머(Encoder-decoder Transformer) 아키텍처를 도입했습니다.
        * 적응형 노이즈 스케줄(Adaptive noise schedule)과 자가 조건화(Self-conditioning) 기법을 활용하여 텍스트 생성의 효율성을 확보했습니다.
            * Adaptive Noise Schedule: 전통적인 확산 모델은 모든 타임스텝(timestep)에 미리 정해진 고정된 양의 노이즈를 추가, SeqDiffuSeq는 이를 동적으로 조절
            * Self-conditioning: 모델이 현재 단계에서 노이즈를 제거할 때, 이전 단계에서 자신이 예측했던 결과물($x_0$에 대한 추정치)을 현재 단계의 추가적인 입력(조건)으로 다시 사용
    * Lovelace et al. (2024):
        * 텍스트를 먼저 연속적인 잠재 공간(Continuous latent space)으로 인코딩한 뒤, 해당 공간 안에서 연속 확산 모델을 사용하여 샘플링하는 방법론을 제시했습니다.


#### 2.2.4 Audio Generation

1) 오디오 생성의 고유한 도전 과제
    * 오디오 데이터는 이미지나 텍스트와 다른 특성을 가지고 있어 가속화가 더 까다롭습니다.
    * 강한 시간적 연속성: 고해상도 오디오를 생성하려면 시간 영역(Time-domain)과 주파수 영역(Frequency-domain)의 정보를 모두 정확하게 재구성해야 합니다.
    * 인간의 민감도: 오디오의 미세한 왜곡이나 노이즈는 인간의 귀에 매우 쉽게 포착되어 청취 경험에 큰 영향을 미칩니다.
    * 저지연성 요구: 음성 합성(TTS)이나 실시간 대화 시스템과 같은 응용 분야에서는 매우 낮은 지연 시간(Low-latency)이 필수적입니다.
    * 다차원적 특성: 스테레오, 공간 오디오 등 오디오의 다차원적 요소를 유지하면서 생성 속도를 높여야 합니다.

2) 주요 가속화 모델 및 기법
    * WaveGrad (Chen et al., 2020) & DiffWave (Kong et al., 2020)
        * 확산 단계(Diffusion steps)의 수를 줄여 가속화
    * FastDPM (Kong & Ping, 2021)
        * 이산적 단계를 연속적인 단계로 일반화, 노이즈 수준 사이의 일대일 매핑(Bijective mapping)을 사용  


#### 2.2.5 3D Generation

1) 3D 생성의 고유한 도전 과제
    * 3D 생성은 2D 이미지 생성보다 훨씬 더 복잡하며 계산 집약적
    * 데이터의 특성: 3D 데이터는 볼륨 데이터(Volumetric data)나 포인트 클라우드(Point clouds)와 같은 고해상도 요소를 포함하고 있어 연산 요구량이 기하급수적으로 증가
        *  볼륨 데이터는 3D 공간을 격자(Grid) 형태의 '복셀(Voxel)' 단위로 나누어 표현하는 방식
        *  포인트 클라우드는 물체의 표면이나 공간에 흩어져 있는 '수많은 점(Points)'들의 집합입니다. 각 점은 3차원 좌표( $x, y, z$ )와 경우에 따라 색상(RGB) 정보를 가집니다.
    * 계산 부담: 2D 이미지와 비교했을 때, 3D 공간의 표현과 생성은 하드웨어 자원을 매우 많이 소모하는 작업  

2) 효율성 향상을 위한 주요 전략
    1) 샘플링 스케줄 최적화 (Efficient Sampling Schedules)
        * 더 큰 샘플링 단계(Step size)를 사용하거나, 2D와 3D 사이의 샘플링 전략을 수정하고, 다중 뷰 병렬 처리(Multi-view parallelism)를 도입
        * 관련 연구: Bieder et al. (2023), Li et al. (2024c), Yu et al. (2024b)
    2) 혁신적인 아키텍처 도입 (Novel Architectures)
        * 방법: 상태 공간 모델(State-space models)이나 경량화된 특징 추출기(Lightweight feature extractors)를 사용하여 3D 데이터를 처리할 때 발생하는 계산 부담을 완화
        * 3D Gaussians생성에 드는 비용을 크게 줄일 수 있다.
    3) 주요 응용 및 평가 (Table 1 참고)
        * 주요 데이터셋: ShapeNet, Objaverse, HumanML3D, BraTS2020(의료) 등.
        * 평가 지표: CD(Chamfer Distance), EMD(Earth Mover's Distance), FID, 다양성(Diversity) 등이 품질 측정에 사용됩니다


---

## 3. Algorithm-Level Efficiency Optimization

<p align = 'center'>
<img width="832" height="318" alt="image" src="https://github.com/user-attachments/assets/45f4ef2b-5526-4245-adcd-4596fcfdde1e" />
</p>

### 3.1 Efficient Training

#### 3.1.1 Latent Space

* 픽셀 공간(Pixel Space)의 비효율성
    * DDPM과 같은 초기 모델은 이미지의 픽셀 단위에서 직접 노이즈를 추가하고 제거
    * 이미지 해상도가 높아질수록(예: 512x512) 고차원 연산에 따른 계산량과 메모리 오버헤드가 기하급수적으로 증가
* 잠재 공간(Latent Space)의 해결책
    * 데이터를 더 낮고 압축된 차원인 잠재 표현(Latent Representation)으로 변환하여 확산 과정을 수행 
* 오토인코더(Autoencoder)를 통한 압축
    * VAE (Variational Autoencoder)
        * 이미지를 잠재 공간의 가우시안 분포로 매핑
        * 특정 샘플이 아닌 데이터의 전체 분포를 학습하여 과적합을 방지하지만, 규제가 강할 경우 결과물이 다소 흐릿해질 수 있다는 단점
    * VQ-VAE (Vector Quantized VAE)
        * 벡터 양자화를 통해 이산적인(Discrete) 잠재 공간을 도입하여 압축 효율을 더욱 높입니다.
    * LDM (Latent Diffusion Models)
        * 오토인코더와 확산 모델을 결합한 형태로, 현재 가장 유명한 Stable Diffusion의 기반이 되었습니다.


#### 3.1.2 Loss Formulation

1) 점수 매칭 (Score Matching) 개선
    1) Score Matching은 데이터 분포의 로그 밀도 그래디언트인 점수 함수를 추정하는 방식
        * 헤시안(Hessian) 계산의 병목: 점수 매칭은 모델의 점수 함수와 실제 데이터의 점수 함수 사이의 차이를 최소화합니다. 이 과정에서 헤시안 행렬의 트레이스(trace, 대각합)를 계산해야 하는데, 데이터의 차원이 커질수록 이 계산량이 기하급수적으로 늘어납니다.
        * 비효율적인 역전파: 트레이스 계산에는 일반적인 그래디언트 계산보다 훨씬 많은 횟수의 역전파(backward passes)가 필요하므로, 고차원 이미지와 같은 데이터에는 적용하기가 매우 어렵습니다.
    * Sliced Score Matching: 고차원 점수 함수를 무작위 방향( $v$ )으로 투영(projection)하여 문제를 1차원으로 축소함으로써 헤시안 전체 계산을 피하고 확장성을 확보
        * 확장성(Scalability): 전체 헤시안 행렬을 계산할 필요 없이 무작위 방향으로의 투영만 사용하므로 계산 복잡도가 크게 낮아집니다.
        * 효율적 학습: 고차원 데이터에서도 적은 자원으로 점수 기반 모델을 학습할 수 있는 길을 열어주었습니다
        * 모델의 점수 함수 $s_m(x;\theta)$와 데이터의 점수 함수 $s_d(x)$를 무작위 벡터 $v$ 방향으로 투영한 뒤, 해당 방향에서의 차이만 비교

$$L(\theta;p_{v})=\frac{1}{2}\mathbb{E}_{p_{v}}\mathbb{E}_{p_{d}(x)}[(v^{\top}s_{m}(x;\theta)-v^{\top}s_{d}(x))^{2}]$$

* 데이터 점수 함수( $s_d(x)$ )를 알 수 없다는 문제, 부분 적분(Integration by parts)을 적용하여 다음과 같은 계산 가능한 형태로 변환

$$J(\theta;p_{v})=\mathbb{E}_{p_{v}}\mathbb{E}_{p_{d}(x)}[v^{\top}\nabla_{x}s_{m}(x;\theta)v+\frac{1}{2}(v^{\top}s_{m}(x;\theta))^{2}]$$ 


1) 점수 매칭 (Score Matching) 개선

    2) NCSN (Noise Conditional Score Networks): 데이터 밀도가 낮은 영역에서는 점수 추정이 부정확해지는 문제를 해결하기 위해, 여러 수준의 노이즈를 순차적으로 추가하며 점수를 학습하여 전체 데이터 공간에서 견고한 추정을 가능하게 합니다.
        * 데이터 샘플이 부족한 곳에서는 그래디언트 정보가 부족하여 생성 과정(Langevin dynamics)이 제대로 작동하지 않게 됩니다.

        * NCSN은 단일 네트워크( $s_{\theta}(x, \sigma)$ )를 사용하여 여러 수준의 노이즈가 섞인 데이터 분포의 점수를 동시에 학습
        * 기하급수적 노이즈 스케일: $\sigma_{3} > \sigma_{2} > \sigma_{1}$과 같이 기하급수적으로 줄어드는 노이즈 레벨 시퀀스를 설정
        * 평활화 (Smoothing): 가장 높은 노이즈 레벨($\sigma_{3}$)은 데이터가 없는 빈 공간을 노이즈로 채워(평활화), 점수 함수가 모든 공간에서 잘 정의되도록 만듭
        * 구조 보존: 가장 낮은 노이즈 레벨($\sigma_{1}$)은 원래 데이터의 세밀한 구조를 최대한 보존하는 역할
        * 통합 학습 (Unified Training): 하나의 네트워크가 모든 노이즈 레벨에 조건화되어 학습되므로, 전체 데이터 공간에서 견고한 점수 추정이 가능
        * 어닐링된 랑주뱅 역학 (Annealed Langevin Dynamics): 생성 시에는 높은 노이즈 레벨에서 시작하여 점차 노이즈를 줄여가며 데이터를 샘플링합니다. 이는 마치 금속을 서서히 식히며 결정 구조를 만드는 '어닐링' 과정과 유사하게 작동하여 고품질의 샘플을 생성
        * 강력한 추정: 데이터 밀도가 낮은 영역에서도 노이즈 덕분에 방향성을 잃지 않고 데이터가 있는 곳으로 이동할 수 있습니다.
        * 고품질 생성: 다양한 해상도와 복잡한 데이터 분포에서도 안정적으로 고품질 샘플을 생성할 수 있는 토대를 마련했습니다.
    3) 결합 랑주뱅 역학 (CLD): 데이터 공간에 직접 노이즈를 넣는 대신 데이터와 결합된 보조 변수에 노이즈를 주입하여 모델이 학습해야 할 대상을 단순화
        * 기존의 점수 매칭(Score Matching) 목적 함수를 새로운 프레임워크 내에서 재정의한 방법
        * 전통적 방식: 기존의 점수 매칭 방법들은 데이터 공간(Data space)에 직접 노이즈를 주입합니다.
        * CLD 방식: 데이터와 결합된 보조 변수( $v_t$ )를 도입하고, 노이즈를 데이터가 아닌 이 보조 변수에 주입합니다.
        * 


2) 정류 흐름 (Rectified Flow) 및 흐름 매칭 (Flow Matching)
    * 정류 흐름은 두 분포( $\pi_0$와 $\pi_1$ ) 사이를 연결할 때, 가장 효율적인 직선 수송 경로(Straight transport paths)를 학습하는 방식
    * 직선 궤적의 원리: 기존 확산 모델은 데이터 생성 시 우회하는 경로(Roundabout path)를 따르는 경향이 있지만, 정류 흐름은 단순한 최적화 문제를 통해 분포 간의 가장 직접적인 루트를 보장
    * 학습 효율성: 직선 흐름을 생성함으로써 학습 효율을 높일 뿐만 아니라, 생성된 샘플의 품질도 개선    
    * 연속 정규화 흐름(Continuous Normalizing Flows, CNFs)을 기반으로 하는 생성 모델 패러다임으로, 기존 확산 모델보다 더 큰 규모의 훈련과 높은 효율성을 제공
    * 최적 수송(Optimal Transport, OT): 평균( $\mu_t$ )과 표준편차( $\sigma_t$ )가 시간에 따라 선형적으로 변하는 가우시안 분포를 사용하여, 노이즈에서 데이터로 직선 궤적을 따라 이동하게 합니다.
    * 벡터장(Vector Field) 학습: 분포 간의 변환 맵을 직접 캡처하는 시간 의존적 벡터장 $v_t$를 학습합니다.
    * 수학적 모델링: 벡터장 $v_t$는 다음의 상미분 방정식(ODE)을 통해 시간 의존적인 미분동형사상(Diffeomorphic map)을 정의합니다

$$\frac{d}{dt}\phi_{t}(x)=v_{t}(\phi_{t}(x)), \quad \phi_{0}(x)=x$$


#### 3.1.3 Training Tricks

1) 데이터 의존적 적응형 사전 확률 (Data-Dependent Adaptive Priors)
    1)  기존 확산 모델은 초기 분포(사전 확률)를 표준 가우시안 분포($\mathcal{N}(0, I)$)로 가정하지만, 이는 실제 데이터 분포와 일치하지 않아 학습 효율을 저해할 수 있습니다.
    2)  특정 작업이나 데이터셋의 특성에 맞춰 사전 분포를 조정함으로써 학습 수렴을 가속화하고 생성 품질을 높입니다.
2) 추가 손실 감독 (Extra Loss Supervision)
    1) 표준적인 노이즈 제거 목적 함수 외에 추가적인 손실 항을 도입하여 모델을 데이터 분포에 더 잘 맞추고 학습을 가속화
    2) REPA (Representation Alignment): 노이즈 제거 네트워크의 숨겨진 상태를 DINOv2와 같은 사전 학습된 시각 인코더의 표현과 일치시킵니다 (17.5배 가속화)
    3) VA-VAE: VAE의 잠재 공간을 사전 학습된 시각 기반 모델과 정렬시킵니다. (기존 DiT 모델 대비 21배 가속)
3) 노이즈 스케줄 (Noise Schedule)
    1) 체계적 노이즈 추가 (Fixed/Systematic)
        1) Cosine 스케줄 (IDDPM): 선형 스케줄보다 노이즈 분포를 최적화
            1) $\beta_t = 1 - \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$
        2) Laplace 스케줄: 데이터 구조 학습에 가장 효과적인 중간 강도의 노이즈 영역에 계산 자원을 집중
    2) 동적 노이즈 조정 (Dynamic/Adaptive)
        1) DDIM: 비마르코프(Non-Markovian) 순방향 과정을 도입
        2) ResShift: 고해상도 이미지와 저해상도 이미지 사이의 잔차(Residual)를 이동시키는 방식을 사용하여 초해상도(Super-resolution) 작업의 학습 효율을 높입니다.
        3) Immiscible Diffusion: 미니 배치 내의 이미지와 가장 가까운 노이즈를 할당하여 노이즈 제거의 복잡성을 줄이고 효율성을 높입니다. 

### 3.2 Efficient Fine-Tuning

1) LoRA (Low-Rank Adaptation)
    1) LoRA는 사전 학습된 모델의 가중치를 고정한 상태에서, 각 트랜스포머 레이어에 저차원 분해 행렬(Low-rank decomposition matrices)을 삽입하여 효율적으로 학습하는 방법입니다
    2) 수학적 원리: 사전 학습된 가중치 행렬 $W_0 \in \mathbb{R}^{d \times k}$에 대해 가중치 업데이트를 $\Delta W = BA$로 표현하며, 여기서 $B \in \mathbb{R}^{d \times r}$와 $A \in \mathbb{R}^{r \times k}$는 학습 가능한 저차원 행렬( $r \ll \min(d, k)$ )입니다.
    3) 효율성: 메모리 요구 사항을 최대 90%까지 절감하며, 추론 시 추가적인 지연 시간 없이 표준 절차를 사용할 수 있습니다.
        1) 학습할때 필요한 메모리가 줄어든다  
    4) 주요 변형
        1) LCM-LoRA: 샘플링 단계를 약 4단계로 줄여주는 범용 가속 모듈입니다.
        2) Concept Sliders: 이미지 생성 시 특정 속성(예: 나이, 스타일)을 정밀하게 제어할 수 있게 합니다.
        3) LoRA-Composer: 여러 개의 LoRA를 결합하여 한 이미지 내에서 여러 개념을 동시에 구현합니다.
2). 어댑터 (Adapter)
    1) 어댑터는 사전 학습된 모델 내부에 삽입되는 가벼운 모듈로, 원래의 가중치는 동결한 채 특정 작업에 필요한 특징을 학습합니다.
    2) 구조: 일반적으로 다운 프로젝션(Down-projection), 비선형성(Nonlinearity), 업 프로젝션(Up-projection) 단계로 구성되며, 트랜스포머 블록의 정규화 레이어와 피드 포워드 레이어 사이에 위치합니다.
    3) 주요 모델
        1) T2I-Adapter: 스케치, 깊이 맵(Depth map) 등 구조적 가이드를 텍스트-이미지 생성에 추가합니다.
        2) IP-Adapter: 이미지 프롬프트를 사용하여 참조 이미지와 시각적 일관성이 높은 결과를 생성하며, 분리된 교차 주의(Decoupled cross-attention) 전략을 사용합니다.
        3) CTRL-Adapter: 감정이나 객체 유형과 같은 특정 속성을 정밀하게 제어합니다
3) ControlNet
    1) ControlNet은 사전 학습된 확산 모델에 공간적 조건 제어(Spatial conditioning control)를 추가하는 강력한 아키텍처입니다.
    2) 제로 컨볼루션(Zero Convolution): 가중치가 0으로 초기화된 컨볼루션 레이어를 사용하여 사전 학습된 모델의 안정성을 해치지 않으면서 조건 정보를 서서히 통합합니다.
    3) 제어 종류: 가장자리(Canny edge), 스크리블(Scribble), 인체 포즈(OpenPose) 등 다양한 공간 정보를 활용할 수 있습니다.
    4) 주요 변형
        1) ControlNet-XS: 통신 대역폭을 개선하여 매개변수 수를 줄이면서도 추론 및 학습 속도를 약 2배 가속했습니다
        2) ControlNet++: 생성된 이미지와 제어 조건 사이의 정렬을 높이기 위해 픽셀 수준의 사이클 일관성 최적화를 도입했습니다
        3) ControlNeXt: 무거운 추가 브랜치를 간소화된 구조로 대체하여 학습 가능한 매개변수를 기존 대비 90% 감소시켰습니다



### 3.3 Efficient Sampling

#### 3.3.1 Efficient Solver

1) SDE 솔버 (SDE Solver)
    1) SDE 기반 솔버는 무작위성(Stochasticity)을 활용하여 생성 과정에서의 오류를 보정하는 데 강점이 있습니다.
    2) DiffFlow & Normalizing Flow 활용: 학습 가능한 드리프트(Drift) 항을 도입하거나 정규화 흐름(Normalizing Flow)을 결합하여, 데이터를 노이즈로 변환하거나 그 반대의 과정을 더 빠르게 수렴하도록 돕습니다.
    3) 적응형 단계 크기(Adaptive Step-size): 고정된 단계가 아닌, 오차 허용 범위에 따라 단계 크기를 동적으로 조절하여 불필요한 계산을 줄입니다.
    4) GMS (Gaussian Mixture Solver): 역전파 과정이 가우시안 분포를 따른다는 기존 가정을 버리고, 더 유연한 가우시안 혼합 모델(GMM)을 사용하여 적은 단계에서도 정확도를 높입니다.
    5) SA-Solver (Stochastic Adams Solver): 다단계 수치 해석 기법인 아담스 방법(Adams method)과 예측자-수정자(Predictor-corrector) 구조를 결합하여 속도와 품질의 균형을 맞춥니다.
    6) Restart: SDE의 오류 수축 능력과 ODE의 효율성을 결합하여, 노이즈 주입과 결정론적 단계를 교차하며 샘플링합니다

2) ODE 솔버 (ODE Solver)
    1) ODE 기반 솔버는 결정론적(Deterministic) 궤적을 가지므로 SDE보다 빠르게 수렴하는 경향이 있습니다.
    2) DDIM: 확산 모델 가속화를 위해 ODE를 명시적으로 활용한 초기 모델로, 비마르코프(Non-Markovian) 과정을 통해 샘플링 단계를 획기적으로 줄였습니다.
    3) PNDM: 수치 솔버를 그래디언트와 전송 성분으로 분해하여 2차 수렴을 달성, 품질 저하 없이 20배 가속을 구현했습니다.
    4) DPM-solver & DEIS: 확률 흐름 ODE의 반선형(Semi-linear) 구조를 활용한 맞춤형 솔버로, 단 10~20회 반복만으로 고품질 샘플을 생성합니다.
    5) 속도 매개변수화 (Velocity Parameterization): 속도 변화( $dx$ )를 직접 예측하고 최적화하여 궤적 추정의 정밀도를 높입니다.
    6) DMCMC: MCMC 기법을 통합하여 초기점을 노이즈가 적은 상태 근처에서 생성함으로써 ODE 솔버의 부담을 줄입니다.
    7) TrigFlow: 대규모 일관성 모델(Consistency Models)을 위한 통합 프레임워크로, 단 2단계의 샘플링만으로 최첨단 성능을 보여줍니다.

#### 3.3.2 Sampling Scheduling

* 샘플링 단계의 시점과 순서를 구조화하여 효율성과 품질을 동시에 개선하는 전략을 다룹니다.

1) 병렬 샘플링 (Parallel Sampling)
    1) 전통적인 확산 모델은 수천 개의 노이즈 제거 단계를 순차적으로 실행해야 하므로 속도가 매우 느립니다. 병렬 샘플링은 멀티코어 GPU의 성능을 활용해 여러 단계를 동시에 계산함으로써 이 문제를 해결
    2) DEQ (Deep Equilibrium) 모델: 샘플링 시퀀스를 다변량 고정점 시스템(fixed-point system)으로 간주합니다. 모든 상태를 동시에 해결하여 여러 GPU에서 병렬 처리가 가능하도록 합니다.
        1) 상태 벡터 (Multivariate State) 디퓨전 모델이 다루는 데이터(이미지, 오디오 등)는 수만 개 이상의 차원을 가진 다변량 벡터입니다. 샘플링의 각 단계 $x_t$는 이 고차원 공간 내의 한 점이며, 시스템의 상태를 나타냅니다.
        2) 연산자로서의 신경망 ( $\Phi$ )학습된 역확산 연산자(Reverse operator)는 현재 상태 $x_t$를 입력받아 다음 상태 $x_{t-1}$을 출력합니다.
            1) 결정론적 샘플링(DDIM 등): $x_{t-1} = \Phi(x_t, t)$ 이때 $\Phi$는 데이터 분포의 매니폴드(Manifold)로 향하는 벡터장을 정의합니다.
        3) 고정점(Fixed-point)의 의미샘플링의 최종 목적지인 $x_0$는 더 이상 노이즈 제거가 필요 없는 상태입니다. 즉, 이상적인 디노이저(Ideal Denoiser) 관점에서 $x_0$는 다음과 같은 조건을 만족하는 고정점에 가깝습니다.
        4) 고정점 시스템의 수렴 속도를 높이는 수치 최적화 기법(Newton's method 등)을 적용하여 샘플링 단계를 1000단에서 10~20단으로 줄일 수 있습니다.
        5) 수렴성(Convergence): 다변량 시스템에서 $\Phi$가 축약 사상(Contraction Mapping) 조건을 만족한다면, 어떤 노이즈에서 시작하더라도 유일한 데이터 포인트로 수렴하게 됩니다.
    3) ParaDiGMS: 피카르 반복(Picard iterations)을 사용하여 ODE 해를 병렬로 근사합니다. 슬라이딩 윈도우 프레임워크 내에서 여러 상태 전이를 동시에 업데이트합니다.
        1) 피카르 반복(Picard Iteration)은 상미분 방정식(ODE)의 해를 구하기 위해 사용되는 반복법으로, 미분 방정식을 적분 방정식 형태로 변환하여 해에 점진적으로 수렴하게 만드는 기법
        2) 다변량 고정점 시스템이나 디퓨전 모델의 병렬 샘플링 관점에서 피카르 반복은 매우 핵심적인 역할
        3) $\frac{dy}{dt} = f(t, y), \quad y(t_0) = y_0$
        4) $y(t) = y_0 + \int_{t_0}^t f(s, y(s)) \, ds$
        5) $y_{n+1}(t) = y_0 + \int_{t_0}^t f(s, y_n(s)) \, ds$
        6) 이 과정을 반복하면 $y_n(t)$는 실제 해 $y(t)$에 수렴
        7) 일반적인 ODE Solver(예: Euler, RK4)는 시간을 잘게 쪼개어 $t_k$의 값을 알아야만 $t_{k+1}$을 계산할 수 있는 순차적(Sequential) 방식
        8) 함수 전체의 업데이트: 피카르 반복은 특정 시점의 점이 아니라, 시간 구간 $[t_0, T]$ 전체에 대한 궤적(Trajectory) $y(t)$ 자체를 하나의 상태로 취급
        9) 독립적 적분: $y_{n+1}(t)$를 계산할 때, 모든 $t$에 대한 적분 값들을 이전 단계의 궤적 $y_n$ 정보를 이용해 동시에(Parallel) 계산할 수 있습니다
    4) ParaTAA: 고정점 반복을 통해 삼각 비선형 방정식을 해결합니다. 계산 속도와 안정성을 높이기 위해 '삼각 앤더슨 가속(Triangular Anderson Acceleration)' 기법을 도입했습니다.
        1) 기본 아이디어: 고정점 반복 $x_{n+1} = \Phi(x_n)$을 수행할 때, 단순히 직전의 결과값($x_n$)만 사용하는 것이 아니라, 과거 $m$개의 반복 결과들의 선형 결합(Linear Combination)을 사용하여 다음 단계 $x_{n+1}$을 예측합니다.
        2) 최적화: 과거의 잔차(Residual, $r_i = \Phi(x_i) - x_i$)들의 가중치 합이 최소가 되도록 계수($\alpha$)를 결정
    5) StreamDiffusion: 실시간 인터랙션을 위해 배치 단위 노이즈 제거와 RCFG(Residual Classifier-Free Guidance)를 결합합니다. 또한 확률적 유사도 필터링을 통해 유사한 프레임의 처리를 건너뛰어 에너지 효율을 높입니다.
        1) 기존의 CFG는 조건부 모델( $\epsilon_\theta(x_t, c)$ )과 무조건부 모델( $\epsilon_\theta(x_t, \emptyset)$ ) 사이의 차이를 증폭시켜 가이드합니다.
        2) $\epsilon_{cfg} = \epsilon_\theta(x_t, \emptyset) + w(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$
        3) RCFG는 이를 잔차(Residual) 형태로 재구성하여, 수치적 안정성을 높이고 수렴 속도를 최적화
        4) 피카르 반복에서의 고정점 연산자 $\Phi$는 다음 단계의 궤적을 계산하는 함수, 이때 RCFG를 적용하면 연산자가 정의하는 벡터장(Vector Field)이 더 매끄러워(Smoother)집니다.
        5) RCFG로 계산된 가이드 벡터를 바탕으로, 시간 $t=0$부터 $T$까지의 전체 궤적을 병렬로 업데이트


2) 타임스텝 스케줄 (Timestep Schedule)
    1) 타임스텝 스케줄은 이산적인 샘플링 단계들을 어떻게 선택하고 배열할지에 대한 전략
    2) FastDPM: 이산적인 확산 단계를 연속적인 단계로 일반화하고, 이를 노이즈 레벨과 일대일 매핑(bijective mapping)합니다. 이를 통해 원래 단계( $T$ )보다 훨씬 적은 단계( $S \ll T$ )만으로도 샘플링이 가능해집니다.
    3) Watson et al. (2021): 동적 계획법(Dynamic Programming)을 사용하여 최적의 스케줄을 결정합니다. ELBO(Evidence Lower Bound) 최적화 문제로 접근하여, 단 32단계만으로 수천 단계의 모델과 대등한 성능을 냈습니다.
    4) AYS (Align Your Steps): 샘플링 스케줄과 솔버를 공동 최적화합니다. KL 발산의 상한선(KLUB)을 최소화하는 적응형 스케줄을 도출하여, 품질 저하 없이 샘플링 단계를 약 40% 절감했습니다.

#### 3.3.3 Truncated Sampling

* 확산 과정의 특정 단계에서 중복되는 계산을 전략적으로 줄여 샘플 생성 효율을 높이는 기술을 다룹니다.

1) 조기 종료 (Early Exit)
    1) 충분히 확신을 가질 수 있는 단계에 도달했을 때, 모델의 깊은 레이어나 나중 단계의 계산을 건너뛰는 방식
    2) 노이즈 상태에 가까울수록 점수 추정(Score estimation)이 쉬워지고, 데이터 상태에 가까울수록 더 많은 계산 자원이 필요하다는 관찰에 기반
    3) ASE (Adaptive Score Estimation): 타임스텝에 따라 종료 스케줄을 동적으로 조정합니다. 노이즈 영역에 가까울수록 더 많은 블록을 건너뛰어 계산 효율을 최적화
    4) DeeDiff: 레이어별 예측 불확실성을 평가하는 '불확실성 추정 모듈(UEM)'을 도입합니다. 특정 레이어의 불확실성이 설정된 임계값보다 낮으면 해당 단계에서 즉시 종료하여 하위 레이어 계산을 생략 

2) 검색 가이드 초기화 (Retrieval-Guided Initialization)
    1) 데이터베이스에서 입력 프롬프트와 관련된 샘플을 검색하여 모델의 초기 상태를 더 고도화된 정보로 제공함으로써, 반복적인 정제 과정을 대폭 단축
    2) kNN-Diffusion: CLIP 인코더를 사용하여 입력 텍스트와 가장 유사한 이미지 임베딩을 검색합니다. 검색된 임베딩은 생성 과정의 조건부 정보로 활용되어 모델이 더 빠르게 수렴하도록 돕습니다.
    3) 

#### 3.3.4 Knowledge Distillation

### 3.3 Efficient Sampling

### 3.4 Compression

#### 3.4.1 Quantization

#### 3.4.2 Pruning

#### 

---

## 4. System-Level Efficiency Optimization

### 4.1 Hardware-Software Co-Design

---

## 5. Frameworks

---

## 6. Future Work

---

