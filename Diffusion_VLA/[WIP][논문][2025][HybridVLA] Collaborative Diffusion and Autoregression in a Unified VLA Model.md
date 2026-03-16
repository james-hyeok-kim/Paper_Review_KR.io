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

### 1. 논문 요약 (Summary)

* HybridVLA는 로봇 조작(Manipulation)을 위해 Diffusion + Autoregressive를 하나의 LLM에 통합한 비전-언어-행동(VLA) 모델
    * Image + Text(command) $\rightarrow$ LLM prefix $\rightarrow$ Iteration 4 times $\rightarrow$ MLP $\rightarrow$ 로봇 행동의 물리적 수치
    * 첫 Image는 Gaussian Noise
    * 예측 하는 대상이 행동이 아니라 노이즈(제거 될 노이즈)
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



---

---
