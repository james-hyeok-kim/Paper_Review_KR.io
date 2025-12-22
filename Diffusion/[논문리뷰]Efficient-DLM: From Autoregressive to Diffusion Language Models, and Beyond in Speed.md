
# Efficient-DLM: From Autoregressive to Diffusion Language Models, and Beyond in Speed

저자 : Yonggan Fu, Lexington Whalen, Zhifan Ye, Xin Dong, Shizhe Diao, Jingyu Liu, Chengyue Wu, Hao Zhang, Enze Xie, Song Han, Maksim Khadkevich, Jan Kautz, Yingyan Celine Lin, Pavlo Molchanov

출간 : arXiv preprint arXiv:2512.14067, 2025

논문 : [PDF](https://arxiv.org/pdf/2512.14067)

---

## 1. Introduction

<p align = 'center'>
<img width="400" height="350" alt="image" src="https://github.com/user-attachments/assets/d8c0cf54-a257-40c1-a667-89504104a2a1" />
</p>

### 1. 배경 및 문제 제기

* 자기회귀(AR) 모델의 한계: 현재 대규모 언어 모델(LLM)의 성공을 이끄는 AR 방식은 토큰을 하나씩 순차적으로 생성해야 합니다.
    * 이는 특히 메모리 대역폭이 제한적인 상황에서 하드웨어 활용도를 낮추고 생성 속도(처리량)를 제한하는 병목 현상을 야기합니다.
    * Memory Bound

* 확산 모델(dLM)의 등장: 이에 대한 대안으로 등장한 dLM(Diffusion Language Models)은 반복적인 노이즈 제거(denoising) 단계를 통해 여러 토큰을 병렬로 생성할 수 있어 더 높은 처리량을 기대할 수 있습니다.
    * dLM은 각 정제 단계에서 여러 위치의 토큰을 한꺼번에 예측하고 확정

* 기존 dLM의 문제점: 하지만 실제로는 KV 캐싱(Key-Value caching)과의 호환성 부족과 디코딩 시 병렬 처리의 한계로 인해 AR 모델보다 속도가 빠르지 못한 경우가 많았습니다. 또한, dLM을 처음부터 학습시키는 데 드는 비용이 매우 막대하다는 단점이 있습니다.


### 2. 핵심 아이디어: AR-to-dLM 변환

* 이미 잘 학습된 사전 학습된 AR 모델을 초기값으로 사용하여 효율적인 dLM으로 변환하는 'AR-to-dLM' 방식을 제안합니다.
* 학습 효율성: 적절한 훈련 체계를 갖추면 약 100억(10B) 토큰 수준의 낮은 비용으로도 AR 모델을 빠른 dLM으로 변환할 수 있습니다.
* 성능 유지: 기존 AR 모델의 작업 정확도를 유지하면서도 생성 속도만 획기적으로 높이는 것이 목표입니다.

### 3. 주요 제안 방법

* 블록 단위 어텐션 (Block-wise Attention): 전체 문장을 한꺼번에 보는 대신 블록 단위로 나누어 처리함으로써 AR 모델의 가중치 분포를 더 잘 보존하고, KV 캐싱을 기본적으로 지원하게 합니다.

* 위치 기반 토큰 마스킹 (Position-dependent Token Masking): 훈련 시와 테스트 시의 토큰 분포 차이를 줄이기 위해, 문장 뒷부분의 토큰에 더 높은 마스킹 확률을 부여하는 전략을 사용합니다.

### 4. 연구 성과 (Efficient-DLM 모델 제품군)

* 이러한 방법론을 통해 탄생한 Efficient-DLM 모델들은 최신 AR 모델 및 dLM 대비 뛰어난 정확도와 속도 간의 균형(Trade-off)을 보여줍니다.
    * 예시: Efficient-DLM 8B 모델은 Qwen3 8B와 대등하거나 약간 더 나은 정확도를 유지하면서, 기존 dLM(Dream 7B) 및 AR 모델(Qwen3 4B) 대비 각각 4.5배 및 2.7배 높은 처리량을 달성했습니다.

---

## 2. Efficient-DLM: A Study of Attention Patterns

* Fully bidirectional attention
    * 특정 토큰을 예측할 때, 그 토큰보다 앞에 있는 단어뿐만 아니라 뒤에 나오는 단어(미래의 정보)까지 모두 한꺼번에 고려하여 연산
    * Training: 학습할 때는 정답지(전체 문장)를 이미 알고 있기 때문에 미래 정보를 참조할 수 있는 것이고, 생성할 때는 가상의 미래(노이즈)를 채워 넣고 시작하기 때문
    * Inference: 가상의 미래(노이즈)를 깔아놓는다
        * 1단계: [나는] [M] [M] [M] (일부만 예측)
        * 2단계: [나는] [학교] [M] [간다] (뒤쪽 정보가 동시에 조금씩 드러남)

### 2.1. 기존 방식의 한계와 블록 단위 어텐션의 제안

* 기존 연구들이 채택했던 완전 양방향 어텐션은 다음과 같은 세 가지 주요 단점

1. KV 캐싱 적용의 어려움: 모든 토큰이 서로를 참조하기 때문에 효율적인 KV 캐싱이 불가능하여 생성 속도가 제한됩니다.
2. 과도한 문맥 손상: 특히 문장 뒷부분의 토큰들이 너무 많은 노이즈(Mask 토큰)에 노출되어 학습 난이도가 급격히 상승합니다.
3. 가중치 편차(Weight Drift) 발생: AR 모델의 인과적(Causal) 특성과 너무 동떨어진 패턴을 학습하게 되어, 원래 모델이 가졌던 능력을 잃어버리기 쉽습니다.

* 이를 해결하기 위해 본 논문은 블록 단위 어텐션(Block-wise Attention)을 도입
    * 구조적 특징: 블록 사이에는 인과적(Causal) 관계를 유지하여 AR 모델과 유사성을 확보하고, 블록 내부에서는 양방향(Bidirectional) 모델링을 허용합니다.
    * 장점: 이 방식은 KV 캐싱을 기본적으로 지원하며, AR 모델의 가중치 분포를 더 잘 보존하여 변환 효율성을 높입니다.


### 2.2. 클린 컨텍스트(Clean Context) 조건화

* 단순한 블록 단위 어텐션을 넘어, 학습과 테스트 사이의 간극을 줄이기 위해 '클린 컨텍스트'를 활용하는 방식을 강조합니다.

* 학습 방식: 현재 정제(Denoising) 중인 블록 앞의 모든 블록은 노이즈가 없는 '깨끗한 토큰(Clean tokens)' 상태로 입력됩니다.
    * 이유: 실제 추론(Inference) 시에는 앞 블록들이 이미 완성된 상태에서 다음 블록을 생성하기 때문입니다. 이를 학습에 반영함으로써 성능을 9.46%나 향상시켰습니다.


### 2.3. 주요 실험 결과 및 통찰 (Takeaways)

1. 최적의 조합: 블록 단위 어텐션 + 클린 컨텍스트 사용 + 토큰 시프트 제거 조합이 AR 모델의 능력을 보존하면서 dLM으로 변환하는 데 가장 효과적입니다.
2. 가중치 변화 최소화: 이 방식은 완전 양방향 방식보다 어텐션 및 FFN 레이어의 가중치 변화(Weight Drift)를 훨씬 적게 일으킵니다.
3. 학습 속도 및 정확도 향상: 기존 방식보다 평균 정확도를 약 19.12% 향상시켰으며, 더 적은 학습 데이터로도 고성능 dLM을 구현할 수 있음을 증명했습니다.


<p align = 'center'>
<img width="1079" height="486" alt="image" src="https://github.com/user-attachments/assets/b3671b4d-a131-4e4a-b22b-d589054646be" />
</p>

* (b) Bidirectional (완전 양방향)
* (c) Block-wise w/o Clean Context (클린 컨텍스트 없는 블록 단위)
* (d) Block-wise w/ Clean Context (클린 컨텍스트를 포함한 블록 단위 - 저자 제안)
* (e) Weight Change after Continuous Pretraining (가중치 변화량)
    * Bidirectional(파란색 선) 방식이 가중치 변화가 가장 큽니다.
    * 중치가 많이 변했다는 것은 AR 모델이 원래 가지고 있던 지식이나 능력을 잃어버릴 위험이 크다는 뜻입니다.
    * 반면, 저자들이 제안한 Block-wise w/ Clean Context(주황색 선)는 변화가 가장 적어 원래의 성능을 가장 잘 보존

### 학습 목적 함수(Training Objective)

$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim U[0, 1]} \mathbb{E}_{\tilde{x}_t^b \sim q(\cdot|x^b)} \left[ -\frac{1}{t} \sum_{b=1}^{B} \log p_{\theta}(x^b | \tilde{x}_t^b, x^{<b}) \right] \quad (1)$$


* $x^b$: 복원해야 할 현재 블록의 정답 토큰들입니다.
* $\tilde{x}_t^b$: 현재 블록 $x^b$에 노이즈 레벨 $t$만큼의 노이즈(마스킹)가 추가된 상태입니다.
* $x^{<b}$: 현재 블록보다 앞에 있는 '깨끗한(Clean)' 이전 블록들입니다.
* $p_{\theta}(x^b | \tilde{x}_t^b, x^{<b})$: 깨끗한 이전 문맥($x^{<b}$)과 노이즈 섞인 현재 블록($\tilde{x}_t^b$)이 주어졌을 때, 원래 블록($x^b$)을 맞출 확률입니다.
* $t \sim U[0, 1]$: 노이즈의 양($t$)을 0에서 1 사이에서 무작위로 선택하여 다양한 오염 수준을 학습합니다.


### Table 1

* 사전 학습된 AR(자기회귀) 모델인 Qwen2.5 1.5B를 dLM(확산 모델)으로 전환할 때, 어텐션 패턴과 학습 설정

<img width="1134" height="364" alt="image" src="https://github.com/user-attachments/assets/3a78c155-bd3c-4aad-9293-84fb79053bb5" />

* Attn Pattern: 시퀀스 전체를 한꺼번에 보는지(Bidirectional), 아니면 블록 단위로 나누어 보는지(Block-wise)를 나타냅니다.
* Clean Context: 현재 블록을 복원할 때 이전 블록들을 노이즈가 제거된 '깨끗한' 상태로 보게 할 것인지 여부입니다.
* Token Shift: AR 모델처럼 다음 토큰을 예측하는 방식($\checkmark$)을 유지할지, 아니면 마스크 토큰 자체를 직접 예측($\times$)할지를 결정합니다.
    * 예시처럼 순서대로 다음 단어 맞히지 않는 방식
    * 나는, [M], 간다
    * [M] 자리에 들어갈 단어인 학교에
* KV Cache: 추론 시 효율적인 메모리 재활용 기술인 KV 캐싱을 네이티브하게 지원하는지 여부입니다.

* Row (a) - 기준점 (AR Baseline): 원본 Qwen2.5 1.5B 모델의 성능으로, 평균 정확도는 41.79%입니다.
* Row (b, c) - 기존 dLM 방식 (Bidirectional): 기존 연구(Dream 등)처럼 전체 시퀀스를 양방향으로 학습시킨 경우입니다.
    * 정확도가 18~19%대로 급격히 떨어지며, AR 모델의 지식을 잘 보존하지 못함을 보여줍니다.
* Row (d) - 블록 단위 도입: 블록 단위 어텐션만 도입해도 정확도가 28.23%로 상승하며, KV 캐싱도 가능해집니다.
* Row (f) - 클린 컨텍스트의 위력: 블록 단위 학습 시 이전 문맥을 깨끗하게(Clean Context) 제공하면 정확도가 37.69%까지 올라갑니다.
    * 이는 노이즈 섞인 문맥을 쓰는 것보다 약 9.46% 높은 수치입니다.
* Row (g) - 최종 최적화 (Efficient-DLM): 클린 컨텍스트에 토큰 시프트까지 제거한 설정입니다. 평균 정확도 38.41%를 기록하며 AR 원본 성능에 가장 가깝게 도달했습니다.


### Figure 3

* 학습(Training) 시의 블록 크기와 평가(Evaluation) 시의 블록 크기

<p align = 'center'>
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/00cd8876-f2ec-4cd9-8813-e23ffd03fca0" />
</p>

* 색상 및 숫자: 6가지 작업(수학, 코딩 등)에 대한 평균 정확도를 나타내며, 노란색에 가까울수록 성능이 높음을 의미합니다

1. 너무 작은 블록은 정보가 부족하고, 너무 큰 블록은 가중치를 너무 많이 변화시킵니다.
2. 모델이 클수록 더 큰 블록 크기를 선호합니다.
3. 적절한 크기(1.5B는 16, 4B 이상은 64)로 학습하면 다양한 상황에서 높은 정확도를 낼 수 있습니다.


---


## 3. Efficient-DLM: Positiondependent Token Masking


### 3.1. 학습과 테스트 사이의 간극 (The Training-Test Gap)

* 기존 방식 (Uniform Masking): 기존 dLM은 문장의 어느 위치든 상관없이 무작위로 토큰을 마스킹합니다.
* 실제 생성 시 현상 (Left-to-right tendency): 하지만 실제로 모델이 문장을 생성할 때(신뢰도 기반 샘플링), 앞쪽 토큰이 뒤쪽 토큰보다 훨씬 먼저 확정되는 '왼쪽에서 오른쪽으로의 경향성'이 강하게 나타납니다.
* 발견된 사실: Figure 6를 보면 문장의 뒤쪽으로 갈수록 노이즈 제거(Denoising) 단계가 더 많이 필요하며, 손실(Loss) 또한 더 큽니다.
    * 즉, 문장 뒷부분이 모델에게는 더 '어려운 문제'라는 뜻입니다.

### 3.2. 위치 기반 토큰 마스킹 전략

* 문장의 뒷부분일수록 더 자주 마스킹하는 전략을 수식 (2)를 통해 제안합니다.
* 핵심 원리: 노이즈 레벨( $t$ )이 낮을 때(즉, 거의 다 완성되어 가는 단계일 때), 문장의 뒷부분에 있는 토큰들이 마스크([M])로 선택될 확률을 높입니다.
* 효과: 이는 모델이 실제 추론 상황과 비슷하게 "앞쪽 문맥이 거의 완성된 상태에서 뒷부분의 어려운 빈칸을 채우는 법"을 집중적으로 학습하게 만듭니다.

$$w_{i}(t) = \exp[\beta(1-t)i] \quad (2)$$

* $w_i(t)$: 블록 내의 $i$번째 위치에 있는 토큰이 마스킹(가려짐)될 가중치(확률)입니다.
* $i$: 블록 내에서의 상대적인 토큰 위치입니다 ($i \in [1, L']$).
* $t$: 노이즈 레벨로, 0(깨끗한 상태)에서 1(완전 노이즈 상태) 사이의 값을 가집니다.
* $\beta$: 위치 기반 편향(Positional bias)의 강도를 조절하는 하이퍼파라미터입니다
* $\beta=0$이면 모든 위치가 동일한 확률로 마스킹되는 일반적인 방식(Uniform sampling)이 됩니다.

### 3.3. 실험 결과 및 효과 (Table 2)

<p align = 'center'>
<img width="1134" height="311" alt="image" src="https://github.com/user-attachments/assets/ba10acd7-9bba-46f4-8eed-04335b794ea4" />
</p>


* 정확도 향상: 단순히 무작위로 마스킹할 때보다 위치 기반 마스킹을 사용했을 때 평균 정확도가 향상되었습니다.
* 병렬 생성 능력 강화: 특히 한 번에 많은 토큰을 생성해야 하는 공격적인 설정(TPF가 높을 때)에서 최대 4.38%의 성능 향상을 보였습니다.
* 주의점: 하지만 너무 극단적으로 뒷부분만 마스킹(Right-to-left)하면 오히려 성능이 떨어집니다. 11모델이 양방향 문맥을 활용하는 법을 배울 기회를 잃기 때문입니다.

---


## 4. Analysis of Training Dynamics

### 4.1. 학습 비용에 따른 정확도 회복

* 신속한 회복: AR 모델에서 dLM으로 전환된 모델은 약 10B(100억) 토큰 정도의 상대적으로 적은 학습량만으로도 원래의 작업 정확도를 대부분 회복할 수 있습니다.
    * 적은 데이터로 빠른 학습
* 지속적 향상: 학습 데이터(토큰)를 더 많이 투입할수록 우도(Likelihood) 추정 능력이 꾸준히 개선되며, 이는 곧 여러 선택지 중 정답을 고르는 능력의 향상
    * 데이터 넣을수록 꾸준히 성능 향상

 ### 4.2. 우도 추정과 병렬 생성의 관계

* 우도(Likelihood) 추정 성능이 좋아질수록 병렬 생성 능력도 함께 강력해진다
* 신뢰도 점수의 정확성: 더 오래 학습된 모델일수록 각 토큰에 대해 더 정확하고 신뢰할 수 있는 자신감 점수(Confidence score)를 생성
* 공격적인 병렬화 가능: 모델이 자신의 예측에 대해 더 확신을 가질 수 있게 되므로, 한 번의 연산으로 더 많은 토큰을 동시에 생성

### 4.3. 실험 결과 시각화 (Figure 8)

<p align = 'center'>
<img width="1120" height="309" alt="image" src="https://github.com/user-attachments/assets/87ab7c7f-8829-4ea5-9995-2339a741afeb" />
</p>

* (a) Likelihood Tasks: 학습량이 늘어날수록 ARCE, Hellaswag 등 선택형 작업의 성능이 안정적으로 우상향합니다.
* (b-d) Accuracy-NFE Trade-offs: GSM8K, MBPP 등 생성 작업에서 학습 토큰 수가 많아질수록(예: 200B) 적은 단계(NFE)만 거치고도 높은 정확도를 유지하는 것을 볼 수 있습니다. 이는 학습을 오래 할수록 모델이 "한 번에 더 많은 답을 찍어낼 수 있는 수준"이 된다는 것을 의미합니다.


---
## 5. Efficient-DLM: A New Family of Efficient dLMs

### 5.1. Efficient-DLM 모델군의 구성

* 모델 라인업: 1.5B, 4B, 8B 모델로 구성됩니다.
* 기반 모델: Qwen2.5-1.5B 및 Qwen3-4B/8B의 AR 모델을 초기값으로 사용하여 지속적으로 사전 학습(Continuous Pre-training)했습니다.
* 적용 기술
    * 블록 단위 어텐션(Block-wise attention)과 클린 컨텍스트(Clean context)를 적용
    * 토큰 시프트를 제거
    * $\lambda=0.1$ 설정의 위치 기반 토큰 마스킹(Position-dependent masking)을 사용했습니다.
    * 1.5B/4B 모델은 300B 토큰, 8B 모델은 500B 토큰만큼 충분히 학습시켜 병렬 생성 능력을 극대화했습니다.

### 5.2. 성능 벤치마크 결과 (Table 3 & Figure 1)

<p align = 'center'>
<img width="600" height="450" alt="image" src="https://github.com/user-attachments/assets/13264171-9557-46ec-9051-0c926fad8adf" />
</p>

* SOTA dLM 대비 우위
    * Efficient-DLM 8B는 기존 dLM인 Dream 7B보다 평균 정확도가 5.35% 높으며, 처리량(Throughput)은 4.5배 더 빠릅니다.
    * 이는 완전 양방향 모델링 대신 블록 단위 어텐션을 선택한 설계의 승리로 분석됩니다.

* SOTA AR 모델 대비 효율성
    * Efficient-DLM 8B/4B는 Qwen3 4B/1.7B와 비교했을 때, 각각 2.68배/1.82배 더 높은 처리량을 기록하면서도 정확도는 오히려 더 높았습니다

### 5.3. 주요 특징 및 장점

* One-for-All 유연성: 하나의 모델로 신뢰도 임계값(Confidence threshold)만 조절하면, 정확도 우선 모드와 속도 우선 모드를 자유롭게 전환할 수 있습니다. 이는 단일 모델이 여러 배포 시나리오에 대응할 수 있음을 의미합니다.
    * 모델은 각 denoising 단계에서 마스킹된 모든 토큰에 대해 예측 확률(Confidence score)을 계산합니다. 사용자가 설정한 임계값을 넘는 점수를 받은 토큰들만 정답으로 확정하고, 나머지는 다음 단계에서 다시 계산합니다

|임계값 설정|생성 방식|장점|단점|
|:---:|:---:|:------|:------|
|높은 임계값 (예: 0.9 이상)|신중한 생성|정확도 상승: 확실한 토큰만 확정하므로 오류가 적음 |속도 저하: 한 번에 생성되는 토큰(TPF)이 적어 연산 횟수(NFE)가 늘어남 |
|낮은 임계값|공격적인 병렬 생성 |속도 상승: 한 번에 많은 토큰을 동시에 확정하여 처리량(Throughput) 극대화 |정확도 하락: 확신이 낮은 토큰도 생성에 포함되어 품질이 떨어질 수 있음|

* 텍스트 임베딩 성능 향상: dLM의 양방향 모델링 특성 덕분에 텍스트 임베딩 작업에서 동일 크기의 AR 모델보다 뛰어난 성능을 보였습니다 (1.5B 기준 7.71%, 4B 기준 9.91% 우세).

* 메모리 제한 상황에 최적: 배치 사이즈(Batch size)가 작은 메모리 제한 시나리오에서 dLM의 속도 이점이 가장 두드러지게 나타납니다.

---

## 6. Related Work

* Ablation Study

<p align = 'center'>
<img width="1155" height="275" alt="image" src="https://github.com/user-attachments/assets/8160be13-6489-49a6-911a-cf42110b9bcf" />
</p>


---

