# PTQ4DiT: Post-training Quantization for Diffusion Transformers

저자 : Junyi Wu1,3,∗ Haoxuan Wang1,∗ Yuzhang Shang2 Mubarak Shah3 Yan Yan1,†

1University of Illinois Chicago, 2 Illinois Institute of Technology 3University of Central Florida

Git : https://github.com/adreamwu/PTQ4DiT

출간 : Advances in neural information processing systems, 2024 (NeurIPS)

논문 : [PDF](https://arxiv.org/pdf/2405.16005)

---

## 1. Introduction
<p align = 'center'>
<img width="807" height="406" alt="image" src="https://github.com/user-attachments/assets/b4931967-d06f-45d5-8907-1f9f409b973e" />
</p>

### 1. Diffusion Transformer (DiT)의 등장과 배경
* U-Net $\rightarrow$ Transformer

### 2. 해결하고자 하는 문제: 높은 연산 비용
* 연산 복잡도
* 모델 크기가 커지거나 해상도가 높아질수록 실제 환경에서 사용하기 어렵

### 3. DiT 양자화의 두 가지 핵심 과제

* 두드러진 채널(Salient Channels)의 존재: 가중치(Weights)와 활성화 값(Activations) 모두에서 극단적인 크기를 가진 채널들이 발견되며, 이를 저비트(Low-bit)로 표현할 때 큰 오차가 발생
* 시간적 변동성(Temporal Variation): 확산 모델의 추론 과정은 여러 타임스텝(Timestep)을 거치는데, 이때 활성화 값의 분포가 시간에 따라 급격하게 변하여 고정된 양자화 전략을 쓰기 어렵

### 4. 제안된 솔루션: PTQ4DiT
<p align = 'center'>
<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/efc6dc94-4aca-41e5-a078-d3bf793ac04f" />
</p>

* CSB (Channel-wise Salience Balancing): 활성화 값과 가중치의 채널 간 극단적인 값들이 서로 보완적인 관계(동시에 극단적이지 않음)에 있다는 점을 이용해, 이 분포를 재배치하여 양자화 난이도를 낮춥니다.
* SSC (Spearman's p-guided Salience Calibration): 시간에 따른 활성화 값의 변화를 캡처하기 위해, 보완성이 큰 타임스텝에 가중치를 더 두어 동적으로 salience를 조정합니다.
    * 보완성이 작은데, abs value가 높은건 quantization error 생기고 이거는 수정이 어차피 안된다. (Bit width 결정)
    * 그래서 이 논문에서는 balancing 할수있는건 해보고 그상태에서 quantization을 하자 (Outlier가 줄어 Bit width가 줄길 기대)
* 오프라인 재파라미터화(Offline Re-parameterization): 양자화 과정에서 발생하는 추가 연산을 추론 단계가 아닌 오프라인에서 미리 처리하여 추론 속도 저하를 방지합니다

### 5. 연구 결과 요약
* 8비트 양자화(W8A8)에서 원본 풀프레시전(FP) 모델과 대등한 생성 능력을 유지했습니다.
* 4비트 가중치 양자화(W4A8) 환경에서도 효과적인 이미지 생성이 가능함을 최초로 입증했습니다.

---

## 2 Backgrounds and Related Works

### 2.1 Diffusion Transformers

#### 2. DiT 블록의 구조
* DiT는 $n_B$개의 블록으로 구성되며, 각 블록은 다음과 같은 핵심 모듈을 포함합니다
    * MHSA (Multi-Head Self-Attention): 토큰 간의 관계를 학습합니다.
    * PF (Pointwise Feedforward): 각 토큰에 대해 비선형 변환을 수행합니다.
    * adaLN (Adaptive Layer Norm): MHSA와 PF 모듈 앞에 위치하여 조건부 정보를 주입합니다.

#### 3. 데이터 처리 및 조건 주입 방식
* 토큰화(Tokenization): 노이즈가 섞인 잠재값(Noised latent)과 조건부 정보(Conditional information)는 모두 저차원 잠재 공간(Latent space)에서 토큰 형태로 표현되어 처리됩니다.
* adaLN 매커니즘: 조건부 입력 $c$는 MLP를 통해 스케일($\gamma$) 및 시프트($\beta$) 파라미터로 변환됩니다. 이 파라미터들은 다음과 같은 수식으로 잠재 토큰 $Z$에 주입됩니다

$$adaLN(Z) = LN(Z) \odot (1 + \gamma) + \beta \quad (1)$$

이 과정은 모델이 다양한 조건에 동적으로 적응하게 하여 생성 품질을 높여줍니다.

#### 4. 한계점: 연산 비용

* 구조적 우수성에도 불구하고, DiT는 수많은 트랜스포머 블록을 반복적으로 사용하고 긴 반복 추론 과정을 거치기 때문에 막대한 연산 자원을 요구합니다. 이는 실전 배포를 가로막는 장애물이 되며, 본 논문이 제안하는 양자화 기술(PTQ4DiT)의 필요성을 뒷받침합니다.

---

## 3. Diffusion Transformer Quantization Challenges
<p align = 'center'>
<img width="800" height="300" alt="image" src="https://github.com/user-attachments/assets/73a81744-c1c9-40df-8f36-11ce8b19f0ce" />
<img width="800" height="300" alt="image" src="https://github.com/user-attachments/assets/43494ad1-d1b0-4bb2-9843-f19188523fe3" />
</p>

### 1. 두드러진 채널에서의 뚜렷한 양자화 오차 (Pronounced Quantization Error in Salient Channels)
* Salient Channels의 정의: 활성화 값(Activation)과 가중치(Weight) 채널 중 일반적인 범위를 크게 벗어나는 극단적인 값을 가진 채널을 'Salient Channels'라고 정의합니다.
    * Outlier
* 오차 발생 원인: 균등 양자화(Uniform quantization)를 적용할 때, 대다수의 표준 채널들의 정밀도를 유지하기 위해 이러한 극단값들을 절단(Truncation)해야 하는 경우가 많습니다.
* 누적 결과: 샘플링 과정이 진행됨에 따라 이러한 절단 오차는 원래의 풀프레시전 분포에서 크게 벗어나게 만들며, 특히 반복적인 추론 패러다임을 가진 DiT 아키텍처에서 그 영향이 두드러집니다.

### 2. 활성화 값의 시간적 변동성 (Temporal Variation in Salient Activation)
* 동적 추론 특성: DiT는 무작위 노이즈에서 이미지를 생성하기 위해 일련의 타임스텝을 거치며 작동합니다.
* 분포의 변화: 신호를 주도하는 Salient Channels의 최대 절댓값이 타임스텝별로 급격하게 변하는 것이 관찰되었습니다.
* 최적화의 어려움: 특정 타임스텝에 효과적인 양자화 파라미터가 다른 타임스텝에서는 부적합할 수 있으며, 이러한 불일치는 양자화 오차를 악화시켜 최종 생성 이미지의 품질을 저하시킵니다.
* 따라서 DiT를 정확하게 양자화하기 위해서는 전체 디노이징(Denoising) 과정 동안 진화하는 Salient Channels의 특성을 포착하는 것이 필수적입니다.

---

## 4. PTQ4DiT

### 4.1 Channel-wise Salience Balancing (CSB)

* 배경 및 동기
    * 문제점: $X$와 $W$ 모두에서 특정 채널이 매우 큰 값을 갖는 '두드러진 채널(Salient Channels)'이 존재하며, 이는 심각한 양자화 오차를 유발
    * 보완성(Complementarity)의 발견: 연구 결과, 활성화 값의 큰 값과 가중치의 큰 값이 동일한 채널에서 동시에 발생하지 않는다는 흥미로운 특성을 발견했습니다.


* CSB의 작동 원리
    * CSB는 이 보완성을 활용하여 한쪽의 극단적인 값을 다른 쪽으로 재분배함으로써 양쪽 모두 양자화하기 쉬운 형태로 만듭니다.
    * 행렬 변환: 대각 행렬인 Salience Balancing Matrices ( $B^X, B^W$ )를 도입하여 활성화 값과 가중치를 각각 변환합니다.
        * $\tilde{X} = XB^X \quad(3)$
        * $\tilde{W} = B^WW \quad(3)$
    * 채널 Salience 측정: 각 채널 $j$의 salience($s$)를 해당 채널 요소 중 최대 절댓값으로 정의합니다.
    * 균형 잡힌 Salience ( $\tilde{s}$ ) 계산: 기하 평균을 사용하여 활성화 값과 가중치 사이의 평형점을 계산합니다.

$$\tilde{s}(X_j, W_j) = (s(X_j) \cdot s(W_j))^{1/2} \quad (5)$$

* 결과 및 효과
    * 평형 달성: 변환 후에는 각 채널에서 활성화 값과 가중치의 salience가 동일해집니다 ( $s(\tilde{X}_j) = s(\tilde{W}_j)$ ).
    * 양자화 난이도 완화: 결과적으로 전체적인 채널 salience의 최대치가 감소하여, 균등 양자화 시 발생하는 오차를 효과적으로 줄일 수 있습니다.

### 4.2 Spearman’s ρ-guided Salience Calibration (SSC)

#### SSC의 핵심 아이디어

* SSC는 모든 타임스텝의 활성화 값 salience를 종합적으로 평가하되
* 가중치($W$)와 보완성(Complementarity)이 가장 큰 타임스텝에 더 많은 비중을 두어 계산합니다
* 보완성이 크다는 것은 활성화 값과 가중치의 salience 분포 사이의 상관관계가 낮다는 것을 의미하며, 이때 CSB를 통한 오차 감소 효과가 극대화됩니다

#### 수학적 구성 및 작동 방식
* Spearman's $\rho$ 통계량 활용: 타임스텝 $t$에서의 활성화 값 salience( $s(X^{(t)})$ )와 가중치 salience( $s(W)$ ) 사이의 상관계수 $\rho$를 계산합니다.
* 가중치( $\eta_t$ ) 결정: 역(Inverse) Spearman 상관계수를 소프트맥스(Softmax)와 유사한 형태의 지수 함수로 정규화하여 가중치를 할당합니다.
    * 상관계수($\rho$)가 낮을수록: 가중치($\eta_t$)가 커지며, 해당 타임스텝의 특성을 더 많이 반영합니다.
* 시간적 Salience 통합: 각 타임스텝의 salience를 위에서 구한 가중치로 가중 평균하여 최종 $s_{\rho}$를 산출합니다.

$$s_{\rho}(X^{(1:T)}) = \sum_{t=1}^{T} \eta_t \cdot s(X^{(t)}) \quad(10)$$

* 이렇게 계산된 $s_{\rho}$는 앞서 설명한 CSB(Channel-wise Salience Balancing) 공식에 대입되어, 시간에 따른 변화까지 완벽히 고려된 정교한 Salience Balancing Matrices ( $B_{\rho}^X, B_{\rho}^W$ )를 생성하는 데 사용됩니다.
* 이를 통해 DiT의 전체 디노이징 과정에서 발생하는 양자화 오차를 전략적으로 최소화할 수 있습니다


### 4.3 Re-Parameterization
* Salience Balancing Matrices( $B_{\rho}^X, B_{\rho}^W$ )를 모델에 적용할 때, 추론 단계에서 추가적인 연산 비용이 발생하지 않도록 모델 구조에 미리 통합하는 방법

#### 1. 수학적 등가성 (Mathematical Equivalence)

* 원리: $B_{\rho}^X$와 $B_{\rho}^W$는 서로 역행렬 관계에 있는 대각 행렬입니다.
* 수식: $\tilde{X} \cdot \tilde{W} = (X B_{\rho}^X) \cdot (B_{\rho}^W W) = X \cdot (B_{\rho}^X B_{\rho}^W) \cdot W = X \cdot W$

#### 2. 오프라인 통합 전략 (Offline Integration)

* 가중치 업데이트: 가중치 행렬 $W$를 $\tilde{W}$로 오프라인에서 미리 업데이트하여 선형 계층 $f$에 적용합니다.
* 활성화 값 조정 ( $B_{\rho}^X$ )의 통합
    * Post-adaLN (adaLN 이후): 선형 계층(Projection1, FC1)이 adaLN 뒤에 오는 경우, $B_{\rho}^X$를 adaLN의 파라미터( $\gamma, \beta$ )를 생성하는 MLP의 가중치와 편향(Bias)에 미리 곱해둡니다. 이를 통해 별도의 행렬 곱셈 없이 조정된 활성화 값을 얻을 수 있습니다.
    * Post-Matrix-Multiplication (행렬 곱셈 이후): 선형 계층(Projection2)이 어텐션 메커니즘의 행렬 곱셈 뒤에 오는 경우, $B_{\rho}^X$를 이전 단계의 역양자화(De-quantization) 함수 내부에 직접 흡수시킵니다.
 
* 연산 효율성: 이러한 재파라미터화 설계를 통해 PTQ4DiT는 추론 과정에서 추가적인 연산 부담(Overhead)을 전혀 주지 않습니다.
* 실용성: 수학적으로는 동일한 결과를 내면서도 양자화 오차만 효과적으로 줄여주므로, 실시간 애플리케이션 배포에 매우 적합합니다


---
## 5. Experiments

### 5.1 Experimental Settings

#### 1. 모델 및 데이터셋 (Models and Datasets)
* 평가 데이터셋: 대규모 이미지 데이터셋인 ImageNet에서 평가를 진행했습니다.
* 대상 모델: 사전 학습된 클래스 조건부(class-conditional) DiT-XL/2 모델을 사용했습니다.
* 해상도: $256\times256$ 및 $512\times512$ 두 가지 이미지 해상도에서 실험을 수행했습니다.

#### 2. 생성 및 샘플링 설정 (Generation and Sampling)
* 솔버 및 단계: DDPM 솔버를 기본으로 사용하며, 표준 생성 과정은 250회의 샘플링 단계를 거칩니다.
* 강건성 테스트: 방법론의 견고함을 평가하기 위해 샘플링 단계를 100회 및 50회로 줄인 추가 실험도 병행했습니다

#### 3. 양자화 및 보정 상세 설정 (Quantization and Calibration)
* 양자화 방식: 모든 방법론은 가중치에 대해 채널별(channel-wise) 양자화를, 활성화 값에 대해 텐서별(tensor-wise) 균등 양자화기를 사용했습니다.
* 보정 데이터셋 구축
    * $256\times256$ 해상도: 25개 타임스텝을 균등하게 선택했습니다.
    * $512\times512$ 해상도: 10개 타임스텝을 선택했습니다.
    * 각 선택된 타임스텝에서 32개의 샘플을 생성하여 보정에 활용했습니다.
* 최적화: 양자화 파라미터 최적화는 기존 Q-Diffusion의 구현 방식을 따랐습니다.

#### 4. 하드웨어 및 평가 지표 (Hardware and Metrics)
* 인프라: 모든 실험은 PyTorch 기반으로 구축되었으며, NVIDIA RTX A6000 GPU에서 실행되었습니다.
* 평가 지표
    * FID (Fréchet Inception Distance)
    * sFID (spatial FID)
    * IS (Inception Score)
    * Precision
* 샘플 수: $256\times256$ 해상도는 10,000장, $512\times512$ 해상도는 5,000장의 이미지를 샘플링하여 지표를 계산했습니다18.


### 5.2 Quantization Performance
#### 1. Baselines
* PTQ4DM, Q-Diffusion, PTQD: 확산 모델을 위해 설계된 기존의 주요 PTQ 방법들입니다.
* $RepQ^{*}$: 비전 트랜스포머(ViT)용 SOTA 방법인 RepQ-ViT를 DiT의 특성에 맞춰 개선(타임스텝 동동역학 반영 등)한 버전입니다

#### 2. 양자화 성능 분석

<p align = 'center'>
<img width="700" height="700" alt="image" src="https://github.com/user-attachments/assets/319d6355-d936-4696-b8c5-496556568506" />
</p>

* 8비트 양자화 (W8A8) 결과
    * 성능 유지: PTQ4DiT는 8비트 정밀도에서 풀프레시전(FP) 모델의 생성 능력과 거의 대등한 성능을 보여주었습니다.
    * 기준 모델 대비 우위: 대부분의 기준 모델들이 성능 저하를 겪는 것과 달리, 우리 방법은 높은 이미지 품질을 유지했습니다.
    * ImageNet 256x256 (250 steps): PTQ4DiT의 FID는 4.63으로, FP 모델의 4.53에 근접한 수치를 기록했습니다.
<p align = 'center'>
<img width="518" height="467" alt="image" src="https://github.com/user-attachments/assets/f2babc0e-68c1-4403-a454-0bba16e32e0e" />
<img width="600" height="350" alt="image" src="https://github.com/user-attachments/assets/d2c6d954-e02f-400a-933f-b12eb49d29f7" />
</p>


* 4비트 가중치 양자화 (W4A8) 결과
    * 독보적인 회복력: 훨씬 더 까다로운 4비트 설정에서 타 방법들은 극심한 성능 저하를 보였으나, PTQ4DiT는 안정적인 결과를 냈습니다.
    * 오차 폭 비교 (250 steps): PTQ4DM이 FID가 68.05나 급증한 반면, PTQ4DiT는 단 2.56 증가에 그쳤습니다.
    * 고해상도($512\times512$) 성과: Q-Diffusion 대비 FID를 41.26, sFID를 9.83 낮추며 압도적인 성능 격차를 증명했습니다.

#### 3. 효율성 및 강건성 (Efficiency & Robustness)

* 샘플링 단계 축소: 샘플링 단계를 100회, 50회로 줄여도 PTQ4DiT는 여전히 높은 성능을 유지하며 자원이 제한된 환경에서의 견고함을 보여주었습니다. 
* 비용 대비 효과: 아래 표에서 보듯, PTQ4DiT는 FP 모델과 비슷한 수준의 성능을 내면서도 계산 비용을 크게 절감했습니다.

### 5.3 Ablation Study
<p align = 'center'>
<img width="550" height="150" alt="image" src="https://github.com/user-attachments/assets/81574025-0173-4998-8b06-0e6b5053ba89" />
</p>



---


