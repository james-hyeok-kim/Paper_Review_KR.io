# LRQ-DiT: Log-Rotation Post-Training Quantization of Diffusion Transformers for Image and Video Generation

저자 : Lianwei Yang∗1,2, Haokun Lin∗1,2,5, Tianchen Zhao∗3, Yichen Wu 4,5, Hongyu Zhu 3, Ruiqi Xie 3, Zhenan Sun 1,2, Yu Wang 3, Qingyi Gu†

1 Institute of Automation, Chinese Academy of Sciences

2 School of Artificial Intelligence, University of Chinese Academy of Sciences

3 Department of Electronic Engineering, Tsinghua University

4 School of Engineering and Applied Sciences, Harvard University

5 Department of Computer Science, City University of Hong Kong

발행 : arXiv preprint arXiv:2508.03485, 2025

논문 : [PDF](https://arxiv.org/pdf/2508.03485)

---

## 1. Introduction

<p align = 'center'>
<img width="876" height="700" alt="image" src="https://github.com/user-attachments/assets/829e5bc7-996d-4d37-8aaf-f84d7691ccee" />
</p>

### 2. 저비용 양자화의 두 가지 주요 장애물
* 가우시안 분포의 가중치: DiT 모델의 가중치는 0을 중심으로 꼬리가 긴(long-tail) 가우시안 형태를 띱니다.
    * 일반적인 균등 양자화(Uniform Quantization)는 데이터가 밀집된 중앙 영역에 간격을 적절히 할당하지 못해 큰 양자화 오차를 발생시킵니다.
* 활성화 함수의 이상치(Outliers): 활성화 값에서 두 종류의 이상치가 발견되었습니다
    * 완만한 이상치(Mild Outliers): 값이 약간 높은 수준으로 전체적으로 분포함.
    * 핵심 이상치(Salient Outliers): 특정 채널에 집중된 매우 큰 값으로, 양자화 과정을 심각하게 방해함.

### 3. 제안된 솔루션: LRQ-DiT

|핵심 구성 요소|설명|효과|
|:---:|:------|:------|
|Twin-Log Quantization (TLQ)|가중치의 가우시안 분포에 맞춰 양수와 음수 부분을 별도로 로그 기반 양자화를 수행합니다.|밀집 영역에 더 많은 간격을 할당하여 오차를 줄입니다.|
|Adaptive Rotation Scheme (ARS)|활성화 값의 변동성(metric J)을 측정하여 하다마르(Hadamard) 회전 또는 이상치 인식 회전을 동적으로 적용합니다.|두 종류의 활성화 이상치 영향을 효과적으로 억제합니다.|

### 4. 연구의 주요 기여 (Main Contributions)
* 활성화 값의 이상치를 식별하는 지표를 도입하고, 맞춤형 회전을 수행하는 ARS를 통해 이상치를 억제했습니다.
* PixArt, FLUX, OpenSORA 등 다양한 이미지 및 비디오 생성 모델에서 실험을 진행하여, 저비용 설정에서도 생성 품질을 유지하면서 기존 베이스라인 모델들보다 우수한 성능을 입증했습니다.


---

## 2. Related Works

### 2.1 Text-to-Image Models

#### 1. 초기 모델 및 기반 기술

* Stable Diffusion: 텍스트와 이미지 양식(modality) 사이의 의미론적 차이를 좁히며 큰 진전을 이루었습니다.
* Stable Diffusion 3 (SD3): 멀티모달 융합 능력을 더욱 강화한 후속 모델입니다.
* SDXL: 고품질 이미지를 효율적으로 생성할 수 있는 강력한 기반을 제공했습니다. 

#### 2. 아키텍처의 패러다임 전환

* U-Net에서 Transformer로: 최근 확산 모델(Diffusion Models)에서는 기존의 U-Net 백본을 트랜스포머(Transformer) 기반 아키텍처가 대체하고 있습니다.
* DiT의 장점: 이러한 변화는 다운스트림 애플리케이션을 위한 새로운 가능성을 열어주었습니다. 

#### 3. 주요 DiT 기반 모델

* PixArt-α: 이미지 품질을 유지하면서도 훈련 비용을 절감하는데 집중했습니다.
* PixArt-Σ: 4K 해상도 이미지 생성 효율성을 대폭 개선했습니다.
* FLUX (schnell 및 dev): 초고해상도 및 고충실도(High-fidelity) 이미지를 생성하는 능력을 입증했습니다. 

#### 4. 현재의 과제

* 복잡성 증가: 모델 아키텍처가 발전함에 따라 파라미터 수와 계산 복잡성이 끊임없이 증가하고 있습니다.
* 배포의 어려움: 이러한 증가세는 향후 기술 발전과 실제 환경에서의 모델 배포에 큰 어려움을 초래하고 있습니다.

### 2.2 Text-to-Video Models


#### 1. 비디오 생성으로의 확장

#### 2. 주요 텍스트-투-비디오(T2V) 모델

* Latte: 잠재 확산 트랜스포머(Latent Diffusion Transformer) 아키텍처를 통해 효율적이고 고품질의 비디오 생성을 달성했습니다.
* Sora: 텍스트 명령으로부터 선명하고 현실적인 장면 비디오를 생성하도록 훈련되었습니다.
* OpenSORA: 시공간 확산 트랜스포머(Spatial-Temporal Diffusion Transformer, STDiT) 기술을 채택하여 고해상도 비디오 합성을 포함한 다양한 시각적 생성 작업을 수행합니다.
* CogVideoX: 다중 해상도 프레임 패킹(Multi-resolution frame packing)을 활용한 점진적 훈련을 통해 일관성 있는 고품질의 장편 비디오를 생성합니다.
* Hunyuan Video: 130억 개 이상의 파라미터를 가진 비디오 생성 모델을 훈련시켰습니다.
* Wan: DiT 프레임워크를 기반으로 한 혁신적인 방법들을 도입하여 비디오 생성 분야에서 상당한 진전을 이루었습니다.

#### 3. 직면한 과제

* 이러한 모델들은 우수한 결과를 얻었으나, 모델 파라미터 수와 계산 복잡도가 지속적으로 증가하고 있습니다.

### 2.3 Model Quantization

#### 1. 양자화의 정의와 이점
* 이점: 모델 가중치를 양자화하면 메모리 사용량이 감소하고, 활성화(Activation) 값을 양자화하면 추론 속도를 높일 수 있습니다.
* PTQ의 선호: 사후 학습 양자화(PTQ)는 고비용의 재훈련 과정이 필요 없기 때문에 거대 모델 배포 시 선호됩니다.

#### 2. 아키텍처별 주요 양자화 연구
* 비전 트랜스포머(ViTs): 데이터 특성에 맞추기 위해 트윈 균등(Twin uniform), $log2$, $log\sqrt{2}$, 시프트-균등-log2 등 더 정밀한 양자화 기법들이 제안되었습니다.
* 거대 언어 모델(LLMs): 일반적인 값보다 훨씬 큰 이상치(Outliers)가 주요 도전 과제입니다.
    * 이를 해결하기 위해 난이도를 가중치로 전이하는 'Smooth Quant'나 회전 변환을 사용하는 'QuaRot', 'DuQuant' 등이 활용됩니다.
* 확산 모델(DMs): 타임스텝(Time-step)에 따라 달라지는 특징을 고려하여 양자화 파라미터를 결정하는 'Q-Diffusion'이나 'PTQ4DM' 등의 연구가 진행되었습니다.
* 확산 트랜스포머(DiTs): DiT의 특성을 반영하여 채널별 파라미터 할당(Q-DiT), 타임스텝별 채널 재할당(PTQ4DiT), 저계수 분기(Low-rank branches)를 통한 이상치 보호(SVDQuant), 미세 그룹화 및 동적 양자화(ViDiT-Q) 등이 제안되었습니다.


#### 3. 기존 연구의 한계

* 기존의 DiT 양자화 방법들은 8비트(W8A8)나 4비트(W4A8) 수준에서는 잘 작동하지만, 극도로 낮은 비트(예: 3비트 가중치) 설정에서는 여전히 심각한 성능 저하를 보입니다.
* 본 논문은 이러한 한계를 극복하기 위해 가중치와 활성화 값의 고유한 분포 특성을 분석하고 LRQ-DiT 프레임워크를 제안하는 동기가 되었습니다


---

## 3. Preliminaries
<p align = 'center'>
<img width="983" height="339" alt="image" src="https://github.com/user-attachments/assets/2fb5bb21-b583-41d9-87f1-a5eb58c774d4" />
</p>

* X축 (Layer Name): PixArt 모델 내의 다양한 트랜스포머 레이어
* Y축 (Quantization Error): 양자화 오차를 나타내는 지표인 $||W_{q}-W||_{2}$ 값

### 3.1 Quantization

$$x_{q}=clamp(\lfloor\frac{x}{s}\rfloor-z,0,2^{b}-1)\quad (1)$$

### 3.2 Rotation Transformation

$$Y=XW^{\top}=(XH)(WH)^{\top}\quad(2)$$

* 하다마드 행렬 (Hadamard Matrix)
    * 일반적으로 회전 행렬 $H$로는 구현이 간단한 하다마드 행렬이 사용됩니다.
    * 이 행렬은 원소가 ${+1, -1}$로만 구성된 직교 행렬입니다.
    * $2^{n}$ 크기의 하다마드 행렬은 다음과 같이 재귀적으로 정의됩니다.

$$H_{2}=\frac{1}{\sqrt{2}}\begin{bmatrix}1&1\\ 1&-1\end{bmatrix}, \quad H_{2^{n}}=H_{2}\otimes H_{2^{n-1}} \quad(3)$$


---
## 4.  Method


### 4.1 Twin-Log Quantization for Weights

#### 1. 배경 및 동기 (Motivation)
* DiT 모델의 가중치는 대부분 0을 중심으로 하는 가우시안(Gaussian) 형태의 분포를 따르며, 양 끝단에 긴 꼬리(long tails)가 존재하는 특성이 있습니다.
* 기존의 균등 양자화(Uniform Quantization)는 두 가지 한계
    * 조밀한 영역에 충분한 양자화 간격을 할당하지 못함
    * 긴 꼬리 영역의 극단값들을 처리하지 못함

#### 2. Twin-Log Quantization (TLQ) 방법론

* TLQ는 로그 변환을 기반으로 중앙 밀집 영역에 더 많은 양자화 간격을 할당

* 로그 변환: 가중치 $W$에 대해 $W' = \log_2(|W|)$를 수행하여 데이터의 범위를 압축
* 부호별 분리: 양수 위치 마스크 $M^+$와 음수 위치 마스크 $M^-$를 사용하여 가중치를 분리

$$W^+ = W' \cdot M^+, \quad W^- = W' \cdot M^-$$

* 개별 양자화: 양수와 음수 부분을 각각 별도의 스케일 인자( $s^+, s^-$ )와 제로 포인트( $z^+, z^-$ )를 사용하여 양자화합니다.

$$W_q = W_q^+ - W_q^-$$

#### 3. 검색 기반 클리핑 (Search-based Clipping)

* 긴 꼬리 영역의 극단값으로 인한 오차를 줄이기 위해 검색 기반 클리핑 전략을 도입
* 두 개의 하이퍼파라미터 $\alpha, \beta$를 도입하여 가중치 분포 양단의 극단값을 잘라냅니다.
    * $\alpha$: 양수 가중치($W^+$) 영역의 최대값을 조절하여 어느 지점까지 보존하고 나머지를 잘라낼지 결정합니다.
    * $\beta$: 음수 가중치($W^-$) 영역의 최대값을 조절하여 클리핑 범위를 결정합니다.
* 그리드 검색(Grid Search)을 통해 최적의 $\alpha, \beta$ 값을 결정하며, 이를 기반으로 최종 복원된 가중치 $W_f$는 다음과 같이 계산됩니다.
 
$$W_f = 2^{s^+(W_q^+ + z^+)} \cdot M^+ - 2^{s^-(W_q^- + z^-)} \cdot M^- \quad (7)$$

#### 4. 하드웨어 효율성 및 가속 (Hardware Implementation)

* 지수 분해: 지수 항을 정수 부분( $f$ )과 잔차 부분( $r$ )으로 나눕니다.
    * 정수 부분 ($f$): $s(W_q + z)$의 소수점 아래를 버린 정수 값입니다.
    * 잔차 부분 ($r$): 전체 값에서 정수 부분을 뺀 나머지 값으로, $0 \le r < 1$ 범위를 가집니다.
        * 지수 항은 $2^{f+r} = 2^{f} \cdot 2^{r}$로 표현될 수

$$f^{+} = \lfloor s^{+}(W_{q}^{+} + z^{+}) \rfloor$$
$$r^{+} = s^{+}(W_{q}^{+} + z^{+}) - f^{+}$$

* 정수 근사: 정수화 인자 $2^{-I}$를 사용하여 잔차 부분을 정수로 근사합니다.
    * 근사 원리: $2^r$ 값을 $2^{-I}$의 배수인 정수 $\mathbb{I}^{r}$로 근사
* 가속 연산: 이를 통해 실제 연산 시 복잡한 곱셈 대신 비트 시프트(SHIFT) 연산과 비트 단위 AND 연산을 활용하여 효율적인 하드웨어 배포가 가능해집니다.
* 실험 결과, 이러한 INT8 구현은 FP16 대비 약 2.15배에서 2.45배의 속도 향상을 달성했습니다.

### 4.2 Adaptive Rotation Scheme for Activations (ARS)

#### 1. 두 가지 유형의 아웃라이어 식별
* Mild Outliers (경미한 아웃라이어): 대부분의 아웃라이어로, 일반적인 수준을 약간 상회하며 보통 5를 넘지 않습니다.
* Salient Outliers (현저한 아웃라이어): 특정 채널에 집중된 매우 큰 값으로, PixArt 모델의 경우 최대 245에 달하기도 합니다.
* 이 아웃라이어들은 프롬프트에 무관하게 특정 채널에서 일관되게 나타나며, 모든 토큰에 걸쳐 분포되어 있어 양자화 정밀도를 크게 떨어뜨립니다.

#### 2. 적응형 지표 $J$ (Adaptive Metric)

* LRQ-DiT는 레이어별 활성화 변동성을 측정하는 지표 $J$를 도입

$$J=\frac{||X||_{F}}{\sqrt{BNC}} \quad (11)$$

* $||X||_{F}$는 전체 활성화 값의 크기(Frobenius norm)를 캡처합니다.
    * 행렬의 모든 원소를 제곱하여 더한 후 루트를 씌운 값, 활성화 텐서 전체의 에너지 합계 또는 물리적 크기를 나타냄
* $\sqrt{BNC}$는 레이어의 형상(Batch, Number of tokens, Channels)에 따른 스케일을 정규화합니다.
    * 단위 원소당 평균적인 크기

* 저변동 레이어 (Low Fluctuation): $J$ 값이 낮으면 활성화 값이 비교적 고르게 분포되어 있거나 경미한 아웃라이어(Mild Outliers)만 존재함을 의미합니다.
* 고변동 레이어 (High Fluctuation): $J$ 값이 높으면 특정 채널에 수백 단위의 현저한 아웃라이어(Salient Outliers)가 집중되어 있어 양자화 오류가 발생할 위험이 큼을 나타냅니다.


#### 3. 적응형 회전 기법 (ARS)의 작동 원리

지표 $J$를 기준값(Threshold, 일반적으로 1로 설정)과 비교하여 두 가지 회전 방식 중 하나를 동적으로 선택합니다. 

$$
\mathbf{X} = \begin{cases} 
\mathbf{X} \cdot \mathbf{R}_H & \text{if } \mathbf{J} < \text{Threshold}, \\ 
\mathbf{X} \cdot \hat{\mathbf{R}}_1 \cdot \mathbf{P} \cdot \hat{\mathbf{R}}_2 & \text{if } \mathbf{J} \geq \text{Threshold}. 
\end{cases}
\quad (12)
$$


* Hadamard Rotation ( $R_H$ ): $J$가 낮을 때 적용하며, 가볍고 단순한 연산으로 경미한 아웃라이어를 매끄럽게 만듭니다.
* Dual Transformation (DuQuant 기반): $J$가 높을 때(약 5%의 레이어) 적용하며, 아웃라이어 채널을 고려한 그리디(Greedy) 회전( $R_1$ ), 채널 순서 재배열( $P$ ), 그리고 추가 회전( $R_2$ )을 결합하여 강력하게 아웃라이어를 억제합니다.

#### 4. 특징 및 효과

* 훈련 불필요 (Training-free): 수백 개의 선형 레이어를 최적화할 필요 없이 보정(Calibration) 데이터만으로 즉시 적용 가능합니다.
* 무시할 수 있는 오버헤드: 복잡한 처리는 전체 레이어의 약 5%에만 적용되므로 추론 시 발생하는 추가 연산 비용이 매우 적습니다.
* 시너지 효과: 이 기법은 앞서 설명한 Twin-Log Quantization (TLQ)과 결합하여 저비트 환경에서도 생성 품질을 보존하는 데 결정적인 역할을 합니다.


---

## Appendix

### ViTs Quantization

* 데이터 특성에 맞추기 위해 트윈 균등(Twin uniform), $log2$, $log\sqrt{2}$, 시프트-균등-log2 등 더 정밀한 양자화 기법들이 제안

* 트윈 균등(Twin Uniform): 데이터 분포를 단순히 하나로 보지 않고, 두 개의 영역(예: 양수/음수 또는 서로 다른 밀도 구간)으로 나누어 각각 최적의 균등 스케일을 적용함으로써 정밀도를 높입니다.
* 로그 기반 양자화 ( $log2$, $log\sqrt{2}$ )
    * 선형 간격이 아닌 로그 스케일을 사용하여 간격을 설정합니다.
    * 값이 작은(0에 가까운) 밀집 지역에는 촘촘한 간격을, 값이 큰(꼬리 부분) 희소 지역에는 넓은 간격을 할당합니다.
    * 이는 데이터의 동적 범위(Dynamic range)를 더 효율적으로 표현할 수 있게 해줍니다.
* 시프트-균등-log2 (Shift-Uniform-Log2): 하드웨어에서 연산이 빠른 비트 시프트(Shift) 연산의 장점과 균등/로그 양자화의 정확도를 결합하여, 연산 효율성과 정밀도 사이의 균형을 맞춘 하이브리드 방식입니다.


---
