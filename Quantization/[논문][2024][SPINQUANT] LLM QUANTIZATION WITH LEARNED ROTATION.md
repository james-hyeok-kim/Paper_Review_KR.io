# SPINQUANT: LLM QUANTIZATION WITH LEARNED ROTATION

저자 : Zechun Liu∗ / Changsheng Zhao∗ / Igor Fedorov / Bilge Soran / Dhruv Choudhary / Raghuraman Krishnamoorthi / Vikas Chandra / Yuandong Tian / Tijmen Blankevoort

Meta

출간 : arXiv preprint arXiv:2405.16406, 2024

논문 : [PDF](https://arxiv.org/pdf/2405.16406)

---

## 1. Introduction

### 1. 핵심 문제: 양자화와 이상치(Outliers)

* 이상치(Outliers)의 존재: 가중치나 활성화 행렬에 존재하는 이상치 값들은 양자화 범위를 넓혀버려, 대부분의 일반적인 값들이 사용할 수 있는 유효 비트 수를 줄어들게 하고 큰 양자화 오차를 유발합니다.

* 기존 연구들은 가중치와 활성화값 간의 양자화 난이도를 조절하거나 혼합 정밀도(mixed-precision)를 사용하는 방식으로 이를 완화하려 했습니다.

### 2. 해결책: 학습된 회전 (Learned Rotations)

<p align = 'center'>
<img width="420" height="214" alt="image" src="https://github.com/user-attachments/assets/62a381c4-935b-48b3-91bf-afc4ca652723" />
</p>

<p align = 'center'>
<img width="567" height="217" alt="image" src="https://github.com/user-attachments/assets/091ed03a-48bd-4587-a95a-9671a99000c0" />
</p>

* 회전 불변성(Rotational Invariance): LLM의 가중치 행렬에 회전 행렬을 곱해도 전체 네트워크의 출력값은 변하지 않는 특성을 활용합니다.

* 회전의 효과: 무작위 회전(Random rotation)을 적용하면 이상치(outliers)가 줄어들고 분포가 평탄해져 양자화가 쉬워집니다.

* 무작위 회전 vs. 학습된 회전: 기존 연구(예: QuaRot)처럼 무작위 회전을 사용할 경우, 어떤 회전 행렬을 쓰느냐에 따라 성능 편차가 매우 큽니다 (제로샷 추론 과제에서 최대 13점 차이).

* 최적화: SpinQuant는 운에 맡기는 무작위 회전 대신, 양자화 손실을 최소화하는 회전 행렬을 직접 학습하고 최적화(Cayley SGD 사용)하여 일관된 고성능을 보장합니다.


### 3. 두 가지 SpinQuant 전략

* $SpinQuant_{no had}$:
    * 학습된 회전 행렬을 기존 가중치에 흡수(merge)시키는 방식입니다.
    * 장점: 추론(Inference) 시 모델 구조를 변경하거나 추가 연산을 할 필요가 없습니다.

* $SpinQuant_{had}$:
    * 4-bit와 같은 극도로 낮은 비트의 활성화값(Activation)이나 KV 캐시(KV cache) 양자화가 필요한 경우 사용합니다.
    * 장점: 온라인 하다마드(Hadamard) 회전을 추가하여 내부 이상치를 더욱 효과적으로 제어합니다.

### 4. 주요 성과

* LLaMA-2 7B 모델: 가중치, 활성화값, KV 캐시를 모두 4-bit로 양자화했을 때, 전체 정밀도(Full precision) 모델과의 점수 차이를 불과 2.9점으로 좁혔습니다. 이는 LLM-QAT보다 19.1점, SmoothQuant보다 25.0점 더 높은 성능입니다.

* LLaMA-3 8B 모델: 양자화가 어려운 이 모델에서도 경쟁 기술인 QuaRot 대비 성능 격차를 최대 45.1% 줄였습니다


---

## 2. Motivation

### 2.1 Outlier Reduction

* 양자화의 문제점: LLM(대규모 언어 모델)을 양자화할 때 가장 큰 걸림돌은 이상치(Outliers)입니다

* 회전(Rotation)의 역할: 가중치나 활성화 행렬에 회전 행렬을 곱하면, 큰 값(이상치)과 작은 값들이 통계적으로 섞이게 됩니다. 이로 인해 데이터 분포가 이상치가 적은, 다루기 쉬운 형태(Gaussian-like)로 변합니다.

* 증거 (Kurtosis): 저자들은 분포의 뾰족한 정도를 나타내는 첨도(Kurtosis, $\kappa$)를 측정했습니다.
    * 회전 전: 특정 레이어의 첨도가 200을 넘을 정도로 이상치가 많음
    * 회전 후: 모든 레이어에서 첨도가 약 3(정규 분포 수준)으로 떨어짐
    * 결과: 분포가 고르게 펴지면서 양자화 오차가 크게 감소했습니다.

### 2.2 Random Rotations Produce Large Variance

<p align = 'center'>
<img width="608" height="375" alt="image" src="https://github.com/user-attachments/assets/61402208-5c56-458e-9e23-88c87f586fcd" />
</p>

* "모든 무작위 회전이 동일한 결과를 내지 않는다"는 점을 발견했습니다.

* 실험 결과: LLaMA-2 7B 모델(W4A4 양자화)에 100가지의 서로 다른 무작위 회전을 적용해 본 결과, 성능 편차가 매우 컸습니다.
    * 최고 성능과 최저 성능의 차이가 무려 13점에 달했습니다.
    * 상대적으로 성능이 좋은 '무작위 하다마드(Hadamard) 행렬'조차도 최대 6점의 성능 차이를 보였습니다.

---

## 3. Method

* 병합 가능한 회전 ( $R_1, R_2$ ):특징: 이 두 행렬은 가중치(Weight) 행렬에 병합(Merge)될 수 있습니다.
* 온라인 하다마드 회전 ($R_3, R_4$):특징: 가중치에 합치지 않고, 추론 실행 중에(Online) 연산되는 하다마드(Hadamard) 회전입니다.
    * 효과: 활성화값(Activation)이나 KV 캐시(KV-cache)를 극단적으로 낮은 비트(예: 4비트)로 양자화해야 할 때 사용합니다. 남아있는 이상치(Outlier)를 더욱 강력하게 제거하는 역할을 합니다.

* 슈티펠 다양체 (Stiefel Manifold)
    * 타겟 손실(Target Loss)을 줄이는 방향으로 이 회전 행렬들을 최적화
    * 회전 행렬의 성질을 깨뜨리지 않고 학습하기 위한 제약 조건
    * SpinQuant의 요구사항: SpinQuant는 가중치를 회전시켜도 원래 모델의 출력값과 수학적으로 동일(Rotational Invariance)해야 합니다. 이를 위해서는 학습 중인 행렬 $R$이 언제나, 매 순간 완벽한 직교 행렬(Orthonormal matrix)이어야만 합니다

### 3.1 ROTATION PARAMETERIZATION

#### 1. 잔차 연결(Residual Stream) 회전: $R_1$

* 위치: 임베딩(Embedding) 직후의 출력값 $X$에 $R_1$을 곱합니다.
* 목적: 잔차 연결을 통해 흐르는 활성화값의 이상치(Outlier)를 제거하여, 이 값을 입력으로 받는 모든 완전 연결 계층(Fully-connected layers)의 양자화를 돕습니다
* 수치적 불변성 유지 (원상복구): $R_1$으로 회전된 값을 그대로 어텐션(Attention)이나 FFN 블록에 넣으면 결과가 달라집니다. 따라서 각 블록(Attention, FFN)으로 들어가기 직전에 역행렬( $R_1^T = R_1^{-1}$ )을 곱해서 회전을 취소해 줍니다.
* 흡수(Merge): 수학적으로 이 역행렬( $R_1^T$ )은 각 블록의 입력 가중치 행렬( $W_q, W_k, W_v$ 등)에 미리 곱해서 합칠 수 있습니다. 즉, 추론 시에는 $R_1$을 위한 별도의 연산이 필요 없습니다.


#### 2. 어텐션 내부 회전: $R_2$

* 어텐션 블록(Attention Block) 내부의 이상치를 잡기 위한 추가 회전입니다.
* 위치: 어텐션의 Value 행렬( $W_v$ ) 출력과 출력 프로젝션( $W_o$ ) 입력 사이에 적용됩니다.
* 작동 방식: Value 행렬($W_v$)에 $R_2$를 곱하고, $W_o$에 들어가는 입력에 $R_2^T$를 곱합니다.
* 헤드별 적용 (Head-wise): $R_2$는 어텐션 헤드(Head) 크기에 맞는 작은 행렬( $(D_{head}, D_{head})$ )이며, 각 레이어마다 독립적으로 선택됩니다
* 효과: Value 캐시(KV Cache 중 V)와 $W_o$ 입력의 이상치를 줄여줍니다. $W_v$와 $W_o$ 사이에는 비선형 함수(ReLU 등)가 없으므로, 이 두 회전은 서로 상쇄되어 전체 결과에 영향을 주지 않습니다.
* $SpinQuant_{no had}$ 전략:위에서 설명한 **$R_1$과 $R_2$**만을 사용하여 최적화하는 방식입니다. 

#### 3. 추가적인 온라인 회전 (Hadamard): $R_3, R_4$

* 피드포워드(FFN) 내부 ( $R_4$ ): FFN 블록 내부, 다운 프로젝션(Down projection) 레이어의 입력값에 있는 이상치를 줄이기 위해 하다마드 행렬 $R_4$를 곱합니다
* KV 캐시 ( $R_3$ ): KV 캐시를 낮은 비트로 양자화해야 할 때, Key/Value 저장 직전에 하다마드 행렬 $R_3$를 곱해줍니다.
* 특징: 이들은 실시간으로 계산해야 하므로 연산량이 적은 하다마드 행렬(Hadamard Matrix)을 사용합니다. 이를 통해 추론 지연(latency)을 최소화합니다
* $SpinQuant_{had}$ 전략:학습된 회전( $R_1, R_2$ )에 온라인 하다마드 회전( $R_3, R_4$ )까지 모두 포함하여 성능을 극대화한 방식입니다.

### 3.2 Cayley-OPTIMIZED ROTATION

* SpinQuant의 핵심 알고리즘인 "어떻게 회전 행렬의 성질(직교성)을 깨뜨리지 않고 최적의 회전 각도를 학습할 것인가?"

* 양자화된 네트워크의 최종 손실(Loss, 예: 교차 엔트로피)을 최소화하는 $R_1, R_2$를 찾는 것

$$\arg \min_{R \in \mathcal{M}} \mathcal{L}_Q(R_1, R_2 | W, X) \quad (2)$$

* Cayley 변환(Cayley Transform)을 이용해 곡면(다양체)을 따라 이동하게 만듭니다.

$$R' = (I - \frac{\alpha}{2}Y)^{-1}(I + \frac{\alpha}{2}Y)R \quad (3)$$

* 장점: 이전 단계의 $R$이 직교 행렬이었다면, 업데이트된 $R'$도 수학적으로 반드시 직교 행렬임이 보장

* 결과: 가중치 파라미터는 전혀 건드리지 않고(Frozen), 전체 가중치 크기의 0.26%에 불과한 회전 행렬만 학습하여 양자화 성능을 극대

---

## 4. Experiments

#### 1. 실험 설정 (Experimental Settings) 

* 대상 모델: LLaMA-2 (7B/13B/70B), LLaMA-3 (1B/3B/8B), Mistral (7B) 등 7가지 최신 LLM.

* 평가 지표: 8가지 제로샷 상식 추론 과제(BoolQ, PIQA 등)의 평균 정확도 및 WikiText2의 퍼플렉시티(Perplexity).

#### 2. 주요 결과 (Main Results)

* SpinQuant는 두 가지 모드($no\_had$, $had$) 모두에서 기존 방법론들을 압도했습니다.

* (1) W4A8 (가중치 4-bit, 활성화 8-bit) 설정 $SpinQuant_{no had}$ (가중치 흡수 방식)만으로도 충분히 강력합니다.
    * 성과: LLaMA-3 8B 모델에서 전체 정밀도(Full Precision) 모델과의 점수 차이를 불과 1.0점으로 좁혔습니다.
    * 이는 추가적인 온라인 연산 없이 가중치 교체만으로 달성한 결과입니다.
* (2) W4A4 (가중치 4-bit, 활성화 4-bit) 설정 - "가장 어려운 구간"이 설정에서는 대부분의 기존 방법들이 실패하지만, $SpinQuant_{had}$는 뛰어난 성능을 보였습니다.
    * 비교: LLaMA-2 7B 모델 기준, 기존의 LLM-QAT 방식보다 19.1점, SmoothQuant보다 25.0점 더 높은 정확도를 기록했습니다.
    * 전체 정밀도 모델과의 성능 차이를 2.9점까지 줄였습니다5.

#### 3. 비교 실험 (Ablation Studies)

* 학습된 회전 vs. 무작위 회전
    * 단순히 무작위 하다마드 회전을 쓰는 것보다, 학습된 회전을 썼을 때 성능이 일관되게 높았습니다.
    * Mistral-7B 모델의 경우, 학습된 회전이 무작위 회전보다 15.7점 더 높은 점수를 기록했습니다.

* GPTQ와의 호환성
    * SpinQuant는 GPTQ와 함께 사용할 때 시너지가 납니다. 활성화 양자화 오차는 회전으로 잡고, 가중치 양자화 오차는 GPTQ로 잡는 전략이 가장 유효했습니다.

* 경쟁작(QuaRot) 대비 우위
    * 유사한 컨셉인 QuaRot과 비교했을 때, 더 적은 온라인 회전을 사용하면서도 2.0 ~ 28.6점 더 높은 정확도를 보였습니다

#### 4. 속도 및 효율성 (Speed Measurement)

* 추론 속도: M1 Pro CPU에서 테스트한 결과, 4-bit 양자화 모델은 16-bit 모델보다 약 3배 빠릅니다.
* 오버헤드: 성능을 위해 온라인 하다마드 회전($R_3, R_4$)을 추가한 $SpinQuant_{had}$를 사용하더라도, 지연 시간(Latency) 증가는 약 8%에 불과했습니다.
---

## 5. DISTRIBUTION VISUALIZATIONS BEFORE AND AFTER ROTATION

|구분|회전 전 (Before)|회전 후 (After)|
|:---:|:------|:------|
|목표|데이터의 분산(정보량)을 최대로 설명하는 수학적 축 생성|사람이 이해하기 쉬운 해석 가능한 축으로 재조정|
|변수의 위치|축과 축 사이(중간)에 흩어져 있음|특정 축(요인)에 가깝게 밀집됨|
|해석 난이도|어려움 (이것도 저것도 아닌 상태)|쉬움 (명확한 특성 구분)|
|대표 기법|초기 주성분/요인 추출|"Varimax (직각 회전), Promax (사각 회전)"|


<p align = 'center'>
<img width="500" height="800" alt="image" src="https://github.com/user-attachments/assets/64369e45-2e85-4981-97f4-0e238b43b6c7" />
<img width="500" height="800" alt="image" src="https://github.com/user-attachments/assets/0cad8ded-db8b-4492-99ea-0f3d78794965" />
</p>


<p align = 'center'>
<img width="500" height="800" alt="image" src="https://github.com/user-attachments/assets/c0e7cb29-f0d7-4f44-aeab-412d0fdc2f38" />
<img width="500" height="800" alt="image" src="https://github.com/user-attachments/assets/d94621e8-a00a-4dcb-a371-4649e24eb443" />
</p>



---

## Appendix

### 1. 회전행렬

#### 회전 행렬의 기초 (수학적 개념)

* 모양은 변하지 않음: 회전 행렬을 곱하면 벡터의 방향은 바뀌지만, 길이(크기)는 변하지 않습니다.
* 직교성 (Orthogonality): 회전 행렬 $R$은 "직교 행렬"이라는 특별한 성질을 가집니다. 이는 $R$의 역행렬( $R^{-1}$ )이 전치 행렬( $R^T$, 행과 열을 바꾼 행렬 )과 같다는 뜻입니다 ( $R^T = R^{-1}$ ).
    * 쉽게 말해, 어떤 물체를 오른쪽으로 회전시킨 뒤( $R$ ), 다시 반대로 돌려놓으려면( $R^T$ ) 아주 간단한 계산만으로 원상복구가 가능하다는 뜻입니다
 
#### 예제

$$X = \begin{bmatrix} 10 \\ 0 \end{bmatrix}$$

$$
R =
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
\approx
\begin{bmatrix}
0.707 & -0.707 \\
0.707 & 0.707
\end{bmatrix}
$$


$$
X' = R \times X =
\begin{bmatrix}
0.707 & -0.707 \\
0.707 & 0.707
\end{bmatrix}
\begin{bmatrix}
10 \\
0
\end{bmatrix}
$$

$$
X' =
\begin{bmatrix}
7.07 \\
7.07
\end{bmatrix}
$$

$$\sqrt{10^2} = 10, \sqrt{7.07^2 + 7.07^2} \approx 10$$

### 2. 하다마드(Hadamard Rotation)

* 일반적인 회전 행렬은 $\sin, \cos$ 값(실수)으로 복잡하게 이루어져 있지만, 하다마드 행렬( $H$ )은 오직 $+1$과 $-1$로만 이루어진 아주 단순하고 강력한 행렬, 정규화를 위해 상수가 곱해지긴 합니다.

* 하다마드 회전은 $+1, -1$로 이루어진 특수 행렬로, 데이터를 가장 골고루 섞어주는 회전입니다
* 계산 속도가 매우 빨라서, 실시간으로 데이터가 변하는 활성화값(Activation)이나 KV 캐시를 회전시키는 데 사용됩니다 ( $R_3, R_4$ ).
* SpinQuant는 "정밀한 학습된 회전( $R_1, R_2$ )"과 "빠른 하다마드 회전( $R_3, R_4$ )"을 조합하여 속도와 정확도를 모두 잡았습니다.

---
