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
\<img width="567" height="217" alt="image" src="https://github.com/user-attachments/assets/091ed03a-48bd-4587-a95a-9671a99000c0" />
</p>

* 회전 불변성(Rotational Invariance): LLM의 가중치 행렬에 회전 행렬을 곱해도 전체 네트워크의 출력값은 변하지 않는 특성을 활용합니다.

* 회전의 효과: 무작위 회전(Random rotation)을 적용하면 이상치(outliers)가 줄어들고 분포가 평탄해져 양자화가 쉬워집니다.

* 무작위 회전 vs. 학습된 회전: 기존 연구(예: QuaRot)처럼 무작위 회전을 사용할 경우, 어떤 회전 행렬을 쓰느냐에 따라 성능 편차가 매우 큽니다 (제로샷 추론 과제에서 최대 13점 차이).

* 최적화: SpinQuant는 운에 맡기는 무작위 회전 대신, 양자화 손실을 최소화하는 회전 행렬을 직접 학습하고 최적화(Cayley SGD 사용)하여 일관된 고성능을 보장합니다.


### 3. 두 가지 SpinQuant 전략

* SpinQuant$_{no_had}$:
    * 학습된 회전 행렬을 기존 가중치에 흡수(merge)시키는 방식입니다.
    * 장점: 추론(Inference) 시 모델 구조를 변경하거나 추가 연산을 할 필요가 없습니다.

* SpinQuant$_{had}$:
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


---

---

## Appendix

#### 1. 회전 행렬의 기초 (수학적 개념)

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

---
