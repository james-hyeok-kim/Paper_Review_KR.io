# TQ-DiT: Efficient Time-Aware Quantization for Diffusion Transformers 

저자 : 

출간 : arXiv 2025

논문 : [PDF](https://arxiv.org/pdf/2502.04056)

---
## I. INTRODUCTION

<p align = 'center'>
<img width="506" height="407" alt="image" src="https://github.com/user-attachments/assets/f55d01d3-ae32-45fb-be97-622abf425284" />
</p>

### 1. 연구 배경: 확산 모델과 DiT의 부상
* 확산 모델의 발전 U-Net $\rightarrow$ DiT(Diffusion Transformers)
* 높은 연산 비용

### 2. 기존 DiT 양자화의 주요 한계점

<p align = 'center'>
<img width="572" height="501" alt="image" src="https://github.com/user-attachments/assets/aeb0d49f-eb23-4ad6-9dc7-ce0e3400dde1" />
</p>

* 비대칭적 활성값 분포
    * DiT 블록 내의 Softmax 이후 값은 0 근처에 집중되어 있고
    * GELU 이후 값은 넓게 퍼진 음의 왜곡을 보이는 등 분포가 불균일하여 단일 양자화 파라미터로 정확히 포착하기 어렵습니다.

<p align = 'center'>
<img width="553" height="413" alt="image" src="https://github.com/user-attachments/assets/beefaa07-311b-4d6f-a541-4a4f00617729" />
</p>

* 타임스텝에 따른 분포 변화
    * 반복적인 확산 과정에서 타임스텝마다 활성값의 분포가 크게 변합니다.
    * 특정 타임스텝에 최적화된 양자화 파라미터는 다른 단계에서 큰 오차를 발생시켜 전체 성능을 저하시킵니다.

* 하늘색 막대 (상자, Box)
    * 핵심 범위: 상자의 길이는 전체 데이터 중 가운데 50%가 모여 있는 구간(사분위 범위, IQR)을 나타냅니다. 상자 내부의 주황색 선은 데이터의 중앙값(Median)입니다.
    * 중앙값과 밀집도: 상자의 아래쪽 선은 데이터의 하위 25% 지점( $Q_{1}$ ), 위쪽 선은 상위 25% 지점( $Q_{3}$ )을 의미합니다.
    * 해석: 막대가 짧으면 해당 타임스텝의 값들이 중앙값 근처에 촘촘하게 모여 있다는 뜻이고, 막대가 길면 값들이 넓게 퍼져 있다는 의미입니다.

* 위아래로 뻗은 선 (수염, Whisker)
    * 전체 범위: 상자 밖으로 뻗어 나온 선은 이상치를 제외한 데이터의 최솟값과 최댓값을 나타냅니다.
    * 변동성: 이 선이 길게 뻗어 있을수록 데이터의 전체적인 변동 폭이 크다는 것을 알 수 있습니다.

* 상단에 찍힌 점 (이상치, Outliers)
    * 극단적인 값: 선(수염)보다 훨씬 높은 곳에 찍힌 점들은 일반적인 분포에서 크게 벗어난 이상치(Outliers)입니다.
    * 중요성: 이미지 하단 설명에 따르면, 특히 초기 타임스텝(0~100 사이)에서 매우 높은 값(0.20 근처)들이 점으로 나타나는데, 이는 채널별 최대값의 변동이 매우 심하다는 것을 보여줍니다.


### 3. TQ-DiT의 제안 및 기여

* 시간 그룹화 양자화(TGQ): 타임스텝을 그룹화하고 각 그룹에 최적화된 파라미터를 할당하여 시간적 변동성 문제를 해결합니다.


---

## II. BACKGROUND AND RELATED WORKS

### A. 확산 모델 (Diffusion Models)

* 순방향 과정 (Forward Process)
* 역방향 과정 (Reverse Process)
* 노이즈 예측


### B. 확산 트랜스포머 (Diffusion Transformers, DiTs)

<p align = 'center'>
<img width="667" height="580" alt="image" src="https://github.com/user-attachments/assets/d353f12b-0a2c-4a4f-977c-586c9848d458" />
</p>


### C. 모델 양자화 (Model Quantization)

$$\hat{x}=s\cdot clip(\lfloor\frac{x}{s}\rfloor+z,0,2^{k}-1)-z \quad (5)$$

* 작업 인식 양자화 (Task-aware Quantization): 단순히 파라미터 오차를 줄이는 대신, 모델의 최종 성능 손실(Loss)을 최소화하는 방향으로 양자화 파라미터를 최적화합니다.
    * 수치적 오차보다 최종 작업의 손실 함수($\mathcal{L}$)를 최소화
 
$$min_{\Delta}\mathbb{E}[\mathcal{L}(\hat{w})] \quad (8)$$

* 사후 훈련 양자화 (PTQ)
* 기존 연구의 한계: 기존의 DiT용 PTQ 방식(예: PTQ4DiT)은 이상치 문제를 해결하려 했으나, 대규모 캘리브레이션 데이터와 긴 시간이 소요되어 자원 소모가 크다는 한계가 있었습니다.

#### Calibration

* 캘리브레이션 데이터를 모델에 입력하여 각 레이어에서 발생하는 값들의 분포를 확인한 후, 이 값들을 가장 잘 표현할 수 있는 $s$와 $z$를 계산
    * 캘리브레이션 데이터: 학습 데이터의 아주 작은 일부

---

## III. METHODOLOGY

<p align = 'center'>
<img width="707" height="258" alt="image" src="https://github.com/user-attachments/assets/fe5930b4-487a-4d39-b605-e8fe5dea4799" />
</p>


### 1. 시간 그룹화 양자화 (Time-Grouping Quantization, TGQ)
* 타임스텝에 따른 데이터 변동성을 관리하기 위해 제안

* 데이터 구축: 전체 타임스텝 $\{0, 1, ..., T-1\}$을 $G$개의 그룹( $\mathcal{G}_i$ )으로 나눕니다.

$$\mathcal{G}_{i}=\{t|t\in[\frac{(i-1)T}{G},\frac{iT}{G}-1]\} , \forall i \in \{1,..., G\} \quad(9)$$

* 균형 잡힌 샘플링: 각 그룹에서 $n$개의 샘플을 무작위로 선택하여 시간대별 활성값 분포를 포착하는 캘리브레이션 데이터셋 $\mathcal{D}_{cal}^{TG}$를 구성합니다.

$$\mathcal{D}_{cal}^{TG} = \bigcup_{i=1}^{G} \mathcal{D}_{cal}^{\mathcal{G}_{i}} \quad \text{where} \quad |\mathcal{D}_{cal}^{TG}|=n \cdot G \quad(10)$$

* 개별 최적화: 각 그룹별로 별도의 양자화 파라미터( $\Delta_{A}^{l, \mathcal{G}_i}$ )를 할당함으로써, 시간 경과에 따른 데이터 분포의 변화를 더 정확하게 반영하고 왜곡을 줄입니다.

$$\Delta_{A}^{l,\mathcal{G}_{i}}=arg~min_{\Delta}\mathbb{E}[||\epsilon_{\hat{\theta}}^{l}(x_{t},t;\Delta)-\epsilon_{\theta}^{l}(x_{t},t)||^{2}] \quad(12)$$


### 2. 헤시안 가이드 최적화 (Hessian-Guided Optimization, HO)

* 단순한 수치 오차 대신 모델의 최종 손실(Loss)에 미치는 영향을 고려하여 최적화를 수행합니다.

* 가중치 부여: 테일러 급수 전개(Taylor series expansion)를 사용하여 양자화로 인한 손실 변화를 근사합니다.
* 민감도 분석: 2차 미분 정보인 헤시안(Hessian) 행렬을 통해, 최종 출력에 더 큰 영향을 주는 '민감한' 활성값에 더 높은 가중치를 두어 최적화합니다.
* TGQ와의 결합: 이 방식은 앞서 설명한 시간 그룹화(TGQ)와 결합되어, 생성 작업에서 더 정교한 양자화 성능을 발휘합니다.


#### 1. 헤시안 가이드 최적화(HO)의 기본 원리
* 기존의 MSE 방식이 단순한 수치적 거리만 측정했다면, HO는 데이터의 민감도를 측정합니다.
* 테일러 급수 근사: 가중치 변화( $\Delta\theta$ )에 따른 손실 함수($\mathcal{L}$)의 변화를 테일러 전개로 근사하면, 1차 항(그래디언트)과 2차 항(헤시안)으로 나타낼 수 있습니다.
* 2차 항 최소화: 모델이 수렴했다고 가정하면 1차 항은 무시할 수 있으므로, 양자화 오차 최적화 문제는 2차 항($\Delta\theta^{T}\overline{H}^{(\theta)}\Delta\theta$)을 최소화하는 문제로 단순화됩니다.
* 핵심 아이디어: 헤시안(Hessian) 행렬값이 큰 위치의 오차는 최종 손실을 크게 키우므로, 이 부분의 오차를 더 엄격하게 관리하여 성능을 보존합니다


#### 수식(13) - (17)

$$\mathbb{E}[\mathcal{L}(\theta+\Delta\theta)-\mathcal{L}(\theta)]\approx\Delta\theta^{T}\overline{g}^{(\theta)}+\frac{1}{2}\Delta\theta^{T}\overline{H}^{(\theta)}\Delta\theta \quad(13)$$

* 양자화로 인한 파라미터 변화( $\Delta\theta$ )가 모델의 전체 손실( $\mathcal{L}$ )을 얼마나 변화시키는지 테일러 급수(Taylor series)로 근사한 식
* $\overline{g}^{(\theta)}$는 그래디언트(기울기), $\overline{H}^{(\theta)}$는 헤시안(2차 미분) 행렬을 의미
* 모델이 이미 수렴했다고 가정하면 그래디언트 항은 무시할 수 있으므로, 2차 항인 헤시안 부분($\Delta\theta^{T}\overline{H}^{(\theta)}\Delta\theta$)을 최소화하는 것이 양자화의 핵심 목표

$$min\mathbb{E}[\Delta z^{(l),T}H^{(z^{(l)})}\Delta z^{(l)}] \quad \text{--- (14)}$$$$min\mathbb{E}[\Delta z^{(l),T}diag((\frac{\partial\mathcal{L}}{\partial z_{1}^{(l)}})^{2},\cdot\cdot\cdot,(\frac{\partial\mathcal{L}}{\partial z_{a}^{(l)}})^{2})\Delta z^{(l)}] \quad \text{--- (15)}$$

* (14) 전체 모델 대신 각 레이어 $l$의 프리 활성화(pre-activation) 출력값 $z^{(l)}$에 대해 최적화를 수행
* (15) 연산 비용이 너무 크기 때문에, 피셔 정보 행렬(Fisher Information Matrix, FIM)을 사용하여 대각 성분(그래디언트의 제곱값)으로 근사

$$min\mathbb{E}[\Delta\epsilon^{(l)}(x_{t},t)^{T}G^{(l)}\Delta\epsilon^{(l)}(x_{t},t)]\quad (16)$$

* 확산 모델의 특성에 맞춰, 레이어 $l$의 예측 노이즈 오차( $\Delta\epsilon^{(l)}$ )와 해당 노이즈의 중요도 가중치( $G^{(l)}$ )를 결합하여 최적화 문제를 정의

$$\Delta_{A}^{l,\mathcal{G}i}=arg~min~\Delta\mathbb{E}_{t\in\mathcal{G}_{i}}[\Delta\epsilon^{(l)}(x_{t},t)^{T}G^{(l)}\Delta\epsilon^{(l)}(x_{t},t)] \quad(17)$$

*  HO 목적 함수를 특정 타임스텝 그룹($\mathcal{G}_{i}$)에 적용한 최종 공식

### 3. 멀티 지역 양자화 (Multi-Region Quantization, MRQ)

* DiT 블록 내의 불균일한 활성값 분포(Softmax, GELU)를 처리하기 위한 기술입니다.
* 영역 분할: 데이터 범위를 두 개의 지역( $R_1, R_2$ )으로 나눕니다
    * Softmax: 0에 집중된 작은 값( $R_1$ )은 최적화된 스텝 크기 $s_1$을 사용하고, 큰 값( $R_2$ )은 고정된 스텝 크기를 사용합니다.
        * 지역 1 ( $R_1$ ): $[0, 2^{k-1}s_1)$ 구간으로, 값이 밀집된 0 근처 영역입니다. 이 영역은 최적화된 스텝 크기 $s_1$을 사용하여 정밀하게 양자화합니다.
        * 지역 2 ( $R_2$ ): $[2^{k-1}s_1, 1]$ 구간으로, 상대적으로 값이 적은 영역입니다. 여기서는 고정된 스텝 크기 $s_2 = \frac{1}{2^{k-1}}$을 사용합니다.
    * GELU: 비대칭적인 양수와 음수 분포를 각각 별도의 스텝 크기로 캘리브레이션하여 오차를 최소화합니다.
        * 영역 분할: 음수 영역 $R_1 = [-2^{k-1}s_1^g, 0)$과 양수 영역 $R_2 = [0, 2^{k-1}s_2^g)$으로 나눕니다.
* 효과: 이 방식은 비균일한 분포로 인해 발생하는 성능 저하를 방지하며 이미지 생성 품질을 보존합니다.



### 4. TQ-DiT 프레임워크 (Algorithm 1)

<img width="337" height="688" alt="image" src="https://github.com/user-attachments/assets/9e93d24b-ebb4-4f00-aaed-755d406906b4" />


* Phase 1: 데이터 생성
    * 타임스텝 그룹화 및 캘리브레이션 데이터셋 구축
* Phase 2: 레이어별 계산
    * "순방향 전파(FP)로 출력값을 얻고, 역방향 전파(BP)로 기울기를 계산함 "
* Phase 3: 시간 인식 양자화
    * "HO, MRQ, TGQ를 적용하여 레이어별 최적의 양자화 파라미터 결정 "


---
