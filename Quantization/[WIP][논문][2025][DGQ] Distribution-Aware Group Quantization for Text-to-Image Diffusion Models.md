# DGQ: Distribution-Aware Group Quantization for Text-to-Image Diffusion Models

저자 : Hyogon Ryu, NaHyeon Park, Hyunjung Shim∗

Korea Advanced Institute of Science and Technology (KAIST)

{hyogon.ryu, julia19, kateshim}@kaist.ac.kr

출간 : arXiv preprint arXiv:2501.04304, 2025

논문 : [PDF](https://arxiv.org/pdf/2501.04304)

---

## 1. Introduction

<p align = 'center'>
<img width="427" height="286" alt="image" src="https://github.com/user-attachments/assets/796952ef-92b7-4f4d-ae96-45961b8c964b" />
</p>

### 핵심 분석

* Activatioin Outliers: Outlier가 이미지 품질 유지에 결정적 역할, Layer-wise 양자화는 Outlier가 보존되지 않으므로 성능 저하가 크다

* Cross Attention: Self-attention과 달리 `Start`토큰과 나머지 토크들에 대응하는 두개의 뚜렷한 피크를 가진다. Uniform Quantizer를 사용하면 텍스트 이미지 간 정렬 성능이 떨어진다.


### 제안 방법

* 각 그룹별 맞춤 양자화 스케일 적용
* Start 토큰은 별도로 분리하여 고정밀도로 유지 나머지 점수는 로그 양자화를 수행 



### 주요 기여 및 성과

<p align = 'center'>
<img width="376" height="369" alt="image" src="https://github.com/user-attachments/assets/e1616671-7e53-474b-9201-13b2644a21f6" />
</p>

* 최초의 저비트 양자화 성공: Weight Fine-tuning 없이도 텍스트-이미지 확산 모델에서 8비트 미만(예: 6비트)의 양자화를 성공 첫 번째 사례입니다. 

* 우수한 성능: MS-COCO 데이터셋에서 Full Precision 모델보다 낮은 FID 점수(13.15)를 기록하면서도, 연산량(BOPS)을 93.7%까지 절감했습니다. 

* 범용성: 하드웨어 친화적인 설계를 통해 실제 엣지 디바이스 등 다양한 환경에서 확산 모델을 효율적으로 배포할 수 있는 기반을 마련했습니다.

---

## 2. RELATED WORK

### 1. 모델 양자화 기법 (PTQ vs. QAT)

* 양자화 후 미세 조정: 최근 연구들은 PTQ 이후 미세 조정을 수행하기도 하지만 여전히 추가 연산 비용이 발생합니다. 반면, DGQ는 이러한 미세 조정이 필요 없다는 강점이 있습니다.

### 2. 확산 모델 전용 양자화 연구
* 타임스텝 고려: 초기 연구들은 확산 모델의 타임스텝(단계)별 특성에 집중했습니다.
    * Q-Diffusion: 타임스텝별 활성화 다양성을 고려해 보정 데이터셋을 구축했습니다.
    * TFMQ-DM: 시간적 특징을 더 잘 보존하기 위해 재구성 블록 구조를 변경했습니다.
* 한계: 이러한 방식들은 텍스트 조건(Text-condition)에 따른 특성을 고려하지 않았다는 단점이 있습니다.

### 3. 텍스트 조건 및 혼합 정밀도 연구
* MixDQ: 텍스트 조건에 따른 레이어별 민감도를 측정하고 혼합 정밀도(Mixed Precision)를 적용했습니다.
* PCR: 타임스텝에 따라 비트 정밀도를 동적으로 조절하는 매커니즘을 도입했습니다.
* 한계: 이 기법들은 서로 다른 비트 수를 섞어 쓰는 혼합 정밀도 방식이라 실제 하드웨어 구현이 어렵습니다.

### 4. 활성화 이상치(Outlier) 관련 연구
* QuEST: 활성화 값의 이상치가 중요함을 강조하며 가중치 양자화를 가변적으로 적용했습니다.
* DGQ와의 차이: QuEST와 달리 DGQ는 활성화 양자화(Activation Quantization)에 집중하며, 두 방법은 동시에 적용될 수도 있습니다.

---

## 3. METHOD

### 3.1 PRELIMINARY: HARDWARE-FRIENDLY QUANTIZATION
#### 1. 선형 양자화 (Linear/Uniform Quantization)

* 양자화: $x_{q}=clamp(\lfloor\frac{x}{s}\rfloor+z,0,2^{b}-1)$
* 역양자화: $x_{dq}=s\cdot(x_{q}-z)\approx x$

#### 2. 로그 양자화 (Logarithmic Quantization)

* 양자화: $x_{q}=clamp(\lfloor-log_{2}(\frac{x}{s})\rceil,0,2^{b}-1)$
* 역양자화: $x_{dq}=s\cdot2^{-x_{q}}\approx x$


### 3.2 ANALYZING CHARACTERISTICS OF TEXT-TO-IMAGE DIFFUSION MODELS

#### 1. Activation Outliers

<p align = 'center'>
<img width="796" height="421" alt="image" src="https://github.com/user-attachments/assets/73420330-962f-4508-9118-0dedb66d2581" />
</p>


* 실험 내용: 동일한 텍스트 프롬프트에서 활성화 값을 무작위로 제거(0으로 설정)했을 때와 이상치를 제거했을 때의 결과를 비교했습니다.

* 결과: 무작위 활성화를 제거하면 이미지에 거의 영향이 없었지만, 이상치를 제거하면 이미지 형체가 무너지고 품질이 급격히 저하되었습니다. 이는 특정 이상치들이 이미지 생성의 핵심 정보를 담고 있음을 의미합니다.

* 이상치 발생 패턴: 이상치는 모든 곳에 퍼져 있는 것이 아니라, 특정 채널(Channel)이나 특정 픽셀(Pixel)에 집중되어 나타나는 경향을 보입니다. 또한, 이러한 발생 위치는 레이어마다 다르지만 시드(Seed)나 프롬프트가 바뀌어도 유지되는 모델의 고유한 특징입니다.

#### 2. Cross-attention Scores

<p align = 'center'>
<img width="800" height="458" alt="image" src="https://github.com/user-attachments/assets/2dbbf36d-8eb4-462a-b3c4-047347205b29" />
</p>

* <start> 토큰의 피크(Peak): 교차주의 점수 분포를 보면 1.0에 가까운 값에서 뚜렷한 피크가 관찰되는데, 이는 주로 <start>(또는 <bos> Begining of Sentence) 토큰에 해당합니다.


* 배경 픽셀의 역할: 분석 결과, 이미지의 배경 픽셀들이 <start> 토큰에 대해 매우 높은 주의 점수를 가지는 것으로 나타났습니다. 이 토큰을 제거하거나 값을 조정하면 이미지의 세부 사항이 크게 변하므로, 정밀하게 보존해야 할 대상입니다.
    * 실험적 확인: 이 <start> 토큰의 점수를 강제로 낮추거나 제거할 경우, 이미지의 주요 내용은 유지될 수 있으나 세부적인 디테일과 이미지 품질(Fidelity)이 크게 손상됩니다 Figure 5 - (a).


* 프롬프트 의존성: Self Attention는 입력 프롬프트와 상관없이 분포 범위가 일정하지만, Cross Attention는 입력 프롬프트에 따라 점수 분포가 매우 동적으로 변화합니다. 특정 픽셀과 관련된 텍스트 토큰의 개수가 프롬프트마다 다르기 때문에 점수가 집중되거나 분산되기 때문입니다 Figure 5 - (b).

### 3.3 DISTRIBUTION-AWARE GROUP QUANTIZATION

#### 1. Outlier-preserving group quantization

* 최적 차원 선택: 이상치가 특정 채널에 있는지 아니면 특정 픽셀에 있는지 판단하기 위해 각 차원의 변동성을 측정하는 지표 $D_{d}$를 계산합니다.
    * $D_{d}$가 더 큰 차원(채널 또는 픽셀)을 그룹화 차원으로 선택합니다.
    * $d^{*}=arg~max_{d}D_{d}$

$$D_{d}=(max_{i}a_{i,d}^{max}-min_{i}a_{i,d}^{max})+(max_{i}a_{i,d}^{min}-min_{i}a_{i,d}^{min}) \quad(4)$$



* K-means 클러스터링 기반 그룹화: 선택된 차원( $d^{*}$ )에서 활성화 값들을 $K$개의 그룹으로 나누고, 각 그룹마다 개별적인 양자화 스케일( $s_{k}$ )과 제로 포인트( $z_{k}$ )를 할당합니다.

* 그룹별 양자화 파라미터: 각 그룹( $k$ )마다 독립적인 양자화 스케일( $s_k$ )과 제로 포인트( $z_k$ )를 계산하여 적용합니다
    * 이상치 보존

$$s_{k}=\frac{max~x-min~x}{2^{b}}$, $z_{k}=min~x$$

* 타임스텝 적응: 확산 모델의 역과정(Denoising) 단계마다 활성화 값의 분산이 변하므로, 각 타임스텝별로 별도의 양자화 파라미터를 사용합니다.


#### 2. Attention-aware quantization

* 로그 양자화 (Logarithmic Quantization): Attention scores가 지수적 분포를 따르기 때문에, 선형 방식보다 작은 값을 더 정밀하게 표현할 수 있는 로그 양자화기를 사용합니다.


* <start> 토큰 분리: 분석을 통해 확인된 핵심 요소인 <start> 토큰의 주의 점수는 양자화하지 않고 Full Precision로 유지하여 이미지의 세부 디테일을 보존합니다.


* 프롬프트별 동적 양자화: 입력 프롬프트에 따라 Cross Attention Score의 범위가 동적으로 변하는 특성을 반영하여, 추론 시점(Inference-time)에 해당 프롬프트의 최대값에 맞춰 양자화 스케일을 실시간으로 조정합니다.
    * 추론 시점에 <start> 토큰을 제외한 나머지 점수들의 최대값($s=max(A_{[:,1:]})$)을 기준으로 양자화 스케일을 실시간 조정 

$$\hat{A}=[A_{[:,0]},s\cdot2^{-A_{[:,1:]}^{q}}]$$

$$\hat{A}\hat{V}=[A_{[:,0]}\hat{V}_{[0,:]},s\cdot2^{-A_{[:,1:]}^{q}}\hat{V}_{[1:,:]}]$$ 

* 여기서 $A_{[:,0]}$은 전체 정밀도로 유지되는 <start> 토큰의 점수입니다.


---

## 4. EXPERIMENTS

---
