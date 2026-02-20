# DMQ: Dissecting Outliers of Diffusion Models for Post-Training Quantization

저자 : Dongyeun Lee Jiwan Hur Hyounguk Shon Jae Young Lee Junmo Kim

KAIST

발표 : arXiv 공개: 2025년 7월 17일, ICCV 2025 (International Conference on Computer Vision)

논문 : [PDF](https://arxiv.org/pdf/2507.12933v1)

---

## 1. Introduction

### 2. 기존 기술의 한계와 "아웃라이어" 문제

* 기존의 사후 양자화(PTQ) 방식들은 저비트(예: W4A6) 환경에서는 성능이 급격히 저하되는 한계가 있었습니다

* 동적 활성화 분포: 타임스텝에 따라 활성화 값의 분포가 크게 변합니다.
* 오차 누적: 역확산 과정이 진행됨에 따라 양자화 오차가 누적되어 최종 출력 품질을 해칩니다.
* 채널별 아웃라이어(Outliers): 특정 채널에 분포하는 극단적인 아웃라이어 값이 양자화 범위를 확장시켜, 결과적으로 나머지 일반 채널들의 양자화 정밀도를 떨어뜨립니다.

### 3. 제안 방법론: DMQ (Dissecting Outliers)

* 핵심 기술 구성
    * 학습된 등가 스케일링 (Learned Equivalent Scaling, LES): 채널별 스케일링 인자를 최적화하여 가중치와 활성화 값 사이의 양자화 난이도를 재분배하고 전체 오차를 줄입니다.
    * 적응형 타임스텝 가중치(Adaptive Timestep Weighting): 초기 역확산 단계가 최종 품질에 결정적인 영향을 미친다는 점에 착안하여, 학습 시 이 중요한 단계들에 우선순위를 둡니다.
    * 채널별 2의 거듭제곱 스케일링 (Power-of-Two Scaling, PTS): 스킵 연결(Skip Connection)과 같이 아웃라이어가 극심한 레이어에서 활성화 값을 직접 스케일링하여 아웃라이어를 제거합니다.
    * 투표 알고리즘(Voting Algorithm): 적은 양의 보정 데이터로도 신뢰할 수 있는 PTS 인자를 선택할 수 있도록 설계되었습니다.

<p align = 'center'>
<img width="512" height="400" alt="image" src="https://github.com/user-attachments/assets/b08157f9-5815-4996-a004-68a42e4ee6e4" />
</p>

---

## 2. Related Work

### 2. 확산 모델을 위한 양자화 연구


* QAT 접근법: 타임스텝 인식 스무딩 연산을 제안한 Q-DM이나, 저비트 어댑터(LoRA)를 활용한 EfficientDM 등이 있으나 연산 자원이 많이 소모된다는 단점이 있습니다.
* PTQ 접근법: 크게 세 가지 방향으로 연구되어 왔습니다.
    * 보정 데이터 구성: 타임스텝별 데이터 샘플링을 최적화하는 연구 (PTQ4DM, Q-Diffusion, EDA-DM 등).
    * 노이즈 보정: 양자화로 발생한 노이즈를 직접 수정하는 연구 (PTQD, TAC-Diffusion 등).
    * 타임스텝별 파라미터: 각 단계마다 다른 양자화 파라미터를 적용하는 연구 (TFMQ-DM, TDQ 등).
* 기존 연구의 한계: 이러한 노력에도 불구하고, 아웃라이어(극단값) 문제를 간과했기 때문에 활성화 값이 8비트 미만으로 떨어지면 성능이 크게 저하됩니다.

### 3. 채널별 등가 스케일링 (Equivalent Channel-wise Scaling)

* 기본 원리: 활성화 채널의 크기를 줄이는 대신 가중치를 그만큼 키워 수학적 동등성을 유지하면서 양자화 난이도를 조절하는 방식입니다.
* 기존 한계: 대규모 언어 모델(LLM)에서 성공적이었던 SmoothQuant 방식을 확산 모델(DiT 등)에 적용하려는 시도가 있었으나, 스테이블 디퓨전과 같은 모델에서는 성능이 최적화되지 않았습니다.
* DMQ의 차별점: 기존 방식은 스케일링 인자가 타임스텝마다 변해 가중치에 융합(fusing)하기 어려운 실질적인 한계가 있었으나, DMQ는 이러한 인자를 학습을 통해 결정하여 효율적으로 동적인 활성화 변화를 포착합니다.

---

## 3. Preliminaries

### 1. 확산 모델 (Diffusion Models)

* 확산 모델은 노이즈 제거 과정을 통해 데이터 분포 $p(x_0)$를 근사하는 생성 모델입니다.
* 순방향 마르코프 연쇄: 데이터를 가우시안 노이즈로 점진적으로 오염시키며, $q(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_tI)$로 정의됩니다.
* 역방향 회복 과정: 학습된 신경망 $\mu_\theta$와 $\Sigma_\theta$를 사용하여 $x_t$로부터 $x_{t-1}$의 평균과 분산을 예측하며 원래 데이터를 복구합니다.
* 양자화의 어려움: 타임스텝 임베딩을 통해 파라미터를 공유하므로 단계마다 활성화 값의 분포가 크게 변합니다. 특히 각 단계의 양자화 오차가 누적되는 성질 때문에 모델이 원래 분포에서 벗어나게 됩니다.

### 2. 양자화 (Quantization)

* 본 연구는 부동소수점을 저비트 정수로 변환하는 균일 양자화(Uniform Quantization) 방식을 채택합니다.
* 기본 공식
    * 텐서 $X$의 양자화 및 역양자화는 $X \approx Q(X) = s \cdot \tilde{X}$로 표현됩니다.
    * 여기서 $\tilde{X} = \text{clamp}(\lfloor X/s \rceil, l, u)$이며, $s$는 스케일 인자, $[l, u]$는 비트 너비에 의해 결정되는 범위입니다.
* 선형 레이어의 적용
    * 출력 $Y$는 $Y = XW \approx (s^{(X)}\tilde{X})(s^{(W)}\tilde{W})$와 같이 활성화와 가중치 각각의 스케일을 사용하여 계산됩니다.
* 효율적 정수 연산
    * 하드웨어 가속을 위해서는 고정밀 스케일 값 $s$를 행렬 곱셈 합산 기호 밖으로 분리해야 합니다. 이를 위해 활성화는 텐서 단위(Per-tensor)로, 가중치는 채널 단위(Per-channel)로 양자화하는 것이 일반적입니다.
* 성능과 비용의 균형
    * 활성화를 샘플 단위(Per-sample)로 양자화하면 정밀도는 높지만 실시간 연산 비용이 발생하므로, 본 논문에서는 효율성을 위해 정적 텐서 단위 양자화를 사용합니다.

---

## 4. Method

### 4.1. Learned Equivalent Scaling

#### 1. 채널별 등가 스케일링의 원리

* 이 기술의 핵심은 수학적 동등성을 유지하면서 활성화( $X$ )와 가중치( $W$ ) 사이의 양자화 난이도를 재분배하는 것입니다.
* 다음과 같이 행렬 곱셈을 재구성합니다

$$Y=(X/\tau)(\tau^{\top}\odot W)=\hat{X}\hat{W}$$

* $\tau$ (스케일링 인자): 활성화의 각 채널을 $\tau$로 나누고, 가중치에는 $\tau$를 곱합니다.
* 목적: 활성화에 있는 극심한 아웃라이어를 가중치 쪽으로 일부 넘겨주거나 그 반대로 하여, 양자화하기 더 수월한 분포( $\hat{X}, \hat{W}$ )를 만드는 것입니다.
* 차별점: 기존의 SmoothQuant는 단순히 최대값의 비율로 $\tau$를 정했지만, DMQ는 실제 양자화 오차를 최소화하도록 이 값을 학습합니다.
    * 레이어 단위별로 스케일링 인자만 최적화 (QAT 아니고, PTQ)

$$\mathcal{L}_{i}=||X_{i}W-Q(\hat{X}_{i})Q(\hat{W})||^{2} \quad (6)$$


#### 2. 적응형 타임스텝 가중치 (Adaptive Timestep Weighting)

<p align = 'center'>
<img width="506" height="352" alt="image" src="https://github.com/user-attachments/assets/98c49936-a0df-47ed-8430-8fd1d0c0bc29" />
</p>

* 확산 모델은 여러 단계( $t$ )를 거치는데, 단순히 모든 단계의 오차를 똑같이 취급하면 안 됩니다.
* 오차의 역설: 연구팀의 분석 결과, 노이즈 제거 후반부(작은 $t$ )는 양자화 오차 자체가 크지만, 초기 단계(큰 $t$ )에서 발생한 작은 오차가 연쇄적으로 증폭되어 최종 이미지 품질에 더 치명적인 영향을 줍니다.
    * 왼쪽 그래프: MSE vs. FID (전체 경향)MSE (파란 선): 노이즈 제거 과정이 진행될수록(즉, $t$가 0에 가까워질수록) 평균 양자화 오차는 점점 커집니다.
    * FID (빨간 선): 하지만 최종 이미지 품질(FID)은 MSE와 정비례하지 않습니다.
    * 핵심 통찰: 초기 단계(큰 $t$ )는 MSE가 작음에도 불구하고 FID에 큰 영향을 미칩니다. 이는 초기에 발생한 작은 오차가 전체 과정에 걸쳐 누적(Error Accumulation)되어 최종 결과물을 크게 망치기 때문입니다.
* 가중치 손실 함수: 이를 해결하기 위해 Focal Loss에서 영감을 받은 적응형 손실 함수를 사용합니다.

$$\lambda_{t_{i}}=(1-\frac{\Lambda_{t_{i}}}{\sum_{t^{\prime}\in T}\Lambda_{t^{\prime}}})^{\alpha}$$

* 효과: 누적 오차가 적은(하지만 중요한) 초기 타임스텝에 더 높은 가중치를 부여하여 정밀하게 학습하도록 유도합니다.

<p align = 'center'>
<img width="1021" height="359" alt="image" src="https://github.com/user-attachments/assets/74041a76-245d-40f2-9ade-a526a38dc499" />
</p>

* (a) & (b) 원본 상태 (Original): 활성화와 가중치 모두에 매우 큰 아웃라이어(어두운 부분)가 존재하여 양자화가 어려운 상태입니다.
* (c) & (d) 등가 스케일링 적용 후 (After Equivalent Scaling): LES 기술을 통해 아웃라이어가 활성화와 가중치 양쪽으로 재분배되면서, 두 분포 모두에서 극단적인 값들이 눈에 띄게 줄어든 것을 볼 수 있습니다.
* (e) 2의 거듭제곱 스케일링 적용 후 (After PTS): LES만으로 해결되지 않았던 활성화의 잔여 아웃라이어들을 PTS가 직접적으로 깎아내어, 양자화하기 훨씬 수월한 평탄한 분포를 만들어냅니다.


#### 3. 효율적인 추론을 위한 파라미터 융합

* 학습된 $\tau$를 추론할 때마다 계산하면 오버헤드가 발생합니다.  DMQ는 이를 방지하기 위해 다음과 같은 트릭을 씁니다
* 가중치 융합: 가중치 $W$는 고정된 값이므로 $\tau$를 미리 곱해서 저장해 둡니다.
* 활성화 스케일 통합: 활성화 $X$는 계속 변하지만, 정적 양자화(Static Quantization)를 사용하므로 양자화 스케일( $s^{(X)}$ ) 안에 $\tau$ 값을 통합시켜 계산 과정에서 추가 연산이 없도록 만듭니다. 


### 4.2. Power-of-Two Scaling

<p align = 'center'>
<img width="527" height="498" alt="image" src="https://github.com/user-attachments/assets/a2f15179-9207-4c77-96da-934ec5c73be8" />
</p>

* 스킵 연결(Res. skip)의 극심한 분산: 그래프를 보면 다른 레이어들에 비해 Res. skip (스킵 연결) 레이어의 분산이 압도적으로 높게 나타납니다. 이는 해당 레이어에 매우 강력한 **아웃라이어(극단값)**들이 존재함을 시사합니다. 
* 아웃라이어의 원인: 연구팀은 특히 정규화(Normalization) 과정이 없는 스킵 연결 레이어에서 이러한 채널 간 분산이 심화된다는 점을 발견했습니다.

#### 1. 활성화 아웃라이어의 도전 과제

* 연구팀은 확산 모델의 활성화 분포를 분석한 결과, 특정 레이어에서 심각한 문제가 발생함을 발견했습니다.
* 스킵 연결(Skip Connections)의 문제: 정규화(Normalization) 과정이 없는 스킵 연결 레이어는 채널 간 분산이 매우 크고 극단적인 아웃라이어가 존재합니다.
* LES의 한계: LES는 가중치와 활성화 사이의 난이도를 재분배하지만, 아웃라이어 자체가 너무 극심한 경우에는 이 부담을 가중치로 온전히 넘기기 어려워 양자화 오차가 여전히 발생합니다.

#### 2. Power-of-Two Scaling (PTS)의 정의

* PTS는 활성화의 각 채널을 2의 거듭제곱( $2^{\delta}$ ) 인자로 직접 스케일링하여 동적 범위를 개별적으로 조정합니다.
* 공식: $\tilde{X}=\text{clamp}(\lfloor \frac{X}{2^{\delta}\odot s^{(X)}} \rceil,l,u)$
* 여기서 $\delta$는 채널별 2의 거듭제곱 지수 벡터입니다.

#### 3. 하드웨어 효율성: 비트 시프팅(Bit-shifting)

* PTS의 가장 큰 장점은 연산 오버헤드가 거의 없다는 점입니다.
* 원리: 2의 거듭제곱을 곱하거나 나누는 연산은 하드웨어에서 비트 시프트(<<, >>) 연산과 동일합니다.
* 구현: 가중치를 로드할 때 비트 시프트를 적용하여 연산하므로, 복잡한 부동소수점 곱셈 없이도 아웃라이어를 효과적으로 제거할 수 있습니다. 이 방식은 PyTorch FP32 대비 최대 5.17배의 속도 향상을 보여줍니다.

#### 4. 강건한 투표 알고리즘 (Robust Voting Algorithm)


<p align = 'center'>
<img width="502" height="433" alt="image" src="https://github.com/user-attachments/assets/a0b32ea4-f093-4542-b8a7-2ba50e661476" />
<img width="508" height="465" alt="image" src="https://github.com/user-attachments/assets/4dc8dfc7-aab5-4d12-9e95-5662134230bf" />
</p>


* 채널별 후보 선택: 각 샘플 및 채널별로 양자화 오차를 최소화하는 후보 인자( $2^0, 2^1, ..., 2^D$ )를 평가합니다.
* 투표 및 임계값 적용: 모든 샘플에서 가장 많이 선택된 최빈값( $\delta_{k}^{mode}$ )을 찾고, 이에 동의하는 샘플의 비율( $r_k$ )이 특정 임계값( $\kappa$ )을 넘을 때만 해당 인자를 적용합니다.
* 보수적 접근: 합의가 이루어지지 않으면 스케일링을 적용하지 않음으로써( $\delta_k = 0$ ), 소수의 이상치에 의한 모델 왜곡을 방지합니다.

#### 5. 선택적 적용

* PTS는 모든 레이어에 적용하는 것보다 스킵 연결 레이어에만 선택적으로 적용하는 것이 더 효과적입니다.
* 이는 극단적인 아웃라이어가 발생하는 특정 지점을 집중 타격함으로써 연산 효율과 양자화 정밀도를 동시에 잡는 전략입니다.

---

## 5. Experiments

### 5.1. Main results

#### 1. 비조건부 이미지 생성 (Unconditional Generation)

<p align = 'center'>
<img width="1022" height="535" alt="image" src="https://github.com/user-attachments/assets/c5f5eb56-2772-4d44-8baa-47a2439f5de1" />
</p>


* 실제 배포 환경을 가정하여 20단계의 짧은 샘플링 과정(DDIM)에서 실험을 진행했습니다. 
* 독보적인 성능: LSUN Bedroom, FFHQ 등 다양한 데이터셋에서 기존 방식들보다 뛰어난 FID(이미지 품질) 및 sFID(다양성) 점수를 기록했습니다.
* 저비트에서의 안정성: 특히 가중치 4-bit, 활성화 6-bit(W4A6)라는 극한의 저비트 설정에서 다른 모델들은 성능이 급격히 무너졌으나, DMQ는 고품질 이미지를 안정적으로 생성했습니다.
* 비교 우위: EDA-DM과 같은 기존 최신 기술보다도 더 일관된 성능 향상을 보여주었습니다. 


#### 2. 클래스 조건부 이미지 생성 (Class-conditional Generation)

<p align = 'center'>
<img width="501" height="507" alt="image" src="https://github.com/user-attachments/assets/090acee8-175a-4adb-8073-4a93a0b21961" />
</p>

* ImageNet 데이터셋(256x256)을 대상으로 실험을 진행했습니다.
* 원본 모델에 가장 근접: LPIPS, SSIM, PSNR 지표에서 최고점을 기록하며, 양자화된 모델 중 원본(Full-precision) 모델의 결과물에 가장 가까운 이미지를 만들어냈습니다.
* 일관된 우수성: 모든 설정에서 기존 연구들을 일관되게 앞섰으며, 특히 W4A6 설정에서 성능 격차가 가장 두드러졌습니다. 

#### 3. 텍스트 가이드 이미지 생성 (Text-guided Generation)

<p align = 'center'>
<img width="500" height="254" alt="image" src="https://github.com/user-attachments/assets/80016300-1d3e-472d-8263-3387154a8f3f" />
</p>

* 가장 대중적인 Stable Diffusion v1.4를 사용하여 MS-COCO 프롬프트로 512x512 해상도 이미지를 생성했습니다.
* 의미적 일치(CLIP score): CLIP 점수에서 가장 높은 성적을 거두어, 생성된 이미지가 입력된 텍스트의 의미를 가장 정확하게 반영하고 있음을 입증했습니다.
* 시각적 품질: 구조적 유사성(SSIM)과 지각적 유사성(LPIPS) 모두에서 최고 성능을 보여주며 텍스트 기반 생성에서도 그 강력함을 증명했습니다.


### 5.2. Ablation study


#### 1. 구성 요소별 기여도 (Table 5)

<p align = 'center'>
<img width="496" height="195" alt="image" src="https://github.com/user-attachments/assets/9cbdeb91-69b3-4dbf-9bbb-cc2a431db08c" />
</p>

* DMQ의 핵심 기술들을 하나씩 추가했을 때 지표가 어떻게 개선되는지 보여줍니다.
* Learned Equivalent Scaling (LES): 기본 베이스라인에 LES를 추가하면 아웃라이어로 인한 양자화 난이도가 재분배되어 성능이 크게 향상됩니다.
* Adaptive Timestep Weighting: 여기에 적응형 가중치를 더하면, 오차가 작지만 최종 품질에 결정적인 초기 역확산 단계의 중요도를 반영하여 성능이 한층 더 강화됩니다.
* Power-of-Two Scaling (PTS): 마지막으로 분산이 큰 스킵 연결 레이어에 PTS를 적용했을 때 가장 최적의 결과(FID 30.37)를 얻었습니다. 흥미롭게도 최종 결과는 원본 부동소수점 모델(FID 31.34)보다도 수치상 우수한 지표를 보였습니다.


#### 2. 타임스텝 가중치 전략 비교 (Table 6)

<p align = 'center'>
<img width="491" height="194" alt="image" src="https://github.com/user-attachments/assets/aa71f872-37ac-40e3-8c6c-f4f7f9cd59a2" />
</p>

* 단순한 수치적 오차(MSE)를 줄이는 것보다 '어떤 단계'를 중시하느냐가 핵심임을 증명합니다.
* 고정형 가중치(Linear/Quadratic): 초기 단계에 무조건 높은 점수를 주는 선형(Linear)이나 이차(Quadratic) 방식은 오히려 균일(Uniform) 방식보다 성능이 낮았습니다.
* 이유: 레이어마다 오차 발생 패턴이 다른데, 고정된 가중치를 쓰면 특정 레이어의 중요한 후반부 단계를 놓칠 수 있기 때문입니다.
* 적응형 전략(Adaptive): DMQ의 방식은 각 단계의 누적 오차를 동적으로 관찰하여 가중치를 조절하므로 가장 뛰어난 성능을 보입니다.

#### 3. PTS 스케일링 및 투표 알고리즘 분석 (Table 7)

<p align = 'center'>
<img width="496" height="191" alt="image" src="https://github.com/user-attachments/assets/c590dbc3-53cc-4517-8a32-6c7b1fcbc383" />
</p>

* PTS를 '어떻게', '어디에' 적용할지에 대한 분석입니다.
* 투표 알고리즘의 승리: 단순히 오차(MSE)를 최소화하는 인자를 고르면 보정 데이터에 과적합(Overfitting)되어 실제 성능이 떨어집니다. 반면, DMQ의 투표 방식은 여러 샘플의 합의를 이끌어내어 훨씬 강건한 결과를 만듭니다.
* 타겟 레이어 선정: PTS를 모든 레이어에 적용하는 것보다, 아웃라이어 문제가 심각한 스킵 연결(Skip connection) 레이어에만 집중적으로 적용하는 것이 연산 효율과 성능 면에서 더 효과적이었습니다.


---

## 6. Conclusion

### 1. 연구의 핵심 요약

* LES & 적응형 가중치: 학습된 등가 스케일링(LES)과 적응형 타임스텝 가중치를 결합하여 가중치와 활성화 값 사이의 양자화 난이도를 최적으로 재분배하고, 이미지 품질에 결정적인 영향을 미치는 타임스텝을 우선적으로 보호했습니다.
* PTS & 투표 알고리즘: 일반적인 방식으로는 해결되지 않는 스킵 연결(Skip connection)의 극심한 아웃라이어를 처리하기 위해 2의 거듭제곱 스케일링(PTS)과 강건한 투표 알고리즘을 도입했습니다. 

### 2. 실험적 성과 및 의의

* 성능 우위: 다양한 데이터셋과 생성 작업에 걸친 광범위한 실험을 통해, DMQ가 기존의 사후 양자화(PTQ) 기술들보다 일관되게 우수한 성능을 보임을 증명했습니다.
* 저비트 한계 돌파: 특히 기존 방식들이 대부분 실패했던 W4A6(가중치 4비트, 활성화 6비트) 와 같은 초저비트 환경에서도 높은 이미지 품질을 유지하며 모델을 안정적으로 양자화할 수 있음을 보여주었습니다.


---
