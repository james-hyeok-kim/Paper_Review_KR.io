# Cambricon-D: Full-Network Differential Acceleration for Diffusion Models

저자 : Weihao Kong (제1저자), Qi Guo (교신 저자) ...

발표 : ISCA(International Symposium on Computer Architecture), 2024

논문 : [PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10609724)

---

## 1. Introduction

<p align = 'center'>
<img width="458" height="419" alt="image" src="https://github.com/user-attachments/assets/e982bfdb-0541-4554-9e53-d6a68422aaa7" />
</p>


### 1.1. 배경 및 문제점

* 디퓨전 모델의 비효율성: Stable Diffusion이나 OpenAI Sora와 같은 디퓨전 모델은 이미지 생성 과정에서 수많은 반복(iteration) 과정을 거칩니다. 각 단계(timestep)마다 전체 네트워크를 다시 계산해야 하므로 연산량과 하드웨어 비용이 매우 큽니다.
* 연산 중복성(Computational Redundancy): 인접한 단계 사이의 입력 데이터는 매우 유사하며, 실제로는 아주 작은 차이(변화량)만 발생합니다. 따라서 매번 전체를 새로 계산하는 것은 심각한 낭비입니다.

### 1.2. 기존 해결책의 한계

* 미분 연산(Differential Computing): 입력값 자체가 아닌 이전 단계와의 차이값( $\Delta$, delta)만을 계산하여 비트수를 줄이고 연산 속도를 높이는 방식입니다.
* 성능 저하의 역설: 미분 연산을 적용하면 산술 연산 비용은 줄어들지만, 오히려 전체 성능은 약 23.4% 하락하는 현상이 발생합니다.
    * 메모리 트래픽이 5.78배 증가
    * 비선형 함수의 방해: ReLU와 같은 비선형 활성화 함수는 미분 값($\Delta$)만으로는 계산이 불가능합니다. 즉, $f(x + \Delta) \neq f(x) + f(\Delta)$이기 때문
    * 데이터의 반복 로드: 이전 단계의 무거운 원본 데이터(Raw Input)를 외부 메모리(DRAM)에서 다시 읽어와 델타값과 합치는 과정이 필요
    * DRAM 읽기 $\rightarrow$ 연산 $\rightarrow$ DRAM 쓰기
* 메모리 액세스 병목: ReLU와 같은 비선형 활성화 함수를 만날 때마다 정확한 계산을 위해 델타값과 원본 데이터를 합쳐야 합니다. 이 과정에서 무거운 원본 데이터를 메모리에서 반복적으로 읽어와야 하므로, 메모리 액세스량이 약 5.78배나 급증하게 됩니다.


### 1.3. Cambricon-D의 제안

* Sign-Mask 데이터플로우: 원본 데이터 전체를 읽는 대신 1비트 부호(Sign bit)만 로드하여 비선형 연산을 처리합니다. 이를 통해 전체 네트워크에서 미분 연산을 끊김 없이 유지하며(Full-network differential computing), 메모리 액세스를 66%~82% 절감합니다.
* Outlier-Aware PE 배열: 델타값 중 범위를 벗어나는 특이값(Outliers)을 효율적으로 처리하기 위한 하드웨어 설계입니다. 특이값 처리를 정형화하고 동기화하여 연산 효율을 높였습니다.


### 1.4. 실험 결과

* NVIDIA A100 GPU 대비 1.46배에서 2.38배의 속도 향상을 달성했습니다.추가되는 하드웨어 면적 오버헤드는 단 3.6% 수준입니다.

---

## 2. Background

### 2.A. Diffusion Models

#### 1. 기본 개념: 노이즈 제거 프로세스

* 순방향 확산(Forward Diffusion): 훈련 중에 이미지에 점진적으로 노이즈를 추가하여 완전한 가우시안 노이즈로 만드는 과정입니다.
* 역방향 노이즈 제거(Reverse Denoising): 추론(Inference) 단계에서 수행되며, 완전한 노이즈 상태에서 시작해 조금씩 노이즈를 제거하며 이미지를 생성해 나가는 과정입니다.
* 논문은 이 역방향 과정의 가속화에 집중하고 있습니다.

#### 2. 반복적 연산 특성 (Timesteps)

* 입력 데이터(이미지)는 타임스텝 간에 아주 조금씩만 변하기 때문에, 인접한 단계 사이의 입력값들은 매우 높은 유사성을 가집니다.

#### 3. 백본 네트워크: U-Net

* 주요 레이어: 잔차 연결(residual connection)이 포함된 합성곱 신경망(CNN)을 기반으로 하며, SiLU 활성화 함수, 그룹 정규화(Group Normalization), 드롭아웃, 업샘플링 및 다운샘플링 레이어로 구성됩니다.
* 추가 메커니즘: 특정 텍스트 프롬프트나 클래스에 맞춰 이미지를 생성하도록 안내하는 어텐션 메커니즘(Attention)이나 가이드 메커니즘이 포함되기도 합니다.
    * Attention
        * Middle Block: 인코더와 디코더 사이의 가장 깊은 병목(Bottleneck) 구간에서 글로벌한 문맥을 파악하기 위해 어텐션 레이어가 들어갑니다.
        * Upsampling / Downsampling Blocks: 해상도가 낮은 특정 레이어(예: $32 \times 32$ 또는 $16 \times 16$ 크기)들에 Transformer Block이 포함되어 있습니다.
* 연산 비중: 프로파일링 결과, 합성곱(Convolution) 연산이 전체 연산 시간의 약 76.4%를 차지하는 핵심 병목 구간입니다.이 모델의 핵심은 결국 "매우 비슷한 입력을 가지고 똑같은 모델을 수십 번 반복 계산한다"는 점이며, Cambricon-D는 바로 이 지점에서 발생하는 연산 중복을 제거하고자 합니다.

### 2.B. Differential Computing

#### 1. 핵심 원리: 계산 중복성(Computational Redundancy) 제거

* 입력의 유사성: 디퓨전 모델의 노이즈 제거 과정에서 각 타임스텝은 이미지를 아주 조금씩만 변화시킵니다.
* 델타( $\Delta$ ) 활용: 따라서 타임스텝 $t$의 활성화 값 $X_t$를 처음부터 다 계산하는 대신, 이전 단계 $X_{t-1}$과의 작은 차이인 $\Delta X_t$만 사용하여 $X_t = X_{t-1} + \Delta X_t$로 표현합니다.
* 연산량 감소: 이 $\Delta X_t$는 수치적 범위가 매우 작기 때문에(Fig. 1 참조), 정밀도 손실 없이 훨씬 적은 비트(예: INT3)로 표현이 가능하여 연산기 회로를 단순화하고 속도를 높일 수 있습니다.

#### 2. 선형 연산에서의 적용 (Convolution)

* 수학적 근거: 합성곱(Convolution)과 같은 선형 연산자는 $Conv(X_t) = Conv(X_{t-1} + \Delta X_t) = Conv(X_{t-1}) + Conv(\Delta X_t)$ 성질이 성립합니다.
* 가속 방법: 이미 알고 있는 이전 결과값 $Conv(X_{t-1})$에, 훨씬 적은 비트로 계산한 $Conv(\Delta X_t)$만 더하면 현재 결과값을 얻을 수 있습니다.
* 비유: $123 \times 7 = 861$임을 알 때, $124 \times 7$을 새로 계산하는 대신 차이인 $1 \times 7$을 구해 861에 더하는 것과 같습니다.

#### 3. 시간적 미분(Temporal Differential)의 선택

* 기존 방식(Diffy)과의 차이: 기존의 Diffy는 이미지 내 인접 픽셀 간의 차이를 이용하는 공간적 미분(Spatial Differential)을 사용했습니다.
* 디퓨전 모델의 특성: 하지만 디퓨전 모델의 중간 데이터는 노이즈 성분이 많아 공간적으로는 매끄럽지 않습니다.
* 효율성: 따라서 이 논문은 타임스텝 간의 차이를 이용하는 시간적 미분이 델타 값을 더 작고 집중되게 만들어(Fig. 5 참조) 훨씬 효율적임을 밝혀냈습니다.

---


---


