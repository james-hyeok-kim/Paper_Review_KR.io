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
    * 현재 값($Y_t$)의 부호와 이전 단계($Y_{t-1}$)의 부호는 대부분 일치
    * 실험 결과 Stable Diffusion 모델에서 이 가정은 99.59%의 확률로 일치하는 것
    * 복잡한 ReLU 연산을 1비트 부호 비트를 이용한 마스킹(AND 연산)으로 대체

$$ReLU(Y_t) = Y_t \cdot sgn(Y_t)$$

$$\Delta Y'_t = ReLU(Y_t) - ReLU(Y_{t-1})$$

$$\Delta Y'_t \approx \Delta Y_t \cdot sgn(Y_{t-1})$$

* Sign-Mask 주요 작동 단계 (5단계)
    * 가중치 로드: 해당 레이어의 가중치( $W$ )를 온칩 버퍼로 읽어옵니다.
    * 미분 연산: PE 배열이 온칩에 남아있던 이전 레이어의 델타값( $\Delta X_t$ )과 가중치를 곱해 출력 델타( $\Delta Y_t$ )를 계산합니다.
    * 부호 비트 로드: DRAM에서 이전 단계의 1비트 부호( $Sgn_{t-1}$ )만 읽어옵니다.
    * 로그 기록: 현재의 부호( $Sgn_t$ )를 업데이트하기 위해 결과값을 DRAM에 전송합니다.마스킹 처리: SFU 유닛에서 델타값과 부호 비트를 결합(Masking)하여 최종 활성화 델타( $\Delta Y'_t$ )를 생성합니다.

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

## 3. Motivation

### 1. 성능 저하의 원인: 메모리 트래픽 폭증

* 역설적인 결과: 미분 연산은 계산 비트 수를 줄여 연산 시간을 단축시키지만, 실제로는 5.78배 더 많은 메모리 트래픽을 발생시켜 결과적으로 23.4%의 성능 손실을 초래합니다.
* 워크로드의 특성: 기존 연구인 Diffy 에서는 메모리가 주요 병목이 아니었으나, 디퓨전 모델은 메모리 집약적인(memory intensive) 특성을 가지기 때문에 이 문제가 두드러집니다.
  
### 2. 비선형 활성화 함수의 제약

* 선형성 위반: ReLU와 같은 활성화 함수는 비선형적이기 때문에 $f(Y + \Delta Y) \neq f(Y) + f(\Delta Y)$입니다.
* 단편적인 연산: 이로 인해 기존 방식은 합성곱(Convolution) 레이어에서만 미분 연산을 수행하고, 활성화 함수 전후로 데이터를 다시 원본 값(Raw value)으로 변환해야 하는 파편화된(fragmented) 구조를 가집니다.

### 3. 데이터 변환 단계의 오버헤드 (Fig. 6a 분석)

<p align = 'center'>
<img width="421" height="323" alt="image" src="https://github.com/user-attachments/assets/ca968506-e7f4-4f64-a4b9-7c262e5b483d" />
</p>

* 기존 방식(Diffy 방식)이 레이어마다 거치는 복잡한 단계들이 메모리 병목을 만듭니다
* 델타 생성 (단계 2-4): 미분 연산을 준비하기 위해 DRAM에서 이전 단계의 활성화 값( $X_{t-1}$ )을 읽어와 현재 값과의 차이( $\Delta X_t$ )를 계산하고, 현재 값을 다시 DRAM에 기록해야 합니다.
* 원본 복구 (단계 6-8): 미분 합성곱 결과( $\Delta Y_t$ )를 비선형 활성화 함수에 넣기 위해 다시 원본 값( $Y_t$ )으로 변환하는 과정이 필요합니다.
* 온칩 저장 불가: 전체 U-Net 패스의 활성화 데이터 크기는 약 1.1GB에 달해 칩 내부에 저장하는 것이 불가능하며, 반드시 외부 DRAM을 거쳐야만 합니다.

### 4. 결론 및 목표

* 해결 과제: 디퓨전 모델을 효율적으로 가속하려면 비선형 활성화 함수조차도 미분 방식으로 계산할 수 있는 아키텍처가 필요합니다.
* 제안 방향: Cambricon-D는 메모리 오버헤드 없이 네트워크 전체에서 미분 연산을 유지하는 'Sign-Mask Dataflow'를 도입하여 이 문제를 해결하고자 합니다.

---

## 4. Design

### 4.A. Sign-Mask 데이터플로우 (Sign-Mask Dataflow)

* 기존 방식이 레이어마다 무거운 원본 데이터를 읽어 병목을 일으켰던 문제를 해결하기 위한 혁신적인 데이터 흐름입니다.
* 비선형성의 한계 극복: ReLU 함수는 $ReLU(Y) = Y \cdot sgn(Y)$로 정의됩니다. 타임스텝 간의 차이( $\Delta Y_t$ )가 매우 작기 때문에 $sgn(Y_t) \approx sgn(Y_{t-1})$이라는 근사치를 사용합니다.
* 1비트 부호 비트 활용: 16비트의 원본 활성화 텐서 대신, DRAM에서 1비트 부호 비트( $sgn(Y)$ )만 읽어옵니다. 이를 통해 메모리 트래픽을 획기적으로 절감합니다.
* 연산의 단순화: 활성화 함수 계산이 특수 함수 유닛(SFU)에서 단순한 AND 마스킹 연산으로 변환됩니다.
* 전체 네트워크 미분 유지: 레이어마다 원본 값으로 복구할 필요가 없으므로, 여러 레이어에 걸쳐 미분 모드를 유지하는 Full-network differential method를 구현합니다.

### 4.B. PE 설계 개선 (Further Improvements on PE design)

#### 1. 특이값(Outlier) 문제 해결

* 롱테일 분포: 디퓨전 모델의 델타 값은 대부분 작지만(Inliers), 범위를 벗어나는 특이값(Outliers)이 존재하며(약 1.86%), 이를 무시하면 정밀도가 크게 떨어집니다.
* Outlier-Aware 설계: 일반적인 값은 저정밀도(INT3)로, 특이값은 고정밀도(FP16)로 처리하는 방식을 채택합니다.

#### 2. 하드웨어-소프트웨어 협력 설계 (Co-design)

* 구조적 제약 도입: 각 연산 그룹 내에서 처리할 수 있는 최대 특이값 개수( $m$ )를 고정합니다. 제한을 초과하는 특이값은 클리핑(clipping)하여 일반 값으로 처리합니다.
* 동기화 오버헤드 제거: 특이값 계산을 정렬된 그룹 내로 제한함으로써, 일반 연산과 특이값 연산이 완전 동기화(lock-step)되어 실행되도록 하여 하드웨어 복잡도를 낮췄습니다.


---

## 5. Implementation Details

<p align = 'cetner'>
<img width="346" height="267" alt="image" src="https://github.com/user-attachments/assets/73247aa6-a311-4559-b282-19baa7fe0ff9" />
</p>

### 5.A. 부호 비트 관리 및 액세스 (Accessing and Maintaining the Sign Bit)

* Sign-mask 데이터플로우를 구현할 때 가장 큰 과제는 1비트 단위의 데이터를 효율적으로 읽어오는 것입니다.
* 별도 텐서 저장: DRAM에서 부호 비트(sign bits)만 효율적으로 읽을 수 있도록, 각 원본 입력 텐서에 대응하는 부호 비트들만 모아 별도의 메모리 위치에 텐서 형태로 관리합니다.
* NDP(Near-Data-Processing) 기술: 원본 데이터가 변경될 때마다 부호 비트를 업데이트해야 하는데, 이 과정에서 발생하는 데이터 전송 오버헤드를 막기 위해 DRAM 장치 근처에 전용 회로(NDP 엔진)를 배치했습니다.
* 델타 압축 전송(Compression): 업데이트 시 델타( $\Delta$ ) 값만 메모리 인터페이스로 전송하며, 델타값은 수치 범위가 좁아 압축이 용이하므로 전송량을 더욱 줄일 수 있습니다. 

### 5.B. 기타 연산자의 미분 계산 (Miscellaneous Operators)

* 활성화 함수 외에도 선형적이지 않아 미분 가속을 방해하는 다른 연산자들을 처리하는 방법입니다. 

#### 1) 그룹 정규화 (Group Normalization, GN)

* 비선형성 문제: GN에서 사용하는 평균( $\mu$ )과 분산( $\sigma^2$ )은 입력값에 따라 매번 변하기 때문에 비선형적입니다.
* 해결책: 각 타임스텝의 통계치를 독립적으로 계산하되, 인접한 두 타임스텝의 평균값을 미분 연산에 사용합니다. 결과: 타임스텝 간 통계치 변화가 미미하기 때문에 이 방식을 적용해도 모델 정밀도 하락이 측정되지 않았습니다.

#### 2) 어텐션 메커니즘 (Attention Mechanism)

* 복잡한 비선형성: 어텐션은 행렬 곱셈(Bilinear)과 Softmax 함수를 포함하고 있어 미분 형태로 표현하기 매우 까다롭습니다.
* 우회 전략: 프로파일링 결과 어텐션이 전체 연산 시간에서 차지하는 비중은 단 0.9%에 불과합니다.
* 설계 단순화: 따라서 어텐션 부분은 복잡하게 설계하는 대신, 단순히 외부 메모리에서 원본 데이터를 읽어와 일반적인 방식으로 계산하도록 하여 아키텍처 복잡도를 최소화했습니다. 

#### 3) 기타 선형 연산자

* 잔차 연결(Residual connections), 드롭아웃(Dropouts), 업샘플링 및 다운샘플링은 선형 변환 조건을 만족하므로 문제없이 미분 연산으로 처리됩니다.
    * 선형변환 $f(ax + by) = af(x) + bf(y)$
    * Residual Connection 은 더하기(Addition)
    * Dropout은 Mask 연산, 미분값으로 0(정보차단), 1(정보전달) 로 mapping 가능   


---

## 6. Architecture

<p align = 'cetner'>
<img width="346" height="267" alt="image" src="https://github.com/user-attachments/assets/73247aa6-a311-4559-b282-19baa7fe0ff9" />
</p>

### 6.A. 아키텍처 개요 (Overview)

* PE Array (Processing Element Array): 가중치와 델타 입력을 곱하는 텐서 연산을 수행하여 다음 레이어를 위한 델타 출력을 생성합니다.
* SFU (Special Function Unit): 원본 데이터를 가져오지 않고도 ReLU와 같은 비선형 함수의 미분 연산을 실행합니다.
* Compression Unit: DRAM에 쓰기 전 델타 값을 압축하여 데이터 전송량을 줄입니다.
* On-chip Buffers: 가중치(32MB), 입력(64MB), 출력(64MB)을 위한 전용 버퍼를 갖추고 있습니다.
* NDP Engine: DRAM 쪽에 위치하여 델타 값의 압축을 해제하고, 원본 값을 업데이트하며, 부호 비트(sign bits)를 관리합니다.

### 6.B. 상세 데이터플로우 (Detailed Dataflow)

* 가중치(Weight): DRAM에서 온칩 버퍼로 로드되어 PE 배열로 전달됩니다 (Wide/Raw values).
* 델타 입력(Input): 이전 레이어의 ReLU 결과로 생성되어 PE 배열에서 연산에 사용됩니다 (Narrower delta values).
* 델타 출력(Output): PE 배열 연산 결과로, 출력 버퍼에 저장되었다가 SFU로 보내져 ReLU 처리를 거칩니다.
    * 비선형함수에 그대로 넣으면 다른 결과나온다
    * SFU는 Differential Non-linear 유닛 (Sign-Masking)
    * DRAM에서 이전 단계 원본 데이터의 1비트 부호 비트( $sgn(Y_{t-1})$ )를 가져옵니다.
    * 이 부호가 양수(+)면, 활성화된 상태이므로 입력된 델타( $\Delta Y_t$ )를 그대로 통과시킵니다.
    * 이 부호가 음수(-)면, ReLU에 의해 차단된 상태이므로 델타 값을 0으로 마스킹(Masking)해버립니다.
* 부호 비트(Sign bits): DRAM에서 SFU로 직접 로드되어 ReLU 마스킹 연산에 사용됩니다.
* 증분값(Increments): 원본 활성화 값을 업데이트하기 위해 압축 유닛으로 전달됩니다.
* 압축된 증분값: 메모리로 전송되어 최종적으로 원본 데이터를 갱신합니다.

### 6.C. PE 배열 설계 (PE Array)

<p align = 'center'>
<img width="543" height="233" alt="image" src="https://github.com/user-attachments/assets/793b48f2-ce7b-47ac-97b1-eb299761eb55" />
</p>

* Inlier Seg (일반 값 세그먼트): 대다수의 일반적인 델타 값을 처리하기 위해 다수의 int-and-fp 멀티플라이어를 배치했습니다.
* Outlier Seg (특이값 세그먼트): 범위를 벗어나는 소수의 특이값을 처리하기 위해 소수의 fp-and-fp 멀티플라이어를 포함합니다.
* 동작 방식: 입력 데이터가 들어오면 fp2int 모듈이 양자화를 시도하고, 실패 시 오버플로우 플래그(OF)를 발생시켜 특이값 선택 회로가 이를 고정밀도 연산기로 보냅니다.

### 6.D. ReLU 계산 및 부호 비트 유지 (Computing ReLU)

* SFU와 NDP 엔진이 협력하여 비선형 함수를 처리합니다
* SFU의 마스킹: 출력 버퍼에서 델타 값을 읽어와 대응하는 부호 비트가 0인 값을 마스킹(제거)하여 ReLU 결과를 냅니다.
* NDP 엔진의 업데이트: int2fp 변환과 FP Adder 배열을 통해 DRAM 내부에서 직접 원본 값을 읽고-더하고-쓰는(Read-Add-Write) 과정을 거쳐 데이터를 최신 상태로 유지합니다.

### 6.E. Convolution Compute Algorithm

<p align = 'center'>
<img width="427" height="444" alt="image" src="https://github.com/user-attachments/assets/8cab0e67-f297-418a-9c56-e4270b0b705e" />
</p>

#### 1. 루프 구조 및 타일링 (Line 1-3)

* 병렬 처리: 출력 채널( $d_1$ ), 배치 및 출력 해상도( $d_2$ )를 기준으로 병렬 처리가 이루어집니다.
* 타일 분할: 대용량 데이터를 처리하기 위해 입력을 특정 크기의 타일(Tile) 단위로 나누어 루프를 돕니다.

#### 2. 데이터 로드 (Line 4-5)

* 입력 데이터 로드: 입력 버퍼에서 미분 입력값( $Tile_{in}$, 델타값)을 읽어옵니다.
* 가중치 로드: 가중치 버퍼에서 원본 가중치( $Tile_{w}$ )를 읽어옵니다.

#### 3. PE 배열 연산 (Line 6-12)

* 부분합 초기화: 출력 타일을 0으로 초기화합니다.
* 미분 합성곱 수행: PE 배열이 델타 입력과 원본 가중치를 곱하여 부분합( $Tile_{partial}$ )을 계산합니다.
* 누적: 타일별 연산 결과를 출력 타일에 지속적으로 더해 최종 델타 출력값을 완성합니다.

#### 4. 출력 및 SFU 활성화 (Line 13-14)

* 결과 쓰기: 계산된 델타 출력값을 출력 버퍼(OutputBuf)에 기록합니다.
* SFUActivation: 가장 핵심적인 부분으로, SFU가 출력 버퍼에 접근하여 Sign-mask 기반의 비선형 활성화 함수(ReLU 등)를 처리합니다. 이 단계에서 델타값은 다음 레이어의 델타 입력으로 사용될 준비를 마칩니다.


### 6.F. CornerCase

#### 부호 변화 시 연산 단계 (Step-by-Step)

1) 이전 단계 부호 확인 (Sign Bit Fetch)
    1) SFU는 DRAM에서 이전 타임스텝( $t-1$ )의 원본 값에 대한 부호 비트($Sgn_{t-1}$ )를 읽어옵니다.
    2) 이 시점에서 $Sgn_{t-1}$은 양수(+)인 상태입니다.
2) 부호 일치 가정 (Approximation)
    1) 논문은 $\Delta Y_t$가 매우 작기 때문에 $sgn(Y_t) \approx sgn(Y_{t-1})$이라고 가정합니다.
    2) 즉, 실제 원본 값이 음수로 변했더라도, 하드웨어는 일단 이전 단계의 부호인 양수(+)를 기준으로 판단합니다.
3) 델타 마스킹 (Masking/Passing)
    1) $Sgn_{t-1}$이 양수(+)이므로, SFU는 ReLU가 '열려 있는(Pass)' 상태라고 판단하여 PE 배열에서 계산된 델타 출력( $\Delta Y_t$ )을 그대로 통과시킵니다.
    2) 이 결과값이 최종 미분 출력 $\Delta Y'_t$가 되어 다음 레이어로 전달됩니다.
4) DRAM 원본 업데이트 (NDP Update)
    1) 동시에 계산된 델타 값( $\Delta Y_t$ )은 메모리 측의 NDP 엔진으로 전송됩니다.
    2) NDP 엔진 내부의 FP Adder가 실제 $Y_{t-1} + \Delta Y_t$를 계산하여 원본 값을 업데이트합니다.
    3) 이 과정에서 원본 값은 실제로 양수에서 음수로 변하게 됩니다.
5) 부호 비트 갱신 (Sign Bit Update)
    1) NDP 엔진은 업데이트된 원본 값의 새로운 부호(음수)를 확인하고, DRAM에 저장된 부호 비트 텐서를 갱신합니다.
    2) 이 갱신된 부호는 다음 타임스텝( $t+1$ ) 연산에서 $Sgn_t$로 사용되어, 그때부터는 해당 위치의 델타를 0으로 마스킹(차단)하게 됩니다.

6) 부호가 양수(+)에서 음수(-)로 변하는 실제 시나리오 예
    1) 이전 단계 원본 ( $Y_{t-1}$ ): $+0.5$ (양수)
    2) 현재 타임스텝의 변화량 ($\Delta Y_t$): $-0.7$ (음수)
    3) 현재 단계 원본 ($Y_t = Y_{t-1} + \Delta Y_t$): $+0.5 + (-0.7) = \mathbf{-0.2}$ (음수)
    4) 이 경우, 실제 값은 양수에서 음수로 바뀌었지만(Zero-crossing), Cambricon-D의 SFU는 해당 단계에서 이전 부호인 '양수(+)'를 사용하여 연산을 수행하게 됩니다.
7) 왜 이렇게 처리해도 괜찮나요?
    1) 논문은 이 미세한 오차가 치명적이지 않다는 점을 강조합니다
    2) 델타 값( $\Delta Y_t$ ) 자체가 매우 작기 때문에, 부호가 바뀌는 경계선 근처의 값들은 원래 0에 매우 가깝습니다.
    3) 낮은 확률: 부호가 일치하지 않을 확률은 Stable Diffusion 기준 0.41%에 불과합니다.
    4) 자기 교정: NDP 엔진이 다음 단계를 위해 DRAM의 부호 비트를 실제 결과값(-0.2의 부호인 '-')으로 즉시 갱신하므로, 오차는 해당 타임스텝에서만 일시적으로 발생하고 사라집니다.
  

---

## 7. Methodology

### 7.A. 하드웨어 구성 (Hardware Configurations)

* 논문은 공정한 비교를 위해 Cambricon-D의 처리량을 실제 NVIDIA A100 GPU 수준으로 맞추어 설계했습니다.
* Cambricon-D 사양: 128x128 크기의 PE 배열을 사용하며, 1GHz 클럭에서 작동합니다.
* 기술 노드: TSMC 45nm 기술로 설계한 후, 도구를 사용하여 7nm 공정 데이터로 스케일링하여 평가했습니다.
* PE 내부 구성: 각 PE는 60개의 int3-and-fp16 멀티플라이어(Inliers용)와 4개의 fp-and-fp16 멀티플라이어(Outliers용)를 가집니다.
* 메모리 대역폭: A100과 동일한 1.5 TB/s의 메모리 대역폭을 할당했습니다.

### B. 비교 대상 (Baseline & Alternatives)

* Baseline (Systolic): NVIDIA A100의 성능을 대변하는 TPU 스타일의 시스톨릭 배열(Systolic Array) 시뮬레이터입니다.
* DiffyDF (Diffy-Dataflow): Sign-mask 데이터플로우 없이, 레이어마다 원본 값으로 복구하는 기존 Diffy 방식의 데이터 흐름을 사용한 모델입니다.
* DiffyPE (Diffy-PE): 특이값(Outlier) 처리가 없는 Diffy 스타일의 PE를 사용한 모델입니다.
* AsyncPE: 특이값을 처리하기 위해 별도의 비동기식 희소(Sparse) PE 배열을 사용하는 모델입니다.
* DiffyAll: 데이터플로우와 PE 설계 모두 기존 Diffy 방식을 따른 모델입니다.

### C. 벤치마크 모델 (Benchmarks)

* 가장 대중적이고 성능이 검증된 두 가지 디퓨전 모델을 사용했습니다.
* Guided-Diffusion (GUID)
    * 모델 특성: 약 0.4B~0.5B(4억~5억 개)의 파라미터를 가진 조건부 DDPM 모델입니다.
    * 테스트 환경: 다양한 해상도(128x128, 256x256, 512x512)에서 테스트
    * 데이터셋: 침실(bedroom), 고양이(cat) 이미지 데이터셋(LSUN) 및 ImageNet을 사용했습니다.
* Stable Diffusion v1.4 (STBL)
    * 모델 특성: 디퓨전 구성 요소만 약 0.86B(8.6억 개)의 파라미터를 가진 대규모 모델입니다.
    * 테스트 환경: 주로 512x512 해상도에서 테스트되었습니다.
    * 데이터셋: 이미지와 설명글이 포함된 Conceptual Captions 데이터셋을 활용했습니다.


### D. 평가 지표

* 모델 정확도: 생성된 이미지의 품질을 측정하기 위해 Precision과 SSIM(구조적 유사도 지수)을 측정했습니다.
* 성능 및 효율: 속도 향상(Speedup), 메모리 트래픽(Memory Traffic), 에너지 소비량(Energy), 하드웨어 면적(Area)을 종합적으로 평가했습니다.

---

## 8. Experimental Results

### 8.1. 모델 정확도 (Model Accuracy)

<p align = 'center'>
<img width="442" height="203" alt="image" src="https://github.com/user-attachments/assets/97694f6e-e279-4e02-a12f-725a154065ff" />
</p>


* 제안된 미분 양자화 스킴이 이미지 품질에 미치는 영향을 평가했습니다. 
* 정밀도 유지: Cambricon-D의 양자화 방식은 FP16 대비 오차가 0~4% 이내로 매우 적었습니다. 
* 시각적 품질: 생성된 이미지의 SSIM(구조적 유사도 지수)은 약 0.9650으로 측정되었습니다. 일반적으로 0.95 이상이면 인간이 시각적으로 차이를 느끼지 못하는 수준입니다.
* 기존 방식(Diffy)과의 비교: 특이값(Outlier) 처리가 없는 기존 Diffy 방식은 디퓨전 모델에서 심각한 정밀도 하락을 겪는 것으로 나타났습니다. 

### 8.2. 속도 향상 (Speedup)

<p align = 'center'>
<img width="442" height="313" alt="image" src="https://github.com/user-attachments/assets/9e8d7df3-832a-4243-908d-7cead2f4eb65" />
</p>

* 기본 시스톨릭 배열(Baseline) 및 NVIDIA A100 GPU와 비교한 결과입니다. 
* 성능 향상: Cambricon-D는 Baseline 대비 Guided-Diffusion에서 1.46배 ~ 2.38배, Stable Diffusion에서 2.38배의 속도 향상을 달성했습니다.
* 데이터플로우의 중요성: Sign-mask 데이터플로우가 없는 모델(DiffyDF)은 메모리 병목 때문에 오히려 Baseline보다 성능이 60~77% 하락했습니다. 이는 Sign-mask 기술이 미분 가속의 핵심임을 증명합니다.
* 물리적 GPU와 비교: 물리적인 A100 GPU와 비교했을 때도 전용 가속기인 Cambricon-D가 월등히 빠른 성능을 보였습니다. 

### 8.3. 메모리 및 에너지 효율

* 메모리 트래픽: Sign-mask 데이터플로우 덕분에 메모리 액세스량이 기존 미분 가속 방식 대비(Diffy) 66% ~ 82% 절감되었습니다.

<p align = 'center'>
<img width="458" height="296" alt="image" src="https://github.com/user-attachments/assets/4d86e9e0-4beb-4702-95a9-c3179c959be5" />
</p>

* 에너지 소비: Guided-Diffusion 벤치마크 기준, Cambricon-D는 Baseline보다 25.81% ~ 42.16% 적은 에너지를 사용했습니다. 이는 연산 속도 향상으로 인한 동적 전력 감소 덕분입니다. 

### 8.4. 하드웨어 특성 (Area & Power)

<p align = 'center'>
<img width="403" height="226" alt="image" src="https://github.com/user-attachments/assets/7d1b22e6-b071-4411-a65c-8d06824413b3" />
</p>

* 면적 오버헤드: 미분 연산 및 특이값 처리를 위한 추가 회로에도 불구하고, 기존 설계 대비 하드웨어 면적 증가는 단 3.6%에 불과했습니다.
* 칩 사양: 7nm 기술 노드 기준, 가속기 코어의 면적은 $16.24mm^2$, 예상 전력 소모는 73.47W입니다. 


---

## 10. Conclusion

### 10.1. 핵심 성과 및 제안

* 계산 중복성 제거: 디퓨전 모델의 반복적인 특성에서 기인하는 불필요한 연산을 미분 연산(differential computation)을 통해 효과적으로 줄였습니다.
* Full-Network 미분 구현: 기존 방식들과 달리, 비선형 레이어에서 연산이 끊기지 않고 네트워크 전체에서 미분 모드를 유지하여 메모리 오버헤드를 근본적으로 해결했습니다.
* Sign-Mask 데이터플로우: 1비트 부호 비트만을 활용해 비선형 활성화 함수를 처리하는 새로운 데이터 흐름을 제안하여 메모리 트래픽을 획기적으로 감소시켰습니다. 

### 10.2. 연구의 의의

* 효율적인 하드웨어 가속: Cambricon-D는 디퓨전 모델의 고유한 특성을 하드웨어 설계에 직접 반영함으로써 고성능 실행을 위한 유망한 솔루션을 제공합니다.
* 비선형성 문제 해결: 미분 연산의 최대 난제였던 비선형 활성화 함수 처리 문제를 해결하여 하드웨어 가속기 설계의 새로운 방향을 제시했습니다.

---
