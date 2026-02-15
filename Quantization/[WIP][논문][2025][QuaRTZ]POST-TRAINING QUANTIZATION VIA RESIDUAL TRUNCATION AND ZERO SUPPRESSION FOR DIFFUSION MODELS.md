# POST-TRAINING QUANTIZATION VIA RESIDUAL TRUNCATION AND ZERO SUPPRESSION FOR DIFFUSION MODELS

저자 : 

Donghoon Kim

Department of Artificial Intelligence

Kyung Hee University

Yongin-si, Gyeonggi-do, South Korea

dhkim2810@khu.ac.kr

Dongyoung Lee

Department of Electrical Engineering

Kyung Hee University

Yongin-si, Gyeonggi-do, South Korea

dylee@khu.ac.kr

Ik Joon Chang

Department of Electrical Engineering

Kyung Hee University

Yongin-si, Gyeonggi-do, South Korea

ichang@khu.ac.kr

Sung-Ho Bae

Department of Computer Science

Yongin-si, Gyeonggi-do, South Korea

shbae@khu.ac.kr

발표 : 2025년 9월 30일, arXiv

논문 : [PDF](https://arxiv.org/pdf/2509.26436)

---

## 1. Introduction

<p align = 'center'>
<img width="748" height="266" alt="image" src="https://github.com/user-attachments/assets/aade2dbf-2acb-4edf-99a0-d8191349a124" />
</p>

### 1. 배경 및 문제 의식

1) 확산 모델의 한계
    1) 확산 모델은 텍스트-이미지 생성에서 뛰어난 성능
    2) 반복적인 정제 과정(iterative refinement process)으로 인해 연산 비용 높음
    3) 리소스가 제한적인 환경에서의 배포가 어렵다

2) 양자화(Quantization)의 필요성
    1) 모델의 메모리 사용량을 줄이고 속도를 높이기 위해 PTQ
    2) 8비트나 6비트 수준에서는 효과가 입증되었으나, 4비트(W4A4) 정밀도로 낮출 경우 라운딩 에러(rounding error)가 누적되어 이미지의 텍스처 품질이 심각하게 저하되는 문제가 발생합니다.

### 2. 기존 접근법의 한계 (Outliers vs. LSBs)

1) 기존 연구의 초점 (Outliers): 이상치(Outliers)를 보존하는 데 집중

2) 저자들의 핵심 발견 (LSBs): 저자들은 기존 방식이 최하위 비트(LSBs, Least Significant Bits)의 손실을 간과하고 있다고 지적
    1) 확산 모델은 0 근처에 밀집된 작은 값들의 미세한 변화를 통해 텍스처와 그라디언트를 형성합니다.
    2) LSB를 잘라내면 미세한 정보가 사라져 이미지가 뭉개지거나 생성이 실패하게 됩니다.

### 3. 제안하는 해결책 (QuaRTZ)

QuaRTZ (Quantization via Residual Truncation and Zero suppression)

1) 1단계 (8-bit Quantization)
    1) Outlier 보존 및 Rounding error 최소화 (step size를 유지해 라운딩 에러를 최소화)

2) 2단계 (4-bit Compression)
    1) 이후 LZS (Leading Zero Suppression) 기술을 사용하여 불필요한 상위 0 비트들을 제거하고, 정보가 담긴 유효한 비트들만 남겨 4비트로 압축합니다.

3) 효과
    1) 이 방식은 이상치(Outlier)의 크기 정보와 LSB의 정밀도(Fine-grained details)를 동시에 보존할 수 있습니다.

### 4. 주요 기여 및 성과

1) 성능: FLUX.1-schnell 모델에서 4비트 QuaRTZ를 적용했을 때 FID 6.98 기록

2) 효율성: 기존의 고성능 방식인 SVDQuant가 추가적인 16비트(FP16) 브랜치를 필요로 했던 것과 달리, QuaRTZ는 보조 브랜치 없이도 더 나은 성능을 내며 메모리 사용량을 16비트 대비 3.8배 줄였습니다.

3) 범용성: UNet 및 DiT(Diffusion Transformer) 기반의 다양한 아키텍처에서 최첨단 성능(SOTA)을 입증했습니다.

---

## 2. RELATED WORKS

### 1. 확산 모델의 발전과 한계

1) 확산 모델 성능 (다양한 분야에서 SOTA(State-of-the-art))
    1) 텍스트-이미지 생성(Text-to-Image)
    2) 초해상도(Super-resolution)
    3) 인페인팅(Inpainting)

2) 구조: 최근에는 트랜스포머(Transformer) 백본을 통합하여 모델의 용량과 제어 능력을 확장하는 추세입니다.



### 2. 양자화(Quantization) 접근법: QAT vs. PTQ

1) QAT (Quantization-Aware Training)

2) PTQ (Post-Training Quantization)

### 3. 기존 PTQ 연구들의 초점: 이상치(Outliers)

1) 기존의 PTQ 연구들은 주로 이상치(Outliers) 처리에 집중해 왔습니다.
2) 대표적인 방법: PTQ4DM, Q-Diffusion, TFQM-DM 등의 연구는 시간적 정렬(Temporal alignment)이나 조건부 스케일링(Condition-aware scaling) 등을 통해 값이 큰 이상치로 인한 오차를 줄이려 했습니다.
3) 최신 연구: 최근 DGQ는 텍스트-이미지 모델을 W4A6(가중치 4비트, 활성화 6비트)로 양자화했고, SVDQuant는 4비트 정밀도에 도달했습니다.

### 4. 기존 연구의 한계와 QuaRTZ의 차별점

1) 6비트 이하의 품질 저하: 기존 연구들은 대부분 6비트 미만의 정밀도에서는 이미지 품질을 유지하는 데 실패했습니다.

2) SVDQuant의 비효율성: SVDQuant는 이미지 품질은 유지했지만, 이를 위해 16비트(FP16) 보조 브랜치(LoRA)를 사용하여 오차를 흡수하는 방식을 썼습니다.

3) 이는 추가적인 파라미터를 필요로 하고, 아키텍처를 수정해야 하며, 혼합 정밀도(Mixed-precision) 연산을 요구하므로 순수한 4비트 양자화의 효율성 이점을 갉아먹습니다.

4) QuaRTZ의 접근: 이에 반해, 본 논문의 QuaRTZ는 보조 브랜치 없이 순수 4비트만으로 미세한 텍스처 품질을 보존하는 것을 목표로 합니다. 기존 연구가 '이상치'에만 집중했던 것과 달리, QuaRTZ는 '이상치'와 '최하위 비트(LSB)'를 동시에 고려한다는 점에서 차별화됩니다.

---

## 3. QUANTIZATION VIA RESIDUAL TRUNCATION AND ZERO SUPPRESSION

<p align = 'center'>
<img width="704" height="367" alt="image" src="https://github.com/user-attachments/assets/a7b68d52-f495-49af-a9a9-12d296ed8ec9" />
</p>

### 1. 핵심 가설 (Hypothesis)

* 이상치(Outliers): 큰 값을 가지며, 이미지의 주요 특징(salient features)과 큰 폭의 수정(correction)을 담당합니다.

* 최하위 비트(LSBs): 0에 가까운 작은 값들의 미세한 변화를 나타내며, 텍스처(textures)와 부드러운 그라디언트(smooth gradients)를 형성하는 데 결정적입니다.

* 기존의 4비트 양자화는 이 LSB를 잃어버리기 때문에 이미지가 뭉개지는 현상이 발생했습니다.

### 2. 2단계 양자화 프로세스 (Two-Stage Quantization)

1) 1단계: 8비트 정수 양자화 (8-bit Integer Quantization)
    1) 이유: 4비트로 바로 변환하면 단계(step size)가 너무 커져서 오차가 크지만, 8비트는 단계가 촘촘하여 라운딩 에러(rounding error)를 최소화할 수 있습니다.
    2) 이 과정에서 데이터의 전체 범위(이상치 포함)를 표현할 수 있게 됩니다.

2) 2단계: 4비트 압축 (4-bit Compression via LZS)
    1) 8비트 데이터에는 불필요한 정보(Redundancy)가 많다는 점을 이용해 이를 4비트로 압축합니다. 특히 작은 값들은 앞부분(상위 비트)이 모두 0으로 채워져 있다는 점에 착안하여 LZS (Leading Zero Suppression) 기법을 사용합니다.
    2) 그룹화 (Grouping): 데이터를 작은 그룹(예: 16개 또는 32개 단위)으로 묶습니다.
    3) FLAG 계산: 각 그룹 내에서 가장 큰 값(가장 높은 활성 비트 위치)을 기준으로 FLAG를 계산합니다.
        1) FLAG는 해당 그룹의 값들이 공통적으로 가지고 있는 '앞쪽의 0(Leading Zeros)'의 개수를 기반으로 결정됩니다.
    4) 시프트 및 압축 (Shifting): 계산된 FLAG만큼 비트를 오른쪽으로 시프트(Right-shift)하여 4비트 공간에 밀어 넣습니다.
        1) 작은 값 (대부분의 경우): 앞쪽의 불필요한 0들을 제거하고, 뒤쪽의 LSB(정밀한 정보)를 그대로 보존합니다.
        2) 큰 값 (이상치): 큰 자릿수(Magnitude)를 보존하고, 필요하다면 하위 비트를 일부 희생합니다.

### 3. 결과 및 추론 (Inference)

* 이 방식을 통해 4비트 용량만 차지하면서도 이상치의 크기와 작은 값의 정밀도(LSB)를 모두 잡을 수 있습니다.
* 추론 시에는 저장해둔 FLAG 값을 이용해 연산 결과(MMA output)를 다시 복원(조정)할 수 있어 추가적인 고정밀도 연산 장치가 필요 없습니다.

#### 4. 예제

공식, 여기서 clz는 앞쪽 0의 개수

$$FLAG = \max(29 - \text{clz}(m), 0)$$ 

$$[3, -2, 5, 0], Flag = 0 (정보 손실 없음)$$

$$[100, 3, -2, 10], Flag = 4 (right \quad shift 4)$$

$$ 3>>4 = 0, -2 >> 4 = 0, 10 >> 4 = 0 $$


---


---

