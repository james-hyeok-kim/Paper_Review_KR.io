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

### 4. 예제

공식, 여기서 clz는 앞쪽 0의 개수

$$FLAG = \max(29 - \text{clz}(m), 0)$$ 

$$[3, -2, 5, 0], Flag = 0 (정보 손실 없음)$$

$$[100, 3, -2, 10], Flag = 4 (right \quad shift 4)$$

$$ 3>>4 = 0, -2 >> 4 = 0, 10 >> 4 = 0 $$



---

## 4. Analysis of QuaRTZ

<p align = 'center'>
<img width="704" height="364" alt="image" src="https://github.com/user-attachments/assets/80f77d80-a997-4b32-9fbb-0e42342139e0" />
</p>

### 1. 이론적 검증: 왜곡 분석 (Distortion Analysis)

저자들은 수학적으로 QuaRTZ의 오차가 일반적인 4비트 양자화보다 작다는 것을 증명합니다.

* 비교: 일반적인 4비트 균일 양자화(Naive INT4)의 오차( $E_q^4$ )와 QuaRTZ 방식(8비트 양자화 후 압축)의 오차( $E_{total}$ )를 비교합니다.
* 핵심 정리: 확률 밀도 함수에서 "절댓값이 큰 값(이상치)의 비중이 전체의 절반 미만이라면", QuaRTZ의 총 오차가 일반 4비트 양자화 오차보다 항상 작다는 것을 부등식으로 증명했습니다 ( $E_{total} < E_q^4 \quad (3)$ ).

* 의미: 확산 모델의 활성화 값(Activation)은 대부분 0 근처에 몰려 있고 큰 값은 드물기 때문에, 이 조건은 항상 만족되며 이론적으로 더 우월함을 보장합니다.


#### 증명

1) 단계 크기 ( $s$ )의 정의
    1) 4비트 양자화: $2^3$ (부호 제외 3비트) $\rightarrow$ 8개 구간
    2) 8비트 양자화: $2^7$ (부호 제외 7비트) $\rightarrow$ 128개 구간
    3) $s_4 = 16 s_8$

2) 부등식 설정: 일반적인 양자화 오차는 단계 크기의 절반($s/2$)을 넘지 않습니다.
    1) 일반 4비트 오차 상한 ( $E_q^4$ ): $\frac{s_4}{2} = \frac{16s_8}{2} = 8s_8$
    2) 8비트 오차 상한 ( $E_q^8$ ): $\frac{s_8}{2} = 0.5s_8$ 

3) QuaRTZ의 총 오차( $E_{total}$ )는 "1차 8비트 양자화 오차"와 "2차 LZS 압축 오차( $E_{LZS}$ )"의 합

$$E_{total} \le E_q^8 + \mathbb{E}[E_{LZS}] < E_q^4$$

$$0.5s_8 + \mathbb{E}[E_{LZS}] < 8s_8$$

$$\therefore \mathbb{E}[E_{LZS}] < 7.5s_8$$

4) 압축으로 인해 발생하는 오차의 평균(기댓값)이 $7.5s_8$보다 작다면 QuaRTZ가 이득

5) LZS 오차의 분석 (Worst Case)
    1) LZS는 값이 작을 때는 오차가 0이고, 값이 클 때(이상치)만 하위 비트를 잘라내어 오차가 발생합니다.
    2) 오차가 발생하는 조건: 8비트 정수 기준 절댓값이 8 이상일 때 ( $|m| \ge 8$ ).
    3) 최악의 경우 (Worst Case): 가장 큰 값을 표현하기 위해 비트를 가장 많이 잘라내는 경우입니다. 이때 최대 4비트가 잘려나가며, 손실되는 값의 크기는 최대 $(2^4 - 1) = 15$배의 $s_8$입니다.

$$E_{LZS}(max) \approx 15 s_8$$

6) 확률 조건 도출

"이상치가 발생할 확률"을 $P$라고 할 때

$$\mathbb{E}[E_{LZS}] \approx (\text{최대 오차}) \times P$$

$$\mathbb{E}[E_{LZS}] \le 15s_8 \times P$$

* 이 값이 앞서 구한 조건( $7.5s_8$ )보다 작아야 하므로

$$15s_8 \times P < 7.5s_8$$

양변을 $15s_8$로 나누면 최종 조건이 증명됩니다.

$$P < \frac{7.5}{15} = 0.5$$

### 2. 정보 효율성: 비트 단위 엔트로피 (Bit-wise Entropy)

<p align = 'center'>
<img width="711" height="333" alt="image" src="https://github.com/user-attachments/assets/a1dfb35d-b7cf-4364-8d5f-50216608cb82" />
</p>

* 엔트로피 증가: Figure 4를 통해, 모든 레이어에서 QuaRTZ가 일반 INT4보다 더 높은 엔트로피를 가짐을 보여줍니다.

* 의미: 일반 INT4는 0이 많아 비트 낭비가 심한 반면, QuaRTZ는 불필요한 0을 제거(Leading Zero Suppression)했기 때문에 4개의 비트가 모두 고르게 사용되며 정보 밀도가 높습니다.


### 3. 경험적 분석: 분포 보존 (Empirical Analysis)

* INT4의 문제: 일반 INT4는 0 근처에서 값들이 듬성듬성하게 찍히는(Rounding error) 현상이 발생하여 미세한 텍스처 정보를 잃습니다.
* QuaRTZ의 장점: QuaRTZ는 0 근처의 미세한 값들(LSB)을 잘 보존하여 원래 분포와 유사한 형태를 유지하면서도, 동시에 이상치(Outlier)의 크기 정보도 놓치지 않았습니다.

### 4. 하드웨어 효율성: 지연 시간 (Latency Analysis)

A100 GPU 기준

<p align = 'center'>
<img width="685" height="286" alt="image" src="https://github.com/user-attachments/assets/6b2b30ac-311e-45f3-9cae-a4ba875484ec" />
</p>

* 연산 비용: 압축을 풀 때 사용하는 비트 시프트(Bit-shift) 연산은 GPU의 텐서 코어 연산(MMA)에 비해 비용이 거의 들지 않습니다 (무시할 수준).
* 메모리 이득: 데이터 전송량(Traffic)이 INT8 대비 거의 절반으로 줄어듭니다.
* 속도: 표 1(Table 1)에 따르면, PyTorch 기본 구현에 비해 월등히 빠른 속도를 보여줍니다 (예: $4096 \times 4096$ 레이어 기준 PyTorch 5.4ms vs QuaRTZ 0.18ms).


---

## 5. EXPERIMENTS AND ANALYSIS

### 5.1 SETUP

1) 실험 환경 (Setup)
    1) 모델
        1) UNet 기반: LDM, Stable Diffusion (SD) v1.4, SDXL-Turbo.
        2) DiT (Transformer) 기반: PixArt-$\Sigma$, FLUX.1-schnell.
    2) 평가 지표: FID (이미지 품질), CLIP Score (텍스트 정합성), ImageReward (인간 선호도), LPIPS/PSNR (유사도) 등을 사용했습니다.
    3) 비교 대상: TFMQ-DM, DGQ, SVDQuant 등 최신 양자화 기법들과 비교했습니다.

### 5.2 MAIN RESULTS

2) 주요 결과 (Main Results)
    1) W4A4 (가중치 4비트, 활성화 4비트) 설정에서 실험한 결과입니다.
    2) 비조건부 생성 (Unconditional Generation)
        1) LDM 모델에서 일반적인 4비트 양자화(Naive INT4)는 이미지가 완전히 붕괴(FID 327.01)되었으나, QuaRTZ는 FID 7.11을 기록하며 8비트 모델(W4A8)에 근접한 성능을 보였습니다.
    3) 텍스트-이미지 생성 (Text-to-Image)
        1) SDv1.4 & SDXL-Turbo: 기존 W4A6(6비트 활성화) 모델들보다 W4A4(4비트)인 QuaRTZ가 더 뛰어난 성능을 보였습니다.
        2) FLUX.1-schnell: FID 6.98을 기록하며, 보조 가지(Auxiliary Branch)를 사용하는 SVDQuant(FID 7.07)보다 더 나은 성능을 달성했습니다. 이는 QuaRTZ가 순수 4비트만으로도 고성능을 낼 수 있음을 증명합니다.
        3) 한계점: PixArt-$\Sigma$ 모델에서는 SVDQuant보다 성능이 다소 떨어졌는데, 이는 SVDQuant가 이상치(Outlier) 보정을 위한 별도의 모듈을 사용하기 때문으로 분석했습니다.

### 5.3 ABLATION STUDY

3) 절제 연구 (Ablation Study)
    1) 그룹 크기의 영향양자화 그룹 크기(Group Size, $G_s$ )가 성능에 미치는 영향을 분석했습니다.
    2) 트레이드오프: 그룹 크기가 커질수록( $G_s$ 증가), 그룹 내에 이상치가 포함될 확률이 높아집니다. 이상치가 포함되면 전체를 많이 시프트(Shift)해야 하므로 작은 값들(LSB)이 잘려 나갑니다.
    3) 결과: 그룹 크기가 커질수록 FID 점수가 선형적으로 나빠집니다.
    4) 권장: 저자들은 지연 시간과 이미지 품질의 균형을 위해 그룹 크기 16 또는 32를 권장합니다.


4) 확장성: LLM에의 적용 (Potential Applications to LLMs)
    1) 대상: Qwen2, LLaMA2, LLaMA3 모델.
    2) 결과: 4비트 QuaRTZ를 적용했을 때 펄플렉서티(Perplexity) 증가율이 +4.5% ~ +10.7% 수준으로 억제되었습니다.
    3) 의미: 이는 LSB(최하위 비트) 보존이 언어 모델에서도 유효하며, QuaRTZ가 범용적인 저비트 양자화 기술로 쓰일 수 있음을 시사합니다

---






