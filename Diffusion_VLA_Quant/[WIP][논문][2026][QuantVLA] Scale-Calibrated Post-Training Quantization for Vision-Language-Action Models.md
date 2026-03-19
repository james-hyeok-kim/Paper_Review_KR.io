# QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models

저자 : 

Jingxuan Zhang2*, Yunta Hsieh3*, Zhongwei Wan1, Haokun Lin4,

Xin Wang1, Ziqi Wang1, Yingtie Lei1, Mi Zhang1†

1The Ohio State University, 2

Indiana University, 3University of Michigan, 4City University of Hong Kong

[Git](https://quantvla.github.io/)

[Hugging Face](https://huggingface.co/papers/2602.20309)


발표 : 2026년 2월 27일 arXiv

논문 : [PDF](https://arxiv.org/pdf/2602.20309)

---

## 0. Summary

* 세계 최초의 VLA 시스템 전용 PTQ 방식이며, 특히 민감한 Diffusion Transformer(DiT) 액션 헤드를 성공적으로 양자화한 첫 사례

### 핵심 기술 구성 (Scale-Calibrated Components)

* 선택적 양자화 레이아웃 (Selective Quantization Layout)
    * 언어 백본(LLM)의 모든 선형 레이어와 DiT의 MLP 레이어는 정수형(Integer)으로 양자화합니다.
    * 가장 민감한 Attention Projection(Q, K, V, O)은 부동 소수점(Floating Point) 상태로 유지하여 오차 누적을 방지합니다.
* 어텐션 온도 매칭 (Attention Temperature Matching, ATM)
    * 양자화로 인해 어텐션 로그(logits)의 온도가 변하는 현상을 방지하기 위해 헤드별 스케일링 메커니즘을 적용하여 안정화합니다.
* 출력 헤드 밸런싱 (Output Head Balancing, OHB)
    * 레이어별 잔차 인터페이스(residual interface)를 보정하여 양자화 후 발생하는 에너지 드리프트 현상을 완화합니다.
  
### 주요 성능 및 결과

* 메모리 절감: 양자화된 구성 요소에서 약 70%의 상대적 메모리 절감을 달성했습니다.
* 작업 성공률 유지 및 향상: LIBERO 시뮬레이터 테스트 결과, $\pi0.5$ 모델에서 평균 성공률 97.6%를 기록하며 풀 프리시전(Full-precision) 베이스라인과 대등하거나 오히려 상회하는 성능을 보였습니다.
* 효율성: 추가 학습이 필요 없으며, 라벨이 없는 소량의 보정 데이터(Calibration buffer)만 사용하여 실제 배포에 매우 실용적입니다.


### Model Architecture

$$ \text{Vision Encoder(SigLIP2 or DINOv2)} \rightarrow \text{LLM(Cross Attention)} \rightarrow DiT$$


#### 모델별 파라미터 수(Parameter Count) 구성 비교

| 모델 | 컴포넌트 | 모델/레이어 상세 정보 | 파라미터 수 (추정치) | 전체 합계 |
| :--- | :--- | :--- | :--- | :--- |
| **$\pi$ 0.5** | **Vision Encoder** | SigLIP (So400m급) | **약 0.4B (400M)** | **총 약 3.3B** |
| | **LLM Backbone** | 126개 선형 레이어 (18개 블록) | **약 2.6B** | (VLA 통합형) |
| | **DiT Action Head**| 126개 선형 레이어 (18개 블록) | **약 0.3B (300M)** | |
| **GR00T N1.5**| **Vision Encoder** | SigLIP2 또는 DINOv2 | **약 0.4B ~ 0.6B** | **총 약 3.0B** |
| | **LLM Backbone** | 84개 선형 레이어 (약 12개 블록) | **약 1.8B ~ 2.0B** | (3B 변형 모델) |
| | **DiT Action Head**| 96개 선형 레이어 (16개 블록) | **약 0.5B ~ 0.6B** | |

#### VLA 모델 경량화(QuantVLA) 전후 상세 비교

| 모델 명칭 | 상태 | 경량화 레이어 수 (W4A8) | 유지 레이어 (FP16) | 메모리 점유 | 절감률 | 평균 성공률 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **π0.5** | **Baseline** | 0개 | 252개 | 4.27 GB | 0.0% | 97.1% |
| **π0.5** | **QuantVLA** | **180개** (LLM 126 + DiT MLP 54) | **72개** (DiT Attn) | **1.28 GB** | **70.0%** | **97.6%** |
| | | | | | | |
| **GR00T N1.5** | **Baseline** | 0개 | 180개 | 2.02 GB | 0.0% | 86.5% |
| **GR00T N1.5** | **QuantVLA** | **116개** (LLM 84 + DiT MLP 32) | **64개** (DiT Attn) | **0.91 GB** | **55.0%** | **88.0%** |

#### Vision Encoder 모델 사이즈 및 메모리 (FP16 기준)

| 모델 종류 | 아키텍처 변형 (Variant) | 파라미터 수 (Params) | 메모리 점유 (FP16) | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **SigLIP2 / SigLIP** | **So400m** (표준) | 약 400M (0.4B) | **0.8 GB** | $\pi$ 0.5(PaliGemma 기반)에서 주로 사용 |
| | **Base** (B/16) | 약 86M | **0.17 GB** | 경량화 버전 |
| | **Large** (L/16) | 약 303M | **0.6 GB** | 고성능 범용 |
| | **Giant** (g/14) | 약 1,000M (1B) | **2.0 GB** | 최상위 성능 모델 |
| **DINOv2** | **ViT-L/14** (Large) | 약 300M (0.3B) | **0.6 GB** | GR00T 및 로봇 제어에서 가장 선호됨 |
| | **ViT-g/14** (Giant) | 약 1,100M (1.1B) | **2.2 GB** | 고해상도 및 세밀한 특징 추출용 |
| | **ViT-S/14** (Small) | 약 21M | **0.04 GB** | 초고속 실시간 처리용 |


#### 표준 DiT 아키텍처 변형 (참조용)

| DiT 변형 | 블록 수 (Depth) | Hidden Dim | 파라미터 수 | 메모리 (FP16) | 비고 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **DiT-S** (Small) | 12 | 384 | 33M | **약 0.07 GB** | 초경량 실시간 제어 |
| **DiT-B** (Base) | 12 | 768 | 130M | **약 0.26 GB** | 범용 액션 헤드 |
| **DiT-L** (Large) | 24 | 1024 | 458M | **약 0.92 GB** | 고성능 공간 추론 |
| **DiT-XL** (XLarge) | 28 | 1152 | 675M | **약 1.35 GB** | 복잡한 멀티태스킹 |


---

## 1. Introduction

### 2. 배포의 걸림돌: 계산 및 메모리 병목

* 연구 결과, 연산 오버헤드의 상당 부분은 시각적 인식보다 다운스트림 추론 및 제어(언어 백본 및 DiT 액션 헤드)에서 발생

### 3. 기존 효율성 프레임워크의 한계

* 기존 연구들은 주로 시각 인코더를 효율화하거나 아키텍처 자체를 다시 설계(Pruning, Caching 등)하는 데 집중해 왔습니다.특히 DiT 액션 헤드는 성능에 매우 민감하고 언어 백본과 긴밀하게 결합되어 있어, 기존 방식으로는 성능 저하 없이 수정하기가 매우 어려웠습니다.
* 기존의 포스트 트레이닝 양자화(PTQ) 기법들은 VLA 시스템 특유의 복잡한 활성화 및 정밀도 특성을 제대로 포착하지 못했습니다.

### 4. QuantVLA의 제안 배경

* QuantVLA는 양자화로 인한 '스케일 드리프트(Scale Drift)'가 어텐션 온도와 잔차 에너지를 변화시켜 성능을 떨어뜨린다는 분석 결과에 기반하여 설계되었습니다.
    * 어텐션 온도: 어텐션 로그 $L$에서 온도가 낮아진다(분산이 증가) softmax 쏠리게 되고(Over-sharpening), 온도가 높다 (분산이 감소) Softmax 고르게(Over-flattening)

### 5. 핵심 기여도 (Main Contributions)

* 최초의 분석: VLA 모델(특히 DiT 액션 헤드)이 왜 양자화에 민감한지, 어떤 실패 모드가 발생하는지에 대한 최초의 체계적 분석을 제공합니다.
* 최초의 프레임워크: VLA 시스템을 위한 최초의 훈련이 필요 없는(Training-free) 로테이션 기반 PTQ 프레임워크를 제안합니다.
* 우수한 성능: 저정밀도 환경에서도 상당한 메모리 절감을 달성함과 동시에 최첨단(SOTA) 성능을 유지하거나 상회함을 입증했습니다.

---

## 2. Related Work
<p align = 'center'>
<img width="1100" height="400" alt="image" src="https://github.com/user-attachments/assets/1d3208d1-f40d-4ab6-b64b-7f29445258cc" />
</p>

### 1. Vision-Language-Action (VLA) 모델

* 초기 접근법: ALOHA, RT-1과 같이 처음부터 트랜스포머를 학습시키는 방식은 정확도는 높지만 일반화 성능이 제한적.
* 사전 학습 모델 기반: RT-2, OpenVLA 등은 언어 모델을 기반으로 액션을 토큰처럼 생성하여 뛰어난 추론 능력을 보여주었으나, 동작의 부드러움이 부족.
* 디퓨전 기반 및 하이브리드: 최근에는 $\pi0.5$, GR00T N1.5처럼 언어적 추론과 디퓨전 기반의 정밀 제어를 결합한 하이브리드 아키텍처가 주류.

### 2. 경량화 및 컴팩트 VLA 모델

* 경량 아키텍처: TinyVLA, SmolVLA 등은 모델의 백본 자체를 작게 설계하거나 비동기 추론 스택을 채택하여 속도를 높였습니다.
    * 비동기 추론 스택: Asynchronous Inference Stack은 로봇이 명령을 처리하는 '생각(추론)' 단계와 실제 '움직임(제어)' 단계를 분리하여 병렬로 진행하는 방식
* QuantVLA와의 차이: 이들은 새로운 모델 설계와 재학습이 필요하지만, QuantVLA는 기존 아키텍처를 그대로 유지하면서 양자화만 적용하므로 기존 대형 모델에도 즉시 적용 가능하다는 장점이 있습니다. 

### 3. 기존 효율화 프레임워크

* 모델 구조를 바꾸지 않고 추론 과정을 최적화하는 기법들. 
* 주요 기법: 레이어 가지치기(EfficientVLA), 시각적 토큰 캐싱(VLA-Cache), 동적 레이어 스킵(MoLe-VLA) 등이 제안되었습니다.
* QuantVLA와의 차이: 기존 기법들은 수치적 정밀도(Precision)를 유지한 채 연산량을 줄이려 하지만, QuantVLA는 수치 정밀도 자체를 낮추어(양자화) 메모리와 대역폭을 직접적으로 절감합니다. 

### 4. 포스트 트레이닝 양자화 (PTQ) 

* LLM 양자화: SmoothQuant나 DuQuant 같은 기법들이 제안되어 언어 모델의 이상치(Outlier) 문제를 해결해 왔습니다.
* DiT 양자화: 이미지/비디오 생성용 DiT를 위해 SVDQuant, ViDiT-Q 등이 연구되었습니다.
* VLA의 특수한 어려움: VLA는 멀티모달 추론과 디퓨전 롤아웃이 긴밀하게 결합되어 있어, 기존 PTQ를 그대로 적용하면 모달리티 간의 스케일 불일치와 오차 누적으로 인해 제어 안정성이 급격히 무너집니다. 

---

## 3. Method

<p align = 'center'>
<img width="890" height="508" alt="image" src="https://github.com/user-attachments/assets/9d401a3d-5650-441e-9dca-60876ef0c695" />
</p>


### 3.1 디퓨전 기반 VLA 모델의 기초 (Preliminaries)

* 입력 처리: 시각 정보는 SigLIP2 또는 DINOv2 인코더를 통해 이미지 토큰으로 변환되고, 언어 지침은 사전 학습된 LLM 백본에 의해 임베딩됩니다.
* 멀티모달 통합: 두 토큰은 공유 트랜스포머 공간에서 어텐션을 통해 결합되어 태스크 조건부 표현 $F_{VL}$을 형성합니다.
* 액션 생성 (DiT): 정책 헤드인 Diffusion Transformer(DiT)는 $F_{VL}$과 로봇의 상태 정보를 조건으로 받아 다음 수식과 같이 반복적으로 액션 잠재 변수를 정제합니다.

$$x_{t-1}=f_{\theta}(x_{t},F_{VL},t)$$

### 3.2 양자화 설정 및 DiT의 민감도 분석

* 기본 프레임워크: 트랜스포머 스택에서 안정성이 검증된 DuQuant의 재매개변수화(Reparameterization) 기법을 채택하여 선형 레이어의 이상치를 재분배합니다.
* 핵심 과제:입력 드리프트: 상류(Upstream) LLM 백본이 양자화되면서 발생하는 미세한 섭동이 하류의 DiT 액션 정책으로 전축됩니다.정밀도 손실: DiT는 실제 로봇 제어를 위한 정밀한 토큰을 생성해야 하므로, 아주 작은 반올림 오차나 스케일 불일치도 큰 제어 오류로 이어집니다.수학적 분석: 분석 결과, 양자화 오차는 어텐션의 **로그 온도($s_{q}s_{k}$)**와 **잔차 스트림 에너지($s_{v}s_{o}$)**라는 두 가지 결정적 요인을 왜곡시키는 것으로 나타났습니다.

### 3.3 QuantVLA 프레임워크

위의 분석을 바탕으로 QuantVLA는 훈련이 필요 없는(Training-free) 세 가지 핵심 전략을 통합합니다.

#### 1. 선택적 양자화 레이아웃 (Selective Quantization Layout)LLM: 모든 선형 레이어를 정수형(Integer)으로 양자화합니다.

* DiT: MLP 블록은 양자화하되, 가장 민감한 어텐션 프로젝션( $W_q, W_k, W_v, W_o$ )은 부동 소수점(FP) 상태로 유지하여 드리프트 증폭을 막습니다.

#### 2. 어텐션 온도 매칭 (ATM: Attention Temperature Matching)

* 양자화로 인해 어텐션 분포가 너무 날카롭거나 평평해지는 것을 방지합니다.
* 헤드별 스칼라( $\alpha$ )를 사용하여 원본(Teacher)과 양자화된(Student) 로그의 표준편차를 정렬합니다.
    * Multi-Head Attention의 Scale 값
    * 이 값은 소량의 무라벨 데이터로 계산되며, 추론 시에는 기존 스케일에 병합되어 추가 연산이 발생하지 않습니다.

$$\alpha = \frac{Std(L_{Teacher})}{Std(L_{Quantized})}\quad (12)$$ 

#### 3. 출력 헤드 밸런싱 (OHB: Output Head Balancing)

* 출력 프로젝션 이후의 에너지 변화를 측정하여 잔차 주입(Residual injection)의 이득을 복원합니다.
    * 트랜스포머 블록: [원래 입력값] + [새로 계산된 값] 구조
    * $\beta$ 활용하여 출력 밸런싱 (OHB)
* 레이어별 스칼라( $\beta$ )를 사용하여 출력값의 RMS(에너지 강도)를 원본 모델과 일치시킵니다.
    * 이를 통해 깊은 DiT 스택 내에서 에너지가 누적되거나 유실되는 문제를 해결하고 레이어 노멀라이제이션의 안정성을 확보합니다. 

$$\beta = \frac{RMS(Z_T)}{RMS(Z_Q)}$$

$$Z_{corrected} = \frac{Z_Q}{\beta} \quad (17)$$


---

## 4. Experiment

### 1. 실험 설정 (Experimental Settings)

* 대상 모델: 효율적인 추론을 중시하는 OpenPI $\pi0.5$ 와 고성능 휴머노이드 제어를 목표로 하는 GR00T N1.5를 평가 대상.
* 벤치마크: 로봇의 4가지 핵심 능력을 테스트하는 LIBERO 시뮬레이터를 사용했습니다.
    * Spatial: 관계 추론 및 정밀 배치 테스트.
    * Object: 물체 잡기 및 조작 테스트.
    * Goal: 지시사항 준수 및 목표 달성 측정.
    * Long: 장기적인 제어 능력 및 오차 누적 관리 테스트.
* 양자화 설정: 기본적으로 W4A8(가중치 4비트, 활성화 8비트) 설정을 적용했으며, NVIDIA A100 GPU에서 실험을 진행했습니다.

### 2. 선택적 양자화 레이아웃 검증 (Selective Layout Validation)

<p align = 'center'>
<img width="986" height="355" alt="image" src="https://github.com/user-attachments/assets/17f9735c-cab1-415b-abd0-e4f3e9989864" />
</p>


* 분석 결과: DiT 액션 헤드 전체를 양자화하거나 모델 전체를 양자화할 경우, 특히 Long(장기 작업) 테스트에서 성능이 급격히 저하되었습니다.
* 최적의 조합: LLM 백본 전체와 DiT의 MLP 블록만 양자화하고, 어텐션 프로젝션은 FP16으로 유지하는 것이 메모리 절감과 성능 보존의 가장 좋은 균형점으로 나타났습니다.

### 3. ATM 및 OHB 보정 효과 (Calibration Effect)

<p align = 'center'>
<img width="999" height="354" alt="image" src="https://github.com/user-attachments/assets/003ace45-382d-4cb6-a9de-9f7cdd22dd54" />
</p>

* ATM (온도 매칭): 양자화로 인해 어텐션 로그(Logits)의 표준편차가 원본과 달라지는 것을 확인했으며, ATM을 적용했을 때 모든 블록에서 원본(Teacher) 분포에 가깝게 복원되었습니다.
* OHB (에너지 밸런싱): 출력 프로젝션 이후의 RMS 에너지를 원본과 맞춤으로써, 특히 깊은 레이어에서 발생하는 에너지 드리프트를 효과적으로 억제했습니다.

### 4. 주요 결과 (Main Results on LIBERO)

<p align = 'center'>
<img width="992" height="320" alt="image" src="https://github.com/user-attachments/assets/8e500111-1dd9-4e84-a85c-f0a4a136055d" />
</p>

* QuantVLA는 기존 양자화 방식(DuQuant)의 성능 저하 문제를 해결하며 최첨단 성능을 달성했습니다.
* 특이사항: 두 모델 모두 경량화 후에도 원본(FP16) 베이스라인과 대등하거나 오히려 성능이 소폭 향상되는 결과를 보였습니다.
* 기존 방식 비교: 일반적인 DuQuant를 전체 적용했을 때 $\pi0.5$의 성공률이 76.3%까지 떨어졌던 것과 비교하면 매우 뛰어난 보존력입니다.

### 5. 강건성 및 효율성 (Robustness & Efficiency)

<p align = 'center'>
<img width="493" height="221" alt="image" src="https://github.com/user-attachments/assets/957af4e2-45d1-4292-bfb3-457dfd61a12a" />
<img width="488" height="220" alt="image" src="https://github.com/user-attachments/assets/22753f50-8c51-4557-9779-1bed403f26c8" />
</p>

* 초저정밀도 테스트: 더 공격적인 W4A4 설정에서도 $\pi0.5$ 모델은 95.3%의 높은 성공률을 유지하며 안정적인 동작을 보였습니다.
* 노이즈 저항성: 디퓨전의 디노이징 스텝(Denoising steps) 변화에도 성능이 일정하게 유지되어 다양한 추론 환경에 적합함을 증명했습니다.
* 실제 가치: 약 70%의 메모리 절감을 통해 동일한 하드웨어 예산으로 더 긴 입력 호라이즌을 처리하거나 여러 정책을 동시에 실행할 수 있는 확장성을 확보했습니다.


---

## 5. Conclusion




---
