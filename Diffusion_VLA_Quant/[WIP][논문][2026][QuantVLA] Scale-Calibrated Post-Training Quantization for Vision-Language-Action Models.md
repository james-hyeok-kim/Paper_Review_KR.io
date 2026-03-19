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

<img width="533" height="161" alt="image" src="https://github.com/user-attachments/assets/1d3208d1-f40d-4ab6-b64b-7f29445258cc" />


---

## 3. Method

---

## 4. Experiment


---

## 5. Conclusion

---
