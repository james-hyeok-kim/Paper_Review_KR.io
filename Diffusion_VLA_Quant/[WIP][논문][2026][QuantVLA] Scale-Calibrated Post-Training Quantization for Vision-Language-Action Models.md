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

| 모델 | 컴포넌트 | 모델/레이어 정보 | 메모리 (FP16 기준) |
| :--- | :--- | :--- | :--- |
| **$\pi$ 0.5** | Vision Encoder | SigLIP2 또는 DINOv2 | - |
| | LLM Backbone | 126개 선형 레이어 (18개 블록 수준) | **4.27 GB** |
| | DiT Action Head | 126개 선형 레이어 | (LLM+DiT 합산 관리) |
| **GR00T N1.5** | Vision Encoder | SigLIP2 또는 DINOv2 | - |
| | LLM Backbone | 84개 선형 레이어 | **2.02 GB** |
| | DiT Action Head | 96개 선형 레이어 | (LLM+DiT 합산 관리) |


#### Vision Encoder 모델 사이즈 및 메모리 (FP16 기준)

| 모델 종류 | 아키텍처 변형 (Variant) | 파라미터 수 (Params) | 메모리 점유 (FP16) | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **SigLIP2 / SigLIP** | **So400m** (표준) | 약 400M (0.4B) | **0.8 GB** | $\pi$0.5(PaliGemma 기반)에서 주로 사용 |
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

