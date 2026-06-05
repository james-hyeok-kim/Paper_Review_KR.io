# FlatQuant: Flatness Matters for LLM Quantization

저자 : 

Yuxuan Sun * 1 Ruikang Liu * 2 Haoli Bai † 1 Han Bao 1 Kang Zhao 1 Yuening Li 3

Jiaxin Hu 1 Xianzhi Yu 1 Lu Hou 1 Chun Yuan 1 Xin Jiang 1 Wulong Liu 1 Jun Yao 1

*Equal contribution 

1Huawei Noah’s Ark Lab 

2Shenzhen International Graduate School, Tsinghua University 

3The Chinese University of Hong Kong. 

발표 : ICML 2025 (PMLR 267), Huawei Noah's Ark Lab 외. arXiv 최초 공개 2024년 10월, v4는 2025년 8월.

논문 : [PDF](https://arxiv.org/pdf/2410.09426)

---

## 0. Summary

<p align = 'cetner'>
<img width="1083" height="526" alt="image" src="https://github.com/user-attachments/assets/bb2b0ed8-6700-480d-8c1b-d6a33ae78d0f" />
</p>

### 0.0. 문제

* LLM 양자화 시 weight/activation 분포의 flatness(평탄성) 가 중요하다는 관찰에서 출발. 
* 기존 per-channel scaling이나 Hadamard 변환은 outlier를 충분히 평탄화하지 못함.

### 0.1. 핵심 아이디어

* 이를 해결하기 위해 각 linear layer마다 학습 가능한 affine 변환 행렬 P를 도입하되, P를 그대로 쓰면 연산·메모리 부담이 크므로 Kronecker product (P = P₁ ⊗ P₂) 로 두 개의 작은 행렬로 분해.
    * 수학적 트릭: 선형 레이어 Y = XWᵀ에서, 가역행렬 P를 끼워 넣어도 결과는 동일합니다.
    * $Y = XW^\top = (XP)(P^{-1}W^\top)$
* 기존 방법들과의 차이
    * Per-channel scaling (SmoothQuant): P를 대각행렬(diag(c))로 제한 → 표현력이 약함
    * Hadamard 변환 (QuaRot): P를 고정된 ±1 행렬로 사용 → 모든 layer에 동일하게 적용, layer별 특성 무시
    * FlatQuant: P를 학습 가능한 일반 가역행렬로 두고, layer별로 따로 최적화
* 추가로 per-channel scaling, learnable clipping threshold를 결합하고, calibration 데이터 128문장으로 block-wise MSE 최소화 학습.
* affine 변환과 양자화를 단일 Triton 커널로 fusion해 메모리 접근 오버헤드 최소화.

### 0.2. Quantization 설정

* 메인: W4A4KV4 (weight, activation, KV cache 모두 INT4)
* weight는 per-channel symmetric, activation은 per-token symmetric, KV cache는 group-size 128 asymmetric
* RTN과 GPTQ 둘 다 weight quantizer로 사용 가능 (RTN만으로도 SOTA 달성)
    * GPTQ: Hessian 기반으로 weight를 한 번에 하나씩 양자화하면서, 양자화 오차를 나머지 weight에 분산시켜 보정.
    * 반적으로 W4A4 같은 저비트 환경에서는 GPTQ가 거의 필수였습니다.
* 추가로 W4A16, W3A16 (weight-only), KV cache-only 등 다양한 설정 지원
    * W4A4KV4용으로 학습한 변환 행렬을 그대로 W4A16이나 KV4-only에 재사용해도 성능이 거의 그대로 나옵니다

### 0.3. 적용 모델

* LLaMA-2 (7B/13B/70B), LLaMA-3 (8B/70B), LLaMA-3.1-8B-Instruct, Qwen-2.5-Instruct (7B/32B), DeepSeek-V3-Base, DeepSeek-R1 (671B MoE).


### 0.4. 벤치마크

* Perplexity: WikiText-2, C4
* Zero-shot QA: ARC-C/E, HellaSwag, LAMBADA, PIQA, WinoGrande
* 추가: MT-Bench, C-Eval, MMLU, AIME2024

### 0.5. 효과 (주요 수치)

* LLaMA-3-70B W4A4 기준 정확도 손실 1% 미만 (FP16 79.95 → 79.01, RTN 기준). 동일 조건 SpinQuant 대비 +7.5%p, QuaRot 대비 +43%p 이상.
* LLaMA-2-70B WikiText-2 PPL: FP16 3.32 → FlatQuant 3.55 (gap 0.23)
* LLaMA-3-8B WikiText-2 PPL: SpinQuant 7.39 → FlatQuant 6.98
* 추론 속도: FP16 대비 prefill 2.3배, decoding 1.7배 (batch 64, RTX 3090)
* Calibration cost: LLaMA-3-8B 기준 단일 GPU에서 약 0.9시간, 26GB 메모리
* 추가 파라미터/연산 오버헤드: LLaMA-2-7B 기준 FLOPs의 2.61%, 메모리 3.41MB로 매우 가벼움

---
