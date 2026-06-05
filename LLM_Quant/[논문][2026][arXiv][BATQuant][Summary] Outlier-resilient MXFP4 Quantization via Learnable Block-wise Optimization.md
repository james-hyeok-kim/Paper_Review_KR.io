# BATQuant: Outlier-resilient MXFP4 Quantization via Learnable Block-wise Optimization

* 저자 :

Ji-Fu Li1, Manyi Zhang1⋆, Xiaobo Xia2, Han Bao1, Haoli Bai1,Zhenhua Dong1, and Xianzhi Yu1

1 Huawei Technologies

2 University of Science and Technology of China

{lijifu4, zhangmanyi6}@huawei.com

* 발표 : arXiv 공개 2026년 3월

* 논문 : [PDF](https://arxiv.org/pdf/2603.16590)

---

## 0. Summary

<p align='center'>
<img width="707" height="559" alt="image" src="https://github.com/user-attachments/assets/589129b3-2fe8-4a91-b22a-403c126a11a9" />
</p>

### 0.0. 문제

* MXFP4(Microscaling FP4) 양자화 라는 새로운 하드웨어 포맷(NVIDIA Blackwell, AMD CDNA4 등이 지원)에서는
    * 32개 element가 하나의 block을 이루고 block마다 별도의 scaling factor를 공유합니다.
    * 그런데 기존 INT4용 SOTA 방법들(QuaRot, SpinQuant, FlatQuant)을 MXFP4에 그대로 적용하면 RTN보다도 못한 성능 붕괴가 일어납니다. 
* 원인은 두 가지:
    * Cross-block energy transfer:
        * global rotation은 모든 채널을 섞기 때문에, outlier가 많은 block의 에너지를 outlier 없는 block들로 옮겨놓음 →
        * 멀쩡했던 block들이 새로운 outlier를 떠안게 되어 block-wise scaling factor 효율이 떨어짐.
    * Bimodal distribution 문제:
        * Hadamard matrix는 ±1 원소로 구성되어 있어, 극단적 outlier를 가진 block에 적용하면 분포가 양봉(bimodal) 형태가 됨 →
        * MXFP4의 한정된 7개 양자화 grid를 효율적으로 활용하지 못함.

### 0.1. 핵심 아이디어

* Block-wise Affine Transformation (BAT):
    * 변환 행렬 P를 MXFP block 크기(32)에 정확히 맞춘 block-diagonal 구조로 제한.
    * P = diag(P₁, P₂, ..., Pₖ)로 각 Pᵢ ∈ ℝ³²ˣ³².
    * 이렇게 하면 변환이 block 내부에서만 일어나므로 cross-block energy transfer가 원천 차단됨.
    * 또한 orthogonality 제약을 풀고 affine 변환으로 학습해 분포 형태를 자유롭게 reshape 가능.

* Global and Private Kronecker (GPK) 분해:
    * block마다 독립적인 Pᵢ를 저장하면 파라미터가 너무 많아짐.
    * Pᵢ = Bᵢ ⊗ A로 분해하되, A는 모든 block에서 공유, Bᵢ만 block별로 고유.
    * 파라미터 수를 g₁² + k·g₂²로 압축 (FlatQuant 대비 74% 감소, Naive Kronecker 대비 79% 감소).
    * 기본 설정 g₁=8, g₂=4.

* Block-wise Learnable Clipping:
    * 각 block마다 학습 가능한 clipping threshold $⁡\beta_i^{\min}, \beta_i^{\max}$​을 두어,
    * 변환 후에도 남은 residual outlier를 block 단위로 억제.
    * Sigmoid로 (0, 1) 범위 ratio를 학습.

* 학습 목표는 FlatQuant와 동일하게 layer-wise MSE 최소화 (full-precision 출력과 quantized 출력의 차이).

### 0.2. Quantization 설정

* MXFP 포맷 (정수 INT4가 아님)
    * MXFP4는 1 sign + 2 exp + 1 mantissa로 7개 양수값만 표현 ({0.5, 1, 1.5, 2, 3, 4, 6})
    * block size 32
    * scaling factor는 UE8M0
* 주요 설정: W4A8KV16 (가벼움), W4A4KV16 (가장 도전적), W4A8KV8, W4A8KV4
* Weight quantizer로 GPTQ가 RTN보다 일관되게 우수하므로 GPTQ 통합본을 기본으로 사용
* 모든 linear layer에 저비트 적용, layer norm·RoPE·attention score는 BF16 유지

### 0.3. 적용 모델

* LLM: Qwen3-8B
* MLLM: Qwen3-VL-8B-Instruct (text model + ViT 모두에 BAT 적용, ViT는 KV cache 없으므로 Pqkv, Po만 사용)

### 0.4. 벤치마크

* Multimodal: MME, OCRBench, DocVQA, RealWorldQA, VLMBlind
* Non-reasoning: PIQA, Winogrande, HellaSwag, ARC-E, ARC-C
* Reasoning: GSM8K, MATH-500, AIME24, AIME25, GPQA-D
* 비교 baseline: RTN, QuaRot, SpinQuant, BRQ (block rotation), FlatQuant, SmoothQuant, GPTQ

### 0.5. 효과 (주요 수치)

| 설정 | Qwen3-VL-8B Recovery | Qwen3-8B Reasoning Recovery |
| :--- | :--- | :--- |
| **W4A8KV16** | 99.29% (1% 미만 손실) | 97.46% |
| **W4A4KV16** | 96.43% (FlatQuant 대비 +1.64%p) | 92.45% (BRQ 76.25%, SpinQuant 76.35% 대비 큰 격차) |
| **W4A8KV8** | 98.89% | 96.22% |
| **W4A8KV4** | 97.51% | 94.00% |


* MXFP4에서 rotation 기반 방법(QuaRot, SpinQuant)이 RTN보다도 못한 성능 붕괴를 보이는 환경에서 BATQuant는 SOTA 안정성 확보.
    * 특히 LLaMA-3-70B처럼 어려운 reasoning task일수록 차이가 큼.
    * W4A4KV16 reasoning에서 SpinQuant 60.99 → BATQuant 71.91 (avg accuracy, 약 +11%p).
* Block-diagonal 변환의 block size를 g=32(MXFP block 크기)로 정확히 맞췄을 때 최적.
    * g보다 작거나 크면 모두 성능 저하 (16일 때 67.68, 128일 때 68.42 → 32일 때 68.70).
* Qualitative case study:
    * VLMBlind 선분 교차 개수 세기, OCRBench 열차 번호 인식 등에서 BRQ가 hallucination/truncation을 일으키는 입력에 BATQuant는 BF16과 동일한 정답을 회복.

---
