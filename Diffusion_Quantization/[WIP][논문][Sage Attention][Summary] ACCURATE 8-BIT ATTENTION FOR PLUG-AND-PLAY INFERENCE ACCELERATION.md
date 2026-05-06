# SAGEATTENTION: ACCURATE 8-BIT ATTENTION FOR PLUG-AND-PLAY INFERENCE ACCELERATION

저자 : 

Jintao Zhang, Jia Wei, Haofeng Huang, Pengle Zhang, Jun Zhu, Jianfei Chen∗

Dept. of Comp. Sci. & Tech., Institute for AI, BNRist Center,

Tsinghua-Bosch Joint ML Center, THBI Lab, Tsinghua University

{zhang-jt24@mails., jianfeic@, dcszj@}tsinghua.edu.cn


발표 : ICLR 2025 (International Conference on Learning Representations)

논문 : [PDF](https://arxiv.org/pdf/2410.02367)

--

## 0. Summary

<p align = 'center'>
<img width="851" height="330" alt="image" src="https://github.com/user-attachments/assets/17f53ba1-a6b6-4efa-8955-1d9079182082" />
</p>


### 0.1. 핵심 아이디어 (Key Ideas)

* INT8양자화 도입 (Q,K)
    * SageAttention은 기존 8비트 양자화 방법들이 어텐션 메커니즘에서 정확도가 급격히 떨어지는 문제를 해결하기 위해 다음과 같은 기술을 제안합니다.
* K 매트릭스 평활화 (Smoothing Matrix K)
    * K 매트릭스에 존재하는 채널별 아웃라이어(Outlier)를 처리하기 위해 모든 토큰의 평균값을 빼주는 방식을 사용합니다.
* 이 과정은 정확도를 대폭 높이면서도 연산 오버헤드는 0.2% 미만으로 매우 적습니다.
* $P \times V$ 행렬 FP16 어큐뮬레이터 활용:
    * $P \times V$ 행렬 곱셈 시, INT8 양자화 대신 FP16 데이터 타입을 유지하면서 FP16 어큐뮬레이터를 사용합니다.
    * 이를 통해 정확도 손실 없이 연산 속도를 2배 높였습니다.
* 적응형 양자화 (Adaptive Quantization):
    * 모델의 각 레이어별로 최적의 정확도와 속도를 제공하는 커널을 자동으로 선택하여, 정확도를 유지하면서도 속도를 최대 12% 추가 향상시킵니다.
    * 커널 퓨전 (Kernel Fusion): 양자화 과정을 RoPE(Rotary Position Embedding)와 같은 이전 연산과 병합하여 메모리 I/O 오버헤드를 최소화했습니다.

### 어텐션 메커니즘 핵심 기술 비교표

| 비교 항목 | 표준 어텐션 (Standard) | FlashAttention (1, 2) | SageAttention |
| :--- | :--- | :--- | :--- |
| **핵심 최적화 전략** | 없음 (수학적 정의 충실) | 메모리 I/O 최적화 (Tiling, Online Softmax) | 연산 속도 및 정밀도 최적화 (INT8 Quantization) |
| **데이터 정밀도** | FP16 / BF16 (고정밀) | FP16 / BF16 (고정밀) | **INT8 (Q, K) / FP16 (P, V)** (혼합 정밀도) |
| **하드웨어 가속기 활용** | 일반적인 매트릭스 연산 유닛 | FP16/BF16 Tensor Core 활용 | **INT8 mma & FP16 Accumulator** 최적 활용 |
| **처리 속도 (TOPS)** | 가장 낮음 | 중간 (RTX4090 기준 약 165 TOPS) | **가장 높음 (RTX4090 기준 약 341 TOPS)** |
| **정확도 보정 기술** | 불필요 | 불필요 | **Smoothing Matrix K** (아웃라이어 제거) |
| **성능 특징** | 메모리 부족 및 속도 저하 발생 | 메모리 효율 대폭 향상, I/O 병목 해결 | **FlashAttn2 대비 2.1배 가속**, 정확도 손실 거의 없음 |

## 0.2. 주요 효과 (Effects)

* 획기적인 속도 향상: 기존 최적화 기술인 FlashAttention2 대비 약 2.1배, xformers 대비 약 2.7배 빠른 속도를 보여줍니다.
* 높은 연산 효율: RTX4090 GPU에서 340 TOPS(초당 테라 연산 수)를 달성하여 이론적 한계치의 52%에 도달했습니다.
* 높은 정확도 유지: 거대 언어 모델(LLM)뿐만 아니라 이미지 및 비디오 생성 모델에서도 성능 저하가 거의 없는(평균 0.2% 미만) 수준의 정확도를 보여줍니다.
* 편의성: 추가 학습이 필요 없는 사후 양자화(Post-training Quantization) 방식으로, 기존 모델에 즉시 교체하여 사용할 수 있는 '플러그 앤 플레이' 기능을 제공합니다.

