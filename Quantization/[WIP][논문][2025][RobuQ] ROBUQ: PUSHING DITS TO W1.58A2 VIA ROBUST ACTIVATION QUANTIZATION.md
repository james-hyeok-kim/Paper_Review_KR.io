# ROBUQ: PUSHING DITS TO W1.58A2 VIA ROBUST ACTIVATION QUANTIZATION

저자 : Kaicheng Yang1∗, Xun Zhang1∗, Haotong Qin2,

Yucheng Lin1, Kaisen Yang3, Xianglong Yan1, Yulun Zhang1†

1Shanghai Jiao Tong University, 2ETH Zurich, 3Tsinghua University

발표 : 2025년 9월 28일에 arXiv

논문 : [PDF](https://arxiv.org/pdf/2509.23582)


---

*QAT*

| 레이어 구분 | 가중치 비트 폭 (Weight) | 활성화 비트 폭 (Activation) | 비고 |
| :--- | :--- | :--- | :--- |
| **일반 선형 레이어**<br>(qkv, proj, fc1, fc2) | 1.58-bit (Ternary) | 1~4 bit (가변) | AMPN 알고리즘을 통해 레이어별로 최적의 비트를 할당합니다. |
| **SVD 저역통과 분기**<br>(Low-rank branch) | 32-bit (FP) | 32-bit (FP) | 양자화로 인한 정보 손실을 보상하는 보조 경로입니다. |
| **Attention Scores**<br>(A-A Matrix Mult) | 8-bit (연산 정밀도) | 8-bit | 활성화 값 간의 행렬 곱으로, 높은 정밀도가 필요한 핵심 구간입니다. |
| **adaLN-Zero 레이어** | 4-bit | 4-bit | 모델 안정성을 위해 4-bit로 고정하여 처리합니다. |
| **임베딩 및 최종 레이어** | 32-bit (FP) | 32-bit (FP) | 입출력 데이터의 품질을 위해 양자화를 적용하지 않습니다. |

---

