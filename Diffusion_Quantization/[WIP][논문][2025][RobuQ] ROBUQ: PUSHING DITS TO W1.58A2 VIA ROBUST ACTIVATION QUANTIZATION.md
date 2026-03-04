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


| 레이어 이름 (Layer Name) | W1.58A2 (평균 2-bit 목표) | W1.58A3 (평균 3-bit 목표) | 비고 |
| :--- | :---: | :---: | :--- |
| **attn.proj** (Attention Projection) | **약 2.75 bit** | **약 3.89 bit** | 두 설정 모두에서 가장 높은 정밀도 유지 |
| **attn.qkv** (QKV Projection) | 약 2.21 bit | 약 2.96 bit | Attention 관련 레이어 우선 할당 |
| **mlp.fc1** (MLP First Layer) | 약 2.00 bit | 약 3.04 bit | 목표 평균치에 근접한 할당량 |
| **mlp.fc2** (MLP Second Layer) | 약 1.64 bit | 약 2.31 bit | 상대적으로 낮은 비트 할당 (예산 절감) |


<img width="457" height="638" alt="image" src="https://github.com/user-attachments/assets/2818e7cb-d7e1-43fc-910e-92249deb9bf6" />

<img width="1127" height="549" alt="image" src="https://github.com/user-attachments/assets/db7efff8-2123-4f3c-864e-55d0ac47aa80" />


---

