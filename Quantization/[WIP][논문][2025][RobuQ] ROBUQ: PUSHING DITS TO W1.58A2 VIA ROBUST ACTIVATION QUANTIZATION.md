# ROBUQ: PUSHING DITS TO W1.58A2 VIA ROBUST ACTIVATION QUANTIZATION

저자 : Kaicheng Yang1∗, Xun Zhang1∗, Haotong Qin2,

Yucheng Lin1, Kaisen Yang3, Xianglong Yan1, Yulun Zhang1†

1Shanghai Jiao Tong University, 2ETH Zurich, 3Tsinghua University

발표 : 2025년 9월 28일에 arXiv

논문 : [PDF](https://arxiv.org/pdf/2509.23582)


---

*QAT*

레이어 구분,활성화 비트 폭 (Activation Bit-width),비고
"일반 선형 레이어 (qkv, proj, fc1, fc2)",1-bit ~ 4-bit (가변) ,평균 2-bit(A2) 또는 3-bit(A3) 목표치에 맞춰 DP 알고리즘으로 자동 할당됩니다.+1
Attention Scores (A-A Matrix Mult),8-bit +3,매우 민감하지만 전체 연산량에서 차지하는 비중이 작아 높은 정밀도를 유지합니다.+2
adaLN-Zero 레이어,4-bit +1,안정적인 학습을 위해 4비트로 고정합니다.+1
임베딩 및 최종 레이어,32-bit (FP) +1,가중치와 마찬가지로 활성화 값도 양자화하지 않습니다.

---

