# TerDiT: Ternary Diffusion Models with Transformers

저자 : Xudong Lu, Aojun Zhou, Ziyi Lin, Qi Liu, Yuhui Xu, Renrui Zhang, Xue Yang, Member, IEEE,

Junchi Yan, Senior Member, IEEE, Peng Gao, Hongsheng Li, Member, IEEE

발표 : 2025년 4월 6일에 arXiv

논문 : [PDF](https://arxiv.org/pdf/2405.14854)

---

## 1. Introduction

---

*QAT*


레이어,가중치 (Weight),활성화값 (Activation),비고
"Self-attention, Feedforward, MLP 선형 레이어 +1","삼진 (Ternary, 1.58-bit) +4",Full-precision (FP) ,가중치 전용(Weight-only) 양자화 방식을 사용함.+1
adaLN 모듈 내 MLP 레이어 +3,"삼진 (Ternary, 1.58-bit) ",Full-precision (FP) ,삼진 양자화로 인한 불안정성을 해결하기 위해 레이어 뒤에 RMS Norm을 추가함.+2

---
