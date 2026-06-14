# BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation

저자 : 

발표 : 

논문 : [PDF]()

---

## 0. Summary

* PTQ(Post-Training Quantization)아니라, 모델 설계와 최적화 단계에서부터 양자화를 통합한 '네이티브 1-bit VLA' 모델
* Quantize-then-Distill이라는 양자화 인식 학습(QAT) 전략

### BitVLA 양자화 적용 레이어 및 정밀도 요약

| 구성 요소 (Module) | 레이어 유형 (Layer Type) | Weight 정밀도 | Activation 정밀도 | 양자화 여부 |
| :--- | :--- | :--- | :--- | :--- |
| **Vision Encoder** (SigLIP-L) | 모든 선형 레이어 (Linear Layers) | 1.58-bit (Ternary) | INT8 (8-bit) | 적용 (Quantize-then-Distill) |
| **LLM Backbone** (BitNet b1.58) | 모든 선형 레이어 (Linear Layers) | 1.58-bit (Ternary) | INT8 (8-bit) | 적용 (Native 1-bit) |
| **Connector** (2-layer MLP) | MLP Layers | BF16 (Full Precision) | BF16 (Full Precision) | 제외 |
| **Action Head** (MLP) | Output Layers | BF16 (Full Precision) | BF16 (Full Precision) | 제외 |
| **Embeddings** | 입/출력 임베딩 (In/Out Embeddings) | BF16 (Full Precision) | BF16 (Full Precision) | 제외 |


---

## 1. Introductions



---

---
