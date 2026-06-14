# DySL-VLA: Efficient Vision-Language-Action Model Inference via Dynamic-Static Layer-Skipping for Robot Manipulation

저자 : 

Zebin Yang1,2, Yijiahao Qi4, Tong Xie1,2, Bo Yu3∗, Shaoshan Liu3, Meng Li1,2,5∗

1 Institute for Artificial Intelligence

2School of Integrated Circuits, Peking University, Beijing, China

3Shenzhen Institute of Artificial Intelligence and Robotics for Society, Shenzhen, China

4School of Electronics Engineering and Computer Science, Peking University, Beijing, China

5Beijing Advanced Innovation Center for Integrated Circuits, Beijing, China

발표 : 2026년 3월 17일 arXiv

논문 : [PDF](https://arxiv.org/pdf/2602.22896)


---

## 0. Summary

### Layer Skip을 통해 속도를 높임, 오차보정 Adapter와 Controller만 재학습 하여 효율성 높임

* 동적-정적 레이어 스킵 (Dynamic-Static Layer Skipping): 모든 레이어가 동일하게 중요하지 않다는 관찰에 기반하여, 정보량이 많은 '정적 레이어'는 항상 유지하고, 정보량이 적은 '동적 레이어'는 상황에 따라 선택적으로 건너뜁니다.
* 사전-사후 스킵 가이드 (Prior-Post Skipping Guidance): 로봇의 움직임이 매끄러운(Continuity가 높은) 비임계 상황에서는 레이어를 많이 건너뛰고, 물체를 잡거나 놓는 등 정교한 동작이 필요한 임계 상황에서는 레이어를 더 많이 활성화하여 정확도를 보장합니다.
* 스킵 인지 2단계 지식 증류 (Skip-aware Two-stage Knowledge Distillation): 전체 백본을 재학습하는 대신, 레이어를 건너뛴 후의 오차를 보정하는 가벼운 어댑터(Adapter)와 스킵 여부를 결정하는 컨트롤러만을 효율적으로 학습시킵니다.

### 연구의 의의

* 실시간성 확보: 기존 RoboFlamingo 대비 3.75배의 추론 속도 향상을 달성하여, 저사양 임베디드 장치(Jetson Orin 등)에서도 20Hz 이상의 제어 주기를 확보할 수 있음을 보여주었습니다.
* 효율적인 학습: LLM 백본을 동결한 채 가벼운 모듈만 학습시키므로, 기존 방식(DeeR-VLA 등) 대비 학습 파라미터 수를 85.7배 줄이고 학습 시간도 대폭 단축했습니다.
* 성능 유지: 속도를 높이면서도 Calvin 데이터셋에서 오히려 성공률(Success Length)을 2.1% 향상시키는 등, 효율성과 정확도의 트레이드오프를 성공적으로 극복했습니다.

### Model Architecture

$$ \text{Vision + Text Encoder} \rightarrow \text{LLM(Llama-2-7B)} \rightarrow \text{Action Head}$$

* Action Head
    * no Diffusion
    * no EMB
    * Direct Prediction Explicit policy

---
