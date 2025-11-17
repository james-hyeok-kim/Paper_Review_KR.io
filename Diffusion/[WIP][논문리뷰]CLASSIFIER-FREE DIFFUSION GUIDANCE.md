# CLASSIFIER-FREE DIFFUSION GUIDANCE
저자 : Jonathan Ho & Tim Salimans Google Research, Brain team

출간 : NeurIPS, 2022.

논문 : [PDF](https://arxiv.org/pdf/2207.12598)

---

### Abstract

* Classifier 없이 Diffusion Model 품질 향상

#### 배경

* Classifier Guidance는 Classifier를 따로 훈련 시켜야 한다는 단점
* Classifier 없이 Guidance 수행할 수 있는가?
* Conditional & Unconditional Diffusion Model을 Jointly Train

---

### Introduction

#### 기존 문제점

* 필요성: 생성 모델(예: BigGAN, Glow)에서는 샘플의 다양성을 조금 희생하더라도 개별 샘플의 품질을 높이기 위해 '(low temperature sampling)'이나 '(truncation)' 기법을 사용
* Diffusion Model 한계: 점수 벡터를 스케일링하거나 노이즈를 줄이는 단순한 방식으로는 이러한 효과를 얻을 수 없음

#### Classifier Guidance 해결책

* 단점 및 의문점
  * 복잡성 : 훈련 복잡성 (Classifier)
  * Adversarial Attack 의혹 : 이미 분류기를 속이려는 적대적 공격과 유사
    * IS / FID가 이미지가 좋아저서 좋아진건지 단순 분류기 지표를 속여서 좋아진 것인지 에 대한 의문

---
