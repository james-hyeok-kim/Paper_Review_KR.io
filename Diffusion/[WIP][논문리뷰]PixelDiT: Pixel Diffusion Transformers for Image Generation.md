# PixelDiT: Pixel Diffusion Transformers for Image Generation

저자 : Yongsheng Yu1,2 * Wei Xiong1 † Weili Nie1 Yichen Sheng1 Shiqiu Liu1 Jiebo Luo2 / 1 NVIDIA, 2 University of Rochester, † Project Lead and Main Advising

출간 : CVPR(	Computer Vision and Pattern Recognition), 2025

논문 : [PDF](https://arxiv.org/pdf/2511.20645)

---

## 1. Introduction

#### 1. 연구 배경: 잠재 확산 모델(LDMs)의 구조적 한계

* 목표 불일치: 확산 모델이 별도로 학습된 오토인코더(Autoencoder)에 의존하는데, 오토인코더의 '재구성 목표'와 생성 모델의 '생성 목표'가 완전히 일치하지 않습니다.
* 손실 재구성(Lossy Reconstruction): 오토인코더의 압축 과정에서 고주파 디테일(예: 텍스트, 미세한 질감)이 손실됩니다. 이로 인해 확산 모델 성능이 아무리 좋아도 최종 이미지 품질에 한계가 생깁니다


#### 2. 문제 제기: 픽셀 공간 확산의 어려움

* 오토인코더 없는 픽셀 공간 확산을 다시 도입하려 했으나
* 픽셀 모델링(Pixel Modeling)이라는 핵심 난관에 봉착

* 트레이드오프(Trade-off)
    * 큰 패치를 사용하면 연산은 줄지만 디테일 표현이 약해집니다.
    * 작은 패치를 사용하면 디테일은 살지만, 연산 비용이 제곱(Quadratic)으로 증가하여 학습이 너무 느려집니다.
* 즉, 글로벌한 의미(Semantics)와 픽셀 단위의 세밀한 업데이트를 동시에 효율적으로 처리할 메커니즘이 부족했습니다


#### 3. 해결책: PixelDiT (Dual-level Architecture)

* 이중 레벨 설계 (Dual-level Design): 이미지의 의미 학습과 픽셀 학습을 분리했습니다.
    * Patch-level DiT: 큰 패치 단위를 사용하여 이미지의 전반적인 구조와 내용을 학습합니다.
    * Pixel-level DiT: 픽셀 단위의 미세한 텍스처를 정교하게 다듬습니다.
* 핵심 기술
    * 이를 효율적으로 구현하기 위해 픽셀 단위 AdaLN 변조(Pixel-wise AdaLN modulation)와
    * 픽셀 토큰 압축(Pixel token compaction) 기술을 도입했습니다.

#### 4. 주요 성과 및 기여

* 압도적 성능: ImageNet 256x256에서 FID 1.61을 기록하며 기존 픽셀 생성 모델들을 크게 앞섰습니다
* 확장성: 텍스트-이미지 생성(Text-to-Image)으로 확장하여 $1024^2$ 해상도(메가픽셀 단위) 학습에 성공했습니다
* 편집 용이성: VAE의 압축 손실이 없기 때문에, 이미지 편집 시 배경의 작은 텍스트나 디테일이 뭉개지지 않고 잘 보존됩니다

---


