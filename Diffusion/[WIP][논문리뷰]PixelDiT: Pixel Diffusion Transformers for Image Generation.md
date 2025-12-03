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

## 2. Related Work

<p align = 'center'>
<img width="813" height="330" alt="image" src="https://github.com/user-attachments/assets/b73d4b6e-ec33-42ea-bfd2-092780a4f356" />
</p>

### 2.1. Latent Diffusion Models with Autoencoders

#### 1. LDM의 표준 방식과 장점
* 작동 원리: LDM은 오토인코더(Autoencoder)를 사용해 이미지를 압축된 표현 공간(Latent space)으로 변환한 뒤, 그 공간에서 노이즈 제거(Denoising)를 수행합니다.
* 이점: 이 방식은 연산량과 메모리 사용량을 크게 절약해 주며, 한정된 자원 내에서도 더 큰 백본 모델을 사용해 고해상도 학습을 가능하게 합니다.
* 일반적 구조: 대부분의 LDM은 재구성 정확도와 압축률 사이의 균형을 맞추기 위해 변분 오토인코더(VAE)를 사용합니다.


#### 2. 왜 다시 픽셀 공간인가? (LDM의 한계)

* 효율성에도 불구하고 LDM은 오토인코더에 의존함으로 인해 발생하는 명확한 병목 현상(Bottleneck)을 가지고 있습니다.


* 재구성 병목 (Reconstruction Bottleneck)
    * 생성되는 샘플의 충실도(Fidelity)가 오토인코더의 성능에 의해 제한됩니다.
    * 과감한 압축 과정에서 고주파 디테일이나 미세한 구조가 제거되는 경향이 있습니다.


* 학습 비용 증가
    * 대규모 오토인코더를 사전에 학습하거나 함께 학습시키는 것은 픽셀 공간 학습에 비해 추가적인 데이터와 연산 비용을 발생시킵니다.


* 목표 불일치 (Objective Misalignment)
    * 오토인코더의 '재구성 목표'와 확산 모델의 '생성 목표'가 완전히 일치하지 않습니다.
    * 이로 인해 잠재 공간 내에서 분포 변화(예: 텍스처가 뭉개지거나 색상이 변함)가 발생하고, 확산 모델이 이를 보정해야 하는 부담을 안게 됩니다.


* 추론 지연
    * 샘플링 단계에서 잠재 표현을 다시 이미지로 변환(Decoding)하는 과정이 추가적인 시간 비용을 발생시킵니다.


### 2.2. Pixel-Space Diffusion Models

#### 1. 초기 역사와 핵심 난관초기 연구

* 픽셀 공간 확산 모델(Diffusion)은 LDM보다 먼저 연구되었으며, 픽셀 공간에서 직접 노이즈를 제거하여 고화질 이미지를 생성할 수 있음을 입증했습니다

* 핵심 난관 (계산 비용): 하지만 이미지 해상도가 높아질수록 연산량과 메모리 비용이 제곱(Quadratic)으로 증가하는 문제가 있어, 메가픽셀(Megapixel) 단위의 고해상도 이미지를 엔드투엔드(End-to-end)로 학습하는 것은 비용적으로 거의 불가능했습니다
    * Quadratic 이유: Self-Attention


#### 2. 최근의 시도들 (PixelDiT 이전의 접근법)

* JetFormer: 원본 픽셀과 텍스트에 대해 자기회귀(Autoregressive) 방식을 적용했습니다
* Simple Diffusion: 스킵 연결(Skip connections)이 있는 효율적인 합성곱 신경망(CNN) 구조를 제안했습니다
* Fractal Generative Models: 장거리 구조 학습을 위해 프랙탈 설계를 도입했습니다5.PixelFlow & PixNerd: 계층적 흐름(Hierarchical flow) 모델이나 경량화된 신경 필드(Neural field) 레이어를 사용하여 효율성을 높였습니다6.동시대 연구 (Concurrent Works):EPG: 자기 지도 사전 학습(Self-supervised pre-training)과 생성 미세 조정(Generative fine-tuning)을 연결하는 2단계 프레임워크를 채택했습니다7.FARMER: 고차원 픽셀 처리를 위해 자기회귀 모델링과 정규화 흐름(Normalizing flows)을 결합했습니다8.JiT: 일반적인 트랜스포머(Plain Transformers)가 고차원 데이터를 효율적으로 모델링할 수 있음을 보여주었습니다9.


3. PixelDiT의 차별점위의 연구들과 달리, PixelDiT는 다음과 같은 독보적인 특징을 가집니다.순수 트랜스포머 기반: 복잡한 하이브리드 구조 대신 순수한 트랜스포머 아키텍처를 사용합니다10.직접적인 고해상도 학습: 별도의 단계나 우회적인 방법 없이, $1024^2$ 해상도에서 직접 엔드투엔드 학습이 가능한 픽셀 공간 확산 모델을 제시합니다11.

---










