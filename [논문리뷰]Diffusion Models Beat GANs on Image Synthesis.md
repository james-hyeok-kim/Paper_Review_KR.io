# Diffusion Models Beat GANs on Image Synthesis
저자 : Prafulla Dhariwal∗ OpenAI, Alex Nichol∗ OpenAI

출간 : NeurIPS, 2021.

논문 : [PDF](https://arxiv.org/abs/2105.05233)

---

### Abstract
* Classifier Guidance라는 새롭고 효율적인 방법 도입

* ImageNet
  * 128x128 : FID 2.97
  * 256x256 : FID 4.59
  * 512x512 : FID 7.72

* Classifier Guidance + upsampling
  * 256x256 : FID 3.94
  * 512x512 : FID 3.85
 
### Introduction

#### GAN 한계
* 학습이 불안정
* 다양성 부족

#### Likelihood 모델
* VAE or Diffusion (Likelihood-based) Model 다양성 우수
* 학습 안정적
* 샘플 Quality는 GAN보다 뒤처짐

#### 연구제안
* 모델 아키텍처 개선 - unconditional 이미지 생성
* Classifier Guidance  도입 - 다양성 조절 및 품질 극대화


### Background
#### Forward (Noising)
* 가우시안 노이즈 추가
#### Backward (Denoising)
* 노이즈 제거, 이미지 생성
#### Training
* 손실함수 MSE
* $\left\| \epsilon_\theta(x_t, t) - \epsilon \right\|^2$
#### Sampling
* Backward process에서 진행
* $x_t$ 에서 노이즈 $\epsilon_\theta$를 예측하며 $x_{t-1}, \mu_\theta$ 를 계산
#### Main Improvement
* Learned Variance
  * DDPM에서는 분산이 상수로 고정했지만 이 논문은 분산을 학습 하도록 했음
  * 50단계 미만의 샘플링은 DDIM 방식을 채택
#### 성능 평가 지표
* FID(Frechet Inception Distance) : 이미지 품질과 다양성 판단
* IS(Inception Score) : 특정 클래스 분류 되는지 품질과 다양성 측정
* Precision and Recall : 품질과 다양성 분리하여 측정



