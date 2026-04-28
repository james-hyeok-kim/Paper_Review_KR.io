# PixelGen: Pixel Diffusion Beats Latent Diffusion with Perceptual Loss

저자 : Zehong Ma 1 Ruihan Xu 1 Shiliang Zhang 1

1State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University.

Project Page: https://zehong-ma.github.io/PixelGen

발표 : 2026년 2월 (arXiv 공개일: 2026년 2월 2일).

논문 : [PDF](https://arxiv.org/pdf/2602.02493)

---

## 0. Summary

<p align = 'center'>
<img width="361" height="561" alt="image" src="https://github.com/user-attachments/assets/fc84e660-74cb-43c1-9094-72e2e83e46c1" />
</p>

### 0.1  핵심 아이디어 (Core Idea)

* 엔드투엔드 픽셀 디퓨전: VAE를 사용하는 2단계 잠재 디퓨전(Latent Diffusion)과 달리, 픽셀 공간에서 직접 이미지를 생성하여 VAE의 아티팩트와 성능 병목 현상을 제거했습니다.
* 지각적 매니폴드(Perceptual Manifold) 집중: 고차원의 복잡한 전체 이미지 매니폴드 대신, 인간의 지각에 중요한 신호가 담긴 '지각적 매니폴드'를 학습하도록 유도합니다.
* 지각적 손실(Perceptual Loss) 도입: $x$-prediction 방식을 채택하여 다음 두 가지 보완적인 손실 함수를 적용했습니다:
    * LPIPS Loss: 국소적 텍스처와 세부 디테일 강화.
        * VGG 네트워크의 다층 특징 활성화를 비교하여 이미지의 국소적 패턴과 텍스처 유사성을 측정합니다.        
    * P-DINO Loss: DINOv2 특징 정렬을 통한 전역적 시맨틱 및 구조 일관성 확보.
        * PixelGen에서 새롭게 제안한 P-DINO(Perceptual DINO) 손실은 DINOv2 인코더의 패치 단위 특징을 정렬하여 전역적인 장면 배치와 객체의 시맨틱 일관성을 강화

* LPIPS Loss
    * $l$: VGG 네트워크의 레이어 인덱스입니다.
    * $f_{VGG}^{l}$: 동결된(frozen) 사전 학습 VGG 네트워크에서 추출한 $l$번째 레이어의 특징 맵입니다.
    * $w_{l}$: 해당 레이어의 채널별 가중치 벡터입니다.
    * $x_{\theta}$: 디퓨전 모델이 예측한 이미지( $x$-prediction)입니다.
    * $x$: 실제 정답(Ground-truth) 이미지입니다.

$$\mathcal{L}_{LPIPS}=\sum_{l}||w_{l}\odot(f_{VGG}^{l}(x_{\theta})-f_{VGG}^{l}(x))||_{2}^{2}$$

* P-DINO Loss
    * $f_{DINO}^{p}$: 동결된 DINOv2-B 인코더에서 추출한 $p$번째 패치의 특징 벡터입니다.
    * $\mathcal{P}$: 이미지 내의 모든 패치 집합입니다.
    * $cos(\cdot, \cdot)$: 두 특징 벡터 사이의 코사인 유사도(Cosine Similarity)를 의미합니다.

$$\mathcal{L}_{P\cdot DINO}=\frac{1}{|\mathcal{P}|}\sum_{p\in\mathcal{P}}(1-cos(f_{DINO}^{p}(x_{\theta}),f_{DINO}^{p}(x)))$$

* Total Loss
    * PixelGen은 기존의 플로우 매칭(Flow Matching) 손실에 위 두 지각적 손실과 표현 정렬 손실(REPA)을 결합하여 최종 학습을 진행합니다.
    * $\lambda_{1}, \lambda_{2}$: 디퓨전 목적 함수와 지각적 감독 신호 사이의 균형을 맞추는 하이퍼파라미터입니다.
    * 논문의 실험 결과, $\lambda_{1}=0.1$, $\lambda_{2}=0.01$을 사용할 때 가장 좋은 성능(Trade-off)을 보였습니다.

$$\mathcal{L}=\mathcal{L}_{FM}+\lambda_{1}\mathcal{L}_{LPIPS}+\lambda_{2}\mathcal{L}_{P-DINO}+\mathcal{L}_{REPA}$$


### 0.2. 효과 (Effect)

* 효율적 학습: 단 80 epoch 학습만으로도 수백 epoch를 학습한 기존의 강력한 잠재 디퓨전 모델(REPA, DDT 등)보다 우수한 FID 성능을 기록했습니다.
* 품질 향상: VAE 재구성 과정에서의 정보 손실이 없어 더 선명한 텍스처와 정확한 전역 구조를 생성합니다.
* 단순화된 파이프라인: VAE, 잠재 표현, 보조 단계가 필요 없는 단순한 생성 패러다임을 제공합니다.

### 0.3. 수치적 성능 및 모델 (Quantitative Results)

* 적용 모델 (Model Variants):
    * PixelGen-L/16: 459M 파라미터.
    * PixelGen-XL/16: 676M 파라미터.
    * PixelGen-XXL/16: 1.1B 파라미터 (Text-to-Image용).
* 주요 수치 성능 (ImageNet-256, CFG 미사용 기준):
    * FID: 5.11 (80 epoch 기준) - REPA-XL/2(5.90, 800 epoch) 및 DDT-XL/2(6.27, 400 epoch)를 능가.
    * Precision/Recall: 각각 0.72 / 0.63 기록.

### 0.4. 벤치마크 핵심

* Class-to-Image (ImageNet-256): CFG 없이도 잠재 디퓨전 모델의 성능 상한을 넘어서며 픽셀 디퓨전의 경쟁력을 입증했습니다.
* Text-to-Image (GenEval): Qwen3-1.7B를 텍스트 인코더로 사용해 0.79점을 기록, FLUX.1-dev(12B) 등 훨씬 거대한 모델들과 대등한 수준의 정렬 성능을 보여주었습니다.
* Ablation Study: LPIPS와 P-DINO 손실이 각각 국소/전역 품질에 기여함을 확인했으며, 고노이즈 단계에서는 지각적 손실을 끄는 Noise-Gating 전략이 다양성(Recall) 유지에 필수적임을 밝혔습니다.  


---
