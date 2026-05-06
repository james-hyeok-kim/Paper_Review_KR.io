# SenCache: Accelerating Diffusion Model Inference via Sensitivity-Aware Caching

저자 : 

Yasaman Haghighi, Alexandre Alahi

Ecole Polytechnique F ´ ed´ erale de Lausanne (EPFL) ´

yasaman.haghighi@epfl.ch alexandre.alahi@epfl.ch

발표 : 2026년 2월 arXiv

논문 : [PDF](https://arxiv.org/pdf/2602.24208)

---

## 0. Summary

<p align = 'center'>
<img width="751" height="403" alt="image" src="https://github.com/user-attachments/assets/d8742909-c1ea-4887-99c0-a08259067ce1" />
</p>

### 0.1 핵심 아이디어

* 비디오 확산 모델(Video Diffusion Model)의 추론 속도를 높이기 위한 민감도 인식 캐싱(Sensitivity-Aware Caching, SenCache) 프레임워크를 제안합니다.  
* 기존의 경험적(Heuristic) 캐싱 방식과 달리, 노이즈가 포함된 잠재 변수(noisy latent)와 타임스텝(timestep)의 섭동(perturbation)에 대한 네트워크의 국소적 민감도(local sensitivity)를 이론적 기준으로 삼아 캐싱을 수행합니다.  
* 자코비안 노름(Jacobian norm) 기반의 1차 민감도 근사치를 통해 출력 변화량을 예측하고, 이 점수가 허용 오차보다 작을 때만 이전 단계의 연산 결과를 재사용합니다.  

### 0.2. 효과

* 동일한 연산 예산(NFE 기준) 하에서 기존의 캐싱 방법(TeaCache, MagCache)보다 비디오의 시각적 품질(LPIPS, PSNR, SSIM)을 더 높게 보존합니다.  
* 고정된 스케줄이 아닌, 각 샘플의 복잡도와 동적 특성에 맞춰 캐시 재사용 여부를 동적으로 결정할 수 있습니다.  
* 추가적인 모델 학습이나 구조 변경 없이(training-free) 적용이 가능합니다. 

캐싱 적용 전 NFE (Baseline NFE)실험에 사용된 세 가지 모델(Wan 2.1, CogVideoX, LTX-Video) 모두 캐싱 적용 전 기본 NFE(총 타임스텝 수)는 50이었습니다.  

* SenCache는 특정 내부 레이어(예: Attention 층이나 MLP 층)의 결과를 저장하는 것이 아니라, 노이즈 제거 네트워크(denoiser network)의 전체 결과물(Full-Forward Caching)을 저장하고 재사용합니다.
    * 즉, 이전 단계에서 연산한 전체 모델의 예측 출력값인 $f_\theta(x_t, t, c)$ 자체를 캐싱합니다.
* 캐싱을 적용한 타임스텝 (Timestep)특정 타임스텝을 미리 고정해두고 스킵하는 것이 아니라, 매 단계마다 계산된 민감도 점수(Sensitivity score)에 따라 동적으로(adaptively) 재사용 여부를 결정합니다.
    * 다만, 초기 노이즈 제거 단계가 최종 품질에 매우 중요하기 때문에 처음 20%의 타임스텝 구간에서는 오차 허용치( $\epsilon$ )를 0.01로 매우 엄격하게 설정하여 연산을 최대한 수행(캐싱 최소화)하도록 했습니다.
    * 그 이후의 나머지 80% 구간에서는 모델에 따라 오차 허용치를 0.1~0.6으로 더 높게(느슨하게) 설정하여 적극적으로 연산을 스킵했습니다.



| 모델 (Model) | 방법론 (Method) | NFE (↓) | 캐시 비율 (Cache Ratio) |
| :--- | :--- | :---: | :---: |
| **Wan 2.1** | TeaCache-slow | 33 | 34% |
| (T=50) | MagCache-slow | 25 | 50% |
| | **SenCache-slow (Ours)** | **25** | **50%** |
| | TeaCache-fast | 25 | 50% |
| | MagCache-fast | 21 | 58% |
| | **SenCache-fast (Ours)** | **21** | **58%** |
| **Cog VideoX** | TeaCache | 22 | 56% |
| (T=50) | MagCache | 23 | 54% |
| | **SenCache (Ours)** | **22** | **56%** |
| **LTX-Video** | TeaCache | 32 | 36% |
| (T=50) | MagCache | 28 | 44% |
| | **SenCache (Ours)** | **27** | **46%** |

### 0.3. 적용 모델

* 제안된 알고리즘은 최신 비디오 확산 모델(DiT 기반)인 Wan 2.1, CogVideoX, LTX-Video 모델에 적용되어 평가되었습니다.  

### 0.4. 벤치마크

* VBench: 주요 성능 평가를 위해 전체 프롬프트 세트를 활용했습니다.  
* T2V-CompBench: 파라미터 절제 연구(Ablation study)를 위해 무작위로 70개의 프롬프트를 선정하여 비디오를 생성했습니다.  
* MixKit 데이터셋: 모델의 민감도 점수를 캘리브레이션하기 위해 8개의 비디오 샘플을 사용했습니다.
