# DeepCache: Accelerating Diffusion Models for Free

저자 :

* Xinyin Ma Gongfan Fang Xinchao Wang*

* National University of Singapore

* {maxinyin, gongfan}@u.nus.edu, xinchao@nus.edu.sg

발표 : CVPR 2024

논문 : [PDF](https://arxiv.org/pdf/2312.00858)

---

## 0. Summary

<p align = 'center'>
<img width="542" height="432" alt="image" src="https://github.com/user-attachments/assets/b26c688c-fbd2-483c-bdb7-c342e68e20ba" />
</p>

### 1. 핵심 아이디어 (Key Idea)

* DeepCache는 추가적인 학습(Retraining) 없이 Diffusion 모델의 추론 속도를 획기적으로 높이는 Training-free 가속화 알고리즘.
* 인접 단계의 특징 일관성 (Temporal Consistency): Diffusion 디노이징(Denoising) 과정에서 인접한 스텝 간에 고수준(High-level) 특징들이 매우 유사.
* U-Net 구조 활용: U-Net의 메인 브랜치에서 생성되는 고수준 특징은 캐시(Cache)에 저장하여 다음 스텝에서 재사용, 상대적으로 계산량이 적은 Skip 브랜치의 저수준(Low-level) 특징만 매 스텝 업데이트
    * 1:N 전략: 한 번의 Full Inference 이후 $N-1$번의 Partial Inference를 수행하는 방식입니다.
        * Uniform 1:N: 일정한 간격으로 캐시를 업데이트합니다.
        * Non-uniform 1:N: 특징 유사도가 급격히 변하는 특정 구간에서 더 자주 업데이트하여 품질 저하를 방지합니다.

### 2. 실험 결과 (Experimental Results)

* Stable Diffusion v1.5: CLIP Score 하락을 0.05 수준으로 억제하면서 2.3배의 속도 향상을 달성했습니다.
* LDM-4-G (ImageNet): FID가 0.22 정도만 소폭 증가하는 조건에서 4.1배의 가속이 가능하며, 설정을 통해 최대 10배 이상의 가속도 보여주었습니다.

   
---

## 1. Introduction

### 2. 기존 압축 방식의 문제점

* 대부분 대규모 데이터셋을 활용한 재학습(Retraining) 과정을 필수적으로 수반.
* 이러한 재학습 과정은 비용이 많이 들고, Stable Diffusion과 같은 대규모 사전 학습 모델에 적용하기에는 비실용적인 경우가 많습니다.

### 3. DeepCache의 핵심 발견: 시간적 중복성 (Temporal Redundancy)

* 연구진은 디노이징 과정의 인접한 스텝들 사이에서 고수준(High-level) 특징들이 매우 유사하게 유지된다는 점에 주목했습니다.
* 실험 결과, 어떤 타임스텝이든 인접한 스텝의 최소 10%는 현재 스텝과 0.95 이상의 높은 유사도를 보인다는 사실을 확인했습니다.
* 따라서 매번 비슷한 특징 맵을 생성하기 위해 막대한 연산 자원을 쏟아붓는 것은 비효율적이라고 판단했습니다.

### 4. 제안하는 해결책: 학습이 필요 없는 가속화

* DeepCache는 모델 아키텍처 관점에서 접근하는 Training-free(학습이 필요 없는) 가속화 패러다임입니다.
* U-Net 구조 활용: U-Net의 메인 브랜치에서 생성된 고수준 특징은 캐싱하여 재사용하고, 계산량이 매우 적은 Skip 브랜치를 통해 저수준(Low-level) 특징만 매 스텝 업데이트합니다.

### 5. 주요 기여 및 성과 (Contributions)

* 성능: Stable Diffusion v1.5에서 CLIP Score 하락을 최소화(0.05)하면서 2.3배의 속도 향상을, LDM-4-G에서는 4.1배의 가속을 달성했습니다.
* 유연성: 기존의 빠른 샘플러(DDIM, PLMS 등)와 호환되며, 재학습이 필요한 기존 방식들보다 더 우수한 효율성을 입증했습니다.
* 전략: 긴 캐싱 간격에서도 품질을 유지하기 위한 비균일(Non-uniform) 1:N 전략을 도입했습니다.

---

## 2. Related Work 

### 1. 이미지 생성 모델의 발전 배경

* 과거 모델:
    * 초기에는 GAN(Generative Adversarial Networks) 과 VAE(Variational Autoencoders) 가 주도했으나,
    * 학습 불안정성이나 모드 붕괴(Mode Collapse)와 같은 확장성 문제에 직면했습니다.
* 디퓨전 모델의 등장:
    * 최근에는 디퓨전 확률 모델(Diffusion Probabilistic Models)이 뛰어난 이미지 생성 품질을 제공하며 이 분야를 선도하고 있습니다.
    * 하지만 역방향 디퓨전 과정의 본질적인 특성 때문에 추론 속도가 매우 느리다는 단점이 있습니다.

### 2. 가속화 전략의 두 가지 갈래

현재 연구들은 추론 속도를 높이기 위해 다음 두 가지 방법 중 하나에 집중하고 있습니다:

#### A. 샘플링 효율성 최적화 (Optimized Sampling Efficiency)

디노이징을 위해 필요한 샘플링 단계(Step)의 총 횟수를 줄이는 방식입니다.

* 주요 기법:
    * DDIM: 비마르코프(non-Markovian) 과정을 탐구하여 스텝 수를 줄임.
    * 고성능 솔버: SDE 또는 ODE의 빠른 수치 해석 솔버를 사용하여 샘플링 효율화.
    * 증류 및 모델 변환: 점진적 증류(Progressive Distillation) 나 컨시스턴시 모델(Consistency Model) 을 통해 단 한 번의 평가로 이미지를 생성함.
    * 병렬 샘플링: 푸리에 신경 연산자 등을 활용해 병렬로 디코딩을 수행함.

#### B. 구조적 효율성 최적화 (Optimized Structural Efficiency)

각 샘플링 단계에서 발생하는 모델의 추론 오버헤드 자체를 줄이는 방식입니다.

* 주요 기법:
    * 구조적 가지치기(Pruning): 네트워크 아키텍처를 재설계하거나 불필요한 레이어를 제거함.
    * 양자화(Quantization): 가중치와 활성함수를 낮은 정밀도의 데이터 타입으로 변환함.
    * 조기 종료(Early Stopping): 특정 단계에서 추론을 미리 종료하거나 타임스텝마다 다른 sub-network를 사용함.
    * 입력 최적화: 토큰 병합(Token Merging)을 통해 어텐션 모듈의 계산 효율을 높임.

### 3. DeepCache의 위치와 차별점

* 카테고리: DeepCache는 두 번째 갈래인 '단계당 평균 추론 시간 최소화'에 속합니다.
* 차별화된 가치: DeepCache는 재학습 없이도 매 스텝 실행되는 모델의 실질적인 크기를 획기적으로 줄여 디노이징 속도를 높입니다.


---

## 3. Methodology

### 3.1. Preliminary (기초 개념)

* Forward and Reverse Process: 데이터에 노이즈를 추가하는 과정과, 학습된 네트워크 $\epsilon_{\theta}(x_{t}, t)$를 사용하여 노이즈로부터 데이터를 복원하는 역과정을 설명합니다.

* U-Net의 특징 추출: U-Net은 다운샘플링( $D_i$ )과 업샘플링( $U_i$ ) 블록으로 구성되며, Skip Connection을 통해 저수준(Low-level) 정보와 고수준(High-level) 정보를 결합합니다.
    * 메인 브랜치: 이전 업샘플링 블록( $U_{i+1}$ )에서 가공된 고수준 특징을 제공합니다.
    * Skip 브랜치: 대칭되는 다운샘플링 블록( $D_i$ )에서 풍부한 저수준 특징을 직접 전달합니다.

$$Concat(D_i(\cdot), U_{i+1}(\cdot))$$

### 3.2. Feature Redundancy in Sequential Denoising (특징 중복성 관찰)

* 저자들은 디노이징 과정 전반을 재검토하여 효율화할 수 있는 지점을 찾아냈습니다.

* 관찰 결과: 시간적 유사성(Temporal Similarity)
* 데이터 기반 증거: Stable Diffusion, LDM 등 주요 모델에서 모든 타임스텝의 최소 10%는 인접 스텝과 0.95 이상의 유사도를 가집니다.
* 결론: 매 스텝 막대한 자원을 투입해 유사한 특징 맵을 다시 계산하는 것은 비효율적이

### 3.3. Deep Cache For Diffusion Models (핵심 알고리즘)

* 관찰된 중복성을 활용하여 실제로 추론을 가속하는 방법을 제안합니다.

#### 1. Cacheable Features (캐싱 메커니즘)

* 전략: 무거운 메인 브랜치의 연산 결과는 캐싱하여 재사용하고, 가벼운 Skip 브랜치 연산만 수행합니다 .
* 작동 방식:
    * Full Inference (스텝 $t$ ): 전체 U-Net을 실행하고 고수준 특징을 캐시에 저장합니다 ( $F_{cache}^{t} \leftarrow U_{m+1}^{t}(\cdot)$ )
    * Partial Inference (스텝 $t-1$ ): 전체 네트워크를 돌리지 않고, 선택된 $m$번째 Skip 브랜치까지만 계산한 뒤 캐시된 $F_{cache}^{t}$를 결합합니다.
* 이 방식은 $m=1$과 같이 얕은 레이어를 선택할수록 가속 효과가 극대화됩니다.

#### 2. Extending to 1:N Inference

* 캐시된 특징을 단 한 번이 아니라 연속된 $N-1$개의 스텝에서 재사용하여 효율성을 더 높입니다.
* 전체 $T$ 스텝 중 캐시를 업데이트하는 횟수는 $k = \lceil T/N \rceil$이 됩니다 .

#### 3. Non-uniform 1:N Inference (비균일 전략)

* 특징 유사도가 항상 일정하지 않으므로, 유사도가 급격히 변하는 구간(주로 디노이징 중반부)에서는 더 자주 업데이트하는 전략입니다 .
* 중심 타임스텝( $c$ )과 거듭제곱 계수( $p$ )를 활용한 수식을 통해 업데이트 스케줄을 동적으로 조정합니다 


---

## 4. Experiment

<p align = 'center'>
<img width="1058" height="475" alt="image" src="https://github.com/user-attachments/assets/9d7abc34-2c72-4d21-9eeb-d04d6b26d76a" />
</p>

### 4.1. Experimental Settings (실험 설정)

* 모델 및 샘플러: DDPM, LDM, Stable Diffusion 모델을 사용했습니다. 가속 효과를 극대화하기 위해 1000스텝 대신 DDIM(100/250스텝) 및 PLMS(50스텝)와 같은 빠른 샘플러 위에서 구동했습니다.
* 데이터셋: CIFAR10, LSUN(Bedroom/Churches), ImageNet, MS-COCO 2017, PartiPrompts 등 6개의 데이터셋을 활용했습니다.
* 평가 지표: 생성 품질 측정을 위해 FID, SFID, IS, Precision/Recall 및 CLIP Score를 사용했습니다.
* 비교 대상: 재학습이 필요한 기존의 압축 및 증류 방식인 Diff-Pruning, BK-SDMs 등을 대조군으로 설정했습니다.

### 4.2. Complexity Analysis (복잡도 분석)

* 가속 원리: U-Net의 Skip Connection을 기준으로 레이어를 분할하여, 인접 스텝에서 불필요한 연산을 제거(Layer removal)함으로써 속도를 높입니다.
* 모델별 연산 분포: Stable Diffusion은 레이어별 연산량이 비교적 균등하지만, DDPM은 초기 레이어에 계산량이 집중되어 있습니다. 본 실험에서는 모델별로 최적의 Skip Branch(3번/1번/2번)를 선택하여 가속했습니다.
* 하드웨어 성능: 모든 Throughput 측정은 단일 RTX 2080 GPU 환경에서 이루어졌습니다.

### 4.3. Comparison with Compression Methods (기존 압축 기법과의 비교)

* LDM-4-G (ImageNet): 4.1배 가속 시 FID 저하가 매우 적었으며(3.39 → 3.59), 재학습이 필요한 Pruning/Distillation 방식보다 생성 품질이 뛰어났습니다.
* DDPM (CIFAR-10/LSUN): 재학습 비용이 전혀 없음에도 불구하고, 재학습을 거친 모델들보다 더 좋은 성능을 보여주었습니다.
* Stable Diffusion: BK-SDM의 세 가지 변체(Base, Small, Tiny) 모두와 비교했을 때, 더 빠른 속도에서도 높은 CLIP Score를 유지하며 원본 모델과의 일관성이 더 높았습니다.

### 4.4. Comparison with Fast Samplers (빠른 샘플러와의 비교)

* 상호 보완적 관계: DeepCache는 빠른 샘플러와 대립하는 것이 아니라, 그 위에 추가로 적용할 수 있는 기술입니다.
* 성능 우위: 동일한 시간(Throughput)이 주어졌을 때, 단순히 스텝을 줄인 샘플러보다 DeepCache를 적용한 결과가 품질 면에서 더 우수했습니다.

### 4.5. Analysis & Ablation Study (분석 및 절제 실험)

* 캐시된 특징의 중요성: 캐시된 특징을 사용하지 않고 0(Zero) 행렬로 대체할 경우, FID가 급격히 나빠져 모델이 정상적으로 작동하지 않음을 확인했습니다.

* 얕은 네트워크 추론의 효과: DeepCache를 통해 수행되는 얕은(Shallow) U-Net 추론이 단순한 샘플링보다 더 나은 결과를 낸다는 점을 수치적으로 증명했습니다.

* 간격( $N$ )에 따른 변화: 캐싱 간격 $N$이 커질수록 시간은 단축되지만, 옷의 색상이나 동물의 형태 같은 세부 디테일에서 점진적인 변화가 발생합니다. $N < 5$일 때는 품질 하락이 매우 미미합니다.


---

## 5. Limitations

### 1. 사전 학습된 모델 구조에 대한 의존성

* DeepCache의 성능은 사용하려는 사전 학습된 디퓨전 모델의 기존 구조(pre-defined structure)에 전적으로 의존.
* 모델의 아키텍처를 직접 수정하는 것이 아니라 기존의 Skip Connection을 활용하는 방식이기 때문입니다.

### 2. 가속 효율의 제약 (Computation Balance)

* 모델의 가장 얕은(shallowest) Skip 브랜치가 전체 연산량에서 차지하는 비중이 클 경우 가속 효과가 제한적일 수 있습니다.
* 예를 들어, 가장 얕은 브랜치만 수행하더라도 전체 모델 계산량의 50%를 차지한다면, 이론적으로 달성 가능한 최대 속도 향상 폭은 작아질 수밖에 없습니다.

### 3. 큰 캐싱 간격( $N$ )에서의 성능 저하

* 캐싱 간격( $N$ )을 매우 크게 설정할 경우(예: $N=20$), 무시할 수 없는 수준의 성능 저하(Performance degradation)가 발생.
* 이는 결과적으로 알고리즘이 제공할 수 있는 가속 비율의 상한선(Upper limit)을 제한하는 요소가 됩니다.


---

## 6. Conclusion 

### 1. 새로운 가속화 패러다임 제시

### 2. 특징 유사성 및 구조적 속성 활용: 고수준(High-level) 특징의 유사성

### 3. Stable Diffusion v1.5의 경우 2.3배, LDM-4-G의 경우 4.1배 이상의 속도 향상

### 4. 기존 기법 대비 우수성 및 확장성

---



