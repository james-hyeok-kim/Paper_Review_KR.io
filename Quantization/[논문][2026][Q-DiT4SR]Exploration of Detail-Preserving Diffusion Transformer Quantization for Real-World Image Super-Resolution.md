# Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer Quantization for Real-World Image Super-Resolution

저자 : Xun Zhang 1 Kaicheng Yang 1 Hongliang Lu 1 Haotong Qin 2 Yong Guo 3 Yulun Zhang∗ 1

1 Shanghai Jiao Tong University 

2ETH Zurich

3Huawei. 

발표 : 2026년 2월 1일 (arXiv 제출 기준)

논문 : [PDF](https://arxiv.org/pdf/2602.01273)

---

## 1. Introduction

### 1. 기존 양자화 방식의 한계

* 고주파 세부 정보 손실: 기존의 행렬 분해 방식은 모델 표현을 너무 단순화하여, 텍스처 복원에 필수적인 미세한 디테일을 보존하지 못합니다.

* 레이어별 특성 무시: 모델 레이어마다 민감도가 다름에도 불구하고, 데이터 없이 비트 폭(bit-width)을 유연하게 할당하는 메커니즘이 부족합니다.

* 시점별 동역학 무시: 확산 모델(Diffusion)의 시간 단계(timestep)별 활성화 데이터 특성이 변함에도 기존 방식은 고정된 정밀도만을 사용합니다.

### 2. 제안하는 솔루션: Q-DiT4SR

* H-SVD (Hierarchical SVD): 글로벌 저계수(low-rank) 분기뿐만 아니라 로컬 블록 단위의 분기를 통합하여 미세한 텍스처를 더 잘 보존합니다.

* VaSMP (Variance-aware Spatio Mixed Precision): 별도의 교정 데이터 없이(data-free) 레이어 간 가중치 비트를 효율적으로 할당합니다.

* VaTMP (Variance-aware Temporal Mixed Precision): 확산 시간 단계에 따라 활성화 정밀도를 동적으로 조절합니다

### 3. 연구의 의의

<p align = 'center'>
<img width="600" height="362" alt="image" src="https://github.com/user-attachments/assets/b9d7fd64-0b39-4a52-9cd4-f79196188362" />
<img width="610" height="374" alt="image" src="https://github.com/user-attachments/assets/0147523e-d6f5-4168-b78a-a226593b2427" />
</p>

* 제안된 방식은 W4A4(가중치 4비트, 활성화 4비트)라는 매우 공격적인 환경에서도 성능 저하를 최소화하면서 모델 크기를 5.8배 줄이고 연산량을 60배 이상 절감

---

## 2. Related Work

### 2.1. Image Super-Resolution

1) 딥러닝 기반 SR의 진화
    1) 초기 모델: 초해상도 기술은 초기 CNN에서 시작, Residual Learning 도입을 거쳐 Transformer 구조로 꾸준히 발전
    2) 실세계 초해상도(Real-ISR): GAN 기반 방식들이 지각적으로 사실적인 복원을 주도

2) 확산 모델(Diffusion Models)의 도입
     1) 최근에는 잠재 확산 모델(Latent Diffusion Models)을 활용한 방식이 큰 돌파구를 마련
     2) SOTA 기술: StableSR, DiffBIR, SeeSR과 같은 최신 확산 기반 SR 모델들은 의미론적 가이드(Semantic guidance)와 픽셀 단위 조건(Pixel-aware conditioning)을 결합하여 강력한 품질을 보여줍니다.

3) Diffusion Transformer (DiT)의 등장과 과제
    1) 최근 연구(DiT4SR, DreamClear 등)에 따르면, 모든 구조를 선형 레이어(Linear layers)와 Self-attention으로 구성한 DiT 아키텍처가 매우 뛰어난 성능
    2) 양자화의 난점: 전통적인 SR 모델들은 주로 컨볼루션 레이어(Convolutional layers)를 사용했기 때문에 기존의 양자화 기법들도 이에 최적화
    3) 반면 DiT 기반 SR 모델은 주로 선형 레이어에 의존하므로, 이를 위한 효율적인 저비트(low-bit) 양자화 방식이 새롭게 필요한 시점

### 2.2. Quantization of Diffusion Models

1) 확산 모델 양자화의 기술적 난제
    1) 반복적인 노이즈 제거: 확산 모델은 여러 단계의 반복적인 프로세스를 거치기 때문에 양자화 오류가 누적되기 쉽습니다.
    2) 활성화 이상치(Outliers): 시간 단계(timestep)별로 활성화 데이터의 분포가 크게 변하며 이상치가 존재하여 정밀한 양자화가 어렵습니다.

2) 주요 양자화 접근 방식
    1) QAT (Quantization-Aware Training): 매우 낮은 비트에서도 높은 성능을 내지만, 모델을 다시 훈련해야 하므로 비용이 많이 들고 대규모 데이터셋이 필요합니다.
    2) PTQ (Post-Training Quantization): 가벼운 미세 조정만으로 배포가 가능해 훨씬 효율적입니다.
        1) 초기 PTQ 방식: Q-Diffusion이나 PTQD 등은 시간 단계에 따른 분포를 고려한 교정(Calibration) 전략을 사용했습니다.
        2) DiT 전용 방식: PTQ4DiT(채널 재정렬), Q-DiT(윈도우 어텐션 고려 반올림), SVDQuant(저계수 분해를 통한 이상치 완화) 등이 제안되었습니다.
        3) 초해상도(SR) 전용 방식: PassionSR과 같은 모델은 1단계(one-step) 모델에서 적응형 스케일을 조정하는 방식을 사용합니다.

3) 기존 연구와의 차별점 (Q-DiT4SR의 강점)
    1) 최소한의 Calibration 오버헤드: 대부분의 기존 PTQ 방식은 여전히 많은 양의 Calibration 데이터와 반복적인 최적화 과정을 요구합니다.
    2) 효율적인 비트 할당: 본 연구의 VaSMP는 별도의 Calibration 데이터 없이 오프라인에서 가중치 정밀도를 할당하며, VaTMP는 아주 작은 교정 세트만으로도 활성화 정밀도를 결정할 수 있어 매우 경제적입니다.

### 2.3. Mixed-Precision Quantization

1) 기존 혼합 정밀도 연구의 흐름
    1) 초기 연구: HAWQ 계열은 헤시안(Hessian) 기반의 민감도를 사용하여 비트를 할당했으며, BRECQ는 블록 단위의 재구성을 통해 정밀도를 높였습니다.
    2) 최신 확장: 최근에는 Transformer 및 확산 모델(Diffusion)로 기술이 확장되었습니다.
        1) MixDQ: 몇 단계의 텍스트-이미지 생성 모델을 위해 민감도를 분리하여 분석했습니다.
        2) HQ-DiT: FP4 하이브리드 양자화를 도입했습니다.
        3) MPQ-DM 및 MPQ-DMv2: 시간적 증류(Temporal distillation)와 잔차 혼합 정밀도를 적용했습니다.
        4) 대규모 언어 모델(LLM): IMPQ나 AMQ 같은 기술들이 자동화된 혼합 정밀도 전략을 위해 제안되었습니다. 

2) 기존 방식의 한계점: 대부분, 최적의 비트 할당을 위해 대규모 데이터셋을 활용한 비용이 많이 드는 교정(Calibration) 과정이나 반복적인 순전파(Forward pass) 과정을 거쳐야 한다는 단점이 있습니다. 

2) Q-DiT4SR의 차별화된 접근 (VaSMP 및 VaTMP): 기존 방식들과 대조적으로, 본 논문에서 제안하는 방식은 매우 효율적입니다:
    1) VaSMP: 가중치 정밀도를 할당할 때 어떠한 교정 데이터도 필요 없는 오프라인 데이터 프리(Data-free) 방식을 사용합니다.
    2) VaTMP: 활성화 정밀도를 결정할 때 아주 작은 교정 세트만을 사용하여 분산 통계량을 계산하므로 연산 오버헤드가 거의 없습니다.

---


## 3. Method

<img width="525" height="275" alt="image" src="https://github.com/user-attachments/assets/a02ab7a3-bf8b-44c0-8058-be8bcf39bb50" />


### 3.1. Preliminaries

#### 1. 활성값 양자화 (Activation Quantization)
* 하다마르 변환 적용: 입력 활성값 $X$에 정규화된 하다마르 행렬 $H_n$을 곱하여 $Z = XH_n$으로 변환합니다.
* 가우시안 가정: 변환된 활성값 $Z$는 토큰 단위로 가우시안 분포 $Z_{t,:} \approx \mathcal{N}(0, \sigma_t^2 I)$를 따른다고 가정합니다. 여기서 $\sigma_t$는 각 토큰의 분산을 추정한 값입니다.
* 균등 양자화: $\mathcal{N}(0, 1)$에 최적화된 대칭 균등 양자화기 $Q_{uni}(\cdot)$를 사용하여 다음과 같이 양자화합니다:

$$Q_G(x) = \sigma_t Q_{uni}\left(\frac{Hx}{\sigma_t}\right) \quad (2)$$

#### 2. 가중치 양자화 (Weight Quantization)

가중치 역시 활성값과 유사한 방식으로 변환 및 분해 과정을 거칩니다:도메인 변환: 가중치 행렬 $W$에도 하다마르 변환을 적용하여 $W_H = WH_n$을 얻습니다.저계수 분해(Low-rank Decomposition): $W_H$에서 중요한 정보를 담고 있는 저계수 분기 $W_{LRB}$(예: rank $r=32$)를 추출하여 이는 풀-프리시전(FP)으로 유지합니다.잔차 양자화: 남은 잔차 부분인 $W_{res} = W_H - W_{LRB}$에 대해 출력 채널별로 분산을 추정하여 양자화를 수행합니다.최종 재구성: 양자화된 잔차와 FP 저계수 분기를 더한 후, 역 변환을 통해 최종 가중치 $\hat{W}$를 복원합니다:
$$\hat{W} = (W_{LRB} + Q_w(W_{res}))H_n^\top$$

#### 하다마르 적용 예시

* 데이터에 큰 이상치(10.0)가 포함된 경우

* 입력 벡터 $x = [10.0, 1.0, 2.0, 1.0]$ 가 있다고 가정해 봅시다. 여기서 '10.0'은 다른 값들에 비해 매우 커서 양자화 시 오차를 크게 유발하는 이상치입니다.

* 변환 과정 ($z = x H_n$)

$$
\begin{aligned}
z &= \begin{bmatrix} 10.0 & 1.0 & 2.0 & 1.0 \end{bmatrix}
\times \begin{bmatrix} 
0.5 & 0.5 & 0.5 & 0.5 \\ 
0.5 & -0.5 & 0.5 & -0.5 \\ 
0.5 & 0.5 & -0.5 & -0.5 \\ 
0.5 & -0.5 & -0.5 & 0.5 
\end{bmatrix} \\
&= \begin{bmatrix} 7.0 & 5.0 & 4.0 & 4.0 \end{bmatrix}
\end{aligned}
$$

### 3.2. Hierarchical SVD

### 3.3. Variance-Aware Spatio Mixed Precision

### 3.4. Variance-Aware Temporal Mixed Precision


---

## 4. Experiments

### 4.1. Settings

### 4.2. Main Results

### 4.3. Ablation Study


---

## 5. Conclusion

### 


---





