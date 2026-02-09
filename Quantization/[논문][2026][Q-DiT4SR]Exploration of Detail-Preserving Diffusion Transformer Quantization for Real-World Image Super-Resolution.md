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

* 도메인 변환: 가중치 행렬 $W$에도 하다마르 변환을 적용하여 $W_H = WH_n$을 얻습니다.
* 저계수 분해(Low-rank Decomposition): $W_H$에서 중요한 정보를 담고 있는 저계수 분기 $W_{LRB}$(예: rank $r=32$)를 추출하여 이는 풀-프리시전(FP)으로 유지합니다.
* 잔차 양자화: 남은 잔차 부분인 $W_{res} = W_H - W_{LRB}$에 대해 출력 채널별로 분산을 추정하여 양자화를 수행합니다.
* 최종 재구성: 양자화된 잔차와 FP 저계수 분기를 더한 후, 역 변환을 통해 최종 가중치 $\hat{W}$를 복원합니다:

$$\hat{W} = (W_{LRB} + Q_w(W_{res}))H_n^\top \quad (4)$$

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

#### 1. 도입 배경 및 동기 (Motivation)

* 디테일의 중요성: 확산 모델 기반의 초해상도(SR) 모델은 고주파 세부 정보에 매우 민감합니다.
* 기존 방식의 한계: 일반적인 SVD(전역적 저계수 분해)는 모델의 지배적인 구조는 잘 잡지만, 텍스트 복원에 필수적인 국부적인 미세 정보를 담고 있는 잔차(Residual) 부분을 단순화하여 양자화 오류에 취약하게 만듭니다.
* 제안된 아이디어: 전역적 구조를 잡는 분기(Global)와 국부적 구조를 잡는 분기(Local)를 결합한 계층적 구조를 통해, 동일한 파라미터 예산 내에서 원래 가중치를 더 정밀하게 근사합니다.

#### 2. H-SVD의 구성 단계

1) 전역 SVD 구축 (Global SVD Construction)
    1) 먼저 가중치 행렬에 하다마르 변환을 적용한 $W_H$에서 전역적인 특징을 추출합니다.
    2) $W_{SVD-G}$ 추출: 잘린 SVD(Truncated SVD)를 사용하여 가중치 행렬의 지배적인 저주파 성분을 캡처합니다.
    3) 잔차 계산: $W_{res} = W_H - W_{SVD-G}$를 통해 전역 분기가 잡지 못한 미세 정보를 남깁니다.

2) 국부 SVD 구축 (Local SVD Construction)
    1) 남겨진 잔차( $W_{res}$ )를 다시 한번 분해하여 세부 사항을 보강합니다.블록 분할: 잔차 행렬을 겹치지 않는 작은 블록($s_o \times s_i$)들로 나눕니다.
    2) Rank-1 근사: 각 작은 블록마다 개별적으로 Rank-1 SVD를 적용하여 국부적 특징을 추출합니다
    3) 예산 최적화: 국부 분기( $W_{SVD-L}$ )의 파라미터 총량이 기존 전역 SVD 방식의 예산을 초과하지 않도록 블록 크기를 탐색하여 결정합니다.

$$W^{(p,q)} \approx \hat{W}^{(p,q)} = \sigma_{p,q} u_{p,q} v_{p,q}^\top \quad (5)$$



#### 3. 최종 가중치 재구성 및 연산

<img width="552" height="426" alt="image" src="https://github.com/user-attachments/assets/c6dca471-c5fe-48d2-b76d-4bafeb65b0d1" />


$$\hat{W} = (W_{SVD-G} + W_{SVD-L})H_n^\top + Q_w(W_{res} - W_{SVD-L})H_n^\top \quad (10)$$

1) H-SVD를 통해 최종적으로 재구성된 가중치 $\hat{W}$는 다음과 같이 계산됩니다:
2) 효과: 전역(Global)과 국부(Local) 구조를 모두 모델링함으로써, 고정된 파라미터 내에서 풀-프리시전(FP) 모델의 정보 흐름을 가장 유사하게 유지합니다.
3) 실증적 증거: PCA 분석 결과, H-SVD 방식이 일반 SVD보다 원래 모델의 주요 성분들과 훨씬 더 높은 유사도를 보임을 확인했습니다 (그림 4 참조).

### 3.3. Variance-Aware Spatio Mixed Precision

#### 1. 도입 동기 (Motivation)

* 레이어별 민감도 차이: 분석 결과, DiT 기반 SR 모델의 레이어들은 가중치의 분산값이 서로 크게 다릅니다(수십 배 이상 차이).
* 분산과 오류의 관계: 이론적으로 양자화 오류(Distortion)는 해당 가중치의 분산( $\sigma^2$ )에 비례합니다.
* 결론: 분산이 큰 레이어는 양자화에 더 민감하므로 더 높은 비트를 할당하고, 분산이 작은 레이어는 낮은 비트를 할당하여 효율성을 높여야 합니다.


#### 2. 비트 할당 프로세스 (3단계)

1) 오프라인 통계 (Offline Statistics)
    1) 교정 데이터 없이 모델 자체의 가중치 값만 사용하여 각 레이어 $l$의 평균 출력 채널 분산( $\overline{\sigma}_l^2$ )과 파라미터 수( $N_l$ )를 계산합니다.
2) 연속적 완화 (Continuous Relaxation)
    1) 목표로 하는 평균 비트(예: 4비트) 제약 조건 하에서, 전체 모델의 양자화 오류를 최소화하는 최적의 비트 폭 $b_l^*$을 수학적으로 계산합니다.
    2) 이 단계에서는 비트 값이 $3.7$이나 $4.2$처럼 소수점 형태로 산출됩니다.
3) 탐욕적 이산화 (Greedy Discretization)
    1) 소수점 형태의 비트 값을 실제 하드웨어에서 사용 가능한 정수(3, 4, 5비트 등)로 변환합니다.
    2) 우선순위 점수: 비트를 1비트 높였을 때 오류가 얼마나 줄어드는지( $Gain \propto \overline{\sigma}_l^2 4^{-b_l}$ )를 계산하여, 효율이 가장 좋은 레이어부터 높은 비트를 우선 배정합니다.

#### 3. VaSMP의 주요 특징 및 장점


<img width="462" height="356" alt="image" src="https://github.com/user-attachments/assets/bd32bb03-c1e1-490e-b34f-e4bd019078ca" />


* 완전한 데이터 프리 (Data-free): 이미지를 단 한 장도 넣지 않고 가중치의 통계값만으로 비트를 결정할 수 있습니다.
* 레이어별 이질성 반영: 모든 레이어에 똑같은 4비트를 주는 방식보다 훨씬 더 선명한 디테일을 유지할 수 있습니다.
* 낮은 오버헤드: 복잡한 반복 최적화 과정이 없으므로 비트 할당에 드는 연산 시간이 매우 짧습니다.

### 3.4. Variance-Aware Temporal Mixed Precision

<img width="461" height="339" alt="image" src="https://github.com/user-attachments/assets/68ad5af2-dd72-41b3-bee1-b43ed62d9992" />


#### 1. 도입 동기 (Motivation)

* 시간에 따른 분산 변화: 확산 모델의 노이즈 제거 과정에서 토큰 활성화 값은 시간 단계에 따라 뚜렷한 분산 변화 추세를 보입니다.
* 시간별 민감도 차이: 양자화 왜곡은 분산 크기에 비례하기 때문에, 분산이 큰 특정 시간 단계는 양자화 오류에 더 민감할 수밖에 없습니다.
* 기존 방식의 문제: 대부분의 기존 방식은 모든 시간 단계에 동일한 정밀도를 적용하여, 민감한 단계에서 발생하는 누적 오류를 제대로 제어하지 못합니다.


#### 2. 비트 할당 메커니즘① 민감도 측정 (Sensitivity Metric)각 레이어 $l$과 시간 단계 $t$에서의 평균 토큰 분산 $v_{l,t}$를 계산하여 이를 시간적 민감도 지표로 활용합니다.이를 위해 아주 적은 양의 저해상도(LR) 이미지로 구성된 교정 세트를 사용하여 통계량을 수집합니다.② 왜곡 모델링 (Distortion Modeling)활성화 값이 가우시안 분포($\mathcal{N}(0, v_{l,t})$)를 따른다는 가정하에, 비트 폭 $b$에 따른 기대 왜곡량을 다음과 같이 정의합니다:
$$D_{l,t}(b) = v_{l,t} \kappa(b)$$여기서 $\kappa(b)$는 표준 가우시안 분포에서의 최적 클리핑을 적용한 왜곡 계수입니다.③ 시간적 스케줄링 (Temporal Scheduling)목표: 고정된 전체 활성화 비트 예산(예: 평균 4비트) 내에서 모든 시간 단계의 총 왜곡을 최소화하는 것입니다.해결 방법: 동적 프로그래밍(Dynamic Programming, DP)을 사용하여 각 시간 구간별로 최적의 비트 폭을 할당하는 '계단식 스케줄'을 생성합니다.3. VaTMP의 효과적응형 정밀도: 분산이 높아 오류에 취약한 초기 단계에는 더 높은 비트(예: 5비트)를 할당하고, 상대적으로 덜 민감한 단계에는 낮은 비트(예: 3비트)를 할당하여 자원을 효율적으로 배분합니다.세부 구조 보존: 특히 W4A4와 같이 활성화 비트가 매우 낮은 공격적인 설정에서 미세한 질감과 국부 구조를 훨씬 더 잘 보존합니다.요약하자면, VaTMP는 "데이터가 복잡하게 변하는 시점에는 정밀하게, 단순해지는 시점에는 가볍게" 활성화 값을 처리하여 저비트에서도 고화질을 유지하는 기술입니다.


---

## 4. Experiments

### 4.1. Settings

### 4.2. Main Results

### 4.3. Ablation Study


---

## 5. Conclusion

### 


---





