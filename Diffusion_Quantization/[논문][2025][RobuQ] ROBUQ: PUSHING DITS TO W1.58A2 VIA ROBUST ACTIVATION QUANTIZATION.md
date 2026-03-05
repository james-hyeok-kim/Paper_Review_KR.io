# ROBUQ: PUSHING DITS TO W1.58A2 VIA ROBUST ACTIVATION QUANTIZATION

저자 : Kaicheng Yang1∗, Xun Zhang1∗, Haotong Qin2,

Yucheng Lin1, Kaisen Yang3, Xianglong Yan1, Yulun Zhang1†

1Shanghai Jiao Tong University, 2ETH Zurich, 3Tsinghua University

발표 : 2025년 9월 28일에 arXiv

논문 : [PDF](https://arxiv.org/pdf/2509.23582)


---

## 0. Summary

### QAT

<div align = 'center'>
  
| 레이어 구분 | 가중치 비트 폭 (Weight) | 활성화 비트 폭 (Activation) | 비고 |
| :--- | :--- | :--- | :--- |
| **일반 선형 레이어**<br>(qkv, proj, fc1, fc2) | 1.58-bit (Ternary) | 1~4 bit (가변) | AMPN 알고리즘을 통해 레이어별로 최적의 비트를 할당합니다. |
| **SVD 저역통과 분기**<br>(Low-rank branch) | 32-bit (FP) | 32-bit (FP) | 양자화로 인한 정보 손실을 보상하는 보조 경로입니다. |
| **Attention Scores**<br>(A-A Matrix Mult) | 8-bit (연산 정밀도) | 8-bit | 활성화 값 간의 행렬 곱으로, 높은 정밀도가 필요한 핵심 구간입니다. |
| **adaLN-Zero 레이어** | 4-bit | 4-bit | 모델 안정성을 위해 4-bit로 고정하여 처리합니다. |
| **임베딩 및 최종 레이어** | 32-bit (FP) | 32-bit (FP) | 입출력 데이터의 품질을 위해 양자화를 적용하지 않습니다. |

</div>


<div align = 'center'>

| 레이어 이름 (Layer Name) | W1.58A2 (평균 2-bit 목표) | W1.58A3 (평균 3-bit 목표) | 비고 |
| :--- | :---: | :---: | :--- |
| **attn.proj** (Attention Projection) | **약 2.75 bit** | **약 3.89 bit** | 두 설정 모두에서 가장 높은 정밀도 유지 |
| **attn.qkv** (QKV Projection) | 약 2.21 bit | 약 2.96 bit | Attention 관련 레이어 우선 할당 |
| **mlp.fc1** (MLP First Layer) | 약 2.00 bit | 약 3.04 bit | 목표 평균치에 근접한 할당량 |
| **mlp.fc2** (MLP Second Layer) | 약 1.64 bit | 약 2.31 bit | 상대적으로 낮은 비트 할당 (예산 절감) |

</div>


<p align = 'center'>
<img width="600" height="850" alt="image" src="https://github.com/user-attachments/assets/2818e7cb-d7e1-43fc-910e-92249deb9bf6" />
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/db7efff8-2123-4f3c-864e-55d0ac47aa80" />
</p>


---

## 1. Introduction

### 1. 핵심 문제 의식: 활성화 양자화의 병목 현상

* 비대칭적 난이도: 기존 연구들은 가중치(Weight)를 삼진(ternary) 수준으로 양자화해도 정확도 손실이 거의 없음을 보여주었습니다. 하지만 활성화(Activation) 양자화는 훨씬 더 어려우며, 특히 ImageNet-1K와 같은 대규모 데이터셋에서는 4비트 미만으로 내리기가 쉽지 않았습니다.
* DiT의 특수성: DiT는 깊은 구조와 복잡한 활성화 값 분포를 가지고 있어 양자화 시 고유한 도전 과제에 직면합니다. 연구진은 DiT를 초저비트(ultra-low bit) 설정으로 밀어붙이는 데 있어 활성화 양자화가 주요 병목 지점임을 식별했습니다.

### 2. 주요 제안 방법론

* 강력한 베이스라인: 가중치를 Ternary( $W1.58$ )하고 활성화를 4비트( $A4$ )로 설정한 강력한 DiT 양자화 베이스라인을 먼저 구축했습니다.
* RobustQuantizer: 하다마르 변환(Hadamard transform)을 활용하여 복잡하고 불규칙한 활성화 분포를 표준 정규 분포(standard normal form)로 변환합니다. 이를 통해 분포에 구애받지 않는 효율적인 양자화가 가능해집니다.
* AMPN (Activation-only Mixed-Precision Network): DiT를 위한 최초의 활성화 전용 혼합 정밀도 파이프라인입니다. 가중치는 전체 네트워크에 ternary 방식을 적용하되, 각 레이어의 중요도에 따라 서로 다른 활성화 비트 폭을 할당하여 정보 병목을 제거합니다.

### 3. 연구 성과

* 초저비트 달성: RobuQ 프레임워크를 통해 DiT의 활성화를 평균 2비트( $W1.58A2$ ) 수준까지 낮추면서도 안정적이고 경쟁력 있는 이미지 생성 성능을 달성했습니다.
* SOTA 성능: 무조건부 및 조건부 이미지 생성 실험 모두에서 4비트 미만 양자화 구성 중 최고 수준(State-of-the-Art)의 성능을 입증했습니다.


---

## 2. Realted Works

### 2.1 확산 트랜스포머 (Diffusion Transformers)

* 발전 배경: U-Net to DiT
* DiT의 특징: 확장성(scalability)
* 한계점: 높은 메모리 사용량과 처리 능력을 요구하며

### 2.2 양자화 (Quantization)

* 기본 개념: 가중치와 활성화 값의 정밀도를 낮춤, 압축하고 가속화하는 기술
* 초저비트 연구: Ternary, Binary, 2-bit
* 직교 변환의 도입: SVD, 하다마르(Hadamard) 변환과 같은 직교 변환이 도입
혼합 정밀도(Mixed-precision): 모든 레이어에 동일한 비트 폭을 할당하는 대신, 민감한 구성 요소에는 높은 정밀도를, 그렇지 않은 곳에는 낮은 정밀도를 할당하여 효율성과 성능의 균형을 맞추는 전략이 효과적인 방법으로 부상했습니다.


---

## 3. Method

### 3.1 분석 (Analysis)

1) DiT 모델이 기존 U-Net 기반 모델에 비해 초저비트 환경에서 성능이 떨어지는 이유
    1) 초저비트 QAT 탐색 부족: 기존 연구는 주로 사후 양자화(PTQ)에 집중되어 있으며, 양자화 인식 훈련(QAT)을 통한 극한의 활성화 비트 폭 탐색이 부족했습니다.
    2) 다양하고 복잡한 활성화 분포: DiT는 레이어와 토큰별로 활성화 분포가 매우 다양하여 통일된 양자화기를 적용하기 어렵습니다.
    3) 활성화 비트 폭 병목 현상: 특정 레이어들이 활성화 압축에 극도로 민감하여 전체 모델의 비트 폭을 낮추는 데 걸림돌이 됩니다.

### 3.2 RobustQuantizer: 하다마르 변환의 효과적 활용

<img width="817" height="382" alt="image" src="https://github.com/user-attachments/assets/82853826-9938-42c7-940a-1e6f40c31213" />


* Activations control: 하다마르 변환(Hadamard Transform) 도입, $X \rightarrow HX$
* Weight도 하다마르 변환 적용, $W \rightarrow HW$
* 

#### 3.2.1 초기 베이스라인 및 전략

* 가중치 양자화: channel-wise ternarization를 사용하여 FP 가중치를 $\{-1, 0, 1\}$ 값으로 매핑합니다.
* 활성화 양자화: 토큰별 min-max 양자화 전략을 기본으로 사용합니다.

#### 3.2.2 강화된 베이스라인 (Enhanced Baseline)

* 초기 베이스 라인 + 아래 추가적인 technic 
* SVD 기반 저역 행렬 분기 (LRB): FP 정밀도를 유지하는 저역 행렬( $A, B$ ) 분기를 추가하여 양자화 오류를 보상합니다.
* 전체 레이어 하다마르 적용: 일부 레이어가 아닌 모든 선형 레이어에 하다마르 변환을 적용하여 활성화 분포를 안정화했습니다.

#### 3.2.3 정규 분포화 이론 (Theoretical Foundation)

<p align = 'center'>
<img width="843" height="355" alt="image" src="https://github.com/user-attachments/assets/1eb1804a-b89f-4767-bfb3-cedad24989ea" />
</p>

* 핵심 원리: 하다마르 변환은 중심 극한 정리(CLT)에 따라 임의의 활성화 분포를 예측 가능한 토큰별 정규 분포(per-token normal distribution)로 변환합니다.
    * 중심 극한 정리: 표본의 크기가 커지면 그 평균들의 분포는 정규분포에 가까워진다
    * 수많은 독립적인 채널 값들이 하다마르 행렬에 의해 무작위로 더해지면서, 원래의 불규칙했던 분포(Unknown Distribution)가 자연스럽게 정규 분포(Normal Distribution)로 수렴하게 되는 것 (Inter-channel Hadamard)
* 이점: 복잡한 다변량 양자화 문제를 독립적인 가우시안 변수 양자화 문제로 단순화하여 양자화 효율을 극대화합니다.

### 3.3 활성화 전용 혼합 정밀도 네트워크 (AMPN)

* 모든 레이어에 동일한 비트 폭을 적용하는 대신, 레이어별 민감도에 따라 비트 폭을 다르게 할당합니다.

#### 3.3.1 파이프라인 설계

<p align = 'center'>
<img width="831" height="530" alt="image" src="https://github.com/user-attachments/assets/de55b05c-a076-449d-84db-cfe540c795d6" />
</p>

* 동적 계획법(DP) 활용: 제한된 평균 활성화 비트 폭 예산 내에서 전체 정확도 손실을 최소화하도록 비트 폭을 할당합니다.
* 고정 정밀도: 매우 민감하지만 연산량이 적은 어텐션 스코어(8비트)와 adaLN 레이어(4비트)는 고정된 정밀도를 유지하여 안정성을 확보합니다.
    * QKV projection , Output projection은 비중이 크지만, $QK_T = S(Score)$, $SV = Attention$은 연산량이 작다
    * 하지만 민감도가 높아서 안정성이 크게 흔들린다.

<div align = 'center'>
  
| 항목 | 정밀도 | GFLOPS | 비중 |
| :--- | :--- | :--- | :--- |
| W-A 행렬 곱 (주요 레이어) | 1.58/4 bits | 6.9213 | 대부분  |
| A-A 행렬 곱 (어텐션 스코어 등) | 8/8 bits | 1.0133 | 소수  |
| adaLN-Zero | 4 bits | 0.2026 | 극소  |

</div>

<p align = 'center'>
<img width="852" height="252" alt="image" src="https://github.com/user-attachments/assets/67d7eb71-dda8-42f4-9145-c58e50ee88f4" />
</p>


#### 3.3.2 초저비트 QAT (Ultra-low-bit QAT)

<p align = 'center'>
<img width="832" height="406" alt="image" src="https://github.com/user-attachments/assets/527185f1-af8c-4875-aa7d-feaef772b412" />
</p>

* QAT의 적응성: 사후 양자화(PTQ)에서 큰 오류를 보이던 레이어도 QAT 과정을 통해 보정될 수 있음을 확인했습니다.
* 최적화 단계: 실험 결과, 약 1,000단계의 QAT를 거친 후 레이어 민감도를 측정하는 것이 최종 수렴 성능을 가장 잘 예측하는 것으로 나타났습니다.


---

## 4. Experiments

<p align = 'center'>
<img width="787" height="1095" alt="image" src="https://github.com/user-attachments/assets/eac72319-ff91-484a-baf1-fcc3b62387cb" />
</p>

### 4.1 실험 설정 (Setup)

* 데이터셋 및 모델: 클래스 조건부 DiT-XL/2 모델을 사용하여 ImageNet-1K ($256\times256$)와 FFHQ ( $256\times256$ ) 데이터셋에서 평가를 진행했습니다. 
* 평가 지표: 생성된 이미지의 품질을 측정하기 위해 FID(Fréchet Inception Distance), sFID, IS(Inception Score), Precision의 4가지 지표를 사용했습니다. 
* 비교 대상: BitNetv2, QueST, PTQ4DiT, Q-DiT, BinaryDM 등 최신 PTQ 및 QAT 양자화 방법들과 비교했습니다. 
* 학습 상세: PyTorch 환경에서 단일 NVIDIA RTX A6000 GPU를 사용했으며, AdamW 옵티마이저로 35만 회(350k) 반복 학습을 수행했습니다. 임베딩 및 마지막 레이어는 풀 정밀도(FP)를 유지했습니다.

### 4.2 주요 결과 (Main Results)

* 압도적 성능: ImageNet-1K와 FFHQ 모든 비트 폭 설정에서 기존의 모든 양자화 모델보다 우수한 성능을 보여주었습니다. 
* 초저비트 안정성: 특히 W1.58A2 설정에서도 학습이 붕괴되지 않고 안정적으로 고품질 이미지를 생성하는 세계 최초의 성과를 거두었습니다. 
* 지표 역전 현상: 높은 가이드 스케일( $cfg=4.0$ )에서는 모든 양자화 모델이 오히려 FP 모델보다 나은 FID를 기록하는 이상 현상이 발견되어, 초저비트 환경을 위한 더 정밀한 지표의 필요성을 시사했습니다. 

### 4.3 절제 연구 (Ablation Study)

<p align = 'center'>
<img width="783" height="350" alt="image" src="https://github.com/user-attachments/assets/0b00a461-2ec4-40c5-8951-40edd4bb9c98" />
</p>

* 베이스라인 강화 (Table 2a): 기본 BitNetv2에 저역 행렬 분기(LRB)를 추가하면 FID가 41.59에서 29.59로 개선되었고, 전체 하다마르 변환까지 적용하면 20.82까지 낮아졌습니다. 
* 양자화기 선택 (Table 2b): 이론적으로 최적인 비균등(Non-uniform) 양자화기보다 균등(Uniform) 양자화기가 실제 활성화 값의 미세한 오차에 더 강건하여 최종 선택되었습니다. 
* QAT 단계의 중요성 (Table 2c): 혼합 정밀도 할당을 위한 민감도 측정 시, 1,000단계의 QAT를 거친 후 측정하는 것이 비용 대비 가장 뛰어난 성능(FID 30.30)을 보였습니다.

### 4.4 효율성 분석 (Efficiency Analysis)

<p align = 'center'>
<img width="823" height="136" alt="image" src="https://github.com/user-attachments/assets/9712bd4b-5fd6-4ccd-aa2b-8f2480d72dbd" />
</p>

* 이론적 이득: W1.58A2 모델은 FP 모델 대비 이론적으로 17.3배의 속도 향상과 13.2배의 모델 압축률을 달성했습니다. 
* 실제 배포 (Table 5): NVIDIA RTX 4090에서의 실제 테스트 결과, FP 모델 대비 체크포인트 크기는 15.2배 감소했으며, 추론 속도는 3.5배 가속되었습니다.
* 하드웨어 최적화: 현재는 전용 저비트 컴퓨팅 프레임워크의 부재로 가속 잠재력이 완전히 실현되지 않았으나, 향후 최적화된 라이브러리가 도입되면 더 큰 이점을 보일 것으로 기대됩니다. 



---

## 5. Conclusion 


* 핵심 병목 지점 식별: 연구진은 확산 트랜스포머(DiT)의 양자화 문제를 재검토한 결과, 초저비트(ultra-low-bit) 배포를 가로막는 가장 큰 병목이 활성화 경로(activation pathway)에 있음을 확인했습니다.

* 강력한 베이스라인 구축: SVD로 초기화된 저역 행렬 분기(low-rank branch)와 모든 레이어에 적용된 하다마르 믹싱(all-layer Hadamard mixing)을 특징으로 하는 강력한 W1.58A4 베이스라인을 성공적으로 수립했습니다.

* 이론적 입증과 도구 개발: 하다마르 변환이 토큰별 활성화를 효과적으로 가우시안화(Gaussianizes)한다는 것을 이론적으로 증명하였으며, 이를 통해 분포에 구애받지 않는 RobustQuantizer를 개발했습니다.

* 실용적인 최적화: 하드웨어 친화적인 균등 양자화(uniform implementation)를 활성화 전용 혼합 정밀도 네트워크(AMPN)와 결합하여 안정적인 학습을 달성하고 품질을 대폭 개선했습니다.

* 최종 성과: 이러한 기술적 진보를 통해 양자화된 DiT의 새로운 SOTA(State-of-the-Art) 결과를 도출했으며, 최종적으로 DiT의 성능을 W1.58A2 구성까지 끌어올리는 데 성공했습니다.


---




