# HBVLA: Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models

저자 : 

Xin Yan1, Zhenglin Wan2, Feiyang Ye, Xingrui Yu3*, Hangyu Du4, Yang You2, Ivor Tsang3

1School of Artificial Intelligence, Beijing Normal University, Beijing, China

2Department of Computer Science, National University of Singapore, Singapore

3Centre for Frontier AI Research, Agency for Science, Technology and Research (A*STAR), Singapore

4College of Design and Engineering, National University of Singapore, Singapore

발표 : 2026년 2월 14일, arXiv

논문 : [PDF](https://arxiv.org/pdf/2602.13710)

---

## 0. Summary

* 가중치를 1비트(Binarization) 수준, PTQ
* Step 1: Policy-Aware Weight Partitioning (정책 인식 가중치 분할)
    * 로봇의 행동 생성에 직접적인 영향을 주는 '핵심(Salient) 가중치'를 식별합니다.
        * 배경 노이즈 + 토큰 불균형(시각 토큰 > 텍스트 행동 토큰)
    * Rectified Hessian(교정된 헤시안) 기법을 통해 배경 노이즈나 시각적 토큰 불균형의 영향을 제거하고, 작업에 필수적인 신호만을 잡아냅니다.
        * 토큰 중요도 행렬( $S$ )
        * 블록 단위 그래디언트 조사(Block-wise Gradient Probe): 특정 어텐션 블록 내에서만 가벼운 역전파(Backpropagation)를 수행
        * 손실 함수( $\mathcal{L}_{blk}$ ) 정의 $\rightarrow$ 중요도( $s_t$ ) 계산, 그래디언트의 $l_2$ 노름(norm)
* Step 2: Haar Domain Hybrid Quantization (하르 도메인 혼합 양자화)
    * Sparse Orthogonal Transform (P): 서로 다른 모달리티가 섞여 발생하는 노이즈를 억제하기 위해 가중치 열을 정렬합니다.
        * 행렬의 열 순서를 재배치하는 순열 행렬(Permutation Matrix)
            * 가중치 열($W_{l,non-sal}$)들을 유사한 값끼리 인접하도록 재정렬
            * 유사한 값들을 모음으로써 고주파 에너지를 억제하고, 양자화하기 쉬운 '저엔트로피(Low-entropy)' 중간 상태를 만듭
            * $P$는 직교 행렬(Orthogonal matrix)이므로 변환 후에도 가중치 행렬의 원래 기하학적 특성(Frobenius geometry)이 엄격하게 보존
        * $W = [10.2, -9.8, 9.9, -10.1]$
            * 첫 번째 쌍 (10.2, -9.8): 차이(High-pass) = $\frac{10.2 - (-9.8)}{2} = \mathbf{10.0}$
            * 두 번째 쌍 (9.9, -10.1): 차이(High-pass) = $\frac{9.9 - (-10.1)}{2} = \mathbf{10.0}$
        * 새로운 순서: $[10.2, 9.9, -9.8, -10.1]$ (비슷한 10 근처 값끼리, -10 근처 값끼리 모음)
            * 첫 번째 쌍 (10.2, 9.9): 차이(High-pass) = $\frac{10.2 - 9.9}{2} = \mathbf{0.15}$
            * 두 번째 쌍 (-9.8, -10.1): 차이(High-pass) = $\frac{-9.8 - (-10.1)}{2} = \mathbf{0.15}$
    * Haar Wavelet Transform (H): 가중치를 주파수 영역으로 변환하여 정보 손실을 최소화하며 1비트 양자화를 적용합니다.
        * 하르 변환은 짝을 이룬 두 가중치 ( $w_{2k}, w_{2k+1}$ )를 다음과 같이 두 개의 서브밴드(Subband)로 분해
        * 저주파(Low-pass) 성분 ( $w^{lo}$ )
            * $w_k^{lo} = \frac{1}{2}(w_{2k} + w_{2k+1})$
        * 고주파(High-pass) 성분 ( $w^{hi}$ )
            * $w_k^{hi} = \frac{1}{2}(w_{2k} - w_{2k+1})$
        * '에너지 응집(Energy Compaction)' 효과
            * 앞서 설명한 $P$ 행렬(순열)로 가중치를 잘 정렬해두면, 대부분의 에너지가 저주파( $w^{lo}$ )에 몰리고 고주파( $w^{hi}$ )는 0에 가까워짐 
    * Hybrid Strategy
        * 치명적인 영향을 주는 가중치(Salient)와 그렇지 않은 가중치(Non-salient)를 구분하여 서로 다른 정밀도로 양자화하는 전략
        * 비핵심 가중치는 효율적인 공유 평균(Shared-mean) 양자화를 적용해 효율성을 극대화합니다.
            * 행(Row) 기반 공유 평균
        * 핵심 가중치는 고밀도 양자화
            * 비핵심 가중치에서 잔차 만들어서 오차 보정 (1000개중 100개 핵심이면, 100개 잔차 존재)
            * $R_l = W_l - \hat{W}_{l,non-sal}$
            * 핵심도, 비핵심도 모두 1-bit precision + 잔차도 1-bit precision

| 컴포넌트 (Component) | 가중치 (Weight) | 활성값 (Activation) | 특징 및 적용 방식 |
| :--- | :---: | :---: | :--- |
| **Vision Encoder** | **1.08-bit** | **BF16** | 가중치는 1비트로 압축하되, 활성값은 고정밀 유지 (시각 노이즈 방어) |
| **Language Model** | **1.08-bit** | **BF16** | 핵심 가중치는 잔차 보정 적용, 활성값은 가중치 중요도 판단 도구로 활용 |
| **Projector** | **BF16** | **BF16** | 모달리티 연결부로, 정밀도 손실 시 성능 저하가 심해 양자화 제외 |
| **Action Head** | **BF16** | **BF16** | 로봇의 미세한 움직임을 결정하는 핵심단으로, 모든 정밀도 유지 |


---

## 1. Introductions

<p align ='center'>
<img width="1006" height="430" alt="image" src="https://github.com/user-attachments/assets/cac86629-3661-4ca3-a3b4-c6ec72c1db47" />
</p>

### 1. 배경: VLA 모델의 등장과 한계

* VLA 모델의 유용성: RT-2, OpenVLA, CogACT와 같은 VLA 모델은 시각 지각, 언어 이해, 행동 생성을 하나의 정책으로 통합하여 복잡한 로봇 조작 작업을 수행할 수 있게 해줍니다.
* 배포의 어려움: 계산 및 메모리 요구량이 매우 높습니다. 이는 실시간 제어가 필요한 자원 제한적 로봇 플랫폼이나 엣지 디바이스 배포에 큰 장애물이 됩니다.

### 2. 문제점: 기존 양자화 방식의 부적합성

초저비트 양자화의 필요성: 기존의 8비트나 4비트 사후 양자화(PTQ)보다 더 공격적인 압축을 위해 가중치를 1비트로 줄이는 이진화(Binarization)가 주목받고 있습니다.

* VLA 특수성 무시: 거대 언어 모델(LLM)이나 시각-언어 모델(VLM)에서 사용되는 이진화 기법을 VLA에 그대로 적용하면 효과가 떨어집니다.

* 오차의 증폭: LLM은 텍스트의 정확도(Perplexity)를 중시하지만, VLA는 물리적 환경에서 연속적인 행동을 출력합니다. 아주 미세한 양자화 오차라도 폐쇄 루프(Closed-loop) 실행 과정에서 증폭되어 물체를 놓치거나 경로가 이탈하는 등 치명적인 실패로 이어질 수 있습니다.

### 3. 분석: HBVLA 개발의 세 가지 핵심 관점

저자들은 VLA 모델의 양자화 특성을 분석하여 세 가지 중요한 사실을 발견했습니다.

* 컴포넌트별 민감도 차이
    * 분석 결과 비전 모델은 상대적으로 견고(Robust)한 반면, 프로젝터와 행동 모델은 양자화에 매우 민감했습니다.
* 이중 지배 문제(Dual Dominance Problem)
    * 가중치의 중요도를 평가할 때 사용하는 기존의 헤시안(Hessian) 방식은 VLA에서 부정확합니다.
    * 배경의 노이즈(Outliers)
        * 헤이시안 Hessian ( $H=XX^{\top}$ )
            * 입력값( $X$ )의 크기가 클수록 그 가중치가 중요하다고 판단
            * 로봇이 사과를 집어야 하는 상황에서, 배경에 아주 반짝이는 물병이나 복잡한 무늬의 벽지가 있다면 사과를 집지 않는 현상
    * 시각적 토큰의 수적 불균형
        * 로봇이 보는 전체 화면 중에서 로봇 팔이나 목표 물체(사과)가 차지하는 비중은 아주 작습니다. 나머지 90% 이상은 그냥 바닥, 테이블, 벽 같은 '배경'
        * 수만 개의 배경 토큰들이 내는 작은 소음들이 합쳐져서 단 몇 개뿐인 '행동 결정 토큰(예: 사과의 위치)'의 신호를 완전히 삼켜
* 모달리티 혼합으로 인한 노이즈
    * VLA 가중치는 여러 모달리티가 섞여 있어 표준적인 Haar 변환을 적용할 경우 큰 값의 변화(Value jumps)가 발생하며, 이것이 1비트 양자화의 정확도를 떨어뜨리는 노이즈가 됩니다.


### 4. 제안: HBVLA 프레임워크

* 정책 인식 가중치 분할: 행동 생성에 중요한 가중치를 보호하기 위해 토큰별 중요도를 반영한 정정된 헤시안을 사용합니다.
* 희소 직교 변환: 중요하지 않은 가중치에는 변환을 적용하여 노이즈를 억제하고 이진화 효율을 높입니다.
* Haar 도메인 양자화: 주파수 인식을 통한 그룹화 양자화를 수행하여 저장 효율과 성능 유지 사이의 균형을 맞춥니다.


---

## 2. Related Work

### 1. 시각-언어-행동(VLA) 모델의 두 갈래

* VLA 모델은 멀티모달 관측 데이터를 로봇 제어 명령으로 직접 매핑하는 엔드투엔드(End-to-End) 정책을 지향합니다.
* 이산형 토큰 방식: RT-2, OpenVLA, UniVLA와 같은 모델은 기존의 시각-언어 모델(VLM)을 확장하여, 로봇의 행동을 토큰 형태로 이산화(Discretize)하여 생성합니다.
* 연속형 생성 방식: 시간적 연속성과 고주파수 제어를 위해 행동을 연속 영역에서 직접 모델링합니다.
    * Diffusion Policy: Octo, CogACT 등에서 사용되는 방식입니다.
    * Flow-matching: 최근 $\pi_{o}$ 모델에서 채택한 방식입니다.
* 한계: 이러한 생성형 정책들은 높은 성능을 보이지만, 메모리 요구량이 너무 커서 실제 배포가 어렵다는 단점이 있습니다.

### 2. 로봇 인공지능에서의 양자화 현황

* QAT 중심 연구: BitVLA나 SQIP 같은 기존 연구들은 대부분 양자화 인식 훈련(QAT)에 의존합니다.
* 현실적인 제약: QAT는 효과적이지만 계산 비용이 매우 크고, 방대한 양의 로봇 데이터셋에 다시 접근해야 하므로 빠른 적응이 필요한 상황에서는 비실용적입니다.
* 연구의 부재: 결과적으로, 로봇 AI 분야에서 저비트 사후 양자화(PTQ)의 잠재력은 아직 충분히 탐구되지 않은 상태입니다.

### 3. 네트워크 이진화 기술의 진화

가중치를 $\pm 1$로 극단적으로 줄이는 이진화 기술은 CNN에서 Transformer, 그리고 LLM으로 확장되어 왔습니다.
* 초기 모델: BinaryConnect, BWN, XNOR-Net 등이 CNN 기반의 이진화를 개척했습니다.
* Transformer 기반: BiT(탄성 이진화)와 EcoFormer(해싱 기반) 등이 등장하며 Transformer 구조에 최적화된 전략을 제시했습니다.
* LLM 및 VLM 확장
    * LLM: BiLLM, PB-LLM, 그리고 Haar 변환을 사용하여 표현력을 높인 HBLLM 등이 최근 연구되었습니다.
    * VLM: Bi-VLM은 가우시안 분위수 분할을 통해 멀티모달 모델로 확장했으나, 행동 제어에 핵심적인 활성화 값의 특정 열(Critical activation columns)을 포착하지 못한다는 한계가 있습니다.

---

## 3. Methodology

<p align = 'center'>
<img width="1003" height="722" alt="image" src="https://github.com/user-attachments/assets/46f0020d-b29a-4b4d-a037-433822839b01" />
</p>

### 1단계: 정책 인식 가중치 분할 (Policy-Aware Weight Partitioning)

* 기존 방식의 문제: 표준 Hessian( $H=XX^{\top}$ )은 단순히 입력값의 크기에 의존하므로, 배경 노이즈(Outliers)나 숫자가 많은 시각 토큰에 의해 중요도가 왜곡되는 '이중 지배 문제'가 발생합니다.
* 정정된 Hessian ( $\tilde{H}$ ): 각 토큰이 행동 생성에 미치는 영향력을 반영하는 토큰 중요도 행렬( $S$ )을 도입하여 $XSX^{\top}$ 형태로 Hessian을 재구성합니다.
* 중요도 산출: 효율성을 위해 전체 네트워크를 다시 학습하는 대신, 특정 블록 내에서 행동 경로(Action pathway)를 따라 로컬 그래디언트를 역전파하여 각 토큰의 인과적 영향력을 계산합니다.
* 결과: 이를 통해 가중치를 핵심(Salient) 열과 비핵심(Non-salient) 열로 분리합니다.

#### 블록정렬 손실 ( $\mathcal{L}_{blk}$ )

* 모델 전체를 한꺼번에 보는 대신, 모델을 구성하는 하나의 층(Attention Block)만 똑 떼어내서 관찰
* 정밀한 결과 ( $Z$ ): 가중치를 줄이지 않은 원래 상태의 출력값입니다.
* 거친 결과 ( $\hat{Z}$ ): 가중치를 1비트로 줄였을 때의 출력값입니다.이 두 값의 차이가 바로 양자화로 인해 발생한 '왜곡(오차)'이며, 이를 최소화하는 것이 목표

#### 로컬 그래디언트 역전파

* 이 '왜곡'이 어떤 토큰(이미지 조각이나 단어) 때문에 생겼는지 역추적

### 행동 경로(Action Pathway)

* 단순히 이미지를 잘 복원하는 가중치가 아니라, "팔을 10cm 뻗어라"라는 명령을 정확하게 전달하는 가중치를 골라내기 위함

### 2단계: 하이브리드 이진화 전략 (Hybrid Binarization)

#### 비핵심 가중치(Non-salient) 처리

* 희소 직교 변환 ( $P$ ): 서로 다른 모달리티가 섞여 있어 발생하는 급격한 값 변화(noise)를 줄이기 위해, Haar 변환 전 열의 순서를 재배치하는 순열 행렬 $P$를 적용합니다.
* 탐욕적 페어링 (Greedy Pairing): 유사한 열끼리 묶어 고주파 에너지를 최소화하는 알고리즘을 사용하여 변환 효율을 극대화합니다.
    * 당장 눈앞에 보이는 가장 이득이 되는 선택(최적해)을 즉시 내리기 때문
* 저장 효율화: 동일한 행과 주파수 대역 내에서는 평균값( $\mu_g$ )을 공유하도록 설정하여 메타데이터 오버헤드를 줄입니다.

#### 핵심 가중치(Salient) 처리

* 잔차 기반 보상: 비핵심 가중치 양자화에서 발생한 오차를 보완하기 위해, 원래 가중치와 양자화된 비핵심 가중치 사이의 잔차(Residual)에 대해 핵심 열들을 양자화합니다.
* 열 단위 Haar 변환: 핵심 가중치는 열 단위로 Haar 변환을 적용하여 세밀한 정보를 보존하고 행동 정확도를 유지합니다.

### 최종 재구성 (Reconstruction)

$$\hat{W}_{l} = \hat{W}_{l,non-sal} + \hat{W}_{l,sal}$$


---

## 4. Experiment

### 1. 실험 환경 및 설정

* 하드웨어: 모든 실험은 NVIDIA A800 GPU에서 진행되었습니다.
* 평가 지표: 로봇 조작의 성공 여부를 판단하는 성공률(Success Rate, SR)을 주요 지표로 사용했습니다.
* 비교 대상(Baselines): 최근 발표된 1비트 PTQ 기법들인 HBLLM, BiLLM, BiVLM을 대조군으로 선정했습니다.
* 사용 모델: OpenVLA, OpenVLA-OFT, CogACT 등 대표적인 VLA 모델들을 양자화 대상으로 삼았습니다. 

### 2. 주요 벤치마크 실험 결과

세 가지 주요 환경에서 실험을 수행했으며, HBVLA는 모든 환경에서 기존 1비트 방식들을 압도했습니다. 

| 벤치마크 | 특징 및 과업 | HBVLA 성능 요약 |
| :--- | :--- | :--- |
| **LIBERO** | 지식 전이 및 평생 학습 평가 (Spatial, Object, Goal, Long)  | 기존 1비트 방식 대비 평균 **11.1%~32.6% 높은 성공률**을 기록하며 성능 격차를 대폭 축소함. |
| **SIMPLER** | 실제 로봇 환경(Google Robot)을 고정밀로 재현하여 Coke 집기, 서랍 열기 등 수행  | 기존 SOTA 대비 평균 **3.1%~41.2%의 절대적인 성공률 향상**을 달성하며 압도적인 효율성을 입증함. |
| **Real-world** | Mobile ALOHA 로봇을 이용한 실제 사물 조작 (수건 접기, 하노이의 탑 등)  | 전정밀도(FP) 모델 대비 **성공률 저하가 미미**하며, 하드웨어 제약이 큰 환경에서도 안정적인 배포 가능성을 보여줌. |


### 3. 심층 분석 (Sensitivity Analysis)

* 비전 엔코더 (Vision Encoder): 양자화에 가장 강력(Robust)하며, 성능 변화가 거의 없었습니다. 
* 언어 모델 (Language Model): 어느 정도 민감도를 보였습니다.
* 프로젝터 및 행동 헤드 (Projector & Action Head): 가장 민감한 부분으로, 미세한 정밀도 손실로도 성능이 크게 떨어졌습니다. 

### 4. 절제 연구 (Ablation Study)

<p align ='center'>
<img width="412" height="119" alt="image" src="https://github.com/user-attachments/assets/3603277c-454f-4680-bf48-c465ed8534b9" />
<img width="388" height="126" alt="image" src="https://github.com/user-attachments/assets/335b308a-2568-43ba-b7b9-6040da8b57de" />
</p>

* 정정된 Hessian의 효과: 표준 Hessian 대신 정책 인식(Policy-Aware) Hessian을 사용했을 때, 시각적 노이즈를 효과적으로 필터링하여 성공률이 유의미하게 상승했습니다.
* 순열 기준(Permutation Criterion): 비핵심 열을 정렬할 때 $l_1$-norm보다 $l_2$-norm을 사용하는 것이 에너지 분포를 더 잘 포착하여 양자화 오차를 줄였습니다. 


---

## 5. Conclusion

### 핵심 기술적 기여

* 정책 인식 가중치 분할: 정정된 Hessian(Rectified Hessian)을 활용하여 로봇의 행동 생성에 결정적인 가중치를 식별하고 보호함으로써 양자화로 인한 행동 성능 저하 문제를 해결했습니다.
* Haar 도메인 최적화: 희소 직교 변환(Sparse Orthogonal Transform)을 제안하여 가중치의 기하학적 구조를 최적화하고, 모달리티 간의 이질성 및 고주파 노이즈를 효과적으로 억제했습니다.

### 실험적 성과 및 가치

* SOTA 성능 달성: LIBERO, SIMPLER 시뮬레이션 환경 및 실제 Mobile ALOHA 플랫폼에서의 광범위한 실험을 통해 기존의 모든 1비트 양자화 기법을 뛰어넘는 세계 최고 수준(SOTA)의 성능을 입증했습니다.
* 실제 배포의 기반 마련: 본 연구는 거대한 VLA 모델을 연산 자원이 한정된 실제 로봇 플랫폼에 안정적으로 배포할 수 있는 실질적인 기술적 토대를 제공합니다.
* 성능 유지율: 특히 quantized OpenVLA-OFT는 LIBERO에서 전정밀도 대비 92.2%, quantized CogAct는 SimplerEnv에서 93.6%의 성능을 유지하는 놀라운 효율성을 보여주었습니다.

---

