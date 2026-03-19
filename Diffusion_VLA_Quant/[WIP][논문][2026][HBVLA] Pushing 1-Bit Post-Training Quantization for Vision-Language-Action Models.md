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

---


---


---


---
