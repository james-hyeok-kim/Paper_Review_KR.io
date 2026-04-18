# Learning-to-Cache: Accelerating Diffusion Transformer via Layer Caching

저자 : Xinyin Ma1 Gongfan Fang1 Michael Bi Mi2 Xinchao Wang1∗

National University of Singapore1 Huawei Technologies Ltd.2

maxinyin@u.nus.edu, xinchao@nus.edu.sg 

발표일: NeurIPS 2024 (arXiv:2406.01733v2, 2024년 11월 16일)

논문 : [PDF](https://arxiv.org/pdf/2406.01733)

---

## 0. Summary

<p align = 'center'>
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/9ec2a164-3af6-4aaa-968d-72f5fcb68f26" />
</p>

### 0.1. 핵심 아이디어

* 핵심 메커니즘:
    * 전체 추론(느리지만 정확)과 이전 스텝 재사용(빠르지만 부정확)의 두 극단을 보간(Interpolation)으로 연결
    * 각 레이어에 대해 "캐시할지 계산할지"를 결정하는 라우터 β를 학습
    * 라우터는 입력 불변(input-invariant), 타임스텝 가변(timestep-variant) → 정적 계산 그래프 생성 가능
    * 지수적 탐색 공간을 미분 가능한 최적화 문제로 변환하여 해결
    * 모델 파라미터는 전혀 수정하지 않고 β만 학습 (DiT-XL/2 기준 훈련 변수 560개에 불과)


### 0.2. 효과

* U-ViT-H/2에서 캐시 스텝의 최대 93.68% 레이어 제거 가능 (FID 손실 < 0.01)
* DiT-XL/2에서 1.27× 가속, FID 3.46 (DDIM 16스텝 FID 4.68 대비 우수)
* 동일 추론 속도에서 DDIM, DPM-Solver 및 기존 캐시 기반 방법 전체 능가
* 레이어 드롭아웃 대비 압도적 우세 (FID 3.47 vs 17.35)

### 0.3. 모델

* DiT-XL/2 (256×256, 512×512)
* DiT-L/2 (256×256)
* U-ViT-H/2 (256×256)


### 0.4. 비교 대상

* DDIM, DPM-Solver-2: 스텝 수 감소 샘플러
* DeepCache: U-Net 기반 캐시
* Faster Diffusion: U-Net 구조 캐시
* Manual (Top/Bottom/Random/Metric): 휴리스틱 레이어 선택
* Learning-to-Drop: 레이어 드롭아웃 변형

### 0.5. 데이터셋

* ImageNet (256×256, 512×512) — 클래스 조건부 이미지 생성


### 0.6. 평가 지표

* FID ↓ — 생성 품질 (주요 지표)
* sFID ↓ — 공간적 품질
* IS (Inception Score) ↑ — 다양성 및 품질
* Precision / Recall ↑ — 정밀도 및 재현율
* MACs — 연산량
* Latency (s) — 실제 추론 시간
* Speedup (×) — 가속 배율
* NFE (Number of Function Evaluations) — 모델 호출 횟수

---
