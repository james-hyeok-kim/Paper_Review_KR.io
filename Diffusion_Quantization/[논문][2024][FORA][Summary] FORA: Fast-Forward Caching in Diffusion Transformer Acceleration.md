# FORA: Fast-Forward Caching in Diffusion Transformer Acceleration

저자 : 

* Pratheba Selvaraju1 Tianyu Ding2† Tianyi Chen2 Ilya Zharkov2 Luming Liang2
* 1UMass Amherst 2Microsoft
* pselvaraju@cs.umass.edu, {tianyuding,tiachen,zharkov,lulian}@microsoft.com

발표 : arXiv:2407.01425v1, 2024년 7월 1일

논문 : [PDF](https://arxiv.org/pdf/2407.01425)

---

## 0. Summary

<p align  = 'center'>
<img width="992" height="603" alt="image" src="https://github.com/user-attachments/assets/eb91fdae-ea46-4e71-becf-7bcbaf00c4ff" />
</p>

### 0.1. 핵심 아이디어

* 핵심 관찰 3가지:
    * 연속 타임스텝 간 시각적·특징(feature) 유사도가 매우 높음 (특히 디노이징 후반부)
    * 디퓨전 모델: 기본 구조는 전체 과정에서 일정하게 유지됨
    * 연산 오버헤드의 대부분이 Self-Attention과 MLP 레이어에 집중됨

* 동작 방식 (캐시 인터벌 N):
    * t mod N = 0 인 스텝 → 전체 순전파 + 캐시 저장
        * 인터벌 당 한번만 계산
        * T10 → T0로 이미지 만들어 낼때, T10 부터 무조건 계산 (처음은 무조건 계산)
    * 나머지 N-1 스텝 → 저장된 캐시 재사용 (계산 생략)
    * 추가 학습 불필요, 기존 DiT 모델에 플러그앤플레이로 적용 가능

### 0.2. 효과

* DiT-XL/2에서 N=3 기준 2.80× 가속, FID 2.82 (원본 2.27 대비 소폭 저하)
* DiT-XL/2에서 N=5 기준 4.57× 가속
* PIXART-α + IDDPM에서 1.86×, + DPM-Solver에서 1.56× 가속
* N=7까지는 수용 가능한 품질 유지, N=7 초과 시 FID 급격히 저하


### 0.3. 모델

* DiT-XL/2 (ImageNet 클래스 조건부, 256×256)
* PIXART-α (텍스트 조건부 이미지 생성, 256×256)


### 0.4. 비교 대상

* DiT-XL/2-G (cfg=1.5): 기준선 (DDIM 250스텝)
* IDDPM: 기본 샘플러
* ADM, ADM-G, ADM-U: U-Net 기반 디퓨전
* LDM: Latent Diffusion Model
* DeepCache: U-Net 기반 캐시 (재학습 필요)
* Block Caching블록 단위 캐시 (경량 재학습 필요)

### 0.5. 데이터셋

* ImageNet 256×256 — 클래스 조건부 이미지 생성 (50K 샘플)
* MS-COCO 256×256 (30K) — 텍스트 조건부 이미지 생성 (Zero-shot)

### 0.6. 평가 지표

* FID ↓ — 생성 이미지 품질 (주요 지표)
* sFID ↓ — 공간적 FID
* IS (Inception Score) ↑ — 다양성 및 품질
* Precision / Recall ↑ — 정밀도 및 재현율
* FID-30K (Inception / CLIP) ↓ — MS-COCO 평가
* Speedup (×) ↑ — 추론 가속 배율
* Recomp% — 캐시 재계산 비율


---

