# From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers

저자 : 

* Jiacheng Liu1,2*, Chang Zou1,3∗, Yuanhuiyi Lyu4, Junjie Chen1, Linfeng Zhang1†
* 1Shanghai Jiao Tong University
* 2Shandong University
* 3University of Electronic Science and Technology of China
* 4The Hong Kong University of Science and Technology (Guangzhou)

발표 : arXiv:2503.06923v2, 2025년 8월 11일

논문 : [PDF](https://arxiv.org/pdf/2503.06923)


---

## 0. Summary

<p align = 'center'>
<img width="800" height="450" alt="image" src="https://github.com/user-attachments/assets/d4552952-4df7-4675-95a4-9bb549cbe9e0" />
</p>

### 0.1. 핵심 아이디어

* 기존 캐시 방법의 패러다임인 "cache-then-reuse"(캐시 후 재사용) 의 근본적 한계를 극복하고, 
* "cache-then-forecast"(캐시 후 예측) 라는 새로운 패러다임을 제안합니다.

#### 기존 방법의 문제:

* 타임스텝 간격이 커질수록 피처 유사도가 지수적으로 감소
* 고가속비(4× 이상)에서 급격한 품질 저하 발생

#### 핵심 관찰:

* DiT 모델의 피처가 타임스텝에 걸쳐 안정적인 궤적(trajectory) 을 형성
* 피처의 1차 미분(도함수)도 인접 스텝 간 유사하여 예측이 가능

#### TaylorSeer 동작 방식: 현재 미분값을 알면 근처의 미래값을 예측

* 이전 타임스텝들의 피처 값으로 유한차분법(finite difference) 을 통해 고차 미분을 근사
* 테일러 급수 전개로 미래 타임스텝의 피처를 예측

* 0차(m=0): 기존 캐시 재사용 (단순 복사)
* 1차(m=1): 피처 변화 추세를 선형 예측
* 2차(m=2): 비선형 궤적을 포물선 예측
* M차(m=M): M개 이전 스텝 활용, 고정밀 예측
* 추가 학습 불필요, 추가 파라미터 없음

### 0.2. 효과

* FLUX (텍스트→이미지): 4.99× 가속, ImageReward 1.0039 (원본 수준 유지)
* HunyuanVideo (텍스트→비디오): 5.00× 가속, VBench 79.93%
* DiT-XL/2 (클래스→이미지): 4.53× 가속, FID 2.65 (기존 SOTA 대비 3.41 낮음)
* 6× 이상 가속에서 기존 방법 전부 실패하는 구간에서도 성능 유지


### 0.3. 모델 (베이스 모델)

* FLUX.1-dev — 텍스트→이미지 (50스텝 Rectified Flow)
* HunyuanVideo — 텍스트→비디오 (50스텝)
* DiT-XL/2 — 클래스 조건부 이미지 생성 (50스텝 DDIM)


### 0.4. 비교 대상

* FORA: DiT용 정적 어텐션/MLP 캐시
* ToCa: 토큰 기반 동적 캐시
* DuCa: 이중 캐시 전략
* TeaCache: 타임스텝 임베딩 기반 적응형 캐시
* ∆-DiT: 잔차 기반 캐시
* AdaCache: 콘텐츠 복잡도 기반 캐시
* DDIM (스텝 수 감소): 샘플러 기반 가속

### 0.5. 데이터셋 / 벤치마크

* DrawBench (200 프롬프트) — FLUX 텍스트→이미지 평가
* VBench (946 프롬프트, 4730 비디오) — HunyuanVideo 평가
* ImageNet 256×256 (50K 샘플) — DiT-XL/2 클래스 조건부 평가


### 0.6. 평가 지표

* FID-50k / sFID(↓): 이미지 생성 품질 (DiT)
* Inception Score(↑): 다양성·품질 (DiT)
* ImageReward(↑): 인간 선호도 (FLUX)
* CLIP Score(↑): 텍스트-이미지 정합성
* VBench Score (%)(↑): 비디오 종합 품질
* PSNR / SSIM(↑): 원본 대비 충실도
* LPIPS(↓): 지각적 유사도
* FLOPs (T)(↓): 연산량
* Latency (s)(↓): 추론 시간
* Speedup (×)(↑): 가속 배율



---
