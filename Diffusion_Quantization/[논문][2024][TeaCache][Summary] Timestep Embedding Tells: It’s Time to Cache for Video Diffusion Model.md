# Timestep Embedding Tells: It’s Time to Cache for Video Diffusion Model

저자 : 

* Feng Liu1* Shiwei Zhang2† Xiaofeng Wang1,3 Yujie Wei4 Haonan Qiu5
* Yuzhong Zhao1 Yingya Zhang2 Qixiang Ye1 Fang Wan1‡
    * 1University of Chinese Academy of Sciences
    * 2Alibaba Group
    * 3Institute of Automation, Chinese Academy of Sciences
    * 4Fudan University
    * 5Nanyang Technological University

Project Page: https://liewfeng.github.io/TeaCache

발표일: arXiv:2411.19108v2, 2025년 3월 18일

논문 : [PDF](https://arxiv.org/pdf/2411.19108)


---

## 0. Summary

<p align = 'center'>
<img width="847" height="357" alt="image" src="https://github.com/user-attachments/assets/b80c7719-b7a3-4381-847e-3971948cb8ee" />
</p>

### 0.1. 핵심 아이디어

* 기존 균일 캐시(Uniform Cache) 방식의 한계를 극복하여, 타임스텝별 모델 출력 차이가 균일하지 않다는 사실에 착안한 적응형 동적 캐싱 프레임워크를 제안합니다.

* 핵심 관찰:
    * 텍스트 임베딩 → 불변(사용 불가)
    * 노이즈 입력 → 타임스텝에 둔감(상관관계 약)
    * 타임스텝 임베딩 변조 노이즈 입력 → 모델 출력과 강한 상관관계 ✓

* 2단계 전략:
    * Naive Caching: 타임스텝 임베딩의 변조 노이즈(Modulation)누적 → 상대 L1 거리를 임계값 δ와 비교하여 캐시 여부 결정
        * δ가 크면 → 더 오래 재사용 → 빠르지만 품질 약간 저하
        * δ가 작으면 → 자주 새로 계산 → 느리지만 품질 유지
    * Rescaled Caching: 입력 차이와 출력 차이 간의 스케일 편향을 다항식 피팅(Polynomial Fitting) 으로 보정하여 추정 정확도 향상

* 장점: 추가 학습 불필요, 모델 구조 변경 없음, DiT 기반 모든 모델에 적용 가능

### 0.2. 효과

* Open-Sora-Plan에서 최대 6.83× 가속, 품질 유지 기준 4.41× 가속 (-0.07% VBench)
 Latte에서 3.28× 가속 (PAB 1.34× 대비)
* Open-Sora에서 2.25× 가속 (PAB 1.40× 대비)
* 다중 GPU(DSP) 환경에서 Open-Sora-Plan 221프레임: 8×A800 기준 32.02× 가속


### 0.3. 모델 (베이스 모델)

Open-Sora 1.2 (51 frames, 480P, T=30 steps)
Open-Sora-Plan (65 frames, 512×512, T=150 steps)
Latte (16 frames, 512×512, T=50 steps)


### 0.4. 비교 대상

* PAB (Pyramid Attention Broadcast): 어텐션 블록별 균일 캐시, 비디오 특화
* ∆-DiT: DiT용 잔차 캐시 (이미지 생성 원설계)
* T-GATE: 크로스 어텐션 재사용 (이미지 생성 원설계)
* DeepCache: U-Net 기반 피처 캐시
* FasterCache: CFG 중복 활용 최적화
* AdaCache: 콘텐츠 복잡도 기반 동적 캐시

### 0.5. 데이터셋 / 벤치마크

* VBench — 비디오 생성 품질 종합 평가 (주요 기준)
* T2V-CompBench — 다항식 피팅 캘리브레이션용 70개 프롬프트 샘플링
* 평가 해상도: 480P, 512×512, 360P, 720P / 프레임: 16~240


### 0.6. 평가 지표

* VBench Score (%) ↑ — 인간 선호도 기반 비디오 품질 (참조 불필요)
* LPIPS ↓ — 지각적 유사도 (낮을수록 원본과 유사)
* SSIM ↑ — 구조적 유사도
* PSNR (dB) ↑ — 신호 대 잡음비
* FLOPs (P) ↓ — 연산량
* Latency (s) ↓ — 추론 시간
* Speedup (×) ↑ — 가속 배율

---


---
