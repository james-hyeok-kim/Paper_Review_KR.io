# UNIFIED DIFFUSION VLA: VISION-LANGUAGEACTION MODEL VIA JOINT DISCRETE DENOISING DIFFUSION PROCESS

저자 : 

Jiayi Chen1,∗ Wenxuan Song1,∗,‡ Pengxiang Ding2,3 Ziyang Zhou1 Han Zhao2,3, Feilong Tang4 Donglin Wang2 Haoang Li 1,

1HKUST(GZ) 2Westlake University 3Zhejiang University 4Monash University

∗Equal contribution ‡Project lead: songwenxuan0115@gmail.com  Corresponding author

발표 : ICLR 2025((International Conference on Learning Representations 2025), 5월

논문 : [PDF](https://arxiv.org/pdf/2511.01718)

---

## 0. Summary

### 0.1. UD-VLA 모델 요약
* 핵심 메커니즘 (JD3P): 'Joint Discrete Denoising Diffusion Process'라는 독자적인 확산 공정을 제안했습니다.
    * 미래 이미지와 액션 토큰을 하나의 동기화된 궤적 내에서 동시에 정제(Denoising)
    * 시각적 예측과 물리적 행동이 서로 시너지를 내도록 설계.
* 하이브리드 어텐션 (Hybrid Attention)
    * 이미지 전역적 일관성 + 액션의 상관관계 파악, 양방향 어텐션
    * 액션이 미래 이미지를 미리 참조하여 '역운동학(Inverse Kinematics)'적으로 동작할 수 있도록 인과적(Causal) 구조를 결합했습니다.
    * Block 내에서는 양방향 어텐션: 텍스트, 이미지, 미래 이미지, 액션 블록
    * Casual Mask Attention : 미래 이미지 $\rightarrow$ 액션 사이, 미래 이미지 $\rightarrow$ 현재 이미지 사이
    * 금지사항: 액션 $\rightarrow$ 미래 이미지, 미래 정보 $\rightarrow$ 현재 입력
* 학습 방식: 사전 학습된 시각-언어 모델(VLM)을 기반으로 하며, 비디오 데이터를 통한 '미래 상태 모델링(1단계)'과 로봇 데이터를 통한 '공동 최적화(2단계)'의 파이프라인을 거쳐 학습됩니다.
* 성능: CALVIN, LIBERO, SimplerEnv 등 주요 로봇 벤치마크에서 기존 최고 수준(SOTA)을 경신했으며, 기존 자기회귀(Autoregressive) 방식보다 4배 빠른 추론 속도를 구현했습니다.

### 0.2. 모델의 주요 의의

#### 1. 시각적 예견과 행동의 진정한 통합

* UD-VLA는 이미지 생성과 액션 결정을 하나의 확산 과정으로 묶어 두 과정이 유기적으로 서로를 강화하게 만들었습니다.

#### 2. '시각적 사고의 사슬(Visual CoT)' 구현

* 추론 과정에서 액션 토큰이 반복적으로 미래 이미지 정보를 참조
* 인간이 행동하기 전 머릿속으로 미래를 그려보는 것과 유사
* 복잡한 장기 과제(Long-horizon tasks)에서 훨씬 정교한 계획

#### 3. 효율성과 정밀도의 동시 달성

* 이산 확산(Discrete Diffusion) 방식을 채택하여 고해상도 연속 이미지 생성의 부담을 줄임
* 병렬 디코딩과 KV-캐시 등의 기법을 통해 로봇 제어에 필수적인 실시간성(Low Latency)을 확보했습니다.

#### 4. 강력한 일반화 성능

* 실제 로봇 실험 결과, 학습 과정에서 보지 못한 새로운 물체나 배경(Unseen targets/backgrounds)에 대해서도 미래 이미지를 올바르게 생성
* 정확한 동작을 수행하는 높은 제로샷(Zero-shot) 능력을 증명했습니다.

---

---
