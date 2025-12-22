# Diffusion Model Quantization: A Review

저자 : Qian Zeng, Chenggong Hu, Mingli Song, Jie Song

Zhejiang University

출간 : arXiv preprint arXiv:2505.05215, 2025 

논문 : [PDF](https://arxiv.org/pdf/2505.05215)

---
## 1. Introduction

### 기존의 해결 노력

* 샘플러 최적화: 역시간 SDE나 ODE를 수치적으로 풀어 샘플링 단계 자체를 줄이려는 시도입니다.
* 모델 압축: 가지치기(Pruning), 지식 증류(Distillation), 그리고 양자화(Quantization) 기법이 적용되고 있습니다.
    * 가지치기: 가중치 구조를 파괴하여 처음부터 재학습해야 하는 경우가 많습니다.
    * 지식 증류: 성능은 좋으나 막대한 데이터와 계산 자원(예: 199 A100 GPU days)이 필요합니다.
    * 양자화: 표현의 충실도와 계산 효율성 사이에서 효과적인 균형을 맞추며, 특히 엣지 디바이스 배포를 위한 가속화 솔루션으로 주목받고 있습니다

### 양자화 연구의 흐름과 한계

* 초기 연구: 가우시안 분포 기반 타임스텝 샘플링을 통한 보정(Calibration) 기법인 PTQ4DM이 기반을 닦았습니다.
* 발전된 기법: 타임스텝별 동적 양자화(TDQ), 미분 가능한 타임스텝 그룹화, 비디오 생성을 위한 시간적 정렬 등이 제안되었습니다.
* 구조별 대응: U-Net뿐만 아니라 확산 트랜스포머(DiT)의 특성에 맞춘 그룹 양자화 및 채널 평활화 기법들이 등장했습니다.


### 4. 주요 양자화 기법 (Key Quantization Methods)

* 초기 및 기반 기술
    * 가우시안 분포 기반 타임스텝 샘플링(PTQ4DM) [40]
    * 분포 정렬 보정(Liu et al.) [50]
* 시간적 동적 양자화 (Temporal/Dynamic)
    * 단계별 활성화 양자화(TDQ) [45]
    * 타임스텝 그룹화(Wang et al.) [51]
    * 시간적 특징 유지(TFMQ-DM) [52]
    * 비디오 생성을 위한 시간 정렬(Tian et al.) [53]
* 학습 기반 및 미세 조정 (QAT/LoRA)
    * 양자화 인식 학습(Q-DM [54]
    * QuEST [55], MEFT-QDM [56])
    * LoRA 기반의 4비트 양자화(QaLoRA [57], IntLoRA [58])
* 극단적 양자화 (Binary/Mixed-Precision)
    * 1~2비트 이진화 및 초저비트 연구(BLD [59], BinaryDM [60], BiDM [61])
    * 혼합 정밀도 전략(BitsFusion [62], BDM [63])
* 오류 보정 및 트랜스포머 대응
    * 오류 보정 메커니즘(PTQD [44], $D^2$-DPM [46], Tac-QDM [64])
    * 확산 트랜스포머(DiT) 양자화 기법(A-QDiT [66], Q-DiT [67], PTQ4DiT [68], DIT-AS [69], ViDiT-Q [70], HQ-DiT [71])

---

##

---

##

---

##

---
