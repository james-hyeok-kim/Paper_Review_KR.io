# Generative Modeling via Drifting

저자: Mingyang Deng 1 He Li 1 Tianhong Li 1 Yilun Du 2 Kaiming He 1

1 MIT 2 Harvard University.

발표: 2026년 2월 4일 (arXiv:2602.04770)

논문: [PDF](https://arxiv.org/pdf/2602.04770)

---

<p align = 'center'>
<img width="717" height="285" alt="image" src="https://github.com/user-attachments/assets/58021a55-4c01-49ab-9e3f-6bcb15880ec4" />
</p>

* Drifting Models는 생성 모델링(Generative Modeling)을 위한 새로운 패러다임
* 기존의 확산(Diffusion) 또는 플로우 매칭(Flow Matching) 모델들이 추론 시에 여러 번의 반복적인 단계를 거쳐 노이즈를 데이터로 변환하는 것
* Drifting 모델은 학습 시간 동안의 반복적인 최적화 과정을 분포의 진화로 활용하여 추론 시에는 단 한 번의 단계(one-step inference)만으로 고품질의 데이터를 생성합니다.

---
## 1. Introduction

### 1. 핵심 개념: 학습 시간 동안의 분포 진화

* 푸시포워드(Pushforward) 모델링
    * 사전 분포 $p_{prior}$를 데이터 분포 $p_{data}$로 매핑하는 함수 $f$를 학습하는 과정으로 정의
    * 학습 과정의 활용: 딥러닝 최적화(예: SGD)는 본질적으로 반복
    * Drifting 모델은 이 학습 과정을 통해 푸시포워드 분포 $q = f_{#}p_{prior}$가 점진적으로 데이터 분포에 맞춰지도록 합니다.
    * 드리프팅 필드(Drifting Field): 샘플의 이동을 제어하는 '드리프팅 필드'를 도입합니다. 이 필드는 생성된 분포 $q$와 실제 데이터 분포 $p$ 사이의 상호작용에 기반하며, 두 분포가 일치할 때 0이 되어 평형 상태에 도달합니다.
 

2. 주요 장점 및 성능효율적인 추론: 별도의 반복적인 추론 절차가 필요 없는 단일 패스(single-pass) 네트워크로 구성되어 있어 효율적입니다.상태 최첨단(SOTA) 결과: ImageNet $256\times256$ 데이터셋에서 단일 단계 생성 모델 중 가장 뛰어난 성능을 보였습니다.잠재 공간(Latent Space): FID 1.54 달성.+2픽셀 공간(Pixel Space): FID 1.61 달성.+1모드 붕괴(Mode Collapse) 방지: 실제 데이터 분포의 여러 모드가 샘플을 끌어당기기 때문에 특정 모드로 뭉치는 현상에 강한 내성을 가집니다.+13. 학습 메커니즘손실 함수: 드리프팅 필드의 제곱 노름(squared norm)을 최소화하는 간단한 학습 목표를 사용합니다.+2특징 공간 활용: 고차원 데이터 생성을 위해 원본 데이터 공간 대신 사전 학습된 자기 자기주도 학습(SSL) 모델의 **특징 공간(Feature Space)**에서 드리프팅 손실을 계산하여 풍부한 그래디언트 정보를 얻습니다.+2

---


---
