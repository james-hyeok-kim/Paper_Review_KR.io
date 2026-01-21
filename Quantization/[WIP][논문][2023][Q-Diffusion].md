# Q-Diffusion: Quantizing Diffusion Models

저자 : 
Xiuyu Li, UC Berkeley

Yijiang Liu, Nanjing University

Long Lian, UC Berkeley

Huanrui Yang, UC Berkeley

Zhen Dong, UC Berkeley

Daniel Kang, UIUC

Shanghang Zhang, Peking University

Kurt Keutzer, UC Berkeley

출간 : Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023

논문 : [PDF](https://arxiv.org/pdf/2302.04304)

---

## 1. Introduction

### 1. 효율성 병목 현상 (Efficiency Bottleneck)

* 느린 추론 속도: 50회에서 1,000회까지 반복적인 노이즈 추정 단계
* 높은 자원 요구량
* 긴 지연 시간: GAN 모델이 1초 미만에 여러 이미지를 만드는 것과 달리, 확산 모델은 단일 이미지 생성에 수 초가 걸립니다.

### 2. 양자화의 주요 과제

* 변화하는 분포: 타임스텝에 따라 노이즈 추정 네트워크의 출력 및 활성화(activation) 분포가 크게 달라
* 오차 누적: 반복적인 추론 특성상, 각 단계에서 발생하는 양자화 오차가 다음 단계로 전이
* 구조적 문제: UNet 아키텍처 내의 숏컷(shortcut) 레이어들이 양자화하기 까다로운 이봉(bimodal) 활성화 분포

### 3. 제안된 솔루션: Q-Diffusion

* 타임스텝 인식 교정 (Timestep-aware Calibration): 모든 타임스텝의 활성화 분포를 대변할 수 있도록 데이터를 균일하게 샘플링하여 교정 품질
* 분할 숏컷 양자화 (Split Shortcut Quantization): 숏컷 레이어의 특이한 분포를 처리하기 위한 전용 양자화 기법을 도입

### 4. 연구의 주요 성과

* 4비트 양자화 실현
* 스테이블 디퓨전(Stable Diffusion) 적용


---

---


---


---
