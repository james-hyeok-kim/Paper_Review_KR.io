# Post-training Quantization on Diffusion Models

저자 : Yuzhang Shang1,4*, Zhihang Yuan2*, Bin Xie1, Bingzhe Wu3, Yan Yan1†

1 Illinois Institute of Technology, 2Houmo AI, 3Tencent AI Lab, 4Cisco Research

출간 : CVPR(	Computer Vision and Pattern Recognition), 2023

논문 : [PDF](https://arxiv.org/pdf/2211.15736)

---

## 1. Introduction

<p align = 'center'>
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/7ad236d9-8092-4571-88f2-cc9ae7fe77d3" />
</p>

### 배경 및 문제점

* 느린 속도의 원인: 논문은 속도 저하의 원인을 두 가지로 분석합니다.
    * 노이즈를 제거하는 반복 과정(iteration)이 너무 깁니다.
    * 각 반복 단계에서 노이즈를 추정하는 신경망 연산이 매우 무겁습니다

### 기존 연구의 한계

* 반복 횟수(step)를 줄이는 데에만 집중
* 반복마다 수행되는 무거운 신경망의 연산 비용 자체를 줄이는 문제는 간과

### 제안하는 해결책: 학습 후 양자화 (PTQ)

* PTQ 선택 이유: 재학습(Retraining)이나 미세 조정(Fine-tuning)이 필요한 압축 방식 대신 학습 후 양자화(PTQ)를 선택
    * 데이터 접근성: Dall-E2와 같은 거대 모델의 학습 데이터는 개인정보 보호나 상업적 이유로 공개되지 않는 경우가 많습니다.
    * 비용: 모델을 재학습하는 데 막대한 GPU 자원과 시간이 소모


### 기술적 난관과 혁신 (PTQ4DM)

* 기존 PTQ의 문제점
    * 일반적인 PTQ 방식은 단일 시간 단계(single-time-step)를 가정하고 설계
    * 확산 모델은 시간 단계(time-step)에 따라 노이즈 추정 네트워크의 출력 분포가 계속 변함
    * 따라서 기존 방식을 그대로 적용하면 성능이 크게 떨어집니다.
* 해결책 (NDTC: Normally Distributed Time-step Calibration)
    * 새로운 보정(calibration) 방법을 개발
    * 확산 모델의 다중 시간 단계 구조에 맞춰 최적화된 방식

### 주요 기여 및 성과

* 최초의 시도
    * 확산 모델 가속화를 위해 재학습 없는(training-free) 네트워크 압축을 시도한 최초의 연구입니다.
* 성능
    * 제안된 방식(PTQ4DM)은 전체 정밀도(full-precision) 모델을 8-bit 모델로 양자화하면서도
    * 성능 저하가 거의 없거나 오히려 향상된 결과를 보여줍니다.
* 호환성
    * DDIM과 같은 기존의 고속 샘플링 방식과 함께 사용할 수 있는 플러그 앤 플레이(plug-and-play) 모듈로 작동합니다.

---
## 2. Related Work

### 2.1. Diffusion Model Acceleration

#### 1. 기존 접근 방식(더 짧은 샘플링 경로 탐색)

* 기존의 확산 모델 가속화 연구들은 주로 성능을 유지하면서 더 짧은 샘플링 경로(shorter sampling trajectories)
* 즉 반복 횟수(time-steps)를 줄이는 방법을 찾는 데 집중


##### 주요 연구 사례
* Chen et al.: 그리드 서치(grid search)를 통해 6단계만의 짧은 경로를 찾았지만, 시간이 기하급수적으로 늘어나 긴 경로에는 적용하기 어렵습니다. (탐색시간이 너무 길다)
* Watson et al.: 경로 탐색을 동적 프로그래밍(dynamic programming) 문제로 모델링했습니다.
* Song et al. (DDIM): 학습 목표는 같지만 역방향 프로세스(reverse process)에서 샘플링 속도가 훨씬 빠른 비마르코프(non-Markovian) 프로세스를 구성했습니다.
* ODE/SDE Solver 활용: 확산 모델을 미분방정식(ODE/SDE) 형태로 정의하고, 더 빠른 솔버(solver)를 사용하여 효율성을 높였습니다.
* 근사법 및 기타: 몬테카를로 방법을 이용한 근사법 이나, 여러 단계를 한 단계로 압축하는 지식 증류(Knowledge Distillation) 방식 등이 제안되었습니다.

##### 기존 방식의 한계점
대부분의 기존 방식들(예: 지식 증류 등)은 사전 학습된 모델을 얻은 후에도 **추가적인 학습(additional training)**이 필요합니다. 이는 많은 자원을 소모하며, 훈련 데이터 접근이 어려운 상황에서는 사용하기 어렵습니다.

##### 본 연구의 차별점: 네트워크 압축 (Network Compression)

* 이 논문은 기존 연구들과 달리 반복 횟수를 줄이는 것이 아니라, 각 반복 단계에서 실행되는 네트워크 자체를 압축하는 데 초점을 맞춥니다.
* 따라서 기존의 고속 샘플링 방법(예: DDIM) 위에 플러그 앤 플레이(plug-and-play) 모듈처럼 추가하여 함께 사용할 수 있습니다.
* 최초의 시도: 저자들의 지식에 따르면, 확산 모델에 대해 학습 후 양자화(Post-Training Quantization)를 적용한 첫 번째 연구입니다.

---

