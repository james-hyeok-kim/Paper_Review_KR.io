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

<p align = 'center'>
<img width="574" height="317" alt="image" src="https://github.com/user-attachments/assets/4968ae3f-bb34-4196-8ad2-785c39cf7f5c" />
</p>


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

## 2. Related work

### 1. 확산 모델 (Diffusion Models)

* 순방향 과정 (Forward Process)

$$q(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_tI)$$

* 역방향 과정 (Reverse Process)

$$p_{\theta}(x_{t-1}|x_t)=\mathcal{N}(x_{t-1};\tilde{\mu}_{\theta,t}(x_t),\overline{\beta}_tI)$$

* 모델구조
    * 주로 UNet, 최근 Transformer
 
### 2. 가속화된 확산 프로세스 (Accelerated Diffusion Process)

* 샘플링 최적화
    * DDIM
    * 고차 솔버(high-order solvers)
* 기타 기법
    * 특징 맵(feature map)을 캐싱
    * 모델을 적은 타임스텝으로 증류(distillation)하는 방식
* Q-Diffusion의 차별점
    * Training-Free

### 3. 훈련 후 양자화 (Post-training Quantization, PTQ)

* 양자화 공식

$$\hat{w}=s\cdot clip(round(\frac{w}{s}),c_{min},c_{max})$$

* 기존 PTQ 연구: 분류(classification)나 검출(detection) 작업에서는 EasyQuant, BRECQ, ZeroQ, SQuant 등 다양한 교정(calibration) 기법이 연구
* 확산 모델 양자화의 선행 연구 (PTQ4DM): 8-bit
* Q-Diffusion: 4-bit, 어텐션 활성값(act-to-act matmuls)까지 완전히 양자화하여 더 높은 효율성을 달성
* 
  

---

## 3. Method

### 3.1. Challenges under the Multi-step Denoising

<p align = 'center'>
<img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/335f8beb-9954-409f-93e9-5d07995bb4e3" />
<img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/c088e38f-0379-48e5-bf81-63031958dbde" />
</p>

#### Challenge 1: Quantization errors accumulate across time steps

* 오차의 증폭: 확산 모델에서 $t$ 단계의 입력( $x_t$ )은 이전 단계( $t+1$ )의 출력 결과물, 오차의 누적
* 실험적 증거 (Figure 3): CIFAR-10 데이터셋 실험 결과, 비트로 정밀도를 낮추면 양자화 오차(MSE)가 급격히 증가


#### Challenge 2: Activation distributions vary across time steps.

<p align = 'center'>
<img width="735" height="207" alt="image" src="https://github.com/user-attachments/assets/187abec5-42fe-4195-a8c4-36d557be88bb" />
</p>

* 분포의 변화 (Figure 5): UNet 모델의 출력 활성화 분포는 타임스텝에 따라 서서히 변화
* 데이터 샘플링의 어려움: 특정 시점의 데이터로만 모델을 Calibration 하면, Overfitting
* 실험적 증거 (Figure 4): 처음(First), 중간(Mid), 혹은 마지막(Last) 일부 단계의 데이터만 사용하여 4비트 양자화를 진행했을 때, 전체 단계를 아우르지 못하면 FID 점수가 크게 나빠지는(품질 저하) 현상이 관찰

### 3.2. Challenges on Noise Estimation Model Quantization

#### 1. UNet Architecture & Shortcut Layers

* deep features + shallow features : shortcut (concatenation)

#### 2. Abnormal Activation Ranges

<p align = 'center'>
<img width="704" height="257" alt="image" src="https://github.com/user-attachments/assets/c470bcd2-0af1-4375-b72a-c50c5b55ce2e" />
</p>

* 극단적인 값의 차이: 최대 200배
* 특정 레이어(주로 숏컷 관련)에서 활성화 범위가 비정상적으로 솟구치는 것


#### 3. Bimodal Distribution Problem

<p align = 'center'>
<img width="708" height="245" alt="image" src="https://github.com/user-attachments/assets/0b873b79-560f-4120-8a1b-4a5198b472ee" />
</p>

* Concatenation
* Bimodal 가중치 분포
* 하나의 양자화기로는 한계



---


---
