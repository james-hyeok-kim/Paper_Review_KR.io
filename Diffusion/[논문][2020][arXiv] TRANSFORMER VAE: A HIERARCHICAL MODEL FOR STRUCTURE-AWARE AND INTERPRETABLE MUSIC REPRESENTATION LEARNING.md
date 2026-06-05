# TRANSFORMER VAE: A HIERARCHICAL MODEL FOR STRUCTURE-AWARE AND INTERPRETABLE MUSIC REPRESENTATION LEARNING

저자 : Junyan Jiang1,2, Gus G. Xia2, Dave B. Carlton3, Chris N. Anderson3, Ryan H. Miyakawa3

출신 : 1 School of Computer Science, Carnegie Mellon University, 2 Music X Lab, New York University Shanghai, 3 Hooktheory, LLC

출간 : ICASSP(International Conference on Acoustics, Speech, and Signal Processing), 2020

논문 : [PDF](https://ieeexplore.ieee.org/document/9054554)


<p align = 'center'>
<img width="444" height="632" alt="image" src="https://github.com/user-attachments/assets/d373f975-1d98-4e71-b2bb-8f414b616941" />
</p>


---

## Abstract
* 딥러닝 음악 생성 분야의 두 가지 획기적인 기술
    * 구조적 인식(Structure Awareness)
    * 해석 가능성(Interpretability)
* Music Transformer: 어텐션(Attention) 메커니즘을 사용하여 음악의 장기적인 의존성을 학습합니다.
* Deep Music Analogy(심층 음악 유추): 조건부 VAE(Conditional-VAE)를 사용하여 해석 가능한 잠재 표현(Latent representations)을 학습

* 작동 원리 (계층적 구조)
    * 하위 계층 (Bottom Layer): 여러 개의 로컬 인코더가 병렬로 작동하며 각 마디(Bar)의 잠재 표현을 학습합니다.
    * 상위 계층 (Top Layer): 마스크드 어텐션(Masked Attention) 블록을 사용하여 마디 간의 의존성, 즉 전역적 구조(Global Structure)를 추출

* 문맥 민감형 계층적 표현: 로컬 표현을 '문맥(Context)'으로, 로컬 표현 간의 의존성을 '전역 구조(Global Structure)'로 간주하여 학습
* 문맥 전이 (Context Transfer): 사용자가 모델과 상호작용하며 "만약 이 곡이 다른 곡의 음악적 흐름(구조)을 따라 전개된다면 어떨까?"라는 가상의 상황을 실현

---

## 1. Introduction

#### 기존 연구의 한계점 (Research Gap)

* VAE 기반 모델 (예: Deep Music Analogy): 해석 가능성(Interpretability)은 뛰어나지만, 시계열 구조(Time-series structure)를 잘 다루지 못해 프레이즈 단위와 같이 긴 음악에서는 성능이 크게 떨어집니다.
* 어텐션 기반 모델 (예: Music Transformer): 긴 호흡의 구조(Structure Awareness)는 잘 다루지만, 내부의 잠재 상태(Latent states)를 사람이 해석하기 어려워 제어가 불가능

#### 핵심 매커니즘 및 기여점
1. 효과적인 암기: 구조화된 긴 신호를 효과적으로 기억할 수 있습니다.
2. 문맥 민감형 학습: 문맥에 따라 달라지는 표현 학습이 가능합니다.
3. 상호작용적 생성 (Context Transfer): "만약 이 곡이 다른 곡의 흐름(구조)을 따라간다면 어떨까?"와 같은 상상의 시나리오를 '문맥 전이'를 통해 실제로 구현할 수 있습니다.

---

## 2. Proposed Method

<p align = 'center'>
<img width="444" height="632" alt="image" src="https://github.com/user-attachments/assets/d373f975-1d98-4e71-b2bb-8f414b616941" />
</p>


#### Transformer VAE
 
* 하위 계층 (Local Layer): 각 마디(Bar)를 개별적으로 처리합니다
    * Local Encoder: 각 마디의 멜로디 $x_i$를 입력받아 마디 수준의 표현(embedding)인 $h_i^e$로 변환합니다
    * Local Decoder: 상위 계층에서 처리된 결과($g_i^d$)를 다시 멜로디 $\hat{x}_i$로 복원합니다
* 상위 계층 (Transformer Layer): 마디 간의 관계(전역 구조)를 처리합니다
    * Transformer Encoder: 마디 표현 $h_{1..T}^e$를 입력받아 잠재 코드(Latent code) $z$의 파라미터($\mu, \sigma$)를 계산합니다
    * Transformer Decoder: 잠재 코드 $z$와 이전에 복원된 마디들의 정보를 결합하여 순차적으로 다음 마디를 생성합니다


##### 전역 문맥 학습 (Transformer Encoding)

$$g_{1..T}^{e}=E_{Transformer}(h_{1..T}^{e}) \quad \quad (1)$$

* 로컬 인코더를 통해 얻은 각 마디의 표현들( $h_{1..T}^{e}$ )을 Transformer 인코더에 넣습니다
* 의미: 단순히 개별 마디만 보는 것이 아니라, 어텐션 메커니즘을 통해 마디들 사이의 관계(문맥)를 파악하여 전역 정보가 담긴 벡터 $g_{1..T}^{e}$를 만듭니다.

##### VAE 파라미터 계산

$$[\mu_{i},log(\sigma_{i}^{2})]=W_{i}g_{i}^{e}+b_{i} \quad \quad (2)$$

* 트랜스포머의 출력값 $g_{i}^{e}$에 선형 변환(가중치 $W$, 편향 $b$)을 적용
* VAE는 데이터를 하나의 고정된 점이 아니라 확률 분포(가우시안 분포)로 저장
    * 이 수식은 그 분포를 정의하는 평균( $\mu$ )과 분산( $\sigma^2$, 정확히는 로그 분산)을 계산하는 과정

##### 잠재 변수 샘플링 (Sampling)

$$z_{i}\sim\mathcal{N}(\mu_{i},\sigma_{i}^{2}) \quad \quad (3)$$

* 계산된 평균($\mu_i$)과 분산($\sigma_i^2$)을 따르는 정규분포에서 잠재 변수(Latent Variable) $z_i$를 뽑아냅니다(샘플링)
* $z_i$가 바로 음악의 '구조적 특징'이 압축된 코드

##### 전역 디코딩 (Transformer Decoding)

$$g_{1..i}^{d}=D_{Transformer}(z_{1...i},h_{0...i-1}^{d}) \quad \quad (4)$$

* Transformer 디코더는 두 가지 입력을 받습니다
* 현재 시점까지의 잠재 변수들 ( $z_{1...i}$ )이전 시점까지 이미 생성된 마디들의 로컬 표현 ( $h_{0...i-1}^{d}$ )
* 의미: 새로운 마디를 만들기 위해 "이 곡의 전체적인 구조( $z$ )"와 "방금 전까지 연주된 내용( $h^d$ )"을 동시에 고려하여 디코더 출력값 $g^d$

##### 로컬 디코딩 (Local Decoding)

$$\hat{x}_{i}=D_{local}(g_{i}^{d}) \quad \quad (5)$$

* 트랜스포머 디코더의 출력값 $g_{i}^{d}$를 로컬 디코더($D_{local}$)에 넣습니다
* 의미: 추상적인 표현을 다시 구체적인 음표(Token)들로 변환하여 최종적으로 **재구성된 멜로디 $\hat{x}_i$**를 출력


#### Context-sensitive Representation

* 자기 주의(Self-Attention) 메커니즘을 통해 중복을 제거하고 효율적인 표현을 학습
* 1번째 마디와 5번째 마디가 똑같은 멜로디
* 5번째 마디를 인코딩할 때 1번째 마디를 참조(attend)할 수 있습니다
* 따라서 $z_5$(5번째 마디의 잠재 코드)에 멜로디 정보를 다시 저장하는 대신, "1번째 마디와 같음" 정보 전달
* 이를 통해 특정 마디의 '문맥(이전 마디들의 흐름)'만 바꾸면 전체 음악이 그에 맞춰 변하는 문맥 전이(Context Transfer)가 가능

#### Dependency Control via Masked Attention

<img width="522" height="461" alt="image" src="https://github.com/user-attachments/assets/38bace10-3857-4693-a829-c49ff5cc2ec4" />


* 마스크가 없다면 반복되는 정보가 $z_1$에 저장될지, $z_5$에 저장될지, 혹은 분산될지 알 수 없습니다
* 상삼각 마스크(Upper triangular mask)를 적용
    * $i$번째 마디는 자신의 왼쪽(과거)인 $x_{1..i}$만 볼 수 있습니다

####  A Conditional VAE View

<img width="526" height="315" alt="image" src="https://github.com/user-attachments/assets/bcea50f5-bff5-4102-bf69-e1f7bf379bcc" />

* 일반 VAE: 아무런 참고 자료 없이 정답($x$)을 써내야 합니다. 따라서 머릿속($z$, 잠재 변수)에 정답에 대한 모든 정보를 암기
    * $z_1$에 "도레미파" 저장
    * $z_5$에도 "도레미파" 저장. (정보 낭비)
* 조건부 VAE (CVAE): "참고서 몇 페이지를 베껴라" 또는 "참고서 내용에서 이 부분만 바꿔라" 정도의 정보만 저장
    * $z_5$에 멜로디를 다시 저장하지 않고, "1마디와 똑같이 연주해"

--- 

## 3. Experiments

#### 데이터셋 및 실험 환경 (Dataset & Setup)
 
 * 데이터 출처: Hooktheory에서 제공하는 대중가요(Popular music) 데이터를 사용했습니다
 * 데이터 규모: 4/4 박자의 8마디 길이 곡 16,142개를 사용했으며, 이 중 80%는 훈련용, 20%는 테스트용으로 나누었습니다
 * 전처리: 데이터 부족 문제를 해결하기 위해 -4에서 +4 반음(semitone) 범위 내에서 데이터 증강(Data Augmentation)을 수행했습니다
 * 데이터 표현: 16분 음표 단위로 토큰화하였으며, 총 45개의 음정(pitch) 상태와 지속(sustain), 쉼표(silence) 상태를 포함합니다
 * 하이퍼파라미터: 각 마디의 잠재 차원( $z_i$ )은 64로 설정하였고, $\beta$-VAE 손실 함수( $\beta=1$ )를 사용하여 학습했습니다

#### 정량적 비교 평가 (Comparative Results)

<img width="521" height="293" alt="image" src="https://github.com/user-attachments/assets/d9283254-dddc-4861-88f2-a07a0600907b" />


##### 비교 대상 모델

* Proposed+A: 제안된 Transformer VAE (상삼각 마스크 어텐션 적용)
* Proposed-A: 어텐션 메커니즘을 뺀 Transformer VAE (대각 마스크 적용)
* 1x8-bar LSTM: 8마디 전체를 한 번에 암기하려는 Music VAE 베이스라인 (무식한 암기 방식) .
* 8x1-bar LSTM: 문맥 고려 없이 1마디짜리 VAE 8개를 병렬로 돌리는 방식 .

#### Interactive Generation via Context Transfer

<img width="526" height="346" alt="image" src="https://github.com/user-attachments/assets/b3976c0a-0275-4411-abb8-b2df8512d389" />


* 실험 방법:두 곡($x^{(1)}$, $x^{(2)}$)을 인코딩한 뒤, **첫 번째 마디의 잠재 표현($z_1$)만 서로 교체(Swap)**하여 새로운 곡을 생성했습니다
    * 실험 결과 (Fig. 5):단지 첫 마디의 코드만 바꿨음에도 불구하고, 생성된 전체 곡에서 흥미로운 변화가 관찰되었습니다
    * 전역 구조 유지 (Global structure remains): 원래 곡의 반복 구조(예: A-A 형식)는 그대로 유지되었습니다
    * 리듬 패턴 변화 (Rhythmic pattern changes): 생성된 곡은 교체된 새로운 첫 마디의 리듬 패턴을 따라가기 시작했습니다
    * 음역대 변화 (Pitch range changes): 전체적인 음 높낮이 범위도 새로운 첫 마디의 음역대에 맞춰 조정되었습니다
* 한계점:모델이 단순 반복은 잘 잡아내지만, 역행(retrograde)이나 전위(inversion) 같은 복잡한 음악적 관계는 아직 완벽하게 포착하지 못하는 한계가 있었습니다 .



---
