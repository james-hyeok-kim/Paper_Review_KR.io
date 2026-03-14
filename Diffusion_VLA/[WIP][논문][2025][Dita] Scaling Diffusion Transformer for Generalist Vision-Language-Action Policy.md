# Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy

저자 : 

Zhi Hou1* Tianyi Zhang2,1* Yuwen Xiong1 Haonan Duan5 Hengjun Pu3,1 Ronglei Tong5

Chengyang Zhao4,1 Xizhou Zhu6,1 Yu Qiao1, Jifeng Dai6,1 Yuntao Chen7†

1 Shanghai AI Lab 2 College of Computer Science and Technology, Zhejiang University

3 MMLab, The Chinese University of Hong Kong 4 Peking University 5 SenseTime Research

6 Tsinghua University 7 HKISI, CAS

https://robodita.github.io

발표 : ICCV 2025 (International Conference on Computer Vision 2025), 10월 21일

논문 : [PDF](https://arxiv.org/pdf/2503.19757)



---

## 0. Summary

### 핵심 기술 및 구조

* 통합 확산 트랜스포머 (Unified DiT)
* 인컨텍스트 컨디셔닝 (In-context Conditioning)
* 효율적인 설계: 334M(약 3.3억 개)의 파라미터

### 논문의 의의 (Significance)

* 액션 생성 패러다임의 전환 (헤드에서 몸통으로): 거대 모델 자체가 직접 액션
* 데이터 효율성과 일반화 능력의 조화: 대규모 데이터(OXE)로 사전 학습한 지식을 바탕, 극소수의 샘플(10-shot)만으로 환경에 빠르게 적응할 수 있음을 증명
* 가볍고 강력한 오픈 소스 베이스라인 제공: (334M)

---

## 1. Introduction

### 주요 특징 및 아키텍처

* 확산 트랜스포머(DiT) 구조

* 인컨텍스트 컨디셔닝(In-context Conditioning)
    * 여러개의 정보를 하나의 Ca
    * 기존: 이미지/텍스트 $\rightarrow$ 트랜스포머 $\rightarrow$ 압축된 임베딩 $\rightarrow$ MLP $\rightarrow$ action
    * In-Context: [언어 토큰 + 이미지 토큰 + 노이즈 섞인 액션 토큰] $\rightarrow$ 하나의 거대한 트랜스포머(DiT) $\rightarrow$ 디노이징된 액션
* 효율적인 모델 크기: 약 3억 3,400만 개(334M)의 파라미터를 가진 가볍고 깨끗한 베이스라인 모델로 설계되었습니다.
* 단일 시점 입력: 기본적으로 단일 3인칭 카메라(Third-person camera) 입력과 언어 명령만을 사용하여 구동되며, 필요에 따라 손목 카메라나 로봇 상태 등의 추가 입력을 통합할 수 있는 유연성을 제공합니다.


### 성능 및 장점

* 강력한 일반화 능력: 배경 변화, 대상이 아닌 물체의 배치, 까다로운 조명 조건 등 다양한 환경 변화 속에서도 높은 견고성을 보여줍니다.
* 장기 작업 수행: "서랍을 열고 물체를 넣은 뒤 다시 닫기"와 같은 복잡한 다단계 작업을 성공적으로 수행할 수 있습니다.
* 최첨단(SOTA) 성능: SimplerEnv, LIBERO, CALVIN, ManiSkill2 등 다양한 시뮬레이션 벤치마크에서 기존의 OpenVLA나 Octo 같은 모델들을 능가하거나 대등한 성능을 기록했습니다.
* 빠른 수렴: 기존의 확산 액션 헤드(Diffusion Action Head) 방식보다 명확하게 빠른 학습 수렴 속도를 보여주어 모델의 확장성을 입증했습니다.
    * 기존 확산 액션 헤드
    * 거대한 트랜스포머 (Backbone): 이미지와 언어 명령을 입력받아 임베딩(Embedding, 벡터 데이터) 하나만 결과물로
    * 작은 확산 헤드 (Diffusion Head): 보통 3개 층 정도의 가벼운 MLP(Multi-Layer Perceptron) 네트워크로 구성, 이 헤드는 트랜스포머가 준 '요약본(임베딩)'에만 의존해서 연속적인 로봇의 움직임을 생성
        * 한계
        * 정보의 손실: 거대한 모델이 아무리 많은 정보를 처리해도, 마지막엔 작은 헤드가 처리할 수 있게 정보를 아주 작게 요약해야 합니다. 
        * 작은 헤드의 한계: 데이터셋이 커지고 환경이 복잡해질수록(Cross-embodiment), 단 몇 층짜리 가벼운 MLP 헤드만으로는 그 복잡한 움직임을 다 감당하기 어렵습니다.
        * 비효율적인 정렬: 행동을 결정할 때 원본 이미지의 픽셀 정보를 직접 보는 게 아니라, 이미 가공된 요약본만 보기 때문에 환경 변화에 둔감할 수 있습니다.


---

## 2. Related Work

### 1. 확산 정책 디노이징 (Diffusion Policy Denoising)

* 배경: 최근 확산 모델은 이미지 생성뿐만 아니라 다중 모드(multi-modal) 로봇 액션 모델링에서도 뛰어난 숙련도를 입증해 왔습니다.
* 기존 방식의 한계: 기존의 확산 기반 조작 정책은 주로 U-Net 아키텍처나 단일 작업을 위해 설계된 얕은 교차 주의(cross-attention) 네트워크에 의존하여, 다중 모드 애플리케이션으로의 확장성에 한계가 있었습니다.
* 최신 트렌드: 최근 모델(예: Octo, $\pi_0$ )은 시각-언어 모델(VLM) 임베딩과 조밀한 MLP 디퓨저를 결합하거나, bimanual(양손) 조작을 위해 DiT 디코더를 활용하기도 합니다.
* Dita의 차별점: Dita는 인컨텍스트 컨디셔닝(in-context conditioning)을 갖춘 확장 가능한 DiT를 제안하여, 인과적 트랜스포머 아키텍처를 통해 과거 관찰 데이터를 직접 처리함으로써 다중 모드 액션 생성을 위한 표현력과 일반화 능력을 강화했습니다.

### 2. 일반주의 로봇 정책 (Generalist Robot Policies)

* 언어 조건부 정책(Language-conditioned Policy): 자연어 명령으로 행동 수행.
* VLA 프레임워크의 부상
* 비디오 백본 활용: 일부 접근 방식은 인터넷 규모 대규모 비디오 백본을 통합하기도 합니다.
* Dita의 접근 방식
    * Dita는 단순한 시각 표현 학습보다는 액션 생성(action generation)에 집중하며, 확산 기반 모델이 더 표현력 있는 대안이 될 수 있다고 봅니다.
    * 최근 PaliGemma 등을 사용하는 연구들과 달리, Dita는 LLaMA 스타일의 인과적 트랜스포머를 채택하여 정책 학습을 수행합니다.
    * 로봇의 액션을 언어 지침 및 시각 관찰과 인컨텍스트 방식으로 정렬함으로써, 다양한 로봇 개체(embodiments)에 걸친 일반화 성능을 크게 향상시켰습니다.


---

## 3. Method
<p align = 'center'>
<img width="662" height="303" alt="image" src="https://github.com/user-attachments/assets/cf63264a-e341-40c6-a234-10c7447797e3" />
</p>


### 3.1 아키텍처 (Architecture)

Dita는 언어 지침과 3인칭 카메라 이미지만을 입력으로 사용합니다. 

* 입력 토큰화 (Tokenization)
    * 언어: 고정된 CLIP 모델을 사용하여 토큰화합니다.
    * 이미지: DINOv2를 통해 패치 특징을 추출하며, 로봇 데이터에 최적화하기 위해 Dita와 함께 엔드투엔드(End-to-End)로 공동 최적화됩니다.
    * 효율성: 계산 비용을 줄이기 위해 Q-Former를 도입, 명령어 문맥(FiLM 컨디셔닝 적용)에 따라 필요한 이미지 특징만 선택합니다.
        * Q-Former: BLIP-2라는 시각-언어 사전 학습 모델에서 처음 제안된 핵심 모듈
        * 고해상도 이미지의 방대한 정보를 로봇 제어에 필요한 핵심 특징으로 압축하는 '정보 집약형 브릿지' 구조

<p align = 'center'>
<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/8b17d7d4-b154-42ed-9186-09a22f229ca7" />
<img width="900" height="350" alt="image" src="https://github.com/user-attachments/assets/e3e8e05b-0bf1-4d0c-8c50-a09c82824e9a" />
</p>


* 액션 전처리 (Action Preprocess)
    * 로봇의 동작을 7차원 벡터(평행 이동 3, 회전 3, 그리퍼 위치 1)로 표현합니다.
    * 이미지/언어 토큰과 차원을 맞추기 위해 0으로 패딩하며, 확산 공정 중에는 이 7차원 벡터에만 노이즈가 추가됩니다.
* 모델 설계 (Model Design)
    * 인컨텍스트 컨디셔닝: 별도의 확산 헤드 대신, 인과적 트랜스포머(Causal Transformer) 내부에 언어 토큰, 이미지 특징, 타임스텝 임베딩, 그리고 노이즈가 섞인 액션을 모두 결합하여 입력합니다.
    * 이 설계는 트랜스포머의 확장성을 유지하면서, 모델이 이미지 패치를 직접 참조하여 과거 관찰 데이터 속의 미세한 액션 변화(Action Deltas)를 포착할 수 있게 합니다. 

### 3.2 학습 목적 함수 (Training Objective)

* Dita는 평균 제곱 오차(MSE) 손실을 최소화하는 것을 목표로 합니다.
* 훈련: Gaussian 노이즈 $x^t$를 액션 $a$에 추가하여 노이즈 섞인 토큰을 만듭니다.
* 네트워크 $\mathcal{E}_{\theta}$는 무작위로 샘플링된 타임스텝 $t$에서 추가된 노이즈 벡터 $\hat{x}$를 예측하도록 학습됩니다.
* 추론: DDIM 스케줄러를 사용하여 훈련 시보다 훨씬 적은 단계( $N_{eval}$ )를 반복함으로써 안정적인 액션을 생성합니다. 

### 3.3 & 3.4 데이터 및 구현 세부 사항

* 사전 학습 데이터: 대규모 로봇 데이터셋인 OXE(Open X-Embodiment)를 사용하여 모델을 사전 학습합니다.
* 세부 설정:모델 구조: LLaMA2 스타일의 아키텍처를 채택했으며, 총 3억 3,400만 개(334M)의 파라미터를 가집니다.
* 프레임 예측: 2개의 과거 이미지 프레임을 보고 미래의 16개 액션 청크(Action Chunks)를 예측합니다.
* 하드웨어: 32개의 NVIDIA A100 GPU를 사용하여 학습되었습니다. 

---

---

---
