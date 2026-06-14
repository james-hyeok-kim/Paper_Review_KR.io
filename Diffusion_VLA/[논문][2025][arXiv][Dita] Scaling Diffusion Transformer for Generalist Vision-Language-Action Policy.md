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

## 4. Simulation Experiments

### 1. 주요 벤치마크 및 목적

* SimplerEnv (Google Robot): 실제 로봇 데이터를 시뮬레이션 환경에서 평가하는 Real-to-Sim 플랫폼으로, 제로샷(Zero-shot) 적응 능력을 측정합니다.
* LIBERO: 다중 작업(Multitask) 및 평생 학습(Lifelong learning)에서의 지식 전이 능력을 평가합니다.
* CALVIN: 환경이 바뀐 상황에서 언어 지침을 따르는 장기 호흡(Long-horizon) 작업 능력을 평가합니다.
* ManiSkill2: 30만 개의 무작위 카메라 풀을 사용하여 새로운 카메라 시점(Camera view)에 대한 일반화 성능을 측정합니다.

### 2. 실험 결과 요약

#### A. SimplerEnv: 제로샷 일반화 (Table 1)
* Dita는 배경, 질감, 물체 위치 등이 변하는 다양한 시나리오에서 가장 높은 성공률을 보였습니다.
* Coke Can (variant): Dita 85.5% vs OpenVLA 54.5%.
* Move Near (variant): Dita 73.0% vs OpenVLA 47.7%.
* 의의: 인컨텍스트 컨디셔닝 덕분에 3인칭 시점 이미지만으로도 미세한 차이를 포착하여 더 신뢰할 수 있는 액션을 생성합니다.

#### B. LIBERO: 다중 작업 적응 (Table 2)

* Dita는 전체 평균 성공률에서 기존 모델 대비 약 6% 향상된 성능을 보였으며, 특히 장기 작업에서 강점을 드러냈습니다.
* LIBERO-LONG: Dita 63.8% vs OpenVLA 53.7%.
* 평균 성공률: Dita 82.4% vs OpenVLA 76.5%.

#### C. CALVIN: 장기 호흡 작업 (Table 3)

* 5개의 연속적인 하위 작업을 완수해야 하는 환경(ABC→D)에서 단일 RGB 카메라만으로 우수한 성적을 거두었습니다.
* 평균 성공 길이 (Avg.Len.): Dita 3.61 (5개 중 평균 3.6개 성공).
* 5개 연속 성공률: Dita 50.0%.

#### D. ManiSkill2: 시점 일반화 (Table 4)

* 학습 시 보지 못한 새로운 카메라 각도에서도 안정적으로 동작함을 입증했습니다.
* 전체 평균: Dita 65.8% vs Diffusion Head 41.0%.
* 특히 복잡한 작업인 PickClutterYCB에서 Diffusion Head 방식보다 12% 높은 성능을 보였습니다.

### 3. 핵심 분석: 왜 Dita가 더 잘하나?

* 수렴 속도: Dita는 기존의 '확산 액션 헤드' 방식보다 훨씬 더 빠르게 학습이 수렴됩니다.
* 인컨텍스트의 힘: 액션 예측 시 요약된 임베딩이 아닌 원본 이미지 토큰을 직접 참조하기 때문에, 물체의 정확한 위치 파악과 정교한 조작이 가능합니다.

---

## 5. Real-Robot Experiments

### 5.1 하드웨어 및 실험 환경

* 로봇 구성: 7자유도의 Franka Emika Panda 로봇 팔과 Robotiq 2F-85 그리퍼를 사용합니다.
* 시각 센서: 로봇에서 약 1.5m 떨어진 곳에 RealSense D435i 카메라를 배치하여 3인칭 시점(Third-person view) 영상만 사용합니다.
* 제어 시스템: NVIDIA A100 GPU 1대를 탑재한 서버를 통해 3Hz의 제어 주기로 작동합니다.

### 5.2 10-Shot Finetuning 과제

<p align = 'center'>
<img width="860" height="384" alt="image" src="https://github.com/user-attachments/assets/a6bbab9a-9f78-47d1-90c2-8507ce0e2b24" />
</p>

* Dita는 OXE 데이터셋으로 사전 학습된 모델을 바탕으로, 새로운 환경에서 단 10개의 데이터만 사용하여 다음의 도전적인 과제들을 수행했습니다
* 기본 조작
    * 바나나/키위 집어서 상자에 넣기
    * 물과 커피 원두 붓기.
* 장기 호흡(Long-Horizon) 과제
    * 서랍 안의 그릇을 꺼내 커피 원두 붓기.
    * 라켓을 집어 공을 골대에 넣기.
    * 서랍을 열고 물체를 넣은 뒤 다시 닫기 등 3단계 이상의 복잡한 동작.
* 복잡한 3D 회전: 바나나를 좁은 연필꽂이에 삽입하거나, 뚜껑이 있는 상자를 열고 안의 물체 집기.

### 5.3 실험 결과 및 성능 분석

* 정량적 성과: Dita는 2단계로 구성된 복잡한 과제에서 63.8%의 평균 성공률을 기록했으며, 이는 Octo(45.0%)나 OpenVLA(41.2%)보다 월등히 높은 수치입니다.
* 장기 과제 수행력: 특히 두 번째 단계(2nd stage)의 성공 기여도가 매우 높아, 긴 호흡의 작업을 끝까지 완수하는 능력이 뛰어남을 입증했습니다.
* 견고성(Robustness)
    * 배경 변화: 테이블보 색상이나 배경막이 바뀌어도 안정적으로 동작합니다.
    * 방해물 존재: 작업과 상관없는 물체들이 어질러진(Cluttered) 상황에서도 목표물에 집중합니다.
    * 조명 변화: 실내 전등을 꺼서 조도가 급격히 변하는 상황에서도 견고함을 유지합니다.

### 5.4 기존 모델과의 비교 (Qualitative Comparison)

* OpenVLA/Octo의 한계: 10-shot 설정에서 기존 모델들은 물체를 집을 정확한 위치를 잡지 못하거나, 잡은 후 다음 동작(붓기, 삽입 등)을 이해하지 못하고 멈추는 경우가 많았습니다.
* Dita의 강점: 인컨텍스트 컨디셔닝 덕분에 시각적 미세 단서를 놓치지 않고, 복잡한 3D 회전과 정밀한 삽입 동작을 성공적으로 수행해냈습니다.


---
## 6. Conclusion

### 1. Dita의 핵심 성과 요약

* 통합 아키텍처: 트랜스포머 기반의 확산 모델을 활용하여 연속적인 액션 시퀀스를 직접 디노이징하는 아키텍처를 제시했습니다.
* 인컨텍스트 컨디셔닝: 별도의 외부 모듈 없이 메인 트랜스포머 내부에서 시각 정보와 액션을 직접 정렬함으로써 모델의 표현력을 극대화했습니다.
* 강력한 일반화 능력: 대규모 교차 개체(Cross-embodiment) 데이터셋의 확장성을 활용하여, 다양한 시뮬레이션 벤치마크에서 견고한 일반화 성능을 입증했습니다.

### 2. 실세계 적응 및 효율성

* Few-shot 적응: 새로운 로봇 환경과 장기 호흡(Long-horizon) 과제에 대해 최소한의 샘플(10-shot)만으로도 성공적으로 적응하는 능력을 보여주었습니다.
* 간결하고 가벼운 설계: 334M 파라미터라는 효율적인 크기와 단일 3인칭 카메라 입력만으로도 뛰어난 성능을 달성하여, 실용적인 일반주의 정책의 베이스라인이 될 잠재력을 증명했습니다.
* 오픈 소스 기여: 모델을 오픈 소스로 공개함으로써 유연하고 확장 가능한 솔루션을 제공하며 향후 연구에 기여하고자 합니다.

### 3. 연구의 의의

* 이 연구는 거대한 VLA 모델이 반드시 엄청나게 큰 파라미터를 가질 필요는 없으며, 아키텍처의 정교한 설계(인컨텍스트 컨디셔닝 등)와 확산 모델의 확장성을 결합할 때 얼마나 강력한 효율성을 낼 수 있는지를 보여주었습니다.

---
