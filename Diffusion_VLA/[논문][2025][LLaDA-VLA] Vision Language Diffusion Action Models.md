# LLaDA-VLA: Vision Language Diffusion Action Models

저자 : 

Yuqing Wen1* Hebei Li1* Kefan Gu2* Yucheng Zhao3†

Tiancai Wang3 Xiaoyan Sun1‡

1University of Science and Technology of China,

2Nanjing University, 3Dexmal

Project Page: https://wenyuqing.github.io/llada-vla/

발표 : arXiv(2025년 2월 14일)

논문 : [PDF](https://arxiv.org/pdf/2509.06932)

---

## 0. Summary

* **기존의 자기회귀(Autoregressive) 방식이 아닌 마스크 확산(Masked Diffusion) 방식을 VLA(Vision-Language-Action) 모델**
* **세계 최초의 시각-언어-확산-동작 모델**

### 두 가지 핵심 설계 (Key Designs)

* 국소적 특수 토큰 분류 (Localized Special-token Classification): 전체 어휘 사전이 아닌 로봇 동작을 위한 특수 토큰(Action Token)에만 분류를 집중하여 학습의 난이도를 낮추고 도메인 적응력을 높였습니다.
* 계층적 동작 구조 디코딩 (Hierarchical Action-Structured Decoding): 동작 간(Inter-action) 및 동작 내(Intra-action) 의존성을 고려하여 계층적으로 디코딩함으로써, 더 정교하고 일관된 로봇 경로를 생성합니다.

### 벤치마크 최고 성능(SOTA)

* SimplerEnv: 기존 OpenVLA 대비 평균 성공률이 51.3% 향상되었습니다.
* CALVIN: 연속 작업 완료 지표(Avg. Len.)에서 4.01을 기록하며 기존 모델들(OpenVLA 3.27 등)을 크게 상회했습니다.
* 실제 로봇: WidowX 로봇 실험에서 $\pi_0$와 CogACT 대비 각각 23%, 28%의 성공률 향상을 보였습니다.
* 탁월한 일반화 능력: 학습되지 않은 물체(큐브), 용기(종이 상자), 방해 요소가 있는 환경에서도 높은 성공률을 유지하며 뛰어난 적응력을 증명했습니다.

---

## 1. Introductdion 

<p align = 'center'>
<img width="660" height="348" alt="image" src="https://github.com/user-attachments/assets/eef09e2e-e885-4ec8-bdf9-2f936bfd7486" />
</p>


### 1. 자기회귀(AR) 기반 VLA의 지배와 한계

* 한계 1 (효율성): AR 모델은 토큰을 하나씩 순차적으로 생성해야 하므로 생성 속도가 제한적입니다.
* 한계 2 (유연성): 단방향(Unidirectional) 생성 방식은 복잡하고 다각적인 로봇 작업에서 유연한 대응을 어렵게 만듭니다.

### 2. 대안으로서의 마스크 확산 모델(MDM)

* 대안: 마스크 확산 모델(Masked Diffusion Models, MDMs)
    * Mask 값에서 원하는 단어를 만드는 것
    * [MASK] [MASK] [MASK] [MASK] (초기 상태)
    * [MASK] 고양이 [MASK] 잠잔다 (자신 있는 단어부터 예측)
    * 검은 고양이 위에서 잠잔다 (남은 빈칸 채우기)
    * 검은 고양이가 소파 위에서 잠잔다 (최종 완성)
* 병렬 디코딩: 토큰을 순차적으로 만드는 대신, 전체 시퀀스를 병렬로 생성하고 반복적인 노이즈 제거(Denoising) 과정을 통해 결과를 정제합니다.
* 검증된 성능: LLaDA, LLaDA-V와 같은 최신 연구들은 확산 모델 기반의 VLM(d-VLMs)이 기존 AR 모델에 필적하는 성능과 확장성을 가짐을 증명했습니다.

### 3. VLA 적용 시 직면하는 두 가지 도전 과제

* 도메인 격차 (Domain Gap): 일반적인 d-VLM은 고수준의 의미론적 데이터로 학습되는 반면, VLA는 정밀한 동작 생성을 위해 저수준(Low-level)의 시각적 단서를 해석해야 합니다.
* 구조적 의존성 결여: 기존의 확산 모델 디코딩 전략은 로봇 동작 시퀀스가 가진 계층적이고 구조적인 특성(동작 간의 선후 관계 등)을 충분히 반영하지 못합니다.

### 4. LLaDA-VLA의 핵심 제안

* 국소적 특수 토큰 분류: 동작 토큰(Action) 에만 분류를 한정하여 도메인 적응 난이도를 대폭 낮춥니다.
* 계층적 동작 구조 디코딩: 동작 간 및 동작 내 의존성을 명시적으로 고려하여 일관성 있는 궤적을 생성합니다.
    * 동작 간(Inter-action) 의존성: 1초 뒤의 동작은 현재 동작과 자연스럽게 이어져야 합니다.
    * 동작 내(Intra-action) 의존성: 하나의 동작을 이루는 7개 토큰(위치, 회전 등)은 서로 일관성이 있어야 합니다.
    * 동작별 신뢰도 점수( $C_a$ )를 계산
    * 가장 점수가 높은(확실한) 동작을 먼저 선택하고, 나머지 동작들은 다시 마스킹하여 다음 차례로 미룹니다.
    * 부드러운 움직임 (일관성 있고 부드러운 궤적)


---

## 2. Related Work

### 2.1. 대규모 언어 확산 모델 (Large Language Diffusion Models)

* 이산적 텍스트의 처리: 텍스트 토큰은 본질적으로 이산적(Discrete)이기 때문에, 픽셀과 같은 연속적인 공간에서 작동하는 확산 모델을 직접 적용하기 어렵습니다.
* 주요 접근법: 텍스트를 연속적인 표현으로 학습하거나, 마스크 확산 모델(Masked Diffusion Models)과 같은 이산적 확산 모델을 개발하는 두 가지 방향이 존재합니다.
* 성능 및 확장성: Dream7B나 LLaDA와 같은 연구들은 대규모 언어 사전 학습을 통해 자기회귀 모델(ARM)에 필적하는 성능과 스케일링 특성을 보여주었습니다.
* 멀티모달 확장: 이러한 흐름은 LaViDA(시각 이해), LLaDA-V(시각 지시 튜닝), MMaDA(통합 확산 트랜스포머) 등으로 이어지며 시각-언어 영역(d-VLMs)으로 확장되었습니다.
* 연구의 공백: 하지만 이러한 발전에도 불구하고, 대규모 언어 확산 모델을 로봇 조작(Robotic Manipulation)에 적용하려는 시도는 거의 이루어지지 않았습니다.

### 2.2. 시각-언어-동작 모델 (Vision-Language-Action Models)

* 선구적 연구: RT-2는 웹 스케일 데이터와 로봇 시연 데이터를 함께 학습하여 다양한 조작 작업에서 강력한 성능을 보여주었습니다.
* 오픈소스 및 발전: OpenVLA는 최초의 오픈소스 VLA 모델로서 연구를 가속화했으며, 이후 궤적 주석을 활용한 LLARVA나 연속 제어를 위한 확산 헤드를 도입한 CogACT 등이 등장했습니다.
    * 궤적 주석(Trajectory Annotation)
        * 로봇이 특정 작업을 수행할 때 거쳐가는 경로(궤적)에 대해 구체적인 정보를 사람이 직접 또는 자동화된 방식으로 달아주는 것
        * "사과를 집어"라는 명령을 받았을 때, 단순히 결과만 보는 것이 아니라 어떤 경로로 가는 것이 최적인지를 학습할 수 있습니다.
    * 확산 헤드
        * AI 모델은 "입력(이미지/언어) → 본체(Backbone) → 헤드(Head)"의 구조 (로봇의 경우 손이 헤드) 
* 최신 기법: $\pi_0$는 흐름 매칭(Flow-matching) 전략과 대규모 멀티태스크 데이터셋을 결합하여 탁월한 성능을 달성했습니다.
* 차별점: 그러나 기존의 VLA 모델들은 거의 예외 없이 자기회귀(Autoregressive) 방식에 기반하고 있으며, 확산 기반 VLM의 잠재력은 여전히 미개척 상태로 남아있습니다.


---

## 3. Method

### 3.2 LLaDA-VLA의 상세 구조

#### 3.2.1 모델 아키텍처 (Model Architecture)

<p align = 'center'>
<img width="389" height="263" alt="Image" src="https://github.com/user-attachments/assets/834aa933-3b1a-4f59-be25-2f21cde98093" />
</p>


* 구성 요소: 언어 백본(LLaDA: Large Language Diffusion Models), 시각 인코더(SigLIP-2), 그리고 이를 연결하는 프로젝터(MLP)로 구성됩니다.
    * MLP: Projector
* 입력 처리: 언어 명령과 로봇의 정면 RGB 이미지를 입력받아 시각 특징을 텍스트 공간으로 투영한 뒤, 함께 결합하여 확산 모델에 입력합니다.
* 동작 토큰화 (Action Tokenization): 연속적인 로봇 동작 값을 7개의 특수 동작 토큰(위치, 회전, 그리퍼)으로 이산화하여 처리합니다.

#### 3.2.2 국소적 특수 토큰 분류 (Localized Special-token Classification)

* 전략: 수만 개의 단어가 있는 전체 어휘 사전 대신, 로봇 동작을 위해 추가된 32개의 특수 토큰에 대해서만 분류를 수행합니다.
* 의의: 학습 목표를 동작 관련 토큰에만 집중시킴으로써 도메인 적응 난이도를 대폭 낮추고 학습 효율을 높였습니다.

#### 3.2.3 계층적 동작 구조 디코딩 (Hierarchical Action-Structured Decoding)

* 가장 혁신적인 부분으로, 동작 시퀀스의 구조적 의존성을 고려한 디코딩 방식입니다.
* 동작 점수 계산 ( $C_a$ ): 개별 토큰의 신뢰도를 합산하여 각 동작(Action Chunk 내의 개별 스텝)의 점수를 매깁니다.
* 계층적 선택
    * 1. Action-level Remask: 가장 신뢰도가 높은 동작 단위를 먼저 선택하고 나머지는 다시 마스킹합니다.
    * 2. Token-level Remask: 선택된 동작 안에서도 신뢰도가 낮은 토큰은 다시 마스킹하여 정밀도를 높입니다.
    * 결과: 이 과정을 통해 물리적으로 훨씬 부드럽고 일관된 로봇의 움직임이 생성됩니다. 


---

## 4. Experiment


### 1. 실험 환경 설정 (Setup)

* SimplerEnv: 현실 물리 법칙과 시각적 외형을 정교하게 모방한 시뮬레이션으로, WidowX 로봇을 사용해 4가지 조작 작업을 평가합니다.
* CALVIN: 긴 호흡(Long-horizon)의 언어 조건부 로봇 조작을 위한 벤치마크로, 5개의 연속적인 작업을 얼마나 잘 수행하는지(Avg. Len.)를 측정합니다.
* 실제 WidowX 로봇: 실제 환경에서 8가지 작업(학습된 작업 4개, 일반화 작업 4개)을 수행하며 실질적인 성능을 테스트합니다.

### 2. 주요 정량적 결과 (Quantitative Results)

<p align = 'center'>
<img width="488" height="317" alt="Image" src="https://github.com/user-attachments/assets/84cdf44e-e426-488a-9f44-a321d85c7446" />
</p>

* 실험 결과 LLaDA-VLA는 기존의 자기회귀(AR) 기반 모델들을 압도하는 성능을 보여주었습니다.
* 시뮬레이션 최고 성능: CALVIN 벤치마크에서 OpenVLA 대비 평균 연속 작업 성공 지표가 0.74 향상되어 4.01을 기록했습니다.
    * Avg.Len: 연속 작업 완료 평균 길이(Average Episode Length), 얼마나 오랫동안 실수 없이 연속적으로 임무를 수행할 수 있는지 측정하는 지표

<p align = 'center'>
<img width="762" height="254" alt="Image" src="https://github.com/user-attachments/assets/c0f5e1a8-b196-4fea-993f-230ae6b962bd" />
</p>

* 실제 로봇 성공률: 실제 환경에서 평균 성공률 58%를 달성하며, 강력한 경쟁 모델인 $\pi_0$(35%)와 CogACT(30%)를 크게 앞질렀습니다.
* 강력한 일반화: 학습 데이터에 없던 새로운 물체나 방해 요소가 있는 환경(OOD)에서도 $\pi_0$보다 25% 높은 평균 성공률을 기록했습니다. 

### 3. 소거 연구 (Ablation Study)

<p align = 'center'>
<img width="381" height="343" alt="Image" src="https://github.com/user-attachments/assets/57d2db24-5a12-4fab-8bd4-1d7e59bd3b35" />
</p>

* Localized Special-token Classification (LSC): 이를 적용했을 때 성공 지표가 0.79 상승하며, 동작 토큰에 집중하는 것이 학습 난이도를 크게 낮춘다는 것을 증명했습니다.
* Hierarchical Action-structured Decoding (HAD): 일반 디코딩 대신 이 방식을 썼을 때 성능이 0.58 추가로 상승하여, 계층적 구조가 부드러운 궤적 생성에 필수적임을 보여주었습니다.
* 동작 덩어리 크기 (Action Chunk Size): 한 번에 5개의 동작 스텝을 생성할 때 가장 최적의 성능(4.01)을 보였으며, 너무 많거나 적으면 성능이 하락했습니다. 

---

## 5. Conclusion

### 1. 연구의 핵심 요약

* 본 논문은 사전 학습된 확산 기반 시각-언어 모델(d-VLMs)을 로봇 제어에 활용한 최초의 Vision-Language-Diffusion-Action 모델(LLaDA-VLA)을 제안했습니다.
    * LLaDA-VLA가 '최초'라고 주장하는 이유는 "사전 학습된 언어 확산 모델(d-VLM)을 그대로 로봇 백본으로 사용한 최초의 사례"
* 연구진은 기존의 자기회귀(AR) 방식이 아닌 확산 패러다임을 통해 로봇 정책 학습의 새로운 가능성을 열었습니다. 

### 2. 기술적 기여 (Key Designs)

* 국소적 특수 토큰 분류 (LSC): 일반 어휘와 로봇 동작 사이의 도메인 격차를 해소하여 학습 효율을 극대화했습니다.
* 계층적 동작 구조 디코딩 (HAD): 마스크 확산 모델이 구조화된 동작 궤적을 생성할 수 있게 하여, 물리적으로 더 일관성 있고 타당한 결과를 만들어냈습니다. 

### 3. 실험적 성과 및 의의

* 성능 입증: 제안된 설계들을 통해 LLaDA-VLA는 여러 시뮬레이션 벤치마크와 실제 로봇 실험에서 SOTA(State-of-the-Art, 최고 수준) 성능을 달성했습니다. 
* 미래 가치: 이번 연구 결과는 로봇 조작 분야에서 확산 기반 모델(d-VLMs)의 응용 가능성을 입증하는 견고한 토대가 되었으며, 향후 관련 연구들이 나아갈 길을 제시했습니다. 

---

## Appendix

<div align = 'center'>

| 구분 | $\pi_0$ (pi-0) | LLaDA-VLA |
| :---: | :---: | :---: |
| **기반 모델 (Backbone)** | 기존의 자기회귀(AR) 방식인 PaliGemma를 기반으로 함 <br> LLM AR(inference) + Diffusion(Action) | 사전 학습된 확산 기반 VLM(d-VLM)인 LLaDA-V를 백본으로 사용 <br> Diffusion(Inference + Action) |
| **확산 방식** | Flow-matching 기법을 별도로 결합하여 동작을 생성 | 모델 자체가 마스크 확산(Masked Diffusion) 방식으로 토큰을 예측 |
| **데이터 형태** | 연속적인(Continuous) 수치 데이터로 동작을 출력함 | 동작을 이산적인(Discrete) 토큰으로 변환하여 언어처럼 처리함 |
</div>

---

