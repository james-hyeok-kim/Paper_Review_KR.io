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

최근 확산 모델(Diffusion Models)은 시각 분야에서 비약적인 발전을 이루었으며, 이를 텍스트 생성으로 확장하려는 시도가 이어지고 있습니다.이산적 텍스트의 처리: 텍스트 토큰은 본질적으로 이산적(Discrete)이기 때문에, 픽셀과 같은 연속적인 공간에서 작동하는 확산 모델을 직접 적용하기 어렵습니다.주요 접근법: 텍스트를 연속적인 표현으로 학습하거나, 마스크 확산 모델(Masked Diffusion Models)과 같은 이산적 확산 모델을 개발하는 두 가지 방향이 존재합니다.성능 및 확장성: Dream7B나 LLaDA와 같은 연구들은 대규모 언어 사전 학습을 통해 자기회귀 모델(ARM)에 필적하는 성능과 스케일링 특성을 보여주었습니다.멀티모달 확장: 이러한 흐름은 LaViDA(시각 이해), LLaDA-V(시각 지시 튜닝), MMaDA(통합 확산 트랜스포머) 등으로 이어지며 시각-언어 영역(d-VLMs)으로 확장되었습니다.연구의 공백: 하지만 이러한 발전에도 불구하고, 대규모 언어 확산 모델을 **로봇 조작(Robotic Manipulation)**에 적용하려는 시도는 거의 이루어지지 않았습니다.

### 2.2. 시각-언어-동작 모델 (Vision-Language-Action Models)

일반화된 로봇 정책을 구축하기 위해 자기회귀 VLM의 강력한 이해 능력을 활용하는 VLA 모델들이 개발되어 왔습니다.선구적 연구: RT-2는 웹 스케일 데이터와 로봇 시연 데이터를 함께 학습하여 다양한 조작 작업에서 강력한 성능을 보여주었습니다.오픈소스 및 발전: OpenVLA는 최초의 오픈소스 VLA 모델로서 연구를 가속화했으며, 이후 궤적 주석을 활용한 LLARVA나 연속 제어를 위한 확산 헤드를 도입한 CogACT 등이 등장했습니다.최신 기법: $\pi_0$는 흐름 매칭(Flow-matching) 전략과 대규모 멀티태스크 데이터셋을 결합하여 탁월한 성능을 달성했습니다.차별점: 그러나 기존의 VLA 모델들은 거의 예외 없이 자기회귀(Autoregressive) 방식에 기반하고 있으며, 확산 기반 VLM의 잠재력은 여전히 미개척 상태로 남아있습니다.



---


---
