# SnapFlow: One-Step Action Generation for Flow-Matching VLAs via Progressive Self-Distillation

저자 : 

* Wuyang Luan, Jilin University, luanwy25@mails.jlu.edu.cn
* Junhui Li, Chongqing University, junhuili@stu.cqu.edu.cn
* Weiguang Zhao, University of Liverpool, weiguang.zhao@liverpool.ac.uk
* Wenjian Zhang∗, GenY, zhangwenjian@genycc.cn
* Tieru Wu, Jilin University, wutr@jlu.edu.cn
* Rui Ma†, Jilin University, ruim@jlu.edu.cn

발표 : 2026년 4월 7일에 arXiv

논문 : [PDF](https://arxiv.org/pdf/2604.05656)


---

## 0. Summary 

<p align = 'center'>
<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/a6b9d2f7-e3ae-4fdb-ab07-6e6a4f31b647" />
</p>

### 0.1. 문제

* $\pi_0$, $\pi_{0.5}$, SmolVLA와 같은 최신 VLA(Vision-Language-Action) 로봇 제어 모델들은 '유동 일치(Flow-Matching)' 방식을 사용하여 뛰어난 성능을 보입니다. 
* 하지만 행동을 생성할 때 보통 10번의 반복적인 디노이징(10 Euler steps) 과정을 거쳐야 하므로 추론 속도가 매우 느리다는 치명적인 단점이 있습니다.

### 0.1 핵심 아이디어

* SnapFlow의 해결책: 다단계로 이루어진 디노이징 과정을 단 한 번의 연산(1-NFE, One-Step)으로 압축하는 '점진적 자가 증류(Progressive Self-Distillation)' 기법을 제안합니다.

### 0.2 작동원리

* 작동 원리: 기존의 표준 Flow-Matching 학습 방식에, 모델 스스로의 예측을 활용해 중간 지점을 거쳐 가는 '2단계 오일러 숏컷(Two-Step Euler Shortcut)' 방식을 혼합하여 학습시킵니다.

* 왜 그냥 1단계로 학습시키면 안 되나요?
    * 기존의 Flow-Matching 모델은 아주 짧은 거리를 조금씩 이동하는 법(10단계)만 배웠기 때문에, 갑자기 긴 거리를 한 번에 가라고 하면 궤적을 벗어나는 '궤적 드리프트(Trajectory Drift)' 현상이 발생합니다.
    * 단순히 정답( $x_0$ )과 노이즈( $\epsilon$ )를 잇는 직선으로 학습시키면, 데이터의 복잡한 곡률을 무시하게 되어 성능이 떨어집니다.
* 2단계 오일러 숏컷의 작동 순서
    * SnapFlow는 학습 중에 모델에게 다음과 같은 '중간 점검' 과정을 거치게 합니다.
    * 시작점 확인: 완전한 노이즈 상태( $t=1$ )에서 모델에게 현재 속도( $v_1$ )가 얼마인지 물어봅니다.
    * 중간지점 이동: 모델이 대답한 속도를 따라 딱 절반( $0.5$ 단계 )만큼만 가봅니다. 이 지점을 중간지점( $x_{0.5}$ )이라고 부릅니다.
        * 1단계 다 가는것보다 0.5단계만 가는게 더 정확하기 때문에  
    * 중간점 속도 측정: 그 중간지점에서 다시 모델에게 "여기서의 속도( $v_{0.5}$ )는 뭐야?"라고 물어봅니다.
    * 최종 지름길 생성: 처음에 말한 속도( $v_1$ )와 중간에서 측정한 속도( $v_{0.5}$ )를 평균 냅니다. 이것이 바로 타겟 속도( $v_{target}$ )가 됩니다.
    * 1단계 점프 학습: 이제 모델에게 "너는 다음에 노이즈를 만났을 때, 다른 생각 하지 말고 바로 이 타겟 속도로 한 번에 가!"라고 가르칩니다.

* '혼합 학습(Mixing)'의 의미 ( $\alpha=0.5$ )
    * SnapFlow는 숏컷만 배우는 것이 아니라, 기존의 공부 방식과 병행합니다.
    * 표준 Flow-Matching ( $\alpha$ 비중): 모델이 기본적인 물리 법칙(속도 필드)을 잊지 않도록 짧은 단계를 예측하는 법을 계속 복습합니다.
    * 숏컷 증류 ( $1-\alpha$ 비중): 위에서 만든 2단계 숏컷 타겟을 보며 한 번에 점프하는 법을 익힙니다.
    * 시너지 효과: 표준 학습이 속도 예측의 정확도를 유지해주고, 숏컷 학습이 그 능력을 1단계로 압축해주어 속도와 정확도를 모두 잡는 결과를 냅니다.

* 혼합 학습
    * 학습 할 데이터를 50(FM):50(ShortCut) 으로 나눈다
    * ShortCut 만 Distilation 학습
        * 0.5간 다음 속도 측정, 1 간 다음 속도 측정 및 이를 평균 ( $v_{target} = \frac{v_1 + v_{0.5}}{2}$ )
        * Teacher모델은 10 -> 5 -> 1만 수행해서 보여줌
      
$$\mathcal{L} = \alpha \cdot \mathcal{L}_{FM} + (1-\alpha) \cdot \lambda \cdot \mathcal{L}_{shortcut}$$

### 0.3. 핵심 결과 요약

* 수학적 정교함: 단순히 정답 데이터를 보고 배우는 것보다, 모델 자신의 예측치를 활용해 사다리꼴 공식(Trapezoidal rule)으로 적분 오차를 줄였기 때문에 더 정확합니다.
* 인프라 효율: 이 모든 과정이 별도의 외부 교사 모델 없이 모델 스스로 수행(Self-distillation)되므로 비용이 적게 듭니다.
* 장점(Plug-and-play)
    * 외부의 교사(Teacher) 모델이 필요 없고, 기존 모델의 구조를 바꿀 필요도 없습니다.
    * 단지 '0으로 초기화된 타겟 시간 임베딩(Zero-initialized target-time embedding)'만 추가하면 되며, 단일 A800 GPU에서 약 12시간 만에 학습이 완료됩니다.

### 0.4. 벤치마크 및 효과성 요약

* 평가 환경 (Benchmarks)
    * 30억(3B) 파라미터 크기의 $\pi_{0.5}$ 모델을 대상으로 로봇 조작 벤치마크인 LIBERO (4개 제품군, 40개 태스크, 400개 에피소드)에서 평가를 진행했습니다.
    * 5억(500M) 파라미터 크기의 경량 모델인 SmolVLA를 대상으로도 성능을 검증했습니다.

* 효과성 - 작업 성공률 (Quality)
    * $\pi_{0.5}$ 모델에 SnapFlow를 적용해 1-Step으로 구동했을 때, LIBERO 벤치마크 평균 성공률 98.75%를 달성했습니다.
    * 이는 기존 10단계 모델의 성공률인 97.75%를 오히려 능가하는 수치입니다.
    * SmolVLA에서는 평균 제곱 오차(MSE)를 8.3% 감소시켰습니다.

* 효과성 - 속도 향상 (Speed)
    * 디노이징 단계에서만 9.6배의 속도 향상을 이루어냈습니다.
    * 결과적으로 전체(End-to-End) 지연 시간을 274ms에서 83ms로 대폭 단축시켰습니다 (약 3.3배 전체 속도 향상).
    * SmolVLA 모델에서도 전체 속도를 3.56배 향상시켰습니다.

* 안정성 유지: 한 번의 예측으로 여러 스텝을 실행하는 환경( $n_{act}=5$ )에서도 기존 모델(90%)보다 높은 93%의 성공률을 기록하며 뛰어난 안정성을 증명했습니다.



---
