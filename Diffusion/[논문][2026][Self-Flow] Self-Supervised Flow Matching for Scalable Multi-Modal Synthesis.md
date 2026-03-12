# Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis

저자 : 

Hila Chefer * 1 Patrick Esser * 1

Dominik Lorenz 1 Dustin Podell 1 Vikash Raja 1 Vinh Tong 1 Antonio Torralba 1 2 Robin Rombach 1

1Black Forest Labs 

2MIT. ∗Equal contribution. 

발표 : 2026년 3월, arXiv

논문 : [PDF](https://arxiv.org/pdf/2603.06507)

---

## 0. Summary 및 의의

* 기존의 생성 모델(Diffusion, Flow Matching)이 성능 향상을 위해 DINOv2와 같은 외부의 사전 학습된 인코더

### 핵심 메커니즘: Dual-Timestep Scheduling

* 정보 비대칭성 생성: 입력 데이터의 토큰들에 서로 다른 노이즈 수준( $t, s$ )을 적용합니다.
* 자가 학습 구조: 학생(Student) 모델은 노이즈가 심한 입력을 받고, 더 깨끗한 입력을 본 교사(EMA Teacher) 모델의 내부 표현(representation)을 복구하도록 학습됩니다.
* 통합 목적 함수: 생성(Flow Matching) 손실과 표현 학습(Alignment) 손실을 동시에 최적화하여 생성 능력과 의미론적 이해도를 함께 높입니다. 

### 의의

1) 외부 모델 의존성 및 병목 현상 해결
2) 뛰어난 확장성(Scalability)과 수렴 속도
3) 범용 멀티모달 합성의 가능성 입증
4) 로보틱스 및 월드 모델로의 확장


### 생성 모델 방법론 비교: REPA vs. Self-Flow

<div align = 'center'>

| 비교 항목 | 기존 방식 (REPA 등) | **Self-Flow (제안 방식)** |
| :--- | :--- | :--- |
| **의존성** | DINO, SigLIP 등 외부 인코더 필수 | 모델 내부 자가 학습 (Self-Supervised) |
| **수렴 속도** | 상대적으로 느림 (성능 정체 발생 가능) | REPA 대비 **약 2.8배 빠른 수렴** |
| **양식 확장성** | 비디오/오디오에서 성능 저하 위험 존재 | 이미지, 비디오, 오디오 모든 양식에서 우수 |
| **스케일링** | 모델 확장 시 효율 저하 및 병목 현상 발생 | 연산량과 모델 크기에 비례해 성능 지속 향상 |
| **학습 메커니즘** | 외부 특징값에 생성 특징을 강제로 정렬 | 정보 비대칭 스케줄링을 통한 자가 추론 학습 |

</div>


---

## 1. Introduction

<p align = 'center'>
<img width="1077" height="515" alt="image" src="https://github.com/user-attachments/assets/85680b72-1aa9-4ef3-bde8-599880018f5b" />
</p>

### 1. 배경 및 문제 제기

* 외부 인코더의 활용
    * 최근의 생성 모델들(Stable Diffusion, DiT 등)은 DINO와 같은 고정된 외부 이미지 인코더를 사용하여 성능 향상
* 표현력의 간극
    * DINO와 같은 인코더는 생성이 아닌 '판별(Semantically clustering images)'을 목적으로 학습
    * 외부 모델이 생성에 도움을 준다는 사실
    * 정작 흐름(Flow) 모델 자체는 생성에는 능숙하지만 강한 의미론적 표현(Semantic representations)을 스스로 학습하지 못하고 있음
* 외부 정렬의 실용적 한계
    * 스케일링 법칙의 불일치: 더 강력한 외부 인코더를 사용하더라도 생성 품질이 비례해서 좋아지지 않거나, 오히려 악화되는 현상이 발생.
    * 모달리티 일반화 실패: 비디오나 오디오 생성 시, 이미지용 외부 인코더를 적용하면 성능이 저하되는 경우가 많아 멀티모달 모델에 적합하지 않음.
    * 예측 불가능성: 텍스트 감독을 받은 SigLIP 2보다 DINOv2가 텍스트-이미지 생성에 더 효과적인 등, 어떤 인코더가 특정 작업에 유리할지 예측하기 어렵습니다.

### 2. Self-Flow의 제안

*  자가 지도 학습(Self-supervised learning) 프레임워크를 흐름 매칭(Flow matching)에 직접 통합하는 방식
*  Dual-Timestep Scheduling: 입력 토큰의 일부에는 강한 노이즈를, 일부에는 약한 노이즈를 적용하여 '정보 비대칭' 상태를 만듭니다.
*  자가 학습 메커니즘: 모델(학생)은 노이즈가 심한 토큰으로부터, 상대적으로 깨끗한 토큰을 본 교사(EMA 버전의 자기 자신)의 특징을 예측하도록 학습됩니다.
    * Exponential Moving Average, 지수 이동 평균
    * $EMA_{new} = \alpha \cdot EMA_{old} + (1 - \alpha) \cdot Model_{current}$ 
*  기대 효과: 이 방식은 외부 모델 없이도 모델 내부에서 강한 의미론적 특징과 구조적 일관성을 동시에 학습하게 하며, 이미지·비디오·오디오 등 다양한 양식에 걸쳐 범용적으로 적용 가능합니다.

### 3. 주요 기여점

* 수렴 속도 향상: 외부 정렬 방식(REPA)보다 약 2.8배 빠른 수렴을 달성했습니다.
* 스케일링 효율성: 모델의 크기와 연산량이 늘어남에 따라 생성 품질이 정비례하여 향상되는 일관된 스케일링 거동을 보였습니다.
* 품질 개선: 얼굴, 손과 같은 복잡한 구조의 묘사 정확도와 비디오의 시간적 일관성을 크게 개선했습니다.


---


## 2. Related Work

### 2.1. 흐름 매칭 및 확산 모델 (Flow Matching and Diffusion Models)

* 배경: 확률적 미분 방정식(SDE)이나 상미분 방정식(ODE)을 이용해 노이즈로부터 데이터를 생성하는 방식이 표준이 되었습니다.
* 최신 트렌드: 특히 Flow Matching(FM)은 확산 모델(Diffusion)보다 학습이 효율적이고 샘플링 경로가 직선에 가까워 성능이 뛰어납니다.
* Backbone: 본 논문은 이 과정에서 DiT(Diffusion Transformer) 구조를 채택하여 모델 크기를 키웠을 때의 확장성(Scalability)을 확보했습니다.

#### 2.1.1. Flow Matching 학습 원리

* 일반적인 확산 모델(Diffusion)이 복잡한 곡선 경로를 따라 노이즈를 제거한다면, Flow Matching은 직선 경로를 따라 데이터를 생성하도록 학습합니다.
* 선형 보간 (Linear Interpolation): 노이즈 $x_0$와 데이터 $x_1$ 사이의 임의의 지점 $x_t$를 생성합니다.
    * $x_t = (1 - t)x_0 + t x_1$ (여기서 $t \in [0, 1]$ )
* 목표 속도 (Target Velocity): $x_t$가 $x_1$으로 이동하는 속도는 $x_1 - x_0$로 고정된 직선 형태가 됩니다.
* 학습 (Training): 신경망 $v_\theta$가 현재 시점 $t$와 상태 $x_t$를 입력받아, 목표 속도인 $x_1 - x_0$를 예측하도록 학습합니다

$$\mathcal{L}_{FM} = \| v_\theta(x_t, t) - (x_1 - x_0) \|^2$$

#### 2.1.2. Self-Flow의 특수 학습 전략 (Dual-Timestep)

* 이 논문은 여기에 Dual-Timestep Scheduling을 추가하여 성능을 높입니다.
* 두 개의 시점 샘플링: $t$(학생용, 노이즈 많음)와 $s$(교사용, 노이즈 적음)를 샘플링합니다 ( $0 \leq s < t \leq 1$ ).
* 자가 정렬 (Self-Alignment): 모델의 가중치를 복사한 교사 모델(EMA)이 $x_s$를 보고 추출한 특징(Representation)을, 노이즈가 더 심한 $x_t$를 본 학생 모델이 예측하도록 추가 손실 함수를 둡니다.

#### 2.1.3. 예제 코드 (PyTorch 스타일)

* Self-Flow의 핵심인 Flow Matching 손실 함수와 Dual-Timestep을 활용한 학습 루프 예시입니다.

```python
import torch
import torch.nn.functional as F

def train_step(model, model_ema, x1, cond):
    """
    x1: 데이터 (Batch, Channel, H, W)
    cond: 텍스트 또는 클래스 조건
    """
    batch_size = x1.shape[0]
    device = x1.device

    # 1. 시점 샘플링 (Dual-Timestep for Self-Flow)
    t = torch.rand(batch_size, device=device)
    # s는 t보다 작거나 같은 값으로 설정 (정보 비대칭 생성)
    s = t * torch.rand(batch_size, device=device) 

    # 2. 노이즈 생성 (x0)
    x0 = torch.randn_like(x1)

    # 3. 선형 보간을 통한 x_t와 x_s 생성
    # x_t = (1-t)*x0 + t*x1
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
    xs = (1 - s[:, None, None, None]) * x0 + s[:, None, None, None] * x1

    # 4. 목표 속도 설정 (Velocity Target)
    target_velocity = x1 - x0

    # 5. 모델 예측 (학생 모델)
    # pred_v: 생성 속도 예측, student_feat: 내부 표현(Representation)
    pred_v, student_feat = model(xt, t, cond, return_feat=True)

    # 6. 교사 모델(EMA)의 특징 추출 (no_grad)
    with torch.no_grad():
        _, teacher_feat = model_ema(xs, s, cond, return_feat=True)

    # 7. 손실 함수 계산
    # (1) Flow Matching Loss: 직선 경로 학습
    loss_fm = F.mse_loss(pred_v, target_velocity)

    # (2) Self-Alignment Loss: 학생이 교사의 특징을 예측 (Self-Flow 핵심)
    loss_align = F.mse_loss(student_feat, teacher_feat)

    # 최종 손실 (람다는 가중치 하이퍼파라미터)
    loss = loss_fm + 0.1 * loss_align

    return loss

# --- 보조 함수: 추론(Sampling) 시 사용되는 ODE Step ---
@torch.no_grad()
def sample_step(model, xt, t, dt, cond):
    # 모델이 예측한 속도 방향으로 한 걸음 이동 (Euler method)
    v = model(xt, t, cond)
    return xt + v * dt
```

* ODE(상미분 방정식, Ordinary Differential Equation)
    * 이는 "시간($t$)에 따른 상태($x$)의 변화량은 함수 $v$에 의해 결정된다"는 뜻. 

$$\frac{dx}{dt} = v(x, t)$$


* 오일러 방법 (Euler Method)
    * ODE를 수치적으로 해결하는 가장 고전적인 방법인 오일러 방법(Euler Method)을 구현한 것
    * 미분 방정식 $\frac{dx}{dt} = v$를 풀기 위해 시간 간격 $dt$를 아주 작게 쪼개어 한 걸음씩 나아가는 방식입니다.
    * 연속적인 흐름(Continuous Flow)을 컴퓨터가 이해할 수 있도록 이산적인 단계(Discrete Steps)로 바꾸어 계산하기 때문에 "ODE 솔버(Solver)를 이용한 샘플링"이라고 부릅니다.

$$x(t + \Delta t) \approx x(t) + v(x(t), t) \cdot \Delta t$$


### 2.2. 자가 지도 표현 학습 (Self-Supervised Representation Learning, SSL)

* MAE (Masked Autoencoders): 이미지의 일부를 가리고 복구하는 방식으로 시각적 이해력을 키우는 방식입니다.
* DINO / DINOv2: 자가 증류(Self-distillation)를 통해 객체의 경계나 의미론적 구조를 매우 잘 파악하는 특징을 가집니다.
* 의의: 이러한 SSL 모델들은 "이미지가 무엇인지" 이해하는 데는 탁월하지만, 새로운 이미지를 만드는 "생성" 능력은 없습니다.

### 2.3. 생성과 표현의 정렬 (Aligning Generation and Representation)

* REPA (Representation Alignment): 생성 모델의 내부 특징을 DINOv2 같은 외부 인코더의 특징과 강제로 일치시키는 방식입니다.
* 문제점: 성능은 좋아지지만, 외부 모델에 의존해야 하며 모델을 키울 때 성능 향상이 멈추는(Plateau) 병목 현상이 발생합니다.
* Self-Flow의 차별점: 외부 모델의 도움 없이, 생성 모델 스스로가 SSL 학습(Dual-Timestep Scheduling)을 수행하여 생성과 이해를 동시에 달성하고자 합니다.

### 2.4. 확장 가능한 멀티모달 합성 (Scalable Multi-Modal Synthesis)

* 기존 연구: 비디오나 오디오를 위해 각각 별도의 모델을 만들거나, 매우 복잡한 구조를 사용하는 경우가 많았습니다.
* Self-Flow의 접근: 단일 아키텍처 내에서 이미지, 비디오, 오디오를 통합적으로 학습할 수 있는 확장성 있는 방법을 제시하며, 특히 외부 모델 없이도 모든 양식(Modality)에서 성능이 향상됨을 강조합니다.


---




---
