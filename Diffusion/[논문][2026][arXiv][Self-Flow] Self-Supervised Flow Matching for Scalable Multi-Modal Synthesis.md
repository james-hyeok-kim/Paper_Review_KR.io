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


## 3. Method

<p align = 'center'>
<img width="979" height="457" alt="image" src="https://github.com/user-attachments/assets/cd465e88-7dd6-40f0-a880-ced3d9574a05" />
</p>

### 3.1 Flow Matching Preliminaries (기초)

* 선형 확률 경로 (Linear Probability Path): 노이즈( $x_0$ )에서 데이터( $x_1$ )로 가는 경로를 직선으로 정의합니다.
    * $x_t = (1 - t)x_0 + t x_1$
* 목표 속도 (Target Velocity): 이 경로를 따라갈 때의 일정한 속도는 $v_t(x) = x_1 - x_0$가 됩니다.
* 학습 목표: 신경망 $v_\theta(x_t, t)$가 이 고정된 속도( $x_1 - x_0$ )를 예측하도록 MSE 손실 함수를 사용하여 학습합니다.

### 3.2 Dual-Timestep Scheduling (핵심 혁신)

* 기존의 REPA 방식이 외부 모델을 '선생님'으로 썼다면, Self-Flow는 서로 다른 시점의 자기 자신을 선생님으로 활용합니다.
* 시점 샘플링 ( $t, s$ ): 하나의 데이터에 대해 두 개의 시점을 샘플링합니다. 이때 $s$는 $t$보다 항상 작거나 같도록 설정합니다 ( $0 \leq s \leq t \leq 1$ ).
    * 학생(Student): 더 많은 노이즈가 섞인 $x_t$를 입력받습니다.
    * 교사(Teacher): 노이즈가 적어 정보가 더 뚜렷한 $x_s$를 입력받습니다.
* 정보 비대칭 (Information Asymmetry): 학생은 데이터에 대한 정보가 부족한 상태에서, 더 많은 정보를 가진 교사의 내부 상태(Feature)를 예측해야 하는 과제를 수행하게 됩니다.

### 3.3 Internal Alignment (내부 정렬 학습)

* EMA Teacher: 교사 모델은 현재 학습 중인 모델의 가중치를 지수 이동 평균(EMA)한 버전을 사용하여 학습의 안정성을 높입니다.
* 정렬 손실 함수 ( $L_{align}$ )
    * 학생 모델의 중간 레이어 특징값(Feature)과 교사 모델의 특징값을 비교하여 일치하도록 만듭니다.
    * 이를 통해 모델은 단순한 픽셀 복구(생성)를 넘어, 데이터의 추상적인 구조와 의미(표현)를 스스로 학습하게 됩니다.
* 최종 손실 함수: 생성 손실( $L_{FM}$ )과 정렬 손실( $L_{align}$ )을 합쳐서 동시에 최적화합니다.

### 3.4 Multi-Modal and Multi-Task (확장성)

* Self-Flow 설명
* 통합 아키텍처: 이미지, 비디오, 오디오를 각각의 VAE/오토인코더를 통해 잠재 공간(Latent Space)으로 보낸 뒤, 동일한 DiT(Transformer) 블록에서 처리합니다.
* 유연한 스케줄링: 데이터의 종류(양식)에 상관없이 동일한 Dual-Timestep 방식을 적용할 수 있어, 별도의 튜닝 없이도 멀티모달 확장이 용이합니다.


---

## 4. Experiments

### 1. 실험 설정 (Implementation Details)

* 모델 구조: DiT(Diffusion Transformer)를 기본 아키텍처로 사용하며, 290M부터 최대 4B(40억 개) 파라미터까지 모델 크기를 확장하며 실험했습니다.
* 데이터셋: 이미지(ImageNet, 고해상도 내부 데이터), 비디오(내부 비디오 데이터셋), 오디오(음악 및 효과음 데이터)를 모두 사용했습니다.
* 비교 대상: 외부 인코더(DINOv2)를 사용하는 최신 기법인 REPA와 아무런 추가 기법이 없는 Vanilla Flow Matching을 주요 대조군으로 삼았습니다.

### 2. 텍스트-이미지 생성 및 스케일링 법칙 (Scaling Laws)

<p align = 'center'>
<img width="977" height="276" alt="image" src="https://github.com/user-attachments/assets/53e337fd-02f5-426d-a5f2-c3ad960d5fb0" />
<img width="972" height="350" alt="image" src="https://github.com/user-attachments/assets/e0c35c3a-4ecb-4d46-a8f3-df27b0348350" />
</p>

* 수렴 속도: 동일한 품질(FID 지표 기준)에 도달하는 시간이 REPA보다 약 2.8배 빠릅니다.
* 성능 우위: 625M 파라미터 크기의 Self-Flow 모델이 1B(10억 개) 파라미터 크기의 REPA 모델보다 더 좋은 성능을 기록했습니다.
* 병목 해결: REPA는 외부 모델 성능에 의존하므로 특정 수준에서 성능 향상이 멈추는(Plateau) 현상이 발생하지만, Self-Flow는 모델이 커질수록 성능이 계속해서 향상되는 전형적인 스케일링 법칙을 따릅니다.

### 3. 비디오 및 오디오 생성 결과

* 비디오 (Video): FVD(Fréchet Video Distance) 지표에서 최고 성능을 기록했습니다. 외부 이미지 인코더에 의존하지 않기 때문에 비디오의 시간적 일관성(Temporal Consistency)을 해치지 않고도 학습이 가능했습니다.
* 오디오 (Audio): FAD(Fréchet Audio Distance) 지표를 통해 측정했을 때, 음악 생성과 일반적인 소리 생성 모두에서 Vanilla 모델보다 훨씬 뛰어난 품질을 보였습니다.

### 4. 멀티모달 공동 학습 (Joint Multi-modal Training)

* 상호 보완 효과: 각 양식(Modality)을 따로 학습할 때보다 함께 학습할 때 성능이 더 향상되는 '긍정적 전이(Positive Transfer)' 현상을 관찰했습니다.
* 이는 Self-Flow를 통해 학습된 내부 표현(Representation)이 서로 다른 데이터 형식 간의 공통된 의미 구조를 잘 파악하고 있음을 시사합니다.

### 5. 다운스트림 과제: 로보틱스 (Action Prediction)

<p align = 'center'>
<img width="497" height="496" alt="image" src="https://github.com/user-attachments/assets/a4b201c4-17f1-4212-8230-dba590f9c64f" />
</p>

* 생성 모델이 단순한 '그림 그리기'를 넘어 세상을 이해하는 '월드 모델'로서 기능할 수 있는지 테스트했습니다.
* 실험: 로봇의 시점 비디오를 보고 다음 행동(Action)을 예측하는 과제를 수행했습니다.
* 결과: Self-Flow로 학습된 모델은 시각적 추론 능력이 뛰어나, 기존 방식들보다 로봇의 복잡한 움직임을 더 정확하게 예측했습니다. 이는 모델 내부의 자가 정렬(Self-alignment) 과정이 물리적 세계에 대한 이해도를 높였음을 의미합니다.



---

## 5. Limitations and Future Work

### 1. 한계점 (Limitations)

#### 오토인코더(VAE)에 대한 의존성

* 본 논문은 DINOv2와 같은 '의미론적(Semantic)' 외부 모델 의존성은 제거했지만, 여전히 이미지를 잠재 공간(Latent Space)으로 압축하기 위해 사전 학습된 VAE(오토인코더)를 사용합니다.
* 즉, 픽셀 수준에서 직접 학습하는 것이 아니라 이미 고정된 잠재 공간 위에서 학습한다는 한계가 있습니다.

#### 계산 복잡도 (Training Overhead)

* Dual-Timestep Scheduling 방식은 학생(Student)과 교사(Teacher) 모델이 서로 다른 데이터를 처리해야 하므로, 일반적인 Flow Matching에 비해 단계당 연산량이 다소 많습니다.
* 비록 수렴 속도가 빨라 전체 학습 시간은 단축되었지만, 메모리나 연산 자원이 제한된 환경에서는 부담이 될 수 있습니다.

#### 멀티모달 간의 불균형

* 이미지, 비디오, 오디오를 동시에 학습할 때 각 데이터 간의 샘플링 비율이나 가중치 조절이 완벽하지 않을 수 있습니다.
* 특정 양식(예: 이미지)의 데이터가 압도적으로 많을 경우, 상대적으로 적은 양식(예: 오디오)의 학습 품질이 영향을 받을 수 있는 가능성이 존재합니다.

### 2. 향후 연구 방향 (Future Work)

#### 엔드투엔드(End-to-End) 학습

* 현재는 고정된 VAE를 사용하지만, 향후에는 VAE와 Flow Matching 모델을 동시에(End-to-End) 처음부터 끝까지 학습시키는 방향을 제시합니다.
* 이렇게 하면 잠재 공간 자체가 생성 모델에 최적화되어 더욱 강력한 성능을 낼 수 있을 것으로 기대합니다.

#### 더 큰 규모로의 확장 (Scaling Up)

* 현재 4B(40억 개) 파라미터까지 실험했지만, LLM(거대언어모델)처럼 수십, 수백억 개의 파라미터로 확장했을 때 Self-Flow가 보여줄 잠재력에 대해 기대를 표하고 있습니다.
* 데이터셋의 규모 역시 더 키울 계획임을 밝힙니다.

#### 지능형 에이전트 및 월드 모델(World Models)

* 단순히 고품질의 미디어를 생성하는 것을 넘어, 물리 법칙을 이해하고 행동을 예측하는 '범용 월드 모델'로의 발전을 목표로 합니다. 
* 특히 로보틱스 분야에서 환경을 시뮬레이션하고 다음 행동을 결정하는 핵심 두뇌로 Self-Flow를 활용하는 연구가 유망할 것으로 보고 있습니다.

#### 이산 데이터와 연속 데이터의 통합

* 현재는 시각/청각과 같은 연속적인 데이터를 주로 다루지만, 텍스트나 코드와 같은 이산적인(Discrete) 데이터까지 하나의 Flow Matching 프레임워크 안에서 완벽하게 통합하는 연구를 계획하고 있습니다.


---

