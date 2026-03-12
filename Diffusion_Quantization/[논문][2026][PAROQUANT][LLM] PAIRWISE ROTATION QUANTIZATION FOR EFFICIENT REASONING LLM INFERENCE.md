# PAROQUANT: PAIRWISE ROTATION QUANTIZATION FOR EFFICIENT REASONING LLM INFERENCE

저자 : 
Yesheng Liang3,† Haisheng Chen3,‡ Zihan Zhang3 Song Han1,2 Zhijian Liu3

1NVIDIA 2MIT 3UC San Diego

†Algorithm lead ‡System lead

발표 : ICLR 2026 (International Conference on Learning Representations), arXiv 2026년 2월 14일

논문 : [PDF](https://arxiv.org/pdf/2511.10645)

---

## 0.1. Summary

**LLM Quantization**

* ParoQuant는 추론 능력이 강조된 최신 LLM(Reasoning LLMs)을 타겟으로 하는 사후 훈련 양자화(PTQ) 기법
* 알고리즘-시스템 공동 설계(Co-design) 접근 방식
* 기술적 핵심: Scaled Pairwise Rotation
    * 독립적 기븐스 회전(Independent Givens Rotations)
        * 하드웨어 효율성을 위해 서로 간섭하지 않는 독립적인 채널 쌍을 선택하여 회전을 적용합니다.
        * 이를 통해 GPU 병렬 처리를 극대화하면서 가중치의 아웃라이어(Outlier)를 효과적으로 억제합니다.
    * 채널별 스케일링(Channel-wise Scaling): 회전과 스케일링을 결합하여 가중치의 동적 범위를 좁히고 양자화에 친화적인 형태로 변환합니다.
    * 전용 추론 커널: GPU의 공유 메모리와 레지스터를 활용하는 융합된(Fused) CUDA 커널을 구현하여 추론 지연 시간을 최소화했습니다.


## 0.2. 연구의 의의 및 기여도

* 장기 추론 시 오차 누적 문제 해결
    * 최근의 Reasoning 모델들은 긴 사고 과정(Chain-of-Thought)을 거치며 수만 개의 토큰을 생성합니다.
    * 기존 AWQ(Activation-Aware Weight Quantization) 같은 방식은 토큰 생성 길이가 길어질수록 양자화 오차가 누적되어 정확도가 급격히 떨어지는 한계가 있었으나, ParoQuant는 이를 효과적으로 방어했습니다.

* 성능과 효율의 균형 (Pareto Frontier 확장)
    * 정확도: 가중치 전용 양자화(Weight-only)에서 기존 AWQ 대비 추론 작업 정확도를 평균 2.4% 개선했으며, 벡터 양자화 방식인 QTIP(Quantization with Trellises and Incoherence Processing, vector quantization 기법 중 하나) 수준의 정확도를 달성했습니다.
        * QTIP의 주요 특징
            * 벡터 양자화 기반
            * Hadamard 변환 활용
            * Trellis Quantization: Trellis(격자) 구조를 활용하여 양자화 과정에서 발생하는 오차를 최소화하는 최적의 경로를 찾습니다
    * 속도: 고도로 최적화된 커널 덕분에 AWQ 대비 오버헤드는 10% 미만에 불과하며, Hadamard 변환 기반 방식들보다 15~30% 더 빠른 속도를 보여줍니다.


* 하드웨어 가속 가능성 제시
    * GPU의 병렬 구조를 고려해 회전 단위를 독립적으로 설계함으로써 실제 서비스 환경(Large-batch serving)에서의 배포 가능성을 입증했습니다.
    * 또한, MXFP4와 같은 차세대 하드웨어 네이티브 포맷에서도 우수한 성능을 보였습니다.



### Appendix Trellis Quant

<p align = 'center'>
<img width="928" height="422" alt="image" src="https://github.com/user-attachments/assets/60be5837-42ee-4ae4-a64c-47d3eaf83914" />
</p>

* 개별 오차 측정: 각 노드에서 실제 원본 값과 양자화 후보 값 사이의 거리(오차)를 계산합니다.
* 누적 오차 계산: 첫 번째 데이터부터 마지막 데이터까지 선을 따라 이동하며, 그동안 쌓인 총 오차의 합을 구합니다.
* 비타비 탐색 (Viterbi Search): 단순히 지금 당장 오차가 적은 값을 고르는 게 아니라, 마지막에 도달했을 때 전체 누적 오차가 가장 작아지는 하나의 연결 선(경로)을 선택합니다.

* 격자(Trellis)의 마지막 단계까지 계산이 끝나면, 전체 누적 오차가 가장 적은 최종 노드가 정해짐
* 역추정 하겨 최종 list를 얻게된다
* 최적 경로상의 값이 코드북의 7번 값이라면 '7'이라는 짧은 번호만 저장
    * 개별 저장이나 경로 저장이나 결국 bit width는 같다
    * 나중에 변경할때, 경로로 변경하다보니 전체 모델 선응에 유리하게 선택 가능


---

## 1. Introduction

<p align = 'center'>
<img width="922" height="415" alt="image" src="https://github.com/user-attachments/assets/bfb5248e-29ae-4480-aba6-c1eb39b149ef" />
</p>

### 1.1 LLM 양자화의 필요성과 한계

* 배포 효율성: LLM의 거대한 크기와 메모리 사용량은 추론 비용을 높이고 온디바이스 배포를 어렵게 만듭니다.
* 가중치 및 활성화 양자화: 추론 속도를 높이기 위해 가중치를 4비트(INT4) 등으로 낮추거나, 대규모 서빙 시에는 활성화 값까지 양자화하여 연산 비용을 줄이려 시도합니다.
* 아웃라이어 문제: 가중치와 활성화 값 모두에 존재하는 아웃라이어(Outlier)는 저비트 양자화 시 정밀도를 심각하게 떨어뜨리는 주요 원인이 됩니다.

### 1.2 기존 연구의 문제점

기존의 사후 훈련 양자화(PTQ) 방식들은 정확도와 효율성 사이의 균형을 잡는 데 어려움을 겪고 있습니다.
* AWQ: 널리 사용되는 빠른 방식이지만, Qwen3-4B 모델을 4비트로 양자화했을 때 MMLU-Pro에서 약 2.8%의 정확도 하락이 발생하는 등 정밀도 면에서 아쉬움이 있습니다.
* QTIP: 최첨단 정확도를 보여주지만, 아웃라이어를 처리하기 위한 추가 오버헤드 때문에 AWQ보다 약 30% 정도 느립니다.

### 1.3 새로운 도전: Reasoning LLM (추론형 모델)

* 최근 등장한 OpenAI o1, DeepSeek-R1과 같은 추론형 모델들은 독특한 도전 과제를 제시합니다.
* 오차 누적: 이 모델들은 수만 개의 사고 체인(CoT) 토큰을 생성하는데, 각 생성 단계마다 발생하는 미세한 양자화 오차가 누적되어 긴 생성 결과물에서는 정밀도가 급격히 무너집니다.
* 계산 비용: 긴 시퀀스를 생성해야 하므로 양자화 과정 자체가 성능에 미치는 오버헤드를 최소화하는 것이 매우 중요합니다.

### 1.4 ParoQuant의 핵심 제안

* 회전의 효과성: 회전(Rotation) 변환은 아웃라이어를 억제하는 데 매우 효과적입니다.
* 파라미터 효율성: 전체 회전 행렬 대신 희소하게 매개변수화된(Sparsely parameterized) 회전만으로도 충분한 효과를 낼 수 있습니다.
* 결론적으로 ParoQuant는 알고리즘과 시스템의 공동 설계를 통해 AWQ보다 정확하고(추론 작업에서 평균 2.4% 개선), QTIP보다 빠른(약 25% 가속) 결과를 달성하고자 합니다.

---

## 2. Background and Related Work

### 2.1 LLM 양자화 (LLM Quantization)

$$Q(X) = \text{clamp}\left(\left\lfloor \frac{X}{s} \right\rfloor + z, 0, 2^{b}-1\right) \quad (1)$$

* 선형 양자화 공식: 가장 기본적인 Round-to-Nearest (RTN) 방식은 가중치 $X$를 스케일 인자( $s$ )와 제로 포인트( $z$ )를 이용해 정수로 변환합니다.
* 블록 단위 양자화 (Block-wise Quantization): 행렬 전체를 하나로 양자화하는 대신, 일정 크기( $g$, 예: 128)의 채널 그룹마다 별도의 $s$와 $z$를 계산합니다. 이는 아웃라이어를 특정 그룹 내에 가두어 전체적인 정확도 손실을 줄여줍니다.
* 아웃라이어 문제: LLM에는 특정 채널에 값이 매우 큰 '아웃라이어'가 존재하며, 이들이 제한된 양자화 범위를 독점하여 나머지 일반 값들의 정밀도를 파괴합니다.

#### 아웃라이어 해결을 위한 3가지 기존 전략

* 분리 저장: 아웃라이어만 따로 고정밀도(FP16 등)로 저장하는 방식입니다.
* 비균등 분포 설계: 데이터 분포에 최적화된 복잡한 양자화 알고리즘을 설계하는 방식입니다.
* 가중치 변환 (Weight Transformation): 양자화 전에 가중치를 '양자화하기 쉬운 형태'로 수학적으로 변형하는 방식입니다.

### 2.2 등가 가중치 변환 (Equivalent Weight Transform)

$$Y = XW + b = (XT^{-1})(TW) + b \quad (2)$$

* 수학적 원리: $Y = XW$인 선형 레이어에서 가중치 $W$에 변환 $T$를 적용하고, 입력 $X$에는 그 역행렬인 $T^{-1}$을 적용하면 결과값은 동일하게 유지됩니다 ( $Y = (XT^{-1})(TW)$ ).
* 채널별 스케일링 (Channel-wise Scaling): 대각 행렬을 사용하여 채널별 크기를 맞춥니다. 추가 연산 비용 없이 이전 연산에 병합될 수 있다는 장점이 있습니다.
* 회전 (Rotation): 직교 행렬을 사용하여 채널 간의 상호작용을 통해 아웃라이어를 분산시킵니다. 스케일링보다 성능은 좋지만, 추론 시 실시간 계산이 필요하여 연산 비용(Overhead)이 발생하는 단점이 있습니다.




---

## 3. Motivation

<p align = 'center'>
<img width="617" height="294" alt="image" src="https://github.com/user-attachments/assets/adcea13a-7898-4b76-806a-d7f9f9e994a9" />
</p>

### 3.1. 긴 문장 생성 시의 오차 누적 (Quantization Error Accumulates)

* 추론 모델의 특성: Reasoning 모델은 수만 개의 사고 체인(CoT) 토큰을 생성합니다.
* 오차 누적: 각 디코딩 단계에서 발생하는 아주 작은 양자화 오차가 매 스텝 누적되어, 결과적으로 모델의 최종 답변 정확도를 급격히 떨어뜨립니다.
* 예시: Qwen3-4B 모델을 AWQ로 양자화했을 때, MMLU-Pro 정확도가 71.0에서 68.2로 하락하는 현상이 관찰되었습니다.

### 3.2. 회전 변환의 딜레마 (Expressive but Expensive)

* 성능적 우위: 회전은 단순 스케일링보다 가중치 분포를 양자화하기 좋게 만듭니다.
* 연산 비용 문제: 임의의 회전 행렬을 적용하려면 복잡한 행렬 곱셈(FP16 연산)이 필요하며, 이는 양자화로 얻은 효율성을 상쇄해 버립니다.
* 기존 방식의 한계:
    * SpinQuant: 특정 레이어에만 적용 가능하며 모든 레이어에 범용적으로 쓰기 어렵습니다.
    * Hadamard 기반 (QuIP# 등): 고정된 변환을 사용하여 모델 고유의 가중치 분포를 반영하지 못하며, 추론 속도를 약 30% 정도 늦춥니다.

### 3.3. 회전 파라미터의 중복성 (Many Redundant Parameters)

* 핵심 실험: 전체 채널 쌍 중에서 상위 10%의 중요한 채널 쌍만 선택하여 회전시켜도, 전체 행렬을 회전시킨 것과 거의 대등한 오차 감소 효과가 있었습니다.
* 기회: 아웃라이어 채널과 일반 채널 사이의 회전만 선별적으로 수행한다면, 연산은 훨씬 가볍게 유지하면서도 회전의 강력한 성능을 그대로 가져갈 수 있다는 결론에 도달합니다.


---

## 4. Method

### 4.1 Scaled Pairwise Rotation (계층적 회전 설계)

저자들은 행렬 곱셈을 직접 수행하는 대신, 하드웨어 효율적인 세 단계를 거쳐 변환을 설계했습니다.

#### 4.1.1 기븐스 회전 (Givens Rotation)

* 원리: 전체 행렬 곱셈 대신, 소수의 채널 쌍( $i, j$ )을 선택하여 해당 평면에서만 회전시키는 방식입니다.
* 연산: 아래와 같은 단순한 벡터 연산으로 수행되어 메모리와 연산량을 크게 줄입니다.
    * $W^{(k)}[i,:] = \cos \theta_k \cdot W^{(k-1)}[i,:] - \sin \theta_k \cdot W^{(k-1)}[j,:] \quad (4)$
    * $W^{(k)}[j, :] = \sin \theta_k \cdot W^{(k-1)}[i, :] + \cos \theta_k \cdot W^{(k-1)}[j, :] \quad (4)$ 

#### 4.1.2 독립적 회전 (Independent Rotation)

* 문제 해결: 여러 기븐스 회전이 서로 겹치는 채널을 가지면(종속성), 순차적으로 계산해야 하므로 GPU 병렬 처리가 불가능해집니다.
* 해결책: 각 채널이 최대 하나의 쌍에만 포함되도록 제한하여(Independent Pairs), 모든 회전 연산이 동시에 병렬로 실행될 수 있게 했습니다.
* 블록 단위 호환성: 이 방식은 블록 단위 양자화와 결합하여, 각 그룹 내에서만 최적의 쌍을 선택할 수 있게 해줍니다.
    * 그룹 내의 모든 채널이 각각 하나의 파트너와 짝을 이뤄 동시에 회전

#### 4.1.3 독립적 회전의 연속 적용 (Series of IR)

* 표현력 강화: 독립적인 회전 한 번은 파라미터 수가 적어 복잡한 분포를 잡기 어렵습니다.
* 방법: 소수(예: 8개)의 독립적 회전을 연속적으로 적용하여 전체 회전 행렬에 가까운 표현력을 확보했습니다.
    * 행렬내에서 회전을 여러번 진행, 다른 Pair들 이랑

#### 4.1.4 채널별 스케일링 결합

* 최종 변환: $T_{\mathcal{P},\Theta,\alpha}(W) = (\prod_{t=1}^{K} R(\mathcal{P}_t, \Theta_t)) \cdot \text{diag}(\alpha) \cdot W$
    * $\text{diag}(\alpha)$ (Channel-wise Scaling), 대각 행렬(Diagonal Matrix)
    * $R(\mathcal{P}_t, \Theta_t)$ (Independent Rotation)
        * $\mathcal{P}_t$ (Pairs): 회전시킬 채널들의 짝꿍 조합
        * $\Theta_t$ (Angles): 각 쌍이 어느 정도로 회전할지를 결정하는 각도(파라미터)
* 효과: 스케일링으로 전체적인 크기를 맞추고, 회전으로 국소적인 아웃라이어를 억제합니다

### 4.2 계층별 최적화 (Layer-wise Optimization)

* 변환 파라미터를 찾기 위해 두 단계의 최적화를 거칩니다.
* 1단계 (Transform Optimization): 회전 각도( $\theta$ )와 스케일링 인자( $\alpha$ )를 최적화하여 가중치를 양자화 친화적으로 만듭니다.
* 2단계 (Fine-tuning): EfficientQAT와 유사하게 가중치와 양자화 파라미터( $s, z$ )를 미세 조정하여 오차를 최소화합니다.
* 오차 보상: 이전 레이어의 양자화 오차를 다음 레이어가 보상하도록 설계하여 최종 정확도를 높였습니다.

### 4.3 효율적인 변환 커널 (Algorithm-System Co-design)

* 시스템 수준에서 하드웨어 활용도를 극대화했습니다.
* 3단계 병렬화: 토큰(Token), 채널 그룹(Channel Group), 채널 쌍(Pair) 세 수준에서 모두 병렬 처리를 수행합니다.
* 메모리 최적화: 그룹 크기를 작게(예: 128) 유지하여 활성화 데이터가 On-chip 공유 메모리에 들어가게 함으로써 지연 시간을 획기적으로 줄였습니다.
* Hadamard 대비 우위: 종속성이 많은 Hadamard 변환보다 채널 차원이 클수록 더 높은 가속 성능을 보여줍니다.


---

## 5. Evaluation

### 5.1 실험 설정 (Models & Tasks)

* 대상 모델: LLaMA-2 (7B), LLaMA-3 (8B, 70B), LLaMA-3.1 Instruct (8B), DeepSeek-R1-distilled (8B), Qwen3 (1.7B ~ 14B) 등 최신 모델들을 망라합니다.
* 평가 지표:Perplexity (PPL): WikiText2 및 C4 데이터셋을 통한 언어 모델의 근본적인 정밀도 측정.
* Reasoning Tasks (추론 작업): MMLU-Pro, GPQA Diamond, AIME-24/25 등 긴 사고 과정이 필요한 벤치마크.Non-Reasoning Tasks: BoolQ, ARC, HellaSwag 등 일반 상식 문제.

### 5.2 주요 결과

#### 1. 정확도 결과 (Accuracy)

<p align = 'center'>
<img width="912" height="286" alt="image" src="https://github.com/user-attachments/assets/d0695d51-608c-43bf-bb52-a876e242e8ff" />
</p>

* 상태 최신(SOTA) 달성: 선형 양자화 방식 중 LLaMA-3 및 4B 이하 소형 모델에서 가장 우수한 성능을 보여줍니다.
* Reasoning 강점: 특히 긴 토큰 생성이 필요한 MMLU-Pro에서 AWQ 대비 2.4%, QTIP 대비 0.9% 더 높은 정확도를 기록했습니다. 이는 ParoQuant가 긴 생성 과정에서의 오차 누적을 효과적으로 막아준다는 점을 입증합니다.
* 벡터 양자화와 대등: 훨씬 복잡한 알고리즘인 QTIP(벡터 양자화)와 대등한 정확도를 유지하면서도, 구현은 훨씬 단순한 선형 양자화로 해결했습니다.

#### 2. 효율성 결과 (Efficiency)

<p align = 'center'>
<img width="917" height="527" alt="image" src="https://github.com/user-attachments/assets/d4b0ebf1-1695-47f0-b6d0-3aee8fc8c8bd" />
</p>

* 추론 속도: AWQ보다 불과 10% 미만으로 느리면서 정확도는 훨씬 높고, QTIP보다는 15~30% 더 빠른 속도를 보여줍니다.
* 하드웨어 친화성: 제안된 퓨즈드(Fused) 커널 덕분에 채널 차원이 커질수록 Hadamard 변환 방식보다 더 높은 가속 성능을 보입니다.



### 5.3 소거 연구 (Ablation Study)

<p align = 'center'>
<img width="937" height="417" alt="image" src="https://github.com/user-attachments/assets/2f5fcd4f-be86-4533-bb73-08c7c854ad59" />
</p>

* 회전 수의 영향: 독립적 회전(IR) 횟수가 늘어날수록(최대 8회) 성능이 향상되는 것을 확인했습니다.
* 컴포넌트의 기여: 채널별 스케일링(S)과 독립적 회전(IR)을 결합했을 때, 각각을 따로 쓸 때보다 훨씬 뛰어난 정확도를 얻었습니다.
* Stage 2의 중요성: 변환 후 가중치를 미세 조정하는 두 번째 최적화 단계가 RTN을 직접 적용하는 것보다 성능이 더 좋습니다.


---

## 6. Conclusion

### 핵심 성과 요약

* 고효율 PTQ 방법론 제안: ParoQuant는 최소한의 연산 오버헤드만으로도 상태 최신(SOTA) 수준의 양자화 정확도를 달성하는 효율적인 사후 훈련 양자화(PTQ) 기법입니다. 
* Scaled Pairwise Rotation 설계: "희소하게 매개변수화된 회전이 아웃라이어를 충분히 억제할 수 있다"는 통찰을 바탕으로, 하드웨어 친화적인 독립적 기븐스 회전과 채널별 스케일링을 결합한 변환을 설계했습니다. 

### 연구의 의의 및 기여

* 성능과 속도의 공존: 가장 정교한 기존 양자화 방식들과 대등한 정확도를 유지하면서도 실행 속도는 훨씬 빠르며, 이전의 효율적인 양자화 방식들보다 일관되게 우수한 성능을 보여줍니다.
* Reasoning 모델에 최적화: 특히 긴 사고 과정(CoT) 동안 양자화 오차가 누적되는 추론형 LLM에서 뛰어난 성능을 발휘하여 차세대 모델 배포의 길을 열었습니다.
* 미래 연구 방향 제시: 고충실도(High-fidelity)와 저오버헤드(Low-overhead)를 동시에 잡는 양자화 기술이 향후 추론형 LLM 연구에 중요한 영감을 주기를 기대하고 있습니다.

---

