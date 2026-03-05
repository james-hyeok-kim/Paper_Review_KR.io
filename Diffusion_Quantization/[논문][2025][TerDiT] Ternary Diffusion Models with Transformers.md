# TerDiT: Ternary Diffusion Models with Transformers

저자 : Xudong Lu, Aojun Zhou, Ziyi Lin, Qi Liu, Yuhui Xu, Renrui Zhang, Xue Yang, Member, IEEE,

Junchi Yan, Senior Member, IEEE, Peng Gao, Hongsheng Li, Member, IEEE

발표 : 2025년 4월 6일에 arXiv

논문 : [PDF](https://arxiv.org/pdf/2405.14854)

---


## 0. Summary

### QAT

<div align="center">
   
| 레이어 | 가중치 (Weight) | 활성화값 (Activation) | 비고 |
| :--- | :---: | :---: | :--- |
| **Self-attention, Feedforward, <br>MLP Linear** | (Ternary, <br>1.58-bit) | Full-precision (FP) | 가중치 전용(Weight-only) 양자화 방식을 사용합니다. |
| **adaLN 모듈 내 MLP 레이어** | (Ternary, <br>1.58-bit) | Full-precision (FP) | 삼진 양자화로 인한 불안정성을 해결하기 위해 레이어 뒤에 RMS Norm을 추가합니다. |

</div>
---

## 1. Introduction

**TerDiT의 핵심 제안**

* 최초의 DiT 전용 QAT: DiT 모델을 위한 삼진 양자화(Weight-only)를 최초로 시도하였으며, 600M에서 4.2B 규모의 모델까지 확장 가능함을 보여주었습니다.
* 구조적 개선 (adaLN-RMSNorm): DiT 블록 내 adaLN 모듈 뒤에 RMS Norm을 추가하는 구조적 수정을 제안했습니다.
* 효율적인 배포: 2비트 CUDA 커널을 통해 배포함으로써, 체크포인트 크기를 10배 이상, 추론 메모리 소비를 약 8배 줄이면서도 정밀 모델에 필적하는 이미지 생성 품질을 유지했습니다.



---

## 2. Related Works

<p align  = 'center'>
<img width="962" height="467" alt="image" src="https://github.com/user-attachments/assets/45774145-b356-4e62-a959-a728bd1f7a1d" />
</p>

### C. (Ternary Weight Networks)

* 양자화 방식: 가중치만 양자화하는 방식(Weight-only)과 가중치 및 활성화 함수를 모두 양자화하는 방식(Weight-activation)이 있습니다.
* 최신 동향: 최근 대규모 언어 모델(LLM)에 Ternary Weight를 적용하여 정밀 모델과 대등한 성능을 거둔 사례가 보고되었습니다.
* TerDiT의 위치: 이러한 LLM 연구에서 영감을 받아, DiT 모델에 특화된 QAT 및 효율적 배포 스키마를 처음으로 제안한 것이 본 논문의 핵심입니다. 


---

## 3. TerDiT

### A. 확산 트랜스포머 모델 예비 지식 (Preliminary)

* 구조: DiT는 기존 U-Net 백본을 트랜스포머로 대체하여 잠재 패치(Latent patches) 상에서 작동합니다.
* 조건 삽입: 노이즈 타임스텝이나 클래스 레이블 같은 조건부 정보를 주입하기 위해 adaLN-Zero(Adaptive Layer Normalization) 모듈을 사용합니다.
* 구성: 각 트랜스포머 블록 내의 adaLN 모듈은 MLP 레이어를 포함하며, 이는 전체 모델 파라미터의 약 10%~20%를 차지합니다.

### B. 모델 양자화 (Model Quantization)

* 양자화 함수: BitNet b1.58과 유사한 absmean 양자화 함수를 채택하여 가중치를 $\{-1, 0, +1\}$ 중 하나의 값으로 변환합니다.
* 훈련 방식: STE(Straight-Through Estimator)를 사용하여 미분 불가능한 성분을 통과해 그레이디언트를 전파하며, 훈련 중에는 풀 프리시전(Full-precision) 파라미터를 유지합니다.
    * Forward ternary, Backward pass full precision
    * 훈련 중에는 가중치가 아주 조금씩 변함, 변화량이 반영되지 않아 학습 불가능
* 학습 설정: 저비트 QAT의 빠른 수렴을 위해 기존 DiT보다 큰 초기 학습률( $5 \times 10^{-4}$ )을 사용합니다.

### C. QAT 전용 모델 구조 개선

<p align  = 'center'>
<img width="477" height="364" alt="image" src="https://github.com/user-attachments/assets/79f2e256-a192-4ed4-bd90-276d98f30ef7" />
</p>

* 문제점: Ternary 선형 레이어는 풀 프리시전 레이어에 비해 매우 큰 활성화 값(Activation values)을 생성하여 훈련 불안정성을 초래합니다.
* 해결책 (RMS Normalized adaLN): adaLN 모듈의 MLP 레이어 뒤에 RMS Norm을 추가하여 활성화 값을 적절한 범위로 스케일링합니다.
* 효과: 이 작은 수정을 통해 더 빠른 수렴 속도와 낮은 훈련 손실을 달성할 수 있습니다.

### D. 배포 스키마 (Deployment Scheme)

* 압축: Ternary Weight 4개를 하나의 INT8 값으로 패킹하여 저장합니다.
* 효율성: 2비트 CUDA 커널을 통해 배포함으로써 4.2B 모델의 체크포인트 크기를 16GB에서 1.1GB로, 추론 메모리 사용량을 17GB에서 2GB 미만으로 줄였습니다.

---

## 4. Experiments

<p align  = 'center'>
<img width="874" height="595" alt="image" src="https://github.com/user-attachments/assets/a0c8020b-ae25-4768-be8d-ab397d8cd324" />
</p>

### A. 풀 프리시전 모델과의 비교

* 이미지 품질: $256\times256$ 해상도에서 TerDiT-4.2B-G는 FID 2.42를 기록하며, 풀 프리시전 모델인 Large-DiT-4.2B-G(FID 2.10)와 근소한 차이만을 보였습니다.
* 고해상도 성능: $512\times512$ 해상도에서 TerDiT-4.2B-G는 FID 2.81을 달성하여, 훨씬 더 많은 파라미터를 가진 DiT-XL/2-G(FID 3.04)보다 우수한 성능을 나타냈습니다.
* 시각적 품질: 정성적 분석 결과, TerDiT가 생성한 이미지는 풀 프리시전 모델과 시각적으로 큰 차이가 없는 고화질 결과물을 생성했습니다.

### B. 배포 효율성 (Deployment Efficiency)

<p align  = 'center'>
<img width="883" height="237" alt="image" src="https://github.com/user-attachments/assets/31b886d1-55e9-4a35-932c-76bfcb5e7627" />
</p>

<div align="center">
   
| 지표 | Large-DiT-4.2B (FP) | TerDiT-4.2B (Ternary) | 절감 효과 |
| :--- | :--- | :--- | :--- |
| 체크포인트 크기 | 16GB  | 1.1GB  | 약 14.5배 감소 |
| 최대 메모리 할당 | 17,027MB  | 1,919MB  | 약 8.8배 감소 |

</div>

* 메모리 이점: 4.2B 규모의 거대 모델을 단 2GB 미만의 GPU 메모리로 구동할 수 있어 모바일이나 FPGA 등 저사양 기기 배포 가능성을 확인했습니다.
* 추론 속도: 현재는 전용 하드웨어 가속기 부족으로 인해 언패킹(Unpacking) 과정에서 FP 모델보다 느리지만(97s vs 83s), 향후 하드웨어-소프트웨어 공동 설계를 통해 개선될 여지가 큽니다.

### C. 주요 제거 실험 (Ablation Studies)

* RMS Norm의 효과: adaLN 뒤에 RMS Norm을 추가했을 때 훈련 수렴 속도가 훨씬 빨라졌으며, 최종 FID 점수도 현저히 낮아졌습니다.
* 활성화 값 분석: RMS Norm을 적용한 모델은 삼진 가중치로 인해 발생하는 거대한 활성화 값을 풀 프리시전 모델과 유사한 수준으로 억제하여 훈련 안정성을 높였습니다.
* 학습률 감소(LR Reduction): 훈련 후반부에 학습률을 낮추는 것이 더 세밀한 파라미터 업데이트를 가능하게 하여 성능 향상에 기여했습니다.

<p align  = 'center'>
<img width="963" height="643" alt="image" src="https://github.com/user-attachments/assets/7deccaa6-ec65-45e6-9b0b-a733e09caac8" />
</p>

* Figure 7: Training Loss 비교 (학습 수렴 속도)
    * 손실 함수(Loss)의 변화
    * 600M 및 4.2B 모델 공통 사항: RMS Norm을 추가한 모델(빨간색 선)이 추가하지 않은 베이스라인(파란색 선)보다 훨씬 빠르게 수렴하며, 최종적으로 더 낮은 손실 값을 달성합니다.
    * 4.2B 모델의 특징: 모델 파라미터가 클수록(4.2B) 작을 때(600M)보다 학습이 더 빠르고 안정적으로 진행되는 경향을 보입니다. 이는 Ternary DiT 모델에도 Scaling Law(규모의 법칙)가 적용됨을 시사합니다.

* Figure 8: FID-50k Score 비교 (이미지 생성 품질)
    * 학습 진행도에 따른 FID 점수의 변화를 측정하여 실제 생성된 이미지의 품질을 평가합니다. FID 점수는 낮을수록 실제 이미지 분포와 유사함을 의미합니다.
    * 생성 품질 향상: 100k 스텝부터 400k 스텝까지 모든 구간에서 RMS Norm을 적용한 모델이 현저히 낮은 FID 점수를 기록합니다.
    * 4.2B 모델의 비약적 발전: 특히 4.2B 모델의 경우, RMS Norm 적용 시 200k 스텝 부근에서 FID가 급격히 하락하며 품질이 고도화되는 것을 확인할 수 있습니다. 이는 구조적 개선이 대규모 모델의 양자화 오차 적응에 결정적인 역할을 함을 증명합니다.

### D. 양자화 베이스라인 비교

* PTQ 실패: Q-DiT나 Q-Diffusion 같은 기존의 사후 양자화(PTQ) 방식은 2비트 설정에서 정상적인 이미지를 생성하지 못하고 노이즈만 출력했습니다.
* QAT 비교: BitNet b1.58 구조를 DiT에 그대로 이식했을 때보다 TerDiT의 방식이 더 빠른 훈련 속도와 낮은 FID(4.34 vs 6.60)를 기록했습니다.

---

## 5. Discussion and Future Works

### 주요 논의 사항 (Discussions)

* 훈련 효율성 및 환경적 영향: Ternary DiT 모델을 훈련하는 것은 풀 프리시전 네트워크에 비해 훈련 과정이 덜 안정적이고 더 많은 시간이 소요됩니다. 훈련 속도를 높이기 위해 더 효율적인 DiT 구조나 하드웨어-소프트웨어 공동 개발을 탐구할 필요가 있습니다.
* 양자화 범위의 설정: 본 논문에서는 INT8이나 FP16 양자화와의 직접적인 성능 비교를 수행하지 않았습니다. 이는 INT8/FP16이 '극저비트(Extremely low-bit)' 범주에 속하지 않기 때문입니다.
* 잠재력 비교: FP16이나 INT8은 메모리 사용량을 최대 75%까지 줄여주는 반면, 삼진 양자화는 이론적으로 메모리 사용량을 최대 16배까지 줄일 수 있어 훨씬 더 큰 잠재력을 가지고 있습니다.
* 하드웨어 지원 현황: 현재 Ternary LLM을 CPU에 배포하거나 Ternary/Binary CNN을 FPGA에서 가속하는 연구는 이미 구현되어 있습니다.

### 향후 연구 방향 (Future Works)

* DiT 구조 탐구: 훈련 안정성과 속도를 높이기 위해 다양한 변형 DiT 구조를 연구할 계획입니다.
* 하드웨어 가속 구현: DiT 모델의 실제 하드웨어 가속 구현을 지속적으로 탐구하여, 현재 추론 속도가 풀 프리시전보다 느린 한계를 극복하고자 합니다.
* 배포 생태계 구축: 학계와 공학계의 공동 노력을 통해 삼진 DiT를 위한 효과적인 오픈소스 배포 솔루션을 마련하는 것을 목표로 합니다.

---


