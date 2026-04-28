# DartQuant: Efficient Rotational Distribution Calibration for LLM Quantization

저자 : 

Yuantian Shao1,2∗ Yuanteng Chen2,3,4∗ Peisong Wang2,3† Jianlin Yu5 Jing Lin5 Yiwu Yao5 Zhihui Wei1 Jian Cheng2,3

1Nanjing University of Science and Technology,

2 C2DL, Institute of Automation, Chinese Academy of Sciences,

3School of Artificial Intelligence, University of Chinese Academy of Sciences,

4Zhongguancun Academy,

5Huawei Technologies Co., Ltd.

발표 : NeurIPS 2025 (39th Conference on Neural Information Processing Systems). 

Nanjing University of Science and Technology, Chinese Academy of Sciences, Huawei 공동 연구. 

arXiv 공개 2025년 11월.

논문 : [PDF](https://arxiv.org/pdf/2511.04063)


---

## 0. Summary

<p align ='center'>
<img width="1066" height="467" alt="image" src="https://github.com/user-attachments/assets/77e6428e-8fc3-4800-8fe4-610dfc4661cc" />
</p>

### 0.0. 문제

* LLM 양자화에서 rotation matrix(회전 행렬)를 학습할 때 기존 방법들(SpinQuant, OSTQuant)은 end-to-end fine-tuning 방식으로 task loss를 최소화하는데,
    * 이 방식은 (1) 막대한 GPU 메모리·시간 소모,
    * (2) calibration set에 대한 overfitting,
    * (3) orthogonal manifold 위에서의 복잡한 최적화(Cayley/Riemannian SGD)라는 문제가 있음.


### 0.1. 핵심 아이디어


* DartQuant는 이를 세 가지로 해결:
    * Rotational Distribution Calibration:
        * rotation matrix 최적화를 task loss가 아닌 activation 분포를 양자화하기 좋은 형태로 변환하는 문제로 재정의.
        * End-to-end forward/backward를 거치지 않아 메모리·시간 절약, overfitting 위험 감소.
    * Whip Loss:
        * activation이 Laplace 분포(중심에 뾰족한 peak + 양 끝 outlier)를 따른다는 관찰에서 출발.
        * CDF 변환 원리를 응용해 
        * $\text{Whip} = \sum \exp(-|x_i|)$
        * 0 근처에서 gradient가 크기 때문에 작은 값들을 0에서 멀어지게 밀어내고, rotation의 norm 불변성 때문에 norm 합이 보존되어야 하므로 결과적으로 outlier가 자동으로 작아짐.
        * 분포가 Laplace → Uniform에 가깝게 변환됨.
    * QR-Orth Optimization:
        * Cayley SGD처럼 manifold 위 projection 연산(약 6n³ 추가 비용)을 쓰는 대신, 잠재 파라미터 Z를 학습하고 매 step QR 분해로 orthogonal matrix R을 추출.
        * 추가 비용은 4/3 n³ 수준으로 줄어들고, SGD/Adam 등 표준 optimizer를 그대로 사용 가능.

### 0.2. Quantization 설정

* W4A8KV16, W4A4KV16, W4A4KV4 세 가지 주요 설정
    * Weight는 GPTQ로 reconstruct
    * activation은 per-token asymmetric quantization
    * R1, R2는 학습 가능한 rotation으로 weight에 fuse, R3·R4는 online Hadamard (KV cache 및 down-projection 입력용)
    * 추론 프레임워크는 SpinQuant와 동일 → 동일한 추론 가속 효과

### 0.3. 적용 모델

* LLaMA-2 (7B/13B/70B), LLaMA-3 (8B/70B)
* MoE 모델: Mixtral-8x7B, DeepSeek-MoE-16B

### 0.4. 벤치마크

* Perplexity: WikiText2, PTB, C4
* Zero-shot 9개 task: LAMBADA, HellaSwag, PIQA, WinoGrande, OpenBookQA, SIQA, MMLU, ARC-E, ARC-C
* 비교 baseline: RTN, SmoothQuant, GPTQ, OmniQuant, QuaRot, SpinQuant, OSTQuant, QUIK, Atom

### 0.5. 효과 (주요 수치)

* Calibration 비용 대폭 절감 (LLaMA-2-70B 기준):
    * SpinQuant 42.9 GPU시간 / 238.89 GiB → DartQuant 0.91 GPU시간 / 23.47 GiB → 47배 빠르고 10배 적은 메모리.
    * OSTQuant는 메모리 583.86 GiB 필요한 것 대비 25배 절감.
* 단일 RTX 3090에서 70B 모델 rotation calibration을 약 3시간만에 완료
    * (기존 방법들은 3090 단일 GPU로 불가능). 자원 제약 환경에서 LLM 양자화를 가능케 한 첫 사례.
* W4A4KV16 정확도:
    * LLaMA-2-70B에서 FP16 대비 zero-shot 평균 손실 단 0.5%.
    * LLaMA-3-70B에서 SpinQuant 대비 +3.33%p, OSTQuant 대비 +1.45%p 우수.
* W4A4KV4:
    * LLaMA-3-70B PPL 평균 8.13 (FP16 6.19),
    * zero-shot 69.05% (FP16 72.70%) → 기존 SpinQuant 64.76%,
    * OSTQuant 67.84% 대비 명확한 우위.
* QR-Orth 자체 효과:
    * Cayley SGD 대비 100 iteration 시간 8.2h → 5.7h (1.4배 가속),
    * 수렴 속도 차이까지 합치면 약 41배 가속.
* Activation 분포 측면에서도 outlier 개수와 quantization error 모두 가장 낮음 (Hadamard, SpinQuant, OSTQuant 대비).


---
