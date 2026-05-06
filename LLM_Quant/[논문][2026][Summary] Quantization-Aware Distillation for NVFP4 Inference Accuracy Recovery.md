# Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery

저자 : NVIDA

Meng Xin, Sweta Priyadarshi, Jingyu Xin, Bilal Kartal, Aditya Vavre, Asma Kuriparambil

Thekkumpate, Zijia Chen, Ameya Sunil Mahabaleshwarkar, Ido Shahaf, Akhiad Bercovich, Kinjal

Patel, Suguna Varshini Velury, Chenjie Luo, Zhiyu Cheng, Jenny Chen, Chen-Han Yu, Wei Ping, Oleg

Rybakov, Nima Tajbakhsh, Oluwatobi Olabiyi, Dusan Stosic, Di Wu, Song Han, Eric Chung, Sharath

Turuvekere Sreenivas, Bryan Catanzaro, Yoshi Suhara, Tijmen Blankevoort, Huizi Mao1

발표 : 2026년 3월

출처/소속: 특정 학회가 아닌 NVIDIA 소속 연구진들이 발표한 기술 보고서(Technical Report)입니다

논문 : [PDF](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)

---

## 0. Summary

<p align = 'center'>
<img width="768" height="344" alt="image" src="https://github.com/user-attachments/assets/c7d51fe3-5b8c-405b-b407-d4db9daec806" />
</p>

### 0.1. 핵심 아이디어 (Core Idea)

* NVFP4 포맷으로 양자화된 대형 언어 모델(LLM)과 비전-언어 모델(VLM)의 추론 정확도 손실을 복구하기 위해 양자화 인식 증류(Quantization-Aware Distillation, QAD) 방식을 제안합니다.  
* 기존의 양자화 인식 훈련(QAT)은 다음 토큰 예측과 같은 작업별 손실(예: 교차 엔트로피 손실)을 사용하여 훈련을 진행합니다.
* 반면, 논문에서 제안한 QAD는 KL 발산 손실(KL divergence loss)을 사용하여 전체 정밀도(Full-precision)를 가진 교사 모델(Teacher)의 출력 분포를 양자화된 학생 모델(Student)로 지식 증류(Distillation)하는 것이 핵심입니다.

#### 1. 교사-학생 학습 구조 (Teacher-Student Framework)

* QAD는 기본적으로 지식 증류(Knowledge Distillation) 기법을 활용합니다.
* 교사 모델(Teacher): 원본의 고정밀 BF16 모델을 사용합니다.
* 학생 모델(Student): NVFP4로 양자화된 모델을 사용하며, 교사 모델의 지식을 학습하여 정확도를 복구합니다.
* 이 과정에서 학생 모델은 교사 모델이 출력하는 "Soft Labels"(확률 분포)를 그대로 따라 하도록 훈련됩니다.

#### 2. 손실 함수: KL 발산 (KL Divergence)

* QAD의 핵심은 정답 토큰(Ground Truth)을 맞추는 것이 아니라, 교사 모델의 출력 분포를 얼마나 정확하게 모사하느냐에 있습니다.
* 손실 함수 공식

$$\mathcal{L}_{QAD}=D_{KL}(p_{teacher}||p_{student})=\sum_{y\in V}p_{teacher}(y|x)log\frac{p_{teacher}(y|x)}{p_{student}(y|x)}$$  

* QAT와의 차이: QAT는 다음 토큰 예측(Next-token prediction)을 위해 교차 엔트로피(Cross-entropy)를 사용하지만, QAD는 교사 모델과 학생 모델 사이의 확률 분포 차이인 KL 발산을 최소화하는 방식을 택합니다.
* 이를 통해 모델이 원래 가지고 있던 지식의 분포를 깨뜨리지 않고 양자화로 인한 오차만 보정할 수 있습니다.
* QAT의 손실 함수: 교차 엔트로피 (Cross-Entropy)QAT는 교사 모델이 아닌 데이터셋의 정답 레이블( $q$ )을 기준으로 학습을 진행합니다.

$$\mathcal{L}_{QAT} = -\sum_{y \in V} q(y|x) \log(p_{student}(y|x))$$

### 0.2. 효과 및 장점 (Effects)

<p align = 'center'>
<img width="398" height="241" alt="image" src="https://github.com/user-attachments/assets/58211ee9-e762-48c5-9ef9-045f12f55928" />
</p>

* 복잡한 훈련 파이프라인에서의 안정성:
    * SFT(지도 미세 조정), RL(강화 학습), 모델 병합 등 다단계 사후 훈련(Post-training)을 거친 모델들에서 QAT가 겪는 훈련 불안정성 및 복잡성 문제를 해결합니다.
    * 특히 RL 훈련 모델에 QAT를 적용하면 학습된 성능이 무너지는 문제가 발생하지만, QAD는 성능 저하 없이 정확도를 성공적으로 복구합니다.
* 불완전한 데이터에 대한 강건성(Robustness): 원본 훈련 데이터 전체가 없거나, 수학/코드 등 특정 도메인의 데이터만 부분적으로 존재하는 상황에서도 원래의 정확도를 성공적으로 복구해 내는 교차 도메인 지식 전이 능력을 보여줍니다.
* 원본 모델과의 높은 정렬도: 목표 데이터세트에 대해 다시 학습하는 QAT와 달리, QAD는 원본 BF16 모델의 출력 분포를 거의 완벽하게 보존하여 정렬(Align) 상태를 유지합니다.  

<p align = 'center'>
<img width="774" height="307" alt="image" src="https://github.com/user-attachments/assets/1ea2f54b-693d-4f1a-8305-f308e3271e29" />
  <img width="775" height="406" alt="image" src="https://github.com/user-attachments/assets/cc6ecb78-63b3-415b-a09d-9ba7b843a696" />

</p>

### 0.3. 벤치마크 및 결과 (Benchmarks)

* 평가 모델: AceReason Nemotron, Nemotron 3 Nano, Nemotron Nano V2, Nemotron Nano V2 VL, Llama Nemotron Super v1 등 NVIDIA의 다양한 최신 훈련 모델들이 활용되었습니다.
* 활용 벤치마크: MATH500, AIME24, AIME25, GPQA Diamond(GPQA-D), IFEval, LiveCodeBench, SciCode 등의 수학 및 코딩/추론 벤치마크가 사용되었습니다. (VLM 모델의 경우 AI2D, ChartQA, DocVQA, TextVQA 등이 활용되었습니다).  
* 결과 요약: QAD는 평가된 모든 벤치마크에서 원본 모델인 BF16에 근접하는 정확도를 일관되게 회복했습니다. 특히 AIME25 및 GPQA-D와 같은 고난도 추론 테스트에서 기존 QAT 방식보다 훨씬 뛰어난 성능 향상을 입증했습니다.  


---


