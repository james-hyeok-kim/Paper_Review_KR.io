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


### 0.1. 핵심 아이디어 (Core Idea)

* NVFP4 포맷으로 양자화된 대형 언어 모델(LLM)과 비전-언어 모델(VLM)의 추론 정확도 손실을 복구하기 위해 양자화 인식 증류(Quantization-Aware Distillation, QAD) 방식을 제안합니다.  

* 기존의 양자화 인식 훈련(QAT)은 다음 토큰 예측과 같은 작업별 손실(예: 교차 엔트로피 손실)을 사용하여 훈련을 진행합니다.
* 반면, 논문에서 제안한 QAD는 KL 발산 손실(KL divergence loss)을 사용하여 전체 정밀도(Full-precision)를 가진 교사 모델(Teacher)의 출력 분포를 양자화된 학생 모델(Student)로 지식 증류(Distillation)하는 것이 핵심입니다.  3. 효과 및 장점 (Effects)복잡한 훈련 파이프라인에서의 안정성: SFT(지도 미세 조정), RL(강화 학습), 모델 병합 등 다단계 사후 훈련(Post-training)을 거친 모델들에서 QAT가 겪는 훈련 불안정성 및 복잡성 문제를 해결합니다. 특히 RL 훈련 모델에 QAT를 적용하면 학습된 성능이 무너지는 문제가 발생하지만, QAD는 성능 저하 없이 정확도를 성공적으로 복구합니다.  불완전한 데이터에 대한 강건성(Robustness): 원본 훈련 데이터 전체가 없거나, 수학/코드 등 특정 도메인의 데이터만 부분적으로 존재하는 상황에서도 원래의 정확도를 성공적으로 복구해 내는 교차 도메인 지식 전이 능력을 보여줍니다.  원본 모델과의 높은 정렬도: 목표 데이터세트에 대해 다시 학습하는 QAT와 달리, QAD는 원본 BF16 모델의 출력 분포를 거의 완벽하게 보존하여 정렬(Align) 상태를 유지합니다.  4. 벤치마크 및 결과 (Benchmarks)평가 모델: AceReason Nemotron, Nemotron 3 Nano, Nemotron Nano V2, Nemotron Nano V2 VL, Llama Nemotron Super v1 등 NVIDIA의 다양한 최신 훈련 모델들이 활용되었습니다.  활용 벤치마크: MATH500, AIME24, AIME25, GPQA Diamond(GPQA-D), IFEval, LiveCodeBench, SciCode 등의 수학 및 코딩/추론 벤치마크가 사용되었습니다. (VLM 모델의 경우 AI2D, ChartQA, DocVQA, TextVQA 등이 활용되었습니다).  결과 요약: QAD는 평가된 모든 벤치마크에서 원본 모델인 BF16에 근접하는 정확도를 일관되게 회복했습니다. 특히 AIME25 및 GPQA-D와 같은 고난도 추론 테스트에서 기존 QAT 방식보다 훨씬 뛰어난 성능 향상을 입증했습니다.  

---


