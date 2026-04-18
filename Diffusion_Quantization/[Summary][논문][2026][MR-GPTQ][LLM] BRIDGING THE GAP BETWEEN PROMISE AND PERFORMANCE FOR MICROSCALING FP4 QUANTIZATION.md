# BRIDGING THE GAP BETWEEN PROMISE AND PERFORMANCE FOR MICROSCALING FP4 QUANTIZATION

저자: Vage Egiazarian, Andrei Panferov, Denis Kuznedelev 등 (IST Austria, Yandex, Red Hat AI, ETH Zürich)

발표: ICLR 2026 conference paper (2026년 3월 3일 최종 버전)

논문 : [PDF](https://arxiv.org/pdf/2509.23202)

---

## 0. Summary

### 1. 문제 발견 (2가지)

1) NVFP4는 그룹 크기(16)가 작아 기존 이상치 완화 기법(SmoothQuant, QuaRot 등)의 효과가 이론적으로 무력화.
    1) 블록 내 최대값이 이미 absmax 스케일로 보존되므로 추가적인 이상치 처리가 불필요합니다.

2) MXFP4는 스케일을 E8M0(2의 거듭제곱)으로만 표현하기 때문에 스케일 양자화 오차가 커서 정확도가 약 10% 상대적으로 떨어집니다.

### 2. 해결책 — MR-GPTQ (Micro-Rotated GPTQ)

1) MSE 최적화 그리드: 텐서 전체 스케일과 그룹별 스케일을 교대 최적화로, 초기 양자화 오차를 최소화합니다.
2) Static Activation Reordering:
    1) 기존 GPTQ의 "동적 act-order"는 런타임에 행렬 열을 재배열해 10~20% 속도 저하를 유발합니다.
    2) MR-GPTQ는 양자화 전에 열 순서를 미리 고정(static)해 정확도 향상은 유지하면서 런타임 오버헤드를 제거합니다.
3) Fused Online Rotation:
    1) 블록 단위 Hadamard 변환을 가중치에는 오프라인으로 융합하고, 활성값에는 온라인으로 적용합니다.
    2) MXFP4에 특히 효과적이며 NVFP4에는 선택적으로 적용합니다.

* QuTLASS GPU 커널
    * NVIDIA Blackwell GPU 전용 고성능 커널 라이브러리로, 온라인 회전 + 양자화를 단일 커널에 융합해 오버헤드를 최소화합니다.
    * vLLM에 통합되어 실제 추론에 사용 가능합니다.


### 3. 주요 실험 모델

* Llama 3 계열: Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct
* Qwen 3 계열: Qwen3-8B, Qwen3-14B, Qwen3-32B

* 하드웨어
    * NVIDIA B200 (SM100)
    * NVIDIA RTX5090 (SM120)
* 추론 프레임워크
    * vLLM (QuTLASS 커널 통합)
    * Transformers 라이브러리

### 4. 비교대상

* RTN: Round-to-Nearest, 가장 단순한 베이스라인
* RTN + HT: Hadamard Transform 추가한 RTN
* GPTQ: 2차 정보 기반 가중치 양자화
* SmoothQuant: 채널별 스케일로 활성값 이상치를 가중치로 이전
* QuaRot: Hadamard 회전으로 이상치 제거
* SpinQuant: 학습 가능한 회전 행렬 사용
* AWQ: 활성값 인식 가중치 양자화
* QAT: Quantization-Aware Training
* NVINT4 / MXINT4: 가상의 INT4 기반 Microscaling 포맷 (비교 분석용)


### 5. 데이터셋 & 평가지표

* 캘리브레이션 데이터셋
    * FineWeb 1024개 시퀀스 (GPTQ 캘리브레이션용)
    * Tülu 3 92,995개 샘플의 10% (QAT용)

* MMLU / MMLU-CoT: 세계 지식 및 추론
* GSM8K: 수학 문제 풀이
* HellaSwag: 문장 완성 상식 추론
* WinoGrande: 언어 이해
* PlatinumBench: 더 정밀한 노이즈 감소 벤치마크 (BBH, HotpotQA, SQuAD, DROP 등 포함)


---

