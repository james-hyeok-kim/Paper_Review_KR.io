# QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models

저자 : 

Jingxuan Zhang2*, Yunta Hsieh3*, Zhongwei Wan1, Haokun Lin4,

Xin Wang1, Ziqi Wang1, Yingtie Lei1, Mi Zhang1†

1The Ohio State University, 2

Indiana University, 3University of Michigan, 4City University of Hong Kong

[Git](https://quantvla.github.io/)

[Hugging Face](https://huggingface.co/papers/2602.20309)


발표 : 2026년 2월 27일 arXiv

논문 : [PDF](https://arxiv.org/pdf/2602.20309)

---

## 0. Summary

* 세계 최초의 VLA 시스템 전용 PTQ 방식이며, 특히 민감한 Diffusion Transformer(DiT) 액션 헤드를 성공적으로 양자화한 첫 사례

### 핵심 기술 구성 (Scale-Calibrated Components)

* 선택적 양자화 레이아웃 (Selective Quantization Layout)
    * 언어 백본(LLM)의 모든 선형 레이어와 DiT의 MLP 레이어는 정수형(Integer)으로 양자화합니다.
    * 가장 민감한 Attention Projection(Q, K, V, O)은 부동 소수점(Floating Point) 상태로 유지하여 오차 누적을 방지합니다.
* 어텐션 온도 매칭 (Attention Temperature Matching, ATM)
    * 양자화로 인해 어텐션 로그(logits)의 온도가 변하는 현상을 방지하기 위해 헤드별 스케일링 메커니즘을 적용하여 안정화합니다.
* 출력 헤드 밸런싱 (Output Head Balancing, OHB)
    * 레이어별 잔차 인터페이스(residual interface)를 보정하여 양자화 후 발생하는 에너지 드리프트 현상을 완화합니다.
  
### 주요 성능 및 결과

* 메모리 절감: 양자화된 구성 요소에서 약 70%의 상대적 메모리 절감을 달성했습니다.
* 작업 성공률 유지 및 향상: LIBERO 시뮬레이터 테스트 결과, $\pi0.5$ 모델에서 평균 성공률 97.6%를 기록하며 풀 프리시전(Full-precision) 베이스라인과 대등하거나 오히려 상회하는 성능을 보였습니다.
* 효율성: 추가 학습이 필요 없으며, 라벨이 없는 소량의 보정 데이터(Calibration buffer)만 사용하여 실제 배포에 매우 실용적입니다.


