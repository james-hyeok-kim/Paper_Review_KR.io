# QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

저자 : 

Saleh Ashkboos, ETH Zurich, saleh.ashkboos@inf.ethz.ch

Amirkeivan Mohtashami, EPFL, amirkeivan.mohtashami@epfl.ch

Maximilian L. Croci, Microsoft Research, mcroci@microsoft.com

Bo Li, ETH Zurich, bolibo@ethz.ch

Pashmina Cameron, Microsoft, pcameron@microsoft.com

Martin Jaggi, EPFL, martin.jaggi@epfl.ch

Dan Alistarh, IST Austria & NeuralMagic, dan.alistarh@ist.ac.at

Torsten Hoefler, ETH Zurich, torsten.hoefler@inf.ethz.ch

James Hensman, Microsoft Research, jameshensman@microsoft.com

발표 : 2024년 10월 29일에 arXiv

논문 : [PDF](https://arxiv.org/pdf/2404.00456)

---


## 0. Summary

<p align = 'center'>
<img width="750" height="400" alt="image" src="https://github.com/user-attachments/assets/eedc7f1f-4da9-49c4-a52c-351dd82ac6b2" />
</p>

### 0.1. 문제 정의

* 대규모 언어 모델(LLM)의 추론에는 막대한 연산, 메모리, 에너지가 필요하며, 특히 프리필(prefill) 단계에서 많은 자원이 소모. 
* 양자화(Quantization) 기술이 중요하지만, 활성화(Activations) 값을 4비트로 양자화하는 것은 매우 어렵습니다.  
* 활성화 값 내에 정상 범위를 크게 벗어나는 거대한 '이상치(Outlier)' 값들이 존재하기 때문입니다.  
* 기존 연구들은 이러한 이상치 피처들을 식별하여 추론 시 16비트 같은 고정밀도(higher precision)로 유지해야 했기 때문에 하드웨어 구현이 복잡해지는 한계가 있었습니다.  

### 0.2. 핵심 아이디어 (QuaRot)

* QuaRot은 회전(Rotations)을 기반으로 하는 새로운 양자화 기법입니다.
* '계산 불변성(computational invariance)'이라는 개념을 활용하여 모델의 가중치 행렬에 무작위 아다마르 변환(Randomized Hadamard transformations)을 적용해 모델을 회전시킵니다.
* 이러한 회전 처리는 모델의 최종 출력값을 전혀 바꾸지 않으면서도 활성화 값(은닉 상태)에 존재하는 이상치(Outliers)를 완전히 제거하여 값을 고르게 퍼뜨립니다.
* 이상치가 사라지기 때문에, 특정 채널을 높은 정밀도로 남겨둘 필요 없이 가중치, 활성화 값, KV 캐시 등 모든 요소를 4비트로 쉽게 양자화할 수 있게 됩니다.

### 0.3. 양자화 세부 정보 (Precision, Weights, Activations, Block size 등)

* 정밀도 (Quantization Precision): 예외로 두는 채널 없이, 모델의 가중치, 활성화, KV 캐시를 모두 End-to-End로 4비트 정수(INT4) 정밀도로 양자화합니다.
    * 가중치 (Weights):
        * GPTQ(기본값) 또는 RTN(Round-to-Nearest) 방식을 사용하여 양자화합니다.
        * 컬럼 단위의 대칭 양자화(per-column symmetric quantization)를 적용합니다.
    * 활성화 (Activations):
        * 추론 과정에서 실시간으로 대칭 양자화를 수행하며, 각 토큰(행)마다 단일 스케일을 사용하는 토큰 단위 대칭 양자화(per-token symmetric quantization)를 적용하고 단순 반올림(RTN)을 사용합니다.
    * KV 캐시 (KV Cache): 어텐션 모듈의 Key와 Value 벡터에도 아다마르 회전을 적용하여 이상치를 제거합니다.
        * KV 캐시는 비대칭 양자화(asymmetric quantization)를 사용하며 그룹 크기(Group size / Block size)는 128로 설정합니다.

### 0.4. 효과 및 벤치마크 (효율성 및 성능)

* 효율성 (속도 및 메모리):
    * LLaMA2-70B 모델 적용 시 프리필(prefill) 단계에서 최대 3.33배의 속도 향상을 기록했습니다 (배치 사이즈 64 기준).
    * 디코딩 단계에서는 기존 FP16 대비 최대 3.89배의 메모리를 절약할 수 있습니다.
* 언어 생성 성능:
    * LLaMA2-70B 모델을 4비트로 양자화했을 때 WikiText-2 펄플렉서티(Perplexity) 손실이 최대 0.47에 불과했습니다.
    * 이는 SmoothQuant, OmniQuant, Atom 등 기존 기법을 뛰어넘는 성능입니다.
* 제로샷 벤치마크 (Zero-Shot Tasks):
    * PIQA, WinoGrande, HellaSwag, ARC, LAMBADA 등의 제로샷 평가에서 원래 모델(FP16) 정확도의 99%를 유지했습니다.
    * LLaMA2-70B의 경우 평균 정확도 손실이 1.09%에 불과했습니다.
* 추가 성과:
    * 6비트 및 8비트 모델 생성 시에는 캘리브레이션 데이터 없이 단순 RTN 방식만 적용해도 정확도 손실이 전혀 발생하지 않음(lossless)을 증명했습니다.
    * LLaMA-2 뿐만 아니라 LLaMA-3, Phi-3-mini 모델에서도 우수한 양자화 결과를 보여주었습니다.  

---
