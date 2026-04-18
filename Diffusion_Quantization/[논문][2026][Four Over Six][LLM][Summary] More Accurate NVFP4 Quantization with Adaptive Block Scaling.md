# Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling

저자: Jack Cook, Junxian Guo, Guangxuan Xiao, Yujun Lin, Song Han (MIT & NVIDIA)

최초 제출: 2024년 12월 (arXiv ID의 "2512"로 추정)

논문 : [PDF](https://arxiv.org/pdf/2503.15465)

---

## 0. Summary

<p align = 'center'>
<img width="400" height="500" alt="image" src="https://github.com/user-attachments/assets/3fbebe46-56a9-4093-9db5-366d52426e15" />
</p>


### 1. 문제 NVFP4 Scale에 의한 표현의 공백
* 문제
    * NVFP4는 블록 내 최대값을 6으로 스케일링하는데, FP4의 표현 가능한 값이 {0, 0.5, 1, 1.5, 2, 3, 4, 6}으로 비균일(non-uniform)하게 분포
    * 66.6%~100% 구간의 값을 전혀 표현할 수 없음
    * 이 "표현 공백"이 성능 저하의 주된 원인임을 실험으로 규명했습니다.

### 2. 해결책 

* Adaptive Block Scaling
    * 각 블록을 6으로 스케일링한 버전과 4로 스케일링한 버전, 두 가지로 동시에 양자화한 뒤, MSE(평균제곱오차)가 더 작은 쪽을 선택합니다.
    * 최대값을 4로 제한하면 75%~100% 구간의 값을 표현할 수 있어 근-최대값(near-maximal values)에 대한 오차가 감소.
    * 모든 블록을 4로 고정하면 오히려 성능이 떨어지므로, 블록별로 적응적(adaptive) 으로 선택하는 것이 핵심입니다.

* 구현 효율: NVIDIA Blackwell GPU의 PTX 명령어(cvt 계열)를 활용해 CUDA 커널로 구현, 양자화 연산 대비 오버헤드 15% 미만으로 경량 적용이 가능합니다.

### 3. 효과

* Pre-training: BF16 대비 학습 손실, NVFP4 대비 22.3% 감소 (구체적 수치는 없음)
* PTQ + AWQ: WikiText-2 perplexity 기준 BF16에 19.9% 더 근접
* PTQ + SmoothQuant: BF16에 5.3% 더 근접
    * perplexity 낮을수록 좋은 값
    * BF16: 7.54
    * AWQ: 8.33
    * AWQ + 4/6: 8.24



### 4. 적용 모델

* Pre-training 실험
    * Nemotron 3 Nano 30B-A3B (NVIDIA):
    * 52개 블록(Self-Attention 6, MoE 23, Mamba-2 23), 총 파라미터 300억 / 활성 파라미터 30억의 하이브리드 Mamba-Transformer 구조

* PTQ 실험
    * Llama 3 계열: Llama-3.2-1B, Llama-3.1-8B, Llama-3.1-70B
    * Qwen 3 계열: Qwen3-1.7B, Qwen3-8B, Qwen3-32B


* 비교 대상
    * BF16: 풀 프리시전 베이스라인
    * MXFP4: 블록당 FP8 E8M0 스케일, 블록 크기 32
    * NVFP4 (M=6): 표준 NVFP4, 최대값을 6으로 스케일
    * NVFP4 (M=4): 전체 블록을 4로만 스케일 (비교용)
    * RTN: Round-to-Nearest, 가장 단순한 PTQ
    * GPTQ: 2차 정보 기반 가중치 양자화
    * AWQ: 활성값 인식 가중치 양자화
    * SmoothQuant: 활성값 이상치를 가중치로 이전


* 데이터셋
    * WikiText-2: 언어 모델 perplexity 평가용 표준 데이터셋
    * C4: Common Crawl 기반 대규모 텍스트, perplexity 평가
    * BoolQ: Yes/No 질의응답
    * ARC-Easy / ARC-Challenge: AI2 추론 벤치마크
    * HellaSwag: 문장 완성 상식 추론
    * Pre-training 실험은 1조 토큰 규모의 NVIDIA 내부 합성 데이터 사용

* 평가지표
    * Word Perplexity (↓): WikiText-2, C4 기준 — 낮을수록 좋음
    * Normalized Accuracy (↑): ARC-Easy, ARC-Challenge, HellaSwag — 토크나이저 차이를 줄이기 위해 정규화 적용
    * Accuracy (↑): BoolQ
    * Training Loss: Pre-training에서 BF16 대비 수렴 곡선 비교


---
