# 6Bit-Diffusion: Inference-Time Mixed-Precision Quantization for Video Diffusion Models

* 저자: Rundong Su, Jintao Zhang, Zhihang Yuan, Haojie Duanmu, Jianfei Chen, Jun Zhu (Fudan / Tsinghua / SJTU)

* 발표일: 2026년 3월 19일, arXiv

* 논문: [PDF](https://arxiv.org/pdf/2603.18742)

---

## 0. Summary

<p align = 'center'>
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/af35cb56-c7b6-49bc-9b26-f8cfd6ac6eb5" />
</p>

### 1. 핵심 아이디어 (3가지 모듈)

* 문제 인식: 기존 정적(static) 혼합 정밀도 양자화는 디노이징 타임스텝마다 레이어의 양자화 민감도가 크게 변한다는 사실을 무시하여 성능이 저하됩니다.

#### 1. DMPQ (Dynamic Mixed-Precision Quantization)

* 블록의 이전 타임스텝 입력-출력 간 상대적 L1 거리(Γ)와 현재 타임스텝의 양자화 오차(Erel) 사이에 강한 선형 상관관계가 있음을 발견했습니다.
* 이를 이용해 각 레이어의 활성값 정밀도를 실시간으로 동적 할당합니다.
    * Γ가 임계값 초과 → 불안정한 레이어 → INT8 사용
    * Γ가 임계값 이하 → 안정적인 레이어 → NVFP4 사용

#### 2.TDC (Temporal Delta Cache)

* 인접 타임스텝 간 Transformer 블록의 잔차(δ)가 매우 유사하다는 점을 활용합니다.
* 이전 두 타임스텝의 δ 변화량으로 현재 타임스텝의 안정성을 예측하여, 안정적인 블록은 계산을 건너뛰고 캐시된 δ를 재사용합니다.
    *  Nmax = 2 (한 블록을 연속으로 최대 2 타임스텝까지만 skip 가능)
* 누적 오차가 임계값을 넘으면 강제로 재계산합니다.

#### 3. PDR (Purified Delta Refresh)

* DMPQ와 TDC를 단순 결합하면 양자화 노이즈가 캐시에 누적되어 영상 품질이 급격히 저하됩니다.
    * skip 횟수 N이 늘어날수록 노이즈가 N배로 선형 증폭
* PDR은 이상치 비율(max/mean)이 높은 레이어는 일시적으로 풀 프리시전(FP16)으로 전환해 δ(Delta) 캐시에 저장되는 값을 정제합니다.
    * 캐시에 저장할 δ를 만들 때, 이상치가 심한 레이어는 아예 양자화를 안 하고 풀 프리시전으로 계산해서 캐시 오염을 원천 차단
    * skip 상태에서 재계산으로 전환될 때 모든 레이어를 INT8로 고정해서 안전하게 처리

### 2. 효과

* 추론 속도 향상: FP16 대비 1.92배 가속
* 메모리 절감: FP16 대비 3.32배 감소
* DMPQ 단독 속도: 1.36배 가속
* CogVideoX-5B W4A6: Aesthetic Quality 0.5724 (ViDiT-Q 0.4433 대비 압도적 우위)
* VQA-A: FP16(74.6) 대비 Full 방법(76.1)이 오히려 초과

### 3. 적용 모델

* CogVideoX-2B: 20억 파라미터 비디오 생성 DiT
* CogVideoX-5B: 50억 파라미터 비디오 생성 DiT
* 두 모델 모두 49프레임 영상 생성, DDIM 스케줄러 50 denoising steps, CFG=6.0으로 실험
* 하드웨어: NVIDIA RTX-5090 단일 GPU


### 4. 비교 대상

* FP16 (Original): 풀 프리시전 베이스라인
* SmoothQuant: 채널별 스케일링으로 활성값 이상치를 가중치로 이전
* QuaRot: 랜덤 Hadamard 회전으로 이상치 제거
* ViDiT-Q: 비디오 DiT용 정적 혼합 정밀도 PTQ (현재 SOTA)
* Uniform W4A4: 전체 균일 4비트 양자화
* Uniform W4A8: 전체 균일 8비트 활성값 양자화


### 5. 데이터셋

* EvalCrafter: 100개 랜덤 샘플 프롬프트 → 캘리브레이션 및 Ablation 평가
* VBench: 비디오 생성 품질 종합 벤치마크 (주요 정량 평가)

---
