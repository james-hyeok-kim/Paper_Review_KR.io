# QuantCache: Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation

저자 : Junyi Wu1*, Zhiteng Li1*, Zheng Hui2, Yulun Zhang1†, Linghe Kong1, Xiaokang Yang1

1Shanghai Jiao Tong University, 2MGTV, Shanhai Academy

발표 : 2025년 3월 9일 (arXiv 제출)

논문 : [PDF](https://arxiv.org/pdf/2503.06545)

---

## 0. Summary

<p align ='center'>
<img width="677" height="471" alt="image" src="https://github.com/user-attachments/assets/f60ba313-2b1a-432e-9806-28101404f1fe" />
</p>


### 0.1. 문제 (Problem)

* Diffusion Transformers(DiTs)는 비디오 생성에서 뛰어난 성능을 보이지만 다음과 같은 한계가 있습니다:
    * 막대한 계산/메모리 비용: Open-Sora로 64프레임, 512×512 해상도 비디오를 NVIDIA A800-80GB GPU에서 생성하는 데 약 130초 소요
    * Self-attention의 2차 복잡도가 긴 timestep과 결합되어 비용 증폭

* 기존 가속 기법들의 한계:
    * 양자화(Quantization)와 캐싱(Caching) 기법이 개별적으로만 적용되고 시너지를 활용하지 못함
    * 정적 휴리스틱에 의존 — 모든 layer/timestep에 동일한 bit-width 적용, 사전 정의된 캐싱 스케줄 사용


### 0.2. 핵심 아이디어 (Key Ideas)

* 저자들은 3가지 기법을 통합 최적화하는 training-free 프레임워크 QuantCache를 제안합니다.

#### 1. Hierarchical Latent Caching (HLC) — timestep 간 캐싱

* Inter-step feature divergence score $D_t^{(l)}$ ​를 계산하여 언제 캐시를 갱신할지 동적으로 결정
    * 이전과 현재 비교해서 앞으로 몇개 caching 할건지 동적으로 결정
* Divergence가 작으면 더 오래 캐시 재사용( $tau_{\max}$​ ), 크면 즉시 재계산( $tau_{\min}$ ​)

#### 2. Adaptive Importance-Guided Quantization (AIGQ) — timestep & layer별 양자화

* Weight 양자화 (offline): layer 민감도(수치 오차, 지각적 왜곡, 시간적 동역학)에 따라 bit-width를 차등 할당. Scaling + rotation으로 outlier 완화
* Activation 양자화 (online): timestep redundancy에 따라 bit-width 동적 조정 — 중복이 높은 step은 낮은 bit, 중요한 전환 step은 높은 bit

#### 3. Structural Redundancy-Aware Pruning (SRAP) — layer 간 가지치기

* 같은 timestep 내 인접 layer 간 cosine similarity 측정
    * 현재 측정으로 미래 결정하는 방식
* 유사도가 높으면(중복) 해당 layer 계산을 건너뜀
* 누적 feature 변화량 $V_t$ ​에 따라 pruning 강도를 적응적으로 조절


### 0.3. 효과 (Results)

* W8A8과 W4A6에서만 진행함
* τmax​,τmid​,τmin​의 실제 값 — 예: 5/3/1 step인지, 10/5/2 step인지 전혀 알 수 없음

* Open-Sora 대비 6.72× end-to-end speedup (NVIDIA A800-80GB)
* 경쟁 기법 대비 압도적: T-Gate(1.10×), PAB(1.34×), ViDiT-Q(1.71×), AdaCache-fast(2.24×)

#### Ablation (누적 효과)

|구성|Speedup|
|---|---|
|Baseline|1.00×|
|+ HLC|4.12×|
|+ HLC + AIGQ|6.33×|
|+ HLC + AIGQ + SRAP (Full)|6.72×|

#### 품질 유지



* W8A8 설정에서 baseline Open-Sora 대비 거의 손실 없음
    * (Aesthetic Quality 60.07 → 58.57, Overall Consistency 26.89 → 26.97로 오히려 소폭 개선)
* W4A6의 공격적 저비트 설정에서도 ViDiT-Q(W4A8)를 능가
    * Q-DiT, PTQ4DiT 같은 기법은 W4A8에서 콘텐츠 생성에 실패

#### CUDA 최적화

* 양자화·rotation·캐싱을 통합한 GEMM CUDA 커널 개발
* Scaling factor를 이전 layer에 흡수(offline)하여 추가 오버헤드 제거


### 0.4. 벤치마크 (Benchmarks)

평가는 Open-Sora 1.2 모델 기반, 100 timesteps로 진행되었습니다.

**1. VBench**

* 비디오 생성 모델용 종합 벤치마크. 8개 핵심 차원에서 평가:
    * Motion Smoothness (모션 부드러움)
    * Background Consistency (배경 일관성)
    * Subject Consistency (피사체 일관성)
    * Aesthetic Quality (미적 품질)
    * Imaging Quality (이미지 품질)
    * Dynamic Degree (역동성)
    * Scene Consistency (장면 일관성)
    * Overall Consistency (전체 일관성)

**2. CLIP 기반 지표**

* CLIPSIM: 텍스트–비디오 정렬도
* CLIP-Temp: 시간적 의미 일관성

**3. DOVER**

* 사용자 생성 콘텐츠 비디오 품질 평가:
    * VQA-Aesthetic: 미학적 관점
    * VQA-Technical: 기술적 관점

**비교 대상 (Baselines)**

* 양자화 기법: Q-diffusion, Q-DiT, PTQ4DiT, SmoothQuant, Quarot, ViDiT-Q
* 캐싱 기법: T-Gate, PAB, AdaCache (slow/fast)

