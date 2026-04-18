# HQ-DiT: Efficient Diffusion Transformer with FP4 Hybrid Quantization

저자 : Wenxuan Liu, Sai Qian Zhang (New York University) 

발표 : 2024년 5월

논문 : [PDF](https://arxiv.org/pdf/2405.19751)

---

## 0. Summary

<p align = 'center'>
<img width="350" height="350" alt="image" src="https://github.com/user-attachments/assets/c13836ab-8e66-4115-b3aa-c572e5fd856c" />
</p>

### 1. Hadamard Transform으로 이상치(Outlier) 제거

* 입력 활성값에 랜덤 Hadamard 행렬을 곱해 분포를 고르게 만들고, 이에 대응하는 변환을 가중치 행렬에도 미리 적용해 둡니다.
* 이렇게 하면 수학적으로 동일한 결과를 유지하면서 이상치가 사라지고 양자화 오차가 크게 줄어듭니다.

### 2. 채널별 데이터 분포 기반 FP 포맷 자동 선택

* E3M0, E2M1, E1M2 등 여러 형태.
* HQ-DiT는 가중치의 최대값/최솟값 비율(sw)을 계산해 각 레이어에 맞는 최적 포맷.

### 3. 효과 (실험 결과)

* 메모리 절감: FP32 대비 2.13배 감소
* 속도 향상: FP32 대비 5.09배 빠름
* W4A4 이미지 품질
    * sFID 기준 FP32 대비 +0.12 수준
    * W4A4에서 FPQ의 FID 231 vs HQ-DiT 23.91
* W4A4(가중치·활성값 모두 4비트) 조건에서 기존 방법들은 사람이 알아볼 수 없는 노이즈 이미지를 생성
* HQ-DiT는 풀 프리시전(FP32) 수준에 가까운 이미지를 생성
* 심지어 IS·FID 기준으로 기존 풀 프리시전 LDM 모델을 능가하기도 했습니다.

### 4. 적용 모델

* 기본 모델: DiT-XL/2
    * DiT-XL/2 256×256 (denoising steps = 100)
    * DiT-XL/2 512×512 (denoising steps = 20)
    * 비교 대상으로 LDM-4 (Latent Diffusion Model, U-Net 기반 풀 프리시전 모델)도 함께 언급

* 비교 대상 (Baseline) 방법들
    * MinMax (FP): 채널별 FP 양자화, 가장 단순한 베이스라인
    * SmoothQuant: LLM용 PTQ, 활성값 난이도를 가중치로 이전
    * FPQ: 모든 FP 포맷 조합을 탐색해 최적 선택
    * GPTQ: 선형 양자화 기반 PTQ, 가중치만 양자화

* 데이터셋
    * ImageNet (256×256, 512×512)
    * Conditional / Unconditional 생성 모두 평가
    * Classifier-Free Guidance Scale(cfg)은 1.5와 4.0 두 가지 설정
 
* 평가방법
    * 50,000장 이미지 샘플링 후 ADM의 TensorFlow Evaluation Suite로 최종 수치 산출

----
