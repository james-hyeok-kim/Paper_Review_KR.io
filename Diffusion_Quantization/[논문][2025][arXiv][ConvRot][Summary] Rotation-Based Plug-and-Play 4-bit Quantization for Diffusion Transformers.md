# ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers

저자 :

* Feice Huang1†∗, Zuliang Han2∗, Xing Zhou2, Yihuang Chen2, Lifei Zhu2, Haoqian Wang1✉

* 1SIGS, Tsinghua University
* 2Central Media Technology Institute, Huawei

발표 : 2025년 12월 3일 (arXiv:2512.03673v1)

논문 : [PDF](https://arxiv.org/pdf/2512.03673)

---

## 0. Summary

<p align ='center'>
<img width="1106" height="531" alt="image" src="https://github.com/user-attachments/assets/2aaa8a5f-a441-4027-90e2-5a28195f40a4" />
</p>

### 0.1. 핵심아이디어

* ConvRot (Group-wise Regular Hadamard Transform)
    * 기존 Hadamard 변환은 column-wise outlier만 처리 가능했으나, Diffusion Transformer에는 row-wise outlier도 존재
    * Regular Hadamard Matrix(RHT)를 사용해 row-wise + column-wise outlier를 동시에 억제
    * 채널을 N0 크기의 블록으로 나눠 그룹 단위 회전 → 복잡도를 O(K²) → O(K)로 선형 감소


* ConvLinear4bit
    * Rotation + Quantization + INT4 GEMM + Dequantization을 하나의 레이어에 통합
    * 기존 Linear 레이어를 그대로 교체(plug-and-play) 가능
    * 재학습(retraining) 불필요

<p align = 'center'>
<img width="811" height="322" alt="image" src="https://github.com/user-attachments/assets/383669e4-a35c-455f-96e2-376637973e45" />
</p>


#### Standard vs Regular Hadamard


**Standard Hadamard**

<p align = 'center'>
<img width="161" height="78" alt="image" src="https://github.com/user-attachments/assets/cdff9800-5abe-4a88-b711-897a55aee9d9" />
</p>

* 모태는 2x2 행렬

<p align = 'center'>
<img width="389" height="87" alt="image" src="https://github.com/user-attachments/assets/f7f25d17-afdd-4292-96e5-1295995e925d" />
</p>

* 자기자신과 곱하면서 행렬 크기를 키워간다

<p align = 'center'>
<img width="267" height="145" alt="image" src="https://github.com/user-attachments/assets/6e1db301-3de9-4448-9131-df94d5e01489" />
</p>


**Regular Hadamard**

* 크기($n$)가 완전제곱수(Perfect Square), $n=4^k$ (4, 16, 64, 256...)

* 수식:
    * $H_4^{reg} = J_4 - 2I_4$
    * $J_4$: 모든 원소가 1인 $4 \times 4$ 행렬
    * $I_4$: 단위 행렬 (Identity matrix)

<p align = 'center'>
<img width="627" height="142" alt="image" src="https://github.com/user-attachments/assets/151e3e8a-ef49-411f-babb-7f0b73d9495d" />
</p>

* 이 행렬을 보면 모든 행의 합이 $1+1+1-1 = 2$가 됩니다. ( $n=4$일 때 $s = \sqrt{4} = 2$ )




### 0.2. 효과 (FLUX.1-dev 기준, RTX 4090)

* 추론 속도 향상: 2.26× 빠름
* 메모리 절감: 4.05× 감소
* DiT 메모리: 22.7 GiB → 5.6 GiB
* 추론 시간: 54.6s → 23.2s

* Mixed precision 전략(20% INT8 혼합) 적용 시 SVDQuant 수준의 이미지 품질 회복


### 0.3. 실험 모델

* FLUX.1-dev (12B, 50 steps)
* FLUX.1-schnell (12B, 4 steps)


### 0.4. 비교 대상 (Baselines)

* BF16: 원본 풀 정밀도 기준선
* NF4: W4A16, QLoRA용 4-bit 포맷
* SVDQuant: W4A4 + 16-bit LoRA 브랜치 병렬 유지
* QuaRot: LLM용 rotation 기반 양자화 (비교용)

### 0.5. 데이터셋

* MJHQ-30K: 5K 프롬프트 샘플링, 다양한 카테고리
* sDCI (summarized Densely Captioned Images): 상세 캡션 기반 이미지 평가셋

### 0.6. 평가 지표

* FID↓: 이미지 품질
* ImageReward (IR)↑ 높을수록 좋음: 사람 선호도
* LPIPS↓: 원본과의 지각적 유사도
* PSNR↑: 원본과의 픽셀 유사도
* DiT Memory↓: GPU 메모리 사용량
* Latency↓: 추론 시간


---
