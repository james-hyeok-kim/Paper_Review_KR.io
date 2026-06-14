# DuQuant++: Fine-grained Rotation Enhances Microscaling FP4 Quantization

저자 :

Haokun Lin∗ 1,6, Xinle Jia∗ 2, Haobo Xu3, Bingchen Yao4, Xianglong Guo1,

Yichen Wu5,6, Zhichao Lu6, Ying Wei4, Qingfu Zhang6, Zhenan Sun1

∗Equal Contribution 1 CASIA 2 NJU 3 THU 4 ZJU 5 Harvard 6 CityU

발표 : 2026년 4월 21일 (arXiv 업데이트 기준).

논문 : [PDF](https://arxiv.org/pdf/2604.17789)

---

## 0. Summary

<p align ='center'>
<img width="750" height="300" alt="image" src="https://github.com/user-attachments/assets/2fc26a34-f2b4-4aa6-a28e-52e780441c9e" />
</p>

### 0.1. 문제 정의: MXFP4와 이상치(Outlier)의 충돌

* MXFP4의 구조적 한계: MXFP4 포맷은 32개 요소를 하나의 블록으로 묶어 하나의 스케일(E8M0)을 공유합니다.  
* 이상치 영향 극대화: 블록 내에 단 하나의 거대한 이상치만 존재해도 공유 스케일 값이 커지게 되어, 나머지 요소들의 다이내믹 레인지가 압축되고 양자화 오차가 급증합니다.  
* 기존 방식의 한계: 기존의 아다마르 회전(Hadamard Rotation) 등은 데이터의 실제 이상치 분포를 고려하지 않는 '데이터 무관(data-agnostic)' 방식이라 최적의 효과를 내지 못합니다.  

### 0.2. 핵심 아이디어: DuQuant++

* Fine-grained Outlier-aware Rotation: 기존 DuQuant의 이상치 인지 회전 기법을 MXFP4에 맞게 개선했습니다.  
* 블록 크기 일치 (B=32): 회전 블록 크기를 MXFP4의 마이크로스케일링 그룹 크기인 32와 정확히 일치시켰습니다.  
* 파이프라인 단순화 (Single Rotation):
    * 기존 DuQuant는 2번의 회전과 순열(permutation)이 필요했으나,
    * DuQuant++은 각 그룹이 독립적인 스케일을 갖는 MXFP4의 특성을 이용해 단 한 번의 회전으로 프로세스를 줄였습니다.  
* 효과: 온라인 회전 연산 비용을 절반으로 줄이면서도 이상치를 효과적으로 분산시켜 가중치 분포를 부드럽게 만듭니다.  

### 0.3. 양자화 세부 정보 (Quant Precision & Block Size)

* 양자화 정밀도: W4A4 (가중치 4비트, 활성화 값 4비트).  
* 포맷: MXFP4 (E8M0 스케일 공유).  
* 블록 사이즈 (Group Size): 32 (회전 블록과 양자화 그룹 크기 동일).  
* 보조 기법: GPTQ와 결합하여 가중치 양자화 오차를 추가로 보정합니다.  

### 0.4. 벤치마크 및 효과

* 대상 모델: LLaMA-3 제품군 (8B, 3.2-3B, Instruct 모델 등).  
* 언어 모델링 성능 (WikiText2 Perplexity):
    * LLaMA3-8B: 6.88 달성 (FP16 baseline: 6.14, 기존 QuaRot: 8.07).
    * LLaMA3.2-3B: QuaRot(17.95) 대비 8.87로 성능을 50% 이상 개선했습니다.  
* 제로샷 정확도 (Zero-shot Accuracy):
    * LLaMA3-8B 기준 평균 67.1%를 기록하여 기존 최강 베이스라인인 MR-GPTQ(66.1%)를 상회했습니다.  

---

