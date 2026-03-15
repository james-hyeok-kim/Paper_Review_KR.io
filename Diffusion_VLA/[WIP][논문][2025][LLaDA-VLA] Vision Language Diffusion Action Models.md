# LLaDA-VLA: Vision Language Diffusion Action Models

저자 : 

Yuqing Wen1* Hebei Li1* Kefan Gu2* Yucheng Zhao3†

Tiancai Wang3 Xiaoyan Sun1‡

1University of Science and Technology of China,

2Nanjing University, 3Dexmal

Project Page: https://wenyuqing.github.io/llada-vla/

발표 : arXiv(2025년 2월 14일)

논문 : [PDF](https://arxiv.org/pdf/2509.06932)

---

## 0. Summary

* **기존의 자기회귀(Autoregressive) 방식이 아닌 마스크 확산(Masked Diffusion) 방식을 VLA(Vision-Language-Action) 모델**
* **세계 최초의 시각-언어-확산-동작 모델**

### 두 가지 핵심 설계 (Key Designs)

* 국소적 특수 토큰 분류 (Localized Special-token Classification): 전체 어휘 사전이 아닌 로봇 동작을 위한 특수 토큰(Action Token)에만 분류를 집중하여 학습의 난이도를 낮추고 도메인 적응력을 높였습니다.
* 계층적 동작 구조 디코딩 (Hierarchical Action-Structured Decoding): 동작 간(Inter-action) 및 동작 내(Intra-action) 의존성을 고려하여 계층적으로 디코딩함으로써, 더 정교하고 일관된 로봇 경로를 생성합니다.

### 벤치마크 최고 성능(SOTA)

* SimplerEnv: 기존 OpenVLA 대비 평균 성공률이 51.3% 향상되었습니다.
* CALVIN: 연속 작업 완료 지표(Avg. Len.)에서 4.01을 기록하며 기존 모델들(OpenVLA 3.27 등)을 크게 상회했습니다.
* 실제 로봇: WidowX 로봇 실험에서 $\pi_0$와 CogACT 대비 각각 23%, 28%의 성공률 향상을 보였습니다.
* 탁월한 일반화 능력: 학습되지 않은 물체(큐브), 용기(종이 상자), 방해 요소가 있는 환경에서도 높은 성공률을 유지하며 뛰어난 적응력을 증명했습니다.

---



---
