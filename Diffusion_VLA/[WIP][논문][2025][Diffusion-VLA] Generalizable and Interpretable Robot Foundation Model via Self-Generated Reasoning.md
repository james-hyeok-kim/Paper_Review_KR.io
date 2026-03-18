# Diffusion-VLA: Generalizable and Interpretable Robot Foundation Model via Self-Generated Reasoning

저자 : 

Junjie Wen 1 2 * Yichen Zhu 1 * † Minjie Zhu 1 2 * Zhibin Tang 1 Jinming Li 3

Zhongyi Zhou 2 Xiaoyu Liu 3 Chaomin Shen 2 Yaxin Peng 3 Feifei Feng 1

* Equal contribution

work done during Junjie Wen and Minjie Zhu’s internship at Midea Group. 

1 Midea Group, Shanghai,China 

2 East China Normal University, Shanghai, China 

3 Shanghai University, Shanghai, China. 

Correspondence to: Yichen Zhu <zhuyc25@midea.com>.

발표 : ICML 2025, arXiv 2025년 6월 4일

논문 : [PDF](https://arxiv.org/pdf/2412.03293)


---

<p align = 'center'>
<img width="716" height="558" alt="image" src="https://github.com/user-attachments/assets/3d50dde2-7c11-4762-83b9-744bfed2835c" />
</p>

## 0. Summary

* 정밀한 로봇 행동 생성에 한계가 있어 Diffusion을 도입했지만, Diffusion은 추론 능력이 부족
* 추론 주입 모듈 도입: VLM이 스스로 생성한 추론 문구를 Diffusion 학습 과정에 직접 삽입하여 로봇 행동의 정밀도 상승

### 의의

* 뛰어난 시각적 일반화 및 제로샷(Zero-shot) 성능
* 로봇 행동의 해석 가능성(Interpretability) 확보
    * 내부생각을 자연어로 보여줌
* 실시간 제어가 가능한 압도적인 추론 속도
    * DiVLA-2B: 82Hz (초당 82번)
    * DiVLA-7B: 42Hz
* 다양한 환경 적응 및 확장성(Scalability)
* 대화 및 시각적 질의응답(VQA) 능력 유지  


---


---


---
