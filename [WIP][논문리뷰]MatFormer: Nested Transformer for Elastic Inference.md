# MatFormer: Nested Transformer for Elastic Inference
저자 : Devvrit∗∆⋄, Sneha Kudugunta∗†⋄, Aditya Kusupati∗†⋄+, Tim Dettmers† Kaifeng Chen⋄, Inderjit Dhillon⋄∆, Yulia Tsvetkov†, Hannaneh Hajishirzi†, Sham Kakade‡, Ali Farhadi†, Prateek Jain⋄+

출간 : ICML(International Conference on Machine Learning), 2023.

논문 : [PDF](https://arxiv.org/pdf/2310.07707)

---

### Introduction

* 해당 논문은 하나의 모델로 여러 크기의 모델을 지원하고자함
* 중첩된 FFN구조 사용
  * 작은 모델의 파라미터가 큰 모델의 파라미터에 완전히 포함되는 방식
* 폭넓은 적용성
  * 언어와 비전모델 모두 적용가능한 모델

