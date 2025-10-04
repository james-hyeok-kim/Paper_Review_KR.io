# MatFormer: Nested Transformer for Elastic Inference
저자 : Devvrit∗∆⋄, Sneha Kudugunta∗†⋄, Aditya Kusupati∗†⋄+, Tim Dettmers† Kaifeng Chen⋄, Inderjit Dhillon⋄∆, Yulia Tsvetkov†, Hannaneh Hajishirzi†, Sham Kakade‡, Ali Farhadi†, Prateek Jain⋄+

출간 : ICML(International Conference on Machine Learning), 2023.

논문 : [PDF](https://arxiv.org/pdf/2310.07707)

---
<p align = "center">
<img width="733" height="352" alt="image" src="https://github.com/user-attachments/assets/f1b87824-5c5b-4995-96da-b3b41b640adb" />
</p>

### Introduction

* 해당 논문은 하나의 모델로 여러 크기의 모델을 지원하고자함
* 중첩된 FFN구조 사용
  * 작은 모델의 파라미터가 큰 모델의 파라미터에 완전히 포함되는 방식
* 폭넓은 적용성
  * 언어와 비전모델 모두 적용가능한 모델


### Related Work
#### 기존 연구의 한계 

1. 모델 패밀리를 독립적으로 학습시, 훈련비용 증가
2. 사후(Post-Hoc) 최적화: Pruning, Quantization, Distillation 기법으로 배포시 추가적인 튜닝이나 학습 필요
3. Speculative Decoding은 작고 가벼운 Draft(초안) + 크고 정학한 Verifier(검증) 모델이 필요한데, 행동의 일관될때 효율적. 독립적으로 학습된 모델은 일관성을 보장하기 어렵다

* 때문에 이는 탄력적(Elastic)해결책으로 보기 어렵다.

#### 핵심 연구
1. OFA(Once for All)
1-1. 기존 교사 - 학생 모델 구조를 사용하면 증류(Distillation)기법에 의존
1-2. MatFormer는 증류가 필요 없으며, 중첩(Nested)구조를 사용하여 여러 모델을 동시에 서비스 할때, 메모리 효율성이 높습니다.

2. HAT (Hardware-aware Transformers)
2-1. 기존 HAT는 최적의 아키텍처를 탐색(NAS) 한뒤 처음부터 다시 학습새야한다
2-2. MatFormer는 추가학습 없이 Mix'n Match라는 간단한 방법으로 최적화 하위모델을 얻을 수 있다.

3. DynaBERT와 차이점
3-1. DynaBERT도 MatFormer처럼 고정된 개수의 하위모델을 공동 최적화
3-2. 학습시 Gradient 업데이트 횟수가 적어 동일 학습자원대비 효율이 떨어진다.

* 이전 연구는 대부분 기하급수적으로 많은 하위 모델을 최적화 하려고 노력하지만 MatFormer는 4개와 같이 매우 적은 수의 중첩된 하위 모델만 명시적으로 최적화


### MatFormer

#### Nested Architecture(중첩구조)
* 러시아 인형 마트료시카 모델
* 가장 작은 모델이 가장 중요하다고 생각하여 정렬
* $M_1$모델이 가장 첫 $m_1$개 뉴런 사용, 다음 크기 $M_2$는 첫 $m_2$개 뉴런 사용 ...
* 예를 들어 FFN의 비율이 {0.5, 1, 2, 4}로 조절

#### 학습방식 (무작위 샘플링)

* 매 학습 단계마다 g개의 하위 모델중 하나를 무작위 선택
* 선택된 하위 모델 $M_i$에 대해 일반적인 경사하강법으로 손실 계산하고 파라미터 업데이트
* g개의 모델뿐만 아니라, 이들을 조합하여 만들 수 있는 수많은 모델들까지 높은 성능

#### Mix'n'Match로 수많은 모델 추출하기
* 각 레이어마다 서로다른 크기의 블록을 조합
* 최적 조합 탐색
* 간단한 휴리스틱으로 높은 효율, 점진적으로 깊은 레이어에서 더 큰 블록을 사용\[S, S, M, L, L\]처럼 크기를 늘리는 것이 \[S, XL, S, XL \]처럼 들쭉날쭉한 구성보다 성능이 좋다

#### 배포 (정적 및 동적 환경 활용)

* 정적 환경에서는 34B, 70B처럼 고정되는게 아니라 40B처럼 원하는 크기 모델 추출
* 동적 환경에서는 시스템 부하에 따라 모델 크기 조정
* 그럼에도 불구하고 높은 일관성 유지


