# Score-Based Generative Modeling through Stochastic Differential Equations
저자 : Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole

논문 : [PDF](https://arxiv.org/pdf/2011.13456)

일자 : Submitted on 26 Nov 2020 (v1), last revised 10 Feb 2021 (this version, v2), ICLR


<img width="666" height="307" alt="image" src="https://github.com/user-attachments/assets/1ba90f03-e42b-4107-bbde-947fc08329fa" />

## 핵심 아이디어
* Forward Process를 연속적인 시간의 흐름으로 보고 이 과정을 수학적으로 완벽히 되돌리면 노이즈에서 실제 이미지가 발생
* 이모든 과정을 확률적 미분 방정식(SDE)를 통해 구현

 
 역방향 SDE를 풀기 위해 필요한 정보는 단 하나, 바로 스코어(score) 함수입니다.

* 스코어 함수 $(\nabla x \log p_t(x))$
  * 특정 시간 t에서의 노이즈 낀 데이터 $x(t)$의 확률 분포 $(p_t(x))$ 에 로그를 씌운 뒤, 데이터 x에 대해 미분(gradient)한 값입니다.
  * 직관적으로 이 '스코어'는 노이즈 낀 데이터 상태에서 어느 방향으로 가야 더 진짜 데이터에 가까워지는지 알려주는 역할
  * 즉, 데이터의 밀도가 높은 방향을 가리키는 벡터입니다.
 
### SDE(Stochastic Differential Equation, 확률적 미분 방정식)
* 어떤 시스템의 시간에 따른 변화에 예측 불가능한 '무작위성(randomness)'이 포함될 때, 그 움직임을 설명하는 수학적 도구입니다.
* 일반적인 미분 방정식이 예측 가능한 움직임만을 다룬다면, SDE는 여기에 '랜덤한 충격'이 계속해서 더해지는 상황을 모델링합니다.

$$
dx=f(x,t)dt+g(t)dw
$$

1. Drift Term (드리프트 항): $f(x,t)dt$

* 시간이 dt만큼 흘렀을 때, 현재 상태 x와 시간 t에 따라 시스템이 평균적으로 어느 방향으로 얼마만큼 움직이는지를 결정합니다.
* 비유: 강물에 떠 있는 나뭇잎이 있을 때, 강의 주된 흐름이 바로 드리프트입니다. 이 흐름은 나뭇잎을 대체로 하류로 밀어냅니다.

2. Diffusion Term (확산 항): $g(t)dw$

* 이 부분은 시스템의 예측 불가능한, 무작위적(stochastic)인 움직임을 나타냅니다.
* dw는 '위너 과정(Wiener Process)' 또는 '브라운 운동(Brownian Motion)'이라고 불리는 순수한 랜덤 노이즈를 의미합니다.
* 매 순간 아주 작은 랜덤한 충격을 주는 것이라고 생각할 수 있습니다.
* g(t)는 **확산 계수(diffusion coefficient)**로, 이 랜덤한 충격의 **세기(강도)**를 조절합니다.
* 시간이 지남에 따라 랜덤성의 영향이 커지거나 작아지게 할 수 있습니다.
* 비유: 강물 위의 나뭇잎이 물의 소용돌이나 바람 때문에 예측 불가능하게 이리저리 흔들리는 움직임이 바로 확산입니다.

