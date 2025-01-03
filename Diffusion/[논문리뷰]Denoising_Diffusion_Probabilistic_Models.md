# Denoising Diffusion Probabilistic Models
저자(소속) : Jonathan Ho (UC Berkeley), Ajay Jain(UC Berkeley), Pieter Abbeel(UC Berkeley)
논문 : [PDF](https://arxiv.org/pdf/2006.11239)
일자 : 16 Dec 2020

## 초록

## 도입
![image](https://github.com/user-attachments/assets/1351575a-8638-446c-9a9b-d5d9dc8db15c)

Markov chain forwarding 방식으로 noise를 더하고, reverse방식으로 noise에서 이미지를 생성



## 실험

## 결과

## 부록
### Markov Chain
#### 마르코프 성질 + 이산시간 확률 과정
마르코프 체인은 '마르코프 성질'을 가진 '이산시간 확률과정' 입니다.
마르코프 성질 - 과거와 현재 상태가 주어졌을 때의 미래 상태의 조건부 확률 분포가 과거 상태와는 독립적으로 현재 상태에 의해서만 결정됨
이산시간 확률과정 - 이산적인 시간의 변화에 따라 확률이 변화하는 과정
![image](https://github.com/user-attachments/assets/7ae5afbc-7884-4e35-a570-cb87513daaf7)

#### 결합확률분포(Joint Probability Distribution)
예를 들어 확률 변수 $X_1,X_2, ... , X_n$ 이 있다고 가정하면,
일반적으로 이 확률변수들의 결합확률분포는 다음과 같이 계산할 수 있다.

$$ P(X_1,X_2, ... , X_n) = P(X_1) \times P(X_2|X_1) \times P(X_3|X_2,X_1)\times  ...  \times P(X_n|X_{n-1}, X_{n_2} , ... , X_1) $$
 
하지만 마르코프 성질을 이용하면 위 보다 더 단순한 계산을 통해 결합확률분포를 구할 수 있다.

$$ P(X_n|X_{n-1}, X_{n_2} , ... , X_1) = P(X_{t+1}|X_t) $$
 

만약 어떠한 상태의 시점이고, 확률분포가 마르코프 성질을 따른다면 

$$ P(X_1,X_2, ... , X_n) = P(X_1) \times P(X_2|X_1) \times P(X_3|X_2)\times  ...  \times P(X_n|X_{n-1}) $$

단순화 할 수 있고 일반화를 적용하면 이전에 결합확률분포의 계산을 다음과 같이 단순화 가능하다.
