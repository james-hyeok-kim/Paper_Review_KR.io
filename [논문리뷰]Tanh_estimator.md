# Tanh Estimator
Impact of Data Normalization on Deep Neural Network for Time Series Forecasting

저자 : Samit Bhanja and Abhishek Das

소속 : IEEE meber

논문 : [PDF](https://arxiv.org/pdf/1812.05519)

## 초록

Serial Data의 여러 Normalization 효과를 알아보는 논문

## 도입


## 제안기법

### Min-Max Normalization
$$x_{norm} = \frac{(high-low) * (x-X_{min})}{X_{max} - X_{min}}$$

### Decimal Scaling Normalization
* 소수점 변환
* 최대값을 따라 자리수 변환
* $d : 최소정수 Max(|x_{norm}|) < 1$

$$x_{norm} = \frac{x}{10^d} $$


### Z-Score Normalization
* 0과 표준편차의 범위로 전환
* $\mu(X):x의 평균$
* $\sigma(X):x의 표준편차 - 모집단$
* $\delta(X):x의 표준편차 - 표분$

$$x_{norm} = \frac{x-\mu(X)}{\delta(X)} $$

### Median Normalization
* 중앙값에 의한 정규화

$$ x_{norm} = \frac{x}{median(X)}$$

### Sigmoid Normalization
* sigmoid 함수 이용 정규화

$$ x_{norm} = \frac{1}{1-e^{-x}} $$

### Tanh estimator
* 아래 공식으로 정규화
* $\mu(X):x의 평균$
* $\sigma(X):x의 표준편차 - 모집단$
* $\delta(X):x의 표준편차 - 표분$

$$ x_{norm}=0.5[tanh[\frac{0.01(x-\mu)}{\delta(X)}]+1] $$

## 결론

## 실험결과

## 부록 - 추가공부
