# Routing Attention
### Efficient Content-Based Sparse Attention with Routing Transformers
저자 : Aurko Roy and Mohammad Saffar and Ashish Vaswani and David Grangier

소속 : Google Research - {aurkor, msaffar, avaswani, grangier}@google.com

논문 : [PDF](https://arxiv.org/pdf/2003.05997)

## 제안 기법
Transforemr에서 FFN의 weight loading 시간과, computing 시간을 줄이기 위한 내용으로 
중요한 부분만 선택적으로 계산하는 방법

K-means clustering (Maximum Inner Product Search)- MIPS problem



## 실험

## 비교 기법

## 추가로 알게된 사항
sparsemax & entmax

Batch Normalization
- Batch dimension (N or B) Normalization - Batch size 작은 LSTM/RNN에서 불리
- Multi Batch with 1 channel, 1 Width total Height or 1 Height total Width

Layer Normalization
- Sequence dimension (X-Y) normalization
-  1channel Total Width Height

$$μ^l = \frac{1}{H} \displaystyle\sum_{i=1}^{H}a_i^l$$
>$$a_i^l$$ $$l^{th}$$ layer의 $$i^{th}$$ hidden unit으로 들어가는 인풋 총합의 정규화 값

$$ \sigma^l = \sqrt{\frac{1}{H}{\displaystyle\sum_{i=1}^{H}(a_i^l - μ^l)^2}}$$

> Covariate Shift
> 
> 특정 layer output의 변화가 다음 layer로의 인풋 총합에 correlated 변화를 크게 일으킨다.
이러한 covariate shift 문제는, 각 layer에서의 인풋 총합의 mean과 variance를 고정시킴으로써 해결할 수 있다.

<img src = "https://github.com/user-attachments/assets/40a8c561-97d4-446a-8eb0-017c3ac7c95f" width="40%" height="40%">




## 추가 공부 필요한 사항
1. Attention with Temporal Sparsity
   
2. Attention with Content-Based Sparsity

* Fixed bounded local context - 
* Decreasing the temporal resolution of context -

3. Sparse Computation beyond Attention
  * Gating techniques
  * Sparsely gated Mixture-of-experts
  * Key-value lookups to replace the feed forward network in the Transformer

4. Self-Attentive Auto-regressive Sequence Modeling
5. 
