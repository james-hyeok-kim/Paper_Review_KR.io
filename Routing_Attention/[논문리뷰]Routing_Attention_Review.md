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
