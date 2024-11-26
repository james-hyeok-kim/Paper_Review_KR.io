# Routing Attention
### Efficient Content-Based Sparse Attention with Routing Transformers
저자 : Aurko Roy and Mohammad Saffar and Ashish Vaswani and David Grangier

소속 : Google Research - {aurkor, msaffar, avaswani, grangier}@google.com

논문 : [PDF](https://arxiv.org/pdf/2003.05997)

## 초록
Self-Attention의 연산과 메모리 overhead를 줄이는 것
1. Content-based 접근으로 local, temporal sparse attention보다 효율성을 더 높인다.
2. 연산 복잡도를 $O(N^{2}d) \rightarrow O(N^{1.5}d)$로 줄인다.

## 도입
* RNN CNN의 경우 Local neighborhood context는 연산하지만 self-attention의 경우에는 $N^2$의 복잡도를 가질만큼 전체 sequence length를 연산한다.
  
* n보다 작은 i의 case에서 복잡도는 아래와 같다

$$ \displaystyle\sum_{i\leq n} = n(n-1)/2 $$


* 때문에, Memory efficient attention은 "data independent" 하거나, "fixed sparsity patterns"을 가져야 한다.

## 기존 기법의 특징
### Attention with Temprosal Sparisty
* Fixed temporal context : Dynamically segment the sequence into variable sized chunks 에서 활용할 수 없다.
  - Strided attention : sub-sampled temporal resolution (Sparse attention)
* Hierarchical attention : contiguous neighborhood of selected area에서 활용
  - 추후 coarse layers with the local layers로 간략화 되었다.

### Attention with Content-Based Sparsity
* Sparsemax - using entmax : efficient inference - arbitrary saprsity attention patterns
  - Space and Time Complexity 개선이 없다. (Sparsemax가 연산 자체를 줄여주진 않는다)
* spherical $k$-means outperform LSH(Locality Sensitive Hashing) in MIPS(Maximum Inner product Search)

### Sparse Computation beyond Attention
* Gating for conditional Computation (조건부 연산)
  - 대표적인 예시 MOE(Mixture of experts)
* Key-Value lookup으로 FFN 대체

## Self-Attentive Auto-regressive Sequence Modeling
* 긴 sequence length + $n^2$ complexity application을 타겟으로 진행

## Efficient Content-Dependent Sparse Attention
* Block sparse attention : Half the hedas - Local attention, Half the heads - strided attention
* Local attention relative positional encoding scheme : Good baseline 
* Routing attention 논문의 Target : more generic formulation, 전체 attention matrix가 sparsity구할때 필요없는 경우


## Routing Attention with Clustering
* Queries 와 Keys 값을 Cluster에 할당
* Centroid(중심점) parameter sequence(xy)방향으로 공유
  - Q와 K 중심점은 $\mu$에 모두 포함 (Nearest centroid) $\mu(Q) = \mu(K) = \mu$
* $O(nkd+n^2d/k)$
  - $$nkd = vector\ (n) \times TopK\ centroids\(k) \times space\ of\ size\ (d)$$ - Clustering assignment
  - $$n^2d/k = Query\ (n)\ \times \ Key\ (n)\ \times \ Dimension\ (d) / TopK\ (k)$$ - Query Key dot products 
* Optimal choice of $k$ is $\sqrt{n}$ [PDF](https://arxiv.org/pdf/1904.10509)

### Training centroid $\mu$ update
$$\mu \leftarrow \lambda\mu + \frac{(1-\lambda)}{2} \displaystyle\sum_{i:\mu(Q_i)=\mu}{Q_i}+\frac{(1-\lambda)}{2}\displaystyle\sum_{i:\mu(K_j)=\mu}{K_j}+\frac{(1-\lambda)}{2}$$

$$\lambda = 0.999$$

## 제안 기법
* Transforemr에서 FFN의 weight loading 시간과, computing 시간을 줄이기 위한 내용으로 중요한 부분만 선택적으로 계산하는 방법
* 중요한 부분을 선택할 때는 K-means clustering 을 사용하여 TopK만 연산
<p align="center">
<img src = "https://github.com/user-attachments/assets/205093f1-1bcf-406c-9168-4bb984adbd9f" width="60%" height="60%">
</p> 
<p align="center">
<img src = "https://github.com/user-attachments/assets/f2853b12-2604-4e54-9404-8c612644cd42" width="40%" height="40%">
</p>


## 실험
* Bits/dim이 가장 낮은 것은 Routing Transformer Heads 4, Layers 4, Attention window 1024
* Steps/sec가 가장 많은 것은 Routing Heads 2, Layers 2, Attention window 512

<p align="center">
<img src = "https://github.com/user-attachments/assets/49d86288-f6f5-4bfa-888f-6cc0f704bbd6" width="50%" height="50%">
</p>

* 비교군 대비, Perplexity와 Bits/dim 모두 낮은 수준 (좋은 기법)

<p align="center">
<img src = "https://github.com/user-attachments/assets/95ce8dbd-5bce-4d87-a18a-306ed2c0b08f" width="45%" height="50%">
<img src = "https://github.com/user-attachments/assets/3320cac0-856b-493f-ab9a-b8907ebde081" width="45%" height="50%">
</p>


---

## 부록 - 추가로 공부한 내용
### Sparsemax & Entmax
#### Sparsemax
Sparse.Softmax를 의미하여 0이 아닌, i,j만 softmax처리하고 나머지는 0으로 처리되는 것을 의미

$$ Softmax{(x_{i})} = \frac{ \exp{(x_{i})}}{\displaystyle\sum_i\exp{(x_i})} $$

>where $i,j$ run over sparse tensor indices and unspecified entries are ignores. 
>
>This is equivalent to defining unspecified entries as negative infinity so that 
$exp(x_k)=0$ when the entry with index $k$ has not specified.

#### Entmax
[PDF](https://arxiv.org/pdf/1905.05702)

<p align="center">
<img src = "https://github.com/user-attachments/assets/41c6f17e-b362-4736-99fc-400f7f272fab" width="40%" height="40%">
</p>


#### Batch Normalization
- Batch dimension (N or B) Normalization - Batch size 작은 LSTM/RNN에서 불리
- Multi Batch with 1 channel, 1 Width total Height or 1 Height total Width

#### Layer Normalization
- Sequence dimension (X-Y) normalization
-  1channel Total Width Height

$$μ^l = \frac{1}{H} \displaystyle\sum_{i=1}^{H}a_i^l$$
>$$a_i^l$$ $$l^{th}$$ layer의 $$i^{th}$$ hidden unit으로 들어가는 인풋 총합의 정규화 값

$$ \sigma^l = \sqrt{\frac{1}{H}{\displaystyle\sum_{i=1}^{H}(a_i^l - μ^l)^2}}$$

> Covariate Shift
> 
> 특정 layer output의 변화가 다음 layer로의 인풋 총합에 correlated 변화를 크게 일으킨다.
이러한 covariate shift 문제는, 각 layer에서의 인풋 총합의 mean과 variance를 고정시킴으로써 해결할 수 있다.

<p align="center">
<img src = "https://github.com/user-attachments/assets/40a8c561-97d4-446a-8eb0-017c3ac7c95f" width="40%" height="40%">
</p>

### K-mean clustering
#### 작동원리
1. 군집의 개수(K) 설정
2. 초기 중심점 설정
3. 데이터 군집에 할당(배정)
4. 중심점 재설정(갱신)
5. 데이터를 군집에 재할당(배정) - 4,5 반복


#### example
1. 군집 개수 설정 - 3
<P align="center"> <img src="https://github.com/user-attachments/assets/39c7f619-35b3-4690-bf12-a86c0cdea71f" width="40%" height="40%"></P>

2. 초기 중심점 설정
<P align="center"> <img src="https://github.com/user-attachments/assets/54e43cf1-b6ec-49dd-8644-797330987ae5" width="40%" height="40%"></P>

3. 데이터 군집에 할당(배정)
<P align="center"> 
<img src = "https://github.com/user-attachments/assets/854ab366-0224-475c-baee-11b3927e1522" width="40%" height="40%">
<img src = "https://github.com/user-attachments/assets/b1e0ea74-1448-406d-aafb-d73a728e1a72" width="40%" height="40%">
<img src = "https://github.com/user-attachments/assets/234ddfb8-1895-4537-9471-fc34a28b866b" width="40%" height="40%">
<img src = "https://github.com/user-attachments/assets/9c76c656-6384-4a14-b128-0a3bf417a3c5" width="40%" height="40%">
<img src = "https://github.com/user-attachments/assets/82a1fba3-acd6-43b5-9c03-8f28ebdb3096" width="40%" height="40%">
<img src = "https://github.com/user-attachments/assets/59a4db89-6233-4f50-a267-b890dcd24b1e" width="40%" height="40%">
</P>

4. 중심점 재설정(갱신)
<P align="center"> 
<img src = "https://github.com/user-attachments/assets/66d13162-9801-42ee-b84e-cbf834bf920c" width="40%" height="40%">
<img src = "https://github.com/user-attachments/assets/3a9b81cb-8745-421a-9f35-0bbddb952e77" width="40%" height="40%">
</P>

5. 데이터 군집에 재할당(배정)
<P align="center"> 
<img src = "https://github.com/user-attachments/assets/ee868ac2-b0a6-4206-9929-5d53192fff24" width="40%" height="40%">
</P>


#### K-mean Algorithm 초기 중심점 random select의 문제
1. 클러스터 중심(센트로이드)을 초기에 랜덤하게 위치시키기 때문에, 매번 결과가 달라질 수도 있다.
2. 한 번에 k개의 센트로이드를 랜덤하게 생성하기 때문에, 각 센트로이드 사이의 거리가 짧으면 분류가 제대로 이루어지지 않을 수 있다.

#### K-means++ Algorithm
1. 센트로이드를 한 번에 k개 모두 생성하는 것이 아니라, 데이터 포인터 중에서 무작위로 '1개'를 선택하여, 이 데이터를 첫 번째 센트로이드로 지정한다.
<P align="center"> <img src = "https://github.com/user-attachments/assets/d5363478-3d64-487e-9563-6015a4c69bea" width="40%" height="40%"> </P>
2. 나머지 데이터 포인터들과 센트로이드 사이의 거리를 계산한다.
<P align="center"> <img src = "https://github.com/user-attachments/assets/cd6aca41-de06-4700-af5f-cd36b8602ffc" width="40%" height="40%"> </P>
3. 그다음 생성할 센트로이드들의 위치는, 데이터 포인터들과 2번 과정에서 계산한 센트로이드 사이의 거리비례 확률에 따라 선정된다.
<P align="center"> <img src = "https://github.com/user-attachments/assets/8a50840b-2aab-4fc5-890d-eae450c710b5" width="40%" height="40%"> </P>
4. 위 과정을 k번 반복하여 총 k개의 센트로이드를 생성한다.
5. 센트로이드 사이의 거리를 최대한 멀리 위치시키는 방향으로 1개씩 총 k번 반복하여 k개의 클러스터를 만들어낸다는 뜻이다.


#### Locality-sensitive hasing
비슷한 자료를 같은 Buckets(바구니) 에 넣어서 찾는 알고리즘
* 서로 가까운 포인트들은 같은 비구니에, 먼 포인터 들은 다른 바구니에 남겨지는 확률적 알고리즘
<P align="center"> <img src = "https://github.com/user-attachments/assets/c33b9aad-189c-42c2-af72-ad6365d57b25" width="30%" height="30%"> </P>

1. Shingling
*  Shingle(조약돌) 로 만드는 단계
*  "Nadal" \rightarrow "Na", "ad", "da", "al" 로 만드는 단계 (2-shingles)

2. Jaccard Index(유사성)
* $$J(A,B) = \frac{|A \cap B|}{|A \cup B|}$$
* A:{Na, ad, da, al}
* B:{Na, ad, di, ia}
* Jaccard Index = $\frac{2}{6}$

3. hashing
* Input : $d$, Hash function : $H()$
* $d_1$, $d_2$ 유사성 높으면 $H(d_1)$ $H(d_2)$ 유사성도 높다

3-1. Min-hasing
* 1. 문서의 shingle 행렬의 행 인덱스를 랜덤으로 섞는다. 
<P align="center"> <img src = "https://github.com/user-attachments/assets/e17fd4cc-e6f3-4e6e-8521-218ca9649a43" width="30%" height="30%"> </P>
* 2. 왼쪽 갈색 Index 행렬의 1이 있는 곳을 보면 2번째와 4번째 열에 해당한다.
     그러므로, Signature 행렬에서 1이라는 행의 인덱스를 2번째와 4번째 열에 넣어 준다.
     그리고 2번째 행 인덱스에서의 행의 값을 보면, 1이 1번째와 3번째에서 등장한다.
     그러면 2라는 행의 인덱스를 해당 열에 넣어 준다.
     Signature matrix M이 다 차면 끝
<P align="center"> <img src = "https://github.com/user-attachments/assets/09bbaf25-5810-4365-8658-2078f3d3df24" width="30%" height="30%"> </P>
* 3. 3가지 랜덤 인덱스의 Signature 값
<P align="center"> <img src = "https://github.com/user-attachments/assets/cc4e3343-85c8-48ea-b939-293a9790aab7" width="30%" height="30%"> </P>
* 4. Signature 행렬의 Jaccard Index 유사성을 판별
<P align="center"> <img src = "https://github.com/user-attachments/assets/2d0f03fa-5c3d-4e3e-8456-7072b0927371" width="30%" height="30%"> </P>

4. Locality-sensitive hashing
* LSH의 일반적인 아이디어는 2개의 문서의 signature를 만들었을 때, 그것이 이 두 문서들이 쌍인지 아닌지를 판별할 수 있는 알고리즘을 찾는 것이다.
* Band partitioning을 하여서 해쉬 함수를 나누고 이를 bucket에 넣어 유사한것 끼리 비교한다.(bxr은 상수, b가 줄어들면 r이 늘고)
<P align="center"> <img src = "https://github.com/user-attachments/assets/dda943ec-a277-41b8-88a1-ac1f9087723b" width="30%" height="30%">
<img src = "https://github.com/user-attachments/assets/0f6ea840-a717-4e60-a1b0-0271b2dc1aab" width="30%" height="30%"> </P>


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


## 추가 공부 필요한 논문

