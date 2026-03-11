# FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling

저자 : 

Ted Zadouri*1,6 Markus Hoehnerbach*2 Jay Shah*3

Timmy Liu4 Vijay Thakkar2,5 Tri Dao1,6

1Princeton University 2Meta 3Colfax Research 4NVIDIA 5Georgia Tech 6Together AI

발표 : 2026년 3월 5일, arXiv

논문 : [PDF](https://arxiv.org/pdf/2603.05451)

---

## 1. Introductdion

### 1. 배경 및 동기: 하드웨어 진화와 새로운 병목 현상

* Transformer의 핵심 병목: Attention
* 긴 문맥(Long-context)의 필요성: 다중 문서 추론, 전체 코드베이스 모델링, 고해상도 비디오 처리
* 비대칭적 하드웨어 확장(Asymmetric Scaling): 최신 GPU(Blackwell 등)는 행렬 연산(MMA) 성능은 급격히 빨라지는 반면, 메모리 대역폭이나 지수 연산(Exponential) 같은 전용 유닛의 속도는 상대적으로 천천히 개선
* Blackwell 아키텍처로의 전환: 업계가 Hopper(H100)에서 Blackwell(B200, GB200)로 빠르게 이동함에 따라, 기존 FlashAttention-3(Hopper 최적화)만으로는 새로운 하드웨어의 특성을 완전히 활용하기 어려워졌습니다.

### 2. FlashAttention-4의 핵심 기술

* 파이프라인 재설계: Blackwell의 완전 비동기 MMA(Matrix Multiply-Accumulate) 연산과 더 커진 타일 크기를 활용하여, 연산과 메모리 이동 간의 오버랩을 극대화했습니다.
* 지수 연산 병목 완화: 소프트웨어적으로 에뮬레이션된 지수 함수(Software-emulated exponential)와 조건부 Softmax 재스케일링 기술을 도입하여, 비-행렬 연산(non-matmul)의 부하를 크게 줄였습니다.
    * Software-emulated exponential: GPU에 내장된 전용 하드웨어 가속기(MUFU)를 사용하는 대신, 일반적인 연산 유닛(FMA)을 사용해 수학적으로 지수 값을 근사해서 계산하는 방식을
    * Blackwell(B200)
        * 행렬 연산(MMA): 텐서 코어(Tensor Core) 덕분에 초당 8192회 연산
        * 지수 연산(MUFU): 전용 다기능 유닛(MUFU)은 초당 16회 연산
    * MUFU: exponent 80%~85% + S/W FMA 연산: 15%~20%
    * $e^x$를 $2^y$ 형태($y = x \cdot \log_2 e$)로 취급
    * 정수와 수소 부분 으로 나눠서 연산
    * 정확도: BF16 정밀도 환경에서는 다항식 근사로 인한 오차가 하드웨어 연산과 비교해 거의 무시할 수 있는 수준(1 BF16 ULP 이내)임이 입증되었습니다.

$$2^y = 2^{\lfloor y \rfloor} \cdot 2^{y - \lfloor y \rfloor}$$

* 메모리 트래픽 최적화: Blackwell의 새로운 하드웨어 기능인 Tensor Memory(TMEM)와 2-CTA MMA 모드를 활용하여 공유 메모리(Shared Memory) 트래픽과 원자적 덧셈(Atomic adds)을 줄였습니다.

### 3. 주요 성과 및 구현 방식

* 탁월한 성능: B200 GPU(BF16 정밀도)에서 cuDNN 9.13 대비 최대 1.3배, Triton 대비 최대 2.7배의 속도 향상을 달성했습니다. 이는 최대 1613 TFLOPs/s(이론상 성능의 71%)에 달하는 수치입니다.
* Python 기반 CuTe-DSL 사용: 전통적인 C++ 템플릿 방식 대신 Python에 내장된 CuTe-DSL로 전체를 구현했습니다. 이를 통해 기존 방식보다 20~30배 빠른 컴파일 속도를 확보하면서도 최저준위 하드웨어 제어 능력을 유지했습니다.

---

## 2. Background

### 2.1 멀티 헤드 어텐션 (Multi-Head Attention)

어텐션 메커니즘은 입력 시퀀스 $Q(Query)$, $K(Key)$, $V(Value)$ 사이의 관계를 계산합니다.기본 연산: 출력 $O$는 다음과 같은 과정을 거쳐 계산됩니다.유사도 점수 ($S$): $S = \alpha QK^\top$ ($\alpha$는 스케일링 인자).소프트맥스 ($P$): $P = \text{softmax}(S)$ (수치적 안정을 위해 각 행의 최댓값을 뺌).최종 출력 ($O$): $O = PV$.역전파 (Backward Pass): 출력의 그래디언트 $dO$가 주어지면, $dV$, $dk$, $dQ$를 계산하기 위해 5번의 행렬 곱셈(MMA) 연산이 필요합니다.

### 2.2 GPU 하드웨어 특성 (Blackwell 아키텍처)

<p align = 'center'>
<img width="903" height="657" alt="image" src="https://github.com/user-attachments/assets/cba4c9a5-d682-4738-a83a-a168ced0df14" />
<img width="559" height="292" alt="image" src="https://github.com/user-attachments/assets/8f914eb5-c123-4d55-9273-0d867ca13e18" />
</p>

1) 새로운 메모리 계층: 텐서 메모리 (TMEM)
    1) Blackwell은 기존의 공유 메모리(SMEM)와 레지스터(RMEM) 외에 텐서 메모리(TMEM)라는 새로운 계층을 도입했습니다.
    2) 용량 및 역할: SM당 256KB가 할당되며, 텐서 코어 연산의 중간 결과를 저장합니다.
    3) 장점: 텐서 코어가 연산 결과를 레지스터를 거치지 않고 TMEM에 직접 비동기적으로 쓰기 때문에, 레지스터 압박(Register Pressure)을 크게 줄여 더 큰 타일 크기를 사용할 수 있게 합니다.
2) 하드웨어 비대칭성과 병목의 변화
    1) 연산 능력 폭증: Blackwell의 텐서 코어 처리량은 Hopper 대비 2배로 증가했습니다 (2.25 PFLOPS vs 1 PFLOPS).
    2) 정체된 유닛: 반면 공유 메모리 대역폭과 지수 연산 유닛(MUFU)의 속도는 그대로이거나 아주 느리게 향상되었습니다.
    3) 결과: 이제 연산 성능보다는 메모리 트래픽과 Softmax 연산(지수 함수) 자체가 전체 성능을 결정짓는 핵심 병목이 되었습니다.
3) 2-CTA 텐서 코어 모드
    1) 두 개의 CTA(Threadblock)가 한 쌍이 되어 하나의 MMA 연산을 협력해서 수행하는 모드입니다.
    2) 이 모드를 사용하면 각 CTA가 필요한 데이터( $B$ 타일)의 절반만 로드하면 되므로, 공유 메모리 대역폭 소모를 절반으로 줄일 수 있습니다.


---

---

---

