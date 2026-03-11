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

* 기본 연산: 출력 $O$는 다음과 같은 과정을 거쳐 계산됩니다.
* 유사도 점수 ( $S$ ): $S = \alpha QK^\top$ ($\alpha$는 스케일링 인자).
* 소프트맥스 ( $P$ ): $P = \text{softmax}(S)$ (수치적 안정을 위해 각 행의 최댓값을 뺌).
* 최종 출력 ( $O$ ): $O = PV$.
* 역전파 (Backward Pass): 출력의 그래디언트 $dO$가 주어지면, $dV$, $dk$, $dQ$를 계산하기 위해 5번의 행렬 곱셈(MMA) 연산이 필요합니다.

### 2.2 GPU 하드웨어 특성 (Blackwell 아키텍처)

<p align = 'center'>
<img width="903" height="657" alt="image" src="https://github.com/user-attachments/assets/cba4c9a5-d682-4738-a83a-a168ced0df14" />
<img width="707" height="384" alt="image" src="https://github.com/user-attachments/assets/4e195832-54de-403c-8846-9a6e446e858b" />
</p>

1) 새로운 메모리 계층: 텐서 메모리 (TMEM)
    1) Blackwell은 기존의 공유 메모리(SMEM)와 레지스터(RMEM) 외에 텐서 메모리(TMEM)라는 새로운 계층을 도입했습니다.
    2) 용량 및 역할: SM당 256KB가 할당되며, 텐서 코어 연산의 중간 결과를 저장합니다.
    3) 장점: 텐서 코어가 연산 결과를 레지스터를 거치지 않고 TMEM에 직접 비동기적으로 쓰기 때문에, 레지스터 압박(Register Pressure)을 크게 줄여 더 큰 타일 크기를 사용할 수 있게 합니다.
2) 하드웨어 비대칭성과 병목의 변화
    1) 연산 능력 폭증: Blackwell의 텐서 코어 처리량은 Hopper 대비 2배로 증가했습니다 (2.25 PFLOPS vs 1 PFLOPS).
    2) 정체된 유닛: 반면 공유 메모리 대역폭과 지수 연산 유닛(MUFU)의 속도는 그대로 이거나 아주 느리게 향상되었습니다.
    3) 결과: 이제 연산 성능보다는 메모리 트래픽과 Softmax 연산(지수 함수) 자체가 전체 성능을 결정짓는 핵심 병목이 되었습니다.
3) 2-CTA 텐서 코어 모드
    1) 두 개의 CTA(Threadblock)가 한 쌍이 되어 하나의 MMA 연산을 협력해서 수행하는 모드입니다.
    2) 이 모드를 사용하면 각 CTA가 필요한 데이터( $B$ 타일)의 절반만 로드하면 되므로, 공유 메모리 대역폭 소모를 절반으로 줄일 수 있습니다.


---

## 3. Algorithm

### 3.1 어텐션 순전파 (Forward Pass)

<p align = 'center'>
<img width="600" height="499" alt="image" src="https://github.com/user-attachments/assets/6c4a5613-b8c2-4927-99db-f84645047199" />
</p>

* 루프라인 분석 (Roofline Analysis): 분석 결과, Blackwell 아키텍처에서는 행렬 연산(MMA)이 너무 빨라진 나머지, 지수 연산 유닛(Exp)과 공유 메모리(SMEM) 트래픽이 동일하거나 더 큰 병목이 되었음을 확인했습니다.

<p align = 'center'>
<img width="984" height="336" alt="image" src="https://github.com/user-attachments/assets/54c1f6b7-3153-413c-bb87-ba5ffb724e0b" />
</p>

* 소프트웨어 지수 함수 에뮬레이션: 느린 하드웨어 전용 유닛(MUFU) 대신, 상대적으로 자원이 넉넉한 일반 연산 유닛(FMA)을 사용하여 $2^x$를 수학적으로 근사 계산했습니다. 이를 통해 지수 연산 처리량을 획기적으로 높였습니다.
    * 3차 다항식만 써도 오차가 매우 작아서, 이를 **BF16으로 변환(반올림)하는 순간 발생하는 손실(양자화 오차)**이 다항식 자체의 계산 오차보다 훨씬 커지게 됩니다



* 온라인 소프트맥스 재스케일링 생략: 매 단계마다 수행하던 재스케일링을 조건부로 변경했습니다. 새로운 최댓값( $m_j$ )과 이전 최댓값( $m_{j-1}$ )의 차이가 특정 임계값( $\tau$ )보다 클 때만 재스케일링을 수행하여 연산 횟수를 줄였습니다.
    * 문제 상황: 새로운 블록을 계산했는데, 이전 블록들보다 더 큰 최댓값이 나타나면 어떻게 될까요? 이전에 계산해둔 결과값($O_{j-1}$)들을 새로운 최댓값에 맞춰서 다시 조정해줘야 합니다. 이 과정을 재스케일링이라고 하며, 이때 벡터 곱셈 연산이 발생
    * FlashAttention-4는 새로운 최댓값( $m_j$ )과 이전 최댓값( $m_{j-1}$ )의 차이가 임계값( $\tau$ )보다 클 때만 재스케일링을 수행
    * 보통 $\tau = 8.0$ (즉, $log_2(256)$ )으로 설정하는데, 이는 값이 약 256배 이상 차이 날 때만 정밀하게 조정

* TMEM 기반 파이프라인: 텐서 메모리(TMEM)를 활용해 출력 타일을 저장함으로써 레지스터 압박을 줄이고, 한 타일이 MMA를 수행하는 동안 다른 타일은 소프트맥스를 계산하는 '핑퐁(Ping-pong)' 스케줄링을 구현했습니다.
    * 회색 블록(MMA)과 노란색/주황색 블록(Softmax)이 수직으로 겹쳐서 동시에 진행되는 것


### 3.2 어텐션 역전파 (Backward Pass)

* 역전파에서는 순전파보다 더 많은 5번의 MMA 연산이 발생하며, 공유 메모리 대역폭이 가장 큰 병목이 됩니다.
* 2-CTA MMA 모드: Blackwell의 새로운 기능을 활용하여 두 개의 CTA(Threadblock)가 한 쌍으로 동작하게 했습니다. 이를 통해 각 CTA는 필요한 데이터( $B$ 타일)의 절반만 로드하면 되므로 공유 메모리 트래픽이 감소합니다.
* dQ 원자적 연산(Atomic Adds) 절반 감소: 2-CTA 모드에서 $dS$ 데이터를 분산 공유 메모리(DSMEM)로 교환하여 $dQ$를 계산함으로써, 전역 메모리에 결과를 쓸 때 발생하는 비싼 원자적 연산 횟수를 1-CTA 방식 대비 절반으로 줄였습니다.
* 결정론적(Deterministic) 모드: 훈련의 재현성을 위해 세마포어 락(Semaphore lock)을 사용한 순차적 연산 모드를 제공하며, 성능 저하를 최소화하기 위한 최적의 CTA 처리 순서를 설계했습니다.

### 3.3 스케줄링 및 최적화 (Scheduling)

* LPT (Longest-Processing-Time-first): 연산 시간이 가장 길 것으로 예상되는 타일을 SM(Streaming Multiprocessor)에 먼저 할당하여 전체 작업 완료 시간을 단축했습니다.
* Causal Masking 최적화: 마스킹으로 인해 삼각형 모양으로 줄어든 작업량을 효율적으로 배분하여, 기존 방식 대비 4~14%의 성능 향상을 달성했습니다.
* 가변 시퀀스 길이 (Varlen): 배치마다 시퀀스 길이가 다른 경우, 전처리 커널을 통해 작업량을 미리 계산하고 정렬하여 SM 간의 부하를 균등하게 맞췄습니다.

---

---

