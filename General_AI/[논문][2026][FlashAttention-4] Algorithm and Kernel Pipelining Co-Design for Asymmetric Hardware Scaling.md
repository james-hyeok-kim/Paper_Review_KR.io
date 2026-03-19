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
* Blackwell 아키텍처로의 전환: 기존 FlashAttention-3(Hopper 최적화)만으로는 새로운 하드웨어의 특성을 완전히 활용하기 어려워졌습니다.

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
* Attention Score ( $S$ ): $S = \alpha QK^\top$ ($\alpha$는 스케일링 인자).
* 소프트맥스 ( $P$ ): $P = \text{softmax}(S)$ (수치적 안정을 위해 각 행의 최댓값을 뺌).
* 최종 출력 ( $O$ ): $O = PV$.
* 역전파 (Backward Pass): 출력의 그래디언트 $dO$가 주어지면, $dV$, $dk$, $dQ$를 계산하기 위해 5번의 행렬 곱셈(MMA) 연산이 필요합니다.

### 2.2 GPU 하드웨어 특성 (Blackwell 아키텍처)

<p align = 'center'>
<img width="903" height="657" alt="image" src="https://github.com/user-attachments/assets/cba4c9a5-d682-4738-a83a-a168ced0df14" />
<img width="707" height="384" alt="image" src="https://github.com/user-attachments/assets/4e195832-54de-403c-8846-9a6e446e858b" />
</p>

### 2.2.1 새로운 메모리 계층: 텐서 메모리 (TMEM)
    
1) Blackwell은 기존의 공유 메모리(SMEM)와 레지스터(RMEM) 외에 텐서 메모리(TMEM)라는 새로운 계층을 도입했습니다.
2) 용량 및 역할: SM당 256KB가 할당되며, 텐서 코어 연산의 중간 결과를 저장합니다.
3) 장점: 텐서 코어가 연산 결과를 레지스터를 거치지 않고 TMEM에 직접 비동기적으로 쓰기 때문에, 레지스터 압박(Register Pressure)을 크게 줄여 더 큰 타일 크기를 사용할 수 있게 합니다.

### 2.2.2 하드웨어 비대칭성과 병목의 변화

1) 연산 능력 폭증: Blackwell의 텐서 코어 처리량은 Hopper 대비 2배로 증가했습니다 (2.25 PFLOPS vs 1 PFLOPS).
2) 정체된 유닛: 반면 공유 메모리 대역폭과 지수 연산 유닛(MUFU)의 속도는 그대로 이거나 아주 느리게 향상되었습니다.
3) 결과: 이제 연산 성능보다는 메모리 트래픽과 Softmax 연산(지수 함수) 자체가 전체 성능을 결정짓는 핵심 병목이 되었습니다.

### 2.2.3 2-CTA 텐서 코어 모드

<p align = 'center'>
   <img width="1200" height="1000" alt="image" src="https://github.com/user-attachments/assets/b80c88bf-de22-4bf4-9e20-4e761ec68416" />
   <img width="714" height="483" alt="image" src="https://github.com/user-attachments/assets/5db2ee0c-b91c-44c5-92eb-4896a87a083e" />
</p>


1) 두 개의 CTA(Threadblock)가 한 쌍이 되어 하나의 MMA 연산을 협력해서 수행하는 모드입니다.
2) 이 모드를 사용하면 각 CTA가 필요한 데이터( $B$ 타일)의 절반만 로드하면 되므로, 공유 메모리 대역폭 소모를 절반으로 줄일 수 있습니다.


---

## 3. Algorithm

### 3.1 어텐션 순전파 (Forward Pass)

<p align = 'center'>
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/6c4a5613-b8c2-4927-99db-f84645047199" />
</p>

#### 루프라인 분석 (Roofline Analysis)

 * 지수 연산 유닛(Exp)과 공유 메모리(SMEM) 트래픽이 동일하거나 더 큰 병목
 * Blackwell 아키텍처에서는 행렬 연산(MMA)이 빨라짐

<p align = 'center'>
<img width="700" height="250" alt="image" src="https://github.com/user-attachments/assets/54c1f6b7-3153-413c-bb87-ba5ffb724e0b" />
</p>

#### 소프트웨어 지수 함수 에뮬레이션

* 느린 하드웨어 전용 유닛(MUFU) 대신, 일반 연산 유닛(FMA)을 사용하여 $2^x$를 수학적으로 근사
    * 3차 다항식만 써도 오차가 매우 작아서, 이를 BF16으로 변환(반올림)하는 순간 발생하는 손실(양자화 오차)이 다항식 자체의 계산 오차보다 훨씬 커지게 됩니다

#### 온라인 소프트맥스 재스케일링 생략

* 매 단계마다 수행하던 재스케일링을 조건부로 변경
* 새로운 최댓값( $m_j$ )과 이전 최댓값( $m_{j-1}$ )의 차이가 특정 임계값( $\tau$ )보다 클 때만 재스케일링을 수행하여 연산 횟수를 줄였습니다.
    * 문제 상황: 새로운 블록을 계산했는데, 이전 블록들보다 더 큰 최댓값이 나타나면 어떻게 될까요? 이전에 계산해둔 결과값( $O_{j-1}$ )들을 새로운 최댓값에 맞춰서 다시 조정해줘야 합니다. 이 과정을 재스케일링이라고 하며, 이때 벡터 곱셈 연산이 발생
    * FlashAttention-4는 새로운 최댓값( $m_j$ )과 이전 최댓값( $m_{j-1}$ )의 차이가 임계값( $\tau$ )보다 클 때만 재스케일링을 수행
    * 보통 $\tau = 8.0$ (즉, $log_2(256)$ )으로 설정하는데, 이는 값이 약 256배 이상 차이 날 때만 정밀하게 조정

#### TMEM 기반 파이프라인

* 텐서 메모리(TMEM)를 활용해 출력 타일을 저장함으로써 레지스터 압박을 줄이고, 한 타일이 MMA를 수행하는 동안 다른 타일은 소프트맥스를 계산하는 '핑퐁(Ping-pong)' 스케줄링을 구현했습니다.
    * 회색 블록(MMA)과 노란색/주황색 블록(Softmax)이 수직으로 겹쳐서 동시에 진행되는 것

### 3.2 어텐션 역전파 (Backward Pass)

* 역전파에서는 순전파보다 더 많은 5번의 MMA 연산이 발생하며, 공유 메모리 대역폭이 가장 큰 병목이 됩니다.

#### 1. 연산 횟수 비교: 2회 vs 5회

* 순전파 (Forward Pass): 단 2회의 MMA 연산만 수행.

1) $S = \alpha QK^\top$ (Query와 Key의 유사도 계산).
2) $O = PV$ (Softmax 결과와 Value의 가중 합산).

#### 2. 역전파에서 수행되는 5가지 MMA 연산

1) $S$ 재계산 ($S^\top = KQ^\top$): 메모리 절약을 위해 순전파 때의 $S$를 저장하지 않으므로, 역전파 때 다시 계산해야 합니다.
2) $dP$ 계산 ($dP^\top = VdO^\top$): 출력의 변화( $dO$ )와 $V$를 곱해 Softmax 결과의 변화량을 구합니다.
3) $dV$ 계산 ($dV = P^\top dO$): $V$에 대한 그래디언트를 구하는 과정입니다.
4) $dQ$ 계산 ($dQ = \alpha dSK$): $Q$에 대한 그래디언트를 구하는 과정입니다.
5) $dK$ 계산 ($dK = \alpha dS^\top Q$): $K$에 대한 그래디언트를 구하는 과정입니다.

#### 2-CTA MMA 모드

##### 요약

* A, B 행렬 및 결과값(Accumulator)을 다음과 같이 나누어 처리

* 기존 $B$ 전체(예: 100MB)
    * $C = A \times B$
    * CTA 0은 $A_0 \times B$를 계산하고, CTA 1은 $A_1 \times B$를 계산

* 2-CTA
    * CTA 0은 $B$의 앞부분 절반(50MB)
    * CTA 1은 $B$의 뒷부분 절반(50MB)
    * CTA 0는 $B$의 뒷부분 CTA 1에서 Read 가능
    * CTA 1는 $B$의 앞부분 CTA 0에서 Read 가능
  

##### 설명

* Blackwell의 새로운 기능을 활용하여 두 개의 CTA(Threadblock)가 한 쌍으로 동작하게 했습니다. 이를 통해 각 CTA는 필요한 데이터( $B$ 타일)의 절반만 로드하면 되므로 공유 메모리 트래픽이 감소합니다.
* dQ 원자적 연산(Atomic Adds) 절반 감소: 2-CTA 모드에서 $dS$ 데이터를 분산 공유 메모리(DSMEM)로 교환하여 $dQ$를 계산함으로써, 전역 메모리에 결과를 쓸 때 발생하는 비싼 Atomic 연산 횟수를 1-CTA 방식 대비 절반으로 줄였습니다.
    * $dQ$를 구하는 연산( $dQ = \alpha dSK$ )은 KV 시퀀스 차원을 따라 결과값을 계속 더해나가는 리덕션(Reduction) 과정
    * 기존 1-CTA 방식에서는 여러 CTA가 전역 메모리(GMEM)의 동일한 $dQ$ 위치에 동시에 값을 쓰려 하기 때문에, 데이터 오염을 막기 위해 비싼 **원자적 연산(Atomic Add)**을 사용
    * CTA 0과 CTA 1은 서로가 가진 $dS$ 타일의 절반을 분산 공유 메모리(DSMEM)를 통해 맞바꿉니다.
    * 타일 재구조화 (Repacking): 데이터를 교환한 후, 각 CTA는 이제 전체 행( $M$ ) 중 자신이 담당하는 절반의 행($M/2$)에 대해서만 모든 열( $2N$ )에 대한 리덕션을 수행
    * CTA 0: $dQ$의 상단 $M/2$ 행만 계산 및 기록 CTA 1: $dQ$의 하단 $M/2$ 행만 계산 및 기록
        * M: Query의 tiling, 128 or 256, 결과물인 $O$(Output)와 $dQ$의 행 크기를 결정 (Sub Sequence Length, 128, 256, 512)
        * N: Key, Value의 Tiling, 128 $S = \alpha QK^\top$ (Sequence Length, 100K, 300K)
        * MMA 연산 횟수: Blackwell의 MMA(행렬 연산) 지침은 $128 \times 128$ 크기를 기본으로 처리하므로, $M \times N$ 크기의 결과물을 만들기 위해 $\lceil M/128 \rceil \times \lceil N/128 \rceil$번의 연산이 필요함을 계산하는 척도가 됨
* 결정론적(Deterministic) 모드: 훈련의 재현성을 위해 세마포어 락(Semaphore lock)을 사용한 순차적 연산 모드를 제공하며, 성능 저하를 최소화하기 위한 최적의 CTA 처리 순서를 설계했습니다.

### 3.3 스케줄링 및 최적화 (Scheduling)

* LPT (Longest-Processing-Time-first): 연산 시간이 가장 길 것으로 예상되는 타일을 SM(Streaming Multiprocessor)에 먼저 할당하여 전체 작업 완료 시간을 단축했습니다.
    * Casual Masking 윗 부분말고 아랫부분 할당, 더 많은 연산
* Causal Masking 최적화: 마스킹으로 인해 삼각형 모양으로 줄어든 작업량을 효율적으로 배분하여, 기존 방식 대비 4~14%의 성능 향상을 달성했습니다.
    * 역순 처리하여 최적화 + L2 Cache swizzling으로 최적화
    * L2캐시 용량 넘지 않도록 일정한 세션으로 나눈다 + 섹션 내 순회 
* 가변 시퀀스 길이 (Varlen): 배치마다 시퀀스 길이가 다른 경우, 전처리 커널을 통해 작업량을 미리 계산하고 정렬하여 SM 간의 부하를 균등하게 맞췄습니다.

---

## 4. Language and Framework

### 1. CuTe-DSL 및 Python 기반 구현

* 탈 CUDA C++: FlashAttention-4는 CUDA C++ 구성 요소 없이 전체를 Python에 내장된 CuTe-DSL로 작성했습니다.
* 컴파일 과정: Python 소스 코드를 입력받아 중간 단계인 PTX로 낮춘 뒤, 최종적으로 GPU 어셈블리 코드(SASS)를 생성하는 방식을 사용합니다.

### 2. 압도적인 컴파일 속도 향상 (20~30배)

* 기존의 문제: 과거 FlashAttention 버전들은 복잡한 C++ 템플릿 메타프로그래밍 때문에 컴파일 시간이 매우 길어 개발 효율을 떨어뜨렸습니다.
* JIT 컴파일의 도입: Python 기반의 적시(Just-In-Time) 컴파일 방식을 도입하여, FlashAttention-3 대비 컴파일 속도를 20~30배 끌어올렸습니다.
* 수치 비교: 단일 커널 기준, 순전파는 55초에서 2.5초로, 역전파는 45초에서 1.4초로 컴파일 시간이 대폭 단축되었습니다.

### 3. 하위 수준 제어와 생산성의 조화

* 표현력 유지: CUTLASS C++과 동등한 수준의 추상화를 제공하여 저수준 GPU 프로그래밍의 세밀한 제어 능력을 그대로 유지합니다.
* 탈출구(Escape Hatch): 프레임워크가 지원하지 않는 특수 기능이 필요한 경우, PTX(저수준 지침)를 직접 사용할 수 있는 통로를 열어두어 하드웨어 성능을 끝까지 활용할 수 있게 했습니다.

### 4. 개발 장벽 완화 및 모듈화

* 연구 접근성 향상: 깊은 C++ 메타프로그래밍 지식이 없어도 수개월 정도의 GPU 프로그래밍 경험만 있다면 새로운 어텐션 변종을 직접 구현할 수 있을 만큼 진입 장벽을 낮췄습니다.
* 구성 가능한 프리미티브: 블록 희소 패턴(Block-sparse), 마스킹 전략, 스케줄링 등을 독립적이고 조합 가능한 구성 요소(Primitives)로 분리했습니다. 이를 통해 연구자들은 핵심 프레임워크를 수정하지 않고도 새로운 아이디어를 빠르게 프로토타이핑할 수 있습니다.


---

## 5. Empirical Evaluation

### 1. 평가 개요 및 실험 설정

* 목표: FlashAttention-4의 효율성을 다양한 오픈소스 및 벤더(Vendor) 제공 라이브러리와 비교 평가합니다.
* 비교 대상(Baselines): PyTorch 표준 구현, FlashAttention-2, Triton(B200 전용 지침 사용), Gluon, 그리고 NVIDIA가 B200에 최적화해 내놓은 cuDNN 9.13 등을 포함합니다.
* 환경: NVIDIA B200 GPU에서 BF16 정밀도를 사용하여 측정했습니다.
* 데이터 조건: 시퀀스 길이를 1k에서 32k까지 변화시켰으며, 배치 사이즈는 전체 토큰 수가 32k가 되도록 조정했습니다. 또한 DeepSeek V3에서 사용하는 (192, 128) 헤드 차원 설정도 포함되었습니다.

### 2. 주요 성능 결과 (핵심 수치)

<p align = 'center'>
<img width="1073" height="399" alt="image" src="https://github.com/user-attachments/assets/836b6d95-55fa-4fc6-b609-59556e07bd54" />
</p>

* 속도 향상: cuDNN 9.13 대비 최대 1.3배, Triton 대비 최대 2.7배의 성능 향상을 달성했습니다.
* 연산 처리량: 최대 1613 TFLOPs/s에 도달했으며, 이는 B200 GPU 이론적 최대 성능의 약 71%에 해당하는 높은 수치입니다.

### 3. 세부 분석 결과

#### 순전파 (Forward Pass)

<p align ='center'>
<img width="1060" height="430" alt="image" src="https://github.com/user-attachments/assets/20c1891b-1e2b-4c6c-8836-b8eba930db32" />
</p>

* 4k 이상의 중간 및 긴 시퀀스 길이에서 모든 베이스라인을 지속적으로 압도합니다.
* 특히 인과적 마스킹(Causal Masking)이 적용된 경우 성능 이득이 더 컸는데, 이는 논문에서 제안한 LPT(최장 작업 우선) 스케줄러가 작업 불균형을 효과적으로 해결했기 때문입니다.

#### 역전파 (Backward Pass)

<p align ='center'>
<img width="1075" height="352" alt="image" src="https://github.com/user-attachments/assets/d6bc6f0e-2e9d-4294-b8b2-5e8254c9835f" />
</p>

* 긴 시퀀스 길이와 인과적 마스킹 상황에서 일관된 속도 향상을 보여줍니다.이는 공유 메모리 트래픽을 줄여주는 2-CTA 역전파 기술이 실제로 효과적임을 입증합니다.
* 결정론적 역전파 (Deterministic Backward Pass) 재현성을 위해 연산 순서를 고정하는 결정론적 모드에서도 뛰어난 성능을 보였습니다.
    * 동일한 입력값에 대해 실행할 때마다 비트 단위까지 완벽하게 동일한 결과값을 내놓는 계산 방식을 의미
    * 결정론적 역전파가 속도가 75%수준으로 올라왔기 때문에(Flashattention-4), 디버깅 및 재현에 용이
* 정교한 스위즐링(Swizzling)과 SPT/LPT 스케줄링을 통해, 최적화되지 않은 일반 역전파(1-CTA) 성능의 75% 수준까지 속도를 끌어올렸습니다.


---

## 6. Discussion and Conclusion

### 1. 하드웨어 비대칭성 해결 (Addressing Asymmetric Scaling)

* 텐서 코어가 너무 빠르기 때문에 이제 성능의 지배적인 병목은 행렬 연산 자체가 아니라, 공유 메모리 트래픽과 지수 연산 처리량으로 옮겨갔음을 강조합니다. 

### 2. 제안된 기술적 해결책의 요약

* 파이프라인 재설계: 완전히 비동기적인 MMA(Matrix Multiply-Accumulate) 연산을 중심으로 파이프라인을 재설계하여, 더 큰 타일 크기를 사용하면서도 소프트맥스 연산과 행렬 연산을 겹쳐서 처리(Overlap)했습니다.
* 연산 부하 감소: 소프트웨어적으로 에뮬레이션된 지수 함수와 조건부 소프트맥스 재스케일링 기술을 도입하여 비-행렬 연산(non-matmul)의 비중을 줄였습니다.
* 메모리 효율화: 텐서 메모리(TMEM)와 2-CTA MMA 모드를 적극적으로 활용하여 공유 메모리 트래픽을 줄였습니다.
* 원자적 연산 최적화: 특히 2-CTA 모드를 통해 전역 메모리의 원자적 누적(Global atomic accumulation) 횟수를 절반으로 줄여 역전파 속도를 높였습니다. 

### 3. 구현 프레임워크의 혁신

* FlashAttention-4는 C++ 템플릿 기반의 복잡한 커널 대신, Python에 내장된 CuTe-DSL만을 사용하여 구현되었습니다.
* 이를 통해 하위 수준의 하드웨어 제어 능력을 유지하면서도, 기존 C++ 방식보다 20~30배 빠른 컴파일 속도를 달성하여 개발 생산성을 획기적으로 높였습니다. 

### 4. 결론 및 향후 전망

* 이 논문에서 제안한 알고리즘은 Blackwell GPU에 최적화되어 있지만, 연산 능력이 메모리나 특수 유닛의 속도를 앞지르는 비대칭성은 다른 가속기에서도 공통적으로 나타나는 추세입니다.
* 따라서 FlashAttention-4의 설계 원칙은 향후 등장할 다양한 차세대 가속기 하드웨어로 확장되어 적용될 수 있을 것으로 기대하며 마무리됩니다.

---
