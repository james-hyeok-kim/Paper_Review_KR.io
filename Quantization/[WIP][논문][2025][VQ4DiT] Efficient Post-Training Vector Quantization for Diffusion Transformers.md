# VQ4DiT: Efficient Post-Training Vector Quantization for Diffusion Transformers

저자 : Juncan Deng 1*, Shuaiting Li1*, Zeyu Wang 1, Hong Gu 2, Kedong Xu 2, Kejie Huang 1†

1Zhejiang University

2vivo Mobile Communication Co., Ltd

dengjuncan@zju.edu.cn, list@zju.edu.cn, wangzeyu2020@zju.edu.cn, guhong@vivo.com, xukedong@vivo.com,huangkejie@zju.edu.cn

출간 : Proceedings of the AAAI Conference on Artificial Intelligence, 2025

논문 : [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/33782)

---

## 1. Introduction

#### 1. 확산 트랜스포머(DiT)의 부상과 장점

* UNet $\rightarrow$ DiT
* 성능과 확장성

#### 2. 배포 시 발생하는 과제
* 막대한 자원 요구
* 엣지 디바이스의 한계

#### 3. 기존 양자화 방식의 한계
* 초저비트에서의 성능 저하
* 전통적 벡터 양자화(VQ)의 문제
    * 기존 VQ 방식은 '코드북(가중치 대표값)'만 보정하고 '할당(어떤 대표값을 쓸지 결정)'은 보정하지 않습니다.

#### 4. 제안된 해결책: VQ4DiT
* 핵심 혁신
    * 동시 보정: 코드북과 할당(assignments)을 동시에 보정하여 오류 누적을 방지합니다.
    * 데이터 프리(Zero-Data): 별도의 보정용 데이터셋 없이도 부동 소수점 모델과 유사한 성능을 낼 수 있는 '제로 데이터 및 블록 단위 보정' 전략을 사용합니다.
* 성능: DiT 가중치를 2비트 수준으로 압축하면서도 수용 가능한 이미지 품질을 유지하며, 20분에서 5시간 내에 양자화가 가능합니다.

<p align = 'center'>
<img width="900" height="450" alt="image" src="https://github.com/user-attachments/assets/803f3fad-3740-44ef-811f-2f25888272f0" />
</p>


---

## 2. Backgrounds and Related Works

### 모델 양자화 (Model Quantization)
* 균등 양자화(Uniform Quantization, UQ)의 한계: 2비트와 같은 초저비트에서는 가중치를 일정한 간격으로만 재구성할 수 있다는 제약 때문에 오차가 매우 커집니다.
* 벡터 양자화 (Vector Quantization, VQ): 가중치를 서브 벡터로 나누고 이를 코드북의 대표값(코드워드) 인덱스로 대체하는 방식입니다. UQ보다 훨씬 유연하여 동일 비트에서 양자화 오차가 더 작습니다.


### DiT를 위한 VQ의 도전 과제
* 코드북 크기의 트레이드오프: 코드워드의 수( $k$ )와 차원( $d$ )이 커지면 오차는 줄어들지만, 코드북이 차지하는 메모리와 클러스터링 시간이 늘어나는 문제가 있습니다.
* 미세 조정의 어려움: 기존 CNN 모델에서 사용하던 방식처럼 DiT 전체를 미세 조정(Fine-tuning)하는 것은 막대한 컴퓨팅 자원과 시간이 소요됩니다.
* 그래디언트 충돌 (핵심 문제): 동일한 인덱스에 할당된 서브 벡터들이 서로 다른 방향의 그래디언트(기울기)를 가질 경우, 코드북을 업데이트할 때 오차가 누적되어 최적의 결과를 얻지 못하게 됩니다.


<p align = 'center'>
<img width="1058" height="448" alt="image" src="https://github.com/user-attachments/assets/bab134a2-fce1-4ecc-9012-3c9153d00494" />
</p>


---

## 3. VQ4DiT

### VQ4DiT의 핵심 메커니즘
* 후보 할당 세트(Candidate Assignment Sets) 구성: 각 가중치 서브 벡터에 대해 유클리드 거리가 가장 가까운 상위 $n$개의 코드워드를 후보로 선정합니다.
* 동시 보정(Simultaneous Calibration): 코드북만 업데이트하는 대신, 소프트맥스(Softmax) 비율을 적용하여 후보 세트 내에서 어떤 할당이 최적인지를 코드북과 함께 동시에 학습합니다.
* 제로 데이터 및 블록 단위 보정: 별도의 보정 데이터셋 없이도, 원본(FP) 모델과 양자화 모델 간의 블록 단위 출력 오차를 최소화하는 방식으로 효율적인 보정을 수행합니다.

#### 1. 후보 할당 세트 구성 (Candidate Assignment Sets)
* 거리 계산: 각 가중치 서브 벡터( $w_{o,i/d}$ )와 코드북 내의 모든 코드워드 사이의 유클리드 거리를 계산합니다.
* 상위 n개 선택: 계산된 거리를 바탕으로 가장 가까운 상위 $n$개의 코드워드 인덱스를 추출하여 후보 할당 세트 $A_c$를 구성합니다.
* 수식 표현: 이를 수식으로 나타내면 $A_{c}=\{a_{o,i/d}\}_{n}=arg~min_{k}^{n}||w_{o,i/d}-c(k)||_{2}^{2} \quad(5)$ 와 같습니다.
* 가정: 연구진은 이 후보 세트 안에 해당 서브 벡터를 가장 잘 표현할 수 있는 최적의 할당(optimal assignment)이 포함되어 있다고 가정합니다.

#### 2. 동시 보정 (Simultaneous Calibration)
* 후보 세트가 정해지면, 어떤 코드워드가 실제로 가장 적합한지 결정하기 위해 비율(ratio) 개념을 도입합니다.
* 소프트맥스 비율( $R$ ): 후보 세트의 각 멤버에게 소프트맥스 함수를 통한 비율 $R$을 할당합니다.
* 초기화 및 가중 평균: 모든 비율은 초기에 $1/n$으로 균등하게 설정되며, 양자화된 가중치($\hat{W}$)는 후보 코드워드들의 가중 평균(weighted average)으로 재구성됩니다.
* 동시 업데이트: 보정 과정에서 코드북( $C$ )의 값뿐만 아니라 각 후보의 비율 ( $R$ )도 그래디언트를 통해 동시에 업데이트됩니다.
* 최적 할당 결정: 특정 비율이 임계값 이상으로 높아지면 해당 코드워드를 최적 할당으로 선택합니다. 이는 서로 다른 방향의 그래디언트가 충돌하여 코드북 업데이트가 방해받는 현상을 방지합니다.

$$R=\{r_{o,i/d}\}_{n}=\{\frac{e^{z_{o,i}/d,n}}{\sum_{j=1}^{n}e^{z_{o,i/d,j}}}\}_{n},\sum_{n}\{r_{o,i/d}\}_{n}=1; \quad(6)$$


#### 3. 제로 데이터 및 블록 단위 보정 (Zero-Data and Block-Wise Calibration)
* 이 과정은 방대한 ImageNet 데이터셋 없이도 효율적으로 모델을 최적화할 수 있게 해줍니다.
* 데이터 프리(Zero-Data): DiT 학습에 쓰이는 대규모 데이터셋을 사용하는 대신, 별도의 외부 데이터 없이 보정을 수행하는 전략을 사용합니다.
* 입력 생성: 초기 단계의 입력으로는 가우시안 노이즈( $\epsilon \sim \mathcal{N}(0,I)$ )를 사용하며, 이후 단계에서는 누적 오차로 인한 보정 붕괴(calibration collapse)를 막기 위해 원본(FP) 모델의 이전 블록 출력을 입력으로 사용합니다.
* 블록 단위 손실 함수( $\mathcal{L}_d$ ): 동일한 입력에 대해 원본 모델 블록( $d_{fp}^l$ )과 양자화 모델 블록( $d_{q}^l$ )의 출력 사이의 평균 제곱 오차(MSE)를 계산하여 최소화합니다.
* 비율 손실 함수( $\mathcal{L}_r$ ) 추가: 최적 할당을 더 빠르게 찾기 위해 비율 보정을 돕는 추가적인 손실 함수를 결합하여 최종 목적 함수를 구성합니다.


$$\mathcal{L}_{d}=\mathbb{E}_{x,y,d,t}[\sum_{l}||d_{fp}^{l}(z_{t},y,t,W)-d_{q}^{l}(z_{t},y,t,\hat{W})||_{2}^{2}] \quad(8)$$

* 동일한 입력($z_t, y, t$)에 대해 부동 소수점 모델 블록($d_{fp}^l$)과 양자화된 모델 블록($d_q^l$)의 출력값 사이의 **평균 제곱 오차(MSE)**를 계산

$$\mathcal{L}_{r}=\sum_{o,i/d,n}(1-|2\times\{r_{o,i/d}\}_{n}-1|)/(\frac{o\times i}{d}) \quad(9)$$

* 할당(Assignment)을 더 빠르게 찾기 위해 후보 할당 세트의 비율($R$)에 적용하는 보조 손실 함수
* 초기값인 0.5 근처에 있는 비율들을 0 또는 1에 가깝게 밀어내는 역할을 합니다

$$\mathcal{L}=\lambda_{d}\mathcal{L}_{d}+\lambda_{r}\mathcal{L}_{r} \quad(10)$$

* 구성: 출력 오차를 줄이는 $\mathcal{L}_d$와 최적 할당을 찾는 $\mathcal{L}_r$에 각각 가중치($\lambda$)를 곱해 더합니다.

### 3. 주요 성능 및 특징
* 압도적인 압축 효율: 가중치를 2비트 수준으로 정밀하게 양자화하면서도 이미지 생성 품질을 안정적으로 유지합니다.
* 빠른 실행 속도: 단일 NVIDIA A100 GPU에서 모델 설정에 따라 20분에서 5시간 내에 양자화를 완료할 수 있습니다.
* 성능 비교: 기존의 GPTQ, Q-DiT, RepQ-ViT와 같은 강력한 베이스라인들이 2비트에서 성능이 완전히 무너지는(collapse) 것과 달리, VQ4DiT는 높은 정밀도를 유지합니다.


---

## Appendix

#### 벡터 양자화(Vector Quantization, VQ)

* 모델의 가중치를 직접 숫자로 저장하는 대신, 대표값들의 집합인 코드북(Codebook)과 각 데이터가 어떤 대표값에 해당하는지를 나타내는 할당(Assignments) 지표로 분해하여 저장하는 압축 기술

* 작동 원리
    * 서브 벡터 분할: 가중치 행렬 $W$를 일정한 길이 $d$를 가진 행 단위의 서브 벡터( $w_{i,j}$ )들로 나눕니다.
    * 코드북 생성: K-평균(K-Means) 클러스터링 알고리즘 등을 사용하여 전체 서브 벡터들을 잘 대표할 수 있는 $k$개의 코드워드(Codewords)를 생성하고 이를 코드북 $C$에 저장합니다.
    * 인덱스 할당: 각 서브 벡터와 가장 유사한(유클리드 거리가 가장 가까운) 코드워드를 찾아 그 인덱스를 할당( $A$ )합니다.
    * 가중치 복원: 실제 연산 시에는 저장된 인덱스를 보고 코드북에서 해당 코드워드를 꺼내와 원래의 가중치 모양으로 복원($\hat{W} = C[A]$)하여 사용합니다.
 
* 전통적 방식의 한계 (본 논문의 지적)
    * 코드북만 보정: 기존 방식은 코드북의 값은 업데이트하지만, 한 번 정해진 할당(Assignments)은 변경하지 않습니다.


---


