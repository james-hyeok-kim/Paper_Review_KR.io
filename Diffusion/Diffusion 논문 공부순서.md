확산 모델(Diffusion Model)을 처음부터 깊이 있게 이해하고 싶으시다면, 몇 가지 핵심 논문을 순서대로 읽어보시는 것이 좋습니다. 모델의 기본 아이디어부터 최신 기술까지의 흐름을 파악할 수 있도록 리스트를 정리해 드립니다.

1. 기초 및 핵심 개념 정립 (The Foundations)
이 논문들은 확산 모델의 근본적인 아이디어를 제시하고, 오늘날 모든 확산 모델의 기반을 마련했습니다.

📜 Deep Unsupervised Learning using Nonequilibrium Thermodynamics
저자/연도: Jascha Sohl-Dickstein, et al. (2015)

중요성: 확산 모델의 시초. 🔥 열역학의 비평형 통계 물리학에서 영감을 받아, 데이터에 점진적으로 노이즈를 추가했다가(forward process) 다시 제거하는(reverse process) 과정을 통해 이미지를 생성하는 아이디어를 최초로 제안했습니다. 확산 모델을 이해하기 위한 가장 첫 번째 필독 논문입니다.

📜 Denoising Diffusion Probabilistic Models (DDPM)
저자/연도: Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)

중요성: 현대 확산 모델의 부흥을 이끈 논문. 🚀 위 2015년 논문의 아이디어를 단순화하고 성능을 대폭 개선하여 GAN과 비견될 만한 고품질 이미지 생성을 처음으로 보여주었습니다. Loss 함수를 단순화하여 구현이 쉬워졌고, 이후 대부분의 확산 모델 연구는 이 논문을 기반으로 발전했습니다.

2. 성능 및 속도 개선 (Improvements & Speed)
DDPM의 성공 이후, 생성 속도를 높이고 샘플링 방식을 개선하려는 연구들이 활발하게 진행되었습니다.

📜 Denoising Diffusion Implicit Models (DDIM)
저자/연도: Jiaming Song, Chenlin Meng, Stefano Ermon (2020)

중요성: 빠른 샘플링 속도. ⚡ DDPM은 수천 번의 스텝을 거쳐야 하나의 이미지를 생성할 수 있어 매우 느렸습니다. DDIM은 이 과정을 non-Markovian으로 재정의하여, 훨씬 적은 스텝(예: 20~50 스텝)으로도 고품질 이미지를 생성할 수 있게 했습니다. 결정론적(deterministic) 샘플링이 가능해져 잠재 공간(latent space)을 다루기 용이해졌습니다.

📜 Score-Based Generative Modeling through Stochastic Differential Equations (Score SDE)
저자/연도: Yang Song, Jascha Sohl-Dickstein, et al. (2021)

중요성: 이론적 통합. DDPM과 Score-based 모델(NCSN)이 사실상 같은 프레임워크임을 확률적 미분 방정식(SDE) 관점에서 통합하고 증명했습니다. 모델의 이론적 배경을 더 깊이 이해하는 데 중요한 역할을 합니다.

3. 고해상도 및 조건부 생성 (High-Resolution & Control)
최신 확산 모델의 핵심인 고해상도 이미지 생성과 텍스트 등의 조건을 제어하는 기술에 대한 논문들입니다.

📜 Diffusion Models Beat GANs on Image Synthesis
저자/연도: Prafulla Dhariwal, Alex Nichol (2021)

중요성: GAN을 뛰어넘는 성능. 아키텍처 개선과 Classifier Guidance라는 기법을 도입하여, 이미지 품질 평가 척도인 FID score에서 처음으로 GAN을 능가하는 성능을 보여주었습니다. 이를 통해 확산 모델이 이미지 생성 분야의 새로운 표준이 될 수 있음을 증명했습니다.

📜 Classifier-Free Diffusion Guidance
저자/연도: Jonathan Ho, Tim Salimans (2022)

중요성: 유연한 조건부 생성. 이전의 Classifier Guidance는 원하는 결과(예: "강아지" 이미지)를 얻기 위해 별도의 분류기(classifier) 모델이 필요했습니다. 이 논문은 별도의 모델 없이, 확산 모델 자체만으로 조건부 생성을 가능하게 하는 Classifier-Free Guidance를 제안했습니다. 현재 Text-to-Image 모델의 핵심 기술 중 하나입니다.

📜 High-Resolution Image Synthesis with Latent Diffusion Models (LDM)
저자/연도: Robin Rombach, Andreas Blattmann, et al. (2022)

중요성: Stable Diffusion의 탄생. 🌟 고화질 이미지의 픽셀 공간에서 직접 확산을 진행하는 대신, VAE(Variational Autoencoder)를 이용해 저차원의 **잠재 공간(Latent Space)**에서 확산을 진행하는 방식을 제안했습니다. 이를 통해 계산량을 획기적으로 줄여, 적은 컴퓨팅 자원으로도 고해상도 이미지를 빠르고 효율적으로 생성할 수 있게 되었습니다.

이 리스트를 순서대로 따라가시면 확산 모델의 역사와 핵심 아이디어를 체계적으로 이해하는 데 큰 도움이 될 것입니다.
