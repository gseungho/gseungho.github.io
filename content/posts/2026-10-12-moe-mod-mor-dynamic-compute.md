---
title: "MoE · MoD · MoR: 모든 token에 같은 계산을 써야 하는가"
date: "2026-10-12"
category: "논문리뷰"
subcategory: "Deep Learning/Dynamic Compute"
tags: ["논문리뷰", "Mixture-of-Experts", "Mixture-of-Depths", "Mixture-of-Recursions"]
description: "모든 token에 같은 계산을 쓰지 않는 세 가지 방법을 비교한다. MoE는 token마다 expert를 고르고 MoD는 일부 token만 block에 통과시키며 MoR은 token마다 block을 반복하는 횟수를 정한다. 각 routing 방식을 어떻게 학습하고 어떤 제약이 따르는지도 함께 살펴본다."
---

dense Transformer는 예측하기 쉬운 자리든 어려운 자리든 모든 token을 똑같이 다룬다. "the" 같은 흔한 단어와 처음 보는 고유명사가 모두 같은 weight로 같은 수의 layer를 거친다.

> **모든 token에 같은 계산량이 필요한가?**

세 논문은 모두 **router**라는 작은 부품을 붙여 이 질문에 답하되 router에 맡기는 결정만 서로 다르다. MoE는 어떤 weight를 쓸지를, MoD는 계산을 할지 말지를, MoR은 몇 번 반복할지를 정하게 한다.

세 논문을 이어 보면 앞 논문이 내놓은 해법이 다음 논문에서는 풀어야 할 문제가 된다. MoE는 처음 제안된 Shazeer et al. (2017)보다 Transformer에 맞게 단순화한 Switch Transformer(Fedus et al., 2022)를 기준으로 본다.

## MoE·MoD·MoR의 routing 비교

| | token마다 달라지는 것 | token당 FLOPs | 전체 파라미터 |
|---|---|---|---|
| Dense | 없음 | 고정 | 기준 |
| MoE | 어떤 expert FFN을 쓰는가 | **고정** | 크게 늘어남 |
| MoD | block을 통과하는가, 건너뛰는가 | **가변** | 그대로 |
| MoR | block을 몇 번 반복하는가 | **가변** | 약 $1/N_r$로 줄어듦 |

MoE는 token마다 다른 expert를 쓰지만 파라미터만 늘어날 뿐 token 하나에 드는 계산량은 dense 모델과 같다. token마다 계산량이 실제로 달라지는 모델은 MoD와 MoR부터다. MoR은 block을 재사용하기까지 해서 파라미터가 오히려 줄어든다.

MoE가 처음 질문에 직접 답을 내놓는 것은 아니다. 다만 뒤의 두 논문이 가져다 쓰는 router와 그 학습 방법이 MoE에서 나왔으므로 먼저 살펴보자.

## MoE가 token마다 다른 FFN을 고르는 방식

### 여러 expert 중 하나만 선택하는 Switch Transformer

<Fig src="/images/posts/moe-mod-mor-dynamic-compute/sw_fig2.png" alt="Switch Transformer encoder block">Figure 2. FFN 자리에 expert 네 개와 router가 들어간 Switch Transformer block (Fedus et al., 2022)</Fig>

Transformer block에서 attention은 그대로 두고 FFN 하나를 $N$개로 늘려 각각을 **expert**라고 부른다. router는 token마다 보낼 expert를 고른다.

2017년의 MoE는 token 하나를 top-$k$개의 expert에 보내 출력을 가중합했다. Switch Transformer는 이를 $k = 1$로 줄여 token 하나를 expert **하나**에만 보낸다. 그림에서 "More"는 확률 0.65인 expert로, "Parameters"는 0.8인 다른 expert로 갔다.

expert가 몇 개든 token 하나가 FFN을 한 번만 통과하므로 token당 계산량은 그대로 유지된다.

### router는 어떻게 학습되는가

router는 행렬 $W_r \in \mathbb{R}^{N \times d}$ 하나로 동작한다. token 표현 $x \in \mathbb{R}^d$에 이 행렬을 곱해 expert 수만큼 logit을 만들고 softmax를 씌운 것이 논문의 식 (1)이다.

$$h(x) = W_r\, x, \qquad p_i(x) = \frac{e^{h(x)_i}}{\sum_{j=1}^{N} e^{h(x)_j}}$$

$p_i(x)$는 "이 token에 expert $i$가 얼마나 맞는가"를 나타내는 값이다. 그림의 0.65와 0.8이 바로 이 값이고 Switch는 그중 가장 큰 값 하나를 고른다.

$$i^\ast = \arg\max_i\, p_i(x), \qquad y = p_{i^\ast}(x)\, E_{i^\ast}(x)$$

눈여겨볼 점은 출력에 $p_{i^\ast}(x)$를 **다시 곱한다**는 것이다. $\arg\max$는 미분할 수 없으므로 어느 expert를 골랐다는 사실만으로는 gradient가 생기지 않는다. 고른 expert의 출력에 그 확률을 곱해 두어야 loss에서 router까지 gradient가 전달된다. 논문은 이 값을 gate value라고 부르며 router를 미분 가능하게 만드는 것도 이 값이라고 적어 둔다.

gradient를 직접 써 보면 router가 무엇을 배우는지 보인다.

$$\frac{\partial \mathcal{L}}{\partial p_{i^\ast}} = \Big\langle \frac{\partial \mathcal{L}}{\partial y},\; E_{i^\ast}(x) \Big\rangle$$

고른 expert의 출력 $E_{i^\ast}(x)$가 loss를 줄이는 방향이면 이 내적이 음수가 되어 gradient descent가 $p_{i^\ast}$를 키우고 반대면 줄인다. $p$는 softmax를 거친 값이라 선택되지 않은 expert의 logit도 함께 움직인다.

$$\frac{\partial p_{i^\ast}}{\partial h_j} = p_{i^\ast}\,(\delta_{i^\ast j} - p_j)$$

$j \neq i^\ast$여도 이 편미분은 0이 아니므로 $W_r$의 모든 행이 gradient를 받는다. router에 "이 token은 몇 번 expert"라는 정답을 따로 주지 않아도 language model loss만으로 어디로 보낼지를 배운다.

Shazeer et al. (2017)은 expert를 둘 이상 비교해야 routing이 학습될 거라고 추측했다. 반면 Switch는 $k = 1$로도 품질이 유지된다는 점을 실험으로 보였다.

### expert capacity를 넘긴 token의 처리

<Fig src="/images/posts/moe-mod-mor-dynamic-compute/sw_fig3.png" alt="capacity factor에 따른 token routing">Figure 3. capacity factor 1.0(왼쪽)과 1.5(오른쪽). 빨간 점선이 넘쳐서 계산되지 않는 token (Fedus et al., 2022)</Fig>

TPU와 GPU에서는 tensor 크기가 미리 정해져 있어야 하지만 어느 expert에 token이 몇 개 갈지는 router가 그때그때 정한다. 이 제약을 맞추기 위해 논문은 식 (3)처럼 expert마다 받을 수 있는 token 수를 고정한다.

$$\text{expert capacity} = \frac{\text{tokens per batch}}{\text{number of experts}} \times \text{capacity factor}$$

예를 들어 batch에 token이 1,024개, expert가 8개, capacity factor가 1.25라면 expert 하나가 160개까지 받는다. 균등하게 나뉘면 128개씩이니 25%의 여유를 둔 셈이다.

router가 한 expert에 그보다 많이 보내면 token이 넘친다. 논문은 이 경우를 이렇게 처리한다.

> 계산을 건너뛰고 token 표현을 residual connection으로 다음 layer에 그대로 넘긴다.

**capacity를 넘긴 token은 그 layer의 FFN을 통과하지 않고 residual 경로로 지나간다.** Switch 입장에서는 피하고 싶은 부작용이다. capacity factor를 키우면 덜 넘치지만 빈 자리에 쓰는 계산과 메모리가 늘어난다. 논문은 실험 전반에서 넘치는 비율이 보통 1% 미만이었다고 보고한다. 뒤에서 볼 MoD는 바로 이 건너뛰기 동작을 설계의 중심으로 가져간다.

### load balancing을 위한 보조 loss

router를 language model loss만으로 학습시키면 몇몇 expert만 계속 고르게 된다. 많이 뽑힌 expert가 더 빨리 좋아지고 좋아진 만큼 더 뽑히기 때문이다. 이를 막으려 논문의 식 (4)부터 (6)인 보조 loss를 더한다.

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \sum_{i=1}^{N} f_i\, P_i$$

$$f_i = \frac{1}{T}\sum_{x \in \mathcal{B}} \mathbb{1}\{\arg\max\, p(x) = i\}, \qquad P_i = \frac{1}{T}\sum_{x \in \mathcal{B}} p_i(x)$$

$T$는 batch $\mathcal{B}$의 token 수다. $f_i$는 expert $i$로 **실제로 간** token의 비율이고 $P_i$는 router가 expert $i$에 준 확률의 **평균**이다.

둘이 왜 따로 필요한지는 미분 가능성을 보면 알 수 있다. $f_i$는 $\arg\max$로 세는 값이라 gradient가 없고 $P_i$만 미분된다. 그러니 이 loss는 $P_i$를 $f_i$만큼의 세기로 눌러 token이 많이 몰린 expert일수록 확률을 더 세게 깎는다.

균등할 때 최소가 된다는 것도 확인할 수 있다. $f_i$와 $P_i$가 거의 같이 움직인다고 보면 합은 $\sum_i P_i^2$이고 $\sum_i P_i = 1$이라는 조건에서 코시-슈바르츠 부등식으로

$$\sum_{i=1}^{N} P_i^2 \;\ge\; \frac{1}{N}$$

이며 등호는 모든 $P_i$가 $1/N$일 때 성립한다. 식 앞에 $N$을 곱해 둔 것도 균등할 때 합이 $1/N$이니 expert 수와 무관하게 loss 크기를 일정하게 맞추기 위해서다.

계수는 $\alpha = 10^{-2}$다. 논문은 $10^{-1}$부터 $10^{-5}$까지 시험했을 때 이 값이 본래의 loss를 방해하지 않으면서 빠르게 균형을 잡았다고 적는다.

### 같은 품질에 도달하는 학습 시간 비교

<Fig src="/images/posts/moe-mod-mor-dynamic-compute/sw_fig5.png" alt="학습 시간에 따른 Switch-Base와 T5-Base의 품질" size="md">Figure 5. 가로축이 학습 시간, 세로축이 negative log perplexity (Fedus et al., 2022)</Fig>

example당 FLOPs를 맞춘 비교에서 expert 64개를 쓴 Switch-Base는 T5-Base가 도달한 품질에 **7분의 1 시간** 만에 도달한다. 같은 32 TPUv3 core와 같은 계산량 조건에서 달라진 것은 파라미터 수뿐이다.

Switch가 보인 결론은 **파라미터 수와 계산량은 따로 움직일 수 있다**는 점이다. expert를 늘려 모델이 커져도 token당 계산은 그대로 유지되기 때문이다.

다만 처음 질문에 비추어 보면 여전히 달라진 점이 없다. 쉬운 token과 어려운 token 모두 **같은 양의 계산**을 수행하며, 계산량을 token마다 다르게 쓰는 접근은 다음 논문에서 시작된다.

## MoD는 일부 token만 block에 통과시킨다

### block을 건너뛰는 방식과 early-exit의 차이

<Fig src="/images/posts/moe-mod-mor-dynamic-compute/mod_fig1.png" alt="Mixture-of-Depths 구조와 routing 결과">Figure 1. 왼쪽은 block 구조, 오른쪽 위는 학습된 모델의 routing 결과, 오른쪽 아래는 vanilla·early-exit과의 비교 (Raposo et al., 2024)</Fig>

Mixture-of-Depths(Raposo et al., 2024)는 MoE의 router를 가져오되 선택지를 **block 통과와 residual 건너뛰기** 둘로 줄인다. MoE와 달리 적용 대상도 FFN만이 아니라 **self-attention과 MLP를 합친 block 전체**다.

그림 오른쪽 위는 학습된 모델의 실제 routing 결과로 가로가 sequence이고 세로가 layer다. 보라색이 block을 통과한 token이고 주황색이 건너뛴 token이다.

아래쪽의 early-exit과 비교하면 구조적 차이가 드러난다. early-exit은 token마다 멈출 시점을 정해 그 뒤 layer를 전부 건너뛰므로 한 번 빠지면 돌아오지 않는다. 반면 MoD의 token은 중간 block을 건너뛰었다가 **뒤쪽 block에 다시 들어올 수 있다.** 논문은 이 점을 early-exit 방식과 구별되는 특징으로 든다.

### 처리할 token 수를 미리 정하는 top-C routing

어떤 token을 통과시킬지는 top-$k$로 정한다. layer $l$에서 router가 token마다 scalar 점수를 낸다.

$$r_i^l = w_\theta^\top x_i^l$$

sequence 길이를 $S$, 통과시킬 token 수를 $C$라고 하면 block의 출력은 논문의 식 (1)로 정해진다.

$$
x_i^{l+1} =
\begin{cases}
r_i^l\, f_i(\tilde{X}^l) + x_i^l, & r_i^l > P_\beta(R^l) \\[1mm]
x_i^l, & \text{otherwise}
\end{cases}
$$

- $R^l$은 그 layer의 router 점수 전체이고 $P_\beta(R^l)$는 그중 $\beta$ 분위수다($\beta = 1 - C/S$)
- $\tilde{X}^l$은 점수가 그 문턱을 넘은 token들, 즉 상위 $C$개다
- $f$는 self-attention과 그 뒤의 MLP다

논문의 최적 설정으로 숫자를 넣어 보면 $S = 2048$, $C = 256$이라 $\beta = 0.875$다. 2,048개 중 256개(12.5%)만 block에 들어가고 나머지 1,792개는 $x_i^{l+1} = x_i^l$로 그대로 지나간다.

이 식에서 block 출력에 $r_i^l$을 곱하는 이유는 Switch의 gate value와 같다. 이 곱셈이 있어야 router 점수가 gradient 경로에 놓여 language model loss로 학습된다.

한편 **$C$는 학습 전에 사람이 정한다.** 정확히 $C$개만 통과하므로 계산 그래프와 tensor 크기가 학습 내내 고정된다. 달라지는 것은 "어떤 token이 그 $C$개에 드는가"뿐이다. 논문은 ACT(Adaptive Computation Time)처럼 입력마다 계산량이 달라지는 방식이 동적 계산 그래프 때문에 지금의 하드웨어와 잘 맞지 않는다고 본다. MoD는 예산을 미리 고정해 이 문제를 피한다.

### token-choice와 expert-choice의 차이

<Fig src="/images/posts/moe-mod-mor-dynamic-compute/mod_fig2.png" alt="token-choice와 expert-choice routing">Figure 2. 왼쪽부터 token-choice, expert-choice, MoD가 쓰는 expert-choice (Raposo et al., 2024)</Fig>

routing 방식은 둘로 나뉜다.

- token-choice에서는 token이 자기가 갈 경로를 고르며 Switch가 이 방식을 쓴다. 한 경로로 몰리면 넘치는 token이 생겨 balancing loss가 필요했다
- expert-choice에서는 경로가 token을 고른다. 각 경로가 점수 상위 $k$개를 가져가므로 **항상 정확히 $k$개**가 들어간다

MoD는 expert-choice를 쓴다. 경로가 block 하나뿐이라 top-$k$ 한 번으로 "통과"와 "건너뜀"이 나뉘며 균형도 구조적으로 보장되므로 balancing loss가 필요 없다.

Switch에서는 넘친 token이 **어쩌다** 계산을 건너뛰었던 반면 MoD는 그 건너뛰기를 의도된 **설계**로 삼았다. 논문은 이 대비를 MoE와 MoD를 합친 모델(MoDE)로 직접 실험했다. 아무 계산도 하지 않는 "no-op" expert를 하나 둔 쪽이 expert capacity를 줄여 token이 넘치게 둔 쪽보다 분명히 좋았다. 저자들의 해석은 이렇다.

> token이 residual 경로를 **스스로 고르도록 배우는 것**과, 원하는 expert가 있었는데 **밀려나는 것**은
> 다르다.

### 생성 중에는 알 수 없는 top-k 순위

expert-choice에는 대가가 따른다. top-$k$는 **sequence 전체**의 점수를 놓고 순위를 매긴다. 문장 전체가 주어지는 학습 때와 달리 생성할 때는 네 번째 token을 만드는 시점에 다섯 번째 이후가 아직 존재하지 않는다. 어떤 token이 상위 $k$에 드는지는 뒤에 올 token의 점수에 달려 있어 미리 알 수 없으며 논문은 이를 두고 top-$k$가 non-causal하다고 표현한다.

첫 번째 해결책은 **보조 loss**다. router 출력을 logit으로, "top-$k$에 들었는가"를 정답으로 삼아 binary cross-entropy를 건다. 그러면 뽑힌 token의 점수는 sigmoid 기준 0.5 위로, 아닌 token은 0.5 아래로 밀린다. 생성할 때는 순위를 매길 필요 없이 점수가 0.5를 넘는지만 보면 되며 language modeling 성능은 0.2에서 0.3% 정도 손해를 본다.

두 번째 해결책은 **작은 predictor**다. router와 같은 입력을 받는 작은 MLP를 따로 두고 "이 token이 top-$k$에 들 것인가"를 맞히게 한다. stop-gradient를 걸어 두어 본래의 loss에는 영향이 없고 문제 자체가 쉬워 학습 초반에 정확도가 97%를 넘고 곧 99%에 이른다.

실제로 학습된 모델의 router 점수 분포를 보면 보조 loss가 의도한 대로 약 12.5%가 0.5 위, 87.5%가 아래에 놓인다.

### block 생략 비율에 따른 성능 비교

<Fig src="/images/posts/moe-mod-mor-dynamic-compute/mod_fig3.png" alt="MoD hyperparameter 탐색 결과">Figure 3. 왼쪽은 같은 FLOPs로 학습한 MoD 변형들의 loss(회색 상자 안이 최적 baseline보다 좋은 모델), 오른쪽은 일부 모델의 학습 곡선 (Raposo et al., 2024)</Fig>

저자들은 routing을 모든 block에 걸지 한 block 걸러 걸지, capacity를 12.5%부터 95%까지 어디에 둘지 비교했다. 가장 좋았던 설정은 **한 block 걸러 routing하고 routing block의 capacity를 12.5%로 두는 것**이다. 절반의 block에서는 token 8개 중 7개가 계산 없이 지나간다.

다만 routing은 한 block 걸러 걸어야 성능이 나온다는 조건이 붙는다. 논문의 정리는 "전체 token이 참여하는 block이 자주 있기만 하면 capacity는 과감하게 줄여도 된다"이다.

이 설정으로 학습한 모델의 실험 결과를 수치로 비교해 보자.

- 그림의 3번 모델(220M)은 같은 FLOPs 예산의 최적 baseline(역시 220M)과 **같은 loss**에 도달하면서 한 step이 **66% 빠르다.** forward pass당 FLOPs가 그만큼 적기 때문이다
- router 대신 **무작위로** 고르면 baseline보다도 크게 나빠진다

계산을 줄이는 일 자체보다 **무엇을 줄일지를 학습했기 때문에** 손해가 없다는 점이 두 번째 결과에서 확인된다.

### 계산을 더 받는 token

<Fig src="/images/posts/moe-mod-mor-dynamic-compute/mod_fig5.png" alt="MoD의 routing 결정과 router 점수 분포">Figure 5. 왼쪽은 layer별 routing 결정, 오른쪽은 router 점수의 분포 (Raposo et al., 2024)</Fig>

그럼 router는 어떤 token에 계산을 쓸까?

실제 routing에서는 token마다 성향이 갈린다. 어떤 token은 깊이 방향으로 거의 모든 block에 들어가는 반면 어떤 token은 건너뛸 수 있을 때마다 건너뛴다. 저자들은 block에 자주 들어가는 token일수록 **출력 예측의 entropy가 높았다**고 보고한다. 다음 token을 맞히기 어려운 자리에 계산을 더 쓴다는 해석이다.

다만 논문 스스로 "preliminary analyses"라고 단서를 달았다. 수치나 그림으로 제시된 결과가 아니어서 확인된 사실로 받아들이기는 이르다.

## block 반복 횟수를 token마다 정하는 MoR

### block 하나를 반복해서 쓰기

Mixture-of-Recursions(Bae et al., 2025)는 Recursive Transformer 구조를 바탕으로 한다. 이는 layer마다 다른 weight를 두는 대신 **같은 layer 묶음을 $N_r$번 반복**해 통과시키는 방식으로 깊이는 그대로인데 고유 파라미터는 줄어든다.

논문의 360M 설정을 예로 들어 계산해 보자. 원래 모델은 32 layer에 non-embedding 파라미터가 315M인데 첫 layer와 마지막 layer만 따로 두고 가운데를 공유하면 아래 표처럼 파라미터가 줄어든다.

| $N_r$ | layer 구성 | 고유 layer 수 | 파라미터 |
|---|---|---|---|
| 1 (vanilla) | 32 | 32 | 315M |
| 2 | 1 + 15 × 2 + 1 | 17 | 167M |
| 3 | 1 + 10 × 3 + 1 | 12 | 118M |

layer 하나가 약 9.8M이니 17 layer면 167M, 12 layer면 118M으로 맞아떨어진다.

### 네 가지 weight 공유 방식 비교

어느 layer를 어떻게 공유할지는 정해진 답이 없어 논문은 9 layer를 3번 반복하는 기준으로 네 가지 방식을 비교한다.

- Cycle은 (0 1 2)(0 1 2)(0 1 2)처럼 묶음 전체를 다시 돈다
- Sequence는 (0 0 0)(1 1 1)(2 2 2)처럼 한 layer를 연달아 쓰고 다음으로 넘어간다
- Middle-Cycle과 Middle-Sequence는 위와 같되 첫 layer와 마지막 layer는 공유하지 않는다

| 방식 | validation NLL ↓ |
|---|---|
| Cycle | 2.8487 |
| Sequence | 2.9467 |
| **Middle-Cycle** | **2.8295** |
| Middle-Sequence | 2.9303 |

360M 모델, $N_r = 2$, 10B token 학습 기준이다. NLL은 정답 token의 음의 로그우도로 validation loss와 같은 뜻이다.

Sequence 계열은 성능이 뚜렷하게 나쁜데 같은 변환을 연달아 적용하면 비슷한 feature가 반복될 뿐이라는 게 논문의 설명이다. Middle-Cycle의 NLL이 가장 낮은 데에는 입력을 받는 layer와 출력을 내는 layer에 고유한 변환이 필요하다는 해석이 붙는다.

다만 Middle-Cycle이 항상 앞서는 것은 아니다. 135M 모델의 $N_r = 2$에서는 Cycle이 3.0071로 Middle-Cycle의 3.0330보다 좋았다. 논문 역시 최적의 공유 방식은 아직 풀리지 않은 문제라고 적고 있으며 이후 실험은 전부 Middle-Cycle을 쓴다.

### router로 token별 반복 횟수 정하기

<Fig src="/images/posts/moe-mod-mor-dynamic-compute/mor_fig1.png" alt="Mixture-of-Recursions 개요">Figure 1. 왼쪽은 반복 단계 하나, 가운데는 전체 구조, 오른쪽은 token별 반복 횟수의 예 (Bae et al., 2025)</Fig>

Recursive Transformer는 모든 token을 똑같이 $N_r$번 돌리지만 MoR은 여기에 router를 붙여 **token마다 반복 횟수를 다르게** 조절한다. MoD의 결정이 "할까 말까"였다면 MoR은 "몇 번 할까"다.

그림 오른쪽의 실제 routing 예시에서는 색이 진한 칸이 계산이 일어난 자리다. "defensively"가 "def"와 "ensively"로 쪼개졌을 때 논문은 단어의 뒤쪽 조각처럼 예측하기 쉬운 경우에는 적은 단계로 끝난다고 설명한다.

### expert-choice 대 token-choice

<Fig src="/images/posts/moe-mod-mor-dynamic-compute/mor_fig2.png" alt="MoR의 routing 방식과 KV caching 방식">Figure 2. (a) expert-choice routing, (b) token-choice routing, (c) KV caching 방식 두 가지 (Bae et al., 2025)</Fig>

MoR은 MoD에서 본 두 routing 방식을 반복 횟수에 적용해 둘 다 실험한다.

**expert-choice**는 반복 단계 하나하나를 expert로 보아 단계 $r$의 router가 점수를 내고 식 (2.1)처럼 상위 $k$개만 그 단계의 block을 통과시킨다.

$$
\mathcal{H}_t^{r+1} =
\begin{cases}
g_t^r\, f(\mathcal{H}_t^r, \Phi') + \mathcal{H}_t^r, & g_t^r > P_\beta(G^r) \\[1mm]
\mathcal{H}_t^r, & \text{otherwise}
\end{cases}
$$

$g_t^r = \mathcal{G}(\theta_r^\top \mathcal{H}_t^r)$는 단계 $r$에서 token $t$가 받은 점수이고 $\mathcal{G}$는 sigmoid 같은 activation 함수다. $\Phi'$는 공유되는 block의 파라미터다. 식의 모양은 MoD와 같지만 **이전 단계에서 뽑힌 token만 다음 단계의 후보가 된다**는 점이 다르다. 한 번 빠진 token은 돌아오지 않으므로 단계가 올라갈수록 남는 token이 줄어든다.

각 단계에 얼마나 남길지도 미리 정해 단계 $j$가 처리하는 비율을 $(N_r - j + 1)/N_r$로 둔다. $N_r = 3$이면 세 단계가 차례로 $3/3$, $2/3$, $1/3$을 처리한다.

**token-choice**는 처음에 한 번만 정한다. 첫 hidden state로 $N_r$개 expert의 점수를 내고 가장 높은 expert $i$를 골라 그 token을 $i$번 반복시킨다.

| $N_r = 3$, 같은 FLOPs | validation NLL ↓ | few-shot 평균 ↑ |
|---|---|---|
| expert-choice | **2.7925** | **42.6** |
| token-choice | 2.9163 | 40.0 |

few-shot 평균은 LAMBADA, HellaSwag, PIQA, WinoGrande, ARC, MMLU 여섯 benchmark의 정확도 평균이다. 표에서 보듯 expert-choice가 뚜렷하게 좋다.

두 방식은 앞 논문들의 문제를 하나씩 물려받는다. token-choice는 Switch가 겪은 쏠림 문제 때문에 balancing loss가 필요하다. expert-choice는 MoD처럼 top-$k$가 non-causal하므로 보조 loss나 보조 router가 있어야 한다.

### 반복을 일찍 끝낸 token의 KV cache

생성할 때는 이전 token들의 key와 value를 cache해 두고 쓴다. 그런데 어떤 token이 첫 단계에서 빠지면 둘째·셋째 단계의 key와 value가 **계산된 적이 없어** 뒤에 오는 token이 참조할 대상이 비어 있다.

논문은 이를 해결하는 두 가지 caching 방식을 비교한다.

- recursion-wise caching은 각 단계에서 그 단계까지 살아남은 token의 KV만 저장하고 attention도 그 token들끼리만 계산한다
- recursive sharing은 모든 token이 첫 단계는 통과하므로 첫 단계의 KV를 뒤 단계에서 재사용한다

비용을 따져 보면 recursion-wise caching에서 KV cache 크기는 각 단계의 처리 비율을 평균한 값이 되어 차이가 분명해진다.

$$\frac{1}{N_r}\sum_{j=1}^{N_r}\frac{N_r - j + 1}{N_r} = \frac{N_r + 1}{2N_r}$$

$N_r = 3$일 때 이 비율은 $2/3$이다. 단계마다 query도 key도 $k$개로 줄어들어 attention 계산은 vanilla의 $(k/N_{\text{ctx}})^2$이 된다. 반면 recursive sharing은 key가 전체 길이를 유지하므로 $k/N_{\text{ctx}}$만큼만 줄어든다.

| expert-choice, $N_r = 3$ | validation NLL ↓ | few-shot 평균 ↑ |
|---|---|---|
| recursion-wise caching | **2.7925** | **42.6** |
| recursive sharing | 2.7983 | 41.9 |

성능 역시 recursion-wise caching이 조금 나아서 본 실험은 이 방식을 쓴다.

### 파라미터 수와 학습 비용을 맞춘 성능 비교

논문의 Table 3에서 주요 행을 옮겨 보았다.

| 모델 | $N_r$ | 파라미터 | 학습 FLOPs | 학습 token | NLL ↓ | few-shot ↑ |
|---|---|---|---|---|---|---|
| Vanilla | - | 315M | 16.5e18 | 20B | 2.7824 | 42.3 |
| Recursive | 2 | 167M | 16.5e18 | 20B | 2.8079 | 42.6 |
| **MoR** | 2 | 167M | 16.5e18 | **27B** | **2.7511** | **43.1** |
| MoR | 3 | 118M | 16.5e18 | 30B | 2.7925 | 42.6 |
| MoR | 4 | 98M | 16.5e18 | 30B | 2.8204 | 41.6 |
| MoR (token 수 고정) | 2 | 167M | **12.3e18** | 20B | 2.7749 | 42.9 |

같은 학습 FLOPs(16.5e18)에서 $N_r = 2$인 MoR은 파라미터가 절반 수준(167M 대 315M)인데도 validation loss가 더 낮고 few-shot 평균도 높다. 이는 MoR이 token당 계산이 적어 같은 FLOPs 안에서 **27B token**을 학습해 20B를 본 vanilla보다 데이터를 35% 더 봤기 때문이다.

학습 token 수를 20B로 맞춘 마지막 행을 보면 MoR은 FLOPs를 **25% 덜 쓰고도** NLL이 2.7749로 vanilla의 2.7824보다 낮다. 논문은 이 설정에서 학습 시간이 19%, 최대 메모리 사용량이 25% 줄었다고 보고한다.

다만 표에서 **vanilla를 이기는 건 $N_r = 2$뿐이다.** $N_r = 3$은 NLL이 2.7925, $N_r = 4$는 2.8204로 데이터를 30B나 보고도 vanilla보다 높다. few-shot 평균은 $N_r = 3$에서 42.6으로 vanilla를 조금 앞서지만 $N_r = 4$에서는 41.6으로 뒤집힌다. 논문 본문도 이 구간은 "vanilla와 근소한 차이 안에 머문다"고만 표현한다.

반면 같은 $N_r$의 Recursive Transformer보다는 일관되게 좋다. $N_r = 2$에서 2.7511 대 2.8079의 차이를 보이며, block을 재사용하는 구조 안에서는 token마다 반복 횟수를 다르게 하는 쪽이 낫다는 것까지는 분명하다.

### 모델 규모별 성능과 속도 측정의 조건

<Fig src="/images/posts/moe-mod-mor-dynamic-compute/mor_fig3.png" alt="모델 크기별 validation loss">Figure 3. 135M, 360M, 730M, 1.7B 모델의 FLOPs 예산별 validation loss (Bae et al., 2025)</Fig>

이 실험은 반복을 세 번으로 고정해 모델 크기를 네 가지로 바꿔 가며 비교한 결과다. 가장 작은 135M에서는 MoR이 vanilla보다 나쁜데 논문은 재귀 구조의 capacity가 부족해서일 것으로 본다. 규모가 커지면 격차가 빠르게 줄고 FLOPs 예산이 작거나 중간일 때는 특히 360M부터 vanilla와 같거나 앞서는 경우가 많다.

생성 속도도 개선되어 360M 모델 기준으로 반복을 4번까지 늘린 MoR-4가 vanilla 대비 최대 **2.06배**의 처리량을 냈다. 일찍 빠진 token의 자리를 대기 중인 다른 요청으로 바로 채우는 continuous depth-wise batching 덕분이다.

다만 이 결과에는 본문이 아니라 부록에 적힌 단서가 따른다.

**1.7B에서는 vanilla가 앞선다.** 같은 FLOPs에서 MoR이 token을 더 봤는데도(26B 대 20B) few-shot 평균이 vanilla 48.9, MoR 48.4다. 저자들의 문장을 옮기면 이렇다.

> 이는 각 모델의 최적 scaling 정책에서 최적 token 수가 다르기 때문일 수 있지만, 현재의 MoR 구조가
> scaling에 적합하지 않다는 신호일 수도 있다.

**처리량 측정에서 KV cache 시간이 빠져 있다.** 부록 E에는 "구현 편의상 KV pair를 cache하고 update하는 데 드는 시간은 제외했다"고 적혀 있다. 엔지니어링으로 줄일 수 있는 부분이라는 이유다. 하지만 MoR의 KV cache는 단계마다 길이가 달라 다루기가 까다로우므로 실제 배포 속도는 2.06배보다 낮을 수 있다.

## Routing 방식마다 다른 제약

세 논문의 router 역할과 그에 따른 제약을 나란히 비교해 보자.

| | MoE (Switch) | MoD | MoR |
|---|---|---|---|
| router가 정하는 것 | 어떤 expert | 통과 / 건너뜀 | 반복 횟수 |
| token당 FLOPs | 고정 | **가변** | **가변** |
| 파라미터 | 크게 늘어남 | 그대로 | 줄어듦 |
| routing 방식 | token-choice | expert-choice | expert-choice |
| 그래서 생기는 문제 | load balance | top-$k$가 non-causal | KV cache |

표의 마지막 두 행처럼 MoE에서는 token이 expert를 고르기 때문에 한쪽으로 쏠리는 현상이 생긴다. 이를 막으려 balancing loss와 capacity를 넣었고 그 부작용으로 넘친 token이 계산 없이 지나갔다.

MoD는 그 부작용을 설계로 뒤집어 경로가 token을 고르게 함으로써 쏠림을 구조적으로 없앴다. 대신 top-$k$가 미래 token을 보게 되는 문제가 생겨 보조 loss와 predictor로 메웠다.

MoR은 MoD의 routing을 깊이가 아니라 반복 횟수에 적용했다. 그러자 token마다 지나온 단계 수가 달라져 KV cache가 어긋났고 단계별로 cache를 따로 두는 방법이 나왔다.

MoR 논문의 한계 절에는 또 다른 제약이 이미 적혀 있다. 보조 loss를 쓰면 뽑힌 token과 아닌 token의 router 출력이 거의 완전히 갈라져서 **학습이 끝난 뒤에는 top-$k$ 값을 바꾸기 어렵다.** 추론할 때 예산을 조절하고 싶어도 바꿀 수 없으므로 expert-choice가 준 "고정된 예산"이라는 장점이 다시 제약으로 돌아온다.

## 정리

- MoE는 파라미터 수를 계산량에서 떼어냈지만 token당 계산은 그대로 두었다
- MoD는 미리 정한 예산 안에서 token이 block을 건너뛰게 했다
- MoR은 block 하나를 재사용하면서 token마다 반복 횟수를 다르게 했다

처음 질문의 답은 "아니다"에 가깝다. MoD에서는 token의 87.5%가 절반의 block을 건너뛰어도 loss가 같았고 MoR은 FLOPs를 25% 줄이고도 더 낮은 loss를 냈다.

다만 각 결과가 확인된 조건은 다르다. "어려운 token에 계산을 더 쓴다"는 MoD의 관찰은 예비 분석이고 MoR의 우위는 $N_r = 2$와 중간 규모에서 확인된 것이며 1.7B에서는 뒤집혔다.

세 논문은 데이터와 모델, 평가 지표가 모두 달라 수치를 직접 비교할 수 없다. 나란히 비교할 대상은 성능 수치가 아니라 구조와 문제의식이다.

## 이어지는 글

- [GPT-1 · BERT · GPT-2 · GPT-3: 모델은 무슨 task인지 어디서 아는가](/posts/2026-10-12-gpt-bert-in-context-learning)
- [Attention Is All You Need: 설명이 항상 멈추는 자리들](/posts/2026-09-01-attention-is-all-you-need)
- [ViT 리뷰: inductive bias는 데이터로 대체할 수 있는가](/posts/2026-09-01-vit-an-image-is-worth-16x16-words)

## 참고문헌

- Shazeer, N. et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR.
- Fedus, W., Zoph, B., Shazeer, N. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR.
- Raposo, D. et al. (2024). *Mixture-of-Depths: Dynamically allocating compute in transformer-based language models*. arXiv:2404.02258.
- Bae, S. et al. (2025). *Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation*.
