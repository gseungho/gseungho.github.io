---
title: "Attention Is All You Need: 설명이 항상 멈추는 자리들"
date: "2026-09-01"
category: "논문리뷰"
subcategory: "Deep Learning/Foundations"
tags: ["논문리뷰", "Transformer", "Attention", "딥러닝"]
description: "Transformer를 공부하다 보면 설명이 멈추는 자리가 몇 군데 있다. 위치 인코딩을 임베딩에 더하는 이유, Q·K·V가 서로 다르게 학습되는 방식, attention scaling, residual connection과 LayerNorm의 역할을 질문별로 짚는다. 직접 계산한 결과와 논문이 제시하는 근거를 함께 본다."
---

Transformer를 공부하면서 구조 설명만으로 이해하기 어려운 부분이 있었다.

"위치 정보를 더해준다"라는 설명은 자주 보았지만 *더하면 단어 정보랑 섞이지 않을까*는 여전히 궁금했고 "$\sqrt{d_k}$로 나눈다"라는 식을 보고도 *왜 하필 제곱근인가*가 궁금했다. Q, K, V는 "질문·열쇠·값"이라는 비유만으로는 이해하기 어려웠다. 특히 *같은 loss로 세 행렬이 어떻게 서로 다르게 학습되는가*를 더 따져 보고 싶었다.

공부하면서 막혔던 질문을 중심으로 정리했다. 몇 가지는 직접 계산해 확인해보았고 논문에 이유가 제시되어 있지 않은 부분도 그대로 적었다.

<Fig src="/images/posts/attention-is-all-you-need/fig_architecture.png" alt="Transformer 전체 구조" size="sm">Figure 1. 왼쪽이 encoder, 오른쪽이 decoder (Vaswani et al., 2017)</Fig>

기준으로 삼는 모델은 원 논문의 base 모델이다. 설정은 $N=6$ layer, $d_{\text{model}}=512$, $d_{ff}=2048$, head $h=8$개, $d_k=d_v=64$, 파라미터 65M이다.

이 논문을 먼저 읽고 그다음에 ViT를 읽었다. 이어지는 글은 [ViT 리뷰: inductive bias는 데이터로 대체할 수 있는가](/posts/2026-09-01-vit-an-image-is-worth-16x16-words)에 있다.

## 위치 정보를 임베딩에 더하는 이유

### sin/cos로 위치를 표현하는 방법

$$
PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad
PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

식만 보면 $pos$와 $i$가 각각 무엇인지부터 헷갈린다.

- $pos$ = 문장 내 단어 위치 ($0, 1, 2, \dots$)
- $i$ = 차원 쌍의 인덱스 ($0 \dots 255$). $d=512$면 차원 0~511을 둘씩 묶어 $i$번째 쌍이 차원 $2i$(sin)와 $2i{+}1$(cos)을 채운다.

$w_i = 10000^{-2i/d}$로 두면 식이 $\sin(pos \cdot w_i)$, $\cos(pos \cdot w_i)$가 된다. **$w_i$는 그 차원 쌍의 각속도다.**

- $i=0$이면 $w_0 = 1$, 주기 $2\pi \approx 6.3$으로 빠르게 도는 시계가 된다.
- $i=255$면 $w \approx 1/9500$, 주기 약 62800으로 거의 안 도는 시계가 된다.

논문 표현으로는 파장이 $2\pi$부터 $10000 \cdot 2\pi$까지 **등비수열**을 이룬다. 즉 **512개 차원에 빠른 시계부터 느린 시계까지 골고루 깔아 둔 것**이다. 이진수에서 최하위 비트가 빠르게, 최상위 비트가 느리게 바뀌는 것과 같은 발상이다. 여러 해상도의 주기 함수를 겹쳐 위치마다 구별되는 벡터를 만든다.

참고로 **상수 10000을 선택한 이유는 논문에 제시되어 있지 않다.**

### sin과 cos를 쌍으로 쓰는 이유

덧셈정리를 적용하면 회전행렬이 나온다.

$$
\begin{bmatrix}PE_{(pos+k,\,2i)}\\PE_{(pos+k,\,2i+1)}\end{bmatrix}
=\begin{bmatrix}\cos k w_i & \sin k w_i\\ -\sin k w_i & \cos k w_i\end{bmatrix}
\begin{bmatrix}PE_{(pos,\,2i)}\\PE_{(pos,\,2i+1)}\end{bmatrix}
$$

이 행렬은 **오직 상대 거리 $k$에만 의존**하고 $pos$와는 무관하다. 따라서 위치 $pos+k$의 인코딩은 위치 $pos$ 인코딩의 선형변환으로 표현된다. 논문은 이 성질 덕분에 모델이 상대 위치를 학습하기 쉬울 것이라고 **"가정했다(hypothesized)"고** 적었다.

다만 이 성질만으로 attention이 상대 거리만 계산하도록 강제되는 것은 아니다.

실제 attention의 position-position 항은 $p_i M p_j^\top$ 형태이고 여기서 $M = W_Q W_K^\top$이다. 이 값이 상대 거리에만 의존하려면 **$M$이 모든 회전행렬과 교환 가능해야(commute)** 하는데 학습되는 일반 행렬이 그럴 이유가 없다. 즉 **"가능성"이지 "보장"이 아니다.** 나중에 나온 RoPE가 바로 이 조건을 아키텍처 차원에서 강제한 연구다.

논문이 "hypothesized"라는 단어를 쓴 것도 이 때문일 것이다. 많은 설명 자료가 이 단어를 빼고 단정하는데 원문은 그렇게 말하지 않는다.

### 상대 거리별 코사인 유사도 계산

말로만 "가까운 위치는 비슷하다"고 하면 확인이 안 되니 직접 계산해보았다.

먼저 norm을 구해 보면 $\|p_{pos}\|^2 = \sum_i (\sin^2 + \cos^2) = d/2$이므로 **모든 위치의 norm이 $\sqrt{d/2}$로 동일**하다. $d=512$면 정확히 **16.0**이 된다. 위치와 무관하게 크기가 같기 때문에 코사인 유사도가 내적만으로 결정된다.

$$\cos\_sim(p_{pos},\,p_{pos+k}) = \frac{2}{d}\sum_i \cos(k\,w_i)$$

$d=512$로 계산한 결과다.

| 상대 거리 $k$ | 1 | 5 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|
| cos sim | 0.973 | 0.741 | 0.616 | 0.512 | 0.437 | 0.350 |

<Fig src="/images/posts/attention-is-all-you-need/fig_sinusoidal.png" alt="sinusoidal PE의 코사인 유사도">직접 계산한 sinusoidal PE: 왼쪽은 상대 거리 k에 따른 감소 곡선, 오른쪽은 위치 0~49의 유사도 행렬 (d=512)</Fig>

가까울수록 1에 가깝고 멀수록 감소한다. **의도한 성질이 실제로 나온다.**

다만 **완벽한 단조 감소는 아니다.** $k \approx 25 \sim 35$ 부근에 미세한 굴곡(ripple)이 있다. 가장 빠른 시계($w_0 = 1$, 주기 6.28)가 주기를 한 바퀴 돌면서 생기는 aliasing이다. 256개 주파수가 겹치면서 대부분 상쇄되기 때문에 전체 추세는 뚜렷하게 감소한다.

### 더하면 정보가 섞이지 않는가

내가 가장 오래 막혔던 질문이다. 100이라는 값을 보고 99+1인지 98+2인지 어떻게 알까?

**직관 자체는 정확하다.** 스칼라라면 구분할 수 없다. $D$차원에서 이야기가 달라지는 데는 두 가지 이유가 있다.

**① 선형연산에는 분배법칙이 성립한다.** $(a+b)W = aW + bW$이므로 attention score가 자동으로 네 항으로 쪼개진다.

$$
S_{ij} = \underbrace{(e_iW_Q)(e_jW_K)^\top}_{\text{content-content}}
+ \underbrace{(e_iW_Q)(p_jW_K)^\top}_{\text{content-position}}
+ \underbrace{(p_iW_Q)(e_jW_K)^\top}_{\text{position-content}}
+ \underbrace{(p_iW_Q)(p_jW_K)^\top}_{\text{position-position}}
$$

**모델이 $e$와 $p$를 따로 분리하지 않아도 내적이라는 연산 자체가 이처럼 네 항으로 쪼개진다.** $a$와 $b$를 몰라도 "$a+b$에 5를 곱하면?"에는 답할 수 있는 것과 같다.

**② 고차원에서 무작위 두 벡터는 거의 직교한다.** 코사인 유사도가 대략 $1/\sqrt{d}$인데 $d=512$면 0.044, $d=768$이면 0.036이다.

따라서 정확한 표현은 "복원한다"기보다 **"거의 직교한 방향에 겹쳐 저장해서 간섭이 거의 없다"고** 이해하는 편이다. 이 두 성질 덕분에 $e$와 $p$를 따로 저장할 필요가 없다.

### 절대 위치와 상대 위치가 전달되는 방식

둘 중 하나를 고르는 질문이 아니었다. **둘 다 접근 가능하고 보장의 강도가 다르다.**

| | 메커니즘 | 보장 정도 |
|---|---|---|
| **절대 위치** | $p_i$가 residual stream에 실려 모든 layer에 전달되고 FFN이 패턴을 탐지 | 강함: 구조적으로 항상 존재 |
| **상대 위치** | attention의 position-position 항 | **조건부**: $M$이 회전과 commute할 때만 |

앞선 네 항 중에서도 **position-position만 상대적**이고 나머지 cross term은 절대 위치에 의존한다. 이 불일치가 Transformer-XL 이후 relative position 계열 연구가 파고든 지점이다.

### 임베딩에 $\sqrt{d_{\text{model}}}$을 곱하는 이유

논문 3.4절에는 딱 한 문장만 나온다. "In the embedding layers, we multiply those weights by $\sqrt{d_{\text{model}}}$." 근거는 적혀 있지 않다.

단서는 **그 문장이 weight tying 선언 바로 뒤에 온다**는 점이다. 이 단서로 이유를 추론해보았다.

같은 행렬 $W$가 두 역할을 한다.

- output layer로 쓸 때는 평범한 linear라서 원소 표준편차가 $\approx 1/\sqrt{d}$여야 한다.
- 입력 임베딩으로 쓸 때는 한 행의 norm이 $\sqrt{d \cdot \frac{1}{d}} = 1$밖에 안 된다.

| | 임베딩 norm | PE norm | 비율 |
|---|---|---|---|
| 곱하기 전 | 1.0 | 16.0 | **1 : 16** |
| 곱한 후 | 22.6 | 16.0 | 1.4 : 1 |

곱하기 전에는 **위치 정보가 단어 정보를 16배로 압도**하므로 위의 4항 분해에서 content 항들이 묻혀 버린다. $\sqrt{d}$를 곱하면 둘이 비슷한 크기가 된다.

이 설명은 초기화 방식에 따라 달라진다. 임베딩이 $\mathcal{N}(0, 1/d)$로 초기화됐을 때만 성립한다. PyTorch `nn.Embedding` 기본값인 $\mathcal{N}(0,1)$이면 곱한 뒤 norm이 **511.6**이 되어 이번엔 반대로 PE가 묻힌다. **초기화 방식과 함께 확인해야 하는 부분**이고 구현체마다 이 부분이 다른 이유이기도 하다.

## Self-attention을 공부하며 생긴 질문

<Fig src="/images/posts/attention-is-all-you-need/fig_attention.png" alt="Scaled Dot-Product Attention과 Multi-Head Attention">Figure 2. 왼쪽이 Scaled Dot-Product Attention, 오른쪽이 Multi-Head Attention (Vaswani et al., 2017)</Fig>

### 같은 loss에서 갈라지는 Q, K, V

$W_Q$, $W_K$, $W_V$는 전부 같은 스칼라 loss 하나로 학습된다. 그런데 어떻게 각자 다른 역할을 맡게 될까?

**"같은 loss"와 "같은 gradient"는 다르다.** 셋은 계산 그래프에서 위치가 달라서 **미분식 자체가 다르다.**

$$
\frac{\partial L}{\partial W_V}\propto\sum_{i,j}\alpha_{ij}\,x_j^\top(\cdot)
\qquad \text{(softmax 미분을 안 거침)}
$$

$$
\frac{\partial L}{\partial W_Q}\propto\sum_{i,j}g_{ij}\,x_i^\top K_j,\qquad
\frac{\partial L}{\partial W_K}\propto\sum_{i,j}g_{ij}\,x_j^\top Q_i
$$

$W_V$는 V가 가중합에만 들어가므로 softmax 미분을 아예 거치지 않는다. $W_Q$와 $W_K$는 둘 다 거치지만 **미분식에 들어가는 행렬이 각각 $K$와 $Q$로 다르다.**

실제로 측정해보았다. $d{=}64$, $d_k{=}32$, $n{=}16$에서 seed 300개로 gradient 코사인 유사도의 절댓값을 구한 결과다.

| 쌍 | 평균 | 중앙값 |
|---|---|---|
| $W_Q$, $W_K$ | 0.045 | 0.039 |
| $W_Q$, $W_V$ | 0.047 | 0.041 |
| $W_K$, $W_V$ | 0.023 | 0.020 |
| *무작위 기준선* | *0.018* | - |

**셋 다 거의 직교한다.** 하나의 스칼라 loss에서 출발하지만 서로 무관한 방향으로 갱신된다. 랜덤 초기화가 대칭성을 깨고 구조적 차이가 셋을 계속 벌려 놓기 때문이다.

> 처음엔 시드 **하나**로만 측정해서 $W_Q$와 $W_K$는 0.02,
> $W_Q$와 $W_V$는 0.34가 나왔고 "Q와 K가 특히 더 직교하다"는 결론을 낼 뻔했다. 시드를
> 300개로 늘리니 **그 구분이 사라졌다.** 당시 세팅이 $d{=}6$, $d_k{=}4$, $n{=}3$으로 차원이
> 아주 작았던 게 원인이었다. 저차원일수록 코사인 유사도의 분산이 크다($1/\sqrt{D}$ 성질).
> 단일 실행 결과를 일반화하면 안 된다는 걸 직접 확인했다.

"질문한다·답한다·의미를 담는다"는 표현은 **사람이 붙인 해석**일 뿐 gradient descent가 그 의미를 이해하고 최적화하지는 않는다. 확실한 사실은 "구조적으로 다른 gradient를 받는다"까지다.

### $\sqrt{d_k}$로 나누는 이유

$q$와 $k$의 각 성분이 독립이고 평균 0, 분산 1이라고 가정하자(학습 초기의 근사다).

$$
\text{Var}(q\cdot k)=\text{Var}\Big(\sum_{i=1}^{d_k}q_ik_i\Big)
=\sum_i\text{Var}(q_ik_i)=d_k
$$

독립 확률변수를 더하면 **분산도 그대로 더해진다는 성질**을 쓴 것이며 $\text{Var}(q_ik_i)=E[q_i^2]E[k_i^2]=1$이다. 즉 **내적의 분산이 차원에 정비례**하고 표준편차는 $\sqrt{d_k}$다.

이게 왜 문제일까? softmax의 미분이 $\text{softmax}(z)_i(1-\text{softmax}(z)_i)$라서 출력이 0이나 1에 가까워지면 **gradient가 죽기** 때문이다.

key 20개를 두고 softmax를 적용했을 때 가장 큰 weight의 평균을 측정해보았다.

| $d_k$ | 4 | 32 | 128 | 512 |
|---|---|---|---|---|
| scaling 없음 | 0.401 | 0.769 | 0.887 | **0.943** |
| $\div\sqrt{d_k}$ | 0.208 | 0.216 | 0.217 | **0.217** |

<Fig src="/images/posts/attention-is-all-you-need/fig_sqrt_dk.png" alt="d_k에 따른 내적 분산과 softmax 붕괴">왼쪽: 내적의 표준편차가 정확히 $\sqrt{d_k}$를 따라감. 오른쪽: scaling 없이는 한 key가 확률을 독식하고 나누면 $d_k$와 무관하게 일정해진다</Fig>

scaling 없이 $d_k{=}512$면 **한 key가 94%를 가져가** 사실상 one-hot이 되고 gradient가 거의 0이 된다. 반면 $\sqrt{d_k}$로 나누면 $d_k$와 **무관하게** 0.21로 안정된다. 균등분포(0.05)보다는 뾰족하지만 확률이 한곳에 몰리는 현상은 줄어든다.

**왜 하필 제곱근인지도 여기서 나온다.** $d_k$로 나눴다면 분산이 $1/d_k$가 되어 이번엔 반대로 균등분포로 뭉개진다. **$\sqrt{d_k}$는 분산을 정확히 1로 맞추는 유일한 지수다.**

참고로 논문은 이 논증을 본문이 아닌 각주 4에 적어 두었고 표현도 "We suspect that..."이다.

이는 Transformer 전체를 관통하는 원칙이기도 하다. LayerNorm과 $\times\sqrt{d_{\text{model}}}$ 역시 결국 **분산을 1 근처로 유지한다**는 같은 이야기다.

### Head 수와 head별 차원의 관계

"8개 만드니까 조합이 다양해진다"는 설명은 방향은 맞지만 원리를 정확히 짚어야 한다. 이는 **"같은 계산을 8번 반복"하는 구조가 아니다.**

다양성은 **오직 head마다 $W_Q$, $W_K$, $W_V$가 독립이라는 데서 나오고** 그 다양성이 **두 층위**에서 동시에 생긴다.

- $W_Q$, $W_K$가 다름 → 어떤 단어에 주목할지($\alpha_{ij}$)가 달라진다.
- $W_V$가 다름 → 같은 단어를 봐도 거기서 무엇을 읽어낼지가 달라진다.

단일 head라면 "동사-목적어 관계"와 "주어-동사 수 일치"를 **하나의 가중합으로 타협**해야 한다.

하지만 많을수록 좋은 건 아니다. Table 3 (A)에서 $h \times d_k = 512$를 고정하고 head 수만 바꾼 결과다.

| $h$ | 1 | 4 | 8 (base) | 16 | 32 |
|---|---|---|---|---|---|
| $d_k = d_v$ | 512 | 128 | 64 | 32 | 16 |
| BLEU | 24.9 | 25.5 | **25.8** | **25.8** | 25.4 |

**$h=32$에서 꺾인다.** head 수가 늘면 다양성은 올라가지만 $d_k$가 줄어 head별 표현력이 떨어지는 trade-off다.

덧붙이면 학습된 head 중 상당수가 redundant해서 일부를 제거해도 성능이 거의 떨어지지 않는다는 후속 관찰이 있다. 출처를 확인하지 못해 단정하지는 않겠지만 **"다양성이 이론적으로 가능하다"와 "실제로 8개가 다 다른 역할을 한다"는 별개**라는 점은 기억해 둘 만하다.

## Residual connection과 LayerNorm의 역할

Figure 1에 "Add & Norm"이라고 한 덩어리로 그려져 있어 하나로 묶어 외우기 쉽지만 **둘은 서로 다른 문제를 풀기 때문에** 분리해서 봐야 한다.

### Residual connection이 gradient를 살리는 방식

$$\frac{\partial}{\partial x}\big(x+F(x)\big) = I + \frac{\partial F}{\partial x}$$

backprop은 layer마다 이 값을 **곱한다.** residual connection이 없으면 $\prod_\ell \partial F/\partial x$가 되어 각 항이 조금만 1보다 작아도 지수적으로 0에 수렴한다. 반면 residual connection이 있으면 $F$의 미분이 작아져도 **$I$가 남아서** 통로가 보장된다.

$L=32$, $\partial F/\partial x = 0.1$일 때의 계산 결과다.

| | 32 layer 통과 후 |
|---|---|
| residual connection 없음 | $10^{-32}$ |
| residual connection 있음 | $21.1$ |

다만 **$F$의 미분이 작아지는 것 자체는 막히지 않는다.** 바뀌는 것은 작은 값들이 최종 결과에 **결합되는 방식**이다.

- 곱셈 구조 $\prod \delta_\ell$에서는 항 하나만 작아도 전체가 지수적으로 붕괴한다.
- $\prod(1+\delta_\ell)$ 구조에서는 $\delta$가 0이어도 각 항이 최소 1이라 절대 0으로 수렴하지 않는다.

### LayerNorm으로 activation 크기 조절하기

residual connection은 gradient를 살리는 대신 **activation이 layer마다 커지는 부작용**을 만든다. $\delta > 0$이 반복되면 이번엔 값이 발산하는 쪽이 문제가 되므로 Norm이 매 block에서 다시 눌러 준다. **그래서 residual connection과 norm은 항상 세트다.**

### BatchNorm 대신 LayerNorm

NLP의 입력은 $(N, T, D)$이고 BatchNorm은 batch 축으로 통계를 내기 때문에 두 가지가 걸린다.

1. 문장 길이가 다르다. `<pad>` token이 통계를 오염시킨다. masking을 해도 위치 $t$가 클수록 실제 token이 있는 문장 수가 batch마다 달라져 **통계 신뢰도가 위치마다 달라진다.**
2. "같은 위치의 token"끼리는 의미상 서로 관계가 없다. 이미지의 "같은 채널"은 "같은 종류의 특징"이라 batch로 묶는 게 타당하지만 "3번째 단어"는 문장마다 완전히 다른 단어다.

LayerNorm은 **token 하나의 $D$개 feature만으로** 통계를 낸다. 그래서 batch 크기가 1이거나 추론에서 한 단어씩 넣어도 문제가 없다.

### Post-LN과 Pre-LN의 차이

원 논문은 $\text{LN}(x + F(x))$, 즉 **Post-LN**이다. 그런데 GPT-2와 ViT는 $x + F(\text{LN}(x))$, **Pre-LN**을 쓴다.

$$
\text{Post-LN: } J_{\text{LN}}\Big(I+\tfrac{\partial F}{\partial x}\Big)
\qquad\qquad \text{Pre-LN: } I+\delta_\ell
$$

**Post-LN은 $J_{\text{LN}}$이 괄호 바깥**에 있어서 항등 경로에도 LayerNorm의 미분이 곱해진다. 반면 **Pre-LN은 LN이 $\delta_\ell$ 안에만** 들어가므로 항등 경로에는 LayerNorm의 미분이 곱해지지 않는다.

$J_{\text{LN}} \approx 0.95$로 두고 layer 수만큼 곱해 보면 차이가 분명해진다.

| layer 수 | $J_{\text{LN}}^{L}$ |
|---|---|
| 6 layer (원 논문) | 0.735 |
| 32 layer (ViT-Huge) | 0.194 |

**원 논문이 6 layer라 Post-LN으로 버틴 것이다.** 수십 layer부터는 Pre-LN이 표준이 됐다. 그 대신 residual stream이 정규화 없이 누적되므로 **마지막에 LN을 한 번 더** 붙여야 한다.

여기서 쓴 $\delta_\ell$은 항등 경로를 뺀 **나머지 항 전부**를 줄여 적은 축약 기호다. 정확히는 $\delta_\ell = \frac{\partial F_\ell}{\partial(\text{LN})}\cdot J_{\text{LN}}$인 $d \times d$ **Jacobian 행렬**이다(위에서 스칼라처럼 쓴 건 직관용 단순화다). Pre-LN에서도 LN은 그대로 존재하지만 $\delta$ 안에 갇혀 있어 $I$를 건드리지 못한다.

ViT-Huge가 32 layer이기 때문에 이 이야기는 [ViT 리뷰](/posts/2026-09-01-vit-an-image-is-worth-16x16-words)에서 이어진다.

## FFN의 차원을 늘렸다 줄이는 이유

$512 \rightarrow 2048 \rightarrow 512$. 왜 굳이 늘렸다가 다시 줄일까?

**늘리는 이유는 표현력이다.** ReLU 뉴런 하나가 입력 공간의 "꺾임" 하나에 대응하므로 은닉 뉴런 수($d_{ff}$)가 많을수록 조합할 수 있는 조각별 선형함수가 정교해진다(universal approximation의 직관이다).

**줄이는 이유는 구조적 제약이다.** residual connection이 $x + \text{FFN}(x)$를 계산하려면 shape이 같아야 하기 때문이다.

역할 분담으로 보면 **attention은 가로세로 양방향으로 단어끼리 정보를 섞고**, **FFN은 가로 방향으로만 각 위치를 독립적으로 변환한다.**

확장비 4는 **경험값이고 이론적 최적이 아니다.** Table 3 (C)의 실험 결과다.

| $d_{ff}$ | 1024 | 2048 (base) | 4096 |
|---|---|---|---|
| BLEU | 25.4 | 25.8 | **26.2** |
| params | 53M | 65M | 90M |

$d_{ff}$를 키우면 성능은 올라가지만 파라미터도 같이 늘어난다. **4가 최적이라는 논증은 논문에 없다.**

## Softmax 전에 미래 token을 가리는 이유

decoder에서 미래 token을 가릴 때 아주 큰 음수로 대체해 $e^{-\infty} = 0$이 되게 한다. 그런데 왜 굳이 softmax **전**일까? 후에 0으로 만들면 안 될까?

**안 된다. softmax는 전체 합이 1이 되도록 정규화**하기 때문이다. softmax를 먼저 통과시키면 이미 **분모에 미래 token이 들어가** 있다. 나중에 지우면 남은 weight의 합이 1이 아니게 되고 미래 정보가 분모를 거쳐 결과에 스며든다.

추론할 때도 똑같이 masking하므로 단어를 더 뽑아도 **이전 결과가 바뀌지 않는다.**

## PPL과 BLEU가 보는 것은 다르다

**PPL**은 "평균 몇 가지 중에 헷갈렸나"를 뜻한다. 정답 문장 확률의 기하평균의 역수이며 작을수록 좋다. 100단어 중 균등하게 찍으면 PPL이 100이고 Cross-Entropy loss에 $\exp$를 취하면 PPL을 얻는다.

**BLEU**는 $BP\cdot\prod_{n=1}^{N}p_n^{w_n}$이고 보정이 세 가지 들어간다.

- $n$-gram은 unigram만 쓰면 순서를 뒤섞어도 점수가 같아지기 때문에 사용한다.
- clipping은 같은 단어를 반복해 점수를 부풀리지 못하도록 정답에 존재하는 개수까지만 인정한다.
- brevity penalty는 너무 짧게 번역해 정보를 날리는 걸 막는다. 길게 하는 건 어차피 precision에서 손해라 따로 벌점이 필요 없다.

Table 3 (D)를 보면 **label smoothing을 켜면 PPL은 나빠지는데 BLEU는 좋아진다.**

| label smoothing | PPL (dev) | BLEU (dev) |
|---|---|---|
| 0.0 | **4.67** | 25.3 |
| 0.1 (base) | 4.92 | **25.8** |
| 0.2 | 5.47 | 25.7 |

모델이 덜 확신하게 만들면 확률 예측의 품질(PPL)은 나빠지지만 번역 품질(BLEU)은 올라간다. **두 지표가 서로 다른 것을 보고 있음**이 드러난다.

## 설계 선택을 뒷받침하는 근거

설계 선택마다 논문에 제시된 근거의 강도가 다르다.

| 선택 | 논문의 근거 |
|---|---|
| $\div\sqrt{d_k}$ | **있음**: 분산이 커져 softmax gradient가 죽는다 (각주 4) |
| multi-head | **있음**: Table 3 (A) ablation |
| sinusoidal PE | **부분적**: 상대 위치의 선형 표현을 "가정", 학습형과 차이 없음을 Table 3 (E)로 확인 |
| FFN 확장비 4 | **없음**: 크기를 바꾼 실험은 있으나 4가 최적이라는 논증은 없음 |
| $\times\sqrt{d_{\text{model}}}$ | **없음**: 한 문장, 근거 없음 |
| Post-LN | **없음**: 이후 Pre-LN으로 뒤집힘 |
| 상수 10000 | **없음**: 값만 제시 |

"Transformer는 이렇게 설계되어 있다"를 전부 같은 확신으로 받아들이면 단순한 관행까지 원리로 착각하게 된다. 실제로 Post-LN은 몇 년 뒤에 뒤집혔다.

논문에서도 sinusoidal을 고른 이유를 성능 때문이 아니라 **"학습 때보다 긴 sequence로 외삽할 수 있을지도 모른다"는** 기대로 밝히고 있다. Table 3 (E)를 보면 learned PE와 BLEU가 25.7 대 25.8로 사실상 같기 때문이다.

## 공부하면서 바로잡은 오해

정리하면서 내가 틀렸던 것들도 남겨 둔다.

1. "더하면 정보가 섞인다"는 직관은 맞지만 고차원에서 벡터끼리 거의 직교하기 때문에 실제로는 문제가 되지 않는다.
2. sinusoidal의 회전행렬 성질이 상대 위치를 "보장"한다고 읽었으나 이는 $M$이 회전과 commute해야만 성립하는 조건부 성질이다.
3. 단일 seed 실험 결과를 일반화할 뻔했으나 gradient 코사인 유사도를 300개 seed로 다시 측정하니 결론이 바뀌었다.

2번과 3번이 특히 기억에 남는다. 둘 다 **"그럴듯한 설명"에서 멈추지 않고 한 번 더 따져보았을 때만** 드러나는 것들이었다.

이 논문을 읽고 나서 [ViT: An Image is Worth 16×16 Words](/posts/2026-09-01-vit-an-image-is-worth-16x16-words)를 읽었다. 여기서 다룬 Pre-LN, CLS token, learned positional embedding이 전부 그쪽에서 다시 나온다. 특히 **"위치 임베딩이 2D 구조를 스스로 배운다"는** 실험은 이 글에서 본 sinusoidal의 성질과 **같은 결과인데 출처가 정반대**라서 나란히 놓고 보면 재미있다.

---

**참고문헌**

- Vaswani et al., *Attention Is All You Need*, NIPS 2017
- Dosovitskiy et al., *An Image is Worth 16×16 Words*, ICLR 2021
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021

*(Figure 1, 2는 원논문에서 가져왔고 sinusoidal 유사도와 $\sqrt{d_k}$ 그림은 직접 계산해 그렸다. 본문의 측정값들도 모두 직접 재현한 것이다.)*
