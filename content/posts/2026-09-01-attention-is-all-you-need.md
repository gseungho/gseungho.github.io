---
title: "Attention Is All You Need — 설명이 항상 멈추는 자리들"
date: "2026-09-01"
category: "논문리뷰"
subcategory: "Deep Learning/Transformer"
tags: ["논문리뷰", "Transformer", "Attention", "딥러닝"]
description: "트랜스포머 구조 소개가 아니라, 공부하면서 실제로 막혔던 질문들로 논문을 다시 읽었습니다. 직접 계산해 검증한 수치와, 논문이 끝내 근거를 대지 않은 선택들까지."
---

트랜스포머 설명 자료는 넘칩니다. 그런데 읽다 보면 **거의 모든 자료가 같은 자리에서 멈춥니다.**

"위치 정보를 더해준다"까지는 어디에나 있는데 *더하면 단어 정보랑 섞이지 않나*에는 답이 없고,
"$\sqrt{d_k}$로 나눈다"는 있는데 *왜 하필 제곱근인가*는 없습니다. Q, K, V는 "질문·열쇠·값"이라는
비유로 넘어가고, 정작 *같은 loss에서 나온 세 행렬이 어떻게 다른 걸 배우는가*는 다루지 않습니다.

그래서 이 글은 구조 순서대로 가지 않습니다. **제가 실제로 막혔던 질문 순서로 갑니다.** 몇 개는
직접 계산해서 확인했고, 몇 개는 논문이 끝내 답을 안 해준다는 사실을 확인하는 것으로 끝났습니다.
후자도 그대로 적었습니다.

<figure className="my-6 text-center">
  <img src="/images/posts/attention-is-all-you-need/fig_architecture.png" alt="Transformer 전체 구조" className="rounded-lg mx-auto max-w-sm" />
  <figcaption className="mt-2 text-sm text-gray-500">Figure 1 — 왼쪽이 encoder, 오른쪽이 decoder (Vaswani et al., 2017)</figcaption>
</figure>

기본 설정만 먼저 깔아두겠습니다. base 모델은 $N=6$층, $d_{\text{model}}=512$, $d_{ff}=2048$,
head $h=8$개, $d_k=d_v=64$, 파라미터 65M입니다.

이 논문을 먼저 읽고 그다음에 ViT를 읽었습니다. 이어지는 글은
[ViT 리뷰 — inductive bias는 데이터로 대체할 수 있는가](/posts/2026-09-01-vit-an-image-is-worth-16x16-words)에
있습니다.

## 위치 정보를 왜 "더하는가"

### sin/cos가 하는 일은 시계 512개를 깔아두는 것이다

$$
PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad
PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

식만 보면 $pos$와 $i$가 각각 뭔지부터 헷갈립니다.

- $pos$ = 문장 내 단어 위치 ($0, 1, 2, \dots$)
- $i$ = **차원 쌍의 인덱스**입니다. $d=512$면 차원 0~511을 둘씩 묶어서, $i$번째 쌍이
  차원 $2i$(sin)와 $2i{+}1$(cos)을 채웁니다. 그래서 $i = 0 \dots 255$

$w_i = 10000^{-2i/d}$로 두면 식이 그냥 $\sin(pos \cdot w_i)$, $\cos(pos \cdot w_i)$가 됩니다.
**$w_i$는 그 차원 쌍의 각속도예요.**

- $i=0$: $w_0 = 1$, 주기 $2\pi \approx 6.3$ — 빠르게 도는 시계
- $i=255$: $w \approx 1/9500$, 주기 약 62800 — 거의 안 도는 시계

논문 표현으로는 파장이 $2\pi$부터 $10000 \cdot 2\pi$까지 **등비수열**을 이룹니다. 즉
**512개 차원에 빠른 시계부터 느린 시계까지 골고루 깔아둔 것**입니다. 이진수에서 최하위 비트가
빠르게, 최상위 비트가 느리게 바뀌는 것과 같은 발상이에요. 여러 해상도를 겹쳐서 위치마다 고유한
지문을 만듭니다.

참고로 **"왜 하필 10000인가"는 논문이 설명하지 않습니다.** 이 글에서 앞으로 이런 항목이 여러 번
나올 텐데, 그때마다 명시하겠습니다.

### 왜 하필 sin/cos "쌍"인가 — 그리고 여기 함정이 있다

덧셈정리를 적용하면 회전행렬이 나옵니다.

$$
\begin{bmatrix}PE_{(pos+k,\,2i)}\\PE_{(pos+k,\,2i+1)}\end{bmatrix}
=\begin{bmatrix}\cos k w_i & \sin k w_i\\ -\sin k w_i & \cos k w_i\end{bmatrix}
\begin{bmatrix}PE_{(pos,\,2i)}\\PE_{(pos,\,2i+1)}\end{bmatrix}
$$

이 행렬은 **오직 상대 거리 $k$에만 의존**하고 $pos$와는 무관합니다. 그래서 위치 $pos+k$의 인코딩이
위치 $pos$ 인코딩의 선형변환으로 표현되고, 논문은 이 성질 덕분에 모델이 상대 위치를 학습하기 쉬울
것이라고 **"가정했다(hypothesized)"** 고 씁니다.

**여기가 함정입니다.** 이건 "attention이 상대 거리만 계산하도록 강제된다"는 뜻이 아닙니다.

실제 attention의 위치–위치 항은 $p_i M p_j^\top$ 형태이고 여기서 $M = W_Q W_K^\top$입니다.
이 값이 상대 거리에만 의존하려면 **$M$이 모든 회전행렬과 교환 가능해야(commute)** 하는데,
학습되는 일반 행렬이 그럴 이유가 없습니다. 즉 **"가능성"이지 "보장"이 아닙니다.** 나중에 나온
RoPE가 바로 이 조건을 아키텍처 차원에서 강제한 연구고요.

논문이 "hypothesized"라는 단어를 쓴 것도 이 때문일 겁니다. 많은 설명 자료가 이 단어를 빼고
단정하는데, 원문은 그렇게 말하지 않습니다.

### 실제로 계산해보면 의도한 성질이 나오긴 한다

말로만 "가까운 위치는 비슷하다"고 하면 확인이 안 되니 직접 계산해봤습니다.

먼저 norm입니다. $\|p_{pos}\|^2 = \sum_i (\sin^2 + \cos^2) = d/2$이므로 **모든 위치의 norm이
$\sqrt{d/2}$로 동일**합니다. $d=512$면 정확히 **16.0**이에요. 위치와 무관하게 크기가 같다는 게
중요합니다. 그래서 코사인 유사도가 내적만으로 결정되고,

$$\cos\_sim(p_{pos},\,p_{pos+k}) = \frac{2}{d}\sum_i \cos(k\,w_i)$$

이 값을 $d=512$로 계산하면 이렇게 나옵니다.

| 상대 거리 $k$ | 1 | 5 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|
| cos sim | 0.973 | 0.741 | 0.616 | 0.512 | 0.437 | 0.350 |

<figure className="my-6 text-center">
  <img src="/images/posts/attention-is-all-you-need/fig_sinusoidal.png" alt="sinusoidal PE의 코사인 유사도" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">직접 계산한 sinusoidal PE — 왼쪽은 상대 거리 k에 따른 감소 곡선, 오른쪽은 위치 0~49의 유사도 행렬 (d=512)</figcaption>
</figure>

가까울수록 1에 가깝고 멀수록 감소합니다. **의도한 성질이 실제로 나옵니다.**

다만 **완벽한 단조 감소는 아닙니다.** $k \approx 25 \sim 35$ 부근에 미세한 굴곡(ripple)이 있어요.
가장 빠른 시계($w_0 = 1$, 주기 6.28)가 주기를 한 바퀴 돌면서 생기는 aliasing입니다. 256개
주파수가 겹치면서 대부분 상쇄되기 때문에 전체 추세는 뚜렷하게 감소하고요.

### 더하면 정보가 섞이지 않나

제가 가장 오래 막혔던 질문입니다. 100이라는 값을 보고 99+1인지 98+2인지 어떻게 아나요.

**직관 자체는 정확합니다.** 스칼라라면 구분할 수 없어요. $D$차원에서 얘기가 달라지는 이유가
두 가지입니다.

**① 선형연산은 분배됩니다.** $(a+b)W = aW + bW$이므로 attention score가 자동으로 네 항으로
쪼개집니다.

$$
S_{ij} = \underbrace{(e_iW_Q)(e_jW_K)^\top}_{\text{content–content}}
+ \underbrace{(e_iW_Q)(p_jW_K)^\top}_{\text{content–position}}
+ \underbrace{(p_iW_Q)(e_jW_K)^\top}_{\text{position–content}}
+ \underbrace{(p_iW_Q)(p_jW_K)^\top}_{\text{position–position}}
$$

핵심은 **모델이 $e$와 $p$를 분리해서 계산하는 게 아니라, 내적이라는 연산 자체가 그렇게
쪼개진다**는 점입니다. $a$와 $b$를 몰라도 "$a+b$에 5를 곱하면?"에는 답할 수 있는 것과 같아요.

**② 고차원에서 무작위 두 벡터는 거의 직교합니다.** 코사인 유사도가 대략 $1/\sqrt{d}$인데,
$d=512$면 **0.044**, $d=768$이면 **0.036**입니다.

그래서 정확한 표현은 "복원한다"가 아니라
**"거의 직교한 방향에 겹쳐 저장해서 간섭이 거의 없다"입니다.** 따라서 $e$와 $p$를 따로 저장할
필요가 **없습니다.**

### 그래서 모델이 아는 건 절대 위치인가, 상대 위치인가

either/or 질문이 아니었습니다. **둘 다 접근 가능하고, 보장의 강도가 다릅니다.**

| | 메커니즘 | 보장 정도 |
|---|---|---|
| **절대 위치** | $p_i$가 residual stream에 실려 모든 층에 전달되고, FFN이 패턴을 탐지 | 강함 — 구조적으로 항상 존재 |
| **상대 위치** | attention의 위치–위치 항 | **조건부** — $M$이 회전과 commute할 때만 |

게다가 위의 네 항 중 **position–position만 상대적**이고 나머지 cross term은 절대 위치에
의존합니다. 이 비일관성이 Transformer-XL 이후 relative position 계열 연구가 파고든 지점이고요.

### 임베딩에 √d_model을 곱하는 이유는 논문에 없다

논문 3.4절에 딱 한 문장 있습니다. "In the embedding layers, we multiply those weights by
$\sqrt{d_{\text{model}}}$." 근거는 없습니다.

단서는 **그 문장이 weight tying 선언 바로 뒤에 온다**는 것입니다. 아래는 제 추론입니다.

같은 행렬 $W$가 두 역할을 합니다.

- **출력층**으로 쓸 때: 평범한 linear라서 원소 표준편차가 $\approx 1/\sqrt{d}$여야 합니다
- **입력 임베딩**으로 쓸 때: 그러면 한 행의 norm이 $\sqrt{d \cdot \frac{1}{d}} = 1$밖에 안 됩니다

| | 임베딩 norm | PE norm | 비율 |
|---|---|---|---|
| 곱하기 전 | 1.0 | 16.0 | **1 : 16** |
| 곱한 후 | 22.6 | 16.0 | 1.4 : 1 |

곱하기 전에는 **위치 정보가 단어 정보를 16배로 압도**합니다. 위의 4항 분해에서 content 항들이
묻혀버려요. $\sqrt{d}$를 곱하면 둘이 비슷한 크기가 됩니다.

**그런데 함정이 있습니다.** 이 논리는 임베딩이 $\mathcal{N}(0, 1/d)$로 초기화됐을 때만 성립합니다.
PyTorch `nn.Embedding` 기본값인 $\mathcal{N}(0,1)$이면 곱한 뒤 norm이 **511.6**이 되어 이번엔
반대로 PE가 묻힙니다. **초기화와 세트로 봐야 하는 디테일**이고, 구현체마다 이 부분이 다른
이유이기도 합니다.

## Self-attention에서 자주 틀리는 것들

<figure className="my-6 text-center">
  <img src="/images/posts/attention-is-all-you-need/fig_attention.png" alt="Scaled Dot-Product Attention과 Multi-Head Attention" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">Figure 2 — 왼쪽이 Scaled Dot-Product Attention, 오른쪽이 Multi-Head Attention (Vaswani et al., 2017)</figcaption>
</figure>

### 같은 loss에서 나온 Q, K, V가 어떻게 갈라지나

$W_Q$, $W_K$, $W_V$는 전부 같은 스칼라 loss 하나로 학습됩니다. 그런데 어떻게 각자 다른 역할을
맡게 될까요.

**"같은 loss"와 "같은 gradient"는 다릅니다.** 셋은 계산 그래프에서 위치가 달라서 **미분식 자체가
다릅니다.**

$$
\frac{\partial L}{\partial W_V}\propto\sum_{i,j}\alpha_{ij}\,x_j^\top(\cdot)
\qquad \text{(softmax 미분을 안 거침)}
$$

$$
\frac{\partial L}{\partial W_Q}\propto\sum_{i,j}g_{ij}\,x_i^\top K_j,\qquad
\frac{\partial L}{\partial W_K}\propto\sum_{i,j}g_{ij}\,x_j^\top Q_i
$$

$W_V$는 V가 가중합에만 들어가므로 softmax 미분을 아예 거치지 않습니다. $W_Q$와 $W_K$는 둘 다
거치지만 **곱해지는 파트너가 다릅니다** ($K$ vs $Q$).

실제로 재봤습니다. $d{=}64$, $d_k{=}32$, $n{=}16$에서 시드 300개로 gradient 코사인 유사도의
절댓값을 측정한 결과입니다.

| 쌍 | 평균 | 중앙값 |
|---|---|---|
| $W_Q$–$W_K$ | 0.045 | 0.039 |
| $W_Q$–$W_V$ | 0.047 | 0.041 |
| $W_K$–$W_V$ | 0.023 | 0.020 |
| *무작위 기준선* | *0.018* | — |

**셋 다 거의 직교합니다.** 하나의 스칼라 loss에서 나왔는데 사실상 무관한 방향으로 갑니다. 랜덤
초기화가 대칭성을 깨고, 구조적 차이가 셋을 계속 벌려놓는 거예요.

> **여기서 방법론을 하나 배웠습니다.** 처음엔 시드 **하나**로만 재서 $W_Q$–$W_K$ = 0.02,
> $W_Q$–$W_V$ = 0.34가 나왔고, "Q와 K가 특히 더 직교하다"는 결론을 낼 뻔했습니다. 시드를
> 300개로 늘리니 **그 구분이 사라졌습니다.** 당시 세팅이 $d{=}6$, $d_k{=}4$, $n{=}3$으로 차원이
> 아주 작았던 게 원인이었어요 — 저차원일수록 코사인 유사도의 분산이 큽니다($1/\sqrt{D}$ 성질).
> 단일 실행 결과를 일반화하면 안 된다는 걸 제 손으로 확인한 셈입니다.

한 가지 덧붙이면, "질문한다·답한다·의미를 담는다"는 건 **사람이 붙인 해석**입니다. gradient
descent가 그 의미를 이해하고 최적화하는 게 아니에요. 확실한 건 "구조적으로 다른 gradient를
받는다"까지입니다.

### 왜 √d_k로 나누나

$q$와 $k$의 각 성분이 독립이고 평균 0, 분산 1이라고 가정하면(학습 초기의 근사입니다),

$$
\text{Var}(q\cdot k)=\text{Var}\Big(\sum_{i=1}^{d_k}q_ik_i\Big)
=\sum_i\text{Var}(q_ik_i)=d_k
$$

독립 확률변수 합의 **분산 가법성**을 쓴 것이고, $\text{Var}(q_ik_i)=E[q_i^2]E[k_i^2]=1$입니다.
즉 **내적의 분산이 차원에 정비례**하고 표준편차는 $\sqrt{d_k}$입니다.

이게 왜 문제냐면, softmax의 미분이 $\text{softmax}(z)_i(1-\text{softmax}(z)_i)$라서 출력이 0이나
1에 가까워지면 **gradient가 죽기** 때문입니다.

key 20개를 두고 softmax 최대 가중치의 평균을 측정해봤습니다.

| $d_k$ | 4 | 32 | 128 | 512 |
|---|---|---|---|---|
| scaling 없음 | 0.401 | 0.769 | 0.887 | **0.943** |
| $\div\sqrt{d_k}$ | 0.208 | 0.216 | 0.217 | **0.217** |

<figure className="my-6 text-center">
  <img src="/images/posts/attention-is-all-you-need/fig_sqrt_dk.png" alt="d_k에 따른 내적 분산과 softmax 붕괴" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">왼쪽 — 내적의 표준편차가 정확히 √d_k를 따라감. 오른쪽 — scaling 없이는 한 key가 확률을 독식하고, 나누면 d_k와 무관하게 일정해진다</figcaption>
</figure>

scaling 없이 $d_k{=}512$면 **한 key가 94%를 가져갑니다.** 사실상 one-hot이고 gradient가 거의 0이
됩니다. 나누면 $d_k$와 **무관하게** 0.21로 안정됩니다. 균등분포(0.05)보다는 뾰족하지만 붕괴하지는
않는, 딱 좋은 지점이에요.

**왜 하필 제곱근인지도 여기서 나옵니다.** $d_k$로 나눴다면 분산이 $1/d_k$가 되어 이번엔 반대로
균등분포로 뭉개집니다. **$\sqrt{d_k}$는 분산을 정확히 1로 맞추는 유일한 지수입니다.**

참고로 논문은 이 논증을 본문이 아니라 각주 4에 적어뒀고, 표현도 "We suspect that..."입니다.

그리고 이건 트랜스포머 전체를 관통하는 원칙이기도 합니다. LayerNorm도,
$\times\sqrt{d_{\text{model}}}$도 결국 같은 이야기예요 — **분산을 1 근처로 유지한다.**

### multi-head는 많을수록 좋지 않다

"8개 만드니까 조합이 다양해진다"는 설명은 방향은 맞지만 이유가 중요합니다. **"같은 계산을
8번 반복"이 아닙니다.**

다양성의 **유일한 원천은 head마다 $W_Q$, $W_K$, $W_V$가 독립**이라는 사실이고, 다양성이
**두 층위**에서 동시에 생깁니다.

- $W_Q$, $W_K$가 다름 → **어떤 단어에 주목할지**($\alpha_{ij}$)가 달라짐
- $W_V$가 다름 → 같은 단어를 봐도 **거기서 무엇을 읽어낼지**가 달라짐

단일 head라면 "동사–목적어 관계"와 "주어–동사 수 일치"를 **하나의 가중합으로 타협**해야 합니다.

그런데 많을수록 좋은 건 아닙니다. Table 3 (A)에서 $h \times d_k = 512$를 고정하고 head 수만
바꾼 결과입니다.

| $h$ | 1 | 4 | 8 (base) | 16 | 32 |
|---|---|---|---|---|---|
| $d_k = d_v$ | 512 | 128 | 64 | 32 | 16 |
| BLEU | 24.9 | 25.5 | **25.8** | **25.8** | 25.4 |

**$h=32$에서 꺾입니다.** head 수가 늘면 다양성은 올라가지만 $d_k$가 줄어 head별 표현력이
떨어지는 trade-off예요.

덧붙이면, 학습된 head 중 상당수가 redundant해서 일부를 제거해도 성능이 거의 떨어지지 않는다는
후속 관찰이 있습니다. 제가 출처를 확인하지 못해 단정하지는 않겠지만, **"다양성이 이론적으로
가능하다"와 "실제로 8개가 다 다른 역할을 한다"는 별개**라는 점은 기억해둘 만합니다.

## Add & Norm은 하나가 아니라 둘이다

Figure 1에 "Add & Norm"이라고 한 덩어리로 그려져 있어서 하나로 묶어 외우기 쉬운데,
**둘은 서로 다른 문제를 풉니다.** 분리해서 봐야 합니다.

### Add는 gradient 통로를 확보한다

$$\frac{\partial}{\partial x}\big(x+F(x)\big) = I + \frac{\partial F}{\partial x}$$

backprop은 층마다 이 값을 **곱합니다.** skip이 없으면 $\prod_\ell \partial F/\partial x$가 되어
각 항이 조금만 1보다 작아도 지수적으로 0에 수렴합니다. skip이 있으면 $F$가 죽어도 **$I$가 남아서**
통로가 보장되고요.

$L=32$, $\partial F/\partial x = 0.1$인 경우로 계산하면 이렇습니다.

| | 32층 통과 후 |
|---|---|
| skip 없음 | $10^{-32}$ |
| skip 있음 | $21.1$ |

여기서 한 가지 정확히 해둘 게 있습니다. **$F$의 미분이 작아지는 것 자체는 막히지 않습니다.**
바뀌는 건 작은 값들이 최종 결과에 **결합되는 방식**이에요.

- 곱셈 구조 $\prod \delta_\ell$: 항 하나만 작아도 전체가 지수적으로 붕괴
- $\prod(1+\delta_\ell)$: $\delta$가 0이어도 각 항이 **최소 1**이라 절대 0으로 수렴하지 않음

### Norm은 그 부작용을 뒤처리한다

skip은 gradient를 살리는 대신 **activation이 층마다 커지는 부작용**을 만듭니다. $\delta > 0$이
반복되면 이번엔 값이 발산하는 쪽이 문제가 되죠. Norm이 매 블록에서 다시 눌러줍니다.
**그래서 skip과 norm은 항상 세트입니다.**

### 왜 BatchNorm이 아닌가

NLP의 입력은 $(N, T, D)$이고 BatchNorm은 배치 축으로 통계를 냅니다. 두 가지가 걸립니다.

1. **문장 길이가 다릅니다.** `<pad>` 토큰이 통계를 오염시켜요. masking을 해도, 위치 $t$가 클수록
   실제 토큰이 있는 문장 수가 배치마다 달라져서 **통계 신뢰도가 위치마다 달라집니다.**
2. **"같은 위치의 토큰들"이 의미론적으로 무관합니다.** 이미지의 "같은 채널"은 "같은 종류의 특징"이라
   배치로 묶는 게 타당하지만, "3번째 단어"는 문장마다 완전히 다른 단어예요.

LayerNorm은 **토큰 하나의 $D$개 feature**에 대해서만 통계를 냅니다. 그래서 배치 크기가 1이어도,
추론에서 한 단어씩 넣어도 문제가 없습니다.

### Post-LN에서 Pre-LN으로 뒤집힌 이유

원 논문은 $\text{LN}(x + F(x))$, 즉 **Post-LN**입니다. 그런데 GPT-2와 ViT는 $x + F(\text{LN}(x))$,
**Pre-LN**을 씁니다.

$$
\text{Post-LN: } J_{\text{LN}}\Big(I+\tfrac{\partial F}{\partial x}\Big)
\qquad\qquad \text{Pre-LN: } I+\delta_\ell
$$

**Post-LN은 $J_{\text{LN}}$이 괄호 바깥**에 있어서 애써 만든 $I$를 오염시킵니다.
**Pre-LN은 LN이 $\delta_\ell$ 안에만** 들어가서 $I$가 무손상으로 남고요.

$J_{\text{LN}} \approx 0.95$로 두고 층수만큼 곱해보면 차이가 분명해집니다.

| 층 수 | $J_{\text{LN}}^{L}$ |
|---|---|
| 6층 (원 논문) | 0.735 |
| 32층 (ViT-Huge) | 0.194 |

**원 논문이 6층이라 Post-LN으로 버틴 겁니다.** 수십 층부터는 Pre-LN이 표준이 됐고요. 대가는
residual stream이 정규화 없이 누적된다는 것이라, **마지막에 LN을 한 번 더** 붙여야 합니다.

$\delta_\ell$이라는 기호가 헷갈릴 수 있는데, 표준 기호가 아니라 축약입니다. skip 경로가 아닌
**나머지 전부**를 가리켜요. 정확히는
$\delta_\ell = \frac{\partial F_\ell}{\partial(\text{LN})}\cdot J_{\text{LN}}$이고 $d \times d$
**Jacobian 행렬**입니다(위에서 스칼라처럼 쓴 건 직관용 단순화입니다). Pre-LN에서 LN이 사라지는 게
아니라 **$\delta$ 안에 갇혀서 $I$를 건드리지 못하는 것**이 핵심입니다.

이 이야기는 [ViT 리뷰](/posts/2026-09-01-vit-an-image-is-worth-16x16-words)에서 이어집니다.
ViT-Huge가 32층이거든요.

## FFN은 왜 늘렸다 줄이나

$512 \rightarrow 2048 \rightarrow 512$. 왜 굳이 늘렸다가 다시 줄일까요.

**늘리는 이유는 표현력입니다.** ReLU 뉴런 하나가 입력 공간의 "꺾임" 하나에 대응하므로, 뉴런이
많을수록 조합할 수 있는 조각별 선형함수가 정교해집니다(universal approximation의 직관이에요).
은닉 뉴런 수가 곧 $d_{ff}$입니다.

**줄이는 이유는 구조적 제약입니다.** skip-connection이 $x + \text{FFN}(x)$를 계산하려면 shape이
같아야 하니까요.

역할 분담으로 보면 이렇습니다. **attention은 가로세로 둘 다** 건드리고(단어끼리 섞음),
**FFN은 가로로만** 건드립니다(각 위치를 독립적으로 가공).

확장비 4는 **경험값이고 이론적 최적이 아닙니다.** Table 3 (C)를 보면,

| $d_{ff}$ | 1024 | 2048 (base) | 4096 |
|---|---|---|---|
| BLEU | 25.4 | 25.8 | **26.2** |
| params | 53M | 65M | 90M |

키우면 좋아지긴 하는데 파라미터도 같이 늘어납니다. **4가 최적이라는 논증은 논문에 없습니다.**

## masking은 왜 softmax 전에 하나

decoder에서 미래 토큰을 가릴 때, 아주 큰 음수로 바꿔치기해서 $e^{-\infty} = 0$이 되게 합니다.
그런데 왜 굳이 softmax **전**일까요. 후에 0으로 만들면 안 되나요.

**안 됩니다. softmax는 전체 합이 1이 되도록 정규화**하기 때문입니다. softmax를 먼저 통과시키면
이미 **분모에 미래 단어가 들어가** 있어요. 나중에 지우면 남은 가중치의 합이 1이 아니게 되고,
미래 정보가 분모를 통해 결과에 스며듭니다. **순서가 본질입니다.**

추론할 때도 똑같이 masking하므로, 단어를 더 뽑아도 **이전 결과가 바뀌지 않습니다.**

## PPL과 BLEU는 서로 다른 것을 잰다

**PPL**은 "평균 몇 가지 중에 헷갈렸나"입니다. 정답 문장 확률의 기하평균의 역수이고 작을수록
좋습니다. 100단어 중 균등하게 찍으면 PPL이 100이에요. Cross-Entropy loss에 $\exp$를 취하면 PPL이
됩니다.

**BLEU**는 $BP\cdot\prod_{n=1}^{N}p_n^{w_n}$이고, 보정이 세 가지 들어갑니다.

- **$n$-gram**: unigram만 쓰면 순서를 뒤섞어도 점수가 같아지니까
- **clipping**: 같은 단어를 반복해 점수를 부풀리는 걸 막으려고(정답에 존재하는 개수까지만 인정)
- **brevity penalty**: 너무 짧게 번역해 정보를 날리는 걸 막으려고. 길게 하는 건 어차피
  precision에서 손해라 따로 벌점이 필요 없습니다

재미있는 사례가 Table 3 (D)에 있습니다. **label smoothing을 켜면 PPL은 나빠지는데 BLEU는
좋아집니다.**

| label smoothing | PPL (dev) | BLEU (dev) |
|---|---|---|
| 0.0 | **4.67** | 25.3 |
| 0.1 (base) | 4.92 | **25.8** |
| 0.2 | 5.47 | 25.7 |

모델이 덜 확신하게 만들면 확률 예측의 품질(PPL)은 나빠지지만 번역 품질(BLEU)은 올라갑니다.
**두 지표가 서로 다른 것을 재고 있다**는 걸 보여주는 좋은 예입니다.

## 논문이 정당화한 것과 관행을 상속한 것

리뷰를 마치면서 정리해두고 싶은 표입니다. 논문의 설계 선택들이 **다 같은 무게가 아닙니다.**

| 선택 | 논문의 근거 |
|---|---|
| $\div\sqrt{d_k}$ | **있음** — 분산이 커져 softmax gradient가 죽는다 (각주 4) |
| multi-head | **있음** — Table 3 (A) ablation |
| sinusoidal PE | **부분적** — 상대 위치의 선형 표현을 "가정", 학습형과 차이 없음을 Table 3 (E)로 확인 |
| FFN 확장비 4 | **없음** — 크기를 바꾼 실험은 있으나 4가 최적이라는 논증은 없음 |
| $\times\sqrt{d_{\text{model}}}$ | **없음** — 한 문장, 근거 없음 |
| Post-LN | **없음** — 이후 Pre-LN으로 뒤집힘 |
| 상수 10000 | **없음** — 값만 제시 |

**논문을 읽을 때 이 구분을 하는 게 생각보다 중요합니다.** "트랜스포머는 이렇게 설계되어 있다"를
전부 같은 확신으로 받아들이면, 사실은 그냥 관행인 것까지 원리로 착각하게 되니까요. 실제로
Post-LN은 몇 년 뒤에 뒤집혔습니다.

덧붙이면 논문 스스로도 sinusoidal을 고른 이유를 성능이 아니라 **"학습 때보다 긴 시퀀스로
외삽할 수 있을지도 모른다"는** 기대로 적고 있습니다. Table 3 (E)를 보면 learned PE와 BLEU가
25.7 대 25.8로 사실상 같거든요.

## 공부하면서 실제로 틀렸던 것들

정리하면서 제가 틀렸던 것들도 남겨둡니다.

1. **"더하면 정보가 섞인다"** — 직관은 맞지만 고차원 준직교성 때문에 실제로는 문제가 되지
   않습니다
2. **sinusoidal의 회전행렬 성질이 상대 위치를 "보장"한다고 읽음** — $M$이 회전과 commute해야만
   성립하는 조건부 성질입니다
3. **단일 시드 실험 결과를 일반화할 뻔함** — gradient 코사인 유사도를 300 시드로 다시 재니
   결론이 바뀌었습니다

2번과 3번이 특히 기억에 남습니다. 둘 다 **"그럴듯한 설명"에서 멈추지 않고 한 번 더 따져봤을 때만**
드러나는 것들이었거든요.

## 다음 글

이 논문을 읽고 나서 **[ViT — An Image is Worth 16×16 Words](/posts/2026-09-01-vit-an-image-is-worth-16x16-words)** 를
읽었습니다. 여기서 다룬 Pre-LN, CLS 토큰, learned positional embedding이 전부 그쪽에서 다시
나옵니다. 특히 **"위치 임베딩이 2D 구조를 스스로 배운다"** 는 실험은, 이 글에서 본
sinusoidal의 성질과 **같은 결과인데 출처가 정반대**라서 나란히 놓고 보면 재미있습니다.

---

**참고문헌**

- Vaswani et al., *Attention Is All You Need*, NIPS 2017
- Dosovitskiy et al., *An Image is Worth 16×16 Words*, ICLR 2021
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021

*(Figure 1, 2는 원논문에서 가져왔고, sinusoidal 유사도와 $\sqrt{d_k}$ 그림은 직접 계산해
그렸습니다. 본문의 측정값들도 모두 직접 재현한 것입니다.)*
