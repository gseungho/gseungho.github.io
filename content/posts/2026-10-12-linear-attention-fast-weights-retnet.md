---
title: "Linear Attention · Fast Weight · RetNet: 크기가 고정된 memory에 어떻게 쓸 것인가"
date: "2026-10-12"
category: "논문리뷰"
subcategory: "Deep Learning/Sequence Models"
tags: ["논문리뷰", "Linear Attention", "Fast Weight", "RetNet"]
description: "Linear Attention, Fast Weight와 RetNet의 고정 크기 상태를 비교하고 정보의 누적·갱신·감쇠와 delta rule의 경사하강법 해석을 정리했다."
---

softmax attention은 **과거의 key와 value를 개별 항목으로 보관한다.** 지나간
token의 key와 value를 손대지 않고 쌓아 두었다가 새 token이 올 때마다 전부 다시 읽는다. 요약하지도
고치지도 잊지도 않는다.

그래서 저장 공간이 sequence 길이 $N$에 비례해 자라고 token 하나를 만들 때마다 $N$개를 훑는다.
KV cache가 커지는 이유다.

이 글에서 다루는 논문들은 반대로 간다. 들고 다니는 memory의 크기를 **먼저 고정한다.** 그러면
읽는 비용은 줄지만 질문이 바뀐다. 무엇을 읽을지가 아니라 다음을 정해야 한다.

> **크기가 고정된 memory에 새 정보를 어떻게 쓸 것인가?**

세 논문이 세 가지 답을 낸다. 그냥 더하거나, 지우고 쓰거나, 오래된 것을 잊거나. 그리고 글의
마지막에는 그 "쓰기"가 사실 gradient descent 한 step이었다는 데까지 간다. 다음에 읽을 Test-Time
Training으로 이어지는 지점이다.

## KV cache와 고정 크기 상태

용어 하나를 먼저 정해 두자. token 하나를 처리한 뒤 다음 token으로 넘길 때 들고 가는 것 전부를
**state**라고 부르겠다.

- softmax attention의 state는 KV cache다. token이 올 때마다 key와 value가 리스트에 하나씩
  덧붙어서 크기가 $N \times d$로 자란다
- 이 글의 모델들은 state가 $d \times d$ 행렬 하나다. 새 token이 와도 리스트에 붙이지 않고 그 행렬
  위에 더한다. 크기가 변하지 않는다

$d = 64$로 숫자를 세어 보면 차이가 분명하다. head 하나, layer 하나 기준이다.

| | token 1,000개 처리 후 | token 100,000개 처리 후 |
|---|---|---|
| softmax (KV cache) | $2 \times 1000 \times 64 = 128{,}000$개 | $12{,}800{,}000$개 |
| 고정 크기 state | $64^2 + 64 = 4{,}160$개 | $4{,}160$개 |

읽는 비용도 따라간다. cache를 읽으려면 query를 key $N$개 전부와 비교해야 하지만 행렬은 벡터
하나를 곱하면 끝이다.

대신 쓰기가 문제가 된다. softmax는 공간이 계속 생기니 새 $(k, v)$를 얹기만 하면 된다. 고정
크기 state는 천 개든 십만 개든 같은 4,160개 숫자 안에 접어 넣어야 하고 어떻게 접을지를 정해야
한다.

| | 쓰는 방법 | 논문 |
|---|---|---|
| Linear attention | 더한다 | Katharopoulos et al., 2020 |
| Fast weight (delta rule) | 읽고, 지우고, 쓴다 | Schlag et al., 2021 |
| Retention | 감쇠시키고 더한다 | Sun et al., 2023 |

Performer (Choromanski et al., 2021)는 축이 조금 다르다. 쓰기가 아니라 **읽기**를 고치는 논문이라
뒤에서 따로 짧게 본다.

## Linear attention: 정보를 누적하는 메모리

### 유사도 함수를 내적으로 표현하기

attention의 한 행을 일반적인 꼴로 쓰면 이렇다. "Transformers are RNNs" 논문의 식 (3)이다.

$$V_i' = \frac{\sum_{j=1}^{N} \text{sim}(Q_i, K_j)\, V_j}{\sum_{j=1}^{N} \text{sim}(Q_i, K_j)}$$

softmax attention은 $\text{sim}(q, k) = \exp(q^\top k / \sqrt{D})$인 특수한 경우다.

관찰이 하나 필요하다. attention이 $\text{sim}$에 요구하는 건 **음수가 아니라는 것**뿐이다.
가중평균의 가중치니까. 그래서 유사도를 feature map $\phi$ 두 개의 내적으로 바꾼다.

$$\text{sim}(q, k) = \phi(q)^\top \phi(k), \qquad \phi(x) = \text{elu}(x) + 1$$

$\text{elu}(x) + 1$은 어디서나 양수다. relu를 쓰지 않은 이유도 논문에 있다. $x$가 음수일 때
gradient가 0이 되는 걸 피하기 위해서다.

짚어둘 점이 있다. **이건 softmax의 근사가 아니다.** 다른 attention이다. 논문도 근사라고
주장하지 않는다. 오히려 exp kernel에 대응하는 feature map은 무한 차원이라 softmax를 정확히
선형화할 수는 없다고 적는다. 그 근사를 시도한 것이 Performer다.

### 행렬곱의 순서와 계산 복잡도

<figure className="my-6 text-center">
  <img src="/images/posts/linear-attention-fast-weights-retnet/perf_fig1.png" alt="attention 행렬을 만들지 않고 계산하는 순서" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">왼쪽은 L×L 행렬을 먼저 만드는 순서, 오른쪽은 작은 행렬을 먼저 만드는 순서 (Choromanski et al., 2021, Figure 1)</figcaption>
</figure>

$\phi$로 바꾸면 무엇이 달라지는지가 이 그림이다. Performer 논문의 그림인데 아이디어가 가장 잘
보여서 가져왔다. 그림의 $L$이 sequence 길이(이 글의 $N$), $r$이 feature map의 폭(이 글의
$d$)이다.

$$\underbrace{\big(\phi(Q)\,\phi(K)^\top\big)\,V}_{O(N^2 d)} \;=\; \underbrace{\phi(Q)\,\big(\phi(K)^\top V\big)}_{O(N d^2)}$$

논문의 식 (6)이다. 쓴 것은 **행렬곱의 결합법칙** 하나다. 왼쪽은 $N \times N$ 행렬을 먼저 만들고
$V$를 곱한다. 오른쪽은 $\phi(K)^\top V$라는 $d \times d$ 행렬을 먼저 만들고 $\phi(Q)$를 곱한다.
결과는 같은데 $N \times N$ 행렬이 한 번도 만들어지지 않는다.

softmax에서는 이 괄호를 옮길 수 없다. $\exp(q^\top k)$가 $q$만의 함수와 $k$만의 함수의 유한
차원 내적으로 나뉘지 않기 때문이다. $\phi(q)^\top\phi(k)$는 처음부터 나뉘어 있다.

### Causal linear attention의 재귀식

언어 모델은 causal mask를 쓴다. $i$번째 token은 자기까지만 본다. 그러면 합의 범위가
$j \le i$로 줄어든다.

$$V_i' = \frac{\phi(Q_i)^\top \sum_{j \le i} \phi(K_j)\, V_j^\top}{\phi(Q_i)^\top \sum_{j \le i} \phi(K_j)}$$

분자와 분모의 합은 둘 다 **누적합**이다. $i$번째 값은 $i-1$번째 값에 항 하나를 더한 것이다.
그러니 과거 token을 들고 있을 필요가 없고 누적값만 넘기면 된다. 논문의 식 (16)부터
(20)이다.

$$
\begin{aligned}
S_i &= S_{i-1} + \phi(K_i)\, V_i^\top \\
z_i &= z_{i-1} + \phi(K_i) \\
y_i &= \frac{\phi(Q_i)^\top S_i}{\phi(Q_i)^\top z_i}
\end{aligned}
$$

$S \in \mathbb{R}^{d \times d}$, $z \in \mathbb{R}^{d}$이고 둘 다 $N$과 무관하다. 근사는 어디에도
없다. 같은 식을 다시 쓴 것뿐이다.

논문 제목이 "Transformers are RNNs"인 이유다. RNN의 일반형과 나란히 놓으면 그 말이 왜
정확한지 보인다.

$$h_i = f(h_{i-1},\, x_i), \qquad y_i = g(h_i,\, x_i)$$

RNN의 조건은 둘이다. 다음 상태가 **직전 상태와 지금 입력만으로** 정해질 것, 출력이 그 상태에서
나올 것. 위의 세 줄은 $(S, z)$를 상태로 두면 이 조건을 그대로 만족한다.

| | RNN | linear attention |
|---|---|---|
| 상태 | 벡터 $h$ | 행렬 $S$와 벡터 $z$ |
| update $f$ | 학습되는 $W_h$를 거치는 비선형 함수 | $\phi(k)\,v^\top$를 **더한다** |
| 출력 $g$ | $W_y h_i$ | $\phi(q_i)$로 상태를 읽는다 |

차이는 update가 덧셈이라는 점인데 이게 오히려 결정적이다. 재귀가 선형이고 덧셈이면 $S_i$는
누적합이고 누적합은 순서대로 돌지 않아도 한 번에 계산할 수 있다. 그래서 **학습은 Transformer처럼
병렬로 하고 생성만 RNN처럼** 할 수 있다. tanh가 끼어 있는 보통의 RNN은 앞 step을 끝내야 다음
step을 시작할 수 있다.

### 상태 행렬 S에 저장되는 정보

$S$를 풀어 쓰면 바깥곱(outer product)의 합이다.

$$S_i = \sum_{j \le i} \phi(k_j)\, v_j^\top$$

key와 value를 짝지어 행렬에 더해 넣는 이 구조는 고전적인 **연상 메모리**(associative memory)
그대로다. 읽는 쪽을 풀어 쓰면 무슨 일이 일어나는지 보인다.

$$S_i^\top \phi(q) = \sum_{j \le i} \big(\phi(k_j)^\top \phi(q)\big)\, v_j$$

읽은 결과는 저장된 **모든** value의 가중합이고 가중치는 각 key와 query의 내적이다. query가
저장된 key 하나와 같고 key들이 서로 직교한다면 다른 항이 전부 0이 되어 원하는 value만 나온다.
그렇지 않으면 섞여서 나온다.

오해하기 쉬운 것이 하나 있다. 고정 크기라고 해서 오래된 token을 **버리는 게 아니다.** 최근
$M$개만 남기는 sliding window와 다르다. 첫 번째 token의 정보도 $S$ 안에 그대로 있다. 다만
다른 모든 token과 **겹쳐져** 있다.

| | 저장하는 것 | 잃는 것 |
|---|---|---|
| softmax | $(k_j, v_j)$ 쌍 $N$개를 그대로 | 없음. 대신 공간이 자람 |
| sliding window | 최근 $M$개만 그대로 | 오래된 token 전체 |
| linear attention | 전부를 하나로 합산 | 개별 token을 분리해 꺼내는 능력 |

종이 한 장에 문장을 계속 덧쓰는 것과 같다. 전부 거기 있는데 겹쳐서 흐릿하다. 얼마나 겹치면
못 읽게 되는지가 다음 논문의 주제다.

### 시퀀스 길이별 비용과 이미지 생성 속도

<figure className="my-6 text-center">
  <img src="/images/posts/linear-attention-fast-weights-retnet/trnn_fig1.png" alt="sequence 길이에 따른 시간과 GPU 메모리" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">Figure 1. sequence 길이에 따른 시간(왼쪽)과 GPU 메모리(오른쪽). 검은 선이 linear, 빨간 선이 softmax, 파란 선이 Reformer (Katharopoulos et al., 2020)</figcaption>
</figure>

두 축이 모두 log라서 기울기가 곧 지수다. 검은 선(linear)은 기울기 1, 빨간 선(softmax)은 기울기
2다. softmax는 길이 4,096에서 GPU 메모리가 차서 그 뒤로는 측정되지 않았다.

차이가 가장 크게 드러나는 건 autoregressive 이미지 생성이다.

| | bits/dim ↓ | images/sec ↑ |
|---|---|---|
| MNIST, softmax | **0.621** | 0.45 |
| MNIST, linear | 0.644 | **142.8** (317배) |
| CIFAR-10, softmax | 3.47 | 0.004 |
| CIFAR-10, linear | **3.40** | **17.85** (4,462배) |

bits/dim은 픽셀 하나를 표현하는 데 드는 비트 수로 낮을수록 좋다. 품질은 거의 같은데 속도가
수백 배에서 수천 배 차이 난다. 이미지 생성은 픽셀을 하나씩 뽑는 작업이라 추론 전체가 순차적이고
token 하나당 $O(1)$과 $O(N)$의 차이가 그대로 쌓이기 때문이다.

## Fast weight: 저장된 값을 갱신하는 메모리

### Linear attention과 Fast Weight Programmer의 관계

Schlag et al. (2021)의 출발점은 앞 절의 $S$가 새로운 것이 아니라는 관찰이다. 전치하고 정규화
항을 빼서 쓰면 이렇다. 논문의 식 (17)과 (19)에 해당한다.

$$W^{(i)} = W^{(i-1)} + v^{(i)} \otimes \phi(k^{(i)}), \qquad y^{(i)} = W^{(i)}\, \phi(q^{(i)})$$

이 구조는 Schmidhuber가 1991년에 제안한 **Fast Weight Programmer**와 같다. 이름에 뜻이 다
들어 있다.

- **slow weight**: $W_k, W_v, W_q$ 같은 보통의 파라미터다. 경사하강법으로 학습되고 학습이 끝나면
  고정된다
- **fast weight**: $W$다. 문장이 시작될 때 0에서 출발해 token마다 바뀌고 문장이 끝나면 버려진다

slow network가 입력에서 $k, v, q$를 만들면 이 셋이 fast weight를 어떻게 바꿀지 정하는 **명령**이
된다. 그래서 programmer다.

헷갈리기 쉬운 점이 있다. **$W$는 파라미터가 아니다.** forward 중에 만들어졌다 사라지는 중간
계산값이고 checkpoint에 저장되지 않는다. 학습되는 파라미터는 softmax Transformer와 똑같이 다
있고 개수도 같다. gradient는 $W$를 갱신하려고 흐르는 게 아니라 $W$를 **통과해서** $W_k$와
$W_v$까지 간다. 생긴 것이 행렬이고 벡터에 곱해서 쓰니까 weight라는 이름이 붙었을 뿐이다.

softmax attention도 구조는 같다. $W_k, W_v, W_q$가 slow이고 KV cache가 fast 쪽이다. 차이는
fast 쪽을 자라는 리스트로 들고 있느냐 고정 크기 행렬로 들고 있느냐뿐이다.

그리고 위의 update는 Hebb 규칙(1949) 그대로다. 순수하게 **더하기만** 한다. 한 번 쓴 것을
고칠 방법이 없다.

### 고정 크기 메모리의 용량 한계

앞에서 본 읽기 식을 다시 가져와 보자.

$$W\phi(q) = \sum_j v_j\, \big(\phi(k_j)^\top \phi(q)\big)$$

$v_3$를 꺼내려고 $\phi(q) = \phi(k_3)$를 넣으면 $v_3$ 하나만 나오지 않는다.

$$W\phi(k_3) = \lVert\phi(k_3)\rVert^2\, v_3 \;+\; \sum_{j \ne 3} \big(\phi(k_j)^\top\phi(k_3)\big)\, v_j$$

둘째 항이 간섭이다. key들이 서로 직교해야 0이 된다.

그런데 $d_{\text{dot}}$차원 공간에 서로 직교하는 벡터는 **최대 $d_{\text{dot}}$개**다. 그보다
많은 연관을 저장하면 간섭은 생길 수도 있는 게 아니라 생길 수밖에 없다. 논문은 sequence 길이가
$d_{\text{dot}}$을 넘는 상태를 overcapacity라고 부른다.

실제 sequence는 거의 항상 key 차원보다 길다. 그러니 이 한계는 예외가 아니라 기본 상황이다.
softmax에는 이 한계가 없다. 저장 공간이 자라니까. 저장 공간을 고정한 대신 정보 간 간섭이 생기는 것이다.
고정 크기 state는 속도를 위한 트릭이기 전에 **기억할 수 있는 양의 상한**이다.

### 저장한 key 수에 따른 검색 성능

<figure className="my-6 text-center">
  <img src="/images/posts/linear-attention-fast-weights-retnet/fwp_fig2.png" alt="저장한 key 수에 따른 retrieval loss" className="rounded-lg mx-auto max-w-xl" />
  <figcaption className="mt-2 text-sm text-gray-500">Figure 2. 가로축이 저장한 key-value 쌍의 수, 세로축이 retrieval loss (Schlag et al., 2021)</figcaption>
</figure>

합성 과제다. 서로 다른 key-value 쌍을 20개부터 600개까지 늘려 가며 전부 저장하게 한 뒤 제대로
꺼내는지를 본다.

- **초록(Linear)**: $d_{\text{key}} = d_{\text{dot}} = 64$인 linear attention이다. **60개쯤부터**
  loss가 오른다. 이론상 한계가 64인데 거의 그 자리에서 무너진다
- **DPFP-1, 2, 3**: 이 논문이 제안한 feature map으로 key를 각각 128, 256, 384차원으로 펼친다.
  무너지기 시작하는 지점도 그 근처다
- **회색(Softmax)**: 500개 근처까지 거의 0을 유지한다
- **FAVOR+**: Performer의 random feature다. 이 과제에서는 어떤 설정에서도 loss가 0에 닿지
  못했다

앞 절의 논증이 그림 한 장에 그대로 나타난다.

### Delta rule로 저장된 값 갱신하기

논문은 쓰는 규칙을 바꾸자고 제안한다. 식 (20)부터 (24)다.

$$
\begin{aligned}
\bar{v}^{(i)} &= W^{(i-1)}\, \phi(k^{(i)}) \\
\beta^{(i)} &= \sigma\big(W_\beta\, x^{(i)}\big) \\
W^{(i)} &= W^{(i-1)} + \beta^{(i)}\,\big(v^{(i)} - \bar{v}^{(i)}\big) \otimes \phi(k^{(i)})
\end{aligned}
$$

세 단계다.

1. 새 key로 지금 memory를 **먼저 읽는다.** 그 key 자리에 현재 저장돼 있는 값이 $\bar{v}$다
2. 얼마나 세게 덮어쓸지를 정하는 $\beta$를 만든다. sigmoid를 거치니 0과 1 사이다
3. 새 값과 기존 값의 **차이**에 $\beta$를 곱해 더한다

마지막 줄을 풀면 왜 "지우고 쓰기"인지 보인다. $v_{\text{new}} = \beta v + (1-\beta)\bar{v}$로
두면 이렇게 나뉜다.

$$W^{(i)} = W^{(i-1)} \underbrace{+\; v_{\text{new}}^{(i)} \otimes \phi(k^{(i)})}_{\text{쓰기}} \underbrace{-\; \bar{v}^{(i)} \otimes \phi(k^{(i)})}_{\text{지우기}}$$

$\beta = 0$이면 기존 값을 그대로 두고 $\beta = 1$이면 완전히 바꾼다.

$\beta$를 두고 헷갈리기 쉬운 것이 둘 있다. 학습되는 것은 $\beta$가 아니라 $W_\beta$다.
$\beta^{(i)}$는 $k, v, q$처럼 token마다 입력에서 계산되는 값이다. 그리고 $W_\beta$까지 gradient가
닿는 길은 $W$를 지난다. $\beta^{(i)}$가 $W^{(i)}$를 바꾸고 그 $W$가 이후의 출력을 전부 만드니
$\beta^{(i)}$는 그 시점 이후의 모든 loss에서 신호를 받는다.

이 규칙은 Widrow와 Hoff의 delta rule(1960)이다. 다른 점은 learning rate에 해당하는 $\beta$를
사람이 정하지 않고 network가 token마다 직접 만든다는 것이다.

state 크기는 그대로다. 용량이 늘어난 게 아니라 **관리되기 시작한** 것이다.

### 같은 key의 값을 갱신하는 실험

<figure className="my-6 text-center">
  <img src="/images/posts/linear-attention-fast-weights-retnet/fwp_fig3.png" alt="update 규칙별 학습 곡선" className="rounded-lg mx-auto max-w-xl" />
  <figcaption className="mt-2 text-sm text-gray-500">Figure 3. update 규칙별 학습 곡선. 세로축이 log scale의 loss (Schlag et al., 2021)</figcaption>
</figure>

delta rule이 필요한 상황을 직접 만든 실험이다. 같은 key가 sequence 안에서 여러 번 나오고 그때마다
다른 value가 붙는다. 모델은 가장 최근 값을 답해야 한다. 누적이 아니라 **갱신**이 필요한
과제다. sequence 길이가 40이고 key가 20종류라 용량 자체는 넘지 않는다.

- **주황(DPFP-1 delta)**: 이 논문의 규칙이다. 수백 step 만에 loss가 $10^{-3}$ 아래로 내려간다
- **초록(DPFP-1 sum)**: 더하기만 하는 규칙이다. $10^{-1}$ 근처에서 움직이지 않는다

같은 feature map에 쓰는 규칙만 바꾼 비교라서 차이의 원인이 분명하다. 나머지 두 선은 이전의 fast
weight 모델이다. 범례의 "Schlag (2021)"은 이 논문이 아니라 같은 저자가 같은 해에 낸 다른
논문(Schlag, Munkhdalai & Schmidhuber, ICLR 2021)을 가리킨다.

softmax attention에는 갱신이라는 개념이 없다. 논문의 표현대로 key와 value 쌍을 바꿀 수 없는
채로 이어 붙일 뿐이다. 고정 크기 memory는 그럴 수 없어서 갱신 규칙이 있어야 한다.

## RetNet: 오래된 정보를 감쇠시키는 메모리

### 감쇠 계수 γ와 기억 범위

Retentive Network (Sun et al., 2023)는 같은 문제에 다른 답을 낸다. 논문의 식 (6)이다.

$$S_n = \gamma\, S_{n-1} + K_n^\top V_n, \qquad \text{Retention}(X_n) = Q_n S_n$$

linear attention의 식과 두 군데가 다르다.

첫째, 이전 state에 **$\gamma$를 곱한다.** 1보다 작은 스칼라라서 매 step마다 과거 전체가 조금씩
옅어진다. 둘째, **분모 $z$가 없다.** token마다 하던 정규화를 하지 않는다. (실제 식에는
$Q$와 $K$에 위치에 따른 회전이 곱해지는데 여기서는 생략했다.)

$\gamma$는 학습하지 않는다. head마다 다른 값을 고정해서 쓴다.

$$\gamma = 1 - 2^{-5 - \text{arange}(0,\, h)}$$

앞의 세 head는 $\gamma$가 $1 - 1/32$, $1 - 1/64$, $1 - 1/128$이다. 기여가 절반으로 줄어드는
거리로 바꾸면 각각 약 22, 44, 88 token이다. 가까운 문맥을 보는 head와 먼 문맥을 보는 head를
섞어 두는 셈이다.

### 재귀식에서 감쇠 행렬 D 유도하기

RetNet은 같은 계산을 병렬 형태로도 쓴다. 논문의 식 (5)다.

$$\text{Retention}(X) = \big(QK^\top \odot D\big)\,V, \qquad D_{nm} = \begin{cases} \gamma^{\,n-m}, & n \ge m \\ 0, & \text{otherwise} \end{cases}$$

$D$를 성분별로 곱하는 이유는 재귀식을 펼쳐 보면 나온다.

$$
\begin{aligned}
S_1 &= K_1^\top V_1 \\
S_2 &= \gamma K_1^\top V_1 + K_2^\top V_2 \\
S_3 &= \gamma^2 K_1^\top V_1 + \gamma K_2^\top V_2 + K_3^\top V_3
\end{aligned}
$$

$m$번째에 써 넣은 항은 $n$번째에 읽힐 때까지 $\gamma$가 $n - m$번 곱해진다. 일반항과 출력은
이렇다.

$$S_n = \sum_{m \le n} \gamma^{\,n-m} K_m^\top V_m, \qquad Q_n S_n = \sum_{m \le n} \underbrace{\gamma^{\,n-m}}_{\text{위치}}\, \underbrace{\big(Q_n K_m^\top\big)}_{\text{내용}}\, V_m$$

$V_m$들의 가중합이고 가중치가 "내용으로 정해지는 값"과 "위치로 정해지는 값"의 곱이다. 앞의
것이 $QK^\top$의 $(n, m)$ 성분, 뒤의 것이 $D$의 $(n, m)$ 성분이다. 쌍마다 계수가 따로 있으니
성분별 곱이어야 한다.

그러니 $D$는 설계해서 얹은 것이 아니라 재귀가 이미 하고 있는 일을 펼쳐 쓴 결과다. 행렬 하나가
두 가지 일을 한다. 위쪽 삼각의 0은 미래를 가리는 causal mask이고 아래쪽의 $\gamma^{n-m}$은
감쇠다. softmax의 mask는 볼 수 있느냐 없느냐 둘 중 하나인데 $D$는 "볼 수는 있지만 멀수록
흐리게"를 표현한다.

### 병렬·재귀·청크 단위 계산

<figure className="my-6 text-center">
  <img src="/images/posts/linear-attention-fast-weights-retnet/ret_dual.png" alt="RetNet의 병렬 형태와 재귀 형태" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">Figure 3. (a) 병렬 형태, (b) 재귀 형태. GN은 GroupNorm (Sun et al., 2023)</figcaption>
</figure>

두 그림은 **같은 함수**다. 같은 weight를 두고 계산 순서만 바꾼 것이다.

- **병렬 형태**는 학습에 쓴다. sequence 전체를 한 번에 계산하니 GPU를 가득 채워 쓸 수 있다
- **재귀 형태**는 생성에 쓴다. state 하나만 들고 다니면 되고 KV cache가 필요 없다
- **chunkwise 형태**도 있다. 덩어리 안에서는 병렬로, 덩어리 사이는 재귀로 계산한다. 긴
  sequence를 학습할 때 쓰는 절충이다

그림의 GN은 GroupNorm이다. 앞에서 분모 $z$가 없다고 했는데 정규화를 없앤 게 아니라 **layer
출력으로 옮긴** 것이다. head마다 $\gamma$가 달라 출력의 분산이 제각각이어서 head별로 따로
정규화하고 그래서 LayerNorm이 아니라 GroupNorm을 쓴다. 논문의 ablation에서 이걸 빼면 성능이
떨어진다.

논문이 보고한 숫자는 이렇다. 7B 모델, 길이 8k 기준으로 KV cache를 쓰는 Transformer보다 생성이
8.4배 빠르고 메모리를 70% 덜 쓴다. 학습에서는 메모리가 25%에서 50% 줄었다. language modeling
perplexity는 1.3B, 2.7B, 6.7B 세 크기에서 Transformer와 비슷했고 2B를 넘으면 RetNet이 앞서는
경향이 있었다고 한다.

### 성능 저하에 대한 서로 다른 설명

<figure className="my-6 text-center">
  <img src="/images/posts/linear-attention-fast-weights-retnet/ret_tri.png" alt="불가능한 삼각형" className="rounded-lg mx-auto max-w-sm" />
  <figcaption className="mt-2 text-sm text-gray-500">Figure 2. 학습 병렬성, 저렴한 추론, 좋은 성능을 동시에 얻기 어렵다는 "impossible triangle" (Sun et al., 2023)</figcaption>
</figure>

RetNet의 주장은 이 삼각형으로 요약된다. 세 꼭짓점을 다 가진 구조가 없었는데 RetNet이 그 자리에
들어간다는 것이다. 그 주장을 표로 적은 것이 논문의 Table 1이다.

| 구조 | 학습 병렬화 | 추론 비용 | 긴 sequence 메모리 | 성능 |
|---|---|---|---|---|
| Transformer | ✔ | $O(N)$ | $O(N^2)$ | ✔✔ |
| Linear Transformer | ✔ | $O(1)$ | $O(N)$ | ✘ |
| Recurrent NN | ✘ | $O(1)$ | $O(N)$ | ✘ |
| RWKV | ✘ | $O(1)$ | $O(N)$ | ✔ |
| H3/S4 | ✔ | $O(1)$ | $O(N \log N)$ | ✔ |
| Hyena | ✔ | $O(N)$ | $O(N \log N)$ | ✔ |
| RetNet | ✔ | $O(1)$ | $O(N)$ | ✔✔ |

저자들이 자기 논문에 만든 표라서 결과라기보다 **주장**으로 읽어야 한다. 성능 칸의 기호는
측정값이 아니다.

눈에 띄는 건 Linear Transformer 줄의 ✘다. RetNet이 대는 이유는 **위치 정보**다. 2.4절에
"linear attention은 위치 정보를 효과적으로 인코딩하지 못해 성능이 떨어진다"고 적혀 있다. 앞 절의
용량 이야기는 나오지 않는다.

두 논문이 **같은 증상에 다른 진단**을 내렸다.

| | 진단 | 처방 |
|---|---|---|
| Schlag et al. | 용량이 차서 값이 섞인다 | delta rule. **내용**을 보고 지운다 |
| Sun et al. | 위치 정보를 담지 못한다 | $\gamma$ 감쇠. **위치**를 보고 잊는다 |

각자의 처방이 자기 진단을 그대로 따른다. 어느 쪽이 맞는지는 이 논문들만으로는 가릴 수 없다.
두 진단을 같은 조건에서 맞세운 실험이 없기 때문이다.

## Performer: softmax kernel의 근사

앞에서 미뤄 둔 질문이 있다. $\phi$를 아무 양수 함수로 바꾸면 그건 더 이상 softmax가 아니다.
softmax에 가까운 $\phi$는 없을까?

정확한 것은 없다. exp kernel의 feature map이 무한 차원이니까. Performer는 이걸 **random
feature로 추정**한다. 논문의 Lemma 1이다.

$$\exp(x^\top y) = \mathbb{E}_{\omega \sim \mathcal{N}(0, I_d)}\Big[\exp\!\Big(\omega^\top x - \frac{\lVert x\rVert^2}{2}\Big)\, \exp\!\Big(\omega^\top y - \frac{\lVert y\rVert^2}{2}\Big)\Big]$$

$\omega$를 $r$개 뽑아 평균을 내면 기댓값이 정확히 softmax kernel인 추정량이 된다. 그리고 각 항이
지수함수라 **항상 양수**다.

양수라는 점이 이 논문의 핵심이다. 더 흔히 쓰이던 추정량은 sin과 cos을 쓰는데 그러면 값이 음수가
될 수 있다. attention 값은 대부분 0 근처이고 바로 그 구간에서 삼각함수 추정량의 분산이
커진다. 정규화 항이 음수가 되면서 학습이 아예 안 되거나 성능이 크게 떨어진다고 논문은 보고한다.
양수 feature에 직교성을 더한 것이 FAVOR+다. 이름의 positive와 orthogonal이 그 두 가지다.

다만 근사에는 대가가 있다. 앞의 용량 실험에서 FAVOR+는 어떤 설정에서도 loss 0에 닿지
못했다. 정확히 꺼내야 하는 과제에서는 추정 오차가 그대로 남는다.

## Delta rule을 경사하강법으로 해석하기

마지막으로 delta rule을 다른 각도에서 보자. memory $W$가 하는 일을 회귀 문제로 적어 본다.
$\phi(k)$를 넣었을 때 $v$가 나오게 하고 싶다.

$$\ell(W) = \frac{1}{2}\,\big\lVert W\phi(k) - v \big\rVert^2$$

$W$로 미분하면

$$\nabla_W\, \ell = \big(\underbrace{W\phi(k)}_{\bar{v}} - v\big)\,\phi(k)^\top$$

이고 괄호 안의 $W\phi(k)$가 앞에서 "먼저 읽은 값" $\bar{v}$다. learning rate $\beta$로 경사하강법으로 한 번
갱신하면 이렇게 된다.

$$W \leftarrow W - \beta\,\nabla_W\, \ell = W + \beta\,(v - \bar{v})\,\phi(k)^\top$$

**앞에서 본 delta rule과 같은 갱신식이다.** write strength라고 부르던 $\beta$는
이 해석에서는 학습률에 해당한다. 메모리 갱신을 학습 단계로 볼 수 있는 셈이다. 이 모델은 생성하는 동안 token
하나마다 SGD 갱신에 해당하는 연산을 하고 있었던 것이다.

순수한 linear attention도 같은 틀에 들어간다. $\bar{v}$를 뺀 꼴이다. 이미 무엇이 들어 있는지
확인하지 않고 정보를 더한다. 그래서 저장한 정보가 겹치면서 간섭이 생긴다.

여기서 다음 질문이 나온다. state가 행렬이고 쓰기가 gradient step 한 번이라면 state를 더 표현력
있는 것으로 바꾸고 쓰기를 제대로 된 학습으로 바꾸면 어떻게 될까? Test-Time Training 계열이 그
방향이다. state를 행렬이 아니라 작은 model의 파라미터로 두고 쓰기를 그 model을 학습시키는 gradient
step으로 정의한다. 다음에 읽을 논문이다.

그리고 쓰기 한 번이 gradient step 한 번의 비용이라면
[앞 글](/posts/2026-10-12-moe-mod-mor-dynamic-compute)의 질문이 그대로 돌아온다. 모든 token이 그
step을 받아야 할까?

## 정리

| | 쓰는 규칙 | state 크기 | token당 생성 비용 |
|---|---|---|---|
| Softmax | 없음. cache에 덧붙임 | $N$에 비례 | $O(N)$ |
| Linear | $S \mathrel{+}= \phi(k)\,v^\top$ | $d \times d$ | $O(1)$ |
| Delta | $W \mathrel{+}= \beta\,(v - \bar{v}) \otimes \phi(k)$ | $d \times d$ | $O(1)$ |
| Retention | $S \leftarrow \gamma S + k^\top v$ | $d \times d$ | $O(1)$ |

첫 줄 아래의 셋은 사실상 같은 구조다. 다른 것은 memory에 무엇을 남길지 정하는 **한 줄**뿐이다.

- 유사도를 내적으로 바꾸면 attention은 고정 크기 state를 가진 RNN이 된다. 유도는 행렬곱 결합법칙
  하나다
- 고정 크기 state에는 용량이 있다. key 차원보다 많은 연관을 넣으면 꺼낼 때 섞인다
- 그 뒤는 설계 결정 하나다. 무엇을 쓰고 무엇을 포기할 것인가

softmax attention은 key와 value를 개별 항목으로 보관하는 대신 저장 공간이 계속 늘어난다. 고정 크기 상태로 정보를 합치면 어떤 정보를 남기고 어떻게 갱신할지 정해야 한다.

실험 결과를 해석할 때는 과제와 평가 조건도 구분해야 한다. Katharopoulos et al.의 속도 수치는 이미지 생성 과제에서 나왔고 Schlag et
al.의 용량 실험은 합성 과제다. 큰 언어 모델에서 linear attention이 softmax에 뒤진다는 것은
RetNet의 표에 기호로만 나온다. 그 이유는 두 논문의 설명이 서로 다르다.

## 이어지는 글

- [MoE · MoD · MoR: 모든 token에 같은 계산을 써야 하는가](/posts/2026-10-12-moe-mod-mor-dynamic-compute)
- [GPT-1 · BERT · GPT-2 · GPT-3: 모델은 무슨 task인지 어디서 아는가](/posts/2026-10-12-gpt-bert-in-context-learning)
- [Attention Is All You Need: 설명이 항상 멈추는 자리들](/posts/2026-09-01-attention-is-all-you-need)

## 참고문헌

- Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. ICML.
- Schlag, I., Irie, K., Schmidhuber, J. (2021). *Linear Transformers Are Secretly Fast Weight Programmers*. ICML.
- Sun, Y. et al. (2023). *Retentive Network: A Successor to Transformer for Large Language Models*. arXiv:2307.08621.
- Choromanski, K. et al. (2021). *Rethinking Attention with Performers*. ICLR.
