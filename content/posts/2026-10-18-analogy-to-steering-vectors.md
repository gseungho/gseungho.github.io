---
title: "Analogy에서 steering vector까지: 선형 표현 가설"
date: "2026-10-18"
category: "논문리뷰"
subcategory: "Deep Learning/Interpretability"
tags: ["논문리뷰", "Steering Vector", "Linear Representation Hypothesis", "Word Embedding"]
description: "단어 벡터의 차이가 관계를 나타낸다는 analogy의 가정에서 출발해 같은 차이 벡터를 언어 모델의 hidden state에 더하는 steering vector 연구를 논문 순서대로 따라간다. Bolukbasi의 성별 방향, Subramani의 steering vector, ActAdd와 CAA가 차이 벡터를 만드는 방법을 비교한다. Park et al.의 선형 표현 가설이 이들을 어떻게 묶는지와 각 논문이 보이지 못한 것도 함께 정리한다."
---

[자연어처리 (3)](/posts/2026-10-18-nlp-3-evaluating-embeddings-glove)에서 analogy 문제를 풀었다.
man : woman :: king : ?의 답을 $\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}}$에
가장 가까운 단어로 찾는 방법이다. 그런데 이상하지 않은가? $\mathbf{v}_{\text{woman}} - \mathbf{v}_{\text{man}}$은
숫자 수백 개를 늘어놓은 것뿐인데 king에 더하면 "남성을 여성으로 바꾸는" 연산처럼 움직인다. 그렇다면
이런 차이 벡터를 단어 벡터가 아니라 언어 모델 안의 hidden state에 더하면 어떻게 될까?

> **차이 벡터 하나가 관계 하나를 담는다면 그 벡터를 언어 모델 안에 더했을 때 모델이 만드는 문장도 그 관계만큼 바뀌는가?**

이 질문은 2013년의 analogy 논문에서 시작해 2024년의 steering vector 논문과 선형 표현
가설(linear representation hypothesis)까지 이어진다. 이 글은 그 사이의 논문 일곱 편을 순서대로
읽으면서 차이 벡터를 만드는 방법과 더하는 자리, 더하는 크기를 비교한다. 각 논문이 숫자로 보인 것과
보이지 못한 것을 같이 적었다.

먼저 읽으면 좋은 글은 둘이다.

- [자연어처리 (2): Word2Vec, 주변 단어를 맞히며 배우는 임베딩](/posts/2026-10-18-nlp-2-word2vec):
  입력 임베딩 행렬 $\mathbf{W}$와 출력 행렬 $\mathbf{W}'$. 이 글의 뒤쪽에서 두 행렬의 구분이 다시 중요해진다.
- [자연어처리 (3): 임베딩 평가와 GloVe](/posts/2026-10-18-nlp-3-evaluating-embeddings-glove): analogy
  문제와 cosine으로 답을 고르는 방법.

## 차이 벡터를 어디에 더하는가

일곱 편을 먼저 한 표에 놓아 보자.

| 논문 | 차이 벡터의 재료 | 더하는 곳 | 모델 |
|---|---|---|---|
| Mikolov, Yih & Zweig (2013) | 단어 쌍 하나 | 단어 벡터 | RNN 언어 모델의 입력 weight |
| Bolukbasi et al. (2016) | 성별 단어 쌍 10개의 PCA | 단어 벡터 (빼기) | word2vec (Google News) |
| Subramani et al. (2022) | 문장마다 최적화한 벡터의 평균 차 | layer 7 hidden state | GPT-2 117M |
| Turner et al. (2023), ActAdd | prompt 쌍 하나의 activation 차 | layer $l$ residual stream | GPT-2-XL, OPT, LLaMA-3 |
| Panickssery et al. (2024), CAA | 객관식 쌍 290~1,000개의 activation 평균 차 | layer 13 근처 residual stream | Llama 2 Chat |
| Li et al. (2023), ITI | 참 답과 거짓 답 activation의 평균 차 | attention head 출력 | LLaMA 계열 |
| Park, Choe & Veitch (2024) | 단어 쌍 4~63개의 unembedding 평균 차 | 마지막 표현 $\lambda(x)$ | LLaMA-2 7B |

위의 둘은 단어 벡터를, 아래의 다섯은 언어 모델의 내부 표현을 다룬다. 마지막 Park et al.은 이 덧셈이
왜 되는지를 정리(theorem)로 설명하려 한다.

## 차이 벡터가 관계라는 가정

Mikolov, Yih & Zweig (2013)가 쓴 단어 벡터는 word2vec이 아니라 RNN 언어 모델의 입력층 weight다. one-hot 입력에 곱해지는 행렬
$\mathbf{U}$의 열 하나가 단어 하나의 벡터가 된다(논문 3절). 학습 데이터는 Broadcast News 320M 단어,
어휘는 82k다.

5절의 vector offset method는 다음 가정에서 출발한다.

> 같은 관계를 공유하는 단어 쌍은 임베딩 공간에서 모두 같은 상수 offset만큼 떨어져 있다.

이 가정이 맞다면 a : b :: c : ?의 답 d는 $\mathbf{x}_c + (\mathbf{x}_b - \mathbf{x}_a)$ 근처에 있어야 한다.
논문은 세 벡터를 단위 길이로 맞춘 뒤 이 점을 만들고 cosine이 가장 큰 단어를 고른다.

$$
\mathbf{y} = \mathbf{x}_b - \mathbf{x}_a + \mathbf{x}_c, \qquad w^\ast = \operatorname*{argmax}_{w}\, \frac{\mathbf{x}_w^\top \mathbf{y}}{\lVert\mathbf{x}_w\rVert\,\lVert\mathbf{y}\rVert} \tag{1}
$$

$\mathbf{x}_w$는 단어 $w$의 벡터다. [자연어처리 (3)](/posts/2026-10-18-nlp-3-evaluating-embeddings-glove)에서
본 식과 같은 모양이다. 식 (1)을 조금 다르게 읽어 보자. $\mathbf{x}_b - \mathbf{x}_a$를 한 번 계산해 두면
어떤 $\mathbf{x}_c$에든 더할 수 있다. **차이 벡터가 단어에 적용하는 연산이 되는 셈이다.** 이 글의
나머지 내용은 모두 이 한 줄에서 나온다.

가정은 얼마나 맞았을까? 논문이 만든 문법 analogy 문제 8,000개(형용사의 비교급과 최상급, 명사의
복수형과 소유격, 동사의 시제)의 정답률이다(Table 2).

| 차원 | 80 | 320 | 640 | 1600 |
|---|---|---|---|---|
| LSA | 12.8% | 16.5% | 11.3% | |
| RNN | 16.2% | 28.5% | 34.7% | 39.6% |

무작위로 고르면 1%에도 한참 못 미친다고 하니 RNN 벡터에 관계가 방향으로 들어 있는 것은 분명하다.
의미 관계를 재는 SemEval-2012 Task 2에서도 1600차원 벡터[^1]가 Spearman 0.275로 이전 최고 0.230을
넘었다(Table 4).

다만 숫자를 거꾸로 읽으면 가장 좋은 벡터도 문제 열 개 중 여섯 개를 틀린다. "모든 쌍이 같은
offset"은 대략적으로만 맞는 가정이고 논문은 offset이 왜 생기는지 설명하지 않는다.

## Bolukbasi: 쌍 열 개로 찾은 성별 방향

### 쌍 하나는 왜 불안정한가

Bolukbasi et al. (2016)는 Google News로 학습한 공개 word2vec 벡터(단어와 구 300만 개, 300차원)에
"man is to computer programmer as woman is to x"를 물으면 x = homemaker가 나온다는 결과로 유명하다(1절).
이 글에서 눈여겨볼 것은 편향 자체보다 성별 방향을 찾는 방법이다.

5.1절은 쌍 하나의 차를 그대로 쓰면 안 된다고 말한다. man은 "oh man!" 같은 감탄사로도, "man the
station" 같은 동사로도 쓰인다. grandfather에도 "to grandfather a regulation"이라는 동사 용법이 있다.
$\mathbf{v}_{\text{woman}} - \mathbf{v}_{\text{man}}$에 성별 말고 다른 성분이 섞이는 이유다. 논문은
she−he, her−his, woman−man, Mary−John, herself−himself, daughter−son, mother−father, gal−guy,
girl−boy, female−male의 열 쌍을 모아 이 잡음을 줄인다.

### PCA는 무엇을 최대화하는가

모으는 방법은 PCA다. 6절의 Step 1을 쌍에 맞춰 풀어 보자. $i$번째 쌍을 $D_i = \{\mathbf{x}_i, \mathbf{y}_i\}$,
평균을 $\boldsymbol\mu_i = (\mathbf{x}_i + \mathbf{y}_i)/2$라 하자. 논문이 SVD를 하는 행렬은

$$
\mathbf{C} = \sum_{i=1}^{n} \frac{1}{\lvert D_i\rvert}\sum_{\mathbf{w} \in D_i} (\mathbf{w} - \boldsymbol\mu_i)(\mathbf{w} - \boldsymbol\mu_i)^\top
$$

이다. 쌍에는 원소가 둘뿐이라 $\mathbf{x}_i - \boldsymbol\mu_i = \tfrac12\mathbf{d}_i$, $\mathbf{y}_i - \boldsymbol\mu_i = -\tfrac12\mathbf{d}_i$다.
여기서 $\mathbf{d}_i = \mathbf{x}_i - \mathbf{y}_i$는 쌍의 차이 벡터다. 넣어서 정리하면 다음과 같다.

$$
\mathbf{C} = \sum_{i=1}^{n} \frac12\Big(\tfrac14\mathbf{d}_i\mathbf{d}_i^\top + \tfrac14\mathbf{d}_i\mathbf{d}_i^\top\Big) = \frac14\sum_{i=1}^{n}\mathbf{d}_i\mathbf{d}_i^\top \tag{2}
$$

그러면 첫 번째 고유벡터 $\mathbf{g}$는 차이 벡터들을 투영한 값의 제곱합을 최대로 만드는 방향이다.

$$
\mathbf{g} = \operatorname*{argmax}_{\lVert\mathbf{u}\rVert = 1}\, \sum_{i=1}^{n} \big(\mathbf{u}^\top\mathbf{d}_i\big)^2 \tag{3}
$$

곧 차이 벡터 열 개에 가장 잘 맞는 방향 하나다. 열 개가 거의 같은 쪽을 가리키면 식 (3)의 답은 그
평균 방향과 거의 같아진다. 뒤에서 볼 CAA의 평균과 여기서 이어진다. 논문 그림 6에서 첫 주성분이
설명하는 분산이 나머지보다 뚜렷하게 커서 저자들은 $\mathbf{g}$ 하나가 성별을 담는다는 가설을 세운다.

### 방향을 빼는 Neutralize

찾은 방향은 지우는 데 쓴다. 6절의 Neutralize는 성별과 무관해야 할 단어(직업명 등)에서
$\mathbf{g}$ 성분을 빼고 길이를 다시 1로 맞춘다.

$$
\mathbf{w} \;\leftarrow\; \frac{\mathbf{w} - (\mathbf{w}^\top\mathbf{g})\,\mathbf{g}}{\big\lVert\mathbf{w} - (\mathbf{w}^\top\mathbf{g})\,\mathbf{g}\big\rVert} \tag{4}
$$

결과는 8절에 있다. she−he와 비슷한 관계로 자동 생성한 analogy 상위 150개 가운데 crowdworker 과반이
고정관념이라고 판정한 비율이 19%에서 6%로 줄었다. "he : doctor :: she : x"의 답은 nurse에서
physician으로 바뀌었다. 그러면서 analogy 정답률은 57.0에서 57.0, 유사도 평가 RG는 62.3에서 62.4로
거의 그대로였다(Table 1).

더하는 대신 빼는 쪽이지만 **방향 하나를 찾아 표현을 직접 고친다는 생각은 이미 여기에 있다.**
한계도 논문이 직접 적는다. receptionist가 football보다 softball에 가깝게 놓이는 것 같은 간접 편향은
정답이 없어 정량화하기 어렵다면서 정성적 예시로만 보인다. 성별이 정말 한 방향인지도 그림 6의
고윳값 감소에서 세운 가설이다.

## 단어 벡터에서 hidden state로

word2vec의 단어 벡터는 단어마다 하나로 정해져 있다. Transformer 언어 모델은 다르다. layer $l$에서
위치 $t$의 hidden state $\mathbf{h}^l_t \in \mathbb{R}^d$는 앞 문맥에 따라 매번 달라진다. 그래서 차이
벡터를 더할 자리도 단어 벡터에서 hidden state로 옮겨 간다.

$$
\mathbf{h}^l_t \;\leftarrow\; \mathbf{h}^l_t + c\,\mathbf{v} \tag{5}
$$

$\mathbf{v}$가 steering vector, $c$는 얼마나 세게 더할지 정하는 scalar다. 덧셈은 한 위치에서
일어나지만 효과는 거기서 끝나지 않는다. 뒤의 layer와 뒤의 token이 모두 바뀐 $\mathbf{h}^l_t$를 보고
계산하므로 다음 token의 분포가 달라지고 그 뒤의 생성 전체가 달라진다.

<figure className="my-6 text-center">
  <img src="/images/posts/2026-10-18-analogy-to-steering-vectors/offset-vs-steering.svg" alt="단어 벡터의 offset과 hidden state에 더하는 steering 방향 비교" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">그림 1. 왼쪽은 단어 쌍마다 같은 offset이 있다는 analogy의 가정, 오른쪽은 prompt 쌍의 activation 차를 사용자 prompt의 hidden state마다 더하는 steering. 두 그림의 벡터는 설명을 위해 만든 2차원 값이다.</figcaption>
</figure>

그림 1의 두 그림은 같은 연산을 그린다. 화살표를 더하는 대상이 고정된 단어 벡터인지 문맥마다 다른
hidden state인지만 다르다. 그러면 남는 질문은 둘이다. **$\mathbf{v}$는 어떻게 구하고 $l$과 $c$는
어떻게 정하는가?** 다음 세 논문은 이 두 질문에 서로 다르게 답한다.

## Subramani의 steering vector는 문장 하나를 만든다

Steering vector라는 이름은 Subramani et al. (2022)이 붙였고 ActAdd도 이 논문을 이름의 출처로 단다.
GPT-2 117M($d = 768$)을 고정해 두고 목표 문장 $\mathbf{x} = (x_1, \dots, x_T)$마다 벡터
$\mathbf{z}_{\text{steer}}$를 하나씩 찾는다. hidden state에 이 벡터를 더했을 때 모델이 바로 그 문장을
생성하도록 log-likelihood를 최대화한다. 논문의 식 (1)이다.

$$
\hat{\mathbf{z}}_{\text{steer}} = \operatorname*{argmax}_{\mathbf{z}_{\text{steer}}}\, \sum_{t=1}^{T} \log p\big(x_t \mid x_{<t},\, \mathbf{z}_{\text{steer}}\big) \tag{6}
$$

모델 weight는 그대로 두고 $\mathbf{z}_{\text{steer}}$만 Adam으로 update한다. 생성할 때는 문장 시작
token과 $\mathbf{z}_{\text{steer}}$만 넣고 greedy decoding을 한다. 어디에 더하는지가 결과를 가른다.
재구성한 문장과 원래 문장의 BLEU-4다(Table 1의 일부).

| 더한 위치 | 더한 시점 | BLEU-4 |
|---|---|---|
| 임베딩 layer | 모든 시점 | 33.99 |
| layer 6 (self-attention 뒤) | 모든 시점 | 100.0 |
| layer 6 (self-attention 뒤) | 첫 시점 | 99.80 |
| layer 7 (feed-forward 뒤) | 첫 시점 | 99.25 |
| 모든 layer | 첫 시점 | 91.72 |
| LM head 직전 | 모든 시점 | 6.72 |

12 layer의 가운데에 넣으면 첫 시점에 한 번만 더해도 문장이 거의 그대로 나온다. 입력 임베딩이나
출력 직전에 넣으면 거의 되지 않는다. 뒤의 ActAdd와 CAA도 가운데 layer를 쓴다.

4절에서 이 벡터로 analogy와 같은 산술을 한다. Yelp validation set에서 부정 문장 100개와 긍정 문장
100개의 steering vector를 각각 평균낸다. 그 차를 바꿀 문장의 벡터에 $\lambda$배 해서 더한다. 논문의
식 (2), (3)이다.

$$
\mathbf{z}_{\text{to target}} = \bar{\mathbf{z}}_{\text{target}} - \bar{\mathbf{z}}_{\text{source}}, \qquad \hat{\mathbf{z}}_{\text{target}} = \mathbf{z}_{\text{source}} + \lambda\,\mathbf{z}_{\text{to target}} \tag{7}
$$

$\mathbf{v}_{\text{king}} + (\mathbf{v}_{\text{woman}} - \mathbf{v}_{\text{man}})$과 같은 꼴이다. 차를 쌍
하나가 아니라 100개씩의 평균으로 잡고 더하는 크기를 $\lambda$로 조절한다는 점만 다르다. 이 실험에서는
7번째 self-attention layer 뒤, 첫 시점에 넣었다. Table 2의 예시를 옮기면 이렇다.

| $\lambda$ | 부정 → 긍정 | 긍정 → 부정 |
|---|---|---|
| 입력 | the desserts were very bland. | the taste is excellent! |
| 0.5 | the desserts were very bland. | the taste is excellent! |
| 1.0 | the desserts were very bland . | the taste is excellent! |
| 1.5 | the desserts were very tasty. | the taste is bitter and bitter taste is bitter taste is bitter |
| 2.0 | the desserts were very tasty. | the taste is unpleasant. |

두 예 모두 $\lambda$가 1 이하일 때는 그대로이다가 1.5에서 뒤집힌다. 긍정 → 부정 쪽은 같은 말을
되풀이한다. 논문도 $\lambda$가 크면 유창성이 떨어지고 단어나 구가 반복된다고 적는다. 논문 그림 3에서
$\lambda$를 키울수록 감정 분류 정확도는 오르고 원문과의 self-BLEU는 내려간다.

한계도 분명하다. 문장마다 gradient descent를 돌려야 벡터를 얻는다. 감정 바꾸기 성능은 autoencoder
기반 모델과 "비슷한" 수준이고 비지도 번역 기법을 쓴 방법들이 확실히 낫다는 것도 논문이 인정한다.

## ActAdd: prompt 한 쌍이면 충분한가

### 최적화 없이 activation 차를 더한다

Turner et al. (2023)의 ActAdd는 gradient descent를 아예 뺀다. Algorithm 1의 순서는 이렇다.

1. 원하는 성질을 담은 prompt $p_+$(예: "Love")와 반대 prompt $p_-$(예: "Hate")를 준비하고 짧은 쪽을
   공백으로 채워 token 길이를 맞춘다.
2. 두 prompt를 각각 forward pass하고 layer $l$의 입력, 곧 residual stream activation $\mathbf{h}^l_+$와
   $\mathbf{h}^l_-$를 기록한다.
3. 사용자 prompt $p^\ast$를 layer $l$까지 계산하고 그 activation에 차를 $c$배 해서 더한 뒤 forward
   pass를 이어 간다.

$$
\mathbf{h}^l_{p^\ast} \;\leftarrow\; \mathbf{h}^l_{p^\ast} + c\,\big(\mathbf{h}^l_+ - \mathbf{h}^l_-\big) \tag{8}
$$

$c$를 논문은 injection coefficient라 부른다. analogy로 읽으면 "Love" − "Hate"가 woman − man 자리에,
사용자 prompt의 activation이 king 자리에 들어간다. 그림 2가 이 흐름이다.

<figure className="my-6 text-center">
  <img src="/images/posts/2026-10-18-analogy-to-steering-vectors/steering-pipeline.svg" alt="prompt 쌍에서 steering vector를 만들고 layer l에 더하는 과정" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">그림 2. 위: prompt 쌍을 같은 모델에 넣어 layer l의 activation 차를 구한다. 아래: 사용자 prompt를 계산하다가 layer l의 입력에 그 차를 c배 해서 더하고 나머지 layer를 마저 계산한다.</figcaption>
</figure>

식 (8)에서 짚어 둘 것이 있다. $\mathbf{h}^l_\pm$는 벡터 하나가 아니라 prompt의 token 위치마다 하나씩
있는 activation 묶음이다. ActAdd는 이 묶음을 사용자 prompt의 앞쪽 위치부터 맞춰 더한다(부록 C의 정렬
위치 $a = 1$). 그래서 steering prompt가 token 두 개면 사용자 prompt의 처음 몇 위치만 바뀐다. 게다가
첫 위치는 tokenizer가 앞에 붙이는 endoftext token이라 두 prompt에서 같으므로 차가 0이다(부록 F).
뒤에서 볼 CAA가 벡터 하나를 뒤쪽 모든 위치에 더하는 것과 다르다.

### 숫자로 본 효과

GPT-2-XL(1.5B)에 결혼 주제 벡터를 넣으면 layer 6 근처에서 효과가 가장 커서 생성문의 90% 넘게 결혼
관련 단어가 들어갔다. 조작하지 않은 모델은 2%다(4.1.3절). 독성 줄이기는 RealToxicityPrompts에서
1,000개를 뽑아 Perspective API로 쟀다(Table 3). steering vector는 "Love" − "Hate"다.

| 모델 | 방법 | toxicity ↓ | disfluency ↓ | relevance ↑ |
|---|---|---|---|---|
| OPT 6.7B | 조작 없음 | .134 | 8.9 | .369 |
| OPT 6.7B | prompting | .200 | 54.3 | .294 |
| OPT 6.7B | PREADD-D | .122 | 56.6 | .326 |
| OPT 6.7B | **ActAdd** | **.112** | 13.8 | .329 |
| LLaMA-3-8B | 조작 없음 | .114 | 6.3 | .391 |
| LLaMA-3-8B | **ActAdd** | **.108** | 6.7 | .365 |

IMDb 감정 바꾸기(Table 4)에서는 LLaMA-3-8B의 부정 → 긍정 성공률이 0.417에서 0.669로 올랐다.

### 손으로 고른 $l$과 $c$, 그리고 실패한 벡터

표의 굵은 숫자만 보면 깔끔하지만 조건을 같이 읽어야 한다.

먼저 $l$과 $c$는 사람이 고른다. 재현성 절은 대개 $c \in [3, 20]$, $l \in [6, 24]$에서 단순 grid search를
했다고 쓰고 이 탐색에는 "완전한 통계가 없다"고 밝힌다. 정성 예시(부록 Table 6)는 seed 0으로 만든 처음
세 completion 가운데 가장 그럴듯한 것을 골랐다.

둘째, 듣지 않는 벡터가 있다. 부록 Table 7의 실패 예다.

| $p_+$ | $p_-$ | $l$ | $c$ | 결과 |
|---|---|---|---|---|
| I talk about weddings constantly | I do not talk about weddings constantly | 20 | +100 | 결혼 이야기는 나오지만 문장이 깨진다 |
| Love | (공백) | 10 | +3 | "I hate you because" 뒤가 여전히 욕설이다 |
| Whenever I say the word goose I follow it with the special phrase AAAAHHHH | I can say goose | 10 | +7 | 의도한 문구 대신 부서진 문장이 나온다 |

두 번째 줄이 흥미롭다. "Love"만 넣고 반대쪽을 공백으로 두면 듣지 않는데 "Love" − "Hate"는 듣는다.
반대 prompt가 있어야 방향이 정해진다. 부록 E.2에서는 Llama-1-13B로 옮겼을 때 Paris → Rome, anger,
harm 벡터가 재현되지 않았다.

셋째, 비교 조건이 같지 않다. Table 3의 PREADD와 prompting 수치는 Pei et al. (2023)이 보고한 값이다.
재현성 절에 따르면 ActAdd는 frequency penalty 1.0, `top_p` 0.3으로 생성했는데 옮겨 온 숫자는 frequency
penalty 0, `top_p` 1.0에서 나왔다.

## CAA: 쌍 수백 개의 평균

### 평균 차 벡터

Panickssery et al. (2024)에는 ActAdd의 제1저자 Turner가 공저자로 들어 있다. 관련 연구 절은 ActAdd가
행동마다 일관되게 듣지 않고 prompt가 바뀌면 흔들린다고 지적한다[^2]. 해법은 쌍 하나 대신 데이터셋
전체의 평균이다. Bolukbasi가 쌍 열 개를 모은 것과 같은 이유다.

쌍을 만드는 방법이 영리하다. 같은 객관식 질문 끝에 목표 행동에 해당하는 답 글자("(A")를 붙인
prompt와 반대 답 글자("(B")를 붙인 prompt를 만든다. 두 prompt는 마지막 token 하나만 다르다. 답 글자
위치에서 layer $l$의 residual stream activation을 뽑아 차를 평균낸다. 논문의 식 (1)이다.

$$
\mathbf{v}_{\text{MD}} = \frac{1}{\lvert\mathcal{D}\rvert} \sum_{(p,\, y_+,\, y_-) \in \mathcal{D}} \Big[\mathbf{a}_l(p, y_+) - \mathbf{a}_l(p, y_-)\Big] \tag{9}
$$

$p$는 질문, $y_+$와 $y_-$는 행동에 맞는 답 글자와 반대 답 글자, $\mathbf{a}_l$은 그 답 글자 위치의
activation이다(논문은 layer를 $L$로 쓴다). MD는 Mean Difference의 약자다. 생성할 때는 prompt가 끝난
뒤의 모든 token 위치에 $\mathbf{v}_{\text{MD}}$를 multiplier만큼 곱해 더한다. 음수를 곱하면 행동을
줄이는 쪽으로 간다.

논문은 MD 벡터가 PCA 같은 다른 방법으로 얻은 벡터와 비슷하다는 Tigges et al. (2023)을 인용한다.
식 (3)에서 본 대로 차이 벡터들이 한 방향으로 모여 있으면 Bolukbasi의 PCA와 CAA의 평균은 같은 방향을
찾는 두 가지 계산이 된다.

실험은 Llama 2 7B Chat과 13B Chat에서 했다. 행동은 일곱 가지(AI Coordination, Corrigibility,
Hallucination, Myopic Reward, Survival Instinct, Sycophancy, Refusal)다. 행동마다 쌍 290~1,000개로
벡터를 만들고(부록 Table 9) 따로 떼어 둔 객관식 50문항으로 평가했다. layer를 전부 훑었더니 7B는
layer 13 근처, 13B는 대개 14나 15에서 효과가 가장 컸다(4.1절).

### multiplier를 키우면 행동도 늘어나는가

<figure className="my-6 text-center">
  <img src="/images/posts/2026-10-18-analogy-to-steering-vectors/caa-multiplier.svg" alt="CAA multiplier에 따른 행동별 답 확률" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">그림 3. multiplier를 −1에서 +1로 바꾸면 행동에 맞는 답의 확률이 오르는가? Sycophancy는 +1에서 오히려 내려간다. Panickssery et al. (2024) Table 3에서 Llama 2 13B Chat, layer 13, system prompt 없음 열을 옮겨 그렸다.</figcaption>
</figure>

그림 3의 세로축은 50문항에서 행동에 맞는 답 글자에 모델이 준 확률의 평균이다[^3]. 일곱 행동 중
여섯은 multiplier를 −1에서 +1로 바꾸면 확률이 단조롭게 오른다. Corrigibility는 0.45, 0.57, 0.83이다.

Sycophancy는 0.56, 0.63, 0.60이다. +1을 곱해 더했는데 steering하지 않았을 때보다 낮다. **같은 방법으로
만든 평균 차 벡터라도 행동마다 듣는 정도가 다르다.** 일반 능력은 거의 그대로였다. MMLU(Table 5, 13B,
layer 14)에서 정답에 준 평균 확률은 steering 없이 0.63, +1에서 0.59~0.65, −1에서 0.57~0.64였다.

한계는 논문 10절과 4.2절에 있다. multiplier를 키우면 생성문 품질이 떨어져서 범위를 좁혔다. 그리고
residual stream의 norm이 layer를 지날수록 커지는데 layer마다 multiplier를 따로 찾지 않았으니 "최적
layer"라는 결론이 흔들릴 수 있다고 저자들 스스로 적는다.

### 같은 무렵의 ITI

Li et al. (2023)의 ITI(Inference-Time Intervention)도 평균 차를 쓰지만 residual stream이 아니라
attention head의 출력에 더한다. linear probe 정확도로 고른 상위 $K$개 head에서만 참 답과 거짓 답
activation의 평균을 잇는 방향으로 표준편차의 $\alpha$배만큼 옮긴다. $K = 48$, $\alpha = 15$에서
Alpaca의 TruthfulQA `true*informative`가 32.5%에서 65.1%로 올랐다.

## 선형 표현이란 무엇을 말하는가

### 세 가지 뜻

여기까지 보면 질문이 생긴다. 차이 벡터는 왜 이렇게 잘 듣는가? 그리고 언제 듣지 않는가? Park, Choe &
Veitch (2024)는 이 질문을 정리로 다룬다. 1절은 "linear representation"이라는 말이 세 가지 뜻으로
쓰여 왔다고 정리한다.

| 뜻 | 내용 | 이 글에서 본 예 |
|---|---|---|
| subspace | woman − man과 queen − king 같은 차가 한 방향에 놓인다 | Mikolov et al., Bolukbasi et al. |
| measurement | linear probe로 개념의 확률을 읽을 수 있다 | ITI의 head별 probe |
| intervention | 어떤 벡터를 더하면 다른 개념은 그대로 두고 그 개념만 바뀐다 | ActAdd, CAA |

논문은 셋이 서로 어떻게 이어지는지 처음부터 분명하지는 않다고 쓴다. 그리고 셋을 같은 언어로
정의한 뒤 그 사이를 정리로 잇는다.

### 두 공간: embedding과 unembedding

언어 모델은 문맥 $x$를 벡터 $\lambda(x) \in \mathbb{R}^d$로 보내고 단어 $y$마다 벡터
$\gamma(y) \in \mathbb{R}^d$를 둔다. 다음 단어 분포는 softmax다.

$$
P(y \mid x) \propto \exp\big(\lambda(x)^\top \gamma(y)\big) \tag{10}
$$

논문은 $\lambda$를 embedding, $\gamma$를 unembedding이라 부른다. 이름이 헷갈리기 쉬운데 $\lambda(x)$는
입력 token의 임베딩이 아니라 문맥 전체를 거쳐 softmax 바로 앞에 나온 표현이다. $\gamma(y)$는 출력
행렬에서 단어 $y$에 해당하는 벡터다. [자연어처리 (2)](/posts/2026-10-18-nlp-2-word2vec)의 skip-gram
$P(o \mid c) \propto \exp(\mathbf{u}_o^\top\mathbf{v}_c)$와 나란히 놓으면 center word 벡터
$\mathbf{v}_c$(입력 행렬 $\mathbf{W}$의 행)가 $\lambda(x)$ 자리에, context word 벡터 $\mathbf{u}_o$(출력
행렬 $\mathbf{W}'$의 열)가 $\gamma(y)$ 자리에 온다.

개념은 male⇒female처럼 값이 둘인 변수 $W$로 다룬다. $W$의 counterfactual 쌍 $(Y(0), Y(1))$은
("man", "woman"), ("king", "queen")처럼 그 개념만 다른 단어 쌍이다. 두 개념 $W$, $Z$를 서로 독립적으로
바꿀 수 있으면 causally separable하다고 한다. English⇒French와 male⇒female은 "king", "queen", "roi",
"reine"가 다 있으니 separable하다. English⇒French와 English⇒Russian은 따로 바꿀 수 없다.

**정의 2.1 (unembedding representation).** $W$의 모든 counterfactual 쌍에서

$$
\gamma(Y(1)) - \gamma(Y(0)) \in \mathrm{Cone}(\bar\gamma_W) = \{\kappa\,\bar\gamma_W : \kappa > 0\}
$$

이면 $\bar\gamma_W$를 $W$의 unembedding representation이라 한다. queen − king이 woman − man과 같은
방향이라는 analogy의 가정을 출력 단어 공간에서 정의로 삼았다. 방향만 같으면 되고 길이 $\kappa$는
쌍마다 달라도 된다.

이 정의만으로 measurement가 나온다(정리 2.2). 단어가 $Y(0)$과 $Y(1)$ 중 하나라고 알 때 $Y(1)$일
logit은 식 (10)에서 두 확률의 비를 구해 log를 씌우면 나온다.

$$
\operatorname{logit} P\big(Y = Y(1) \mid Y \in \{Y(0), Y(1)\}, \lambda\big) = \lambda^\top\big(\gamma(Y(1)) - \gamma(Y(0))\big) = \kappa\,\lambda^\top\bar\gamma_W \tag{11}
$$

정규화 상수가 분자와 분모에서 지워져 첫 등호가 성립하고 둘째 등호는 정의 2.1이다. 그러니
$\bar\gamma_W$는 개념 $W$를 읽는 linear probe 방향이다.

**정의 2.3 (embedding representation).** "He is the monarch of England, "과 "She is the monarch of England, "처럼
목표 개념의 분포만 다르고 다른 개념의 분포는 같은 문맥 쌍 $\lambda_0, \lambda_1$을 생각한다. 그런 쌍의
차 $\lambda_1 - \lambda_0$가 모두 $\mathrm{Cone}(\bar\lambda_W)$에 들어가면 $\bar\lambda_W$가 embedding
representation이다. 문맥 쌍의 표현 차를 쓴다는 점에서 ActAdd의 prompt 쌍과 같은 생각이다.

### 정리 2.5: 왜 다른 개념은 움직이지 않는가

보조정리 2.4는 두 표현 사이의 관계를 준다. $Z$가 $W$와 causally separable하면

$$
\bar\lambda_W^\top\bar\gamma_W > 0, \qquad \bar\lambda_W^\top\bar\gamma_Z = 0
$$

이다. 이제 $\lambda$에 $c\,\bar\lambda_W$를 더해 보자. $Z$의 값을 $z$로 고정한 쌍 $(Y(0, z), Y(1, z))$는
$W$의 counterfactual 쌍이므로 차가 $\kappa\bar\gamma_W$다. 마찬가지로 $W$를 $w$로 고정한 쌍
$(Y(w, 0), Y(w, 1))$의 차는 $\kappa'\bar\gamma_Z$다($\kappa, \kappa' > 0$). 식 (11)에 넣으면

$$
\begin{aligned}
\operatorname{logit}_W(c) &= \kappa\,\lambda^\top\bar\gamma_W + c\,\kappa\,\bar\lambda_W^\top\bar\gamma_W \\
\operatorname{logit}_Z(c) &= \kappa'\,\lambda^\top\bar\gamma_Z + c\,\kappa'\,\bar\lambda_W^\top\bar\gamma_Z = \kappa'\,\lambda^\top\bar\gamma_Z
\end{aligned} \tag{12}
$$

이다. $\operatorname{logit}_W(c)$는 $Y(0, z)$와 $Y(1, z)$ 중 $Y(1, z)$를 고를 logit이고
$\operatorname{logit}_Z(c)$도 같은 방식으로 정의한다. 첫 줄의 기울기 $\kappa\,\bar\lambda_W^\top\bar\gamma_W$는
양수이고 둘째 줄의 기울기는 0이다. 이것이 정리 2.5다. **$\bar\lambda_W$를 더하면 목표 개념의 logit은
$c$에 비례해 오르고 따로 바꿀 수 있는 개념의 logit은 움직이지 않는다.** steering vector가 "다른 것은
건드리지 않고 그 성질만 바꾼다"는 주장이 이 조건 아래에서 증명된 셈이다.

식 (12)를 성립시키는 것은 $\bar\lambda_W^\top\bar\gamma_Z = 0$ 하나다. 그런데 analogy처럼 출력 쪽 차이
벡터 $\bar\gamma_W$를 그대로 더하면 어떻게 될까? 기울기가 $\bar\gamma_W^\top\bar\gamma_Z$가 되므로 두
방향이 Euclidean 의미로 직교하지 않는 한 $Z$도 같이 움직인다.

<figure className="my-6 text-center">
  <img src="/images/posts/2026-10-18-analogy-to-steering-vectors/park-geometry.svg" alt="embedding representation과 unembedding representation을 더했을 때 logit 변화 비교" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">그림 4. 왼쪽: 회색 선은 Z의 logit이 같은 점들이다. embedding representation λ̄ 방향으로 가면 그 선을 따라 움직이고 unembedding representation γ̄ 방향으로 가면 선을 가로지른다. 오른쪽: 두 방향으로 c만큼 더했을 때 W와 Z의 logit 변화. 두 개념의 γ̄는 (1, 0.25)와 (0.55, 1)로 설명을 위해 만든 값이다.</figcaption>
</figure>

그림 4가 이 차이를 보여 준다. 만든 2차원 예에서 $\bar\gamma_W^\top\bar\gamma_Z = 0.8$이라 $\bar\gamma_W$를
더하면 $Z$의 logit이 $0.8c$만큼 같이 오른다. $\bar\lambda_W$는 $\bar\gamma_Z$와 직교하도록 고른
방향이어서 $Z$의 logit이 0에 머문다. 그렇다면 $\bar\gamma_W$에서 $\bar\lambda_W$를 어떻게 얻을까?
다음 절의 내적 문제가 여기서 나온다.

## 내적은 누가 정하는가

### 학습은 내적을 정해 주지 않는다

analogy에서 cosine을 쓸 때 우리는 Euclidean 내적을 당연하게 썼다. Park et al. 3절은 이 내적이 학습으로
정해지는 양이 아니라고 지적한다. 가역 행렬 $\mathbf{A}$와 상수 벡터 $\boldsymbol\beta$로 두 공간을
바꿔 보자(논문 식 3.1).

$$
\gamma(y) \mapsto \mathbf{A}\gamma(y) + \boldsymbol\beta, \qquad \lambda(x) \mapsto \mathbf{A}^{-\top}\lambda(x)
$$

그러면 $\lambda^\top\mathbf{A}^{-1}(\mathbf{A}\gamma + \boldsymbol\beta) = \lambda^\top\gamma + \lambda^\top\mathbf{A}^{-1}\boldsymbol\beta$다.
둘째 항은 $y$와 무관해서 softmax에서 지워지므로 식 (10)의 확률은 그대로다. 학습의 loss는 이 확률만
보기 때문에 $\mathbf{A}$를 정해 주지 못한다. 반면 Euclidean 내적은 $\langle\bar\gamma_W, \bar\gamma_Z\rangle$에서
$\langle\mathbf{A}\bar\gamma_W, \mathbf{A}\bar\gamma_Z\rangle$로 바뀐다. **같은 모델을 두 좌표계로 적었을
뿐인데 cosine이 달라진다.**

### causal inner product: 따로 바뀌는 개념끼리 직교

그래서 논문은 내적을 원리로 고른다. causally separable한 개념끼리 직교하게 만드는 내적
$\langle\cdot, \cdot\rangle_{\mathrm{C}}$를 causal inner product라 한다(정의 3.1). 내적을
$\langle\mathbf{a}, \mathbf{b}\rangle_{\mathrm{C}} = \mathbf{a}^\top\mathbf{M}\,\mathbf{b}$($\mathbf{M}$은 대칭 양의 정부호)로 쓰면
정리 3.2는 다음을 말한다.

$$
\bar\lambda_W = \mathbf{M}\,\bar\gamma_W
$$

unembedding representation에 $\mathbf{M}$을 곱하면 embedding representation, 곧 더할 방향이 된다.
보조정리 2.4의 조건이 실제로 맞는지 확인해 보자. $\bar\lambda_W^\top\bar\gamma_Z = \bar\gamma_W^\top\mathbf{M}\bar\gamma_Z = \langle\bar\gamma_W, \bar\gamma_Z\rangle_{\mathrm{C}} = 0$이고
$\bar\lambda_W^\top\bar\gamma_W = \langle\bar\gamma_W, \bar\gamma_W\rangle_{\mathrm{C}} > 0$이다. 그림 4의
$\bar\lambda_W$도 이렇게 계산했다.

$\mathbf{M}$은 어떻게 구할까? 논문은 어휘에서 균등하게 뽑은 단어가 French인지 안다고 해서 그 단어가
남자를 가리키는지는 거의 알 수 없다고 가정한다(가정 3.3). 이 가정 아래 정리 3.4는 $\mathbf{M}^{-1} = \mathbf{G}\mathbf{G}^\top$과
$\mathbf{G}^\top\mathrm{Cov}(\gamma)^{-1}\mathbf{G} = \mathbf{D}$를 준다. $\mathbf{G}$는 개념 표현들을 열로
모은 행렬, $\mathbf{D}$는 양수 대각행렬이다.

여기서 $\mathbf{D}$는 자유 매개변수이고 논문은 $\mathbf{D}$를 하나로 고를 원리가 없다고 쓴다. 실험에서는
$\mathbf{D} = \mathbf{I}$를 택해 $\mathbf{M} = \mathrm{Cov}(\gamma)^{-1}$을 쓴다. 논문의 식 (3.3)과 (4.1)이다.

$$
\langle\bar\gamma, \bar\gamma'\rangle_{\mathrm{C}} := \bar\gamma^\top\mathrm{Cov}(\gamma)^{-1}\bar\gamma', \qquad \bar\lambda_W := \mathrm{Cov}(\gamma)^{-1}\bar\gamma_W \tag{13}
$$

$\mathrm{Cov}(\gamma)$는 어휘 전체의 unembedding 벡터로 계산한 공분산이다.

### LLaMA-2 7B에서 확인한 것

실험 모델은 LLaMA-2 7B(어휘 32,000개, 4,096차원)다. 개념은 27개로, analogy 문제 모음인 BATS 3.0의
범주에서 온 22개(verb⇒3pSg, male⇒female, country⇒capital 등)와 언어 쌍 4개, frequent⇒infrequent 하나다.
$\bar\gamma_W$는 개념마다 counterfactual 쌍의 차를 평균내서 추정한다. 쌍은 단일 token 단어만 써서
개념당 4~63개이고 male⇒female은 11개다(부록 Table 2).

- 쌍 하나를 빼고 추정한 방향에 그 쌍의 차를 투영하면 무작위 단어 쌍보다 뚜렷하게 크다. 예외는
  thing⇒part로, 선형 표현이 없는 것으로 보인다.
- 추정한 내적에서 causally separable한 개념끼리는 거의 직교한다.
- 식 (13)의 $\bar\lambda_W$를 $\lambda(x)$에 $\alpha$배 더하면 male⇒female 방향에서 queen 대 king의 log
  확률비는 오르고 King 대 king의 비는 그대로다(논문 그림 5).

"Long live the " 뒤의 다음 단어 상위 5개가 $\alpha$에 따라 바뀌는 모습이다(Table 1).

| 순위 | $\alpha = 0$ | 0.1 | 0.2 | 0.3 | 0.4 |
|---|---|---|---|---|---|
| 1 | king | Queen | queen | queen | queen |
| 2 | King | queen | Queen | Queen | Queen |
| 3 | Queen | king | `_` | lady | lady |
| 4 | queen | King | lady | woman | woman |
| 5 | `_` | `_` | king | women | women |

`_`는 논문 표에 밑줄로 찍힌 token이다. $\alpha = 0.2$에서 king이 5위로 밀리고 0.3부터는 상위 5개에서
빠진다.

부록 D.2에는 뜻밖의 결과도 있다. LLaMA-2에서는 Euclidean 내적으로도 causally separable한 개념
대부분이 꽤 직교했다(저자들도 "놀랍게도"라고 쓴다). Gemma-2B에서는 Euclidean 내적이 의미 구조를
놓쳤고 causal inner product는 잡았다.

## 아직 이어지지 않은 고리

Park et al.로 analogy와 steering이 한 그림에 들어왔다. 하지만 그 그림과 실제 steering 논문 사이에는
빈 곳이 남는다.

**입력 벡터와 출력 벡터.** word2vec 배포본이 저장하는 단어 벡터는 입력 쪽 행렬이다. `word2vec.c`의
"Save the word vectors" 부분은 `syn0`, 곧 [자연어처리 (2)](/posts/2026-10-18-nlp-2-word2vec)의
$\mathbf{W}$를 파일에 쓴다. Mikolov, Yih & Zweig도 RNN의 입력 weight $\mathbf{U}$를 썼다. 그런데 Park
et al.의 subspace 정의(정의 2.1)는 출력 쪽 $\gamma$에 있고 steering은 문맥 쪽 $\lambda$에 더한다. 그래서
"analogy가 되니 steering도 된다"로 곧장 넘어갈 수 없다. 둘을 잇는 것은 정리 3.2와 causal inner
product다. 그 내적도 $\mathbf{D} = \mathbf{I}$로 고른 여러 후보 중 하나다.

**마지막 layer와 중간 layer.** Park et al.의 $\lambda(x)$는 softmax 바로 앞의 표현이다. 논문 5절은 중간
layer의 activation 해석은 다루지 않는다고 분명히 적는다. ActAdd와 CAA가 실제로 더하는 곳은 layer 6이나
13 같은 중간 residual stream이다. 정리 2.5가 그 자리를 직접 설명하지는 않는다.

**개념의 범위.** 정리의 개념은 값이 둘인 변수이고 실험의 개념은 단일 token 단어 쌍으로 정의된다.
Sycophancy나 Refusal처럼 문장 전체에 걸친 행동이 이 틀에 들어맞는지는 이 일곱 편 어디에서도 보이지
않았다. 정리 2.5는 embedding representation이 있다고 가정하고 출발할 뿐 그것이 있다는 것을 보이지는
않는다.

## 정리

일곱 편의 차이를 다시 정리하면 이렇다. 차이 벡터의 재료는 쌍 하나(Mikolov, ActAdd), 쌍 열 개의
주성분(Bolukbasi), 쌍 수백 개의 평균(CAA)으로 갈린다. 더하는 자리는 단어 벡터에서 언어 모델의 중간
layer로 옮겨 갔다. 크기는 $\lambda$, $c$, multiplier라는 이름으로 늘 사람이 정했다.

처음 질문으로 돌아가자. 차이 벡터를 언어 모델 안에 더하면 문장이 그 관계만큼 바뀌는가? 감정, 주제,
독성, 일부 행동에서는 그렇다. 정리 2.5는 개념들이 따로 바뀔 수 있고 방향을 알맞은 내적으로 잡았을 때
왜 그런지 보여 준다. 듣지 않는 벡터(ActAdd Table 7)와 거꾸로 움직인 행동(CAA의 Sycophancy)은 정리
밖에 있다. **조건이 맞는지는 개념마다, layer마다 따로 확인해야 하고 지금까지의 실험은 그 확인을 일부만
했다.**

## 이어지는 글

- [자연어처리 (2): Word2Vec, 주변 단어를 맞히며 배우는 임베딩](/posts/2026-10-18-nlp-2-word2vec)
- [자연어처리 (3): 임베딩 평가와 GloVe](/posts/2026-10-18-nlp-3-evaluating-embeddings-glove)

## 참고문헌

- Mikolov, T., Yih, W., Zweig, G. (2013). *Linguistic Regularities in Continuous Space Word Representations*. NAACL-HLT.
- Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., Kalai, A. (2016). *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings*. arXiv:1607.06520.
- Subramani, N., Suresh, N., Peters, M. E. (2022). *Extracting Latent Steering Vectors from Pretrained Language Models*. arXiv:2205.05124.
- Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., MacDiarmid, M. (2023). *Steering Language Models With Activation Engineering*. arXiv:2308.10248 (v5, 2024).
- Panickssery, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., Turner, A. M. (2024). *Steering Llama 2 via Contrastive Activation Addition*. arXiv:2312.06681.
- Li, K., Patel, O., Viégas, F., Pfister, H., Wattenberg, M. (2023). *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model*. NeurIPS.
- Park, K., Choe, Y. J., Veitch, V. (2024). *The Linear Representation Hypothesis and the Geometry of Large Language Models*. ICML.
- word2vec 원본 C 코드 `word2vec.c` (Mikolov et al., 단어 벡터 저장 부분).

[^1]: 1600차원은 여러 시스템의 벡터를 이어 붙인 것이다(논문 6절). 모든 RNN 벡터는 같은 Broadcast News 데이터로 학습했다.
[^2]: CAA 논문(arXiv v4, 2024년 7월)은 ActAdd가 GPT-2-XL에서만 실험되었다고도 적는다. 이 글이 읽은 ActAdd 논문(arXiv v5, 2024년 10월)에는 OPT와 LLaMA-3 결과가 더해져 있다.
[^3]: 4.1절에 따르면 13B의 최적 layer는 대개 14나 15인데 Table 3은 layer 13에서 steering했다. 같은 논문의 MMLU 평가(Table 5)는 layer 14를 썼다.
