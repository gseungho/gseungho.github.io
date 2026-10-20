---
title: "자연어처리 (4): subword, 문장 벡터, 문맥화 임베딩"
date: "2026-10-18"
category: "공부"
subcategory: "Deep Learning/Natural Language Processing"
tags: ["자연어처리", "fastText", "attention", "ELMo"]
description: "단어를 character n-gram의 합으로 나타내는 fastText와 BPE로 처음 보는 단어 문제를 푼다. 단어 벡터를 합, 평균, 최댓값, 가중합, attention으로 합쳐 문장 벡터를 만들고 각 방법이 잃는 것을 같은 예제로 비교한다. 정적 임베딩이 다의어를 가르지 못하는 이유와 문맥화 임베딩을 정리하며 단어 임베딩 단원을 마친다."
---

자연어처리 시리즈의 네 번째 글이다. 세종대학교 2026년 2학기 자연어처리 수업을 따라가며 공부한 내용이고 Stanford [CS224N](https://web.stanford.edu/class/cs224n/) 강의 노트와 Jurafsky & Martin의 [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)(3판 초안)을 함께 참고했다.

> **단어 벡터 다음의 문제: 처음 보는 단어, 길이가 제각각인 문장, 뜻이 여러 개인 단어는 어떻게 벡터로 나타낼까?**

[3편](/posts/2026-10-18-nlp-3-evaluating-embeddings-glove)은 어휘에 없는 단어가 모두 UNK 하나로 뭉개진다는 문제로 끝났다. 이번 글은 단어를 더 작은 조각으로 쪼개 이 문제를 푼다. 이어서 단어 벡터 여러 개를 문장 벡터 하나로 합치는 방법을 보고 같은 단어라도 context마다 다른 벡터를 주는 문맥화 임베딩으로 단어 임베딩 단원을 마무리한다. skip-gram의 기호는 [2편](/posts/2026-10-18-nlp-2-word2vec)과 같다.

## 단어 안에도 구조가 있다

friend, friended, friendless, friendly, friendship, unfriend, unfriendly를 보자. 일곱 단어는 friend라는 조각을 공유한다. [3편](/posts/2026-10-18-nlp-3-evaluating-embeddings-glove)의 GloVe 6B 어휘를 뒤져 보면 여섯 개는 있지만 unfriend는 없다. 그래도 사람은 unfriend를 처음 봐도 un + friend로 뜻을 짐작한다.

word2vec과 GloVe는 이런 내부 구조를 보지 않는다. D2L 15.6절의 말대로 helps, helped, helping은 파라미터를 하나도 나누지 않는 별개의 벡터다. 영어는 그나마 낫다. 같은 절에 따르면 프랑스어와 스페인어 동사는 활용형이 40개가 넘고 핀란드어 명사는 격이 15개까지 된다. 한국어도 먹다, 먹었다, 먹었고, 먹으니처럼 어미가 붙을 때마다 형태가 바뀐다. **형태마다 벡터를 따로 배우면 드문 형태는 학습 신호가 부족하고 아예 못 본 형태는 UNK가 된다.**

## fastText: 단어를 조각 벡터의 합으로

fastText(Bojanowski et al., 2017)는 단어를 character n-gram의 묶음으로 본다. 3.2절의 방법은 이렇다. 단어 앞뒤에 경계 기호 `<`와 `>`를 붙이고 길이 3~6인 조각을 모두 뽑는다. 단어 자체도 조각 목록에 하나로 넣는다. where를 쪼개면 그림 1이다.

<Fig src="/images/posts/2026-10-18-nlp-4-subword-composition-contextual/fasttext-where.svg" alt="where를 경계 기호와 함께 3-gram부터 6-gram, 단어 자체까지 15개 조각으로 나눈 그림">그림 1. where는 몇 개의 조각으로 나뉘는가? 3-gram 5개, 4-gram 4개, 5-gram 3개, 6-gram 2개에 단어 자체까지 15개다.</Fig>

경계 기호를 붙이면 글자가 $L + 2$개이므로 길이 $n$인 조각은 $L + 3 - n$개다. where는 $L = 5$라 5 + 4 + 3 + 2개에 `<where>` 하나를 더해 15개다. friend는 $L = 6$이라 6 + 5 + 4 + 3 + 1 = 19개다. 단어 $w$의 조각 집합을 $\mathcal{G}_w$, 조각 $g$의 벡터를 $\mathbf{z}_g$라 하면 단어 벡터는 조각 벡터의 합이다.

$$
\mathbf{v}_w = \sum_{g\in\mathcal{G}_w}\mathbf{z}_g, \qquad s(w, c) = \sum_{g\in\mathcal{G}_w}\mathbf{z}_g^\top\mathbf{u}_c \tag{1}
$$

skip-gram의 center word 벡터 $\mathbf{v}_w$ 자리에 이 합을 넣은 것이고 나머지는 그대로다(D2L 15.6.1절). $s(w, c)$는 center word $w$와 context word $c$의 점수로 논문 3.2절의 식이다(논문은 context word 벡터를 $\mathbf{v}_c$로 쓴다).

경계 기호에는 역할이 있다. 논문은 where 안의 3-gram `her`와 단어 her를 나타내는 `<her>`가 서로 다른 조각이라고 강조한다. `<wh`는 단어 첫머리에서만, `re>`는 끝에서만 나오니 접두사와 접미사도 구별된다.

### 조각을 나눠 쓰는 이점

식 (1)의 합을 미분하면 각 항의 계수가 1이라서 $\partial J/\partial\mathbf{z}_g = \partial J/\partial\mathbf{v}_w$가 $\mathcal{G}_w$의 모든 조각에 똑같이 전해진다. friend라는 조각은 friendly, friendship, unfriendly 가운데 어느 단어가 학습되든 update된다. **드문 단어도 흔한 조각 덕분에 학습 신호를 받는다.**

처음 보는 단어에는 단어 자체의 벡터가 없으니 아는 조각의 벡터만 더한다. 논문 5.1절의 단어 유사도 실험(Table 1)에서 이렇게 벡터를 만들어 준 경우(sisg)는 0벡터를 준 경우(sisg-)보다 늘 같거나 나았다. 메모리를 아끼려고 조각은 hashing으로 $K = 2\times10^6$개 칸에 나눠 담고 같은 칸에 든 조각은 벡터를 공유한다(3.2절).

형태 정보가 실제로 듣는 곳은 문법 쪽이다. 영어 analogy에서 문법 문제 정확도는 skip-gram 70.1%에서 fastText 74.9%로 올랐고 의미 문제는 78.5%에서 77.8%로 비슷했다(논문 Table 2). [3편](/posts/2026-10-18-nlp-3-evaluating-embeddings-glove) 끝에서 Mikolov et al.(2013a)이 단어 구조를 넣으면 문법 문제가 나아질 것이라고 한 예상과 맞는다.

### BPE: 자주 붙는 쌍부터 합치기

조각을 만드는 다른 방법이 BPE(byte pair encoding)다. D2L 15.6.2절의 예로 따라가 보자. 말뭉치에 fast가 4번, faster가 3번, tall이 5번, taller가 4번 나오고 단어 끝에는 `_`를 붙인다. 처음에는 글자 하나하나가 기호다. 단어 안에서 이웃한 기호 쌍을 세면 (t, a), (a, l), (l, l)이 tall과 taller에서 5 + 4 = 9번으로 가장 많다. (f, a)나 (e, r)는 4 + 3 = 7번이다. 가장 많은 쌍을 새 기호로 합치고 다시 센다. 10번 합치면 이렇게 된다.

| 합친 순서 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 새 기호 | ta | tal | tall | fa | fas | fast | er | `er_` | `tall_` | `fast_` |

이제 faster는 fast + `er_`, taller는 tall + `er_`로 잘린다. 말뭉치에 없던 tallest도 tall + e + s + t + `_`로, fatter는 fa + t + t + `er_`로 잘려서 UNK가 되지 않는다. 글자 하나하나가 처음부터 기호로 들어 있어서 처음 보는 단어도 끝까지 쪼갤 수 있다.

fastText와 BPE는 쪼개는 목적이 다르다. fastText는 길이가 정해진 조각을 모두 쓰고 그 합으로 단어 벡터를 만든다. 모델이 보는 단위는 여전히 단어다. BPE는 말뭉치를 보고 조각 어휘 자체를 정하고 단어를 그 조각의 열로 자른다. 모델이 보는 단위가 조각이 되고 어휘 크기는 합치는 횟수로 정할 수 있다. GPT-2가 BPE를 쓰고(D2L 15.6.2절) BERT는 30,000개짜리 WordPiece 어휘를 쓴다(Devlin et al. 2019, 3절).

## 길이가 다른 문장을 벡터 하나로

이제 단어 벡터는 있다. 영화 리뷰가 긍정인지 부정인지 분류하려면 문장 전체를 벡터 하나로 만들어 분류기에 넣어야 한다. 그런데 문장마다 token 수가 다르고 대부분의 분류기는 고정된 길이의 입력을 받는다. 핵심 문제는 이것이다.

**길이가 제각각인 벡터 열을 고정된 크기의 벡터 하나로 어떻게 줄일 것인가?**

SLP3 6.5절은 두 길을 제시한다. 벡터를 모두 이어 붙이는 concatenation은 정보를 다 남기지만 망이 커지고 token 수가 정해져 있어야 한다. 벡터들을 하나로 합치는 pooling은 크기가 $H$로 고정되고 작지만 정보가 줄어든다. SLP3는 감성 분류처럼 great가 몇 번째 단어인지보다 great가 있느냐가 중요한 task에서 pooling이 특히 쓸모 있다고 설명한다. 여기서는 pooling을 본다. "I loved the movie !"의 다섯 token에 3차원 벡터를 주었다(설명을 위해 지어낸 값이다).

| token | $\mathbf{x}_i$ |
| --- | --- |
| I | [0.2, 0.1, 0.9] |
| loved | [1.8, 0.3, 0.1] |
| the | [0.0, 0.1, 1.2] |
| movie | [0.3, 1.5, 0.2] |
| ! | [0.6, 0.0, 0.5] |

### 합, 평균, 최댓값, 가중합

가장 단순한 것은 합 $\mathbf{d} = \sum_i \mathbf{x}_i$다. 첫 차원은 $0.2 + 1.8 + 0.0 + 0.3 + 0.6 = 2.9$이고 $\mathbf{d} = [2.9, 2.0, 2.9]$다. 평균은 이것을 $n = 5$로 나눈 $[0.58, 0.40, 0.58]$이다. 합은 문장이 길수록 커진다. 같은 내용의 리뷰라도 두 배 길면 합도 두 배가 되어 선형 분류기의 점수가 한쪽으로 치우친다. 평균은 길이와 상관없이 비슷한 크기에 머문다.

평균은 단순해도 강하다. Iyyer et al.(2015)의 deep averaging network(DAN)는 임베딩을 평균 낸 뒤 feedforward 층 몇 개를 지나 분류한다. 초록에 따르면 이 망이 감성 분석과 factoid 질의응답에서 구문 구조를 쓰는 모델과 겨룰 만했고 일부에서는 더 나았으며 학습 시간은 그 모델들의 일부만 들었다.

최댓값은 차원마다 다섯 값 가운데 가장 큰 것을 고른다. 첫 차원은 loved의 1.8, 둘째는 movie의 1.5, 셋째는 the의 1.2라서 $\mathbf{d} = [1.8, 1.5, 1.2]$다. 차원마다 다른 token에서 온 값이 섞이므로 결과는 어느 token의 벡터와도 같지 않다. 학습할 때 gradient는 차원마다 최댓값을 낸 token으로만 흐른다.

가중합은 token마다 중요도 $a_i$를 다르게 준다. $a = (0.05, 0.55, 0.05, 0.20, 0.15)$로 loved에 가장 큰 무게를 주면 첫 차원은

$$
0.05(0.2) + 0.55(1.8) + 0.05(0.0) + 0.20(0.3) + 0.15(0.6) = 1.15
$$

이고 $\mathbf{d} = [1.150, 0.475, 0.275]$다. 모든 $a_i = 1$이면 합, 모든 $a_i = 1/n$이면 평균이니 가중합은 앞의 둘을 품는다. 네 결과를 나란히 놓은 것이 그림 2다(attention은 바로 다음에 계산한다). 남는 질문은 $a_i$를 누가 정하느냐다.

<Fig src="/images/posts/2026-10-18-nlp-4-subword-composition-contextual/composition.svg" alt="같은 다섯 token 벡터를 합, 평균, 최댓값, 고정 가중합, attention으로 합친 결과 막대그래프">그림 2. 같은 다섯 벡터를 다섯 방법으로 합치면 무엇이 달라지는가? 합과 평균은 방향이 같다. 가중합과 attention은 loved의 벡터 [1.8, 0.3, 0.1] 쪽으로 기운다.</Fig>

### Attention: 가중치도 학습한다

$a_i$를 손으로 정하지 말고 학습하자는 것이 attention이다. "중요한 단어"를 나타내는 벡터 $\mathbf{v} \in \mathbb{R}^H$를 하나 두고 token마다 점수를 매긴 뒤 softmax로 합이 1인 가중치를 만든다.

$$
r_i = \mathbf{v}^\top\mathbf{x}_i, \qquad \mathbf{a} = \operatorname{softmax}(\mathbf{r}), \qquad \mathbf{d} = \sum_i a_i\mathbf{x}_i \tag{2}
$$

$\mathbf{v} = [2.0, 0.5, -1.0]$이라 하자. loved의 점수는 $2.0(1.8) + 0.5(0.3) - 1.0(0.1) = 3.65$이고 다섯 점수는 $\mathbf{r} = [-0.45, 3.65, -1.15, 1.15, 0.70]$이다. 지수를 취하면 $[0.638, 38.475, 0.317, 3.158, 2.014]$, 합이 44.601이니

$$
\mathbf{a} = [0.014,\ 0.863,\ 0.007,\ 0.071,\ 0.045], \qquad \mathbf{d} = \sum_i a_i\mathbf{x}_i = [1.604,\ 0.367,\ 0.144]
$$

이다. 그림 3에서 보듯 점수 3.65가 다른 점수보다 2.5 이상 커서 가중치의 86%가 loved에 몰렸다.

<Fig src="/images/posts/2026-10-18-nlp-4-subword-composition-contextual/attention-weights.svg" alt="다섯 token의 attention 점수와 softmax 가중치 막대그래프">그림 3. 점수 차이가 softmax를 지나면 얼마나 커지는가? 점수 3.65인 loved가 가중치 0.863을 가져간다.</Fig>

$\mathbf{v}$에는 정답이 따로 없다. 분류 손실의 gradient가 $\mathbf{a}$를 거쳐 $\mathbf{v}$까지 흘러 분류에 도움이 되는 token에 큰 점수를 주는 쪽으로 학습된다. Yang et al.(2016)의 Hierarchical Attention Network가 같은 생각이다. 논문 식 (5)~(7)은 양방향 GRU가 만든 단어 표현을 한 층 MLP에 넣은 뒤 학습되는 벡터 $\mathbf{u}_w$와 내적하고 softmax로 가중합한다. 논문은 $\mathbf{u}_w$를 "what is the informative word"라는 고정된 질문으로 설명하고 무작위로 초기화해 다른 파라미터와 함께 학습한다.

softmax의 양 끝을 보면 attention의 성격이 보인다. 점수 하나가 훨씬 크면 $\mathbf{a}$가 one-hot에 가까워져 token 하나를 고르는 셈이다. 점수가 모두 같으면 $\mathbf{a} = (1/n, \dots, 1/n)$이라 평균이 된다. 최댓값이 차원마다 다른 token을 고르는 것과 달리 attention은 token 단위로 고르고 그 정도를 학습한다.

**주의:** 가중치가 큰 token이 곧 예측의 이유라고 단정하면 안 된다. Jain & Wallace(2019)는 여러 NLP task에서 attention 가중치가 gradient 기반 중요도와 자주 어긋나며 가중치 분포가 전혀 다른데도 같은 예측을 내는 경우를 찾을 수 있다고 보고했다. Wiegreffe & Pinter(2019)는 그 결론이 설명을 어떻게 정의하느냐에 달렸다고 반박하며 attention을 설명으로 쓸 수 있는지 가리는 시험 네 가지를 내놓았다. 그림 3 같은 막대는 참고 자료로 읽는 편이 안전하다.

### 순서를 버린 대가

다섯 방법에는 공통된 약점이 있다. 두 리뷰를 보자.

- the movie was not good, it was bad
- the movie was not bad, it was good

단어 목록이 똑같다. 합, 평균, 최댓값은 token의 순서를 보지 않으니 두 문장에 같은 벡터를 준다. 식 (2)의 attention도 점수 $r_i$가 $\mathbf{x}_i$ 하나로만 정해지므로 마찬가지다. 뜻은 정반대인데 말이다. **pooling은 어떤 단어가 있는지만 보고 어디에 있는지는 보지 않는다.** Iyyer et al.(2015) 5.2절은 부정어와 but이 함께 든 문장 92개로 DAN을 따로 시험했다. 부정어가 보이면 DAN은 60.9%를 부정으로 예측했고 긍정으로 예측한 것은 35.9%였다. 구문 구조를 쓰는 비교 모델(DRecNN)도 55.4%를 부정으로 예측해 비슷하게 치우쳤으니 순서 정보만 넣는다고 바로 풀리는 문제는 아니다. 순서를 보는 CNN과 RNN은 시리즈 뒤쪽에서 다룬다.

다시 정리하면 합과 평균은 모든 token을 똑같이, 가중합은 정해 둔 무게로, attention은 학습한 무게로 섞고 최댓값은 차원마다 token 하나를 고른다. 다섯 방법 모두 문장 길이와 상관없이 $H$차원 벡터 하나를 내놓는 대신 순서를 버린다.

## 정적 임베딩은 왜 다의어에 약한가

먼저 용어 둘을 나누자. type은 사전에 실리는 단어 하나이고 token은 문장 안에서 실제로 쓰인 한 번이다. "the cat sat on the mat"은 token이 6개, type이 5개다(the가 두 번). [2편](/posts/2026-10-18-nlp-2-word2vec)의 어휘 크기 $V$가 type 수다.

word2vec, GloVe, fastText는 type마다 벡터가 하나인 정적(static) 임베딩이다. 문장이 무엇이든 같은 단어는 같은 벡터를 받는다. D2L 15.8.1절의 예로 crane은 "a crane is flying"에서는 두루미이고 "a crane driver came"에서는 기중기인데 둘 다 같은 벡터다. 50차원 GloVe에서 crane과 cosine이 큰 단어 12개를 뽑으면 실제로 뜻이 섞여 나온다.

| 단어 | GloVe 50d에서 가장 가까운 단어 (순서대로) |
| --- | --- |
| crane | rex, ichabod, gantry, wolf, decker, sarus, sandhill, bell, reed, carpenter, cliff, elevator |
| bank | banks, securities, banking, investment, exchange, financial, credit, lender, capital, brokerage, currency, investments |

crane의 목록에는 두루미 종 이름(sarus crane, sandhill crane), 기중기 종류(gantry crane), 단편 소설 「슬리피 할로의 전설」의 주인공 Ichabod Crane이 함께 섞여 있다. bank는 12개가 모두 금융 쪽이고 강둑이라는 뜻은 보이지 않는다. Arora et al.(2018)은 정적 벡터가 뜻별 벡터를 음이 아닌 계수로 더한 것에 가깝다는 것을 이론과 실험으로 뒷받침했다(1절 식 (1)). 2.2절의 Theorem 2에서 계수는 각 뜻이 말뭉치에 나온 횟수의 비율이다. **정적 벡터는 여러 뜻 사이 어딘가, 자주 쓰이는 뜻 쪽으로 기운 자리에 놓인다.** bank처럼 한 뜻이 압도적이면 나머지 뜻은 거의 묻힌다.

D2L은 이 차이를 함수로 적는다. 정적 임베딩은 단어 $x$만 받는 $f(x)$이고 문맥화(contextualized) 임베딩은 단어와 그 context를 함께 받는 $f(x, c(x))$다.

## 문맥화 임베딩: 문장을 읽고 나서 정하는 벡터

대표적인 예가 ELMo(Peters et al., 2018)다. 앞에서 뒤로 다음 단어를 맞히는 언어 모델과 뒤에서 앞으로 이전 단어를 맞히는 언어 모델을 함께 학습한다(3.1절, biLM). 입력층은 글자 단위 CNN이라 학습 때 못 본 단어에도 표현을 만들 수 있고 그 위에 두 방향 LSTM이 2층 쌓여 있다(3.4절). 그래서 token 하나마다 층별 표현 $\mathbf{h}_{k,0}, \mathbf{h}_{k,1}, \mathbf{h}_{k,2}$가 생긴다. ELMo는 이것을 가중합한다(논문 식 (1)).

$$
\text{ELMo}_k = \gamma\sum_{j=0}^{L} s_j\,\mathbf{h}_{k,j} \tag{3}
$$

$s_j$는 softmax로 정규화한 층별 가중치이고 $\gamma$는 전체 크기를 맞추는 스칼라로, 둘 다 task마다 학습한다. **앞 절의 가중합을 token이 아니라 층에 건 모양이다.** 쓰는 방법도 간단하다. 사전학습한 biLM의 가중치는 고정하고 task 모델의 원래 입력 $\mathbf{x}_k$(예: GloVe 벡터)에 $\text{ELMo}_k$를 이어 붙인다(3.3절). 논문 1절에 따르면 아래층은 품사 같은 구문 정보를, 위층은 context에 따른 뜻을 더 잘 담는다. 층마다 담는 것이 다르니 어느 층을 얼마나 쓸지는 task가 가중치 $s_j$로 정한다.

논문 5.3절은 다의어 play로 차이를 보여 준다. GloVe에서 play와 가까운 단어는 played, playing, player, game처럼 품사는 여럿인데 뜻은 운동 경기 쪽에 몰려 있다. biLM의 문맥화 표현으로 가장 가까운 문장을 찾으면 원래 문장 속 play의 품사와 뜻이 맞는 문장이 나온다.

BERT(Devlin et al., 2019)도 입력층에서는 표를 쓴다. WordPiece token 임베딩에 문장 구분용 segment 임베딩과 위치 임베딩을 더해 입력을 만들고(3절) Transformer를 지나면서 context 정보가 섞인다. 첫 token `[CLS]`의 마지막 hidden state를 분류용 문장 표현으로 쓴다. 앞 절의 pooling과는 다른 방식으로 문장 벡터를 얻는 셈이다. ELMo와 달리 사전학습한 망 전체를 task 데이터로 fine-tuning한다. 자세한 구조는 Transformer를 다루는 글에서 본다.

| | 정적(static) 임베딩 | 문맥화(contextualized) 임베딩 |
| --- | --- | --- |
| 단위 | type마다 벡터 하나 | token마다 다른 벡터 |
| 얻는 법 | 표에서 행을 꺼낸다 | 문장 전체를 망에 넣어 계산한다 |
| 예 | word2vec, GloVe, fastText | ELMo, BERT |
| 다의어 | 뜻이 한 벡터에 섞인다 | context로 가른다 |

## 단어 임베딩 단원 정리

네 편을 한 줄씩 정리하면 이렇다.

- [1편](/posts/2026-10-18-nlp-1-distributional-hypothesis-tf-idf): 단어의 뜻은 함께 나오는 단어에 담겨 있다는 분포 가설에서 출발해 횟수를 센 희소(sparse) 벡터와 cosine similarity, TF-IDF를 만들었다.
- [2편](/posts/2026-10-18-nlp-2-word2vec): skip-gram으로 주변 단어를 맞히며 밀집(dense) 벡터를 학습했다. one-hot에 임베딩 행렬을 곱하는 일은 행 하나를 꺼내는 일이었다.
- [3편](/posts/2026-10-18-nlp-3-evaluating-embeddings-glove): 임베딩을 사람의 유사도 순위와 analogy로 평가했고 동시 출현 확률의 비에서 GloVe를 유도했다.
- 4편(이 글): 조각 단위의 fastText와 BPE로 처음 보는 단어를 다뤘고 pooling과 attention으로 문장 벡터를 만들었으며 문맥화 임베딩으로 다의어 문제를 넘었다.

**단어 하나에 벡터 하나를 주던 표가 문장을 읽고 token마다 벡터를 계산하는 망으로 바뀌었다.** 그 표는 사라지지 않았다. BERT의 입력층에도 여전히 남아 있다.

다음 글에서는 이번 글에서 만든 문장 벡터 $\mathbf{d}$를 받아 label을 내는 text classification을 다룬다.

---

**참고 문헌**

- Jurafsky & Martin, [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/), 3판 초안, 6.5절
- Zhang et al., [Dive into Deep Learning](https://d2l.ai/), 15.6, 15.8절
- Bojanowski et al. (2017), [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606), TACL
- Iyyer et al. (2015), Deep Unordered Composition Rivals Syntactic Methods for Text Classification, ACL
- Yang et al. (2016), Hierarchical Attention Networks for Document Classification, NAACL
- Jain & Wallace (2019), [Attention is not Explanation](https://arxiv.org/abs/1902.10186), NAACL
- Wiegreffe & Pinter (2019), [Attention is not not Explanation](https://arxiv.org/abs/1908.04626), EMNLP
- Arora et al. (2018), Linear Algebraic Structure of Word Senses, with Applications to Polysemy, TACL
- Peters et al. (2018), [Deep Contextualized Word Representations](https://arxiv.org/abs/1802.05365), NAACL
- Devlin et al. (2019), [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), NAACL
- Mikolov et al. (2013a), [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- Pennington, Socher, Manning (2014), [GloVe](https://nlp.stanford.edu/projects/glove/): 표의 최근접 단어는 `glove.6B.50d.txt`로 직접 계산했다
- Wikipedia, Sarus crane, Sandhill crane, Gantry crane, Ichabod Crane 항목 (crane 이웃 단어의 뜻 확인)
