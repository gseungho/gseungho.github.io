---
title: "자연어처리 (3): 임베딩 평가와 GloVe"
date: "2026-10-18"
category: "공부"
subcategory: "Deep Learning/Natural Language Processing"
tags: ["자연어처리", "word embedding", "GloVe", "analogy"]
description: "학습한 단어 임베딩을 사람의 유사도 순위와 analogy로 평가하는 방법을 다룬다. 공개된 사전학습 임베딩을 살펴보고 동시 출현 확률의 비에서 GloVe의 목적함수를 유도한다. 어휘에 없는 단어가 남기는 문제로 글을 맺는다."
---

자연어처리 시리즈의 세 번째 글이다. 세종대학교 2026년 2학기 자연어처리 수업을 따라가며 공부한 내용이고 Stanford [CS224N](https://web.stanford.edu/class/cs224n/) 강의 노트와 Jurafsky & Martin의 [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)(3판 초안)을 함께 참고했다.

> **임베딩 평가가 묻는 것: 벡터 사이의 거리와 방향이 사람이 아는 단어의 뜻과 얼마나 맞을까?**

[2편](/posts/2026-10-18-nlp-2-word2vec)에서 skip-gram으로 단어마다 밀집(dense) 벡터를 학습했다. 이번 글은 그 벡터가 좋은지 평가하는 방법에서 시작해 남이 학습해 공개한 임베딩, 그중 하나인 GloVe의 목적함수로 넘어간다. cosine similarity는 [1편](/posts/2026-10-18-nlp-1-distributional-hypothesis-tf-idf)에서 정의한 것을 그대로 쓴다.

## 좋은 임베딩이란 무엇인가

임베딩은 결국 다른 task에 쓰려고 학습한다. 그러니 가장 정직한 평가는 임베딩을 감성 분류나 개체명 인식 같은 실제 task의 입력으로 넣고 그 task의 성능을 측정하는 것이다. 이것을 extrinsic 평가라 하고 SLP3 5.9절도 가장 중요한 평가로 꼽는다.

문제는 비용이다. CS224N 노트 2의 2.1절처럼 질의응답 시스템을 만든다고 생각해 보자. 단어를 벡터로 바꾸고 그 벡터를 파라미터가 수백만 개인 큰 망에 넣어 답을 만든다. 임베딩 차원 같은 하이퍼파라미터를 하나 바꿀 때마다 그 큰 망을 처음부터 다시 학습해야 한다면 실험이 끝나지 않는다. 성능이 나쁘게 나와도 임베딩 탓인지 그 위의 망 탓인지 가리기 어렵다(같은 노트 2.2절).

그래서 임베딩만 떼어 작은 부속 문제로 시험하는 intrinsic 평가를 함께 쓴다. 단어 쌍의 유사도를 맞히거나 analogy를 푸는 식이라 금방 끝난다. 조건이 하나 붙는다. **intrinsic 점수는 최종 task 성능과 함께 움직일 때만 쓸모가 있다**(같은 노트 2.1절). 늘 그렇지는 않아서 Schnabel et al.(2015)은 평가 방법을 바꾸면 임베딩 방법들의 순위가 바뀐다고 보고했다.

## 단어 유사도: 사람이 매긴 순위와 비교하기

### WordSim-353

가장 흔한 intrinsic 평가는 사람이 매긴 유사도와 비교하는 것이다. 대표 자료인 WordSim-353(Finkelstein et al., 2002)은 단어 쌍 353개에 0~10점을 붙였다. 배포 파일의 안내문은 평가자에게 0은 전혀 관계없음, 10은 매우 밀접함으로 점수를 매기라고 했고 153쌍은 13명, 200쌍은 16명의 점수를 평균했다.

| 단어 1 | 단어 2 | 사람 점수 |
| --- | --- | --- |
| midday | noon | 9.29 |
| journey | voyage | 9.29 |
| car | automobile | 8.94 |
| professor | cucumber | 0.31 |
| king | cabbage | 0.23 |

절차는 단순하다. 쌍마다 두 단어 벡터의 cosine을 구하면 cosine 353개와 사람 점수 353개가 생기고 이 두 열의 상관계수가 임베딩의 점수가 된다.

### 왜 순위 상관을 쓰는가

GloVe 논문(Pennington et al., 2014) Table 3과 fastText 논문(Bojanowski et al., 2017) 5.1절은 모두 Spearman 순위 상관을 쓴다. 사람 점수는 0~10이고 cosine은 −1~1이라 눈금이 달라서일까? 그건 이유가 아니다. Pearson 상관계수도 한 변수에 양수를 곱하고 상수를 더해도 그대로다. 이유는 우리가 묻는 질문에 있다. cosine이 사람 점수의 일차함수이기를 바라는 게 아니다. **사람이 더 비슷하다고 한 쌍에 임베딩도 더 큰 cosine을 주는지, 곧 순서가 맞는지를 묻는다.** Spearman은 두 변수를 각각 순위로 바꾼 뒤 구한 Pearson이고 동점이 없으면 이렇게 쓴다.

$$
\rho = 1 - \frac{6\sum_{i=1}^{n} d_i^2}{n(n^2-1)}, \qquad d_i = (\text{사람 점수 순위}) - (\text{cosine 순위}) \tag{1}
$$

Stanford가 공개한 50차원 GloVe 벡터(뒤에서 다시 나온다)로 네 쌍을 계산해 보자.

| 쌍 | 사람 점수 | 사람 순위 | cosine | cosine 순위 | $d_i$ |
| --- | --- | --- | --- | --- | --- |
| midday, noon | 9.29 | 1 | 0.830 | 1 | 0 |
| car, automobile | 8.94 | 2 | 0.696 | 2 | 0 |
| professor, cucumber | 0.31 | 3 | −0.130 | 4 | −1 |
| king, cabbage | 0.23 | 4 | 0.113 | 3 | 1 |

$\sum d_i^2 = 2$이므로 $\rho = 1 - 6\cdot2/(4\cdot15) = 0.8$이다. 같은 네 쌍의 Pearson은 0.97이다. 높은 두 쌍과 낮은 두 쌍이 멀리 떨어져 있어서 Pearson은 그 큰 간격에 끌려 높게 나온다. 아래 두 쌍의 순서가 뒤집힌 것은 Spearman만 잡아낸다.

<figure className="my-6 text-center">
  <img src="/images/posts/2026-10-18-nlp-3-evaluating-embeddings-glove/ws353-scatter.svg" alt="WordSim-353 353쌍의 사람 점수와 GloVe 50차원 cosine 산점도" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">그림 1. 사람 점수가 높을수록 cosine도 커지는가? 353쌍 전체의 Spearman은 0.503이다. 파란 점이 표의 네 쌍이다.</figcaption>
</figure>

353쌍 전체가 그림 1이다. 오른쪽 위로 올라가는 경향은 뚜렷하지만 흩어짐도 크다. 50차원 GloVe의 Spearman은 0.503이고 같은 말뭉치의 300차원 GloVe는 논문 Table 3에서 0.658이었다[^ws].

### similarity와 relatedness의 차이

"비슷하다"에는 두 뜻이 섞여 있다. SLP3 5.1절의 예로 cat과 dog은 비슷한(similar) 단어이고 coffee와 cup은 비슷하지 않지만 함께 쓰이는 관련된(related) 단어다. WordSim-353은 둘을 가르지 않았다. 안내문은 반의어도 같은 영역에 있으니 비슷하다고 보라고 했다. 그 결과 cup과 coffee가 6.58점으로 car와 train(6.31점)보다 높다. Hill et al.(2015)은 1절에서 이 쌍을 문제로 들며 similarity만 평가하는 SimLex-999를 만들었다.

어느 쪽을 잘 잡는지는 학습할 때 정한 window 크기에 따라서도 달라진다. SLP3 5.7절이 소개한 Levy & Goldberg(2014)의 실험에서 skip-gram의 window를 앞뒤 2단어로 잡자 Hogwarts에 가장 가까운 단어로 Sunnydale, Evernight 같은 다른 소설 속 학교가 나왔다. 앞뒤 5단어로 넓히자 Dumbledore, Malfoy처럼 해리 포터 이야기에 함께 나오는 단어가 나왔다. **좁은 window는 similarity 쪽, 넓은 window는 relatedness 쪽으로 기운다.** 어느 자료로 재느냐에 따라 좋은 임베딩의 기준이 달라지는 셈이다.

## 차이 벡터로 푸는 analogy

두 번째 intrinsic 평가는 "Germany와 Berlin의 관계는 France와 무엇의 관계와 같은가"를 푸는 analogy다. Mikolov et al.(2013a) 4절은 이것을 벡터 덧셈으로 푼다. a : b :: c : ?의 답은 $\mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c$와 cosine이 가장 큰 단어다.

$$
\hat d = \operatorname*{argmax}_{i \notin \{a,\, b,\, c\}} \cos\big(\mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c,\ \mathbf{v}_i\big) \tag{2}
$$

$\mathbf{v}_b - \mathbf{v}_a$를 a에서 b로 가는 관계로 보고 그 관계를 c에 더하는 셈이다. 관계 하나가 공간의 방향 하나라면 man → woman, king → queen, uncle → aunt 화살표는 모두 나란해야 한다.

<figure className="my-6 text-center">
  <img src="/images/posts/2026-10-18-nlp-3-evaluating-embeddings-glove/analogy-pca.svg" alt="GloVe 50차원 벡터 열 개를 PCA로 2차원에 그린 그림, 남성에서 여성으로 가는 화살표 다섯 개" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">그림 2. 50차원 GloVe 벡터 열 개를 PCA로 2차원에 눌렀다. 남성에서 여성으로 가는 화살표가 거의 나란하고 점선과 함께 평행사변형을 이룬다.</figcaption>
</figure>

그림 2는 실제 50차원 GloVe 벡터다. SLP3 5.7절에 따르면 analogy를 이런 평행사변형으로 푸는 모형은 Rumelhart & Abrahamson(1973)이 먼저 내놓았고 word2vec 이후 다시 주목받았다. 다만 2차원 그림은 보기 좋게 눌러 놓은 것이라 50차원에서 다섯 차이 벡터끼리의 cosine은 0.60~0.83이다. 나란하긴 해도 똑같지는 않다.

### 입력 세 단어는 왜 빼는가

식 (2)의 $i \notin \{a, b, c\}$는 지나치기 쉽다. Mikolov et al.(2013a) 4절은 입력 단어를 검색에서 뺀다고 적었고 word2vec 배포본의 `compute-accuracy.c`도 세 단어를 건너뛴다. 빼지 않으면 어떻게 될까? 50차원 GloVe에서 man : woman :: king : ?를 배포 코드처럼 단위 벡터로 풀면 다음과 같다.

| 순위 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 단어 | king | queen | throne | prince |
| cosine | 0.885 | 0.852 | 0.766 | 0.759 |

1등은 king이다. 차이 벡터 $\mathbf{v}_{\text{woman}} - \mathbf{v}_{\text{man}}$이 king을 queen까지 옮길 만큼 길지 않아서 결과가 king 근처에 머문다. 입력 단어를 빼야 queen이 남는다. SLP3 5.7절도 가장 가까운 단어는 대개 입력 세 단어 가운데 하나라서 명시적으로 빼야 한다고 적었다. 채점도 엄격하다. 1등 단어가 정답과 똑같아야 맞힌 것이고 동의어는 틀린 것으로 센다(Mikolov et al. 2013a, 4.1절).

### 14개 범주의 성적표

Mikolov et al.(2013a)의 문제 모음은 word2vec 배포본에 `questions-words.txt`로 들어 있다. 19,544문제가 의미 범주 5개(8,869문제)와 문법 범주 9개(10,675문제)로 나뉜다.

| a | b | c | 정답 | 범주 |
| --- | --- | --- | --- | --- |
| Baltimore | Maryland | Oakland | California | 도시와 그 도시가 있는 주 |
| possibly | impossibly | certain | uncertain | 반의어 |
| think | thinking | look | looking | 현재분사 |
| shrinking | shrank | slowing | slowed | 과거형 |

첫 줄은 수도 문제가 아니다. 논문 Table 1은 이 범주를 City-in-state라 부르고 수도는 나라와 수도를 묻는 별도 범주에 있다. 4.1절에 따르면 미국 대도시 68곳과 그 도시가 속한 주를 짝지은 뒤 두 쌍씩 무작위로 묶어 이 범주의 문제 약 2,500개를 만들었다. 50차원 GloVe로 전체 문제를 풀어 범주별 정확도를 낸 것이 그림 3이다.

<figure className="my-6 text-center">
  <img src="/images/posts/2026-10-18-nlp-3-evaluating-embeddings-glove/analogy-categories.svg" alt="questions-words.txt 14개 범주별 GloVe 50차원 analogy 정확도 막대그래프" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">그림 3. 어떤 관계는 잘 풀고 어떤 관계는 못 푸는가? 50차원 GloVe, 어휘 40만 개 전체에서 푼 결과다. 주황 점은 입력 단어를 빼지 않았을 때 1등이 입력 단어인 비율이다.</figcaption>
</figure>

전체 정확도는 46.2%이고 범주마다 차이가 크다. 나라와 국적 형용사(86%)나 나라와 수도(68~79%)는 잘 풀고 화폐(8%), 반의어(9%), 도시와 주(15%)는 거의 못 푼다. 입력 단어를 빼지 않으면 전체 문제의 47.5%에서 1등이 입력 단어다. 앞의 king은 예외가 아니었다.

### 말뭉치가 정답과 편향을 정한다

**analogy가 보여 주는 것은 일부 관계가 공간에서 대략 일정한 방향으로 놓인다는 사실까지다.** SLP3 5.7절은 이 방법이 흔한 단어, 가까운 거리, 나라와 수도나 동사 활용 같은 일부 관계에서만 잘 된다고 정리한다. 그림 3에서도 나라와 수도는 잘 풀리지만 화폐나 반의어는 거의 풀리지 않는다.

정답도 말뭉치에 따라 흔들린다. CS224N 노트 2의 2.3절에 따르면 미국에는 Phoenix라는 곳이 적어도 10곳 있어서 "Chicago : Illinois :: Phoenix : ?"의 답이 Arizona 하나라고 하기 어렵다. 수도도 바뀐다. 카자흐스탄의 수도는 1997년 전에는 Almaty였다.

더 곤란한 것은 편향이다. Bolukbasi et al.(2016) 1절에 따르면 Google News로 학습한 공개 word2vec 벡터는 "man is to computer programmer as woman is to x"에 x = homemaker라고 답하고 father : doctor :: mother : x에는 nurse라고 답한다. 같은 논문 그림 1에서 she 쪽 끝에 놓인 직업은 homemaker, nurse, receptionist 순이었다. 텍스트에 담긴 고정관념이 그대로 방향이 된다.

이 차이 벡터가 언어 모델 안에서 어떻게 쓰이는지는 [별도 글](/posts/2026-10-18-analogy-to-steering-vectors)에서 다룬다.

## 내려받아 쓰는 사전학습 임베딩

임베딩을 매번 직접 학습할 필요는 없다. 큰 말뭉치로 한 번 학습해 공개한 벡터를 받아 쓰면 된다.

| | word2vec (Google News) | GloVe 6B |
| --- | --- | --- |
| 말뭉치 | Google News 일부, 약 1,000억 단어 | Wikipedia 2014 + Gigaword 5, 60억 token |
| 어휘 | 단어와 구 300만 개 | 40만 개 (소문자) |
| 차원 | 300 | 50, 100, 200, 300 |

word2vec 쪽은 프로젝트 페이지, GloVe 쪽은 Stanford GloVe 페이지의 설명이다. GloVe 파일은 한 줄에 단어 하나와 숫자 50개가 있다.

| 단어 | 1 | 2 | 3 | 4 | … | 50 |
| --- | --- | --- | --- | --- | --- | --- |
| the | 0.418 | 0.24968 | −0.41242 | 0.1217 | … | −0.78581 |
| of | 0.70853 | 0.57088 | −0.4716 | 0.18048 | … | −0.80375 |

첫 다섯 줄은 the, 쉼표, 마침표, of, to다. GloVe 코드의 `vocab_count.c`가 어휘를 등장 횟수 순으로 정렬하므로 위쪽일수록 흔한 단어다. 파일 전체가 [2편](/posts/2026-10-18-nlp-2-word2vec)의 임베딩 행렬 $\mathbf{W}$($400{,}000 \times 50$)이고 한 줄이 한 행 $\mathbf{v}_w$다. 단어를 벡터로 바꾸려면 그 줄을 찾기만 하면 된다.

**주의:** GloVe 6B는 모두 소문자로 바꿔 학습했다. 회사 Apple과 과일 apple이 같은 줄 하나를 나눠 쓴다는 뜻이다. 한 단어에 벡터가 하나뿐이라는 이 성질은 [4편](/posts/2026-10-18-nlp-4-subword-composition-contextual)의 문맥화 임베딩에서 다시 문제가 된다.

## GloVe: 동시 출현 확률의 비를 벡터의 차로

GloVe(Pennington et al., 2014)는 출발점이 word2vec과 다르다. word2vec은 window를 밀며 주변 단어를 하나씩 예측했다. GloVe는 말뭉치를 한 번 훑어 동시 출현 횟수 $X_{ij}$(단어 $i$의 context에 단어 $j$가 나온 횟수)를 먼저 세고 그 표를 직접 맞춘다. [1편](/posts/2026-10-18-nlp-1-distributional-hypothesis-tf-idf)의 term-document 행렬에서 열을 문서 대신 주변 단어로 바꾼 $V \times V$ 행렬이다(SLP3 5.3절의 word-context 행렬). 논문은 앞뒤 10단어를 context로 잡고 $d$단어 떨어진 쌍은 $1/d$만큼만 세었다(4.2절). $X_i = \sum_k X_{ik}$로 두면 $P(j \mid i) = X_{ij}/X_i$다. 논문은 확률보다 확률의 비가 뜻을 잘 드러낸다는 관찰에서 출발한다(논문 Table 1).

| $k$ | solid | gas | water | fashion |
| --- | --- | --- | --- | --- |
| $P(k \mid \text{ice})$ | $1.9\times10^{-4}$ | $6.6\times10^{-5}$ | $3.0\times10^{-3}$ | $1.7\times10^{-5}$ |
| $P(k \mid \text{steam})$ | $2.2\times10^{-5}$ | $7.8\times10^{-4}$ | $2.2\times10^{-3}$ | $1.8\times10^{-5}$ |
| 비 | 8.9 | 0.085 | 1.36 | 0.96 |

확률 하나하나는 너무 작아서 읽기 어렵다. 비를 보면 다르다. ice에만 어울리는 solid는 비가 크고 steam에만 어울리는 gas는 작다. 둘 다와 관련된 water나 둘 다와 무관한 fashion은 1 근처다.

논문은 이 비를 벡터의 차로 나타내자는 데서 시작한다. 단어 벡터 $\mathbf{v}$와 context word 벡터 $\mathbf{u}$를 따로 두고 몇 가지 조건을 걸면 다음 꼴에 이른다(논문 식 (3)~(6)).

$$
\exp\big((\mathbf{v}_i - \mathbf{v}_j)^\top\mathbf{u}_k\big) = \frac{P(k \mid i)}{P(k \mid j)}, \qquad \mathbf{v}_i^\top\mathbf{u}_k = \log P(k \mid i) = \log X_{ik} - \log X_i
$$

차이 벡터 $\mathbf{v}_{\text{ice}} - \mathbf{v}_{\text{steam}}$과 $\mathbf{u}_k$의 내적이 그 비의 로그다. 그러니 이 차이 벡터는 context word $k$마다 ice 쪽에서 얼마나 더 자주 나오는지를 담는다. context word마다 woman과 man의 비가 queen과 king의 비와 비슷하다면 두 차이 벡터도 비슷해지고 그림 2의 평행사변형이 생긴다.

$\log X_i$는 $k$와 무관하니 bias $b_i$에 넣고 대칭을 맞추려고 context 쪽 bias $c_k$도 둔다(식 (7)). 이것을 가중 최소제곱으로 푼 것이 GloVe의 목적함수, 논문의 식 (8)이다.

$$
J = \sum_{i,j=1}^{V} f(X_{ij})\,\big(\mathbf{v}_i^\top\mathbf{u}_j + b_i + c_j - \log X_{ij}\big)^2 \tag{3}
$$

### 가중치 $f$는 무엇을 막는가

$X_{ij}$는 대부분 0이다. 논문은 어휘와 말뭉치에 따라 75~95%가 0이라고 적었다. 0에는 로그를 취할 수 없고 한두 번 나온 쌍은 잡음이 많다. 반대로 너무 흔한 쌍이 목적함수를 휩쓸어도 곤란하다. 그래서 논문은 $f(0) = 0$일 것, 드문 쌍을 키우지 않도록 감소하지 않을 것, 큰 $x$에서 비교적 작을 것을 요구하고 이 꼴을 골랐다(식 (9)).

$$
f(x) = \begin{cases} (x/x_{\max})^{\alpha} & x < x_{\max} \\ 1 & x \ge x_{\max} \end{cases} \tag{4}
$$

<figure className="my-6 text-center">
  <img src="/images/posts/2026-10-18-nlp-3-evaluating-embeddings-glove/glove-weight.svg" alt="GloVe 가중치 함수 f(x), alpha 3/4와 1, x_max 100에서 1로 고정" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">그림 4. 가중치 함수 f. 0에서 시작해 x_max = 100까지 올라간 뒤 1에 머문다. α = 3/4는 작은 횟수에 α = 1보다 조금 더 무게를 준다.</figcaption>
</figure>

그림 4가 이 함수다. 논문은 모든 실험에서 $x_{\max} = 100$, $\alpha = 3/4$를 썼고 $\alpha = 3/4$가 $\alpha = 1$보다 조금 나았다고 한다. 한 번 나온 쌍의 가중치는 $\alpha = 3/4$에서 $0.01^{3/4} = 0.032$, $\alpha = 1$에서 0.010이다. 드문 쌍을 버리지는 않되 흔한 쌍보다 훨씬 가볍게 본다. $f(0) = 0$이므로 0인 칸은 계산에서 빠지고 0이 아닌 칸만 돌면 된다. **$f$가 $x_{\max}$에서 멈추기 때문에 아무리 흔한 쌍도 가중치가 1을 넘지 못한다.**

word2vec과는 어떻게 이어질까? D2L 15.5절은 skip-gram의 손실을 같은 $(i, j)$끼리 묶으면 $-\sum_i X_i \sum_j P(j \mid i)\log Q_{ij}$라는 가중 cross-entropy가 된다는 것을 보인다. $Q_{ij}$는 모델의 softmax 확률이라 분모에 어휘 전체의 합이 들어간다. GloVe는 이것을 로그 값의 제곱오차로 바꿔 그 분모를 없앤 셈이다. 말뭉치 전체의 동시 출현 표를 쓰는 점은 count 기반을, 내적으로 값을 맞히는 점은 예측 기반을 닮았다. $X_{ij} = X_{ji}$라 두 벡터의 역할이 대칭이어서 논문은 둘을 더한 $\mathbf{W} + \tilde{\mathbf{W}}$를 단어 벡터로 쓴다(4.2절).

다시 정리하면 GloVe는 동시 출현 표를 먼저 세고 단어 벡터와 context word 벡터의 내적이 로그 횟수를 맞히도록 가중 최소제곱을 푼다. 확률의 비가 벡터의 차로 나타나도록 처음부터 식을 골랐으니 차이 벡터가 관계를 담는 것은 우연이 아니라 설계의 결과다.

## 어휘에 없는 단어는 어떻게 되는가

사전학습 임베딩은 자주 나온 단어에는 좋은 벡터를 준다. 어휘 밖의 단어가 문제다. 표에 없는 단어는 모두 UNK라는 기호 하나로 바뀌어 같은 벡터를 받는다. GloVe 6B의 어휘 40만 개를 뒤져 보면 friended는 있지만 unfriend는 없고 selfie와 covid도 없다. 어휘를 작게 잡으면 UNK는 훨씬 많아진다. D2L 15.3절은 PTB 말뭉치에서 10번 미만 나온 단어를 모두 `<unk>`로 바꿔 어휘를 6,719개로 만든다.

사람은 unfriend를 처음 봐도 un과 friend로 뜻을 짐작한다. UNK는 그 단서를 모두 버린다. Mikolov et al.(2013a) 4.1절도 모델이 단어의 형태 정보를 입력으로 받지 않으니 100% 정확도는 불가능에 가깝고 단어 구조를 넣으면 특히 문법 문제가 나아질 것이라고 적었다. **단어를 더 쪼갤 수 없는 단위로 보는 한 처음 보는 단어에 줄 벡터가 없다.**

## 다음 글

[4편](/posts/2026-10-18-nlp-4-subword-composition-contextual)에서는 단어를 character n-gram으로 쪼개는 fastText와 BPE로 이 문제를 푼다. 이어서 단어 벡터를 문장 벡터로 합치는 방법과 context에 따라 벡터가 바뀌는 문맥화 임베딩까지 다룬다. 이전 글은 [2편](/posts/2026-10-18-nlp-2-word2vec)의 Word2Vec이다.

---

**참고 문헌**

- Jurafsky & Martin, [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/), 3판 초안, 5.1, 5.3, 5.7, 5.9절
- Stanford CS224N, Lecture Notes Part II: Word Vectors II (2019), 2.1~2.3절
- Zhang et al., [Dive into Deep Learning](https://d2l.ai/), 15.3, 15.5절
- Finkelstein et al. (2002), Placing Search in Context: The Concept Revisited, ACM TOIS. 데이터: [WordSim-353](http://www.gabrilovich.com/resources/data/wordsim353/wordsim353.html)
- Hill, Reichart, Korhonen (2015), SimLex-999: Evaluating Semantic Models with (Genuine) Similarity Estimation, Computational Linguistics
- Schnabel et al. (2015), Evaluation Methods for Unsupervised Word Embeddings, EMNLP
- Levy & Goldberg (2014), Dependency-Based Word Embeddings, ACL
- Mikolov et al. (2013a), [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781). 코드와 `questions-words.txt`: [word2vec](https://code.google.com/archive/p/word2vec/)
- Bolukbasi et al. (2016), [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://arxiv.org/abs/1607.06520), NeurIPS
- Pennington, Socher, Manning (2014), [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf), EMNLP. 벡터와 코드: [GloVe](https://nlp.stanford.edu/projects/glove/)
- Bojanowski et al. (2017), [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606), TACL

[^ws]: 그림 1과 그림 3의 숫자는 Stanford의 `glove.6B.50d.txt`와 WordSim-353 원본, `questions-words.txt`로 직접 계산했다. 유사도는 두 벡터의 cosine 그대로이고 동점 순위는 평균 순위로 처리했다. 논문 Table 3은 차원마다 어휘 전체에서 정규화한 뒤 cosine을 구했으므로 두 값을 그대로 비교할 수는 없다.
