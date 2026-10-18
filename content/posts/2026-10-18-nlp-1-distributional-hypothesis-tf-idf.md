---
title: "자연어처리 (1): 단어를 벡터로, 분포 가설과 TF-IDF"
date: "2026-10-18"
category: "공부"
subcategory: "Deep Learning/Natural Language Processing"
tags: ["자연어처리", "분포 가설", "TF-IDF", "cosine similarity"]
description: "단어를 신경망에 넣으려면 벡터로 바꿔야 하는데 정수 번호와 one-hot은 단어 사이의 유사도를 담지 못한다. 분포 가설에서 출발해 단어가 어느 문서에 몇 번 나오는지 센 term-document 행렬을 만들고 cosine similarity와 TF-IDF로 비교한다. IDF가 무엇을 바꾸고 무엇을 바꾸지 않는지도 계산으로 확인한다."
---

자연어처리 시리즈의 첫 번째 글이다. 세종대학교 2026년 2학기 자연어처리 수업을 따라가며 공부한 내용이고 Stanford [CS224N](https://web.stanford.edu/class/cs224n/) 강의 노트와 Jurafsky & Martin의 [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)(3판 초안, 이하 SLP3)을 함께 참고했다.

> **단어를 벡터로 바꾸는 법: 무엇을 세어야 뜻이 비슷한 단어끼리 벡터도 비슷해질까?**

신경망은 숫자만 계산한다. 문장을 분류하든 번역하든 첫 단계는 단어를 숫자 벡터로 바꾸는 일이다. 이 글에서는 그 첫 단계를 가장 단순하게, 세는 방법으로 해 본다. 단어가 어느 문서에 몇 번 나오는지 세어 벡터를 만들고 그 벡터끼리 얼마나 닮았는지 비교한다. [다음 글](/posts/2026-10-18-nlp-2-word2vec)의 Word2Vec은 세는 대신 맞히는 방법이다.

## 번호와 one-hot으로는 왜 안 되는가

가장 먼저 떠오르는 방법은 단어에 번호를 붙이는 것이다. 개 = 1, 고양이 = 2, 자동차 = 3이라고 해 보자. 숫자에는 크기와 순서가 따라오니 고양이가 개보다 "크고" (개 + 자동차)/2가 고양이가 된다. 개와 고양이가 가깝고 개와 자동차가 멀다는 결론도 번호를 매긴 순서에서 나온다. 단어에 없는 구조를 억지로 씌운 셈이다.

순서를 없애려면 단어마다 축을 하나씩 주면 된다. 어휘가 $V$개이면 길이 $V$인 벡터에서 그 단어 자리만 1로 둔다. 이것이 one-hot 벡터다. 서로 다른 단어 $i \ne j$의 one-hot $\mathbf{e}_i$, $\mathbf{e}_j$는 1이 다른 자리에 있으므로

$$
\mathbf{e}_i^\top\mathbf{e}_j = 0, \qquad \|\mathbf{e}_i - \mathbf{e}_j\| = \sqrt{1^2 + (-1)^2} = \sqrt{2}
$$

이다. 어느 쌍을 골라도 내적은 0, 거리는 $\sqrt2$다. 개와 고양이는 개와 자동차만큼 다르다. 순서 문제는 사라졌지만 이번에는 유사도를 아예 나타낼 수 없다(D2L 15.1.1). 게다가 $V$는 수만에서 수십만이다.

그렇다면 우리가 고민해야 할 핵심 문제는 이것이다. **뜻이 비슷한 단어끼리 벡터도 비슷하게 만들려면 단어의 뜻을 어디서 가져와야 하는가?** 글자 모양에서는 얻을 수 없다. "dog"라는 세 글자에 개의 성질이 들어 있지는 않으니까.

## 분포 가설: 함께 다니는 단어를 보면 안다

답은 텍스트에서 그 단어가 쓰이는 모습이다. 자주 인용되는 문장은 언어학자 Firth(1957)의 "You shall know a word by the company it keeps"다(CS224N 강의 노트 1, Eisenstein 14.1). 단어는 함께 다니는 단어들을 보면 알 수 있다는 말이다. SLP3 5장에 따르면 1950년대에 Joos, Harris, Firth 같은 언어학자들이 같은 생각을 내놓았다. Harris(1954)는 두 단어의 뜻이 다른 정도가 두 단어가 나타나는 환경이 다른 정도와 대략 맞아떨어진다고 보았다. 철학 쪽에서는 Wittgenstein(1953)이 단어의 뜻을 그 단어가 언어에서 쓰이는 방식으로 보았다(SLP3 5장 Historical Notes). **비슷한 context에 나오는 단어는 뜻도 비슷하다는 이 생각을 분포 가설(distributional hypothesis)이라고 한다.**

직접 해 보자. 아래 네 문장의 빈칸에 들어갈 단어는 무엇일까?

- everyone likes `____`
- a bottle of `____` is on the table
- `____` makes you drunk
- a cocktail with `____` and seltzer

주변 단어만 보고도 wine이나 gin 같은 술이 떠오른다. 네 문장에 모두 들어맞는 단어들은 context를 공유하고 그래서 뜻도 비슷하다. 앞의 세 문장은 Lin(1998)이 낯선 단어 tezgüino로 든 예에서 왔다. Lin은 이런 문장들만 보고도 tezgüino가 옥수수로 빚은 술일 것이라고 짐작할 수 있다고 썼다. SLP3 5.2의 ongchoi도 같은 예다. 이 낯선 단어가 "delicious sauteed with garlic", "superb over rice", "leaves with salty sauces" 같은 문장에 나온다고 하자. 시금치나 근대도 garlic, rice, leaves 근처에 나온다는 것을 알고 있다면 ongchoi도 잎채소라고 짐작할 수 있다. 실제로 ongchoi는 영어로 water spinach라고도 부르는 Ipomoea aquatica다.

Eisenstein(14.1)은 이 예를 한 걸음 더 끌고 간다. 비슷한 문장 네 개를 두고 후보 단어가 빈칸에 들어갈 수 있으면 1, 없으면 0을 적는다. 그러면 단어마다 길이 4인 0/1 벡터가 생긴다. tezgüino는 (1, 1, 1, 1), wine은 (1, 1, 1, 0), loud는 (0, 0, 0, 0)이다. 벡터가 가장 닮은 wine이 뜻도 가장 가깝다. context를 차원으로 삼아 단어를 벡터로 만든다는 생각이 여기서 벌써 나온다.

### Distributional과 distributed

비슷하게 생긴 두 말을 구별해 두자. Eisenstein 14.1에 따르면 distributional은 의미를 context 통계에서 얻는다는 뜻이고 distributed는 의미를 기호가 아니라 숫자 벡터에 담는다는 뜻이다. 다음 글의 Word2Vec은 둘 다에 해당한다. 라벨이 있는 데이터로 지도학습한 분류기가 만든 임베딩은 context 통계를 쓰지 않았으니 distributed이기만 하다. 이 시리즈에서 distributed representation은 단어가 나타나는 context의 분포를 담은 벡터를 가리키고 벡터, representation, 임베딩을 같은 뜻으로 섞어 쓴다.

Distributed라는 말은 신경망 연구에서 왔다. Hinton, McClelland, Rumelhart(1986)는 대상 하나를 계산 단위 하나에 맡기는 방식을 local representation이라 불렀다. 대상 하나를 여러 단위에 걸친 활성 패턴으로 나타내고 단위 하나가 여러 대상을 나타내는 데 함께 쓰이면 distributed representation이다. 단어마다 축을 하나씩 주는 one-hot은 local 쪽이다. 1이 켜진 칸 하나가 단어 하나를 맡는다.

Context를 무엇으로 정할지도 골라야 한다. 같은 문서에 나오는 단어로 정할 수도 있고 앞뒤 몇 단어 안에 나오는 단어로 정할 수도 있다. 이 선택에 따라 결과가 꽤 달라진다. Eisenstein 14.2.2가 소개한 LSA 결과에서 dog와 가장 가까운 단어는 앞뒤 2단어를 context로 보면 cat, horse, fox, pet, rabbit 같은 다른 동물이다. 앞뒤 30단어로 넓히면 kennel, puppy, terrier, to bark처럼 개와 관련된 말이 된다. 좁은 window는 문법적 쓰임이 같은 단어를, 넓은 window는 주제가 같은 단어를 모은다. 이 글은 문서를 context로 쓰고 다음 글의 Word2Vec은 앞뒤 몇 단어의 window를 쓴다.

## 희곡 여덟 편의 term-document 행렬

Context를 "같은 문서에 나온다"로 정하면 단어 × 문서 빈도 행렬이 나온다. 셰익스피어 희곡 여덟 편에서 단어 다섯 개가 나온 횟수를 세었더니 다음과 같았다고 하자.

| | Hamlet | Macbeth | Romeo & Juliet | Richard III | Julius Caesar | Tempest | Othello | King Lear |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| knife | 1 | 1 | 4 | 2 | 0 | 2 | 0 | 10 |
| dog | 0 | 0 | 0 | 6 | 12 | 2 | 0 | 0 |
| sword | 2 | 2 | 7 | 5 | 0 | 5 | 0 | 17 |
| love | 64 | 0 | 135 | 63 | 0 | 12 | 0 | 48 |
| like | 75 | 38 | 34 | 36 | 34 | 41 | 27 | 44 |

행 하나가 그 단어의 벡터이고 길이는 문서 수 8이다.

$$
\textit{knife} = [1, 1, 4, 2, 0, 2, 0, 10], \qquad \textit{sword} = [2, 2, 7, 5, 0, 5, 0, 17]
$$

**knife와 sword는 비슷한 문서에 비슷한 비율로 나온다.** 둘 다 King Lear에 가장 많고 Julius Caesar와 Othello에는 없다. dog는 Julius Caesar에 몰려 있다.

같은 행렬을 세로로 읽으면 문서 벡터가 된다. Hamlet은 knife, dog, sword, love, like 순으로 $[1, 0, 2, 64, 75]$다. SLP3 11.1.1도 같은 식으로 셰익스피어 희곡 네 편을 battle, good, fool, wit의 횟수로 나타낸다. 거기서도 희극 두 편(As You Like It, Twelfth Night)이 fool과 wit이 많고 battle이 적다는 점에서 서로 닮는다. 정보 검색은 문서 벡터(열)끼리 비교하고(SLP3 11.1.1) 단어 임베딩은 단어 벡터(행)끼리 비교한다. 이 구별은 뒤에서 IDF를 볼 때 다시 쓴다.

## Cosine similarity는 왜 길이로 나누는가

두 벡터가 얼마나 닮았는지 어떻게 수치로 나타낼까? 가장 많이 쓰는 것은 두 벡터 사이 각의 cosine이다.

$$
\cos(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}^\top\mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|} = \frac{\sum_i a_ib_i}{\sqrt{\sum_i a_i^2}\,\sqrt{\sum_i b_i^2}} \tag{1}
$$

같은 방향이면 1, 직교하면 0, 반대 방향이면 −1이다. knife와 sword로 계산해 보자.

$$
\begin{aligned}
\textit{knife}^\top\textit{sword} &= 1\cdot2 + 1\cdot2 + 4\cdot7 + 2\cdot5 + 2\cdot5 + 10\cdot17 = 222\\
\|\textit{knife}\|^2 &= 126, \qquad \|\textit{sword}\|^2 = 396\\
\cos(\textit{knife}, \textit{sword}) &= \frac{222}{\sqrt{126\times396}} = \frac{222}{223.4} = 0.994
\end{aligned}
$$

dog는 knife와 Richard III, Tempest 두 편에서만 겹친다. 내적이 $2\cdot6 + 2\cdot2 = 16$, $\|\textit{dog}\|^2 = 184$라서 cosine은 $16/\sqrt{126\times184} = 0.105$다. 다섯 단어를 모두 계산하면 다음과 같다.

| knife와 비교한 단어 | knife | dog | sword | love | like |
| --- | --- | --- | --- | --- | --- |
| cosine | 1.000 | 0.105 | 0.994 | 0.648 | 0.613 |

그런데 왜 굳이 길이로 나눌까? 내적만 쓰거나 Euclidean distance를 써도 되지 않을까? 셋을 나란히 놓아 보자.

<figure className="my-6 text-center">
  <img src="/images/posts/2026-10-18-nlp-1-distributional-hypothesis-tf-idf/knife-similarity.svg" alt="knife와 sword, love, like, dog 사이의 내적, Euclidean distance, cosine 막대그래프" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">그림 1. knife와 다른 단어의 거리를 세 가지 방법으로 계산했다. 내적은 자주 나오는 love와 like를 가장 가깝다고 하고 Euclidean distance는 dog가 love보다 knife에 가깝다고 한다.</figcaption>
</figure>

그림 1의 왼쪽에서 보듯 내적은 긴 벡터를 편애한다. love는 몇몇 희곡에 아주 많이 나와서 벡터 길이가 169.5로 가장 길고 knife와의 내적이 1234나 된다. sword(222)의 다섯 배가 넘는다. 자주 나오는 단어는 어느 단어와 곱해도 내적이 크니 내적은 뜻보다 빈도에 좌우된다(SLP3 5.4). Euclidean distance는 거꾸로 긴 벡터를 멀리 둔다. love는 knife가 나오는 희곡들에 함께 나오는데도 거리가 162.5로 가장 멀고 겹치는 희곡이 두 편뿐인 dog(16.7)가 더 가깝다.

Cosine은 이 문제가 없다. 어떤 단어가 모든 문서에서 두 배씩 나온다고 해 보자. 벡터에 양수 $\alpha$를 곱해도

$$
\cos(\alpha\mathbf{a}, \mathbf{b}) = \frac{\alpha\,\mathbf{a}^\top\mathbf{b}}{\alpha\|\mathbf{a}\|\,\|\mathbf{b}\|} = \cos(\mathbf{a}, \mathbf{b}) \tag{2}
$$

라서 cosine은 그대로다. knife와 2×knife는 cosine이 1인데 Euclidean distance는 $\|\textit{knife}\| = 11.2$나 된다. **Cosine은 길이를 나눠 없애고 방향만 본다. 얼마나 자주 나오는지가 아니라 어느 문서에 어떤 비율로 나오는지를 비교하는 셈이다.** 벡터마다 미리 자기 길이로 나눠 단위 벡터로 만들어 두면 cosine은 그냥 내적이 된다(SLP3 5.4). 다음 글의 Word2Vec이 두 단어의 유사도를 내적으로 계산하는 것도 같은 생각에서 나왔다(SLP3 5.5.1).

**주의:** 등장 횟수에는 음수가 없어서 count 벡터끼리의 cosine은 0과 1 사이에 있다(SLP3 5.4). 음수 쪽 값은 원소에 음수가 있는 임베딩, 곧 다음 글의 Word2Vec부터 나온다.

## TF-IDF: 어디에나 나오는 단어는 덜 믿는다

모든 단어가 똑같이 쓸모 있지는 않다. like는 여덟 편에 모두 나온다. 어느 문서에나 있으니 문서를 가려내는 데 도움이 안 된다. dog는 세 편에만 나오고 그중 Julius Caesar에 12번 몰려 있다. dog가 보이면 어떤 문서인지 꽤 좁혀진다. 그래서 단어마다 가중치를 다르게 준다. **한 문서 안에서 자주 나오는 단어에는 크게, 여러 문서에 두루 나오는 단어에는 작게 준다.**

$$
\operatorname{tf\text{-}idf}(t,d) = \operatorname{tf}(t,d)\times\operatorname{idf}(t), \qquad \operatorname{idf}(t) = \log_{10}\frac{N}{\operatorname{df}(t)} \tag{3}
$$

$\operatorname{tf}(t,d)$는 문서 $d$에서 단어 $t$가 나온 횟수, $N$은 전체 문서 수, $\operatorname{df}(t)$는 $t$가 나온 문서 수다. IDF는 Sparck Jones(1972)가 제안한 가중치다(SLP3 11.1.2). 모든 문서에 나오는 단어는 $\log 1 = 0$을 받고 드물수록 값이 커진다[^smooth]. 우리 행렬은 $N = 8$이다.

| 단어 | df | IDF |
| --- | --- | --- |
| knife | 6 | 0.125 |
| dog | 3 | 0.426 |
| sword | 6 | 0.125 |
| love | 5 | 0.204 |
| like | 8 | 0 |

<figure className="my-6 text-center">
  <img src="/images/posts/2026-10-18-nlp-1-distributional-hypothesis-tf-idf/idf-curve.svg" alt="df가 1에서 8까지 늘 때 IDF가 줄어드는 곡선과 다섯 단어의 위치" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">그림 2. df에 따른 IDF. 한 편에만 나오는 단어라면 0.903을 받고 여덟 편에 모두 나오는 like는 0을 받는다.</figcaption>
</figure>

TF-IDF는 횟수에 이 값을 곱한 것이다. Julius Caesar의 dog는 $12 \times 0.426 = 5.11$, Romeo & Juliet의 love는 $135 \times 0.204 = 27.56$, King Lear의 knife는 $10 \times 0.125 = 1.25$가 된다. like 행은 전부 0이 된다.

### 횟수를 그대로 곱해도 되는가

위 계산에서 love는 IDF가 dog의 절반도 안 되는데 TF-IDF는 다섯 배가 넘는다. 135번이라는 횟수가 그대로 곱해졌기 때문이다. 그런데 한 문서에 100번 나온 단어가 1번 나온 단어보다 100배 중요할까? SLP3 11.1.2는 그렇지 않다고 보고 TF로 $1 + \log_{10}\operatorname{count}(t,d)$를 쓴다(횟수가 0이면 0). 1번이면 1, 10번이면 2, 100번이면 3이다. 이 TF로 다시 계산하면 Romeo & Juliet의 love는 $(1 + \log_{10}135)\times0.204 = 3.13\times0.204 = 0.64$, Julius Caesar의 dog는 $(1 + \log_{10}12)\times0.426 = 2.08\times0.426 = 0.89$다. 순서가 뒤집혀 드문 단어 쪽이 더 무거워진다. 이 글의 나머지 계산은 단순하게 횟수를 그대로 쓴다. log의 밑을 바꾸면 모든 IDF에 같은 상수가 곱해질 뿐이라 어느 밑을 써도 순위는 같다.

## IDF는 단어 벡터의 cosine을 바꾸지 않는다

TF-IDF로 바꾸면 knife와 sword의 cosine은 어떻게 될까? 그대로다. 한 행의 원소는 모두 같은 단어의 값이라 IDF는 행 전체에 같은 상수를 곱할 뿐이다. knife 행도 sword 행도 0.125배가 된다. 식 (2)에서 본 것처럼 양수 상수배는 cosine을 바꾸지 않는다. 실제로 계산해도 dog 0.105, sword 0.994, love 0.648이 그대로 나온다. like 행은 영벡터가 되어 분모가 0이니 cosine을 정의할 수 없다.

**IDF가 효과를 내는 곳은 문서 벡터(열)의 비교다.** 문서 벡터의 차원은 단어라서 like 차원은 0배가 되고 dog 차원은 상대적으로 커진다. 세 쌍으로 확인해 보자.

| 문서 쌍 | 횟수 그대로 | TF-IDF |
| --- | --- | --- |
| Hamlet, Romeo & Juliet | 0.815 | 1.000 |
| Hamlet, Julius Caesar | 0.717 | 0.000 |
| Richard III, Julius Caesar | 0.492 | 0.195 |

<figure className="my-6 text-center">
  <img src="/images/posts/2026-10-18-nlp-1-distributional-hypothesis-tf-idf/doc-cosine-idf.svg" alt="세 문서 쌍의 cosine을 횟수 그대로와 TF-IDF로 비교한 막대그래프" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">그림 3. IDF를 곱하기 전과 후의 문서 cosine. like가 빠지자 Julius Caesar는 dog가 나오는 문서하고만 닮는다.</figcaption>
</figure>

횟수 그대로이면 모든 쌍이 꽤 비슷해 보인다. like가 모든 문서에 수십 번씩 나와 문서 벡터의 방향을 끌고 가기 때문이다. Hamlet $[1, 0, 2, 64, 75]$와 Julius Caesar $[0, 12, 0, 0, 34]$는 겹치는 단어가 like 하나뿐인데도 0.717이 나온다. IDF를 곱하면 Julius Caesar에는 dog(5.11) 하나만 남고 Hamlet에는 dog가 없으니 cosine이 0이 된다. dog가 6번 나오는 Richard III는 0.195가 남는다. Hamlet과 Romeo & Juliet은 둘 다 love가 압도적이라 1.000이 된다. 단어를 다섯 개만 남긴 작은 행렬이라 생긴 극단값이다. 어휘 전체를 쓰면 다른 단어들이 함께 방향을 정한다. 같은 이유로 다섯 단어 중 like만 나오는 Othello는 TF-IDF 벡터가 영벡터가 된다.

다시 정리하면 IDF는 행(단어)을 비교할 때는 상수배라서 아무 일도 하지 않고 열(문서)을 비교할 때 흔한 단어의 차원을 줄인다. 단어 벡터 쪽에서 이런 가중치가 효과를 내려면 차원이 문서가 아니라 단어여야 한다. 앞뒤 몇 단어 안에 함께 나온 횟수를 세는 word-context 행렬(SLP3 5.3)이 그런 경우다. 차원 하나하나가 주변 단어이니 like 같은 흔한 주변 단어의 차원이 줄어든다.

## 길고 대부분 0인 벡터

우리 표는 단어 다섯 개, 문서 여덟 편이라 작아 보인다. 실제 term-document 행렬은 행이 어휘 크기 $V$만큼 있고 문서 벡터의 길이도 $V$다. 한 문서에 나오는 단어는 그중 일부라서 대부분의 칸이 0이다. Word-context 행렬은 아예 $V \times V$다(SLP3 5.3). 이렇게 길고 대부분 0인 벡터를 희소(sparse) 벡터라고 한다.

One-hot도 sparse지만 둘은 다르다. One-hot은 0이 아닌 칸이 하나뿐이라 단어 번호 말고는 아무 정보가 없다. Term-document 벡터는 0이 아닌 칸에 context 정보가 들어 있어서 cosine에 뜻이 있다. 그래도 sparse 벡터에는 약점이 둘 있다. 하나는 크기다. 단어를 50,000차원으로 나타내면 그 위에 올리는 분류기도 차원마다 가중치를 배워야 한다. SLP3 5.5는 300차원이면 배울 가중치가 훨씬 적어 일반화에 유리할 수 있다고 본다. 다른 하나는 동의어다. 주변 단어를 차원으로 쓰는 sparse 벡터에서 car와 automobile은 서로 다른 차원이다. 그래서 car 옆에 나온 단어와 automobile 옆에 나온 단어가 비슷하다는 사실을 놓친다(SLP3 5.5).

**그래서 다음 질문은 이것이다. 길이 $V$의 sparse 벡터 대신 길이 수백의 짧고 빽빽한 벡터를 얻을 수는 없는가?** SLP3 5.5가 소개하는 임베딩이 그런 벡터다. 차원은 보통 50~1000이고 원소는 음수도 될 수 있는 실수이며 차원 하나하나에 뚜렷한 뜻이 붙어 있지 않다.

## 다음 글

[자연어처리 (2)](/posts/2026-10-18-nlp-2-word2vec)에서는 횟수를 세는 대신 주변 단어를 맞히는 문제를 풀게 해서 밀집(dense) 벡터를 얻는 Word2Vec을 다룬다. 이번 글의 cosine과 내적이 거기서도 그대로 쓰인다.

[^smooth]: 문서 모음에 없는 단어가 검색어로 들어오면 $\operatorname{df} = 0$이라 식 (3)의 분모가 0이 된다. 이를 막으려고 $\log\frac{N}{1 + \operatorname{df}(t)}$를 쓰기도 하는데 이 식은 $\operatorname{df} = N$일 때 음수가 된다. like라면 $\log_{10}(8/9) = -0.051$이다. scikit-learn의 `TfidfTransformer`는 기본값(`smooth_idf=True`)으로 $\ln\frac{1+N}{1+\operatorname{df}(t)} + 1$을 쓴다. 모든 단어가 한 번씩 나오는 문서를 하나 더 본 것처럼 계산하는 식이라 df가 0이어도 정의되고 값이 늘 1 이상이다.

---

**참고 문헌**

- D. Jurafsky, J. H. Martin, [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/), 3판 초안: 5.1~5.5절, 5장 Historical Notes, 11.1절
- J. Eisenstein, Natural Language Processing (2018년 11월 초안): 14.1, 14.2.2절
- Stanford CS224N, Lecture Notes 1: Introduction and Word2vec (Winter 2023 초안)
- A. Zhang et al., [Dive into Deep Learning](https://d2l.ai/)(D2L): 15.1.1절
- G. E. Hinton, J. L. McClelland, D. E. Rumelhart (1986), Distributed Representations, *Parallel Distributed Processing* 1권 3장
- D. Lin (1998), Automatic Retrieval and Clustering of Similar Words, COLING-ACL
- scikit-learn, `TfidfTransformer` 문서(`sklearn/feature_extraction/text.py`)
