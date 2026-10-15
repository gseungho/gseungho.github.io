---
title: "GPT-1 · BERT · GPT-2 · GPT-3: 모델은 무슨 task인지 어디서 아는가"
date: "2026-10-12"
category: "논문리뷰"
subcategory: "Deep Learning/Foundations"
tags: ["논문리뷰", "GPT", "BERT", "In-Context Learning"]
description: "GPT-1과 BERT의 미세조정에서 GPT-2와 GPT-3의 프롬프트 기반 적응으로 이어지는 변화를 구조, 학습 목적, 평가 결과로 비교했다."
---

Transformer 자체는 2017년 이후 거의 바뀌지 않았다. 그런데 2018년의 GPT-1과 2020년의 GPT-3는
쓰는 방식이 완전히 다르다. 하나는 dataset마다 모델을 통째로 다시 학습시키고 다른 하나는 weight를
한 번도 건드리지 않고 예시 몇 개만 보여준다.

네 논문을 읽으면서 궁금했던 것은 다음과 같다.

> 모델은 자기가 지금 무슨 task를 해야 하는지를 **어디에서** 아는가?
> 번역인지 분류인지를 알려주는 정보는 어디에 담겨 있는가?

구조를 순서대로 훑는 글은 이미 많다. 여기서는 그 질문에 대한 답이 네 논문을 거치며 어떻게
이동하는지, 그리고 각 논문이 자기 주장을 실제로 어떤 숫자로 뒷받침했는지를 본다. 뒷받침하지
못한 것도 같이 적었다.

## 네 모델의 구조와 과제 적응 방식

| | GPT-1 (2018) | BERT (2018) | GPT-2 (2019) | GPT-3 (2020) |
|---|---|---|---|---|
| Backbone | decoder | encoder | decoder | decoder |
| Pre-training | next token | MLM + NSP | next token | next token |
| 최대 모델 | 117M | 340M | 1.5B | 175B |
| **task 정보 위치** | **weight** | **weight** | **prompt** | **prompt + 예시** |
| downstream update | 전체 fine-tune | 전체 fine-tune | 없음 | 없음 |

backbone 줄을 보면 BERT만 encoder이고 나머지는 전부 decoder다. 구조는 거의 안 바뀌었다.
정작 달라진 건 아래 두 줄, **task 정보가 weight에 들어가는가 context에 들어가는가**이다.

과제에 맞게 가중치를 바꾸는 방식에서, 입력 문맥으로 과제를 알려주는 방식으로 이어지는 변화다.

## GPT-1: 사전학습과 과제별 미세조정

### 두 단계 학습과 보조 손실

1단계는 label 없는 텍스트에서 다음 token을 예측하는 pre-training이다. 데이터는 출간되지 않은
책 7,000권 이상을 모은 BooksCorpus이고 모델은 12층 decoder에 768차원, head 12개다.

$$L_1(\mathcal{U}) = \sum_i \log P(u_i \mid u_{i-k}, \dots, u_{i-1};\, \Theta)$$

2단계에서는 데이터셋마다 출력층을 붙여 지도 미세조정을 한다. 이때 손실 함수에는 분류 항과 언어 모델링 항을 함께 사용한다.

$$L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda\, L_1(\mathcal{C}), \qquad \lambda = 0.5$$

두 항뿐이라 단순해 보이지만 성격이 전혀 다르다. $L_2$는 **task마다 달라지는** 항이다. label도
다르고 output head $W_y$도 다르다. 반면 $L_1$은 모든 task에서 똑같은 next-token prediction이고
representation이 pre-training에서 너무 멀어지지 않게 잡아두는 보조 regularizer다.

fine-tuning 때 새로 생기는 파라미터는 $W_y$와 delimiter embedding뿐이다. 그런데 **학습은 전체
파라미터에 대해 일어난다.** 그래서 dataset 하나당 fine-tuned checkpoint 하나가 생긴다. 12개
dataset이면 12개 모델이다.

보조 loss가 실제로 얼마나 기여하는지는 논문 Table 5에 있다.

| 설정 | Avg | CoLA | SST-2 | MRPC | STS-B | QQP | MNLI | QNLI | RTE |
|---|---|---|---|---|---|---|---|---|---|
| Transformer + aux LM | 74.7 | 45.4 | 91.3 | 82.3 | 82.0 | **70.3** | **81.8** | **88.1** | **56.0** |
| Transformer, aux LM 제거 | **75.0** | **47.9** | **92.0** | **84.9** | **83.2** | 69.8 | 81.1 | 86.9 | 54.4 |
| **pre-training 제거** | **59.9** | 18.9 | 84.0 | 79.4 | 30.9 | 65.5 | 75.7 | 71.2 | 53.8 |
| LSTM + aux LM | 69.1 | 30.3 | 90.5 | 83.2 | 71.8 | 68.1 | 73.7 | 81.1 | 54.6 |

보조 loss는 평균으로 보면 오히려 손해다(74.7 대 75.0). 데이터가 큰 MNLI, QQP, QNLI에서만
도움이 되고 논문도 "큰 dataset은 득을 보고 작은 dataset은 아니다"라고 쓴다. 다만 RTE는 2.5k로
작은데도 득을 봤으니 법칙은 아니고 경향이다.

훨씬 큰 효과는 다른 줄에 있다. **pre-training을 빼면 평균이 14.8점 떨어진다.** LSTM으로
바꾸면 5.6점이다. "Transformer가 좋아서인가, pre-training이 좋아서인가"라는 질문에는 이 두
숫자로 답이 된다. 둘 다지만 pre-training 쪽이 두 배 반쯤 크다.

### 과제별 입력 형식과 출력층

<figure className="my-6 text-center">
  <img src="/images/posts/gpt-bert-in-context-learning/g1_fig1_right.png" alt="GPT-1의 task별 입력 변환" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">Figure 1 (right). task마다 입력을 하나의 token sequence로 바꾼다 (Radford et al., 2018)</figcaption>
</figure>

GPT-1의 덜 유명하지만 중요한 아이디어다. task마다 모델 구조를 새로 설계하는 대신, **입력을
전부 하나의 token sequence로 직렬화한다.**

- **Entailment**: 전제와 가설 사이에 delimiter를 끼운다.
- **Similarity**: 두 문장에 자연스러운 순서가 없으니 두 순서를 **모두** 인코딩해서 더한다.
- **Multiple choice**: 선택지를 한꺼번에 보여주지 않는다. `context + answer`를 후보마다 따로
  만들어 점수를 내고 softmax로 비교한다.

바뀌는 건 입력 형식과 마지막 linear head뿐이다. Transformer 본체는 건드리지 않는다.

### GELU의 확률적 해석

GPT-1은 ReLU 대신 GELU를 쓴다. 보통 "부드러운 ReLU" 정도로 설명하고 넘어가는데 원 논문의
유도를 따라가면 성격이 더 분명해진다.

입력 $x$를 확률 $\Phi(x)$로 통과시키는 확률적 gate를 생각해 보자. $\Phi$는 표준정규분포의
누적분포함수다.

$$m \sim \mathrm{Bernoulli}\big(\Phi(x)\big) \;\Rightarrow\; \mathbb{E}[m\,x] = x\,\Phi(x)$$

이 기댓값이 정확히 GELU다. **$x$가 클수록 통과 확률이 높아지는 gate의 평균값**인 셈이다.

dropout과 비교하면 차이가 선명해진다. dropout도 값을 확률적으로 통과시키지만 확률 $p$가
입력과 무관하다.

$$m \sim \mathrm{Bernoulli}(p) \;\Rightarrow\; \mathbb{E}[m\,x] = p\,x$$

기댓값이 $x$에 대해 **선형**이다. 그래서 dropout은 activation이 될 수 없고 regularization에만
쓰인다. 통과 확률이 입력에 의존하느냐 아니냐가 비선형성의 유무를 가른다.

<figure className="my-6 text-center">
  <img src="/images/posts/gpt-bert-in-context-learning/gelu.png" alt="GELU, ReLU, dropout 기댓값 비교" className="rounded-lg mx-auto max-w-md" />
  <figcaption className="mt-2 text-sm text-gray-500">GELU는 단조함수가 아니다. x ≈ −0.75에서 −0.17까지 내려갔다 올라온다</figcaption>
</figure>

그래프에서 눈에 띄는 건 GELU가 **단조증가가 아니라는 점**이다. 음수 구간에서 한 번 내려갔다가
올라온다. ReLU는 음수를 전부 0으로 죽이고 dropout의 기댓값은 원점을 지나는 직선이다.

### 미세조정 없이 평가한 성능

개인적으로 GPT-1에서 가장 흥미로웠던 분석이다.

저자들은 fine-tuning을 전혀 하지 않고 pre-trained 모델만으로 task를 푸는 방법을 만들었다.
감정 분류라면 문장 끝에 `very`를 붙이고 다음 단어로 `positive`와 `negative` 중 어느 쪽 확률이
높은지 본다. RACE라면 지문이 주어졌을 때 평균 log-probability가 가장 높은 선택지를 고른다.

<figure className="my-6 text-center">
  <img src="/images/posts/gpt-bert-in-context-learning/g1_fig2_right.png" alt="pre-training update에 따른 zero-shot 성능" className="rounded-lg mx-auto max-w-sm" />
  <figcaption className="mt-2 text-sm text-gray-500">Figure 2 (right). 실선이 Transformer, 점선이 LSTM (Radford et al., 2018)</figcaption>
</figure>

pre-training update 횟수에 따라 이 zero-shot 성능을 그린 게 위 그래프다. 실선인 Transformer는
꾸준히 올라가고 점선인 LSTM은 더 낮고 들쭉날쭉하다.

이게 **2018년에 이미 등장한 prompt**다. 문장 끝에 `very`를 붙여서 모델에게 "지금 감정을
판단하라"고 알려주는 것이기 때문이다. 그리고 곡선이 $10^6$ update에서도 아직 올라가는 중이다.
데이터와 모델을 더 키우면 어떻게 될까? 이 질문이 그대로 GPT-2의 출발점이 된다.

물론 GPT-1의 주력 결과는 여전히 fine-tuning이다. 12개 dataset 중 9개에서 SOTA를 갱신했고
Story Cloze 86.5(+8.9), RACE 59.0(+5.7), GLUE 72.8(이전 68.9)이었다.

## BERT: 양방향 문맥을 활용한 사전학습

### 양방향 문맥이 필요한 이유

<figure className="my-6 text-center">
  <img src="/images/posts/gpt-bert-in-context-learning/b_fig3.png" alt="BERT, GPT, ELMo의 pre-training 구조 비교" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">Figure 3. 왼쪽부터 BERT, OpenAI GPT, ELMo (Devlin et al., 2019)</figcaption>
</figure>

가운데 GPT는 각 token이 자기 왼쪽만 본다. 생성에는 자연스럽다. 다음 단어를 만들 때 오른쪽은
아직 존재하지도 않으니까. 하지만 NLI나 extractive QA처럼 문장 전체를 이해해야 하는 task에서는
오른쪽 정보를 쓰는 게 유리할 수 있다.

오른쪽 ELMo는 양방향 LSTM을 **따로** 학습해서 마지막에만 이어붙인다. 왼쪽 BERT는 encoder라서
모든 층에서 양쪽 문맥을 동시에 본다.

모든 token이 양쪽 문맥을 보면 자기 자신도 보게 된다. 그러면
"이 token을 맞혀라"는 문제가 공짜가 된다. 그래서 **다른 objective가 필요하다.**

### segment embedding은 순서가 아니라 소속을 표시한다

<figure className="my-6 text-center">
  <img src="/images/posts/gpt-bert-in-context-learning/b_fig2.png" alt="BERT의 입력 임베딩 구성" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">Figure 2. token + segment + position (Devlin et al., 2019)</figcaption>
</figure>

BERT의 입력은 세 임베딩의 합이다. token embedding은 어떤 subword인지, position embedding은
몇 번째인지를 나타낸다.

헷갈리기 쉬운 건 segment embedding이다. 이건 **순서가 아니다.** `[CLS] A [SEP] B [SEP]`처럼
두 문장을 넣었을 때 각 token이 A에 속하는지 B에 속하는지를 표시하는 값이다. 그림의 $E_A$, $E_B$가
그 값이다. 순서는 position embedding이 이미 담당하고 있다.

두 문장을 한 번에 넣는 이유는 NSP(Next Sentence Prediction) 때문이다. B가 실제로 A 다음 문장인지
맞히는 이진 분류이고 절반은 진짜 다음 문장, 절반은 무작위 문장이다. 최종 모델은 97~98% 정확도를
낸다. 참고로 이후 연구에서 NSP가 필수는 아니라는 결과가 여럿 나왔는데 이 글은 원 논문 기준으로
본다.

### MLM의 80 / 10 / 10 비율 비교

MLM은 token의 15%를 예측 대상으로 고르고 그중 80%는 `[MASK]`로, 10%는 무작위 token으로 바꾸고
10%는 그대로 둔 다음 원래 값을 맞히게 한다.

| 비율 | 입력 예시 |
|---|---|
| 80% | `my dog is [MASK]` |
| 10% | `my dog is apple` (무작위) |
| 10% | `my dog is hairy` (그대로) |

전부 `[MASK]`로 바꾸지 않는 이유로 보통 드는 설명은 train/test mismatch다. fine-tuning 때는
`[MASK]` token이 등장하지 않으니까. 설명은 그럴듯한데 **그 mismatch가 실제로 얼마나 문제일까?**

논문 부록 Table 8에 답이 있다.

| MASK | Same | Rnd | MNLI (fine-tune) | NER (fine-tune) | NER (feature) |
|---|---|---|---|---|---|
| **80%** | **10%** | **10%** | **84.2** | **95.4** | **94.9** |
| 100% | 0% | 0% | 84.3 | 94.9 | **94.0** |
| 80% | 0% | 20% | 84.1 | 95.2 | 94.6 |
| 80% | 20% | 0% | 84.4 | 95.2 | 94.7 |
| 0% | 20% | 80% | 83.7 | 94.8 | 94.6 |
| 0% | 0% | 100% | 83.6 | 94.9 | 94.6 |

MNLI를 fine-tuning으로 풀면 80/10/10이 84.2, 100% `[MASK]`가 84.3이다. 80/10/10 쪽이 **오히려
0.1 낮다.** 차이가 드러나는 건 맨 오른쪽 열, BERT를 freeze하고 feature extractor로만 쓰는
NER에서다. 94.9 대 94.0으로 거의 1점 벌어진다.

해석은 자연스럽다. fine-tuning을 하면 모델이 mismatch에 **적응할 기회**가 있지만 weight를
고정하면 그럴 수 없다. 그러니까 80/10/10은 "mismatch를 없애는 장치"라기보다 "weight를 고정해
쓸 때를 대비한 보험"에 가깝다.

### 같은 모델 크기에서 비교한 GLUE 성능

| | MNLI (m/mm) | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | Avg |
|---|---|---|---|---|---|---|---|---|---|
| *train 크기* | *392k* | *363k* | *108k* | *67k* | *8.5k* | *5.7k* | *3.5k* | *2.5k* | |
| OpenAI GPT | 82.1/81.4 | 70.3 | 87.4 | 91.3 | 45.4 | 80.0 | 82.3 | 56.0 | 75.1 |
| BERT-BASE | 84.6/83.4 | 71.2 | 90.5 | 93.5 | 52.1 | 85.8 | 88.9 | 66.4 | 79.6 |
| BERT-LARGE | **86.7/85.9** | **72.1** | **92.7** | **94.9** | **60.5** | **86.5** | **89.3** | **70.1** | **82.1** |

GLUE test 기준이다. QQP와 MRPC는 F1, STS-B는 Spearman, 나머지는 accuracy다.

여기서 중요한 건 BERT-BASE가 **일부러 GPT와 같은 크기로 맞춘 모델**이라는 점이다. 크기를 통제한
상태에서 평균 4.5점이 올라갔으니 bidirectional representation과 MLM objective의 효과로 볼 수
있다.

차이가 큰 곳도 패턴이 있다. CoLA가 +6.7, RTE가 +10.4인데 둘 다 train 데이터가 작은 쪽이다.
representation이 좋을수록 데이터가 부족한 task에서 이득이 크다.

다만 **BERT도 dataset마다 전체 파라미터를 update한다.** representation은 더 일반적이 됐지만
downstream 적응 방식 자체는 GPT-1과 똑같다. task 정보는 여전히 weight에 들어간다.

## GPT-2: 프롬프트로 과제 알려주기

### 과제를 텍스트로 전달한다는 가설과 WebText

GPT-2는 질문 자체를 바꾼다. 입력만이 아니라 입력과 **task**가 주어졌을 때 출력을 내는 모델,
즉 $p(\text{output} \mid \text{input}, \text{task})$를 모델링하되 그 task를 별도 head가 아니라
**텍스트로** 전달하겠다는 것이다.

그럼 모델은 task를 어디서 배울까? 논문은 학습 데이터에서 실제로 찾은 문장을 보여준다.
"프랑스어로 말하자면, Je ne suis pas un imbecile" 같은 문장이 글 속에 자연스럽게 섞여 있다.
충분히 크고 다양한 텍스트로 학습하면 이런 대응 관계도 함께 배운다는 가정이다.

그래서 WebText를 만든다. Reddit에서 3 karma 이상 받은 외부 링크를 모은 것으로 약 800만 문서
40GB다. 모델은 최대 48층 1,542M이고 context는 512에서 **1,024로** 늘렸다. vocab은 50,257이다.

| Params | Layers | $d_{\text{model}}$ |
|---|---|---|
| 117M | 12 | 768 |
| 345M | 24 | 1024 |
| 762M | 36 | 1280 |
| **1542M** | **48** | **1600** |

구조에서 하나 바뀐 게 LayerNorm을 각 block의 **입력 쪽으로** 옮긴 것이다. 흔히 Pre-LN이라고
부르는 배치다. [ViT 리뷰](/posts/2026-09-01-vit-an-image-is-worth-16x16-words)에서
ViT가 Pre-LN을 쓴다고 적었는데 GPT-2가 그보다 먼저 같은 배치를 쓰고 있었다.

### `TL;DR:`를 붙였을 때의 요약 성능

GPT-2는 모든 task를 next-token prediction으로 바꾼다.

| task | prompt |
|---|---|
| 요약 | `<article> TL;DR:` |
| 번역 | `en = fr` × k, 그리고 `en =` |
| 질의응답 | `<doc> <history> A:` |

여기서 zero-shot은 "학습을 안 했다"가 아니다. **평가 task의 데이터로 weight를 update하지
않았다**는 뜻이다. WebText로는 당연히 학습했다.

task 정보가 정말 prompt에 들어 있다는 가장 직접적인 증거가 요약 실험이다.

| CNN / Daily Mail | ROUGE avg |
|---|---|
| Bottom-Up Sum (SOTA) | 32.75 |
| Lede-3 | 31.55 |
| Seq2Seq + Attn | 23.99 |
| **GPT-2, TL;DR: 붙임** | **21.40** |
| Random-3 | 20.98 |
| **GPT-2, 힌트 제거** | **15.03** |

같은 모델에서 프롬프트만 바꾸었는데 **점수가 6.4점 달라진다.** 두 토큰짜리 힌트가 출력에 영향을 준다는 뜻이다.

동시에 이 표는 GPT-2의 한계도 같이 보여준다. 21.40은 기사 앞 세 문장을 그냥 가져오는
Lede-3(31.55)보다 한참 낮고 무작위로 세 문장 뽑는 Random-3(20.98)보다 겨우 0.42 높다.
prompt가 작동하긴 하는데 잘하지는 못한다.

### GPT-2와 GPT-3에서 달라진 zero-shot의 정의

위 표에서 번역 prompt를 다시 보자. `en = fr` 예시 쌍을 **여러 개 넣고** 시작한다.

weight update가 없다는 의미에서는 분명히 zero-shot이다. 하지만 context에 풀린 예시가 들어
있다는 점에서는 GPT-3가 1년 뒤에 few-shot이라고 부르게 될 바로 그 구조다. 같은 논문 안에서도
요약은 `TL;DR:`만 붙이는 진짜 zero-shot인데 번역은 예시를 넣는다.

용어가 정리되기 전의 과도기라고 보는 게 맞을 것 같다. GPT-3가 zero / one / few를 명시적으로
나눈 게 그래서 의미가 있다.

### 과제별 성능 차이

| Task (metric) | GPT-2 1.5B | 비교 대상 |
|---|---|---|
| LAMBADA (acc) | **63.24** | SOTA 59.23 |
| LAMBADA (PPL ↓) | **8.63** | SOTA 99.8 |
| CoQA (F1) | 55 | human 89 |
| CNN/DM (ROUGE avg) | 21.40 | SOTA 32.75 |
| WMT-14 Fr→En (BLEU) | 11.5 | best unsupervised MT 33.5 |
| WMT-14 En→Fr (BLEU) | 5 | 단어 단위 치환 수준 |
| Natural Questions (EM) | 4.1% | 없음 |

전부 zero-shot이다. 패턴이 보이는데 **language modeling과 닮은 task일수록 강하고 특정 출력
형식이 필요할수록 약하다.** LAMBADA는 긴 문단 끝 단어를 맞히는 거라 next-token prediction
그 자체라서 SOTA를 넘었다. perplexity는 99.8에서 8.63으로 한 자릿수가 됐다. 반면 요약과
질의응답은 "이런 형태로 답하라"는 규약이 필요해서 크게 밀린다.

번역 쪽에 재미있는 이야기가 하나 있다. WebText를 만들 때 **영어가 아닌 페이지를 일부러
걸러냈다.** 그런데 언어 감지기를 돌려 확인해 보니 프랑스어가 10MB 남아 있었다. 기존 번역
연구가 쓰던 코퍼스보다 약 **500배 작은** 양이다. 그 정도만으로도 프랑스어→영어 11.5 BLEU가
나왔다.

반대 방향인 영어→프랑스어는 5 BLEU로 훨씬 낮은데 논문은 강한 영어 language model에 기대고
있기 때문이라고 해석한다. 영어를 만들어내는 능력은 좋지만 프랑스어를 만들어내는 능력은 없는 셈이다.

그래서 GPT-2의 기여는 benchmark 정복이 아니다. 고정된 모델 하나가 자연어 지시만으로 여러 task를
"하기 시작했다"는 방향성이다.

## GPT-3: 모델 규모와 문맥 속 예시

### 예시를 문맥에 넣는 방식과 미세조정의 차이

zero-shot은 task 설명과 query만, one-shot은 예시 하나, few-shot은 예시 여럿을 context에 넣는다.

```text
Zero-shot
  Translate English to French:
  cheese =>

One-shot
  Translate English to French:
  sea otter => loutre de mer
  cheese =>

Few-shot
  Translate English to French:
  sea otter => loutre de mer
  peppermint => menthe poivrée
  plush girafe => girafe peluche
  cheese =>
```

demonstration은 **정답이 포함된 입출력 쌍**이다. 직관적으로는 attention할 재료를 더 많이 주는
것으로 보면 된다. query 위치의 hidden state가 앞의 예시들을 attention하면서 task의 형식과
mapping을 추론하는 것이다.

fine-tuning이었다면 예시를 학습에 사용해 가중치를 갱신했을 것이다. 여기서는 세 설정 모두 **weight가
그대로다.** context를 지우면 적응한 상태도 같이 사라진다.

### GPT-2 이후의 모델·데이터 규모 확대

| Model | $n_{\text{params}}$ | Layers | $d_{\text{model}}$ | Heads | Batch | LR |
|---|---|---|---|---|---|---|
| Small | 125M | 12 | 768 | 12 | 0.5M | $6.0\times10^{-4}$ |
| Medium | 350M | 24 | 1024 | 16 | 0.5M | $3.0\times10^{-4}$ |
| Large | 760M | 24 | 1536 | 16 | 0.5M | $2.5\times10^{-4}$ |
| XL | 1.3B | 24 | 2048 | 24 | 1M | $2.0\times10^{-4}$ |
| 2.7B | 2.7B | 32 | 2560 | 32 | 1M | $1.6\times10^{-4}$ |
| 6.7B | 6.7B | 32 | 4096 | 32 | 2M | $1.2\times10^{-4}$ |
| 13B | 13.0B | 40 | 5140 | 40 | 2M | $1.0\times10^{-4}$ |
| **GPT-3** | **175.0B** | **96** | **12288** | **96** | **3.2M** | $\mathbf{0.6\times10^{-4}}$ |

batch size는 token 단위다. 8개 크기 모두 **300B token**을 학습하고 context는 2,048이다.

architecture는 GPT-2와 같다. 예외가 하나 있는데 layer마다 dense attention과 locally banded
sparse attention을 번갈아 썼다는 **한 문장**이다.

<figure className="my-6 text-center">
  <img src="/images/posts/gpt-bert-in-context-learning/attn_patterns.png" alt="dense, banded, strided, fixed attention 패턴" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">행이 query, 열이 key. Child et al. (2019)의 정의대로 n=16, l=4로 다시 그렸다</figcaption>
</figure>

왼쪽 둘이 GPT-3가 명시한 것이다. dense는 앞의 모든 token을 보고 banded는 가까운 token만 봐서
계산량이 줄어든다. 오른쪽 strided와 fixed는 Sparse Transformer가 제안한 다른 패턴인데
**GPT-3가 이 넷을 다 썼다고 말하면 안 된다.** 논문이 명시한 건 dense와 banded의 교대뿐이고
band 폭도 layer 비율도 적혀 있지 않다.

표에서 눈에 띄는 또 하나는 모델이 커질수록 batch는 커지고 learning rate는 작아진다는 점이다.
보편 법칙이라기보다 큰 모델의 학습 안정성을 위한 경험적 선택, 즉 이 논문의 recipe로 보는 게
정확하다.

### 문맥 속 예시를 늘렸을 때의 성능 변화

<figure className="my-6 text-center">
  <img src="/images/posts/gpt-bert-in-context-learning/g3_fig3_8.png" alt="SuperGLUE 성능과 context 내 예시 수" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">Figure 3.8. 오른쪽은 175B 모델에서 context 내 예시 수 K를 늘린 결과 (Brown et al., 2020)</figcaption>
</figure>

오른쪽 그래프는 이 논문에서 특히 인상적인 그림이다. $K$를 1에서 8로 늘렸을 때의 향상
폭이, 점선으로 그어진 **BERT-Large와 BERT++의 차이**와 비슷하다.

BERT++는 SuperGLUE 전에 MultiNLI 392k와 SWAG 113k로 추가 fine-tuning을 거친 모델이다. 수십만 건의
labeled 데이터로 가중치를 갱신해서 얻은 향상 폭을, **예시 7개를 더 넣어서** 비슷하게 얻었다는
이야기다.

다만 조심할 게 있다. GPT-3 곡선은 dev set이고 점선은 그렇지 않아서 완전히 같은 조건이
아니다. "대체했다"가 아니라 "향상 폭이 비슷하다" 정도로 말하는 게 맞다.

### GPT-2와 GPT-3의 공통 평가 결과

네 논문을 억지로 한 표에 넣으면 조건이 달라 공정한 비교가 안 된다. 그래서 GPT-2와 GPT-3가
**둘 다 보고한** task만 골랐다. LAMBADA와 CoQA 두 개다.

| | GPT-2 zero | GPT-3 zero | GPT-3 one | GPT-3 few |
|---|---|---|---|---|
| LAMBADA (acc) | 63.2 | 76.2 | **72.5** | **86.4** |
| CoQA (F1) | 55 | 81.5 | 84.0 | **85.0** |

두 단계가 깔끔하게 보인다. 먼저 **scale이 zero-shot을 끌어올린다.** 63에서 76으로, 55에서
82로. 같은 방식인데 모델만 커졌다. 그다음 **weight update 없이 demonstration이 추가 이득을
준다.** 76에서 86으로.

그런데 LAMBADA 줄의 one-shot 72.5를 보자. **zero-shot 76.2보다 낮다.** 예시를 하나 줬더니
오히려 나빠진 것이다.

논문의 설명이 흥미롭다. GPT-3는 LAMBADA few-shot에서 빈칸 채우기 형식을 쓴다.

```text
Alice was friends with Bob. Alice went to visit her friend ___. → Bob
```

예시가 하나뿐이면 모델이 이 **형식**을 알아채지 못하고 여러 개 주면 알아채서 86.4까지 올라간다.
그러니까 demonstration이 전달하는 건 정답 내용만이 아니라 **task의 출력 형식**이기도 하다.

### 모델 규모와 자릿수별 산술 성능

<figure className="my-6 text-center">
  <img src="/images/posts/gpt-bert-in-context-learning/g3_fig3_10.png" alt="모델 크기별 산술 정확도" className="rounded-lg mx-auto max-w-md" />
  <figcaption className="mt-2 text-sm text-gray-500">Figure 3.10. 자릿수별 덧셈·뺄셈 정확도 (Brown et al., 2020)</figcaption>
</figure>

산술은 scale에 따라 매끄럽게 좋아지지 않는다. 13B는 두 자리 덧셈도 절반밖에 못 맞히는데
175B는 다르다.

| 문제 | 175B few-shot |
|---|---|
| 두 자리 덧셈 | 100.0% |
| 세 자리 덧셈 | 80.4% |
| 네 자리 덧셈 | 25.5% |
| 다섯 자리 덧셈 | 9.3% |

두 자리는 완벽하고 세 자리도 80%인데 네 자리부터 무너진다. 개별 능력이 어느 규모를 넘을 때
갑자기 나타나는 것처럼 보일 수 있다는 예시로는 좋지만 **연산을 배웠다고 말하기는 어렵다.**
진짜 덧셈을 배웠다면 자릿수에 따라 이렇게 붕괴하지 않을 것이다.

논문에는 세상에 없는 단어를 정의해 주고 문장에 써 보게 한 예시도 있다. "farduddle이란 빠르게
위아래로 뛰는 것을 뜻한다"고 알려주자 그 단어를 넣은 문장을 만들어낸다. 다만 이건 논문이
골라서 보여준 사례라 benchmark가 아니라 **demo**로 봐야 한다.

### 모델을 키워도 해결되지 않은 과제

| | GPT-3 175B | 비교 대상 |
|---|---|---|
| ANLI R3 (acc) | few 40.2 | 찍기 33.3 · fine-tuned SOTA 48.3 |
| RACE-m (acc) | few 58.1 | **fine-tuned GPT-1 62.9** |
| RACE-h (acc) | few 46.8 | **fine-tuned GPT-1 57.4** |
| LAMBADA (acc) | one 72.5 | 자기 자신의 zero-shot 76.2 |

ANLI Round 3은 사람이 모델을 일부러 틀리게 만들도록 설계한 적대적 NLI인데 few-shot 40.2%는
찍었을 때 33.3%와 그리 멀지 않다. fine-tuned SOTA 48.3과는 차이가 크다.

RACE가 더 극적이다. **175B GPT-3가 이 글 맨 앞에서 본 fine-tuned GPT-1보다 낮다.**
파라미터가 약 1,500배 차이인데 그 task에 직접 학습한 117M 모델이 이긴다. 규모가 적응 방식을
언제나 대체하지는 못한다는 것을 보여주는 비교다.

그리고 contamination 문제가 있다. 웹 데이터로 학습했으니 benchmark 테스트셋이 섞여 들어갔을
수 있다. 저자들도 지우려고 했는데 논문에 이렇게 적혀 있다.

> 안타깝게도 **필터링의 버그 때문에** 일부 중복을 놓쳤고, 학습 비용 때문에 **모델을 다시 학습시키는
> 것은 현실적이지 않았다.**

대신 겹치지 않는 clean subset으로 따로 평가했다. LAMBADA는 실제로 contamination이 있었지만
clean subset과 점수 차이가 0.5% 이내였다고 보고한다. 정직하게 적어둔 건 좋지만 숫자를 읽을 때
염두에 둬야 하는 조건이다.

그래서 GPT-3의 결론은 "fine-tuning은 끝났다"가 아니다. **in-context learning이라는 새로운 적응
방식이 가능해졌다**는 쪽이 정확하다.

## 정리

| | GPT-1 | BERT | GPT-2 | GPT-3 |
|---|---|---|---|---|
| Backbone | causal decoder | encoder | causal decoder | causal decoder |
| Pre-training | next token | MLM + NSP | next token | next token |
| Downstream | 전체 fine-tune | 전체 fine-tune | prompt | prompt + 예시 |
| weight 변경 | 예 | 예 | **아니오** | **아니오** |
| task 정보 위치 | weight | weight | 문구와 형식 | 문구 + 예시 |

GPT-1과 BERT는 backbone도 objective도 다르지만 **dataset마다 weight를 update한다**는 점에서
같은 범주다. task 정보가 weight에 들어간다.

GPT-2와 GPT-3는 decoder를 유지한 채 task를 prompt로 옮긴다. GPT-2는 문구와 형식을, GPT-3는
예시까지 전달한다.

정리하면 이렇다. **Transformer 구조는 거의 그대로였고 바뀐 것은 task 설명을 어디에 두느냐와
downstream 학습이 언제 일어나느냐다.**

## GPT-3 논문이 제안한 양방향 모델

흐름이 한 방향으로만 가지는 않는다. GPT-3 논문의 Limitations 절에 이런 대목이 있다.

> 우리 실험에는 bidirectional architecture가 전혀 포함되지 않았다. (...) 경험적으로
> bidirectionality의 덕을 보는 task에서는 성능이 더 나빴을 수 있다.

저자들이 직접 WiC, ANLI, RACE를 그런 task로 지목한다. 두 문장을 비교하거나 긴 지문을 여러 번
되짚어야 하는 것들이다. 그리고 **few-shot 능력을 가진 큰 bidirectional 모델**이 "두 세계의 장점을
모두" 가질 수 있을 거라고 쓴다.

BERT가 2018년에 제기한 문제를, 네 번째 논문이 2020년에 다시 열어둔 채로 끝난다. 네 논문을
한 줄로 읽으면 깔끔한 진보처럼 보이지만 실제로는 한 축을 밀어붙이는 동안 다른 축을 미뤄둔
것에 가깝다.

## 세미나에서 다룬 추가 질문

**bigger model은 그냥 overfit하는 것 아닌가?**

BERT 논문 Table 6을 보면 train 데이터가 3,600개뿐인 MRPC에서도 모델이 클수록 계속 좋아진다.

| L | H | A | LM ppl | MNLI-m | MRPC | SST-2 |
|---|---|---|---|---|---|---|
| 3 | 768 | 12 | 5.84 | 77.9 | 79.8 | 88.4 |
| 6 | 768 | 3 | 5.24 | 80.6 | 82.2 | 90.7 |
| 6 | 768 | 12 | 4.68 | 81.9 | 84.8 | 91.3 |
| 12 | 768 | 12 | 3.99 | 84.4 | 86.7 | 92.9 |
| 12 | 1024 | 16 | 3.54 | 85.7 | 86.9 | 93.3 |
| 24 | 1024 | 16 | 3.23 | **86.6** | **87.8** | **93.7** |

pre-training으로 이미 좋은 representation을 갖고 출발하기 때문이다.

<figure className="my-6 text-center">
  <img src="/images/posts/gpt-bert-in-context-learning/g2_fig4.png" alt="GPT-2의 train/test perplexity" className="rounded-lg mx-auto max-w-xs" />
  <figcaption className="mt-2 text-sm text-gray-500">Figure 4. train과 test perplexity가 함께 떨어진다 (Radford et al., 2019)</figcaption>
</figure>

GPT-2 쪽은 더 직접적이다. train과 test perplexity가 **함께** 떨어지고 있어서 1.5B도 아직
underfit 상태라는 뜻이다. GPT-3에서 규모를 더 키운 근거가 여기 있다.

**metric이 논문마다 달라서 숫자를 직접 비교하면 안 되는 경우가 있다**

STS-B가 대표적이다. **GPT-1 표는 Pearson, BERT의 GLUE 표는 Spearman**을 쓴다. 둘 다
상관계수지만 전자는 선형 상관, 후자는 순위 상관이라 같은 예측에서도 값이 다르게 나온다.

CoLA에 MCC를 쓰는 이유도 비슷하다. label이 불균형할 때 accuracy가 과대평가되는 걸 막기
위해서다.

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

perplexity는 cross-entropy에 $\exp$를 취한 값인데 "평균적으로 몇 개의 후보 사이에서 헷갈리는가"로
읽으면 직관적이다. GPT-2의 LAMBADA perplexity가 99.8에서 8.63으로 떨어졌다는 건 후보가 백 개
수준에서 열 개 아래로 줄었다는 뜻이다.

**in-context learning은 fine-tuning과 뭐가 다른가**

weight가 바뀌지 않는다. 적응이 forward pass 안에서 attention으로 일어나고 context를 지우면
사라진다. fine-tuning은 예시마다 gradient step을 밟아 파라미터에 흔적을 남긴다.

답하기 까다로운 질문도 있었다. "BERT가 더 좋았는데 왜 이후에 GPT 계열이 주류가 됐는가"인데
**이 네 논문만으로는 답할 수 없다.** 논문들이 보여주는 건 GPT-3가 prompt만으로 적응하는
방식을 열었다는 데까지이고 정작 GPT-3 저자들도 bidirectional 모델이 fine-tuning에서는 더 강할
것이라고 봤기 때문이다.

## 이어지는 글

- [Attention Is All You Need: 설명이 항상 멈추는 자리들](/posts/2026-09-01-attention-is-all-you-need)
- [ViT 리뷰: inductive bias는 데이터로 대체할 수 있는가](/posts/2026-09-01-vit-an-image-is-worth-16x16-words)

## 참고문헌

- Radford, A., Narasimhan, K., Salimans, T., Sutskever, I. (2018). *Improving Language Understanding by Generative Pre-Training*.
- Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners*.
- Brown, T. et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS.
- Hendrycks, D., Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs)*.
- Child, R., Gray, S., Radford, A., Sutskever, I. (2019). *Generating Long Sequences with Sparse Transformers*.
