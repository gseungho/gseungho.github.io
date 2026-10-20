---
title: "ViT 리뷰: inductive bias는 데이터로 대체할 수 있는가"
date: "2026-09-01"
category: "논문리뷰"
subcategory: "Deep Learning/Foundations"
tags: ["논문리뷰", "ViT", "Transformer", "컴퓨터비전"]
description: "An Image is Worth 16×16 Words(ICLR 2021) 리뷰다. CNN보다 이미지 구조를 덜 가정한 Transformer가 대규모 pre-training으로 CNN을 넘어서는 과정을 따라간다. 데이터 규모에 따라 결론이 뒤집히는 실험과 ViT가 학습한 위치 임베딩, attention distance도 함께 살펴본다."
---

학부 논문 리뷰 세미나에서 발표한 **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale** (Dosovitskiy et al., ICLR 2021) 리뷰다. 발표 내용을 글로 옮기면서 발표 때 시간에 쫓겨 줄였던 부분을 다시 채워 넣었다.

논문 제목에는 핵심 아이디어가 담겨 있다. **이미지를 16×16 패치로 잘라서 그 패치들을 단어처럼 취급하겠다**는 뜻이다.

이 글 전체를 관통하는 질문은 하나다.

> **ViT에는 2D 구조를 얼마나 미리 알려 줘야 하는가?**

구조와 실험 결과를 이 질문에 맞춰 살펴보자.

이 글은 [Attention Is All You Need 리뷰](/posts/2026-09-01-attention-is-all-you-need)의 다음 편이다. 거기서 다룬 Pre-LN, positional encoding, residual connection 이야기가 여기서 그대로 이어지니 Transformer 구조가 낯설다면 그쪽을 먼저 봐도 좋다.

## 2020년의 NLP와 컴퓨터 비전

2020년 당시 상황을 보면 NLP와 Vision이 완전히 다른 길을 가고 있었다.

**NLP 쪽**은 Transformer가 이미 표준이었다. 큰 코퍼스로 pre-train하고 작은 task로 fine-tune하는 방식이 정착됐고 파라미터를 100B까지 키워도 성능이 포화될 기미가 보이지 않았다.

**Vision 쪽**은 여전히 CNN이 지배하고 있었다. attention을 붙이려는 시도는 많았지만 대부분 CNN 구조를 그대로 두고 일부만 교체하는 식이었다. 아예 conv를 다 걷어낸 연구들도 있었는데 특수한 attention 패턴을 쓰다 보니 GPU/TPU에서 가속이 잘 되지 않았다. 이론적으로 효율적이어도 실제로는 느렸던 것이다.

그래서 이 논문의 질문은 단순하다. **CNN을 완전히 버리고 표준 Transformer를 최소한만 고쳐서 그대로 쓰면 어떻게 되는가?**

## 이 논문의 진짜 기여는 아키텍처가 아니다

ViT의 기여를 이해하려면 앞선 연구와 무엇이 다른지 살펴봐야 한다.

초록만 읽으면 "Transformer를 이미지에 처음 적용했다"로 들린다. 그런데 그건 사실이 아니다.

Related Work에서 저자들이 직접 밝히고 있는데 **Cordonnier et al. (2020)이** 이미 2×2 패치를 뽑아 full self-attention을 적용했다. 구조적으로 ViT와 매우 유사하고 논문도 "most related to ours"라고 인정한다.

차이는 두 가지다.

- 패치가 2×2로 너무 작아서 **저해상도 이미지에만** 쓸 수 있었다
- 결정적으로 **대규모 pre-training을 하지 않았다**

그래서 **이 논문의 진짜 기여는 아키텍처가 아니라 증거다.** "vanilla Transformer도 pre-training 규모만 충분하면 SOTA CNN을 이긴다"는 것을 중간 해상도에서 3억 장 규모로 실증한 것이다.

이 글에서는 특히 대규모 pre-training이 성능에 어떤 영향을 주었는지 살펴본다.

## 이미지 패치에서 분류 결과까지

<Fig src="/images/posts/vit-16x16-words/fig_overview.png" alt="ViT 전체 구조와 Transformer Encoder block">Figure 1. 왼쪽이 ViT 전체, 오른쪽이 encoder block 하나 (Dosovitskiy et al., 2021)</Fig>

ViT는 **Transformer의 encoder만 쓴다.** decoder가 없다.

번역기는 출력이 문장이니까 한 단어씩 순차적으로 뽑아야 했고 그래서 masked self-attention과 decoder가 필요했다. 그런데 **분류는 출력이 클래스 하나**다. 순차적으로 뽑을 게 없으니 decoder가 할 일이 없다. 덕분에 masking, decoder self-attention, encoder-decoder attention이 전부 사라진다. 그만큼 단순해진다.

왼쪽 아래부터 위로 네 단계로 따라가 보자.

### 1단계: 이미지를 패치로 나누기

Transformer는 `(B, sequence 길이, D)` 이 모양만 받는다. 이미지를 어떻게든 그 모양으로 바꾸는 게 첫 번째 문제다.

224×224 이미지를 16×16 패치로 자르면 다음과 같다.

- 패치 개수 $N = 224 \times 224 / 16^2 = 196$
- 패치 하나를 펴면 길이가 $16 \times 16 \times 3 = 768$

그래서 `(B, 3, 224, 224)` → `(B, 196, 768)`이 된다. **여기서 196이 곧 sequence 길이다.** 문장에서 단어 개수에 해당하는 자리다. 패치 하나 = 단어 하나. 제목의 "16×16 words"가 이 얘기다.

**여기서 vanilla와 첫 번째 차이가 나온다.** NLP에서는 단어가 one-hot 인덱스로 들어와서 `nn.Embedding`이 테이블에서 행을 뽑아온다. 그런데 패치는 연속적인 픽셀 값이다. 1482번째 행 같은 걸 뽑을 수가 없다. 그래서 lookup이 아니라 **선형변환**이 필요하다.

실제 구현에서는 이걸 한 줄로 처리한다.

```python
# 자르기 + 선형변환을 동시에
patch_embed = nn.Conv2d(3, D, kernel_size=16, stride=16)
```

커널 크기와 stride를 둘 다 패치 크기로 맞추면 패치끼리 겹치지 않고 딱 잘리기 때문이다. 수학적으로 완전히 같다.

### 2단계: 패치 임베딩

Figure 1의 분홍색 박스에 "Linear Projection of Flattened Patches"라고 적혀 있는데 **이게 패치 임베딩이다.** 같은 변환을 가리킨다. 나는 처음에 이게 패치 임베딩 앞에 오는 별도 단계인 줄 알았는데 아니었다.

`(B, 196, 768)`을 행렬 $E$에 통과시키면 `(B, 196, D)`가 된다. 두 가지만 짚고 넘어가자.

**첫째, $E$는 학습되는 행렬 하나이고 196개 패치가 전부 같은 $E$를 통과한다.** 패치마다 다른 행렬을 쓰지 않는다. weight sharing이다.

**둘째, 그래서 패치 개수 $N$이 바뀌어도 $E$를 그대로 쓸 수 있다.** 이 성질은 뒤에서 해상도를 바꿀 때 다시 살펴본다.

여기서 입출력 차원이 같다고 변환이 없는 것은 아니다. ViT-B/16은 우연히 $16^2 \times 3 = 768$이고 $D$도 768이라 입출력 차원이 똑같다. "그냥 통과시키는 거 아닌가" 싶지만 우연의 일치일 뿐이고 여전히 학습되는 선형변환이다. ViT-L/16을 보면 768 → 1024로 명확히 바뀐다.

### 3단계: CLS token과 위치 임베딩

논문의 식 (1)이다.

$$\mathbf{z}_0 = [\,\mathbf{x}_{\text{class}};\ \mathbf{x}_p^1 E;\ \mathbf{x}_p^2 E;\ \cdots;\ \mathbf{x}_p^N E\,] + E_{\text{pos}}$$

식에서 concat과 덧셈을 구분해야 한다. **대괄호는 concat**, 세로로 쌓는 것이다. 그런데 뒤에 붙은 $+\,E_{\text{pos}}$는 **진짜 덧셈**이다. 한 줄에 두 개의 다른 연산이 섞여 있다. Figure 1 그림에서는 위치 임베딩이 concat처럼 보이게 그려져 있는데 더하는 게 맞다.

- $\mathbf{x}_{\text{class}}$는 BERT의 `[class]` token이다. 이미 $D$차원인 **학습 파라미터**라서 $E$를 통과하지 않는다. 통과시킬 원본 픽셀이 없기 때문이다.
- $E_{\text{pos}}$는 $(N+1) \times D$, 즉 **197행**이다. CLS 자리까지 포함한 수다.

정리하면 `(B, 196, D)` → CLS 붙여서 `(B, 197, D)` → $E_{\text{pos}}$ 더해서 그대로 `(B, 197, D)`.

**여기가 vanilla와의 두 번째 차이다.** 2017년 Transformer는 sin/cos 공식으로 만든 **고정된** positional **encoding**을 썼다. ViT는 **학습되는** positional **embedding**을 쓴다. 공식이 아니라 파라미터다. 뒤에서는 학습된 위치 임베딩의 패턴을 살펴본다.

### CLS token은 왜 필요한가

분류를 하려면 결국 고정 크기 벡터 하나가 필요한데 encoder를 나오면 197개가 남아 있다. 이 중에 뭘 써야 할까?

가장 단순한 방법은 196개를 그냥 평균내는 것(GAP)이다. 그런데 이건 모든 패치를 똑같은 weight $1/196$으로 취급하는 것이다. 배경 하늘 패치나 고양이 얼굴 패치나 똑같이.

CLS는 다른 접근이다. **정보가 하나도 없는 빈 슬롯**을 앞에 하나 끼워 넣고 self-attention을 거치면서 다른 패치들의 정보를 **학습된 weight로** 끌어모으게 한다. 즉 "무엇을 요약할지"까지 학습 대상이 되는 것이다. attention의 weighted sum을 요약이라는 목적에 그대로 쓰는 셈이다.

여기서 decoder의 `<sos>` token과 헷갈리면 안 된다.

- `<sos>`는 **masking 아래에서** 다음 token을 하나씩 예측하기 시작하는 신호다
- **CLS는 masking이 없어서** 첫 layer부터 196개 패치를 전부 본다. 그리고 한 번의 최종 출력으로 끝난다

둘 다 "앞에 붙이는 특수 token"이라 비슷해 보이지만 역할이 완전히 다르다.

### 4단계: Transformer encoder

식 (2), (3), (4)다.

$$\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}, \qquad \ell = 1 \ldots L$$

$$\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell, \qquad \ell = 1 \ldots L$$

$$\mathbf{y} = \text{LN}(\mathbf{z}_L^0)$$

식 (2)가 attention block, 식 (3)이 MLP block이다. 보면 **LN을 먼저 하고 block을 통과하고 그 다음에 더한다.** 2017년 논문의 순서와 다르다. 이건 다음 절에서 따로 다루겠다.

식 (4)가 중요하다. $\mathbf{z}_L$의 위첨자 0, 즉 마지막 layer의 **0번째 행**이다. **CLS 자리다.** 나머지 196개는 분류에 쓰지 않고 버린다. 여기에 LN을 한 번 더 걸고 MLP head로 보낸다.

모델 크기는 세 가지다.

| Model | Layers | Hidden size $D$ | MLP size | Heads | Params |
|---|---|---|---|---|---|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

Base와 Large는 **BERT의 설정을 그대로 가져온 것**이고 Huge만 새로 추가했다. "최소한만 바꾼다"는 논문의 태도와 맞아떨어진다.

모델 이름의 표기도 살펴보자. ViT-L/16은 "Large 크기 + 패치 16×16"이다. 그리고 sequence 길이는 패치 크기의 제곱에 반비례하니까 /16이 /32보다 패치가 4배 많고 훨씬 비싸다.

## 원래 Transformer와 달라진 여섯 가지

| | 2017 Transformer | ViT |
|---|---|---|
| 구성 | encoder + decoder | **encoder만** |
| LayerNorm | Post-LN | **Pre-LN** |
| activation 함수 | ReLU | **GELU** |
| 위치 정보 | sinusoidal encoding (고정) | **learned embedding** |
| 요약 token | 없음 | **CLS token** |
| 입력 임베딩 | 테이블에서 행 뽑기 | **선형변환** |

**Pre-LN과 GELU 조합은 ViT가 발명한 게 아니다.** GPT-2가 이미 쓰던 방식이다. 즉 저자들은 새로운 걸 만든 게 아니라 **NLP에서 이미 검증된 걸 그대로 들고 온 것**이다.

이것도 논문의 태도와 맞아떨어진다. Method 섹션 첫 문장이 "원래 Transformer를 가능한 한 가깝게 따랐다"인데 NLP에서 쓰던 구현을 거의 그대로 활용할 수 있다는 실용적인 이유도 제시한다.

### Pre-LN을 쓰는 이유

LN을 앞에 두느냐 뒤에 두느냐, 괄호 위치만 바뀐 것처럼 보이는데 왜 중요할까? residual connection을 통과하는 gradient를 보면 드러난다.

**Post-LN**은 $\text{LN}(x + F(x))$ 형태다. 미분하면 identity 항 $I$가 **LayerNorm의 Jacobian과 곱해진다.** residual connection을 만든 이유가 "$I$를 남겨서 gradient가 사라지지 않게 하는 것"이었는데 그 $I$가 다른 것과 곱해지면서 보장이 깨지는 것이다.

**Pre-LN**은 $x + F(\text{LN}(x))$ 형태다. 여기서는 $x$ 자체가 LN을 거치지 않는다. LN은 $F$의 입력을 만드는 데만 쓰이고 $x$는 더하기 직전까지 원본 그대로 살아남는다. 그래서 미분하면 $I + \delta$로 깔끔하게 남는다.

$L$개 layer를 지나면 차이가 명확해진다.

$$\prod_{\ell=1}^{L} (I + \delta_\ell) \qquad \text{vs.} \qquad \prod_{\ell=1}^{L} J_{\text{LN}}^{(\ell)} (I + \delta_\ell)$$

왼쪽은 $\delta$가 0이어도 각 항이 최소 1이라 절대 0으로 가지 않는다. 오른쪽은 $J_{\text{LN}}$이 layer마다 한 번씩 곱해지면서 누적된다.

layer 수가 늘어나면 이 차이가 더 중요해진다. 2017년 encoder는 6 layer라 Post-LN으로도 견뎠다. 그런데 ViT-Huge는 32 layer다. 이 정도 깊이에서는 Pre-LN이 아니면 학습이 불안정해진다.

대가가 없는 건 아니다. Pre-LN은 residual stream이 정규화 없이 계속 누적되니까 마지막에 LN을 한 번 더 걸어 줘야 한다. 그게 식 (4)의 $\text{LN}(\mathbf{z}_L^0)$이다. 습관적으로 붙인 게 아니라 이유가 있는 것이다.

LayerNorm의 Jacobian을 layer 수만큼 곱하면 6 layer와 32 layer에서 각각 얼마가 되는지, residual connection이 아예 없으면 어떻게 되는지는 [앞 글의 Add & Norm 절](/posts/2026-09-01-attention-is-all-you-need)에서 수치로 확인했다.

## CNN과 ViT의 inductive bias

두 모델의 데이터 규모별 성능 차이를 이해하려면 inductive bias를 살펴볼 필요가 있다.

**inductive bias**는 모델 구조 자체에 미리 반영한 **가정**이다. "세상이 이렇게 생겼을 것이다"라는 사전 지식이다.

**CNN은 세 가지를 모든 layer에 내장하고 있다.**

- **locality**: 가까운 픽셀끼리 먼저 본다. conv 커널이 3×3이면 애초에 멀리는 못 본다
- **2D neighborhood structure**: 무엇이 무엇의 이웃인지 안다
- **translation equivariance**: 고양이가 왼쪽에 있든 오른쪽에 있든 같은 특징이 잡힌다

**ViT는 이걸 딱 두 군데에서만 넣어 준다.**

1. 이미지를 패치로 자를 때
2. fine-tuning에서 해상도를 바꿀 때 (마지막 절에서 다룬다)

그 외에는 self-attention이 **첫 layer부터 전역**(global)이다. 196개 패치가 서로를 다 볼 수 있다. 그리고 $E_{\text{pos}}$는 초기화 시점에 2D 위치를 전혀 모른다. 그냥 197개의 랜덤 벡터다. 인접한 패치가 무엇인지 임베딩 구조로 지정하지 않는다.

**여기서 예측이 하나 나온다.**

> CNN이 공짜로 갖고 있던 걸 버렸으니 **ViT는 데이터가 적으면 불리해야 한다.** 그 정보를 어딘가에서 채워야 하는데 구조에서 못 받으면 데이터에서 받아야 하기 때문이다.

이 예측이 맞는지가 실험 파트 전체의 주제다.

## 정확도와 pre-training 비용 비교

Table 2다. 정확도부터 보면 ViT-H/14가 ImageNet 88.55%로 BiT-L의 87.54%를 이긴다. 그런데 Noisy Student가 88.4~88.5니까 **정확도만 보면 사실 큰 차이가 아니다.**

| | ViT-H/14 (JFT) | ViT-L/16 (JFT) | ViT-L/16 (I21k) | BiT-L (R152x4) | Noisy Student |
|---|---|---|---|---|---|
| ImageNet | **88.55** | 87.76 | 85.30 | 87.54 | 88.4 / 88.5 |
| ImageNet ReaL | 90.72 | 90.54 | 88.62 | 90.54 | 90.55 |
| CIFAR-100 | 94.55 | 93.90 | 93.25 | 93.51 | - |
| VTAB (19 tasks) | 77.63 | 76.28 | 72.72 | 76.29 | - |
| **TPUv3-core-days** | **2.5k** | **0.68k** | **0.23k** | **9.9k** | **12.3k** |

pre-training 비용은 맨 아랫줄에 있다. ViT-L/16은 BiT-L을 **모든 task에서 이기면서 pre-training 비용은 약 1/15**이다. Noisy Student 대비로는 거의 1/18이다.

정확도 차이와 함께 그 성능에 도달하는 데 드는 연산량도 살펴봐야 한다.

**이 비교는 통제된 비교가 아니다.** optimizer도 다르고 학습 스케줄도 다르고 weight decay도 다르다. **에폭 수부터 다르다.** ViT는 7/14 에폭인데 BiT-L은 원 논문 기준 30 에폭이고 BiT-L은 ResNet152x4라 모델 크기도 다르다. 논문도 이걸 인정하고 4.4절에서 통제된 비교를 따로 한다.

**통제된 숫자는 Appendix의 Table 6에 있다.** 같은 14 에폭끼리 비교하면 다음과 같다.

| 모델 (JFT, 14 epochs) | ImageNet | exaFLOPs |
|---|---|---|
| ViT-L/16 | 87.12 | **1567** |
| ResNet200x3 | 87.22 | **3306** |

거의 같은 정확도에 연산은 **약 2.1배** 차이다. 15배가 아니라 2배가 이 논문이 실제로 보여 준 숫자다. 인용한다면 이쪽을 인용하는 게 맞다.

**"ViT가 CNN보다 가볍다"는 명제 자체는 틀렸다.** 이미지 한 장을 처리하는 비용만 보면 ViT-B/16이 약 17.4 GMACs, ResNet-50이 약 4.1 GMACs로 **ViT가 4배 넘게 비싸다.** 논문의 주장은 "같은 성능에 도달하는 데 드는 총 pre-training 연산이 더 적다"이지 "장당 연산이 적다"가 아니다. 이 둘은 전혀 다른 얘기다.

## 데이터 규모가 결론을 뒤집는다

데이터 규모별 성능을 비교하면 앞의 예측을 확인할 수 있다.

<div className="flex flex-col md:flex-row gap-4 justify-center my-6">
  <figure className="text-center m-0 flex-1">
    <img src="/images/posts/vit-16x16-words/fig_dataset.png" alt="pre-training dataset 크기별 ImageNet 정확도" className="rounded-lg mx-auto" />
    <figcaption className="mt-2 text-sm text-gray-500">Figure 3. 회색 띠가 BiT ResNet들의 범위</figcaption>
  </figure>
  <figure className="text-center m-0 flex-1">
    <img src="/images/posts/vit-16x16-words/fig_fewshot.png" alt="JFT 부분집합 크기별 linear 5-shot 정확도" className="rounded-lg mx-auto" />
    <figcaption className="mt-2 text-sm text-gray-500">Figure 4. JFT를 900만~3억으로 잘라 가며 측정</figcaption>
  </figure>
</div>

Figure 3의 x축이 pre-training dataset 크기다.

- **ImageNet(130만 장)만 쓰면**: ViT가 CNN보다 **못하다.** 심지어 **ViT-Large가 ViT-Base보다 나쁘다.** 모델을 키웠는데 성능이 떨어진 것이다
- **ImageNet-21k(1400만 장)**: 비슷해진다
- **JFT-300M(3억 장)**: **역전된다.** 그리고 **큰 모델일수록 더 이득**을 본다

Figure 4도 같은 얘기다. **ResNet은 작은 데이터에서 더 좋지만 일찍 포화되고 ViT는 늦게 출발해서 계속 올라간다.**

실험 결과는 앞의 예측과 일치한다. inductive bias를 버린 대가를 데이터가 적을 때 치르고 데이터가 많으면 그 대가를 회수하는 것이다. 논문은 이를 "**large scale training trumps inductive bias**"라고 표현한다. 대규모 학습이 inductive bias를 압도한다는 뜻이다.

참고로 작은 데이터에서 큰 모델이 더 나쁜 이유는 표현력이 큰 모델이 적은 데이터에 overfit하기 쉽기 때문이다. 논문도 regularization을 조절해 봤지만 한계가 있었다고 적고 있다.

## CNN을 결합한 hybrid 모델

자연스럽게 나오는 질문이 있다. "그럼 CNN의 inductive bias를 다시 넣어 주면 더 좋아지지 않을까?" 논문도 이걸 실험했다.

**hybrid**는 raw 패치 대신 **ResNet이 뽑은 feature map**을 token으로 쓴다. 이때 패치 크기가 $P=1$이다. feature map의 공간 위치 하나하나가 token 하나가 되는 것이다. 그래서 flatten이랄 것도 없이 채널 수가 그대로 $E$의 입력 차원이 된다. **CNN의 locality를 명시적으로 주입해 준 셈이다.**

| 모델 | ImageNet | exaFLOPs |
|---|---|---|
| ViT-B/32 | 80.73 | 55 |
| R50x1 + ViT-B/32 | **84.90** | 106 |
| ViT-L/16 | 87.12 | 1567 |
| R50x1 + ViT-L/16 | **87.12** | 1668 |

- **작은 모델**(ViT-B/32): 80.73 → 84.90. **4.2%p나 오른다**
- **가장 큰 모델**(ViT-L/16): 87.12 → 87.12. **소수점까지 똑같다.** 그런데 연산은 1567에서 1668로 **더 쓴다**

<Fig src="/images/posts/vit-16x16-words/fig_compute.png" alt="pre-training 연산량 대비 transfer 정확도">Figure 5. 연산량이 커질수록 hybrid(주황)와 ViT(파랑)가 겹쳐진다</Fig>

**충분한 스케일에서는 inductive bias를 넣어 줘도 아무 이득이 없다.** 학습이 이미 그걸 대체하고 있는 것이다. 논문도 이 결과를 "somewhat surprising"이라고 표현한다. conv의 지역적 특징 추출이 어떤 크기에서든 도움이 될 거라고 예상했을 텐데 그렇지 않았다는 것이다.

## ViT가 학습한 이미지의 공간 구조

앞 절에서 "학습이 대체하고 있다"고 했는데 정말 그런지 논문이 직접 들여다본다.

### 1. Attention이 보는 범위

**attention distance**는 CNN의 receptive field에 대응하는 지표다. query 패치와 key 패치 사이의 픽셀 거리를 attention weight로 가중평균한 값이다. 작으면 "가까운 것만 본다", 크면 "멀리까지 본다"는 뜻이다.

<Fig src="/images/posts/vit-16x16-words/fig_attndist.png" alt="layer 깊이별 mean attention distance">Figure 7 (right). x축이 layer 깊이, y축이 mean attention distance(픽셀), 점 하나가 head 하나</Fig>

- **낮은 layer에서는 head들이 크게 갈린다.** 어떤 head는 이미 이미지 전체를 보고 어떤 head는 **아주 좁은 영역만** 본다
- **깊어질수록** 모든 head가 넓게 본다

여기서 좁게 보는 low-layer head들이 흥미롭다. 이건 사실상 **CNN의 초기 conv layer가 하는 일을 ViT가 스스로 학습한 것**이기 때문이다. 아무도 locality를 알려 주지 않았는데 일부 head가 알아서 국소적으로 보기로 한 것이다.

**그리고 이걸 뒷받침하는 증거가 hybrid에 있다.** hybrid에서는 이 국소적 head가 훨씬 덜 나타난다. 왜일까? **ResNet이 이미 그 일을 해 줬기 때문이다.** 이 관찰도 CNN이 초기의 국소 특징 추출을 맡는다는 해석을 뒷받침한다.

### 2. 학습된 패치 임베딩 필터

<Fig src="/images/posts/vit-16x16-words/fig_pca.png" alt="학습된 패치 임베딩 필터의 주성분 28개">Figure 7 (left). ViT-L/32의 RGB embedding filters, 상위 28개 주성분</Fig>

이 그림은 패치 임베딩 행렬 $E$의 필터를 분석한 결과다. $E$의 각 열이 conv filter 하나에 대응한다. ViT-L/32면 $32 \times 32 \times 3 = 3072$차원짜리 필터가 $D$개 있는 셈이다. 이걸 다 보여 줄 순 없으니 **$D$개 필터를 데이터 포인트로 삼아 PCA를 돌리고** 나온 주성분을 다시 패치 모양으로 되돌려 그린 것이다.

여기서 표현 하나를 정정하고 싶다. 흔히 "각 필터에 대해 PCA를 했다"고 설명하는데 **PCA는 샘플이 여러 개 있어야 계산된다.** 필터 하나로는 PCA를 할 수 없다. 정확히는 **$D$개 필터를 모아서 한 PCA**다.

convolution의 locality를 전혀 주지 않았는데 **CNN 초기 layer가 수렴하는 것과 똑같은 패턴**이 나타난다. 선, 격자, 색 대비 blob 같은 것들이다.

### 3. 위치 임베딩이 2D를 재발견한다

개인적으로 이 논문에서 제일 인상적이었던 부분이다.

<Fig src="/images/posts/vit-16x16-words/fig_posemb.png" alt="학습된 위치 임베딩의 코사인 유사도">Figure 7 (center). ViT-L/32의 position embedding similarity</Fig>

일단 이 그림은 패치가 **49개**다. 앞에서 예시로 쓴 것과 모델이 다르기 때문이다.

- 예시로 쓴 건 **ViT-B/16**: 패치 크기 16이라 $224 \div 16 = 14$, 그래서 $14 \times 14 = 196$개
- Figure 7은 **ViT-L/32**: 패치 크기 32라 $224 \div 32 = 7$, 그래서 $7 \times 7 = 49$개

**그림 읽는 법이 좀 까다롭다.** 학습이 끝난 $E_{\text{pos}}$의 행들끼리 코사인 유사도를 계산한 건데 구조가 격자 안에 격자라 처음 보면 헷갈린다.

- 전체가 **49개의 타일**로 되어 있고 $7 \times 7$로 배열돼 있다. **타일 하나가 "기준 패치 하나"다**
- 각 타일 안이 또 $7 \times 7$인데 이건 **그 기준 패치가 49개 패치 전부(자기 자신 포함)와 얼마나 비슷한지**를 그린 것이다

그러니까 맨 왼쪽 위 타일은 "좌상단 패치 기준으로 본 49개와의 유사도"고 3행 5열 타일은 "3행 5열 패치 기준으로 본 49개와의 유사도"다. 타일 49개 × 각 타일 안 49칸 = 총 $49 \times 49$ 유사도 행렬을 2D 배치로 펼쳐 놓은 것이다.

그렇게 보면 두 가지가 눈에 들어온다.

- **각 타일에서 제일 밝은 점이 그 타일 자기 좌표에 있다.** 자기 자신과의 유사도가 1이니 당연하고 그 주변이 함께 밝다. **가까운 패치일수록 유사도가 높다**는 뜻이다
- **같은 행, 같은 열에 있는 패치들이 함께 밝아진다.** 십자 패턴이다

**중요한 건 ViT가 이 패치들이 7×7 격자라는 걸 모른다는 점이다.** 그냥 1번, 2번, … 49번으로 1차원 번호만 매겨 각각 독립적인 벡터를 학습시켰다. 그런데 **2D 구조를 스스로 복원해 낸 것이다.**

비교를 위해 sinusoidal PE의 코사인 유사도를 직접 계산해 봤다.

<Fig src="/images/posts/vit-16x16-words/fig_sinusoidal.png" alt="sinusoidal positional encoding의 코사인 유사도">직접 계산한 sinusoidal PE의 유사도: 왼쪽은 상대 거리별 곡선, 오른쪽은 유사도 행렬 (d=512)</Fig>

식을 정리하면 상대 거리 $k$에 대해 $\frac{2}{d}\sum_i \cos(k \cdot w_i)$가 되는데 이건 **거리가 멀어지면 감소하도록 설계 단계에서 강제된** 성질이다. $d=512$에서 계산해 보면 $k=1$일 때 0.97, $k=200$일 때 0.35로 떨어진다.

위치 간 유사도가 나타나는 과정은 두 방식에서 다르다.

- vanilla는 **sin/cos 공식으로 넣어준 것**
- ViT는 1차원 인덱스만 줬는데 **분류 loss만으로 스스로 알아낸 것**

왼쪽 곡선을 자세히 보면 중간중간 살짝 출렁이는 구간이 있는데 가장 빠른 주파수 성분이 주기를 한 바퀴 돌면서 생기는 aliasing이다. 256개 주파수가 겹쳐서 대부분 상쇄된다.

## 2D 위치 정보를 명시한 실험

바로 나오는 질문이다. "어차피 배울 거면 아예 2D 구조를 알려 주면 더 빨리 배우지 않을까?" 논문이 Appendix D.4에서 실험했다.

세 가지를 비교하는데 각각이 뭔지부터 짚어 보자.

- **1-D** (기본값): 패치를 좌→우, 위→아래 순서로 일렬로 세우고 **순번마다 독립적인 벡터**를 학습. $197 \times D$개 파라미터. 구조적 힌트가 전혀 없다
- **2-D**: X축용, Y축용 embedding 테이블을 **따로** 학습하고 **concat**한다. 각각 $D/2$ 크기다. 그러면 **같은 행의 패치들은 Y 부분을 공유**하게 된다. 앞에서 "학습으로 발견됐다"고 감탄한 십자 패턴이 여기서는 설계로 못 박혀 있는 셈이다
- **relative**: 이건 층위가 다르다. **입력에 더하지 않는다.** 대신 두 패치의 offset마다 embedding을 하나씩 두고 그걸로 계산한 값을 softmax 직전에 **attention logit에 bias로 더한다.** 절대 위치라는 개념 자체가 없다

| Pos. Emb. | Default/Stem | Every Layer | Every Layer-Shared |
|---|---|---|---|
| No Pos. Emb. | 0.61382 | N/A | N/A |
| 1-D Pos. Emb. | **0.64206** | 0.63964 | 0.64292 |
| 2-D Pos. Emb. | 0.64001 | 0.64046 | 0.64022 |
| Rel. Pos. Emb. | 0.64032 | - | - |

- **없음 → 있음**은 0.614 → 0.642, 약 **2.8%p** 차이다. 위치 정보 자체는 확실히 필요하다
- **그런데 1D / 2D / relative 사이는 소수점 셋째 자리다.** 사실상 차이가 없다

논문의 설명은 이렇다. encoder가 **pixel level(224×224)이 아니라 patch level(14×14)에서** 돌기 때문에 공간 차원이 훨씬 작다. 이 정도 해상도의 공간 관계는 어떤 방식으로 알려 주든 똑같이 쉽게 학습된다는 것이다.

앞에서 학습된 위치 관계를 관찰했고 이 실험에서는 2D 구조를 명시한 임베딩으로 바꾸어도 성능이 거의 달라지지 않았다.

## 해상도를 바꿀 때 필요한 위치 임베딩 보간

fine-tuning에서 입력 해상도를 바꾸면 위치 임베딩도 조정해야 한다.

fine-tuning은 pre-training(224)보다 **높은 해상도(384)에서** 하면 성능이 좋아진다. 그런데 패치 크기는 16으로 고정이다. 그러면 $384 / 16 = 24$, 즉 패치가 **576개**가 된다.

문제가 생긴다. $E_{\text{pos}}$는 197행만 학습됐는데 **577행이 필요하다.** 학습된 적 없는 파라미터를 그대로 사용할 수는 없다.

앞에서 $E$는 재사용된다고 했던 걸 떠올려 보자. $E$는 패치 하나를 받아 $D$차원으로 바꾸는 거라 패치 개수와 무관하다. **그런데 $E_{\text{pos}}$는 패치 개수만큼 행이 있어야 하니 재사용이 안 된다.** 이게 learned PE의 유일한 실질적 약점이다.

논문의 해법은 **2D interpolation**이다.

1. **CLS 행을 떼어 낸다**: 공간 좌표가 없으니 보간 대상이 아니다
2. 남은 `(196, D)`를 **`(14, 14, D)`로 reshape**: 여기서 "이게 사실 14×14 격자였다"고 사람이 알려 주는 것이다
3. **bilinear로 `(24, 24, D)`로 확대**
4. 다시 flatten하고 CLS를 앞에 붙인다

**이게 정당한 이유가 앞 절에 있다.** 가까운 위치의 임베딩이 실제로 비슷하다는 게 확인됐기 때문에 공간적으로 보간하는 게 의미가 있는 것이다. 만약 이웃한 행들이 아무 관계 없는 랜덤 벡터였다면 그 둘을 반반 섞어 봐야 아무 의미 없는 값이 나온다.

그리고 이 지점이 중요하다. ViT에 2D inductive bias를 손으로 넣어 주는 곳은 패치 자르기와 **바로 여기, 딱 두 군데**다. 논문이 직접 그렇게 밝히고 있다.

## 정리

처음 질문으로 돌아가 보자. **"ViT에는 2D inductive bias가 얼마나 필요한가?"** 지금까지 본 근거들이 각각 답을 하나씩 준다.

| 근거 | 답 |
|---|---|
| Figure 3, 4 | 데이터가 적으면 **많이 필요하다**: CNN에게 진다 |
| Table 5 (hybrid) | 데이터가 많으면 **넣어 줘도 소용없다** |
| Figure 7 center | 안 알려 줘도 **스스로 배운다** |
| Table 8 | 그래서 **어떻게 알려 주든 차이가 없다** |
| §3.2 interpolation | 단, 해상도가 바뀌면 **손으로 넣어 줘야 한다** |

이 결과에서는 **데이터가 충분하면 이미지에 관한 가정을 구조에 넣지 않아도 된다.** 둘 중 하나는 있어야 하는데 3억 장이 있으면 구조에 박아 둘 필요가 없다는 것이다.

### 실험의 한계

- **JFT-300M은 구글 내부 비공개 dataset이다.** 이 논문의 핵심 결과를 **외부에서 재현할 수 없다.** 논문 리뷰 관점에서 이건 꽤 큰 문제다
- **Table 2의 비교는 통제되지 않았다.** optimizer, 스케줄, weight decay가 전부 다르다
- **self-supervised pre-training은 supervised보다 여전히 4%p 뒤처진다.** NLP에서 BERT가 self-supervised로 성공한 것과 대비되는 지점이고 논문도 future work로 남겼다

### 학습된 위치 패턴은 설정에 따라 달라진다

<Fig src="/images/posts/vit-16x16-words/fig_hparam.png" alt="hyperparameter에 따라 달라지는 위치 임베딩 패턴">Figure 10. 같은 ViT-L/16인데 learning rate와 weight decay에 따라 패턴이 달라진다</Fig>

앞에서 감탄했던 "위치 임베딩이 2D를 재발견한다"를 반박하는 쪽에 가까운 자료다. 같은 ViT-L/16인데 learning rate와 weight decay를 바꾸면 패턴이 눈에 띄게 달라진다. **"2D를 스스로 배운다"는 건 경험적 관찰이지 보장된 성질이 아니다.**

보간과 관련해서도 살펴볼 점이 있다. **이 격자 구조가 흐릿한 모델일수록 앞 절의 interpolation이 위험하다.** 보간은 "가까운 위치끼리 비슷하다"는 가정이 있어야만 말이 되기 때문이다.

## 세미나에서 나온 질문들

발표 후 Q&A에서 나온 것과 대비해 갔던 질문들 중 몇 개를 남겨 둔다.

**Q. 패치 임베딩과 위치 임베딩을 더하면 정보가 섞여서 구분이 안 되지 않을까?**

스칼라라면 맞다. 100이 99+1인지 98+2인지 모르니까. 그런데 고차원에서는 다르다. 첫째, 선형연산에는 분배법칙이 성립하니까 $(a+b)W = aW + bW$이고 attention score가 자동으로 네 항으로 쪼개진다. 둘째, $D=768$에서 무작위 두 벡터의 코사인 유사도는 대략 $1/\sqrt{768} \approx 0.036$으로 거의 직교한다. "복원한다"기보다 **"거의 직교한 방향에 겹쳐 저장해서 간섭이 거의 없다"가** 정확한 표현이다.

**Q. CLS 대신 그냥 평균 내면 안 될까?**

논문이 실험했는데 여기서 재미있는 일이 있었다. 처음엔 GAP이 훨씬 나빴는데 조사해 보니 extra token 때문도 GAP 연산 때문도 아니고 **learning rate 때문**이었다. lr만 낮추니 CLS와 비슷하거나 더 나아졌다.

즉 "구조 A가 B보다 나쁘다"는 결론이 사실은 **"A에 B의 하이퍼파라미터를 그대로 썼다"는** 실험 설계 문제였던 것이다. ViT 이해에 꼭 필요한 얘기는 아니지만 **아키텍처 비교 논문을 읽을 때 항상 의심해야 할 지점**이라 적어 둔다. 그리고 저자들이 이걸 스스로 찾아 부록에 적어 뒀다는 게 오히려 논문의 신뢰도를 높인다고 생각한다.

**Q. sinusoidal PE를 ViT에 쓰면 어떨까?**

Table 8에 직접 비교는 없지만 반대 방향 실험은 원래 Transformer 논문에 있다. sinusoidal 대신 learned를 썼는데 BLEU 차이가 거의 없었다. 그리고 sinusoidal을 쓰면 interpolation 문제가 애초에 생기지 않는다. 공식이니까 몇 번째 위치든 그냥 계산하면 되기 때문이다.

앞 글 [Attention Is All You Need: 설명이 항상 멈추는 자리들](/posts/2026-09-01-attention-is-all-you-need)에서 본 sinusoidal positional encoding이 "수식으로 강제한" 성질을 ViT의 learned embedding이 "분류 loss만으로 스스로 배운" 과정과 나란히 놓고 보면 이 논문의 Figure 7이 훨씬 잘 읽힌다.

---

**참고문헌**

- Dosovitskiy et al., *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*, ICLR 2021
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
- Cordonnier et al., *On the Relationship between Self-Attention and Convolutional Layers*, ICLR 2020
- Kolesnikov et al., *Big Transfer (BiT)*, ECCV 2020

*(그림은 모두 원논문 Figure 1, 3, 4, 5, 7, 10에서 가져왔고 sinusoidal PE 유사도 그림만 직접 계산해 그렸다.)*
