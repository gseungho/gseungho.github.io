---
title: "시계열분석 (2): 추세와 계절성을 구하는 두 방법, 이동평균과 회귀"
date: "2026-10-15"
category: "공부"
subcategory: "Statistics/Time Series Analysis"
tags: ["시계열분석", "시계열 분해", "이동평균", "추세 회귀"]
description: "이동평균으로 성분을 나누는 classical decomposition과 STL, 시간과 dummy 변수에 회귀하는 추세와 계절 모형, AIC와 SIC로 모형을 고르는 과정을 정리했다."
---

시계열분석 시리즈의 두 번째 글이다. 세종대학교 인공지능데이터사이언스학과 2026년 2학기 시계열분석 수업을 따라가며 공부한 내용이고 주 교재 [FPP3](https://otexts.com/fpp3/)(Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed.)를 함께 참고했다.

> **이번 글의 질문: 관측값에 뒤섞인 추세와 계절성을 어떻게 떼어 낼까?**

[1편](/posts/2026-10-15-time-series-1-forecasting-basics)에서 시계열을 추세, 계절성, 나머지로 나누어 본다는 이야기를 했다. 이번에는 그 성분을 실제로 구해 보자. 길은 크게 두 갈래다.

- **평활**: 이동평균으로 자료를 매끄럽게 만들어 추세를 뽑고 남은 것에서 계절 성분을 평균으로 구한다.
- **회귀**: 추세를 시간의 함수로, 계절성을 dummy 변수로 두고 최소제곱으로 계수를 추정한다.

처음 공부할 때 나는 이 둘을 차례로 거치는 단계로 착각했다. 평활로 추세를 구한 뒤 그 위에 회귀를 하는 줄 알았다. **실제로는 같은 일을 하는 서로 다른 도구다.** 왜 그런지는 회귀 부분에서 다시 짚어 보자.

## 이동평균으로 추세 뽑기

추세(정확히는 trend-cycle)를 구하는 가장 단순한 방법이 이동평균(moving average)이다. Order가 $m$인 이동평균 $m$-MA는 시점 $t$를 가운데 두고 앞뒤로 $k$개씩, 모두 $m$개를 평균한다.

$$
\hat{T}_t = \frac{1}{m}\sum_{j=-k}^{k} y_{t+j}, \qquad k = \frac{m-1}{2}
$$

5-MA와 3-MA를 풀어 쓰면 이렇다.

$$
\text{5-MA}:\ \tfrac{1}{5}(y_{t-2} + y_{t-1} + y_t + y_{t+1} + y_{t+2}), \qquad \text{3-MA}:\ \tfrac{1}{3}(y_{t-1} + y_t + y_{t+1})
$$

평균을 내면 무작위 변동이 서로 상쇄되어 매끄러운 곡선이 남는다. window를 한 칸 옮길 때마다 가장 오래된 값 하나가 빠지고 새 값 하나가 들어온다. 3-MA라면 $\hat T_2$는 1번째부터 3번째 값까지, $\hat T_3$은 2번째부터 4번째 값까지의 평균이다. $\hat T_1$은 앞쪽 값이 없어서 구할 수 없다.

FPP3의 호주 수출(GDP 대비 %) 자료에 5-MA를 적용한 앞부분을 보자.

| 연도 | 수출 | 5-MA |
| --- | --- | --- |
| 1960 | 12.99 | |
| 1961 | 12.40 | |
| 1962 | 13.94 | 13.46 |
| 1963 | 13.01 | 13.50 |
| 1964 | 14.94 | 13.61 |
| 1965 | 13.22 | 13.40 |

1962년의 13.46은 1960년부터 1964년까지 다섯 값의 평균이다. 맨 앞 두 해는 비어 있고 맨 뒤 두 해도 마찬가지로 빈다.

Order를 바꾸면 무엇이 달라질까? 세 가지다.

- Order가 클수록 곡선이 더 매끄럽고 평평해진다.
- Order가 클수록 양 끝에서 값을 구할 수 없는 시점이 늘어난다. 11-MA면 앞뒤로 다섯 개씩 잃는다.
- **Order를 계절이나 주기의 길이와 같게 잡으면 평균을 낼 때 그 패턴이 상쇄되어 사라진다.**

마지막 성질이 뒤에서 계절성을 떼어 내는 열쇠다.

**주의:** 여기서 말하는 이동평균은 추세를 구하는 평활법이다. 뒤에 ARIMA에서 나올 MA(moving average) 모형과는 이름만 같고 다른 것이다.

### 짝수 order는 왜 두 번 평균하는가

4-MA는 $t$를 정확히 가운데 둘 수 없다. 앞으로 치우친 평균과 뒤로 치우친 평균, 두 가지가 나온다.

$$
\tfrac{1}{4}(y_{t-2} + y_{t-1} + y_t + y_{t+1}) \qquad \text{또는} \qquad \tfrac{1}{4}(y_{t-1} + y_t + y_{t+1} + y_{t+2})
$$

그래서 이 둘을 한 번 더 평균한다. 4-MA 뒤에 2-MA를 거는 셈이라 $2 \times 4$-MA라고 부른다.

$$
\hat{T}_t = \frac{1}{2}\left[\tfrac{1}{4}(y_{t-2} + y_{t-1} + y_t + y_{t+1}) + \tfrac{1}{4}(y_{t-1} + y_t + y_{t+1} + y_{t+2})\right]
= \tfrac{1}{8}y_{t-2} + \tfrac{1}{4}y_{t-1} + \tfrac{1}{4}y_t + \tfrac{1}{4}y_{t+1} + \tfrac{1}{8}y_{t+2}
$$

가중치가 $t$를 중심으로 대칭이 됐다. 호주 분기별 맥주 생산량으로 확인해 보자.

| 분기 | 생산량 | 4-MA | 2×4-MA |
| --- | --- | --- | --- |
| 1992 Q1 | 443.00 | | |
| 1992 Q2 | 410.00 | 451.25 | |
| 1992 Q3 | 420.00 | 448.75 | 450.00 |
| 1992 Q4 | 532.00 | 451.50 | 450.12 |
| 1993 Q1 | 433.00 | 449.00 | 450.25 |

451.25는 1992년 네 분기의 평균이고 448.75는 한 칸 민 네 값의 평균이다. 그 둘을 다시 평균한 것이 450.00이다.

다른 조합도 가능하다. $3 \times 3$-MA를 풀어 쓰면 가운데로 갈수록 무거운 5항 가중평균이 나온다.

$$
\frac{1}{3}\left[\tfrac{1}{3}(y_{t-2} + y_{t-1} + y_t) + \tfrac{1}{3}(y_{t-1} + y_t + y_{t+1}) + \tfrac{1}{3}(y_t + y_{t+1} + y_{t+2})\right]
= \tfrac{1}{9}y_{t-2} + \tfrac{2}{9}y_{t-1} + \tfrac{1}{3}y_t + \tfrac{2}{9}y_{t+1} + \tfrac{1}{9}y_{t+2}
$$

세 가지 다섯 항 이동평균의 가중치를 나란히 그리면 그림 1과 같다.

<figure className="my-6 text-center">
  <img src="/images/posts/2026-10-15-time-series-2-decomposition-trend-regression/ma_weights.png" alt="5-MA, 3×3-MA, 2×4-MA의 가중치 막대그래프" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">그림 1. 모두 다섯 항을 쓰지만 무게를 나누는 방식이 다르다. 셋 다 좌우 대칭이다.</figcaption>
</figure>

5-MA는 다섯 점에 똑같이 1/5을 주고 $3 \times 3$-MA는 가운데에 무게를 몰아준다. $2 \times 4$-MA는 양 끝에만 절반을 준다. 대칭을 지키려면 짝수 order 뒤에는 짝수 order를, 홀수 order 뒤에는 홀수 order를 붙여야 한다.

### 계절성을 지우는 2×m-MA

분기 자료에 $2 \times 4$-MA를 쓰면 다섯 항이 차례로 작년 4분기, 1분기, 2분기, 3분기, 올해 4분기에 걸린다. 처음과 끝이 같은 분기라서 가중치 $\tfrac18$ 두 개를 합치면 $\tfrac14$이 된다. **네 분기가 모두 $\tfrac14$씩 같은 무게를 받으니 계절 변동은 평균에서 상쇄되어 사라지고 추세만 남는다.**

일반화하면 이렇다.

- $2 \times m$-MA는 order가 $m+1$인 가중 이동평균이다. 양 끝 두 항은 $\tfrac{1}{2m}$, 나머지는 $\tfrac{1}{m}$씩 받는다.
- 계절 주기 $m$이 짝수면 $2 \times m$-MA, 홀수면 $m$-MA를 쓴다. 월별 자료의 연간 계절성에는 $2 \times 12$-MA, 일별 자료의 주간 계절성에는 7-MA다.

그림 2는 계절성이 있는 월별 시뮬레이션 계열 위로 $2 \times 12$-MA의 window(13개월)가 한 칸씩 옮겨 가며 추세를 그려 가는 모습이다.

<figure className="my-6 text-center">
  <img src="/images/posts/2026-10-15-time-series-2-decomposition-trend-regression/ma_window.gif" alt="2×12-MA window가 월별 계열 위를 움직이며 추세선을 그리는 애니메이션" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">그림 2. 주황 띠가 window다. window 하나가 1년치 계절을 통째로 덮어서 오르내림이 상쇄되고 파란 추세선만 남는다. 처음과 마지막 6개월은 window가 자료 밖으로 나가 추세를 구할 수 없다.</figcaption>
</figure>

시뮬레이션 계열은 선형 추세 $100 + 0.35t$에 진폭이 다른 두 사인파와 잡음을 더해 만들었다. 관측값은 1년 주기로 크게 출렁이지만 추세선은 거의 직선으로 올라간다.

지금까지 본 조합을 모두 묶으면 가중 이동평균이라는 일반형이 된다.

$$
\hat{T}_t = \sum_{j=-k}^{k} a_j\, y_{t+j}, \qquad \sum_{j=-k}^{k} a_j = 1, \qquad a_j = a_{-j}
$$

항은 $2k+1$개이고 가중치는 합이 1이며 좌우 대칭이다.

## Classical decomposition의 네 단계

이동평균으로 성분을 나누는 가장 오래된 방법이 1920년대의 classical decomposition이다.[^history]

모형은 성분을 더하거나 곱하는 두 가지다.

$$
y_t = T_t + S_t + R_t \quad (\text{additive}), \qquad y_t = T_t \times S_t \times R_t \quad (\text{multiplicative})
$$

곱하는 모형은 log를 씌우면 $\ln y_t = \ln T_t + \ln S_t + \ln R_t$로 더하는 모형이 된다.

Additive 분해의 절차를 따라가 보자. $m$은 계절 주기다.

1. $\hat T_t$를 구한다. $m$이 짝수면 $2 \times m$-MA, 홀수면 $m$-MA다.
2. 추세를 뺀 계열 $d_t = y_t - \hat T_t$를 만든다.
3. 계절 성분을 구한다. (i) 계절마다 $d_t$를 평균해 $\tilde s_1, \dots, \tilde s_m$을 얻고 (ii) 그 합이 0이 되도록 조정해 $\hat S_t$를 만든다.
4. 나머지 $\hat R_t = y_t - \hat T_t - \hat S_t$를 구한다.

3단계 (i)은 표로 생각하면 쉽다. 행을 연도, 열을 월로 놓고 칸에 $d_t$를 적은 뒤 **열 방향으로 평균한다.** 1월 열이 50, 52, 48, …, 53이면 그 평균 49가 1월의 계절 값 $\tilde s_1$이다. 같은 해의 열두 달을 가로로 평균하는 게 아니다.

네 단계를 분기별 시뮬레이션 계열 40개에 그대로 적용하면 그림 3이 나온다. 계열은 추세 $400 + 2t$에 계절 값 $-30, -10, +10, +30$과 표준편차 6인 잡음을 더해 만들었다.

<figure className="my-6 text-center">
  <img src="/images/posts/2026-10-15-time-series-2-decomposition-trend-regression/classical_decomposition.png" alt="분기별 시뮬레이션 계열의 classical decomposition 결과: 관측값, 추세, 계절, 나머지" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">그림 3. 위에서부터 관측값, 추세, 계절, 나머지. 추세는 처음과 마지막 두 분기가 비어 있고 계절 성분은 해마다 똑같은 모양을 되풀이한다.</figcaption>
</figure>

계절 평균 $\tilde s_j$는 $-30.96, -11.51, 13.51, 29.31$로 참값 $-30, -10, +10, +30$에 가깝게 나왔다. 넷의 평균 $c$는 0.087이라 (ii)에서 각 값에서 0.087을 빼면 $\hat S_j = -31.05, -11.60, 13.42, 29.22$가 된다. 나머지는 0 주위에서 잡음 크기만큼 흔들린다.

### 합을 0으로 맞추는 단계는 왜 있는가

3단계 (ii)는 건너뛰어도 분해식 자체는 성립한다. 그래서 선택 사항처럼 보인다. 나는 처음에 이 단계를 4단계와 같은 일로 오해했다. "나머지의 평균을 0으로 만드는 과정 아닌가" 하고 말이다. 둘은 다른 일이다.

(ii)는 $\tilde s_j$들의 평균 $c$를 각 값에서 빼서 계절 성분을 조정한다.

$$
c = \frac{1}{m}\sum_{j=1}^{m} \tilde{s}_j, \qquad \hat{S}_j = \tilde{s}_j - c, \qquad \sum_{j=1}^{m} \hat S_j = 0
$$

4단계는 그다음에 남은 것을 계산할 뿐이다. 오히려 (ii)를 건너뛰면 나머지의 평균이 정확히 0이다. $\tilde s_j$가 그 계절 $d_t$의 평균이기 때문이다. (ii)를 하면 $\hat S$가 $c$만큼 내려가고 $\hat R$이 그만큼 올라가서 나머지의 평균이 $c$가 된다. $\hat T$는 1단계에서 정해진 뒤 바뀌지 않는다.

열두 달 평균이 다음과 같다고 해 보자(설명을 위한 예시 값이다).

| 월 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 합 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $\tilde{s}_j$ | 49 | 43 | 30 | 12 | -3 | -14 | -19 | -21 | -16 | -9 | -18 | -22 | 12 |
| $\hat{S}_j$ | 48 | 42 | 29 | 11 | -4 | -15 | -20 | -22 | -17 | -10 | -19 | -23 | 0 |

합이 12라서 $c = 1$이고 열두 값에서 모두 1을 뺀다. 표 안의 $d_t$는 그대로 두고 맨 아래 열 평균만 조정하는 것이다. 1961년 1월의 $d_t$가 50이었다면 나머지는 (ii) 전에 $50 - 49 = 1$, 후에 $50 - 48 = 2$다. 뺀 1이 나머지로 옮겨 간 셈이다.

그렇다면 왜 굳이 할까? **수준과 계절을 나누는 방법이 하나로 정해지지 않기 때문이다.** 어떤 상수 $c$를 잡아도 $T_t + S_t = (T_t + c) + (S_t - c)$라서 규칙이 없으면 분해가 무수히 많다. 분기 평균이 100, 120, 140, 160인 자료를 예로 들면 두 가지 쓰는 법이 모두 맞다.

| 규칙 | 수준 | 1분기 | 2분기 | 3분기 | 4분기 |
| --- | --- | --- | --- | --- | --- |
| 1분기 기준 | 100 | 0 | +20 | +40 | +60 |
| 합을 0으로 | 130 | -30 | -10 | +10 | +30 |

Classical decomposition은 아래 줄을 고른다. $\hat T_t$가 $t$를 가운데 둔 1년치 평균이라 연평균 수준을 뜻하기 때문이다. 그러면 $\hat S_j$는 그 달이 연평균보다 얼마나 높거나 낮은지를 뜻하므로 1년 동안 더하면 0이어야 한다. 앞의 표에서 1월 $\hat S_1 = 48$은 1월이 연평균보다 48 높다는 뜻이다. 조정 전처럼 합이 12라면 열두 달이 평균 1씩 연평균보다 높다는 말이 된다. 모든 달이 평균보다 높을 수는 없다.

계절 조정 계열 $y_t - \hat S_t$에서도 차이가 드러난다. 1년치를 더하면 $\sum y_t - \sum_j \hat S_j$이므로 $\hat S$의 합이 0이어야 계절 조정 전후의 연간 합계가 같다. 합이 120($c = 10$)인 채로 두면 계절 조정 계열이 원자료보다 평균 10 낮아진다. 계절 조정은 1년 안의 오르내림만 걷어내야 하는데 수준까지 바꾸는 셈이다.

$c$는 보통 0에 가깝다. $2 \times m$-MA가 1년치를 평균하므로 수준은 이미 추세에 들어가 있다. 선형 추세와 합이 0인 계절 성분에 잡음을 더한 월별 120개 계열로 시뮬레이션해 보니 $c = -0.025$였다. 그림 3의 분기 계열에서도 0.087이었다. 그래도 잡음 때문에 정확히 0인 경우는 거의 없어서 (ii)는 매번 하는 작은 보정에 가깝다.

추세가 곡선이면 $c$가 한쪽으로 치우치기도 한다. $y_t = at^2$에 $2 \times 12$-MA를 쓰면 가중치 $a_j$로 계산한 $\sum_j a_j (t+j)^2 = t^2 + \sum_j a_j j^2$에서 다음 값이 나온다.

$$
\sum_j a_j j^2 = 2 \cdot \tfrac{1}{12}(1 + 4 + 9 + 16 + 25) + 2 \cdot \tfrac{1}{24} \cdot 36 = 12.1\overline{6}
$$

즉 $\hat T_t = a(t^2 + 12.17)$이고 추세를 뺀 값이 모든 시점에서 $-12.17a$다. 이동평균이 볼록한 곡선을 위로 끌어올리기 때문이다. $a = 0.1$이면 $c \approx -1.2$이고 이 치우침은 잡음과 달리 해가 바뀌어도 상쇄되지 않는다.

### 나누기로 바뀌는 multiplicative

곱하는 모형은 빼기를 나누기로 바꾸면 된다.

1. $\hat T_t$는 같은 방법으로 구한다.
2. 추세를 제거한 계열은 $y_t / \hat T_t$다.
3. 계절마다 평균한 뒤 $m$개의 합이 $m$이 되도록 조정한다. 각 값을 평균으로 나누면 된다.
4. $\hat R_t = y_t / (\hat T_t \hat S_t)$다.

조정 기준이 0이 아니라 $m$인 이유는 같다. 곱하는 모형에서는 계절 성분이 1 주위에서 움직여야 수준이 전부 추세에 담기니까 평균이 1, 합이 $m$이어야 한다. 미국 소매업 고용을 분해해 보면 additive의 계절 성분은 대략 -200에서 600 사이, multiplicative의 계절 성분은 0.98에서 1.04 사이에 있다.

### Classical decomposition의 한계

- 처음과 끝의 몇 시점에서는 추세를 구할 수 없다. 그림 2와 그림 3에서 보듯 이동평균의 window가 자료 밖으로 나간다.
- 계절 성분이 해마다 똑같이 반복된다고 가정한다. 그림 3의 계절 패널이 그렇다. 1960년대와 지금의 전기 수요 패턴이 같다고 보기는 어렵다.
- 이상값에 약하다. Covid-19나 노사 분규 같은 사건이 평균을 그대로 흔든다.

## STL은 무엇이 다른가

STL(Seasonal and Trend decomposition using Loess)은 이 세 한계를 겨냥한 분해법이다. 수업 자료에는 이름과 분해 결과 그림만 나와서 [FPP3 3.6절](https://otexts.com/fpp3/stl.html)과 방법을 정리한 논문[^stl]을 보고 정리했다.

식은 classical과 같은 $y_t = T_t + S_t + R_t$다. 다른 것은 성분을 구하는 도구다. Classical은 이동평균과 단순 평균을 쓰고 STL은 loess를 쓴다.

### Loess, 주변 점에 직선 맞추기

Loess는 국소 가중 회귀(locally weighted regression)다. 시점 $t$의 매끄러운 값을 구할 때 $t$에 가까운 관측값 $q$개만 쓴다. 가까운 점일수록 큰 가중치를 주고 직선을 가중최소제곱으로 맞춘 뒤 그 직선이 $t$에서 지나는 값을 취한다. 이 계산을 모든 $t$에서 되풀이한다. 가중치는 tricube 함수로 주고 $d_{\max}$는 고른 $q$개 중 $t$에서 가장 먼 점까지의 거리다.

$$
w_i = \left(1 - \left(\frac{|t_i - t|}{d_{\max}}\right)^3\right)^3
$$

그림 4는 시뮬레이션 계열 40개에서 $q = 11$로 두 시점의 loess 값을 구하는 모습이다.

<figure className="my-6 text-center">
  <img src="/images/posts/2026-10-15-time-series-2-decomposition-trend-regression/loess_local_line.png" alt="가운데 시점과 마지막 시점에서 loess가 가까운 11개 점에 가중 직선을 맞추는 모습" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">그림 4. 주황 점이 쓰인 11개 이웃이고 점이 클수록 가중치가 크다. 파란 직선이 맞춘 직선, 파란 점이 그 시점의 loess 값이다. 마지막 시점에서도 왼쪽 이웃만으로 값이 나온다.</figcaption>
</figure>

이동평균도 같은 틀로 볼 수 있다. window 안의 점에 같은 가중치를 주고 직선 대신 상수를 맞추면 $m$-MA다. 여기서 가중치만 바꾸면 가중 이동평균이다. Loess는 상수 대신 직선을 맞추고 가중치를 거리로 정한다는 점이 다르다. **직선을 맞추니까 끝점에서도 값이 나온다.** 그림 4의 오른쪽처럼 마지막 시점이라면 왼쪽 이웃만으로 직선을 맞추면 된다.

### 반복하는 classical

STL의 inner loop는 classical의 순서를 여러 번 되풀이하는 구조다.

1. 추세를 뺀다: $y_t - T_t$. 처음에는 $T_t = 0$으로 둔다.
2. 추세를 뺀 값을 같은 달끼리 모은다. 1월 계열은 해마다 1월 값 하나씩으로 이루어진다. 열두 계열을 각각 loess로 다듬는다.
3. 2의 결과에서 느린 변동을 빼서 $S_t$를 얻는다. 이 단계가 없으면 추세 일부가 계절 성분에 섞여 든다.
4. 계절 조정 계열 $y_t - S_t$를 loess로 다듬어 새 $T_t$를 얻고 1로 돌아간다.

robust 옵션을 켜면 outer loop가 하나 더 붙는다. inner loop가 끝나면 $R_t = y_t - T_t - S_t$를 구하고 잔차가 큰 시점에 작은 가중치 $\rho_t$를 준다.

$$
\rho_t = B\!\left(\frac{|R_t|}{6\,\operatorname{median}|R_t|}\right), \qquad B(u) = \begin{cases} (1-u^2)^2 & 0 \le u < 1 \\ 0 & u \ge 1 \end{cases}
$$

다음 inner loop에서는 loess 가중치에 $\rho_t$를 곱한다. 잔차가 중앙값의 6배를 넘는 시점은 가중치가 0이 되어 추세와 계절 추정에서 빠진다. 그 값은 $R_t$에 그대로 남는다.

Classical의 한계와 맞대 보자.

| Classical의 한계 | STL의 대응 |
| --- | --- |
| 처음과 끝 몇 시점에 추세가 없다 | loess가 한쪽 이웃만으로 직선을 맞춰 끝까지 구한다 |
| 계절 성분이 해마다 같다 | 같은 달 계열을 loess로 다듬어 계절 성분이 천천히 변한다 |
| 이상값에 약하다 | robust 옵션을 켜면 이상값이 나머지로 빠진다 |

둘째 줄은 앞의 1월 열 예와 맞춰 보면 잘 보인다. Classical은 1월 열을 통째로 평균해 모든 해의 1월에 49를 준다. STL은 이 열을 loess로 다듬으니 1990년 1월과 2010년 1월의 계절 성분이 달라질 수 있다.

조절하는 값은 둘이다. Trend window는 추세 값 하나를 구할 때 쓰는 연속 관측값 수, season window는 계절 값 하나를 구할 때 쓰는 연도 수다. 값이 작을수록 성분이 빨리 변한다. Season window를 무한대로 두면(R fable의 `season(window = "periodic")`) 계절 성분이 해마다 같아져 classical과 같은 가정이 된다.

장점이 하나 더 있다. X-11과 SEATS는 월별과 분기별 자료만 다루지만 STL은 주기가 무엇이든 쓸 수 있어서 일별 자료의 주간 계절성도 나눌 수 있다. 단점은 거래일 수나 명절 같은 달력 효과를 자동으로 처리하지 않는다는 것, 그리고 additive 분해만 한다는 것이다. Multiplicative가 필요하면 log를 씌운 뒤 STL을 하고 성분을 되돌린다.

## 시간에 회귀하는 추세 모형

이제 두 번째 길로 가 보자. 추세를 시간의 함수로 두고 계수를 추정한다. $TIME_t$는 첫 시점에 1, 둘째 시점에 2를 주는 변수다.

| 모형 | 식 | 특징 |
| --- | --- | --- |
| Linear | $T_t = \beta_0 + \beta_1 TIME_t$ | $\beta_0$은 $t = 0$에서의 추세 값 |
| Quadratic | $T_t = \beta_0 + \beta_1 TIME_t + \beta_2 TIME_t^2$ | 휘어진 추세. $\beta_2 = 0$이면 linear |
| Exponential | $T_t = \beta_0 e^{\beta_1 TIME_t}$ | $\ln T_t = \ln \beta_0 + \beta_1 TIME_t$로 log에서 선형 |

Quadratic은 계수의 부호에 따라 모양이 갈린다.

| 부호 | 모양 |
| --- | --- |
| $\beta_1 > 0,\ \beta_2 > 0$ | 단조 증가, 비선형 |
| $\beta_1 < 0,\ \beta_2 < 0$ | 단조 감소, 비선형 |
| $\beta_1 < 0,\ \beta_2 > 0$ | U자 |
| $\beta_1 > 0,\ \beta_2 < 0$ | 뒤집힌 U자 |

고차 다항식을 쓰면 표본 안에서는 잘 맞지만 추세가 매끄럽지 않게 되므로 낮은 차수를 쓴다. Exponential trend는 log-linear trend라고도 하고 경영, 금융, 경제 자료에서 매우 흔하다. 성장률이 일정한 계열이 여기에 해당한다.

추정은 최소제곱이다.

$$
\hat{\boldsymbol\beta} = \arg\min_{\boldsymbol\beta} \sum_{t=1}^{T} \big(y_t - T_t(\boldsymbol\beta)\big)^2
$$

### 회귀식에 넣는 값은 원자료다

이 식의 $y_t$는 무엇일까? 나는 처음에 "이동평균을 거친 값"으로 읽었다. 앞에서 추세를 MA로 구했으니 그 위에 회귀를 하는 줄 알았다. 그렇지 않다.

**주의: 회귀식의 $y_t$는 이동평균을 거치지 않은, 관측한 계열 그대로다.**

뒤에 나올 소매 판매 예제가 근거다. 표본이 1955년 1월부터 1993년 12월까지이고 관측값이 468개다. 39년 × 12개월 = 468이라 빠진 달이 없다. $2 \times 12$-MA를 거쳤다면 앞뒤 6개월씩 값이 없어 456개가 됐을 것이다. 예측식이 맞히려는 것도 실제 관측값이고 구간 폭을 정하는 $\sigma$도 실제 값과 추세선 사이의 오차다.

그러니까 두 방법은 같은 추세 $T_t$를 다른 방식으로 뽑는 것이다. STL의 loess까지 한 줄에 놓으면 차이가 선명해진다.

| | 이동평균 | Loess (STL) | OLS 추세 |
| --- | --- | --- | --- |
| 맞추는 것 | 주변 점에 상수 | 주변 점에 직선 | 전체 점에 직선이나 곡선 하나 |
| $T_t$의 형태 | 정해진 식 없음 | 정해진 식 없음 | 시간의 함수 |
| 끝점 | 값이 없다 | 있다 | 있다 |
| 미래 | 구할 수 없다 | 구할 수 없다 | $TIME_{T+h}$를 넣으면 나온다 |

계절성도 마찬가지다. 이동평균 쪽은 계절마다 평균을 내고 회귀 쪽은 뒤에 나올 seasonal dummy의 계수로 잡는다.

MA한 값에 다시 OLS를 맞추면 안 되는 이유도 있다. OLS는 전체 자료로 직선을 맞추면서 잡음을 이미 평균으로 걸러 내니 MA를 먼저 할 필요가 없다. MA를 먼저 하면 양 끝 관측값을 잃는다. 잔차도 인위적으로 매끈해져서 $R^2$나 회귀 표준오차의 의미가 달라진다. 특히 표준오차가 작게 나와 예측 구간이 실제 흔들림보다 좁아진다. 직선 추세라면 계수는 거의 같게 나온다. 대칭 이동평균은 직선을 바꾸지 않기 때문이다.

### 행렬로 쓰면 공식 하나

Linear, quadratic, seasonal dummy 모형은 모두 다중회귀 $\mathbf{y} = \mathbf{X}\boldsymbol\beta + \boldsymbol\varepsilon$의 특수한 경우다. 관측값이 $n$개, 설명변수가 $k$개면 $\mathbf{X}$는 $n \times (k+1)$ 행렬이고 첫 열은 절편을 위한 $\mathbf{1}$이다. 잔차 제곱합을 벡터로 미분해 0으로 두면 정규방정식이 나온다.

$$
Q(\boldsymbol\beta) = (\mathbf{y} - \mathbf{X}\boldsymbol\beta)^\top(\mathbf{y} - \mathbf{X}\boldsymbol\beta), \qquad
\frac{\partial Q}{\partial \boldsymbol\beta} = -2\mathbf{X}^\top\mathbf{y} + 2\mathbf{X}^\top\mathbf{X}\boldsymbol\beta = \mathbf{0}
$$

$$
\hat{\boldsymbol\beta} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}, \qquad
V(\hat{\boldsymbol\beta}) = \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}, \qquad
\hat\sigma^2 = s_e^2 = \frac{\mathbf{e}^\top\mathbf{e}}{n-k-1} \tag{1}
$$

$(\mathbf{X}^\top\mathbf{X})^{-1}$이 존재하려면 $\mathbf{X}$의 $k+1$개 열이 선형독립이어야 한다. 이 조건은 뒤의 seasonal dummy에서 다시 문제가 된다. $\hat\beta_i$의 표준오차는 $s_e\sqrt{C_{ii}}$이고 $C_{ii}$는 $(\mathbf{X}^\top\mathbf{X})^{-1}$의 $i$번째 대각원소다. 회귀 프로그램 출력의 S.E. of regression이 $s_e$, Sum squared resid가 $\mathbf{e}^\top\mathbf{e}$, Std. Error가 $s_e\sqrt{C_{ii}}$다.

Linear trend는 $\mathbf{X}$의 둘째 열이 $1, 2, \dots, n$이다.

$$
\mathbf{X} = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ \vdots & \vdots \\ 1 & n \end{pmatrix}, \qquad
\mathbf{X}^\top\mathbf{X} = \begin{pmatrix} n & \sum t \\ \sum t & \sum t^2 \end{pmatrix}, \qquad
\mathbf{X}^\top\mathbf{y} = \begin{pmatrix} \sum y_t \\ \sum t\,y_t \end{pmatrix}
$$

2×2 역행렬을 풀면 단순회귀 공식 $\hat\beta_1 = S_{ty}/S_{tt}$, $\hat\beta_0 = \bar y - \hat\beta_1 \bar t$가 된다. 소매 판매 예제의 출력으로 검산해 보자. $n = 468$이면 $S_{tt} = \sum(t - \bar t)^2 = n(n^2 - 1)/12 = 8{,}541{,}897$이고 $s_e = 15866.12$다.

$$
\operatorname{SE}(\hat\beta_1) = \frac{s_e}{\sqrt{S_{tt}}} = 5.4287, \qquad
\operatorname{SE}(\hat\beta_0) = s_e\sqrt{\frac{1}{n} + \frac{\bar t^2}{S_{tt}}} = 1469.18
$$

출력의 5.428670, 1469.177과 같다. $TIME$이 1부터 468까지라는 것도 이 계산으로 확인된다.

Quadratic은 셋째 열에 $t^2$을 넣으면 된다. 2차식을 선형회귀로 풀어도 될까? 된다. **선형회귀의 "선형"은 계수 $\boldsymbol\beta$의 1차식이라는 뜻이다.** $\beta_0 + \beta_1 t + \beta_2 t^2$은 $t$로 보면 2차지만 $\boldsymbol\beta$로 보면 1차라서 $\mathbf{X}\boldsymbol\beta$로 쓸 수 있다. $t^2$은 $a + bt$ 꼴로 만들 수 없으니 세 열은 선형독립이다. 다만 상관은 크다. $n = 468$에서 $t$와 $t^2$의 상관계수가 0.968이라 실무에서는 $t$ 대신 $t - \bar t$를 넣기도 한다. 그러면 $t - \bar t$와 $(t - \bar t)^2$의 상관이 0이 되고 적합값은 같다.

Seasonal dummy만 넣은 모형은 $\mathbf{X}$의 각 행에 1이 하나뿐이다. 그래서 $\mathbf{X}^\top\mathbf{X} = \operatorname{diag}(n_1, \dots, n_{12})$($n_j$는 $j$월 관측값 수)이고 $\mathbf{X}^\top\mathbf{y}$는 달마다 $y_t$의 합이다. 식 (1)에 넣으면 $\hat\beta_j$는 $j$월의 평균이다. 이 사실은 뒤의 주택 착공 예제에서 숫자로 확인할 수 있다.

### 지수 추세를 추정하는 두 방법

Exponential trend는 $\beta_1$이 지수 안에 있어서 그대로는 $\mathbf{X}\boldsymbol\beta$ 꼴이 아니다. 방법은 둘이다.

첫째는 log를 씌우는 방법이다. 오차가 곱으로 붙어 있다고 보면 양변에 log를 씌워 선형회귀로 만들 수 있다.

$$
y_t = \beta_0 e^{\beta_1 TIME_t} e^{\epsilon_t} \;\Rightarrow\; \ln y_t = \underbrace{\ln \beta_0}_{b_0} + \beta_1 TIME_t + \epsilon_t
$$

$\mathbf{X}$는 linear와 같고 $\mathbf{y}$ 자리에 $\mathbf{z} = (\ln y_1, \dots, \ln y_n)^\top$를 넣는다. 추정한 $\hat b_0$에서 $\hat\beta_0 = e^{\hat b_0}$로 되돌리면 된다.

둘째는 nonlinear least squares(NLS)다. 오차가 더해져 있다고 보면($y_t = \beta_0 e^{\beta_1 TIME_t} + \epsilon_t$) log로 깔끔하게 나눌 수 없으니 원래 식의 제곱오차를 그대로 줄인다. 편미분을 0으로 놓아도 $\beta_1$이 지수 안에 있어 닫힌 해가 없다.

$$
\frac{\partial Q}{\partial \beta_0} = -2\sum_t \big(y_t - \beta_0 e^{\beta_1 t}\big)e^{\beta_1 t} = 0, \qquad
\frac{\partial Q}{\partial \beta_1} = -2\sum_t \big(y_t - \beta_0 e^{\beta_1 t}\big)\beta_0 t\, e^{\beta_1 t} = 0
$$

그래서 반복 계산으로 푼다. 대표적인 Gauss-Newton은 현재 추정값 $\boldsymbol\beta^{(i)}$ 근처에서 $f_t(\boldsymbol\beta) = \beta_0 e^{\beta_1 t}$를 1차 테일러 전개해 선형회귀 꼴로 만든다. $\mathbf{J}$는 $t$번째 행이 $(e^{\beta_1 t},\ \beta_0 t\, e^{\beta_1 t})$인 $n \times 2$ 야코비 행렬이고 $\mathbf{r} = \mathbf{y} - \mathbf{f}(\boldsymbol\beta^{(i)})$는 지금의 잔차다.

$$
\boldsymbol\beta^{(i+1)} = \boldsymbol\beta^{(i)} + (\mathbf{J}^\top\mathbf{J})^{-1}\mathbf{J}^\top\mathbf{r}
$$

식 (1)의 OLS 공식에서 $\mathbf{X}$ 자리에 $\mathbf{J}$, $\mathbf{y}$ 자리에 $\mathbf{r}$을 넣은 꼴이다. 값이 거의 바뀌지 않을 때까지 되풀이한다. 468개짜리 가상 지수 추세 자료로 돌려 보니 log-linear 추정값에서 출발해 6번 만에 수렴했다.

소매 판매 예제에서 두 방법의 결과를 비교해 보자.

| | log-linear (OLS) | exponential (NLS) |
| --- | --- | --- |
| 종속변수 | $\ln y_t$ | $y_t$ |
| $\beta_0$ | $e^{9.389975} \approx 11967.8$ | 11967.80 |
| $\beta_1$ | 0.005931 | 0.005944 |

$\beta_1$이 왜 조금 다를까? **두 방법이 줄이는 오차가 다르기 때문이다.** Log OLS는 비율 오차를 줄이니까 1955년(약 15,000)의 1% 오차와 1993년(약 180,000)의 1% 오차를 같게 센다. NLS는 금액 오차를 줄이니까 1993년의 1%(1,800)가 1955년의 1%(150)보다 제곱해서 144배 무겁다. NLS 추정은 값이 큰 끝부분에 끌려간다.

$\beta_1 = 0.005931$은 한 달에 약 0.59%씩 성장한다는 뜻이고 연으로는 $e^{0.005931 \times 12} - 1 \approx 7.4\%$다. 1편에서 본 "log 값의 차이는 퍼센트 변화"가 여기서 쓰인다.[^median]

## 추세로 예측하면 구간 폭이 일정하다

계절 조정된 자료에서 $y_t = T_t + \epsilon_t$이고 추세가 선형이라고 하자. $h$시점 뒤의 값은 다음과 같다.

$$
y_{T+h} = \beta_0 + \beta_1 TIME_{T+h} + \epsilon_{T+h}
$$

$\epsilon_{T+h}$는 시점 $T$의 정보로 알 수 없으니 기댓값 0으로 놓는다.

$$
\hat{y}_{T+h \mid T} = \hat{\beta}_0 + \hat{\beta}_1 TIME_{T+h}
$$

이것이 점예측이고 밀도 예측(density forecast)은 $N(\hat{y}_{T+h \mid T}, \sigma^2)$, 95% 구간 예측은 $\hat{y}_{T+h \mid T} \pm 1.96\hat{\sigma}$다.

1편에서 본 AR 식과 나란히 놓으면 차이가 보인다. AR에서는 2-step 앞에 잡음이 두 개 들어간다.

$$
y_{T+2} = a\,y_{T+1} + b + \epsilon_{T+2} = a(a\,y_T + b + \epsilon_{T+1}) + b + \epsilon_{T+2}
$$

멀리 갈수록 잡음이 쌓여 구간이 넓어진다. 추세 모형에서는 $TIME_{T+h}$가 이미 아는 값이라 $y_{T+h}$에 들어가는 잡음이 $\epsilon_{T+h}$ 하나뿐이다. **그래서 분산이 $h$와 상관없이 $\sigma^2$이고 구간 폭이 일정하다.**[^se]

## 소매 판매 예제: 어떤 추세가 맞는가

미국 경상 달러 소매 판매(계절 조정된 월별 자료)다. Diebold 교재의 예제로 1955년 1월부터 1993년 12월까지의 468개로 추정하고 1994년 12개월로 평가한다. 비선형 추세가 뚜렷한 계열이다. 네 모형의 회귀 결과를 옮기면 다음과 같다.

| 모형 | 계수 | $R^2$ | 회귀 표준오차 | Durbin-Watson | AIC | SIC |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | $\beta_0 = -16391.25$, $\beta_1 = 349.7731$ | 0.899076 | 15866.12 | 0.004682 | 19.34815 | 19.36587 |
| Quadratic | $\beta_0 = 18708.70$, $\beta_1 = -98.31130$, $\beta_2 = 0.955404$ | 0.997022 | 2728.205 | 0.151089 | 15.82919 | 15.85578 |
| Log-linear | $b_0 = 9.389975$, $\beta_1 = 0.005931$ | 0.987076 | 0.091879 | 0.019949 | -4.770302 | -4.752573 |
| Exponential (NLS) | $\beta_0 = 11967.80$, $\beta_1 = 0.005944$ | 0.988796 | 5286.406 | 0.040527 | 17.15005 | 17.16778 |

Log-linear만 종속변수가 log 판매액이고 나머지 셋은 판매액 그대로다.

Linear trend의 잔차는 처음에 전부 추세선 위에 있다가 한동안 아래에 있고 끝에서 다시 위로 올라간다. 직선이 휘어진 추세를 못 따라간다는 뜻이다. 실제 자료 대신 비슷한 모양으로 만든 시뮬레이션 계열로 이 현상을 다시 그려 보자. 한 달에 0.59%씩 성장하는 지수 추세에 3% 정도의 곱셈 잡음(multiplicative noise)을 얹은 468개 계열이다.

<figure className="my-6 text-center">
  <img src="/images/posts/2026-10-15-time-series-2-decomposition-trend-regression/linear_vs_quadratic.png" alt="볼록한 시뮬레이션 계열에 linear와 quadratic 추세를 맞춘 결과와 linear 잔차" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">그림 5. 왼쪽: 볼록한 시뮬레이션 계열에 맞춘 직선(파랑)과 2차 곡선(주황). 오른쪽: 직선의 잔차는 처음과 끝이 양수, 가운데가 음수인 긴 덩어리로 갈린다.</figcaption>
</figure>

그림 5의 직선 잔차는 대략 113번째 시점부터 390번째 시점까지 음수이고 그 앞뒤는 양수다. 잔차가 무작위라면 위아래를 자주 오가야 하는데 긴 덩어리로 몰려 있으니 직선이 놓친 구조가 남았다는 신호다. 2차 곡선은 계열의 휘어짐을 거의 따라간다.

모형 선택에는 AIC와 SIC를 쓴다.

| | Linear | Quadratic | Exponential |
| --- | --- | --- | --- |
| AIC | 19.35 | 15.83 | 17.15 |
| SIC | 19.37 | 15.86 | 17.17 |

두 기준 모두 quadratic이 가장 작아서 quadratic trend를 고른다. 1994년 예측을 보면 quadratic의 점예측은 실제 값에 가깝고 실제 값이 95% 구간 안에 대체로 들어온다. Linear는 점예측이 15만 근처인데 실제 값은 18만에서 19만 사이라 크게 빗나간다.

Durbin-Watson 통계량도 눈여겨볼 만하다. 잔차에 1차 자기상관이 없으면 2 근처인데 네 모형 모두 0에 가깝다. **추세를 걷어내도 잔차에 강한 양의 자기상관이 남아 있다는 뜻이다.** 이 남은 구조는 뒤에서 ARIMA로 다룬다.

### AIC와 SIC가 매기는 것

둘 다 모형 선택 기준이다. 잔차가 얼마나 작은지에 모수 개수만큼 벌점을 더한 값이고 **작을수록 좋은 모형이다.** 위 표의 값은 다음 식으로 나온다. $T$는 관측값 수 468, $k$는 회귀계수 개수, SSR은 잔차 제곱합이다.

$$
\text{AIC} = \ln\!\Big(\frac{\text{SSR}}{T}\Big) + \frac{2k}{T}, \qquad \text{SIC} = \ln\!\Big(\frac{\text{SSR}}{T}\Big) + \frac{k\ln T}{T} \tag{2}
$$

Linear trend로 검산해 보자. 회귀 표준오차에서 $\text{SSR} = s_e^2(T - k) = 15866.12^2 \times 466 = 1.173 \times 10^{11}$이고 $\ln(\text{SSR}/T) = 19.3396$이다. 여기에 $2k/T = 4/468 = 0.0085$를 더하면 AIC 19.34815, $k\ln T/T = 2 \times 6.1485/468 = 0.0263$을 더하면 SIC 19.36587이 나온다. 나머지 세 모형도 같은 식으로 출력 값이 반올림 차이 안에서 그대로 나온다.

벌점은 왜 필요할까? 답은 $R^2$의 성질에 있다. 설명변수를 늘리면 SSR은 줄기만 하고 $R^2$는 늘기만 한다. $R^2$로 고르면 늘 가장 큰 모형이 뽑힌다. 그런데 표본 안에서 잘 맞춘 만큼 표본 밖 예측이 좋아지지는 않는다(overfitting). AIC와 SIC는 $k$가 늘수록 벌점을 키워 이 경향을 누른다.

두 기준의 차이는 벌점 크기다. 모수 하나당 AIC는 $2/T$, SIC는 $\ln T/T$를 더한다. 그림 6은 두 벌점에 $T$를 곱해 나란히 그린 것이다.

<figure className="my-6 text-center">
  <img src="/images/posts/2026-10-15-time-series-2-decomposition-trend-regression/aic_sic_penalty.png" alt="관측값 수 T에 따른 AIC와 SIC의 모수 1개당 벌점" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">그림 6. AIC의 벌점은 2로 고정이고 SIC의 벌점은 ln T로 계속 커진다. 두 선은 T = e² ≈ 7.4에서 만난다.</figcaption>
</figure>

$T > e^2 \approx 7.4$이면 $\ln T > 2$라 SIC가 더 무겁게 벌한다. 이 예제에서는 $\ln 468 = 6.15$라 SIC 벌점이 AIC의 약 3배이고 SIC가 더 작은 모형 쪽으로 기운다. 참 모형이 후보 안에 있으면 $T$가 커질 때 SIC가 그 모형을 고를 확률은 1로 간다(일치성). AIC는 일치성이 없는 대신 예측오차가 작은 모형을 고르는 쪽에 맞춰져 있다. SIC는 BIC(Bayesian information criterion)라고도 부른다.

이 예제에서 linear와 quadratic의 AIC 차이 3.52는 거의 다 $\ln(\text{SSR}/T)$의 차이다. $e^{3.52} \approx 34$라 linear의 잔차 제곱 평균이 quadratic의 약 34배다. 모수가 하나 많은 quadratic이 더 받는 벌점은 AIC로 $2/468 = 0.004$, SIC로 $6.15/468 = 0.013$이라 결과에 거의 영향이 없다. **벌점이 결론을 가르는 건 적합도가 비슷한 모형끼리 비교할 때다.**

우도로 다시 쓸 수도 있다. 정규 오차 회귀의 로그우도는 다음과 같다.

$$
\ln L = -\frac{T}{2}\Big[1 + \ln(2\pi) + \ln\Big(\frac{\text{SSR}}{T}\Big)\Big]
$$

Linear trend 값을 넣으면 $-5189.53$으로 출력의 Log likelihood($-5189.529$)와 같다. 흔히 쓰는 정의 $\text{AIC} = -2\ln L + 2k$, $\text{SIC} = -2\ln L + k\ln T$를 $T$로 나누면 식 (2)에 상수 $1 + \ln(2\pi)$가 더해진 꼴이다. 상수는 순위를 바꾸지 않으니 어느 정의로 계산해도 같은 모형을 고른다.[^diebold]

### log-linear는 왜 비교표에서 빠졌는가

모형 선택 표의 exponential 열에는 log-linear의 AIC(-4.77)가 아니라 NLS의 AIC(17.15)가 들어가 있다.

**주의: AIC는 종속변수가 같은 모형끼리만 비교할 수 있다.** Log-linear의 우도는 $\ln y_t$의 우도라서 $y_t$의 우도와 기준이 다르다. -4.77을 표에 그대로 넣으면 log-linear가 압도적으로 좋아 보이지만 그건 착시다.

그래도 비교하고 싶다면 어떻게 할까? $y_t$ 기준으로 옮기면 된다. 변수 변환의 야코비안 때문에 $\ln L$에서 $\sum_t \ln y_t$를 빼야 한다. 식 (2) 꼴의 AIC로는 $\ln y_t$ 평균의 2배를 더하는 것과 같다. Log-linear 회귀 출력의 종속변수 평균 10.78072를 쓰면 다음과 같다.

$$
\text{AIC}_y = -4.770302 + 2 \times 10.78072 = 16.79, \qquad \text{SIC}_y = -4.752573 + 2 \times 10.78072 = 16.81
$$

표에 넣으면 quadratic(15.83)보다 크고 NLS(17.15)보다 작다. 고르는 모형은 그대로 quadratic이다. 지수 추세끼리만 보면 오차를 곱으로 둔 log-linear가 오차를 더한 NLS보다 AIC가 작다.

## Seasonal dummy로 계절성 잡기

회귀 쪽에서 계절성을 잡는 핵심 도구는 seasonal dummy다. 계절이 네 개($s = 4$)면 dummy 네 개를 만든다. $D_i$는 $i$분기에 1, 나머지 분기에 0인 변수다.

$$
D_1 = (1,0,0,0,1,0,0,0,\dots),\quad D_2 = (0,1,0,0,0,1,0,0,\dots),\quad \dots,\quad D_4 = (0,0,0,1,0,0,0,1,\dots)
$$

$$
S_t = \sum_{i=1}^{s} \gamma_i D_{it}
$$

$S_t$는 계절마다 정해진 값으로 바뀌는 절편이고 $\gamma_i$를 seasonal factor라고 부른다. 추정은 OLS다. $y_t$를 $D_1, \dots, D_s$에 회귀하거나 추세까지 넣어 $TIME, TIME^2, D_1, \dots, D_s$에 회귀한다. $y_t = S_t + \epsilon_t$라면 예측은 해당 계절의 계수다.

$$
\hat{y}_{T+h \mid T} = \sum_{i=1}^{s} \hat{\gamma}_i D_{i,T+h}
$$

추세 모형과 같은 이유로 95% 구간 $\hat{y}_{T+h \mid T} \pm 1.96\hat{\sigma}$의 폭은 일정하다.

미국 주택 착공 건수(1946년 1월부터 1993년 12월까지, 576개)에 월 dummy 12개를 절편 없이 넣은 결과를 보자. 역시 Diebold 교재의 예제다.

| D1 | D2 | D3 | D4 | D5 | D6 |
| --- | --- | --- | --- | --- | --- |
| 86.50417 | 89.50417 | 122.8833 | 142.1687 | 147.5000 | 145.9979 |

| D7 | D8 | D9 | D10 | D11 | D12 |
| --- | --- | --- | --- | --- | --- |
| 139.1125 | 138.4167 | 130.5625 | 134.0917 | 111.8333 | 92.15833 |

**계수가 곧 월별 평균이다.** 1월이 가장 낮고 5월이 가장 높다. 표준오차는 모두 4.029055로 같다. 달마다 관측값 수가 48개로 같으니 $(\mathbf{X}^\top\mathbf{X})^{-1}$의 대각원소가 모두 $1/48$이기 때문이다. 12개 계수의 평균 123.3944는 출력의 종속변수 평균과 같다. 행렬 부분에서 본 "$\hat\beta_j$는 $j$월의 평균"을 숫자로 확인한 셈이다. $R^2 = 0.383780$, Durbin-Watson은 0.154140이었다.

계수가 월별 평균이라는 점에서 이 회귀는 classical decomposition의 3단계와 같은 계산을 한다. 연도를 행, 월을 열로 놓고 열마다 평균하는 바로 그 표다.

### dummy를 전부 넣으면 생기는 일

추세와 계절성을 한 회귀에 넣을 수도 있다. 호주 분기별 맥주 생산량 모형이 그 예다.

$$
y_t = \beta_0 + \beta_1 t + \beta_2 d_{2,t} + \beta_3 d_{3,t} + \beta_4 d_{4,t} + \epsilon_t
$$

$d_{i,t}$는 $t$가 $i$분기면 1, 아니면 0이다. $\beta_1 t$가 추세, $\beta_2 \sim \beta_4$가 계절성을 맡는다.

1분기 dummy는 왜 뺐을까? 절편이 있는데 dummy 네 개를 모두 넣으면 $d_1 + d_2 + d_3 + d_4 = 1$이 절편 열 $\mathbf{1}$과 같아진다. **$\mathbf{X}$의 열이 선형독립이 아니게 되어 $(\mathbf{X}^\top\mathbf{X})^{-1}$이 존재하지 않는다(dummy trap).** 해결책은 둘이다. 주택 착공처럼 절편 없이 dummy를 모두 넣거나 맥주 모형처럼 절편을 두고 하나를 빼면 된다.

하나를 빼면 나머지 계수는 그 분기와 1분기의 차이를 잰다. 분기 평균이 100, 120, 140, 160이라면 100은 절편 $\beta_0$으로 가고 dummy 계수는 +20, +40, +60이 된다. 앞에서 classical decomposition의 (ii) 단계를 설명할 때 본 표의 위쪽 줄이 바로 이 방식이다. 수준과 계절을 나누는 기준이 하나로 정해지지 않는다는 같은 문제를 classical은 "합을 0으로", dummy 회귀는 "1분기를 기준으로" 푼 것이다.

적합 결과는 실제 값을 가로축, 적합값을 세로축에 놓은 산점도로 확인한다. 점이 대각선 $y = x$에 가까울수록 잘 맞은 것이다.

## 두 방법을 나란히 놓고 보면

다시 정리하면 추세와 계절성을 구하는 두 길은 이렇게 갈린다.

| | 이동평균 분해 (classical, STL) | 회귀 (추세 함수, dummy) |
| --- | --- | --- |
| 추세 | 시점마다 주변 평균이나 국소 직선 | 시간의 함수 하나 |
| 계절성 | 계절마다 평균 (STL은 천천히 변함) | dummy 계수 |
| 끝점과 미래 | classical은 끝점이 없고 둘 다 미래를 직접 주지 않음 | $TIME_{T+h}$와 dummy를 넣으면 나옴 |
| 주로 쓰는 곳 | 성분을 보고 계절 조정하기 | 예측 |

회귀 쪽은 예측식을 바로 주지만 추세와 계절성이 영원히 같은 모양이라고 가정한다. 소매 판매의 Durbin-Watson이 보여 주듯 그렇게 걷어내고도 잔차에는 구조가 남는다. 그렇다면 성분을 고정하지 않고 관측값이 들어올 때마다 조금씩 업데이트하면 어떨까? [3편](/posts/2026-10-15-time-series-3-exponential-smoothing)에서 다루는 지수평활법이 바로 그 방법이다.

---

**참고 문헌**

- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. [3장 Time series decomposition](https://otexts.com/fpp3/decomposition.html), [7장 Time series regression models](https://otexts.com/fpp3/regression.html)
- F. X. Diebold, *Forecasting in Economics, Business, Finance and Beyond* (2017), 3.3.12절(AIC, SIC), 5.2.4절(소매 판매), 5.3.4절(주택 착공). 저자 홈페이지에서 무료로 공개되어 있다: [Textbooks](https://www.sas.upenn.edu/~fdiebold/Textbooks.html)
- R. B. Cleveland, W. S. Cleveland, J. E. McRae, I. Terpenning (1990), STL: A Seasonal-Trend Decomposition Procedure Based on Loess, *Journal of Official Statistics* 6(1)

[^history]: 이후 1957년 Census II가 나왔고 이것이 X-11, X-12-ARIMA, X-13-ARIMA로 이어졌다. STL은 1983년, TRAMO/SEATS는 1990년대에 나왔다.

[^stl]: Cleveland, Cleveland, McRae & Terpenning (1990), *Journal of Official Statistics*. 아래 참고 문헌 참고.

[^median]: 예측할 때 $e^{\hat b_0 + \hat\beta_1 TIME_{T+h}}$는 평균이 아니라 중앙값이다. 평균을 원하면 $e^{\hat\sigma^2/2}$를 곱해야 한다. 이 예에서는 $\hat\sigma = 0.091879$라 $e^{\hat\sigma^2/2} \approx 1.0042$, 약 0.4% 차이다.

[^se]: 이 구간은 $\hat{\boldsymbol\beta}$의 추정 오차를 무시한 것이다. 실제로는 먼 미래일수록 추세선 자체의 불확실성이 커진다.

[^diebold]: Diebold 교재 3.3.12절은 $\text{AIC} = e^{2k/T}\,\frac{\text{SSR}}{T}$, $\text{SIC} = T^{k/T}\,\frac{\text{SSR}}{T}$로 정의한다. 회귀 출력의 값은 이 둘에 log를 씌운 것이라 식 (2)와 같다.
