---
title: "시계열분석 (4): 지수평활법, 관측값마다 업데이트하는 예측"
date: "2026-10-15"
category: "공부"
subcategory: "Statistics/Time Series Analysis"
tags: ["시계열분석", "지수평활법", "SES", "Holt-Winters"]
description: "관측값이 하나 들어올 때마다 수준을 업데이트하는 단순 지수평활(SES)에서 출발한다. 여기에 기울기를 더한 Holt 방법과 damped trend, 계절 성분까지 다루는 Holt-Winters를 차례로 유도하고 추세와 계절의 조합 아홉 가지를 정리한다."
---

세종대학교 인공지능데이터사이언스학과 2026년 2학기 시계열분석 수업을 따라가며 정리하는 시리즈의 네 번째 글이다. 수업 내용과 교재 [Forecasting: Principles and Practice, 3rd ed.](https://otexts.com/fpp3/)(이하 FPP3) 8장 앞부분(8.1~8.4절)을 바탕으로 썼다.

> **지수평활이 말하는 것: 새 관측값이 하나 들어올 때마다 지금의 수준을 얼마나 업데이트해야 할까?**

[2편](/posts/2026-10-15-time-series-2-decomposition)과 [3편](/posts/2026-10-15-time-series-3-trend-regression)에서는 추세와 계절성을 이동평균이나 회귀로 자료 전체에서 한 번에 구했다. 이번 글의 지수평활법(exponential smoothing)은 접근이 반대다. 관측값이 들어올 때마다 수준, 기울기, 계절 성분을 조금씩 업데이트하며 예측한다.

## 성분마다 하나씩 붙는 평활 모수

지수평활법은 시계열을 세 성분으로 나눠 따로 추적한다. 수준(level) $\ell_t$, 기울기(trend) $b_t$, 계절(seasonality) $s_t$다. **성분마다 평활 모수가 하나씩 붙고 그 모수가 성분이 얼마나 빨리 바뀌는지를 정한다.**

| 모수 | 성분 | 0이면 | 1이면 |
| --- | --- | --- | --- |
| $\alpha$ | 수준 $\ell_t$ | 수준이 바뀌지 않음 (상수) | 매번 새 관측값으로 바뀜 (naive) |
| $\beta$ | 기울기 $b_t$ | 기울기 고정 (직선 추세) | 관측값마다 기울기가 크게 바뀜 |
| $\gamma$ | 계절 $s_t$ | 계절 성분 고정 (해마다 같음) | 계절 성분이 매번 새로 바뀜 |

$\gamma = 0$은 [2편](/posts/2026-10-15-time-series-2-decomposition) classical decomposition의 가정과 같다. 거기서는 계절마다 평균을 내서 해마다 똑같은 계절 값을 썼다.

세 성분은 직접 보이지 않는다. 보이는 것은 관측값 $y_t$뿐이고 성분은 관측값으로 추정해야 한다. 그래서 이 성분들을 상태(state)라고 부르고 한 시점의 계산은 다음 순서로 돈다.

$$
(\ell_{t-1}, b_{t-1}, s_{t-1}) \;\Rightarrow\; \hat{y}_{t|t-1} \;\Rightarrow\; y_t \text{ 관측} \;\Rightarrow\; (\ell_t, b_t, s_t) \;\Rightarrow\; \hat{y}_{t+1|t}
$$

$\hat{y}_{t|t-1}$은 시점 $t-1$까지의 정보로 낸 시점 $t$의 예측이다. 직전 상태로 다음 값을 예측하고 실제 값을 본 뒤 상태를 업데이트한다. 업데이트한 상태로 그다음 값을 다시 예측한다.

### 수준이란 무엇일까

**수준은 관측값에서 그때그때의 흔들림을 걷어 낸 "지금 이 계열의 높이"다.** 관측값은 수준에 흔들림이 더해진 값이다.

시험 점수로 생각해 보자. 점수는 실력에 그날 컨디션이 더해진 값이고 여기서 실력이 수준이다. 점수 다섯 개가 70, 90, 60, 80, 85이고 처음 짐작한 실력이 $\ell_0 = 75$라고 하자. 시험을 볼 때마다 다음 식으로 실력 추정을 업데이트한다.

$$
\ell_t = \alpha y_t + (1-\alpha)\ell_{t-1} \tag{1}
$$

| $t$ | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 점수 $y_t$ | 70 | 90 | 60 | 80 | 85 |
| $\ell_t$ ($\alpha = 0.3$) | 73.5 | 78.4 | 72.9 | 75.0 | 78.0 |
| $\ell_t$ ($\alpha = 0.9$) | 70.5 | 88.0 | 62.8 | 78.3 | 84.3 |

<Fig src="/images/posts/2026-10-15-time-series-4-exponential-smoothing/level-update.gif" alt="시험 점수 다섯 개를 보며 α = 0.3과 α = 0.9의 수준이 한 회차씩 업데이트되는 애니메이션">그림 1. 60점을 받은 뒤 수준은 얼마나 내려갈까? $\alpha = 0.3$(파랑)은 72.9, $\alpha = 0.9$(주황)는 62.8까지 내려간다.</Fig>

그림 1에서 보듯 $\alpha = 0.3$이면 시험 한 번으로 판단을 크게 바꾸지 않는다. 60점을 받아도 추정은 78.4에서 72.9로 내려갈 뿐이다. $\alpha = 0.9$는 최근 점수를 거의 그대로 실력으로 본다. 어느 쪽이 맞을까? 실력이 실제로 얼마나 빨리 바뀌느냐에 달렸다.

- $\alpha = 0$: 실력은 그대로이고 점수 차이는 전부 컨디션 탓이라고 본다. 그러면 다섯 점수의 평균 77이 가장 좋은 추정이다.
- $\alpha = 1$: 점수가 달라진 만큼 실력도 달라졌다고 본다. 마지막 점수 85가 곧 실력이다.

추세가 있으면 수준은 지금 시점의 높이를 맡고 기울기가 한 시점에 오르는 양을 맡는다. 계절성이 있으면 수준은 계절 효과까지 뺀 높이가 된다. 여름에 잘 팔리는 상품이라면 여름이라서 더 팔린 몫은 계절 성분이 가져간다. 두 경우 모두 뒤의 Holt와 Holt-Winters에서 식으로 다시 확인해 보자.

## naive와 average 사이 어딘가

첫 번째 방법은 수준만 쓰는 단순 지수평활(simple exponential smoothing, SES)이다. 추세와 계절성이 뚜렷하지 않은 자료에 쓴다. FPP3의 예제는 1960~2017년 알제리의 GDP 대비 수출 비율이다.

출발점은 아주 단순한 두 예측이다. $T$는 마지막 관측 시점이고 $h$는 몇 step 앞을 예측하는지를 나타낸다.

$$
\text{naive: } \hat{y}_{T+h|T} = y_T, \qquad \text{average: } \hat{y}_{T+h|T} = \frac{1}{T}\sum_{t=1}^{T} y_t
$$

Naive는 마지막 관측값에 가중치를 전부 주고 average는 모든 관측값에 같은 가중치를 준다. 둘 다 극단이다. 그 사이 어딘가에 답이 있지 않을까? **SES는 최근 값에 큰 가중치를 주고 과거로 갈수록 가중치를 등비로 줄이는 가중평균이다.**

$$
\hat{y}_{T+1|T} = \alpha y_T + \alpha(1-\alpha) y_{T-1} + \alpha(1-\alpha)^2 y_{T-2} + \cdots, \qquad 0 \le \alpha \le 1
$$

### 가중치는 등비로 줄어든다

한 칸 과거로 갈 때마다 가중치에 $1-\alpha$가 곱해진다. $\alpha = 0.2$면 0.8배씩, $\alpha = 0.4$면 0.6배씩 줄어드는 식이다.

| | $\alpha = 0.2$ | $\alpha = 0.4$ | $\alpha = 0.6$ | $\alpha = 0.8$ |
| --- | --- | --- | --- | --- |
| $y_T$ | 0.2000 | 0.4000 | 0.6000 | 0.8000 |
| $y_{T-1}$ | 0.1600 | 0.2400 | 0.2400 | 0.1600 |
| $y_{T-2}$ | 0.1280 | 0.1440 | 0.0960 | 0.0320 |
| $y_{T-3}$ | 0.1024 | 0.0864 | 0.0384 | 0.0064 |
| $y_{T-4}$ | 0.0819 | 0.0518 | 0.0154 | 0.0013 |
| $y_{T-5}$ | 0.0655 | 0.0311 | 0.0061 | 0.0003 |

<Fig src="/images/posts/2026-10-15-time-series-4-exponential-smoothing/ses-weights.svg" alt="α = 0.2, 0.5, 0.8일 때 최근 열 개 관측값에 붙는 SES 가중치 막대그래프">그림 2. $\alpha$가 클수록 가중치가 최근 값에 쏠린다. $\alpha = 0.8$이면 세 시점 전 값의 가중치가 벌써 0.0064다.</Fig>

$\alpha$가 클수록 가중치가 빨리 줄어 최근 값에 쏠린다(그림 2). $\alpha = 0.8$이면 다섯 시점 전 값의 가중치가 0.0003밖에 안 된다.

가중치를 모두 더하면 1이라서 이름 그대로 가중평균이다. 관측값이 $T$개뿐이면 합이 유한해지고 남는 몫은 시작값 $\ell_0$이 받는다(바로 다음 절에서 나온다). 등비수열의 합으로 확인해 보자.

$$
\sum_{j=0}^{T-1}\alpha(1-\alpha)^j + (1-\alpha)^T = \big[1 - (1-\alpha)^T\big] + (1-\alpha)^T = 1
$$

딥러닝을 공부했다면 낯익은 식이다. Adam의 1차 모멘트 $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$나 BatchNorm의 running mean이 같은 점화식이다. Adam의 기본값 $\beta_1 = 0.9$는 $\alpha = 0.1$인 SES와 같다.

### 같은 식, 세 가지 꼴

위 합을 점화식으로 쓰면 다음 예측은 최근 관측값과 직전 예측의 가중평균이 된다. 이 꼴을 weighted average form이라고 부른다.

$$
\hat{y}_{t+1|t} = \alpha y_t + (1-\alpha)\hat{y}_{t|t-1}
$$

표본 안의 각 시점에서 이렇게 낸 $\hat{y}_{t|t-1}$을 fitted value라고 하고 $e_t = y_t - \hat{y}_{t|t-1}$이 잔차다. 이미 관측한 값을 모형이 1-step 앞서 얼마나 맞혔는지 보는 셈이다.

점화식에는 시작점이 필요하다. $\hat{y}_{1|0}$은 정의되지 않으므로 이 자리를 $\ell_0$이라는 값으로 둔다. 한 줄씩 대입해 보자.

$$
\begin{aligned}
\hat{y}_{2|1} &= \alpha y_1 + (1-\alpha)\ell_0 \\
\hat{y}_{3|2} &= \alpha y_2 + (1-\alpha)\hat{y}_{2|1} = \alpha y_2 + \alpha(1-\alpha) y_1 + (1-\alpha)^2 \ell_0 \\
\hat{y}_{4|3} &= \alpha y_3 + \alpha(1-\alpha) y_2 + \alpha(1-\alpha)^2 y_1 + (1-\alpha)^3 \ell_0
\end{aligned}
$$

같은 규칙을 $T$까지 이어 가면 앞의 가중합이 나온다.

$$
\hat{y}_{T+1|T} = \sum_{j=0}^{T-1} \alpha(1-\alpha)^j y_{T-j} + (1-\alpha)^T \ell_0
$$

$T$가 크면 $\ell_0$의 가중치 $(1-\alpha)^T$는 거의 0이 된다.

세 번째 꼴이 component form이다. 예측식과 평활식을 나눠 쓰고 평활식이 바로 식 (1)이다.

$$
\text{Forecast: } \hat{y}_{t+h|t} = \ell_t, \qquad \text{Smoothing: } \ell_t = \alpha y_t + (1-\alpha)\ell_{t-1}
$$

$\ell_t$ 자리에 $\hat{y}_{t+1|t}$를, $\ell_{t-1}$ 자리에 $\hat{y}_{t|t-1}$을 넣으면 weighted average form으로 돌아간다. 이름만 바꾼 것 같지만 성분을 따로 적어 두는 이 꼴이 Holt와 Holt-Winters로 넓히는 발판이 된다.

### 예측은 왜 평평할까

Component form의 예측식을 보면 $h$가 얼마든 예측이 $\ell_t$ 하나다. 2-step 앞도 10-step 앞도 같은 값이다. 왜 그럴까? 아직 관측하지 않은 $y_{T+1}$ 자리에 그 예측값 $\ell_T$를 넣어 보자.

$$
\begin{aligned}
\hat{y}_{T+1|T} &= \ell_T \\
\hat{y}_{T+2|T} &= \hat{\ell}_{T+1} = \alpha\,\hat{y}_{T+1|T} + (1-\alpha)\ell_T = \alpha\ell_T + (1-\alpha)\ell_T = \ell_T
\end{aligned}
$$

**새로 알게 된 정보가 없으니 수준이 그대로다.** 시험 점수로 말하면 실력이 앞으로 오를지 내릴지 모르니 다음 시험도 그다음 시험도 지금 실력으로 예상하는 것이다.

같은 시점을 예측해도 언제 예측했느냐에 따라 값은 달라진다. $\hat{y}_{t+2|t} \ne \hat{y}_{t+2|t+1}$이다. 앞은 $\ell_t$이고 뒤는 $y_{t+1}$을 보고 업데이트한 $\ell_{t+1}$이다.

### $\alpha$와 $\ell_0$는 어떻게 고르는가

1-step 예측오차의 제곱합(SSE)을 최소로 하는 값을 고른다.

$$
\text{SSE} = \sum_{t=1}^{T}\big(y_t - \hat{y}_{t|t-1}\big)^2 = \sum_{t=1}^{T} e_t^2
$$

$\hat{y}_{t|t-1}$ 안에 $\alpha(1-\alpha)^j$가 들어 있어서 SSE는 $\alpha$의 고차 다항식이다. 회귀처럼 공식 한 줄로 풀리지 않으니 수치 최적화로 푼다.

알제리 자료($t = 1, \dots, 58$)에서는 $\hat\alpha = 0.84$, $\hat\ell_0 = 39.54$가 나온다.

| 연도 | $t$ | $y_t$ | $\ell_t$ | $\hat{y}_{t\mid t-1}$ |
| --- | --- | --- | --- | --- |
| 1959 | 0 | | 39.54 | |
| 1960 | 1 | 39.04 | 39.12 | 39.54 |
| 1961 | 2 | 46.24 | 45.10 | 39.12 |
| 1962 | 3 | 19.79 | 23.84 | 45.10 |
| 1963 | 4 | 24.68 | 24.55 | 23.84 |
| ⋮ | | | | |
| 2017 | 58 | 22.64 | 22.44 | 21.43 |

첫 줄을 직접 계산해 보자.

$$
\ell_1 = 0.84 \times 39.04 + 0.16 \times 39.54 = 39.12
$$

이 값이 그대로 1961년의 예측이 된다. 수준 열의 값이 한 줄 아래 예측 열로 내려가는 모습이 $\hat{y}_{t+1|t} = \ell_t$ 그 자체다. 앞 네 줄은 다시 계산해도 39.12, 45.10, 23.84, 24.55로 같았다[^algeria]. 2018~2022년 예측은 모두 $\ell_{58} = 22.44$로 평평하다.

$\hat\alpha = 0.84$는 꽤 큰 값이라 수준이 최근 관측값을 빠르게 따라간다는 뜻이다. 반대로 $\alpha = 0$으로 두면 모든 예측이 $\ell_0$ 하나가 되고 SSE를 최소로 하는 $\ell_0$은 표본 평균이다. $\alpha = 1$이면 naive다. 앞에서 본 두 극단이 $\alpha$의 양 끝이고 SES는 그 사이를 자료로 고르는 방법이다.

## 추세를 따라가는 Holt 방법

Holt(1957)는 SES에 기울기를 더해 추세가 있는 자료로 확장했다.

$$
\begin{aligned}
\text{Forecast: } & \hat{y}_{t+h|t} = \ell_t + h b_t \\
\text{Level: } & \ell_t = \alpha y_t + (1-\alpha)(\ell_{t-1} + b_{t-1}) \\
\text{Trend: } & b_t = \beta^*(\ell_t - \ell_{t-1}) + (1-\beta^*) b_{t-1}
\end{aligned}
$$

두 평활식을 하나씩 읽어 보자.

- Level 식의 $\ell_{t-1} + b_{t-1}$은 직전 시점에서 낸 예측 $\hat{y}_{t|t-1}$이다. 그래서 $\ell_t = \alpha y_t + (1-\alpha)\hat{y}_{t|t-1}$로 SES와 같은 꼴이다.
- Trend 식은 이번에 실제로 오른 양 $\ell_t - \ell_{t-1}$과 원래 알던 기울기 $b_{t-1}$의 가중평균이다.

계산 순서도 정해져 있다. $y_t$를 관측한 뒤 $\ell_t$를 먼저 구하고 그 $\ell_t$로 $b_t$를 구한다. 예측식에 $h = 0$을 넣으면 $\ell_t$가 남으니 수준은 시점 $t$에서 그은 직선의 높이이고 $b_t$는 그 직선의 기울기다.

**주의:** 여기서 쓰는 $\beta^*$와 표의 $\beta$는 다른 모수다. [5편](/posts/2026-10-15-time-series-5-ets-models)에서 다룰 상태공간 꼴에서 $\beta = \alpha\beta^*$라는 관계로 이어진다. 별표는 그 구분을 위한 표시다.

### 미래의 수준과 기울기

예측식은 왜 $h$의 1차식일까? SES의 평평한 예측과 같은 방법으로 확인할 수 있다. 모르는 $y_{T+1}$ 자리에 예측값 $\ell_T + b_T$를 넣어 보자.

$$
\begin{aligned}
\hat{\ell}_{T+1} &= \alpha(\ell_T + b_T) + (1-\alpha)(\ell_T + b_T) = \ell_T + b_T \\
\hat{b}_{T+1} &= \beta^*(\hat{\ell}_{T+1} - \ell_T) + (1-\beta^*) b_T = \beta^* b_T + (1-\beta^*) b_T = b_T \\
\hat{y}_{T+2|T} &= \hat{\ell}_{T+1} + \hat{b}_{T+1} = \ell_T + 2b_T
\end{aligned}
$$

Level 식은 같은 값 두 개의 가중평균이 되고 Trend 식의 "실제로 오른 양"도 $b_T$가 된다. 그러니 둘 다 그대로 이어진다. 같은 일을 $h$번 되풀이하면 다음을 얻는다.

$$
\hat{\ell}_{T+h} = \ell_T + h b_T, \qquad \hat{b}_{T+h} = b_T
$$

**기울기는 그대로 두고 수준만 매 step 기울기만큼 올리는 셈이다.** 여기서 $\ell_T, b_T$는 $T$까지의 자료로 구한 추정값이고 $\hat{\ell}_{T+h}, \hat{b}_{T+h}$는 $T$ 시점에서 본 미래 상태의 예측값이다. 예측식 $\hat{y}_{T+h|T} = \ell_T + hb_T$가 $\hat{\ell}_{T+h}$와 같은 값이라는 것도 보인다.

### 호주 인구에서 $\alpha$가 거의 1인 이유

FPP3의 호주 인구 예제에서는 $\hat\alpha = 0.9999$, $\hat\beta^* = 0.3267$이 나온다. 인구는 추세가 강하고 잡음이 거의 없이 매끈하게 늘어나는 계열이라 수준이 관측값을 거의 그대로 따라간다. $\hat\beta^*$도 작지 않아서 기울기가 자주 업데이트된다. 시작값은 $\ell_0 = 10.05$, $b_0 = 0.22$다.

| 연도 | $t$ | $y_t$ | $\ell_t$ | $b_t$ | $\hat{y}_{t\mid t-1}$ |
| --- | --- | --- | --- | --- | --- |
| 1959 | 0 | | 10.05 | 0.22 | |
| 1960 | 1 | 10.28 | 10.28 | 0.22 | 10.28 |
| 1961 | 2 | 10.48 | 10.48 | 0.22 | 10.50 |
| 1965 | 6 | 11.39 | 11.39 | 0.22 | 11.39 |
| 1966 | 7 | 11.65 | 11.65 | 0.23 | 11.61 |
| ⋮ | | | | | |
| 2017 | 58 | 24.60 | 24.60 | 0.37 | 24.57 |

수준 열이 관측값 열과 똑같다. 1960년 예측은 $\ell_0 + b_0$이다[^pop]. 1966년 예측 11.61도 $\ell_{1965} + b_{1965} = 11.39 + 0.22$다. 기울기 한 줄도 손으로 따라가 보면 표와 맞는다.

$$
b_1 = 0.3267 \times (10.28 - 10.05) + 0.6733 \times 0.22 = 0.223
$$

### 끝없이 오르는 직선을 꺾는 $\phi$

Holt의 예측은 같은 기울기로 끝없이 오르거나 내려가서 먼 미래를 과대예측하기 쉽다. Gardner와 McKenzie(1985)는 추세를 언젠가 평평하게 꺾는 감쇠 모수 $\phi$를 넣었다.

$$
\begin{aligned}
\text{Forecast: } & \hat{y}_{t+h|t} = \ell_t + (\phi + \phi^2 + \cdots + \phi^h) b_t \\
\text{Level: } & \ell_t = \alpha y_t + (1-\alpha)(\ell_{t-1} + \phi b_{t-1}) \\
\text{Trend: } & b_t = \beta^*(\ell_t - \ell_{t-1}) + (1-\beta^*)\phi b_{t-1}
\end{aligned}
\qquad 0 < \phi < 1
$$

$\phi = 1$이면 Holt와 같다. 앞에서처럼 $y_{T+1}$ 자리에 예측값 $\ell_T + \phi b_T$를 넣으면 $\hat{b}_{T+1} = \beta^*\phi b_T + (1-\beta^*)\phi b_T = \phi b_T$가 된다. 기울기가 step마다 $\phi$배로 줄고 그 기울기를 차례로 더한 것이 예측식의 $\phi + \phi^2 + \cdots + \phi^h$다. 등비급수라서 $h \to \infty$이면 예측이 한 값으로 모인다.

$$
\hat{y}_{T+h|T} \;\to\; \ell_T + \frac{\phi}{1-\phi}\, b_T
$$

<Fig src="/images/posts/2026-10-15-time-series-4-exponential-smoothing/holt-vs-damped.svg" alt="같은 마지막 상태에서 출발한 Holt 예측 직선과 감쇠 추세 예측 곡선">그림 3. 같은 마지막 상태에서 출발해도 Holt(파랑)는 직선으로 오르고 감쇠 추세(주황, $\phi = 0.9$)는 $\ell_T + 9b_T = 79.4$ 아래에서 멈춘다(모의 자료).</Fig>

$\phi = 0.9$면 $\phi/(1-\phi) = 9$라서 아무리 멀리 가도 기울기를 $9b_T$까지만 더한다. 그림 3의 모의 자료는 마지막 상태가 $\ell_T = 67.94$, $b_T = 1.269$이고 감쇠 예측은 40-step 뒤에도 79.19로 상한 79.36 아래에 머문다. **가까운 미래는 추세를 따르고 먼 미래는 평평해진다.** FPP3는 $\phi$가 0.8보다 작으면 감쇠가 너무 세고 1에 가까우면 감쇠 없는 모형과 구분이 안 되어 보통 0.8~0.98로 둔다고 설명한다. 여러 계열을 자동으로 예측할 때 가장 널리 쓰이는 방법 가운데 하나이기도 하다.

## 계절까지 다루는 Holt-Winters

Holt-Winters는 Holt에 계절 성분 $s_t$를 하나 더 얹는다. $m$은 계절 주기(분기 자료면 4, 월별 자료면 12)이고 $k$는 $(h-1)/m$의 정수 부분이다.

$$
\begin{aligned}
\text{Forecast: } & \hat{y}_{t+h|t} = \ell_t + h b_t + s_{t+h-m(k+1)} \\
\text{Level: } & \ell_t = \alpha(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1}) \\
\text{Trend: } & b_t = \beta^*(\ell_t - \ell_{t-1}) + (1-\beta^*) b_{t-1} \\
\text{Season: } & s_t = \gamma(y_t - \ell_{t-1} - b_{t-1}) + (1-\gamma) s_{t-m}
\end{aligned}
$$

Level 식이 평활하는 값은 $y_t$가 아니라 작년 같은 철의 계절 값을 뺀 $y_t - s_{t-m}$이다. 앞에서 말한 "계절 효과까지 뺀 높이"가 이 식이다. Season 식은 반대로 관측값에서 예상 높이 $\ell_{t-1} + b_{t-1}$을 뺀 나머지를 계절 성분의 새 정보로 쓴다. 여름이라 더 팔린 몫은 $s_t$가 가져간다. 그 몫을 빼고도 예상보다 높을 때만 수준이 오른다.

FPP3는 Season 식을 $s_t = \gamma^*(y_t - \ell_t) + (1-\gamma^*) s_{t-m}$으로 쓰기도 한다. 여기에 Level 식의 $\ell_t$를 대입하면 위 식이 되고 두 모수 사이에는 $\gamma = \gamma^*(1-\alpha)$라는 관계가 있다.

### 26개월 뒤 예측에 왜 작년 계절을 쓸까

예측식의 $s_{t+h-m(k+1)}$은 처음 보면 낯설다. 월별 자료($m = 12$)에서 26개월 뒤를 예측한다고 해 보자. $k = \lfloor 25/12 \rfloor = 2$이므로 계절 첨자는 이렇게 된다.

$$
t + h - m(k+1) = t + 26 - 36 = t - 10
$$

$t+26$과 $t-10$은 정확히 36개월 차이라 같은 달이고 $t-10$은 이미 관측한 마지막 1년 안에 있다. 예측하려는 달과 같은 달 가운데 가장 최근에 추정한 계절 값, 말하자면 "작년의 계절성"을 쓰는 것이다. $h \le m$이면 $k = 0$이라 그냥 $s_{t+h-m}$이다.

왜 작년 값을 그대로 쓸까? Holt에서처럼 확인할 수 있다. $y_{T+1}$ 자리에 예측값 $\ell_T + b_T + s_{T+1-m}$을 넣으면 Season 식의 $y_{T+1} - \ell_T - b_T$가 $s_{T+1-m}$이 된다.

$$
\hat{s}_{T+1} = \gamma\, s_{T+1-m} + (1-\gamma)\, s_{T+1-m} = s_{T+1-m}
$$

**미래의 계절 값은 작년 같은 철의 값을 그대로 이어받아 $m$ 주기로 되풀이된다.** 수준과 기울기는 Holt와 똑같이 $\ell_T + hb_T$, $b_T$로 이어진다.

### 계절 변동이 커지면 곱셈으로

계절 성분을 빼는 대신 나누는 방식도 있다.

$$
\begin{aligned}
\text{Forecast: } & \hat{y}_{t+h|t} = (\ell_t + h b_t)\, s_{t+h-m(k+1)} \\
\text{Level: } & \ell_t = \alpha\frac{y_t}{s_{t-m}} + (1-\alpha)(\ell_{t-1} + b_{t-1}) \\
\text{Trend: } & b_t = \beta^*(\ell_t - \ell_{t-1}) + (1-\beta^*) b_{t-1} \\
\text{Season: } & s_t = \gamma\frac{y_t}{\ell_{t-1} + b_{t-1}} + (1-\gamma) s_{t-m}
\end{aligned}
$$

<Fig src="/images/posts/2026-10-15-time-series-4-exponential-smoothing/additive-vs-multiplicative.svg" alt="같은 수준 위에 덧셈 계절과 곱셈 계절을 얹은 두 모의 계열">그림 4. 덧셈 계절(왼쪽)은 진폭이 일정하고 곱셈 계절(오른쪽)은 수준이 오를수록 진폭도 커진다(모의 자료).</Fig>

곱셈 방식은 그림 4의 오른쪽처럼 계절 변동의 폭이 수준에 비례해 커지는 계열에 쓴다. [1편](/posts/2026-10-15-time-series-1-forecasting-basics)의 additive와 multiplicative 분해를 고르던 기준과 같다. FPP3의 호주 국내 숙박 여행(분기별) 예제에 두 방법을 모두 써 보면 예측이 꽤 비슷하게 나온다. 성분을 그려 보면 덧셈 쪽 계절 성분은 0 주위에서 움직이고 곱셈 쪽은 1 주위에서 움직인다. [2편](/posts/2026-10-15-time-series-2-decomposition) classical decomposition의 덧셈 분해와 곱셈 분해 그림에서 보던 차이와 같다.

감쇠 추세와 곱셈 계절을 합친 Holt-Winters damped도 있다. 예측식은 $\hat{y}_{t+h|t} = \big[\ell_t + (\phi + \cdots + \phi^h) b_t\big]\, s_{t+h-m(k+1)}$이고 평활식은 앞의 두 방법에서 $b_{t-1}$을 $\phi b_{t-1}$로 바꾼 꼴이다.

## 아홉 가지 조합

추세를 없음(N), 덧셈(A), 감쇠 덧셈($\text{A}_d$) 셋 중에서 고르고 계절을 없음(N), 덧셈(A), 곱셈(M) 셋 중에서 고르면 아홉 가지 방법이 나온다. **약칭의 앞 글자가 추세, 뒤 글자가 계절이다.** 지금까지 본 방법은 이 표의 여섯 칸을 차지한다.

| 약칭 | 방법 |
| --- | --- |
| (N,N) | Simple exponential smoothing |
| (A,N) | Holt's linear method |
| ($\text{A}_d$,N) | Additive damped trend method |
| (A,A) | Additive Holt-Winters' method |
| (A,M) | Multiplicative Holt-Winters' method |
| ($\text{A}_d$,M) | Holt-Winters' damped method |

지금까지 본 것은 모두 점예측을 내는 계산 규칙이다. 예측구간을 내려면 오차가 어떤 분포를 따르는지까지 정한 모형이 필요하다. 이 표의 방법들에 오차 분포를 붙인 ETS 모형은 [5편](/posts/2026-10-15-time-series-5-ets-models)에서 이어 간다.

---

**참고**

- FPP3 [8.1 Simple exponential smoothing](https://otexts.com/fpp3/ses.html), [8.2 Methods with trend](https://otexts.com/fpp3/holt.html), [8.3 Methods with seasonality](https://otexts.com/fpp3/holt-winters.html), [8.4 A taxonomy of exponential smoothing methods](https://otexts.com/fpp3/taxonomy.html)

[^algeria]: 그 뒤로는 $\alpha$를 0.84로 반올림해 계산한 탓에 소수 둘째 자리가 0.01씩 어긋난다.

[^pop]: $10.05 + 0.22 = 10.27$이 아니라 10.28로 찍힌 것은 반올림하기 전의 $\ell_0$, $b_0$로 계산했기 때문이다.
