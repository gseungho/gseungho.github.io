---
title: "시계열분석 (3): 지수평활법과 ETS, 관측값마다 고쳐 쓰는 예측"
date: "2026-10-15"
category: "공부"
subcategory: "Statistics/Time Series Analysis"
tags: ["시계열분석", "지수평활법", "ETS", "Holt-Winters"]
description: "SES, Holt, Holt-Winters를 유도하고 오차 분포를 붙인 ETS 모형으로 예측구간을 구합니다. 수준의 직관, 모수 추정, 베이즈 해석, 회귀와의 차이도 함께 정리했습니다."
---

세종대학교 인공지능데이터사이언스학과 2026년 2학기 시계열분석 수업을 따라가며 정리하는 시리즈의 세 번째 글입니다. 수업 내용과 교재 [Forecasting: Principles and Practice, 3rd ed.](https://otexts.com/fpp3/)(이하 FPP3) 8장을 바탕으로 썼습니다.

[2편](/posts/2026-10-15-time-series-2-decomposition-trend-regression)에서는 추세와 계절성을 이동평균이나 회귀로 **자료 전체에서 한 번에** 구했습니다. 이번 글의 지수평활법(exponential smoothing)은 접근이 반대예요. 관측값이 하나 들어올 때마다 수준, 기울기, 계절 성분을 조금씩 고쳐 가며 예측합니다.

## 성분마다 하나씩 붙는 평활 모수

지수평활법은 시계열을 세 성분으로 나눠 들고 다닙니다. 수준(level) $\ell_t$, 기울기(trend) $b_t$, 계절(seasonality) $s_t$예요. 성분마다 평활 모수가 하나씩 붙는데 이 모수가 그 성분이 얼마나 빨리 바뀌는지를 정합니다.

| 모수 | 성분 | 0이면 | 1이면 |
| --- | --- | --- | --- |
| $\alpha$ | 수준 $\ell_t$ | 수준이 바뀌지 않음 (상수) | 매번 새 관측값으로 바뀜 (naive) |
| $\beta$ | 기울기 $b_t$ | 기울기 고정 (직선 추세) | 관측값마다 기울기가 크게 바뀜 |
| $\gamma$ | 계절 $s_t$ | 계절 성분 고정 (해마다 같음) | 계절 성분이 매번 새로 바뀜 |

$\gamma = 0$은 2편 classical decomposition의 가정과 같습니다. 거기서는 계절마다 평균을 내서 해마다 똑같은 계절 값을 썼거든요.

세 성분은 직접 보이지 않습니다. 보이는 것은 관측값 $y_t$뿐이고 성분은 관측값으로 추정해야 하죠. 그래서 이 성분들을 상태(state)라고 부르고 한 시점의 계산은 다음 순서로 돕니다.

$$
(\ell_{t-1}, b_{t-1}, s_{t-1}) \;\Rightarrow\; \hat{y}_{t|t-1} \;\Rightarrow\; y_t \text{ 관측} \;\Rightarrow\; (\ell_t, b_t, s_t) \;\Rightarrow\; \hat{y}_{t+1|t}
$$

$\hat{y}_{t|t-1}$은 시점 $t-1$까지의 정보로 낸 시점 $t$의 예측입니다. 직전 상태로 다음 값을 예측한 뒤 실제 값을 보고 상태를 고칩니다. 고친 상태로 그다음 값을 다시 예측하고요.

### 수준이란 무엇일까

수준은 관측값에서 그때그때의 흔들림을 걷어 낸 "지금 이 계열의 높이"입니다. 관측값은 수준에 흔들림이 더해진 값이고요.

시험 점수로 생각하면 쉽습니다. 점수는 실력에 그날 컨디션이 더해진 값이고 여기서 실력이 수준이에요. 점수 다섯 개가 70, 90, 60, 80, 85이고 처음 짐작한 실력이 $\ell_0 = 75$라고 해 보겠습니다. 시험을 볼 때마다 다음 식으로 실력 추정을 고칩니다.

$$
\ell_t = \alpha y_t + (1-\alpha)\ell_{t-1}
$$

| $t$ | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 점수 $y_t$ | 70 | 90 | 60 | 80 | 85 |
| $\ell_t$ ($\alpha = 0.3$) | 73.5 | 78.4 | 72.9 | 75.0 | 78.0 |
| $\ell_t$ ($\alpha = 0.9$) | 70.5 | 88.0 | 62.8 | 78.3 | 84.3 |

$\alpha = 0.3$이면 시험 한 번으로 판단을 크게 바꾸지 않습니다. 60점을 받아도 추정은 78.4에서 72.9로 내려갈 뿐이죠. $\alpha = 0.9$는 최근 점수를 거의 그대로 실력으로 봅니다. 어느 쪽이 맞는지는 실력이 실제로 얼마나 빨리 바뀌느냐에 달렸어요.

- $\alpha = 0$: 실력은 그대로이고 점수 차이는 전부 컨디션 탓이라고 봅니다. 그러면 다섯 점수의 평균 77이 가장 좋은 추정입니다.
- $\alpha = 1$: 점수가 달라진 만큼 실력도 달라졌다고 봅니다. 마지막 점수 85가 곧 실력이에요.

추세가 있으면 수준은 지금 시점의 높이를 맡고 기울기가 한 시점에 오르는 양을 맡습니다. 계절성이 있으면 수준은 계절 효과까지 뺀 높이가 되고요. 여름에 잘 팔리는 상품이라면 여름이라서 더 팔린 몫은 계절 성분이 가져갑니다. 두 경우 모두 뒤의 Holt와 Holt-Winters에서 식으로 다시 확인하겠습니다.

## naive와 average 사이 어딘가

첫 번째 방법은 수준만 쓰는 단순 지수평활(simple exponential smoothing, SES)입니다. 추세와 계절성이 뚜렷하지 않은 자료에 씁니다. FPP3의 예제는 1960~2017년 알제리의 GDP 대비 수출 비율이에요.

출발점은 아주 단순한 두 예측입니다. $T$는 마지막 관측 시점이고 $h$는 몇 걸음 앞을 예측하는지를 나타냅니다.

$$
\text{naive: } \hat{y}_{T+h|T} = y_T, \qquad \text{average: } \hat{y}_{T+h|T} = \frac{1}{T}\sum_{t=1}^{T} y_t
$$

Naive는 마지막 관측값에 가중치를 전부 주고 average는 모든 관측값에 같은 가중치를 줍니다. SES는 그 사이에서 최근 값에 큰 가중치를 주고 과거로 갈수록 줄입니다.

$$
\hat{y}_{T+1|T} = \alpha y_T + \alpha(1-\alpha) y_{T-1} + \alpha(1-\alpha)^2 y_{T-2} + \cdots, \qquad 0 \le \alpha \le 1
$$

### 가중치는 등비로 줄어든다

한 칸 과거로 갈 때마다 가중치에 $1-\alpha$가 곱해집니다. $\alpha = 0.2$면 0.8배씩, $\alpha = 0.4$면 0.6배씩 줄어드는 식이에요.

| | $\alpha = 0.2$ | $\alpha = 0.4$ | $\alpha = 0.6$ | $\alpha = 0.8$ |
| --- | --- | --- | --- | --- |
| $y_T$ | 0.2000 | 0.4000 | 0.6000 | 0.8000 |
| $y_{T-1}$ | 0.1600 | 0.2400 | 0.2400 | 0.1600 |
| $y_{T-2}$ | 0.1280 | 0.1440 | 0.0960 | 0.0320 |
| $y_{T-3}$ | 0.1024 | 0.0864 | 0.0384 | 0.0064 |
| $y_{T-4}$ | 0.0819 | 0.0518 | 0.0154 | 0.0013 |
| $y_{T-5}$ | 0.0655 | 0.0311 | 0.0061 | 0.0003 |

$\alpha$가 클수록 가중치가 빨리 줄어 최근 값에 쏠립니다. $\alpha = 0.8$이면 다섯 시점 전 값의 가중치가 0.0003밖에 안 되죠.

가중치를 모두 더하면 1이라서 이름 그대로 가중평균입니다. 관측값이 $T$개뿐이면 합이 유한해지고 남는 몫은 시작값 $\ell_0$이 받습니다(바로 다음 절에서 나와요). 등비수열의 합으로 확인하면 이렇습니다.

$$
\sum_{j=0}^{T-1}\alpha(1-\alpha)^j + (1-\alpha)^T = \big[1 - (1-\alpha)^T\big] + (1-\alpha)^T = 1
$$

딥러닝을 공부했다면 낯익은 식일 거예요. Adam의 1차 모멘트 $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$나 BatchNorm의 running mean이 같은 점화식이거든요. Adam의 기본값 $\beta_1 = 0.9$는 $\alpha = 0.1$인 SES와 같습니다.

### 같은 식, 세 가지 모양

위 합을 점화식으로 쓰면 다음 예측은 최근 관측값과 직전 예측의 가중평균이 됩니다. 이 꼴을 weighted average form이라고 부릅니다.

$$
\hat{y}_{t+1|t} = \alpha y_t + (1-\alpha)\hat{y}_{t|t-1}
$$

표본 안의 각 시점에서 이렇게 낸 $\hat{y}_{t|t-1}$을 fitted value라고 하고 $e_t = y_t - \hat{y}_{t|t-1}$이 잔차입니다. 이미 관측한 값을 모형이 한 걸음 앞서 얼마나 맞혔는지 보는 셈이죠.

점화식에는 시작점이 필요합니다. $\hat{y}_{1|0}$은 정의되지 않으므로 이 자리를 $\ell_0$이라는 값으로 둡니다. 한 줄씩 대입해 보겠습니다.

$$
\begin{aligned}
\hat{y}_{2|1} &= \alpha y_1 + (1-\alpha)\ell_0 \\
\hat{y}_{3|2} &= \alpha y_2 + (1-\alpha)\hat{y}_{2|1} = \alpha y_2 + \alpha(1-\alpha) y_1 + (1-\alpha)^2 \ell_0 \\
\hat{y}_{4|3} &= \alpha y_3 + \alpha(1-\alpha) y_2 + \alpha(1-\alpha)^2 y_1 + (1-\alpha)^3 \ell_0
\end{aligned}
$$

같은 규칙을 $T$까지 이어 가면 앞의 가중합이 나옵니다.

$$
\hat{y}_{T+1|T} = \sum_{j=0}^{T-1} \alpha(1-\alpha)^j y_{T-j} + (1-\alpha)^T \ell_0
$$

$T$가 크면 $\ell_0$의 가중치 $(1-\alpha)^T$는 거의 0이 됩니다.

세 번째 모양이 component form입니다. 예측식과 평활식을 나눠 씁니다.

$$
\text{Forecast: } \hat{y}_{t+h|t} = \ell_t, \qquad \text{Smoothing: } \ell_t = \alpha y_t + (1-\alpha)\ell_{t-1}
$$

$\ell_t$ 자리에 $\hat{y}_{t+1|t}$를, $\ell_{t-1}$ 자리에 $\hat{y}_{t|t-1}$을 넣으면 weighted average form으로 돌아갑니다. 이름만 바꾼 것 같지만 성분을 따로 적어 두는 이 꼴이 Holt와 Holt-Winters로 넓히는 발판이 돼요.

### 예측은 왜 평평할까

Component form의 예측식을 보면 $h$가 얼마든 예측이 $\ell_t$ 하나입니다. 2걸음 앞도 10걸음 앞도 같은 값이죠. 아직 관측하지 않은 $y_{T+1}$ 자리에 그 예측값 $\ell_T$를 넣어 보면 이유가 보입니다.

$$
\begin{aligned}
\hat{y}_{T+1|T} &= \ell_T \\
\hat{y}_{T+2|T} &= \hat{\ell}_{T+1} = \alpha\,\hat{y}_{T+1|T} + (1-\alpha)\ell_T = \alpha\ell_T + (1-\alpha)\ell_T = \ell_T
\end{aligned}
$$

새로 알게 된 정보가 없으니 수준이 그대로입니다. 시험 점수로 말하면 실력이 앞으로 오를지 내릴지 모르니 다음 시험도 그다음 시험도 지금 실력으로 예상하는 거예요.

같은 시점을 예측해도 언제 예측했느냐에 따라 값은 달라집니다. $\hat{y}_{t+2|t} \ne \hat{y}_{t+2|t+1}$이에요. 앞은 $\ell_t$이고 뒤는 $y_{t+1}$을 보고 고친 $\ell_{t+1}$이니까요.

### α와 ℓ₀는 어떻게 고르나

한 걸음 예측오차의 제곱합(SSE)을 최소로 하는 값을 고릅니다.

$$
\text{SSE} = \sum_{t=1}^{T}\big(y_t - \hat{y}_{t|t-1}\big)^2 = \sum_{t=1}^{T} e_t^2
$$

$\hat{y}_{t|t-1}$ 안에 $\alpha(1-\alpha)^j$가 들어 있어서 SSE는 $\alpha$의 고차 다항식입니다. 회귀처럼 공식 한 줄로 풀리지 않으니 수치 최적화로 풀어요.

알제리 자료($t = 1, \dots, 58$)에서는 $\hat\alpha = 0.84$, $\hat\ell_0 = 39.54$가 나옵니다.

| 연도 | $t$ | $y_t$ | $\ell_t$ | $\hat{y}_{t\mid t-1}$ |
| --- | --- | --- | --- | --- |
| 1959 | 0 | | 39.54 | |
| 1960 | 1 | 39.04 | 39.12 | 39.54 |
| 1961 | 2 | 46.24 | 45.10 | 39.12 |
| 1962 | 3 | 19.79 | 23.84 | 45.10 |
| 1963 | 4 | 24.68 | 24.55 | 23.84 |
| ⋮ | | | | |
| 2017 | 58 | 22.64 | 22.44 | 21.43 |

첫 줄을 직접 계산해 보면 이렇습니다.

$$
\ell_1 = 0.84 \times 39.04 + 0.16 \times 39.54 = 39.12
$$

이 값이 그대로 1961년의 예측이 됩니다. 수준 열의 값이 한 줄 아래 예측 열로 내려가는 모습이 $\hat{y}_{t+1|t} = \ell_t$ 그 자체예요. 앞 네 줄은 다시 계산해도 39.12, 45.10, 23.84, 24.55로 같았습니다. 그 뒤로는 $\alpha$를 0.84로 반올림한 탓에 소수 둘째 자리가 0.01씩 어긋나고요. 2018~2022년 예측은 모두 $\ell_{58} = 22.44$로 평평합니다.

$\hat\alpha = 0.84$는 꽤 큰 값이라 수준이 최근 관측값을 빠르게 따라간다는 뜻입니다. 반대로 $\alpha = 0$으로 두면 모든 예측이 $\ell_0$ 하나가 되고 SSE를 최소로 하는 $\ell_0$은 표본 평균입니다. $\alpha = 1$이면 naive고요. 앞에서 본 두 극단이 $\alpha$의 양 끝이고 SES는 그 사이를 자료로 고르는 방법입니다.

## 추세를 따라가는 Holt 방법

Holt(1957)는 SES에 기울기를 더해 추세가 있는 자료로 넓혔습니다.

$$
\begin{aligned}
\text{Forecast: } & \hat{y}_{t+h|t} = \ell_t + h b_t \\
\text{Level: } & \ell_t = \alpha y_t + (1-\alpha)(\ell_{t-1} + b_{t-1}) \\
\text{Trend: } & b_t = \beta^*(\ell_t - \ell_{t-1}) + (1-\beta^*) b_{t-1}
\end{aligned}
$$

두 평활식을 하나씩 읽어 보겠습니다.

- Level 식의 $\ell_{t-1} + b_{t-1}$은 직전 시점에서 낸 예측 $\hat{y}_{t|t-1}$입니다. 그래서 $\ell_t = \alpha y_t + (1-\alpha)\hat{y}_{t|t-1}$로 SES와 같은 꼴이에요.
- Trend 식은 이번에 실제로 오른 양 $\ell_t - \ell_{t-1}$과 원래 알던 기울기 $b_{t-1}$의 가중평균입니다.

계산 순서도 정해져 있습니다. $y_t$를 관측한 뒤 $\ell_t$를 먼저 구하고 그 $\ell_t$로 $b_t$를 구해요. 예측식에 $h = 0$을 넣으면 $\ell_t$가 남으니 수준은 시점 $t$에서 그은 직선의 높이이고 $b_t$는 그 직선의 기울기입니다. $\beta^*$에 별표가 붙은 것은 뒤의 상태공간 꼴에서 $\beta = \alpha\beta^*$라는 다른 모수가 나오기 때문입니다.

### 미래의 수준과 기울기

예측식이 왜 $h$의 1차식인지는 SES의 평평한 예측과 같은 방법으로 확인할 수 있습니다. 모르는 $y_{T+1}$ 자리에 예측값 $\ell_T + b_T$를 넣어 보죠.

$$
\begin{aligned}
\hat{\ell}_{T+1} &= \alpha(\ell_T + b_T) + (1-\alpha)(\ell_T + b_T) = \ell_T + b_T \\
\hat{b}_{T+1} &= \beta^*(\hat{\ell}_{T+1} - \ell_T) + (1-\beta^*) b_T = \beta^* b_T + (1-\beta^*) b_T = b_T \\
\hat{y}_{T+2|T} &= \hat{\ell}_{T+1} + \hat{b}_{T+1} = \ell_T + 2b_T
\end{aligned}
$$

Level 식은 같은 값 두 개의 가중평균이 되고 Trend 식의 "실제로 오른 양"도 $b_T$가 됩니다. 그러니 둘 다 그대로 이어지죠. 같은 일을 $h$번 되풀이하면 다음을 얻습니다.

$$
\hat{\ell}_{T+h} = \ell_T + h b_T, \qquad \hat{b}_{T+h} = b_T
$$

기울기는 그대로 두고 수준만 매 걸음 기울기만큼 올리는 셈이에요. 여기서 $\ell_T, b_T$는 $T$까지의 자료로 구한 추정값이고 $\hat{\ell}_{T+h}, \hat{b}_{T+h}$는 $T$ 시점에서 본 미래 상태의 예측값입니다. 예측식 $\hat{y}_{T+h|T} = \ell_T + hb_T$가 $\hat{\ell}_{T+h}$와 같은 값이라는 것도 보이고요.

### 호주 인구에서 α가 거의 1인 이유

FPP3의 호주 인구 예제에서는 $\hat\alpha = 0.9999$, $\hat\beta^* = 0.3267$이 나옵니다. 인구는 추세가 강하고 잡음이 거의 없이 매끈하게 늘어나는 계열이라 수준이 관측값을 거의 그대로 따라갑니다. $\hat\beta^*$도 작지 않아서 기울기가 자주 고쳐지고요. 시작값은 $\ell_0 = 10.05$, $b_0 = 0.22$입니다.

| 연도 | $t$ | $y_t$ | $\ell_t$ | $b_t$ | $\hat{y}_{t\mid t-1}$ |
| --- | --- | --- | --- | --- | --- |
| 1959 | 0 | | 10.05 | 0.22 | |
| 1960 | 1 | 10.28 | 10.28 | 0.22 | 10.28 |
| 1961 | 2 | 10.48 | 10.48 | 0.22 | 10.50 |
| 1965 | 6 | 11.39 | 11.39 | 0.22 | 11.39 |
| 1966 | 7 | 11.65 | 11.65 | 0.23 | 11.61 |
| ⋮ | | | | | |
| 2017 | 58 | 24.60 | 24.60 | 0.37 | 24.57 |

수준 열이 관측값 열과 똑같습니다. 1960년 예측은 $\ell_0 + b_0$인데 $10.05 + 0.22 = 10.27$이 아니라 10.28로 찍힌 것은 반올림하기 전 값으로 계산했기 때문이에요. 1966년 예측 11.61도 $\ell_{1965} + b_{1965} = 11.39 + 0.22$입니다. 기울기 한 줄도 손으로 따라가 보면 표와 맞습니다.

$$
b_1 = 0.3267 \times (10.28 - 10.05) + 0.6733 \times 0.22 = 0.223
$$

### 끝없이 오르는 직선을 꺾는 φ

Holt의 예측은 같은 기울기로 끝없이 오르거나 내려가서 먼 미래를 과대예측하기 쉽습니다. Gardner와 McKenzie(1985)는 추세를 언젠가 평평하게 꺾는 감쇠 모수 $\phi$를 넣었어요.

$$
\begin{aligned}
\text{Forecast: } & \hat{y}_{t+h|t} = \ell_t + (\phi + \phi^2 + \cdots + \phi^h) b_t \\
\text{Level: } & \ell_t = \alpha y_t + (1-\alpha)(\ell_{t-1} + \phi b_{t-1}) \\
\text{Trend: } & b_t = \beta^*(\ell_t - \ell_{t-1}) + (1-\beta^*)\phi b_{t-1}
\end{aligned}
\qquad 0 < \phi < 1
$$

$\phi = 1$이면 Holt와 같습니다. 앞에서처럼 $y_{T+1}$ 자리에 예측값 $\ell_T + \phi b_T$를 넣으면 $\hat{b}_{T+1} = \beta^*\phi b_T + (1-\beta^*)\phi b_T = \phi b_T$가 돼요. 기울기가 걸음마다 $\phi$배로 줄고 그 기울기를 차례로 더한 것이 예측식의 $\phi + \phi^2 + \cdots + \phi^h$입니다. 등비급수라서 $h \to \infty$이면 예측이 한 값으로 모입니다.

$$
\hat{y}_{T+h|T} \;\to\; \ell_T + \frac{\phi}{1-\phi}\, b_T
$$

$\phi = 0.9$면 $\phi/(1-\phi) = 9$라서 아무리 멀리 가도 기울기를 $9b_T$까지만 더합니다. 가까운 미래는 추세를 따르고 먼 미래는 평평해지는 거죠. FPP3는 $\phi$가 0.8보다 작으면 감쇠가 너무 세고 1에 가까우면 감쇠 없는 모형과 구분이 안 되어 보통 0.8~0.98로 둔다고 설명합니다. 여러 계열을 자동으로 예측할 때 가장 널리 쓰이는 방법 가운데 하나이기도 해요.

## 계절까지 다루는 Holt-Winters

Holt-Winters는 Holt에 계절 성분 $s_t$를 하나 더 얹습니다. $m$은 계절 주기(분기 자료면 4, 월별 자료면 12)이고 $k$는 $(h-1)/m$의 정수 부분입니다.

$$
\begin{aligned}
\text{Forecast: } & \hat{y}_{t+h|t} = \ell_t + h b_t + s_{t+h-m(k+1)} \\
\text{Level: } & \ell_t = \alpha(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1}) \\
\text{Trend: } & b_t = \beta^*(\ell_t - \ell_{t-1}) + (1-\beta^*) b_{t-1} \\
\text{Season: } & s_t = \gamma(y_t - \ell_{t-1} - b_{t-1}) + (1-\gamma) s_{t-m}
\end{aligned}
$$

Level 식이 평활하는 값은 $y_t$가 아니라 작년 같은 철의 계절 값을 뺀 $y_t - s_{t-m}$입니다. 앞에서 말한 "계절 효과까지 뺀 높이"가 이 식이에요. Season 식은 반대로 관측값에서 예상 높이 $\ell_{t-1} + b_{t-1}$을 뺀 나머지를 계절 성분의 새 정보로 씁니다. 여름이라 더 팔린 몫은 $s_t$가 가져갑니다. 그 몫을 빼고도 예상보다 높을 때만 수준이 올라요.

FPP3는 Season 식을 $s_t = \gamma^*(y_t - \ell_t) + (1-\gamma^*) s_{t-m}$으로 쓰기도 합니다. 여기에 Level 식의 $\ell_t$를 대입하면 위 식이 되고 두 모수 사이에는 $\gamma = \gamma^*(1-\alpha)$라는 관계가 있습니다.

### 왜 t − 10이 작년의 계절일까

예측식의 $s_{t+h-m(k+1)}$은 처음 보면 낯섭니다. 월별 자료($m = 12$)에서 26개월 뒤를 예측한다고 해 보겠습니다. $k = \lfloor 25/12 \rfloor = 2$이므로 계절 첨자는 이렇게 됩니다.

$$
t + h - m(k+1) = t + 26 - 36 = t - 10
$$

$t+26$과 $t-10$은 정확히 36개월 차이라 같은 달이고 $t-10$은 이미 관측한 마지막 1년 안에 있습니다. 예측하려는 달과 같은 달 가운데 가장 최근에 추정한 계절 값, 말하자면 "작년의 계절성"을 쓰는 거예요. $h \le m$이면 $k = 0$이라 그냥 $s_{t+h-m}$입니다.

왜 작년 값을 그대로 쓰는지도 Holt에서처럼 확인할 수 있습니다. $y_{T+1}$ 자리에 예측값 $\ell_T + b_T + s_{T+1-m}$을 넣으면 Season 식의 $y_{T+1} - \ell_T - b_T$가 $s_{T+1-m}$이 됩니다.

$$
\hat{s}_{T+1} = \gamma\, s_{T+1-m} + (1-\gamma)\, s_{T+1-m} = s_{T+1-m}
$$

미래의 계절 값은 작년 같은 철의 값을 그대로 이어받아 $m$ 주기로 되풀이됩니다. 수준과 기울기는 Holt와 똑같이 $\ell_T + hb_T$, $b_T$로 이어지고요.

### 계절 변동이 커지면 곱셈으로

계절 성분을 빼는 대신 나누는 방식도 있습니다.

$$
\begin{aligned}
\text{Forecast: } & \hat{y}_{t+h|t} = (\ell_t + h b_t)\, s_{t+h-m(k+1)} \\
\text{Level: } & \ell_t = \alpha\frac{y_t}{s_{t-m}} + (1-\alpha)(\ell_{t-1} + b_{t-1}) \\
\text{Trend: } & b_t = \beta^*(\ell_t - \ell_{t-1}) + (1-\beta^*) b_{t-1} \\
\text{Season: } & s_t = \gamma\frac{y_t}{\ell_{t-1} + b_{t-1}} + (1-\gamma) s_{t-m}
\end{aligned}
$$

계절 변동의 폭이 수준에 비례해 커지는 계열에 씁니다. [1편](/posts/2026-10-15-time-series-1-forecasting-basics)의 additive와 multiplicative 분해를 고르던 기준과 같아요. FPP3의 호주 국내 숙박 여행(분기별) 예제에 두 방법을 모두 써 보면 예측이 꽤 비슷하게 나옵니다. 성분을 그려 보면 덧셈 쪽 계절 성분은 0 주위에서 움직이고 곱셈 쪽은 1 주위에서 움직입니다. 2편 classical decomposition의 덧셈 분해와 곱셈 분해 그림에서 보던 차이와 같죠.

감쇠 추세와 곱셈 계절을 합친 Holt-Winters damped도 있습니다. 예측식은 $\hat{y}_{t+h|t} = \big[\ell_t + (\phi + \cdots + \phi^h) b_t\big]\, s_{t+h-m(k+1)}$이고 평활식은 앞의 두 방법에서 $b_{t-1}$을 $\phi b_{t-1}$로 바꾼 꼴입니다.

## 아홉 가지 조합

추세를 없음(N), 덧셈(A), 감쇠 덧셈($\text{A}_d$) 셋 중에서 고르고 계절을 없음(N), 덧셈(A), 곱셈(M) 셋 중에서 고르면 아홉 가지 방법이 나옵니다. 약칭의 앞 글자가 추세, 뒤 글자가 계절이에요.

| 약칭 | 방법 |
| --- | --- |
| (N,N) | Simple exponential smoothing |
| (A,N) | Holt's linear method |
| ($\text{A}_d$,N) | Additive damped trend method |
| (A,A) | Additive Holt-Winters' method |
| (A,M) | Multiplicative Holt-Winters' method |
| ($\text{A}_d$,M) | Holt-Winters' damped method |

## 방법에서 모형으로

지금까지 본 것은 점예측을 내는 계산 규칙(method)이었습니다. 예측구간을 내려면 오차가 어떤 분포를 따르는지까지 정한 통계 모형(model)이 필요해요. FPP3는 이것을 innovation state space model이라고 부르고 ETS(Error, Trend, Seasonal) 세 글자로 이름을 붙입니다. 방법의 약칭 앞에 오차 글자 하나가 더 붙는 거예요. 오차가 더해지면 A, 곱해지면 M입니다. 덧셈 Holt-Winters를 덧셈 오차로 쓰면 ETS(A,A,A), 곱셈 Holt-Winters를 곱셈 오차로 쓰면 ETS(M,A,M)이 되죠.

### 오차가 더해지는 SES

SES의 평활식을 조금 정리하면 "오차 보정(error correction)" 꼴이 나옵니다.

$$
\ell_t = \alpha y_t + (1-\alpha)\ell_{t-1} = \ell_{t-1} + \alpha(y_t - \ell_{t-1}) = \ell_{t-1} + \alpha e_t, \qquad e_t = y_t - \hat{y}_{t|t-1}
$$

수준은 직전 수준에 이번 예측오차의 $\alpha$배만큼 보정한 값입니다. 이제 이 오차 $e_t$를 확률변수 $\varepsilon_t \sim N(0, \sigma^2)$로 봅니다. 평활식에서는 상태 방정식이 나오고 $e_t = y_t - \ell_{t-1}$을 $y_t$로 정리하면 관측 방정식이 나와요. 이 모형이 ETS(A,N,N)입니다.

$$
\begin{aligned}
y_t &= \ell_{t-1} + \varepsilon_t && (\text{관측 방정식}) \\
\ell_t &= \ell_{t-1} + \alpha\varepsilon_t && (\text{상태 방정식})
\end{aligned}
\qquad \varepsilon_t \sim N(0, \sigma^2)
$$

각 관측값은 직전 수준에 오차 하나를 더한 값입니다. 앞에서 수준을 "흔들림을 걷어 낸 높이"라고 했던 말을 식으로 쓴 것이기도 해요.

### α의 두 끝은 상수와 random walk다

$\alpha$의 양 끝을 넣어 보면 모형의 성격이 바로 보입니다.

- $\alpha = 0$이면 $\ell_t = \ell_{t-1}$이라 수준이 상수 $k$입니다. 모형은 $y_t = k + \varepsilon_t$가 되고 일정한 값 주위에서 흔들리기만 하죠.
- $\alpha = 1$이면 $\ell_t = \ell_{t-1} + \varepsilon_t = y_t$라서 $y_t = y_{t-1} + \varepsilon_t$, random walk입니다. 같은 규칙으로 만들어도 경로가 제각각 멀리 떠돌아다니는 계열이에요.

### 오차가 곱해지는 SES

오차를 예측값으로 나눈 상대 오차로 둘 수도 있습니다.

$$
\varepsilon_t = \frac{y_t - \hat{y}_{t|t-1}}{\hat{y}_{t|t-1}} = \frac{y_t - \ell_{t-1}}{\ell_{t-1}}, \qquad \varepsilon_t \sim N(0, \sigma^2)
$$

그러면 $e_t = y_t - \ell_{t-1} = \ell_{t-1}\varepsilon_t$이고 이것을 평활식에 넣으면 ETS(M,N,N)이 됩니다.

$$
\begin{aligned}
y_t &= \ell_{t-1}(1 + \varepsilon_t) && (\text{관측 방정식}) \\
\ell_t &= \ell_{t-1} + \alpha\ell_{t-1}\varepsilon_t = \ell_{t-1}(1 + \alpha\varepsilon_t) && (\text{상태 방정식})
\end{aligned}
$$

ETS(A,N,N)과 ETS(M,N,N)은 평활식이 똑같이 $\ell_t = \alpha y_t + (1-\alpha)\ell_{t-1}$입니다. 모수가 같으면 점예측도 같아요. 오차를 어떻게 보느냐만 달라서 예측구간이 달라집니다.

### Holt를 상태공간 꼴로

Holt 방법도 같은 방식으로 모형이 됩니다. 한 걸음 예측오차가 $\varepsilon_t = y_t - (\ell_{t-1} + b_{t-1})$이니 관측 방정식은 바로 나와요. 이것을 Level 식에 넣습니다.

$$
\ell_t = \alpha(\ell_{t-1} + b_{t-1} + \varepsilon_t) + (1-\alpha)(\ell_{t-1} + b_{t-1}) = \ell_{t-1} + b_{t-1} + \alpha\varepsilon_t
$$

그러면 $\ell_t - \ell_{t-1} = b_{t-1} + \alpha\varepsilon_t$이고 이것을 Trend 식에 넣습니다.

$$
b_t = \beta^*(b_{t-1} + \alpha\varepsilon_t) + (1-\beta^*) b_{t-1} = b_{t-1} + \alpha\beta^*\varepsilon_t
$$

$\alpha\beta^*$를 $\beta$로 쓰면 ETS(A,A,N)이 완성됩니다.

$$
\begin{aligned}
y_t &= \ell_{t-1} + b_{t-1} + \varepsilon_t \\
\ell_t &= \ell_{t-1} + b_{t-1} + \alpha\varepsilon_t \\
b_t &= b_{t-1} + \beta\varepsilon_t
\end{aligned}
\qquad \beta = \alpha\beta^*
$$

이 상태 방정식으로 보면 앞의 "미래의 수준과 기울기"도 한 줄로 설명됩니다. 미래 오차를 기댓값 0으로 두면 보정 항이 사라지니 기울기는 그대로이고 수준은 매 걸음 기울기만큼 오르죠. 덧셈 오차 9개, 곱셈 오차 9개 모형 모두 상태 방정식이 "직전 상태 + 모수 × 오차" 꼴입니다. 곱셈 오차 쪽은 오차에 그 시점의 예측 수준이 곱해진다는 점만 달라요.

## 예측구간은 왜 점점 넓어질까

모형이 생기면 예측구간을 낼 수 있습니다.

$$
\hat{y}_{T+h|T} \pm c\,\sigma_h
$$

$c$는 80% 구간에서 1.28, 95% 구간에서 1.96입니다. $\sigma_h^2$는 $h$걸음 예측오차의 분산인데 모형마다 식이 복잡해요(Hyndman et al. 2008, 6장). 가장 단순한 ETS(A,N,N)은 직접 유도할 수 있습니다. 관측 방정식과 상태 방정식을 번갈아 대입해 $\ell_t$까지 내려가면 됩니다.

$$
\begin{aligned}
y_{t+h} &= \ell_{t+h-1} + \varepsilon_{t+h} \\
&= \ell_{t+h-2} + \alpha\varepsilon_{t+h-1} + \varepsilon_{t+h} \\
&\;\;\vdots \\
&= \ell_t + \alpha\varepsilon_{t+1} + \alpha\varepsilon_{t+2} + \cdots + \alpha\varepsilon_{t+h-1} + \varepsilon_{t+h}
\end{aligned}
$$

시점 $t$에서 $\ell_t$는 이미 아는 값이고 오차들은 서로 독립입니다. 독립인 확률변수의 합의 분산은 분산의 합이고 $\operatorname{Var}(aX) = a^2\operatorname{Var}(X)$이니 바로 계산됩니다.

$$
\sigma_h^2 = \operatorname{Var}(y_{t+h} \mid t) = \underbrace{\alpha^2\sigma^2 + \cdots + \alpha^2\sigma^2}_{h-1\text{개}} + \sigma^2 = \sigma^2\big[1 + \alpha^2(h-1)\big]
$$

멀리 볼수록 아직 일어나지 않은 충격이 수준에 쌓이니 구간이 넓어집니다. 양 끝을 보면 앞 글들과 이어져요.

- $\alpha = 0$이면 $\sigma_h^2 = \sigma^2$로 $h$와 상관없이 일정합니다. 2편의 결정적 추세 모형에서 예측구간 폭이 일정했던 것과 같습니다.
- $\alpha = 1$(random walk)이면 $\sigma_h^2 = h\sigma^2$로 $h$에 비례해 커집니다. 1편에서 AR 모형 $y_t = ay_{t-1} + b + \varepsilon_t$의 예측 분산을 구할 때 $a = 1$이면 끝없이 커지던 바로 그 경우예요.

## 모수는 무엇으로 추정할까

SES는 SSE로 $\alpha$와 $\ell_0$을 고른다고 했습니다. Holt와 Holt-Winters도 원리는 같아요. 평활 모수와 초기 상태를 한꺼번에 움직이며 한 걸음 예측오차의 SSE를 최소로 합니다. FPP3도 호주 인구 예의 $\alpha, \beta^*, \ell_0, b_0$를 이렇게 구했다고 씁니다.

$$
\text{SSE}(\boldsymbol\theta) = \sum_{t=1}^{T}\big(y_t - \hat{y}_{t|t-1}(\boldsymbol\theta)\big)^2
$$

$\boldsymbol\theta$에 들어가는 값은 방법마다 늘어납니다. 초기 상태도 추정 대상이라는 점이 중요해요.

| 방법 | 평활 모수 | 초기 상태 | 자유로운 값의 개수 |
| --- | --- | --- | --- |
| SES | $\alpha$ | $\ell_0$ | 2 |
| Holt | $\alpha, \beta^*$ | $\ell_0, b_0$ | 4 |
| Damped | $\alpha, \beta^*, \phi$ | $\ell_0, b_0$ | 5 |
| Holt-Winters | $\alpha, \beta^*, \gamma$ | $\ell_0, b_0, s_0, s_{-1}, \dots, s_{1-m}$ | $m + 4$ (분기면 8) |

### 초기 계절 값의 합을 0으로 묶는 이유

Holt-Winters의 초기 계절 값은 $m$개인데 표에서는 $m-1$개로 셌습니다. 보통 합을 0으로 묶기 때문이에요(곱셈 계절이면 합을 $m$으로). 묶지 않으면 문제가 생깁니다. $\ell_0$에 $c$를 더하고 초기 계절 값 $m$개에서 모두 $c$를 빼 볼게요.

- 첫 예측 $\ell_0 + b_0 + s_{1-m}$은 그대로입니다.
- Level 식에서 $\ell_1$은 $\alpha c + (1-\alpha)c = c$만큼 오릅니다.
- Season 식에서 $s_1$은 $-\gamma c - (1-\gamma)c = -c$만큼 내려갑니다.
- $\ell_1 - \ell_0$이 그대로라 $b_1$도 그대로입니다.

그러면 다음 예측도 같고 이 일이 끝까지 이어집니다. SSE가 같은 점이 끝없이 많아서 최소점이 하나로 정해지지 않아요. 분기 자료로 직접 돌려 보니 $c = 7$을 옮겨도 한 걸음 예측의 차이가 최대 $4 \times 10^{-14}$였습니다. 2편 회귀에서 dummy 변수를 모두 넣으면 계수가 정해지지 않던 dummy trap, classical decomposition에서 계절 값의 합을 0으로 맞추던 조정과 같은 문제입니다.

모수의 범위도 정해 둡니다. 평활식을 가중평균으로 읽으려고 $0 < \alpha, \beta^*, \gamma^*, \phi < 1$로 둡니다. 상태공간 꼴의 $\beta = \alpha\beta^*$, $\gamma = (1-\alpha)\gamma^*$로 옮기면 $0 < \alpha < 1$, $0 < \beta < \alpha$, $0 < \gamma < 1 - \alpha$가 됩니다. FPP3의 fable 패키지는 계산이 불안정해지지 않도록 $\phi$를 $0.8 < \phi < 0.98$로 더 좁혀요.

### 모형이라면 우도로

ETS 모형으로 보면 우도를 최대로 하는 값을 고릅니다. 이것이 SSE와 같은 답을 주는지는 오차 종류에 따라 갈립니다.

오차가 더해지는 모형은 $\varepsilon_t = y_t - \hat{y}_{t|t-1} \sim N(0, \sigma^2)$입니다.

$$
\log L = -\frac{T}{2}\log(2\pi\sigma^2) - \frac{\text{SSE}}{2\sigma^2}
$$

$\sigma^2$로 미분해 0으로 두면 $\hat\sigma^2 = \text{SSE}/T$이고 이를 다시 넣으면 $\log L = -\frac{T}{2}\log\text{SSE} + \text{상수}$입니다. SSE가 작을수록 우도가 크니 SSE 최소화와 답이 같아요.

오차가 곱해지는 모형은 $y_t = \mu_t(1 + \varepsilon_t)$, $\mu_t = \hat{y}_{t|t-1}$이라 $y_t$의 조건부 분포가 $N(\mu_t, \mu_t^2\sigma^2)$입니다. 분산이 예측값에 따라 달라지죠.

$$
\log L = -\frac{T}{2}\log(2\pi\sigma^2) - \sum_{t=1}^{T}\log\lvert\mu_t\rvert - \frac{1}{2\sigma^2}\sum_{t=1}^{T}\varepsilon_t^2, \qquad \varepsilon_t = \frac{y_t - \mu_t}{\mu_t}
$$

같은 방법으로 $\sigma^2$를 없애면 다음을 최소로 하는 문제가 됩니다.

$$
T\log\sum_{t=1}^{T} \varepsilon_t^2 + 2\sum_{t=1}^{T} \log\lvert\mu_t\rvert
$$

오차의 제곱합이 아니라 상대 오차의 제곱합이고 $\log\lvert\mu_t\rvert$ 항까지 붙으니 SSE와는 답이 다릅니다. 예측값을 키우면 상대 오차는 작아 보이지만 분산 $\mu_t^2\sigma^2$도 함께 커지거든요. 이 항이 그만큼 벌점을 줍니다.

우도가 필요한 이유가 하나 더 있습니다. 여러 ETS 모형 가운데 하나를 고를 때 FPP3는 2편에서 본 AIC를 쓰는데 AIC는 우도로 계산해요. 덧셈 오차 모형의 SSE와 곱셈 오차 모형의 상대 오차 제곱합은 단위가 달라 서로 비교할 수 없습니다.

## 베이즈 업데이트로 다시 읽기

관측값이 들어올 때마다 상태를 고친다는 설명을 듣다 보면 이런 생각이 듭니다. 수준을 모르는 모수로 보고 한 스텝마다 베이즈 업데이트를 하는 것과 같지 않을까요? 결론부터 말하면 그렇게 볼 수 있습니다. 다만 고정된 모수가 아니라 매 시점 조금씩 움직이는 모수로 봐야 SES와 같은 식이 나와요.

잡음이 두 개인 local level model로 따져 보겠습니다. $\mu_t$가 수준입니다.

$$
y_t = \mu_t + v_t,\ \ v_t \sim N(0, V) \qquad\qquad \mu_t = \mu_{t-1} + w_t,\ \ w_t \sim N(0, W)
$$

$V$는 관측 잡음의 분산이고 $W$는 수준이 한 시점에 움직이는 정도입니다. 시점 $t-1$까지 본 뒤 수준의 사후분포가 $N(m_{t-1}, C_{t-1})$이라고 하면 한 스텝은 이렇게 돕니다.

1. 사전분포: 수준이 $w_t$만큼 움직이므로 $\mu_t \mid y_{1:t-1} \sim N(m_{t-1}, R_t)$, $R_t = C_{t-1} + W$입니다.
2. 가능도: $y_t \mid \mu_t \sim N(\mu_t, V)$입니다.
3. 사후분포: 정규-정규 켤레라 사후분포도 정규분포입니다.

$$
m_t = K_t\, y_t + (1 - K_t)\, m_{t-1}, \qquad K_t = \frac{R_t}{R_t + V}, \qquad C_t = K_t V
$$

사후 평균은 사전 평균 $m_{t-1}$과 새 관측값 $y_t$의 가중평균이고 가중치는 각자의 정밀도(분산의 역수)에 비례합니다. SES 평활식의 $\alpha$ 자리에 $K_t$가 들어간 꼴이죠. 수준을 잘 모를수록($R_t$가 클수록) 새 관측값을 많이 믿고 관측값이 시끄러울수록($V$가 클수록) 덜 믿습니다.

$K_t$는 처음 몇 스텝만 바뀌고 곧 상수로 수렴합니다. 수렴하면 $R = KV + W$이고 $K = R/(R+V)$에서 $R = KV/(1-K)$입니다. 두 식을 같게 놓고 $q = W/V$로 두면 $K^2 + qK - q = 0$이 나오고 그 양의 근이 SES의 $\alpha$예요.

$$
\alpha = \frac{-q + \sqrt{q^2 + 4q}}{2}, \qquad q = \frac{\alpha^2}{1-\alpha}
$$

| $q = W/V$ | 0.01 | 0.1 | 1 | 10 | 100 |
| --- | --- | --- | --- | --- | --- |
| $\alpha$ | 0.095 | 0.27 | 0.618 | 0.916 | 0.99 |

$q = 1$로 30개를 시뮬레이션해 보니 $K_t$가 1, 0.667, 0.625, 0.619, 0.618로 금방 수렴했고 $t = 10$부터 $\alpha = 0.618$인 SES를 돌린 값과 사후 평균의 차이는 최대 $5 \times 10^{-9}$였습니다. $\alpha$는 결국 "수준이 움직이는 정도 대 관측 잡음"의 비율로 정해지는 셈입니다. 호주 인구에서 $\hat\alpha$가 거의 1이었던 것도 이 관점으로 보면 관측 잡음 $V$가 아주 작은 계열이라는 뜻이에요.

두 극단도 여기서 나옵니다.

- $W = 0$(수준이 정말 고정된 모수): 사전 분산이 계속 줄어서 처음 사전분포가 아주 넓으면 $K_t = 1/t$가 되고 사후 평균은 표본 평균입니다. average 예측이죠.
- $V = 0$(관측 잡음이 없음): $K = 1$이라 naive입니다.

고정된 모수에 베이즈 업데이트를 하면 가중치가 $1/t$로 줄어 결국 average가 됩니다. SES의 $\alpha$가 0으로 줄지 않는 것은 수준이 매 시점 $W$만큼 움직인다고 보기 때문이에요. 사전 분산이 $W$ 아래로 내려가지 않으니 옛 관측값을 계속 잊어 갑니다.

주의할 점이 하나 있습니다. ETS(A,N,N) 자체는 이 모형과 다릅니다. ETS는 관측식과 상태식에 같은 $\varepsilon_t$ 하나가 들어가서 $\ell_0$와 $\alpha$가 정해지면 $\ell_t$가 자료로 정확히 계산돼요. $\ell_t$의 사후분포라는 것이 따로 없습니다. 베이지안 해석은 잡음이 두 개인 위 모형의 것이고 수렴한 뒤에 두 모형의 점예측과 예측 분산이 같아지는 거예요. ETS의 $\sigma^2$를 한 걸음 예측오차의 분산 $R + V$로 맞추면 $\alpha^2\sigma^2 = R^2/(R+V) = W$라서 $\sigma^2[1 + \alpha^2(h-1)] = R + V + (h-1)W$가 됩니다. 위 모형에서 직접 구한 $h$걸음 예측 분산과 같은 값이에요.

추세와 계절성이 있으면 상태가 벡터가 되고 같은 계산을 행렬로 합니다. 선형 정규 상태공간모형에서 상태의 사후분포를 한 스텝씩 구하는 이 계산이 Kalman filter이고 베이지안 쪽에서는 West와 Harrison의 동적선형모형(dynamic linear model)이 같은 틀입니다.

## 2편의 회귀와 무엇이 다를까

2편처럼 분해하고 회귀해서 예측할 수도 있고 이번 글처럼 지수평활로 예측할 수도 있습니다. 둘 다 추세와 계절성을 나눠 다루지만 성분을 보는 관점이 다릅니다.

| | 2편 (회귀) | 3편 (ETS) |
| --- | --- | --- |
| 성분 | 시간의 고정된 함수 ($\beta_0 + \beta_1 t$, dummy) | 시점마다 바뀌는 상태 ($\ell_t, b_t, s_t$) |
| 지금의 성분을 정하는 관측값 비중 | 자료 전체가 같음 (OLS) | 최근일수록 큼 ($\alpha(1-\alpha)^j$) |
| 예측 | 미래 시점 $t = T+h$를 식에 대입 | 마지막 상태에서 앞으로 늘림 |
| 예측 분산 | $h$와 상관없이 $\sigma^2$ ($\hat\beta$ 추정 오차 무시) | $h$가 커질수록 커짐 |
| 잘 맞는 자료 | 패턴이 오래 그대로인 계열 | 패턴이 서서히 바뀌는 계열 |

2편의 이동평균, classical decomposition, STL은 그 자체로는 예측 방법이 아니었습니다. 성분을 나눠 보고 계절 조정을 하는 도구였죠. 이동평균은 양 끝 몇 시점에서 값이 비고 미래로 늘릴 식도 없으니까요. 예측식이 나온 곳은 시간과 dummy 변수에 회귀한 부분이었습니다.

두 방법은 따로 떨어진 것도 아닙니다. ETS(A,A,A)에서 $\alpha = \beta = \gamma = 0$으로 두면 상태가 전혀 바뀌지 않거든요.

$$
\ell_t = \ell_0 + b_0 t, \quad b_t = b_0, \quad s_t = s_{t-m} \quad\Rightarrow\quad y_t = \ell_{t-1} + b_{t-1} + s_{t-m} + \varepsilon_t = \ell_0 + b_0 t + s_{t-m} + \varepsilon_t
$$

선형 추세에 해마다 같은 계절 값을 더한 식이라 2편의 맥주 생산량 회귀(선형 추세 + 분기 dummy)와 같은 모형입니다. 계절 값을 1분기 기준으로 적느냐 합을 0으로 맞추느냐만 다를 수 있어요. 2편의 회귀는 "성분은 절대 바뀌지 않는다"고 못 박은 ETS인 셈이고 ETS는 $\alpha, \beta, \gamma$를 추정해 성분이 얼마나 빨리 바뀌는지를 자료가 정하게 합니다. 추정값이 0에 가까우면 회귀와 거의 같은 예측이 나오고 크면 최근 값을 따라갑니다.

섞어 쓰는 방법도 있습니다. FPP3 5.7절은 STL로 계절 성분을 떼어 낸 뒤 계절 조정된 계열은 Holt 같은 비계절 방법으로, 계절 성분은 마지막 1년을 반복하는 seasonal naive로 예측해 더합니다. 2편에서 계절 조정된 자료에 추세 회귀를 했던 것과 같은 짜임이고 추세 회귀 자리에 Holt가 들어간 거예요. 어느 쪽이 나은지는 자료마다 다르니 뒤쪽 구간을 남겨 두고 예측 정확도를 비교해 고릅니다.

## ARIMA로 이어지는 다리

다음 편부터는 ARIMA를 다룹니다. 성분을 나누지 않고 $y_t$와 과거 값, 과거 충격의 관계를 직접 모형으로 쓰는 방법이에요. 그런데 지수평활과 아예 다른 세계는 아닙니다. ETS(A,N,N)의 관측값을 한 번 차분해 보면 이렇습니다.

$$
y_t - y_{t-1} = (\ell_{t-1} + \varepsilon_t) - (\ell_{t-2} + \varepsilon_{t-1}) = \alpha\varepsilon_{t-1} + \varepsilon_t - \varepsilon_{t-1} = \varepsilon_t - (1-\alpha)\varepsilon_{t-1}
$$

차분한 계열이 이번 충격과 직전 충격 하나로 설명되는 꼴입니다. 다음 편에서 다룰 MA(1) 모형이고 계수는 $\theta_1 = \alpha - 1$이에요. 정상성과 차분, AR과 MA를 정리한 뒤 이 연결을 ARIMA의 언어로 다시 보겠습니다.
