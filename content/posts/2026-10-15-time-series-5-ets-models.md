---
title: "시계열분석 (5): ETS 모형, 오차 분포를 붙여 예측구간까지"
date: "2026-10-15"
category: "공부"
subcategory: "Statistics/Time Series Analysis"
tags: ["시계열분석", "ETS", "상태공간모형", "Kalman filter"]
description: "지수평활법에 오차 분포를 붙여 통계 모형인 ETS로 만들고 예측구간이 왜 점점 넓어지는지 유도한다. 모수를 SSE와 우도로 추정하는 방법, 수준을 베이즈로 업데이트하는 관점, 3편의 추세 회귀와 무엇이 다른지도 함께 다룬다."
---

세종대학교 인공지능데이터사이언스학과 2026년 2학기 시계열분석 수업을 따라가며 정리하는 시리즈의 다섯 번째 글이다. 수업 내용과 교재 [Forecasting: Principles and Practice, 3rd ed.](https://otexts.com/fpp3/)(이하 FPP3) 8장 뒷부분(8.5~8.7절)을 바탕으로 썼다.

> **ETS가 말하는 것: 지수평활의 점예측에 오차 분포를 붙이면 앞으로의 불확실성은 얼마나 커질까?**

[4편](/posts/2026-10-15-time-series-4-exponential-smoothing)에서는 관측값이 들어올 때마다 수준 $\ell_t$, 기울기 $b_t$, 계절 성분 $s_t$를 업데이트하는 지수평활법을 유도했다. 이번 글에서 계속 쓰는 것은 SES의 평활식과 Holt의 두 평활식이다.

$$
\ell_t = \alpha y_t + (1-\alpha)\ell_{t-1} \tag{1}
$$

$$
\begin{aligned}
\text{Level: } & \ell_t = \alpha y_t + (1-\alpha)(\ell_{t-1} + b_{t-1}) \\
\text{Trend: } & b_t = \beta^*(\ell_t - \ell_{t-1}) + (1-\beta^*) b_{t-1}
\end{aligned}
$$

## 방법에서 모형으로

4편에서 본 SES, Holt, Holt-Winters는 점예측을 내는 계산 규칙(method)이었다. 예측구간을 내려면 무엇이 더 필요할까? 오차가 어떤 분포를 따르는지까지 정한 통계 모형(model)이다. **계산 규칙에 오차 분포를 붙이면 예측구간을 낼 수 있는 모형이 된다.** FPP3는 이것을 innovation state space model이라고 부르고 ETS(Error, Trend, Seasonal) 세 글자로 이름을 붙인다. 4편 표에 나온 방법의 약칭 앞에 오차 글자 하나가 더 붙는 것이다. 오차가 더해지면 A, 곱해지면 M이다. 덧셈 Holt-Winters를 덧셈 오차로 쓰면 ETS(A,A,A), 곱셈 Holt-Winters를 곱셈 오차로 쓰면 ETS(M,A,M)이 된다.

**주의:** SES, Holt, Holt-Winters는 방법(method)이고 ETS(·,·,·)는 모형(model)이다. 방법 하나에 덧셈 오차 모형과 곱셈 오차 모형 두 개가 대응한다. 두 모형은 점예측이 같아도 예측구간이 다르다.

### 오차가 더해지는 SES

SES의 평활식을 조금 정리하면 "오차 보정(error correction)" 꼴이 나온다.

$$
\ell_t = \alpha y_t + (1-\alpha)\ell_{t-1} = \ell_{t-1} + \alpha(y_t - \ell_{t-1}) = \ell_{t-1} + \alpha e_t, \qquad e_t = y_t - \hat{y}_{t|t-1}
$$

수준은 직전 수준에 이번 예측오차의 $\alpha$배만큼 보정한 값이다. 이제 이 오차 $e_t$를 확률변수 $\varepsilon_t \sim N(0, \sigma^2)$로 보자. 평활식에서는 상태 방정식이 나오고 $e_t = y_t - \ell_{t-1}$을 $y_t$로 정리하면 관측 방정식이 나온다. 이 모형이 ETS(A,N,N)이다.

$$
\begin{aligned}
y_t &= \ell_{t-1} + \varepsilon_t && (\text{관측 방정식}) \\
\ell_t &= \ell_{t-1} + \alpha\varepsilon_t && (\text{상태 방정식})
\end{aligned}
\qquad \varepsilon_t \sim N(0, \sigma^2) \tag{2}
$$

각 관측값은 직전 수준에 오차 하나를 더한 값이다. 4편에서 수준을 "흔들림을 걷어 낸 높이"라고 했던 말을 식으로 쓴 것이기도 하다.

### α의 두 끝은 상수와 random walk다

$\alpha$의 양 끝을 식 (2)에 넣어 보면 모형의 성격이 바로 보인다.

- $\alpha = 0$이면 $\ell_t = \ell_{t-1}$이라 수준이 상수 $k$다. 모형은 $y_t = k + \varepsilon_t$가 되고 일정한 값 주위에서 흔들리기만 한다.
- $\alpha = 1$이면 $\ell_t = \ell_{t-1} + \varepsilon_t = y_t$라서 $y_t = y_{t-1} + \varepsilon_t$, random walk다. 같은 규칙으로 만들어도 경로가 제각각 멀리 떠돌아다니는 계열이다.

### 오차가 곱해지는 SES

오차를 예측값으로 나눈 상대 오차로 둘 수도 있다.

$$
\varepsilon_t = \frac{y_t - \hat{y}_{t|t-1}}{\hat{y}_{t|t-1}} = \frac{y_t - \ell_{t-1}}{\ell_{t-1}}, \qquad \varepsilon_t \sim N(0, \sigma^2)
$$

그러면 $e_t = y_t - \ell_{t-1} = \ell_{t-1}\varepsilon_t$이고 이것을 평활식에 넣으면 ETS(M,N,N)이 된다.

$$
\begin{aligned}
y_t &= \ell_{t-1}(1 + \varepsilon_t) && (\text{관측 방정식}) \\
\ell_t &= \ell_{t-1} + \alpha\ell_{t-1}\varepsilon_t = \ell_{t-1}(1 + \alpha\varepsilon_t) && (\text{상태 방정식})
\end{aligned}
$$

ETS(A,N,N)과 ETS(M,N,N)은 평활식이 똑같이 식 (1)이다. 모수가 같으면 점예측도 같다. 오차를 어떻게 보느냐만 달라서 예측구간이 달라진다.

### Holt를 상태공간 꼴로

Holt 방법도 같은 방식으로 모형이 된다. 1-step 예측오차가 $\varepsilon_t = y_t - (\ell_{t-1} + b_{t-1})$이니 관측 방정식은 바로 나온다. 이것을 Level 식에 넣어 보자.

$$
\ell_t = \alpha(\ell_{t-1} + b_{t-1} + \varepsilon_t) + (1-\alpha)(\ell_{t-1} + b_{t-1}) = \ell_{t-1} + b_{t-1} + \alpha\varepsilon_t
$$

그러면 $\ell_t - \ell_{t-1} = b_{t-1} + \alpha\varepsilon_t$이고 이것을 Trend 식에 넣는다.

$$
b_t = \beta^*(b_{t-1} + \alpha\varepsilon_t) + (1-\beta^*) b_{t-1} = b_{t-1} + \alpha\beta^*\varepsilon_t
$$

$\alpha\beta^*$를 $\beta$로 쓰면 ETS(A,A,N)이 완성된다.

$$
\begin{aligned}
y_t &= \ell_{t-1} + b_{t-1} + \varepsilon_t \\
\ell_t &= \ell_{t-1} + b_{t-1} + \alpha\varepsilon_t \\
b_t &= b_{t-1} + \beta\varepsilon_t
\end{aligned}
\qquad \beta = \alpha\beta^*
$$

이 상태 방정식으로 보면 4편의 "미래의 수준과 기울기"도 한 줄로 설명된다. 미래 오차를 기댓값 0으로 두면 보정 항이 사라지니 기울기는 그대로이고 수준은 매 step 기울기만큼 오른다. 덧셈 오차 9개, 곱셈 오차 9개 모형 모두 상태 방정식이 "직전 상태 + 모수 × 오차" 꼴이다. 곱셈 오차 쪽은 오차에 그 시점의 예측 수준이 곱해진다는 점만 다르다.

## 예측구간은 왜 점점 넓어질까

모형이 생기면 예측구간을 낼 수 있다.

$$
\hat{y}_{T+h|T} \pm c\,\sigma_h
$$

$c$는 80% 구간에서 1.28, 95% 구간에서 1.96이다. $\sigma_h^2$는 $h$-step 예측오차의 분산인데 모형마다 식이 복잡하다[^hyndman]. 가장 단순한 ETS(A,N,N)은 직접 유도할 수 있다. 식 (2)의 관측 방정식과 상태 방정식을 번갈아 대입해 $\ell_t$까지 내려가 보자.

$$
\begin{aligned}
y_{t+h} &= \ell_{t+h-1} + \varepsilon_{t+h} \\
&= \ell_{t+h-2} + \alpha\varepsilon_{t+h-1} + \varepsilon_{t+h} \\
&\;\;\vdots \\
&= \ell_t + \alpha\varepsilon_{t+1} + \alpha\varepsilon_{t+2} + \cdots + \alpha\varepsilon_{t+h-1} + \varepsilon_{t+h}
\end{aligned}
$$

시점 $t$에서 $\ell_t$는 이미 아는 값이고 오차들은 서로 독립이다. 독립인 확률변수의 합의 분산은 분산의 합이고 $\operatorname{Var}(aX) = a^2\operatorname{Var}(X)$이니 바로 계산된다.

$$
\sigma_h^2 = \operatorname{Var}(y_{t+h} \mid t) = \underbrace{\alpha^2\sigma^2 + \cdots + \alpha^2\sigma^2}_{h-1\text{개}} + \sigma^2 = \sigma^2\big[1 + \alpha^2(h-1)\big] \tag{3}
$$

<figure className="my-6 text-center">
  <img src="/images/posts/2026-10-15-time-series-5-ets-models/interval-width.svg" alt="α = 0.2, 0.5, 1일 때 예측 시점 h에 따른 95% 예측구간 half-width" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">그림 1. h = 12에서 95% 구간 half-width는 α = 0.2면 2.35, 0.5면 3.80, 1이면 6.79다(σ = 1). 시작은 모두 1.96이다.</figcaption>
</figure>

**멀리 볼수록 아직 일어나지 않은 충격이 수준에 쌓이니 구간이 넓어진다.** 그림 1에서 보듯 얼마나 빨리 넓어지는지는 $\alpha$가 정한다. 양 끝을 보면 앞 글들과 이어진다.

- $\alpha = 0$이면 $\sigma_h^2 = \sigma^2$로 $h$와 상관없이 일정하다. 3편의 결정적 추세 모형에서 예측구간 폭이 일정했던 것과 같다.
- $\alpha = 1$(random walk)이면 $\sigma_h^2 = h\sigma^2$로 $h$에 비례해 커진다. 1편에서 AR 모형 $y_t = ay_{t-1} + b + \varepsilon_t$의 예측 분산을 구할 때 $a = 1$이면 끝없이 커지던 바로 그 경우다.

## 모수는 무엇으로 추정할까

4편에서 SES는 SSE로 $\alpha$와 $\ell_0$을 고른다고 했다. Holt와 Holt-Winters도 원리는 같다. 평활 모수와 초기 상태를 한꺼번에 움직이며 1-step 예측오차의 SSE를 최소로 한다. FPP3도 4편의 호주 인구 예의 $\alpha, \beta^*, \ell_0, b_0$를 이렇게 구했다고 쓴다.

$$
\text{SSE}(\boldsymbol\theta) = \sum_{t=1}^{T}\big(y_t - \hat{y}_{t|t-1}(\boldsymbol\theta)\big)^2
$$

$\boldsymbol\theta$에 들어가는 값은 방법마다 늘어난다. **평활 모수만이 아니라 초기 상태도 추정 대상이다.**

| 방법 | 평활 모수 | 초기 상태 | 자유 모수(free parameter)의 수 |
| --- | --- | --- | --- |
| SES | $\alpha$ | $\ell_0$ | 2 |
| Holt | $\alpha, \beta^*$ | $\ell_0, b_0$ | 4 |
| Damped | $\alpha, \beta^*, \phi$ | $\ell_0, b_0$ | 5 |
| Holt-Winters | $\alpha, \beta^*, \gamma$ | $\ell_0, b_0, s_0, s_{-1}, \dots, s_{1-m}$ | $m + 4$ (분기면 8) |

### 초기 계절 값의 합을 0으로 묶는 이유

Holt-Winters의 초기 계절 값은 $m$개인데 표에서는 $m-1$개로 셌다. 보통 합을 0으로 묶기 때문이다(곱셈 계절이면 합을 $m$으로). 묶지 않으면 무슨 일이 생길까? $\ell_0$에 $c$를 더하고 초기 계절 값 $m$개에서 모두 $c$를 빼 보자.

- 첫 예측 $\ell_0 + b_0 + s_{1-m}$은 그대로다.
- Level 식에서 $\ell_1$은 $\alpha c + (1-\alpha)c = c$만큼 오른다.
- Season 식에서 $s_1$은 $-\gamma c - (1-\gamma)c = -c$만큼 내려간다.
- $\ell_1 - \ell_0$이 그대로라 $b_1$도 그대로다.

그러면 다음 예측도 같고 이 일이 끝까지 이어진다. SSE가 같은 점이 끝없이 많아서 최소점이 하나로 정해지지 않는다. 분기 자료로 직접 돌려 보니 $c = 7$을 옮겨도 1-step 예측의 차이가 최대 $4 \times 10^{-14}$였다. 3편 회귀에서 dummy 변수를 모두 넣으면 계수가 정해지지 않던 dummy trap, 2편 classical decomposition에서 계절 값의 합을 0으로 맞추던 조정과 같은 문제다.

모수의 범위도 정해 둔다. 평활식을 가중평균으로 읽으려고 $0 < \alpha, \beta^*, \gamma^*, \phi < 1$로 둔다. 상태공간 꼴의 $\beta = \alpha\beta^*$, $\gamma = (1-\alpha)\gamma^*$로 옮기면 $0 < \alpha < 1$, $0 < \beta < \alpha$, $0 < \gamma < 1 - \alpha$가 된다. FPP3의 fable 패키지는 계산이 불안정해지지 않도록 $\phi$를 $0.8 < \phi < 0.98$로 더 좁힌다.

### 모형이라면 우도로

ETS 모형으로 보면 우도를 최대로 하는 값을 고른다. 이것이 SSE와 같은 답을 줄까? **덧셈 오차면 같은 답이 나오고 곱셈 오차면 다르다.**

오차가 더해지는 모형은 $\varepsilon_t = y_t - \hat{y}_{t|t-1} \sim N(0, \sigma^2)$이다.

$$
\log L = -\frac{T}{2}\log(2\pi\sigma^2) - \frac{\text{SSE}}{2\sigma^2}
$$

$\sigma^2$로 미분해 0으로 두면 $\hat\sigma^2 = \text{SSE}/T$이고 이를 다시 넣으면 $\log L = -\frac{T}{2}\log\text{SSE} + \text{상수}$다. SSE가 작을수록 우도가 크니 SSE 최소화와 답이 같다.

오차가 곱해지는 모형은 $y_t = \mu_t(1 + \varepsilon_t)$, $\mu_t = \hat{y}_{t|t-1}$이라 $y_t$의 조건부 분포가 $N(\mu_t, \mu_t^2\sigma^2)$이다. 분산이 예측값에 따라 달라진다.

$$
\log L = -\frac{T}{2}\log(2\pi\sigma^2) - \sum_{t=1}^{T}\log\lvert\mu_t\rvert - \frac{1}{2\sigma^2}\sum_{t=1}^{T}\varepsilon_t^2, \qquad \varepsilon_t = \frac{y_t - \mu_t}{\mu_t}
$$

같은 방법으로 $\sigma^2$를 없애면 다음을 최소로 하는 문제가 된다.

$$
T\log\sum_{t=1}^{T} \varepsilon_t^2 + 2\sum_{t=1}^{T} \log\lvert\mu_t\rvert
$$

오차의 제곱합이 아니라 상대 오차의 제곱합이고 $\log\lvert\mu_t\rvert$ 항까지 붙으니 SSE와는 답이 다르다. 예측값을 키우면 상대 오차는 작아 보이지만 분산 $\mu_t^2\sigma^2$도 함께 커진다. 이 항이 그만큼 벌점을 준다.

우도가 필요한 이유가 하나 더 있다. 여러 ETS 모형 가운데 하나를 고를 때 FPP3는 3편에서 본 AIC를 쓰는데 AIC는 우도로 계산한다. 덧셈 오차 모형의 SSE와 곱셈 오차 모형의 상대 오차 제곱합은 단위가 달라 서로 비교할 수 없다.

## 베이즈 업데이트로 다시 읽기

관측값이 들어올 때마다 상태를 업데이트한다는 설명을 듣다 보면 이런 생각이 든다. 수준을 모르는 모수로 보고 한 step마다 베이즈 업데이트를 하는 것과 같지 않을까? 그렇게 볼 수 있다. 다만 고정된 모수가 아니라 매 시점 조금씩 움직이는 모수로 봐야 SES와 같은 식이 나온다.

잡음이 두 개인 local level model로 따져 보자. $\mu_t$가 수준이다.

$$
y_t = \mu_t + v_t,\ \ v_t \sim N(0, V) \qquad\qquad \mu_t = \mu_{t-1} + w_t,\ \ w_t \sim N(0, W)
$$

$V$는 관측 잡음의 분산이고 $W$는 수준이 한 시점에 움직이는 정도다. 시점 $t-1$까지 본 뒤 수준의 사후분포가 $N(m_{t-1}, C_{t-1})$이라고 하면 한 step은 이렇게 돈다.

1. 사전분포: 수준이 $w_t$만큼 움직이므로 $\mu_t \mid y_{1:t-1} \sim N(m_{t-1}, R_t)$, $R_t = C_{t-1} + W$다.
2. 가능도: $y_t \mid \mu_t \sim N(\mu_t, V)$다.
3. 사후분포: 정규-정규 켤레라 사후분포도 정규분포다.

$$
m_t = K_t\, y_t + (1 - K_t)\, m_{t-1}, \qquad K_t = \frac{R_t}{R_t + V}, \qquad C_t = K_t V
$$

사후 평균은 사전 평균 $m_{t-1}$과 새 관측값 $y_t$의 가중평균이고 가중치는 각자의 정밀도(분산의 역수)에 비례한다. 식 (1)의 $\alpha$ 자리에 $K_t$가 들어간 꼴이다. 수준을 잘 모를수록($R_t$가 클수록) 새 관측값을 많이 믿고 관측 잡음의 분산 $V$가 클수록 덜 믿는다.

$K_t$는 처음 몇 step만 바뀌고 곧 상수로 수렴한다. 수렴하면 $R = KV + W$이고 $K = R/(R+V)$에서 $R = KV/(1-K)$다. 두 식을 같게 놓고 $q = W/V$로 두면 $K^2 + qK - q = 0$이 나오고 그 양의 근이 SES의 $\alpha$다.

$$
\alpha = \frac{-q + \sqrt{q^2 + 4q}}{2}, \qquad q = \frac{\alpha^2}{1-\alpha}
$$

| $q = W/V$ | 0.01 | 0.1 | 1 | 10 | 100 |
| --- | --- | --- | --- | --- | --- |
| $\alpha$ | 0.095 | 0.27 | 0.618 | 0.916 | 0.99 |

<figure className="my-6 text-center">
  <img src="/images/posts/2026-10-15-time-series-5-ets-models/kalman-gain.svg" alt="q = 0.1, 1, 10일 때 칼만 이득 K_t가 상수 α로 수렴하는 모습" className="rounded-lg mx-auto" />
  <figcaption className="mt-2 text-sm text-gray-500">그림 2. 칼만 이득 K_t는 몇 step 만에 상수로 수렴하고 그 값이 SES의 α다. q = 1이면 1, 0.667, 0.625, 0.619, 0.618로 수렴한다.</figcaption>
</figure>

그림 2는 처음 사전분포를 아주 넓게 두고 $K_t$를 계산한 것이다. $q = 1$이면 1, 0.667, 0.625, 0.619, 0.618로 금방 수렴하고 $q = 0.1$은 0.270, $q = 10$은 0.916에 붙는다. 수렴한 뒤로는 SES와 같은 점화식이다[^sim]. **SES의 $\alpha$는 결국 "수준이 움직이는 정도 대 관측 잡음"의 비율로 정해지는 셈이다.** 4편의 호주 인구에서 $\hat\alpha$가 거의 1이었던 것도 이 관점으로 보면 관측 잡음 $V$가 아주 작은 계열이라는 뜻이다.

4편에서 본 두 극단(average와 naive)도 여기서 나온다.

- $W = 0$(수준이 정말 고정된 모수): 사전 분산이 계속 줄어서 $K_t = 1/t$가 되고 사후 평균은 표본 평균이다[^vague]. average 예측이다.
- $V = 0$(관측 잡음이 없음): $K = 1$이라 naive다.

고정된 모수에 베이즈 업데이트를 하면 가중치가 $1/t$로 줄어 결국 average가 된다. SES의 $\alpha$가 0으로 줄지 않는 것은 수준이 매 시점 $W$만큼 움직인다고 보기 때문이다. 사전 분산이 $W$ 아래로 내려가지 않으니 옛 관측값을 계속 잊어 간다.

**주의:** ETS(A,N,N) 자체는 이 모형과 다르다. ETS는 식 (2)처럼 관측식과 상태식에 같은 $\varepsilon_t$ 하나가 들어가서 $\ell_0$와 $\alpha$가 정해지면 $\ell_t$가 자료로 정확히 계산된다. $\ell_t$의 사후분포라는 것이 따로 없다. 베이지안 해석은 잡음이 두 개인 위 모형의 것이고 수렴한 뒤에 두 모형의 점예측과 예측 분산이 같아지는 것이다. ETS의 $\sigma^2$를 1-step 예측오차의 분산 $R + V$로 맞추면 $\alpha^2\sigma^2 = R^2/(R+V) = W$라서 식 (3)의 $\sigma^2[1 + \alpha^2(h-1)]$이 $R + V + (h-1)W$가 된다. 위 모형에서 직접 구한 $h$-step 예측 분산과 같은 값이다.

추세와 계절성이 있으면 상태가 벡터가 되고 같은 계산을 행렬로 한다. 선형 정규 상태공간모형에서 상태의 사후분포를 한 step씩 구하는 이 계산이 Kalman filter이고 베이지안 쪽에서는 West와 Harrison의 동적선형모형(dynamic linear model)이 같은 틀이다.

## 3편의 회귀와 무엇이 다를까

2편처럼 분해하고 3편처럼 회귀해서 예측할 수도 있고 4편과 이번 글처럼 지수평활로 예측할 수도 있다. 둘 다 추세와 계절성을 나눠 다루지만 성분을 보는 관점이 다르다.

| | 3편 (회귀) | 4·5편 (ETS) |
| --- | --- | --- |
| 성분 | 시간의 고정된 함수 ($\beta_0 + \beta_1 t$, dummy) | 시점마다 바뀌는 상태 ($\ell_t, b_t, s_t$) |
| 지금의 성분을 정하는 관측값 비중 | 자료 전체가 같음 (OLS) | 최근일수록 큼 ($\alpha(1-\alpha)^j$) |
| 예측 | 미래 시점 $t = T+h$를 식에 대입 | 마지막 상태에서 앞으로 늘림 |
| 예측 분산 | $h$와 상관없이 $\sigma^2$ ($\hat\beta$ 추정 오차 무시) | $h$가 커질수록 커짐 |
| 잘 맞는 자료 | 패턴이 오래 그대로인 계열 | 패턴이 서서히 바뀌는 계열 |

2편의 이동평균, classical decomposition, STL은 그 자체로는 예측 방법이 아니었다. 성분을 나눠 보고 계절 조정을 하는 도구였다. 이동평균은 양 끝 몇 시점에서 값이 비고 미래로 늘릴 식도 없다. 예측식이 나온 곳은 3편에서 시간과 dummy 변수에 회귀한 부분이었다.

그렇다면 두 방법은 완전히 다른 것일까? 그렇지 않다. ETS(A,A,A)에서 $\alpha = \beta = \gamma = 0$으로 두면 상태가 전혀 바뀌지 않는다.

$$
\ell_t = \ell_0 + b_0 t, \quad b_t = b_0, \quad s_t = s_{t-m} \quad\Rightarrow\quad y_t = \ell_{t-1} + b_{t-1} + s_{t-m} + \varepsilon_t = \ell_0 + b_0 t + s_{t-m} + \varepsilon_t
$$

선형 추세에 해마다 같은 계절 값을 더한 식이라 3편의 맥주 생산량 회귀(선형 추세 + 분기 dummy)와 같은 모형이다. 계절 값을 1분기 기준으로 적느냐 합을 0으로 맞추느냐만 다를 수 있다. **3편의 회귀는 "성분은 절대 바뀌지 않는다"고 못 박은 ETS인 셈이다.** ETS는 $\alpha, \beta, \gamma$를 추정해 성분이 얼마나 빨리 바뀌는지를 자료가 정하게 한다. 추정값이 0에 가까우면 회귀와 거의 같은 예측이 나오고 크면 최근 값을 따라간다.

섞어 쓰는 방법도 있다. FPP3 5.7절은 STL로 계절 성분을 떼어 낸 뒤 계절 조정된 계열은 Holt 같은 비계절 방법으로, 계절 성분은 마지막 1년을 반복하는 seasonal naive로 예측해 더한다. 3편에서 계절 조정된 자료에 추세 회귀를 했던 것과 같은 짜임이고 추세 회귀 자리에 Holt가 들어간 것이다. 어느 쪽이 나은지는 자료마다 다르니 뒤쪽 구간을 남겨 두고 예측 정확도를 비교해 고른다.

## ARIMA로 이어지는 다리

6편부터는 ARIMA를 다룬다. 성분을 나누지 않고 $y_t$와 과거 값, 과거 충격의 관계를 직접 모형으로 쓰는 방법이다. 그런데 지수평활과 아예 다른 세계일까? 식 (2)의 관측값을 한 번 차분해 보자.

$$
y_t - y_{t-1} = (\ell_{t-1} + \varepsilon_t) - (\ell_{t-2} + \varepsilon_{t-1}) = \alpha\varepsilon_{t-1} + \varepsilon_t - \varepsilon_{t-1} = \varepsilon_t - (1-\alpha)\varepsilon_{t-1}
$$

차분한 계열이 이번 충격과 직전 충격 하나로 설명되는 꼴이다. **ETS(A,N,N)을 한 번 차분하면 다음 편에서 다룰 MA(1) 모형이 되고 계수는 $\theta_1 = \alpha - 1$이다.** 정상성과 차분, AR과 MA를 정리한 뒤 이 연결을 ARIMA의 언어로 다시 보자.

---

**참고**

- FPP3 [8.5 Innovations state space models for exponential smoothing](https://otexts.com/fpp3/ets.html), [8.6 Estimation and model selection](https://otexts.com/fpp3/ets-estimation.html), [8.7 Forecasting with ETS models](https://otexts.com/fpp3/ets-forecasting.html)
- Hyndman, Koehler, Ord, Snyder (2008), *Forecasting with Exponential Smoothing: The State Space Approach*
- West, Harrison (1997), *Bayesian Forecasting and Dynamic Models*, 2nd ed.

[^hyndman]: 모형별 $\sigma_h^2$ 식은 Hyndman et al. (2008), *Forecasting with Exponential Smoothing: The State Space Approach* 6장에 정리돼 있다.

[^sim]: $q = 1$로 30개를 시뮬레이션해 $t = 10$부터 $\alpha = 0.618$인 SES를 돌린 값과 사후 평균을 비교하니 차이가 최대 $5 \times 10^{-9}$였다.

[^vague]: $K_t = 1/t$가 정확히 나오려면 처음 사전분포의 분산이 아주 커야 한다. 그래야 첫 관측값이 사전 평균을 완전히 덮어쓴다.
