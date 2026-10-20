---
title: "시계열분석 (8): MA 모형과 가역성, 과거 충격으로 현재를 설명하기"
date: "2026-10-17"
category: "공부"
subcategory: "Statistics/Time Series Analysis"
tags: ["시계열분석", "MA 모형", "가역성", "ARIMA"]
description: "과거 충격의 가중합으로 현재를 설명하는 MA 모형을 다룬다. 정상인 AR이 MA(∞)가 되고 가역인 MA가 AR(∞)가 되는 대응을 유도한다. 보이지 않는 충격을 거꾸로 계산해 계수를 추정하는 방법, 가역성이 필요한 이유, 지수평활이 ARIMA의 한 경우가 되는 지점도 정리한다."
---

세종대학교 인공지능데이터사이언스학과 2026년 2학기 시계열분석 수업을 따라가며 정리하는 시리즈의 여덟 번째 글이다. 수업 내용과 교재 [Forecasting: Principles and Practice, 3rd ed.](https://otexts.com/fpp3/)(이하 FPP3) 9.4절, 9.6절, 9.10절을 바탕으로 썼다.

> **MA 모형이 말하는 것: 눈에 보이지 않는 과거 충격만으로 현재를 설명할 수 있을까?**

[7편](/posts/2026-10-17-time-series-7-backshift-ar)의 AR 모형은 과거 값으로 현재를 설명했다. 이번 글의 MA 모형은 과거 충격, 곧 과거의 예측오차로 설명한다. 두 모형은 서로 바꿔 쓸 수 있는데 그 조건이 정상성과 가역성이다. 보이지 않는 충격으로 계수를 어떻게 추정하는지도 본다. 끝에서는 [5편](/posts/2026-10-15-time-series-5-ets-models) 마지막에 미뤄 둔 "ETS를 차분하면 MA(1)"을 다시 본다.

## MA 모형: 보이지 않는 충격의 가중합

MA($q$) 모형은 지금 값을 현재와 과거 $q$개 충격의 가중합으로 쓴다.

$$
y_t = c + \varepsilon_t + \theta_1\varepsilon_{t-1} + \theta_2\varepsilon_{t-2} + \cdots + \theta_q\varepsilon_{t-q} \tag{1}
$$

$\varepsilon_t$는 white noise다. 과거 충격을 설명변수로 쓰는 회귀처럼 생겼지만 그 설명변수를 관측할 수 없다는 점이 다르다.

**주의:** [2편](/posts/2026-10-15-time-series-2-decomposition)의 moving average와 이름만 같다. [2편](/posts/2026-10-15-time-series-2-decomposition)의 이동평균은 관측값 $y$를 평균해 추세를 구하는 평활법이고 여기의 MA는 보이지 않는 충격 $\varepsilon$의 가중합으로 $y$를 만드는 모형이다.

MA 모형은 성질이 단순하다. 모든 $\varepsilon$의 기댓값이 0이니 $E(y_t) = c$이고 서로 독립이니 분산은 $\sigma^2(1 + \theta_1^2 + \cdots + \theta_q^2)$이다. 둘 다 $t$와 상관없다. 공분산은 두 시점에 함께 들어 있는 충격에서만 나온다. MA(1) $y_t = \varepsilon_t + \theta\varepsilon_{t-1}$이라면 $y_t$와 $y_{t-1}$이 $\varepsilon_{t-1}$을 함께 담고 있어

$$
\operatorname{Cov}(y_t, y_{t-1}) = \theta\sigma^2, \qquad \rho_1 = \frac{\theta}{1 + \theta^2}
$$

이고 두 칸 이상 떨어지면 겹치는 충격이 없어 공분산이 0이다. [6편](/posts/2026-10-17-time-series-6-stationarity-differencing)의 식 (4)대로 white noise에 유한한 선형 필터를 건 것이니 **MA 모형은 $\theta$가 무엇이든 늘 정상이다.**

<Fig src="/images/posts/2026-10-17-time-series-8-ma-invertibility/ma-vs-ar.svg" alt="MA(1)과 AR(1)의 시간 그림과 ACF">그림 1. $\theta_1 = 0.8$인 MA(1)과 $\phi_1 = 0.8$인 AR(1)을 200개씩 만들었다. MA(1)의 ACF는 시차 1만 크고 바로 점선 안으로 들어가지만 AR(1)의 ACF는 0.75, 0.54, 0.36으로 천천히 줄어든다.</Fig>

그림 1의 MA(1)은 $\rho_1 = 0.8/1.64 = 0.488$이고 표본에서는 $r_1 = 0.471$이었다. 시차 2부터는 모두 점선 안이다. 같은 0.8을 쓴 AR(1)은 $\rho_k = 0.8^k$라 0.8, 0.64, 0.51로 천천히 줄어든다. **MA($q$)의 ACF는 시차 $q$ 뒤에 끊기고 AR의 ACF는 천천히 줄어든다.** 이 차이로 차수를 고르는 방법은 다음 글에서 다룬다[^band].

## 정상 AR은 MA($\infty$)다

두 모형은 서로 바꿔 쓸 수 있다. 먼저 AR(1)을 MA로 바꿔 보자. 평균이 0이고 $\lvert\phi\rvert < 1$인 AR(1)에서 $y_{t-1}$ 자리에 같은 식을 계속 넣는다.

$$
\begin{aligned}
y_t &= \phi y_{t-1} + \varepsilon_t = \phi(\phi y_{t-2} + \varepsilon_{t-1}) + \varepsilon_t = \phi^2y_{t-2} + \phi\varepsilon_{t-1} + \varepsilon_t \\
&= \cdots = \phi^ky_{t-k} + \phi^{k-1}\varepsilon_{t-k+1} + \cdots + \phi\varepsilon_{t-1} + \varepsilon_t
\end{aligned}
$$

$\lvert\phi\rvert < 1$이면 $k \to \infty$에서 $\phi^ky_{t-k}$가 사라진다.

$$
y_t = \sum_{j=0}^{\infty}\phi^j\varepsilon_{t-j} \tag{2}
$$

계수 $\phi^j$는 [7편](/posts/2026-10-17-time-series-7-backshift-ar)에서 따라간 "충격이 $j$시점 뒤에 남는 양"이다. 정상성 조건 $\lvert\phi\rvert < 1$이 바로 남은 항 $\phi^ky_{t-k}$가 사라지는 조건이기도 하다. **정상인 AR($p$)는 모두 MA($\infty$)로 쓸 수 있다.** Backshift로 쓰면 $(1 - \phi B)y_t = \varepsilon_t$의 양변을 $(1 - \phi B)$로 나눈 것이다. 등비급수로 $(1 - \phi B)^{-1} = \sum_j\phi^jB^j$이고 $\lvert\phi\rvert < 1$일 때 수렴한다. [7편](/posts/2026-10-17-time-series-7-backshift-ar)의 $(\mathbf{I} - \mathbf{L})^{-1}$이 누적합이었던 것과 같은 계산이다.

## 가역성: MA를 AR($\infty$)로 뒤집기

이번에는 반대로 MA(1) $y_t = \varepsilon_t + \theta\varepsilon_{t-1}$을 AR로 바꿔 보자. $\varepsilon_t = y_t - \theta\varepsilon_{t-1}$에서 $\varepsilon_{t-1}$ 자리에 같은 식을 계속 넣는다.

$$
\begin{aligned}
\varepsilon_t &= y_t - \theta(y_{t-1} - \theta\varepsilon_{t-2}) = y_t - \theta y_{t-1} + \theta^2\varepsilon_{t-2} \\
&= \cdots = y_t - \theta y_{t-1} + \theta^2y_{t-2} - \cdots + (-\theta)^ky_{t-k} + (-\theta)^{k+1}\varepsilon_{t-k-1}
\end{aligned}
$$

마지막 항이 사라지려면 $\lvert\theta\rvert < 1$이어야 한다. 그러면

$$
\varepsilon_t = \sum_{j=0}^{\infty}(-\theta)^jy_{t-j} \tag{3}
$$

이고 이것을 $y_t$에 대해 정리하면 과거 값 무한 개로 쓰는 AR($\infty$)다. 이렇게 뒤집을 수 있는 MA를 가역(invertible)이라 한다. Backshift로는 $(1 + \theta B)^{-1} = \sum_j(-\theta)^jB^j$다. 부호가 번갈아 바뀐다는 점을 놓치기 쉽다. 일반적인 MA($q$)는 $\Theta(x) = 1 + \theta_1x + \cdots + \theta_qx^q$의 근이 모두 단위원 밖에 있을 때 가역이다. [7편](/posts/2026-10-17-time-series-7-backshift-ar)의 AR 정상성 조건과 똑같은 모양의 조건을 다른 다항식에 건 셈이다.

### 가역성은 왜 필요한가

MA는 어차피 늘 정상인데 왜 가역성까지 요구할까? 이유는 두 가지다.

첫째, 모형을 하나로 정하기 위해서다. $\rho_1 = \theta/(1 + \theta^2)$에 $\theta$와 $1/\theta$를 넣으면 같은 값이 나온다. $\theta = 0.5$와 $\theta = 2$는 둘 다 $\rho_1 = 0.4$다. 분산까지 맞추면($\theta$와 $\sigma^2$ 대 $1/\theta$와 $\theta^2\sigma^2$) 자기공분산이 완전히 같아서 정규분포 계열이라면 자료로는 둘을 구별할 수 없다. 가역 조건 $\lvert\theta\rvert < 1$이 둘 중 하나를 고른다.

둘째, 지금의 충격을 과거 관측값으로 계산할 수 있어야 한다. 잔차를 구하고 예측하려면 $\varepsilon_t$를 알아야 하는데 그 계산이 식 (3)이다. $\lvert\theta\rvert > 1$이면 가중치 $(-\theta)^j$가 먼 과거로 갈수록 커진다.

<Fig src="/images/posts/2026-10-17-time-series-8-ma-invertibility/invertibility.svg" alt="MA(1)의 시차 1 자기상관과 역변환 가중치">그림 2. 왼쪽: MA(1)의 $\rho_1 = \theta/(1 + \theta^2)$. $\theta = 0.5$와 $\theta = 2$가 같은 0.4를 준다. 오른쪽: 충격을 과거 관측값으로 되돌릴 때의 가중치 $\lvert\theta\rvert^j$. $\theta = 2$이면 10시점 전 관측값의 가중치가 1024다.</Fig>

그림 2 오른쪽에서 $\theta = 0.5$이면 10시점 전 관측값의 가중치가 0.001로 줄지만 $\theta = 2$이면 1024로 커진다. 오래된 관측값일수록 지금의 충격을 더 많이 좌우한다는 말이니 예측 모형으로 쓸 수 없다. **가역성은 지금의 충격을 최근 관측값 위주로 계산할 수 있다는 보증이다.**

[6편](/posts/2026-10-17-time-series-6-stationarity-differencing)에서 본 과차분이 여기와 이어진다. White noise를 차분하면 $\varepsilon_t - \varepsilon_{t-1}$로 $\theta = -1$이다. 근이 단위원 바로 위에 있어서 가중치 $\lvert\theta\rvert^j = 1$이 전혀 줄지 않는다. 필요 없는 차분을 하면 가역성의 경계에 걸린 MA가 생기는 셈이다.

## AR과 MA의 대응

정리하면 두 모형은 다음처럼 짝을 이룬다.

| | AR($p$) | MA($q$) |
| --- | --- | --- |
| 정상성 | $\Phi(x)$의 근이 모두 $\lvert x\rvert > 1$일 때만 | 늘 정상 |
| 가역성 | 늘 가역 | $\Theta(x)$의 근이 모두 $\lvert x\rvert > 1$일 때만 |
| 바꿔 쓰기 | 정상이면 MA($\infty$) | 가역이면 AR($\infty$) |
| 1차일 때 조건 | $\lvert\phi_1\rvert < 1$ | $\lvert\theta_1\rvert < 1$ |
| ACF | 천천히 줄어든다 | 시차 $q$ 뒤에 끊긴다 |

AR은 과거 값 유한 개로 쓰니 늘 가역이고 MA는 충격 유한 개의 합이니 늘 정상이다. 반대쪽으로 바꿔 쓸 때만 조건이 필요하다.

## MA 계수는 어떻게 추정하는가

[7편](/posts/2026-10-17-time-series-7-backshift-ar)의 AR 계수는 1-step 예측오차의 제곱합을 가장 작게 하는 회귀로 구했다. 설명변수 $y_{t-1}$이 관측값이라 제곱합이 $\phi$의 2차식이고 미분하면 바로 풀렸다. MA에도 같은 기준을 쓰고 싶은데 걸리는 것이 있다. 식 (1)의 설명변수 $\varepsilon_{t-1}$이 보이지 않는다. 그렇다면 어떻게 해야 할까?

### 충격을 거꾸로 계산하는 조건부 제곱합

$\theta$ 후보를 하나 정하면 충격을 자료에서 하나씩 계산할 수 있다. 평균이 $\mu$인 MA(1) $y_t = \mu + \varepsilon_t + \theta\varepsilon_{t-1}$을 충격 쪽으로 정리하고 아직 모르는 첫 충격은 0으로 둔다.

$$
e_t = y_t - \mu - \theta e_{t-1}, \qquad e_0 = 0 \tag{4}
$$

이렇게 만든 잔차의 제곱합 $S(\mu, \theta) = \sum_t e_t^2$을 가장 작게 하는 값이 조건부 제곱합(conditional sum of squares, CSS) 추정값이다. [7편](/posts/2026-10-17-time-series-7-backshift-ar)의 조건부 최소제곱이 첫 관측값을 주어진 값으로 두었다면 여기서는 첫 충격을 0으로 둔다. 식 (4)를 풀면 $e_t = \sum_{j=0}^{t-1}(-\theta)^j(y_{t-j} - \mu)$로 식 (3)을 $t$개에서 자른 꼴이다.

기준은 같아도 계산은 다르다. $e_t$가 $\theta$의 $(t-1)$차 다항식이라 $S$는 $\theta$의 고차 다항식이고 미분해 0으로 두어도 손으로 풀리지 않는다. 그래서 컴퓨터로 최솟값을 찾는다. $\theta = 0.6$, $\mu = 5$로 200개를 만들고 $\mu$를 표본평균에 둔 채 $S$를 그리면 그림 3 왼쪽처럼 최솟값 양쪽이 대칭이 아니다. 2차식이 아니라는 뜻이다. $\mu$까지 함께 최소화하면 $\hat\theta = 0.609$가 나온다.

<Fig src="/images/posts/2026-10-17-time-series-8-ma-invertibility/ma-estimation.svg" alt="MA(1)의 조건부 제곱합 곡선과 첫 충격을 0으로 둔 오차가 남는 정도">그림 3. 왼쪽: $\theta = 0.6$인 MA(1) 200개로 계산한 $S(\theta)$. 최솟값 양쪽이 왜 대칭이 아닐까? 오른쪽: 첫 충격을 0으로 둔 오차가 $t$시점 뒤에 남는 정도 $\lvert\theta\rvert^t$. $\lvert\theta\rvert < 1$이어야 사라진다.</Fig>

### 첫 충격을 0으로 둔 대가

$e_0 = 0$은 사실이 아니다. 참값 $\theta$를 넣어도 $e_t - \varepsilon_t = -\theta(e_{t-1} - \varepsilon_{t-1})$이라 처음에 틀린 몫 $-\varepsilon_0$이 $(-\theta)^t$배로 계속 따라온다. 그림 3 오른쪽에서 $\theta = 0.6$이면 50시점 뒤에 $8 \times 10^{-12}$배만 남는다. $\theta = 0.95$이면 0.077배가 남고 $\theta = 1.05$이면 오히려 커진다. $\theta = 2$로 만든 자료에 $\theta = 2$를 넣고 식 (4)를 돌리면 잔차가 $10^{60}$ 규모로 불어난다. **CSS는 식 (3)으로 충격을 되찾는 계산이라 가역인 $\theta$에서만 제대로 된다.** 앞에서 본 가역성의 둘째 이유가 추정에서 그대로 드러나는 셈이다.

### 정확한 우도: 첫 충격을 가정하지 않는다

첫 충격을 아예 정하지 않는 방법도 있다. $\varepsilon_t$가 정규분포이면 관측값을 모은 벡터 $\mathbf{y}$는 다변량정규분포를 따르고 공분산 행렬은 앞에서 구한 자기공분산으로 채운다. MA(1)이면 대각선이 $\sigma^2(1 + \theta^2)$, 바로 옆이 $\theta\sigma^2$이고 나머지는 0이다. 이 분포의 우도를 가장 크게 하는 $(\mu, \theta, \sigma^2)$가 최대가능도 추정값이다. 이것도 컴퓨터로 푼다. FPP3 9.6절은 fable이 ARIMA를 최대가능도로 추정하며 ARIMA에서는 그 값이 오차 제곱합을 최소로 하는 최소제곱 추정값과 비슷하다고 적는다. 같은 표본에서 $\hat\theta = 0.605$로 CSS의 0.609와 거의 같았다.

가역성의 첫째 이유도 여기서 다시 나온다. $\theta$와 $\sigma^2$인 MA(1)과 $1/\theta$와 $\theta^2\sigma^2$인 MA(1)은 자기공분산이 같으니 공분산 행렬도 같고 우도도 같다. $\theta = 2$로 만든 자료에서 $(\theta, \sigma^2) = (2, 1)$과 $(0.5, 4)$의 로그우도가 소수점 여섯째 자리까지 같았다. 자료로는 둘을 가를 수 없으니 추정값은 가역인 쪽으로 정한다.

### 자기상관으로 바로 풀 수는 없는가

더 쉬운 길도 있다. $\rho_1 = \theta/(1 + \theta^2)$에 표본 자기상관 $r_1$을 넣고 $\theta$를 풀면 된다. 가역인 근은 $\hat\theta = \big(1 - \sqrt{1 - 4r_1^2}\big)/(2r_1)$이고 위 표본은 $r_1 = 0.422$라 0.549가 나온다. 다만 $\lvert r_1\rvert \ge 1/2$이면 해가 없다. MA(1)의 $\rho_1$은 그림 2 왼쪽처럼 $\pm 1/2$을 넘지 못하지만 표본 $r_1$은 넘을 수 있다. 같은 설정으로 300번 반복해 세 방법을 비교했다.

| 방법 | 해가 없던 횟수 | $\hat\theta$ 평균 | 표준편차 |
| --- | --- | --- | --- |
| $r_1$로 풀기 | 33 | 0.590 | 0.141 |
| CSS | 0 | 0.605 | 0.059 |
| 최대가능도 | 0 | 0.608 | 0.059 |

$r_1$로 푸는 방법은 해가 나온 경우만 따져도 흔들림이 두 배가 넘는다. **그래서 MA 계수는 수치 최적화로 구한다.** AR 항이 섞인 ARMA도 식 (4)에 AR 항을 더할 뿐이고 AR과 MA 계수를 한꺼번에 추정한다. 그 이야기는 ARIMA를 다루는 글에서 이어 간다.

## 지수평활과 ARIMA는 어떻게 이어지는가

[5편](/posts/2026-10-15-time-series-5-ets-models) 끝에서 ETS(A,N,N)을 한 번 차분해 보았다. 관측 방정식 $y_t = \ell_{t-1} + \varepsilon_t$와 상태 방정식 $\ell_t = \ell_{t-1} + \alpha\varepsilon_t$로부터

$$
y_t - y_{t-1} = (\ell_{t-1} - \ell_{t-2}) + \varepsilon_t - \varepsilon_{t-1} = \varepsilon_t - (1 - \alpha)\varepsilon_{t-1}
$$

이었다. 이제 이 식을 읽을 수 있다. 한 번 차분한 계열이 $\theta = \alpha - 1$인 MA(1)이다. 차분 한 번에 MA(1)을 붙인 이 모형을 ARIMA(0,1,1)이라 부른다. 이름은 다음 글들에서 정리한다.

예측식도 같다. MA(1)의 1-step 예측은 다음 충격만 0으로 두면 되므로

$$
\hat{y}_{t+1|t} = y_t + \theta\varepsilon_t = y_t + \theta(y_t - \hat{y}_{t|t-1}) = \alpha y_t + (1 - \alpha)\hat{y}_{t|t-1}
$$

이다. [4편](/posts/2026-10-15-time-series-4-exponential-smoothing)의 SES 가중평균 꼴 그대로다. 무작위 계열에 두 점화식을 $\alpha = 0.3, 0.8, 1.5$로 돌려 보니 예측의 차이가 최대 $2 \times 10^{-14}$였다. SES 가중치 $\alpha(1 - \alpha)^j$의 $(1 - \alpha)^j$도 이 MA(1)을 식 (3)처럼 뒤집을 때 나오는 $(-\theta)^j$다.

가역 조건 $\lvert\theta\rvert = \lvert\alpha - 1\rvert < 1$을 풀면 $0 < \alpha < 2$다. 가중평균으로 읽으려고 잡은 $0 < \alpha < 1$보다 넓다. $1 < \alpha < 2$이면 가중치의 부호가 번갈아 바뀌지만 예측식으로는 여전히 작동한다는 뜻이다.

**ETS와 ARIMA는 다른 세계가 아니다.** FPP3 9.10절은 덧셈 오차의 선형 ETS 모형이 모두 ARIMA의 특수한 경우라고 정리한다. ETS(A,A,N)은 ARIMA(0,2,2)이고 계절 모형도 대응하는 ARIMA가 있다. 반대로 곱셈 오차처럼 비선형인 ETS는 대응하는 ARIMA가 없고 ETS로 나타낼 수 없는 ARIMA도 많다. 둘은 겹치는 부분이 있는 두 모형족이다.

정상성과 차분([6편](/posts/2026-10-17-time-series-6-stationarity-differencing)), AR([7편](/posts/2026-10-17-time-series-7-backshift-ar)), MA(이번 글)로 ARIMA의 재료가 모였다. 다음 글에서는 AR과 MA의 ACF와 PACF를 유도해 그림 1의 차이로 차수를 고르는 방법을 정리한다. 그 뒤에 차분과 AR, MA를 합친 ARIMA($p$, $d$, $q$)로 넘어간다.

---

**참고**

- FPP3 [9.4 Moving average models](https://otexts.com/fpp3/MA.html), [9.6 Estimation and order selection](https://otexts.com/fpp3/arima-estimation.html), [9.10 ARIMA vs ETS](https://otexts.com/fpp3/arima-ets.html)

[^band]: 그림 1의 점선은 white noise를 기준으로 한 $\pm 1.96/\sqrt{T}$다. MA(1)에서 시차 2 이상은 Bartlett 공식으로 분산이 $(1 + 2\rho_1^2)/T$라서 기준선을 $\pm 0.168$로 넓혀야 맞다. 시뮬레이션으로 구한 $T\operatorname{Var}(r_2)$는 1.42로 이론값 1.48에 가까웠다. 점선을 그대로 쓰면 시차 2 이후 막대 하나가 우연히 점선을 넘는 일이 생각보다 자주 생긴다.
