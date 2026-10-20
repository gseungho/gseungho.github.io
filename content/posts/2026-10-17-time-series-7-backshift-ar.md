---
title: "시계열분석 (7): Backshift 연산자와 AR 모형, 과거 값으로 현재를 설명하기"
date: "2026-10-17"
category: "공부"
subcategory: "Statistics/Time Series Analysis"
tags: ["시계열분석", "AR 모형", "backshift", "정상성"]
description: "시점을 한 칸 미는 backshift 연산자를 행렬로 보고 d번 차분한 식이 (1-B)^d인 이유를 보인다. AR 모형의 평균과 정상성 조건을 충격이 남는 양으로 유도한다. 계수 추정이 자기 자신에 대한 회귀라는 점도 정리한다."
---

세종대학교 인공지능데이터사이언스학과 2026년 2학기 시계열분석 수업을 따라가며 정리하는 시리즈의 일곱 번째 글이다. 수업 내용과 교재 [Forecasting: Principles and Practice, 3rd ed.](https://otexts.com/fpp3/)(이하 FPP3) 9.2~9.3절을 바탕으로 썼다.

> **AR 모형이 말하는 것: 오늘 값을 어제 값에 회귀하면 어떤 계열까지 설명할 수 있을까?**

[6편](/posts/2026-10-17-time-series-6-stationarity-differencing)에서 정상성과 차분을 정리했다. 이번 글은 차분을 짧게 쓰는 기호에서 시작해 정상 계열을 설명하는 첫 모형인 AR로 넘어간다. 계수 추정은 [3편](/posts/2026-10-15-time-series-3-trend-regression)의 회귀와 같은 계산이니 함께 보면 좋다.

## Backshift 연산자: 시점을 한 칸 미는 기호

Backshift 연산자 $B$는 자료를 한 시점 뒤로 민다.

$$
By_t = y_{t-1}, \qquad B(By_t) = B^2y_t = y_{t-2}, \qquad B^{12}y_t = y_{t-12}
$$

월별 자료에서 $B^{12}$는 작년 같은 달이다. [6편](/posts/2026-10-17-time-series-6-stationarity-differencing)의 차분을 $B$로 쓰면 이렇게 된다.

$$
\begin{aligned}
y'_t &= y_t - y_{t-1} = (1 - B)y_t \\
y''_t &= y_t - 2y_{t-1} + y_{t-2} = (1 - 2B + B^2)y_t = (1 - B)^2y_t
\end{aligned}
$$

**주의:** $(1 - B)^2$와 $(1 - B^2)$는 다르다. 앞은 1차 차분을 두 번 한 것이고 뒤는 2시점 전과의 차이 $y_t - y_{t-2}$를 한 번 구한 것이다. 계절 차분 $(1 - B^m)y_t = y_t - y_{t-m}$이 뒤쪽 꼴이다.

계절 차분 뒤 1차 차분은 두 다항식을 곱해 전개하면 된다.

$$
(1 - B)(1 - B^m)y_t = (1 - B - B^m + B^{m+1})y_t = y_t - y_{t-1} - y_{t-m} + y_{t-m-1}
$$

$B$의 다항식끼리는 곱하는 순서를 바꿔도 같다. 그래서 [6편](/posts/2026-10-17-time-series-6-stationarity-differencing)의 "두 차분은 어느 쪽을 먼저 해도 같다"가 곱셈의 교환법칙으로 바로 나온다.

### $B$를 행렬로 쓰면

관측값 $T$개를 벡터 $\mathbf{y} = [y_1, \dots, y_T]^\top$로 두면 $B$는 $T \times T$ 행렬이 된다. 시점 $t$의 값을 시점 $t + 1$ 자리로 옮기고 마지막 값은 밀어내는 행렬이다. $T = 4$로 써 보자.

$$
\mathbf{L} = \begin{bmatrix} 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}, \qquad \mathbf{L}\mathbf{y} = \begin{bmatrix} 0 \\ y_1 \\ y_2 \\ y_3 \end{bmatrix}
$$

대각선 바로 아래만 1인 아래 이동 행렬이다. 첫 성분이 0인 것은 관측하지 않은 $y_0$을 0으로 놓았다는 뜻이다[^circ]. 이 행렬로 보면 두 가지가 눈에 들어온다.

하나, $\mathbf{L}$은 멱영(nilpotent)이다. $\mathbf{L}^k$는 대각선 아래 $k$번째 줄만 1이고 $T$번 밀면 값이 모두 밀려 나가 $\mathbf{L}^T = \mathbf{O}$이다.

둘, 차분을 되돌리는 것은 누적합이다. $\mathbf{L}^T = \mathbf{O}$이므로

$$
(\mathbf{I} - \mathbf{L})(\mathbf{I} + \mathbf{L} + \cdots + \mathbf{L}^{T-1}) = \mathbf{I} - \mathbf{L}^T = \mathbf{I}
$$

이다. $(\mathbf{I} - \mathbf{L})^{-1}$은 대각선과 그 아래가 모두 1인 하삼각행렬이고 $(\mathbf{I} - \mathbf{L})^{-1}\mathbf{y}$의 $t$번째 성분은 $y_1 + \cdots + y_t$다. **ARIMA의 I(integrated)는 차분의 역연산인 누적합을 가리킨다.** 차분한 계열을 다시 쌓아 올려야 원래 계열이 된다.

### $d$번 차분은 왜 $(1 - B)^d$인가

$d$번 차분은 "앞에서 구한 차분 계열을 한 번 더 차분한다"로 정의한다.

$$
\nabla y_t = y_t - y_{t-1}, \qquad \nabla^d y_t = \nabla^{d-1}y_t - \nabla^{d-1}y_{t-1}
$$

어떤 수열이든 $\nabla$를 걸면 $(1 - B)$를 건 것과 같다. 정의의 둘째 식은 $(d-1)$번 차분한 계열에 $(1 - B)$를 한 번 더 건다는 말이라 귀납법으로 $\nabla^d = (1 - B)^d$이다. 남은 일은 전개다.

$$
(1 - B)^d = \sum_{k=0}^{d}(-1)^k\binom{d}{k}B^k \tag{1}
$$

$d = 1$이면 그대로다. $d$에서 식 (1)이 성립한다고 하고 $(1 - B)$를 한 번 더 곱해 보자.

$$
\begin{aligned}
(1 - B)^{d+1} &= \sum_{k=0}^{d}(-1)^k\binom{d}{k}B^k - \sum_{k=0}^{d}(-1)^k\binom{d}{k}B^{k+1} \\
&= \sum_{k=0}^{d+1}(-1)^k\left[\binom{d}{k} + \binom{d}{k-1}\right]B^k = \sum_{k=0}^{d+1}(-1)^k\binom{d+1}{k}B^k
\end{aligned}
$$

둘째 합은 $k + 1$을 새 $k$로 바꿨고 $\binom{d}{-1} = \binom{d}{d+1} = 0$으로 둔다. 마지막은 파스칼 법칙이다. $B^ky_t = y_{t-k}$이므로

$$
\nabla^d y_t = \sum_{k=0}^{d}(-1)^k\binom{d}{k}y_{t-k}
$$

이다. $d = 2$이면 $y_t - 2y_{t-1} + y_{t-2}$, $d = 3$이면 $y_t - 3y_{t-1} + 3y_{t-2} - y_{t-3}$이다.

행렬로 계산한 차분은 앞의 $d$개를 버린 $(T - d) \times T$ 행렬이고 $(\mathbf{I} - \mathbf{L})^d$의 마지막 $T - d$행과 같다. $T = 5$, $d = 2$이면

$$
\mathbf{D}_2 = \begin{bmatrix} 1 & -2 & 1 & 0 & 0 \\ 0 & 1 & -2 & 1 & 0 \\ 0 & 0 & 1 & -2 & 1 \end{bmatrix}
$$

이다. 행마다 첫 0 아닌 성분의 위치가 달라 계수(rank)가 $T - d$이고 차원정리로 영공간은 $d$차원이다. [6편](/posts/2026-10-17-time-series-6-stationarity-differencing)에서 $d$번 차분하면 $d - 1$차 이하 다항식이 0이 된다고 했다. 그 다항식들이 정확히 이 영공간이다. **$d$번 차분은 $d - 1$차 이하 추세만 정확히 지운다.** 지워진 $d$차원(수준, 기울기, …)은 차분 계열만 보고는 되찾을 수 없다. 그래서 차분 계열의 예측을 원래 단위로 되돌리려면 마지막 관측값 $d$개가 필요하다.

## AR 모형: 자기 자신에 대한 회귀

이제 모형으로 넘어가자. AR($p$) 모형은 지금 값을 과거 $p$개 값의 선형결합으로 설명한다.

$$
y_t = c + \phi_1y_{t-1} + \phi_2y_{t-2} + \cdots + \phi_py_{t-p} + \varepsilon_t \tag{2}
$$

$\varepsilon_t$는 white noise다. AR은 autoregressive의 줄임말로 자기 자신의 과거 값을 설명변수로 쓰는 회귀라는 뜻이다. [1편](/posts/2026-10-15-time-series-1-forecasting-basics)의 $y_t = ay_{t-1} + b + \varepsilon_t$가 바로 AR(1)이다.

$\phi_1$ 하나로 AR(1)의 성격이 정해진다.

| $\phi_1$ | 계열 |
| --- | --- |
| $0$ | white noise |
| $1$ | random walk ($c \ne 0$이면 drift) |
| $\lvert\phi_1\rvert < 1$ | 정상 |
| $< 0$ | 평균 위아래로 번갈아 튄다 |

<Fig src="/images/posts/2026-10-17-time-series-7-backshift-ar/ar1-paths.svg" alt="같은 충격으로 만든 AR(1) 세 계열, φ = 0.5, 0.95, −0.8">그림 1. 같은 충격 $\varepsilon$로 만든 AR(1) 세 개. $\phi_1$이 1에 가까울수록 평균에서 오래 떠돌고 음수이면 매 시점 부호가 바뀌며 튄다.</Fig>

그림 1에서 보듯 $\phi_1$이 1에 가까울수록 계열이 평균에서 위아래로 오래 떠돈다. [6편](/posts/2026-10-17-time-series-6-stationarity-differencing)에서 $\phi_1 = 0.99$인 AR(1)이 random walk와 비슷해 보였던 것과 같은 현상이다.

### 평균은 c가 아니다

식 (2)의 $c$는 절편이지 평균이 아니다. 정상이면 $E(y_t) = E(y_{t-1}) = \mu$이므로 양변에 기댓값을 취해 풀 수 있다.

$$
\mu = c + \phi_1\mu + \cdots + \phi_p\mu \quad\Rightarrow\quad \mu = \frac{c}{1 - \phi_1 - \phi_2 - \cdots - \phi_p}
$$

$c = 2$, $\phi_1 = 0.8$이면 평균은 $2/0.2 = 10$이다. 이 관계를 $c = \mu(1 - \phi_1 - \cdots - \phi_p)$로 넣으면 AR(1)을 평균을 뺀 꼴로 쓸 수 있다.

$$
y_t - \mu = \phi_1(y_{t-1} - \mu) + \varepsilon_t \tag{3}
$$

**주의:** 이 유도는 평균이 시점과 상관없다는 정상성을 쓴다. $\phi_1 + \cdots + \phi_p = 1$이면 분모가 0이다. drift가 있는 random walk에는 고정된 평균이 없다.

## 정상성 조건: 충격이 사라져야 한다

AR 모형이 정상이 되려면 계수가 어떤 조건을 만족해야 할까? 충격 하나가 얼마나 오래 남는지 따라가 보자.

AR(1)에서 시점 $t$에 들어온 충격 1은 다음 시점에 $\phi_1$, 그다음에 $\phi_1^2$, $h$시점 뒤에 $\phi_1^h$만큼 남는다. 이것이 0으로 사라지려면 $\lvert\phi_1\rvert < 1$이어야 한다. AR(2) $y_t = \phi_1y_{t-1} + \phi_2y_{t-2} + \varepsilon_t$에서는 충격이 남는 양 $d_h$가 점화식을 따른다($d_1$은 충격 직후).

$$
d_1 = 1, \quad d_2 = \phi_1, \quad d_3 = \phi_1^2 + \phi_2, \quad \dots, \quad d_h = \phi_1d_{h-1} + \phi_2d_{h-2} \tag{4}
$$

$d_h = r^h$ 꼴의 해를 찾아보자. 식 (4)에 넣고 $r^{h-2}$로 나누면

$$
r^2 - \phi_1r - \phi_2 = 0 \quad\xrightarrow{\;x = 1/r\;}\quad \Phi(x) = 1 - \phi_1x - \phi_2x^2 = 0
$$

이다. 충격이 사라지려면 $\lvert r\rvert < 1$, 곧 $\lvert x\rvert = \lvert 1/r\rvert > 1$이어야 한다. $\Phi(x)$를 특성다항식이라 한다.

### $r^h$로 놓아도 되는가

여기서 의문이 생긴다. 실제 충격이 $r^h$ 꼴이라고 가정해도 될까? 된다. **$r^h$는 실제 충격의 모양을 가정한 것이 아니라 점화식의 해를 만드는 재료를 찾은 것이다.**

식 (4)를 만족하는 수열은 처음 두 값 $(d_1, d_2)$를 정하면 나머지가 모두 정해진다. 그래서 해들은 2차원 공간을 이루고 서로 독립인 해 두 개만 찾으면 모든 해를 그 둘의 일차결합으로 쓸 수 있다. 근 $r_1 \ne r_2$가 다르면 $r_1^h$와 $r_2^h$가 그 두 개다. 실제 충격도 $d_h = c_1r_1^h + c_2r_2^h$로 쓸 수 있고 $d_1 = 1$, $d_2 = \phi_1 = r_1 + r_2$를 넣어 풀면

$$
d_h = \frac{r_1^h - r_2^h}{r_1 - r_2}
$$

이다. $\phi = (0.5, 0.3)$이면 $r = 0.852, -0.352$이고 이 식과 식 (4)가 모두 $1, 0.5, 0.55, 0.425, 0.378, \dots$을 준다.

<Fig src="/images/posts/2026-10-17-time-series-7-backshift-ar/shock-response.svg" alt="AR(2) 네 경우의 충격이 남는 양 d_h">그림 2. AR(2)에서 충격 1이 h시점 뒤에 남는 양. 역근이 모두 단위원 안에 있으면 0으로 사라지고(위), 역근 1이 있으면 1.25에 머물며 1보다 큰 역근이 있으면 폭발한다(아래).</Fig>

근의 모양에 따라 그림 2처럼 세 경우가 있다.

- 서로 다른 실근: 위 식 그대로 지수적으로 줄어든다.
- 중근($\phi_1^2 + 4\phi_2 = 0$): $r^h$ 하나로는 재료가 모자라 $hr^h$를 더 쓴다. 실제 충격은 $d_h = hr^{h-1}$이고 $\lvert r\rvert < 1$이면 지수적 감소가 $h$보다 빨라 0으로 간다.
- 복소근($\phi_1^2 + 4\phi_2 < 0$): $r = \rho e^{\pm i\theta}$이고 정리하면 $d_h = \rho^{h-1}\sin(h\theta)/\sin\theta$다. $\rho < 1$이면 진동하며 줄어든다. $\phi = (1, -0.5)$이면 $1, 1, 0.5, 0, -0.25, -0.25, \dots$이다.

반대 방향도 성립한다. 위 식에서 두 근의 계수가 $\pm 1/(r_1 - r_2)$로 둘 다 0이 아니므로, 어느 한 근이라도 $\lvert r\rvert \ge 1$이면 충격이 0으로 가지 않는다. $\phi = (1.2, -0.2)$는 $r = 1, 0.2$라서 충격이 $1/(1 - 0.2) = 1.25$에 영구히 남는다. [6편](/posts/2026-10-17-time-series-6-stationarity-differencing)의 단위근이 이것이다. 일반적인 정리는 다음과 같다.

**AR($p$)가 정상일 필요충분조건은 특성다항식 $\Phi(x) = 1 - \phi_1x - \cdots - \phi_px^p$의 근이 모두 단위원 밖($\lvert x\rvert > 1$)에 있는 것이다.**[^causal]

AR(2)에서 이 조건을 계수로 풀면 세 부등식이 된다.

$$
-1 < \phi_2 < 1, \qquad \phi_1 + \phi_2 < 1, \qquad \phi_2 - \phi_1 < 1
$$

<Fig src="/images/posts/2026-10-17-time-series-7-backshift-ar/ar2-region.svg" alt="AR(2)가 정상인 계수 영역 삼각형과 실근, 복소근의 경계">그림 3. AR(2)가 정상인 $(\phi_1, \phi_2)$ 영역은 삼각형이다. 점선 아래는 복소근이라 충격이 진동하며 줄어든다. 점 네 개는 그림 2의 네 경우다.</Fig>

근이 $x = 1$이나 $x = -1$을 지나 단위원 안으로 들어오지 않으려면 $\Phi(1) = 1 - \phi_1 - \phi_2 > 0$, $\Phi(-1) = 1 + \phi_1 - \phi_2 > 0$이어야 한다. 이것이 삼각형의 두 빗변이다. 역근의 곱이 $-\phi_2$라서 $\lvert\phi_2\rvert < 1$도 필요하다. **주의:** $-1 < \phi_2 < 1$과 $-2 < \phi_1 < 2$만으로는 부족하다. $\phi = (1.5, 0.5)$는 둘 다 만족하지만 $\phi_1 + \phi_2 = 2$라 삼각형 밖이다. 그림 2의 넷째 칸처럼 폭발한다.

선형대수로 보면 이 이야기가 한 줄로 정리된다. 식 (4)를 벡터로 쓰면

$$
\begin{bmatrix} d_h \\ d_{h-1} \end{bmatrix} = \mathbf{A}\begin{bmatrix} d_{h-1} \\ d_{h-2} \end{bmatrix}, \qquad \mathbf{A} = \begin{bmatrix} \phi_1 & \phi_2 \\ 1 & 0 \end{bmatrix}
$$

이고 $\mathbf{A}$의 특성다항식이 $\lambda^2 - \phi_1\lambda - \phi_2$다. 위에서 푼 $r$이 바로 $\mathbf{A}$의 고윳값이다. $\mathbf{A}$를 대각화하면 $\mathbf{A}^n$이 고윳값의 $n$제곱으로 계산되니 $r^h$가 나온다. 충격이 사라진다는 것은 $\mathbf{A}^n \to \mathbf{O}$, 곧 고윳값의 절댓값이 모두 1보다 작다는 것이다. 피보나치 수열을 행렬로 푸는 것과 같은 계산이다.

## 계수를 추정하는 회귀

계수는 어떻게 추정할까? 식 (3)의 평균을 뺀 꼴로 1-step 예측오차의 제곱합을 최소로 한다.

$$
S_c(\phi, \mu) = \sum_{t=2}^{n}\big[(Y_t - \mu) - \phi(Y_{t-1} - \mu)\big]^2
$$

첫 관측값 $Y_1$을 주어진 값으로 두고 $t = 2$부터 더한다고 해서 조건부(conditional) 최소제곱이라 부른다. $\mu$로 미분해 0으로 두면 $\hat\mu$가, $\phi$로 미분해 0으로 두면 $\hat\phi$가 나온다.

### 왜 $\mu$를 넣는가

$\mu$ 없이 $Y_t = \phi Y_{t-1} + \varepsilon_t$로 SSE를 구해 $\phi$만 추정하면 안 될까? 그 식은 평균을 0으로 정해 버린 다른 모형이다. 평균이 0이 아닌 계열에 쓰면 원점을 지나는 회귀 $\hat\phi_0 = \sum Y_tY_{t-1}/\sum Y_{t-1}^2$이 된다. 대수의 법칙으로 이 값은

$$
\hat\phi_0 \;\to\; \frac{E(Y_tY_{t-1})}{E(Y_{t-1}^2)} = \frac{\gamma_1 + \mu^2}{\gamma_0 + \mu^2}
$$

에 가까워진다. $\phi = 0.8$, $\mu = 5$, $\sigma = 1$이면 $\gamma_0 = 2.78$, $\gamma_1 = 2.22$라서 0.980이다. 참값 0.8이 아니라 단위근 근처다. 표본 크기 50, 200, 1000으로 1만 번씩 해 봐도 평균이 0.979, 0.980, 0.980으로 줄지 않았다.

<Fig src="/images/posts/2026-10-17-time-series-7-backshift-ar/lag-scatter.svg" alt="시차 산점도에 절편 있는 회귀선과 원점을 지나는 회귀선">그림 4. $\phi = 0.8$, $\mu = 5$인 AR(1) 200개의 시차 산점도. 원점에 묶인 직선은 (5, 5) 근처의 점구름을 지나려고 기울기가 1 가까이 된다.</Fig>

그림 4를 보면 이유가 보인다. 점 $(Y_{t-1}, Y_t)$는 $(\mu, \mu) = (5, 5)$ 근처에 모여 있다. 원점에서 출발해 그 점구름을 지나는 직선은 기울기가 1에 가까울 수밖에 없다. $(5, 5)$가 대각선 $y = x$ 위에 있기 때문이다. **절편($\mu$)을 빼면 기울기가 수준을 떠안아 1 쪽으로 치우친다.**

### 회귀로 보는 $\hat\phi$

$\hat\phi$는 $Y_2, \dots, Y_n$을 $Y_1, \dots, Y_{n-1}$에 절편과 함께 회귀한 OLS 기울기다. [3편](/posts/2026-10-15-time-series-3-trend-regression)의 단순회귀 공식을 그대로 쓰면

$$
\hat\phi = \frac{\sum_{t=2}^{n}(Y_t - \bar{Y}_{(2)})(Y_{t-1} - \bar{Y}_{(1)})}{\sum_{t=2}^{n}(Y_{t-1} - \bar{Y}_{(1)})^2}, \qquad \hat{c} = \bar{Y}_{(2)} - \hat\phi\bar{Y}_{(1)}
$$

이다. $\bar{Y}_{(2)}$는 반응변수 $Y_2, \dots, Y_n$의 평균, $\bar{Y}_{(1)}$은 설명변수 $Y_1, \dots, Y_{n-1}$의 평균이다. $S_c$를 $(\phi, \mu)$로 최소화한 결과는 이 OLS와 정확히 같다. $(c, \phi)$와 $(\mu, \phi)$가 $c = \mu(1 - \phi)$로 일대일 대응하기 때문이다. 그림 4의 표본에서 OLS는 $\hat\phi = 0.833$, $\hat{c} = 0.854$이고 $\hat{c}/(1 - \hat\phi) = 5.109$가 $S_c$를 수치로 최소화한 $\hat\mu$와 같았다. 두 평균 자리에 모두 $\bar{Y}$를 넣은 근사식도 0.833으로 같았다.

기울기는 상관계수에 두 표준편차의 비를 곱한 값이다. 같은 계열이라 두 표준편차가 거의 같으니 $\hat\phi$는 [6편](/posts/2026-10-17-time-series-6-stationarity-differencing)의 $r_1$과 거의 같다. AR(1)에서 $\rho_1 = \phi_1$이니 앞뒤가 맞는다.

보통의 회귀와 다른 점도 있다. 설명변수 $Y_{t-1}$은 고정된 값이 아니라 과거 충격으로 만들어진 확률변수다. 현재 충격 $\varepsilon_t$와는 독립이지만 $\varepsilon_t$가 다음 시점의 설명변수 $Y_t$에 들어간다. 그래서 회귀분석의 비편향성이 성립하지 않고 작은 표본에서 $\hat\phi$가 아래로 치우친다.

| 표본 크기 $n$ | 50 | 200 | 1000 |
| --- | --- | --- | --- |
| $\hat\phi$의 평균 (1만 번, 참값 0.8) | 0.726 | 0.783 | 0.797 |
| 근사식 $\phi - (1 + 3\phi)/n$[^bias] | 0.732 | 0.783 | 0.797 |

치우침은 $1/n$ 속도로 사라지고 $\hat\phi$는 참값으로 수렴한다. 표본이 크면 회귀분석의 표준오차와 $t$검정을 근사적으로 그대로 쓸 수 있다.

## AR 모형으로 예측하기

AR(1)의 예측은 식 (3)에서 모르는 미래 충격을 기댓값 0으로 두면 된다.

$$
\hat{y}_{T+h|T} = \mu + \phi_1^h(y_T - \mu)
$$

$\lvert\phi_1\rvert < 1$이면 $h$가 커질수록 예측이 평균 $\mu$로 돌아간다. 마지막 관측값이 평균에서 벗어난 만큼이 $\phi_1^h$로 줄어드는 셈이다. 예측오차의 분산은 $\sigma^2\sum_{j=0}^{h-1}\phi_1^{2j}$이고 $\sigma^2/(1 - \phi_1^2)$에 다가간다. [1편](/posts/2026-10-15-time-series-1-forecasting-basics) 그림 1의 부채꼴이 AR(1)의 예측이었다. [4편](/posts/2026-10-15-time-series-4-exponential-smoothing)의 SES는 예측이 평평했고 random walk의 예측은 마지막 값에 머물렀다. **정상 AR의 예측은 평균으로 돌아간다는 점이 다르다.**

[8편](/posts/2026-10-17-time-series-8-ma-invertibility)에서는 과거 값 대신 과거 충격으로 현재를 설명하는 MA 모형을 다룬다. 이번 글의 충격이 남는 양 $\phi_1^h$가 거기서 "정상 AR은 MA($\infty$)다"라는 식으로 다시 나온다.

---

**참고**

- FPP3 [9.2 Backshift notation](https://otexts.com/fpp3/backshift.html), [9.3 Autoregressive models](https://otexts.com/fpp3/AR.html)
- Sørbye, Nicolau, Rue (2020), [Model-based bias correction for short AR(1) and AR(2) processes](https://arxiv.org/abs/2010.05870), arXiv:2010.05870

[^circ]: 마지막 값을 맨 앞으로 돌려 넣는 순환 이동 행렬도 있다. 고윳값과 고유벡터가 깔끔해 푸리에 분석에 쓰지만 $y_0$ 자리에 $y_T$가 들어와 차분에는 맞지 않는다.

[^causal]: 엄밀하게는 과거 충격만으로 쓸 수 있는(인과적, causal) 정상 해가 있을 필요충분조건이다. 근이 단위원 위에 있으면 정상 해가 없다. 근이 단위원 안에 있으면 미래 충격으로 쓰는 정상 해가 따로 있지만 예측에는 쓸 수 없어 정상으로 치지 않는다.

[^bias]: 표본 크기 $n$에서 AR(1) 계수 추정량의 점근 편향 $-(1 + 3\phi)/n$이다. Sørbye, Nicolau, Rue (2020) 4쪽이 Tanaka (1984), Cordeiro and Klein (1994)를 인용해 적은 식이다. 표는 이 식과 내 시뮬레이션을 비교한 것이다.
