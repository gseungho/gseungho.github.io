---
title: "시계열분석 (9): ACF와 PACF, 막대 그림으로 AR과 MA의 차수 읽기"
date: "2026-10-21"
category: "공부"
subcategory: "Statistics/Time Series Analysis"
tags: ["시계열분석", "ACF", "PACF", "Yule-Walker"]
description: "AR과 MA 모형의 ACF와 PACF를 직접 유도한다. 중간 시차의 영향을 걷어 낸 상관인 PACF를 회귀 계수로 읽고 Yule-Walker 방정식과 Durbin-Levinson 점화식으로 계산한다. 두 그림으로 차수를 고르는 규칙과 그 한계도 정리한다."
---

세종대학교 인공지능데이터사이언스학과 2026년 2학기 시계열분석 수업을 따라가며 정리하는 시리즈의 아홉 번째 글이다. 수업 내용과 교재 [Forecasting: Principles and Practice, 3rd ed.](https://otexts.com/fpp3/)(이하 FPP3) 9.5절을 바탕으로 썼다.

> **ACF와 PACF가 말하는 것: 막대 그림 두 장만 보고 AR과 MA의 차수를 알 수 있을까?**

[7편](/posts/2026-10-17-time-series-7-backshift-ar)과 [8편](/posts/2026-10-17-time-series-8-ma-invertibility)에서 AR과 MA를 따로 보았다. [8편](/posts/2026-10-17-time-series-8-ma-invertibility) 그림 1에서 MA(1)의 ACF는 시차 1 뒤에 끊겼고 AR(1)의 ACF는 천천히 줄었다. 이번 글은 그 모양을 식으로 유도한다. AR의 차수는 ACF로 잘 보이지 않아서 PACF라는 그림이 하나 더 필요하다. 먼저 기호부터 정리하자.

## 참값과 표본값의 기호

| 기호 | 뜻 | 정의 |
| --- | --- | --- |
| $\gamma_k$ | 자기공분산 | $\operatorname{Cov}(y_t, y_{t-k})$, 특히 $\gamma_0 = \operatorname{Var}(y_t)$ |
| $\rho_k$ | 자기상관계수 (ACF) | $\gamma_k/\gamma_0$ |
| $r_k$ | 표본 자기상관계수 | 자료로 계산한 $\rho_k$의 추정값 |
| $\alpha_k$ | 편자기상관계수 (PACF) | 아래에서 정의한다 |

그리스 문자는 모형이 정하는 참값이고 로마자는 자료로 계산한 값이다. $\mu$와 $\bar{y}$의 관계와 같다. [6편](/posts/2026-10-17-time-series-6-stationarity-differencing)의 ACF 그림에 그린 막대는 $r_k$이고 이번 글에서 유도하는 것은 $\rho_k$다.

계산은 평균이 0인 계열로 한다. 평균이 $\mu$인 계열이라면 $y_t - \mu$를 새 계열로 두면 된다. 평균이 0이면 공분산이 곱의 기댓값과 같아서 $\gamma_k = E(y_ty_{t-k})$로 쓸 수 있다[^mean].

## PACF: 중간 시차를 걷어 낸 상관

ACF만으로 부족한 이유부터 보자. AR(1) $y_t = \phi y_{t-1} + \varepsilon_t$에서 $y_t$를 만드는 데 직접 쓰이는 과거 값은 $y_{t-1}$ 하나다. 그런데 $y_{t-1}$은 $y_{t-2}$와 상관되어 있다. 그러면 $y_t$와 $y_{t-2}$도 둘 다 $y_{t-1}$과 이어져 있다는 이유만으로 상관이 생긴다. $\phi = 0.7$이면 $\rho_2 = 0.49$다. ACF는 이렇게 건너온 상관까지 모두 센다. **그래서 AR의 ACF는 차수 뒤에도 끊기지 않고 천천히 줄어든다.**

그렇다면 중간 시차를 거쳐 건너온 몫을 걷어 내면 어떨까? $y_t$와 $y_{t-2}$ 각각에서 $y_{t-1}$로 설명되는 몫을 회귀로 빼고 남은 잔차끼리 상관을 구해 보자.

<Fig src="/images/posts/2026-10-21-time-series-9-acf-pacf/partial-correlation.svg" alt="AR(1)에서 y_t와 y_t−2의 산점도, y_t−1로 설명한 몫을 뺀 잔차끼리의 산점도">그림 1. $\phi = 0.7$인 AR(1) 300개. 왼쪽은 $y_{t-2}$와 $y_t$의 상관이 0.49다. 오른쪽처럼 둘에서 $y_{t-1}$로 설명되는 몫을 빼고 나면 상관이 0.00이다.</Fig>

그림 1 오른쪽에서 잔차끼리는 상관이 없다. $y_{t-2}$가 $y_t$에 주는 정보는 모두 $y_{t-1}$을 거쳐 온 것이었다는 뜻이다. 이렇게 시차 $1, \dots, k-1$의 영향을 걷어 낸 뒤 남는 $y_t$와 $y_{t-k}$의 상관을 시차 $k$의 편자기상관(partial autocorrelation)이라 하고 $\alpha_k$로 쓴다. 시차마다 막대로 그린 것이 PACF다.

계산은 회귀로 하는 편이 쉽다. FPP3 9.5절은 $\alpha_k$가 AR($k$) 모형에서 마지막 계수 $\phi_k$의 추정값과 같다고 적는다. $y_t$를 $y_{t-1}, \dots, y_{t-k}$에 회귀한 다중회귀에서 $y_{t-k}$의 계수다. 다중회귀 계수는 다른 설명변수를 고정했을 때의 효과이니 "중간 시차를 걷어 낸다"는 뜻과 맞는다. 그림 1의 자료로 AR(2)를 OLS로 맞추면 $y_{t-2}$의 계수가 0.002로 잔차끼리의 상관 0.002와 같다[^pacf-def].

AR(1)이라면 $\alpha_1 = \phi$이고 $k \ge 2$에서는 $\alpha_k = 0$일 것이다. 계산으로 확인하려면 계수와 자기상관을 잇는 식이 필요하다. AR(1)의 ACF부터 유도해 보자.

## AR(1)의 ACF 유도

평균이 0이고 $\lvert\phi\rvert < 1$인 AR(1) $y_t = \phi y_{t-1} + \varepsilon_t$의 양변에 $y_{t-k}$($k \ge 1$)를 곱하고 기댓값을 취한다.

$$
E(y_ty_{t-k}) = \phi E(y_{t-1}y_{t-k}) + E(\varepsilon_ty_{t-k}) \tag{1}
$$

세 항을 하나씩 보자.

- 왼쪽은 정의대로 $\gamma_k$다.
- 오른쪽 첫 항의 두 시점은 $(t-1) - (t-k) = k-1$만큼 떨어져 있다. 정상 계열의 공분산은 두 시점이 어디 있는지와 상관없이 간격에만 의존한다. 그래서 두 시점을 함께 한 칸 밀어도 같고 $E(y_{t-1}y_{t-k}) = \gamma_{k-1}$이다.
- 마지막 항은 0이다. 이 항이 핵심이다.

$y_{t-k}$는 [8편](/posts/2026-10-17-time-series-8-ma-invertibility)의 식 (2)대로 충격 $\varepsilon_{t-k}, \varepsilon_{t-k-1}, \dots$로만 만들어진다. $\varepsilon_t$는 그 뒤에 새로 들어온 충격이라 $y_{t-k}$와 독립이다. 그래서 $E(\varepsilon_ty_{t-k}) = E(\varepsilon_t)E(y_{t-k}) = 0$이다. **과거 값은 지금의 충격과 독립이다.** 평균이 0이라서 기댓값이 쪼개지는 것은 아니다. 같은 계열에서도 $y_t$ 안에는 $\varepsilon_t$가 그대로 들어 있어서 $E(\varepsilon_ty_t) = \sigma^2$이다. $\phi = 0.7$로 20만 개를 만들어 보면 $y_{t-1}\varepsilon_t$의 평균은 $-0.0025$이고 $y_t\varepsilon_t$의 평균은 0.997이다.

정리하면 식 (1)은 $\gamma_k = \phi\gamma_{k-1}$이고 $k = 0$까지 거슬러 올라가면

$$
\gamma_k = \phi^k\gamma_0, \qquad \rho_k = \phi^k
$$

이다. $\gamma_0$은 양변의 분산을 취해 구한다. $\gamma_0 = \phi^2\gamma_0 + \sigma^2$에서 $\gamma_0 = \sigma^2/(1 - \phi^2)$다.

**주의:** $\sigma^2$은 충격 $\varepsilon_t$의 분산이고 $\gamma_0$은 계열 $y_t$의 분산이다. $\phi = 0.7$, $\sigma^2 = 1$이면 $\gamma_0 = 1.96$으로 충격 분산의 두 배 가까이 된다.

## Yule-Walker 방정식

같은 계산을 AR($p$) $y_t = \phi_1y_{t-1} + \cdots + \phi_py_{t-p} + \varepsilon_t$에 하면 된다. $y_{t-k}$를 곱하고 기댓값을 취하면 충격 항이 같은 이유로 0이 되고 나머지는 모두 자기공분산이다. $\gamma_0$으로 나누면

$$
\rho_k = \phi_1\rho_{k-1} + \phi_2\rho_{k-2} + \cdots + \phi_p\rho_{k-p} \qquad (k \ge 1) \tag{2}
$$

이다. 시차가 음수인 항은 $\rho_{-j} = \rho_j$로 바꾼다. $k = 1, \dots, p$의 $p$개 식을 행렬로 모은 것이 Yule-Walker 방정식이다.

$$
\begin{pmatrix} 1 & \rho_1 & \cdots & \rho_{p-1} \\ \rho_1 & 1 & \cdots & \rho_{p-2} \\ \vdots & & \ddots & \vdots \\ \rho_{p-1} & \rho_{p-2} & \cdots & 1 \end{pmatrix}
\begin{pmatrix} \phi_1 \\ \phi_2 \\ \vdots \\ \phi_p \end{pmatrix}
= \begin{pmatrix} \rho_1 \\ \rho_2 \\ \vdots \\ \rho_p \end{pmatrix} \tag{3}
$$

왼쪽 행렬을 $\mathbf{R}_p$라 하자. $k = 0$이면 $E(\varepsilon_ty_t) = \sigma^2$이 남아서 $\sigma^2 = \gamma_0(1 - \phi_1\rho_1 - \cdots - \phi_p\rho_p)$가 된다. 이 방정식은 세 방향으로 쓰인다.

### 계수에서 ACF로

계수를 알면 식 (2)로 $\rho$를 차례로 구한다. $\phi = (1, -0.5)$인 AR(2)라면 $k = 1$의 식 $\rho_1 = \phi_1 + \phi_2\rho_1$에서 $\rho_1 = \phi_1/(1 - \phi_2) = 0.667$이다. 그다음은 점화식이다.

$$
\rho_2 = 0.667 - 0.5 = 0.167, \qquad \rho_3 = 0.167 - 0.333 = -0.167, \qquad \rho_4 = -0.167 - 0.083 = -0.25
$$

식 (2)는 $k > p$에서도 계속 성립하니 AR의 ACF는 시차 $p$에서 끊기지 않는다. 점화식의 모양이 [7편](/posts/2026-10-17-time-series-7-backshift-ar)에서 충격이 남는 양을 따라간 식 (4)와 같다. 그래서 $\rho_k$도 특성다항식 역근의 거듭제곱의 일차결합이다. 역근이 실수이면 지수적으로 줄고 복소수이면 진동하며 줄어든다. $\phi = (1, -0.5)$는 역근이 복소수라 위처럼 부호가 바뀐다.

### ACF에서 계수로

거꾸로 $\rho$ 자리에 표본값 $r$을 넣고 식 (3)을 풀면 계수의 추정값이 나온다. Yule-Walker 추정이다. $\phi = (1, -0.5)$로 200개를 만들어 보니 $r_1 = 0.604$, $r_2 = 0.093$에서 추정값이 $(0.861, -0.427)$이었다. 절편을 넣은 OLS는 $(0.876, -0.438)$로 거의 같다.

우연이 아니다. $y_t$를 $y_{t-1}, \dots, y_{t-p}$에 회귀하는 OLS의 정규방정식 $\mathbf{X}^\top\mathbf{X}\hat{\boldsymbol\phi} = \mathbf{X}^\top\mathbf{y}$를 $T$로 나눠 보자. $\mathbf{X}^\top\mathbf{X}/T$의 $(i, j)$ 성분은 $\frac{1}{T}\sum_t y_{t-i}y_{t-j} \approx \gamma_{\lvert i-j\rvert}$이고 $\mathbf{X}^\top\mathbf{y}/T$의 $i$번째 성분은 $\approx \gamma_i$다. **Yule-Walker 방정식은 OLS 정규방정식에서 표본평균을 기댓값으로 바꾼 식이다.** 두 방법의 차이는 합의 범위와 양 끝을 다루는 방식에서만 생긴다.

### PACF가 끊기는 이유

$\alpha_k$는 AR($k$)를 맞춘 마지막 계수였다. 그러니 차수 $k$의 Yule-Walker 방정식 $\mathbf{R}_k\boldsymbol\alpha = (\rho_1, \dots, \rho_k)^\top$을 풀어 마지막 성분을 읽으면 된다. $k = 2$이면 두 식 $\rho_1 = \alpha_{2,1} + \alpha_{2,2}\rho_1$, $\rho_2 = \alpha_{2,1}\rho_1 + \alpha_{2,2}$에서 첫 식의 $\alpha_{2,1} = \rho_1(1 - \alpha_{2,2})$를 둘째 식에 넣어

$$
\alpha_2 = \alpha_{2,2} = \frac{\rho_2 - \rho_1^2}{1 - \rho_1^2}
$$

이다. AR(1)에 $\rho_k = \phi^k$를 넣으면 $\alpha_1 = \rho_1 = \phi$이고 $\alpha_2 = (\phi^2 - \phi^2)/(1 - \phi^2) = 0$이다.

AR($p$)에서 $k > p$이면 어떻게 될까? $(\phi_1, \dots, \phi_p, 0, \dots, 0)$을 차수 $k$의 방정식에 넣어 보면 그대로 성립한다. 식 (2)가 모든 $k \ge 1$에서 성립하기 때문이다. 정상인 AR이면 $\mathbf{R}_k$가 가역이라 해가 하나뿐이니 그 해의 마지막 성분 $\alpha_k$는 0이다. **AR($p$)의 PACF는 시차 $p$ 뒤에 끊긴다.**

### Durbin-Levinson 점화식

시차마다 방정식을 새로 풀 필요는 없다. $\alpha_{n,k}$를 AR($n$)을 맞췄을 때의 $k$번째 계수라 하자. PACF는 그 마지막 값 $\alpha_n = \alpha_{n,n}$이다. 차수를 하나씩 키우며 다음 두 식을 번갈아 쓰면 된다. Durbin-Levinson 점화식이라 부른다.

$$
\alpha_{n,n} = \frac{\rho_n - \sum_{k=1}^{n-1}\alpha_{n-1,k}\,\rho_{n-k}}{1 - \sum_{k=1}^{n-1}\alpha_{n-1,k}\,\rho_k}, \qquad \alpha_{n,k} = \alpha_{n-1,k} - \alpha_{n,n}\,\alpha_{n-1,n-k} \quad (1 \le k \le n-1) \tag{4}
$$

$\alpha_{1,1} = \rho_1$에서 출발해 첫 식으로 $\alpha_{n,n}$을 구하고 둘째 식으로 나머지 계수를 업데이트한다. 앞의 계수들은 PACF로 쓰지 않지만 다음 차수를 계산하려면 들고 가야 한다[^dl]. $\phi = (1, -0.5)$로 해 보자.

| $n$ | 첫 식 (분자 / 분모) | 결과 |
| --- | --- | --- |
| 1 | $\rho_1$ | $\alpha_{1,1} = 0.667$ |
| 2 | $(\rho_2 - \alpha_{1,1}\rho_1)/(1 - \alpha_{1,1}\rho_1) = -0.278/0.556$ | $\alpha_{2,2} = -0.5$, $\alpha_{2,1} = 0.667 \times 1.5 = 1$ |
| 3 | $(\rho_3 - \alpha_{2,1}\rho_2 - \alpha_{2,2}\rho_1)/(1 - \alpha_{2,1}\rho_1 - \alpha_{2,2}\rho_2) = 0/0.417$ | $\alpha_{3,3} = 0$, $(\alpha_{3,1}, \alpha_{3,2}) = (1, -0.5)$ |

$n = 2$에서 계수 전체가 참값 $(1, -0.5)$가 되고 $n = 3$부터 PACF가 0이다. $\alpha_{1,1} = 0.667$은 $\phi_1 = 1$과 다르다. PACF의 $\alpha_k$는 AR($k$)를 새로 맞춘 마지막 계수라서 $k = p$일 때만 모형의 $\phi_k$와 같다.

## MA(1)의 ACF와 PACF

이번에는 MA(1) $y_t = \varepsilon_t + \theta\varepsilon_{t-1}$이다. 같은 방법으로 양변에 $y_{t-1}$을 곱해 보자.

$$
\gamma_1 = E(\varepsilon_ty_{t-1}) + \theta E(\varepsilon_{t-1}y_{t-1})
$$

첫 항은 AR처럼 0이다. $y_{t-1} = \varepsilon_{t-1} + \theta\varepsilon_{t-2}$는 과거 충격만 담고 있으니 $\varepsilon_t$와 독립이다. MA는 처음부터 충격 유한 개의 합이라 $\theta$가 무엇이든 그렇다. 둘째 항은 다르다. 오른쪽의 $\varepsilon_{t-1}$이 $y_{t-1}$ 안에도 들어 있어서 $E(\varepsilon_{t-1}y_{t-1}) = \sigma^2$이다. $\theta = 0.6$, $\sigma^2 = 1$로 확인해 보자.

| 짝 | 기댓값 | 시뮬레이션 |
| --- | --- | --- |
| $y_{t-1}$과 $\varepsilon_t$ | 0 (독립) | $-0.004$ |
| $y_{t-1}$과 $\varepsilon_{t-1}$ | $\sigma^2$ | 0.998 |
| $y_t$와 $\varepsilon_{t-1}$ | $\theta\sigma^2$ | 0.598 |

그래서 $\gamma_1 = \theta\sigma^2$이다. 두 칸 이상 떨어지면 함께 담은 충격이 없어 $\gamma_k = 0$이고 $\gamma_0 = \sigma^2(1 + \theta^2)$이니

$$
\rho_1 = \frac{\theta}{1 + \theta^2}, \qquad \rho_k = 0 \quad (k \ge 2)
$$

이다. **MA($q$)의 ACF는 시차 $q$ 뒤에 끊긴다.** 일반적으로 $\theta_0 = 1$로 두면 $\gamma_k = \sigma^2\sum_{j=0}^{q-k}\theta_j\theta_{j+k}$이고 $k > q$이면 겹치는 충격이 없다.

AR과 갈리는 지점도 여기다. AR은 오른쪽이 관측값이라 곱하고 나면 자기공분산끼리의 선형식이 나왔다. MA는 오른쪽이 충격이라 $\theta$와 $\sigma^2$이 섞인 비선형식이 된다. $\rho_1 = \theta/(1 + \theta^2)$을 $\theta$에 대해 풀면 해가 둘이거나 없다. [8편](/posts/2026-10-17-time-series-8-ma-invertibility)에서 $r_1$로 $\theta$를 풀면 300번 중 33번 해가 없었던 것이 이 때문이다. MA에는 Yule-Walker 같은 손쉬운 선형 추정이 없다.

PACF는 식 (4)에 $\rho_1 = \theta/(1 + \theta^2)$, $\rho_k = 0$을 넣으면 된다. $\alpha_1 = \rho_1$이고

$$
\alpha_2 = \frac{0 - \rho_1^2}{1 - \rho_1^2} = \frac{-\theta^2}{1 + \theta^2 + \theta^4}, \qquad \alpha_k = -\frac{(-\theta)^k}{1 + \theta^2 + \cdots + \theta^{2k}}
$$

이다. 절댓값이 지수적으로 줄어들 뿐 0이 되지 않는다. $\theta = 0.8$이면 0.488, $-0.312$, 0.221, $-0.165$, 0.127로 부호도 번갈아 바뀐다. 끊기지 않는 이유는 [8편](/posts/2026-10-17-time-series-8-ma-invertibility)의 가역성에 있다. 가역인 MA(1)은 AR($\infty$)이라 몇 차의 AR을 맞춰도 마지막 계수가 0이 되지 않는다. **AR과 MA는 ACF와 PACF에서 역할이 서로 뒤바뀐다.**

**주의:** $\theta$와 $1/\theta$는 $\rho_1$이 같으니 PACF까지 같다. $\theta = 0.5$와 $\theta = 2$는 PACF가 모두 0.400, $-0.190$, 0.094, …다. ACF와 PACF로는 둘을 가를 수 없고 [8편](/posts/2026-10-17-time-series-8-ma-invertibility)의 가역 조건 $\lvert\theta\rvert < 1$이 하나를 고른다.

## 차수를 고르는 규칙

지금까지를 표로 모으면 이렇다.

| 모형 | ACF | PACF |
| --- | --- | --- |
| AR($p$) | 지수적으로 또는 진동하며 줄어든다 | 시차 $p$ 뒤에 끊긴다 |
| MA($q$) | 시차 $q$ 뒤에 끊긴다 | 지수적으로 또는 진동하며 줄어든다 |
| ARMA($p$, $q$) | 줄어든다 | 줄어든다 |

FPP3 9.5절도 같은 규칙을 준다. 차분한 계열의 ACF가 지수적으로 줄거나 사인파 꼴이고 PACF가 시차 $p$에서 유의한데 그 뒤로는 유의한 막대가 없으면 ARIMA($p$, $d$, 0)을 쓴다. ACF와 PACF를 바꾸면 ARIMA(0, $d$, $q$)의 규칙이다. PACF에 그리는 점선도 ACF와 같은 $\pm 1.96/\sqrt{T}$다.

<Fig src="/images/posts/2026-10-21-time-series-9-acf-pacf/acf-pacf-patterns.svg" alt="AR(2), MA(1), ARMA(1,1) 표본 300개의 ACF와 PACF, 이론값 점">그림 2. 세 모형으로 300개씩 만든 표본 ACF(위)와 PACF(아래). 주황색 점은 이론값이고 점선은 $\pm 1.96/\sqrt{300} = \pm 0.113$이다. AR(2)는 PACF가, MA(1)은 ACF가 끊긴다. ARMA(1,1)은 둘 다 줄어들기만 한다.</Fig>

그림 2의 AR(2)는 ACF가 진동하며 줄고 PACF가 시차 2 뒤에 끊긴다. MA(1)은 반대다. 표본이라 규칙이 깔끔하게 지켜지지는 않는다. AR(2)의 PACF는 시차 7에서 $-0.119$로 점선을 살짝 넘고 MA(1)의 ACF도 시차 5에서 0.122로 넘는다. 점선은 white noise 기준이라 막대 하나쯤 넘는 일은 흔하다([8편](/posts/2026-10-17-time-series-8-ma-invertibility)의 각주). 끊긴 위치는 막대 하나가 아니라 전체 모양으로 판단해야 한다.

규칙이 듣지 않는 경우도 있다. ARMA(1,1)은 ACF가 0.819, 0.573, 0.401로 줄고 PACF도 0.819, $-0.294$, 0.116으로 줄어든다. 끊기는 쪽이 없으니 어느 차수도 읽히지 않는다. FPP3 9.5절도 $p$와 $q$가 둘 다 양수이면 ACF와 PACF가 차수를 정하는 데 도움이 되지 않는다고 적는다. **ACF와 PACF는 순수한 AR이나 MA일 때만 차수를 알려 준다.** 그 밖의 경우에는 후보 모형을 여럿 맞춰 정보기준으로 비교한다.

다시 정리하면 ACF는 건너온 상관까지 모두 세고 PACF는 중간 시차를 걷어 낸 직접 효과만 센다. AR의 차수는 PACF가, MA의 차수는 ACF가 알려 준다. 둘 다 Yule-Walker 방정식 하나에서 나온다. [10편](/posts/2026-10-21-time-series-10-arima)에서는 차분과 AR, MA를 합친 ARIMA($p$, $d$, $q$)를 정의하고 이집트 수출 자료로 차수를 골라 본다. ACF와 PACF로 고른 모형과 정보기준으로 고른 모형이 어떻게 다른지도 거기서 비교한다.

---

**참고**

- FPP3 [9.5 Non-seasonal ARIMA models](https://otexts.com/fpp3/non-seasonal-arima.html)

[^mean]: $\gamma_1 = E[(y_t - \mu)(y_{t-1} - \mu)] = E(y_ty_{t-1}) - \mu^2$이다. 평균을 빼지 않고 곱만 평균 내면 $\mu^2$이 섞인다. $\phi = 0.7$, $\mu = 5$인 AR(1)을 20만 개 만들면 $y_ty_{t-1}$의 평균이 26.37로 $\gamma_1 + \mu^2 = 1.37 + 25$와 같다.

[^pacf-def]: 정상 계열에서는 잔차끼리의 상관과 AR($k$)의 마지막 계수가 같은 값이다. 앞쪽으로 예측할 때와 뒤쪽으로 예측할 때 오차 분산이 같기 때문이다. 표본에서는 계산 방법에 따라 조금씩 다르다. 그림 1의 자료에서 잔차 상관과 OLS 계수는 0.002이고 아래의 Durbin-Levinson 점화식에 $r_k$를 넣은 값은 0.000이었다.

[^dl]: 둘째 식이 맞는 이유는 $\mathbf{R}_{n-1}$이 주대각선뿐 아니라 반대 대각선에 대해서도 대칭이라는 데 있다. 차수 $n-1$의 해를 거꾸로 늘어놓은 벡터는 오른쪽을 $(\rho_{n-1}, \dots, \rho_1)$로 바꾼 방정식의 해다. 이것을 써서 둘째 식의 계수를 차수 $n$의 방정식 앞 $n - 1$개에 넣으면 $\alpha_{n,n}$이 무엇이든 성립한다. 남은 마지막 식을 $\alpha_{n,n}$에 대해 풀면 첫 식이 나온다.
