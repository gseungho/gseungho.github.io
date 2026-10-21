---
title: "시계열분석 (10): ARIMA 모형, 차수를 고르고 계수를 추정하기"
date: "2026-10-21"
category: "공부"
subcategory: "Statistics/Time Series Analysis"
tags: ["시계열분석", "ARIMA", "AICc", "모형 선택"]
description: "차분과 AR, MA를 합친 ARIMA(p, d, q)를 정의하고 이집트 수출 자료로 차수를 고르는 과정을 따라간다. AR과 MA 계수를 한꺼번에 추정해야 하는 이유와 두 다항식이 서로 지우는 공통 인수 문제를 보인다. AICc와 자동 선택 알고리즘이 하는 일도 정리한다."
---

세종대학교 인공지능데이터사이언스학과 2026년 2학기 시계열분석 수업을 따라가며 정리하는 시리즈의 열 번째 글이다. 수업 내용과 교재 [Forecasting: Principles and Practice, 3rd ed.](https://otexts.com/fpp3/)(이하 FPP3) 9.5~9.7절을 바탕으로 썼다.

> **ARIMA가 말하는 것: 차분하고 남은 계열을 과거 값과 과거 충격으로 함께 설명하면 어디까지 갈 수 있을까?**

재료는 모두 모였다. [6편](/posts/2026-10-17-time-series-6-stationarity-differencing)의 차분, [7편](/posts/2026-10-17-time-series-7-backshift-ar)의 AR, [8편](/posts/2026-10-17-time-series-8-ma-invertibility)의 MA, [9편](/posts/2026-10-21-time-series-9-acf-pacf)의 ACF와 PACF다. 이번 글은 이것들을 모형 하나로 합치고 실제 자료에서 차수를 고른다. 계수를 추정할 때 AR과 MA가 섞이면서 생기는 문제도 함께 본다.

## ARIMA($p, d, q$): 차분한 계열의 ARMA

AR과 MA를 한 식에 넣으면 ARMA($p$, $q$)다.

$$
y_t = c + \phi_1y_{t-1} + \cdots + \phi_py_{t-p} + \theta_1\varepsilon_{t-1} + \cdots + \theta_q\varepsilon_{t-q} + \varepsilon_t \tag{1}
$$

설명변수가 과거 값과 과거 충격 둘 다다. AR 계수에는 [7편](/posts/2026-10-17-time-series-7-backshift-ar)의 정상성 조건이, MA 계수에는 [8편](/posts/2026-10-17-time-series-8-ma-invertibility)의 가역 조건이 붙는다. 여기에 차분을 더한 것이 ARIMA($p$, $d$, $q$)다. $d$번 차분한 계열 $y'_t = (1 - B)^dy_t$가 식 (1)의 ARMA를 따른다는 뜻이다. Backshift로 쓰면 괄호 세 개가 된다.

$$
\underbrace{(1 - \phi_1B - \cdots - \phi_pB^p)}_{\text{AR}}\;\underbrace{(1 - B)^d}_{\text{차분}}\;y_t = c + \underbrace{(1 + \theta_1B + \cdots + \theta_qB^q)}_{\text{MA}}\;\varepsilon_t \tag{2}
$$

$p$는 AR 차수, $d$는 차분 횟수, $q$는 MA 차수다. ARIMA(1,1,1)이라면 $(1 - \phi_1B)(1 - B)y_t = c + (1 + \theta_1B)\varepsilon_t$다. 이름의 I(integrated)는 [7편](/posts/2026-10-17-time-series-7-backshift-ar)에서 본 대로 차분의 역연산인 누적합을 가리킨다.

**주의:** 소프트웨어에는 차분한 계열이 아니라 원계열을 넣는다. ARIMA($p$, $d$, $q$)를 맞추면 차분, ARMA 추정, 예측을 원래 단위로 되돌리는 누적합까지 모형 안에서 한다. 사람이 미리 해 둘 것은 분산을 안정시키는 변환 정도다.

지금까지 본 모형은 모두 ARIMA의 특수한 경우다.

| 모형 | ARIMA |
| --- | --- |
| white noise | (0,0,0) |
| random walk | (0,1,0), 상수 없음 |
| drift가 있는 random walk | (0,1,0), 상수 있음 |
| AR($p$) | ($p$,0,0) |
| MA($q$) | (0,0,$q$) |
| ETS(A,N,N), 곧 SES | (0,1,1), $\theta_1 = \alpha - 1$ |
| ETS(A,A,N) | (0,2,2), $\theta_1 = \alpha + \beta - 2$, $\theta_2 = 1 - \alpha$ |

ETS(A,N,N)이 ARIMA(0,1,1)인 이유는 [8편](/posts/2026-10-17-time-series-8-ma-invertibility) 끝에서 유도했다. 관측 방정식과 상태 방정식에서 보이지 않는 수준을 지우면 차분한 계열이 MA(1)로 남는다. 추세가 있는 ETS(A,A,N)은 상태가 두 개라 차분을 두 번 해야 둘 다 지워진다[^aan].

### 상수 $c$는 평균이 아니다

[7편](/posts/2026-10-17-time-series-7-backshift-ar)의 AR처럼 식 (2)의 $c$도 절편이다. $y'_t$의 평균 $\mu$와는 $c = \mu(1 - \phi_1 - \cdots - \phi_p)$의 관계가 있다. 그래서 같은 모형을 평균을 뺀 꼴로도 쓸 수 있다.

$$
(1 - \phi_1B - \cdots - \phi_pB^p)(y'_t - \mu) = (1 + \theta_1B + \cdots + \theta_qB^q)\varepsilon_t
$$

R fable 출력의 constant는 $c$ 쪽이다. 모형 이름 뒤에 붙는 꼬리표는 상수가 있을 때 차분 횟수를 알려 준다. 차분이 없으면 "w/ mean", 한 번이면 "w/ drift"다.

## 이집트 수출: 그림으로 고른 모형과 자동으로 고른 모형

FPP3 9.5절의 예제로 차수를 골라 보자. 1960년부터 2017년까지 이집트의 GDP 대비 수출 비율이다. 관측값은 58개이고 추세 없이 오르내린다. fable의 `ARIMA()`에 맡기면 차분하지 않는 다음 모형을 고른다.

```text
Model: ARIMA(2,0,1) w/ mean
         ar1      ar2      ma1  constant
      1.6764  -0.8034  -0.6896    2.5623
s.e.  0.1111   0.0928   0.1492    0.1161
sigma^2 estimated as 8.046:  log likelihood=-141.6
AIC=293.1   AICc=294.3   BIC=303.4
```

식으로 쓰면 다음과 같다.

$$
y_t = 2.562 + 1.676y_{t-1} - 0.803y_{t-2} - 0.690\varepsilon_{t-1} + \varepsilon_t, \qquad \varepsilon_t \sim N(0, 8.046)
$$

평균은 $2.5623/(1 - 1.6764 + 0.8034) \approx 20.2$로 constant 2.562와 전혀 다르다. AR 부분의 역근은 $0.838 \pm 0.318i$로 복소수이고 크기가 0.896이다. 충격이 주기 약 17년으로 진동하며 줄어든다는 뜻이다.

이번에는 [9편](/posts/2026-10-21-time-series-9-acf-pacf)의 규칙대로 그림을 읽어 보자. FPP3에 따르면 이 자료의 ACF는 사인파 꼴로 줄어들고 PACF의 마지막 유의한 막대가 시차 4다. AR(4)가 보이는 모양이다. ARIMA(4,0,0)을 맞추면 다음과 같다.

```text
Model: ARIMA(4,0,0) w/ mean
         ar1      ar2     ar3      ar4  constant
      0.9861  -0.1715  0.1807  -0.3283    6.6922
s.e.  0.1247   0.1865  0.1865   0.1273    0.3562
sigma^2 estimated as 7.885:  log likelihood=-140.5
AIC=293.1   AICc=294.7   BIC=305.4
```

평균은 $6.6922/0.3330 \approx 20.1$로 거의 같다. AICc는 294.70으로 ARIMA(2,0,1)의 294.29보다 조금 크다. FPP3도 이 모형이 자동으로 고른 모형보다 조금 나쁠 뿐이라고 적는다. 모양이 전혀 다른 두 모형이 왜 이렇게 비슷한 점수를 받았을까?

<Fig src="/images/posts/2026-10-21-time-series-10-arima/egypt-models.svg" alt="이집트 수출에 맞춘 ARIMA(2,0,1)과 ARIMA(4,0,0)의 이론 ACF와 PACF">그림 1. 두 모형의 추정값으로 계산한 이론 ACF(왼쪽)와 PACF(오른쪽). 회색 점선은 $T = 58$의 $\pm 1.96/\sqrt{58} = \pm 0.257$이다. ACF는 거의 겹치고 PACF는 시차 3, 4에서만 갈린다.</Fig>

두 모형이 그리는 자기상관 구조가 거의 같기 때문이다. 그림 1에서 시차 1~4의 ACF는 0.834, 0.595, 0.328, 0.071과 0.834, 0.604, 0.359, 0.073이다. PACF는 시차 4에서 $-0.140$과 $-0.328$로 갈리지만 관측값이 58개뿐이라 점선이 $\pm 0.257$이나 된다. 표본 PACF의 시차 4 막대 하나만으로는 두 모형을 가르기 어렵다. **MA 항 하나가 AR 항 여러 개의 몫을 한다.** [8편](/posts/2026-10-17-time-series-8-ma-invertibility)에서 가역인 MA는 AR($\infty$)였다. 그래서 MA 항 하나로 긴 AR을 대신할 수 있고 모수가 하나 적은 쪽이 AICc에서 이겼다.

## 계수를 한꺼번에 추정하는 이유

ARIMA의 추정은 세 층으로 나뉜다.

| 무엇을 | 어떻게 |
| --- | --- |
| 차분 횟수 $d$ | 추정하지 않는다. 단위근 검정으로 먼저 정하고 차분한다 |
| 계수 $c$, $\phi$, $\theta$, $\sigma^2$ | 차분한 계열에서 모두 한꺼번에 수치 최적화로 추정한다 |
| 차수 $p$, $q$ | 후보마다 위 추정을 하고 AICc로 고른다 |

가운데 줄부터 보자. [8편](/posts/2026-10-17-time-series-8-ma-invertibility)에서 MA 계수는 충격을 거꾸로 계산하는 점화식으로 구했다. ARMA는 그 점화식에 AR 항을 더하면 된다. 차분한 계열을 $w_t$라 하면

$$
e_t = w_t - c - \phi_1w_{t-1} - \cdots - \phi_pw_{t-p} - \theta_1e_{t-1} - \cdots - \theta_qe_{t-q} \tag{3}
$$

이다. 처음 $p$개 관측값은 주어진 값으로 두고 처음 충격은 0으로 둔다. $\sum e_t^2$를 $(c, \phi, \theta)$ 전체에 대해 한 번에 최소화하면 조건부 제곱합(CSS) 추정값이다. 정규분포를 가정한 최대가능도로 풀어도 된다. FPP3 9.6절은 ARIMA에서 최대가능도 추정값이 $\sum\varepsilon_t^2$를 최소로 하는 최소제곱 추정값과 비슷하다고 적는다.

그런데 왜 꼭 한꺼번에 추정해야 할까? AR 계수만 회귀로 먼저 구하면 안 될까? 실험해 보자. $\phi = 0.7$, $\theta = 0.4$인 ARIMA(1,1,1)로 300개를 만들면 한 번 차분해 299개의 ARMA(1,1)이 된다. 이것을 200번 반복했다.

| 방법 | $\hat\phi$ 평균 (표준편차) | $\hat\theta$ 평균 (표준편차) |
| --- | --- | --- |
| CSS | 0.699 (0.047) | 0.399 (0.062) |
| 최대가능도 | 0.697 (0.048) | 0.400 (0.063) |
| AR(1)만 OLS | 0.813 (0.026) | 없음 |
| Hannan-Rissanen | 0.709 (0.047) | 0.387 (0.073) |

MA 항을 빼고 AR(1)만 맞추면 기울기가 $\phi = 0.7$이 아니라 0.813으로 간다. [9편](/posts/2026-10-21-time-series-9-acf-pacf)대로 AR(1) 회귀의 기울기는 $\rho_1$로 가는데 이 ARMA(1,1)은 $\rho_1 = 0.819$이기 때문이다. 빠진 MA 항 $\theta\varepsilon_{t-1}$이 설명변수 $w_{t-1}$과 상관되어 있어서 그 몫을 AR 계수가 떠안는다. [7편](/posts/2026-10-17-time-series-7-backshift-ar)에서 절편을 빼면 기울기가 수준을 떠안았던 것과 같은 구조다. **AR과 MA 계수는 한 목적함수 안에서 함께 움직여야 한다.**

표의 마지막 줄은 충격을 먼저 근사하는 방법이다. 충격을 안다면 ARMA는 그냥 회귀다. 그래서 긴 AR 모형을 맞춘 잔차 $\hat\varepsilon_t$를 충격 대신 쓰고 $w_t$를 $w_{t-1}, \dots, w_{t-p}$와 $\hat\varepsilon_{t-1}, \dots, \hat\varepsilon_{t-q}$에 OLS로 회귀한다. Hannan-Rissanen 방법이다[^hr]. 위 실험에서는 AR(20)으로 잔차를 구했다. 평균은 참값에 가깝지만 $\hat\theta$의 표준편차가 최대가능도보다 크다. 닫힌 꼴로 바로 계산되니 수치 최적화의 시작값으로 쓰기 좋다.

### 공통 인수: AR과 MA가 서로 지울 때

한꺼번에 추정하다 보면 부딪히는 문제가 있다. 두 다항식이 같은 인수를 가지면 약분된다.

$$
(1 - aB)w_t = (1 - aB)\varepsilon_t \quad\Longrightarrow\quad w_t = \varepsilon_t
$$

$a$가 무엇이든 white noise다. $\phi = a$, $\theta = -a$인 ARMA(1,1)은 모두 같은 모형이라는 말이다. 자료가 이 직선 위의 점들을 구별하지 못하니 추정값이 정해지지 않는다.

<Fig src="/images/posts/2026-10-21-time-series-10-arima/css-surface.svg" alt="ARMA(1,1) 자료와 white noise 자료에서 CSS 목적함수의 등고선">그림 2. ARMA(1,1)을 맞출 때 CSS 목적함수의 등고선. 왼쪽 자료($\phi_1 = 0.7$, $\theta_1 = 0.4$, ×)에서는 최솟값(주황 점)이 한 점으로 모인다. 오른쪽 white noise 자료에서는 골짜기가 점선 $\theta_1 = -\phi_1$을 따라 길게 뻗는다.</Fig>

그림 2 왼쪽은 최솟값이 $(0.67, 0.40)$ 한 점에 모인다. 오른쪽 white noise에서는 골짜기가 $\theta_1 = -\phi_1$을 따라 길게 뻗어 있고 최솟값 $(0.10, -0.08)$은 그 골짜기의 아무 점이다. 정확한 우도로 계산하면 차이가 더 분명하다. White noise 200개에서 $(\phi, \theta) = (0, 0)$, $(0.5, -0.5)$, $(-0.3, 0.3)$의 음의 로그우도가 모두 275.482623으로 소수점 여섯째 자리까지 같았다. 같은 white noise에 ARMA(1,1)을 맞추는 일을 반복하면 $(0.48, -0.56)$, $(-0.80, 0.98)$, $(0.84, -0.91)$처럼 $\hat\phi \approx -\hat\theta$인 쌍이 표본마다 제각각 나온다.

**$p$와 $q$를 필요 이상으로 키우면 서로 지우는 계수가 생긴다.** 모수 수에 벌점을 주는 AICc가 그런 모형을 걸러 내는 데 도움이 된다. 이집트 수출의 ARIMA(2,0,1)은 AR 역근이 복소수이고 MA 역근이 실수 0.69라 약분될 인수가 없다.

## AICc로 차수 고르기

남은 것은 $p$와 $q$다. FPP3 9.6절의 정보기준은 다음과 같다.

$$
\text{AIC} = -2\log L + 2(p + q + k + 1), \qquad \text{AICc} = \text{AIC} + \frac{2(p + q + k + 1)(p + q + k + 2)}{T - p - q - k - 2}
$$

$L$은 우도, $T$는 관측값 수이고 $k$는 상수가 있으면 1, 없으면 0이다. 마지막 1은 $\sigma^2$의 몫이다. BIC는 AIC에 $[\log T - 2](p + q + k + 1)$을 더한다. 이집트 수출에서는 ARIMA(2,0,1)의 모수가 5개, ARIMA(4,0,0)이 6개다. 로그우도는 ARIMA(4,0,0)이 $-140.5$로 1.04 더 크다. $-2\log L$로는 2.08 차이라 모수 하나의 벌점 2를 겨우 넘고 AIC는 ARIMA(4,0,0)이 293.05 대 293.14로 아주 조금 낫다[^aic]. 순위를 뒤집는 것은 AICc의 보정항이다. 관측값이 58개뿐이라 보정항이 1.65 대 1.15로 모수가 많은 쪽에 더 큰 벌점을 준다.

**주의:** 차분 횟수가 다른 모형끼리는 AICc를 비교할 수 없다. FPP3 9.6절의 설명대로 차분하면 우도를 계산하는 자료 자체가 바뀌기 때문이다. $d$를 정보기준이 아니라 단위근 검정으로 먼저 정하는 이유가 이것이다.

### 자동 선택: Hyndman-Khandakar 알고리즘

fable의 `ARIMA()`가 이집트 수출에서 ARIMA(2,0,1)을 고른 방법은 FPP3 9.7절에 있다.

1. KPSS 검정을 반복해 $0 \le d \le 2$를 정한다.
2. ARIMA(0,$d$,0), ARIMA(2,$d$,2), ARIMA(1,$d$,0), ARIMA(0,$d$,1) 네 모형을 맞춘다. $d = 2$가 아니면 상수를 넣는다. $d \le 1$이면 상수 없는 ARIMA(0,$d$,0)도 맞춘다. AICc가 가장 작은 모형이 현재 모형이다.
3. 현재 모형에서 $p$나 $q$를 $\pm 1$ 바꾸거나 상수를 넣고 빼 본다. AICc가 더 작은 모형이 나오면 그 모형으로 옮겨 가고 더 줄지 않을 때까지 되풀이한다.

모든 조합을 맞추지 않고 이웃으로만 옮겨 가는 탐색이다. 기본 설정은 속도를 위해 근사도 쓴다. 그래서 AICc가 가장 작은 모형을 놓칠 수 있다. `stepwise = FALSE`로 두면 훨씬 많은 모형을 보고 `approximation = FALSE`로 근사를 끈다. FPP3는 $d > 1$이면 상수를 늘 뺀다고 적는다. 2차 이상의 추세는 예측에서 특히 위험하기 때문이다.

자동 선택이 사람의 판단을 대신하지는 않는다. FPP3 9.7절이 권하는 순서는 이렇다.

1. 시간 그림을 그리고 이상한 관측값을 찾는다.
2. 필요하면 Box-Cox 변환으로 분산을 안정시킨다.
3. 비정상이면 정상이 될 때까지 차분한다.
4. ACF와 PACF를 보고 ARIMA($p$, $d$, 0)이나 ARIMA(0, $d$, $q$)가 맞는지 본다.
5. 고른 모형과 그 주변 모형을 AICc로 비교한다.
6. 잔차의 ACF를 그리고 Ljung-Box 같은 포트맨토 검정을 한다. White noise처럼 보이지 않으면 모형을 바꾼다.
7. 잔차가 white noise처럼 보이면 예측한다.

1~3단계는 [1편](/posts/2026-10-15-time-series-1-forecasting-basics)과 [6편](/posts/2026-10-17-time-series-6-stationarity-differencing), 4단계는 [9편](/posts/2026-10-21-time-series-9-acf-pacf), 5단계는 이번 글에서 다뤘다. 이집트 수출에서는 4단계가 ARIMA(4,0,0)을, 5단계의 자동 탐색이 ARIMA(2,0,1)을 냈다. 두 모형의 AICc 차이는 0.4에 불과하다. 그러니 둘 다 6단계의 잔차 점검을 거쳐 보고 고르는 것이 맞다.

[11편](/posts/2026-10-21-time-series-11-arima-forecasting-seasonal)에서는 7단계, 곧 예측을 다룬다. 예측식에서 아직 오지 않은 충격을 어떻게 처리하는지, 상수 $c$와 차분 횟수 $d$가 장기 예측의 모양을 어떻게 정하는지 본다. 계절이 있는 자료를 위한 계절 ARIMA도 같은 글에서 정리한다.

---

**참고**

- FPP3 [9.5 Non-seasonal ARIMA models](https://otexts.com/fpp3/non-seasonal-arima.html), [9.6 Estimation and order selection](https://otexts.com/fpp3/arima-estimation.html), [9.7 ARIMA modelling in fable](https://otexts.com/fpp3/arima-r.html), [9.10 ARIMA vs ETS](https://otexts.com/fpp3/arima-ets.html)
- statsmodels, [`hannan_rissanen` 소스](https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/arima/estimators/hannan_rissanen.py)

[^aan]: ETS(A,A,N)은 $y_t = \ell_{t-1} + b_{t-1} + \varepsilon_t$, $\ell_t = \ell_{t-1} + b_{t-1} + \alpha\varepsilon_t$, $b_t = b_{t-1} + \beta\varepsilon_t$다. 양변에 $(1 - B)^2$을 곱해 $\ell$과 $b$를 지우면 $(1 - B)^2y_t = \varepsilon_t + (\alpha + \beta - 2)\varepsilon_{t-1} + (1 - \alpha)\varepsilon_{t-2}$가 된다. FPP3 9.10절 표 9.4의 계수와 같다. $\alpha = 0.3$, $\beta = 0.1$로 만든 계열에서 양변의 차이는 최대 $1 \times 10^{-14}$였다.

[^aic]: 출력의 AIC는 둘 다 293.1로 반올림되어 있다. 소수 둘째 자리는 FPP3 본문의 AICc(294.29, 294.70)에서 보정항을 빼 거꾸로 계산한 값이다. 이 값으로 $\log L$을 다시 구하면 $-141.57$, $-140.53$이고 BIC도 출력의 303.4, 305.4와 맞는다.

[^hr]: statsmodels의 구현은 Brockwell and Davis (2016), *Introduction to Time Series and Forecasting* 5.1.4절의 세 단계를 따른다고 적는다. 큰 차수의 AR을 Yule-Walker로 맞춰 잔차를 구하고 최소제곱으로 AR과 MA 계수를 구한 뒤 편향을 보정한다. 위 실험은 AR(20)을 OLS로 맞췄고 편향 보정은 하지 않았다.
