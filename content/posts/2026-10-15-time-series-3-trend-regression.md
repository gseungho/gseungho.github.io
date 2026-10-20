---
title: "시계열분석 (3): 시간에 회귀하는 추세와 계절성, OLS부터 seasonal dummy까지"
date: "2026-10-15"
category: "공부"
subcategory: "Statistics/Time Series Analysis"
tags: ["시계열분석", "추세 회귀", "OLS", "AIC"]
description: "추세를 시간의 함수로, 계절성을 dummy 변수로 두고 OLS로 추정하는 회귀 모형을 다룬다. 행렬로 쓴 추정량, 소매 판매 예제에서 AIC와 SIC로 추세 모형을 고르는 과정, dummy trap을 차례로 정리한다. 이동평균 분해와 회귀가 무엇이 다른지도 비교한다."
---

시계열분석 시리즈의 세 번째 글이다. 세종대학교 인공지능데이터사이언스학과 2026년 2학기 시계열분석 수업을 따라가며 공부한 내용이고 주 교재 [FPP3](https://otexts.com/fpp3/)(Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed.)를 함께 참고했다.

> **이번 글의 질문: 추세와 계절성을 시간의 함수로 놓고 회귀로 구하면 무엇을 얻을까?**

[2편](/posts/2026-10-15-time-series-2-decomposition)에서는 이동평균과 loess로 추세와 계절성을 떼어 냈다. 이번에는 같은 성분을 두 번째 길인 회귀로 구한다. 추세는 시간의 함수로, 계절성은 dummy 변수로 두고 최소제곱으로 계수를 추정한다.

## 시간에 회귀하는 추세 모형

먼저 추세부터 보자. 추세를 시간의 함수로 두고 계수를 추정한다. $TIME_t$는 첫 시점에 1, 둘째 시점에 2를 주는 변수다.

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

이 식의 $y_t$는 무엇일까? [2편](/posts/2026-10-15-time-series-2-decomposition)에서 추세를 MA로 구했으니 이동평균을 거친 값에 회귀를 하는 것처럼 읽기 쉽다. 그렇지 않다.

**주의: 회귀식의 $y_t$는 이동평균을 거치지 않은, 관측한 계열 그대로다.**

뒤에 나올 소매 판매 예제가 근거다. 표본이 1955년 1월부터 1993년 12월까지이고 관측값이 468개다. 39년 × 12개월 = 468이라 빠진 달이 없다. $2 \times 12$-MA를 거쳤다면 앞뒤 6개월씩 값이 없어 456개가 됐을 것이다. 예측식이 맞히려는 것도 실제 관측값이고 구간 폭을 정하는 $\sigma$도 실제 값과 추세선 사이의 오차다.

그러니까 두 방법은 같은 추세 $T_t$를 다른 방식으로 뽑는 것이다. [2편](/posts/2026-10-15-time-series-2-decomposition)에서 본 STL의 loess까지 한 줄에 놓으면 차이가 선명해진다.

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

$\beta_1 = 0.005931$은 한 달에 약 0.59%씩 성장한다는 뜻이고 연으로는 $e^{0.005931 \times 12} - 1 \approx 7.4\%$다. [1편](/posts/2026-10-15-time-series-1-forecasting-basics)에서 본 "log 값의 차이는 퍼센트 변화"가 여기서 쓰인다.[^median]

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

[1편](/posts/2026-10-15-time-series-1-forecasting-basics)에서 본 AR 식과 나란히 놓으면 차이가 보인다. AR에서는 2-step 앞에 잡음이 두 개 들어간다.

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

<Fig src="/images/posts/2026-10-15-time-series-3-trend-regression/linear_vs_quadratic.svg" alt="볼록한 시뮬레이션 계열에 linear와 quadratic 추세를 맞춘 결과와 linear 잔차">그림 1. 왼쪽: 볼록한 시뮬레이션 계열에 맞춘 직선(파랑)과 2차 곡선(주황). 오른쪽: 직선의 잔차는 처음과 끝이 양수, 가운데가 음수인 긴 덩어리로 갈린다.</Fig>

그림 1의 직선 잔차는 대략 113번째 시점부터 390번째 시점까지 음수이고 그 앞뒤는 양수다. 잔차가 무작위라면 위아래를 자주 오가야 하는데 긴 덩어리로 몰려 있으니 직선이 놓친 구조가 남았다는 신호다. 2차 곡선은 계열의 휘어짐을 거의 따라간다.

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

두 기준의 차이는 벌점 크기다. 모수 하나당 AIC는 $2/T$, SIC는 $\ln T/T$를 더한다. 그림 2는 두 벌점에 $T$를 곱해 나란히 그린 것이다.

<Fig src="/images/posts/2026-10-15-time-series-3-trend-regression/aic_sic_penalty.svg" alt="관측값 수 T에 따른 AIC와 SIC의 모수 1개당 벌점">그림 2. AIC의 벌점은 2로 고정이고 SIC의 벌점은 $\ln T$로 계속 커진다. 두 선은 $T = e^2 \approx 7.4$에서 만난다.</Fig>

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

계수가 월별 평균이라는 점에서 이 회귀는 [2편](/posts/2026-10-15-time-series-2-decomposition)에서 본 classical decomposition의 3단계와 같은 계산을 한다. 연도를 행, 월을 열로 놓고 열마다 평균하는 바로 그 표다.

### dummy를 전부 넣으면 생기는 일

추세와 계절성을 한 회귀에 넣을 수도 있다. 호주 분기별 맥주 생산량 모형이 그 예다.

$$
y_t = \beta_0 + \beta_1 t + \beta_2 d_{2,t} + \beta_3 d_{3,t} + \beta_4 d_{4,t} + \epsilon_t
$$

$d_{i,t}$는 $t$가 $i$분기면 1, 아니면 0이다. $\beta_1 t$가 추세, $\beta_2 \sim \beta_4$가 계절성을 맡는다.

1분기 dummy는 왜 뺐을까? 절편이 있는데 dummy 네 개를 모두 넣으면 $d_1 + d_2 + d_3 + d_4 = 1$이 절편 열 $\mathbf{1}$과 같아진다. **$\mathbf{X}$의 열이 선형독립이 아니게 되어 $(\mathbf{X}^\top\mathbf{X})^{-1}$이 존재하지 않는다(dummy trap).** 해결책은 둘이다. 주택 착공처럼 절편 없이 dummy를 모두 넣거나 맥주 모형처럼 절편을 두고 하나를 빼면 된다.

하나를 빼면 나머지 계수는 그 분기와 1분기의 차이를 나타낸다. 분기 평균이 100, 120, 140, 160이라면 100은 절편 $\beta_0$으로 가고 dummy 계수는 +20, +40, +60이 된다. [2편](/posts/2026-10-15-time-series-2-decomposition)에서 classical decomposition의 (ii) 단계를 설명할 때 본 표의 위쪽 줄(1분기 기준)이 바로 이 방식이다. 수준과 계절을 나누는 기준이 하나로 정해지지 않는다는 같은 문제를 classical은 "합을 0으로", dummy 회귀는 "1분기를 기준으로" 푼 것이다.

적합 결과는 실제 값을 가로축, 적합값을 세로축에 놓은 산점도로 확인한다. 점이 대각선 $y = x$에 가까울수록 잘 맞은 것이다.

## 두 방법을 나란히 놓고 보면

[2편](/posts/2026-10-15-time-series-2-decomposition)과 이번 글을 합쳐 다시 정리하면 추세와 계절성을 구하는 두 길은 이렇게 갈린다.

| | 이동평균 분해 (classical, STL) | 회귀 (추세 함수, dummy) |
| --- | --- | --- |
| 추세 | 시점마다 주변 평균이나 국소 직선 | 시간의 함수 하나 |
| 계절성 | 계절마다 평균 (STL은 천천히 변함) | dummy 계수 |
| 끝점과 미래 | classical은 끝점이 없고 둘 다 미래를 직접 주지 않음 | $TIME_{T+h}$와 dummy를 넣으면 나옴 |
| 주로 쓰는 곳 | 성분을 보고 계절 조정하기 | 예측 |

회귀 쪽은 예측식을 바로 주지만 추세와 계절성이 영원히 같은 모양이라고 가정한다. 소매 판매의 Durbin-Watson이 보여 주듯 그렇게 걷어내고도 잔차에는 구조가 남는다. 그렇다면 성분을 고정하지 않고 관측값이 들어올 때마다 조금씩 업데이트하면 어떨까? [4편](/posts/2026-10-15-time-series-4-exponential-smoothing)에서 다루는 지수평활법이 바로 그 방법이다.

---

**참고 문헌**

- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. [7장 Time series regression models](https://otexts.com/fpp3/regression.html)
- F. X. Diebold, *Forecasting in Economics, Business, Finance and Beyond* (2017), 3.3.12절(AIC, SIC), 5.2.4절(소매 판매), 5.3.4절(주택 착공). 저자 홈페이지에서 무료로 공개되어 있다: [Textbooks](https://www.sas.upenn.edu/~fdiebold/Textbooks.html)

[^median]: 예측할 때 $e^{\hat b_0 + \hat\beta_1 TIME_{T+h}}$는 평균이 아니라 중앙값이다. 평균을 원하면 $e^{\hat\sigma^2/2}$를 곱해야 한다. 이 예에서는 $\hat\sigma = 0.091879$라 $e^{\hat\sigma^2/2} \approx 1.0042$, 약 0.4% 차이다.

[^se]: 이 구간은 $\hat{\boldsymbol\beta}$의 추정 오차를 무시한 것이다. 실제로는 먼 미래일수록 추세선 자체의 불확실성이 커진다.

[^diebold]: Diebold 교재 3.3.12절은 $\text{AIC} = e^{2k/T}\,\frac{\text{SSR}}{T}$, $\text{SIC} = T^{k/T}\,\frac{\text{SSR}}{T}$로 정의한다. 회귀 출력의 값은 이 둘에 log를 씌운 것이라 식 (2)와 같다.
