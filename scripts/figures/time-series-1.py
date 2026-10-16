# Figures for 2026-10-15-time-series-1-forecasting-basics.
# Run: python scripts/figures/time-series-1.py
import os
import numpy as np
from blogstyle import *

OUT = os.path.join(POSTS, '2026-10-15-time-series-1-forecasting-basics')

# Fig 1: AR(1) fan chart with simulated futures
rng = np.random.default_rng(7)
a, b, s = 0.8, 2.0, 1.0
T, H = 60, 30
y = np.empty(T); y[0] = 10
for t in range(1, T): y[t] = a*y[t-1] + b + rng.normal(0, s)
paths = np.empty((200, H))
for k in range(200):
    prev = y[-1]
    for h in range(H):
        prev = a*prev + b + rng.normal(0, s); paths[k, h] = prev
hh = np.arange(1, H+1)
mean = np.array([a**h*y[-1] + b*sum(a**j for j in range(h)) for h in hh])
sd = np.array([s*np.sqrt(sum(a**(2*j) for j in range(h))) for h in hh])
fig, ax = plt.subplots(figsize=(W, 3.5))
ax.plot(np.arange(1, T+1), y, color=GRAY, label='관측값')
for k in range(20): ax.plot(T+hh, paths[k], color=GRAY, lw=0.6, alpha=0.35, label='시뮬레이션한 미래 경로 20개' if k == 0 else None)
ax.fill_between(T+hh, mean-1.96*sd, mean+1.96*sd, color=BLUE, alpha=0.13, lw=0, label='95% 예측구간')
ax.plot(T+hh, mean, color=BLUE, lw=1.8, label='점예측 (경로들의 평균)')
ax.axvline(T+0.5, color=AXIS, lw=0.8, ls=(0, (4, 3)))
ax.text(T+1.2, 14.6, '미래 →', color=INK2, va='top', fontsize=9.5)
ax.set_xlabel('시점 $t$'); ytitle(ax, '$y_t$')
top_legend(ax, ncol=4)
save(fig, os.path.join(OUT, 'fan_chart_ar1.svg'))
print('last obs', y[-1], 'band h1', 1.96*sd[0], 'band h30', 1.96*sd[-1]); print('fig1 sd h=1,2,30:', sd[0], sd[1], sd[-1], 'limit', s/np.sqrt(1-a*a))

# Fig 2: forecast SD vs h for a = 0.5, 0.8, 1
fig, ax = plt.subplots(figsize=(W, 3.5))
H2 = np.arange(1, 31)
for a_, c in [(0.5, BLUE), (0.8, ORANGE), (1.0, AQUA)]:
    sdh = np.sqrt([sum(a_**(2*j) for j in range(h)) for h in H2])
    ax.plot(H2, sdh, color=c)
    lab = f'$a = {a_:g}$' + (r'  (random walk, $\sqrt{h}$)' if a_ == 1 else f'  (상한 {1/np.sqrt(1-a_*a_):.2f})')
    ax.text(30.8, sdh[-1], lab, color=INK2, va='center', fontsize=9.5)
    if a_ < 1: ax.hlines(1/np.sqrt(1-a_*a_), 1, 30, color=c, lw=0.9, ls=(0, (1.5, 2)))
    print('a', a_, 'sd h=30', sdh[-1])
ax.set_xlim(0, 42); ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.set_xlabel('몇 step 앞인가 ($h$)'); ytitle(ax, '예측 표준편차 ($\\sigma$ 배)')
save(fig, os.path.join(OUT, 'forecast_sd_by_h.svg'))

# Fig 3: Box-Cox small multiples on a multiplicative seasonal series
rng = np.random.default_rng(3)
n = 80; t = np.arange(n)
level = 20*np.exp(0.03*t)
seas = 1 + 0.25*np.sin(2*np.pi*t/4 + 0.6)
yq = level*seas*np.exp(rng.normal(0, 0.02, n))
fig, axes = plt.subplots(3, 1, figsize=(W, 5.6), sharex=True, gridspec_kw={'hspace': 0.5})
for ax, lam, c in zip(axes, [1, 0.5, 0], [GRAY, ORANGE, BLUE]):
    w = np.log(yq) if lam == 0 else (yq**lam - 1)/lam
    ax.plot(t, w, color=c)
    first = w[:8].max()-w[:8].min(); last = w[-8:].max()-w[-8:].min()
    ax.set_title(f'$\\lambda = {lam:g}$     2년 동안의 변동 폭: 처음 {first:.2f}, 마지막 {last:.2f} ({last/first:.1f}배)')
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    print('lambda', lam, 'amp first/last', first, last, last/first)
axes[-1].set_xlabel('분기')
save(fig, os.path.join(OUT, 'boxcox_lambda.svg'))

# Fig 4: additive vs multiplicative
rng = np.random.default_rng(5)
n = 60; t = np.arange(n)
trend = 50 + 1.5*t
s_add = 8*np.sin(2*np.pi*t/12)
y_add = trend + s_add + rng.normal(0, 1.5, n)
y_mul = trend*(1 + 0.15*np.sin(2*np.pi*t/12))*np.exp(rng.normal(0, 0.015, n))
fig, axes = plt.subplots(1, 2, figsize=(W, 3.0), sharey=True, gridspec_kw={'wspace': 0.12})
for ax, yy, c, name in [(axes[0], y_add, BLUE, '덧셈: 진폭이 일정'), (axes[1], y_mul, ORANGE, '곱셈: 진폭이 수준을 따라 커짐')]:
    ax.plot(t, yy, color=c); ax.plot(t, trend, color=GRAY, lw=1.1, ls=(0, (4, 3)))
    ax.set_title(name)
    ax.set_xlabel('월')
save(fig, os.path.join(OUT, 'additive_vs_multiplicative.svg'))

# Fig 5: seasonality (fixed period) vs cycle (AR(2), irregular period)
rng = np.random.default_rng(11)
n = 120; t = np.arange(n)
seasonal = 10*np.sin(2*np.pi*t/12) + rng.normal(0, 1, n)
p1, p2 = 1.6, -0.9
c = np.zeros(n+50)
e = rng.normal(0, 1, n+50)
for i in range(2, n+50): c[i] = p1*c[i-1] + p2*c[i-2] + e[i]
c = c[50:]
fig, axes = plt.subplots(2, 1, figsize=(W, 4.4), sharex=True, gridspec_kw={'hspace': 0.45})
axes[0].plot(t, seasonal, color=BLUE)
for k in range(0, n, 12): axes[0].axvline(k+3, color=AXIS, lw=0.8, ls=(0, (1.5, 2)))
axes[0].set_title('계절성: 정확히 12개월마다 같은 자리에서 정점')
axes[1].plot(t, c, color=ORANGE)
pk = [i for i in range(1, n-1) if c[i] == c[max(0, i-4):i+5].max() and c[i] > 1]
gaps = np.diff(pk)
axes[1].plot(pk, c[pk], 'o', color=ORANGE)
axes[1].set_title(f'주기: 정점 간격이 매번 다르다 ({gaps.min()}~{gaps.max()}개월)')
axes[1].set_xlabel('월')
print('cycle peaks', pk, 'gaps', gaps.tolist())
save(fig, os.path.join(OUT, 'season_vs_cycle.svg'))
