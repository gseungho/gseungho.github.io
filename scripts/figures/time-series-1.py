# Figures for 2026-10-15-time-series-1-forecasting-basics.
# Run: python scripts/figures/time-series-1.py
import os
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'public', 'images', 'posts'), '2026-10-15-time-series-1-forecasting-basics')
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#898781'
INK, INK2, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7', '#fcfcfb'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'figure.facecolor': SURF,
    'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.grid': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 11, 'legend.frameon': False,
    'lines.linewidth': 2})

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
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(1, T+1), y, color=GRAY, label='관측값')
for k in range(20): ax.plot(T+hh, paths[k], color=GRAY, lw=0.6, alpha=0.5, label='시뮬레이션한 미래 경로 (20개)' if k == 0 else None)
ax.fill_between(T+hh, mean-1.96*sd, mean+1.96*sd, color=BLUE, alpha=0.15, lw=0, label='95% 예측구간')
ax.plot(T+hh, mean, color=BLUE, label='점예측 (경로들의 평균)')
ax.axvline(T+0.5, color=AXIS, lw=1, ls='--')
ax.text(T+1.5, ax.get_ylim()[1]*0.97, '미래 →', color=INK2, va='top')
ax.set_xlabel('시점 t'); ax.set_ylabel('$y_t$')
ax.legend(loc='lower left', ncol=2, fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fan_chart_ar1.png'), dpi=150); plt.close(fig)
print('last obs', y[-1], 'band h1', 1.96*sd[0], 'band h30', 1.96*sd[-1]); print('fig1 sd h=1,2,30:', sd[0], sd[1], sd[-1], 'limit', s/np.sqrt(1-a*a))

# Fig 2: forecast SD vs h for a = 0.5, 0.8, 1
fig, ax = plt.subplots(figsize=(8, 4))
H2 = np.arange(1, 31)
for a_, c in [(0.5, BLUE), (0.8, ORANGE), (1.0, AQUA)]:
    sdh = np.sqrt([sum(a_**(2*j) for j in range(h)) for h in H2])
    ax.plot(H2, sdh, color=c, marker='o', ms=4)
    lab = f'a = {a_:g}' + (r'  (random walk, $\sqrt{h}$)' if a_ == 1 else f'  (상한 {1/np.sqrt(1-a_*a_):.2f})')
    ax.text(30.5, sdh[-1], lab, color=INK, va='center', fontsize=10)
    if a_ < 1: ax.hlines(1/np.sqrt(1-a_*a_), 1, 30, color=c, lw=1, ls=':')
    print('a', a_, 'sd h=30', sdh[-1])
ax.set_xlim(0, 42); ax.set_xlabel('몇 step 앞인가 (h)'); ax.set_ylabel('예측 표준편차 ($\sigma$ 배)')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'forecast_sd_by_h.png'), dpi=150); plt.close(fig)

# Fig 3: Box-Cox small multiples on a multiplicative seasonal series
rng = np.random.default_rng(3)
n = 80; t = np.arange(n)
level = 20*np.exp(0.03*t)
seas = 1 + 0.25*np.sin(2*np.pi*t/4 + 0.6)
yq = level*seas*np.exp(rng.normal(0, 0.02, n))
fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
for ax, lam, c in zip(axes, [1, 0.5, 0], [GRAY, ORANGE, BLUE]):
    w = np.log(yq) if lam == 0 else (yq**lam - 1)/lam
    ax.plot(t, w, color=c, lw=1.6)
    first = w[:8].max()-w[:8].min(); last = w[-8:].max()-w[-8:].min()
    ax.text(0.01, 0.92, f'λ = {lam:g}   2년 동안의 변동 폭: 처음 {first:.2f}, 마지막 {last:.2f} ({last/first:.1f}배)',
            transform=ax.transAxes, color=INK, va='top', fontsize=10)
    print('lambda', lam, 'amp first/last', first, last, last/first)
axes[-1].set_xlabel('분기')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'boxcox_lambda.png'), dpi=150); plt.close(fig)

# Fig 4: additive vs multiplicative
rng = np.random.default_rng(5)
n = 60; t = np.arange(n)
trend = 50 + 1.5*t
s_add = 8*np.sin(2*np.pi*t/12)
y_add = trend + s_add + rng.normal(0, 1.5, n)
y_mul = trend*(1 + 0.15*np.sin(2*np.pi*t/12))*np.exp(rng.normal(0, 0.015, n))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, yy, c, name in [(axes[0], y_add, BLUE, '덧셈: 진폭이 일정'), (axes[1], y_mul, ORANGE, '곱셈: 진폭이 수준을 따라 커짐')]:
    ax.plot(t, yy, color=c, lw=1.6); ax.plot(t, trend, color=GRAY, lw=1.2, ls='--')
    ax.text(0.02, 0.95, name, transform=ax.transAxes, color=INK, va='top', fontsize=10)
    ax.set_xlabel('월')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'additive_vs_multiplicative.png'), dpi=150); plt.close(fig)

# Fig 5: seasonality (fixed period) vs cycle (AR(2), irregular period)
rng = np.random.default_rng(11)
n = 120; t = np.arange(n)
seasonal = 10*np.sin(2*np.pi*t/12) + rng.normal(0, 1, n)
p1, p2 = 1.6, -0.9
c = np.zeros(n+50)
e = rng.normal(0, 1, n+50)
for i in range(2, n+50): c[i] = p1*c[i-1] + p2*c[i-2] + e[i]
c = c[50:]
fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
axes[0].plot(t, seasonal, color=BLUE, lw=1.6)
for k in range(0, n, 12): axes[0].axvline(k+3, color=AXIS, lw=0.8, ls=':')
axes[0].text(0.01, 0.95, '계절성: 정확히 12개월마다 같은 자리에서 정점', transform=axes[0].transAxes, color=INK, va='top', fontsize=10)
axes[1].plot(t, c, color=ORANGE, lw=1.6)
pk = [i for i in range(1, n-1) if c[i] == c[max(0, i-4):i+5].max() and c[i] > 1]
gaps = np.diff(pk)
axes[1].plot(pk, c[pk], 'o', color=ORANGE, ms=5)
axes[1].text(0.01, 0.95, f'주기: 정점 간격이 매번 다르다 ({gaps.min()}~{gaps.max()}개월)', transform=axes[1].transAxes, color=INK, va='top', fontsize=10)
axes[1].set_xlabel('월')
print('cycle peaks', pk, 'gaps', gaps.tolist())
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'season_vs_cycle.png'), dpi=150); plt.close(fig)
