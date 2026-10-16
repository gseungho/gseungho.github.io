# Figures for 2026-10-15-time-series-2-decomposition and 2026-10-15-time-series-3-trend-regression
# (one draft split into two posts). Run: python scripts/figures/time-series-2-3.py
import os
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from blogstyle import *

OUT = os.path.join(POSTS, '2026-10-15-time-series-2-decomposition')
OUT3 = os.path.join(POSTS, '2026-10-15-time-series-3-trend-regression')
os.makedirs(OUT, exist_ok=True); os.makedirs(OUT3, exist_ok=True)
info = {}


def ma2xm(y, m):
    """centred 2 x m moving average (m even); NaN where the window does not fit"""
    k = m // 2
    w = np.r_[0.5, np.ones(m - 1), 0.5] / m
    out = np.full(len(y), np.nan)
    for t in range(k, len(y) - k):
        out[t] = np.dot(w, y[t - k:t + k + 1])
    return out


# ---------- simulated monthly series for the GIF ----------
rng = np.random.default_rng(7)
n = 96
t = np.arange(n)
season12 = 8 * np.sin(2 * np.pi * t / 12) + 3 * np.cos(4 * np.pi * t / 12)
y_m = 100 + 0.35 * t + season12 + rng.normal(0, 2.0, n)
trend_m = ma2xm(y_m, 12)

# ---------- 1. GIF: 2x12-MA window sliding ----------
fig, ax = plt.subplots(figsize=(W, 3.5))
ax.plot(t, y_m, color=GRAY, lw=1.1, label='관측값 (시뮬레이션)')
trend_line, = ax.plot([], [], color=BLUE, lw=1.8, label='2×12-MA 추세')
span = ax.axvspan(0, 1, color=ORANGE, alpha=0.13, lw=0, label='12개월 window')
dot, = ax.plot([], [], 'o', color=BLUE, ms=6.5)
ax.set_xlim(-1, n)
ax.set_ylim(y_m.min() - 4, y_m.max() + 4)
ax.set_xlabel('월')
ytitle(ax, '값')
top_legend(ax, ncol=3)
fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.15)
frames = list(range(6, n - 6, 2))


def update(i):
    c = frames[i]
    span.set_x(c - 6)
    span.set_width(12)
    trend_line.set_data(t[6:c + 1], trend_m[6:c + 1])
    dot.set_data([c], [trend_m[c]])
    return span, trend_line, dot


ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
ani.save(os.path.join(OUT, 'ma_window.gif'), writer=PillowWriter(fps=8), dpi=GIF_DPI)
plt.close(fig)
info['gif_frames'] = len(frames)

# ---------- 2. MA weights ----------
fig, axes = plt.subplots(1, 3, figsize=(W, 2.7), sharey=True, gridspec_kw={'wspace': 0.12})
js = np.arange(-2, 3)
weights = {
    '5-MA': np.full(5, 1 / 5),
    '3×3-MA': np.array([1, 2, 3, 2, 1]) / 9,
    '2×4-MA': np.array([1, 2, 2, 2, 1]) / 8,
}
labels = {'5-MA': ['1/5'] * 5, '3×3-MA': ['1/9', '2/9', '1/3', '2/9', '1/9'],
          '2×4-MA': ['1/8', '1/4', '1/4', '1/4', '1/8']}
for ax, (name, w) in zip(axes, weights.items()):
    ax.bar(js, w, width=0.62, color=BLUE)
    for j, wi, lab in zip(js, w, labels[name]):
        ax.text(j, wi + 0.008, lab, ha='center', va='bottom', fontsize=9, color=INK2)
    ax.set_xticks(js)
    ax.set_xticklabels([f'{j:+d}' if j else '0' for j in js])
    ax.set_title(name, pad=22)  # above the ytitle() of the first panel
    ax.set_xlabel('시차 $j$')
ytitle(axes[0], '가중치 $a_j$')
axes[0].set_ylim(0, 0.4)
save(fig, os.path.join(OUT, 'ma_weights.svg'))

# ---------- 3. classical decomposition, simulated quarterly ----------
rng = np.random.default_rng(11)
nq = 40
tq = np.arange(nq)
s_true = np.array([-30.0, -10.0, 10.0, 30.0])
yq = 400 + 2.0 * tq + s_true[tq % 4] + rng.normal(0, 6, nq)
Tq = ma2xm(yq, 4)
d = yq - Tq
s_tilde = np.array([np.nanmean(d[q::4]) for q in range(4)])
c = s_tilde.mean()
S_hat = s_tilde - c
Sq = S_hat[tq % 4]
Rq = yq - Tq - Sq
info['quarterly_s_tilde'] = np.round(s_tilde, 2).tolist()
info['quarterly_c'] = round(float(c), 3)
info['quarterly_S_hat'] = np.round(S_hat, 2).tolist()
fig, axes = plt.subplots(4, 1, figsize=(W, 6.2), sharex=True, gridspec_kw={'hspace': 0.55})
panels = [(yq, GRAY, '관측값 $y_t$'), (Tq, BLUE, '추세 $\\hat T_t$'),
          (Sq, ORANGE, '계절 $\\hat S_t$'), (Rq, AQUA, '나머지 $\\hat R_t$')]
for ax, (v, col, name) in zip(axes, panels):
    ax.plot(tq, v, color=col)
    ax.set_title(name)
    if name.startswith(('계절', '나머지')):
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
axes[3].set_xlabel('분기')
save(fig, os.path.join(OUT, 'classical_decomposition.svg'))

# ---------- 4. loess at an interior point and at the end point ----------
rng = np.random.default_rng(5)
nl = 40
tl = np.arange(nl, dtype=float)
yl = 10 + 4 * np.sin(tl / 7) + 0.08 * tl + rng.normal(0, 0.8, nl)
q = 11


def loess_at(t0):
    dist = np.abs(tl - t0)
    idx = np.argsort(dist)[:q]
    dmax = dist[idx].max() * 1.0001
    w = (1 - (dist[idx] / dmax) ** 3) ** 3
    X = np.c_[np.ones(q), tl[idx]]
    W_ = np.diag(w)
    b = np.linalg.solve(X.T @ W_ @ X, X.T @ W_ @ yl[idx])
    return idx, w, b


fig, axes = plt.subplots(1, 2, figsize=(W, 3.1), sharey=True, gridspec_kw={'wspace': 0.12})
for ax, t0, lab in [(axes[0], 20.0, '가운데 시점 $t = 20$'), (axes[1], 39.0, '마지막 시점 $t = 39$')]:
    idx, w, b = loess_at(t0)
    ax.scatter(tl, yl, s=14, color=GRAY, zorder=2, linewidth=0)
    ax.scatter(tl[idx], yl[idx], s=14 + 70 * w, color=ORANGE, alpha=0.85, zorder=3,
               edgecolor=SURF, linewidth=0.8)
    xs = np.array([tl[idx].min(), tl[idx].max()])
    ax.plot(xs, b[0] + b[1] * xs, color=BLUE, lw=1.8, zorder=4)
    ax.plot([t0], [b[0] + b[1] * t0], 'o', color=BLUE, ms=7, zorder=5, markeredgewidth=1.2)
    ax.set_title(lab)
    ax.set_xlabel('$t$')
save(fig, os.path.join(OUT, 'loess_local_line.svg'))

# ---------- 5. linear vs quadratic on a convex series ----------
rng = np.random.default_rng(3)
nr = 468
tr = np.arange(1, nr + 1, dtype=float)
yr = 12000 * np.exp(0.0059 * tr) * np.exp(rng.normal(0, 0.03, nr))
X1 = np.c_[np.ones(nr), tr]
X2 = np.c_[np.ones(nr), tr, tr ** 2]
b1 = np.linalg.lstsq(X1, yr, rcond=None)[0]
b2 = np.linalg.lstsq(X2, yr, rcond=None)[0]
f1, f2 = X1 @ b1, X2 @ b2
r1 = yr - f1
sign = np.sign(r1)
runs = 1 + int(np.sum(sign[1:] != sign[:-1]))
info['lin_runs'] = runs
info['lin_resid_first_neg'] = int(np.argmax(r1 < 0)) + 1
info['lin_resid_last_neg'] = int(nr - np.argmax(r1[::-1] < 0))
fig, axes = plt.subplots(1, 2, figsize=(W, 3.2), gridspec_kw={'wspace': 0.22})
ax = axes[0]
ax.plot(tr, yr / 1000, color=GRAY, lw=0.9, label='관측값 (시뮬레이션)')
ax.plot(tr, f1 / 1000, color=BLUE, lw=1.6, label='linear')
ax.plot(tr, f2 / 1000, color=ORANGE, lw=1.6, label='quadratic')
ax.set_xlabel('TIME')
ytitle(ax, '값 (천)')
ax.legend(loc='upper left', bbox_to_anchor=(0, 0.97))
ax = axes[1]
ax.plot(tr, r1 / 1000, color=BLUE, lw=1.0)
ax.axhline(0, color=AXIS, lw=0.9)
ax.fill_between(tr, r1 / 1000, 0, where=r1 > 0, color=BLUE, alpha=0.12, lw=0)
ax.fill_between(tr, r1 / 1000, 0, where=r1 < 0, color=ORANGE, alpha=0.15, lw=0)
ax.set_xlabel('TIME')
ytitle(ax, 'linear 잔차 (천)')
save(fig, os.path.join(OUT3, 'linear_vs_quadratic.svg'))

# ---------- 6. AIC vs SIC penalty per parameter (times T) ----------
T = np.linspace(2, 1000, 2000)
fig, ax = plt.subplots(figsize=(W, 3.2))
ax.plot(T, np.full_like(T, 2.0), color=BLUE, lw=1.8)
ax.plot(T, np.log(T), color=ORANGE, lw=1.8)
ax.set_xscale('log')
ax.axvline(np.e ** 2, color=AXIS, lw=0.8, ls=(0, (4, 3)))
ax.plot([468], [np.log(468)], 'o', color=ORANGE, ms=6.5)
ax.annotate('$T = 468$: $\\ln T = 6.15$', (468, np.log(468)), xytext=(-10, 10),
            textcoords='offset points', ha='right', color=INK2, fontsize=9.5)
ax.text(1000, 2.0 + 0.2, 'AIC: $2$', ha='right', va='bottom', color=INK2)
ax.text(25, np.log(25) + 0.45, 'SIC: $\\ln T$', ha='center', va='bottom', color=INK2)
ax.set_xticks([2, 5, 10, 20, 50, 100, 200, 500, 1000])
ax.set_xticklabels(['2', '5', '10', '20', '50', '100', '200', '500', '1000'])
ax.minorticks_off()
ax.text(np.e ** 2 * 1.06, 0.3, r'$T = e^2 \approx 7.4$', color=INK2, fontsize=9.5)
ax.set_xlabel('관측값 수 $T$ (로그 축)')
ytitle(ax, '모수 1개당 벌점 × $T$')
ax.set_ylim(0, 7.5)
save(fig, os.path.join(OUT3, 'aic_sic_penalty.svg'))

for k, v in info.items():
    print(k, v)
for d in (OUT, OUT3):
    for f in sorted(os.listdir(d)):
        print(f, os.path.getsize(os.path.join(d, f)))
