# Figures for 2026-10-15-time-series-2-decomposition and 2026-10-15-time-series-3-trend-regression
# (one draft split into two posts). Run: python scripts/figures/time-series-2-3.py
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'public', 'images', 'posts')
OUT = os.path.join(BASE, '2026-10-15-time-series-2-decomposition')
OUT3 = os.path.join(BASE, '2026-10-15-time-series-3-trend-regression')
os.makedirs(OUT, exist_ok=True); os.makedirs(OUT3, exist_ok=True)

BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#898781'
INK, INK2, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7', '#fcfcfb'
mpl.rcParams.update({
    'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 10.5,
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'text.color': INK,
    'xtick.color': GRAY, 'ytick.color': GRAY, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.3, 'legend.frameon': False,
})
DPI = 150
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
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.plot(t, y_m, color=GRAY, lw=1.0, label='관측값 (시뮬레이션)')
trend_line, = ax.plot([], [], color=BLUE, lw=1.6, label='2×12-MA 추세')
span = ax.axvspan(0, 1, color=ORANGE, alpha=0.13, lw=0)
dot, = ax.plot([], [], 'o', color=BLUE, ms=5.5)
ax.set_xlim(-1, n)
ax.set_ylim(y_m.min() - 4, y_m.max() + 4)
ax.set_xlabel('월')
ax.set_ylabel('값')
ax.legend(loc='upper left', ncol=2)
fig.tight_layout()
frames = list(range(6, n - 6, 2))


def update(i):
    c = frames[i]
    span.set_x(c - 6)
    span.set_width(12)
    trend_line.set_data(t[6:c + 1], trend_m[6:c + 1])
    dot.set_data([c], [trend_m[c]])
    return span, trend_line, dot


ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
ani.save(os.path.join(OUT, 'ma_window.gif'), writer=PillowWriter(fps=8), dpi=100)
plt.close(fig)
info['gif_frames'] = len(frames)

# ---------- 2. MA weights ----------
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), sharey=True)
js = np.arange(-2, 3)
weights = {
    '5-MA': np.full(5, 1 / 5),
    '3×3-MA': np.array([1, 2, 3, 2, 1]) / 9,
    '2×4-MA': np.array([1, 2, 2, 2, 1]) / 8,
}
labels = {'5-MA': ['1/5'] * 5, '3×3-MA': ['1/9', '2/9', '1/3', '2/9', '1/9'],
          '2×4-MA': ['1/8', '1/4', '1/4', '1/4', '1/8']}
for ax, (name, w) in zip(axes, weights.items()):
    ax.bar(js, w, width=0.6, color=BLUE)
    for j, wi, lab in zip(js, w, labels[name]):
        ax.text(j, wi + 0.008, lab, ha='center', va='bottom', fontsize=9, color=INK2)
    ax.set_xticks(js)
    ax.set_xticklabels([f'{j:+d}' if j else '0' for j in js])
    ax.set_xlabel(f'{name}   (시차 j)')
    ax.grid(axis='x', visible=False)
axes[0].set_ylabel('가중치 $a_j$')
axes[0].set_ylim(0, 0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'ma_weights.png'), dpi=DPI)
plt.close(fig)

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
fig, axes = plt.subplots(4, 1, figsize=(7.5, 6.4), sharex=True)
axes[0].plot(tq, yq, color=GRAY, marker='o', ms=3.5)
axes[0].set_ylabel('관측값')
axes[1].plot(tq, Tq, color=BLUE)
axes[1].set_ylabel('추세 $\\hat T_t$')
axes[2].plot(tq, Sq, color=ORANGE, marker='o', ms=3.5)
axes[2].axhline(0, color=AXIS, lw=0.8)
axes[2].set_ylabel('계절 $\\hat S_t$')
axes[3].plot(tq, Rq, color=AQUA, marker='o', ms=3.5)
axes[3].axhline(0, color=AXIS, lw=0.8)
axes[3].set_ylabel('나머지 $\\hat R_t$')
axes[3].set_xlabel('분기')
for ax in axes:
    ax.yaxis.set_label_coords(-0.08, 0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'classical_decomposition.png'), dpi=DPI)
plt.close(fig)

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
    W = np.diag(w)
    b = np.linalg.solve(X.T @ W @ X, X.T @ W @ yl[idx])
    return idx, w, b


fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.1), sharey=True)
for ax, t0, lab in [(axes[0], 20.0, '가운데 시점 t = 20'), (axes[1], 39.0, '마지막 시점 t = 39')]:
    idx, w, b = loess_at(t0)
    ax.scatter(tl, yl, s=14, color=GRAY, zorder=2)
    ax.scatter(tl[idx], yl[idx], s=14 + 70 * w, color=ORANGE, alpha=0.85, zorder=3,
               edgecolor=SURF, linewidth=0.8)
    xs = np.array([tl[idx].min(), tl[idx].max()])
    ax.plot(xs, b[0] + b[1] * xs, color=BLUE, lw=1.6, zorder=4)
    ax.plot([t0], [b[0] + b[1] * t0], 'o', color=BLUE, ms=6.5, zorder=5,
            markeredgecolor=SURF, markeredgewidth=1)
    ax.set_xlabel(lab)
axes[0].set_ylabel('값')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'loess_local_line.png'), dpi=DPI)
plt.close(fig)

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
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
ax = axes[0]
ax.plot(tr, yr / 1000, color=GRAY, lw=0.9, label='관측값 (시뮬레이션)')
ax.plot(tr, f1 / 1000, color=BLUE, lw=1.6, label='linear')
ax.plot(tr, f2 / 1000, color=ORANGE, lw=1.6, label='quadratic')
ax.set_xlabel('TIME')
ax.set_ylabel('값 (천)')
ax.legend(loc='upper left')
ax = axes[1]
ax.plot(tr, r1 / 1000, color=BLUE, lw=1.0)
ax.axhline(0, color=AXIS, lw=0.9)
ax.fill_between(tr, r1 / 1000, 0, where=r1 > 0, color=BLUE, alpha=0.12, lw=0)
ax.fill_between(tr, r1 / 1000, 0, where=r1 < 0, color=ORANGE, alpha=0.15, lw=0)
ax.set_xlabel('TIME')
ax.set_ylabel('linear 잔차 (천)')
fig.tight_layout()
fig.savefig(os.path.join(OUT3, 'linear_vs_quadratic.png'), dpi=DPI)
plt.close(fig)

# ---------- 6. AIC vs SIC penalty per parameter (times T) ----------
T = np.linspace(2, 1000, 2000)
fig, ax = plt.subplots(figsize=(7.5, 3.0))
ax.plot(T, np.full_like(T, 2.0), color=BLUE, lw=1.6)
ax.plot(T, np.log(T), color=ORANGE, lw=1.6)
ax.set_xscale('log')
ax.axvline(np.e ** 2, color=AXIS, lw=0.8, ls='--')
ax.plot([468], [np.log(468)], 'o', color=ORANGE, ms=5.5)
ax.annotate('T = 468: ln T = 6.15', (468, np.log(468)), xytext=(-10, 12),
            textcoords='offset points', ha='right', color=INK2, fontsize=9.5)
ax.text(1000, 2.0 + 0.2, 'AIC: 2', ha='right', va='bottom', color=INK2)
ax.text(25, np.log(25) + 0.45, 'SIC: ln T', ha='center', va='bottom', color=INK2)
ax.set_xticks([2, 5, 10, 20, 50, 100, 200, 500, 1000])
ax.set_xticklabels(['2', '5', '10', '20', '50', '100', '200', '500', '1000'])
ax.minorticks_off()
ax.text(np.e ** 2 * 1.06, 0.3, r'$T = e^2 \approx 7.4$', color=INK2, fontsize=9.5)
ax.set_xlabel('관측값 수 T (로그 축)')
ax.set_ylabel('모수 1개당 벌점 × T')
ax.set_ylim(0, 7.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT3, 'aic_sic_penalty.png'), dpi=DPI)
plt.close(fig)

for k, v in info.items():
    print(k, v)
for d in (OUT, OUT3):
    for f in sorted(os.listdir(d)):
        print(f, os.path.getsize(os.path.join(d, f)))
