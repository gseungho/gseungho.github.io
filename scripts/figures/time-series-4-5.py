# Figures for 2026-10-15-time-series-4-exponential-smoothing and 2026-10-15-time-series-5-ets-models.
# Run: python scripts/figures/time-series-4-5.py
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Figures were split across two posts (2026-10-15): parts 4 and 5 of the series.
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'public', 'images', 'posts')
OUT4 = os.path.join(ROOT, "2026-10-15-time-series-4-exponential-smoothing")
OUT5 = os.path.join(ROOT, "2026-10-15-time-series-5-ets-models")
POST5_FIGS = {"interval-width.png", "kalman-gain.png"}
for d in (OUT4, OUT5):
    os.makedirs(d, exist_ok=True)

def out_path(name):
    return os.path.join(OUT5 if name in POST5_FIGS else OUT4, name)

SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "legend.frameon": False, "lines.linewidth": 2,
})
DPI = 150

def save(fig, name):
    fig.savefig(out_path(name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# 1. GIF: level update on exam scores
y = np.array([70, 90, 60, 80, 85], float)
def ses(alpha, l0=75.0):
    l = [l0]
    for v in y:
        l.append(alpha * v + (1 - alpha) * l[-1])
    return np.array(l)
L3, L9 = ses(0.3), ses(0.9)
print("levels a=0.3", np.round(L3, 1), "a=0.9", np.round(L9, 1))

fig, ax = plt.subplots(figsize=(6.8, 3.9))
SUB = 6  # frames per step
nframes = 5 * SUB + 8
def draw(fr):
    ax.clear()
    ax.set_xlim(-0.3, 5.6); ax.set_ylim(55, 95)
    ax.set_xticks(range(6)); ax.set_xticklabels(["시작", "1", "2", "3", "4", "5"])
    ax.set_xlabel("시험 회차 t"); ax.set_ylabel("점수")
    ax.grid(True, color=GRID, linewidth=0.8)
    step = min(fr // SUB, 4); frac = min((fr - step * SUB) / (SUB - 1), 1.0) if fr < 5 * SUB else 1.0
    shown = step + 1
    ax.scatter(range(1, shown + 1), y[:shown], s=64, color=MUTED, zorder=3, label="점수 $y_t$")
    for lab, L, c in (("α = 0.3", L3, BLUE), ("α = 0.9", L9, ORANGE)):
        xs = list(range(0, shown)); ys = list(L[:shown])
        done = len(xs)
        if frac >= 1.0:
            xs.append(shown); ys.append(L[shown]); done += 1
        else:
            xs.append(shown - 1 + frac); ys.append(L[shown - 1] + frac * (L[shown] - L[shown - 1]))
        ax.plot(xs, ys, color=c, marker="o", markersize=7, markevery=list(range(done)), label=f"수준 $\ell_t$ ({lab})")
        k = done - 1
        ax.annotate(f"{ys[k]:.1f}", (xs[k], ys[k]), xytext=(7, -12 if c == BLUE else 6),
                    textcoords="offset points", color=INK2, fontsize=10)
    ax.legend(loc="upper left", fontsize=9, ncol=3)
fig.subplots_adjust(bottom=0.16, left=0.1, right=0.97, top=0.97)
anim = FuncAnimation(fig, draw, frames=nframes, interval=180)
gif = out_path("level-update.gif")
anim.save(gif, writer=PillowWriter(fps=6), dpi=110)
plt.close(fig)
print("gif bytes", os.path.getsize(gif))

# 2. SES weights
fig, ax = plt.subplots(figsize=(6.8, 3.6))
j = np.arange(10); w = 0.27
for k, (a, c) in enumerate(((0.2, BLUE), (0.5, ORANGE), (0.8, AQUA))):
    ax.bar(j + (k - 1) * w, a * (1 - a) ** j, width=w - 0.03, color=c, label=f"α = {a}")
ax.set_xticks(j); ax.set_xticklabels(["$y_T$"] + [f"$y_{{T-{i}}}$" for i in j[1:]])
ax.set_ylabel("가중치 $\\alpha(1-\\alpha)^j$"); ax.legend(loc="upper right")
save(fig, "ses-weights.png")
for a in (0.2, 0.5, 0.8):
    print("weights", a, np.round(a * (1 - a) ** j[:6], 4))

# 3. Holt vs damped
rng = np.random.default_rng(7)
T = 40
yy = 20 + 1.2 * np.arange(1, T + 1) + rng.normal(0, 2.0, T)
a, bs = 0.5, 0.2
l, b = yy[0], 1.0
for v in yy:
    lp = l
    l = a * v + (1 - a) * (l + b)
    b = bs * (l - lp) + (1 - bs) * b
H = 40; h = np.arange(1, H + 1); phi = 0.9
holt = l + h * b
damp = l + b * np.cumsum(phi ** h)
asym = l + phi / (1 - phi) * b
print("holt state l,b", round(l, 2), round(b, 3), "asymptote", round(asym, 2), "damped at h=40", round(damp[-1], 2))
fig, ax = plt.subplots(figsize=(6.8, 3.8))
ax.plot(np.arange(1, T + 1), yy, color=MUTED, linewidth=1.6, label="관측값")
ax.plot(T + h, holt, color=BLUE, label="Holt ($\\phi = 1$)")
ax.plot(T + h, damp, color=ORANGE, label="감쇠 추세 ($\\phi = 0.9$)")
ax.hlines(asym, T, T + H, color=ORANGE, linewidth=1, linestyle=(0, (4, 3)))
ax.annotate(f"$\\ell_T + 9b_T$ = {asym:.1f}", (T + H, asym), xytext=(-110, -16), textcoords="offset points", color=INK2, fontsize=10)
ax.axvline(T + 0.5, color=AXIS, linewidth=1)
ax.set_xlabel("t"); ax.set_ylabel("y"); ax.legend(loc="upper left")
save(fig, "holt-vs-damped.png")

# 4. additive vs multiplicative seasonality
t = np.arange(48); seas = np.array([-1, 0.4, 1.0, -0.4])[t % 4]
level = 20 + 1.0 * t
add = level + 6 * seas + rng.normal(0, 0.8, t.size)
mult = level * (1 + 0.25 * seas) + rng.normal(0, 0.8, t.size)
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
for ax, ser, lab in ((axes[0], add, "덧셈: 진폭 일정"), (axes[1], mult, "곱셈: 진폭이 수준에 비례")):
    ax.plot(t, ser, color=MUTED, linewidth=1.6)
    ax.plot(t, level, color=BLUE, linewidth=1.6)
    ax.set_xlabel(f"분기 t\n{lab}")
axes[0].set_ylabel("y")
axes[0].annotate("수준", (40, level[40]), xytext=(4, -16), textcoords="offset points", color=BLUE, fontsize=10)
save(fig, "additive-vs-multiplicative.png")

# 5. interval half-width
hh = np.arange(1, 13)
fig, ax = plt.subplots(figsize=(6.8, 3.6))
for a, c in ((0.2, BLUE), (0.5, ORANGE), (1.0, AQUA)):
    hw = 1.96 * np.sqrt(1 + a ** 2 * (hh - 1))
    ax.plot(hh, hw, color=c, marker="o", markersize=6)
    ax.annotate(f"α = {a:g}", (hh[-1], hw[-1]), xytext=(6, -4), textcoords="offset points", color=INK2, fontsize=10)
    print("halfwidth", a, round(hw[0], 2), round(hw[-1], 2))
ax.set_xlabel("예측 시점 h"); ax.set_ylabel("95% 구간 half-width ($\\sigma = 1$)")
ax.set_xlim(0.5, 13.6); ax.set_xticks(hh)
save(fig, "interval-width.png")

# 6. Kalman gain convergence
fig, ax = plt.subplots(figsize=(6.8, 3.6))
for q, c in ((0.1, BLUE), (1.0, ORANGE), (10.0, AQUA)):
    V, W = 1.0, q; C = 1e6; K = []
    for _ in range(12):
        R = C + W; k = R / (R + V); C = k * V; K.append(k)
    a = (-q + np.sqrt(q * q + 4 * q)) / 2
    ax.plot(range(1, 13), K, color=c, marker="o", markersize=6)
    ax.hlines(a, 0.6, 12.2, color=c, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(f"q = {q:g}, α = {a:.3f}", (12.2, a), xytext=(8, -4), textcoords="offset points", color=INK2, fontsize=10)
    print("K", q, np.round(K[:5], 3), "alpha", round(a, 3))
ax.set_xlabel("step t"); ax.set_ylabel("칼만 이득 $K_t$")
ax.set_xlim(0.5, 15.8); ax.set_ylim(0, 1.05)
save(fig, "kalman-gain.png")

for d in (OUT4, OUT5):
    for f in sorted(os.listdir(d)):
        print(os.path.basename(d), f, os.path.getsize(os.path.join(d, f)))
