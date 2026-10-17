# Figures for 2026-10-17-time-series-6-stationarity-differencing, 2026-10-17-time-series-7-backshift-ar
# and 2026-10-17-time-series-8-ma-invertibility (three posts from the ARIMA 1 notes).
# Run: python scripts/figures/time-series-6-8.py
import os
import numpy as np
from scipy.signal import lfilter
from blogstyle import *

OUT6 = os.path.join(POSTS, "2026-10-17-time-series-6-stationarity-differencing")
OUT7 = os.path.join(POSTS, "2026-10-17-time-series-7-backshift-ar")
OUT8 = os.path.join(POSTS, "2026-10-17-time-series-8-ma-invertibility")
for d in (OUT6, OUT7, OUT8):
    os.makedirs(d, exist_ok=True)
DASH = (0, (4, 3))


def acf(y, K):
    d = y - y.mean()
    return np.array([d[k:] @ d[:-k] / (d @ d) for k in range(1, K + 1)])


def ar1(phi, n, rng, burn=300, sigma=1.0):
    return lfilter([1.0], [1.0, -phi], rng.normal(0, sigma, n + burn))[burn:]


def acf_panel(ax, y, K=25):
    r = acf(y, K)
    band = 1.96 / np.sqrt(len(y))
    ax.bar(np.arange(1, K + 1), r, width=0.4, color=BLUE)
    for b in (band, -band):
        ax.axhline(b, color=BLUE, lw=0.9, ls=DASH)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlim(0, K + 1)
    ax.set_xlabel("시차 $k$")
    return r, band


# ---------- 6-1. ensemble: stationary AR(1) vs random walk ----------
rng = np.random.default_rng(6)
T, P = 100, 40
t = np.arange(1, T + 1)
sd_ar = 1 / np.sqrt(1 - 0.8 ** 2)
ar_paths = np.array([ar1(0.8, T, rng) for _ in range(2000)])
rw_paths = np.cumsum(rng.normal(size=(2000, T)), axis=1)
fig, axes = plt.subplots(1, 2, figsize=(W, 3.3), sharey=True, gridspec_kw={"wspace": 0.1})
for ax, paths, band, name in ((axes[0], ar_paths, np.full(T, 1.96 * sd_ar), "AR(1), $\\phi_1 = 0.8$: 퍼짐이 일정하다"),
                              (axes[1], rw_paths, 1.96 * np.sqrt(t), "random walk: 퍼짐이 $\\sqrt{t}$로 커진다")):
    for p in paths[:P]:
        ax.plot(t, p, color=GRAY, lw=0.6, alpha=0.35)
    ax.plot(t, paths[0], color=BLUE, lw=1.4)
    ax.plot(t, band, color=ORANGE, lw=1.2, ls=DASH)
    ax.plot(t, -band, color=ORANGE, lw=1.2, ls=DASH)
    ax.set_title(name)
    ax.set_xlabel("시점 $t$")
save(fig, os.path.join(OUT6, "ensemble.svg"))
for tt in (25, 100):
    print(f"[6-1] t={tt}: sd across paths AR {ar_paths[:, tt-1].std():.2f} (theory {sd_ar:.2f})  RW {rw_paths[:, tt-1].std():.2f} (theory {np.sqrt(tt):.2f})")

# ---------- 6-2. random walk and its difference with ACFs ----------
rng = np.random.default_rng(11)
T = 251
rw = 100 + np.cumsum(rng.normal(size=T))
drw = np.diff(rw)
fig, axes = plt.subplots(2, 2, figsize=(W, 4.8), gridspec_kw={"hspace": 0.6, "wspace": 0.18})
axes[0, 0].plot(np.arange(T), rw, color=GRAY)
axes[0, 0].set_title("random walk $y_t$")
axes[0, 1].plot(np.arange(1, T), drw, color=GRAY)
axes[0, 1].set_title("차분 $y_t - y_{t-1}$")
for ax in axes[0]:
    ax.set_xlabel("시점 $t$")
r_rw, band = acf_panel(axes[1, 0], rw)
r_d, band_d = acf_panel(axes[1, 1], drw)
axes[1, 0].set_title("random walk의 ACF")
axes[1, 1].set_title("차분의 ACF")
axes[1, 0].set_ylim(-0.3, 1.0); axes[1, 1].set_ylim(-0.3, 1.0)
save(fig, os.path.join(OUT6, "acf-random-walk.svg"))
s = np.sign(rw - rw.mean())
print(f"[6-2] rw r1 {r_rw[0]:.3f} r10 {r_rw[9]:.3f} r25 {r_rw[24]:.3f}; band {band:.3f}; crossings {int(np.sum(s[1:] != s[:-1]))}")
print(f"[6-2] diff: lags outside band {[k+1 for k in range(25) if abs(r_d[k]) > band_d]} values {[round(r_d[k],3) for k in range(25) if abs(r_d[k]) > band_d]}; max|r| {np.abs(r_d).max():.3f}")

# ---------- 6-3. seasonal difference then first difference ----------
rng = np.random.default_rng(3)
T, m = 144, 12
eps = rng.normal(0, 1.0, T)
w = np.cumsum(eps)                       # (1-B^12) y = w, (1-B) w = eps
y = np.empty(T)
y[:m] = 100 + 8 * np.sin(2 * np.pi * np.arange(m) / m)
for i in range(m, T):
    y[i] = y[i - m] + 0.3 + w[i]
d12 = y[m:] - y[:-m]
d12_1 = np.diff(d12)
fig, axes = plt.subplots(3, 1, figsize=(W, 5.8), sharex=True, gridspec_kw={"hspace": 0.55})
tt = np.arange(T)
axes[0].plot(tt, y, color=GRAY)
axes[0].set_title("원계열 $y_t$: 계절성이 있고 수준이 떠돈다")
axes[1].plot(tt[m:], d12, color=ORANGE)
axes[1].set_title("계절 차분 $y_t - y_{t-12}$: 계절성은 사라졌지만 수준이 떠돈다")
axes[2].plot(tt[m + 1:], d12_1, color=BLUE)
axes[2].axhline(0, color=AXIS, lw=0.8)
axes[2].set_title("계절 차분 뒤 1차 차분: 0 주위에서 고르게 흔들린다")
axes[2].set_xlabel("월")
for ax in axes:
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
save(fig, os.path.join(OUT6, "seasonal-difference.svg"))
print(f"[6-3] r1 of d12 {acf(d12, 1)[0]:.3f}, of d12_1 {acf(d12_1, 1)[0]:.3f}")

# ---------- 6-4. over-differencing: variance and r1 by number of differences ----------
rng = np.random.default_rng(18)
R, T = 1000, 250
gens = (("white noise (필요한 $d = 0$)", lambda: rng.normal(size=T), BLUE),
        ("random walk ($d = 1$)", lambda: np.cumsum(rng.normal(size=T)), ORANGE),
        ("random walk의 누적합 ($d = 2$)", lambda: np.cumsum(np.cumsum(rng.normal(size=T))), AQUA))
fig, axes = plt.subplots(1, 2, figsize=(W, 3.4), gridspec_kw={"wspace": 0.28})
ds = np.arange(4)
for name, g, col in gens:
    V = np.zeros((R, 4)); Q = np.zeros((R, 4))
    for i in range(R):
        yy = g()
        for d in ds:
            z = np.diff(yy, n=d) if d else yy
            V[i, d] = z.var(); Q[i, d] = acf(z, 1)[0]
    v, q = np.median(V, 0), Q.mean(0)
    axes[0].plot(ds, v, color=col, marker="o", ms=7, label=name)
    axes[1].plot(ds, q, color=col, marker="o", ms=7)
    print(f"[6-4] {name}: median var {np.round(v, 2)} mean r1 {np.round(q, 3)}")
axes[0].set_yscale("log")
axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
ytitle(axes[0], "차분한 계열의 분산 (로그 축)")
axes[1].axhline(-0.5, color=INK2, lw=0.9, ls=DASH)
axes[1].text(0.05, -0.47, "$-0.5$", color=INK2, va="bottom", fontsize=9.5)
axes[1].axhline(0, color=AXIS, lw=0.8)
ytitle(axes[1], "$r_1$")
for ax in axes:
    ax.set_xticks(ds); ax.set_xlabel("차분 횟수 $d$")
axes[0].legend(loc="lower left", bbox_to_anchor=(-0.01, 1.12), ncol=3, borderaxespad=0)
save(fig, os.path.join(OUT6, "over-differencing.svg"))

# ---------- 7-1. AR(1) paths for three phi ----------
rng = np.random.default_rng(37)
T = 120
e = rng.normal(size=T + 300)
fig, axes = plt.subplots(3, 1, figsize=(W, 5.6), sharex=True, gridspec_kw={"hspace": 0.55})
for ax, phi, name in ((axes[0], 0.5, "$\\phi_1 = 0.5$: 평균으로 빨리 돌아온다"),
                      (axes[1], 0.95, "$\\phi_1 = 0.95$: 평균에서 오래 떠돈다"),
                      (axes[2], -0.8, "$\\phi_1 = -0.8$: 평균 위아래로 번갈아 튄다")):
    yy = lfilter([1.0], [1.0, -phi], e)[300:]
    ax.plot(np.arange(1, T + 1), yy, color=BLUE)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_title(name)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
axes[2].set_xlabel("시점 $t$")
save(fig, os.path.join(OUT7, "ar1-paths.svg"))


# ---------- 7-2. shock response d_h of AR(2) ----------
def shock(p1, p2, H):
    d = [1.0, p1]
    for _ in range(3, H + 1):
        d.append(p1 * d[-1] + p2 * d[-2])
    return np.array(d)


H = 15
cases = ((0.5, 0.3, "서로 다른 실근"), (1.0, -0.5, "복소근: 진동하며 줄어든다"),
         (1.2, -0.2, "역근 1이 있다: 0으로 가지 않는다"), (1.5, 0.5, "역근 1.78: 폭발한다"))
fig, axes = plt.subplots(2, 2, figsize=(W, 4.9), sharex=True, gridspec_kw={"hspace": 0.6, "wspace": 0.2})
for ax, (p1, p2, lab) in zip(axes.flat, cases):
    dh = shock(p1, p2, H)
    r = np.roots([1, -p1, -p2])
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.plot(np.arange(1, H + 1), dh, color=BLUE, marker="o", ms=5)
    if np.iscomplexobj(r) and abs(r[0].imag) > 1e-12:
        rs = f"$|r| = {abs(r[0]):.2f}$"
    else:
        rs = "$r = " + ",\\ ".join(f"{v:.2f}" for v in sorted(r.real, reverse=True)) + "$"
    ax.set_title(f"$\\phi = ({p1:g}, {p2:g})$, {rs}\n{lab}", linespacing=1.5)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    print(f"[7-2] {p1},{p2}: roots {np.round(r, 4)} d_1..8 {np.round(dh[:8], 4)} d_15 {dh[-1]:.4g}")
for ax in axes[1]:
    ax.set_xlabel("$h$")
save(fig, os.path.join(OUT7, "shock-response.svg"))

# ---------- 7-3. AR(2) stationarity triangle ----------
fig, ax = plt.subplots(figsize=(W, 3.9))
tri = np.array([[-2, -1], [2, -1], [0, 1], [-2, -1]])
ax.fill(tri[:, 0], tri[:, 1], color=BLUE, alpha=0.12, lw=0)
ax.plot(tri[:, 0], tri[:, 1], color=BLUE, lw=1.4)
p1 = np.linspace(-2, 2, 200)
ax.plot(p1, -p1 ** 2 / 4, color=ORANGE, lw=1.2, ls=DASH)
pts = ((0.5, 0.3, "(0.5, 0.3)"), (1.0, -0.5, "(1, −0.5)"), (1.2, -0.2, "(1.2, −0.2)"), (1.5, 0.5, "(1.5, 0.5)"))
for x, yv, lab in pts:
    ax.plot(x, yv, "o", color=INK, ms=6)
    ax.annotate(lab, (x, yv), xytext=(7, -14 if lab.startswith("(1,") else 4), textcoords="offset points",
                color=INK2, fontsize=9.5)
ax.text(-0.4, -0.6, "복소근", color=ORANGE, ha="center", fontsize=9.5)
ax.text(-0.4, 0.3, "실근", color=ORANGE, ha="center", fontsize=9.5)
ax.text(-1.55, -0.9, "정상", color=BLUE, fontsize=10)
ax.set_xlim(-2.4, 2.4); ax.set_ylim(-1.3, 1.3)
ax.set_xlabel("$\\phi_1$")
ytitle(ax, "$\\phi_2$")
save(fig, os.path.join(OUT7, "ar2-region.svg"))

# ---------- 7-4. lag scatter: regression with and without intercept ----------
rng = np.random.default_rng(44)
n, phi, mu = 200, 0.8, 5.0
Y = mu + ar1(phi, n, rng)
x, yv = Y[:-1], Y[1:]
X = np.column_stack([np.ones_like(x), x])
c_hat, ph = np.linalg.lstsq(X, yv, rcond=None)[0]
ph0 = (yv @ x) / (x @ x)
fig, ax = plt.subplots(figsize=(W, 4.2))
ax.scatter(x, yv, s=12, color=GRAY, linewidth=0, alpha=0.8)
xs = np.array([0, Y.max() + 0.5])
ax.plot(xs, c_hat + ph * xs, color=BLUE, lw=1.8, label=f"절편 있는 회귀: 기울기 {ph:.2f}")
ax.plot(xs, ph0 * xs, color=ORANGE, lw=1.8, label=f"원점을 지나는 회귀: 기울기 {ph0:.2f}")
ax.plot([0], [0], "o", color=ORANGE, ms=7)
top_legend(ax, ncol=2)
ax.set_xlim(-0.3, xs[1] + 0.2); ax.set_ylim(-0.3, xs[1] + 0.2)
ax.set_xlabel("$Y_{t-1}$")
ytitle(ax, "$Y_t$")
save(fig, os.path.join(OUT7, "lag-scatter.svg"))
print(f"[7-4] OLS phi {ph:.4f} c {c_hat:.4f} mu {c_hat/(1-ph):.4f} Ybar {Y.mean():.4f} no-intercept {ph0:.4f}")

# ---------- 8-1. MA(1) vs AR(1): series and ACF ----------
rng = np.random.default_rng(87)
T = 200
e = rng.normal(size=T + 1)
ma = e[1:] + 0.8 * e[:-1]
ar = ar1(0.8, T, rng)
fig, axes = plt.subplots(2, 2, figsize=(W, 4.8), gridspec_kw={"hspace": 0.6, "wspace": 0.18})
axes[0, 0].plot(np.arange(1, T + 1), ma, color=GRAY); axes[0, 0].set_title("MA(1), $\\theta_1 = 0.8$")
axes[0, 1].plot(np.arange(1, T + 1), ar, color=GRAY); axes[0, 1].set_title("AR(1), $\\phi_1 = 0.8$")
for ax in axes[0]:
    ax.set_xlabel("시점 $t$")
r_ma, _ = acf_panel(axes[1, 0], ma, 15)
r_ar, band = acf_panel(axes[1, 1], ar, 15)
axes[1, 0].set_title("MA(1)의 ACF: 시차 1 뒤에 끊긴다")
axes[1, 1].set_title("AR(1)의 ACF: 천천히 줄어든다")
for ax in axes[1]:
    ax.set_ylim(-0.3, 1.0)
save(fig, os.path.join(OUT8, "ma-vs-ar.svg"))
print(f"[8-1] MA r1..4 {np.round(r_ma[:4], 3)} (theory {0.8/1.64:.3f}); AR r1..4 {np.round(r_ar[:4], 3)}; band {band:.3f}")

# ---------- 8-2. invertibility: rho_1(theta) and inversion weights ----------
fig, axes = plt.subplots(1, 2, figsize=(W, 3.3), gridspec_kw={"wspace": 0.3})
th = np.linspace(-4, 4, 801)
ax = axes[0]
ax.axvspan(-1, 1, color=BLUE, alpha=0.08, lw=0)
ax.plot(th, th / (1 + th ** 2), color=BLUE, lw=1.8)
ax.axhline(0.4, color=INK2, lw=0.9, ls=DASH)
for v in (0.5, 2.0):
    ax.plot(v, v / (1 + v ** 2), "o", color=ORANGE, ms=7)
    ax.annotate(f"$\\theta = {v:g}$", (v, 0.4), xytext=(-8 if v < 1 else 8, -16), textcoords="offset points",
                ha="right" if v < 1 else "left", color=INK2, fontsize=9.5)
ax.text(0, 0.53, "가역: $|\\theta| < 1$", color=BLUE, ha="center", fontsize=9.5)
ax.set_xlabel("$\\theta$")
ytitle(ax, "$\\rho_1 = \\theta / (1 + \\theta^2)$")
ax.set_ylim(-0.6, 0.6)
ax = axes[1]
j = np.arange(0, 11)
for v, col in ((0.5, BLUE), (2.0, ORANGE)):
    ax.plot(j, v ** j, color=col, marker="o", ms=5)
    ax.annotate(f"$\\theta = {v:g}$", (j[-1], v ** j[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", color=INK2, fontsize=9.5)
ax.set_yscale("log")
ax.set_xlim(-0.5, 12.5); ax.set_xticks(range(0, 11, 2))
ax.set_xlabel("$j$ (몇 시점 전 관측값인가)")
ytitle(ax, "가중치 $|\\theta|^j$ (로그 축)")
save(fig, os.path.join(OUT8, "invertibility.svg"))

for d in (OUT6, OUT7, OUT8):
    for f in sorted(os.listdir(d)):
        print(os.path.basename(d)[11:], f, os.path.getsize(os.path.join(d, f)))
