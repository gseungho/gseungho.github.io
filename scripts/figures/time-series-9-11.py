# Figures for 2026-10-21-time-series-9-acf-pacf, 2026-10-21-time-series-10-arima
# and 2026-10-21-time-series-11-arima-forecasting-seasonal (three posts from the ARIMA 2 notes).
# Run: python scripts/figures/time-series-9-11.py
import os
import numpy as np
from scipy.signal import lfilter
from blogstyle import *

OUT9 = os.path.join(POSTS, "2026-10-21-time-series-9-acf-pacf")
OUT10 = os.path.join(POSTS, "2026-10-21-time-series-10-arima")
OUT11 = os.path.join(POSTS, "2026-10-21-time-series-11-arima-forecasting-seasonal")
for d in (OUT9, OUT10, OUT11):
    os.makedirs(d, exist_ok=True)
DASH = (0, (4, 3))
TPAD = 22  # panel title above the ytitle() line


def psi_weights(phi, theta, n):
    """MA(inf) weights of (1 - sum phi B^i) y = (1 + sum theta B^j) e."""
    imp = np.zeros(n)
    imp[0] = 1.0
    return lfilter(np.r_[1.0, theta], np.r_[1.0, -np.asarray(phi, float)], imp)


def arma_acf(phi, theta, K, n=4000):
    psi = psi_weights(phi, theta, n)
    g = np.array([psi[: n - k] @ psi[k:] for k in range(K + 1)])
    return g[1:] / g[0]


def durbin_levinson(rho):
    """PACF alpha_{k,k} from autocorrelations rho_1..rho_K."""
    K = len(rho)
    r = np.r_[1.0, rho]
    out, a = [], np.zeros(0)
    for n in range(1, K + 1):
        num = r[n] - (a @ r[n - 1:0:-1] if n > 1 else 0.0)
        den = 1.0 - (a @ r[1:n] if n > 1 else 0.0)
        ann = num / den
        a = np.r_[a - ann * a[::-1], ann]
        out.append(ann)
    return np.array(out)


def sample_acf(y, K):
    d = y - y.mean()
    return np.array([d[k:] @ d[:-k] / (d @ d) for k in range(1, K + 1)])


def simulate(phi, theta, n, rng, burn=500):
    e = rng.standard_normal(n + burn)
    return lfilter(np.r_[1.0, theta], np.r_[1.0, -np.asarray(phi, float)], e)[burn:]


def bars(ax, vals, band, theory=None, color=BLUE):
    K = len(vals)
    ax.bar(np.arange(1, K + 1), vals, width=0.4, color=color)
    if theory is not None:
        ax.plot(np.arange(1, K + 1), theory, "o", color=ORANGE, ms=4.5, zorder=3)
    for b in (band, -band):
        ax.axhline(b, color=BLUE, lw=0.9, ls=DASH)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlim(0, K + 1)


# ---------- 9-1. partial correlation: y_t vs y_{t-2} before and after removing y_{t-1} ----------
rng = np.random.default_rng(225)
y = simulate([0.7], [], 300, rng)
yt, y1, y2 = y[2:], y[1:-1], y[:-2]
res_t = yt - np.polyval(np.polyfit(y1, yt, 1), y1)
res_2 = y2 - np.polyval(np.polyfit(y1, y2, 1), y1)
r2 = np.corrcoef(yt, y2)[0, 1]
pr = np.corrcoef(res_t, res_2)[0, 1]
X = np.column_stack([np.ones_like(y1), y1, y2])
coef = np.linalg.lstsq(X, yt, rcond=None)[0]
fig, axes = plt.subplots(1, 2, figsize=(W, 3.3), gridspec_kw={"wspace": 0.28})
axes[0].scatter(y2, yt, s=10, color=BLUE, alpha=0.55, edgecolors="none")
axes[0].set_title(f"$y_{{t-2}}$와 $y_t$: 상관 {r2:.2f}", pad=TPAD)
axes[0].set_xlabel("$y_{t-2}$")
axes[1].scatter(res_2, res_t, s=10, color=ORANGE, alpha=0.55, edgecolors="none")
axes[1].set_title(f"$y_{{t-1}}$로 설명한 몫을 뺀 뒤: 상관 {pr:.2f}", pad=TPAD)
axes[1].set_xlabel("$y_{t-2}$의 잔차")
for ax, lab in ((axes[0], "$y_t$"), (axes[1], "$y_t$의 잔차")):
    ax.grid(True, axis="both")
    ytitle(ax, lab)
save(fig, os.path.join(OUT9, "partial-correlation.svg"))
print(f"[9-1] r_2 {sample_acf(y, 2)[1]:.3f}  corr(y_t,y_t-2) {r2:.3f}  partial corr {pr:.3f}  AR(2) OLS coef of y_t-2 {coef[2]:.3f}")
print(f"[9-1] sample PACF via DL lag 2: {durbin_levinson(sample_acf(y, 2))[1]:.3f}")

# ---------- 9-2. ACF / PACF patterns of AR(2), MA(1), ARMA(1,1) ----------
rng = np.random.default_rng(92)
K, n = 15, 300
band = 1.96 / np.sqrt(n)
cases = [("AR(2), $\\phi = (1, -0.5)$", [1.0, -0.5], []),
         ("MA(1), $\\theta_1 = 0.8$", [], [0.8]),
         ("ARMA(1,1), $\\phi_1 = 0.7$, $\\theta_1 = 0.4$", [0.7], [0.4])]
fig, axes = plt.subplots(2, 3, figsize=(W, 4.6), sharey=True, gridspec_kw={"wspace": 0.12, "hspace": 0.55})
for j, (name, phi, theta) in enumerate(cases):
    y = simulate(phi, theta, n, rng)
    r = sample_acf(y, K)
    rho = arma_acf(phi, theta, K)
    bars(axes[0, j], r, band, rho)
    bars(axes[1, j], durbin_levinson(r), band, durbin_levinson(rho))
    axes[0, j].set_title(name, pad=TPAD)
    axes[1, j].set_xlabel("시차 $k$")
    print(f"[9-2] {name}: rho {np.round(rho[:5], 3)}  pacf {np.round(durbin_levinson(rho)[:5], 3)}")
    print(f"[9-2]   sample acf {np.round(r[:6], 3)}  sample pacf {np.round(durbin_levinson(r)[:6], 3)}"
          f"  max|acf| k>=2 {np.abs(r[1:]).max():.3f}  max|pacf| k>=3 {np.abs(durbin_levinson(r)[2:]).max():.3f}")
axes[0, 0].set_ylim(-0.75, 1.0)
ytitle(axes[0, 0], "ACF")
ytitle(axes[1, 0], "PACF")
save(fig, os.path.join(OUT9, "acf-pacf-patterns.svg"))
print(f"[9-2] band +-{band:.3f}")

# ---------- 10-1. Egyptian exports: theoretical ACF/PACF of the two fitted models (FPP3 9.5) ----------
K = 15
m201 = ([1.6764, -0.8034], [-0.6896])
m400 = ([0.9861, -0.1715, 0.1807, -0.3283], [])
a201, a400 = arma_acf(*m201, K), arma_acf(*m400, K)
p201, p400 = durbin_levinson(a201), durbin_levinson(a400)
fig, axes = plt.subplots(1, 2, figsize=(W, 3.2), sharey=True, gridspec_kw={"wspace": 0.12})
k = np.arange(1, K + 1)
for ax, v1, v2, name in ((axes[0], a201, a400, "ACF"), (axes[1], p201, p400, "PACF")):
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.plot(k, v1, "-o", color=BLUE, ms=5, label="ARIMA(2,0,1)")
    ax.plot(k, v2, "-s", color=ORANGE, ms=5, label="ARIMA(4,0,0)")
    for b in (1.96 / np.sqrt(58), -1.96 / np.sqrt(58)):
        ax.axhline(b, color=GRAY, lw=0.9, ls=DASH)
    ax.set_title(name)
    ax.set_xlabel("시차 $k$")
    ax.set_xlim(0, K + 1)
    ax.set_xticks([1, 5, 10, 15])
top_legend(axes[0], 2)
save(fig, os.path.join(OUT10, "egypt-models.svg"))
print(f"[10-1] ACF 201 {np.round(a201[:8], 3)}")
print(f"[10-1] ACF 400 {np.round(a400[:8], 3)}")
print(f"[10-1] PACF 201 {np.round(p201[:6], 3)}  PACF 400 {np.round(p400[:6], 3)}")
print(f"[10-1] max |ACF diff| lags 1-15 {np.abs(a201 - a400).max():.3f}; band {1.96/np.sqrt(58):.3f}")
ar_roots = 1 / np.roots(np.r_[-np.array(m201[0])[::-1], 1.0])
print(f"[10-1] AR inverse roots {np.round(ar_roots, 3)} modulus {np.abs(ar_roots).round(3)} period {2*np.pi/abs(np.angle(ar_roots[0])):.1f}")


# ---------- 10-2. CSS surface of ARMA(1,1): identified vs common factor ----------
def css(w, phi, theta):
    e = np.zeros_like(w)
    for t in range(1, len(w)):
        e[t] = w[t] - phi * w[t - 1] - theta * e[t - 1]
    return (e[1:] ** 2).sum()


rng = np.random.default_rng(102)
w1 = simulate([0.7], [0.4], 299, rng)
w1 = w1 - w1.mean()
w2 = rng.standard_normal(200)
w2 = w2 - w2.mean()
g = np.linspace(-0.95, 0.95, 77)
fig, axes = plt.subplots(1, 2, figsize=(W, 3.6), sharey=True, gridspec_kw={"wspace": 0.12})
for ax, w, name, true in ((axes[0], w1, "ARMA(1,1) 자료 ($\\phi_1 = 0.7$, $\\theta_1 = 0.4$)", (0.7, 0.4)),
                          (axes[1], w2, "white noise 자료", None)):
    S = np.array([[css(w, p, q) for p in g] for q in g])
    L = np.log(S / S.min())
    ax.contourf(g, g, L, levels=[0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 10], cmap="Blues_r", alpha=0.85)
    ax.contour(g, g, L, levels=[0.02, 0.05, 0.1, 0.2, 0.4], colors=SURF, linewidths=0.6)
    iq, ip = np.unravel_index(S.argmin(), S.shape)
    ax.plot(g[ip], g[iq], "o", color=ORANGE, ms=7, zorder=4)
    if true:
        ax.plot(*true, "x", color=INK, ms=7, mew=1.6, zorder=4)
    else:
        ax.plot(g, -g, color=INK, lw=0.9, ls=DASH)
    ax.set_title(name, pad=TPAD)
    ax.set_xlabel("$\\phi_1$")
    ax.grid(False)
    ax.set_aspect("equal")
    print(f"[10-2] {name}: CSS min at phi={g[ip]:.2f}, theta={g[iq]:.2f}")
ytitle(axes[0], "$\\theta_1$")
save(fig, os.path.join(OUT10, "css-surface.svg"))
for a in (0.0, 0.5, -0.3, 0.8):
    print(f"[10-2] WN CSS at (phi, theta) = ({a}, {-a}): {css(w2, a, -a):.4f}")

# ---------- 11-1. future paths of ARIMA(1,1,1) and the 26-step rule ----------
c, phi, th, T = 0.4, 0.5, 0.3, 200
rng = np.random.default_rng(7)
eps = rng.standard_normal(T + 100)
w = np.zeros(T + 100)
for t in range(1, T + 100):
    w[t] = c + phi * w[t - 1] + eps[t] + th * eps[t - 1]
w, eps = w[100:], eps[100:]
y = np.cumsum(w) + 10
e = np.zeros(T)
for t in range(1, T):
    e[t] = w[t] - c - phi * w[t - 1] - th * e[t - 1]
H, N = 12, 400_000
fc, wp, yp = [], w[-1], y[-1]
for h in range(1, H + 1):
    wp = c + phi * wp + (th * e[-1] if h == 1 else 0.0)
    yp = yp + wp
    fc.append(yp)
fc = np.array(fc)
psi = psi_weights([1 + phi, -phi], [th], H)          # (1 - phi B)(1 - B) = 1 - (1 + phi)B + phi B^2
v = np.cumsum(psi ** 2)
fut = rng.standard_normal((N, H))
sim = np.zeros((N, H))
wprev, yprev, eprev = np.full(N, w[-1]), np.full(N, y[-1]), np.full(N, eps[-1])
for h in range(H):
    wh = c + phi * wprev + fut[:, h] + th * eprev
    yprev, wprev, eprev = yprev + wh, wh, fut[:, h]
    sim[:, h] = yprev
fig, ax = plt.subplots(figsize=(W, 3.4))
tt = np.arange(T - 39, T + 1)
th_ = np.arange(T + 1, T + H + 1)
for p in sim[:60]:
    ax.plot(np.r_[T, th_], np.r_[y[-1], p], color=GRAY, lw=0.6, alpha=0.4)
ax.plot(tt, y[-40:], color=BLUE, lw=1.6, label="관측값")
ax.plot(np.r_[T, th_], np.r_[y[-1], fc], color=ORANGE, lw=2, label="점예측")
ax.plot(th_, fc + 1.96 * np.sqrt(v), color=ORANGE, lw=1.2, ls=DASH, label="95% 예측구간")
ax.plot(th_, fc - 1.96 * np.sqrt(v), color=ORANGE, lw=1.2, ls=DASH)
ax.set_xlabel("시점 $t$")
top_legend(ax, 3)
save(fig, os.path.join(OUT11, "future-paths.svg"))
print(f"[11-1] e_T - eps_T {e[-1] - eps[-1]:.1e}")
for h in (1, 2, 5, 12):
    print(f"[11-1] h={h}: rule {fc[h-1]:.3f}  MC mean {sim[:, h-1].mean():.3f}  var formula {v[h-1]:.3f}  MC var {sim[:, h-1].var():.3f}"
          f"  95% MC {np.quantile(sim[:, h-1], 0.025) - fc[h-1]:.2f} {np.quantile(sim[:, h-1], 0.975) - fc[h-1]:.2f}  formula +-{1.96*np.sqrt(v[h-1]):.2f}")
print(f"[11-1] psi {np.round(psi[:5], 3)}")
dfc = np.diff(np.r_[y[-1], fc])
print(f"[11-1] differenced forecasts h=1,2,5,12 {np.round(dfc[[0, 1, 4, 11]], 3)}  limit c/(1-phi) {c/(1-phi):.3f}")

# ---------- 11-2. long-term forecasts decided by c and d ----------
phi, th, n, H = 0.5, 0.3, 80, 50
fig, axes = plt.subplots(2, 3, figsize=(W, 4.6), gridspec_kw={"wspace": 0.25, "hspace": 0.6})
labels = {0: "0으로 간다", 1: "상수", 2: "직선", 3: "평균", 4: "직선", 5: "2차 곡선"}
for i, cc in enumerate((0.0, None)):
    for d in range(3):
        c_d = {0: 1.0, 1: 0.2, 2: 0.1}[d] if cc is None else 0.0
        rng = np.random.default_rng(1131 if (i, d) == (1, 1) else 1120 + 3 * i + d)
        eps = rng.standard_normal(n + 200)
        w = np.zeros(n + 200)
        for t in range(1, n + 200):
            w[t] = c_d + phi * w[t - 1] + eps[t] + th * eps[t - 1]
        w, eps = w[200:], eps[200:]
        e = np.zeros(n)
        for t in range(1, n):
            e[t] = w[t] - c_d - phi * w[t - 1] - th * e[t - 1]
        fw, wp = [], w[-1]
        for h in range(1, H + 1):
            wp = c_d + phi * wp + (th * e[-1] if h == 1 else 0.0)
            fw.append(wp)
        series, fcs = w.copy(), np.array(fw)
        for _ in range(d):                       # undo each difference: cumulative sums from the last value
            series = np.cumsum(series)
            fcs = series[-1] + np.cumsum(fcs)
        ax = axes[i, d]
        ax.plot(np.arange(1, n + 1), series, color=BLUE, lw=1.2)
        ax.plot(np.arange(n, n + H + 1), np.r_[series[-1], fcs], color=ORANGE, lw=1.8)
        cname = "c = 0" if cc == 0.0 else "c \\ne 0"
        ax.set_title(f"${cname}$, $d = {d}$: {labels[3 * i + d]}")
        ax.set_xticks([])
save(fig, os.path.join(OUT11, "long-term.svg"))
print("[11-2] done")

# ---------- 11-3. seasonal MA(1) and AR(1) with m = 12 ----------
K = 48
sma = np.zeros(12)
sma[11] = 0.6
sar = np.zeros(12)
sar[11] = 0.6
a_ma, a_ar = arma_acf([], sma, K), arma_acf(sar, [], K)
fig, axes = plt.subplots(2, 2, figsize=(W, 4.2), sharey=True, sharex=True, gridspec_kw={"wspace": 0.12, "hspace": 0.45})
for j, (a, name) in enumerate(((a_ma, "ARIMA(0,0,0)(0,0,1)$_{12}$, $\\Theta_1 = 0.6$"),
                               (a_ar, "ARIMA(0,0,0)(1,0,0)$_{12}$, $\\Phi_1 = 0.6$"))):
    for i, vals in enumerate((a, durbin_levinson(a))):
        ax = axes[i, j]
        ax.bar(np.arange(1, K + 1), vals, width=0.6, color=BLUE)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_xticks([12, 24, 36, 48])
        ax.set_xlim(0, K + 1)
    axes[0, j].set_title(name, pad=TPAD)
    axes[1, j].set_xlabel("시차 $k$")
    print(f"[11-3] {name}: acf at 12,24,36 {np.round(a[[11, 23, 35]], 3)}  pacf at 12,24,36 {np.round(durbin_levinson(a)[[11, 23, 35]], 3)}"
          f"  max |pacf| off-season {np.abs(np.delete(durbin_levinson(a), [11, 23, 35, 47])).max():.1e}")
ytitle(axes[0, 0], "ACF")
ytitle(axes[1, 0], "PACF")
save(fig, os.path.join(OUT11, "seasonal-acf.svg"))
