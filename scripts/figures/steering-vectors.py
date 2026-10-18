# Figures for 2026-10-18-analogy-to-steering-vectors.
# Run from the repo root: python scripts/figures/steering-vectors.py
# Figures 1 and 4 use made-up 2-D vectors (labelled as such in the captions).
# Figure 3 retypes Panickssery et al. (2024) Table 3 (Llama 2 13B Chat, layer 13, no system prompt).
import os
import numpy as np
from matplotlib.patches import FancyBboxPatch
from blogstyle import *

OUT = os.path.join(POSTS, "2026-10-18-analogy-to-steering-vectors")
os.makedirs(OUT, exist_ok=True)
DASH = (0, (4, 3))


def arrow(ax, p, q, color, lw=1.8, ls="-", shrink=4):
    ax.annotate("", xy=q, xytext=p,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                                shrinkA=shrink, shrinkB=shrink, mutation_scale=12))


def plain(ax):
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---------- 1. offset between word vectors vs. a direction added to hidden states ----------
fig, axes = plt.subplots(1, 2, figsize=(W, 3.3), gridspec_kw={"wspace": 0.12})
ax = axes[0]
d = np.array([0.45, 1.25])                       # the shared male -> female offset
pairs = {"man": np.array([0.6, 0.4]), "king": np.array([3.0, 0.8]), "uncle": np.array([1.8, 0.1])}
fem = {"man": "woman", "king": "queen", "uncle": "aunt"}
for m, x in pairs.items():
    y = x + d
    arrow(ax, x, y, BLUE)
    ax.plot(*x, "o", color=INK2, ms=7)
    ax.plot(*y, "o", color=INK2, ms=7)
    ax.text(x[0] + 0.08, x[1] - 0.22, m, color=INK, fontsize=10)
    ax.text(y[0] + 0.08, y[1] + 0.08, fem[m], color=INK, fontsize=10)
for a, b in (("man", "king"),):
    ax.plot(*zip(pairs[a], pairs[b]), color=GRAY, lw=1.0, ls=DASH)
    ax.plot(*zip(pairs[a] + d, pairs[b] + d), color=GRAY, lw=1.0, ls=DASH)
ax.text(3.55, 1.25, "$\\mathbf{v}_{woman}-\\mathbf{v}_{man}$", color=BLUE, fontsize=10)
ax.set_xlim(0.2, 5.4); ax.set_ylim(-0.3, 2.6)
ax.set_title("정적 임베딩: 단어 쌍마다 같은 offset")
plain(ax)

ax = axes[1]
hp, hm = np.array([4.3, 1.9]), np.array([3.7, 1.2])   # activations of p+ and p-
v = hp - hm
arrow(ax, hm, hp, BLUE)
ax.plot(*hp, "o", color=BLUE, ms=7); ax.plot(*hm, "o", color=BLUE, ms=7)
ax.text(hp[0] + 0.1, hp[1] + 0.05, "$\\mathbf{h}^l_+$ (“Love”)", color=INK, fontsize=10)
ax.text(hm[0] + 0.1, hm[1] - 0.25, "$\\mathbf{h}^l_-$ (“Hate”)", color=INK, fontsize=10)
c = 1.3
users = np.array([[0.4, 0.25], [1.0, -0.1], [1.6, 0.35], [0.9, 0.7]])
for i, h in enumerate(users):
    arrow(ax, h, h + c * v, ORANGE, lw=1.4)
    ax.plot(*h, "o", color=GRAY, ms=7)
    ax.plot(*(h + c * v), "o", color=SURF, mec=ORANGE, mew=1.5, ms=7)
ax.text(0.15, -0.35, "사용자 prompt의 hidden state", color=INK2, fontsize=9.5)
ax.text(2.3, 0.25, "$+\\,c\\,(\\mathbf{h}^l_+-\\mathbf{h}^l_-)$", color=ORANGE, fontsize=10)
ax.set_xlim(0.0, 5.4); ax.set_ylim(-0.5, 2.6)
ax.set_title("언어 모델: hidden state에 같은 방향을 더한다")
plain(ax)
save(fig, os.path.join(OUT, "offset-vs-steering.svg"))


# ---------- 2. steering pipeline ----------
def box(ax, x, y, w, h, text, ec=AXIS, fc=SURF, color=INK, fs=10):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                ec=ec, fc=fc, lw=1.1))
    ax.text(x, y, text, ha="center", va="center", color=color, fontsize=fs)


fig, ax = plt.subplots(figsize=(W, 3.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.6)
plain(ax)
# top: build the vector from a contrast pair
box(ax, 1.0, 4.0, 1.7, 0.55, "$p_+$  “Love”")
box(ax, 1.0, 3.1, 1.7, 0.55, "$p_-$  “Hate”")
box(ax, 3.4, 3.55, 1.9, 1.45, "layer $1, \\dots, l$\n(같은 모델)", ec=GRAY)
box(ax, 5.6, 4.0, 1.0, 0.55, "$\\mathbf{h}^l_+$")
box(ax, 5.6, 3.1, 1.0, 0.55, "$\\mathbf{h}^l_-$")
box(ax, 8.0, 3.55, 2.6, 0.75, "$\\mathbf{v} = \\mathbf{h}^l_+ - \\mathbf{h}^l_-$", ec=BLUE, color=BLUE)
for y in (4.0, 3.1):
    arrow(ax, (1.85, y), (2.45, y), INK2, lw=1.2, shrink=1)
    arrow(ax, (4.35, y), (5.1, y), INK2, lw=1.2, shrink=1)
    arrow(ax, (6.1, y), (6.7, 3.55 + (y - 3.55) * 0.3), INK2, lw=1.2, shrink=1)
ax.text(8.0, 2.95, "CAA: 쌍 290~1,000개의 차를 평균", ha="center", color=INK2, fontsize=9)
# bottom: add it while running the user prompt
yb = 1.0
box(ax, 1.0, yb, 1.7, 0.6, "사용자 prompt")
box(ax, 3.4, yb, 1.9, 0.6, "layer $1, \\dots, l$", ec=GRAY)
ax.add_patch(plt.Circle((5.6, yb), 0.3, ec=ORANGE, fc=SURF, lw=1.6))
ax.text(5.6, yb, "+", ha="center", va="center", color=ORANGE, fontsize=14)
box(ax, 7.4, yb, 2.2, 0.6, "layer $l+1, \\dots, L$", ec=GRAY)
box(ax, 9.45, yb, 0.9, 0.6, "출력")
arrow(ax, (1.85, yb), (2.45, yb), INK2, lw=1.2, shrink=1)
arrow(ax, (4.35, yb), (5.3, yb), INK2, lw=1.2, shrink=1)
arrow(ax, (5.9, yb), (6.25, yb), INK2, lw=1.2, shrink=1)
arrow(ax, (8.55, yb), (8.95, yb), INK2, lw=1.2, shrink=1)
arrow(ax, (7.6, 2.75), (5.75, yb + 0.33), ORANGE, lw=1.4, shrink=2)
ax.text(6.8, 1.95, "$\\times\\, c$", color=ORANGE, fontsize=11)
ax.text(5.6, yb - 0.62, "layer $l$의 입력에 더한다", ha="center", color=INK2, fontsize=9)
save(fig, os.path.join(OUT, "steering-pipeline.svg"))


# ---------- 3. CAA: probability of the behaviour-matching answer vs multiplier ----------
# Panickssery et al. (2024) Table 3, Llama 2 13B Chat, steering at layer 13, system prompt "None".
caa = {
    "AI Coordination": (0.20, 0.22, 0.39),
    "Corrigibility": (0.45, 0.57, 0.83),
    "Hallucination": (0.42, 0.54, 0.78),
    "Myopic Reward": (0.44, 0.49, 0.66),
    "Survival Instinct": (0.28, 0.35, 0.63),
    "Sycophancy": (0.56, 0.63, 0.60),
    "Refusal": (0.56, 0.78, 0.86),
}
label_y = {"Refusal": 0.885, "Corrigibility": 0.825, "Hallucination": 0.765, "Myopic Reward": 0.695,
           "Survival Instinct": 0.635, "Sycophancy": 0.575, "AI Coordination": 0.39}
color = {"Corrigibility": BLUE, "Sycophancy": ORANGE}
m = np.array([-1, 0, 1])
fig, ax = plt.subplots(figsize=(W, 3.6))
for name, ys in caa.items():
    col = color.get(name, GRAY)
    ax.plot(m, ys, "-o", color=col, lw=2 if name in color else 1.2, ms=8 if name in color else 6,
            zorder=3 if name in color else 2)
    ax.text(1.08, label_y[name], f"{name}  {ys[2]:.2f}", va="center", color=col if name in color else INK2,
            fontsize=9.5)
ax.set_xticks(m, ["−1", "0 (steering 없음)", "+1"])
ax.set_xlim(-1.15, 1.75); ax.set_ylim(0.1, 0.95)
ax.set_xlabel("steering multiplier")
ytitle(ax, "행동에 맞는 답에 준 평균 확률")
save(fig, os.path.join(OUT, "caa-multiplier.svg"))


# ---------- 4. Theorem 2.5 geometry with made-up 2-D vectors ----------
gW = np.array([1.0, 0.25])                        # unembedding representation of W (e.g. male => female)
gZ = np.array([0.55, 1.0])                        # unembedding representation of Z (e.g. lower => upper)
G = np.column_stack([gW, gZ])
M = np.linalg.inv(G @ G.T)                        # causal inner product with D = I (Park et al. Thm 3.4)
lW = M @ gW                                       # embedding representation (Thm 3.2): Riesz map
print(f"[4] gW.gZ (Euclid) = {gW @ gZ:.3f}; <gW,gZ>_C = {gW @ M @ gZ:.3f}; lW = {np.round(lW, 3)}; "
      f"lW.gW = {lW @ gW:.3f}; lW.gZ = {lW @ gZ:.3f}")
lam = np.array([-0.7, 0.6])
cs = np.linspace(0, 1.5, 31)

fig, axes = plt.subplots(1, 2, figsize=(W, 3.4), gridspec_kw={"wspace": 0.28, "width_ratios": [1.05, 1]})
ax = axes[0]
for k in np.arange(-2, 3.01, 0.5):                # level lines of logit_Z = lambda . gZ
    t = np.linspace(-3, 3, 2)
    base = k * gZ / (gZ @ gZ)
    perp = np.array([-gZ[1], gZ[0]]) / np.linalg.norm(gZ)
    ax.plot(base[0] + t * perp[0], base[1] + t * perp[1], color=GRID, lw=1.0, zorder=0)
o = np.array([1.2, -0.85])                         # reference arrows drawn away from the paths
arrow(ax, o, o + gW, BLUE, lw=1.4, ls=DASH, shrink=0); arrow(ax, o, o + gZ, GRAY, lw=1.4, ls=DASH, shrink=0)
ax.text(o[0] + gW[0] + 0.05, o[1] + gW[1] - 0.05, "$\\bar{\\gamma}_W$", color=BLUE, fontsize=11)
ax.text(o[0] + gZ[0] + 0.05, o[1] + gZ[1] + 0.02, "$\\bar{\\gamma}_Z$", color=INK2, fontsize=11)
pW = lam + 1.5 * lW
pN = lam + 1.5 * gW
arrow(ax, lam, pW, BLUE, lw=2.0, shrink=0)
arrow(ax, lam, pN, ORANGE, lw=2.0, shrink=0)
ax.plot(*lam, "o", color=INK, ms=7, zorder=4)
ax.text(lam[0], lam[1] + 0.13, "$\\lambda(x)$", color=INK, fontsize=11, ha="center")
ax.text(pW[0] - 0.6, pW[1] - 0.32, "$+\\,c\\,\\bar{\\lambda}_W$", color=BLUE, fontsize=11)
ax.text(pN[0] + 0.05, pN[1] + 0.02, "$+\\,c\\,\\bar{\\gamma}_W$", color=ORANGE, fontsize=11)
ax.set_xlim(-1.05, 2.5); ax.set_ylim(-1.0, 1.3)
ax.set_aspect("equal")
ax.set_title("회색 선: $Z$의 logit이 같은 점들")
plain(ax)

ax = axes[1]
for vec, col, nm in ((lW, BLUE, "$\\bar{\\lambda}_W$"), (gW, ORANGE, "$\\bar{\\gamma}_W$")):
    ax.plot(cs, cs * (vec @ gW), color=col, lw=2, label=f"{nm} 방향, $W$의 logit")
    ax.plot(cs, cs * (vec @ gZ), color=col, lw=2, ls=DASH, label=f"{nm} 방향, $Z$의 logit")
ax.set_xlabel("더한 크기 $c$")
ytitle(ax, "logit 변화 (만든 값)")
ax.legend(loc="upper left", fontsize=9, handlelength=2.2)
ax.set_xlim(0, 1.5); ax.set_ylim(-0.1, 1.9)
save(fig, os.path.join(OUT, "park-geometry.svg"))
print("saved to", OUT)
