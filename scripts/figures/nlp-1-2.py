# Figures for 2026-10-18-nlp-1-distributional-hypothesis-tf-idf and 2026-10-18-nlp-2-word2vec
# (two posts from the Word Embeddings note, part 1).
# Run: python scripts/figures/nlp-1-2.py
import os
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
from blogstyle import *

OUT1 = os.path.join(POSTS, "2026-10-18-nlp-1-distributional-hypothesis-tf-idf")
OUT2 = os.path.join(POSTS, "2026-10-18-nlp-2-word2vec")
for d in (OUT1, OUT2):
    os.makedirs(d, exist_ok=True)
DASH = (0, (4, 3))

# ---------- shared data: term-document matrix (example A) ----------
DOCS = ["Hamlet", "Macbeth", "Romeo & Juliet", "Richard III", "Julius Caesar", "Tempest", "Othello", "King Lear"]
WORDS = ["knife", "dog", "sword", "love", "like"]
C = np.array([[1, 1, 4, 2, 0, 2, 0, 10],
              [0, 0, 0, 6, 12, 2, 0, 0],
              [2, 2, 7, 5, 0, 5, 0, 17],
              [64, 0, 135, 63, 0, 12, 0, 48],
              [75, 38, 34, 36, 34, 41, 27, 44]], float)


def cos(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


def bar_labels(ax, bars, vals, fmt, dy):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(v),
                ha="center", va="bottom", fontsize=9, color=INK2)


# ---------- 1-1. knife vs other words: dot product, distance, cosine ----------
others = ["sword", "love", "like", "dog"]
idx = [WORDS.index(w) for w in others]
k = C[0]
dot = [k @ C[i] for i in idx]
dist = [np.linalg.norm(k - C[i]) for i in idx]
cosv = [cos(k, C[i]) for i in idx]
fig, axes = plt.subplots(1, 3, figsize=(W, 3.0), gridspec_kw={"wspace": 0.3})
colors = [BLUE if w == "sword" else GRAY for w in others]
for ax, vals, title, fmt, top in ((axes[0], dot, "내적\n클수록 가깝다", "{:.0f}", 1450),
                                  (axes[1], dist, "Euclidean distance\n작을수록 가깝다", "{:.1f}", 190),
                                  (axes[2], cosv, "cosine\n클수록 가깝다", "{:.3f}", 1.17)):
    bars = ax.bar(others, vals, width=0.6, color=colors)
    bar_labels(ax, bars, vals, fmt, top * 0.015)
    ax.set_ylim(0, top)
    ax.set_title(title)
    ax.tick_params(axis="x", labelsize=9.5)
axes[2].set_yticks([0, 0.5, 1.0])
save(fig, os.path.join(OUT1, "knife-similarity.svg"))
print("[1-1] dot", dot, "dist", np.round(dist, 2), "cos", np.round(cosv, 3))

# ---------- 1-2. IDF as a function of df ----------
N = 8
df = (C > 0).sum(1)
idf = np.log10(N / df)
fig, ax = plt.subplots(figsize=(W, 2.7))
grid = np.linspace(1, 8, 200)
ax.plot(grid, np.log10(N / grid), color=GRAY, lw=1.4)
ax.plot(np.arange(1, 9), np.log10(N / np.arange(1, 9)), "o", color=GRAY, ms=4)
labels = {"dog": "dog", "love": "love", "knife": "knife, sword", "like": "like"}
for w, lab in labels.items():
    i = WORDS.index(w)
    ax.plot(df[i], idf[i], "o", color=BLUE, ms=8)
    ax.annotate(f"{lab}  (df {df[i]}, IDF {idf[i]:.3f})".replace("0.000", "0"), (df[i], idf[i]), xytext=(8, 6),
                textcoords="offset points", fontsize=9.5, color=INK)
ax.set_xlim(0.7, 9.6)
ax.set_ylim(-0.08, 1.0)
ax.set_xticks(range(1, 9))
ax.set_xlabel("df: 단어가 나온 문서 수 (전체 N = 8)")
ytitle(ax, "IDF = $\\log_{10}(N/\\mathrm{df})$")
save(fig, os.path.join(OUT1, "idf-curve.svg"))
print("[1-2] df", df, "idf", np.round(idf, 3))

# ---------- 1-3. document cosines before and after IDF ----------
T = C * idf[:, None]
pairs = [("Hamlet", "Romeo & Juliet"), ("Hamlet", "Julius Caesar"), ("Richard III", "Julius Caesar")]
raw = [cos(C[:, DOCS.index(a)], C[:, DOCS.index(b)]) for a, b in pairs]
tf = [cos(T[:, DOCS.index(a)], T[:, DOCS.index(b)]) for a, b in pairs]
fig, ax = plt.subplots(figsize=(W, 2.8))
x = np.arange(len(pairs))
b1 = ax.bar(x - 0.18, raw, width=0.34, color=GRAY, label="횟수 그대로")
b2 = ax.bar(x + 0.18, tf, width=0.34, color=BLUE, label="TF-IDF")
bar_labels(ax, b1, raw, "{:.3f}", 0.015)
bar_labels(ax, b2, tf, "{:.3f}", 0.015)
ax.set_xticks(x, [f"{a} / {b}" for a, b in pairs])
ax.set_ylim(0, 1.15)
ax.set_yticks([0, 0.5, 1.0])
ytitle(ax, "문서 벡터 사이의 cosine")
top_legend(ax, 2)
save(fig, os.path.join(OUT1, "doc-cosine-idf.svg"))
print("[1-3] raw", np.round(raw, 3), "tfidf", np.round(tf, 3))

# ---------- 2-1. skip-gram training pairs from a sliding window ----------
sent = ["a", "cocktail", "with", "gin", "and", "seltzer"]
m = 2
fig, ax = plt.subplots(figsize=(W, 3.1))
ax.axis("off")
bw, bh, gap = 1.0, 0.62, 0.12
total = 0
for r, c in enumerate(range(len(sent))):
    y = -r
    n = 0
    for j, wd in enumerate(sent):
        xj = j * (bw + gap)
        if j == c:
            face, edge = (0.922, 0.408, 0.204, 0.22), ORANGE
        elif abs(j - c) <= m:
            face, edge = (0.165, 0.471, 0.839, 0.16), BLUE
            n += 1
        else:
            face, edge = SURF, AXIS
        ax.add_patch(FancyBboxPatch((xj, y - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=face, ec=edge, lw=1.0))
        ax.text(xj + bw / 2, y, wd, ha="center", va="center", fontsize=9.5,
                color=INK if (j == c or abs(j - c) <= m) else MUTED)
    total += n
    ax.text(len(sent) * (bw + gap) + 0.15, y, f"쌍 {n}개", ha="left", va="center", fontsize=9.5, color=INK2)
ax.text(len(sent) * (bw + gap) + 0.15, -len(sent) + 0.25, f"합계 {total}개", ha="left", va="center",
        fontsize=9.5, color=INK)
ax.set_xlim(-0.1, len(sent) * (bw + gap) + 1.4)
ax.set_ylim(-len(sent) + 0.0, 0.5)
ax.set_aspect("auto")
save(fig, os.path.join(OUT2, "skipgram-window.svg"))
print("[2-1] total pairs", total, "max 2mT", 2 * m * len(sent))

# ---------- shared data: skip-gram toy network (example B) ----------
V_WORDS = ["gin", "cocktail", "globe"]
Wm = np.array([[-0.5, 1.3], [0.4, 0.08], [1.7, 3.1]])
Wp = np.array([[4.1, 0.7, 1.3], [-0.9, 1.3, -3.4]])
xvec = np.array([0, 1, 0])
h = xvec @ Wm
z = Wp.T @ h
p = np.exp(z) / np.exp(z).sum()

# ---------- 2-2. skip-gram as a shallow network ----------
fig, ax = plt.subplots(figsize=(W, 3.8))
ax.axis("off")
R = 0.34
xin, xh, xz, xy = 0.9, 3.0, 5.1, 7.1
yin = [1.25, 0.0, -1.25]
yh = [0.62, -0.62]
for i in range(3):
    for j in range(2):
        hot = i == 1
        ax.plot([xin + R, xh - R], [yin[i], yh[j]], color=BLUE if hot else GRID, lw=1.6 if hot else 1.0, zorder=1)
for j in range(2):
    for i in range(3):
        ax.plot([xh + R, xz - R], [yh[j], yin[i]], color=MUTED, lw=1.0, zorder=1)
for i, wd in enumerate(V_WORDS):
    ax.add_patch(Circle((xin, yin[i]), R, fc=SURF, ec=BLUE if xvec[i] else AXIS, lw=1.6 if xvec[i] else 1.0, zorder=2))
    ax.text(xin, yin[i], f"{xvec[i]}", ha="center", va="center", fontsize=10, color=INK, zorder=3)
    ax.text(xin - R - 0.12, yin[i], wd, ha="right", va="center", fontsize=9.5, color=INK2)
for j in range(2):
    ax.add_patch(Circle((xh, yh[j]), R, fc=SURF, ec=BLUE, lw=1.2, zorder=2))
    ax.text(xh, yh[j], f"{h[j]:g}", ha="center", va="center", fontsize=9, color=INK, zorder=3)
for i in range(3):
    ax.add_patch(Circle((xz, yin[i]), R, fc=SURF, ec=MUTED, lw=1.0, zorder=2))
    ax.text(xz, yin[i], f"{z[i]:.3f}", ha="center", va="center", fontsize=8.5, color=INK, zorder=3)
# the output: one vector y-hat with three components
ax.add_patch(FancyBboxPatch((xy - 0.55, -1.75), 1.1, 3.5, boxstyle="round,pad=0,rounding_size=0.15",
                            fc=SURF, ec=ORANGE, lw=1.4, zorder=1))
for i, wd in enumerate(V_WORDS):
    ax.text(xy, yin[i] + 0.12, f"{p[i]:.3f}", ha="center", va="center", fontsize=10, color=INK)
    ax.text(xy, yin[i] - 0.25, wd, ha="center", va="center", fontsize=8.5, color=INK2)
ax.annotate("", (xy - 0.62, 0), (xz + R + 0.12, 0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0))
ax.text((xz + R + 0.12 + xy - 0.62) / 2, 0.12, "softmax", ha="center", va="bottom", fontsize=9, color=INK2)
ax.text((xin + xh) / 2, 1.85, "$\\mathbf{W}$  (3×2)", ha="center", fontsize=10, color=INK)
ax.text((xh + xz) / 2, 1.85, "$\\mathbf{W}'$  (2×3)", ha="center", fontsize=10, color=INK)
for xx, lab in ((xin, "one-hot $\\mathbf{x}$"), (xh, "$\\mathbf{h}$"), (xz, "점수 $\\mathbf{z}$"),
                (xy, "$\\hat{\\mathbf{y}}$: 벡터 하나")):
    ax.text(xx, -2.1, lab, ha="center", va="center", fontsize=10, color=INK)
ax.set_xlim(-0.3, 7.9)
ax.set_ylim(-2.35, 2.15)
ax.set_aspect("equal")
save(fig, os.path.join(OUT2, "skipgram-network.svg"))
print("[2-2] h", h, "z", np.round(z, 3), "p", np.round(p, 3), "loss", round(-np.log(p[0]), 3))

# ---------- 2-3. one gradient step on v_cocktail ----------
E = Wp @ p
grad = -(Wp[:, 0] - E)
lr = 0.5
h2 = h - lr * grad
z2 = Wp.T @ h2
p2 = np.exp(z2) / np.exp(z2).sum()
fig, (ax, bx) = plt.subplots(1, 2, figsize=(W, 3.5), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.3})
ax.grid(True, axis="both")
ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
offs = {"gin": (6, 4), "cocktail": (6, 2), "globe": (6, -4)}
for i, wd in enumerate(V_WORDS):
    u = Wp[:, i]
    ax.annotate("", u, (0, 0), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.0), zorder=2)
    ax.annotate(f"$\\mathbf{{u}}_{{\\mathrm{{{wd}}}}}$", u, xytext=offs[wd], textcoords="offset points",
                fontsize=10, color=INK2)
ax.plot(*E, "X", color=ORANGE, ms=9, zorder=4)
ax.annotate("기대", E, xytext=(-8, -16), textcoords="offset points", fontsize=9.5, color=ORANGE, ha="center")
ax.plot([E[0], Wp[0, 0]], [E[1], Wp[1, 0]], color=ORANGE, lw=1.2, ls=DASH, zorder=3)
ax.text((E[0] + Wp[0, 0]) / 2, E[1] + 0.22, "관측 − 기대", ha="center", va="bottom", fontsize=9, color=ORANGE)
ax.plot(*h, "o", color=SURF, mec=BLUE, mew=1.6, ms=8, zorder=5)
ax.plot(*h2, "o", color=BLUE, ms=8, zorder=5)
ax.annotate("", h2, h, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4, shrinkA=5, shrinkB=5), zorder=5)
ax.annotate("$\\mathbf{v}_{\\mathrm{cocktail}}$", h, xytext=(-6, 10), textcoords="offset points",
            fontsize=10, color=BLUE, ha="center")
ax.set_xlim(-0.6, 4.9)
ax.set_ylim(-3.8, 1.9)
ax.set_aspect("equal")
ax.set_title("출력 벡터와 중심 단어 벡터")
x = np.arange(3)
b1 = bx.bar(x - 0.2, p, width=0.36, color=GRAY, label="update 전")
b2 = bx.bar(x + 0.2, p2, width=0.36, color=BLUE, label="update 후")
bar_labels(bx, b1, p, "{:.3f}", 0.015)
bar_labels(bx, b2, p2, "{:.3f}", 0.015)
bx.set_xticks(x, V_WORDS)
bx.set_ylim(0, 1.1)
bx.set_yticks([0, 0.5, 1.0])
ytitle(bx, "$P(w \\mid \\mathrm{cocktail})$")
top_legend(bx, 2)
save(fig, os.path.join(OUT2, "gradient-step.svg"))
print("[2-3] expected", np.round(E, 3), "grad", np.round(grad, 3), "v'", np.round(h2, 3), "p'", np.round(p2, 3))

# ---------- 2-4. noise distribution: unigram vs unigram^(3/4) ----------
r = np.arange(1, 11)
uni = (1 / r) / (1 / r).sum()
p34 = uni ** 0.75 / (uni ** 0.75).sum()
fig, ax = plt.subplots(figsize=(W, 2.7))
ax.bar(r - 0.18, uni, width=0.34, color=GRAY, label="빈도 그대로 $U(w)$")
ax.bar(r + 0.18, p34, width=0.34, color=BLUE, label="$U(w)^{3/4}/Z$")
ax.set_xticks(r)
ax.set_xlabel("빈도 순위 (Zipf 분포를 따르는 단어 10개)")
ytitle(ax, "negative로 뽑힐 확률")
top_legend(ax, 2)
save(fig, os.path.join(OUT2, "noise-distribution.svg"))
print("[2-4] rank1", round(uni[0], 3), "->", round(p34[0], 3), "rank10", round(uni[-1], 3), "->", round(p34[-1], 3))
