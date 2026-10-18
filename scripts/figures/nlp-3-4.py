# Figures for 2026-10-18-nlp-3-evaluating-embeddings-glove and
# 2026-10-18-nlp-4-subword-composition-contextual (word embeddings, parts 3 and 4 of the NLP series).
# Run from the repo root: python scripts/figures/nlp-3-4.py
#
# The GloVe numbers below were computed once from Stanford's glove.6B.50d.txt (glove.6B.zip,
# Wikipedia 2014 + Gigaword 5), the original WordSim-353 combined.csv (gabrilovich.com) and
# word2vec's questions-words.txt, and are embedded here so the script runs without those files:
#   - WS353: plain cosine of the two 50d vectors for each of the 353 pairs (all pairs in vocab).
#   - ANALOGY: questions-words.txt, lowercased, unit-normalized vectors, full 400K vocabulary,
#     answer = argmax cosine to b - a + c (as compute-accuracy.c does).
#   - PCA: 2-D PCA of the ten 50d vectors listed (fitted on those ten words only).
import os
import numpy as np
from matplotlib.patches import FancyBboxPatch
from blogstyle import *

OUT3 = os.path.join(POSTS, "2026-10-18-nlp-3-evaluating-embeddings-glove")
OUT4 = os.path.join(POSTS, "2026-10-18-nlp-4-subword-composition-contextual")
for d in (OUT3, OUT4):
    os.makedirs(d, exist_ok=True)
DASH = (0, (4, 3))

# ---------------------------------------------------------------- NLP 3
# 3-1. WordSim-353: human score vs GloVe 50d cosine
WS353 = (
    "6.77,0.619 7.35,0.615 10,1.000 7.46,0.665 7.62,0.577 7.58,0.806 5.77,0.666 6.31,0.766 7.5,0.664 "
    "6.77,0.871 7.42,0.761 6.85,0.675 6.19,0.840 5.92,0.712 7,0.798 6.62,0.582 6.81,0.672 4.62,0.407 "
    "5.81,0.618 7.08,0.735 8.08,0.885 1.62,0.524 1.31,0.323 0.92,0.208 1.81,0.221 6.69,0.370 3.73,0.271 "
    "0.92,0.388 7.46,0.606 8.12,0.636 7.73,0.585 9.15,0.899 0.31,-0.130 0.23,0.113 8.58,0.784 5.92,0.254 "
    "6.69,0.504 8.46,0.805 7.65,0.753 1.62,0.338 9.44,0.165 8.62,0.502 9.03,0.896 6.81,0.879 6.63,0.597 "
    "7.56,0.410 6.73,0.695 7.65,0.532 2.5,0.303 8.38,0.669 7.38,0.692 6.19,0.378 6.73,0.459 7.92,0.693 "
    "8.12,0.225 7.35,0.900 4.88,0.333 5.54,0.362 8.46,0.661 8.13,0.716 3.04,0.437 1.31,0.352 5.96,0.440 "
    "6.87,0.751 7.85,0.750 2.65,0.410 8.94,0.696 8.96,0.739 9.29,0.827 8.83,0.428 9.1,0.793 8.87,-0.274 "
    "9.02,0.724 9.29,0.830 8.79,0.665 7.52,0.676 7.1,0.232 7.38,0.449 6.46,0.515 6.27,0.524 2.69,-0.087 "
    "4.46,0.248 5.85,0.391 5,0.051 2.08,0.517 4.42,-0.025 4.38,0.459 1.85,0.340 3.08,0.409 0.92,0.433 "
    "3.15,0.627 0.92,0.373 0.54,0.242 2.08,0.154 0.54,0.220 0.62,0.011 8.42,0.600 9.08,0.899 9.04,0.534 "
    "8.27,0.682 7.57,0.663 7.29,0.464 8.5,0.636 7.73,0.558 6.88,0.315 5.65,0.538 3.31,0.511 8,0.404 "
    "8,0.303 7.08,0.181 6.85,0.341 7,0.454 4.77,0.076 5.62,0.269 5.87,0.367 8.08,0.804 7,0.481 "
    "6.85,0.280 7.42,0.482 6.58,0.406 6.42,0.484 8.21,0.529 7.69,0.505 7.23,0.496 6.71,0.829 5.58,0.526 "
    "7.48,0.659 8.45,0.476 8.06,0.537 8.08,0.652 8.02,0.520 8.11,0.564 7.92,0.716 7.94,0.365 5.85,0.518 "
    "3.85,0.339 2.81,0.194 6.65,0.636 2.5,0.213 1.77,0.193 6.04,0.345 6.58,0.355 6.85,0.098 2.4,0.172 "
    "2.92,-0.046 3.69,0.058 2.15,0.057 7.25,0.390 5,0.318 1.92,0.260 5.9,0.305 7.42,0.373 7.27,0.475 "
    "1.81,0.392 5.06,0.596 5.09,0.439 6.78,0.558 6.06,0.545 6.94,0.464 8.31,0.818 4.59,0.576 2.94,0.392 "
    "5.63,0.396 8.16,0.796 7.53,0.389 4.56,0.479 6.34,0.668 6.56,0.510 2.38,0.268 2.22,0.745 8.66,0.875 "
    "4.47,0.531 5.34,0.392 3.69,0.522 3,0.313 8.13,0.760 6.31,0.645 6.22,0.741 6.5,0.661 3.91,0.376 "
    "2.56,0.378 3,0.392 5.63,0.515 7.59,0.502 3.16,0.541 1.19,0.175 3.31,0.451 6.63,0.625 4.75,0.676 "
    "3.69,0.452 4.25,0.607 6.56,0.713 4.25,0.570 5.88,0.800 5.94,0.561 7.56,0.687 2.75,0.515 7.03,0.441 "
    "5.47,0.334 6.47,0.390 7.91,0.489 4.97,0.572 5,0.342 5.19,0.178 7.03,0.769 3.44,0.679 2.31,0.185 "
    "5.91,0.218 7.38,0.376 8.13,0.960 4.63,0.326 5.25,0.547 5.03,0.592 6.69,0.482 7.88,0.726 4.5,0.657 "
    "4.75,0.498 4.47,0.730 3.25,0.458 5.63,0.408 3.69,0.456 2.94,0.161 5.28,0.450 5,0.470 6.44,0.501 "
    "4.13,0.653 4.75,0.635 2.38,0.306 4.94,0.536 8.06,0.630 5.31,0.576 8.03,0.312 5.94,0.389 6,0.407 "
    "5.41,0.541 1.81,0.469 8.97,0.643 6,0.493 6.72,0.695 8,0.612 4.81,0.692 3.88,0.612 5.16,0.176 "
    "2.25,0.530 6.44,0.583 8.88,0.797 6.88,0.604 4.94,0.356 2.56,0.302 6.38,0.486 7.81,0.604 1.75,0.388 "
    "4.25,0.868 3.88,0.506 2.88,0.509 7.63,0.689 6.48,0.452 8.44,0.740 7.5,0.726 8.59,0.629 6.34,0.631 "
    "3.38,0.432 6,0.674 3.88,0.393 7.63,0.672 7.78,0.745 9.22,0.206 7.38,0.705 6.09,0.512 8.5,0.881 "
    "8.31,0.537 7.13,0.586 5.91,0.581 6.47,0.653 3.38,0.764 3.63,0.447 7.13,0.341 7.89,0.720 5.97,0.700 "
    "7.03,0.657 7.69,0.780 7.47,0.674 6.19,0.726 6.97,0.581 3.56,0.689 7.47,0.371 8.34,0.734 8.7,0.745 "
    "7.81,0.410 5.7,0.474 6.22,0.574 6.34,0.532 4.06,0.783 4.47,0.690 5.97,0.706 7.61,0.443 8.36,0.883 "
    "7.41,0.383 2.69,0.393 3.94,0.791 7.16,0.525 5.63,0.396 7.53,0.675 8.31,0.721 8.81,0.710 6.25,0.534 "
    "8.3,0.886 5.25,0.405 8.53,0.727 7.94,0.536 6.88,0.557 5.94,0.689 4.06,0.680 6.25,0.311 7.72,0.731 "
    "6.19,0.443 2.97,0.336 1.94,0.299 3.75,0.489 3.31,0.518 3.69,0.489 7.44,0.808 6.41,0.700 5.44,0.650 "
    "6.25,0.436 2.63,0.327 0.88,0.137 3.19,0.498 4.69,0.464 6.75,0.622 5.31,0.235 7.31,0.542 5.75,0.341 "
    "3.97,0.538 3.47,0.559 3.63,0.528 5.56,0.771 7.83,0.733 3.88,0.540 5.31,0.468 6.81,0.447 7.59,0.729 "
    "7.19,0.652 4.38,0.395 6.53,0.568 6.19,0.350 7.69,0.482 6.31,0.453 6.03,0.270 8.34,0.576 6.25,0.450 "
    "6.34,0.659 3.78,0.702 "
)
pairs = np.array([[float(v) for v in t.split(",")] for t in WS353.split()])
human, cosv = pairs[:, 0], pairs[:, 1]
assert len(human) == 353


def ranks(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):          # average ranks for ties
        m = x == v
        r[m] = r[m].mean()
    return r


rho = np.corrcoef(ranks(human), ranks(cosv))[0, 1]
pear = np.corrcoef(human, cosv)[0, 1]
print(f"[3-1] WS353 GloVe 50d: Spearman {rho:.3f}, Pearson {pear:.3f}")
EX = [("midday / noon", 9.29, 0.830), ("car / automobile", 8.94, 0.696),
      ("professor / cucumber", 0.31, -0.130), ("king / cabbage", 0.23, 0.113)]
fig, ax = plt.subplots(figsize=(W, 3.6))
ax.scatter(human, cosv, s=18, color=GRAY, alpha=0.55, linewidth=0, label="WordSim-353의 353쌍")
for name, hx, cy in EX:
    ax.scatter([hx], [cy], s=60, color=BLUE, zorder=3)
    dx, ha = (-8, "right") if hx > 5 else (8, "left")
    ax.annotate(name, (hx, cy), xytext=(dx, 0), textcoords="offset points", ha=ha, va="center",
                color=INK, fontsize=9.5)
ax.set_xlim(-0.3, 10.3); ax.set_ylim(-0.35, 1.05)
ax.set_xlabel("사람 점수 (0~10)")
ytitle(ax, "GloVe 50d cosine")
save(fig, os.path.join(OUT3, "ws353-scatter.svg"))

# 3-2. gender pairs in GloVe 50d, 2-D PCA of the ten vectors
PCA = {"man": (-2.66, 1.22), "woman": (-3.12, -0.69), "king": (2.10, 1.70), "queen": (1.38, -1.14),
       "uncle": (0.19, 1.60), "aunt": (-0.31, -1.21), "brother": (-0.19, 1.93), "sister": (-0.77, -1.39),
       "prince": (2.04, 0.66), "princess": (1.36, -2.69)}
GP = [("man", "woman"), ("king", "queen"), ("uncle", "aunt"), ("brother", "sister"), ("prince", "princess")]
fig, ax = plt.subplots(figsize=(W, 4.0))
for a, b in GP:
    (x0, y0), (x1, y1) = PCA[a], PCA[b]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6, shrinkA=6, shrinkB=6, mutation_scale=12))
for a, b in (("man", "king"), ("woman", "queen")):
    (x0, y0), (x1, y1) = PCA[a], PCA[b]
    ax.plot([x0, x1], [y0, y1], color=GRAY, lw=1.2, ls=DASH, zorder=1)
for w, (x, y) in PCA.items():
    ax.scatter([x], [y], s=46, color=INK2, zorder=3)
    off = {"brother": (-8, 6, "right"), "uncle": (8, 4, "left"), "aunt": (8, -2, "left"),
           "sister": (-8, -2, "right"), "man": (-8, 0, "right"), "woman": (-8, 0, "right")}.get(w, (8, 0, "left"))
    ax.annotate(w, (x, y), xytext=off[:2], textcoords="offset points", ha=off[2], va="center", fontsize=10, color=INK)
ax.set_xlim(-4.6, 3.6); ax.set_ylim(-3.2, 2.5)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
ax.spines["bottom"].set_visible(False)
ax.set_aspect("equal")
save(fig, os.path.join(OUT3, "analogy-pca.svg"))

# 3-3. analogy accuracy per category (GloVe 50d)
# name in the post, file section, questions, correct (inputs excluded), top-1 is an input word (no exclusion)
ANALOGY = [
    ("나라와 수도 (주요국)", "capital-common-countries", 506, 401, 126),
    ("나라와 수도 (전 세계)", "capital-world", 4524, 3098, 1297),
    ("나라와 화폐", "currency", 866, 72, 355),
    ("도시와 그 도시가 있는 주", "city-in-state", 2467, 378, 1492),
    ("남성과 여성", "family", 506, 349, 230),
    ("형용사와 부사", "gram1-adjective-to-adverb", 992, 151, 535),
    ("반의어 (un-, im- 등)", "gram2-opposite", 812, 77, 403),
    ("비교급", "gram3-comparative", 1332, 690, 789),
    ("최상급", "gram4-superlative", 1122, 321, 421),
    ("현재분사 (-ing)", "gram5-present-participle", 1056, 439, 827),
    ("나라와 국적 형용사", "gram6-nationality-adjective", 1599, 1375, 143),
    ("과거형", "gram7-past-tense", 1560, 585, 1231),
    ("복수형 명사", "gram8-plural", 1332, 798, 906),
    ("동사의 -s형 (plural verbs)", "gram9-plural-verbs", 870, 299, 536),
]
names = [r[0] for r in ANALOGY]
n = np.array([r[2] for r in ANALOGY]); k = np.array([r[3] for r in ANALOGY]); top = np.array([r[4] for r in ANALOGY])
acc, inp = k / n, top / n
print(f"[3-3] questions {n.sum()} (semantic {n[:5].sum()}, syntactic {n[5:].sum()}); "
      f"accuracy all {k.sum()/n.sum():.3f}, semantic {k[:5].sum()/n[:5].sum():.3f}, syntactic {k[5:].sum()/n[5:].sum():.3f}; "
      f"input word ranked first without exclusion {top.sum()/n.sum():.3f}")
for nm, a_, i_ in zip(names, acc, inp):
    print(f"      {nm}: acc {a_:.3f}, input-first {i_:.3f}")
fig, ax = plt.subplots(figsize=(W, 4.9))
y = np.arange(len(names))[::-1].astype(float)
y[5:] -= 0.6                                  # gap between semantic and syntactic groups
ax.barh(y, acc, height=0.62, color=BLUE, label="정확도 (입력 세 단어를 뺐을 때)")
ax.scatter(inp, y, s=50, color=ORANGE, zorder=3, label="빼지 않으면 1등이 입력 단어인 비율")
for yy, a_ in zip(y, acc):
    ax.text(a_ + 0.012, yy, f"{a_*100:.0f}%", va="center", ha="left", fontsize=9, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(names)
ax.set_xlim(0, 1.0); ax.set_xticks(np.linspace(0, 1, 6)); ax.set_xticklabels([f"{t:.0%}" for t in np.linspace(0, 1, 6)])
ax.grid(False); ax.xaxis.grid(True)
ax.text(-0.5, y[0] + 0.75, "의미 범주", transform=ax.get_yaxis_transform(), fontsize=9.5, color=INK2, ha="left")
ax.text(-0.5, y[5] + 0.75, "문법 범주", transform=ax.get_yaxis_transform(), fontsize=9.5, color=INK2, ha="left")
h_, l_ = ax.get_legend_handles_labels()
top_legend(ax, ncol=2, handles=h_[::-1], labels=l_[::-1])
save(fig, os.path.join(OUT3, "analogy-categories.svg"))

# 3-4. GloVe weighting function
x = np.linspace(0, 200, 801)
f34 = np.where(x < 100, (x / 100) ** 0.75, 1.0)
f1 = np.where(x < 100, x / 100, 1.0)
fig, ax = plt.subplots(figsize=(W, 3.0))
ax.plot(x, f34, color=BLUE, lw=2, label=r"$\alpha = 3/4$ (논문 설정)")
ax.plot(x, f1, color=ORANGE, lw=2, ls=DASH, label=r"$\alpha = 1$")
ax.axvline(100, color=AXIS, lw=0.9)
ax.text(102, 0.06, r"$x_{\max} = 100$", color=INK2, fontsize=9.5)
ax.set_xlim(0, 200); ax.set_ylim(0, 1.1)
ax.set_xlabel(r"동시 출현 횟수 $X_{ij}$")
ytitle(ax, r"가중치 $f(X_{ij})$")
top_legend(ax, ncol=2)
for xx in (1, 10):
    print(f"[3-4] f({xx}) = {(xx/100)**0.75:.3f} (alpha 3/4) vs {xx/100:.3f} (alpha 1)")
save(fig, os.path.join(OUT3, "glove-weight.svg"))

# ---------------------------------------------------------------- NLP 4
# 4-1. fastText n-grams of "where"
word = "<where>"
grams = {n_: [word[i:i + n_] for i in range(len(word) - n_ + 1)] for n_ in range(3, 7)}
rows = [(f"{n_}-gram", [g for g in grams[n_] if g != word]) for n_ in range(3, 7)] + [("단어 자체", [word])]
print("[4-1]", {r[0]: r[1] for r in rows}, "total", sum(len(r[1]) for r in rows))
fig, ax = plt.subplots(figsize=(W, 2.9))
ax.set_xlim(0, 10); ax.set_ylim(-0.4, len(rows) - 0.2)
ax.axis("off")
for r, (lab, gs) in enumerate(rows):
    yy = len(rows) - 1 - r
    ax.text(0.0, yy, lab, va="center", ha="left", fontsize=10, color=INK2)
    for j, g in enumerate(gs):
        x0 = 1.55 + j * 1.28
        is_word = lab == "단어 자체"
        ax.add_patch(FancyBboxPatch((x0, yy - 0.3), 1.12, 0.6, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=SURF, ec=ORANGE if is_word else BLUE, lw=1.3))
        ax.text(x0 + 0.56, yy, g, ha="center", va="center", fontsize=10, color=INK, family="monospace")
ax.text(8.25, 2.0, r"15개 벡터를 더한 것이" "\n" r"where의 벡터 $\mathbf{v}_{\mathrm{where}}$",
        ha="left", va="center", fontsize=10, color=INK)
save(fig, os.path.join(OUT4, "fasttext-where.svg"))

# 4-2. composition on example C (made-up vectors)
tokens = ["I", "loved", "the", "movie", "!"]
X = np.array([[0.2, 0.1, 0.9], [1.8, 0.3, 0.1], [0.0, 0.1, 1.2], [0.3, 1.5, 0.2], [0.6, 0.0, 0.5]])
a_fix = np.array([0.05, 0.55, 0.05, 0.20, 0.15])
q = np.array([2.0, 0.5, -1.0])
r_att = X @ q
a_att = np.exp(r_att) / np.exp(r_att).sum()
res = {"합": X.sum(0), "평균": X.mean(0), "최댓값": X.max(0), "가중합\n(고정 가중치)": a_fix @ X, "attention": a_att @ X}
for kk, v in res.items():
    print("[4-2]", kk.replace("\n", " "), np.round(v, 3))
print("[4-2] max comes from", [tokens[i] for i in X.argmax(0)])
fig, ax = plt.subplots(figsize=(W, 3.3))
xs = np.arange(len(res)); wbar = 0.25
for d_, col in enumerate((BLUE, ORANGE, AQUA)):
    vals = np.array([v[d_] for v in res.values()])
    ax.bar(xs + (d_ - 1) * wbar, vals, width=wbar * 0.92, color=col, label=f"{d_ + 1}번째 차원")
    for xx, vv, nd in zip(xs, vals, (1, 2, 1, 3, 3)):   # decimals as written in the post
        ax.text(xx + (d_ - 1) * wbar, vv + 0.05, f"{vv:.{nd}f}", ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_xticks(xs); ax.set_xticklabels(list(res.keys()))
ax.set_ylim(0, 3.3)
ytitle(ax, r"문장 벡터 $\mathbf{d}$의 성분")
top_legend(ax, ncol=3)
save(fig, os.path.join(OUT4, "composition.svg"))

# 4-3. attention scores and weights
print("[4-3] r", np.round(r_att, 2), "exp", np.round(np.exp(r_att), 3), "sum", round(np.exp(r_att).sum(), 3),
      "a", np.round(a_att, 3))
fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"wspace": 0.25})
axes[0].bar(tokens, r_att, width=0.55, color=[ORANGE if v < 0 else BLUE for v in r_att])
axes[0].axhline(0, color=AXIS, lw=0.8)
for i, v in enumerate(r_att):
    axes[0].text(i, v + (0.12 if v >= 0 else -0.12), f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=9, color=INK2)
axes[0].set_ylim(-1.8, 4.4)
axes[0].set_title(r"점수 $r_i = \mathbf{v}^\top\mathbf{x}_i$")
axes[1].bar(tokens, a_att, width=0.55, color=BLUE)
for i, v in enumerate(a_att):
    axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK2)
axes[1].set_ylim(0, 1.0)
axes[1].set_title(r"가중치 $a_i = \mathrm{softmax}(\mathbf{r})_i$")
save(fig, os.path.join(OUT4, "attention-weights.svg"))
