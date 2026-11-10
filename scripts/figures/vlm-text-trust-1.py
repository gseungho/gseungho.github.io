# Figures for the post vlm-text-trust-1 (public/images/posts/vlm-text-trust-1/).
# Counts are the confirmation run on 73 new fashion anchors (gseungho/vlm-same-item-test,
# results/confirm_fashion/*.jsonl): same-item pairs, answers out of 73 per model and condition.
import os

from blogstyle import *

OUT = os.path.join(POSTS, "vlm-text-trust-1")
os.makedirs(OUT, exist_ok=True)

N = 73
ROWS = ["이름 없음 (Q1)", "다른 이름 (Q1)", "다른 이름 (Q2)", "판매자 줄 (Q1)"]
# (same, different, cannot tell) for each row
COUNTS = {
    "Haiku": [(56, 1, 16), (3, 41, 29), (43, 6, 24), (28, 2, 43)],
    "Sonnet": [(72, 0, 1), (52, 7, 14), (72, 0, 1), (72, 1, 0)],
    "Opus": [(72, 1, 0), (31, 42, 0), (69, 4, 0), (68, 5, 0)],
}
PARTS = [("같다", BLUE, "white"), ("다르다", ORANGE, "white"), ("알 수 없다", GRAY, "white")]

fig, axes = plt.subplots(1, 3, figsize=(W, 2.9), sharey=True)
for ax, (model, rows) in zip(axes, COUNTS.items()):
    y = list(range(len(ROWS)))[::-1]
    left = [0.0] * len(ROWS)
    for k, (label, color, ink) in enumerate(PARTS):
        share = [100 * r[k] / N for r in rows]
        ax.barh(y, share, left=left, color=color, height=0.62, label=label if model == "Haiku" else None)
        for yy, l, s in zip(y, left, share):
            if s >= 14:
                ax.text(l + s / 2, yy, f"{s:.0f}", ha="center", va="center", color=ink, fontsize=8.5)
        left = [l + s for l, s in zip(left, share)]
    ax.set_title(model)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(list(range(len(ROWS)))[::-1])
axes[0].set_yticklabels(ROWS)
fig.supxlabel("앵커 73개 가운데 비율(%)", fontsize=9.5, color=INK2, y=-0.06)
top_legend(axes[0], 3)
fig.subplots_adjust(wspace=0.16)
save(fig, os.path.join(OUT, "answers.svg"))
