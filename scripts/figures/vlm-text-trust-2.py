# Figures for the post vlm-text-trust-2 (public/images/posts/vlm-text-trust-2/).
# Counts come from gseungho/vlm-caption-test: results/caption_confirm/*.jsonl (63 new Commons items)
# and results/caption_wording/*.jsonl (128 items asked QM again and QC). Each tuple is
# (accepted, rejected, cannot tell). Under QM accepting is "Yes"; under QC
# ("Does anything in the photo contradict the caption?") accepting is "No".
import os

from blogstyle import *

OUT = os.path.join(POSTS, "vlm-text-trust-2")
os.makedirs(OUT, exist_ok=True)
PARTS = [("받아들임", BLUE), ("거름", ORANGE), ("알 수 없다", GRAY)]


def stacked(rows, counts, height, path, legend_labels=None):
    fig, axes = plt.subplots(1, 3, figsize=(W, height), sharey=True)
    for ax, (model, data) in zip(axes, counts.items()):
        y = list(range(len(rows)))[::-1]
        left = [0.0] * len(rows)
        for k, (label, color) in enumerate(PARTS):
            share = [100 * r[k] / sum(r) for r in data]
            ax.barh(y, share, left=left, color=color, height=0.62,
                    label=(legend_labels or [p[0] for p in PARTS])[k] if model == "Haiku" else None)
            for yy, l, s in zip(y, left, share):
                if s >= 14:
                    ax.text(l + s / 2, yy, f"{s:.0f}", ha="center", va="center", color="white", fontsize=8.5)
            left = [l + s for l, s in zip(left, share)]
        ax.set_title(model)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.grid(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(list(range(len(rows)))[::-1])
    axes[0].set_yticklabels(rows)
    fig.supxlabel("문항 가운데 비율(%)", fontsize=9.5, color=INK2, y=-0.06)
    top_legend(axes[0], 3)
    fig.subplots_adjust(wspace=0.16)
    save(fig, path)


# Figure 2: the confirmation run, QM on the true caption, the false caption and the other photo (63 items).
stacked(
    ["맞는 캡션", "틀린 캡션", "다른 사건의 사진"],
    {"Haiku": [(3, 0, 60), (0, 34, 29), (0, 44, 19)],
     "Sonnet": [(14, 9, 40), (3, 47, 13), (0, 56, 7)],
     "Opus": [(20, 12, 31), (0, 48, 15), (0, 56, 7)]},
    2.5, os.path.join(OUT, "confirm_qm.svg"), legend_labels=["그렇다", "아니다", "알 수 없다"])

# Figure 3: the wording study. True captions on all 128 items; false captions on the 57 items whose
# changed fact the photo cannot check.
stacked(
    ["맞는 캡션, QM", "맞는 캡션, QC", "확인할 수 없는 틀린 캡션, QM", "확인할 수 없는 틀린 캡션, QC"],
    {"Haiku": [(7, 0, 121), (73, 1, 54), (2, 0, 55), (19, 3, 35)],
     "Sonnet": [(27, 20, 81), (84, 8, 36), (7, 26, 24), (20, 20, 17)],
     "Opus": [(46, 24, 58), (118, 4, 6), (6, 28, 23), (30, 19, 8)]},
    2.9, os.path.join(OUT, "wording.svg"))
