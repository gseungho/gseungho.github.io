# Shared figure style for blog posts. Every figure script starts with
#   from blogstyle import *
# which sets rcParams and exposes the palette, W, POSTS, ytitle(), save() and GIF_DPI.
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
POSTS = os.path.join(HERE, "..", "..", "public", "images", "posts")
for f in ("Pretendard-Regular.otf", "Pretendard-SemiBold.otf"):
    font_manager.fontManager.addfont(os.path.join(HERE, "fonts", f))

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e8e7e1", "#c3c2b7"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", MUTED

# Every figure is W inches wide. The post column is 848 px, so 1 pt shows as about 1.64 px and
# 10 pt text matches the 16 px body text in every figure. Vary only the height.
W = 7.2
# GIFs cannot be vector; W * GIF_DPI = 1440 px stays sharp on 1.5x-2x screens.
GIF_DPI = 200

plt.rcParams.update({
    "font.family": ["Pretendard", "Malgun Gothic"], "font.size": 10, "axes.unicode_minus": False,
    "mathtext.fontset": "cm",                       # math looks like the KaTeX in the body
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "axes.labelsize": 10, "axes.labelpad": 6,
    "axes.titlelocation": "left", "axes.titlesize": 10, "axes.titlecolor": INK2, "axes.titlepad": 6,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.7,
    "xtick.color": AXIS, "xtick.labelcolor": INK2, "xtick.labelsize": 9,
    "xtick.major.size": 3, "xtick.major.width": 0.8, "xtick.minor.size": 0,
    "ytick.labelcolor": INK2, "ytick.labelsize": 9, "ytick.major.size": 0, "ytick.minor.size": 0,
    "ytick.major.pad": 4,
    "lines.linewidth": 1.4, "lines.markersize": 5, "lines.solid_capstyle": "round",
    "lines.markeredgecolor": SURF, "lines.markeredgewidth": 1.0,
    "legend.frameon": False, "legend.fontsize": 9.5, "legend.handlelength": 1.6,
    "legend.columnspacing": 1.6,
    "svg.fonttype": "path", "svg.hashsalt": "gseungho-blog",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
})


def ytitle(ax, text):
    """y-axis name written horizontally above the axis instead of rotated beside it."""
    ax.set_ylabel("")
    ax.text(0, 1.03, text, transform=ax.transAxes, ha="left", va="bottom", color=INK2, fontsize=10)


def top_legend(ax, ncol, **kw):
    """legend in one row above the axis, clear of the ytitle()."""
    return ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.09), ncol=ncol, borderaxespad=0, **kw)


def save(fig, path):
    """SVG for still figures (sharp at any zoom); no date in the file so reruns diff cleanly."""
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
