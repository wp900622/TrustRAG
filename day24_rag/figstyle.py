# -*- coding: utf-8 -*-
"""Day 25 起所有圖共用的配色與畫框工具。

抽出來的理由很實際：五個腳本各自複製一份 NEW_F／NEW_E，
之後要調色就得改五個地方，一定會有一張忘了改、看起來不像同一組。

配色的意思固定不變：
    橘  這支服務今天寫的
    綠  行程外面、多個 worker 共用的東西
    灰  從外面注入或不歸這個檔案管的
"""
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch

FONT_PATH = "C:/Windows/Fonts/msjh.ttc"
F = FontProperties(fname=FONT_PATH)
FB = FontProperties(fname=FONT_PATH, weight="bold")
MONO = FontProperties(family="DejaVu Sans Mono")

INK, MUTED = "#1c1917", "#57534e"
NEW_F, NEW_E = "#fff7ed", "#ea580c"
OUT_F, OUT_E = "#fffbf5", "#fdba74"
STORE_F, STORE_E = "#f0fdf4", "#16a34a"
EDGE_F, EDGE_E = "#fafaf9", "#a8a29e"
LINE = "#ea580c"

SIG, BODY, NOTE = 14.5, 13.5, 12.5


def panel(ax, x, y, w, h, fc=OUT_F, ec=OUT_E, lw=2.2, dashed=False, z=1):
    """一個大的容器框。"""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z,
        linestyle="--" if dashed else "-"))


def card(ax, x, y, w, h, fc=NEW_F, ec=NEW_E, dashed=False, z=2):
    """一個小方塊。"""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
        linewidth=1.8, edgecolor=ec, facecolor=fc, zorder=z,
        linestyle="--" if dashed else "-"))


def sig_row(ax, x, y, sig, why, sig_w=30, color=INK):
    """一行「簽名 → 這個方法在做什麼」。簽名等寬，說明用中文字型。"""
    ax.text(x, y, sig, ha="left", va="center",
            fontproperties=MONO, fontsize=SIG, color=color, zorder=3)
    ax.text(x + sig_w, y, why, ha="left", va="center",
            fontproperties=F, fontsize=BODY, color=MUTED, zorder=3)


def group(ax, x, y, title, color=NEW_E):
    ax.text(x, y, title, ha="left", va="center",
            fontproperties=FB, fontsize=BODY + 1, color=color, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=2.0, dashed=False,
          label=None, label_dy=0.0, boxed=True):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle="--" if dashed else "-",
                                shrinkA=0, shrinkB=0, mutation_scale=16), zorder=2)
    if label:
        kw = dict(bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="none")) \
            if boxed else {}
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + label_dy, label,
                ha="center", va="center", fontproperties=F, fontsize=NOTE,
                color=color, zorder=4, **kw)


def save(fig, stem, base_dir):
    for ext in ("png", "svg"):
        out = base_dir / f"{stem}.{ext}"
        # 先砍再寫。Windows 上如果有預覽視窗開著那個檔，
        # 直接覆寫會拿到 OSError 22，砍掉重建就沒事
        out.unlink(missing_ok=True)
        fig.savefig(out, dpi=170 if ext == "png" else None,
                    facecolor="white", bbox_inches="tight")
        print("寫出", out.name)
