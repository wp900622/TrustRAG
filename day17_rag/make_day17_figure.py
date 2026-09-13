# -*- coding: utf-8 -*-
"""畫 Day 17 的主圖，輸出 day17_recall_gap.png / day17_recall_gap.svg。

這張圖只要講清楚一件事：**兩條線從 k=3 之後就分家了。**
聯集 recall 一路往上爬（62% → 88%），答對率沒有跟著爬，
到 k=15 反而掉到 3/8。Day 15 的結論「答對率天花板 = recall@k」
在多條文題上不成立。

輸入 token 用右軸的灰柱畫，讓「爬上去的代價」跟「爬上去沒用」
出現在同一張圖裡。

數字硬編自 experiment_result_day17.md（那份報表是實跑產物）；
重跑實驗後要回來更新這裡的常數。
字型用微軟正黑體（Windows 內建）；換平台改 FONT_PATH 即可。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

BASE_DIR = Path(__file__).parent
FONT_PATH = "C:/Windows/Fonts/msjh.ttc"
F = FontProperties(fname=FONT_PATH)
FB = FontProperties(fname=FONT_PATH, weight="bold")

INK = "#1c1917"
MUTED = "#57534e"
BAR = "#e7e5e4"
SINGLE = "#94a3b8"      # 單條 recall：舊尺，淡
JOINT = "#2563eb"       # 聯集 recall：今天的新尺
ACC = "#dc2626"         # 答對率：反轉的那條

# ---- 實跑數字（experiment_result_day17.md 第一張表）----
K = [1, 3, 5, 10, 15]
SINGLE_RECALL = [50, 81, 81, 88, 94]
JOINT_RECALL = [0, 62, 62, 75, 88]
ACCURACY = [12, 62, 50, 62, 38]        # 1/8, 5/8, 4/8, 5/8, 3/8
CORRECT = ["1/8", "5/8", "4/8", "5/8", "3/8"]
INPUT_TOKENS = [2239, 4863, 7579, 14334, 20791]


def main() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    x = list(range(len(K)))

    # 背景：輸入 token（右軸）——爬上去的代價
    ax2 = ax.twinx()
    ax2.bar(x, INPUT_TOKENS, width=0.62, color=BAR, zorder=1)
    ax2.set_ylim(0, max(INPUT_TOKENS) * 2.6)
    ax2.set_ylabel("輸入 tokens（灰柱）", fontproperties=F,
                   fontsize=11, color=MUTED)
    ax2.tick_params(axis="y", colors=MUTED, labelsize=9)
    for i, v in enumerate(INPUT_TOKENS):
        ax2.text(i, v + max(INPUT_TOKENS) * 0.03, f"{v:,}", ha="center",
                 va="bottom", fontproperties=F, fontsize=8.5, color=MUTED)

    # 三條線
    for series, color, label, style in (
            (SINGLE_RECALL, SINGLE, "單條 recall（舊尺）", "--"),
            (JOINT_RECALL, JOINT, "聯集 recall（新尺）", "-"),
            (ACCURACY, ACC, "答對率", "-")):
        ax.plot(x, series, style, color=color, linewidth=2.6 if style == "-" else 2.0,
                marker="o", markersize=7, zorder=4, label=label)

    # 標出反轉：k=3 與 k=15
    ax.annotate("", xy=(4, JOINT_RECALL[4]), xytext=(1, JOINT_RECALL[1]),
                arrowprops=dict(arrowstyle="->", color=JOINT, alpha=0.35,
                                linewidth=1.4, linestyle=":"), zorder=3)
    ax.text(2.5, 88, "聯集 recall ＋26 個百分點", ha="center", va="bottom",
            fontproperties=FB, fontsize=10.5, color=JOINT, zorder=5)
    ax.text(2.5, 33, "答對率 5/8 → 3/8", ha="center", va="top",
            fontproperties=FB, fontsize=10.5, color=ACC, zorder=5)

    for i, label in enumerate(CORRECT):
        ax.text(i, ACCURACY[i] - 5, label, ha="center", va="top",
                fontproperties=F, fontsize=9, color=ACC, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in K], fontproperties=F, fontsize=11.5)
    ax.set_ylim(-8, 108)
    ax.set_ylabel("百分比", fontproperties=F, fontsize=11, color=INK)
    ax.tick_params(axis="y", colors=INK, labelsize=9)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.grid(axis="y", color="#e7e5e4", linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#d6d3d1")

    ax.set_title("多條文綜合題：撈得更齊，答得更差（8 題，gpt-4o-mini，temperature=0）",
                 fontproperties=FB, fontsize=13.5, color=INK, pad=16)
    legend = ax.legend(prop=F, fontsize=10.5, loc="lower right",
                       frameon=False, ncol=1)
    for text in legend.get_texts():
        text.set_color(MUTED)

    fig.tight_layout()
    for ext in ("png", "svg"):
        path = BASE_DIR / f"day17_recall_gap.{ext}"
        fig.savefig(path, dpi=200, facecolor="white")
        print(f"已輸出 {path.name}")


if __name__ == "__main__":
    main()
