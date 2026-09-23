# -*- coding: utf-8 -*-
"""畫串流與非串流的時間軸對照，輸出 day26_timeline.png / .svg。

這張圖就是整篇的論點：兩條一樣長的時間軸，總時間幾乎相同，
但使用者盯著空白畫面的那一段差了 6.5 倍。

數字全部讀自 runs_day26.json，不是手打的——
改天重跑，圖會跟著新的數字走。
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from figstyle import (F, FB, INK, MUTED, NEW_E, STORE_E, EDGE_E, save)

BASE_DIR = Path(__file__).parent
BLANK = "#e7e5e4"        # 盯著空白的那一段
WAIT = "#fed7aa"         # 出處在了，還在等答案
GEN = "#fff7ed"          # 答案一個字一個字長出來


def med(rows, key):
    return st.median([r[key] for r in rows if r.get(key) is not None])


def main() -> None:
    rows = json.loads((BASE_DIR / "runs_day26.json").read_text(encoding="utf-8"))
    n = len(rows)
    cit = med(rows, "citations_ms")
    first = med(rows, "first_token_ms")
    done = med(rows, "done_ms")
    plain = med(rows, "plain_wall_ms")

    span = max(done, plain) * 1.02
    fig, ax = plt.subplots(figsize=(13.4, 6.4))
    fig.patch.set_facecolor("white")
    ax.set_xlim(-16, 118)
    ax.set_ylim(0, 58)
    ax.axis("off")

    X0, XW = 18, 92                      # 時間軸在畫布上的起點與寬度
    def px(ms):                          # 毫秒 → 畫布座標
        return X0 + ms / span * XW

    def bar(y, x1, x2, fc, ec):
        ax.add_patch(FancyBboxPatch(
            (px(x1), y), px(x2) - px(x1), 7,
            boxstyle="round,pad=0,rounding_size=0.4",
            linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2))

    def tick(x, y, label, sub, color):
        ax.plot([px(x), px(x)], [y - 1.6, y + 8.6], color=color, lw=1.8, zorder=3)
        ax.text(px(x), y + 10.6, label, ha="center", va="center",
                fontproperties=FB, fontsize=13, color=color, zorder=4)
        ax.text(px(x), y + 7.4 - 10.6, sub, ha="center", va="center",
                fontproperties=F, fontsize=12, color=MUTED, zorder=4)

    # ------------------------------------------------------------ 非串流
    ax.text(X0 - 3, 41.5, "現在（非串流）", ha="right", va="center",
            fontproperties=FB, fontsize=14.5, color=INK)
    bar(38, 0, plain, BLANK, EDGE_E)
    ax.text(px(plain / 2), 41.5, "使用者看著一片空白", ha="center", va="center",
            fontproperties=FB, fontsize=13.5, color=MUTED, zorder=4)
    tick(plain, 38, f"{plain:.0f} ms", "整包 JSON 一次回來", EDGE_E)

    # -------------------------------------------------------------- 串流
    ax.text(X0 - 3, 17.5, "改成串流", ha="right", va="center",
            fontproperties=FB, fontsize=14.5, color=INK)
    bar(14, 0, cit, BLANK, EDGE_E)
    bar(14, cit, first, WAIT, NEW_E)
    bar(14, first, done, GEN, NEW_E)

    ax.text(px(cit / 2), 17.5, "空白", ha="center", va="center",
            fontproperties=FB, fontsize=12, color=MUTED, zorder=4)
    ax.text(px((cit + first) / 2), 17.5, "出處已經在畫面上，還在等答案",
            ha="center", va="center", fontproperties=FB, fontsize=12.5,
            color="#9a3412", zorder=4)
    ax.text(px((first + done) / 2), 17.5, "答案一個字一個字長出來",
            ha="center", va="center", fontproperties=FB, fontsize=12.5,
            color="#9a3412", zorder=4)

    tick(cit, 14, f"{cit:.0f} ms", "citations 事件", NEW_E)
    tick(first, 14, f"{first:.0f} ms", "第一個 token", NEW_E)
    tick(done, 14, f"{done:.0f} ms", "done 事件", NEW_E)

    # --------------------------------------------- 兩條軸之間那個比較箭頭
    ax.annotate("", xy=(px(cit), 32), xytext=(px(plain), 32),
                arrowprops=dict(arrowstyle="<|-|>", color=STORE_E, lw=2.0,
                                shrinkA=0, shrinkB=0, mutation_scale=15), zorder=4)
    ax.text(px((cit + plain) / 2), 34.4,
            f"使用者看到第一個東西的時間：{plain:.0f} → {cit:.0f} ms"
            f"（{plain / cit:.1f} 倍）",
            ha="center", va="center", fontproperties=FB, fontsize=13.5,
            color=STORE_E, zorder=5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none"))

    # ------------------------------------------------------------- 底註
    ax.text(-16, 5.6,
            f"同一組 28 題，串流與非串流交錯跑，取中位數。總時間幾乎沒變"
            f"（{plain:.0f} → {done:.0f} ms），變的是前面那一段。",
            ha="left", va="center", fontproperties=F, fontsize=12.5, color=MUTED)
    ax.text(-16, 2.2,
            "出處能先送，是因為檢索完就定案了，不必等模型開口——"
            "這是串流真正買到的東西，不只是字會跳出來。",
            ha="left", va="center", fontproperties=F, fontsize=12.5, color=MUTED)

    fig.tight_layout(pad=0.5)
    save(fig, "day26_timeline", BASE_DIR)
    print(f"（讀自 runs_day26.json，n={n}）")


if __name__ == "__main__":
    main()
