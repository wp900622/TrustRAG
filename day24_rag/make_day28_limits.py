# -*- coding: utf-8 -*-
"""畫 Day 28 的架構圖，輸出 day28_limits.png / .svg。

講的是一件事：**agent 這條路停下來的方式有兩種，而呼叫端拿到的東西不一樣。**

形狀是一個迴圈加兩個出口。迴圈中間那一格是今天新加的檢查點，
左邊的出口走凍結 agent 自己那條強制作答的路（還有答案），
右邊的出口是直接打斷（沒有答案，但已經查到的條文照樣回去）。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figstyle import (F, FB, MONO, INK, MUTED, NEW_F, NEW_E, OUT_F, OUT_E,
                      STORE_F, STORE_E, EDGE_F, EDGE_E, BODY, NOTE,
                      card, arrow, save)

BASE_DIR = Path(__file__).parent
TITLE = 14.5


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.8, 9.8))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 96)
    ax.axis("off")

    def box(x, y, w, h, title, note=None, fc=NEW_F, ec=NEW_E, mono=False,
            tsize=TITLE):
        card(ax, x, y, w, h, fc=fc, ec=ec)
        cx = x + w / 2
        if note:
            ax.text(cx, y + h * 0.64, title, ha="center", va="center",
                    fontproperties=MONO if mono else FB, fontsize=tsize,
                    color=INK, zorder=3)
            ax.text(cx, y + h * 0.26, note, ha="center", va="center",
                    fontproperties=F, fontsize=NOTE, color=MUTED, zorder=3)
        else:
            ax.text(cx, y + h / 2, title, ha="center", va="center",
                    fontproperties=MONO if mono else FB, fontsize=tsize,
                    color=INK, zorder=3)

    # ------------------------------------------------------------ 進來
    box(26, 88, 76, 6.5, 'POST /ask   {"mode": "agent", "max_steps": 3}',
        fc=EDGE_F, ec=EDGE_E, mono=True, tsize=13.5)
    arrow(ax, 64, 88, 64, 85.5)

    box(26, 76, 76, 8.5, "上限：呼叫端要的，跟服務允許的，取緊的那個",
        "三個都沒設就是不設限，行為跟 Day 24 完全一樣")
    arrow(ax, 64, 76, 64, 73.5)

    # ------------------------------------------------------------ 迴圈
    box(20, 40, 88, 33, "", fc=OUT_F, ec=OUT_E)
    ax.text(64, 70.6, "那支 agent：它自己決定要查幾次、什麼時候交卷",
            ha="center", va="center", fontproperties=FB, fontsize=TITLE,
            color=INK, zorder=3)
    ax.text(64, 67.6, "Day 20／21 寫的，今天一行都沒改",
            ha="center", va="center", fontproperties=F, fontsize=NOTE,
            color=MUTED, zorder=3)

    box(28, 57, 72, 8, "每一次模型呼叫之前，先問一句",
        "時間到了嗎？錢用完了嗎？（今天加的就是這一格）",
        fc=NEW_F, ec=NEW_E)
    arrow(ax, 64, 57, 64, 54.5)

    box(28, 46, 72, 8, "問模型「下一步做什麼」",
        "它會挑一個：再查一次規章／驗一下草稿／交卷", fc=EDGE_F, ec=EDGE_E)

    # 迴圈回去的線
    ax.plot([100, 106, 106, 100], [50, 50, 61, 61], color=MUTED, lw=1.6,
            zorder=1)
    arrow(ax, 106, 61, 100.5, 61, color=MUTED)
    ax.text(107.5, 55.5, "還沒交卷\n就再一輪", ha="left", va="center",
            fontproperties=F, fontsize=NOTE, color=MUTED)

    # ------------------------------------------------------------ 兩個出口
    ax.plot([34, 94], [37.5, 37.5], color=MUTED, lw=1.6, zorder=1)
    ax.plot([64, 64], [40, 37.5], color=MUTED, lw=1.6, zorder=1)
    arrow(ax, 34, 37.5, 34, 34.5)
    arrow(ax, 94, 37.5, 94, 34.5)
    ax.text(31, 35.8, "走到第 N 步", ha="right", va="center",
            fontproperties=F, fontsize=NOTE, color=MUTED)
    ax.text(97, 35.8, "時間或金額先用完", ha="left", va="center",
            fontproperties=F, fontsize=NOTE, color=NEW_E)

    # 左：步數上限
    box(4, 24, 60, 9.5, "跟它說「請你現在交卷」",
        "這條路那支 agent 本來就有", fc=STORE_F, ec=STORE_E)
    arrow(ax, 34, 24, 34, 21.5, color=STORE_E)
    box(4, 11, 60, 9.5, "有答案，有出處",
        'stopped_reason = "steps"', fc=STORE_F, ec=STORE_E)

    # 右：金額／時間上限
    box(66, 24, 58, 9.5, "當場打斷，不再問模型",
        "它寫到一半的草稿拿不到")
    arrow(ax, 94, 24, 94, 21.5)
    box(66, 11, 58, 9.5, "沒有答案，但出處還在",
        'answer = null　／　stopped_reason = "time"')

    # ------------------------------------------------------------ 底
    ax.plot([34, 94], [9, 9], color=MUTED, lw=1.6, zorder=1)
    ax.plot([34, 34], [11, 9], color=MUTED, lw=1.6, zorder=1)
    ax.plot([94, 94], [11, 9], color=MUTED, lw=1.6, zorder=1)
    arrow(ax, 64, 9, 64, 6.5, color=MUTED)
    box(30, 0.5, 68, 6, "兩種都是 HTTP 200",
        "被上限攔下來是預期內的事，不是錯誤", fc=EDGE_F, ec=EDGE_E)

    save(fig, "day28_limits", BASE_DIR)


if __name__ == "__main__":
    main()
