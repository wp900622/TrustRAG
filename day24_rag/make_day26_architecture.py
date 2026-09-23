# -*- coding: utf-8 -*-
"""畫 Day 26 的架構圖，輸出 day26_architecture.png / .svg。

跟 day26_timeline 那張分工：那張講「使用者什麼時候看到東西」，
這張講「程式碼長什麼樣、哪裡分岔、分岔之後多了哪些麻煩」。

構圖的重點是串流那一側的橫線：第一個 byte 一送出去，
狀態碼就定死、header 就改不了、還得自己偵測客戶端走掉。
所以形狀是共用的頭、兩條岔路，串流那條上面橫一刀，
三個後果收在最底下獨立一排——放在岔路旁邊會跟左欄撞在一起。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figstyle import (F, FB, MONO, INK, MUTED, NEW_F, NEW_E, OUT_F, OUT_E,
                      STORE_F, STORE_E, EDGE_F, EDGE_E, LINE, BODY, NOTE,
                      card, arrow, save)

BASE_DIR = Path(__file__).parent
TITLE = 14.5


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.8, 10.4))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 98)
    ax.axis("off")

    def box(x, y, w, h, title, note=None, fc=NEW_F, ec=NEW_E, mono=False,
            tsize=TITLE):
        """有副標的框，標題放上三分之二、副標放下三分之一，都留在框裡。"""
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

    # ================================================ 共用的頭：兩條路一樣
    box(30, 89, 68, 6, "POST /ask", fc=EDGE_F, ec=EDGE_E)
    arrow(ax, 64, 89, 64, 86.5)

    box(30, 77, 68, 9, "契約檢查", "問題長度、k 的範圍。這裡不合格還回得了 422")
    arrow(ax, 64, 77, 64, 74.5)

    box(30, 65, 68, 9, "算問題向量　→　檢索", "走 Redis 快取，命中就不打 API")
    arrow(ax, 64, 65, 64, 62.5)

    box(24, 53, 80, 9, "這一刻，出處已經定案了", "模型還沒有被呼叫",
        fc=STORE_F, ec=STORE_E)

    # 分岔
    ax.plot([32, 96], [50, 50], color=MUTED, lw=1.6, zorder=1)
    ax.plot([64, 64], [53, 50], color=MUTED, lw=1.6, zorder=1)
    arrow(ax, 32, 50, 32, 46.5)
    arrow(ax, 96, 50, 96, 46.5)
    ax.text(28, 48.2, "stream=false", ha="right", va="center",
            fontproperties=MONO, fontsize=NOTE, color=MUTED)
    ax.text(100, 48.2, "stream=true", ha="left", va="center",
            fontproperties=MONO, fontsize=NOTE, color=NEW_E)

    # ====================================================== 左：非串流（舊）
    LX, LW = 4, 56
    box(LX, 37.5, LW, 9, "呼叫模型，等整段講完", "這一秒多，連線開著但什麼都沒傳",
        fc=EDGE_F, ec=EDGE_E)
    arrow(ax, LX + LW / 2, 37.5, LX + LW / 2, 35, color=EDGE_E)
    box(LX, 26, LW, 9, "組一包 JSON，一次送出", "帶 x-total-ms 等計時 header",
        fc=EDGE_F, ec=EDGE_E)
    ax.text(LX + LW / 2, 22.6, "使用者第一次看到東西：1,423 ms",
            ha="center", va="center", fontproperties=FB, fontsize=BODY,
            color=MUTED)

    # ======================================================== 右：串流（新）
    RX, RW = 68, 56
    box(RX, 37.5, RW, 9, "① event: citations", "出處先送。模型還沒被呼叫")

    # 第一個 byte 出去之後，回不去的那條線
    ax.plot([64, 127], [34.6, 34.6], linestyle="--", color=LINE, lw=2.2, zorder=1)
    ax.text(65, 32.9, "第一個 byte 送出去了，以下三件事回不去",
            ha="left", va="top", fontproperties=FB, fontsize=BODY, color=LINE)

    box(RX, 20, RW, 9, "② event: token ×39", "一段一段轉送。每段之間都是一個取消點")
    arrow(ax, RX + RW / 2, 20, RX + RW / 2, 17.5)
    box(RX, 8.5, RW, 9, "③ event: done", "答案、usage，還有真正的 timing")

    # =========================================== 最底下：三件回不去的事
    ax.plot([2, 126], [5.8, 5.8], color=EDGE_E, lw=1.2, zorder=1)
    for i, (head, body) in enumerate([
        # 順序跟文章的小節一致：最貴的那件排第一
        ("客戶端走掉沒人告訴你",
         "模型照樣生完、錢照付。" + chr(10) +
         "產生器要寫成 async 才收得到 aclose"),
        ("狀態碼定死在 200",
         "後面炸掉改不了，" + chr(10) + "只能送 event: error"),
        ("計時 header 會騙人",
         "那時還沒開始做。實測 2.88 ms，" + chr(10) +
         "真正 50.45 ms → 串流不送"),
    ]):
        x = 3 + i * 41.5
        card(ax, x, -8.5, 39, 12.5, fc=OUT_F, ec=OUT_E)
        ax.text(x + 2.2, 1.3, f"{i + 1}. {head}", ha="left", va="center",
                fontproperties=FB, fontsize=BODY, color="#9a3412", zorder=3)
        ax.text(x + 2.2, -3.4, body, ha="left", va="center",
                fontproperties=F, fontsize=NOTE - 0.5, color=MUTED, zorder=3)

    ax.text(3, -11.6,
            "左邊那條是 Day 24 就有的，今天一行沒動。新寫的全在右邊，"
            "以及底下那三格必須跟著補的東西。",
            ha="left", va="center", fontproperties=F, fontsize=NOTE, color=MUTED)
    ax.set_ylim(-14, 98)

    fig.tight_layout(pad=0.5)
    save(fig, "day26_architecture", BASE_DIR)


if __name__ == "__main__":
    main()
