# -*- coding: utf-8 -*-
"""畫 Day 25 的攝取流程圖，輸出 day25_ingest.png / day25_ingest.svg。

為什麼不沿用 Day 24 那張：那張畫的是兩條路（/ask 與 /documents），
這篇只走 /documents 這一條。讀者被迫先解析 mode=pipeline／mode=agent
那些本篇根本不談的東西，還要追一條從右邊橫跨回左邊的 _refresh() 虛線。

這張只講一件事：**橫線以上是回應之前，橫線以下是回應送出去之後還在做的事。**
那條線就是 202 的意義，也是這篇所有麻煩的來源——
回應走了，工作還在跑，所以狀態得放在別人也看得到的地方。

版面規則沿用 Day 24：框裡只放一行，句子留給文章；
不寫「Day 22 那支 OCR」這種只有追連載的人才懂的字眼。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch

BASE_DIR = Path(__file__).parent
FONT_PATH = "C:/Windows/Fonts/msjh.ttc"
F = FontProperties(fname=FONT_PATH)
FB = FontProperties(fname=FONT_PATH, weight="bold")

INK, MUTED = "#1c1917", "#57534e"
NEW_F, NEW_E = "#fff7ed", "#ea580c"      # 這支服務
STORE_F, STORE_E = "#f0fdf4", "#16a34a"  # 行程外面的狀態
EDGE_F, EDGE_E = "#fafaf9", "#a8a29e"    # 外面的世界
LINE = "#ea580c"

STEP, BODY, NOTE = 17, 13.5, 12.5


def step(ax, x, y, w, h, n, title, note=None, fc=NEW_F, ec=NEW_E):
    """一個步驟框。編號放左邊，讓讀者知道先後。"""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
        linewidth=1.8, edgecolor=ec, facecolor=fc, zorder=2))
    if n:
        ax.text(x + 3.4, y + h / 2, n, ha="center", va="center",
                fontproperties=FB, fontsize=STEP + 2, color=ec, zorder=3)
    tx = x + (7.5 if n else w / 2)
    ha = "left" if n else "center"
    if note:
        ax.text(tx, y + h * 0.63, title, ha=ha, va="center",
                fontproperties=FB, fontsize=STEP, color=INK, zorder=3)
        ax.text(tx, y + h * 0.27, note, ha=ha, va="center",
                fontproperties=F, fontsize=BODY, color=MUTED, zorder=3)
    else:
        ax.text(tx, y + h / 2, title, ha=ha, va="center",
                fontproperties=FB, fontsize=STEP, color=INK, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=MUTED, dashed=False, label=None, lw=2.0,
          label_dy=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle="--" if dashed else "-",
                                shrinkA=0, shrinkB=0, mutation_scale=17), zorder=1)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + label_dy, label,
                ha="center", va="center",
                fontproperties=F, fontsize=NOTE, color=color, zorder=4,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 10.5))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 124)
    ax.set_ylim(0, 100)
    ax.axis("off")

    L, W = 6, 68          # 主軸的左緣與寬度
    cx = L + W / 2
    R, RW = 84, 34        # 右側「行程外面」那一欄

    # ------------------------------------------------------------ 呼叫端
    step(ax, L, 92, W, 6.5, None, "呼叫端：curl／前端／另一支服務",
         fc=EDGE_F, ec=EDGE_E)

    # -------------------------------------------- 回應之前（快，毫秒等級）
    arrow(ax, cx, 92, cx, 87.5, label="POST /documents")
    step(ax, L, 80, W, 7.5, "①", "擋掉 ../ 這種路徑",
         "不合法就直接 400，不要等到背景才失敗")

    arrow(ax, cx, 80, cx, 75.5)
    step(ax, L, 68, W, 7.5, "②", "登記一個 job，狀態 queued",
         "拿到一個 job_id")

    arrow(ax, cx, 68, cx, 63.5)
    step(ax, L, 56, W, 7.5, "③", "立刻回 202 ＋ job_id",
         "呼叫端不必等下面那些事做完")

    # ------------------------------------------------- 分界：回應已經送出去
    ax.plot([2, 122], [50, 50], linestyle="--", color=LINE, lw=2.2, zorder=1)
    ax.text(3, 51.4, "以上：呼叫端還在等，只花幾毫秒", ha="left", va="bottom",
            fontproperties=FB, fontsize=BODY + 0.5, color=LINE)
    ax.text(3, 48.6, "以下：回應已經送出去了，這些事還在背景跑",
            ha="left", va="top", fontproperties=FB, fontsize=BODY + 0.5, color=LINE)

    # 202 回到呼叫端：沿右邊繞上去，不穿過中間
    ax.annotate("", xy=(78, 93.5), xytext=(78, 59.75),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6,
                                connectionstyle="arc3,rad=0", mutation_scale=15),
                zorder=1)
    arrow(ax, L + W, 59.75, 78, 59.75, lw=1.6)
    ax.text(79.5, 76, "202", ha="left", va="center", rotation=90,
            fontproperties=F, fontsize=NOTE, color=MUTED)

    # ------------------------------------------- 回應之後（慢，可能好幾分鐘）
    # 這條箭頭刻意穿過分隔線：同一條流程繼續往下走，只是回應已經先走了
    arrow(ax, cx, 56, cx, 44)
    step(ax, L, 36, W, 7.5, "④", "讀檔，沒有文字層就做 OCR",
         "一份 20 頁的掃描 PDF 要 277 秒")

    arrow(ax, cx, 36, cx, 31.5)
    step(ax, L, 24, W, 7.5, "⑤", "切塊，算每一塊的 id",
         "id ＝ 檔名＋條號＋內容雜湊，所以重覆攝取是覆蓋")

    arrow(ax, cx, 24, cx, 19.5)
    step(ax, L, 12, W, 7.5, "⑥", "算向量，寫進索引",
         "寫的時候上兩層鎖：擋自己的執行緒，也擋別的 worker")

    arrow(ax, cx, 12, cx, 7.5)
    step(ax, L, 1.5, W, 6, "⑦", "重建 retriever，把 job 標成 done")

    # --------------------------------------------- 右欄：行程外面的兩個東西
    ax.text(R + RW / 2, 91, "行程外面", ha="center", va="center",
            fontproperties=FB, fontsize=BODY + 1, color=STORE_E)
    ax.text(R + RW / 2, 87.8, "四個 worker 都看得到同一份",
            ha="center", va="center", fontproperties=F, fontsize=NOTE, color=MUTED)

    step(ax, R, 64, RW, 11, None, "JobStore", fc=STORE_F, ec=STORE_E)
    ax.text(R + RW / 2, 67.2, "SQLite（WAL）", ha="center", va="center",
            fontproperties=F, fontsize=BODY, color=MUTED, zorder=3)
    arrow(ax, L + W, 71.75, R, 71.75, color=STORE_E, lw=1.6, label="寫")

    step(ax, R, 14, RW, 11, None, "Redis 快取", fc=STORE_F, ec=STORE_E)
    ax.text(R + RW / 2, 17.2, "算過的向量與答案", ha="center", va="center",
            fontproperties=F, fontsize=BODY, color=MUTED, zorder=3)
    arrow(ax, L + W, 19.5, R, 19.5, color=STORE_E, lw=1.6, label="查／存")
    ax.text(R + RW / 2, 12.5, "算過的向量不必再跟 OpenAI 買一次",
            ha="center", va="top", fontproperties=F, fontsize=NOTE, color=MUTED)

    for i, t in enumerate(["② 登記、⑦ 標成 done 都寫它",
                           "GET /documents/{id} 查進度也是問它",
                           "換一個 worker 來查，一樣查得到"]):
        ax.text(R + RW / 2, 61.5 - i * 3.2, t, ha="center", va="top",
                fontproperties=F, fontsize=NOTE, color=MUTED)

    fig.tight_layout(pad=0.6)
    for ext in ("png", "svg"):
        out = BASE_DIR / f"day25_ingest.{ext}"
        fig.savefig(out, dpi=170 if ext == "png" else None,
                    facecolor="white", bbox_inches="tight")
        print("寫出", out.name)


if __name__ == "__main__":
    main()
