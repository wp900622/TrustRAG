# -*- coding: utf-8 -*-
"""畫 Day 24 的系統架構圖，輸出 day24_architecture.png / day24_architecture.svg。

這張圖只要講清楚一件事：**今天寫的全在那條虛線上面，底下那一塊一個字都沒改。**

橘色是今天新寫的，灰藍色是被它包起來的 RAG 核心，綠色是唯一落地的狀態（SQLite）。

**框裡不要出現只有追這個系列的人才懂的東西。**「Day 14 那條管線」「rag_core」
對第一次點進來的人是空氣，所以一律換成它在做什麼：「固定流程」「切塊」「呼叫模型」。

版面規則：**框裡只放名字與關鍵字，句子留給文章。**
第一版每個框塞三行說明，縮到文章寬度之後整張圖沒人看得懂，
所以這一版把說明砍掉、字級放大，讓它縮小之後還讀得出來。

字型用微軟正黑體（Windows 內建）；換平台改 FONT_PATH 即可。
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

INK = "#1c1917"
MUTED = "#57534e"

NEW_F, NEW_E = "#fff7ed", "#ea580c"      # 今天寫的
FROZE_F, FROZE_E = "#f1f5f9", "#64748b"  # 前 23 天，一個字沒改
STORE_F, STORE_E = "#f0fdf4", "#16a34a"  # 落地的狀態
EDGE_F, EDGE_E = "#fafaf9", "#a8a29e"    # 外面的世界

TITLE = 19
BODY = 14.5
NOTE = 13


def box(ax, x, y, w, h, title, lines=(), fc=NEW_F, ec=NEW_E,
        title_size=TITLE, body_size=BODY, title_color=INK):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
        linewidth=1.8, edgecolor=ec, facecolor=fc, zorder=2))
    cx = x + w / 2
    if lines:
        top = y + h - 2.6
        ax.text(cx, top, title, ha="center", va="top",
                fontproperties=FB, fontsize=title_size, color=title_color, zorder=3)
        for i, line in enumerate(lines):
            ax.text(cx, top - 4.0 - i * 3.2, line, ha="center", va="top",
                    fontproperties=F, fontsize=body_size, color=MUTED, zorder=3)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontproperties=FB, fontsize=title_size, color=title_color, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=2.0, dashed=False, label=None,
          label_dx=0.0, label_dy=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle="--" if dashed else "-",
                                shrinkA=0, shrinkB=0, mutation_scale=18), zorder=1)
    if label:
        ax.text((x1 + x2) / 2 + label_dx, (y1 + y2) / 2 + label_dy, label,
                ha="center", va="center", fontproperties=F, fontsize=NOTE,
                color=color, zorder=4,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 11.0))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 122)
    ax.set_ylim(0, 92)
    ax.axis("off")

    # ---------------------------------------------------------- 外面的世界
    box(ax, 4, 85, 114, 6, "呼叫端：curl／前端／另一支服務",
        fc=EDGE_F, ec=EDGE_E, title_size=16)
    arrow(ax, 61, 85, 61, 81.5)

    # ------------------------------------------------------------ middleware
    box(ax, 4, 70.5, 114, 11, "middleware",
        ["RequestId：一個 id 貫穿 header、body 與 log　｜　"
         "Timing：四段耗時 → x-*-ms"])
    arrow(ax, 30, 70.5, 30, 67)
    arrow(ax, 90, 70.5, 90, 67)

    # ------------------------------------------------------------------ 端點
    box(ax, 4, 53, 54, 14, "POST /ask",
        ["AskRequest 擋格式", "回 answer ＋ citations ＋ timing"])
    box(ax, 64, 53, 54, 14, "POST /documents",
        ["路徑擋掉 ../", "立刻回 202 ＋ job_id"])

    # ------------------------------------------------------- /ask 的兩種答法
    arrow(ax, 17, 53, 17, 49.5)
    arrow(ax, 45, 53, 45, 49.5)
    box(ax, 4, 35.5, 26, 14, "mode=pipeline",
        ["固定流程", "查一次、答一次"])
    box(ax, 32, 35.5, 26, 14, "mode=agent",
        ["會用工具的迴圈", "自己決定查幾次"])

    # ------------------------------------------------------- Retriever 介面
    arrow(ax, 17, 35.5, 17, 32)
    arrow(ax, 45, 35.5, 45, 32, label="RetrieverCollection\n轉接頭", label_dx=16)
    box(ax, 4, 18, 54, 14, "Retriever 介面",
        ["search(vector, k) → list[Hit]", "ChromaRetriever　｜　NumpyRetriever"])

    # ------------------------------------------------------- /documents 的背景
    arrow(ax, 78, 53, 78, 49.5)
    box(ax, 64, 35.5, 26, 14, "store.JobStore",
        ["SQLite（WAL）", "多個 worker 共用"], fc=STORE_F, ec=STORE_E)
    box(ax, 92, 35.5, 26, 14, "BackgroundTasks",
        ["切塊、算向量", "沒文字層就 OCR"])
    arrow(ax, 90, 42.5, 92, 42.5)
    arrow(ax, 105, 35.5, 105, 32)
    box(ax, 64, 18, 54, 14, "upsert 進索引",
        ["chunk_id ＝ 檔名＋條號＋雜湊", "重複攝取是覆蓋，不是追加"])

    # 攝取完回頭重建 retriever
    arrow(ax, 64, 25, 58, 25, dashed=True, color=NEW_E,
          label="_refresh()", label_dy=2.8)

    # --------------------------------------------------------- 那條分界線
    ax.plot([2, 120], [15.0, 15.0], linestyle=(0, (6, 5)), color=NEW_E,
            linewidth=2.0, zorder=1)
    ax.text(61, 15.8, "以上是這支服務　｜　以下是它包起來的 RAG 核心", ha="center",
            va="bottom", fontproperties=FB, fontsize=NOTE + 1, color=NEW_E,
            zorder=4, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none"))

    # ------------------------------------------------------------- 凍結區
    arrow(ax, 31, 18, 31, 13.4)
    arrow(ax, 91, 18, 91, 13.4)
    box(ax, 4, 2.8, 114, 10.6, "RAG 核心（沿用前 23 天，今天沒動）",
        ["切塊　｜　算問題向量　｜　組 prompt　｜　呼叫模型　｜　"
         "向量庫存取　｜　agent 的工具迴圈"],
        fc=FROZE_F, ec=FROZE_E)

    # ------------------------------------------------------------------ 圖例
    legend = [("今天寫的", NEW_F, NEW_E),
              ("沿用，今天沒動", FROZE_F, FROZE_E),
              ("落地的狀態", STORE_F, STORE_E)]
    for i, (label, fc, ec) in enumerate(legend):
        x = 4 + i * 26
        ax.add_patch(FancyBboxPatch((x, 0.0), 3.0, 2.0,
                                    boxstyle="round,pad=0,rounding_size=0.4",
                                    linewidth=1.6, edgecolor=ec, facecolor=fc,
                                    zorder=3))
        ax.text(x + 4.2, 1.0, label, ha="left", va="center",
                fontproperties=F, fontsize=NOTE, color=MUTED, zorder=3)

    fig.tight_layout(rect=(0, 0, 1, 1))
    for ext in ("png", "svg"):
        path = BASE_DIR / f"day24_architecture.{ext}"
        fig.savefig(path, dpi=170 if ext == "png" else None,
                    facecolor="white", bbox_inches="tight")
        print("→", path.name)


if __name__ == "__main__":
    main()
