# -*- coding: utf-8 -*-
"""畫 ingest_document() 那四行，輸出 day25_handler.png / day25_handler.svg。

為什麼值得一張圖：這四行用條列寫會變成流水帳，但它們的重點是**順序**——
三件事都必須發生在 return 之前，各有各的理由，而條列排不出「之前」這個概念。

所以把 return 那條線畫出來，四行擺在線的上面，
背景任務擺在線的下面：函式回來了，它才開始跑。

跟 day25_ingest 那張的關係：那張畫整趟旅程（七步），這張放大其中前三步
落在同一個函式裡的樣子。配色與框線沿用，讀者一看就知道是同一組圖。
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
MONO = FontProperties(family="DejaVu Sans Mono")

INK, MUTED = "#1c1917", "#57534e"
NEW_F, NEW_E = "#fff7ed", "#ea580c"
STORE_F, STORE_E = "#f0fdf4", "#16a34a"
EDGE_F, EDGE_E = "#fafaf9", "#a8a29e"
LINE = "#ea580c"

CODE, WHY, HEAD = 15, 13, 15.5


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8.2))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 124)
    ax.set_ylim(0, 80)
    ax.axis("off")

    L, W = 4, 78          # 程式碼那一欄
    RX = L + W + 4        # 右邊的註解欄

    ax.text(L, 76.8, "def ingest_document(req, background):", ha="left", va="center",
            fontproperties=MONO, fontsize=CODE + 0.5, color=MUTED)

    rows = [
        ("1", "settings.resolve_document(req.path)",
         "擋掉 ../ 這種路徑",
         "不合法就在這裡 400。放到背景才擋的話，\n呼叫端早就拿到 202，以為成功了"),
        ("2", "job = store.create(req.path, req.ocr)",
         "登記一個 job，狀態 queued",
         "要先有 job_id，第 4 步才有東西可以回；\n呼叫端也才查得到後來怎麼了"),
        ("3", "background.add_task(ingest.run, ...)",
         "把慢的事排進背景",
         "只是排隊，還沒開始跑"),
        ("4", "return IngestJob(...)   # 202",
         "回 202 ＋ job_id",
         "呼叫端到這裡就走了"),
    ]

    y = 64
    h = 10.5
    for n, code, title, why in rows:
        ax.add_patch(FancyBboxPatch(
            (L, y), W, h, boxstyle="round,pad=0,rounding_size=0.8",
            linewidth=1.8, edgecolor=NEW_E, facecolor=NEW_F, zorder=2))
        ax.text(L + 3.6, y + h / 2, n, ha="center", va="center",
                fontproperties=FB, fontsize=HEAD + 3, color=NEW_E, zorder=3)
        ax.text(L + 8, y + h * 0.68, code, ha="left", va="center",
                fontproperties=MONO, fontsize=CODE, color=INK, zorder=3)
        ax.text(L + 8, y + h * 0.26, title, ha="left", va="center",
                fontproperties=FB, fontsize=HEAD - 1.5, color=NEW_E, zorder=3)
        ax.text(RX, y + h / 2, why, ha="left", va="center",
                fontproperties=F, fontsize=WHY, color=MUTED, zorder=3)
        y -= h + 2.4

    # 第 2 步寫進 JobStore
    ax.annotate("", xy=(L + W, 52.7), xytext=(L + 26, 52.7),
                arrowprops=dict(arrowstyle="-", color=STORE_E, lw=0, alpha=0), zorder=1)

    # ------------------------------------------------ return 之後才開始那條線
    yline = y + 6.2
    ax.plot([2, 122], [yline, yline], linestyle="--", color=LINE, lw=2.2, zorder=1)
    ax.text(3, yline - 1.6, "函式到這裡就回去了。下面這件事，是回應送出去之後才開始的",
            ha="left", va="top", fontproperties=FB, fontsize=WHY + 1, color=LINE)

    ax.add_patch(FancyBboxPatch(
        (L, yline - 10.5), W, 5.6, boxstyle="round,pad=0,rounding_size=0.8",
        linewidth=1.8, edgecolor=EDGE_E, facecolor=EDGE_F, zorder=2))
    ax.text(L + W / 2, yline - 7.7, "ingest.run()：讀檔／OCR → 切塊 → 算向量 → 寫索引",
            ha="center", va="center", fontproperties=FB, fontsize=HEAD - 1, color=MUTED,
            zorder=3)
    ax.text(RX, yline - 7.7, "可能跑好幾分鐘", ha="left", va="center",
            fontproperties=F, fontsize=WHY, color=MUTED)

    fig.tight_layout(pad=0.5)
    for ext in ("png", "svg"):
        out = BASE_DIR / f"day25_handler.{ext}"
        fig.savefig(out, dpi=170 if ext == "png" else None,
                    facecolor="white", bbox_inches="tight")
        print("寫出", out.name)


if __name__ == "__main__":
    main()
