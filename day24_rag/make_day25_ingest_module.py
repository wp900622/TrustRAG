# -*- coding: utf-8 -*-
"""畫 ingest.py 的三個函式，輸出 day25_ingest_module.png / .svg。

為什麼不留表格：ithelp 的表格寬度會把 run(store, job_id, upsert, on_done)
折成兩行，讀起來很糟。但「表格的截圖」也沒有多出任何資訊，
所以改畫它們真正的關係——這三個不是並列的，`run()` 是外框，
另外兩個是它裡面呼叫的東西。表格排不出包含關係，圖可以。

底下那條狀態帶是這張圖的第二個重點：`run()` 從頭到尾都在更新 job 狀態，
所以呼叫端隨時查得到現在跑到哪。

配色與框線沿用 day25_ingest／day25_handler，讀者一看就知道是同一組。
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
OUT_F, OUT_E = "#fffbf5", "#fdba74"      # run() 的外框，淡一點
EDGE_F, EDGE_E = "#fafaf9", "#a8a29e"    # 從外面注入的
STORE_E = "#16a34a"

SIG, TITLE, BODY, NOTE = 15, 15, 13.5, 12.5


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 8.6))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 112)
    ax.set_ylim(0, 84)
    ax.axis("off")

    # ------------------------------------------------------- run() 的外框
    ax.add_patch(FancyBboxPatch(
        (2, 12), 108, 70, boxstyle="round,pad=0,rounding_size=1.2",
        linewidth=2.2, edgecolor=OUT_E, facecolor=OUT_F, zorder=1))
    ax.text(6, 78.5, "run(store, job_id, upsert, on_done)", ha="left", va="center",
            fontproperties=MONO, fontsize=SIG + 1, color=INK, zorder=3)
    ax.text(6, 74.8, "背景執行緒跑的就是這個。下面兩個函式都是它呼叫的",
            ha="left", va="center", fontproperties=F, fontsize=BODY, color=MUTED,
            zorder=3)

    IL, IW = 7, 66        # 內層方塊
    cx = IL + IW / 2
    RX = IL + IW + 4      # 右邊註解

    def inner(y, h, sig, lines, fc=NEW_F, ec=NEW_E, note=None, mono=True):
        ax.add_patch(FancyBboxPatch(
            (IL, y), IW, h, boxstyle="round,pad=0,rounding_size=0.8",
            linewidth=1.8, edgecolor=ec, facecolor=fc, zorder=2))
        ax.text(IL + 3, y + h - 3.2, sig, ha="left", va="center",
                fontproperties=MONO if mono else FB, fontsize=SIG, color=INK,
                zorder=3)
        for i, t in enumerate(lines):
            ax.text(IL + 5, y + h - 7.6 - i * 3.6, t, ha="left", va="center",
                    fontproperties=F, fontsize=BODY, color=MUTED, zorder=3)
        if note:
            ax.text(RX, y + h / 2, note, ha="left", va="center",
                    fontproperties=F, fontsize=NOTE, color=MUTED, zorder=3)

    def down(y1, y2, label):
        ax.annotate("", xy=(cx, y2), xytext=(cx, y1),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2.0,
                                    shrinkA=0, shrinkB=0, mutation_scale=17), zorder=2)
        ax.text(cx + 1.5, (y1 + y2) / 2, label, ha="left", va="center",
                fontproperties=F, fontsize=NOTE, color=MUTED, zorder=4)

    inner(55, 17, "read_document(path, ocr)",
          ["markdown　　　　　直接讀",
           "PDF 有文字層　　　用 fitz 抽出來",
           "PDF 沒有文字層　　走 OCR（20 頁要 277 秒）"],
          note="所以攝取\n非得丟背景")
    down(55, 49, "一份純文字")

    inner(39, 10, "切塊（不是獨立函式，在 run() 裡）",
          ["照條號切，work_rules.md 切出 59 塊"], mono=False)
    down(39, 33, "59 塊")

    inner(21, 14, "chunk_id(source, meta, text)",
          ["work_rules.md:24:a3f91c02",
           "檔名＋條號＋內容雜湊 → 同樣內容算出同樣 id"],
          note="所以重覆攝取\n是覆蓋不是追加")

    # --------------------------------------------------- 注入進來的兩個東西
    ax.add_patch(FancyBboxPatch(
        (IL, 14.5), IW, 4, boxstyle="round,pad=0,rounding_size=0.6",
        linewidth=1.6, edgecolor=EDGE_E, facecolor=EDGE_F, zorder=2,
        linestyle="--"))
    ax.text(cx, 16.5, "upsert(...) 寫索引　｜　on_done() 重建 retriever",
            ha="center", va="center", fontproperties=F, fontsize=BODY, color=MUTED,
            zorder=3)
    ax.text(RX, 16.5, "這兩個是 main 傳進來的，\n所以這個檔案不認識 Chroma",
            ha="left", va="center", fontproperties=F, fontsize=NOTE, color=MUTED)

    # --------------------------------------------------------- 底下的狀態帶
    ax.text(2, 8, "從頭到尾，job 狀態一直在更新，所以呼叫端隨時查得到跑到哪：",
            ha="left", va="center", fontproperties=F, fontsize=BODY, color=MUTED)
    xs = [4, 32, 62]
    labels = ["queued", "running", "done ／ failed"]
    for i, (x, t) in enumerate(zip(xs, labels)):
        ax.text(x, 3.4, t, ha="left", va="center", fontproperties=FB,
                fontsize=BODY, color=STORE_E)
        if i < len(xs) - 1:
            ax.annotate("", xy=(xs[i + 1] - 2.5, 3.4), xytext=(x + 16, 3.4),
                        arrowprops=dict(arrowstyle="-|>", color=STORE_E, lw=1.5,
                                        shrinkA=0, shrinkB=0, mutation_scale=14))
    ax.text(84, 3.4, "失敗的訊息寫回 job，不是只進 log", ha="left", va="center",
            fontproperties=F, fontsize=NOTE, color=MUTED)

    fig.tight_layout(pad=0.5)
    for ext in ("png", "svg"):
        out = BASE_DIR / f"day25_ingest_module.{ext}"
        fig.savefig(out, dpi=170 if ext == "png" else None,
                    facecolor="white", bbox_inches="tight")
        print("寫出", out.name)


if __name__ == "__main__":
    main()
