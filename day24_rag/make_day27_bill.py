# -*- coding: utf-8 -*-
"""畫 26 天的帳單，輸出 day27_bill.png / .svg。

左邊：錢實際花在哪一筆快取上（由大到小）。
右邊：同一筆錢換一個切法——輸入／輸出／embedding。

論點是右邊那一條：26 天我一直在要求「答案要短」，
而輸出只佔帳單一成；前 13 天調的檢索佔 0.11%。

數字全部讀自 cost_summary_day27.json，不是手打的。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figstyle import F, FB, INK, MUTED, NEW_E, STORE_E, EDGE_E, save

BASE_DIR = Path(__file__).parent
MINI_C = "#fdba74"        # gpt-4o-mini
GPT4O_C = "#ea580c"       # gpt-4o（貴的那個）
EMB_C = "#d6d3d1"         # embedding


def short(label: str) -> str:
    """表格裡的長名字在圖上要短。"""
    table = {
        "`chat_cache_day19.json`（換 gpt-4o 當裁判）": "Day 19 改用 gpt-4o 改考卷",
        "`chat_cache.json`（主線問答＋rubric 判定）": "問答，加上逐項改考卷",
        "`vision_cache_day22.json`（整頁轉錄，全被拒絕）": "Day 22 請模型讀整頁圖（18 次被拒絕）",
        "`review_cache_day18.json`（gpt-4o 盲標覆核）": "Day 18 請 gpt-4o 覆核我的標註",
        "`agent_cache_day20/21.json`（多步 agent）": "Day 20／21 會自己查好幾次的 agent",
        "Day 23 看圖回答（M-mini）": "Day 23 看圖回答（便宜的模型）",
        "Day 23 看圖回答（M-4o）": "Day 23 看圖回答（貴的模型）",
        "`chat_cache_day24_*.json`（服務化的三份）": "Day 24 包成服務時",
        "`embeddings_cache.json`（全部向量）": "26 天所有的向量",
    }
    return table.get(label, label)


def main() -> None:
    d = json.loads((BASE_DIR / "cost_summary_day27.json").read_text(encoding="utf-8"))
    rows = sorted(d["rows"], key=lambda r: r["twd"])
    total = d["total_twd"]

    fig = plt.figure(figsize=(14.4, 7.6))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.45, 1], wspace=0.34)

    # ---------------------------------------------------- 左：一筆一筆的帳
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor("white")
    colors = {"gpt-4o": GPT4O_C, "gpt-4o-mini": MINI_C, "embedding": EMB_C}
    ys = range(len(rows))
    ax.barh(list(ys), [r["twd"] for r in rows],
            color=[colors[r["model"]] for r in rows],
            edgecolor="white", height=0.66)
    for y, r in zip(ys, rows):
        ax.text(r["twd"] + total * 0.012, y,
                f"{r['twd']:.2f} 元　{r['twd'] / total * 100:.0f}%",
                ha="left", va="center", fontproperties=F, fontsize=12, color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([short(r["label"]) for r in rows],
                       fontproperties=F, fontsize=12.5)
    ax.set_xlim(0, total * 0.78)
    ax.set_xticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"26 天總共 NT${total:.2f}，{d['total_calls']:,} 次獨立呼叫",
                 fontproperties=FB, fontsize=16, color=INK, loc="left", pad=16)
    ax.text(0, len(rows) - 0.25,
            "深橘＝貴的模型 gpt-4o　淺橘＝便宜的 gpt-4o-mini　灰＝向量",
            fontproperties=F, fontsize=11.5, color=MUTED)

    # ---------------------------------------------------- 右：同一筆錢的另一種切法
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("white")
    parts = [("輸入（我塞進去的）", d["in_twd"], NEW_E),
             ("輸出（模型講的）", d["out_twd"], MINI_C),
             ("向量（找條文用的）", d["emb_twd"], EMB_C)]
    bottom = 0.0
    for name, val, c in parts:
        ax2.bar([0], [val], bottom=bottom, color=c, edgecolor="white",
                width=0.62, linewidth=1.5)
        pct = val / total * 100
        # 標籤一律放在柱子右邊：0.11% 那一段薄得放不進任何字，
        # 三段都往外拉才會長得像同一組
        y = bottom + val / 2 if pct >= 5 else total * 1.10
        ax2.plot([0.33, 0.44], [bottom + val / 2, y], color=MUTED, lw=1.2)
        ax2.text(0.5, y, f"{name}　{pct:.2f}%" if pct < 5 else f"{name}　{pct:.1f}%",
                 ha="left", va="center", fontproperties=FB if pct >= 5 else F,
                 fontsize=13 if pct >= 5 else 12, color=INK)
        ax2.text(0.5, y - total * 0.042, f"{val:.4f} 元" if pct < 5 else f"{val:.2f} 元",
                 ha="left", va="center", fontproperties=F, fontsize=11.5, color=MUTED)
        bottom += val
    ax2.set_xlim(-0.45, 2.0)
    ax2.set_ylim(0, total * 1.18)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax2.spines[s].set_visible(False)
    ax2.set_title("同一筆錢，換一個切法", fontproperties=FB, fontsize=16,
                  color=INK, loc="left", pad=16)
    ax2.text(-0.45, -total * 0.055,
             "我的 prompt 第一行註解寫著「答案要短，輸出比輸入貴 4 倍」。\n"
             "單價確實是 4 倍，但 RAG 是塞很多、講很少，\n"
             "那個 4 倍沒有機會發生。我省錯地方了。",
             ha="left", va="top", fontproperties=F, fontsize=12, color=MUTED)

    save(fig, "day27_bill", BASE_DIR)


if __name__ == "__main__":
    main()
