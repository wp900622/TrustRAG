# -*- coding: utf-8 -*-
"""畫 Day 24 的圖：**一次 /ask 的時間，四段各佔多少。**

一張圖只講一件事：我前 13 天一直在調的那一段（檢索），
在三條路徑上都只有 1.4% 到 1.6%，那條細到要拉線才看得到的縫。
而真正的大頭會換人：快取全命中的時候是「算問題向量」（其實是在讀一個
10.5 MB 的快取檔），真的打 API 的時候是 LLM。

數字**不硬編**，直接讀 `runs_day24.json`，重跑實驗這張圖就會跟著變。

字型用微軟正黑體（Windows 內建）；換平台改 FONT_PATH 即可。
"""
import json
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
FAINT = "#a8a29e"
GRID = "#e7e5e4"

C_EMBED = "#0891b2"      # 算問題向量
C_RETR = "#dc2626"       # 檢索（今天的主角，因為它小得離譜）
C_LLM = "#7c3aed"        # LLM
C_REST = "#9ca3af"       # 序列化＋框架

SEGMENTS = [("embed", "算問題向量", C_EMBED),
            ("retrieve", "檢索", C_RETR),
            ("llm", "LLM", C_LLM),
            ("rest", "序列化＋框架", C_REST)]

ROWS = [("warm", "寫死管線・快取全命中\n（前 23 天的樣子）"),
        ("cold", "寫死管線・真的打 API\n（上線後的樣子）"),
        ("agent_cold", "agent・真的打 API\n（自己決定查幾次）")]


def load() -> dict:
    return json.loads((BASE_DIR / "runs_day24.json").read_text("utf-8"))


def shares(summary: dict) -> dict:
    """把五段收成四段：序列化與框架合併，因為兩個都小到看不見"""
    sh = dict(summary["share"])
    sh["rest"] = round(sh.pop("serialize", 0) + sh.pop("overhead", 0), 2)
    total = sum(sh[name] for name, _, _ in SEGMENTS)
    # 各段的中位數加起來不會剛好等於 p50，正規化成 100% 才畫得出堆疊條
    return {name: sh[name] / total * 100 for name, _, _ in SEGMENTS}


def main() -> None:
    out = load()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6.0), gridspec_kw={"width_ratios": [2.5, 1]})
    fig.patch.set_facecolor("white")

    ys = list(range(len(ROWS)))[::-1]
    for y, (key, label) in zip(ys, ROWS):
        sh = shares(out[key]["summary"])
        left = 0.0
        for name, text, color in SEGMENTS:
            width = sh[name]
            ax1.barh(y, width, left=left, height=0.5, color=color,
                     edgecolor="white", linewidth=1.4, zorder=3)
            if width >= 7:
                ax1.text(left + width / 2, y, f"{width:.0f}%", ha="center",
                         va="center", color="white", fontproperties=FB,
                         fontsize=12, zorder=4)
            if name == "retrieve":
                # 檢索那一條細到寫不進去，拉一條線出來標
                ax1.annotate(f"檢索 {width:.1f}%",
                             xy=(left + width / 2, y + 0.26),
                             xytext=(left + width / 2, y + 0.46),
                             ha="center", va="bottom", fontproperties=FB,
                             fontsize=11.5, color=C_RETR, zorder=5,
                             arrowprops=dict(arrowstyle="-", color=C_RETR, lw=1.4))
            left += width
        ax1.text(101.5, y, f"p50 {out[key]['summary']['p50']:,.0f} ms",
                 va="center", fontproperties=FB, fontsize=12, color=INK)

    ax1.set_yticks(ys)
    ax1.set_yticklabels([lb for _, lb in ROWS], fontproperties=F, fontsize=12,
                        color=INK)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.6, len(ROWS) - 0.15)
    ax1.set_xlabel("佔一次請求的百分比", fontproperties=F, fontsize=11, color=MUTED)
    ax1.set_title("同一支 /ask，同一組 28 題，三條路徑",
                  fontproperties=FB, fontsize=14, color=INK, pad=16, loc="left")
    for side in ("top", "right", "left"):
        ax1.spines[side].set_visible(False)
    ax1.spines["bottom"].set_color(GRID)
    ax1.tick_params(axis="x", colors=MUTED, labelsize=10)
    ax1.tick_params(axis="y", length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SEGMENTS]
    leg = ax1.legend(handles, [t for _, t, _ in SEGMENTS], loc="lower left",
                     frameon=False, ncol=4, fontsize=11,
                     bbox_to_anchor=(0.0, -0.36))
    for t in leg.get_texts():
        t.set_fontproperties(F)
        t.set_color(MUTED)

    # ------------------------------------------------ 右：檢索那一段的絕對值
    bars = [("chroma", "Chroma", out["warm"]["summary"]["stages"]["retrieve"]),
            ("numpy", "numpy\n（59 條暴力解）",
             out["numpy"]["summary"]["stages"]["retrieve"]),
            ("agent_numpy", "agent 跑在\nnumpy 上",
             out["agent_numpy"]["summary"]["stages"]["retrieve"])]
    xs = range(len(bars))
    vals = [v for _, _, v in bars]
    ax2.bar(xs, vals, width=0.5, color=C_RETR, zorder=3)
    for x, v in zip(xs, vals):
        ax2.text(x, v + max(vals) * 0.03, f"{v:.2f} ms", ha="center", va="bottom",
                 fontproperties=FB, fontsize=12, color=INK, zorder=4)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([lb for _, lb, _ in bars], fontproperties=F, fontsize=11,
                        color=INK)
    ax2.set_ylabel("檢索那一段（毫秒）", fontproperties=F, fontsize=11, color=MUTED)
    ax2.set_title("換一個 retriever：連凍結的 agent 都照跑",
                  fontproperties=FB, fontsize=14, color=INK, pad=16, loc="left")
    ax2.grid(axis="y", color=GRID, zorder=0)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax2.spines[side].set_color(GRID)
    ax2.tick_params(colors=MUTED, labelsize=10)
    ax2.set_ylim(0, max(vals) * 1.25)

    note = (f"同一台機器、同一組 {out['n']} 題、k={out['k']}、59 條規章。"
            f"「算問題向量」在快取命中時量到的其實是把 10.5 MB 的 embedding "
            f"快取檔整包讀進來。序列化與框架合併顯示，兩者都小於 1 ms。"
            f"agent 一題會打好幾次模型，所以 LLM 那一段是累加的。")
    fig.text(0.012, 0.012, note, fontproperties=F, fontsize=9.5, color=FAINT)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "svg"):
        path = BASE_DIR / f"day24_where_time_goes.{ext}"
        fig.savefig(path, dpi=170 if ext == "png" else None,
                    facecolor="white", bbox_inches="tight")
        print("→", path.name)


if __name__ == "__main__":
    main()
