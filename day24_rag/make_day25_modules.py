# -*- coding: utf-8 -*-
"""畫 store.py / cache.py / timing.py 三個模組的圖。

一次寫三張是因為它們的毛病一樣：函式表在 ithelp 上會折行，而且表格本來就
藏掉了每個檔案真正的結構——

    store.py   七個方法不是並列的，分三組，而且 exclusive() 根本不是 job 的事
    cache.py   五個函式是一條時間線：啟動裝上、跑的時候問、關機拆掉
    timing.py  五個都繞著同一個「桶子」轉，差別只在誰寫誰讀

輸出 day25_store.png／day25_cache.png／day25_timing.png（各附 svg）。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figstyle import (F, FB, MONO, INK, MUTED, NEW_F, NEW_E, OUT_F, OUT_E,
                            STORE_F, STORE_E, EDGE_F, EDGE_E, SIG, BODY, NOTE,
                            panel, card, sig_row, group, arrow, save)

BASE_DIR = Path(__file__).parent


# ---------------------------------------------------------------- store.py
def draw_store() -> None:
    """不畫方法清單，畫它們彼此怎麼接。

    重點是中間那條狀態帶：七個方法裡有四個都在動同一列資料，
    誰在什麼時候動它、動完別人看不看得到，才是這個檔案存在的理由。

    另外兩件表格藏不住但也講不出來的事，圖裡各給一格：
      exclusive()  跟 job 無關，它只是借這裡已經有的 SQLite 當跨行程的鎖
      purge()      寫了，但目前沒有任何地方呼叫它
    """
    fig, ax = plt.subplots(figsize=(13.6, 8.8))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 126)
    ax.set_ylim(0, 84)
    ax.axis("off")

    def lane(y, title, sub=""):
        ax.text(2, y, title, ha="left", va="center",
                fontproperties=FB, fontsize=BODY + 0.5, color=MUTED)
        if sub:
            ax.text(2, y - 3.2, sub, ha="left", va="center",
                    fontproperties=F, fontsize=NOTE - 0.5, color=MUTED)

    def call(x, y, w, sig, note="", fc=NEW_F, ec=NEW_E, dashed=False):
        card(ax, x, y, w, 8.2, fc=fc, ec=ec, dashed=dashed)
        ax.text(x + w / 2, y + 5.4, sig, ha="center", va="center",
                fontproperties=MONO, fontsize=NOTE + 0.5, color=INK, zorder=3)
        if note:
            ax.text(x + w / 2, y + 2.2, note, ha="center", va="center",
                    fontproperties=F, fontsize=NOTE - 1, color=MUTED, zorder=3)

    # ------------------------------------------------ 前景：POST /documents
    lane(80, "POST /documents", "前景，幾毫秒")
    call(30, 74.5, 30, "create(path, ocr)", "開一列，回 job_id")
    arrow(ax, 45, 74.5, 45, 68.5, color=STORE_E, lw=1.8)

    # --------------------------------------------------- 中間：那一列的狀態
    card(ax, 28, 58, 94, 10.5, fc=STORE_F, ec=STORE_E)
    ax.text(30, 65.4, "jobs 這一列（SQLite／WAL，四個 worker 共用同一份）",
            ha="left", va="center", fontproperties=F, fontsize=NOTE, color=MUTED,
            zorder=3)
    for i, (x, t) in enumerate([(34, "queued"), (62, "running"),
                                (92, "done ／ failed")]):
        ax.text(x, 61, t, ha="left", va="center", fontproperties=FB,
                fontsize=BODY, color=STORE_E, zorder=3)
        if i < 2:
            ax.annotate("", xy=([62, 92][i] - 2, 61), xytext=(x + 15, 61),
                        arrowprops=dict(arrowstyle="-|>", color=STORE_E, lw=1.4,
                                        shrinkA=0, shrinkB=0, mutation_scale=13),
                        zorder=3)

    # ------------------------------------------------- 背景：ingest.run()
    lane(52, "背景執行緒 ingest.run()", "可能跑好幾分鐘")
    # 方塊之間刻意留 x=51 與 x=83 兩條縫，底下查詢那排的箭頭要從那裡穿上去
    for x, w, sig, note in [
        (28, 20, "get(job_id)", "先確認這列還在"),
        (54, 26, 'update(status="running")', ""),
        (86, 36, 'update(status="done"…)', "或 failed，訊息一起寫回去"),
    ]:
        call(x, 44, w, sig, note)
    arrow(ax, 38, 52.2, 38, 58, color=STORE_E, lw=1.6, label="讀", boxed=False)
    arrow(ax, 67, 52.2, 67, 58, color=STORE_E, lw=1.6, label="寫", boxed=False)
    arrow(ax, 104, 52.2, 104, 58, color=STORE_E, lw=1.6, label="寫", boxed=False)

    # ------------------------------------------------------ 呼叫端隨時來查
    lane(37, "呼叫端隨時可以查", "GET /documents…")
    call(30, 28, 28, "get(job_id)", "查不到 → None → 端點回 404")
    call(66, 28, 28, "recent(limit)", "最近幾筆，給 GET /documents")
    # 從方塊之間那兩條縫穿上去，不要壓到背景執行緒那排
    arrow(ax, 51, 36.7, 51, 58, color=STORE_E, lw=1.4, dashed=True)
    arrow(ax, 83, 36.7, 83, 58, color=STORE_E, lw=1.4, dashed=True)
    ax.plot([44, 51], [36.7, 36.7], color=STORE_E, lw=1.4, ls="--", zorder=2)
    ax.plot([80, 83], [36.7, 36.7], color=STORE_E, lw=1.4, ls="--", zorder=2)
    ax.text(95, 32.6,
            "換一個 worker 來查，一樣查得到；",
            ha="left", va="center", fontproperties=F, fontsize=NOTE - 0.5,
            color=MUTED)
    ax.text(95, 29.4,
            "狀態不能放行程記憶體，就是為了這件事",
            ha="left", va="center", fontproperties=F, fontsize=NOTE - 0.5,
            color=MUTED)

    # ------------------------------------------ 另外三個，跟那一列沒有關係
    ax.plot([2, 124], [22.5, 22.5], linestyle="--", color=EDGE_E, lw=1.6)
    ax.text(2, 19.5, "剩下三個不碰那一列：", ha="left", va="center",
            fontproperties=FB, fontsize=BODY, color=MUTED)

    call(28, 9.5, 34, "exclusive(timeout)",
         "借 BEGIN IMMEDIATE 當跨行程的鎖", fc=OUT_F, ec=OUT_E)
    ax.text(28, 6.4,
            "攝取寫索引時擋別的 worker。它在這裡，只因為需要一把四個行程",
            ha="left", va="top", fontproperties=F, fontsize=NOTE - 0.5,
            color=MUTED)
    ax.text(28, 3.2,
            "都認得的鎖，而 SQLite 已經在這裡了",
            ha="left", va="top", fontproperties=F, fontsize=NOTE - 0.5,
            color=MUTED)

    call(66, 9.5, 22, "close()", "lifespan 收尾時", fc=EDGE_F, ec=EDGE_E)
    call(92, 9.5, 30, "purge(older_than)", "清掉過期的 job",
         fc=EDGE_F, ec=EDGE_E, dashed=True)
    ax.text(92, 6, "寫了，但目前沒有任何地方呼叫它",
            ha="left", va="top", fontproperties=F, fontsize=NOTE - 0.5, color=MUTED)

    fig.tight_layout(pad=0.5)
    save(fig, "day25_store", BASE_DIR)
    plt.close(fig)


# ---------------------------------------------------------------- cache.py
def draw_cache() -> None:
    """五個函式是一條時間線，不是一張清單。"""
    fig, ax = plt.subplots(figsize=(13.2, 6.6))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 60)
    ax.axis("off")

    # 時間軸
    ax.annotate("", xy=(116, 48), xytext=(4, 48),
                arrowprops=dict(arrowstyle="-|>", color=EDGE_E, lw=2.2,
                                shrinkA=0, shrinkB=0, mutation_scale=18), zorder=1)
    for x, t in [(6, "服務啟動"), (52, "服務在跑"), (99, "關機")]:
        ax.text(x, 51.5, t, ha="left", va="center", fontproperties=FB,
                fontsize=BODY + 1, color=MUTED)
        ax.plot([x, x], [46.4, 49.6], color=EDGE_E, lw=2.2, zorder=2)

    def slot(x, w, y, h, sig, lines, fc=NEW_F, ec=NEW_E):
        card(ax, x, y, w, h, fc=fc, ec=ec)
        ax.text(x + 2.5, y + h - 3.4, sig, ha="left", va="center",
                fontproperties=MONO, fontsize=SIG, color=INK, zorder=3)
        for i, t in enumerate(lines):
            ax.text(x + 2.5, y + h - 7.6 - i * 3.5, t, ha="left", va="center",
                    fontproperties=F, fontsize=NOTE, color=MUTED, zorder=3)

    slot(4, 44, 30, 13, "install()",
         ["把 rag_core 的讀寫換掉，回傳實際用到的後端",
          "連不上 Redis 就退回記憶體那版，服務照樣起得來"])
    slot(4, 44, 15, 12, "warm(*paths)",
         ["啟動時先把快取備好，也把版控裡那兩個",
          "JSON 當種子灌進 Redis（已經有的不覆蓋）"])

    slot(52, 44, 30, 13, "stats()",
         ["現在有幾筆、用的是哪個後端",
          "/healthz 會帶回去，不用翻 log 就知道有沒有降級"])
    slot(52, 44, 15, 12, "export(path)",
         ["把 Redis 裡的快取倒回 JSON 檔",
          "版控裡那兩個檔案才跟得上，讀者重跑才拿得到同樣數字"])

    slot(99, 18, 30, 13, "uninstall()",
         ["換回去", "關機與測試之間用"], fc=EDGE_F, ec=EDGE_E)

    ax.text(4, 8, "換掉的是 rag_core 讀寫快取的那兩個函式——"
                  "前 23 天每一篇的數字都靠那個檔案重現，所以不能直接改它。",
            ha="left", va="center", fontproperties=F, fontsize=NOTE, color=MUTED)
    ax.text(4, 4, "代價是這裡多了 install／uninstall 這組膠水；"
                  "如果你的專案沒有這條約束，直接改你自己的快取函式就好。",
            ha="left", va="center", fontproperties=F, fontsize=NOTE, color=MUTED)

    fig.tight_layout(pad=0.5)
    save(fig, "day25_cache", BASE_DIR)
    plt.close(fig)


# --------------------------------------------------------------- timing.py
def draw_timing() -> None:
    """五個函式都繞著同一個桶子轉，差別只在誰寫、誰讀、誰管生命週期。"""
    fig, ax = plt.subplots(figsize=(13.2, 7.0))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 64)
    ax.axis("off")

    # 中間的桶子
    BX, BW = 42, 42
    card(ax, BX, 28, BW, 17, fc=STORE_F, ec=STORE_E)
    ax.text(BX + BW / 2, 41.5, "這次請求的桶子", ha="center", va="center",
            fontproperties=FB, fontsize=BODY + 2, color=INK, zorder=3)
    ax.text(BX + BW / 2, 37.6, "放在 contextvars，一個請求一個",
            ha="center", va="center", fontproperties=F, fontsize=NOTE,
            color=MUTED, zorder=3)
    for i, seg in enumerate(["embed", "retrieve", "llm", "serialize"]):
        ax.text(BX + 3.5 + i * 9.2, 31.5, seg, ha="left", va="center",
                fontproperties=MONO, fontsize=NOTE - 0.5, color=STORE_E, zorder=3)

    # 左邊：寫進去的
    card(ax, 2, 44, 28, 10)
    ax.text(4, 51, "stage(name)", ha="left", va="center",
            fontproperties=MONO, fontsize=SIG, color=INK, zorder=3)
    ax.text(4, 46.8, "配 with 用，量一段", ha="left", va="center",
            fontproperties=F, fontsize=NOTE, color=MUTED, zorder=3)
    card(ax, 2, 30, 28, 10)
    ax.text(4, 37, "record(name, ms)", ha="left", va="center",
            fontproperties=MONO, fontsize=SIG, color=INK, zorder=3)
    ax.text(4, 32.8, "已經自己量好了，直接記", ha="left", va="center",
            fontproperties=F, fontsize=NOTE, color=MUTED, zorder=3)
    arrow(ax, 30, 49, BX, 43, label="累加", boxed=False, label_dy=1.6)
    arrow(ax, 30, 35, BX, 35, label="累加", boxed=False, label_dy=1.8)

    # 右邊：讀出來的
    card(ax, 88, 30, 30, 13)
    ax.text(90, 39.4, "breakdown(total_ms)", ha="left", va="center",
            fontproperties=MONO, fontsize=SIG - 0.5, color=INK, zorder=3)
    ax.text(90, 35.4, "攤成回應要的形狀，", ha="left", va="center",
            fontproperties=F, fontsize=NOTE, color=MUTED, zorder=3)
    ax.text(90, 32.4, "順便算出 overhead", ha="left", va="center",
            fontproperties=F, fontsize=NOTE, color=MUTED, zorder=3)
    arrow(ax, BX + BW, 36.5, 88, 36.5, label="讀", boxed=False, label_dy=1.8)

    # 下面：管生命週期的
    card(ax, 24, 12, 72, 11, fc=EDGE_F, ec=EDGE_E)
    ax.text(26, 19.6, "TimingASGIMiddleware", ha="left", va="center",
            fontproperties=MONO, fontsize=SIG, color=INK, zorder=3)
    ax.text(26, 15.4, "請求進來先 reset() 清空　｜　出去前 current() 取出，"
                      "寫進 x-*-ms header",
            ha="left", va="center", fontproperties=F, fontsize=NOTE, color=MUTED,
            zorder=3)
    arrow(ax, 60, 23, 60, 28, dashed=True)

    ax.text(3, 6.4, "為什麼非得放 contextvars：用模組層級的 dict，"
                    "兩個請求同時進來就會把時間互相加到對方身上。",
            ha="left", va="center", fontproperties=F, fontsize=NOTE, color=MUTED)
    ax.text(3, 2.6, "同一段量兩次會相加，這是要的——agent 回一題會打好幾次模型，"
                    "想看的是總和。",
            ha="left", va="center", fontproperties=F, fontsize=NOTE, color=MUTED)

    fig.tight_layout(pad=0.5)
    save(fig, "day25_timing", BASE_DIR)
    plt.close(fig)


if __name__ == "__main__":
    draw_store()
    draw_cache()
    draw_timing()
