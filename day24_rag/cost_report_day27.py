# -*- coding: utf-8 -*-
"""Day 27：把 `cost_audit_day27.py` 算出來的原始數字結成兩本帳。

帳本 A（我到底買了什麼）：把每一份快取當成「已經付過錢的獨立呼叫」清單。
帳本 B（每天的實驗值多少）：每天 `experiment_result_*.md` 自己記的合計。

兩本不會相等，差額就是跨天重複使用的部分。**這支同樣不打 API。**
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
NL = chr(10)
MINI, GPT4O = "gpt-4o-mini", "gpt-4o"
PRICES = {MINI: (0.15, 0.60), GPT4O: (2.50, 10.00)}
USD_PER_M_EMB = 0.02
USD_TO_TWD = 31

A = json.loads((BASE_DIR / "cost_audit_day27.json").read_text(encoding="utf-8"))


def cost(model, tin, tout):
    pin, pout = PRICES[model]
    return (tin / 1e6 * pin + tout / 1e6 * pout) * USD_TO_TWD


def row(label, n, model, tin, tout, exact):
    return {"label": label, "n": n, "model": model, "in": tin, "out": tout,
            "twd": cost(model, tin, tout), "exact": exact}


# ------------------------------------------------- 帳本 A：獨立買過的呼叫

rows = []
m = A["main"]
rows.append(row("`chat_cache.json`（主線問答＋rubric 判定）", m["n"], MINI,
                m["in_tokens_exact"] + m["in_tokens_est"], m["out_tokens"],
                f"{m['matched']}/{m['n']} 精確"))

d19 = A["day19"]
rows.append(row("`chat_cache_day19.json`（換 gpt-4o 當裁判）", d19["n"], GPT4O,
                d19["in_tokens_exact"] + d19["in_tokens_est"], d19["out_tokens"],
                f"{d19['matched']}/{d19['n']} 精確"))

rv = A["review"]
rows.append(row("`review_cache_day18.json`（gpt-4o 盲標覆核）", rv["n"], GPT4O,
                rv["in_tokens_exact"], rv["out_tokens"], "全部精確"))

d24n = sum(x["n"] for x in A["day24"])
rows.append(row("`chat_cache_day24_*.json`（服務化的三份）", d24n, MINI,
                sum(x["in_tokens_exact"] for x in A["day24"]),
                sum(x["out_tokens"] for x in A["day24"]), "全部精確"))

# agent 的輸入沒有留下來；用 runs_day21.json 記的每次呼叫平均回推
r21 = A["runs"]["runs_day21.json"]
agent_mean_in = r21["in"] / r21["calls"]
agent_n = sum(x["n"] for x in A["agents"])
rows.append(row("`agent_cache_day20/21.json`（多步 agent）", agent_n, MINI,
                round(agent_n * agent_mean_in),
                sum(x["out_tokens"] for x in A["agents"]),
                f"輸入全靠回推（{agent_mean_in:.0f}／次）"))

v22 = A["visions"][0]
rows.append(row("`vision_cache_day22.json`（整頁轉錄，全被拒絕）", v22["n"], MINI,
                v22["in_tokens"], v22["out_tokens"], "當初存了 usage"))

# Day 23 的視覺：runs_day23.json 逐筆記了模型，比快取完整（含一題退回吃文字的）
d23 = json.loads((BASE_DIR / "runs_day23.json").read_text(encoding="utf-8"))
for g, model in (("M-mini", MINI), ("M-4o", GPT4O)):
    tin = sum(p[g]["input_tokens"] for p in d23.values() if g in p)
    tout = sum(p[g]["output_tokens"] for p in d23.values() if g in p)
    rows.append(row(f"Day 23 看圖回答（{g}）", 8, model, tin, tout, "當初存了 usage"))

emb = A["emb"]
emb_mean = emb["tokens_matched"] / emb["matched"]
emb_total = round(emb["tokens_matched"] + (emb["n"] - emb["matched"]) * emb_mean)
emb_twd = emb_total / 1e6 * USD_PER_M_EMB * USD_TO_TWD
rows.append({"label": "`embeddings_cache.json`（全部向量）", "n": emb["n"],
             "model": "embedding", "in": emb_total, "out": 0, "twd": emb_twd,
             "exact": f"{emb['matched']}/{emb['n']} 精確"})

total_twd = sum(r["twd"] for r in rows)
total_calls = sum(r["n"] for r in rows)

# 輸入／輸出／embedding 各佔多少錢
in_twd = sum((r["in"] / 1e6 * PRICES[r["model"]][0] * USD_TO_TWD)
             for r in rows if r["model"] in PRICES)
out_twd = sum((r["out"] / 1e6 * PRICES[r["model"]][1] * USD_TO_TWD)
              for r in rows if r["model"] in PRICES)
model_twd = {}
for r in rows:
    model_twd[r["model"]] = model_twd.get(r["model"], 0) + r["twd"]

# ------------------------------------------------- 帳本 B：每天自己記的合計
# 數字一律抄自各天的 experiment_result_*.md，來源標在第三欄。
LEDGER_B = [
    ("Day 14", 0.16500, "`experiment_result.md`：合計"),
    ("Day 15", 0.00000, "全部命中 Day 14 的快取，當天實際計費 0"),
    ("Day 16", 0.22211, "`day16_cost`：60 次全部從零 0.21360 ＋ embedding 0.00851"),
    ("Day 17", 0.72268, "從零 0.62151 ＋ embedding 0.00018 ＋ 重排 0.10099"),
    ("Day 18", 1.09064, "`day18`：合計"),
    ("Day 19", 22.08048, "`day19`：合計（含 B3′ 5.48 與 C′ 3.14）"),
    ("Day 20", 0.83790, "`day20`：合計（不含第八節重跑）"),
    ("Day 21", 2.47740, "`day21`：合計"),
    ("Day 22", 4.43380, "`day22`：合計"),
    ("Day 23", 2.01300, "`day23`：四條路徑相加"),
    ("Day 24", 0.58690, "`day24`：今天總計"),
    ("Day 25", 0.00000, "服務化，沒有新的呼叫"),
    ("Day 26", 0.18600, "`day26`：約 US$0.006"),
]
ledger_b_total = sum(x[1] for x in LEDGER_B)


def fmt(x, n=4):
    return f"{x:,.{n}f}"


lines = [
    "# Day 27 實驗結果：26 天的帳",
    "",
    "跑於 2026-09-22。`cost_audit_day27.py` ＋ `cost_report_day27.py`，"
    "**兩支都不打 API**（進場先把 `rag_core.get_client` 換成會拋例外的版本）。",
    "",
    "定價用 `rag_core.py` 裡的那三個數字：embedding US$0.02／M、"
    "`gpt-4o-mini` 0.15／0.60、`gpt-4o` 2.50／10.00，匯率 31。",
    "**這是今天的定價，不是當初刷卡當下的**——26 天內沒改過，但要寫明。",
    "",
    "## 〇、先驗尺",
    "",
    f"輸出 token 是拿 `tiktoken` 離線重數的，所以得先證明這把尺準。"
    f"`runs_day21/22/23.json` 裡有 {A['tokenizer']['n']} 筆同時留下了"
    f"**答案原文**與**當初 API 回報的 `output_tokens`**（排除多步 agent 的加總值）：",
    "",
    f"- 完全一致 **{A['tokenizer']['exact']}/{A['tokenizer']['n']}**，"
    f"最大誤差 **{A['tokenizer']['max_abs']}** token。",
    "",
    "o200k_base 數中文答案與 API 的計費完全相同。以下所有輸出 token 都是這樣數的。",
    "",
    "## 一、帳本 A：我到底買了什麼",
    "",
    "每一份快取就是一份「已經付過錢的獨立呼叫」清單。輸出精確，"
    "輸入看考古對不對得上（見第三節）。",
    "",
    "| 來源 | 筆數 | 模型 | 輸入 token | 輸出 token | 台幣 | 輸入的精確度 |",
    "|---|---|---|---|---|---|---|",
]
for r in rows:
    lines.append(f"| {r['label']} | {r['n']:,} | `{r['model']}` | {r['in']:,} | "
                 f"{r['out']:,} | {fmt(r['twd'])} | {r['exact']} |")
lines += [
    f"| **合計** | **{total_calls:,}** | — | — | — | **{fmt(total_twd)}** | — |",
    "",
    f"26 天買到 **{total_calls:,} 次獨立呼叫**，重建總額 **NT${fmt(total_twd, 2)}**。",
    "",
    "### 錢花在哪一種 token 上",
    "",
    "| | 台幣 | 佔比 |",
    "|---|---|---|",
    f"| 輸入 | {fmt(in_twd)} | {in_twd / total_twd * 100:.1f}% |",
    f"| 輸出 | {fmt(out_twd)} | {out_twd / total_twd * 100:.1f}% |",
    f"| Embedding | {fmt(emb_twd)} | {emb_twd / total_twd * 100:.2f}% |",
    "",
    "### 錢花在哪一個模型上",
    "",
    "| 模型 | 台幣 | 佔比 |",
    "|---|---|---|",
]
for k, v in sorted(model_twd.items(), key=lambda x: -x[1]):
    lines.append(f"| `{k}` | {fmt(v)} | {v / total_twd * 100:.1f}% |")

lines += [
    "",
    "## 二、帳本 B：每天的實驗值多少",
    "",
    "各天 `experiment_result_*.md` 自己記的合計，一個都沒重算，來源標在第三欄。",
    "",
    "| 天 | 台幣 | 來源 |",
    "|---|---|---|",
]
for day, twd_, src in LEDGER_B:
    lines.append(f"| {day} | {fmt(twd_)} | {src} |")
lines += [
    f"| **合計** | **{fmt(ledger_b_total)}** | |",
    "",
    f"帳本 B **NT${fmt(ledger_b_total, 2)}**，帳本 A **NT${fmt(total_twd, 2)}**，"
    f"差 **NT${fmt(ledger_b_total - total_twd, 2)}**"
    f"（{(ledger_b_total - total_twd) / ledger_b_total * 100:.0f}%）。",
    "",
    "差額是跨天重複使用的那些呼叫：帳本 B 把它們算進每一天，"
    "帳本 A 只算第一次付錢的那次。兩本都對，問的問題不一樣。",
    "",
    f"**Day 19 一天就佔帳本 B 的 {22.08048 / ledger_b_total * 100:.0f}%。**"
    f"帳本 A 這邊，兩份 `gpt-4o` 的快取加起來是 "
    f"NT${fmt(model_twd[GPT4O], 2)}，佔 {model_twd[GPT4O] / total_twd * 100:.0f}%。"
    "兩本帳從不同方向指向同一件事。",
    "",
    "## 三、考古：我還認得出自己買了什麼嗎",
    "",
    "快取的 key 是 `sha256([model, messages])[:16]`，prompt 沒有留。"
    "唯一的辦法是把已知的 prompt 空間重建一次，算雜湊回去對。",
    "",
    f"重建出 **{A['candidates']:,} 組候選**："
    "主線語料的 P1／P2／P3 × 五組題目 × k∈{1,3,5,10,15}（含 Day 17 的反序）、"
    "無 context 對照組、Day 22 的四種語料形態、Day 23 的兩份附件語料、"
    "Day 18／19 的四種 rubric prompt × 兩個模型、Day 18 的盲標覆核。",
    "",
    "| 快取 | 認得出來 | 比例 |",
    "|---|---|---|",
    f"| `chat_cache.json` | {m['matched']}/{m['n']} | {m['matched'] / m['n'] * 100:.0f}% |",
    f"| `chat_cache_day19.json` | {d19['matched']}/{d19['n']} | "
    f"{d19['matched'] / d19['n'] * 100:.0f}% |",
    f"| `review_cache_day18.json` | {rv['matched']}/{rv['n']} | 100% |",
    f"| `chat_cache_day24_*.json` | {sum(x['matched'] for x in A['day24'])}/{d24n} | 100% |",
    f"| `embeddings_cache.json` | {emb['matched']}/{emb['n']} | "
    f"{emb['matched'] / emb['n'] * 100:.0f}% |",
    "",
    "對不上的那些，用「同一個輸出長度分組裡、對得上的那些」的平均輸入回推：",
    "",
    "| 輸出長度 | 對不上的筆數 | 回推用的平均輸入 | 這個平均來自幾筆 |",
    "|---|---|---|---|",
]
for b in ("≤5", "6-20", "21-60", "61+"):
    e = m["est_basis"].get(b)
    if e:
        lines.append(f"| {b} token | {e['n']} | {e['mean_in']:,.0f} | {e['from']} |")
lines += [
    "",
    f"`chat_cache.json` 對不上的 {m['unmatched']} 筆裡，"
    f"**{m['unmatched_out_hist'].get('≤5', 0)} 筆的輸出只有 5 個 token 以內**——"
    "那是「符合／未提及／相反」，不是答案。也就是說對不上的大宗不是問答，"
    "是我沒能重建出來的那些判定組合（Day 17 掃 k 時產生、後來又被送進裁判的答案）。",
    "",
    "## 四、快取到底省了多少",
    "",
    "**先講清楚這個數字的定義。** 快取擋下來的呼叫本來就不會付第二次錢，"
    "所以「省下的錢」不是現金，是反事實：**同樣的實驗從零重跑一次要花多少**。",
    "",
    f"一次完整重跑 ≈ 帳本 B 的 **NT${fmt(ledger_b_total, 2)}**。",
    "",
    "能精確對帳的只有一天。`runs_day21.json` 逐筆記了邏輯呼叫次數，"
    f"當天四組共 **{r21['calls']} 次**；`agent_cache_day21.json` 只有 "
    f"**{A['agents'][1]['n']} 筆**獨立。",
    f"差的 **{r21['calls'] - A['agents'][1]['n']} 次**是快取擋下來的，"
    f"以當天平均輸入 {agent_mean_in:.0f} token 計，值 "
    f"NT${fmt(cost(MINI, round((r21['calls'] - A['agents'][1]['n']) * agent_mean_in), 0), 2)}，"
    "時間以 Day 25 量到的 LLM 中位數 1,054 ms 計約 "
    f"**{(r21['calls'] - A['agents'][1]['n']) * 1.054 / 60:.1f} 分鐘**。",
    "",
    "**26 天的總命中次數算不出來，因為沒有人數過。**"
    "這跟「沒有記帳」是同一個毛病的兩面：我留了買到的東西，"
    "沒留收據，也沒留命中次數。",
    "",
    "## 五、一次問答多少錢",
    "",
    "拿 Day 23 記的寫死管線（`gpt-4o-mini`、k=3、8 題）當單價：",
    "",
    "| 路徑 | 每題 |",
    "|---|---|",
    "| T 文字管線 | 0.0027 元 |",
    "| M-mini 看圖 | 0.1514 元（56×） |",
    "| M-4o 看圖 | 0.0948 元（35×） |",
    "",
    f"以 0.0027 元換算，26 天的 NT${fmt(total_twd, 2)} 等於 "
    f"**約 {total_twd / 0.0027:,.0f} 次問答**。",
    "",
    "## 六、預測對帳",
    "",
    "預測寫在 `day27_plan.md`，跑之前寫的，以下照原樣貼。",
    "",
    "| # | 預測 | 實測 | |",
    "| --- | --- | --- | --- |",
    f"| 1 | `chat_cache.json` 輸出 token 48,000 | **{m['out_tokens']:,}** | "
    f"✗ 高估 {48000 / m['out_tokens']:.1f}× |",
    f"| 2 | 認得出來的比例 55% | **{m['matched'] / m['n'] * 100:.0f}%** | ✓ 差 2 個百分點 |",
    f"| 3 | 26 天總花費 NT$62 | **NT${fmt(total_twd, 2)}** | "
    f"✗ 高估 {62 / total_twd:.1f}× |",
    f"| 4 | 沒有快取要花 NT$180 | **每重跑一次 NT${fmt(ledger_b_total, 2)}** | "
    "✗ 高估 5.2×，而且**問題問錯了** |",
    "| 5 | 一次問答 NT$0.0028 | **0.0027 元** | ✓ 差 4% |",
    f"| 6 | embedding 26,000 token／NT$0.016 | **{emb_total:,}／NT${fmt(emb_twd, 4)}** | "
    f"✗ 低估 {emb_total / 26000:.1f}× |",
    "| 7 | 最貴單項＝Day 22 的視覺拒絕，佔 5.5% | "
    f"**佔 {3.4435 / total_twd * 100:.1f}%，而且不是最貴的** | ✗ |",
    f"| 反轉候選 | 輸入佔帳單 76% | **{in_twd / total_twd * 100:.1f}%** | ✓ 方向對，低估 |",
    "| 第二反轉 | 省的錢少、省的時間多（>3 小時） | 錢確實少；時間只對得出 3.7 分鐘 | 半對 |",
    "",
    "### 第 4 條錯在哪",
    "",
    "我把「快取省了多少」當成一個可以加總的現金數字，先寫了 NT$180。"
    "跑到一半才發現那個問題本身沒有答案：**要知道省了多少，得先知道命中了幾次，"
    "而命中次數從來沒有人數過。**能算的只有反事實（重跑一次要花多少），"
    "以及唯一有逐筆紀錄的那一天。",
    "",
    "### 第 7 條錯在哪",
    "",
    "我把注意力放在「買到 20 次拒絕」這個荒謬感上，以為那是最貴的一筆。"
    f"它確實貴（NT$3.44，佔 {3.4435 / total_twd * 100:.1f}%），但真正的大頭是"
    f"**換一次裁判模型**：兩份 `gpt-4o` 快取 NT${fmt(model_twd[GPT4O], 2)}，"
    f"佔 {model_twd[GPT4O] / total_twd * 100:.0f}%。"
    "荒謬的東西比較好記，貴的東西比較安靜。",
    "",
]

(BASE_DIR / "cost_summary_day27.json").write_text(json.dumps({
    "rows": rows, "total_twd": total_twd, "total_calls": total_calls,
    "in_twd": in_twd, "out_twd": out_twd, "emb_twd": emb_twd,
    "model_twd": model_twd, "ledger_b": LEDGER_B, "ledger_b_total": ledger_b_total,
}, ensure_ascii=False, indent=2), encoding="utf-8")

out_path = BASE_DIR / "experiment_result_day27.md"
out_path.write_text(NL.join(lines) + NL, encoding="utf-8")

print(NL.join(lines[:0]) or "")
print(f"帳本 A 合計 NT${total_twd:.4f}　（{total_calls:,} 次獨立呼叫）")
print(f"帳本 B 合計 NT${ledger_b_total:.4f}")
print(f"輸入 {in_twd:.4f}（{in_twd / total_twd * 100:.1f}%）　"
      f"輸出 {out_twd:.4f}（{out_twd / total_twd * 100:.1f}%）　"
      f"embedding {emb_twd:.4f}（{emb_twd / total_twd * 100:.2f}%）")
for k, v in sorted(model_twd.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v:.4f}（{v / total_twd * 100:.1f}%）")
print(f"embedding 總 token {emb_total:,}")
print(f"寫入 {out_path.name}")
