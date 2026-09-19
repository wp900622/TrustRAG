# -*- coding: utf-8 -*-
"""Day 20 實驗：會自己決定的迴圈 vs 寫死的管線。

Day 19 文末寫死了今天的題目：

    明天讓模型自己決定：要不要檢索、用什麼字去檢索、撈回來不夠要不要再撈一次。
    用今天選定的尺量它，跟固定 k=3 的單次檢索正面對撞，
    兩個數字並排：答對幾題、花了幾倍的錢。

評估側全部凍結（尺 C 的 prompt、Day 19 的 rubric、Day 18 的三條標註原則）。
**今天唯一的變因在檢索與生成那一側。**

三組題，三種預期（`day20_plan.md` 第二節寫死，跑完不准改）：

    單條文 15 題   預測平手、但貴好幾倍——一次就查得到，迴圈是純成本
    多條文  8 題   預測 agent 贏——Day 17 量過固定 k 撈不齊
    陷阱題  5 題   預測最容易出事——規章根本沒寫，而 agent 有「再查一次」這個逃生口

尺怎麼用：

    單條文／多條文  主尺 C（Day 19 選定）
    多條文那 8 題   另外跑一次 Day 19 修好的 rubric（B3）當第二意見，
                    **兩把尺不一致的我逐筆自己讀**——這是 Day 19 量出來的用法
    陷阱題          尺 C 的 prompt 要條文原文，而陷阱題沒有「必要條文」，
                    所以不套用；改成我人工標註，並把 Day 15 的關鍵詞規則
                    並排列出來當對照（它是 Day 18 判定 15/40 的那把舊尺）

用法：
    python run_experiment_day20.py --label   # 產出標註工作表
    python run_experiment_day20.py           # 標註完成後跑正式報表
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import agent_day20 as ag
import grader
import judges
import judges_day19 as j19
import pipeline
import rag_core
import run_experiment_day18 as d18

BASE_DIR = Path(__file__).parent
RESULT_PATH = BASE_DIR / "experiment_result_day20.md"
LABELS_PATH = BASE_DIR / "human_labels_day20.json"
NL = chr(10)

SETS = [
    ("單條文", BASE_DIR / "questions.json"),
    ("多條文", BASE_DIR / "questions_day17.json"),
    ("陷阱", BASE_DIR / "questions_trap.json"),
]
CONDITIONS = ("baseline", "agent")
COND_LABEL = {"baseline": "對照組（寫死：原句查一次、k=3）",
              "agent": "agent（自己決定查幾次）"}
COND_SHORT = {"baseline": "對照組", "agent": "agent"}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_expected(raw) -> list[int]:
    """`第 24 條` → [24]；`[19, 33]` → [19, 33]；None → []"""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [int(x) for x in raw]
    return [int(n) for n in re.findall(r"\d+", str(raw))]


def load_questions() -> list[dict]:
    out = []
    for group, path in SETS:
        for q in load_json(path):
            out.append({"group": group, "id": q["id"], "question": q["question"],
                        "expected": parse_expected(q.get("expected")),
                        "facts": q.get("facts", []),
                        "forbid": q.get("forbid"),
                        "raw": q})
    return out


# ------------------------------------------------------------------ 跑兩邊

def run_both(collection, questions) -> dict:
    runs = {}
    for q in questions:
        key = f"{q['group']}-{q['id']}"
        runs[key] = {
            "baseline": ag.run_baseline(collection, q["question"]),
            "agent": ag.run_agent(collection, q["question"]),
        }
    return runs


def union_recall(got: list[int], expected: list[int]) -> float | None:
    """必要條文有幾成進到模型手上。沒有必要條文（陷阱題）回 None。"""
    if not expected:
        return None
    return sum(1 for n in expected if n in got) / len(expected)


# ------------------------------------------------------------------ 判定

def judge_all(questions, runs, texts) -> dict:
    """尺 C 判非陷阱題；多條文那組另外跑 B3。"""
    rubrics = {r["id"]: r["checkpoints"]
               for r in load_json(BASE_DIR / "rubric_day18.json")}
    out = {}
    for q in questions:
        key = f"{q['group']}-{q['id']}"
        out[key] = {}
        for cond in CONDITIONS:
            cell = {}
            if q["group"] != "陷阱":
                blob = d18.articles_blob({"expected": q["expected"]}, texts)
                cell["C"] = j19.grade_judge(q["question"], blob,
                                            runs[key][cond]["answer"],
                                            model=j19.MINI)
            if q["group"] == "多條文":
                cell["B3"] = j19.grade_rubric(runs[key][cond]["answer"],
                                              rubrics[q["id"]],
                                              variant="B3", model=j19.MINI)
            if q["group"] == "陷阱":
                verdict, hits = grader.grade_trap(runs[key][cond]["answer"],
                                                  q["raw"])
                cell["day15"] = {"verdict": verdict, "markers": hits}
            out[key][cond] = cell
    return out


# ------------------------------------------------------------------ 標註工作表

def write_label_template(questions, runs, verdicts) -> Path:
    payload = []
    for q in questions:
        key = f"{q['group']}-{q['id']}"
        for cond in CONDITIONS:
            r = runs[key][cond]
            v = verdicts[key][cond]
            payload.append({
                "key": f"{key}-{cond}",
                "group": q["group"], "id": q["id"], "condition": cond,
                "question": q["question"],
                "expected_articles": q["expected"],
                "retrieved": sorted(set(r["article_nos"])),
                "queries": r["queries"],
                "answer": r["answer"],
                "verdict": "",        # ← 填 正確 / 不完整 / 錯誤
                "ruler_C": v.get("C", {}).get("verdict", ""),
                "ruler_B3": v.get("B3", {}).get("verdict", ""),
                "ruler_day15": v.get("day15", {}).get("verdict", ""),
                "note": "",
            })
    target = (LABELS_PATH if not LABELS_PATH.exists()
              else LABELS_PATH.parent / (LABELS_PATH.stem + ".new.json"))
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return target


def load_labels(questions) -> dict:
    if not LABELS_PATH.exists():
        raise SystemExit(f"找不到 {LABELS_PATH.name}。"
                         f"先跑：python {Path(__file__).name} --label")
    labels = {x["key"]: x.get("verdict", "").strip() for x in load_json(LABELS_PATH)}
    missing = [k for k, v in labels.items() if v not in judges.VERDICTS]
    if missing:
        raise SystemExit(f"還有 {len(missing)} 筆沒標：{'、'.join(missing[:8])}"
                         + ("…" if len(missing) > 8 else ""))
    return labels


# ------------------------------------------------------------------ 成本

def cost_twd(r: dict) -> float:
    return (rag_core.twd(rag_core.chat_cost_usd(r["input_tokens"], r["output_tokens"]))
            + rag_core.twd(rag_core.embedding_cost_usd(r["embedding_tokens"])))


def totals(runs, questions, group=None) -> dict:
    sel = [f"{q['group']}-{q['id']}" for q in questions
           if group is None or q["group"] == group]
    out = {}
    for cond in CONDITIONS:
        rows = [runs[k][cond] for k in sel]
        out[cond] = {
            "in": sum(r["input_tokens"] for r in rows),
            "out": sum(r["output_tokens"] for r in rows),
            "emb": sum(r["embedding_tokens"] for r in rows),
            "calls": sum(r["llm_calls"] for r in rows),
            "searches": sum(r["searches"] for r in rows),
            "cap": sum(1 for r in rows if r["hit_cap"]),
            "repeats": sum(r["repeats"] for r in rows),
            "twd": sum(cost_twd(r) for r in rows),
            "n": len(rows),
        }
    return out


# ------------------------------------------------------------------ 報表

def pct(n, d):
    return f"{n / d:.0%}" if d else "—"


def build_report(questions, runs, verdicts, labels) -> str:
    groups = ["單條文", "多條文", "陷阱"]
    out = [
        "# Day 20 實驗結果：會自己決定的迴圈 vs 寫死的管線",
        "",
        "- 題組：" + "、".join(
            f"{g} {sum(1 for q in questions if q['group'] == g)} 題" for g in groups)
        + f"，共 {len(questions)} 題 × 2 組 = {len(questions) * 2} 個答案",
        f"- 兩邊共用：同一份語料、同一個切法、同一個 embedding、"
        f"`{ag.MODEL}`、temperature=0、**每次檢索都 k={ag.SEARCH_K}**",
        f"- agent 的三道防線：最多 {ag.MAX_STEPS} 步、同 query 不重查、撞上限強制作答",
        "- 評估側完全凍結：尺 C 的 prompt、Day 19 的 rubric（B3）、Day 18 的三條標註原則",
        "- **基準是我自己讀完標的**，不是第三方（Day 18 起的慣例：標註來源要寫在正文）",
        "",
        "## 一、答對幾題",
        "",
        "| 題組 | 題數 | 對照組 | agent | 差 |",
        "|---|---|---|---|---|",
    ]
    for g in groups:
        sel = [f"{g}-{q['id']}" for q in questions if q["group"] == g]
        b = sum(1 for k in sel if labels[f"{k}-baseline"] == judges.CORRECT)
        a = sum(1 for k in sel if labels[f"{k}-agent"] == judges.CORRECT)
        out.append(f"| {g} | {len(sel)} | {b}/{len(sel)} | {a}/{len(sel)} | "
                   f"{a - b:+d} |")
    allb = sum(1 for q in questions
               if labels[f"{q['group']}-{q['id']}-baseline"] == judges.CORRECT)
    alla = sum(1 for q in questions
               if labels[f"{q['group']}-{q['id']}-agent"] == judges.CORRECT)
    out += [f"| **合計** | **{len(questions)}** | **{allb}/{len(questions)}** | "
            f"**{alla}/{len(questions)}** | **{alla - allb:+d}** |", "",
            "（判定來源是我的人工標註。尺 C 與 B3 的判定另外列在第五節，"
            "**兩把尺跟我不一致的地方逐筆攤開**。）", ""]

    # ------------------------------------------------------------- 成本
    out += ["## 二、花了幾倍的錢", "",
            "| 題組 | 對照組 | agent | 倍數 |", "|---|---|---|---|"]
    for g in groups + [None]:
        t = totals(runs, questions, g)
        name = g or "**合計**"
        ratio = t["agent"]["twd"] / t["baseline"]["twd"] if t["baseline"]["twd"] else 0
        out.append(f"| {name} | {t['baseline']['twd']:.4f} 元 | "
                   f"{t['agent']['twd']:.4f} 元 | **×{ratio:.1f}** |")
    out += ["", "拆開看錢花在哪裡（全部題目合計）：", "",
            "| 項目 | 對照組 | agent | 倍數 |", "|---|---|---|---|"]
    t = totals(runs, questions)
    for field, name in (("calls", "LLM 呼叫次數"), ("in", "輸入 token"),
                        ("out", "輸出 token"), ("emb", "embedding token"),
                        ("searches", "檢索次數")):
        b, a = t["baseline"][field], t["agent"][field]
        out.append(f"| {name} | {b} | {a} | ×{a / b:.1f} |" if b else
                   f"| {name} | {b} | {a} | — |")
    out += ["",
            f"⚠️ agent 的輸入 token 含**工具定義**：每一輪都要重送一次，"
            f"估 {ag.TOOLS_TOKEN_ESTIMATE} token／輪。"
            "OpenAI 怎麼把工具序列化成 token 沒有公開，**這個數字是估的不是量的**，"
            "所以它被算進上表、但在這裡標出來。", "",
            "agent 真正的成本來源是**迴圈每一輪都要把前面的對話重送一遍**："
            "查第三次的時候，前兩次撈回來的六條條文原文還在 context 裡，"
            "而它們每一輪都要重新計費。", ""]

    # ------------------------------------------------- agent 走了什麼路
    out += ["## 三、agent 走了什麼路", "",
            "| 題組 | 平均步數 | 平均 search 次數 | 撞 6 步上限 | 重複 query |",
            "|---|---|---|---|---|"]
    for g in groups:
        sel = [f"{g}-{q['id']}" for q in questions if q["group"] == g]
        rows = [runs[k]["agent"] for k in sel]
        out.append(f"| {g} | {sum(r['steps'] for r in rows) / len(rows):.1f} | "
                   f"{sum(r['searches'] for r in rows) / len(rows):.1f} | "
                   f"{sum(1 for r in rows if r['hit_cap'])}/{len(rows)} | "
                   f"{sum(r['repeats'] for r in rows)} |")
    dist = Counter(runs[f"{q['group']}-{q['id']}"]["agent"]["searches"]
                   for q in questions)
    out += ["", "search 次數的分布（全部 " + str(len(questions)) + " 題）：", "",
            "| 查幾次 | 幾題 |", "|---|---|"]
    for n in sorted(dist):
        out.append(f"| {n} | {dist[n]} |")
    out += ["",
            "### 它自己寫的 query 長什麼樣", "",
            "**這是今天最值得看的一欄。** 對照組永遠只有一個 query：使用者的原句。", "",
            "| 題組 | 題 | agent 用過的 query |", "|---|---|---|"]
    for q in questions:
        r = runs[f"{q['group']}-{q['id']}"]["agent"]
        if r["searches"] <= 1:
            continue
        out.append(f"| {q['group']} | {q['id']} | "
                   + " ／ ".join(f"`{x}`" for x in r["queries"]) + " |")
    out.append("")

    # --------------------------------------------- recall vs 答對
    out += ["## 四、撈得齊，不等於答得對", "",
            "Day 17 的老問題今天再問一次：必要條文進到模型手上了嗎？進來了就答得對嗎？", "",
            "| 題組 | 對照組 recall | agent recall | 對照組答對 | agent 答對 |",
            "|---|---|---|---|---|"]
    for g in ["單條文", "多條文"]:
        sel = [q for q in questions if q["group"] == g]
        cells = []
        for cond in CONDITIONS:
            rc = [union_recall(runs[f"{g}-{q['id']}"][cond]["article_nos"],
                               q["expected"]) for q in sel]
            rc = [x for x in rc if x is not None]
            cells.append(f"{sum(rc) / len(rc):.0%}" if rc else "—")
        ok = [sum(1 for q in sel
                  if labels[f"{g}-{q['id']}-{cond}"] == judges.CORRECT)
              for cond in CONDITIONS]
        out.append(f"| {g} | {cells[0]} | {cells[1]} | "
                   f"{ok[0]}/{len(sel)} | {ok[1]}/{len(sel)} |")
    out += ["", "逐題（只列兩邊 recall 不同的）：", "",
            "| 題組 | 題 | 必要條文 | 對照組撈到 | agent 撈到 | 對照組 | agent |",
            "|---|---|---|---|---|---|---|"]
    for q in questions:
        if not q["expected"]:
            continue
        key = f"{q['group']}-{q['id']}"
        rb = union_recall(runs[key]["baseline"]["article_nos"], q["expected"])
        ra = union_recall(runs[key]["agent"]["article_nos"], q["expected"])
        if rb == ra:
            continue
        out.append(f"| {q['group']} | {q['id']} | {q['expected']} | "
                   f"{pct(rb, 1)} | {pct(ra, 1)} | "
                   f"{labels[key + '-baseline']} | {labels[key + '-agent']} |")
    out.append("")

    # ------------------------------------------------------------ 陷阱題
    out += ["## 五、陷阱題：agent 多出來的那個逃生口", "",
            "規章根本沒寫這些。寫死的管線查不到就是查不到；"
            "**agent 有一個它沒有的選項：再查一次。**", "",
            "| 題 | 問題 | 對照組 | agent | agent 查幾次 | 撞上限 |",
            "|---|---|---|---|---|---|"]
    for q in [x for x in questions if x["group"] == "陷阱"]:
        key = f"陷阱-{q['id']}"
        r = runs[key]["agent"]
        out.append(f"| {q['id']} | {q['question']} | "
                   f"{labels[key + '-baseline']} | {labels[key + '-agent']} | "
                   f"{r['searches']} | {'⚠️ 是' if r['hit_cap'] else '否'} |")
    out += ["", "Day 15 那把關鍵詞尺（`grader.grade_trap`）今天說什麼——"
            "**它就是 Day 18 量出來只有 15/40 的那把舊尺**，"
            "列在這裡是要看它跟我的標註差多少：", "",
            "| 題 | 組 | 我的標註 | Day 15 舊尺 | 一致 |", "|---|---|---|---|---|"]
    for q in [x for x in questions if x["group"] == "陷阱"]:
        for cond in CONDITIONS:
            key = f"陷阱-{q['id']}-{cond}"
            old = verdicts[f"陷阱-{q['id']}"][cond]["day15"]["verdict"]
            mine = labels[key]
            # 舊尺的四種回傳值裡，只有 proper 代表「它認為這是好行為」。
            # 把它對到我的「正確」，其餘三種對到「不正確」，才比得下去。
            agree = (old == "proper") == (mine == judges.CORRECT)
            out.append(f"| {q['id']} | {COND_SHORT[cond]} | {mine} | `{old}` | "
                       f"{'✅' if agree else '❌ 舊尺錯'} |")
    out.append("")

    # ---------------------------------------------- 尺跟我不一致的地方
    out += ["## 六、兩把尺跟我不一致的地方", "",
            "Day 19 量出來：**兩把尺吵架的位置，就是該人看的位置。**"
            "今天就用它——下表是尺的判定與我的標註不同的每一筆。", "",
            "| 題組 | 題 | 組 | 我的標註 | 尺 C | 尺 B3 |", "|---|---|---|---|---|---|"]
    bad = 0
    for q in questions:
        if q["group"] == "陷阱":
            continue
        key = f"{q['group']}-{q['id']}"
        for cond in CONDITIONS:
            mine = labels[f"{key}-{cond}"]
            c = verdicts[key][cond].get("C", {}).get("verdict", "—")
            b3 = verdicts[key][cond].get("B3", {}).get("verdict", "—")
            if c == mine and (b3 in ("—", mine)):
                continue
            bad += 1
            out.append(f"| {q['group']} | {q['id']} | {COND_SHORT[cond]} | "
                       f"**{mine}** | {c} | {b3} |")
    judged = [(f"{q['group']}-{q['id']}", cond)
              for q in questions if q["group"] != "陷阱" for cond in CONDITIONS]
    c_same = sum(1 for key, cond in judged
                 if verdicts[key][cond]["C"]["verdict"] == labels[f"{key}-{cond}"])
    b3_pairs = [(key, cond) for key, cond in judged if "B3" in verdicts[key][cond]]
    b3_same = sum(1 for key, cond in b3_pairs
                  if verdicts[key][cond]["B3"]["verdict"] == labels[f"{key}-{cond}"])
    out += ["",
            f"- 尺 C 與我的標註一致：**{c_same}/{len(judged)}**",
            f"- 尺 B3 與我的標註一致：**{b3_same}/{len(b3_pairs)}**（只有多條文那組有 rubric）",
            f"- 上表列出的不一致：{bad} 筆", ""]

    # ------------------------------------------------------------ 成本總表
    cin = cout = bin_ = bout = 0
    for q in questions:
        key = f"{q['group']}-{q['id']}"
        for cond in CONDITIONS:
            cell = verdicts[key][cond]
            if "C" in cell:
                cin += cell["C"]["input_tokens"]
                cout += cell["C"]["output_tokens"]
            if "B3" in cell:
                bin_ += cell["B3"]["input_tokens"]
                bout += cell["B3"]["output_tokens"]
    c_twd = j19.cost_twd(j19.MINI, cin, cout)
    b_twd = j19.cost_twd(j19.MINI, bin_, bout)
    tt = totals(runs, questions)
    gen = tt["baseline"]["twd"] + tt["agent"]["twd"]
    out += ["## 七、今天的完整帳單", "",
            "**判定的錢也要算進來。** Day 18 的教訓是「免費的東西不會被檢討」，"
            "所以評估從 Day 18 起就是帳單上的一個項目，不能只報答案的成本。", "",
            "| 項目 | 成本（新台幣） |", "|---|---|",
            f"| 答案生成：對照組（{len(questions)} 題） | {tt['baseline']['twd']:.4f} |",
            f"| 答案生成：agent（{len(questions)} 題） | {tt['agent']['twd']:.4f} |",
            f"| 判定：尺 C（46 個答案） | {c_twd:.4f} |",
            f"| 判定：尺 B3（16 個答案） | {b_twd:.4f} |",
            f"| **合計（不含第八節的重跑）** | **{gen + c_twd + b_twd:.4f}** |",
            "",
            "⚠️ 這張表**不含**上限提示那一節的重跑（`compare_cap_nudge.py`，"
            "3 題、約 0.18 元）。連那一項的總額寫在該檔的報表裡。", "",
            "判定佔了三成——**量一批答案的成本，跟產生它們是同一個數量級。**", ""]

    # ------------------------------------------------------------ 逐題明細
    out += ["## 八、逐題明細", ""]
    for g in groups:
        out += [f"### {g}", ""]
        for q in [x for x in questions if x["group"] == g]:
            key = f"{q['group']}-{q['id']}"
            out += [f"**{q['id']}　{q['question']}**",
                    f"（必要條文：{q['expected'] or '無——規章沒寫'}）", ""]
            for cond in CONDITIONS:
                r = runs[key][cond]
                steps = " → ".join(
                    (f"search «{t['query']}»" + ("（重複）" if t["repeat"] else "")
                     if t["action"] == "search" else t["action"])
                    for t in r["trace"]) or "（寫死：原句查一次）"
                out += [f"- **{COND_LABEL[cond]}**　我的標註：**{labels[f'{key}-{cond}']}**",
                        f"  - 路徑：{steps}",
                        f"  - 撈到：第 {sorted(set(r['article_nos']))} 條"
                        f"　｜　{r['llm_calls']} 次 LLM｜{cost_twd(r):.4f} 元",
                        f"  - 答案：{r['answer'].replace(NL, ' ')}"]
            out.append("")
    return NL.join(out)


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description="Day 20：agent vs 寫死的管線")
    ap.add_argument("--label", action="store_true", help="只產出標註工作表")
    args = ap.parse_args()

    questions = load_questions()
    collection, chunks, articles, _, _ = pipeline.build_index()
    texts = d18.article_text_map(chunks, articles)

    print(f"[1/3] 跑兩邊（{len(questions)} 題 × 2）…")
    runs = run_both(collection, questions)

    print("[2/3] 判定（尺 C／B3／Day 15 舊尺）…")
    verdicts = judge_all(questions, runs, texts)

    if args.label:
        target = write_label_template(questions, runs, verdicts)
        print(f"{NL}標註工作表已寫入 {target.name}（{len(questions) * 2} 筆）。"
              f"{NL}逐字讀完，verdict 填 {'／'.join(judges.VERDICTS)} 之一。")
        return

    labels = load_labels(questions)
    print("[3/3] 產生報表…")
    RESULT_PATH.write_text(
        build_report(questions, runs, verdicts, labels), encoding="utf-8")
    print(f"{NL}報表已寫入 {RESULT_PATH.name}{NL}")

    for g in ["單條文", "多條文", "陷阱"]:
        sel = [f"{g}-{q['id']}" for q in questions if q["group"] == g]
        b = sum(1 for k in sel if labels[f"{k}-baseline"] == judges.CORRECT)
        a = sum(1 for k in sel if labels[f"{k}-agent"] == judges.CORRECT)
        t = totals(runs, questions, g)
        print(f"  {g:<5}｜對照 {b}/{len(sel)}　agent {a}/{len(sel)}"
              f"　｜錢 ×{t['agent']['twd'] / t['baseline']['twd']:.1f}"
              f"　｜撞上限 {t['agent']['cap']}/{len(sel)}")


if __name__ == "__main__":
    main()
