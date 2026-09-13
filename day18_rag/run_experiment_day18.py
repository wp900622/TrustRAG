# -*- coding: utf-8 -*-
"""Day 18 實驗：把判定換掉，然後驗判定本身。

Day 14 的關鍵事實比對在單條文題上撐了四天，Day 17 的多條文題 8 題判錯 5 題
（其中四個誤殺、一個放水，寫 day18_plan.md 時又補到第六筆）。今天把尺換掉。

但換尺有一個先天的困難：**你沒辦法用一把壞掉的尺去驗另一把尺。**
所以今天的基準不是另一把尺，是人工標註——`human_labels_day18.json`。

三個階段：

  一、三把尺量同一批答案（Day 17 的 40 個，8 題 × k=1/3/5/10/15）
      A 關鍵事實比對（免費）／B 逐項 rubric／C LLM 整體裁判
      全部拿去跟人工標註對帳，**誤殺與放水分開算**。

  二、裁判體檢四項（沒量過的裁判不准上線）
      3-1 自我一致性：同一批答案跑 N 遍，會不會改口
      3-2 長度偏誤：正確但極短 vs 正確但冗長
      3-3 位置偏誤：整體裁判 context 裡兩條條文對調
      3-4 負向對照：verbatim／terse／verbose／negated 四種手寫探針

  三、用新尺重判，把 Day 17 的招牌表重畫一次

凍結的東西：語料、切法、embedding、生成模型、temperature=0、prompt P1、
題組、k 值——**今天唯一的變因在判定側**，跟 Day 17 剛好垂直。
被評的 40 個答案全部在 chat_cache.json 裡，重建它們不計費。

⚠️ 3-1 要繞過快取。`rag_core.chat()` 的快取 key 是 model＋messages 的雜湊，
同一個 prompt 問五次會命中五次快取，一致率必然 100%——那是快取的一致性，
不是模型的。所以今天替 chat() 加了 use_cache=False，只給那一節用。

用法：
    python run_experiment_day18.py --label   # 產出人工標註工作表（第一次跑這個）
    python run_experiment_day18.py           # 標註完成後跑正式判定

結果寫入 experiment_result_day18.md。
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import chroma_store
import chunkers
import grader
import judges
import pipeline
import prompts
import rag_core

BASE_DIR = Path(__file__).parent
QUESTIONS_PATH = BASE_DIR / "questions_day17.json"      # 題組沿用，一字未改
RUBRIC_PATH = BASE_DIR / "rubric_day18.json"
PROBES_PATH = BASE_DIR / "probes_day18.json"
LABELS_PATH = BASE_DIR / "human_labels_day18.json"
RESULT_PATH = BASE_DIR / "experiment_result_day18.md"
NL = chr(10)

K_VALUES = (1, 3, 5, 10, 15)
CONSISTENCY_K = 3          # 一致性實驗只做這個 k 的 8 個答案，控制成本
CONSISTENCY_RUNS = 5
PROBE_KINDS = ("verbatim", "terse", "verbose", "negated")
PROBE_EXPECTED = {"verbatim": judges.CORRECT, "terse": judges.CORRECT,
                  "verbose": judges.CORRECT, "negated": judges.WRONG}

RULERS = ("A", "B", "C")
RULER_LABELS = {"A": "A 關鍵事實比對（Day 14 遺物）",
                "B": "B 逐項 rubric",
                "C": "C LLM 整體裁判"}


# ------------------------------------------------------------------ 資料準備

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def article_text_map(chunks: list[dict], articles: list[dict]) -> dict:
    """條號 → 條文原文（沿用 Day 17 的作法）"""
    mapping = {}
    for chunk in chunks:
        covered = chunkers.articles_covering(chunk, articles)
        if covered:
            mapping[covered[0]["no"]] = chunk["text"]
    return mapping


def rebuild_day17_answers(collection, questions: list[dict]) -> tuple[list[dict], int, int]:
    """重建 Day 17 的 40 個答案。

    不是從 experiment_result_day17.md 解析回來的——那樣會被報表的排版
    咬到。這裡原封不動再跑一次同樣的檢索與同樣的 messages，
    chat 快取全部命中，所以拿到的是位元級相同的答案，而且不計費。
    如果這一步出現非零的計費 token，代表 Day 17 之後有東西被改過，
    報表會把它印出來當作警告。
    """
    rows, billed_in, billed_out = [], 0, 0
    for k in K_VALUES:
        for q in questions:
            vectors, _, _ = rag_core.get_embeddings([q["question"]])
            hits = chroma_store.retrieve(collection, vectors[0], k)
            messages = prompts.build_messages(q["question"], hits)
            text, tin, tout = rag_core.chat(messages)
            billed_in, billed_out = billed_in + tin, billed_out + tout
            correct, missing, forbid_hit = grader.grade(
                text, q["facts"], q.get("forbid"))
            rows.append({"key": f"{q['id']}@k{k}", "id": q["id"], "k": k,
                         "answer": text, "facts_correct": correct,
                         "missing": missing, "forbid_hit": forbid_hit,
                         "context_nos": [h["meta"]["article_no"] for h in hits]})
    return rows, billed_in, billed_out


def write_label_template(rows: list[dict], questions: list[dict]) -> Path:
    """產出人工標註工作表。

    已存在就寫到 .new，絕不覆蓋——標註是今天最貴的產出（唯一一筆 API
    付不掉的成本），被腳本蓋掉會是本日最慘的事故。

    `human_labels_day18.seed.json` 若存在，其內容會填進 `verdict_suggested`
    與 `note` 兩欄**供覆核**，刻意不直接填進 `verdict`：
    標註是人的工作，讓一份現成答案躺在正式欄位裡，等於邀請自己按 Enter 過關。
    """
    qmap = {q["id"]: q for q in questions}
    seed_path = LABELS_PATH.parent / (LABELS_PATH.stem + ".seed.json")
    seed = {}
    if seed_path.exists():
        seed = {s["key"]: s for s in load_json(seed_path)}
    payload = []
    for r in rows:
        q = qmap[r["id"]]
        hint = seed.get(r["key"], {})
        payload.append({
            "key": r["key"],
            "question": q["question"],
            "expected_articles": q["expected"],
            "context_nos": r["context_nos"],
            "answer": r["answer"],
            "verdict": "",          # ← 填 正確 / 不完整 / 錯誤
            "verdict_suggested": hint.get("verdict", ""),
            "note": hint.get("reason", ""),
        })
    target = (LABELS_PATH if not LABELS_PATH.exists()
              else LABELS_PATH.parent / (LABELS_PATH.stem + ".new.json"))
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return target


def load_labels(rows: list[dict]) -> dict:
    """讀人工標註並檢查完整性；沒標完就直接停，不要跑出一半的表。"""
    if not LABELS_PATH.exists():
        raise SystemExit(
            f"找不到 {LABELS_PATH.name}。"
            f"{NL}請先跑：python {Path(__file__).name} --label"
            f"{NL}把 40 個答案逐字讀完、每筆填上 正確／不完整／錯誤，再跑一次本腳本。"
            f"{NL}（標註原則見 day18_plan.md 第二節。先寫原則再標，不准邊標邊改。）")
    labels = {item["key"]: item.get("verdict", "").strip()
              for item in load_json(LABELS_PATH)}
    missing = [r["key"] for r in rows
               if labels.get(r["key"]) not in judges.VERDICTS]
    if missing:
        raise SystemExit(
            f"還有 {len(missing)} 筆沒有標註或標註值不合法："
            f"{NL}  " + "、".join(missing[:12]) + ("…" if len(missing) > 12 else "")
            + f"{NL}合法值只有：{'／'.join(judges.VERDICTS)}")
    return labels


# ------------------------------------------------------------------ 三把尺

def articles_blob(q: dict, texts: dict) -> str:
    return NL.join(f"第 {no} 條{NL}{texts.get(no, '（原文缺漏）')}"
                   for no in q["expected"])


def run_rulers(rows, questions, rubrics, texts) -> dict:
    """三把尺各量一次 40 個答案。回傳 {key: {ruler: result}}"""
    qmap = {q["id"]: q for q in questions}
    rmap = {r["id"]: r["checkpoints"] for r in rubrics}
    out = {}
    for r in rows:
        q = qmap[r["id"]]
        out[r["key"]] = {
            "A": {"verdict": judges.grade_facts(r["facts_correct"]),
                  "input_tokens": 0, "output_tokens": 0, "calls": 0},
            "B": judges.grade_rubric(r["answer"], rmap[r["id"]]),
            "C": judges.grade_judge(q["question"], articles_blob(q, texts),
                                    r["answer"]),
        }
    return out


def agreement(rows, verdicts, labels, ruler) -> dict:
    """與人工標註對帳。

    誤殺與放水分開算，而且都以「人工標註」為準：

        誤殺  人判正確，尺判不是正確   → 會讓你去修沒壞的東西
        放水  人判不是正確，尺判正確   → 會讓你以為系統好了
        其他  兩邊都判「不對」，只是不對的種類不同（錯誤 ↔ 不完整）
              → 它不會誤導你的修法方向，所以跟上面兩種分開記

    parse_error 一律計入「其他不一致」並單獨計數，不准當成任何一種判定。
    """
    stat = Counter()
    cases = {"誤殺": [], "放水": [], "其他不一致": []}
    for r in rows:
        human, got = labels[r["key"]], verdicts[r["key"]][ruler]["verdict"]
        if got == judges.PARSE_ERROR:
            stat["parse_error"] += 1
        if got == human:
            stat["一致"] += 1
            continue
        if human == judges.CORRECT:
            stat["誤殺"] += 1
            cases["誤殺"].append((r["key"], human, got))
        elif got == judges.CORRECT:
            stat["放水"] += 1
            cases["放水"].append((r["key"], human, got))
        else:
            stat["其他不一致"] += 1
            cases["其他不一致"].append((r["key"], human, got))
    return {"stat": stat, "cases": cases, "total": len(rows)}


# -------------------------------------------------------- 裁判體檢：一致性

def consistency(rows, questions, rubrics, texts, runs: int) -> dict:
    """3-1：同一批答案跑 runs 遍，看判定會不會改口。**全程繞過快取。**"""
    qmap = {q["id"]: q for q in questions}
    rmap = {r["id"]: r["checkpoints"] for r in rubrics}
    subset = [r for r in rows if r["k"] == CONSISTENCY_K]
    out = {"B": {}, "C": {}, "tokens": {"B": [0, 0], "C": [0, 0]}}
    for r in subset:
        q = qmap[r["id"]]
        b_runs, c_runs = [], []
        for _ in range(runs):
            rb = judges.grade_rubric(r["answer"], rmap[r["id"]], use_cache=False)
            rc = judges.grade_judge(q["question"], articles_blob(q, texts),
                                    r["answer"], use_cache=False)
            b_runs.append(rb["verdict"])
            c_runs.append(rc["verdict"])
            out["tokens"]["B"][0] += rb["input_tokens"]
            out["tokens"]["B"][1] += rb["output_tokens"]
            out["tokens"]["C"][0] += rc["input_tokens"]
            out["tokens"]["C"][1] += rc["output_tokens"]
        out["B"][r["key"]] = b_runs
        out["C"][r["key"]] = c_runs
    return out


# ------------------------------------------- 裁判體檢：長度偏誤與負向對照

def run_probes(questions, rubrics, probes, texts) -> dict:
    """3-2 與 3-4 共用一批探針。

    verbatim 由條文原文即時拼出來，不寫在 probes_day18.json 裡——
    手抄條文進 JSON 遲早會抄錯一個字，而那一個字剛好會讓這節的結論反過來。

    四種變體的期望判定寫在 PROBE_EXPECTED：前三種都該判「正確」
    （照抄、極短、冗長，三種都沒說錯），negated 該判「錯誤」。
    Day 17 的自我檢查只做了 verbatim 那一列，所以 8 題全過，
    然後什麼都沒發現。
    """
    qmap = {q["id"]: q for q in questions}
    rmap = {r["id"]: r["checkpoints"] for r in rubrics}
    out = []
    for p in probes:
        q = qmap[p["id"]]
        blob = articles_blob(q, texts)
        for kind in PROBE_KINDS:
            answer = blob if kind == "verbatim" else p[kind]
            row = {"id": p["id"], "kind": kind, "answer": answer,
                   "expected": PROBE_EXPECTED[kind]}
            correct, _, _ = grader.grade(answer, q["facts"], q.get("forbid"))
            row["A"] = judges.grade_facts(correct)
            rb = judges.grade_rubric(answer, rmap[p["id"]])
            row["B"] = rb["verdict"]
            rc = judges.grade_judge(q["question"], blob, answer)
            row["C"] = rc["verdict"]
            row["C_reason"] = rc["reason"]
            row["tokens"] = (rb["input_tokens"] + rc["input_tokens"],
                             rb["output_tokens"] + rc["output_tokens"])
            out.append(row)
    return out


# ---------------------------------------------------- 裁判體檢：位置偏誤

def position_bias(rows, questions, texts) -> list[dict]:
    """3-3：整體裁判的 context 兩條條文對調，判定會不會變。

    只測尺 C。尺 B 的 checkpoint 根本不看條文原文（它只拿判斷點與答案），
    所以位置偏誤在設計上就進不來——**這本身就是把問題切小的一個好處，
    值得寫進文章**，不是因為懶得測。
    """
    qmap = {q["id"]: q for q in questions}
    out = []
    for r in [x for x in rows if x["k"] == CONSISTENCY_K]:
        q = qmap[r["id"]]
        forward = articles_blob(q, texts)
        reversed_q = {**q, "expected": list(reversed(q["expected"]))}
        backward = articles_blob(reversed_q, texts)
        a = judges.grade_judge(q["question"], forward, r["answer"])
        b = judges.grade_judge(q["question"], backward, r["answer"])
        out.append({"key": r["key"], "forward": a["verdict"],
                    "backward": b["verdict"],
                    "flipped": a["verdict"] != b["verdict"],
                    "tokens": (a["input_tokens"] + b["input_tokens"],
                               a["output_tokens"] + b["output_tokens"])})
    return out


# ---------------------------------------------------------------- 報表

DAY17_TABLE = {1: ("0%", "1/8"), 3: ("62%", "5/8"), 5: ("62%", "4/8"),
               10: ("75%", "5/8"), 15: ("88%", "3/8")}


def pct(n: int, total: int) -> str:
    return f"{n / total:.0%}" if total else "—"


def build_report(rows, questions, verdicts, labels, agree,
                 cons, probes_out, pos, billed, meta) -> str:
    n = len(rows)
    out = [
        "# Day 18 實驗結果：換掉那把尺，然後驗那把尺",
        "",
        f"- 被評的答案：Day 17 的 {n} 個（{len(questions)} 題 × k={K_VALUES}），一字未改",
        "- 生成側完全凍結：同樣的檢索、同樣的 messages，chat 快取全命中",
        f"- 判定模型：`{rag_core.CHAT_MODEL}`、temperature=0",
        "- **基準是人工標註，不是另一把尺**（`human_labels_day18.json`）",
        "",
    ]
    if billed[0] or billed[1]:
        out += [f"> ⚠️ 重建 Day 17 答案時出現計費 token（輸入 {billed[0]}／"
                f"輸出 {billed[1]}）。代表 Day 17 之後有東西被改過，"
                "這批答案與 Day 17 報表可能不是同一批，下面所有對照都要打折看。", ""]

    # ---------------------------------------------------------- 人工標註分布
    dist = Counter(labels[r["key"]] for r in rows)
    out += ["## 一、人工標註（今天唯一的基準）", "",
            "| 標註 | 筆數 | 占比 |", "|---|---|---|"]
    for v in judges.VERDICTS:
        out.append(f"| {v} | {dist[v]} | {pct(dist[v], n)} |")
    out += ["",
            "標註原則（先寫再標，不准邊標邊改）：",
            "1. 「沒說」不等於「說錯」",
            "2. 答反了就是錯，不管關鍵詞多齊全",
            "3. 問題沒問到的細節不列入要求",
            ""]

    # ---------------------------------------------------------- 三把尺對帳
    out += ["## 二、三把尺 vs 人工標註", "",
            "| 尺 | 一致 | 誤殺（人判對→尺判不對） | 放水（人判不對→尺判對） | 其他不一致 | 解析失敗 | 判定 LLM 呼叫 | 判定成本 |",
            "|---|---|---|---|---|---|---|---|"]
    for ruler in RULERS:
        a = agree[ruler]
        calls = sum(verdicts[r["key"]][ruler]["calls"] for r in rows)
        tin = sum(verdicts[r["key"]][ruler]["input_tokens"] for r in rows)
        tout = sum(verdicts[r["key"]][ruler]["output_tokens"] for r in rows)
        money = ("0 元" if not calls else
                 f"{rag_core.twd(rag_core.chat_cost_usd(tin, tout)):.5f} 元")
        out.append(
            f"| {RULER_LABELS[ruler]} | **{a['stat']['一致']}/{n}"
            f"（{pct(a['stat']['一致'], n)}）** | {a['stat']['誤殺']} | "
            f"{a['stat']['放水']} | {a['stat']['其他不一致']} | "
            f"{a['stat']['parse_error']} | {calls} | {money} |")
    out += ["",
            "⚠️ 誤殺與放水一定要分開看。一把誤殺多的尺讓你去修沒壞的東西；"
            "一把放水多的尺讓你以為系統好了。這兩種錯的代價完全不同，"
            "合成一個「準確率」就看不見了。", ""]

    for ruler in RULERS:
        cases = agree[ruler]["cases"]
        lines = []
        for kind in ("誤殺", "放水", "其他不一致"):
            for key, human, got in cases[kind]:
                lines.append(f"| {key} | {kind} | {human} | {got} |")
        if not lines:
            continue
        out += [f"### {RULER_LABELS[ruler]} 的每一筆不一致", "",
                "| 答案 | 類型 | 人工標註 | 尺判定 |", "|---|---|---|---|"] + lines + [""]

    # --------------------------------------------- Day 17 的六筆誤判翻對了嗎
    out += ["### Day 17 那六筆誤判，新尺翻對了嗎", "",
            "（清單見 `day18_plan.md` 第一節。翻不對就別談改善。）", "",
            "| 答案 | 人工標註 | A | B | C |", "|---|---|---|---|---|"]
    watchlist = ["304@k15", "305@k3", "306@k3", "307@k3", "308@k3", "303@k3"]
    for key in watchlist:
        if key not in labels:
            continue
        cells = " | ".join(
            ("✅ " if verdicts[key][r]["verdict"] == labels[key] else "❌ ")
            + verdicts[key][r]["verdict"] for r in RULERS)
        out.append(f"| {key} | {labels[key]} | {cells} |")
    out.append("")

    # ---------------------------------------------------------- 3-1 一致性
    runs_n = max((len(v) for v in cons["B"].values()), default=0)
    out += ["## 三、裁判體檢", "", f"### 3-1 自我一致性（k={CONSISTENCY_K} 的 "
            f"{len(cons['B'])} 個答案 × {runs_n} 遍，**繞過快取**）", "",
            f"| 尺 | {runs_n} 次全同 | 至少翻過一次 |", "|---|---|---|"]
    for ruler in ("B", "C"):
        runs = cons[ruler]
        same = sum(1 for v in runs.values() if len(set(v)) == 1)
        out.append(f"| {RULER_LABELS[ruler]} | {same}/{len(runs)} | "
                   f"{len(runs) - same}/{len(runs)} |")
    out += ["", "| 答案 | " + " | ".join(f"尺 {r} 的 {runs_n} 次判定"
                                        for r in ("B", "C")) + " |",
            "|---|---|---|"]
    for key in cons["B"]:
        out.append(f"| {key} | {'、'.join(cons['B'][key])} | "
                   f"{'、'.join(cons['C'][key])} |")
    out += ["",
            "temperature=0 不等於 deterministic。一把每跑一次就換答案的尺，"
            "你在它身上看到的所有「改善」都可能是噪音。", ""]

    # ------------------------------------------------ 3-2 / 3-4 探針
    out += [f"### 3-2 長度偏誤 ＋ 3-4 負向對照（{len(probes_out)} 支手寫探針）", "",
            "| 變體 | 內容 | 期望判定 | A 判對 | B 判對 | C 判對 |",
            "|---|---|---|---|---|---|"]
    desc = {"verbatim": "必要條文原文直接拼接", "terse": "一句話的正確答案",
            "verbose": "正確但冗長、夾雜無關條文",
            "negated": "與 terse 只差一個否定詞"}
    for kind in PROBE_KINDS:
        group = [p for p in probes_out if p["kind"] == kind]
        cells = []
        for ruler in RULERS:
            ok = sum(1 for p in group if p[ruler] == p["expected"])
            cells.append(f"{ok}/{len(group)}")
        out.append(f"| `{kind}` | {desc[kind]} | {PROBE_EXPECTED[kind]} | "
                   + " | ".join(cells) + " |")
    out += ["",
            "`verbatim` 那一列就是 Day 17 的自我檢查——它 8 題全過，"
            "然後什麼都沒發現。**只測「照抄原文會不會過」，等於只考自己會的那題。**"
            f"{NL}`negated` 那一列才是今天真正的考題：與正確答案只差一個否定詞，"
            "子字串比對看不見那個「不」字。", "",
            "#### 每一支被判錯的探針", "",
            "| 題號 | 變體 | 期望 | A | B | C | 裁判 C 的理由 |",
            "|---|---|---|---|---|---|---|"]
    bad = [p for p in probes_out
           if any(p[r] != p["expected"] for r in RULERS)]
    for p in bad:
        out.append(f"| {p['id']} | `{p['kind']}` | {p['expected']} | "
                   f"{p['A']} | {p['B']} | {p['C']} | "
                   f"{p['C_reason'].replace(NL, ' ')[:60]} |")
    if not bad:
        out.append("| — | — | — | — | — | — | 全部判對 |")
    out.append("")

    # ------------------------------------------------------- 3-3 位置偏誤
    flipped = sum(1 for p in pos if p["flipped"])
    out += [f"### 3-3 位置偏誤（只測尺 C，k={CONSISTENCY_K} 的 {len(pos)} 個答案）", "",
            f"把整體裁判 context 裡的兩條條文對調順序，**{flipped}/{len(pos)} "
            "個答案的判定跟著變**。", "",
            "| 答案 | 正序 | 反序 | 變了嗎 |", "|---|---|---|---|"]
    for p in pos:
        out.append(f"| {p['key']} | {p['forward']} | {p['backward']} | "
                   f"{'⚠️ 是' if p['flipped'] else '否'} |")
    out += ["",
            "尺 B 沒有這一項可測：checkpoint 根本不看條文原文（它只拿判斷點"
            "與答案），位置偏誤在設計上就進不來。**這是把問題切小的好處，"
            "不是我懶得測。**", ""]

    # --------------------------------------------------- 用新尺重算 Day 17
    out += ["## 四、用新尺重算 Day 17 的招牌表", "",
            "| k | 聯集 recall | 舊尺答對率（Day 17） | 人工標註 | 尺 B | 尺 C |",
            "|---|---|---|---|---|---|"]
    per_q = len(questions)
    for k in K_VALUES:
        group = [r for r in rows if r["k"] == k]
        human_ok = sum(1 for r in group if labels[r["key"]] == judges.CORRECT)
        b_ok = sum(1 for r in group
                   if verdicts[r["key"]]["B"]["verdict"] == judges.CORRECT)
        c_ok = sum(1 for r in group
                   if verdicts[r["key"]]["C"]["verdict"] == judges.CORRECT)
        recall, old = DAY17_TABLE[k]
        out.append(f"| {k} | {recall} | {old} | **{human_ok}/{per_q}** | "
                   f"{b_ok}/{per_q} | {c_ok}/{per_q} |")
    out += ["",
            "**Day 17 那個「k 開大反而掉」的反轉，用新尺還在不在？**"
            f"{NL}在 → Day 17 的結論站得住，只是數字不精確；"
            f"不在 → 要在文章裡把 Day 17 收回來。這一格不准挑好看的寫。", ""]

    # --------------------------------------------------------- 逐題明細
    out += ["## 五、逐題明細（答案原文 ＋ 三把尺的判定）", ""]
    for q in questions:
        rub = next(x for x in meta["rubrics"] if x["id"] == q["id"])
        out += [f"### {q['id']}　{q['question']}", "",
                f"- 必要條文：{'、'.join('第 ' + str(a) + ' 條' for a in q['expected'])}",
                f"- 舊尺的判定事實：{'、'.join(q['facts'])}（全部出現才算對）",
                "- 新尺的 checkpoint：", ""]
        for cp in rub["checkpoints"]:
            tag = "**required**" if cp["kind"] == "required" else "bonus"
            out.append(f"  - `{cp['id']}`（{tag}）{cp['point']}")
        out.append("")
        for k in K_VALUES:
            r = next(x for x in rows if x["id"] == q["id"] and x["k"] == k)
            v = verdicts[r["key"]]
            detail = "／".join(f"{d['id']}={d['label']}" for d in v["B"]["detail"])
            out += [f"**k={k}**　人工：{labels[r['key']]}　｜　"
                    f"A：{v['A']['verdict']}　B：{v['B']['verdict']}　"
                    f"C：{v['C']['verdict']}", "",
                    "> " + r["answer"].replace(NL, " "), "",
                    f"- rubric 逐項：{detail}",
                    f"- 裁判 C 的理由：{v['C']['reason'].replace(NL, ' ')}", ""]

    # ------------------------------------------------------------- 成本
    def token_sum(ruler):
        tin = sum(verdicts[r["key"]][ruler]["input_tokens"] for r in rows)
        tout = sum(verdicts[r["key"]][ruler]["output_tokens"] for r in rows)
        return tin, tout

    probe_in = sum(p["tokens"][0] for p in probes_out)
    probe_out_tok = sum(p["tokens"][1] for p in probes_out)
    pos_in = sum(p["tokens"][0] for p in pos)
    pos_out = sum(p["tokens"][1] for p in pos)
    cons_in = cons["tokens"]["B"][0] + cons["tokens"]["C"][0]
    cons_out = cons["tokens"]["B"][1] + cons["tokens"]["C"][1]
    b_in, b_out = token_sum("B")
    c_in, c_out = token_sum("C")

    sections = [
        ("正式判定 尺 B（rubric）", b_in, b_out),
        ("正式判定 尺 C（整體裁判）", c_in, c_out),
        ("3-1 自我一致性（繞過快取，這一項每次跑都全額計費）", cons_in, cons_out),
        ("3-2＋3-4 探針", probe_in, probe_out_tok),
        ("3-3 位置偏誤", pos_in, pos_out),
    ]
    total_in = sum(s[1] for s in sections)
    total_out = sum(s[2] for s in sections)
    out += ["## 六、成本：評估從今天起是帳單上的一個項目", "",
            "token 一律離線重數（`judges.count_messages()`，沿用 Day 16 的作法），"
            "不依賴 API 回報的 usage——快取全命中時 usage 是 0，"
            "只看那個數字會以為評估不用錢。下表是**這批判定實際要花多少**，"
            "重跑時因為快取，實際刷卡金額會更小甚至為 0。", "",
            "| 項目 | 輸入 tokens | 輸出 tokens | 成本（新台幣） |",
            "|---|---|---|---|"]
    for name, tin, tout in sections:
        out.append(f"| {name} | {tin} | {tout} | "
                   f"{rag_core.twd(rag_core.chat_cost_usd(tin, tout)):.5f} |")
    out += [f"| **合計** | **{total_in}** | **{total_out}** | "
            f"**{rag_core.twd(rag_core.chat_cost_usd(total_in, total_out)):.5f}** |",
            "",
            "- 尺 A 是 0 元（純字串比對）。Day 14~17 的評估成本一直是 0，"
            "所以我從來沒把它當成一個項目——**免費的東西不會被檢討**。",
            "- 被評的 40 個答案本身重建不計費（chat 快取全命中）；"
            "生成側的帳在 Day 17 已經付過了。",
            "- 人工標註 40 個答案的時間沒有出現在這張表上，"
            "而那是今天唯一一筆 API 付不掉的成本。",
            ""]
    return NL.join(out)


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Day 18：換尺，並驗尺")
    parser.add_argument("--label", action="store_true",
                        help="只產出人工標註工作表，不跑判定")
    parser.add_argument("--runs", type=int, default=CONSISTENCY_RUNS,
                        help=f"一致性實驗跑幾遍（預設 {CONSISTENCY_RUNS}）")
    args = parser.parse_args()

    questions = load_json(QUESTIONS_PATH)
    rubrics = load_json(RUBRIC_PATH)
    probes = load_json(PROBES_PATH)
    collection, chunks, articles, _, _ = pipeline.build_index()
    texts = article_text_map(chunks, articles)

    print(f"[0/5] 重建 Day 17 的 {len(K_VALUES) * len(questions)} 個答案"
          "（走快取，應為 0 元）…")
    rows, billed_in, billed_out = rebuild_day17_answers(collection, questions)
    if billed_in or billed_out:
        print(f"   ⚠️ 出現計費 token（{billed_in}／{billed_out}）："
              "Day 17 之後有東西被改過，對照要打折看。")

    if args.label:
        target = write_label_template(rows, questions)
        print(f"{NL}人工標註工作表已寫入 {target.name}（{len(rows)} 筆）。"
              f"{NL}逐字讀完每個答案，把 verdict 填成 "
              f"{'／'.join(judges.VERDICTS)} 之一，再跑一次本腳本。"
              f"{NL}標註原則見 day18_plan.md 第二節——先寫原則再標。")
        return

    labels = load_labels(rows)

    print(f"[1/5] 三把尺量 {len(rows)} 個答案…")
    verdicts = run_rulers(rows, questions, rubrics, texts)
    agree = {r: agreement(rows, verdicts, labels, r) for r in RULERS}

    print(f"[2/5] 3-1 自我一致性（k={CONSISTENCY_K} × {args.runs} 遍，繞過快取）…")
    cons = consistency(rows, questions, rubrics, texts, args.runs)

    print(f"[3/5] 3-2＋3-4 探針（{len(probes) * len(PROBE_KINDS)} 支）…")
    probes_out = run_probes(questions, rubrics, probes, texts)

    print("[4/5] 3-3 位置偏誤（只測尺 C）…")
    pos = position_bias(rows, questions, texts)

    print("[5/5] 產生報表…")
    RESULT_PATH.write_text(
        build_report(rows, questions, verdicts, labels, agree, cons,
                     probes_out, pos, (billed_in, billed_out),
                     {"rubrics": rubrics}),
        encoding="utf-8")
    print(f"{NL}報表已寫入 {RESULT_PATH.name}{NL}")

    n = len(rows)
    for ruler in RULERS:
        s = agree[ruler]["stat"]
        print(f"  {RULER_LABELS[ruler]:<22}｜一致 {s['一致']:>2}/{n}"
              f"｜誤殺 {s['誤殺']:>2}｜放水 {s['放水']:>2}")
    for ruler in ("B", "C"):
        runs = cons[ruler]
        same = sum(1 for v in runs.values() if len(set(v)) == 1)
        print(f"  尺 {ruler} 自我一致｜{same}/{len(runs)} 個答案 "
              f"{args.runs} 次全同")
    for kind in PROBE_KINDS:
        group = [p for p in probes_out if p["kind"] == kind]
        cells = "｜".join(
            f"{r}={sum(1 for p in group if p[r] == p['expected'])}/{len(group)}"
            for r in RULERS)
        print(f"  探針 {kind:<9}｜{cells}")


if __name__ == "__main__":
    main()
