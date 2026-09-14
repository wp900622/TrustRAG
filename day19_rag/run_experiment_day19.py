# -*- coding: utf-8 -*-
"""Day 19 實驗：把 Day 18 指認的兩個病灶修掉，並接第二個模型當裁判。

Day 18 文末寫死了今天的題目，驗收標準也一起寫死了：

    修得動就留下來，修完還是輸給 C（33/40），rubric 這條路我就認輸。

四個階段：

  一、B 的 ablation（B0／B1／B2／B3）
      只改 checkpoint 判定的 prompt，rubric JSON 一個字不動。
      **四個變體分開跑**——一次改兩個地方然後說有效，
      是沒有辦法知道哪一個有效的。

  二、探針回歸（32 支手寫探針 × 四個變體）
      放鬆「相反」最直接的風險是誤殺換放水。
      誤殺少幾筆、negated 掉幾支，兩個數字並排寫。

  三、換模型（B3′ 與 C′ 都用 gpt-4o）
      兩把尺都換，才分得出「prompt 修好了」與「模型比較強」。

  四、self-preference 2×2
      只換生成模型（k=3、檢索凍結），產出第二組 8 個答案，
      兩個裁判各判兩組，看差中差 (a−b)−(c−d)。

凍結：語料、切法、embedding、檢索、k 值、題組、P1 prompt、rubric JSON、
Day 17 的那 40 個答案。今天的變因只有 checkpoint 的 prompt 與判定的模型。

用法：
    python run_experiment_day19.py --gen   # 產生 2×2 用的 gpt-4o 答案與標註工作表
    python run_experiment_day19.py         # 標註完成後跑正式實驗
"""
import argparse
import json
from pathlib import Path

import chroma_store
import grader
import judges
import judges_day19 as j19
import pipeline
import prompts
import rag_core
import run_experiment_day18 as d18

BASE_DIR = Path(__file__).parent
QUESTIONS_PATH = BASE_DIR / "questions_day17.json"
RUBRIC_PATH = BASE_DIR / "rubric_day18.json"        # 一個字不動
PROBES_PATH = BASE_DIR / "probes_day18.json"        # 一個字不動
LABELS_PATH = BASE_DIR / "human_labels_day18.json"  # 昨天的基準，沿用
ALT_PATH = BASE_DIR / "answers_day19_gpt4o.json"    # 2×2 用的第二組答案＋標註
RESULT_PATH = BASE_DIR / "experiment_result_day19.md"
NL = chr(10)

K_VALUES = d18.K_VALUES
SELF_PREF_K = 3          # 2×2 只做 k=3，與 Day 18 的裁判體檢同一格
CONS_RUNS_MINI = 5       # 與 Day 18 的 8/8 可直接比
CONS_RUNS_4O = 3         # gpt-4o 每次判定貴一個數量級，次數減下來

# (代號, 標題, 種類, prompt 變體, 模型)
RULER_SPECS = [
    ("B0", "B0 逐項 rubric（Day 18 原版）", "rubric", "B0", j19.MINI),
    ("B1", "B1 只修否定極性（F1）", "rubric", "B1", j19.MINI),
    ("B2", "B2 只修例外條款（F2）", "rubric", "B2", j19.MINI),
    ("B3", "B3 兩個都修（F1＋F2）", "rubric", "B3", j19.MINI),
    ("B3o", "B3′ 兩個都修＋換 gpt-4o", "rubric", "B3", j19.GPT4O),
    ("C", "C LLM 整體裁判（Day 18 原版）", "judge", None, j19.MINI),
    ("Co", "C′ 整體裁判＋換 gpt-4o", "judge", None, j19.GPT4O),
]
RULERS = [s[0] for s in RULER_SPECS]
LABELS = {s[0]: s[1] for s in RULER_SPECS}
SPEC = {s[0]: s for s in RULER_SPECS}

# Day 18 的 9 筆誤殺，今天的主要驗收對象。清單寫死在這裡，
# 免得等一下用「修完之後還誤殺哪幾筆」反推出一份比較好看的名單。
DAY18_B_MISKILLS = ["305@k1", "305@k3", "305@k5", "305@k10", "305@k15",
                    "307@k3", "307@k5", "307@k10", "307@k15"]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ 判定

def grade_one(ruler: str, q: dict, answer: str, checkpoints: list[dict],
              texts: dict, use_cache: bool = True) -> dict:
    _, _, kind, variant, model = SPEC[ruler]
    if kind == "rubric":
        return j19.grade_rubric(answer, checkpoints, variant=variant,
                                model=model, use_cache=use_cache)
    return j19.grade_judge(q["question"], d18.articles_blob(q, texts), answer,
                           model=model, use_cache=use_cache)


def run_rulers(rows, questions, rubrics, texts) -> dict:
    qmap = {q["id"]: q for q in questions}
    rmap = {r["id"]: r["checkpoints"] for r in rubrics}
    out = {}
    for r in rows:
        q = qmap[r["id"]]
        out[r["key"]] = {ruler: grade_one(ruler, q, r["answer"],
                                          rmap[r["id"]], texts)
                         for ruler in RULERS}
    return out


# ----------------------------------------------------------- 探針回歸

def run_probes(questions, rubrics, probes, texts, rulers) -> list[dict]:
    qmap = {q["id"]: q for q in questions}
    rmap = {r["id"]: r["checkpoints"] for r in rubrics}
    out = []
    for p in probes:
        q = qmap[p["id"]]
        blob = d18.articles_blob(q, texts)
        for kind in d18.PROBE_KINDS:
            answer = blob if kind == "verbatim" else p[kind]
            row = {"id": p["id"], "kind": kind, "answer": answer,
                   "expected": d18.PROBE_EXPECTED[kind], "tokens": {}}
            correct, _, _ = grader.grade(answer, q["facts"], q.get("forbid"))
            row["A"] = judges.grade_facts(correct)
            for ruler in rulers:
                res = grade_one(ruler, q, answer, rmap[p["id"]], texts)
                row[ruler] = res["verdict"]
                row["tokens"][ruler] = (res["input_tokens"], res["output_tokens"])
                if ruler == "Co":
                    row["Co_reason"] = res["reason"]
            out.append(row)
    return out


# ------------------------------------------------------------ 自我一致性

def consistency(rows, questions, rubrics, texts, plan) -> dict:
    """plan: {ruler: runs}。全程繞過快取，否則量到的是快取的一致性。"""
    qmap = {q["id"]: q for q in questions}
    rmap = {r["id"]: r["checkpoints"] for r in rubrics}
    subset = [r for r in rows if r["k"] == SELF_PREF_K]
    out = {"runs": {}, "tokens": {}}
    for ruler, runs in plan.items():
        out["runs"][ruler] = {}
        out["tokens"][ruler] = [0, 0]
        for r in subset:
            q = qmap[r["id"]]
            seq = []
            for _ in range(runs):
                res = grade_one(ruler, q, r["answer"], rmap[r["id"]], texts,
                                use_cache=False)
                seq.append(res["verdict"])
                out["tokens"][ruler][0] += res["input_tokens"]
                out["tokens"][ruler][1] += res["output_tokens"]
            out["runs"][ruler][r["key"]] = seq
    return out


# --------------------------------------------------- 2×2：第二組答案

def generate_alt_answers(collection, questions) -> tuple[list[dict], int, int]:
    """只換生成模型，檢索與 prompt 完全凍結。

    用的是同一個 `prompts.build_messages(..., "P1")`、同一批 top-3 條文，
    所以兩組答案之間唯一的差別就是寫答案的模型。
    """
    rows, tin, tout = [], 0, 0
    for q in questions:
        vectors, _, _ = rag_core.get_embeddings([q["question"]])
        hits = chroma_store.retrieve(collection, vectors[0], SELF_PREF_K)
        messages = prompts.build_messages(q["question"], hits)
        text = j19.chat(messages, model=j19.GPT4O)
        tin += judges.count_messages(messages)
        tout += judges.count_text(text)
        rows.append({"key": f"{q['id']}@k{SELF_PREF_K}", "id": q["id"],
                     "k": SELF_PREF_K, "answer": text})
    return rows, tin, tout


def write_alt_template(alt_rows, questions) -> Path:
    """寫出第二組答案的標註工作表。已存在就寫到 .new，絕不覆蓋。"""
    qmap = {q["id"]: q for q in questions}
    payload = [{"key": r["key"], "question": qmap[r["id"]]["question"],
                "expected_articles": qmap[r["id"]]["expected"],
                "answer": r["answer"], "verdict": "", "note": ""}
               for r in alt_rows]
    target = (ALT_PATH if not ALT_PATH.exists()
              else ALT_PATH.parent / (ALT_PATH.stem + ".new.json"))
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return target


def self_preference(rows, alt_rows, questions, rubrics, texts) -> dict:
    """2×2：考生 {mini, 4o} × 裁判 {mini, 4o}，尺 C，k=3 的 8 題。

    指標是「判正確的比例」（leniency），不是準確率——自我偏好問的是
    「誰對誰比較寬」，不是「誰比較準」。準確率另外對帳，兩件事分開寫。
    """
    qmap = {q["id"]: q for q in questions}
    rmap = {r["id"]: r["checkpoints"] for r in rubrics}
    mini_rows = [r for r in rows if r["k"] == SELF_PREF_K]
    cells, detail = {}, []
    for exam_name, group in (("mini", mini_rows), ("4o", alt_rows)):
        for judge_name, ruler in (("mini", "C"), ("4o", "Co")):
            verdicts = []
            for r in group:
                q = qmap[r["id"]]
                res = grade_one(ruler, q, r["answer"], rmap[r["id"]], texts)
                verdicts.append(res["verdict"])
                detail.append({"exam": exam_name, "judge": judge_name,
                               "key": r["key"], "verdict": res["verdict"],
                               "reason": res.get("reason", ""),
                               "tokens": (res["input_tokens"],
                                          res["output_tokens"])})
            cells[(exam_name, judge_name)] = verdicts
    return {"cells": cells, "detail": detail, "n": len(mini_rows)}


def lenient(verdicts: list[str]) -> int:
    return sum(1 for v in verdicts if v == judges.CORRECT)


# ---------------------------------------------------------------- 報表

def pct(n: int, total: int) -> str:
    return f"{n / total:.0%}" if total else "—"


def ruler_cost(verdicts, rows, ruler) -> tuple[int, int, int, float]:
    calls = sum(verdicts[r["key"]][ruler]["calls"] for r in rows)
    tin = sum(verdicts[r["key"]][ruler]["input_tokens"] for r in rows)
    tout = sum(verdicts[r["key"]][ruler]["output_tokens"] for r in rows)
    return calls, tin, tout, j19.cost_twd(SPEC[ruler][4], tin, tout)


def build_report(rows, questions, rubrics, verdicts, labels, agree, probes_out,
                 cons, sp, alt_rows, alt_labels, gen_tokens, billed) -> str:
    n = len(rows)
    out = [
        "# Day 19 實驗結果：修好的尺，以及第二個裁判",
        "",
        f"- 被評的答案：Day 17 的 {n} 個（{len(questions)} 題 × k={K_VALUES}），一字未改",
        "- 基準：`human_labels_day18.json`（Day 18 的標註，今天沒有重標）",
        "- rubric 的 JSON 一個字不動；**今天的變因只有 checkpoint 的 prompt 與判定模型**",
        f"- 驗收標準（Day 18 文末寫死的）：B 修完若還是輸給 C 的 "
        f"{agree['C']['stat']['一致']}/{n}，rubric 認輸",
        "",
    ]
    if billed[0] or billed[1]:
        out += [f"> ⚠️ 重建 Day 17 答案時出現計費 token（{billed[0]}／{billed[1]}），"
                "代表生成側有東西被改過，下面所有對照都要打折看。", ""]

    # ------------------------------------------------- 一、兩個病灶的實際分布
    out += ["## 一、Day 18 那 9 筆誤殺，死在哪一個 checkpoint", "",
            "| 答案 | 基準 | B0 判定 | 被判「相反」的 checkpoint |",
            "|---|---|---|---|"]
    for key in DAY18_B_MISKILLS:
        d = verdicts[key]["B0"]
        bad = "、".join(f"`{x['id']}`（{x['kind']}）"
                        for x in d["detail"] if x["label"] == judges.CONTRADICT)
        out.append(f"| {key} | {labels[key]} | {d['verdict']} | {bad or '—'} |")
    out.append("")

    # ------------------------------------------------------ 二、ablation
    out += ["## 二、四個變體 vs 同一個基準", "",
            "| 尺 | 一致 | 誤殺 | 放水 | 其他不一致 | 解析失敗 | LLM 呼叫 | 成本 |",
            "|---|---|---|---|---|---|---|---|"]
    for ruler in RULERS:
        a = agree[ruler]["stat"]
        calls, tin, tout, money = ruler_cost(verdicts, rows, ruler)
        out.append(f"| {LABELS[ruler]} | **{a['一致']}/{n}（{pct(a['一致'], n)}）** | "
                   f"{a['誤殺']} | {a['放水']} | {a['其他不一致']} | "
                   f"{a['parse_error']} | {calls} | {money:.5f} 元 |")
    bs = [r for r in RULERS if r.startswith("B")]
    out += ["",
            "### 那 9 筆誤殺，各變體救回幾筆", "",
            "| 答案 | 基準 | " + " | ".join(bs) + " |",
            "|---" * (2 + len(bs)) + "|"]
    for key in DAY18_B_MISKILLS:
        cells = []
        for ruler in bs:
            v = verdicts[key][ruler]["verdict"]
            cells.append(("✅ " if v == labels[key] else "❌ ") + v)
        out.append(f"| {key} | {labels[key]} | " + " | ".join(cells) + " |")
    saved = {ruler: sum(1 for k in DAY18_B_MISKILLS
                        if verdicts[k][ruler]["verdict"] == labels[k])
             for ruler in bs}
    out += ["| **救回** | — | " + " | ".join(
        f"**{saved[r]}/{len(DAY18_B_MISKILLS)}**" for r in bs) + " |", ""]

    for ruler in RULERS:
        cases = agree[ruler]["cases"]
        lines = [f"| {key} | {kind} | {human} | {got} |"
                 for kind in ("誤殺", "放水", "其他不一致")
                 for key, human, got in cases[kind]]
        if not lines:
            out += [f"### {LABELS[ruler]}：與基準完全一致", ""]
            continue
        out += [f"### {LABELS[ruler]} 的每一筆不一致", "",
                "| 答案 | 類型 | 基準 | 尺判定 |", "|---|---|---|---|"] + lines + [""]

    # -------------------------------------------------------- 三、探針回歸
    probe_rulers = [r for r in RULERS if r in probes_out[0]]
    out += [f"## 三、探針回歸（{len(probes_out)} 支手寫探針，Day 18 那批，一字未改）", "",
            "放鬆「相反」的判定，最直接的風險是誤殺換放水。"
            "`negated` 那一列是與正確答案只差一個否定詞的假答案——"
            "**它掉幾支，就是這次修法的價錢。**", "",
            "| 變體 | 期望判定 | A | " + " | ".join(probe_rulers) + " |",
            "|---" * (3 + len(probe_rulers)) + "|"]
    for kind in d18.PROBE_KINDS:
        group = [p for p in probes_out if p["kind"] == kind]
        a_ok = sum(1 for p in group if p["A"] == p["expected"])
        cells = [f"{sum(1 for p in group if p[r] == p['expected'])}/{len(group)}"
                 for r in probe_rulers]
        out.append(f"| `{kind}` | {d18.PROBE_EXPECTED[kind]} | {a_ok}/{len(group)} | "
                   + " | ".join(cells) + " |")
    out += ["", "#### `negated` 逐支（誰放水了）", "",
            "| 題號 | 期望 | " + " | ".join(probe_rulers) + " |",
            "|---" * (2 + len(probe_rulers)) + "|"]
    for p in [x for x in probes_out if x["kind"] == "negated"]:
        cells = [("✅ " if p[r] == p["expected"] else "❌ ") + p[r]
                 for r in probe_rulers]
        out.append(f"| {p['id']} | {p['expected']} | " + " | ".join(cells) + " |")
    out.append("")

    # ---------------------------------------------------- 四、自我一致性
    out += ["## 四、自我一致性（繞過快取）", "",
            "| 尺 | 跑幾遍 | 全同 | 至少翻過一次 |", "|---|---|---|---|"]
    for ruler, runs in cons["runs"].items():
        n_runs = max(len(v) for v in runs.values())
        same = sum(1 for v in runs.values() if len(set(v)) == 1)
        out.append(f"| {LABELS[ruler]} | {n_runs} | {same}/{len(runs)} | "
                   f"{len(runs) - same}/{len(runs)} |")
    out += ["", "| 答案 | " + " | ".join(f"{r} 的逐次判定" for r in cons["runs"]) + " |",
            "|---" * (1 + len(cons["runs"])) + "|"]
    for key in next(iter(cons["runs"].values())):
        out.append(f"| {key} | " + " | ".join(
            "、".join(cons["runs"][r][key]) for r in cons["runs"]) + " |")
    out.append("")

    # ------------------------------------------------- 五、self-preference
    cells = sp["cells"]
    a = lenient(cells[("mini", "mini")])
    b = lenient(cells[("4o", "mini")])
    c = lenient(cells[("mini", "4o")])
    d = lenient(cells[("4o", "4o")])
    m = sp["n"]
    out += [f"## 五、self-preference 2×2（尺 C，k={SELF_PREF_K} 的 {m} 題）", "",
            "檢索凍結、prompt 凍結，只換生成模型產出第二組答案。"
            "格子裡是**判「正確」的筆數**——這一節問的是「誰對誰比較寬」，"
            "不是「誰比較準」。", "",
            "| 判「正確」 | 考生 `gpt-4o-mini` | 考生 `gpt-4o` | 對自己 − 對別人 |",
            "|---|---|---|---|",
            f"| 裁判 `gpt-4o-mini` | **{a}/{m}** | {b}/{m} | {(a - b):+d} |",
            f"| 裁判 `gpt-4o` | {c}/{m} | **{d}/{m}** | {(d - c):+d} |",
            "",
            f"**差中差 (a−b) − (c−d) = {(a - b) - (c - d):+d}／{m}**",
            "",
            "兩個裁判都覺得某一組答案比較好（因為它真的比較好），"
            "這個差會互相抵消；只有「偏袒自己人」才會留下殘值。"
            f"{NL}⚠️ 每格 n={m}，小到不能宣稱統計顯著。這句話寫在這裡，不寫在註腳。", "",
            "### 第二組答案（`gpt-4o` 生成）逐筆", "",
            "| 答案 | 我的標註 | 裁判 mini | 裁判 4o |", "|---|---|---|---|"]
    by = {(x["exam"], x["judge"], x["key"]): x for x in sp["detail"]}
    for r in alt_rows:
        out.append(f"| {r['key']} | {alt_labels.get(r['key'], '—')} | "
                   f"{by[('4o', 'mini', r['key'])]['verdict']} | "
                   f"{by[('4o', '4o', r['key'])]['verdict']} |")
    acc = {j: sum(1 for r in alt_rows
                  if by[("4o", j, r["key"])]["verdict"] == alt_labels.get(r["key"]))
           for j in ("mini", "4o")}
    out += ["",
            f"順帶一筆準確率（對第二組答案）：裁判 mini {acc['mini']}/{m}、"
            f"裁判 4o {acc['4o']}/{m}。"
            f"{NL}**這 8 筆標註是我標的，不是第三方**——"
            "Day 18 已經因為「基準不是純人工的」修正過一次，同一個坑不踩第二次。", "",
            "### 第二組答案原文", ""]
    for r in alt_rows:
        out += [f"**{r['key']}**　（我的標註：{alt_labels.get(r['key'], '—')}）", "",
                "> " + r["answer"].replace(NL, " "), ""]

    # --------------------------------------------------- 六、重算 Day 17
    keep = ["B0", "B3", "B3o", "C", "Co"]
    out += ["## 六、再算一次 Day 17 的招牌表", "",
            "| k | 聯集 recall | 基準 | " + " | ".join(keep) + " |",
            "|---" * (3 + len(keep)) + "|"]
    per_q = len(questions)
    for k in K_VALUES:
        group = [r for r in rows if r["k"] == k]
        human_ok = sum(1 for r in group if labels[r["key"]] == judges.CORRECT)
        cells_k = [str(sum(1 for r in group
                           if verdicts[r["key"]][ruler]["verdict"] == judges.CORRECT))
                   + f"/{per_q}" for ruler in keep]
        out.append(f"| {k} | {d18.DAY17_TABLE[k][0]} | **{human_ok}/{per_q}** | "
                   + " | ".join(cells_k) + " |")
    out.append("")

    # ------------------------------------------------------- 七、逐題明細
    out += ["## 七、逐題明細（只列 B0 與 B3 判定不同的答案）", ""]
    qmap = {q["id"]: q for q in questions}
    rmap = {r["id"]: r["checkpoints"] for r in rubrics}
    changed = [r for r in rows
               if verdicts[r["key"]]["B0"]["verdict"]
               != verdicts[r["key"]]["B3"]["verdict"]]
    if not changed:
        out += ["（沒有任何一筆改變——修法完全沒有作用。）", ""]
    for r in changed:
        v = verdicts[r["key"]]
        cp = {x["id"]: x for x in rmap[r["id"]]}
        out += [f"### {r['key']}　{qmap[r['id']]['question']}", "",
                f"基準：**{labels[r['key']]}**　｜　B0：{v['B0']['verdict']}　→　"
                f"B3：{v['B3']['verdict']}　｜　B3′(4o)：{v['B3o']['verdict']}　"
                f"｜　C：{v['C']['verdict']}　C′(4o)：{v['Co']['verdict']}", "",
                "> " + r["answer"].replace(NL, " "), "",
                "| checkpoint | 種類 | B0 | B1 | B2 | B3 | B3′(4o) |",
                "|---|---|---|---|---|---|---|"]
        for i, x in enumerate(v["B0"]["detail"]):
            out.append(f"| `{x['id']}` {cp[x['id']]['point'][:26]}… | {x['kind']} | "
                       f"{x['label']} | {v['B1']['detail'][i]['label']} | "
                       f"{v['B2']['detail'][i]['label']} | "
                       f"{v['B3']['detail'][i]['label']} | "
                       f"{v['B3o']['detail'][i]['label']} |")
        out.append("")

    # ------------------------------------------------------------- 八、成本
    out += ["## 八、成本", "",
            "token 一律離線重數（`judges.count_messages()`），不依賴 API 回報的 usage。"
            "下表是**這批判定實際要花多少**，重跑時因為快取，刷卡金額會更小。", "",
            "| 項目 | 模型 | 輸入 tokens | 輸出 tokens | 成本（新台幣） |",
            "|---|---|---|---|---|"]
    sections = []
    for ruler in RULERS:
        calls, tin, tout, _ = ruler_cost(verdicts, rows, ruler)
        sections.append((f"正式判定 {ruler}（{n} 個答案，{calls} 次呼叫）",
                         SPEC[ruler][4], tin, tout))
    for ruler in probe_rulers:
        tin = sum(p["tokens"][ruler][0] for p in probes_out)
        tout = sum(p["tokens"][ruler][1] for p in probes_out)
        sections.append((f"探針回歸 {ruler}", SPEC[ruler][4], tin, tout))
    for ruler, (tin, tout) in cons["tokens"].items():
        sections.append((f"自我一致性 {ruler}（繞過快取，每次跑都全額計費）",
                         SPEC[ruler][4], tin, tout))
    sp_tokens = {}
    for x in sp["detail"]:
        if x["exam"] != "4o":
            continue                      # mini 那兩格已計在「正式判定」裡
        ruler = "C" if x["judge"] == "mini" else "Co"
        t = sp_tokens.setdefault(ruler, [0, 0])
        t[0] += x["tokens"][0]
        t[1] += x["tokens"][1]
    sections.append((f"2×2 第二組答案生成（{m} 題）", j19.GPT4O, *gen_tokens))
    for ruler, (tin, tout) in sp_tokens.items():
        sections.append((f"2×2 第二組答案判定 {ruler}", SPEC[ruler][4], tin, tout))

    total = 0.0
    for name, model, tin, tout in sections:
        money = j19.cost_twd(model, tin, tout)
        total += money
        out.append(f"| {name} | `{model}` | {tin} | {tout} | {money:.5f} |")
    out += [f"| **合計** | — | — | — | **{total:.5f}** |", ""]

    per_answer = {}
    for ruler in RULERS:
        _, _, _, money = ruler_cost(verdicts, rows, ruler)
        per_answer[ruler] = money / n
    base = per_answer["B0"] or 1e-9
    out += ["每判一個答案要多少錢（正式判定那一列除以 40）：", "",
            "| 尺 | 每個答案 | 相對 B0 |", "|---|---|---|"]
    for ruler in RULERS:
        out.append(f"| {LABELS[ruler]} | {per_answer[ruler]:.5f} 元 | "
                   f"×{per_answer[ruler] / base:.1f} |")
    out.append("")
    return NL.join(out)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Day 19：修尺，並換第二個裁判")
    ap.add_argument("--gen", action="store_true",
                    help="只產生 2×2 用的 gpt-4o 答案與標註工作表")
    args = ap.parse_args()

    questions = load_json(QUESTIONS_PATH)
    rubrics = load_json(RUBRIC_PATH)
    probes = load_json(PROBES_PATH)
    collection, chunks, articles, _, _ = pipeline.build_index()
    texts = d18.article_text_map(chunks, articles)

    print(f"[0] 重建 Day 17 的 {len(K_VALUES) * len(questions)} 個答案"
          "（走快取，應為 0 元）…")
    rows, billed_in, billed_out = d18.rebuild_day17_answers(collection, questions)
    if billed_in or billed_out:
        print(f"   ⚠️ 出現計費 token（{billed_in}／{billed_out}）：生成側被改過。")

    if args.gen:
        alt_rows, tin, tout = generate_alt_answers(collection, questions)
        target = write_alt_template(alt_rows, questions)
        print(f"{NL}第二組答案（{j19.GPT4O}、k={SELF_PREF_K}）已寫入 {target.name}。"
              f"{NL}生成 token：輸入 {tin}／輸出 {tout}，"
              f"約 {j19.cost_twd(j19.GPT4O, tin, tout):.4f} 元"
              f"{NL}逐字讀完，把 verdict 填成 "
              f"{'／'.join(judges.VERDICTS)} 之一，再跑一次本腳本。")
        for r in alt_rows:
            print(f"{NL}  {r['key']}　{r['answer']}")
        return

    if not ALT_PATH.exists():
        raise SystemExit(f"找不到 {ALT_PATH.name}。"
                         f"先跑：python {Path(__file__).name} --gen")
    alt_payload = load_json(ALT_PATH)
    alt_labels = {x["key"]: x.get("verdict", "").strip() for x in alt_payload}
    bad = [k for k, v in alt_labels.items() if v not in judges.VERDICTS]
    if bad:
        raise SystemExit(f"{ALT_PATH.name} 還有 {len(bad)} 筆沒標：{'、'.join(bad)}")
    alt_rows = [{"key": x["key"], "id": int(x["key"].split("@")[0]),
                 "k": SELF_PREF_K, "answer": x["answer"]} for x in alt_payload]
    _, gen_in, gen_out = generate_alt_answers(collection, questions)

    labels = d18.load_labels(rows)

    print(f"[1] {len(RULERS)} 把尺量 {len(rows)} 個答案…")
    verdicts = run_rulers(rows, questions, rubrics, texts)
    agree = {r: d18.agreement(rows, verdicts, labels, r) for r in RULERS}

    print(f"[2] 探針回歸（{len(probes) * len(d18.PROBE_KINDS)} 支）…")
    probes_out = run_probes(questions, rubrics, probes, texts, RULERS)

    print("[3] 自我一致性（繞過快取）…")
    cons = consistency(rows, questions, rubrics, texts,
                       {"B3": CONS_RUNS_MINI, "Co": CONS_RUNS_4O})

    print(f"[4] self-preference 2×2（k={SELF_PREF_K}）…")
    sp = self_preference(rows, alt_rows, questions, rubrics, texts)

    print("[5] 產生報表…")
    RESULT_PATH.write_text(
        build_report(rows, questions, rubrics, verdicts, labels, agree,
                     probes_out, cons, sp, alt_rows, alt_labels,
                     (gen_in, gen_out), (billed_in, billed_out)),
        encoding="utf-8")
    print(f"{NL}報表已寫入 {RESULT_PATH.name}{NL}")

    n = len(rows)
    for ruler in RULERS:
        s = agree[ruler]["stat"]
        print(f"  {LABELS[ruler]:<28}｜一致 {s['一致']:>2}/{n}"
              f"｜誤殺 {s['誤殺']:>2}｜放水 {s['放水']:>2}")
    saved = {r: sum(1 for k in DAY18_B_MISKILLS
                    if verdicts[k][r]["verdict"] == labels[k])
             for r in ("B0", "B1", "B2", "B3", "B3o")}
    print(f"{NL}  9 筆誤殺救回："
          + "｜".join(f"{r} {v}/9" for r, v in saved.items()))
    for kind in d18.PROBE_KINDS:
        group = [p for p in probes_out if p["kind"] == kind]
        print(f"  探針 {kind:<9}｜" + "｜".join(
            f"{r}={sum(1 for p in group if p[r] == p['expected'])}/{len(group)}"
            for r in ("B0", "B3", "B3o", "C", "Co")))


if __name__ == "__main__":
    main()
