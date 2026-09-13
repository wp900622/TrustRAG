# -*- coding: utf-8 -*-
"""Day 17 實驗：多條文綜合題 × k 值 × 條文順序 × LLM reranker。

Day 14~16 的題目全部是「單一條文可答」。那個前提下，recall@k 是答對率的
天花板，而 k=3 就是甜蜜點。今天把題組換成 8 題**必須兩條條文才答得出來**
的綜合題，量三件事：

  一、k 掃描（1/3/5/10/15）
      單條 recall：每條分開算，撈到幾條算幾條 → Day 15 用的那把尺
      聯集 recall：**所有必要條文都進 top-k 才算命中** → 今天的新尺
      兩把尺在單條文題上幾乎一樣，在多條文題上會裂開，裂多大是本日重點。

  二、條文順序（固定 k=10，同一批 context 換三種排法）
      相似度序／反序／必要條文塞中間。撈到了不等於讀到了——
      如果三種排法答對率不同，代表位置會吃掉資訊（lost in the middle）。

  三、LLM reranker（候選 15 → 重排取 3）
      向量檢索問的是「哪一條最像這個提問」，多條文題要的是
      「哪幾條湊起來才答得完」。reranker 能不能補這個落差，
      以及它的帳單長什麼樣。

凍結的東西：語料（59 條）、切法、embedding 模型、生成模型、temperature=0、
prompt 一律用 P1 基本版。今天唯一的變因在檢索側——Day 16 剛好相反。

成本一律用 tiktoken 離線重數（Day 16 的 measure_cost.py 作法），
不依賴 API 回報的 usage，所以快取全命中時報表數字照樣完整；
「本次實際計費」另外單獨列，兩個數字不要混為一談。

結果寫入 experiment_result_day17.md。
"""
import json
from pathlib import Path

import tiktoken

import chroma_store
import chunkers
import grader
import pipeline
import prompts
import rag_core
import reranker

BASE_DIR = Path(__file__).parent
QUESTIONS_PATH = BASE_DIR / "questions_day17.json"
RESULT_PATH = BASE_DIR / "experiment_result_day17.md"
NL = chr(10)

K_VALUES = (1, 3, 5, 10, 15)
ORDER_K = 10                 # 條文順序實驗固定在這個 k
RERANK_POOL = 15             # reranker 的候選數
RERANK_TOP_N = 3             # 重排後送進生成的條數，與 Day 15 的甜蜜點對齊
ORDERINGS = ("sim", "reverse", "middle")
ORDER_LABELS = {"sim": "相似度序（預設）",
                "reverse": "反序（最相關放最後）",
                "middle": "必要條文塞中間"}

CHAT_ENC = tiktoken.get_encoding("o200k_base")   # gpt-4o-mini
EMB_ENC = tiktoken.get_encoding("cl100k_base")   # text-embedding-3-small


# ------------------------------------------------------------------ 小工具

def count_messages(messages: list[dict]) -> int:
    """OpenAI chat 的輸入 token：每則訊息 +3，整體再 +3"""
    total = 0
    for m in messages:
        total += (3 + len(CHAT_ENC.encode(m["role"]))
                  + len(CHAT_ENC.encode(m["content"])))
    return total + 3


def rank_of(hits: list[dict], article_no: int) -> int:
    """指定條號排第幾名（1 起算），沒進清單回 0"""
    for i, hit in enumerate(hits, 1):
        if hit["meta"]["article_no"] == article_no:
            return i
    return 0


def ranks_of(hits: list[dict], expected: list[int]) -> dict:
    return {no: rank_of(hits, no) for no in expected}


def fmt_rank(r: int) -> str:
    return "未進" if r == 0 else f"第{r}名"


def fmt_ranks(ranks: dict) -> str:
    return "、".join(f"第{no}條={fmt_rank(r)}" for no, r in ranks.items())


def article_text_map(chunks: list[dict], articles: list[dict]) -> dict:
    """條號 → 條文原文。判定自我檢查要用。"""
    mapping = {}
    for chunk in chunks:
        covered = chunkers.articles_covering(chunk, articles)
        if covered:
            mapping[covered[0]["no"]] = chunk["text"]
    return mapping


def retrieve(collection, question_text: str, k: int) -> tuple[list[dict], int]:
    vectors, emb_tokens, _ = rag_core.get_embeddings([question_text])
    return chroma_store.retrieve(collection, vectors[0], k), emb_tokens


def generate(question: dict, hits: list[dict]) -> dict:
    """跑一次生成並判定；token 同時記「實際計費」與「離線重數」兩套"""
    messages = prompts.build_messages(question["question"], hits)
    text, billed_in, billed_out = rag_core.chat(messages)
    correct, missing, forbid_hit = grader.grade(
        text, question["facts"], question.get("forbid"))
    return {"id": question["id"], "answer": text,
            "billed_in": billed_in, "billed_out": billed_out,
            "count_in": count_messages(messages),
            "count_out": len(CHAT_ENC.encode(text)),
            "correct": correct, "missing": missing, "forbid_hit": forbid_hit,
            "context_nos": [h["meta"]["article_no"] for h in hits]}


def reorder(hits: list[dict], expected: list[int], mode: str) -> list[dict]:
    """同一批 context 的三種排法。

    middle：把必要條文抽出來塞進中間位置——lost-in-the-middle 的壓力測試。
    注意 sim 這一組跟 k 掃描那一組是同一批 messages，所以它必定命中快取、
    不重複計費，這本身也是一個可以寫進文章的觀察。
    """
    if mode == "sim":
        return list(hits)
    if mode == "reverse":
        return list(reversed(hits))
    required = [h for h in hits if h["meta"]["article_no"] in expected]
    others = [h for h in hits if h["meta"]["article_no"] not in expected]
    half = len(others) // 2
    return others[:half] + required + others[half:]


# -------------------------------------------------------------- 三個階段

def stage_k_sweep(collection, questions: list[dict]) -> dict:
    rows, emb_total = {}, 0
    for k in K_VALUES:
        per_q = []
        for q in questions:
            hits, emb = retrieve(collection, q["question"], k)
            emb_total += emb
            r = generate(q, hits)
            r["ranks"] = ranks_of(hits, q["expected"])
            r["found"] = sum(1 for v in r["ranks"].values() if v > 0)
            r["need"] = len(q["expected"])
            r["joint"] = r["found"] == r["need"]
            per_q.append(r)
        rows[k] = per_q
    return {"rows": rows, "embedding_tokens": emb_total}


def stage_ordering(collection, questions: list[dict]) -> dict:
    rows = {}
    for mode in ORDERINGS:
        per_q = []
        for q in questions:
            hits, _ = retrieve(collection, q["question"], ORDER_K)
            ordered = reorder(hits, q["expected"], mode)
            r = generate(q, ordered)
            r["ranks"] = ranks_of(ordered, q["expected"])
            r["joint"] = all(v > 0 for v in r["ranks"].values())
            per_q.append(r)
        rows[mode] = per_q
    return rows


def stage_rerank(collection, questions: list[dict]) -> list[dict]:
    out = []
    for q in questions:
        pool, _ = retrieve(collection, q["question"], RERANK_POOL)
        reranked, stats = reranker.rerank(q["question"], pool, RERANK_TOP_N)
        r = generate(q, reranked)
        r["ranks"] = ranks_of(reranked, q["expected"])
        r["joint"] = all(v > 0 for v in r["ranks"].values())
        r["pool_ranks"] = ranks_of(pool, q["expected"])
        r["pool_joint"] = all(v > 0 for v in r["pool_ranks"].values())
        r["rerank"] = stats
        out.append(r)
    return out


def self_test(questions: list[dict], texts: dict) -> list[tuple]:
    """尺的自我檢查：把必要條文的原文直接送進 grader，應該 8 題全過。

    這是本系列從 Day 14 就在做的事（README 有寫）：facts 是人挑的，
    挑錯就會誤殺正確答案。全過才代表「答錯」是模型的問題，不是尺的問題。
    """
    results = []
    for q in questions:
        blob = NL.join(texts.get(no, "") for no in q["expected"])
        correct, missing, _ = grader.grade(blob, q["facts"], q.get("forbid"))
        results.append((q["id"], correct, missing))
    return results


# ---------------------------------------------------------------- 報表

def recall_pair(per_q: list[dict]) -> tuple[float, float]:
    """回傳 (單條 recall, 聯集 recall)"""
    found = sum(r["found"] for r in per_q)
    need = sum(r["need"] for r in per_q)
    joint = sum(1 for r in per_q if r["joint"])
    return found / need, joint / len(per_q)


def accuracy(per_q: list[dict]) -> float:
    return sum(1 for r in per_q if r["correct"]) / len(per_q)


def cost_of(per_q: list[dict]) -> tuple[int, int, float]:
    cin = sum(r["count_in"] for r in per_q)
    cout = sum(r["count_out"] for r in per_q)
    return cin, cout, rag_core.chat_cost_usd(cin, cout)


def build_report(questions, sweep, order_rows, rerank_rows, checks) -> str:
    qmap = {q["id"]: q for q in questions}
    n = len(questions)
    rows = sweep["rows"]
    out = [
        "# Day 17 實驗結果：多條文綜合題",
        "",
        f"- 題組：{n} 題，每題都需要兩條條文才答得完（`questions_day17.json`）",
        "- 語料：虛構《員工工作規則》59 條（同 Day 9~16，一字未改）",
        f"- 生成：`{rag_core.CHAT_MODEL}`、temperature=0、prompt 固定用 P1 基本版",
        "- 今天唯一的變因在檢索側：k 值、條文順序、有沒有 rerank",
        "",
        "## 一、k 掃描：兩把 recall 尺",
        "",
        "| k | 單條 recall | 聯集 recall | 答對率 | 輸入 tokens | 輸出 tokens | 成本（新台幣） |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in K_VALUES:
        per_q = rows[k]
        single, joint = recall_pair(per_q)
        cin, cout, usd = cost_of(per_q)
        out.append(f"| {k} | {single:.0%} | **{joint:.0%}** | "
                   f"{accuracy(per_q):.0%}（{sum(1 for r in per_q if r['correct'])}/{n}） | "
                   f"{cin} | {cout} | {rag_core.twd(usd):.5f} |")

    out += ["", "### 逐題：必要條文各排第幾名", "",
            "| 題號 | 問題 | 必要條文 | " + " | ".join(f"k={k}" for k in K_VALUES) + " |",
            "|---|---|---|" + "---|" * len(K_VALUES)]
    for q in questions:
        cells = []
        for k in K_VALUES:
            r = next(x for x in rows[k] if x["id"] == q["id"])
            mark = "✅" if r["joint"] else "❌"
            cells.append(mark + " " + fmt_ranks(r["ranks"]))
        out.append(f"| {q['id']} | {q['question'][:22]}… | "
                   f"{'、'.join('第' + str(a) + '條' for a in q['expected'])} | "
                   + " | ".join(cells) + " |")

    out += ["", "### 逐題：答對了嗎（✅／❌＋漏掉的關鍵事實）", "",
            "| 題號 | " + " | ".join(f"k={k}" for k in K_VALUES) + " |",
            "|---|" + "---|" * len(K_VALUES)]
    for q in questions:
        cells = []
        for k in K_VALUES:
            r = next(x for x in rows[k] if x["id"] == q["id"])
            cells.append("✅" if r["correct"]
                         else "❌ 漏：" + "、".join(r["missing"]))
        out.append(f"| {q['id']} | " + " | ".join(cells) + " |")

    # ----------------------------------------------------------- 順序
    out += ["", f"## 二、條文順序（固定 k={ORDER_K}，同一批 context 換三種排法）", "",
            "| 排法 | 答對率 | 輸入 tokens | 輸出 tokens |", "|---|---|---|---|"]
    for mode in ORDERINGS:
        per_q = order_rows[mode]
        cin, cout, _ = cost_of(per_q)
        out.append(f"| {ORDER_LABELS[mode]} | "
                   f"{accuracy(per_q):.0%}（{sum(1 for r in per_q if r['correct'])}/{n}） "
                   f"| {cin} | {cout} |")
    out += ["",
            "輸入 token 三組幾乎一樣（只有條文順序不同），所以這張表比的是"
            "**同樣的資訊擺在不同位置，模型讀不讀得到**。",
            "", "| 題號 | " + " | ".join(ORDER_LABELS[m] for m in ORDERINGS) + " |",
            "|---|" + "---|" * len(ORDERINGS)]
    for q in questions:
        cells = []
        for mode in ORDERINGS:
            r = next(x for x in order_rows[mode] if x["id"] == q["id"])
            if not r["joint"]:
                cells.append("—（條文沒撈到）")
            else:
                cells.append("✅" if r["correct"]
                             else "❌ 漏：" + "、".join(r["missing"]))
        out.append(f"| {q['id']} | " + " | ".join(cells) + " |")

    # ----------------------------------------------------------- rerank
    base3 = rows[3]
    base15 = rows[15]
    r_single, r_joint = recall_pair(
        [{**r, "found": sum(1 for v in r["ranks"].values() if v > 0),
          "need": len(qmap[r["id"]]["expected"]),
          "joint": r["joint"]} for r in rerank_rows])
    rin = sum(r["count_in"] for r in rerank_rows)
    rout = sum(r["count_out"] for r in rerank_rows)
    rr_in = sum(r["rerank"]["input_tokens"] for r in rerank_rows)
    rr_out = sum(r["rerank"]["output_tokens"] for r in rerank_rows)
    fallbacks = sum(1 for r in rerank_rows if r["rerank"]["fallback"])

    b3_single, b3_joint = recall_pair(base3)
    b15_single, b15_joint = recall_pair(base15)
    b3_in, b3_out, b3_usd = cost_of(base3)
    b15_in, b15_out, b15_usd = cost_of(base15)
    rr_usd = rag_core.chat_cost_usd(rin, rout)

    out += ["", f"## 三、LLM reranker（候選 {RERANK_POOL} → 重排取 {RERANK_TOP_N}）", "",
            "| 組別 | 單條 recall | 聯集 recall | 答對率 | 送進生成的輸入 tokens | 生成成本 |",
            "|---|---|---|---|---|---|",
            f"| 純向量 k=3 | {b3_single:.0%} | {b3_joint:.0%} | "
            f"{accuracy(base3):.0%} | {b3_in} | {rag_core.twd(b3_usd):.5f} 元 |",
            f"| 純向量 k=15 | {b15_single:.0%} | {b15_joint:.0%} | "
            f"{accuracy(base15):.0%} | {b15_in} | {rag_core.twd(b15_usd):.5f} 元 |",
            f"| **rerank {RERANK_POOL}→{RERANK_TOP_N}** | {r_single:.0%} | {r_joint:.0%} | "
            f"{accuracy(rerank_rows):.0%} | {rin} | {rag_core.twd(rr_usd):.5f} 元 |",
            "",
            f"重排本身另外花掉：輸入 {rr_in}／輸出 {rr_out} tokens ≈ "
            f"**新台幣 {rag_core.twd(rag_core.chat_cost_usd(rr_in, rr_out)):.5f} 元**"
            f"（{len(rerank_rows)} 次呼叫，解析失敗退回原順序 {fallbacks} 次）。",
            "",
            "⚠️ 這筆錢要跟生成的錢加在一起看：rerank 省下的是送進生成的 context，"
            "但它自己要先把 15 條完整讀一遍——**候選池的 token 一樣要付，"
            "只是換到重排這一行**。",
            "", "### 逐題：重排挑了哪幾條", "",
            "| 題號 | 必要條文 | 候選 15 名次 | 重排後挑出 | 聯集命中 | 答對 |",
            "|---|---|---|---|---|---|"]
    for r in rerank_rows:
        q = qmap[r["id"]]
        out.append(f"| {r['id']} | "
                   f"{'、'.join('第' + str(a) + '條' for a in q['expected'])} | "
                   f"{fmt_ranks(r['pool_ranks'])} | "
                   f"{'、'.join('第' + str(a) + '條' for a in r['rerank']['picked'])} | "
                   f"{'✅' if r['joint'] else '❌'} | "
                   f"{'✅' if r['correct'] else '❌ 漏：' + '、'.join(r['missing'])} |")

    # ----------------------------------------------------------- 答案原文
    out += ["", "## 四、逐題答案原文（k=3 / k=15 / rerank 三組對照）", ""]
    for q in questions:
        out += [f"### {q['id']}　{q['question']}", "",
                f"- 必要條文：{'、'.join('第 ' + str(a) + ' 條' for a in q['expected'])}",
                f"- 判定事實：{'、'.join(q['facts'])}（全部出現才算對）",
                f"- 為什麼選它：{q['why']}", ""]
        for label, per_q in (("k=3", base3), ("k=15", base15),
                             (f"rerank {RERANK_POOL}→{RERANK_TOP_N}", rerank_rows)):
            r = next(x for x in per_q if x["id"] == q["id"])
            nos = "、".join("第" + str(x) + "條" for x in r["context_nos"])
            out += [f"**{label}**（撈回：{nos}）"
                    f"　{'✅' if r['correct'] else '❌'}",
                    "", "> " + r["answer"].replace(NL, " "), ""]

    # ----------------------------------------------------------- 自我檢查
    out += ["## 五、判定自我檢查（尺會不會誤殺正確答案）", "",
            "把每題必要條文的**原文**直接送進 `grader.grade()`，"
            "全過才代表答錯是模型的問題、不是 facts 挑錯。", "",
            "| 題號 | 通過 | 湊不齊的 fact |", "|---|---|---|"]
    for qid, ok, missing in checks:
        out.append(f"| {qid} | {'✅' if ok else '❌'} | "
                   f"{'、'.join(missing) or '—'} |")
    failed = [c for c in checks if not c[1]]
    out += ["", f"**{len(checks) - len(failed)}/{len(checks)} 通過。**"
            + ("" if not failed else
               "　⚠️ 有題目沒過：那幾題的 facts 必須先改，"
               "否則後面所有答對率都是被尺壓低的。"), ""]

    # ----------------------------------------------------------- 成本
    all_rows = ([r for k in K_VALUES for r in rows[k]]
                + [r for m in ORDERINGS for r in order_rows[m]]
                + rerank_rows)
    billed_in = sum(r["billed_in"] for r in all_rows) + rr_in
    billed_out = sum(r["billed_out"] for r in all_rows) + rr_out
    full_in = sum(r["count_in"] for r in all_rows) + rr_in
    full_out = sum(r["count_out"] for r in all_rows) + rr_out
    emb = sum(len(EMB_ENC.encode(q["question"])) for q in questions)
    billed_usd = rag_core.chat_cost_usd(billed_in, billed_out)
    full_usd = rag_core.chat_cost_usd(full_in, full_out)
    gen_count = len(all_rows)

    out += ["## 六、成本", "",
            f"- 生成次數：{gen_count} 次"
            f"（k 掃描 {len(K_VALUES)}×{n}、順序 {len(ORDERINGS)}×{n}、"
            f"rerank {n}）＋ 重排呼叫 {len(rerank_rows)} 次",
            f"- **本次實際計費**：embedding {sweep['embedding_tokens']} ＋ "
            f"生成輸入 {billed_in}／輸出 {billed_out} tokens ≈ "
            f"**新台幣 {rag_core.twd(billed_usd):.5f} 元**"
            "（快取命中的部分不計費，所以重跑時這個數字會變小甚至為 0）",
            f"- **全部從零跑**：生成輸入 {full_in}／輸出 {full_out} tokens ≈ "
            f"**新台幣 {rag_core.twd(full_usd):.5f} 元**"
            f"（另加 {n} 題問句 embedding {emb} tokens ≈ "
            f"{rag_core.twd(rag_core.embedding_cost_usd(emb)):.5f} 元）",
            "",
            "### 加大 k 的帳單", "",
            "| k | 輸入 tokens | 相對 k=1 | 每題平均輸入 |", "|---|---|---|---|"]
    base_in = cost_of(rows[1])[0]
    for k in K_VALUES:
        cin = cost_of(rows[k])[0]
        out.append(f"| {k} | {cin} | {cin / base_in:.2f}× | {cin // n} |")
    out += ["",
            "k 從 1 開到 15，輸入 token 是線性長的，而聯集 recall 不是——"
            "上面第一張表就是這句話的證據。", ""]
    return NL.join(out)


# ---------------------------------------------------------------- main

def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    collection, chunks, articles, _, _ = pipeline.build_index()
    texts = article_text_map(chunks, articles)

    checks = self_test(questions, texts)
    failed = [c for c in checks if not c[1]]
    if failed:
        print("⚠️ 判定自我檢查未通過（facts 可能挑錯，答對率會被壓低）：")
        for qid, _, missing in failed:
            print(f"   第 {qid} 題湊不齊：{'、'.join(missing)}")
        print("   仍會繼續跑，結果請連同第五節一起讀。")

    print(f"[1/3] k 掃描 {K_VALUES} × {len(questions)} 題…")
    sweep = stage_k_sweep(collection, questions)
    print(f"[2/3] 條文順序 {ORDERINGS} × {len(questions)} 題（k={ORDER_K}）…")
    order_rows = stage_ordering(collection, questions)
    print(f"[3/3] rerank {RERANK_POOL}→{RERANK_TOP_N} × {len(questions)} 題…")
    rerank_rows = stage_rerank(collection, questions)

    RESULT_PATH.write_text(
        build_report(questions, sweep, order_rows, rerank_rows, checks),
        encoding="utf-8")
    print(f"{NL}報表已寫入 {RESULT_PATH.name}{NL}")

    for k in K_VALUES:
        per_q = sweep["rows"][k]
        single, joint = recall_pair(per_q)
        print(f"  k={k:>2}｜單條 recall {single:5.0%}｜聯集 recall {joint:5.0%}"
              f"｜答對 {sum(1 for r in per_q if r['correct'])}/{len(questions)}")
    for mode in ORDERINGS:
        per_q = order_rows[mode]
        print(f"  {ORDER_LABELS[mode]:>12}｜答對 "
              f"{sum(1 for r in per_q if r['correct'])}/{len(questions)}")
    r_joint = sum(1 for r in rerank_rows if r["joint"]) / len(rerank_rows)
    print(f"  rerank {RERANK_POOL}→{RERANK_TOP_N}｜聯集 recall {r_joint:5.0%}"
          f"｜答對 {sum(1 for r in rerank_rows if r['correct'])}/{len(questions)}")


if __name__ == "__main__":
    main()
