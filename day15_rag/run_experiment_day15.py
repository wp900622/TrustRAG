# -*- coding: utf-8 -*-
"""Day 15 主實驗：可信度。同一套檢索，只換 prompt。

Day 14 量到「答案正確率」，也量到那把尺的洞：模型說「條文未提及」一律判錯，
而規章根本沒寫的問題，它照樣掰得出答案（無 context 對照組那 13 題）。
今天補這個洞。

三組對照（檢索側凍結在 Day 14 的甜蜜點 k=3，只改 system prompt）：
    P1 基本版        —— Day 14 的 SYSTEM
    P2 要求引用      —— 加「標出條號」
    P3 引用＋允許拒答 —— 再加「條文未涵蓋就說規章未規定」

題組 20 題 ＝ 15 題原封不動（Day 9~14 沿用）＋ 5 題陷阱題：
    A 類（3 題）語料完全沒有的規定，正確行為只有拒答
    B 類（2 題）語料寫明「依另一辦法規定」，正確行為是轉指、不是編數字

四個新指標的定義寫在 grader.py 的 Day 15 段落。
結果寫入 experiment_result_day15.md（先寫檔再 print——Day 10 的教訓）。
"""
import json
from pathlib import Path

import chroma_store
import grader
import pipeline
import prompts
import rag_core

BASE_DIR = Path(__file__).parent
QUESTIONS_PATH = BASE_DIR / "questions.json"
TRAP_PATH = BASE_DIR / "questions_trap.json"
RESULT_PATH = BASE_DIR / "experiment_result_day15.md"
NL = chr(10)
K = 3
VERSIONS = ("P1", "P2", "P3")


def expected_no(question: dict) -> int:
    """把 "第 24 條" 解析成 24"""
    return int(question["expected"].replace("第", "").replace("條", "").strip())


def ask(collection, question: str, version: str) -> dict:
    """跑一題：檢索 k=3，用指定 prompt 版本生成。

    回傳兩組條號，判定引用時用得到：
      context_nos    餵進去的 k 段各自的條號
      verifiable_nos 上面那些，**再加上條文內文自己寫的交叉引用條號**

    第二組是踩過坑才加的。第 31 條的原文最後一句就是「逾期未補辦者，
    依第 32 條曠職論處」；模型照抄這句、引了第 32 條，只比對 context_nos
    會判成「憑空引用」——那是把模型忠實照抄的交叉引用當成幻覺。
    「引用可查證」的正確定義是「使用者能在我給它的文字裡找到這個條號」，
    不是「這個條號是我撈回來的那幾條之一」。
    """
    vectors, emb_tokens, _ = rag_core.get_embeddings([question])
    hits = chroma_store.retrieve(collection, vectors[0], K)
    messages = prompts.build_messages(question, hits, version=version)
    text, in_tok, out_tok = rag_core.chat(messages)
    context_nos = [h["meta"]["article_no"] for h in hits]
    cross_refs = grader.cited_articles(prompts.build_context(hits))
    return {"answer": text, "hits": hits, "context_nos": context_nos,
            "verifiable_nos": sorted(set(context_nos) | set(cross_refs)),
            "input_tokens": in_tok, "output_tokens": out_tok,
            "embedding_tokens": emb_tokens}


def run_normal(collection, questions: list[dict], version: str) -> list[dict]:
    """跑正常 15 題：沿用 Day 14 的關鍵事實比對，多量過度拒答與引用"""
    rows = []
    for q in questions:
        r = ask(collection, q["question"], version)
        exp = expected_no(q)
        correct, missing, forbid_hit = grader.grade(
            r["answer"], q["facts"], q["forbid"])
        # 過度拒答：正常題本來答得出來，卻用拒答詞把問題推掉。
        # 條件是「出現拒答詞」且「關鍵事實沒給齊」——事實給齊了的話，
        # 前面多一句「條文未明確規定但…」不算把使用者推走。
        over_refusal = grader.has_refusal_word(r["answer"]) and not correct
        rows.append({**r, "id": q["id"], "question": q["question"],
                     "expected": q["expected"], "expected_no": exp,
                     "rank": pipeline.rank_of_expected(r["hits"], exp),
                     "correct": correct, "missing": missing,
                     "forbid_hit": forbid_hit, "over_refusal": over_refusal,
                     "citation": grader.citation_verdict(
                         r["answer"], r["verifiable_nos"], exp),
                     "cited": grader.cited_articles(r["answer"])})
    return rows


def run_trap(collection, traps: list[dict], version: str) -> list[dict]:
    """跑 5 題陷阱題：判定幻覺／正確處理／假拒答／模糊"""
    rows = []
    for q in traps:
        r = ask(collection, q["question"], version)
        verdict, markers = grader.grade_trap(r["answer"], q)
        rows.append({**r, "id": q["id"], "question": q["question"],
                     "kind": q["kind"], "verdict": verdict, "markers": markers,
                     "citation": grader.citation_verdict(
                         r["answer"], r["verifiable_nos"], None),
                     "cited": grader.cited_articles(r["answer"])})
    return rows


def summarize(normal: list[dict], trap: list[dict], version: str) -> dict:
    """一版 prompt 的彙總數字"""
    cited_rows = [r for r in normal + trap if r["citation"] != "none"]
    return {
        "version": version,
        "correct": sum(1 for r in normal if r["correct"]),
        "over_refusal": sum(1 for r in normal if r["over_refusal"]),
        "normal_total": len(normal),
        "hallucination": sum(1 for r in trap
                             if r["verdict"] in ("hallucination", "fake_refusal")),
        "fake_refusal": sum(1 for r in trap if r["verdict"] == "fake_refusal"),
        "proper": sum(1 for r in trap if r["verdict"] == "proper"),
        "vague": sum(1 for r in trap if r["verdict"] == "vague"),
        "trap_total": len(trap),
        "cited_count": len(cited_rows),
        "fabricated": sum(1 for r in cited_rows if r["citation"] == "fabricated"),
        "cite_correct": sum(1 for r in normal if r["citation"] == "correct"),
        "input_tokens": sum(r["input_tokens"] for r in normal + trap),
        "output_tokens": sum(r["output_tokens"] for r in normal + trap),
        "avg_chars": round(sum(len(r["answer"]) for r in normal + trap)
                           / (len(normal) + len(trap)), 1),
    }


VERDICT_LABEL = {"proper": "✅ 正確處理", "hallucination": "❌ 幻覺",
                 "fake_refusal": "⚠️ 假拒答", "vague": "△ 模糊"}
CITE_LABEL = {"none": "未標條號", "fabricated": "❌ 憑空引用",
              "grounded": "△ 條號在 context 但非預期條文", "correct": "✅ 引用正確"}


def one_line(text: str) -> str:
    """答案塞進 markdown 表格前，換行與豎線都要處理掉"""
    return text.replace(NL, " ").replace("|", "／")


def build_report(summaries, normal_rows, trap_rows, index_info) -> str:
    """組出整份報表；文章的表格直接從這裡貼過去"""
    out = ["# Day 15 實驗結果：可信度的代價", "",
           f"- 語料：《員工工作規則》11 章 {index_info['chunks']} 條（虛構，同 Day 9~14）",
           "- 題組：15 題正常（沿用 Day 9~14，一題未改）＋ 5 題陷阱題（A 類 3、B 類 2）",
           f"- 檢索側凍結：k={K}（Day 14 量到的甜蜜點），只改 system prompt",
           f"- 生成模型：`{rag_core.CHAT_MODEL}`（temperature=0）",
           "- 判定：關鍵事實比對＋兩段式拒答判定（grader.py），非 LLM 裁判", "",
           "## 一、招牌總表", "",
           "| prompt 版本 | 正常題答對 | 過度拒答 | 陷阱題幻覺 | 其中假拒答 | "
           "正確處理 | 模糊 | 引用正確（正常題） | 憑空引用 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        n, t = s["normal_total"], s["trap_total"]
        out.append(
            f"| {prompts.PROMPT_LABELS[s['version']]} | **{s['correct']}/{n}** | "
            f"{s['over_refusal']}/{n} | **{s['hallucination']}/{t}** | "
            f"{s['fake_refusal']} | {s['proper']}/{t} | {s['vague']} | "
            f"{s['cite_correct']}/{n} | {s['fabricated']} |")
    out += ["", "> 「幻覺」＝陷阱題中給出語料裡沒有的具體答案，含假拒答；"
                "「假拒答」＝同時說了查不到、又給出具體答案。", ""]

    out += ["## 二、陷阱題逐題 × 三版", ""]
    for q in trap_rows[VERSIONS[0]]:
        nos = "、".join("第 " + str(n) + " 條" for n in q["context_nos"])
        out += [f"### 第 {q['id']} 題（{q['kind']} 類）{q['question']}", "",
                f"- 檢索撈回：{nos}", "",
                "| prompt | 判定 | 命中的幻覺特徵 | 引用 | 答案 |",
                "|---|---|---|---|---|"]
        for v in VERSIONS:
            r = next(x for x in trap_rows[v] if x["id"] == q["id"])
            out.append(f"| {v} | {VERDICT_LABEL[r['verdict']]} | "
                       f"{'、'.join(r['markers']) or '—'} | "
                       f"{CITE_LABEL[r['citation']]} | {one_line(r['answer'])} |")
        out.append("")

    out += ["## 三、正常 15 題：三版的判定差異", "",
            "| # | 題目 | P1 | P2 | P3 | P2 引用 | P3 引用 |",
            "|---|---|---|---|---|---|---|"]
    for i, base in enumerate(normal_rows[VERSIONS[0]]):
        cells = []
        for v in VERSIONS:
            r = normal_rows[v][i]
            cells.append("✅" if r["correct"]
                         else ("🚫 拒答" if r["over_refusal"] else "❌"))
        out.append(f"| {base['id']} | {base['question']} | {cells[0]} | "
                   f"{cells[1]} | {cells[2]} | "
                   f"{CITE_LABEL[normal_rows['P2'][i]['citation']]} | "
                   f"{CITE_LABEL[normal_rows['P3'][i]['citation']]} |")
    out.append("")

    out += ["## 四、判定有變動的正常題：答案原文", ""]
    for i, base in enumerate(normal_rows[VERSIONS[0]]):
        if len({normal_rows[v][i]["correct"] for v in VERSIONS}) == 1:
            continue
        out += [f"### 第 {base['id']} 題 {base['question']}", ""]
        for v in VERSIONS:
            r = normal_rows[v][i]
            flag = "✅" if r["correct"] else "❌"
            out.append(f"- **{v}** {flag}（漏掉：{'、'.join(r['missing']) or '—'}）："
                       f"{one_line(r['answer'])}")
        out.append("")

    out += ["## 五、逐題明細", ""]
    for v in VERSIONS:
        out += [f"### {prompts.PROMPT_LABELS[v]}", "",
                "| # | 題目 | 預期 | 排名 | 判定 | 引用 | 答案 |",
                "|---|---|---|---|---|---|---|"]
        for r in normal_rows[v]:
            rank = "未進 top-k" if r["rank"] == 0 else f"第 {r['rank']} 名"
            out.append(f"| {r['id']} | {r['question']} | {r['expected']} | {rank} | "
                       f"{'✅' if r['correct'] else '❌'} | "
                       f"{CITE_LABEL[r['citation']]} | {one_line(r['answer'])} |")
        for r in trap_rows[v]:
            out.append(f"| {r['id']} | {r['question']} | 陷阱（{r['kind']} 類） | — | "
                       f"{VERDICT_LABEL[r['verdict']]} | "
                       f"{CITE_LABEL[r['citation']]} | {one_line(r['answer'])} |")
        out.append("")

    total_in = sum(s["input_tokens"] for s in summaries)
    total_out = sum(s["output_tokens"] for s in summaries)
    chat_usd = rag_core.chat_cost_usd(total_in, total_out)
    emb_usd = rag_core.embedding_cost_usd(index_info["embedding_tokens"])
    out += ["## 六、成本", "",
            f"- Embedding：本次計費 {index_info['embedding_tokens']} tokens"
            f" ≈ 新台幣 {rag_core.twd(emb_usd):.5f} 元",
            f"- 生成：輸入 {total_in}、輸出 {total_out} tokens"
            f" ≈ 新台幣 {rag_core.twd(chat_usd):.5f} 元"
            "（3 版 × 20 題 = 60 次生成，扣掉 Day 14 已快取的部分）",
            f"- 合計 ≈ **新台幣 {rag_core.twd(emb_usd + chat_usd):.5f} 元**", "",
            "刪除 `embeddings_cache.json` 與 `chat_cache.json` 重跑，"
            "即可重現無快取的完整成本。", ""]
    return NL.join(out)


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    traps = json.loads(TRAP_PATH.read_text(encoding="utf-8"))
    collection, chunks, _, emb_tokens, _ = pipeline.build_index()

    normal_rows, trap_rows, summaries = {}, {}, []
    for v in VERSIONS:
        normal_rows[v] = run_normal(collection, questions, v)
        trap_rows[v] = run_trap(collection, traps, v)
        summaries.append(summarize(normal_rows[v], trap_rows[v], v))

    extra_emb = sum(r["embedding_tokens"]
                    for v in VERSIONS for r in normal_rows[v] + trap_rows[v])
    index_info = {"chunks": len(chunks), "embedding_tokens": emb_tokens + extra_emb}
    RESULT_PATH.write_text(build_report(summaries, normal_rows, trap_rows,
                                        index_info), encoding="utf-8")

    print(f"報表已寫入 {RESULT_PATH.name}")
    for s in summaries:
        print(f"{s['version']}｜正常答對 {s['correct']}/{s['normal_total']}"
              f"｜過度拒答 {s['over_refusal']}"
              f"｜陷阱幻覺 {s['hallucination']}/{s['trap_total']}"
              f"（假拒答 {s['fake_refusal']}）"
              f"｜正確處理 {s['proper']}｜引用正確 {s['cite_correct']}"
              f"｜憑空引用 {s['fabricated']}")


if __name__ == "__main__":
    main()
