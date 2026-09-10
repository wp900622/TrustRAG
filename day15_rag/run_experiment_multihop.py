# -*- coding: utf-8 -*-
"""Day 15 補充實驗：需要兩條條文才回答得出來的題目。

Day 14 的 15 題全部是「單一條文可答」，所以那天量到的
「答對率天花板 = recall@k」只在那個前提下成立。這支腳本測一題多條文綜合題：

    我平日加班 3 小時，加班費要怎麼算？
        第 19 條 → 加成比例（前 2 小時加給三分之一、再延長加給三分之二）
        第 33 條 → 計算基準是本薪加計經常性給與

缺任何一條都算不出來。要看的是兩件事：
    1. 兩條會不會同時被撈進 top-k（單條 recall 好看，聯集可能很難看）
    2. 就算都撈到了，模型會不會把比例抄對

刻意不動 questions.json 與 run_experiment.py——Day 14 的數字已經發布，
題組一改那份報表就對不上了。

結果寫入 experiment_result_multihop.md。
"""
import json
from pathlib import Path

import chroma_store
import grader
import pipeline
import prompts
import rag_core

BASE_DIR = Path(__file__).parent
QUESTIONS_PATH = BASE_DIR / "questions_multihop.json"
RESULT_PATH = BASE_DIR / "experiment_result_multihop.md"
NL = chr(10)
K_VALUES = (1, 3, 5, 10)


def rank_of(hits: list[dict], article_no: int) -> int:
    """指定條號排第幾名（1 起算），沒進 top-k 回 0"""
    for i, hit in enumerate(hits, 1):
        if hit["meta"]["article_no"] == article_no:
            return i
    return 0


def run(collection, question: dict, k: int | None) -> dict:
    """跑一組。k=None 是無 context 對照組"""
    if k is None:
        text, in_tok, out_tok = rag_core.chat(
            prompts.build_messages_no_context(question["question"]))
        hits = []
    else:
        vectors, _, _ = rag_core.get_embeddings([question["question"]])
        hits = chroma_store.retrieve(collection, vectors[0], k)
        text, in_tok, out_tok = rag_core.chat(
            prompts.build_messages(question["question"], hits))
    ranks = {no: rank_of(hits, no) for no in question["expected"]}
    correct, missing, forbid_hit = grader.grade(
        text, question["facts"], question["forbid"])
    return {"k": k, "answer": text, "ranks": ranks, "correct": correct,
            "missing": missing, "forbid_hit": forbid_hit,
            "context_nos": [h["meta"]["article_no"] for h in hits],
            "input_tokens": in_tok, "output_tokens": out_tok}


def build_report(question: dict, rows: list[dict]) -> str:
    out = ["# Day 15 補充實驗：多條文綜合題", "",
           f"- 題目：{question['question']}",
           f"- 需要的條文：{'、'.join('第 ' + str(n) + ' 條' for n in question['expected'])}",
           f"- 判定：{'、'.join(question['facts'])} 全部出現才算對",
           f"- 為什麼選它：{question['why']}", "",
           "## 一、總表", "",
           "| 組別 | 第 19 條排名 | 第 33 條排名 | 兩條都撈到 | 答案正確 | 漏掉的關鍵事實 |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        name = "無 context" if r["k"] is None else f"k={r['k']}"
        if r["k"] is None:
            r19 = r33 = both = "—"
        else:
            r19 = "未進 top-k" if r["ranks"][19] == 0 else f"第 {r['ranks'][19]} 名"
            r33 = "未進 top-k" if r["ranks"][33] == 0 else f"第 {r['ranks'][33]} 名"
            both = "✅" if all(v > 0 for v in r["ranks"].values()) else "❌"
        out.append(f"| {name} | {r19} | {r33} | {both} | "
                   f"{'✅' if r['correct'] else '❌'} | "
                   f"{'、'.join(r['missing']) or '—'} |")

    out += ["", "## 二、逐組答案原文", ""]
    for r in rows:
        name = "無 context" if r["k"] is None else f"k={r['k']}"
        nos = "、".join("第 " + str(n) + " 條" for n in r["context_nos"]) or "（不給條文）"
        out += [f"### {name}", "", f"- 撈回：{nos}", "",
                "> " + r["answer"].replace(NL, " "), ""]

    total_in = sum(r["input_tokens"] for r in rows)
    total_out = sum(r["output_tokens"] for r in rows)
    usd = rag_core.chat_cost_usd(total_in, total_out)
    out += ["## 三、成本", "",
            f"- 生成：輸入 {total_in}、輸出 {total_out} tokens"
            f" ≈ 新台幣 {rag_core.twd(usd):.5f} 元（{len(rows)} 次生成）", ""]
    return NL.join(out)


def main() -> None:
    question = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))[0]
    collection, *_ = pipeline.build_index()
    rows = [run(collection, question, None)]
    rows += [run(collection, question, k) for k in K_VALUES]
    RESULT_PATH.write_text(build_report(question, rows), encoding="utf-8")

    print(f"報表已寫入 {RESULT_PATH.name}")
    for r in rows:
        name = "無 context" if r["k"] is None else f"k={r['k']}"
        ranks = ("—" if r["k"] is None
                 else " ".join(f"第{no}條={v or '未進'}" for no, v in r["ranks"].items()))
        print(f"{name:>12}｜{ranks}｜答對 {'✅' if r['correct'] else '❌'}"
              f"｜漏 {'、'.join(r['missing']) or '—'}")


if __name__ == "__main__":
    main()
