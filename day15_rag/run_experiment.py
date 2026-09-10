# -*- coding: utf-8 -*-
"""Day 14 主實驗：接上生成之後，該用哪把尺？

四組對照，同一組 15 題（沿用 Day 9~11，一題沒動）：
    A. 無 context      —— 不檢索直接問，量「模型本來就會多少」
    B. k=1             —— 只餵第一名，等於把前幾天的 top-1 直接送進去
    C. k=3             —— 餵前三名
    D. k=5             —— 餵前五名

三個指標一起看：
    top-1 命中率  —— Day 8~13 的舊尺（預期條文排第一名）
    recall@k     —— 預期條文有沒有出現在 k 個裡面（生成端真正吃到的）
    答案正確率    —— 新尺，關鍵事實比對（見 grader.py）

結果寫入 experiment_result.md，再 print 摘要（先寫檔後 print：
stdout 重導向遇上 cp950 會炸掉編碼，報表不能跟著遺失——Day 10 的教訓）。
"""
import json
from pathlib import Path

import grader
import pipeline
import rag_core

BASE_DIR = Path(__file__).parent
QUESTIONS_PATH = BASE_DIR / "questions.json"
RESULT_PATH = BASE_DIR / "experiment_result.md"
K_VALUES = (1, 3, 5)


def expected_no(question: dict) -> int:
    """把 "第 24 條" 解析成 24"""
    return int(question["expected"].replace("第", "").replace("條", "").strip())


def run_group(collection, questions: list[dict], k: int | None) -> list[dict]:
    """跑一組（k=None 代表無 context 對照組），回傳每題的紀錄"""
    rows = []
    for question in questions:
        result = pipeline.answer(collection, question["question"],
                                 k=k or 1, with_context=k is not None)
        rank = (pipeline.rank_of_expected(result.hits, expected_no(question))
                if k is not None else -1)
        correct, missing, hit_forbid = grader.grade(
            result.answer, question["facts"], question["forbid"])
        rows.append({
            "id": question["id"], "question": question["question"],
            "expected": question["expected"], "type": question["type"],
            "rank": rank, "correct": correct, "missing": missing,
            "forbid_hit": hit_forbid, "answer": result.answer,
            "refused": grader.looks_like_refusal(result.answer),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "embedding_tokens": result.embedding_tokens,
            "context_chars": result.context_chars,
        })
    return rows


def summarize(rows: list[dict], k: int | None) -> dict:
    """一組的彙總數字"""
    total = len(rows)
    return {
        "k": k,
        "top1": sum(1 for r in rows if r["rank"] == 1),
        "recall": sum(1 for r in rows if r["rank"] > 0),
        "correct": sum(1 for r in rows if r["correct"]),
        "refused": sum(1 for r in rows if r["refused"]),
        "total": total,
        "avg_answer_chars": round(sum(len(r["answer"]) for r in rows) / total, 1),
        "avg_context_chars": round(sum(r["context_chars"] for r in rows) / total, 1),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
    }


def cross_table(rows: list[dict]) -> dict:
    """檢索 × 答案 的 2×2 交叉表：落差到底出在哪一格"""
    cells = {"hit_correct": 0, "hit_wrong": 0, "miss_correct": 0, "miss_wrong": 0}
    for row in rows:
        hit = row["rank"] > 0
        key = f"{'hit' if hit else 'miss'}_{'correct' if row['correct'] else 'wrong'}"
        cells[key] += 1
    return cells


def rank_table(rows: list[dict]) -> dict[int, list[int]]:
    """預期條文排第幾名 → [題數, 答對數]。看模型讀不讀得到後面的名次"""
    table: dict[int, list[int]] = {}
    for row in rows:
        bucket = table.setdefault(row["rank"], [0, 0])
        bucket[0] += 1
        bucket[1] += int(row["correct"])
    return dict(sorted(table.items()))


def fmt_detail(rows: list[dict]) -> str:
    """逐題明細表"""
    lines = ["| # | 題目 | 預期 | 排名 | 判定 | 漏掉的關鍵事實 | 答案 |",
             "|---|------|------|------|------|----------------|------|"]
    for row in rows:
        rank = "未進 top-k" if row["rank"] == 0 else f"第 {row['rank']} 名"
        verdict = "✅" if row["correct"] else "❌"
        missing = "、".join(row["missing"]) or "—"
        if row["forbid_hit"]:
            missing += f"（出現錯誤事實：{'、'.join(row['forbid_hit'])}）"
        answer = row["answer"].replace("\n", " ").replace("|", "／")
        lines.append(f"| {row['id']} | {row['question']} | {row['expected']} | "
                     f"{rank} | {verdict} | {missing} | {answer} |")
    return "\n".join(lines)


def build_report(summaries: list[dict], detail_rows: dict, index_info: dict) -> str:
    """組出整份報表；文章的表格直接從這裡貼過去"""
    out = ["# Day 14 實驗結果：檢索命中 ≠ 答案正確", "",
           f"- 語料：《員工工作規則》11 章 {index_info['chunks']} 條（虛構，同 Day 9~11）",
           f"- 題組：{index_info['questions']} 題（沿用 Day 9~11，一題未改）",
           f"- Embedding 模型：`{rag_core.EMBEDDING_MODEL}`；"
           f"生成模型：`{rag_core.CHAT_MODEL}`（temperature=0）",
           f"- 判定：關鍵事實比對（grader.py），非 LLM 裁判", "",
           "## 一、四組對照總表", "",
           "| 組別 | top-1 命中 | recall@k | 答案正確 | 平均答案字數 | "
           "平均 context 字數 | 計費 tokens（輸入／輸出） |",
           "|------|-----------|----------|----------|-------------|"
           "------------------|--------------------------|"]
    for s in summaries:
        name = "無 context（對照）" if s["k"] is None else f"k={s['k']}"
        top1 = "—" if s["k"] is None else f"{s['top1']}/{s['total']}"
        recall = "—" if s["k"] is None else f"{s['recall']}/{s['total']}"
        out.append(f"| {name} | {top1} | {recall} | "
                   f"**{s['correct']}/{s['total']}** | {s['avg_answer_chars']} | "
                   f"{s['avg_context_chars']} | "
                   f"{s['input_tokens']}／{s['output_tokens']} |")

    out += ["", "> top-1 命中是同一次檢索算出來的，三組 k 值必然相同——"
                "會變的是 recall@k 與答案正確率。", ""]

    out += ["## 二、檢索 × 答案 交叉表", ""]
    for k, rows in detail_rows.items():
        cells = cross_table(rows)
        out += [f"### k={k}", "",
                "| | 答案正確 | 答案錯誤 |", "|---|---|---|",
                f"| 檢索有撈到預期條文 | {cells['hit_correct']} | "
                f"**{cells['hit_wrong']}** |",
                f"| 檢索沒撈到 | **{cells['miss_correct']}** | "
                f"{cells['miss_wrong']} |", "",
                "右上角＝檢索做對了、生成搞砸了；左下角＝檢索沒撈到卻答對了"
                "（要逐題看是模型自己會，還是別的條文剛好也寫了）。", ""]

    out += ["## 三、預期條文的排名 × 答對率", ""]
    for k, rows in detail_rows.items():
        out += [f"### k={k}", "",
                "| 預期條文排名 | 題數 | 答對 |", "|---|---|---|"]
        for rank, (count, correct) in rank_table(rows).items():
            label = "未進 top-k" if rank == 0 else f"第 {rank} 名"
            out.append(f"| {label} | {count} | {correct} |")
        out.append("")

    out += ["## 四、逐題明細", ""]
    for k, rows in detail_rows.items():
        out += [f"### k={k}", "", fmt_detail(rows), ""]

    total_in = sum(s["input_tokens"] for s in summaries)
    total_out = sum(s["output_tokens"] for s in summaries)
    chat_usd = rag_core.chat_cost_usd(total_in, total_out)
    emb_usd = rag_core.embedding_cost_usd(index_info["embedding_tokens"])
    out += ["## 五、成本", "",
            f"- Embedding：本次計費 {index_info['embedding_tokens']} tokens"
            f"（快取命中 {index_info['cached']} 塊）"
            f" ≈ 新台幣 {rag_core.twd(emb_usd):.5f} 元",
            f"- 生成：輸入 {total_in}、輸出 {total_out} tokens"
            f" ≈ 新台幣 {rag_core.twd(chat_usd):.5f} 元",
            f"- 合計 ≈ **新台幣 {rag_core.twd(emb_usd + chat_usd):.5f} 元**", "",
            "刪除 `embeddings_cache.json` 與 `chat_cache.json` 重跑，"
            "即可重現無快取的完整成本。", ""]
    return "\n".join(out)


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    collection, chunks, _, emb_tokens, cached = pipeline.build_index()

    summaries = [summarize(run_group(collection, questions, None), None)]
    detail_rows = {}
    for k in K_VALUES:
        rows = run_group(collection, questions, k)
        detail_rows[k] = rows
        summaries.append(summarize(rows, k))

    index_info = {"chunks": len(chunks), "questions": len(questions),
                  "embedding_tokens": emb_tokens, "cached": cached}
    report = build_report(summaries, detail_rows, index_info)
    RESULT_PATH.write_text(report, encoding="utf-8")  # 先寫檔，再 print

    print(f"報表已寫入 {RESULT_PATH.name}\n")
    for s in summaries:
        name = "無 context" if s["k"] is None else f"k={s['k']}"
        if s["k"] is None:
            top1 = recall = "—"
        else:
            top1 = f"{s['top1']}/{s['total']}"
            recall = f"{s['recall']}/{s['total']}"
        print(f"{name:>12}｜top-1 {top1}｜recall {recall}"
              f"｜答對 {s['correct']}/{s['total']}")


if __name__ == "__main__":
    main()
