# -*- coding: utf-8 -*-
"""Day 8 小實驗：用 10 個測試問題（5 個直白、5 個口語）實測 embedding 檢索的命中率。

判定標準：top-1 條文是否等於 questions.json 標注的預期條文。
執行結果同時輸出到終端機與 experiment_result.md（Markdown 表格，可直接貼進文章）。
"""
import json
from pathlib import Path

import rag_core

BASE_DIR = Path(__file__).parent
QUESTIONS_PATH = BASE_DIR / "questions.json"
RESULT_PATH = BASE_DIR / "experiment_result.md"


def main() -> None:
    articles = rag_core.load_articles()
    vectors, index_tokens, _ = rag_core.get_embeddings(
        [article["text"] for article in articles]
    )
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    rows = []
    misses = []
    question_tokens = 0
    for item in questions:
        results, used_tokens = rag_core.search(item["question"], articles, vectors, top_k=3)
        question_tokens += used_tokens
        top1 = results[0]
        hit = top1["title"].startswith(item["expected"])
        rows.append({**item, "top1": top1["title"], "score": top1["score"], "hit": hit})
        if not hit:
            misses.append((item, results))

    # --- Markdown 表格：可直接貼進文章 ---
    lines = [
        "| # | 類型 | 問題 | Top-1 條文 | 相似度 | 命中 |",
        "|---|------|------|-----------|--------|------|",
    ]
    for row in rows:
        mark = "✅" if row["hit"] else "❌"
        lines.append(f"| {row['id']} | {row['type']} | {row['question']} "
                     f"| {row['top1']} | {row['score']:.4f} | {mark} |")

    # --- 分類統計 ---
    for label in ("直白", "口語"):
        subset = [row for row in rows if row["type"] == label]
        hit_count = sum(row["hit"] for row in subset)
        lines.append(f"\n{label}問法命中率：{hit_count}/{len(subset)}")
    total_hits = sum(row["hit"] for row in rows)
    lines.append(f"總命中率：{total_hits}/{len(rows)}")

    # --- 沒中的題目：把 top-3 攤開，這是文章「卡關段」的素材 ---
    if misses:
        lines.append("\n## 沒命中的題目（top-3 攤開看）\n")
        for item, results in misses:
            lines.append(f"**Q{item['id']}：{item['question']}**（預期 {item['expected']}）")
            for rank, result in enumerate(results, start=1):
                lines.append(f"{rank}. [{result['score']:.4f}] {result['title']}")
            lines.append("")

    # --- 成本統計 ---
    total_tokens = index_tokens + question_tokens
    usd = rag_core.cost_usd(total_tokens)
    lines.append(f"\n本次 API 消耗：{total_tokens} tokens"
                 f" ≈ {usd:.6f} 美元 ≈ 新台幣 {usd * rag_core.USD_TO_TWD:.4f} 元"
                 f"（快取命中的部分不計費）")

    report = "\n".join(lines)
    print(report)
    RESULT_PATH.write_text(report + "\n", encoding="utf-8")
    print(f"\n結果已寫入 {RESULT_PATH.name}，表格可直接貼進文章")


if __name__ == "__main__":
    main()
