# -*- coding: utf-8 -*-
"""Day 9 實驗：三種切法 × 同一組 15 題（10 題沿用 Day 8＋5 題新增），實測 top-1 命中率。

命中判定：top-1 chunk 的「字元範圍」是否與預期條文的範圍重疊。三種切法的邊界
都不同，直接比條號沒有意義；用範圍重疊，三種切法才是同一把尺（判定從寬：沾到就算）。

第二指標「純度」：top-1 命中時，chunk 範圍內屬於預期條文的字元比例。
命中率只回答「找不找得到」，純度回答「找到的那塊有多乾淨」——大塊容易沾到答案
（命中率高估），但沾到的塊裡可能大半是無關內容，這個代價要量出來。
執行結果同時輸出到終端機與 experiment_result.md（Markdown 表格，可直接貼進文章）。
"""
import json
from pathlib import Path

import chunkers
import rag_core

BASE_DIR = Path(__file__).parent
QUESTIONS_PATH = BASE_DIR / "questions.json"
RESULT_PATH = BASE_DIR / "experiment_result.md"


def run_strategy(name: str, chunk_func, text: str,
                 articles: list[dict], questions: list[dict]) -> dict:
    """用指定切法建索引、跑完整組問題，回傳逐題結果與用量統計"""
    chunks = chunk_func(text)
    vectors, index_tokens, index_cached = rag_core.get_embeddings(
        [c["text"] for c in chunks])
    expected_by_no = {a["no"]: a for a in articles}

    rows, misses = [], []
    question_tokens = 0
    for item in questions:
        results, used = rag_core.search(item["question"], chunks, vectors, top_k=3)
        question_tokens += used
        expected_no = int(chunkers.ARTICLE_NO_PATTERN.search(item["expected"]).group(1))
        expected = expected_by_no[expected_no]
        top1 = results[0]
        hit = chunkers.spans_overlap(top1["start"], top1["end"],
                                     expected["start"], expected["end"])
        if hit:
            # 純度＝chunk 與預期條文重疊的字元數 ÷ chunk 總字元數（以原文範圍計）
            overlap_chars = (min(top1["end"], expected["end"])
                             - max(top1["start"], expected["start"]))
            purity = overlap_chars / (top1["end"] - top1["start"])
        else:
            purity = None
        rows.append({**item, "top1_label": chunkers.coverage_label(top1, articles),
                     "score": top1["score"], "hit": hit, "purity": purity})
        if not hit:
            misses.append((item, results))

    avg_len = sum(len(c["text"]) for c in chunks) / len(chunks)
    return {"name": name, "chunk_count": len(chunks), "avg_len": avg_len,
            "rows": rows, "misses": misses, "index_tokens": index_tokens,
            "index_cached": index_cached, "question_tokens": question_tokens}


def hit_stat(rows: list[dict], **filters) -> str:
    """依條件篩選後的命中統計，例如 hit_stat(rows, group="沿用", type="口語")"""
    subset = [r for r in rows if all(r[key] == value for key, value in filters.items())]
    return f"{sum(r['hit'] for r in subset)}/{len(subset)}"


def avg_purity(rows: list[dict]) -> str:
    """命中題的平均純度；全數未命中時回傳「—」"""
    purities = [r["purity"] for r in rows if r["hit"]]
    return f"{sum(purities) / len(purities):.0%}" if purities else "—"


def main() -> None:
    text = rag_core.load_document()
    articles = chunkers.parse_articles(text)
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    reports = [
        run_strategy(name, chunk_func, text, articles, questions)
        for name, chunk_func in chunkers.CHUNKERS.values()
    ]

    lines = [f"# Day 9 實驗結果：三種切法 × {len(questions)} 題\n"]

    # --- 總表：三種切法並排比較 ---
    lines += [
        "| 切法 | chunk 數 | 平均字數 | 沿用10題 | 其中直白 | 其中口語 | 新增5題 | 全部 | 命中純度 |",
        "|------|---------|---------|---------|---------|---------|--------|------|---------|",
    ]
    for report in reports:
        rows = report["rows"]
        lines.append(
            f"| {report['name']} | {report['chunk_count']} | {report['avg_len']:.0f} "
            f"| {hit_stat(rows, group='沿用')} | {hit_stat(rows, group='沿用', type='直白')} "
            f"| {hit_stat(rows, group='沿用', type='口語')} | {hit_stat(rows, group='新增')} "
            f"| {hit_stat(rows)} | {avg_purity(rows)} |"
        )

    # --- 各切法逐題明細 ---
    for report in reports:
        lines += [
            f"\n## 切法：{report['name']}"
            f"（{report['chunk_count']} 塊、平均 {report['avg_len']:.0f} 字）\n",
            "| # | 組 | 類型 | 問題 | Top-1 涵蓋 | 相似度 | 命中 | 純度 |",
            "|---|----|------|------|-----------|--------|------|------|",
        ]
        for row in report["rows"]:
            mark = "✅" if row["hit"] else "❌"
            purity_text = f"{row['purity']:.0%}" if row["hit"] else "—"
            lines.append(f"| {row['id']} | {row['group']} | {row['type']} "
                         f"| {row['question']} | {row['top1_label']} "
                         f"| {row['score']:.4f} | {mark} | {purity_text} |")

        # 沒中的題目：把 top-3 攤開，這是文章分析段的素材
        if report["misses"]:
            lines.append(f"\n### 沒命中的題目（top-3 攤開看）\n")
            for item, results in report["misses"]:
                lines.append(f"**Q{item['id']}：{item['question']}**（預期 {item['expected']}）")
                for rank, result in enumerate(results, start=1):
                    label = chunkers.coverage_label(result, articles)
                    lines.append(f"{rank}. [{result['score']:.4f}] {label}")
                lines.append("")

    # --- 成本統計：計費與快取命中分開列，避免「0 元」被誤讀成實驗不用錢 ---
    total_billed = sum(r["index_tokens"] + r["question_tokens"] for r in reports)
    total_cached = sum(r["index_cached"] for r in reports)
    usd = rag_core.cost_usd(total_billed)
    lines.append(f"\n本次 API 計費：{total_billed} tokens"
                 f" ≈ {usd:.6f} 美元 ≈ 新台幣 {usd * rag_core.USD_TO_TWD:.4f} 元"
                 f"（另有 {total_cached} 塊索引向量命中本機快取、未計費；"
                 f"刪除 embeddings_cache.json 重跑即可重現無快取的完整成本）")

    report_text = "\n".join(lines)
    print(report_text)
    RESULT_PATH.write_text(report_text + "\n", encoding="utf-8")
    print(f"\n結果已寫入 {RESULT_PATH.name}，表格可直接貼進文章")


if __name__ == "__main__":
    main()
