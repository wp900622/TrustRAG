# -*- coding: utf-8 -*-
"""Day 11 實驗：同一份語料、同一組 15 題，把「索引」換成「向量資料庫」（Chroma）。

四個實驗：
1. 換庫對照：59 條入 Chroma（cosine），top-1 是否與 Day 10 Flat 逐題一致
2. metadata 過濾：無過濾／過濾到正確章／過濾到錯誤章，三種條件的命中率
3. 規模與持久化：59＋50,000 塊——建庫時間、磁碟大小、冷開啟（子行程實測）、
   ef_search 掃描的命中率與查詢延遲（Chroma 底層就是 Day 10 的 HNSW）
4. 增量：add／delete 一條條文的耗時（Day 10 對照：加一塊＝重建 265 秒）

命中判定沿用 Day 9 的同一把尺：top-1 的字元範圍是否與預期條文範圍重疊；
top-1 若是合成向量直接算沒中。延遲量的是整個 query()——含把原文與 metadata
從 SQLite 撈回來，不是純向量比對；這正是「資料庫」與「索引」的成本差異。
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import add_demo
import chroma_store
import chromadb
import chunkers
import rag_core

BASE_DIR = Path(__file__).parent
QUESTIONS_PATH = BASE_DIR / "questions.json"
RESULT_PATH = BASE_DIR / "experiment_result.md"

EF_SEARCH_VALUES = (8, 16, 32, 64, 100, 128)  # 100＝Chroma 預設；其餘對齊 Day 10 檔位
LATENCY_REPEAT = 10   # 15 題 × 10 次＝每檔位 150 個延遲樣本
TOP_K = 3             # RAG 常見取用量；命中判定只看 top-1

# Day 10 experiment_result.md「逐題明細（Flat 精確檢索）」的 top-1 條號，
# 用來逐題驗證「換庫不換答案」（來源：day10_vector_index/experiment_result.md）
DAY10_FLAT_TOP1 = {1: 24, 2: 23, 3: 27, 4: 30, 5: 25, 6: 20, 7: 22, 8: 31,
                   9: 47, 10: 29, 11: 18, 12: 11, 13: 40, 14: 7, 15: 54}


def expected_article(item: dict, expected_by_no: dict) -> dict:
    """把題目的「第 N 條」文字換成該條文的字元範圍（判定命中的標準答案）"""
    no = int(chunkers.ARTICLE_NO_PATTERN.search(item["expected"]).group(1))
    return expected_by_no[no]


def top1_of(result: dict, i: int) -> tuple[dict | None, float]:
    """第 i 題的 top-1 metadata 與相似度；過濾條件下查無資料時回 (None, 0)"""
    if not result["ids"][i]:
        return None, 0.0
    return (result["metadatas"][i][0],
            chroma_store.similarity(result["distances"][i][0]))


def is_hit(meta: dict | None, expected: dict) -> bool:
    """Day 9 的尺：top-1 字元範圍與預期條文重疊才算中；合成向量沒有範圍，直接不中"""
    return (meta is not None and meta.get("kind") == "article"
            and chunkers.spans_overlap(meta["start"], meta["end"],
                                       expected["start"], expected["end"]))


def top1_label(meta: dict | None) -> str:
    """top-1 的人看標籤：條號、合成向量、或過濾後查無資料"""
    if meta is None:
        return "（無資料）"
    if meta.get("kind") == "article":
        return f"第 {meta['article_no']} 條"
    return "合成向量"


def mark(hit: bool) -> str:
    return "✅" if hit else "❌"


def bench(collection, question_matrix: np.ndarray,
          repeat: int = LATENCY_REPEAT) -> tuple[float, float]:
    """單題查詢延遲（含 SQLite 撈原文與 metadata）：warm-up 後計時，回 (median, p95) ms"""
    collection.query(query_embeddings=[question_matrix[0].tolist()], n_results=TOP_K)
    samples = []
    for _ in range(repeat):
        for row in question_matrix:
            start = time.perf_counter()
            collection.query(query_embeddings=[row.tolist()], n_results=TOP_K)
            samples.append((time.perf_counter() - start) * 1000)
    return float(np.median(samples)), float(np.percentile(samples, 95))


def run_probe(script: str, *args: str) -> dict:
    """以全新子行程執行探針腳本並收回其 JSON 輸出。

    冷開啟同行程量會被暖快取美化；ef_search 掃描則是剛建完索引的行程裡
    modify() 不會套用到查詢（見 ef_probe.py）——兩者都必須開新行程才量得準。
    """
    proc = subprocess.run(
        [sys.executable, script, *args], cwd=BASE_DIR,
        capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"{script} 失敗：{proc.stderr[-300:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def load_questions() -> tuple[list[dict], dict]:
    """讀題庫與條文範圍對照表（ef_probe.py 也用同一份，判定才是同一把尺）"""
    text = rag_core.load_document()
    articles = chunkers.parse_articles(text)
    expected_by_no = {a["no"]: a for a in articles}
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return questions, expected_by_no


def run_parity(collection, question_matrix, questions, expected_by_no):
    """實驗一：無過濾查 15 題，逐題對照 Day 10 Flat 的 top-1"""
    result = collection.query(query_embeddings=question_matrix.tolist(),
                              n_results=TOP_K)
    rows, hits, consistent = [], 0, 0
    for i, item in enumerate(questions):
        meta, score = top1_of(result, i)
        hit = is_hit(meta, expected_article(item, expected_by_no))
        same = (meta is not None
                and meta.get("article_no") == DAY10_FLAT_TOP1[item["id"]])
        hits += hit
        consistent += same
        rows.append((item, top1_label(meta), score, hit, same))
    return rows, hits, consistent


def run_filter(collection, question_matrix, questions, expected_by_no,
               chapters, chunks):
    """實驗二：無過濾／正確章／錯誤章。錯誤章＝正確章的下一章（循環），固定可重現"""
    rows = []
    totals = {"none": 0, "right": 0, "wrong": 0}
    for i, item in enumerate(questions):
        expected = expected_article(item, expected_by_no)
        right_no = chunkers.chapter_of(expected["start"], chapters)["no"]
        wrong_no = right_no % len(chapters) + 1
        per = {}
        for key, where in (("none", None),
                           ("right", {"chapter_no": right_no}),
                           ("wrong", {"chapter_no": wrong_no})):
            result = collection.query(
                query_embeddings=[question_matrix[i].tolist()],
                n_results=TOP_K, where=where)
            meta, _ = top1_of(result, 0)
            hit = is_hit(meta, expected)
            totals[key] += hit
            per[key] = f"{top1_label(meta)} {mark(hit)}"
        rows.append((item, right_no, wrong_no, per))
    chapter_sizes = {}
    for chunk in chunks:
        chapter_sizes[chunk["chapter_no"]] = chapter_sizes.get(chunk["chapter_no"], 0) + 1
    return rows, totals, chapter_sizes


def run_scale(client, chunks, articles, chunk_matrix):
    """實驗三：50,059 塊——建庫、冷開啟、ef_search 掃描（掃完還原預設值）。

    每個 ef 檔位都交給全新子行程（ef_probe.py）量測：在剛建完索引的行程裡
    modify() 不會套用到查詢，六個檔位會全部量到同一個舊值而不自知。
    """
    collection, build_seconds, reused = chroma_store.ensure_50k(
        client, chunks, articles, chunk_matrix)
    # 建圖參數實讀自 collection（文章的歸因要有實測依據，不能只查文件）
    hnsw_config = collection.configuration_json.get("hnsw", {})
    cold = run_probe("cold_open.py", chroma_store.COLLECTION_50K)
    ef_rows = []
    try:
        for ef in EF_SEARCH_VALUES:
            probe = run_probe("ef_probe.py", str(ef))
            ef_rows.append((probe["ef"], probe["hits"], probe["synth"],
                            probe["median_ms"], probe["p95_ms"]))
    finally:
        # ef_search 的修改會持久化——實驗掃完必須還原，別改掉庫的日常行為
        collection.modify(configuration={"hnsw": {"ef_search": 100}})
    return build_seconds, reused, cold, ef_rows, hnsw_config


def run_incremental(collection):
    """實驗四：add／delete 一條新條文的耗時，用 add_demo 的同一條文與同一題"""
    vectors, used_tokens, cached = rag_core.get_embeddings(
        [add_demo.PET_QUESTION, add_demo.NEW_ARTICLE_TEXT])
    question_vector, article_vector = vectors

    def pet_top1() -> str:
        result = collection.query(query_embeddings=[question_vector.tolist()],
                                  n_results=1)
        meta, score = top1_of(result, 0)
        return f"第 {meta['article_no']} 條 [{score:.4f}]"

    before = pet_top1()
    start = time.perf_counter()
    collection.add(ids=[f"art-{add_demo.NEW_ARTICLE_NO}"],
                   embeddings=[article_vector.tolist()],
                   documents=[add_demo.NEW_ARTICLE_TEXT],
                   metadatas=[add_demo.NEW_ARTICLE_METADATA])
    add_ms = (time.perf_counter() - start) * 1000
    after = pet_top1()
    start = time.perf_counter()
    collection.delete(ids=[f"art-{add_demo.NEW_ARTICLE_NO}"])
    delete_ms = (time.perf_counter() - start) * 1000
    return {"tokens": used_tokens, "cached": cached, "before": before,
            "after": after, "restored": pet_top1(),
            "add_ms": add_ms, "delete_ms": delete_ms}


def render_report(questions, parity, small_latency, filtered, scale,
                  incremental, cost) -> str:
    """把四個實驗的結果組成 Markdown 報表（各段拆開組裝，方便逐段貼文章）"""
    parity_rows, parity_hits, consistent = parity
    filter_rows, filter_totals, chapter_sizes = filtered
    build_seconds, reused, cold, ef_rows, hnsw_config = scale

    lines = [
        "# Day 11 實驗結果：Chroma 向量資料庫（換庫對照、metadata 過濾、規模與增量）",
        "",
        f"環境：Python {platform.python_version()}｜chromadb {chromadb.__version__}"
        f"｜{platform.processor() or platform.machine()}",
        "（延遲絕對值以本機為準；與 Day 10 為不同次量測，跨日比較請看數量級）",
        "", "## 一、換庫對照（59 塊，同 15 題，無過濾）", "",
        "| # | 問題 | top-1 | 相似度 | 命中 | 與 Day 10 Flat 一致 |",
        "|---|------|-------|--------|------|--------------------|"]
    for item, label, score, hit, same in parity_rows:
        lines.append(f"| {item['id']} | {item['question']} | {label} "
                     f"| {score:.4f} | {mark(hit)} | {mark(same)} |")
    lines += [
        "",
        f"命中 {parity_hits}/15；top-1 與 Day 10 Flat 逐題一致 {consistent}/15。",
        f"單題查詢延遲 median {small_latency[0]:.2f} ms、p95 {small_latency[1]:.2f} ms"
        "（含 SQLite 撈原文與 metadata）。",
        "", "## 二、metadata 章別過濾（59 塊）", "",
        "各章塊數：" + "、".join(
            f"第 {no} 章 {count} 塊" for no, count in sorted(chapter_sizes.items())),
        "",
        "| # | 正確章 | 無過濾 | 過濾到正確章 | 過濾到錯誤章 |",
        "|---|--------|--------|--------------|--------------|"]
    for item, right_no, wrong_no, per in filter_rows:
        lines.append(f"| {item['id']} | 第 {right_no} 章 | {per['none']} "
                     f"| {per['right']} | {per['wrong']}（第 {wrong_no} 章） |")
    lines += [
        "",
        f"命中總計：無過濾 {filter_totals['none']}/15、"
        f"過濾到正確章 {filter_totals['right']}/15、"
        f"過濾到錯誤章 {filter_totals['wrong']}/15。",
        "", "## 三、規模與持久化（59＋50,000 塊）", "",
        f"- 建庫（寫入 50,059 塊，含 HNSW 建圖）：{build_seconds:.1f} 秒"
        + ("（沿用既有庫的首次建庫時間）" if reused else ""),
        f"- chroma_db/ 磁碟大小：{chroma_store.db_size_mb():.1f} MB",
        f"- 冷開啟（全新行程實測）：開庫 {cold['open_ms']:.0f} ms、"
        f"首次查詢 {cold['first_query_ms']:.0f} ms（共 {cold['count']:,} 塊）",
        f"- HNSW 設定（實讀自 collection）：M(max_neighbors)="
        f"{hnsw_config.get('max_neighbors')}、ef_construction="
        f"{hnsw_config.get('ef_construction')}、space={hnsw_config.get('space')}"
        "——建庫時只指定 space，其餘為 Chroma 預設",
        "",
        "| ef_search | 命中 | top-1 被合成向量搶走 | 延遲 median (ms) | p95 (ms) |",
        "|-----------|------|----------------------|------------------|----------|"]
    for ef, hits, synth, median_ms, p95_ms in ef_rows:
        note = "（預設值）" if ef == 100 else ""
        lines.append(f"| {ef}{note} | {hits}/15 | {synth} 題 "
                     f"| {median_ms:.2f} | {p95_ms:.2f} |")
    lines += [
        "",
        "此表的兩個偏誤：(1) Chroma 建圖是多執行緒且無法固定，剛灌完大量資料後"
        "還有背景整理（WAL 併入索引）——本表為建庫靜置後的量測，重跑時逐題明細"
        "會在附近浮動，不像 Day 10 的 FAISS 能以單執行緒＋固定 seed 逐題重現；"
        "(2) 延遲含 SQLite 撈原文與 metadata 的固定成本，與 Day 10 的純索引"
        "查詢延遲不可直接相比。"]
    lines += [
        "", "## 四、增量新增與刪除（59 塊庫，操作「第 60 條（寵物友善辦公試辦）」）", "",
        "| 動作 | 耗時 | 查「可以帶寵物來上班嗎？」的 top-1 |",
        "|------|------|-----------------------------------|",
        f"| 新增前 | — | {incremental['before']} |",
        f"| add() 第 60 條 | {incremental['add_ms']:.0f} ms | {incremental['after']} |",
        f"| delete() 第 60 條 | {incremental['delete_ms']:.0f} ms | {incremental['restored']} |",
        "",
        "Day 10 對照：FAISS 索引沒有 add 一筆的選項——50,059 塊重建一次 265 秒。",
        "", cost]
    return "\n".join(lines)


def main() -> None:
    # --- 準備：語料、章別、題目、向量（走快取，成本照系列規則計） ---
    text = rag_core.load_document()
    chunks = chunkers.chunk_by_structure(text)
    articles = chunkers.parse_articles(text)
    chapters = chunkers.parse_chapters(text)
    questions, expected_by_no = load_questions()

    chunk_matrix, index_tokens, index_cached = rag_core.get_embeddings(
        [c["text"] for c in chunks])
    question_matrix, question_tokens, question_cached = rag_core.get_embeddings(
        [q["question"] for q in questions])

    client = chroma_store.get_client()
    small = chroma_store.rebuild_small(client, chunks, articles, chunk_matrix)

    parity = run_parity(small, question_matrix, questions, expected_by_no)
    small_latency = bench(small, question_matrix)
    filtered = run_filter(small, question_matrix, questions, expected_by_no,
                          chapters, chunks)
    scale = run_scale(client, chunks, articles, chunk_matrix)
    incremental = run_incremental(small)

    total_billed = index_tokens + question_tokens + incremental["tokens"]
    total_cached = index_cached + question_cached + incremental["cached"]
    usd = rag_core.cost_usd(total_billed)
    cost = (f"本次 API 計費：{total_billed} tokens"
            f" ≈ {usd:.6f} 美元 ≈ 新台幣 {usd * rag_core.USD_TO_TWD:.4f} 元"
            f"（快取命中 {total_cached} 筆未計費；"
            f"50,000 個合成向量由本機生成，0 元；"
            f"刪除 embeddings_cache.json 重跑即可重現無快取的完整成本）")

    report_text = render_report(questions, parity, small_latency, filtered,
                                scale, incremental, cost)
    # 先寫檔再印：報表含 ✅/❌，stdout 被重導向時可能因編碼炸掉，別讓檔案陪葬
    RESULT_PATH.write_text(report_text + "\n", encoding="utf-8")
    print(report_text)
    print(f"\n結果已寫入 {RESULT_PATH.name}，表格可直接貼進文章")


if __name__ == "__main__":
    main()
