# -*- coding: utf-8 -*-
"""Day 10 實驗：50,059 塊向量下，精確（Flat）vs 近似（HNSW），速度差幾倍、代價是什麼。

索引內容：59 條真實條文向量（id 0～58）＋ 50,000 個合成干擾向量（synth_vectors.py）。
真實問題只有 15 題，命中判定沿用 Day 9 的同一把尺：top-1 的字元範圍是否與預期
條文範圍重疊；top-1 若是合成向量直接算沒中。

三個實驗，全部用同一組 15 題、同一份索引資料：
1. 查詢延遲：單題查詢的 median / p95。計時只含 index.search()，
   問題向量事先算好——量的是索引本身，不是 API 往返
2. recall：以 Flat 的 top-k 為標準答案，量 HNSW 各檔 efSearch 的 recall@1/3/10
3. Day 9 命中率：驗證換了引擎答案變不變、近似檢索從哪個 efSearch 檔位開始掉題

執行結果同時輸出到終端機與 experiment_result.md（Markdown 表格，可直接貼進文章）。
"""
import json
import os
import platform
import time
from pathlib import Path

import faiss
import numpy as np

import chunkers
import rag_core
import synth_vectors

BASE_DIR = Path(__file__).parent
QUESTIONS_PATH = BASE_DIR / "questions.json"
RESULT_PATH = BASE_DIR / "experiment_result.md"

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
EF_SEARCH_VALUES = (8, 16, 32, 64, 128, 256)
TOP_K = 10           # recall@10 需要 10 個結果，延遲量測也統一用 top-10
LATENCY_REPEAT = 20  # 每題重複次數：15 題 × 20 次 ＝ 每個設定 300 個延遲樣本


def bench(index: faiss.Index, queries: np.ndarray,
          repeat: int = LATENCY_REPEAT) -> tuple[float, float]:
    """量單題查詢延遲：先 warm-up，再逐題計時，回傳 (median_ms, p95_ms)"""
    index.search(queries[:2], TOP_K)  # warm-up：讓快取、執行緒池就位
    samples = []
    for _ in range(repeat):
        for i in range(len(queries)):
            start = time.perf_counter()
            index.search(queries[i:i + 1], TOP_K)
            samples.append((time.perf_counter() - start) * 1000)
    return float(np.median(samples)), float(np.percentile(samples, 95))


def day9_hits(ids_matrix: np.ndarray, chunks: list[dict], questions: list[dict],
              expected_by_no: dict) -> list[bool]:
    """Day 9 的同一把尺：每題 top-1 是否命中預期條文（合成向量直接算沒中）"""
    hits = []
    for row, item in zip(ids_matrix, questions):
        top1 = int(row[0])
        expected_no = int(chunkers.ARTICLE_NO_PATTERN.search(item["expected"]).group(1))
        expected = expected_by_no[expected_no]
        hits.append(top1 < len(chunks) and chunkers.spans_overlap(
            chunks[top1]["start"], chunks[top1]["end"],
            expected["start"], expected["end"]))
    return hits


def top1_label(chunk_id: int, chunks: list[dict], articles: list[dict]) -> str:
    """把 top-1 的向量 id 轉成人看的標籤：真實 chunk 給條號，合成向量直說"""
    if chunk_id < len(chunks):
        return chunkers.coverage_label(chunks[chunk_id], articles)
    return f"合成向量 #{chunk_id}"


def peak_rss_mb() -> float | None:
    """量本行程的峰值實體記憶體（MB）；量不到就回 None，不影響實驗"""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            # 64 位元下必須明宣告型別：pseudo-handle（-1）當 int 傳會被截斷、呼叫失敗
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.windll.psapi
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                          ctypes.byref(pmc), ctypes.sizeof(pmc)):
                return pmc.PeakWorkingSetSize / 1e6
            return None
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux 單位是 KB、macOS 是 bytes
        return usage / 1e3 if platform.system() == "Linux" else usage / 1e6
    except Exception:
        return None


def main() -> None:
    # --- 準備資料：59 條真實 chunk ＋ 15 題問題向量（走快取，成本照 Day 9 規則計） ---
    text = rag_core.load_document()
    chunks = chunkers.chunk_by_structure(text)
    articles = chunkers.parse_articles(text)
    expected_by_no = {a["no"]: a for a in articles}
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    chunk_matrix, index_tokens, index_cached = rag_core.get_embeddings(
        [c["text"] for c in chunks])
    question_matrix, question_tokens, question_cached = rag_core.get_embeddings(
        [q["question"] for q in questions])

    distractors = synth_vectors.make_distractors(dim=chunk_matrix.shape[1])
    all_vectors = np.vstack([chunk_matrix, distractors])  # 真實向量佔 id 0～58
    n_total = len(all_vectors)

    # 先記下硬體執行緒數：omp_get_max_threads() 會回傳「當下設定值」，
    # 設成 1 之後再問就只會拿到 1
    max_threads = faiss.omp_get_max_threads()

    # --- 建 Flat 索引並先量完延遲，才建 HNSW：
    #     HNSW 建索引是幾十秒的全核運算，會讓 CPU 降頻，
    #     若 Flat 基準在那之後才量，會被灌水成近兩倍慢，倍率就不誠實 ---
    start = time.perf_counter()
    flat = rag_core.build_flat(all_vectors)
    flat_build_s = time.perf_counter() - start

    # --- 標準答案：Flat 的 top-10（精確暴力掃描，保證正確） ---
    faiss.omp_set_num_threads(1)
    _, gt_ids = flat.search(question_matrix, TOP_K)
    flat_hits = day9_hits(gt_ids, chunks, questions, expected_by_no)

    # 驗證「合成向量沒有污染答案」：Flat 的 top-1/3/10 裡混進幾個合成向量
    synth_in_top = {k: int(np.sum(gt_ids[:, :k] >= len(chunks)))
                    for k in (1, 3, 10)}

    # --- 實驗一：查詢延遲（Flat） ---
    latency_rows = []
    flat_med, flat_p95 = bench(flat, question_matrix)
    latency_rows.append(("Flat（精確）", "1", flat_med, flat_p95))
    faiss.omp_set_num_threads(max_threads)
    med, p95 = bench(flat, question_matrix)
    latency_rows.append(("Flat（精確）", f"{max_threads}", med, p95))
    faiss.omp_set_num_threads(1)

    # --- 建 HNSW：刻意用單執行緒。多執行緒建圖每次插入順序不同，
    #     建出來的圖不一樣、掉的題也不一樣（實測過），數字就不可重現；
    #     單執行緒建圖是決定性的，代價是建索引從約 40 秒變成約 2 分鐘。
    #     建完歇 10 秒讓 CPU 頻率回穩，再量延遲 ---
    faiss.omp_set_num_threads(1)
    start = time.perf_counter()
    hnsw = rag_core.build_hnsw(all_vectors, m=HNSW_M,
                               ef_construction=HNSW_EF_CONSTRUCTION)
    hnsw_build_s = time.perf_counter() - start
    time.sleep(10)

    # --- 實驗二＋三：HNSW 各檔 efSearch 的延遲、recall、Day 9 命中率 ---
    hnsw_rows = []
    drops = []  # (efSearch, 掉的題, top-1 變成了什麼)：Flat 有中、HNSW 沒中的題
    for ef in EF_SEARCH_VALUES:
        hnsw.hnsw.efSearch = ef
        med, p95 = bench(hnsw, question_matrix)
        _, ids = hnsw.search(question_matrix, TOP_K)
        recall = {k: float(np.mean([len(set(ids[i, :k]) & set(gt_ids[i, :k])) / k
                                    for i in range(len(questions))]))
                  for k in (1, 3, 10)}
        hits = day9_hits(ids, chunks, questions, expected_by_no)
        for i, (flat_hit, hnsw_hit) in enumerate(zip(flat_hits, hits)):
            if flat_hit and not hnsw_hit:
                drops.append((ef, questions[i],
                              top1_label(int(ids[i, 0]), chunks, articles)))
        hnsw_rows.append((ef, med, p95, recall, sum(hits)))

    # --- 報表 ---
    data_mb = all_vectors.nbytes / 1e6
    rss = peak_rss_mb()
    lines = [f"# Day 10 實驗結果：{n_total:,} 塊向量，Flat（精確）vs HNSW（近似）\n"]

    lines += [
        "## 實驗設定\n",
        f"- 索引內容：{len(chunks)} 條真實條文向量＋{len(distractors):,} 個合成干擾"
        f"向量（隨機質心＋擾動，固定 seed，見 `synth_vectors.py`），共 {n_total:,} 塊、"
        f"{all_vectors.shape[1]} 維 float32＝{data_mb:.0f} MB",
        f"- 測試問題：Day 9 的同一組 {len(questions)} 題；判定沿用同一把尺"
        "（top-1 範圍與預期條文重疊）",
        f"- 建索引：Flat {flat_build_s:.2f} 秒；HNSW（M={HNSW_M}、"
        f"efConstruction={HNSW_EF_CONSTRUCTION}）{hnsw_build_s:.0f} 秒",
        f"- 延遲量測：單題查詢、top-{TOP_K}，每個設定 {len(questions)} 題 × "
        f"{LATENCY_REPEAT} 次＝{len(questions) * LATENCY_REPEAT} 個樣本，"
        "計時只含 `index.search()`（問題向量事先算好）",
        f"- 機器：{os.environ.get('PROCESSOR_IDENTIFIER', platform.processor())}、"
        f"{os.cpu_count()} 執行緒；Python {platform.python_version()}、"
        f"faiss-cpu {faiss.__version__}、numpy {np.__version__}"
        + (f"；峰值記憶體 {rss:.0f} MB" if rss else ""),
        "",
        "## 合成向量有沒有污染答案\n",
        f"15 題的 Flat 精確結果中，合成向量在 top-1 出現 {synth_in_top[1]} 次、"
        f"top-3（45 個位置）出現 {synth_in_top[3]} 次、"
        f"top-10（150 個位置）出現 {synth_in_top[10]} 次。",
        f"Flat 的 Day 9 題組命中率：{sum(flat_hits)}/{len(questions)}"
        "（與 Day 9 結構化切法在 59 塊索引下的結果對照，驗證換引擎、加干擾後答案變不變）。",
        "",
        "## 表一：單題查詢延遲\n",
        "| 引擎 | 執行緒 | 中位數 (ms) | p95 (ms) | 相對 Flat 單執行緒 |",
        "|------|--------|------------|----------|-------------------|",
    ]
    for name, threads, med, p95 in latency_rows:
        lines.append(f"| {name} | {threads} | {med:.2f} | {p95:.2f} "
                     f"| {flat_med / med:.1f}x |")
    for ef, med, p95, _, _ in hnsw_rows:
        lines.append(f"| HNSW efSearch={ef} | 1 | {med:.3f} | {p95:.3f} "
                     f"| {flat_med / med:.0f}x |")

    lines += [
        "",
        "## 表二：HNSW 的代價——recall 與命中率 vs efSearch\n",
        "recall@k＝HNSW 的 top-k 與 Flat 標準答案 top-k 的重合比例（15 題平均）。\n",
        "| efSearch | 中位數 (ms) | recall@1 | recall@3 | recall@10 | Day 9 題組命中 |",
        "|----------|------------|----------|----------|-----------|---------------|",
        f"| Flat（對照） | {flat_med:.2f} | 1.00 | 1.00 | 1.00 "
        f"| {sum(flat_hits)}/{len(questions)} |",
    ]
    for ef, med, p95, recall, hits in hnsw_rows:
        lines.append(f"| {ef} | {med:.3f} | {recall[1]:.2f} | {recall[3]:.2f} "
                     f"| {recall[10]:.2f} | {hits}/{len(questions)} |")

    # 掉題解剖：Flat 有中、HNSW 沒中的題，top-1 變成了什麼
    lines += ["", "## 掉題解剖：Flat 有中、HNSW 沒中的題\n"]
    if drops:
        lines += ["| efSearch | 題目 | 預期 | HNSW top-1 變成 |",
                  "|----------|------|------|----------------|"]
        for ef, item, label in drops:
            lines.append(f"| {ef} | Q{item['id']}：{item['question']} "
                         f"| {item['expected']} | {label} |")
    else:
        lines.append("（無：所有 efSearch 檔位下，Flat 有中的題 HNSW 也都中）")

    # --- 逐題明細（Flat 精確結果，供對照 Day 9） ---
    scores, _ = flat.search(question_matrix, 1)
    lines += ["", "## 逐題明細（Flat 精確檢索）\n",
              "| # | 問題 | Top-1 涵蓋 | 相似度 | 命中 |",
              "|---|------|-----------|--------|------|"]
    for i, item in enumerate(questions):
        mark = "✅" if flat_hits[i] else "❌"
        label = top1_label(int(gt_ids[i, 0]), chunks, articles)
        lines.append(f"| {item['id']} | {item['question']} | {label} "
                     f"| {scores[i, 0]:.4f} | {mark} |")

    # --- 成本統計：計費與快取命中分開列，合成向量 0 元這件事明講 ---
    total_billed = index_tokens + question_tokens
    usd = rag_core.cost_usd(total_billed)
    lines.append(f"\n本次 API 計費：{total_billed} tokens"
                 f" ≈ {usd:.6f} 美元 ≈ 新台幣 {usd * rag_core.USD_TO_TWD:.4f} 元"
                 f"（快取命中 {index_cached + question_cached} 筆未計費；"
                 f"50,000 個合成向量由本機生成，0 元）")

    report_text = "\n".join(lines)
    print(report_text)
    RESULT_PATH.write_text(report_text + "\n", encoding="utf-8")
    print(f"\n結果已寫入 {RESULT_PATH.name}，表格可直接貼進文章")


if __name__ == "__main__":
    main()
