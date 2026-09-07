# -*- coding: utf-8 -*-
"""Day 11 的 ef_search 掃描探針：在「全新行程」載入 50k 庫、把 ef_search 設為
argv 指定值、查同一組 15 題並量延遲，輸出一行 JSON。

為什麼一定要開新行程：實測發現在「剛建完索引的那個行程」裡，modify() 改的
ef_search 不會套用到後續查詢（記憶體中的索引物件抓著舊設定），六個檔位會
全部量到同一個值而不自知；重新載入索引的行程則立即生效。
run_experiment.py 以子行程呼叫本檔，確保每個檔位量到的是真的那個 ef。
"""
import json
import sys
import time

import numpy as np

import chroma_store
import rag_core
from run_experiment import (TOP_K, LATENCY_REPEAT, expected_article, is_hit,
                            load_questions, top1_of)


def main() -> None:
    ef = int(sys.argv[1])
    client = chroma_store.get_client()
    collection = client.get_collection(chroma_store.COLLECTION_50K)
    collection.modify(configuration={"hnsw": {"ef_search": ef}})

    questions, expected_by_no = load_questions()
    question_matrix, _, _ = rag_core.get_embeddings(
        [q["question"] for q in questions])

    result = collection.query(query_embeddings=question_matrix.tolist(),
                              n_results=TOP_K)
    hits = synth = 0
    for i, item in enumerate(questions):
        meta, _ = top1_of(result, i)
        hits += is_hit(meta, expected_article(item, expected_by_no))
        synth += meta is not None and meta.get("kind") == "synth"

    samples = []
    for _ in range(LATENCY_REPEAT):
        for row in question_matrix:
            start = time.perf_counter()
            collection.query(query_embeddings=[row.tolist()], n_results=TOP_K)
            samples.append((time.perf_counter() - start) * 1000)

    print(json.dumps({"ef": ef, "hits": hits, "synth": synth,
                      "median_ms": round(float(np.median(samples)), 2),
                      "p95_ms": round(float(np.percentile(samples, 95)), 2)}))


if __name__ == "__main__":
    main()
