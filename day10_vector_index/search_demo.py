# -*- coding: utf-8 -*-
"""Day 10 互動 demo：檢索改走 FAISS（精確 Flat 索引），輸入問題看 top-3 與相似度。

用法：
    python search_demo.py 特休沒休完會怎樣
    python search_demo.py --distractors 50000 特休沒休完會怎樣  # 混入合成向量再查
    python search_demo.py                     # 互動模式，空行離開

跟 Day 9 的 demo 對照著玩：同一份語料、同一個問題，答案應該一模一樣——
Flat 是精確檢索，換引擎不換答案。加上 --distractors 後可以驗證
「5 萬個合成向量混進索引，真實問題的 top-3 仍然是真實條文」。
"""
import argparse

import numpy as np

import chunkers
import rag_core
import synth_vectors


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 10 FAISS 檢索 demo")
    parser.add_argument("question", nargs="*", help="要查詢的問題；不給就進入互動模式")
    parser.add_argument("--distractors", type=int, default=0,
                        help="混入索引的合成干擾向量數（預設 0，最多 50000）")
    args = parser.parse_args()

    text = rag_core.load_document()
    articles = chunkers.parse_articles(text)
    chunks = chunkers.chunk_by_structure(text)
    vectors, used_tokens, cached = rag_core.get_embeddings([c["text"] for c in chunks])

    if args.distractors > 0:
        n = min(args.distractors, synth_vectors.N_CENTROIDS * synth_vectors.PER_CENTROID)
        distractors = synth_vectors.make_distractors(dim=vectors.shape[1])[:n]
        vectors = np.vstack([vectors, distractors])  # 真實 chunk 佔 id 0～58

    index = rag_core.build_flat(vectors)
    print(f"索引：{index.ntotal:,} 塊（真實 {len(chunks)}＋合成 {index.ntotal - len(chunks):,}）"
          f"｜本次索引消耗 {used_tokens} tokens（快取命中 {cached} 塊）")

    def ask(question: str) -> None:
        ids, scores, _ = rag_core.search(question, index, top_k=3)
        for rank, (chunk_id, score) in enumerate(zip(ids, scores), start=1):
            if chunk_id < len(chunks):
                label = chunkers.coverage_label(chunks[chunk_id], articles)
                preview = chunks[chunk_id]["text"][:40].replace("\n", " ")
                print(f"  {rank}. [{score:.4f}] {label}｜{preview}…")
            else:
                print(f"  {rank}. [{score:.4f}] 合成向量 #{chunk_id}")

    if args.question:
        ask(" ".join(args.question))
        return

    print("輸入問題（直接按 Enter 離開）：")
    while True:
        question = input("> ").strip()
        if not question:
            break
        ask(question)


if __name__ == "__main__":
    main()
