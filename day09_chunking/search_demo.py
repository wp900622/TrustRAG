# -*- coding: utf-8 -*-
"""Day 9 互動 demo：選一種切法，輸入問題，看 top-3 chunk、涵蓋條文與相似度分數。

用法：
    python search_demo.py 特休沒休完會怎樣
    python search_demo.py --chunker fixed 特休沒休完會怎樣
    python search_demo.py                     # 互動模式，空行離開
"""
import argparse

import chunkers
import rag_core


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 9 Chunking 檢索 demo")
    parser.add_argument("question", nargs="*", help="要查詢的問題；不給就進入互動模式")
    parser.add_argument("--chunker", choices=list(chunkers.CHUNKERS), default="structure",
                        help="切法：fixed（固定字數）、overlap（固定+重疊）、structure（依條文，預設）")
    args = parser.parse_args()

    name, chunk_func = chunkers.CHUNKERS[args.chunker]
    text = rag_core.load_document()
    articles = chunkers.parse_articles(text)
    chunks = chunk_func(text)
    vectors, used_tokens, cached = rag_core.get_embeddings([c["text"] for c in chunks])
    print(f"切法：{name}｜chunk 數：{len(chunks)}"
          f"｜本次索引消耗 {used_tokens} tokens（快取命中 {cached} 塊）")

    def ask(question: str) -> None:
        results, _ = rag_core.search(question, chunks, vectors, top_k=3)
        for rank, result in enumerate(results, start=1):
            label = chunkers.coverage_label(result, articles)
            preview = result["text"][:40].replace("\n", " ")
            print(f"  {rank}. [{result['score']:.4f}] {label}｜{preview}…")

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
