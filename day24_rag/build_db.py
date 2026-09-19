# -*- coding: utf-8 -*-
"""建（或重建）本日的 Chroma 資料庫。

    python build_db.py            # 有現成的庫就沿用
    python build_db.py --rebuild  # 砍掉重建（改過語料或切法時用）

首次執行會呼叫 embedding API 計算 59 條的向量（成本見 README）；
沿用 Day 9~11 的快取檔則不重複計費。
"""
import argparse
import time

import chroma_store
import chunkers
import rag_core


def main() -> None:
    parser = argparse.ArgumentParser(description="建立 Day 14 的 Chroma 資料庫")
    parser.add_argument("--rebuild", action="store_true", help="砍掉重建")
    args = parser.parse_args()

    text = rag_core.load_document()
    chunks = chunkers.chunk_by_structure(text)
    articles = chunkers.parse_articles(text)
    print(f"語料：{len(articles)} 條條文，切成 {len(chunks)} 塊")

    vectors, tokens, cached = rag_core.get_embeddings([c["text"] for c in chunks])
    usd = rag_core.embedding_cost_usd(tokens)
    print(f"Embedding：計費 {tokens} tokens、快取命中 {cached} 塊"
          f" ≈ 新台幣 {rag_core.twd(usd):.5f} 元")

    start = time.perf_counter()
    if args.rebuild:
        collection = chroma_store.rebuild(chroma_store.get_client(),
                                          chunks, articles, vectors)
    else:
        collection = chroma_store.open_or_build(chunks, articles, vectors)
    print(f"Collection「{collection.name}」共 {collection.count()} 筆"
          f"，耗時 {time.perf_counter() - start:.2f} 秒")


if __name__ == "__main__":
    main()
