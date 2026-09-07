# -*- coding: utf-8 -*-
"""Day 11 建庫：把 59 條條文（＋選配 5 萬合成向量）寫進本機持久化的 Chroma。

用法：
    python build_db.py              # 建 work-rules（59 條，含章別 metadata）
    python build_db.py --with-50k   # 另建 work-rules-50k（59＋50,000，量建庫時間）

建完就結束。之後 search_demo.py／add_demo.py／run_experiment.py 直接開庫——
「建一次、每次開檔即用」正是 Day 10 的 FAISS 索引做不到的第一件事
（FAISS 每次行程啟動都得重新灌向量、重新建圖）。
"""
import argparse
import time

import chroma_store
import chunkers
import rag_core


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 11 Chroma 建庫")
    parser.add_argument("--with-50k", action="store_true",
                        help="另建 59＋50,000 塊的 work-rules-50k（約 1～3 分鐘）")
    args = parser.parse_args()

    text = rag_core.load_document()
    chunks = chunkers.chunk_by_structure(text)
    articles = chunkers.parse_articles(text)
    vectors, used_tokens, cached = rag_core.get_embeddings([c["text"] for c in chunks])
    print(f"語料：{len(chunks)} 塊｜本次消耗 {used_tokens} tokens（快取命中 {cached} 塊）")

    client = chroma_store.get_client()
    start = time.perf_counter()
    collection = chroma_store.rebuild_small(client, chunks, articles, vectors)
    print(f"work-rules 建庫完成：{collection.count()} 塊，"
          f"{(time.perf_counter() - start) * 1000:.0f} ms")

    if args.with_50k:
        collection, seconds, reused = chroma_store.ensure_50k(
            client, chunks, articles, vectors)
        if reused:
            print(f"work-rules-50k：{collection.count():,} 塊"
                  f"（沿用既有庫；當時建庫 {seconds:.1f} 秒）")
        else:
            print(f"work-rules-50k 建庫完成：{collection.count():,} 塊，{seconds:.1f} 秒")

    print(f"chroma_db/ 磁碟大小：{chroma_store.db_size_mb():.1f} MB")


if __name__ == "__main__":
    main()
