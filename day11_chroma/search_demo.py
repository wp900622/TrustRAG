# -*- coding: utf-8 -*-
"""Day 11 互動 demo：查本機 Chroma 資料庫，可用章別 metadata 過濾縮小檢索範圍。

用法（先跑 python build_db.py 建庫）：
    python search_demo.py 特休沒休完會怎樣
    python search_demo.py --chapter 5 特休沒休完會怎樣    # 只查第 5 章（請假管理）
    python search_demo.py --collection work-rules-50k 病假連續請幾天以上需要附診斷證明
    python search_demo.py                                 # 互動模式，空行離開

跟 Day 10 對照著玩：同語料同問題，開庫即查、不重建索引；
回傳的不只向量 id，原文與 metadata 一起回來——這就是資料庫與索引的差別。
"""
import argparse
import time

import chroma_store
import rag_core
from chromadb.errors import NotFoundError


def format_row(rank: int, meta: dict, document: str | None, distance: float) -> str:
    """把一筆查詢結果排成人看的一行：條號（章）｜相似度｜原文前 40 字"""
    score = chroma_store.similarity(distance)
    if meta.get("kind") == "article":
        preview = (document or "")[:40].replace("\n", " ")
        return (f"  {rank}. [{score:.4f}] 第 {meta['article_no']} 條"
                f"（{meta['chapter']}）｜{preview}…")
    return f"  {rank}. [{score:.4f}] 合成向量"


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 11 Chroma 檢索 demo")
    parser.add_argument("question", nargs="*", help="要查詢的問題；不給就進入互動模式")
    parser.add_argument("--chapter", type=int, default=None,
                        help="只在此章號內檢索（章號＝章的出現順序，1 起算）")
    parser.add_argument("--collection", default=chroma_store.COLLECTION_SMALL,
                        help=f"要查的 collection（預設 {chroma_store.COLLECTION_SMALL}）")
    args = parser.parse_args()

    start = time.perf_counter()
    client = chroma_store.get_client()
    try:
        collection = client.get_collection(args.collection)
    except NotFoundError:
        print(f"找不到 collection「{args.collection}」：請先執行 python build_db.py"
              f"{'（加 --with-50k）' if args.collection == chroma_store.COLLECTION_50K else ''}")
        return
    open_ms = (time.perf_counter() - start) * 1000
    print(f"開庫 {open_ms:.0f} ms｜{args.collection}（{collection.count():,} 塊）"
          + (f"｜過濾：第 {args.chapter} 章" if args.chapter else ""))

    def ask(question: str) -> None:
        vector, used_tokens, _ = rag_core.get_embeddings([question])
        where = {"chapter_no": args.chapter} if args.chapter else None
        result = collection.query(query_embeddings=[vector[0].tolist()],
                                  n_results=3, where=where)
        if not result["ids"][0]:
            print("  （此過濾條件下沒有任何資料）")
            return
        for rank, (meta, doc, dist) in enumerate(
                zip(result["metadatas"][0], result["documents"][0],
                    result["distances"][0]), start=1):
            print(format_row(rank, meta, doc, dist))
        if used_tokens:
            print(f"  （本題 embedding 消耗 {used_tokens} tokens）")

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
