# -*- coding: utf-8 -*-
"""Day 11 增量 demo：對「已在服務中」的資料庫加一條新條文，立刻可查，再刪掉。

Day 10 的痛點：FAISS 索引加一塊＝整棟重建（50,059 塊單執行緒建圖 265 秒）。
Chroma 是資料庫：add() 一筆、毫秒級，不重建、不停機；delete() 同理。

流程（自我清理，不在庫裡留殘骸）：
1. 查「可以帶寵物來上班嗎？」——59 條裡沒有這種規定，top-1 是不相關條文
2. add() 虛構的「第 60 條（寵物友善辦公試辦）」——計時
3. 再查同一題——top-1 變成第 60 條
4. delete() 第 60 條——查詢結果回到原狀

先跑 python build_db.py 建庫。
"""
import time

import chroma_store
import rag_core
from chromadb.errors import NotFoundError

# 虛構的新條文（語料本身即為虛構，見 work_rules.md 開頭聲明）
NEW_ARTICLE_NO = 60
NEW_ARTICLE_TEXT = (
    "第 60 條（寵物友善辦公試辦）\n"
    "員工得於每週五攜帶已完成疫苗接種之犬、貓進入指定辦公區，"
    "並應於三個工作日前向行政部申請登記。寵物於辦公區內應以牽繩或提籠管理；"
    "如有影響同仁工作或環境衛生之情事，行政部得中止該員工之攜帶資格。"
    "本條為試辦條款，試辦期間至本年度末止。")
NEW_ARTICLE_METADATA = {"kind": "article", "article_no": NEW_ARTICLE_NO,
                        "chapter_no": 11, "chapter": "第十一章 附則",
                        "start": -1, "end": -1}  # 不在原文內，無字元範圍
PET_QUESTION = "可以帶寵物來上班嗎？"


def show_top1(collection, vector) -> None:
    """印出這一題目前的 top-1"""
    result = collection.query(query_embeddings=[vector.tolist()], n_results=1)
    meta = result["metadatas"][0][0]
    score = chroma_store.similarity(result["distances"][0][0])
    print(f"  top-1：第 {meta['article_no']} 條（{meta['chapter']}）[{score:.4f}]")


def main() -> None:
    client = chroma_store.get_client()
    try:
        collection = client.get_collection(chroma_store.COLLECTION_SMALL)
    except NotFoundError:
        print("找不到 collection：請先執行 python build_db.py")
        return

    vectors, used_tokens, cached = rag_core.get_embeddings(
        [PET_QUESTION, NEW_ARTICLE_TEXT])
    question_vector, article_vector = vectors
    print(f"embedding 消耗 {used_tokens} tokens（快取命中 {cached} 筆）\n")

    print(f"1. 新增前查「{PET_QUESTION}」（庫內 {collection.count()} 塊）")
    show_top1(collection, question_vector)

    start = time.perf_counter()
    collection.add(ids=[f"art-{NEW_ARTICLE_NO}"],
                   embeddings=[article_vector.tolist()],
                   documents=[NEW_ARTICLE_TEXT],
                   metadatas=[NEW_ARTICLE_METADATA])
    add_ms = (time.perf_counter() - start) * 1000
    print(f"\n2. add() 第 {NEW_ARTICLE_NO} 條：{add_ms:.0f} ms（不重建、不停機）")

    print(f"\n3. 新增後再查（庫內 {collection.count()} 塊）")
    show_top1(collection, question_vector)

    start = time.perf_counter()
    collection.delete(ids=[f"art-{NEW_ARTICLE_NO}"])
    delete_ms = (time.perf_counter() - start) * 1000
    print(f"\n4. delete() 第 {NEW_ARTICLE_NO} 條：{delete_ms:.0f} ms"
          f"（庫內回到 {collection.count()} 塊）")
    show_top1(collection, question_vector)


if __name__ == "__main__":
    main()
