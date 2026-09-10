# -*- coding: utf-8 -*-
"""Day 14 Chroma 存取層：自 Day 11 沿用並精簡。

- 拿掉 50k 合成向量的大庫：Day 11 用它壓測索引，今天的主題在生成端，
  檢索側刻意保持與 Day 11 完全相同的條件，才能把答案的落差歸因給生成
- 距離空間仍是 cosine，回傳仍是「距離」（＝1−餘弦相似度），
  顯示前用 similarity() 換算——與 Day 8~11 同一把尺
- 內嵌模式（PersistentClient），不開任何網路埠；遙測關閉
"""
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

import chunkers

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION = "work-rules"


def get_client() -> chromadb.api.ClientAPI:
    """開（或初始化）本資料夾的持久化資料庫；遙測一律關閉"""
    return chromadb.PersistentClient(
        path=str(DB_DIR), settings=Settings(anonymized_telemetry=False))


def similarity(distance: float) -> float:
    """把 Chroma 的 cosine 距離換回 Day 8~11 慣用的餘弦相似度"""
    return 1.0 - distance


def article_metadata(chunk: dict, articles: list[dict]) -> dict:
    """一個 chunk 進資料庫時帶的 metadata：條號、章別、字元範圍。

    結構化切法下一個 chunk 就是一條條文，所以 article_no 唯一；
    (start, end) 仍然保留，讓命中判定可以沿用 Day 9~11 的「範圍重疊」。
    """
    covered = chunkers.articles_covering(chunk, articles)
    return {"article_no": covered[0]["no"] if covered else 0,
            "title": covered[0]["title"] if covered else chunk["chapter"],
            "chapter_no": chunk["chapter_no"], "chapter": chunk["chapter"],
            "start": chunk["start"], "end": chunk["end"]}


def rebuild(client, chunks: list[dict], articles: list[dict],
            vectors) -> chromadb.Collection:
    """砍掉重建 59 條的 collection：實驗要可重複，狀態必須從零開始"""
    try:
        client.delete_collection(COLLECTION)
    except NotFoundError:
        pass
    collection = client.create_collection(
        COLLECTION, configuration={"hnsw": {"space": "cosine"}})
    metadatas = [article_metadata(c, articles) for c in chunks]
    collection.add(ids=[f"art-{m['article_no']}" for m in metadatas],
                   embeddings=vectors.tolist(),
                   documents=[c["text"] for c in chunks],
                   metadatas=metadatas)
    return collection


def open_or_build(chunks: list[dict], articles: list[dict],
                  vectors) -> chromadb.Collection:
    """有現成的庫（且塊數對）就直接開，否則重建——塊數不對代表是半成品"""
    client = get_client()
    try:
        collection = client.get_collection(COLLECTION)
        if collection.count() == len(chunks):
            return collection
    except NotFoundError:
        pass
    return rebuild(client, chunks, articles, vectors)


def retrieve(collection, query_vector, k: int) -> list[dict]:
    """取回 top-k：向量、原文、metadata 一次拿齊。

    這裡是 Day 8~11 與今天的接縫——以前 top-k 是給人看的檢索結果，
    今天它要變成 prompt 的一部分，所以連原文一起帶出來。
    """
    result = collection.query(query_embeddings=[query_vector.tolist()],
                              n_results=k,
                              include=["documents", "metadatas", "distances"])
    return [{"text": doc, "meta": meta, "similarity": similarity(dist)}
            for doc, meta, dist in zip(result["documents"][0],
                                       result["metadatas"][0],
                                       result["distances"][0])]
