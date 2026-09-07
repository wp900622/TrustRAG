# -*- coding: utf-8 -*-
"""Day 11 Chroma 存取層：建庫、開庫、批次寫入集中在這裡，demo 與實驗共用同一條路。

- PersistentClient：資料落地在 chroma_db/，行程重啟直接開庫——不重算、不重建
- 距離空間用 cosine，與 Day 8~10 同一把尺。注意 Chroma 回傳的是「距離」
  （cosine distance ＝ 1 − 餘弦相似度），顯示前用 similarity() 換算
- 關閉匿名遙測（anonymized_telemetry=False）：範例程式不該預設對外回報使用資料
- Chroma 底層索引就是 Day 10 的主角 HNSW，預設 ef_search=100、M=16；
  它仍是近似檢索，預設值扛不扛得住是本日實驗的題目之一

資安注意：chromadb 以「Python server」模式對外服務時，有未修補的
CVE-2026-45829（未認證即可遠端執行程式碼，CVSS 10.0）。本資料夾只用
內嵌模式、不開任何網路埠，不受該漏洞影響；細節見 README「資安注意」。
"""
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

import chunkers
import synth_vectors

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_SMALL = "work-rules"        # 59 條真實條文
COLLECTION_50K = "work-rules-50k"      # 59 條＋50,000 個合成干擾向量


def get_client() -> chromadb.api.ClientAPI:
    """開（或初始化）本資料夾的持久化資料庫；遙測一律關閉"""
    return chromadb.PersistentClient(
        path=str(DB_DIR), settings=Settings(anonymized_telemetry=False))


def similarity(distance: float) -> float:
    """把 Chroma 的 cosine 距離換回 Day 8~10 慣用的餘弦相似度"""
    return 1.0 - distance


def article_metadata(chunk: dict, articles: list[dict]) -> dict:
    """一個 chunk 進資料庫時帶的 metadata：條號＋章別（過濾的鑰匙）＋字元範圍（判定命中用）"""
    covered = chunkers.articles_covering(chunk, articles)
    return {"kind": "article",
            "article_no": covered[0]["no"] if covered else 0,
            "chapter_no": chunk["chapter_no"], "chapter": chunk["chapter"],
            "start": chunk["start"], "end": chunk["end"]}


def _add_articles(collection, chunks: list[dict], articles: list[dict],
                  vectors) -> None:
    """把 59 條真實條文寫進 collection：向量、原文、metadata 一起進去——
    這三樣被同一筆資料綁在一起管理，正是「索引」與「資料庫」的分界線"""
    metadatas = [article_metadata(c, articles) for c in chunks]
    collection.add(ids=[f"art-{m['article_no']}" for m in metadatas],
                   embeddings=vectors.tolist(),
                   documents=[c["text"] for c in chunks],
                   metadatas=metadatas)


def rebuild_small(client, chunks: list[dict], articles: list[dict],
                  vectors) -> chromadb.Collection:
    """砍掉重建 59 條的 collection：實驗要可重複，狀態必須從零開始"""
    try:
        client.delete_collection(COLLECTION_SMALL)
    except NotFoundError:
        pass
    collection = client.create_collection(
        COLLECTION_SMALL, configuration={"hnsw": {"space": "cosine"}})
    _add_articles(collection, chunks, articles, vectors)
    return collection


def ensure_50k(client, chunks: list[dict], articles: list[dict],
               vectors) -> tuple[chromadb.Collection, float, bool]:
    """50k collection 有就直接用，沒有（或塊數不對＝半成品）才重建。

    合成向量由固定 seed 生成、內容永遠相同，重建只是浪費時間——
    首次建庫的秒數存在 collection metadata，之後重跑實驗仍能回報。
    回傳：(collection, 建庫秒數, 是否沿用既有庫)
    """
    expected_total = len(chunks) + synth_vectors.N_CENTROIDS * synth_vectors.PER_CENTROID
    try:
        collection = client.get_collection(COLLECTION_50K)
        if collection.count() == expected_total:
            return collection, float(collection.metadata["build_seconds"]), True
        client.delete_collection(COLLECTION_50K)
    except NotFoundError:
        pass

    start = time.perf_counter()
    collection = client.create_collection(
        COLLECTION_50K, configuration={"hnsw": {"space": "cosine"}})
    _add_articles(collection, chunks, articles, vectors)
    distractors = synth_vectors.make_distractors(dim=vectors.shape[1])
    batch = client.get_max_batch_size()  # 1.5.9 實測為 5461，超過會被拒收
    for i in range(0, len(distractors), batch):
        part = distractors[i:i + batch]
        collection.add(ids=[f"synth-{j}" for j in range(i, i + len(part))],
                       embeddings=part.tolist(),
                       metadatas=[{"kind": "synth"}] * len(part))
    elapsed = time.perf_counter() - start
    collection.modify(metadata={"build_seconds": round(elapsed, 1)})
    return collection, elapsed, False


def db_size_mb() -> float:
    """chroma_db/ 在磁碟上的實際大小（MB）"""
    return sum(f.stat().st_size for f in DB_DIR.rglob("*") if f.is_file()) / 1e6
