# -*- coding: utf-8 -*-
"""把檢索抽成一個介面，外加兩個實作。

為什麼今天做這件事：前 13 天我一直在動檢索這一側（切法、k、metadata 過濾、
重排），每一次都直接改 `chroma_store.py`。那沒問題，因為當時只有一個呼叫者。
變成服務之後，`/ask` 不應該知道底下是 Chroma 還是別的東西——
它只需要「給我一個問題向量，還我 k 條帶 metadata 的原文」。

介面刻意只有一個方法。抽象抽得太大是另一種債：
`search(vector, k) -> list[Hit]`，就這樣。過濾、重排、混合檢索都可以
包在實作裡，或者變成裝飾另一個 Retriever 的 Retriever。

兩個實作並排還有一個好處：今天要量「檢索佔延遲的幾成」，
有第二個實作才知道量到的是 Chroma 的成本，還是這件事本身的成本。
"""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from chromadb.errors import NotFoundError


@dataclass
class Hit:
    article_no: int
    title: str
    text: str
    similarity: float
    meta: dict
    """整包 metadata 原封帶著走。

    prompt 要用到 `chapter`，而且 Day 14 的 chat 快取 key 就是整串 messages——
    這裡少帶一個欄位，prompt 就會差一個字，23 天的快取全部失效。"""


@runtime_checkable
class Retriever(Protocol):
    """換 Chroma、Qdrant、pgvector 的時候，要實作的就只有這兩樣。"""

    name: str

    def search(self, query_vector: np.ndarray, k: int) -> list[Hit]:
        ...


class ChromaRetriever:
    """Day 11 起就在用的那一個，包起來而已，檢索行為完全不變。

    唯一多做的事：**不要把 collection 物件抓在手上。**
    Chroma 內嵌模式下，另一個行程（或這個行程的攝取）把 collection 砍掉重建之後，
    舊的 handle 會指向一個已經不存在的 UUID，下一次查詢直接炸：

        NotFoundError: Error getting collection: Collection [003716ce-…] does not exist.

    開兩個 worker 跑攝取時我就踩到這個，6 個 job 死了 2 個。
    所以這裡收的是一個「去把 collection 拿來」的函式，而不是 collection 本身，
    並且在 NotFoundError 的時候重拿一次再試。
    """

    name = "chroma"

    def __init__(self, resolve):
        self._resolve = resolve if callable(resolve) else (lambda: resolve)

    def search(self, query_vector: np.ndarray, k: int) -> list[Hit]:
        try:
            result = self._query(query_vector, k)
        except NotFoundError:
            result = self._query(query_vector, k)      # 重拿一次 handle 再試
        return self._to_hits(result)

    def _query(self, query_vector: np.ndarray, k: int) -> dict:
        return self._resolve().query(
            query_embeddings=[query_vector.tolist()], n_results=k,
            include=["documents", "metadatas", "distances"])

    @staticmethod
    def _to_hits(result: dict) -> list[Hit]:
        return [Hit(article_no=meta["article_no"], title=meta["title"],
                    text=doc, similarity=1.0 - dist, meta=meta)
                for doc, meta, dist in zip(result["documents"][0],
                                           result["metadatas"][0],
                                           result["distances"][0])]


class NumpyRetriever:
    """59 條規章的暴力解：一個 (59, 1536) 的矩陣乘一個向量。

    放這個進來不是為了推薦它，是為了讓「檢索很慢」這句話有個對照組。
    向量都已經正規化過，所以內積就是餘弦相似度。
    語料長到幾萬條的時候這個實作會輸，但那時候你也不會用 59 條的參數。
    """

    name = "numpy"

    def __init__(self, vectors: np.ndarray, chunks: list[dict],
                 metadatas: list[dict]):
        self._matrix = vectors
        self._texts = [c["text"] for c in chunks]
        self._metas = metadatas

    def search(self, query_vector: np.ndarray, k: int) -> list[Hit]:
        scores = self._matrix @ query_vector
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [Hit(article_no=self._metas[i]["article_no"],
                    title=self._metas[i]["title"],
                    text=self._texts[i], similarity=float(scores[i]),
                    meta=self._metas[i])
                for i in top]
