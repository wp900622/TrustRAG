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
from dataclasses import dataclass, field
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

    source: str = ""
    """這一塊是哪份文件來的。

    Day 29 加的。在那之前索引裡只有條號，兩份文件都有「第 3 條」的時候，
    服務分不出誰是誰。預設空字串是為了讓舊的索引照樣讀得起來。"""


@runtime_checkable
class Retriever(Protocol):
    """換 Chroma、Qdrant、pgvector 的時候，要實作的就只有這兩樣。"""

    name: str

    def search(self, query_vector: np.ndarray, k: int,
               where: dict | None = None) -> list[Hit]:
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

    def search(self, query_vector: np.ndarray, k: int,
               where: dict | None = None) -> list[Hit]:
        """`where` 直接交給 Chroma。

        它是先照 metadata 篩、再在剩下的向量裡找最近的 k 個，
        所以「第三章裡最相關的三條」問得出來。這是 Day 29 之前這支服務
        沒有用到、但資料庫本來就有的東西。
        """
        try:
            result = self._query(query_vector, k, where)
        except NotFoundError:
            result = self._query(query_vector, k, where)   # 重拿 handle 再試
        return self._to_hits(result)

    def _query(self, query_vector: np.ndarray, k: int,
               where: dict | None = None) -> dict:
        kwargs = {"query_embeddings": [query_vector.tolist()], "n_results": k,
                  "include": ["documents", "metadatas", "distances"]}
        if where:
            kwargs["where"] = where
        return self._resolve().query(**kwargs)

    @staticmethod
    def _to_hits(result: dict) -> list[Hit]:
        return [Hit(article_no=meta["article_no"], title=meta["title"],
                    text=doc, similarity=1.0 - dist, meta=meta,
                    source=str(meta.get("source", "")))
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

    def search(self, query_vector: np.ndarray, k: int,
               where: dict | None = None) -> list[Hit]:
        """過濾這件事，資料庫免費給你，自己寫就得自己想清楚順序。

        這裡是**先篩再算**：把不符合條件的那幾列的分數壓到負無限大，
        再取前 k 個。算完再丟掉不合格的那種寫法會有一個隱藏的坑——
        要的是「第三章裡最相關的三條」，不是「全庫最相關的三條裡剛好屬於第三章的」，
        後者很可能一條都不剩。
        """
        if not self._texts:
            # 索引被清空過（Day 29 的刪除）。空矩陣乘向量會炸，
            # 而「庫裡什麼都沒有」的正確答案是沒有結果，不是 500
            return []
        scores = self._matrix @ query_vector
        keep = self._mask(where)
        if keep is not None:
            if not keep.any():
                return []
            scores = np.where(keep, scores, -np.inf)
        limit = min(k, int(keep.sum()) if keep is not None else len(scores))
        top = np.argpartition(-scores, min(limit, len(scores) - 1))[:limit]
        top = top[np.argsort(-scores[top])]
        return [Hit(article_no=self._metas[i]["article_no"],
                    title=self._metas[i]["title"],
                    text=self._texts[i], similarity=float(scores[i]),
                    meta=self._metas[i],
                    source=str(self._metas[i].get("source", "")))
                for i in top]

    def _mask(self, where: dict | None):
        """只支援等值比對，跟服務對外開放的過濾條件一樣。

        刻意不去實作 Chroma 那套 `$and` / `$in` / `$gt` 的完整語法。
        兩個實作要能互換，靠的是**服務只承諾兩邊都做得到的東西**，
        而不是讓第二個實作去追第一個的功能表。
        """
        if not where:
            return None
        keep = np.ones(len(self._metas), dtype=bool)
        for key, want in where.items():
            value = want.get("$eq") if isinstance(want, dict) else want
            keep &= np.array([m.get(key) == value for m in self._metas])
        return keep
