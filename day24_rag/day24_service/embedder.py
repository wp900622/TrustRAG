# -*- coding: utf-8 -*-
"""算問題向量的兩個實作：走快取的，跟真的打 API 的。

要有這兩個，是因為前 23 天的數字全部是走快取量出來的。
快取命中的時候 `embed` 這一段量到的其實是「把 10.5 MB 的 json 讀進來」，
那不是上線後的樣子——上線後每一個新問題都沒進過快取。

今天的四段拆解要兩組都跑：
    warm  兩個快取都開   ← 前 23 天腳本的樣子
    cold  兩個快取都關   ← 上線後每一個新問題的樣子
"""
import time

import numpy as np

import rag_core
from . import timing


class CachedEmbedder:
    """Day 8 起的那一個：先查 embeddings_cache.json，沒有才打 API。"""

    name = "cached"

    def embed(self, text: str) -> tuple[np.ndarray, int]:
        with timing.stage("embed"):
            vectors, tokens, _cached = rag_core.get_embeddings([text])
        return vectors[0], tokens


class DirectEmbedder:
    """每次都真的打一次 embedding API，不讀也不寫快取。

    上線之後這才是常態：使用者問的問題你沒看過。
    """

    name = "direct"

    def embed(self, text: str) -> tuple[np.ndarray, int]:
        t0 = time.perf_counter()
        response = rag_core.get_client().embeddings.create(
            model=rag_core.EMBEDDING_MODEL, input=[text])
        vector = np.array(response.data[0].embedding, dtype=np.float32)
        vector /= np.linalg.norm(vector)      # 與快取那條路同一把尺
        timing.record("embed", (time.perf_counter() - t0) * 1000)
        return vector, response.usage.total_tokens


def get(use_cache: bool):
    return CachedEmbedder() if use_cache else DirectEmbedder()
