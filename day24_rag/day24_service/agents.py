# -*- coding: utf-8 -*-
"""把 Day 20／21 那支 agent 接進服務。

系列名的最後一段是「到 AI Agent」，但前面幾節收進來的只有 Day 14 那條寫死管線：
算問題向量、查一次、丟給模型、回答案。agent 那條路（自己決定查幾次、
交卷前自驗）從 Day 20 做到 Day 21，一直留在 `agent_day21.py` 裡，沒有人能打它。

兩個限制決定了這個檔案的長相：

1. **`agent_day21.py` 是凍結的。** 它是 Day 21 那篇文章的證據，改一個字，
   那天的數字就不可重現。所以這裡一行都不動它，只在外面包。
2. **它直接吃 Chroma 的 collection。** `run_agent()` 裡是
   `chroma_store.retrieve(collection, ...)`，而服務這一側已經把檢索抽成
   `Retriever` 介面了。

第 2 點的解法是 `RetrieverCollection`：把任何一個 Retriever 包成
Chroma collection 的樣子。凍結的 agent 不必知道，它照樣可以跑在 numpy 上。
這是今天抽那個介面真正的回報——**介面能不能用，看的是舊程式能不能不改就跑進來。**
"""
import contextlib
import threading
import time

import numpy as np

import agent_day21 as agent21
import rag_core

from . import timing
from .config import settings

# agent 那條路一次只放幾個進來。理由寫在 _instrumented() 的 docstring 裡
_AGENT_SLOTS = threading.Semaphore(max(1, settings.agent_max_concurrency))


class RetrieverCollection:
    """Retriever → Chroma collection 的轉接頭。

    `chroma_store.retrieve()` 只用到 `collection.query()` 這一個方法，
    所以這裡只要把回傳排成它要的形狀就行：三個 key，每個都包一層 list。
    """

    def __init__(self, retriever):
        self._retriever = retriever
        self.name = retriever.name

    def query(self, query_embeddings, n_results, include=None) -> dict:
        vector = np.asarray(query_embeddings[0], dtype=np.float32)
        with timing.stage("retrieve"):
            hits = self._retriever.search(vector, n_results)
        return {"documents": [[h.text for h in hits]],
                "metadatas": [[h.meta for h in hits]],
                # Chroma 回的是距離，這裡換回去：距離 = 1 − 餘弦相似度
                "distances": [[1.0 - h.similarity for h in hits]]}


@contextlib.contextmanager
def _instrumented(use_cache: bool):
    """把 agent 對外的三個出口包上計時，離開時還原。

    agent 一題會打好幾次模型（Day 21 量到平均 3 次以上），
    而 `timing.stage()` 是累加的，所以 `llm` 那一格會自己加總，不必另外算。

    `run_check` 走的是 `rag_core.chat`，而 Day 21 沒有把 `use_cache` 傳下去。
    cold 那組要真的打 API，所以這裡連 `rag_core.chat` 一起換掉，把它釘成不走快取。

    **這個包裝是行程層級的**：替換的是模組上的名字，所以同時有第二個請求
    走進來，兩邊的計時會互相污染。正確的解法是把 retriever 與 embedder 注入
    agent，但那要改那支凍結的檔案。

    在那之前，用一把信號量把 agent 請求序列化（`DAY24_AGENT_CONCURRENCY`，
    預設 1）。**寧可讓第二個請求排隊，也不要回一份錯的計時給它。**
    pipeline 那條路不受影響，它沒有走這裡。
    """
    _AGENT_SLOTS.acquire()
    orig_chat_tools = agent21.chat_tools
    orig_check = agent21.run_check
    orig_embed = rag_core.get_embeddings
    orig_chat = rag_core.chat

    def chat_tools(*args, **kwargs):
        with timing.stage("llm"):
            return orig_chat_tools(*args, **kwargs)

    def run_check(*args, **kwargs):
        with timing.stage("llm"):
            return orig_check(*args, **kwargs)

    def get_embeddings(*args, **kwargs):
        with timing.stage("embed"):
            return orig_embed(*args, **kwargs)

    def chat(messages, use_cache_inner: bool = True):
        return orig_chat(messages, use_cache=use_cache and use_cache_inner)

    agent21.chat_tools = chat_tools
    agent21.run_check = run_check
    rag_core.get_embeddings = get_embeddings
    rag_core.chat = chat
    try:
        yield
    finally:
        agent21.chat_tools = orig_chat_tools
        agent21.run_check = orig_check
        rag_core.get_embeddings = orig_embed
        rag_core.chat = orig_chat
        _AGENT_SLOTS.release()


def run(retriever, question: str, k: int, use_cache: bool,
        self_check: bool) -> dict:
    """跑一題 agent。

    `self_check=False` 是 Day 21 的 C1（工具擺著，它自己決定要不要驗），
    `True` 是 C2（沒驗過就想交卷會被退件）。預設 False，因為 Day 21 量過：
    強制自驗救回 1 題、改壞 1 題，淨值 0，而成本是 ×2.0。
    """
    collection = RetrieverCollection(retriever)
    original_k = agent21.SEARCH_K
    agent21.SEARCH_K = k
    t0 = time.perf_counter()
    try:
        with _instrumented(use_cache):
            result = agent21.run_agent(collection, question,
                                       force_check=self_check,
                                       use_cache=use_cache)
    finally:
        agent21.SEARCH_K = original_k
    result["wall_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result
