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
import json
import threading
import time

import numpy as np

import agent_day20 as agent20
import agent_day21 as agent21
import judges
import rag_core

from . import budget as budget_mod
from . import ledger, timing
from .config import settings

# agent 那條路一次只放幾個進來。理由寫在 _instrumented() 的 docstring 裡
_AGENT_SLOTS = threading.Semaphore(max(1, settings.agent_max_concurrency))


class RetrieverCollection:
    """Retriever → Chroma collection 的轉接頭。

    `chroma_store.retrieve()` 只用到 `collection.query()` 這一個方法，
    所以這裡只要把回傳排成它要的形狀就行：三個 key，每個都包一層 list。
    """

    def __init__(self, retriever, where: dict | None = None):
        self._retriever = retriever
        self._where = where
        self.name = retriever.name
        # 被上限攔下來的時候，凍結 agent 的回傳值拿不到了（例外從中間拋出來），
        # 但它查到的條文經過這裡。記著，停損時還給呼叫端
        self.article_nos: list[int] = []
        # Day 29：條號自己是有歧義的。凍結的 agent 只帶得回條號，
        # 所以（文件, 條號）在這裡成對記下來，引用才查得到對的原文
        self.cited_pairs: list[tuple[str, int]] = []
        self.searches = 0

    def query(self, query_embeddings, n_results, include=None) -> dict:
        """`where` 是請求帶進來的過濾條件，由轉接頭夾帶。

        凍結的 `agent_day21.py` 呼叫的是 `collection.query(...)`，簽名裡沒有
        過濾這回事。但它每一次檢索都會經過這裡，所以「只查第三章」這種限制
        可以在它不知情的情況下生效。改的是外面那一層，不是它。
        """
        vector = np.asarray(query_embeddings[0], dtype=np.float32)
        with timing.stage("retrieve"):
            hits = self._retriever.search(vector, n_results, self._where)
        self.searches += 1
        self.article_nos += [h.meta["article_no"] for h in hits]
        self.cited_pairs += [(h.source, h.meta["article_no"]) for h in hits]
        return {"documents": [[h.text for h in hits]],
                "metadatas": [[h.meta for h in hits]],
                # Chroma 回的是距離，這裡換回去：距離 = 1 − 餘弦相似度
                "distances": [[1.0 - h.similarity for h in hits]]}


def _agent_tokens(messages: list[dict], out: dict) -> tuple[int, int]:
    """一次 `chat_tools()` 的輸入／輸出 token，離線重數。

    **算法逐字照抄 `agent_day21.run_agent()` 裡那兩行**
    （`count_messages(messages) + TOOLS_TOKEN_ESTIMATE`），
    所以帳本記的數字跟 Day 21 那篇公布的是同一套，兩邊對得起來。

    為什麼非得自己數：`chat_tools()` 是凍結檔案裡的函式，它把 API 回報的
    usage 丟掉了，只回訊息本身。改它就會讓 Day 21 的數字不可重現。
    """
    tin = agent20.count_messages(messages) + agent21.TOOLS_TOKEN_ESTIMATE
    tout = judges.count_text(out.get("content") or "")
    for call in (out.get("tool_calls") or []):
        tout += judges.count_text(json.dumps(call.get("function", {}),
                                             ensure_ascii=False))
    return tin, tout


@contextlib.contextmanager
def _instrumented(use_cache: bool, tally: dict, guard=None):
    """把 agent 對外的三個出口包上計時與記帳，離開時還原。

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

    def chat_tools(messages, *args, **kwargs):
        # Day 28：停損掛在這裡，也只掛在這裡。模型呼叫是 agent 唯一會
        # 「花掉時間與錢」的動作，其餘（檢索、算向量）都在它的陰影底下
        if guard is not None:
            guard.check()
        # 記帳跟計時包在同一層。agent 自己的快取 key 也是整份 messages 的雜湊
        # （只是多摻了 TOOLS），所以帳本那套「同一把 key」的做法原封不動就能用
        key = agent21._key(messages)
        # `use_cache=False` 的時候這次一定真的打 API，快取檔裡有沒有這把 key
        # 都不影響。Day 28 之前這裡只看檔案，於是付了錢的呼叫被記成命中，
        # 金額上限看到的花費永遠是 0
        seen = use_cache and key in agent21._load(agent21.CACHE_PATH)
        with timing.stage("llm"):
            out = orig_chat_tools(messages, *args, **kwargs)
        # Day 28：token 改成每一次都數，命中的那些也數。被上限攔下來的時候，
        # 凍結 agent 自己那份計數跟著例外一起消失了，這裡是唯一剩下的來源
        tin, tout = _agent_tokens(messages, out)
        tally["in_tokens"] += tin
        tally["out_tokens"] += tout
        tally["calls"] += 1
        if seen:
            avoided = ledger.hit(key)
            tally["hits"] += 1
            if avoided is None:
                tally["unpriced_hits"] += 1
            else:
                tally["avoided_twd"] += avoided
        else:
            tally["paid_calls"] += 1
            tally["cost_twd"] += ledger.record(key, agent21.MODEL, tin, tout,
                                               estimated=True)
        return out

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
        self_check: bool, limits: budget_mod.Budget | None = None,
        where: dict | None = None) -> dict:
    """跑一題 agent。

    `self_check=False` 是 Day 21 的 C1（工具擺著，它自己決定要不要驗），
    `True` 是 C2（沒驗過就想交卷會被退件）。預設 False，因為 Day 21 量過：
    強制自驗救回 1 題、改壞 1 題，淨值 0，而成本是 ×2.0。

    `limits` 是 Day 28 的停損。不給就是不設限，行為與 Day 24 接進來那天相同。
    步數上限走的是凍結 agent 自己那條強制作答的路，所以它停下來仍然有答案；
    金額與時間上限是直接打斷，停下來只剩下帳單與已經查到的條文。
    """
    limits = limits or budget_mod.Budget()
    collection = RetrieverCollection(retriever, where)
    original_k = agent21.SEARCH_K
    agent21.SEARCH_K = k
    # 這一題自己的帳。agent 一題會打好幾次，其中幾次可能命中、幾次真的付錢，
    # 所以回應裡的金額必須是這一題加總的，不是最後一次那次的
    tally = {"paid_calls": 0, "hits": 0, "unpriced_hits": 0,
             "cost_twd": 0.0, "avoided_twd": 0.0,
             "calls": 0, "in_tokens": 0, "out_tokens": 0}
    guard = budget_mod.Guard(limits, tally)
    t0 = time.perf_counter()
    try:
        # 先進 _instrumented（它握著信號量），再改那個模組常數。
        # 反過來的話，上限會在別人的請求裡生效幾毫秒
        with _instrumented(use_cache, tally, guard),                 budget_mod.step_cap(limits.max_steps):
            result = agent21.run_agent(collection, question,
                                       force_check=self_check,
                                       use_cache=use_cache)
        result["stopped_reason"] = ("steps" if limits.max_steps is not None
                                    and result["hit_cap"] else None)
        result["stopped_at_step"] = (result["steps"]
                                     if result["stopped_reason"] else None)
    except budget_mod.BudgetExceeded as exc:
        result = _stopped(exc, collection, tally)
    finally:
        agent21.SEARCH_K = original_k
    result["wall_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    result["ledger"] = tally
    result["cited_pairs"] = list(dict.fromkeys(collection.cited_pairs))
    return result


def _stopped(exc: budget_mod.BudgetExceeded, collection: RetrieverCollection,
             tally: dict) -> dict:
    """金額或時間上限攔下來之後，把還在手上的東西排成 `run_agent()` 的形狀。

    **答案是沒有的。** 例外從模型呼叫之前拋出來，那一輪的回覆還沒生成，
    前幾輪的草稿也只存在凍結 agent 的區域變數裡。所以這裡不編一個答案出來，
    `answer` 留空，讓 handler 回 null。

    拿得回來的是兩樣東西：它查到哪些條文（經過轉接頭，記在 collection 上），
    以及這一題已經花掉多少（記在 tally 上）。對一個被打斷的請求來說，
    這兩樣比一句半成品的答案有用。
    """
    return {"answer": "", "trace": [], "queries": [],
            "article_nos": list(dict.fromkeys(collection.article_nos)),
            "cited_pairs": list(dict.fromkeys(collection.cited_pairs)),
            "searches": collection.searches, "steps": exc.at_step - 1,
            "repeats": 0, "hit_cap": False, "nudged": False,
            "checks": [], "n_checks": 0, "rejects": 0,
            "flagged": False, "revised": False,
            "input_tokens": tally["in_tokens"],
            "output_tokens": tally["out_tokens"],
            "embedding_tokens": 0, "llm_calls": tally["calls"],
            "stopped_reason": exc.reason, "stopped_at_step": exc.at_step}
