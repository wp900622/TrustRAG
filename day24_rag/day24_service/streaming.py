# -*- coding: utf-8 -*-
"""SSE：一邊生一邊送。

為什麼不直接用 `rag_core.chat()`：它是前 23 天每一篇的證據，不能改，
而它是一次要完整答案的（`stream=False`）。所以這裡自己開一條串流的路。

**但兩條路必須共用同一份快取。** 用的是 `rag_core._chat_cache_key()`
算出來的同一把 key，而讀寫走的是 `rag_core._load_json` / `_save_json`——
也就是 `cache.py` 在啟動時換掉的那兩個。所以串流問過的題目，
非串流再問會命中；反過來也是。不這樣做的話，同一題會被買兩次。

SSE 是什麼：一個普通的 HTTP 回應，只是不關連線，伺服器有東西就往下寫一段。
線路上長這樣，每個事件兩行加一個空行：

    event: citations
    data: {"citations": [...]}
    <空行>

瀏覽器有內建的 EventSource 可以收，curl 也看得到，不需要 WebSocket。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Iterator

import rag_core

log = logging.getLogger("day24")

#: 快取命中時，把存好的答案切成幾段送。見 `replay()` 的說明。
REPLAY_CHUNK = 24


def to_sse(event: str, data: dict) -> bytes:
    """包成 SSE 的線路格式。

    `ensure_ascii=False` 讓中文照原樣走，省掉一半以上的位元組；
    SSE 規定是 UTF-8，所以這樣是合法的。
    """
    body = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n".encode("utf-8")


def stream_chat(messages: list[dict], use_cache: bool = True) -> Iterator[tuple]:
    """一段一段把答案 yield 出來。

    yield 的是 ("delta", 那一段文字) 或 ("final", (完整答案, in_tok, out_tok, 是否命中))。
    最後一個一定是 final，呼叫端靠它拿到寫回應要的統計。

    快取命中時走 `replay()`：不打 API，把存好的答案切段送出去。
    為什麼要假裝串流——見那個函式的說明。
    """
    key = rag_core._chat_cache_key(messages)
    if use_cache:
        cache = rag_core._load_json(rag_core.CHAT_CACHE_PATH)
        if key in cache:
            text = cache[key]
            for piece in replay(text):
                yield "delta", piece
            yield "final", (text, 0, 0, True)
            return

    # stream_options 要開，不然串流模式下拿不到 usage，成本就沒得算
    response = rag_core.get_client().chat.completions.create(
        model=rag_core.CHAT_MODEL, messages=messages, temperature=0,
        stream=True, stream_options={"include_usage": True})

    parts: list[str] = []
    in_tok = out_tok = 0
    for chunk in response:
        if chunk.usage:                      # 最後一個 chunk 才帶 usage
            in_tok = chunk.usage.prompt_tokens
            out_tok = chunk.usage.completion_tokens
        if not chunk.choices:
            continue
        piece = chunk.choices[0].delta.content
        if piece:
            parts.append(piece)
            yield "delta", piece

    text = "".join(parts).strip()

    # 寫快取放在最後：中途斷線就不寫，寧可下次重買，也不要存一個半截的答案
    if use_cache and text:
        cache = rag_core._load_json(rag_core.CHAT_CACHE_PATH)
        cache[key] = text
        rag_core._save_json(rag_core.CHAT_CACHE_PATH, cache)

    yield "final", (text, in_tok, out_tok, False)


def replay(text: str, size: int = REPLAY_CHUNK) -> Iterator[str]:
    """快取命中時，把存好的答案切成幾段送出去。

    這是設計決定不是技術問題：整包一次送也完全可行，而且更快。
    選擇切段的理由只有一個——**前端只要寫一種收法**。
    命中與否是伺服器的事，不該讓呼叫端寫兩條分支。

    代價誠實寫在這裡：命中時串流一定比非串流慢，因為多了分段的來回。
    絕對值很小（毫秒等級），但如果你的場景命中率很高，
    那就不該開串流——它買到的是「等模型時有東西看」，沒在等就沒得買。
    """
    for i in range(0, len(text), size):
        yield text[i:i + size]
