# -*- coding: utf-8 -*-
"""Day 14 的主角：把 Day 8~13 的檢索接上生成，湊成一次完整的 RAG。

一次問答的四步，全部在 answer() 裡：
    問題 →（1）算問題向量 →（2）Chroma 取 top-k →
          （3）top-k 組成 prompt →（4）LLM 產生答案

Day 8~13 只做到第 2 步，把 top-k 印給人看；今天第 2 步的輸出變成第 3 步的
輸入，讀者從人變成模型。這個轉換帶來一個直接後果，也是本日實驗的題目：
給人看的時候，排第一名要是對的（top-1 命中率）；給模型看的時候，
只要正確條文出現在 k 個裡面就有機會（recall@k）。兩把尺不同。
"""
from dataclasses import dataclass, field

import chroma_store
import prompts
import rag_core


@dataclass
class RagResult:
    """一次問答的完整紀錄；實驗報表要的欄位都在這裡"""
    question: str
    answer: str
    hits: list[dict]
    embedding_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    context_chars: int = 0
    hit_ranks: list[int] = field(default_factory=list)


def build_index() -> tuple:
    """讀語料、切塊、算向量、開（或建）Chroma collection。

    回傳 (collection, chunks, articles, embedding token 數, 快取命中數)
    """
    import chunkers
    text = rag_core.load_document()
    chunks = chunkers.chunk_by_structure(text)
    articles = chunkers.parse_articles(text)
    vectors, tokens, cached = rag_core.get_embeddings([c["text"] for c in chunks])
    collection = chroma_store.open_or_build(chunks, articles, vectors)
    return collection, chunks, articles, tokens, cached


def answer(collection, question: str, k: int = 3,
           with_context: bool = True) -> RagResult:
    """回答一個問題。with_context=False 時跳過檢索，作為對照組。"""
    if not with_context:
        text, in_tok, out_tok = rag_core.chat(
            prompts.build_messages_no_context(question))
        return RagResult(question=question, answer=text, hits=[],
                         input_tokens=in_tok, output_tokens=out_tok)

    vectors, emb_tokens, _ = rag_core.get_embeddings([question])
    hits = chroma_store.retrieve(collection, vectors[0], k)
    messages = prompts.build_messages(question, hits)
    text, in_tok, out_tok = rag_core.chat(messages)
    return RagResult(
        question=question, answer=text, hits=hits,
        embedding_tokens=emb_tokens, input_tokens=in_tok, output_tokens=out_tok,
        context_chars=sum(len(h["text"]) for h in hits))


def rank_of_expected(hits: list[dict], expected_no: int) -> int:
    """預期條文排在第幾名（1 起算）；沒進 top-k 回 0。

    這個數字是本日實驗的核心：它同時算得出 top-1 命中率（rank == 1）
    與 recall@k（rank > 0），兩把尺用同一次檢索的結果比較才公平。
    """
    for i, hit in enumerate(hits, 1):
        if hit["meta"]["article_no"] == expected_no:
            return i
    return 0
