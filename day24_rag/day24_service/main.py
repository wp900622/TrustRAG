# -*- coding: utf-8 -*-
"""Day 24：把前 23 天散落的腳本收成一支服務。

    POST /ask              問一題。mode=pipeline 走 Day 14 的寫死管線，
                           mode=agent 走 Day 20／21 那支自己決定查幾次的
    POST /documents        攝取一份文件，立刻回 202 與 job_id（背景跑）
    GET  /documents/{id}   查攝取進度
    GET  /documents        最近的攝取紀錄
    GET  /healthz          活著嗎：索引塊數、retriever、快取、版本
    GET  /readyz           可以收流量了嗎：索引建好了沒

檢索與生成的邏輯一個字都沒改（`chunkers` / `prompts` / `rag_core` 全部沿用），
動的是它們外面那一層：設定、快取、狀態、錯誤、觀測。

handler 刻意寫成同步 `def`：底下那些 OpenAI 呼叫是會阻塞的 sdk，
寫成 `async def` 會把整個 event loop 卡住；同步的會被 Starlette 丟進執行緒池。

跑起來：
    uvicorn day24_service.main:app --port 8024
"""
import logging
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import BackgroundTasks, FastAPI, Response

import chroma_store
import pipeline
import prompts
import rag_core

from . import agents, cache, embedder, errors, ingest, timing
from .config import settings
from .contracts import (AgentTrace, AskRequest, AskResponse, Citation,
                        IngestJob, IngestRequest, Timing, Usage)
from .errors import ServiceError
from .retrievers import ChromaRetriever, NumpyRetriever, Retriever
from .store import JobStore, as_contract

VERSION = "24.0"
log = logging.getLogger("day24")

STATE: dict = {"retrievers": {}, "chunks": 0, "boot_ms": 0.0,
               "collection": None, "chunk_index": {}, "ready": False,
               "store": None, "cache": {}}

# 索引是所有請求共用的，重建的時候不能有人正在查
_INDEX_LOCK = threading.RLock()


def _collection():
    """每次都跟 client 重拿一次 collection，不要把 handle 抓在手上。

    Chroma 內嵌模式下，另一個 worker 把它砍掉重建之後，舊 handle 就指向
    一個不存在的 UUID。重拿很便宜（沒有網路，只是查一次名字）。
    """
    return chroma_store.get_client().get_collection(chroma_store.COLLECTION)


def _refresh(collection=None, chunks=None, articles=None) -> None:
    """重建 retriever 與條號索引。

    攝取完也要呼叫，否則 `?retriever=numpy` 還拿著舊的矩陣，查不到剛進來的東西。
    """
    with _INDEX_LOCK:
        if collection is None:
            collection, chunks, articles, _t, _c = pipeline.build_index()
        vectors, _t, _c = rag_core.get_embeddings([c["text"] for c in chunks])
        metadatas = [chroma_store.article_metadata(c, articles) for c in chunks]
        STATE.update({
            "collection": collection,
            "chunks": collection.count(),
            "retrievers": {"chroma": ChromaRetriever(_collection),
                           "numpy": NumpyRetriever(vectors, chunks, metadatas)},
            # agent 回來的是一串條號，要能換回原文才組得出引用
            "chunk_index": {m["article_no"]: {"text": c["text"], "meta": m}
                            for c, m in zip(chunks, metadatas)}})


def _build_index() -> None:
    t0 = time.perf_counter()
    collection, chunks, articles, _tok, _cached = pipeline.build_index()
    _refresh(collection, chunks, articles)
    STATE["boot_ms"] = round((time.perf_counter() - t0) * 1000, 1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    # 快取換成常駐＋上鎖＋原子換檔的版本。不換的話並發下會讀到寫到一半的檔案
    if settings.resident_cache:
        cache.install()
        STATE["cache"] = cache.warm(rag_core.CACHE_PATH, rag_core.CHAT_CACHE_PATH)
    STATE["store"] = JobStore(settings.job_db)
    _build_index()
    STATE["ready"] = True
    log.info("ready chunks=%s boot_ms=%s cache=%s",
             STATE["chunks"], STATE["boot_ms"], STATE["cache"])
    try:
        yield
    finally:
        STATE["ready"] = False
        if STATE["store"] is not None:
            STATE["store"].close()
        cache.uninstall()


app = FastAPI(title="TrustRAG Day 24", version=VERSION, lifespan=lifespan)
app.add_middleware(errors.RequestIdMiddleware)
if settings.middleware == "asgi":
    app.add_middleware(timing.TimingASGIMiddleware)
elif settings.middleware == "base":
    app.add_middleware(timing.TimingMiddleware)
errors.install(app)


def _retriever(name: str | None) -> Retriever:
    picked = name or settings.default_retriever
    engine = STATE["retrievers"].get(picked)
    if engine is None:
        raise ServiceError(404, "unknown_retriever",
                           f"沒有這個 retriever：{picked}"
                           f"（有的是 {sorted(STATE['retrievers'])}）")
    return engine


def _require_ready() -> None:
    if not STATE["ready"]:
        raise ServiceError(503, "not_ready", "索引還在建，請稍後再試")


# ------------------------------------------------------------------ 健康

@app.get("/healthz")
def healthz() -> dict:
    """活著嗎。這支永遠回 200，讓 orchestrator 分得出「活著」與「能收流量」。"""
    return {"version": VERSION, "ready": STATE["ready"],
            "chunks": STATE["chunks"], "boot_ms": STATE["boot_ms"],
            "retrievers": sorted(STATE["retrievers"]),
            "default_retriever": settings.default_retriever,
            "middleware": settings.middleware,
            "cache": cache.stats() if settings.resident_cache else {}}


@app.get("/readyz")
def readyz() -> Response:
    _require_ready()
    return Response(status_code=204)


# -------------------------------------------------------------------- 問答

@app.post("/ask")
def ask(req: AskRequest, retriever: str | None = None) -> Response:
    _require_ready()
    t0 = time.perf_counter()
    engine = _retriever(retriever)

    with _INDEX_LOCK:
        if req.mode == "agent":
            text, citations, usage, trace = _run_agent(req, engine)
        else:
            text, citations, usage, trace = _run_pipeline(req, engine)

    payload = AskResponse(
        answer=text or None, citations=citations, usage=usage,
        timing=Timing(embed_ms=0, retrieve_ms=0, llm_ms=0, serialize_ms=0,
                      total_ms=0, overhead_ms=0),
        retriever=engine.name, mode=req.mode, agent=trace)

    # 序列化自己量：先算一次並計時，再把 timing 填回去送出。
    # 兩次序列化的成本一樣，取第一次，不然就變成量自己在量的那一次。
    t_ser = time.perf_counter()
    payload.model_dump_json()
    timing.record("serialize", (time.perf_counter() - t_ser) * 1000)
    payload.timing = Timing(**timing.breakdown((time.perf_counter() - t0) * 1000))

    log.info("ask mode=%s retriever=%s k=%s cached=%s total_ms=%.1f request_id=%s",
             req.mode, engine.name, req.k, usage.cached,
             payload.timing.total_ms, errors.current_request_id())
    return Response(content=payload.model_dump_json(),
                    media_type="application/json")


def _run_pipeline(req: AskRequest, engine) -> tuple:
    """Day 14 那條寫死管線：算一次向量、查一次、問一次模型。"""
    vector, emb_tokens = embedder.get(req.use_cache).embed(req.question)
    with timing.stage("retrieve"):
        hits = engine.search(np.asarray(vector), req.k)
    messages = prompts.build_messages(
        req.question, [{"text": h.text, "meta": h.meta} for h in hits])
    with timing.stage("llm"):
        text, in_tok, out_tok = rag_core.chat(messages, use_cache=req.use_cache)
    citations = [Citation(article_no=h.article_no, title=h.title,
                          similarity=round(h.similarity, 4), text=h.text)
                 for h in hits]
    return text, citations, Usage(
        embedding_tokens=emb_tokens, input_tokens=in_tok, output_tokens=out_tok,
        cached=(in_tok == 0 and out_tok == 0)), None


def _run_agent(req: AskRequest, engine) -> tuple:
    """Day 20／21 那支：自己決定查幾次，手上有 search／check／answer 三個工具。

    引用的處理跟 pipeline 不一樣。pipeline 的 k 條是檢索直接給的；
    agent 查了好幾輪，所以回它查過的條號去重之後的結果，
    順序照它查到的順序，因為那就是它讀到的順序。
    """
    result = agents.run(engine, req.question, req.k,
                        use_cache=req.use_cache, self_check=req.self_check)
    index, seen, citations = STATE["chunk_index"], set(), []
    for no in result["article_nos"]:
        if no in seen or no not in index:
            continue
        seen.add(no)
        entry = index[no]
        citations.append(Citation(article_no=no, title=entry["meta"]["title"],
                                  similarity=0.0, text=entry["text"]))
    trace = AgentTrace(
        steps=result["steps"], searches=result["searches"],
        llm_calls=result["llm_calls"], queries=result["queries"],
        repeats=result["repeats"], hit_cap=result["hit_cap"],
        n_checks=result["n_checks"], flagged=result["flagged"],
        revised=result["revised"])
    usage = Usage(embedding_tokens=result["embedding_tokens"],
                  input_tokens=result["input_tokens"],
                  output_tokens=result["output_tokens"],
                  cached=False)      # agent 的 token 是離線重數的，不是 API 回報的
    return result["answer"], citations, usage, trace


# -------------------------------------------------------------------- 攝取

@app.post("/documents", status_code=202)
def ingest_document(req: IngestRequest, background: BackgroundTasks) -> IngestJob:
    """立刻回，慢的事情丟背景。Day 22 量過一份掃描 PDF 走 OCR 要 277 秒。"""
    _require_ready()
    try:
        settings.resolve_document(req.path)       # 先擋路徑，不要等到背景才失敗
    except ValueError as exc:
        raise ServiceError(400, "invalid_path", str(exc)) from exc

    store = STATE["store"]
    job = store.create(req.path, req.ocr)

    def upsert(source: str, chunks, articles, vectors) -> None:
        """同一份文件攝取兩次是覆蓋，不是追加。

        `_INDEX_LOCK` 擋的是這個行程裡的其他執行緒，`store.exclusive()` 擋的是
        別的 worker——Chroma 內嵌模式不是設計給兩個行程同時寫的。
        """
        with _INDEX_LOCK, store.exclusive():
            collection = _collection()
            metadatas = [chroma_store.article_metadata(c, articles)
                         for c in chunks]
            collection.upsert(
                ids=[ingest.chunk_id(source, m, c["text"])
                     for c, m in zip(chunks, metadatas)],
                embeddings=vectors.tolist(),
                documents=[c["text"] for c in chunks],
                metadatas=metadatas)

    background.add_task(ingest.run, store, job["job_id"], upsert, _refresh)
    log.info("ingest queued job_id=%s path=%s request_id=%s",
             job["job_id"], req.path, errors.current_request_id())
    return IngestJob(**as_contract(job))


@app.get("/documents/{job_id}")
def get_job(job_id: str) -> IngestJob:
    job = STATE["store"].get(job_id)
    if job is None:
        raise ServiceError(404, "job_not_found", f"沒有這個 job_id：{job_id}")
    return IngestJob(**as_contract(job))


@app.get("/documents")
def list_jobs(limit: int = 20) -> list[IngestJob]:
    return [IngestJob(**as_contract(j))
            for j in STATE["store"].recent(min(max(limit, 1), 100))]
