# -*- coding: utf-8 -*-
"""Day 24：把前 23 天散落的腳本收成一支服務。

    POST /ask              問一題。mode=pipeline 走 Day 14 的寫死管線，
                           mode=agent 走 Day 20／21 那支自己決定查幾次的
    POST /documents        攝取一份文件，立刻回 202 與 job_id（背景跑）
    GET  /documents/{id}   查攝取進度
    GET  /documents        最近的攝取紀錄
    GET  /healthz          活著嗎：索引塊數、retriever、快取、版本
    GET  /readyz           可以收流量了嗎：索引建好了沒
    GET  /ui/              Day 30：一頁給人用的網頁（模型用的在 mcp_server.py）

檢索與生成的邏輯一個字都沒改（`chunkers` / `prompts` / `rag_core` 全部沿用），
動的是它們外面那一層：設定、快取、狀態、錯誤、觀測。

handler 刻意寫成同步 `def`：底下那些 OpenAI 呼叫是會阻塞的 sdk，
寫成 `async def` 會把整個 event loop 卡住；同步的會被 Starlette 丟進執行緒池。

跑起來：
    uvicorn day24_service.main:app --port 8024
"""
import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import BackgroundTasks, FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

import chroma_store
import chunkers
import prompts
import rag_core
from chromadb.errors import NotFoundError

from . import (agents, budget, cache, embedder, errors, ingest, ledger,
               streaming, timing)
from .config import settings
from .contracts import (AgentTrace, AskRequest, AskResponse, Citation,
                        DeleteResult, IngestJob, IngestRequest, SearchRequest,
                        SearchResponse, SourceSummary, Timing, Usage)
from .errors import ServiceError
from .retrievers import ChromaRetriever, NumpyRetriever, Retriever
from .store import JobStore, as_contract

VERSION = "24.1"

# 開機那份語料的來源名。Day 29 之前索引裡沒有這個欄位，
# 因為在那之前索引裡永遠只有一份文件
BASE_SOURCE = rag_core.DOC_PATH.name
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
    return chroma_store.get_client().get_collection(settings.collection)


def _stamp(metadatas: list[dict], source: str) -> list[dict]:
    """替每一塊補上它是哪份文件來的。

    `chroma_store.article_metadata()` 是 Day 14 的檔案，前 15 天的實驗都靠它，
    所以不去動它，補在服務這一層。多這個欄位不會讓既有的快取失效：
    進 prompt 的只有 `article_no` 與 `chapter`（`prompts.build_messages`）。
    """
    for meta in metadatas:
        meta["source"] = source
    return metadatas


def _where(source: str | None, chapter_no: int | None) -> dict | None:
    """把請求裡的兩個欄位翻成檢索條件。都沒給就回 None，走原本的整庫檢索。

    只做等值比對，因為 `NumpyRetriever` 也要做得到同一件事。
    服務對外承諾的，只能是兩個實作都給得起的東西。
    """
    clauses = {}
    if source:
        clauses["source"] = {"$eq": source}
    if chapter_no is not None:
        clauses["chapter_no"] = {"$eq": chapter_no}
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses
    return {"$and": [{k: v} for k, v in clauses.items()]}


def _upsert(collection, source: str, chunks, articles, vectors) -> None:
    """寫進索引只有這一條路：開機種第一份語料、攝取，都走這裡。

    id 由「文件 + 條號 + 內容雜湊」算出來，同一份文件寫兩次是覆蓋不是追加。
    Day 29 之前開機走的是 `chroma_store.rebuild()`，id 是 `art-{條號}`；
    攝取走的是這一套。同一份文件在索引裡有兩種 id，所以攝取一次就從 59 塊
    變成 118 塊，只是下一步會把整個庫砍掉重建，沒有人看見。
    """
    metadatas = _stamp(
        [chroma_store.article_metadata(c, articles) for c in chunks], source)
    collection.upsert(
        ids=[ingest.chunk_id(source, m, c["text"])
             for c, m in zip(chunks, metadatas)],
        embeddings=vectors.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=metadatas)


def _open_index():
    """開服務自己的 collection。只有第一次（它還不存在）才種開機那份語料。

    Day 28 以前這裡呼叫的是 `pipeline.build_index()`，走到 Day 14 給實驗用的
    `open_or_build()`：塊數不是 59 就當成半成品，砍掉重建。對實驗那是對的，
    每一輪都要從同一個起點開始。對服務，那是在每次攝取完、每次重啟的時候，
    把別人攝取進來的文件全部刪掉。

    判斷的是「collection 在不在」，不是「開機那份在不在」：
    有人用 DELETE /sources 把它拿掉了，重啟之後不該自己長回來。
    """
    client = chroma_store.get_client()
    try:
        return client.get_collection(settings.collection)
    except NotFoundError:
        pass
    text = rag_core.load_document()
    chunks = chunkers.chunk_by_structure(text)
    articles = chunkers.parse_articles(text)
    vectors, _tok, _hit = rag_core.get_embeddings([c["text"] for c in chunks])
    collection = client.create_collection(
        settings.collection, configuration={"hnsw": {"space": "cosine"}})
    _upsert(collection, BASE_SOURCE, chunks, articles, vectors)
    log.info("seeded collection=%s source=%s chunks=%s",
             settings.collection, BASE_SOURCE, collection.count())
    return collection


def _refresh() -> None:
    """拿索引裡現在有的東西，重建兩個 retriever 與條號表。

    開機、攝取完、刪除完都呼叫這一支，而且只讀索引，不讀磁碟上的語料。
    Day 28 以前攝取完呼叫的版本會重讀 `work_rules.md` 來建 numpy 的矩陣，
    所以就算索引沒被砍，`?retriever=numpy` 也只看得到開機那一份。
    兩個 retriever 的資料從同一個地方來，才談得上「兩邊結果一樣」。
    """
    with _INDEX_LOCK:
        collection = _collection()
        got = collection.get(include=["documents", "metadatas", "embeddings"])
        metadatas = [dict(m or {}) for m in got["metadatas"]]
        chunks = [{"text": d} for d in got["documents"]]
        # 全部刪光時 Chroma 回一個空 list，直接 asarray 會得到 shape (0,)，
        # 那個形狀沒辦法跟 1536 維的問題向量相乘
        vectors = (np.asarray(got["embeddings"], dtype=np.float32)
                   if len(got["ids"]) else np.zeros((0, 1), dtype=np.float32))
        STATE.update({
            "collection": collection,
            "chunks": collection.count(),
            "retrievers": {"chroma": ChromaRetriever(_collection),
                           "numpy": NumpyRetriever(vectors, chunks, metadatas)},
            # agent 回來的是一串條號，要能換回原文才組得出引用。
            # Day 29 起 key 是（文件, 條號）：兩份文件都有第 3 條的時候，
            # 只用條號當 key 會讓後進來的那份直接蓋掉前一份
            "chunk_index": {(str(m.get("source", "")), m["article_no"]):
                            {"text": c["text"], "meta": m}
                            for c, m in zip(chunks, metadatas)}})


def _build_index() -> None:
    t0 = time.perf_counter()
    # 兩個 worker 同時開機，只能有一個去建 collection
    with STATE["store"].exclusive():
        _open_index()
    _refresh()
    STATE["boot_ms"] = round((time.perf_counter() - t0) * 1000, 1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    # 快取換成常駐＋上鎖＋原子換檔的版本。不換的話並發下會讀到寫到一半的檔案
    if settings.resident_cache:
        backend = cache.install()
        seeded = cache.warm(rag_core.CACHE_PATH, rag_core.CHAT_CACHE_PATH)
        STATE["cache"] = {"backend": backend, **seeded}
    STATE["store"] = JobStore(settings.job_db)
    # 帳本要在收第一個請求之前讀回來，不然重啟一次命中率就從零開始
    STATE["ledger"] = ledger.load()
    _build_index()
    STATE["ready"] = True
    log.info("ready chunks=%s boot_ms=%s cache=%s ledger=%s",
             STATE["chunks"], STATE["boot_ms"], STATE["cache"], STATE["ledger"])
    try:
        yield
    finally:
        STATE["ready"] = False
        ledger.flush()          # 最後那幾筆還沒落地的命中
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
# 同源掛上來，網頁直接打 /ask，不必處理 CORS
app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "static", html=True),
          name="ui")


def _retriever(name: str | None) -> Retriever:
    picked = name or settings.default_retriever
    engine = STATE["retrievers"].get(picked)
    if engine is None:
        raise ServiceError(404, "unknown_retriever",
                           f"沒有這個 retriever：{picked}"
                           f"（有的是 {sorted(STATE['retrievers'])}）")
    return engine


def _chunk_entry(article_no: int, source: str | None = None) -> dict | None:
    """條號換原文。給了文件就精準查，沒給就退回「第一個有這個條號的」。

    退回的那條路是 agent 留下的：凍結的 `agent_day21.py` 只帶條號回來，
    帶不回它是在哪份文件裡看到的。索引裡不只一份文件時，那個引用本身就有歧義，
    所以 `RetrieverCollection` 現在會把（文件, 條號）成對記下來，
    這個 fallback 只剩下舊資料在用。
    """
    index = STATE["chunk_index"]
    if source is not None and (source, article_no) in index:
        return index[(source, article_no)]
    for (_src, no), entry in index.items():
        if no == article_no:
            return entry
    return None


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


@app.get("/usage")
def usage() -> dict:
    """這支服務到目前為止花了多少、擋下了多少。

    Day 27 要花一整天考古才問得出來的東西，現在是一次 GET。
    `unpriced_hits` 是命中了、但那一筆的金額是這個帳本存在之前買的——
    它不歸零，「省了多少」就還是不完整的，所以它露在回應裡。
    """
    return {"version": VERSION, **ledger.summary()}


# -------------------------------------------------------------------- 問答

@app.post("/ask")
def ask(req: AskRequest, retriever: str | None = None) -> Response:
    _require_ready()
    t0 = time.perf_counter()
    engine = _retriever(retriever)

    if req.stream:
        if req.mode == "agent":
            raise ServiceError(400, "stream_not_supported",
                               "agent 目前不支援串流，它會來回好幾次，"
                               "中間的 token 不是最終答案")
        return _ask_stream(req, engine, t0)

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


_PULL_DONE = object()


def _ask_stream(req: AskRequest, engine, t0: float) -> StreamingResponse:
    """SSE 版的 /ask。跟非串流走同一條管線，差別只在答案怎麼送出去。

    事件順序是刻意的：`citations` 在模型開口之前就送得出去，
    因為出處在檢索完就定案了。這是串流真正買到的東西——
    不只是字會跳出來，是使用者在等模型的那一秒裡有東西可以讀。

    **這個產生器是 async 的，跟服務裡其他 handler 相反。**
    原因只有一個：要偵測得到客戶端中途走掉。
    第一版寫成同步產生器（跟其他 handler 一致，Starlette 會丟進執行緒池），
    結果客戶端斷線時它只是不再被拉取，沒有人關它，
    `except GeneratorExit` 那段從來沒執行過——看起來有處理，其實沒有。
    async 版本才會被 `aclose()`，`CancelledError` 才進得來。

    代價是每一個會阻塞的呼叫都得自己丟進執行緒池，所以底下到處是
    `run_in_threadpool`。漏掉任何一個，整個 event loop 就會被那個
    OpenAI 呼叫卡住。

    串流的另一件麻煩：**出錯了改不了狀態碼。** 第一個 byte 送出去，
    狀態就定在 200。後面炸掉只能靠一個 `error` 事件講，
    不然呼叫端會以為答案就是講到那裡為止。
    """
    async def events():
        first_token_ms = None
        text, in_tok, out_tok, hit = "", 0, 0, False
        emb_tokens = 0
        gen = None
        key = None          # 記帳用，跟快取共用同一把。斷線那條路也要看得到它
        sent = 0            # 已經送出去幾段。斷線時 text 還是空的，要靠它
        sent_chars = 0
        try:
            embed = embedder.get(req.use_cache)
            vector, emb_tokens = await run_in_threadpool(embed.embed, req.question)

            def _retrieve():
                # 只在真的碰索引的時候上鎖。非串流那條抓著 _INDEX_LOCK
                # 走完整個請求沒問題，但串流會抓著它等模型講完話——
                # 那一秒多裡攝取完全沒辦法進來
                with _INDEX_LOCK:
                    with timing.stage("retrieve"):
                        return engine.search(np.asarray(vector), req.k,
                                             _where(req.source, req.chapter_no))

            hits = await run_in_threadpool(_retrieve)

            citations = [Citation(article_no=h.article_no, title=h.title,
                                  similarity=round(h.similarity, 4),
                                  text=h.text, source=h.source)
                         for h in hits]
            # 先送出處。這一刻模型還沒被呼叫
            yield streaming.to_sse("citations", {
                "citations": [c.model_dump() for c in citations],
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)})

            messages = prompts.build_messages(
                req.question, [{"text": h.text, "meta": h.meta} for h in hits])
            key = ledger.chat_key(messages)

            t_llm = time.perf_counter()
            gen = streaming.stream_chat(messages, req.use_cache)
            while True:
                # 一次拉一段。每個 await 都是一個取消點，
                # 客戶端走掉時就停在這裡，不會把整串生完才發現沒人在聽
                kind, payload = await run_in_threadpool(_pull, gen)
                if kind is None:
                    break
                if kind == "delta":
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - t0) * 1000
                    sent += 1
                    sent_chars += len(payload)
                    yield streaming.to_sse("token", {"text": payload})
                else:
                    text, in_tok, out_tok, hit = payload
            timing.record("llm", (time.perf_counter() - t_llm) * 1000)

        except (asyncio.CancelledError, GeneratorExit):
            # 客戶端走了。不能再 yield，只能記下來。
            # 那次呼叫已經花掉的錢是要不回來的——Day 26 寫到這裡就停了，
            # 因為當時講不出它值多少。現在帳本查得到同一把 key 上次付過多少
            burned = ledger.abandoned(key) if key else 0.0
            log.warning("ask stream 客戶端中途斷線 已送出 %d 段／%d 字 "
                        "約燒掉 %.4f 元 elapsed_ms=%.1f request_id=%s",
                        sent, sent_chars, burned,
                        (time.perf_counter() - t0) * 1000,
                        errors.current_request_id())
            raise

        except Exception as exc:                          # noqa: BLE001
            log.exception("ask stream 中途失敗 request_id=%s",
                          errors.current_request_id())
            yield streaming.to_sse("error", {
                "code": "stream_failed",
                "message": f"{type(exc).__name__}: {exc}",
                "partial_answer": text or None,
                "sent_chars": sent_chars,
                "request_id": errors.current_request_id()})
            return

        finally:
            # 關掉底下那個同步產生器，OpenAI 那條連線才會跟著收掉。
            # 不關的話它會一直掛著直到被 GC，而且錢照算
            if gen is not None:
                gen.close()

        total_ms = (time.perf_counter() - t0) * 1000
        usage = _account(key, emb_tokens, in_tok, out_tok)
        yield streaming.to_sse("done", {
            "answer": text,
            "usage": usage.model_dump(),
            "timing": timing.breakdown(total_ms),
            "first_token_ms": round(first_token_ms, 1) if first_token_ms else None,
            "retriever": engine.name})

        log.info("ask stream retriever=%s k=%s cached=%s cost_twd=%.4f "
                 "first_token_ms=%s total_ms=%.1f request_id=%s",
                 engine.name, req.k, usage.cached, usage.cost_twd,
                 round(first_token_ms, 1) if first_token_ms else None,
                 total_ms, errors.current_request_id())

    return StreamingResponse(
        events(), media_type="text/event-stream",
        # 這兩個是給中間那些會幫你緩衝的東西看的。少了它們，
        # nginx 之類的會把整串 SSE 攢起來一次送，串流就白做了
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _pull(gen):
    """從同步產生器拉一段。拉完了回 (None, None)，
    這樣呼叫端就不必在執行緒池那一層處理 StopIteration。"""
    item = next(gen, None)
    return item if item is not None else (None, None)


def _run_pipeline(req: AskRequest, engine) -> tuple:
    """Day 14 那條寫死管線：算一次向量、查一次、問一次模型。"""
    vector, emb_tokens = embedder.get(req.use_cache).embed(req.question)
    with timing.stage("retrieve"):
        hits = engine.search(np.asarray(vector), req.k,
                             _where(req.source, req.chapter_no))
    messages = prompts.build_messages(
        req.question, [{"text": h.text, "meta": h.meta} for h in hits])
    with timing.stage("llm"):
        text, in_tok, out_tok = rag_core.chat(messages, use_cache=req.use_cache)
    citations = [Citation(article_no=h.article_no, title=h.title,
                          similarity=round(h.similarity, 4), text=h.text,
                          source=h.source)
                 for h in hits]
    return (text, citations,
            _account(ledger.chat_key(messages), emb_tokens, in_tok, out_tok), None)


def _account(key: str, emb_tokens: int, in_tok: int, out_tok: int) -> Usage:
    """把這次呼叫記進帳本，順便組出回應要的 usage。

    命中與否只有一個判準：`rag_core.chat()` 命中時回 `(text, 0, 0)`。
    那個 0 以前是資訊的終點（所以 Day 27 得考古），現在是查帳本的訊號。
    """
    if in_tok == 0 and out_tok == 0:
        return Usage(embedding_tokens=emb_tokens, cached=True,
                     cost_twd=ledger.cost_twd(rag_core.CHAT_MODEL, 0, 0, emb_tokens),
                     avoided_twd=ledger.hit(key))
    return Usage(embedding_tokens=emb_tokens, input_tokens=in_tok,
                 output_tokens=out_tok, cached=False,
                 cost_twd=round(ledger.record(key, rag_core.CHAT_MODEL,
                                              in_tok, out_tok, emb_tokens), 6))


def _run_agent(req: AskRequest, engine) -> tuple:
    """Day 20／21 那支：自己決定查幾次，手上有 search／check／answer 三個工具。

    引用的處理跟 pipeline 不一樣。pipeline 的 k 條是檢索直接給的；
    agent 查了好幾輪，所以回它查過的條號去重之後的結果，
    順序照它查到的順序，因為那就是它讀到的順序。
    """
    limits = budget.resolve(req.max_steps, req.max_twd, req.max_wall_ms)
    result = agents.run(engine, req.question, req.k,
                        use_cache=req.use_cache, self_check=req.self_check,
                        limits=limits, where=_where(req.source, req.chapter_no))
    seen, citations = set(), []
    # 轉接頭把（文件, 條號）成對記下來了，條號自己是有歧義的
    for source, no in result.get("cited_pairs") or [
            (None, n) for n in result["article_nos"]]:
        entry = _chunk_entry(no, source)
        if entry is None or (source, no) in seen:
            continue
        seen.add((source, no))
        citations.append(Citation(article_no=no, title=entry["meta"]["title"],
                                  similarity=0.0, text=entry["text"],
                                  source=entry["meta"].get("source", "")))
    trace = AgentTrace(
        steps=result["steps"], searches=result["searches"],
        llm_calls=result["llm_calls"], queries=result["queries"],
        repeats=result["repeats"], hit_cap=result["hit_cap"],
        n_checks=result["n_checks"], flagged=result["flagged"],
        revised=result["revised"],
        stopped_reason=result.get("stopped_reason"),
        stopped_at_step=result.get("stopped_at_step"))
    # agent 的帳是逐次記的：一題打好幾次模型，其中幾次可能命中快取。
    # `cached` 對 agent 不是布林——有的次命中、有的次沒有，所以看的是
    # 「這一題有沒有真的付錢」，細目在 usage 的金額欄位裡
    tally = result["ledger"]
    usage = Usage(embedding_tokens=result["embedding_tokens"],
                  input_tokens=result["input_tokens"],
                  output_tokens=result["output_tokens"],
                  cached=(tally["paid_calls"] == 0),
                  cost_twd=round(tally["cost_twd"], 6),
                  avoided_twd=(round(tally["avoided_twd"], 6)
                               if tally["hits"] else None))
    return result["answer"], citations, usage, trace


# -------------------------------------------------------------------- 檢索

@app.post("/search")
def search(req: SearchRequest, retriever: str | None = None) -> SearchResponse:
    """只檢索，不問模型。

    Day 24 到 Day 28 這支服務只有 `/ask` 一個入口，而它一定會叫模型。
    要知道「這個問題撈得到哪幾條」，得連生成的錢一起付。檢索是一個獨立的能力，
    這裡把它單獨開出來，順便讓過濾這件事有地方被看見。
    """
    _require_ready()
    t0 = time.perf_counter()
    engine = _retriever(retriever)
    where = _where(req.source, req.chapter_no)
    vector, emb_tokens = embedder.get(req.use_cache).embed(req.question)
    with _INDEX_LOCK:
        hits = engine.search(np.asarray(vector), req.k, where)
    if req.min_similarity is not None:
        hits = [h for h in hits if h.similarity >= req.min_similarity]
    return SearchResponse(
        hits=[Citation(article_no=h.article_no, title=h.title,
                       similarity=round(h.similarity, 4), text=h.text,
                       source=h.source) for h in hits],
        retriever=engine.name, k=req.k,
        filtered=bool(where) or req.min_similarity is not None,
        embedding_tokens=emb_tokens,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 2))


@app.get("/sources")
def list_sources() -> list[SourceSummary]:
    """索引裡現在有哪幾份文件，各幾塊。

    `/healthz` 只講得出總塊數。攝取過第二份文件之後，那個數字就不夠用了：
    59 變成 137 的時候，沒有人講得出多出來的 78 塊是誰的。
    """
    _require_ready()
    with _INDEX_LOCK:
        got = STATE["collection"].get(include=["metadatas"])
    grouped: dict[str, list[int]] = {}
    for meta in got["metadatas"]:
        meta = meta or {}
        grouped.setdefault(str(meta.get("source", "")), []).append(
            int(meta.get("article_no", 0)))
    return [SourceSummary(source=name, chunks=len(nos),
                          articles=sorted(set(nos)))
            for name, nos in sorted(grouped.items())]


@app.delete("/sources/{source}")
def delete_source(source: str) -> DeleteResult:
    """把一份文件從索引裡拿掉。

    到今天為止這支服務只進不出：攝取錯一份文件，唯一的救法是砍掉整個
    collection 重建，連對的那幾份一起賠進去。

    刪完要重建 retriever，否則 `?retriever=numpy` 手上還是舊的矩陣，
    刪掉的東西照樣被查得到（攝取那條路 Day 24 就踩過同一個坑）。
    """
    _require_ready()
    store = STATE["store"]
    with _INDEX_LOCK, store.exclusive():
        collection = _collection()
        before = collection.count()
        got = collection.get(where={"source": {"$eq": source}},
                             include=["metadatas"])
        if not got["ids"]:
            raise ServiceError(404, "source_not_found",
                               f"索引裡沒有這份文件：{source}")
        collection.delete(ids=got["ids"])
        after = collection.count()
    _refresh()
    log.info("delete source=%s removed=%s left=%s request_id=%s",
             source, before - after, after, errors.current_request_id())
    return DeleteResult(source=source, removed=before - after,
                        chunks_left=after)


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
            _upsert(_collection(), source, chunks, articles, vectors)

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
