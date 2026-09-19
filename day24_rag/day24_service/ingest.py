# -*- coding: utf-8 -*-
"""攝取：收一份文件，切塊、算向量、放進索引。慢的部分在背景跑。

為什麼一定要走背景：Day 22 量過，一份 20 頁、沒有文字層的 PDF 走離線 OCR
要 277 秒。沒有哪個用戶端會等，中間的反向代理也不會。

第一版有三個問題，這一版都修掉了：

1. **job 狀態放行程記憶體** → 開第二個 worker 就查不到。改放 SQLite（`store.py`）。
2. **chunk 的 id 用 job_id 編號** → 同一份文件攝取兩次會多出一整份，
   索引從 59 塊變 118 塊，檢索開始撈到重複的條文。改成用「文件 + 條號」
   算出穩定 id，重複攝取是覆蓋不是追加。
3. **攝取完沒有重建 retriever** → `?retriever=numpy` 還拿著舊的矩陣，
   查不到剛進來的東西。改成攝取完回呼一次，讓服務重建。
"""
import hashlib
import time
from pathlib import Path

import chunkers
import rag_core

from .config import settings
from .errors import ServiceError


def chunk_id(source: str, meta: dict, text: str) -> str:
    """穩定的 chunk id：同一份文件的同一條，算出來永遠一樣。

    帶上內容的雜湊，所以條文改了字，id 會變、舊的那筆會留下——
    這是刻意的，因為「改過的條文」與「新增的條文」在檢索上是兩件事，
    要清掉舊的請重建索引，不要靠攝取去猜。
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{Path(source).name}:{meta.get('article_no', 0)}:{digest}"


def read_document(path: Path, ocr: bool) -> str:
    """把一份文件變成純文字。

    OCR 那條路直接用 Day 22 的 ingest_day22（含頁首頁尾清理與 NFC 正規化），
    並且沿用它的 ocr_cache_day22.json。那份快取是 277 秒換來的，不能丟。
    """
    if not path.exists():
        raise ServiceError(404, "document_not_found", f"找不到這份文件：{path.name}")
    if path.suffix.lower() != ".pdf":
        return path.read_text(encoding="utf-8")
    if ocr:
        import ingest_day22 as ig22
        return ig22.text_from_pdf_ocr(str(path))
    import fitz
    with fitz.open(str(path)) as doc:
        return chr(10).join(page.get_text() for page in doc)


def run(store, job_id: str, upsert, on_done=None) -> None:
    """背景執行緒跑的東西。

    `upsert` 與 `on_done` 由 main 注入，所以這個檔案整支不認識 Chroma，
    也不認識 retriever。換向量庫的時候這裡一行都不用改。
    """
    job = store.get(job_id)
    if job is None:
        return
    t0 = time.perf_counter()
    store.update(job_id, status="running")
    try:
        path = settings.resolve_document(job["path"])
        text = read_document(path, bool(job["ocr"]))
        chunks = chunkers.chunk_by_structure(text)
        if not chunks:
            raise ServiceError(422, "empty_document", "這份文件切不出任何一塊")
        articles = chunkers.parse_articles(text)
        vectors, _tokens, _hit = rag_core.get_embeddings(
            [c["text"] for c in chunks])
        upsert(path.name, chunks, articles, vectors)
        store.update(job_id, status="done", chunks=len(chunks),
                     elapsed_ms=round((time.perf_counter() - t0) * 1000, 1))
        if on_done is not None:
            on_done()
    except Exception as exc:                  # 失敗要留在 job 上，不是只進 log
        message = (exc.message if isinstance(exc, ServiceError)
                   else f"{type(exc).__name__}: {exc}")
        store.update(job_id, status="failed", error=message[:300],
                     elapsed_ms=round((time.perf_counter() - t0) * 1000, 1))
