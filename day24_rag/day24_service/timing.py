# -*- coding: utf-8 -*-
"""延遲分解：一個 middleware ＋ 一個計時器。

這份檔案是今天最可以直接抄走的東西，而且跟 RAG 一點關係都沒有。

作法：
- `stage("llm")` 這個 context manager 把耗時累加到「這一個請求」的桶子裡
- 桶子放在 `contextvars.ContextVar`，所以並發的請求不會互相加到對方身上
  （用一個模組層級的 dict 就會）
- middleware 負責量整個請求的牆上時間，並把桶子清乾淨

為什麼不用 `time.time()`：它會被系統校時往回調。`perf_counter()` 是單調的，
量區間就該用它。

overhead 的定義寫清楚：`total - 已知的每一段`。裡面是 Pydantic 驗證、
路由、ASGI 的來回。它不是誤差，是真的有人在花那個時間，所以要留在表上。
"""
import contextvars
import time
from contextlib import contextmanager

from starlette.datastructures import MutableHeaders
from starlette.middleware.base import BaseHTTPMiddleware

_STAGES: contextvars.ContextVar[dict] = contextvars.ContextVar("stages")

STAGE_NAMES = ("embed", "retrieve", "llm", "serialize")


def reset() -> dict:
    bucket: dict = {name: 0.0 for name in STAGE_NAMES}
    _STAGES.set(bucket)
    return bucket


def current() -> dict:
    try:
        return _STAGES.get()
    except LookupError:
        return reset()


@contextmanager
def stage(name: str):
    """量一段，累加進這個請求的桶子。同一段量兩次會相加，這是刻意的
    （例如 Day 17 的重排會讓 llm 這一段出現兩次）。"""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        bucket = current()
        bucket[name] = bucket.get(name, 0.0) + (time.perf_counter() - t0) * 1000


def record(name: str, ms: float) -> None:
    """已經自己量好了，直接記進去"""
    bucket = current()
    bucket[name] = bucket.get(name, 0.0) + ms


class TimingMiddleware(BaseHTTPMiddleware):
    """第一版：Starlette 文件上最常見的寫法。

    網路上關於它有兩個很常見的說法，我本來也寫在這裡當成事實：
    （1）`call_next` 把 endpoint 丟到另一個 task，contextvar 傳不過去、
    header 會是空的；（2）掛在回應上的 BackgroundTasks 會被拉到
    middleware 回來之前跑完，於是「立刻回」變成「等完才回」。

    **這兩件事在 starlette 1.0.0 上我都沒有量到。**
    header 10/10 都有值，`POST /documents` 連線重用之後是 3～8 ms。
    我一開始量到的那 1,306 ms 是我自己的量測腳本每開一條新連線的成本：
    對照組（新連線打什麼事都不做的 /healthz）一樣要 915 ms。

    那還有什麼理由用下面那個純 ASGI 版？少一層包裝而已。
    兩者的 /healthz p50 差 0.03 ms，在噪音裡面。
    這個註解留著是為了記住：抄來的結論也要自己量一次。
    """

    async def dispatch(self, request, call_next):
        reset()
        t0 = time.perf_counter()
        response = await call_next(request)
        total = (time.perf_counter() - t0) * 1000
        bucket = current()
        response.headers["x-total-ms"] = f"{total:.2f}"
        for name in STAGE_NAMES:
            if bucket.get(name):
                response.headers[f"x-{name}-ms"] = f"{bucket[name]:.2f}"
        return response


def _is_event_stream(message: dict) -> bool:
    """這個回應是不是 SSE。在 ASGI 這一層只看得到 raw header 的 bytes。"""
    for key, value in message.get("headers", []):
        if key.lower() == b"content-type":
            return value.lower().startswith(b"text/event-stream")
    return False


class TimingASGIMiddleware:
    """第二版：純 ASGI，少一層包裝。預設用這個。

    它沒有比上面那個快（量過，差在噪音裡），選它的理由只是
    「endpoint 跑在同一個 task 裡」這件事比較好推理——
    contextvar、例外、取消都少一層要想。

    這一段是今天最可以直接抄走的東西，而且跟 RAG 一點關係都沒有。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        reset()
        t0 = time.perf_counter()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # 串流不送計時 header。header 在第一個 byte 就出去了，
                # 那時候什麼都還沒做——實測 x-total-ms 會是 2.88，
                # 而真正的總時間是 50.45。寧可不給，也不要給錯的。
                # 串流的數字在最後那個 done 事件裡。
                if not _is_event_stream(message):
                    total = (time.perf_counter() - t0) * 1000
                    headers = MutableHeaders(scope=message)
                    headers.append("x-total-ms", f"{total:.2f}")
                    bucket = current()
                    for name in STAGE_NAMES:
                        if bucket.get(name):
                            headers.append(f"x-{name}-ms", f"{bucket[name]:.2f}")
            await send(message)

        await self.app(scope, receive, send_wrapper)


def breakdown(total_ms: float) -> dict:
    """把桶子攤成回應要的形狀；overhead 是總時間減掉已知的每一段"""
    bucket = current()
    known = sum(bucket.get(name, 0.0) for name in STAGE_NAMES)
    return {f"{name}_ms": round(bucket.get(name, 0.0), 3) for name in STAGE_NAMES} | {
        "total_ms": round(total_ms, 3),
        "overhead_ms": round(max(total_ms - known, 0.0), 3)}
