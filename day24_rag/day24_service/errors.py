# -*- coding: utf-8 -*-
"""統一的錯誤形狀，以及一個跟著請求跑的 id。

腳本時代出錯就是一個 traceback 印在我的終端機，我自己看得懂就好。
服務的錯誤要給三種人看：呼叫端（要知道能不能重試）、我（要能在 log 裡
找到同一次請求）、還有三個月後的我（要知道當時發生什麼事）。

所以每個錯誤都長同一個樣子：

    {"error": {"code": "document_not_found",
               "message": "找不到這份文件：foo.pdf",
               "request_id": "8f1c2b7e"}}

`code` 給程式判斷，`message` 給人看，`request_id` 兩邊都要——
它會同時出現在回應的 header、body，以及服務的 log 裡。
"""
import contextvars
import logging
import uuid

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders

log = logging.getLogger("day24")

_REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar("request_id")


def current_request_id() -> str:
    try:
        return _REQUEST_ID.get()
    except LookupError:
        return "-"


class ServiceError(Exception):
    """服務自己丟的錯。狀態碼與代號一起帶著，handler 不必再判斷一次。"""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message,
                      "request_id": current_request_id()}}


class RequestIdMiddleware:
    """每個請求發一個 id，寫進 contextvar 與回應 header。

    呼叫端已經帶 `x-request-id` 的話就沿用它，這樣跨服務追得下去。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        incoming = dict(scope.get("headers") or {})
        given = None
        for key, value in scope.get("headers") or []:
            if key == b"x-request-id":
                given = value.decode("latin-1")[:64]
                break
        request_id = given or uuid.uuid4().hex[:8]
        token = _REQUEST_ID.set(request_id)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append("x-request-id", request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            _REQUEST_ID.reset(token)


def install(app) -> None:
    """把三種錯誤都收斂成同一個形狀。"""

    @app.exception_handler(ServiceError)
    async def _service_error(request: Request, exc: ServiceError):
        log.warning("service_error code=%s request_id=%s %s",
                    exc.code, current_request_id(), exc.message)
        return JSONResponse(status_code=exc.status_code,
                            content=_body(exc.code, exc.message))

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code,
                            content=_body(f"http_{exc.status_code}",
                                          str(exc.detail)))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        # 契約擋下來的東西要講清楚是哪個欄位，不然呼叫端只會看到 422
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(x) for x in first.get("loc", ())[1:]) or "body"
        return JSONResponse(
            status_code=422,
            content=_body("invalid_request",
                          f"{where}：{first.get('msg', '格式不符')}"))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        # 沒接到的例外不要把 traceback 吐給呼叫端，但 log 裡要留全的
        log.exception("unhandled request_id=%s", current_request_id())
        return JSONResponse(status_code=500,
                            content=_body("internal_error",
                                          "服務發生未預期的錯誤"))
