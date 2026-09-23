# -*- coding: utf-8 -*-
"""串流那條路的測試。

全部走既有快取、不打 API，所以 CI 裡跑幾百次不花錢——
唯一的例外是「中途失敗」那個，它故意讓 stream_chat 炸掉，也沒有真的呼叫。

每個測試都對應一件串流特有、非串流不會遇到的事：
出處能不能先送、狀態碼定死之後怎麼報錯、客戶端走掉會怎樣。
"""
from __future__ import annotations

import json
import time

import pytest

from day24_service import streaming


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """把線路格式拆回 (事件名, 資料) 的序列。"""
    out = []
    event = None
    for line in body.splitlines():
        if line.startswith("event: "):
            event = line[7:].strip()
        elif line.startswith("data: ") and event:
            out.append((event, json.loads(line[6:])))
    return out


def ask_stream(client, **kw):
    body = {"question": "病假請幾天要附診斷證明？", "k": 3, "stream": True}
    body.update(kw)
    r = client.post("/ask", json=body)
    assert r.status_code == 200, r.text
    return r, parse_sse(r.text)


# ------------------------------------------------------------ 事件的形狀

def test_citations_arrive_before_any_token(client):
    """出處要排在第一個 token 前面。這是串流在這裡唯一真正買到的東西，
    順序反過來就白做了。"""
    _, events = ask_stream(client)
    names = [name for name, _ in events]
    assert names[0] == "citations"
    assert "token" in names
    assert names.index("citations") < names.index("token")
    assert names[-1] == "done"


def test_citations_match_the_non_streaming_answer(client):
    """同一題，串流與非串流引用的條號要一樣。
    兩條路共用檢索，這裡若不同就是串流自己走岔了。"""
    _, events = ask_stream(client)
    streamed = [c["article_no"]
                for _, d in events if "citations" in d
                for c in d["citations"]]
    plain = client.post("/ask", json={"question": "病假請幾天要附診斷證明？",
                                      "k": 3}).json()
    assert streamed == [c["article_no"] for c in plain["citations"]]


def test_tokens_join_back_into_the_done_answer(client):
    """把所有 token 事件接起來，要等於 done 裡的完整答案。
    少一段或多一段都代表切段的地方有 bug。"""
    _, events = ask_stream(client)
    joined = "".join(d["text"] for name, d in events if name == "token")
    done = next(d for name, d in events if name == "done")
    assert joined.strip() == done["answer"].strip()


def test_content_type_is_event_stream(client):
    r, _ = ask_stream(client)
    assert r.headers["content-type"].startswith("text/event-stream")
    # 少了這個，前面擺一台 nginx 就會把整串攢起來一次送
    assert r.headers.get("x-accel-buffering") == "no"


# ---------------------------------------------------- 串流特有的兩件麻煩

def test_streaming_does_not_send_lying_timing_headers(client):
    """header 在第一個 byte 就送出去了，那時什麼都還沒做。
    寧可不給，也不要給一個 2.88 ms 的假總時間。"""
    r, events = ask_stream(client)
    assert "x-total-ms" not in r.headers
    assert not [k for k in r.headers if k.endswith("-ms")]
    # 真正的數字在 done 事件裡
    done = next(d for name, d in events if name == "done")
    assert done["timing"]["total_ms"] > 0


def test_non_streaming_still_has_timing_headers(client):
    """上面那條不能誤傷非串流。"""
    r = client.post("/ask", json={"question": "病假請幾天要附診斷證明？"})
    assert float(r.headers["x-total-ms"]) > 0


def test_mid_stream_failure_becomes_an_error_event(client, monkeypatch):
    """狀態碼已經是 200，改不了。炸掉只能用事件講，
    不然呼叫端會以為答案就是講到那裡為止。"""
    def boom(messages, use_cache=True):
        yield "delta", "連續請病假"
        raise RuntimeError("模型連線斷了")

    monkeypatch.setattr(streaming, "stream_chat", boom)
    r, events = ask_stream(client)

    assert r.status_code == 200          # 已經來不及改了
    names = [name for name, _ in events]
    assert names[-1] == "error"
    assert "done" not in names
    err = events[-1][1]
    assert err["code"] == "stream_failed"
    assert err["partial_answer"] is None or "連續" in (err["partial_answer"] or "")
    assert err["request_id"]


# --------------------------------------------------------- 快取命中那條路

def test_cache_hit_replays_without_calling_the_api(client, monkeypatch):
    """命中時不准打 API。打了就是白花錢。"""
    def forbidden(*a, **kw):
        raise AssertionError("命中快取還去呼叫 API")

    monkeypatch.setattr("rag_core.get_client", forbidden)
    _, events = ask_stream(client)
    done = next(d for name, d in events if name == "done")
    assert done["usage"]["cached"] is True
    assert done["usage"]["output_tokens"] == 0


def test_replay_splits_and_rejoins_exactly():
    """切段不能改到內容。"""
    text = "連續請病假三日（含）以上者，應檢附合格醫療院所出具之診斷證明書。"
    pieces = list(streaming.replay(text, size=7))
    assert len(pieces) > 1
    assert "".join(pieces) == text


def test_to_sse_is_wire_format():
    raw = streaming.to_sse("token", {"text": "病假"}).decode("utf-8")
    assert raw.startswith("event: token\ndata: ")
    assert raw.endswith("\n\n")
    # 中文不轉成 \uXXXX，省一半以上的位元組
    assert "病假" in raw


# ------------------------------------------------------------------ 擋掉的

def test_agent_mode_rejects_streaming(client):
    """agent 會來回好幾次，中間吐的 token 不是最終答案，
    直接串出去會讓使用者看到後來被推翻的東西。"""
    r = client.post("/ask", json={"question": "病假請幾天要附診斷證明？",
                                  "stream": True, "mode": "agent"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "stream_not_supported"


@pytest.mark.parametrize("payload", [
    {"question": "", "stream": True},
    {"question": "x", "k": 99, "stream": True},
])
def test_bad_requests_are_still_422_before_the_stream_starts(client, payload):
    """契約在串流開始之前就檢查完了，所以這裡還回得了 422。"""
    assert client.post("/ask", json=payload).status_code == 422


# ------------------------------------------------- 客戶端中途走掉（迴歸測試）

@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_client_disconnect_is_detected_and_logged(client, caplog):
    """把產生器 aclose 掉，模擬客戶端關連線。

    這條是迴歸測試，擋的是第一版那個 bug：產生器寫成同步的，
    Starlette 丟進執行緒池，客戶端走掉時它只是不再被拉取，
    沒有人關它，`except GeneratorExit` 從來沒執行過。
    看起來有處理斷線，實際上沒有——服務會把整段生完、付完錢，
    才發現沒人在聽。改成 async 產生器才會收到 aclose。
    """
    import logging

    from day24_service import main
    from day24_service.contracts import AskRequest

    req = AskRequest(question="病假請幾天要附診斷證明？", k=3, stream=True)
    resp = main._ask_stream(req, main._retriever(None), time.perf_counter())
    it = resp.body_iterator

    first = await it.__anext__()
    assert first.startswith(b"event: citations")

    with caplog.at_level(logging.WARNING, logger="day24"):
        await it.aclose()               # 客戶端走了

    messages = [r.getMessage() for r in caplog.records]
    assert any("客戶端中途斷線" in m for m in messages), messages
