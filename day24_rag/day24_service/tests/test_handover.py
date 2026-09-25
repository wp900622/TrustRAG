# -*- coding: utf-8 -*-
"""Day 30：交出去之後，別人看到的是什麼。

人看到的是 `/ui/`，模型看到的是 `mcp_server` 的兩個工具。兩邊都只是這支服務的客戶端，
所以這裡測的不是「答得對不對」（前 29 天測過了），而是交出去的那一層有沒有把話講清楚：
工具的參數表裡有什麼、沒有什麼；被攔下來、服務壞掉、服務沒開時，模型讀到的是哪一句。

全部走既有快取，不打 API。
"""
import asyncio
import json
import re

import httpx
import pytest

import rag_core

from day24_service import mcp_server

QUESTION = "病假連續請幾天以上需要附診斷證明？"
LONG_QUESTION = "公司有健身房或運動補助嗎？"      # Day 21 撞上限的那題


def _forbidden(*_a, **_kw):
    raise AssertionError("測試不准打 API")


@pytest.fixture
def via_service(client, monkeypatch):
    """MCP 工具接到測試用的 app 上。TestClient 本身就是 httpx.Client。"""
    monkeypatch.setattr(rag_core, "get_client", _forbidden)
    monkeypatch.setattr(mcp_server, "_http", client)
    yield client


def _fake(handler):
    """換一個假的服務：回什麼由 handler 決定，順便記下工具送了什麼出去。"""
    sent = []

    def record(request):
        sent.append(json.loads(request.content))
        return handler(request)
    return httpx.Client(transport=httpx.MockTransport(record),
                        base_url="http://fake"), sent


def _tools():
    return {t.name: t for t in asyncio.run(mcp_server.mcp.list_tools())}


# ------------------------------------------------------------------ 網頁

def test_ui_is_served_from_the_same_origin(client):
    """static/ 是 day30_web（Vue）build 出來的。HTML 只剩一個殼，要順著它找到 JS。"""
    r = client.get("/ui/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    scripts = re.findall(r'src="(/ui/assets/[^"]+\.js)"', r.text)
    assert scripts, "static/ 不是 build 出來的？先在 day30_web 跑 npm run build"
    js = client.get(scripts[0])
    assert js.status_code == 200
    # 網頁打的是同一支 /ask，不是另一套 API，也不是寫死的主機名
    assert '"/ask"' in js.text and "127.0.0.1" not in js.text


# -------------------------------------------------------------- 工具長相

def test_two_tools_and_nothing_else():
    assert set(_tools()) == {"search_rules", "ask"}


def test_filters_are_not_tool_parameters():
    """Day 29：文件篩選是權限，不讓模型填。參數表裡不能出現。"""
    for tool in _tools().values():
        props = tool.inputSchema["properties"]
        assert "source" not in props and "chapter_no" not in props
        # 花費與時間上限也不交給模型，只有步數可以往下調
        assert "max_twd" not in props and "max_wall_ms" not in props


def test_tool_description_tells_the_model_what_not_to_do():
    """工具說明本身就是 prompt。這兩句會改變別人的 agent 要不要多查一次。"""
    assert "不需要再叫另一個 agent" in _tools()["ask"].description
    assert "一定會回最接近的 k 條" in _tools()["search_rules"].description


# --------------------------------------------------------------- 走真服務

def test_search_rules_goes_through_the_service(via_service):
    text = mcp_server.search_rules(QUESTION, k=3)
    assert text.count("（相似度") == 3
    assert text.startswith("[work_rules.md] 第 24 條")   # 帶文件名，不只條號


def test_ask_pipeline_returns_answer_and_citations(via_service):
    text = mcp_server.ask(QUESTION)
    assert "服務查到的條文" in text
    assert "被攔下來" not in text


def test_step_cap_is_readable_by_the_model(via_service):
    """步數上限走凍結 agent 自己的強制作答，所以有答案，但模型要知道它是被截斷的。"""
    text = mcp_server.ask(LONG_QUESTION, mode="agent", max_steps=2)
    assert text.startswith("（被攔下來，查了 2 次")
    assert "答案可能不完整" in text


def test_citations_are_not_called_evidence(via_service):
    """陷阱題：答案是查不到，出處卻有好幾條不相干的。不能標成「根據」。"""
    text = mcp_server.ask(LONG_QUESTION, mode="agent", max_steps=2)
    assert "根據" not in text
    assert "不一定每一條都跟答案有關" in text


def test_called_through_the_mcp_layer(via_service):
    """不只呼叫 Python 函式：經過 FastMCP 的參數驗證與包裝，拿到的仍是那段文字。"""
    result = asyncio.run(mcp_server.mcp.call_tool(
        "search_rules", {"query": QUESTION, "k": 2}))
    blocks = result[0] if isinstance(result, tuple) else result
    assert blocks[0].text.startswith("[work_rules.md] 第 24 條")


# --------------------------------------------------------------- 假服務

def test_model_cannot_ask_for_more_steps(monkeypatch):
    fake, sent = _fake(lambda r: httpx.Response(200, json={
        "answer": "x", "citations": [], "agent": {"stopped_reason": None}}))
    monkeypatch.setattr(mcp_server, "_http", fake)
    mcp_server.ask("q", mode="agent", max_steps=99)
    mcp_server.ask("q", mode="agent", max_steps=2)
    assert [b["max_steps"] for b in sent] == [mcp_server.MAX_STEPS, 2]


def test_cost_stop_is_not_mistaken_for_no_rule(monkeypatch):
    """answer 是 null 的時候，模型最容易講成「規章裡沒有」。這一句要替它擋掉。"""
    fake, _ = _fake(lambda r: httpx.Response(200, json={
        "answer": None,
        "citations": [{"article_no": 47, "title": "工作時間", "text": "……",
                       "source": "work_rules.md", "similarity": 0.0}],
        "agent": {"stopped_reason": "cost", "stopped_at_step": 2}}))
    monkeypatch.setattr(mcp_server, "_http", fake)
    text = mcp_server.ask("q", mode="agent")
    assert "不代表規章裡沒有答案" in text
    assert "[work_rules.md] ……" in text


def test_service_errors_become_one_sentence(monkeypatch):
    fake, _ = _fake(lambda r: httpx.Response(422, json={"error": {
        "code": "invalid_request", "message": "question：太長",
        "request_id": "abc123"}}))
    monkeypatch.setattr(mcp_server, "_http", fake)
    with pytest.raises(mcp_server.ToolError) as exc:
        mcp_server.ask("q")
    msg = str(exc.value)
    assert "invalid_request" in msg and "abc123" in msg and "修正參數" in msg


def test_service_down_is_not_no_result(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)
    fake, _ = _fake(refuse)
    monkeypatch.setattr(mcp_server, "_http", fake)
    with pytest.raises(mcp_server.ToolError) as exc:
        mcp_server.search_rules("q")
    assert "這不是查無資料" in str(exc.value)
