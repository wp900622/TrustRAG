# -*- coding: utf-8 -*-
"""Day 30：把這支服務交給別人的 agent。

    python -m day24_service.mcp_server          # stdio，給 Claude Desktop／Cursor 之類的掛

兩個工具：

    search_rules(query, k)                只檢索，不叫模型，一毛錢都不花
    ask(question, mode, max_steps)        問一題，回答案、出處、有沒有被攔下來

**這支程式不 import `main`，它是 HTTP 的客戶端。** 我第一個想法是直接呼叫
`_run_agent()`，少一跳網路。但那樣 MCP 這條路就繞過了 Day 25 的 request id、
Day 27 的帳本、Day 28 的環境變數硬上限，等於在服務旁邊開了一個沒有門的側門。
走 HTTP 的話，別人的 agent 跟我的 curl 看到的是同一份契約、被同一組上限管。

交出去之前要先想清楚的三件事，都寫在這裡：

1. **上限在服務那邊，不在這裡。** 這支程式設的 `TRUSTRAG_MCP_MAX_STEPS` 只是預設值，
   任何人都可以繞過它直接打 HTTP。真正擋得住的是服務的 `DAY24_AGENT_MAX_*`。
2. **過濾條件不是工具參數。** Day 29 的結論：文件篩選是權限，
   不該讓模型自己填。這裡由啟動的人用 `TRUSTRAG_MCP_SOURCE` 決定。
3. **回給模型的是文字，不是 JSON。** 模型看得懂 `stopped_reason="cost"`，
   但它不知道那代表「沒有答案、不是規章裡沒有」。這一句要替它講出來。
"""
from __future__ import annotations

import os
from typing import Literal

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

BASE_URL = os.getenv("TRUSTRAG_URL", "http://127.0.0.1:8024")
# 空字串＝不篩，看得到索引裡所有文件
SOURCE = os.getenv("TRUSTRAG_MCP_SOURCE", "") or None
MAX_STEPS = int(os.getenv("TRUSTRAG_MCP_MAX_STEPS", "4"))
MAX_WALL_MS = float(os.getenv("TRUSTRAG_MCP_MAX_WALL_MS", "20000"))
TIMEOUT = float(os.getenv("TRUSTRAG_MCP_TIMEOUT", "60"))

mcp = FastMCP("trustrag")

# 測試會把它換成 TestClient（它本身就是一個 httpx.Client），不必真的開服務
_http: httpx.Client | None = None


def _client() -> httpx.Client:
    global _http
    if _http is None:
        _http = httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)
    return _http


def _post(path: str, body: dict) -> dict:
    """打服務。錯誤形狀是 Day 25 定的那個，這裡把它翻成模型讀得懂的一句話。"""
    try:
        r = _client().post(path, json=body)
    except httpx.HTTPError as exc:
        raise ToolError(f"規章服務連不上（{BASE_URL}）：{type(exc).__name__}。"
                        "這不是查無資料，請稍後再試或告訴使用者服務沒開。")
    if r.status_code >= 400:
        err = (r.json().get("error") or {}) if r.headers.get(
            "content-type", "").startswith("application/json") else {}
        code = err.get("code", f"http_{r.status_code}")
        # 4xx 是呼叫的人問錯了，改參數重問有用；5xx 重問同一句沒有用
        hint = "請修正參數後再呼叫。" if r.status_code < 500 else "重試同樣的參數沒有幫助。"
        raise ToolError(f"{code}：{err.get('message', r.text[:200])}"
                        f"（request_id={err.get('request_id', '-')}）{hint}")
    return r.json()


def _cite(c: dict) -> str:
    # 帶文件名。Day 29 的 agent 只看得到「第 3 條」，兩份文件都有第 3 條就分不開
    # 標題本身就是「第 24 條（病假）」，原文第一行又是標題，所以只留原文
    where = f"[{c['source']}] " if c.get("source") else ""
    return f"{where}{c['text'].strip()}"


_STOPPED = {
    "steps": "查詢次數到了上限，已經被要求用手上的資料直接作答。答案可能不完整。",
    "cost": "花費到了上限，這一題沒有答案。下面是停下來之前已經找到的條文，"
            "不代表規章裡沒有答案。",
    "time": "時間到了上限，這一題沒有答案。下面是停下來之前已經找到的條文，"
            "不代表規章裡沒有答案。",
}


@mcp.tool()
def search_rules(query: str, k: int = 5) -> str:
    """在公司工作規則裡找最相關的條文，回條號、標題與原文。不會呼叫語言模型。

    適合：你想自己讀原文再判斷，或要引用確切的條號。
    它一定會回最接近的 k 條，就算規章裡根本沒有相關規定也一樣，
    所以請讀原文判斷有沒有回答到問題；每條後面的相似度只能參考，不是答案的信心。
    同一個問題換個說法再查，通常只會拿到同樣的幾條。
    """
    body = {"question": query, "k": max(1, min(k, 10))}
    if SOURCE:
        body["source"] = SOURCE
    hits = _post("/search", body)["hits"]
    if not hits:
        return "沒有結果。"
    return "\n\n".join(f"{_cite(h)}\n（相似度 {h['similarity']:.2f}）" for h in hits)


@mcp.tool()
def ask(question: str,
        mode: Literal["pipeline", "agent"] = "pipeline",
        max_steps: int | None = None) -> str:
    """問一題公司工作規則的問題，回答案與它根據的條文。

    mode="pipeline"（預設）：查一次、答一次，快又便宜，大部分問題用這個就夠。
    mode="agent"：讓規章服務自己決定要查幾次，適合要比對好幾條的問題，
    但比較慢也比較貴。你自己已經是 agent 的話，通常不需要再叫另一個 agent。
    回傳如果寫「被攔下來」，照那一句的意思處理，不要原樣再問一次。
    """
    body: dict = {"question": question, "mode": mode}
    if SOURCE:
        body["source"] = SOURCE
    if mode == "agent":
        # 呼叫端可以要更少，不能要更多。服務那邊還有一層硬上限
        body["max_steps"] = min(max_steps or MAX_STEPS, MAX_STEPS)
        body["max_wall_ms"] = MAX_WALL_MS
    data = _post("/ask", body)

    parts = []
    trace = data.get("agent") or {}
    reason = trace.get("stopped_reason")
    if reason:
        # 不寫「停在第 N 步」：步數上限 2 的時候服務回的是 3，第 3 步是被逼著交卷的那一步。
        # 模型讀到「上限 2、停在 3」只會以為上限沒生效
        parts.append(f"（被攔下來，查了 {trace.get('searches', 0)} 次："
                     f"{_STOPPED.get(reason, reason)}）")
    if data.get("answer"):
        parts.append(data["answer"].strip())
    elif not reason:
        parts.append("（服務沒有給出答案。）")
    if data.get("citations"):
        # 不叫「根據的條文」。pipeline 回的是檢索撈到的 k 條，agent 回的是它查過的每一條，
        # 兩邊都不保證答案用到了。問「有沒有健身房補助」，答案是查不到，
        # 下面卻列著住宿費和保密義務，標成「根據」的話別人的模型會照引
        parts.append("服務查到的條文（不一定每一條都跟答案有關）：\n\n"
                     + "\n\n".join(_cite(c) for c in data["citations"]))
    return "\n\n".join(parts)


if __name__ == "__main__":
    mcp.run()          # 預設就是 stdio
