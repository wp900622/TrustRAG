# -*- coding: utf-8 -*-
"""Day 30：用真的 MCP client 走一次 stdio，看別人的 agent 拿到的是什麼。

先開服務（stdout 導到檔案，不要接 PIPE，見 Day 24 的坑）：
    uvicorn day24_service.main:app --port 8030 > service_day30.log 2>&1
再跑：
    python smoke_day30.py

只問快取裡有的題目，不打 API。
"""
import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PARAMS = StdioServerParameters(
    command=sys.executable, args=["-m", "day24_service.mcp_server"],
    env={**os.environ, "TRUSTRAG_URL": "http://127.0.0.1:8030",
         "PYTHONIOENCODING": "utf-8"})

CALLS = [
    ("search_rules", {"query": "病假連續請幾天以上需要附診斷證明？", "k": 2}),
    ("ask", {"question": "病假連續請幾天以上需要附診斷證明？"}),
    ("ask", {"question": "公司有健身房或運動補助嗎？", "mode": "agent", "max_steps": 2}),
    ("ask", {"question": ""}),
]


async def main():
    async with stdio_client(PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            for t in tools:
                print(f"## tool {t.name}  params={list(t.inputSchema['properties'])}")
            for name, args in CALLS:
                result = await session.call_tool(name, args)
                print(f"\n## call {name} {args}  isError={result.isError}")
                print(result.content[0].text)


asyncio.run(main())
