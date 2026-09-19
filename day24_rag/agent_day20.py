# -*- coding: utf-8 -*-
"""Day 20：最小的 agent —— 一個會自己決定下一步的迴圈。

前 19 天的檢索是一條寫死的管線：問題原句去查一次、k 是我定的、查完就回答。
今天把那條管線換成迴圈，每一輪模型只能做兩件事之一：

    search(query)   我要查規章，用這串字去查
    answer(text)    我查夠了，這是答案

模型自己決定：第一步要不要查、query 寫什麼（不一定是使用者原句）、
要不要再查一次。**這就是 agent 的最小形式**——不是框架，不是多角色，
不是記憶體，就是一個會自己選下一步的迴圈。

## 一個刻意的限制：每次 search 一律 k=3

跟對照組同一個數字。這樣「agent 比較好」才不會只是因為它偷偷拿了更多條文。
它想要更多條文，得自己多查一次，**而那要付錢**——這正是今天要量的東西。

## 一個刻意的「沒加」：不教它拒答

`SYSTEM` 是 Day 14 `prompts.SYSTEM` 的原文，只把「使用者提供的條文」改成
「查到的條文」（agent 的條文是自己查來的，不改講不通）。除此之外一個字都沒加，
**特別是沒有加「查不到就說規章未規定」**——因為對照組（P1）也沒有這句。

加了，陷阱題上的差別就變成 prompt 的差別，不是迴圈的差別。
今天的變因只准有一個。

## 三道防線（agent 特有的失敗模式）

    最多 6 步        迴圈不收斂會燒到天亮。撞到上限**單獨計數**，不准當成正常結束
    同 query 不重查  模型卡住時最常見的行為就是原地重問
    撞上限強制作答   不准回傳空白——沒有答案也是一種結果，要被尺量到

## token 怎麼算

沿用 Day 16 起的規矩：**離線重數，不看 API 回報的 usage**（快取全命中時
usage 是 0）。但 agent 多了一項算不準的東西：**工具定義每一輪都會被送出去，
而 OpenAI 怎麼把它序列化成 token 沒有公開。**

所以 `TOOLS_TOKEN_ESTIMATE` 是估的，不是量的，而且會在報表裡單獨列一欄，
不混進「輸入 token」裡假裝自己知道。**算不準的東西要標出來，不是藏進總額。**
"""
import json
from pathlib import Path

import tiktoken

import chroma_store
import judges
import prompts
import rag_core

BASE_DIR = Path(__file__).parent
CACHE_PATH = BASE_DIR / "agent_cache_day20.json"

MODEL = "gpt-4o-mini"
SEARCH_K = 3           # 與對照組同一個 k，刻意的
MAX_STEPS = 6
NL = chr(10)

# embedding 用 cl100k_base（text-embedding-3-small），與 chat 的 o200k_base 不同。
# 兩個編碼器混用會讓成本算錯一成以上，所以分開。
EMB_ENC = tiktoken.get_encoding("cl100k_base")

# 工具定義每一輪都會重送。這個數字是把下面 TOOLS 的 JSON 直接數 token
# 再加上 OpenAI 包裝的估計開銷，**是估的不是量的**，報表裡單獨列。
TOOLS_TOKEN_ESTIMATE = None     # 在 module 尾端依 TOOLS 實際算出來

SYSTEM = (
    "你是公司人資的規章問答助理。"
    "請只依據你查到的「規章條文」回答，不要引用條文以外的法律知識或常識。"
    "答案用繁體中文，控制在三句話以內，直接講結論與關鍵數字。"
    "你可以使用 search 查詢規章，需要時可以用不同的關鍵字查詢多次；"
    "查到足夠的條文之後，用 answer 給出最終答案。"
)

TOOLS = [
    {"type": "function", "function": {
        "name": "search",
        "description": f"用關鍵字查詢公司規章，回傳最相關的 {SEARCH_K} 條條文原文。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "查詢用的關鍵字或問句"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "answer",
        "description": "給出最終答案，結束這一題。",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string",
                     "description": "要回覆給使用者的答案，繁體中文，三句話以內"}},
            "required": ["text"]}}},
]

# ⚠️ 這句話的最後一個子句是一個**我自己埋的變因**，寫在這裡不是註解，是自首：
# 「如果查到的條文不足以回答，就在答案裡說明這一點」等於一句拒答提示，
# 而它只有 agent 拿得到、而且只在撞上限那幾題才出現。
# 陷阱題正好有 3 題撞上限，所以 agent 的陷阱題成績被這句話污染了。
# `CAP_NUDGE_NEUTRAL` 是拿掉那個子句的版本，`compare_cap_nudge.py` 用它重跑，
# 把污染量出來——**發現自己埋了變因，要量它，不是在文末道歉。**
CAP_NUDGE = ("查詢次數已達上限，不能再查了。"
             "請立刻用 answer 給出你目前能給的最佳答案；"
             "如果查到的條文不足以回答，就在答案裡說明這一點。")

CAP_NUDGE_NEUTRAL = ("查詢次數已達上限，不能再查了。"
                     "請立刻用 answer 給出你目前能給的最佳答案。")

REPEAT_NOTE = ("（這組關鍵字剛剛查過了，結果與上次相同，沒有再查一次。"
               "請換個說法，或直接用 answer 作答。）")


# ------------------------------------------------------------------ 帶工具的 chat

def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _key(messages: list[dict]) -> str:
    import hashlib
    payload = json.dumps([MODEL, messages, TOOLS], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def chat_tools(messages: list[dict], use_cache: bool = True) -> dict:
    """呼叫一輪，回傳 assistant 訊息（dict 形式，可直接 append 回 messages）。

    快取存的是序列化後的 assistant 訊息，讀者重跑會拿到同一條決策路徑——
    agent 的每一步都被記下來，不是只記最後的答案。
    """
    key = _key(messages)
    if use_cache:
        cache = _load(CACHE_PATH)
        if key in cache:
            return cache[key]

    response = rag_core.get_client().chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS, temperature=0)
    msg = response.choices[0].message
    out = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        out["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name,
                          "arguments": tc.function.arguments}}
            for tc in msg.tool_calls]
    if use_cache:
        cache[key] = out
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False),
                              encoding="utf-8")
    return out


# ------------------------------------------------------------------ token 帳

def count_messages(messages: list[dict]) -> int:
    """離線重數一輪的輸入 token。

    tool_calls 與 tool 回傳的內容都要算——它們是實實在在送出去的字。
    `judges.count_messages` 只認 content 欄位，所以這裡自己算。
    """
    total = 0
    for m in messages:
        total += 3 + len(judges.CHAT_ENC.encode(m.get("role", "")))
        if m.get("content"):
            total += len(judges.CHAT_ENC.encode(m["content"]))
        for tc in m.get("tool_calls", []) or []:
            total += len(judges.CHAT_ENC.encode(
                tc["function"]["name"] + tc["function"]["arguments"]))
    return total + 3


def count_assistant_out(msg: dict) -> int:
    """一輪的輸出 token：文字加上工具呼叫的參數"""
    total = len(judges.CHAT_ENC.encode(msg.get("content") or ""))
    for tc in msg.get("tool_calls", []) or []:
        total += len(judges.CHAT_ENC.encode(
            tc["function"]["name"] + tc["function"]["arguments"]))
    return total


TOOLS_TOKEN_ESTIMATE = len(judges.CHAT_ENC.encode(
    json.dumps(TOOLS, ensure_ascii=False))) + 12


# ------------------------------------------------------------------ 檢索工具

def format_hits(hits: list[dict]) -> str:
    return NL.join(f"[{i}] 第 {h['meta']['article_no']} 條"
                   f"（{h['meta']['chapter']}）{NL}{h['text']}"
                   for i, h in enumerate(hits, 1))


def run_agent(collection, question: str, use_cache: bool = True,
              cap_nudge: str = None) -> dict:
    """跑一題。回傳答案、完整決策軌跡與這一題的帳。

    回傳的 trace 是今天報表的主角：**agent 的答案不重要，它怎麼走到那個
    答案才重要。** 只印最後一句話的 agent 報表，跟只印答對率的評估一樣沒用。
    """
    cap_nudge = CAP_NUDGE if cap_nudge is None else cap_nudge
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"問題：{question}"}]
    trace, queries, seen, article_nos = [], [], set(), []
    tin = tout = emb_tokens = 0
    answer_text, hit_cap, repeats, nudged = None, False, 0, False

    for step in range(MAX_STEPS + 1):
        if step == MAX_STEPS and answer_text is None:
            # 撞上限：再給一輪，但明說不能再查，強制交答案
            messages.append({"role": "user", "content": cap_nudge})
            hit_cap, nudged = True, True

        tin += count_messages(messages) + TOOLS_TOKEN_ESTIMATE
        msg = chat_tools(messages, use_cache=use_cache)
        tout += count_assistant_out(msg)
        messages.append(msg)

        calls = msg.get("tool_calls") or []
        if not calls:
            # 沒叫工具，直接吐文字——當成答案收下
            answer_text = (msg.get("content") or "").strip()
            trace.append({"step": step + 1, "action": "plain_text"})
            break

        done = False
        for tc in calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            if name == "answer":
                answer_text = str(args.get("text", "")).strip()
                trace.append({"step": step + 1, "action": "answer"})
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": "ok"})
                done = True
                continue

            query = str(args.get("query", "")).strip()
            if query in seen:
                repeats += 1
                trace.append({"step": step + 1, "action": "search",
                              "query": query, "repeat": True, "nos": []})
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": REPEAT_NOTE})
                continue

            seen.add(query)
            queries.append(query)
            emb_tokens += len(EMB_ENC.encode(query))
            vectors, _, _ = rag_core.get_embeddings([query])
            hits = chroma_store.retrieve(collection, vectors[0], SEARCH_K)
            nos = [h["meta"]["article_no"] for h in hits]
            article_nos += nos
            trace.append({"step": step + 1, "action": "search",
                          "query": query, "repeat": False, "nos": nos})
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": format_hits(hits)})

        if done:
            break

    if answer_text is None:
        answer_text = ""          # 真的什麼都沒交出來，照實記，不補話

    return {"answer": answer_text, "trace": trace, "queries": queries,
            "article_nos": article_nos, "searches": len(queries),
            "steps": len([t for t in trace]), "repeats": repeats,
            "hit_cap": hit_cap, "nudged": nudged,
            "input_tokens": tin, "output_tokens": tout,
            "embedding_tokens": emb_tokens,
            "llm_calls": len([t for t in trace])}


# --------------------------------------------------------- 對照組：寫死的管線

def run_baseline(collection, question: str, k: int = SEARCH_K) -> dict:
    """Day 14 以來的管線，一字不動：問題原句查一次、k=3、塞進 P1 prompt。

    走 `rag_core.chat` 的快取，所以 Day 14／17 跑過的題目今天是 0 元；
    這也保證對照組的答案與前幾天位元級相同。
    """
    emb = len(EMB_ENC.encode(question))
    vectors, _, _ = rag_core.get_embeddings([question])
    hits = chroma_store.retrieve(collection, vectors[0], k)
    messages = prompts.build_messages(question, hits)
    text, _, _ = rag_core.chat(messages)
    return {"answer": text,
            "article_nos": [h["meta"]["article_no"] for h in hits],
            "queries": [question], "searches": 1, "steps": 1, "repeats": 0,
            "hit_cap": False, "nudged": False,
            "input_tokens": judges.count_messages(messages),
            "output_tokens": judges.count_text(text),
            "embedding_tokens": emb, "llm_calls": 1, "trace": []}
