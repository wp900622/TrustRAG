# -*- coding: utf-8 -*-
"""Day 21：給迴圈加第三個工具 —— 交卷前自己驗一次。

Day 20 的結論是反著的：agent 跟寫死的管線 24:24 打平，貴 4.4 倍。
最硬的那筆證據是多條文題 301——**兩邊撈到一模一樣的條文**，
對照組算對了分段、agent 把它塌成一句話。相同的條文、相同的模型，
一個對一個錯。**問題在生成側，不在檢索側。**

所以今天不再碰檢索，只加一個動作：

    search(query)   我要查規章                    （與 Day 20 一字不差）
    check(draft)    這是我的草稿，對著條文驗一次   ← 今天唯一的新東西
    answer(text)    這是最終答案                  （與 Day 20 一字不差）

## check 不是判斷式，是另一次 LLM 呼叫

把「問題」「agent 自己查到的條文原文」「它寫的草稿」三樣送進去，
要它回報兩種問題：**沒有依據**、**與條文相反**。結果當成 tool 回傳塞回迴圈。

**模型可以不理會自驗的結果。** 我不在程式裡替它改答案——那會變成我在答題。
「標了問題之後它改了沒」是今天要量的東西之一，不是我可以先斬後奏的東西。

## 一個刻意的限制：check 只看得到 agent 自己查到的條文

不是必要條文，不是完整規章。**否則就是把答案卷交給它對。**
308 那題的第 15 條 agent 根本沒撈到，check 一樣看不到——今天量的是自驗，
不是偷看標準答案。

## 一個刻意的取捨：check 吃的是同一個 6 步預算

拉高上限就同時改了兩個變因。所以 `MAX_STEPS` 沿用 Day 20 的 6，
**這會讓已經用滿 6 步的陷阱題對 C 版不利**——寫在這裡，不留到跑完才解釋。
撞上限那一輪強制作答、免驗（不然 C2 會死在互相等待上）。

## 自首：C 版比 A 版多拿到三段文字

Day 20 我從「撞上限的提示句」這個後門漏了一句拒答提示給 agent。
同一種錯不犯第三次，所以先數清楚今天多出來的字：

    1. check 的工具說明     每一輪都隨 TOOLS 送出去
    2. CHECK_SYSTEM         只有 check 那次呼叫看得到——**最危險的一段**，
                            「沒有依據就標出來」離「查不到就說沒規定」只有一步
    3. REJECT_NOTE          只有 C2 有，answer 被擋下來時才出現

**所以 C 版如果在陷阱題上變好，我沒辦法宣稱那是「自驗」的功勞**——
也可能只是這三段文字裡的拒答暗示在起作用。這句話寫在跑之前。

## 上限提示一律用中性版

Day 20 正文那個 5/5 是被我埋的拒答提示餵出來的。今天 A／C1／C2
一律用 `agent_day20.CAP_NUDGE_NEUTRAL`，**A 的陷阱題基準是 4/5。**
"""
import json
from pathlib import Path

import agent_day20 as d20
import chroma_store
import judges
import rag_core

BASE_DIR = Path(__file__).parent
CACHE_PATH = BASE_DIR / "agent_cache_day21.json"

MODEL = d20.MODEL
SEARCH_K = d20.SEARCH_K        # 3，與 Day 20、與對照組同一個數字
MAX_STEPS = d20.MAX_STEPS      # 6，刻意不拉高
NL = chr(10)

# SYSTEM 與 Day 20 一字不差。今天的變因不准包含「我換了一句系統提示」。
SYSTEM = d20.SYSTEM

CHECK_TOOL = {
    "type": "function", "function": {
        "name": "check",
        "description": "把草稿答案送去校對：對照你已經查到的條文原文逐句比對，"
                       "回報沒有依據或與條文相反的地方。",
        "parameters": {"type": "object", "properties": {
            "draft": {"type": "string", "description": "要校對的草稿答案"}},
            "required": ["draft"]}}}

# search 與 answer 的定義直接沿用 Day 20 的物件，一個字都沒有重打。
TOOLS = [d20.TOOLS[0], CHECK_TOOL, d20.TOOLS[1]]

# ⚠️ 今天最危險的一段字。它只出現在 check 那一次呼叫裡，agent 的主迴圈
# 看不到，但它的產出會回到主迴圈。「沒有依據就標出來」跟 Day 20 那句
# 「查不到就說明」是親戚——所以文章裡要把它全文貼出來，讓讀者自己判斷
# 陷阱題上的差別是「自驗」還是「這段字」。
CHECK_SYSTEM = (
    "你是校對員。以下會給你一個問題、一段規章條文原文，以及一份草稿答案。"
    "請逐句比對草稿與條文，只回報兩種問題："
    "（1）沒有依據：草稿講的事，在條文裡找不到；"
    "（2）與條文相反：草稿講的，跟條文寫的不一樣。"
    "每個問題一行，格式為「沒有依據：…」或「與條文相反：…」，"
    "並附上相關條號。沒有問題就只回「通過」兩個字。"
    "不要改寫草稿，不要補充條文以外的知識。")

CHECK_PASS = "通過"

# 只有 C2 會用到。文字寫得越短越好——它是變因，不是功能說明。
REJECT_NOTE = "交卷前請先用 check 校對這份草稿。"

NO_ARTICLES = "（目前一條條文都還沒查到。）"


# ------------------------------------------------------------------ 帶工具的 chat

def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _key(messages: list[dict]) -> str:
    import hashlib
    payload = json.dumps([MODEL, messages, TOOLS], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def chat_tools(messages: list[dict], use_cache: bool = True) -> dict:
    """與 Day 20 同一套快取規則，只是 TOOLS 多了一個，所以 key 全部不同。

    （這代表今天沒有一次呼叫能沿用 Day 20 的快取——多一個工具就是一批新的帳。）
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


TOOLS_TOKEN_ESTIMATE = len(judges.CHAT_ENC.encode(
    json.dumps(TOOLS, ensure_ascii=False))) + 12


# ------------------------------------------------------------------ check

def run_check(question: str, articles: str, draft: str) -> dict:
    """跑一次自驗。回傳校對意見與這一次的帳。

    走 `rag_core.chat`（同一個 gpt-4o-mini、temperature=0、同一份快取），
    token 照 Day 16 起的規矩離線重數，不看 API 回報的 usage。
    """
    messages = [{"role": "system", "content": CHECK_SYSTEM},
                {"role": "user",
                 "content": (f"問題：{question}{NL}{NL}"
                             f"查到的條文：{NL}{articles or NO_ARTICLES}{NL}{NL}"
                             f"草稿答案：{NL}{draft}")}]
    text, _, _ = rag_core.chat(messages)
    text = text.strip()
    return {"verdict": text,
            "passed": text.startswith(CHECK_PASS),
            "input_tokens": judges.count_messages(messages),
            "output_tokens": judges.count_text(text)}


def _norm(s: str) -> str:
    return "".join(s.split())


# ------------------------------------------------------------------ 迴圈

def run_agent(collection, question: str, force_check: bool = False,
              use_cache: bool = True) -> dict:
    """跑一題。`force_check=False` 是 C1（自願），`True` 是 C2（強制）。

    兩者共用同一段程式，差別只有一個布林值——**這樣「強制」的效果才不會
    混進「我順手改了別的地方」。**
    """
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"問題：{question}"}]
    trace, queries, seen, article_nos = [], [], set(), []
    hits_seen: list[dict] = []          # check 看得到的條文＝agent 自己查到的
    checks: list[dict] = []
    tin = tout = emb_tokens = calls = 0
    answer_text, hit_cap, repeats, rejects, checked = None, False, 0, 0, False

    for step in range(MAX_STEPS + 1):
        if step == MAX_STEPS and answer_text is None:
            # 撞上限：中性版提示（Day 20 那個拒答子句已拿掉），強制作答、免驗
            messages.append({"role": "user",
                             "content": d20.CAP_NUDGE_NEUTRAL})
            hit_cap = True

        tin += d20.count_messages(messages) + TOOLS_TOKEN_ESTIMATE
        msg = chat_tools(messages, use_cache=use_cache)
        tout += d20.count_assistant_out(msg)
        calls += 1
        messages.append(msg)

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            answer_text = (msg.get("content") or "").strip()
            trace.append({"step": step + 1, "action": "plain_text"})
            break

        done = False
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            # ---------------------------------------------------- answer
            if name == "answer":
                text = str(args.get("text", "")).strip()
                if force_check and not checked and not hit_cap:
                    # 還沒驗過就想交卷——退件。這是 C2 唯一的額外動作
                    rejects += 1
                    trace.append({"step": step + 1, "action": "rejected"})
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": REJECT_NOTE})
                    continue
                answer_text = text
                trace.append({"step": step + 1, "action": "answer"})
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": "ok"})
                done = True
                continue

            # ---------------------------------------------------- check
            if name == "check":
                draft = str(args.get("draft", "")).strip()
                res = run_check(question, d20.format_hits(hits_seen), draft)
                calls += 1
                tin += res["input_tokens"]
                tout += res["output_tokens"]
                checks.append({"step": step + 1, "draft": draft,
                               "verdict": res["verdict"],
                               "passed": res["passed"]})
                checked = True
                trace.append({"step": step + 1, "action": "check",
                              "passed": res["passed"]})
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": res["verdict"]})
                continue

            # ---------------------------------------------------- search
            query = str(args.get("query", "")).strip()
            if query in seen:
                repeats += 1
                trace.append({"step": step + 1, "action": "search",
                              "query": query, "repeat": True, "nos": []})
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": d20.REPEAT_NOTE})
                continue

            seen.add(query)
            queries.append(query)
            emb_tokens += len(d20.EMB_ENC.encode(query))
            vectors, _, _ = rag_core.get_embeddings([query])
            hits = chroma_store.retrieve(collection, vectors[0], SEARCH_K)
            for h in hits:
                if h["meta"]["article_no"] not in [
                        x["meta"]["article_no"] for x in hits_seen]:
                    hits_seen.append(h)
            nos = [h["meta"]["article_no"] for h in hits]
            article_nos += nos
            trace.append({"step": step + 1, "action": "search",
                          "query": query, "repeat": False, "nos": nos})
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": d20.format_hits(hits)})

        if done:
            break

    if answer_text is None:
        answer_text = ""

    # 自驗標了問題之後，答案到底有沒有動？——今天最可能的失敗模式在這一欄
    last = checks[-1] if checks else None
    revised = bool(last) and _norm(last["draft"]) != _norm(answer_text)
    flagged = bool(last) and not last["passed"]

    return {"answer": answer_text, "trace": trace, "queries": queries,
            "article_nos": article_nos, "searches": len(queries),
            "steps": len(trace), "repeats": repeats, "hit_cap": hit_cap,
            "nudged": hit_cap, "checks": checks, "n_checks": len(checks),
            "rejects": rejects, "flagged": flagged, "revised": revised,
            "input_tokens": tin, "output_tokens": tout,
            "embedding_tokens": emb_tokens, "llm_calls": calls}
