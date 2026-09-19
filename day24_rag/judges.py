# -*- coding: utf-8 -*-
"""Day 18 的兩把新尺：逐項 rubric 與 LLM 整體裁判。

Day 14 的 `grader.grade()`（關鍵事實比對）在單條文題上撐了四天，Day 17 的
多條文題上 8 題判錯 5 題。它的兩個死穴都不是實作 bug，是設計本身：

  1. **看不見否定**：facts 是子字串比對，「簽到」與「不需要簽到」都命中。
  2. **把條文事實當成必答事實**：問「會不會被記過」，正確答案是
     「不會，先書面提醒」，它沒有義務背出懲處清單。

這支檔案提供兩個替代品。兩者都回傳同一組三值判定，才比得下去：

    正確     規章怎麼規定，答案就怎麼說（用詞可以完全不同）
    不完整   沒有講錯，但漏掉問題直接問到的要件
    錯誤     與規章相牴觸，或答反了

為什麼三值而不是二值：Day 17 的 305~307 被判錯的真正原因是「漏了細節」，
跟 308 的「答反了」是完全不同的病。合成一個「答對率」就看不見差別——
而修法也完全不同：前者要加檢索或加 prompt，後者要重新設計 grounding。

## 尺 B：逐項 rubric

每題預先拆成數個 checkpoint（`rubric_day18.json`），每個 checkpoint
單獨問模型一次，只回三個字之一：符合／未提及／相反。

關鍵設計是 checkpoint 分兩種：

    required  問題直接問到的，沒說就是不完整
    bonus     條文裡有、但問題沒問的，沒說不扣分，**說反了照樣算錯**

這個分類就是 Day 17 那一課的程式碼形式。舊尺的 `facts` 只有一種等級，
所以「條文裡有的事實」自動變成「答案必須包含的事實」——四個誤殺全出於此。

把問題切小還有第二個好處：一個 checkpoint 只問一件事，模型不必一次做完
所有取捨，判定會比整體裁判穩。這是個假設，3-1 的一致性實驗就是在驗它。

## 尺 C：LLM 整體裁判

本系列前四天一直拒絕的那個選項（`grader.py` 開頭寫著拒絕的理由）。
今天把它請進來，但**不是因為改變立場，是因為要量它**——拒絕一個方案
和證明它不行是兩件事，Day 14 只做了前者。

裁判拿到題目、必要條文原文、以及答案，回一個 JSON。刻意要求先寫理由
再下判定（理由在前，verdict 在後），因為反過來寫模型會先表態再硬湊理由。

## 兩把尺共同的偏誤（先自首，第三節會逐項量）

- 裁判與考生是同一個 `gpt-4o-mini`：自我偏好量不到，要接第二個模型
- 判定結果本身有隨機性：temperature=0 不保證位元級一致
- 回傳格式靠 prompt 約束，不是 function calling；解析失敗一律計為
  `parse_error` 並單獨列出，**不准悄悄退回「正確」**
"""
import json
import re

import tiktoken

import rag_core

# Day 16 起的規矩：token 一律離線重數，不依賴 API 回報的 usage。
# 快取全命中時 usage 是 0，只看那個數字會以為評估不用錢——
# 而今天整篇文章的主張之一就是「評估是帳單上的一個項目」，
# 帳自己算不清楚就沒資格這樣主張。
CHAT_ENC = tiktoken.get_encoding("o200k_base")


def count_messages(messages: list[dict]) -> int:
    """OpenAI chat 的輸入 token：每則訊息 +3，整體再 +3（同 Day 17 的算法）"""
    total = 0
    for m in messages:
        total += (3 + len(CHAT_ENC.encode(m["role"]))
                  + len(CHAT_ENC.encode(m["content"])))
    return total + 3


def count_text(text: str) -> int:
    return len(CHAT_ENC.encode(text))

CORRECT = "正確"
INCOMPLETE = "不完整"
WRONG = "錯誤"
PARSE_ERROR = "parse_error"
VERDICTS = (CORRECT, INCOMPLETE, WRONG)

MATCH = "符合"
ABSENT = "未提及"
CONTRADICT = "相反"
CHECK_LABELS = (MATCH, ABSENT, CONTRADICT)


# =============================================================== 尺 B：rubric

CHECK_SYSTEM = (
    "你是規章問答的評分助理。使用者會給你一個「判斷點」與一段「答案」。"
    "請判斷這段答案對於該判斷點的態度，只能回覆下列三個詞之一，不要加任何其他文字：\n"
    "符合 —— 答案表達了這個判斷點的意思（用詞不同、更簡短都算符合）\n"
    "未提及 —— 答案沒有談到這件事，但也沒有說出相反的意思\n"
    "相反 —— 答案說出了與這個判斷點牴觸的內容（特別注意否定詞："
    "判斷點說「仍應簽到」而答案說「不需要簽到」，就是相反）\n"
    "只看答案有沒有表達該判斷點，不要評價答案其他部分的優劣或完整性。"
)


def _check_messages(point: str, answer: str) -> list[dict]:
    return [{"role": "system", "content": CHECK_SYSTEM},
            {"role": "user", "content": f"判斷點：{point}\n\n答案：{answer}"}]


def _pick_label(text: str, labels: tuple, ambiguous: tuple = ()) -> str:
    """從模型回覆裡挑出唯一的標籤，挑不出來就認輸。

    這段防禦不是過度工程，是子字串比對自己的教訓：「不符合」裡面有「符合」、
    「不正確」裡面有「正確」。**今天整篇文章都在講一把尺不該被否定詞騙過，
    如果解析器自己踩同一個坑，那就太難看了。**

    規則：
      1. 回覆剛好就是某個標籤 → 用它（模型照指示辦事的正常情況）
      2. 出現任何已知的歧義字串（不符合、不正確⋯）→ parse_error，不猜
      3. 恰好命中一個標籤 → 用它；命中零個或兩個以上 → parse_error
    """
    t = text.strip()
    if t in labels:
        return t
    if any(bad in t for bad in ambiguous):
        return PARSE_ERROR
    hits = [label for label in labels if label in t]
    return hits[0] if len(hits) == 1 else PARSE_ERROR


def check_point(point: str, answer: str, use_cache: bool = True) -> tuple[str, int, int]:
    """問一個 checkpoint。回傳 (符合／未提及／相反／parse_error, 輸入 token, 輸出 token)

    token 是離線重數的，不是 API 回報的 usage：快取命中時 usage 為 0，
    但這次判定實際上就是要花這麼多 token，帳要照實算。
    """
    messages = _check_messages(point, answer)
    text, _, _ = rag_core.chat(messages, use_cache=use_cache)
    return (_pick_label(text, CHECK_LABELS, ambiguous=("不符合",)),
            count_messages(messages), count_text(text))


def aggregate(results: list[tuple[str, str]]) -> str:
    """把 checkpoint 結果聚合成三值判定。

    results 是 [(kind, label), ...]，kind 為 "required" 或 "bonus"。

        任一 checkpoint 判「相反」        → 錯誤
            （bonus 也算：說錯了就是錯，不管必不必要）
        否則任一 required 判「未提及」     → 不完整
        否則                              → 正確
            （bonus 未提及不扣分——這一行就是 Day 17 四個誤殺的解藥）

    任一 checkpoint 解析失敗就整題回 parse_error，不猜。
    """
    labels = [label for _, label in results]
    if PARSE_ERROR in labels:
        return PARSE_ERROR
    if CONTRADICT in labels:
        return WRONG
    if any(label == ABSENT for kind, label in results if kind == "required"):
        return INCOMPLETE
    return CORRECT


def grade_rubric(answer: str, checkpoints: list[dict],
                 use_cache: bool = True) -> dict:
    """用 rubric 判一個答案。回傳判定、逐項明細與 token 帳。"""
    detail, tin, tout = [], 0, 0
    for cp in checkpoints:
        label, a, b = check_point(cp["point"], answer, use_cache=use_cache)
        detail.append({"id": cp["id"], "kind": cp["kind"], "label": label})
        tin, tout = tin + a, tout + b
    verdict = aggregate([(d["kind"], d["label"]) for d in detail])
    return {"verdict": verdict, "detail": detail,
            "input_tokens": tin, "output_tokens": tout,
            "calls": len(checkpoints)}


# ========================================================= 尺 C：LLM 整體裁判

JUDGE_SYSTEM = (
    "你是公司規章問答的評分委員。使用者會給你一個問題、該問題所依據的規章"
    "條文原文、以及一段待評答案。請依規章原文判斷這段答案的品質。\n\n"
    "判定只有三種：\n"
    "正確 —— 答案與規章相符，且回答了問題直接問到的事情。"
    "用詞與條文不同、比條文簡短，都不影響判定。\n"
    "不完整 —— 答案沒有說錯任何事，但漏掉了問題直接問到的要件。\n"
    "錯誤 —— 答案有任何一處與規章牴觸，或把結論答反了。\n\n"
    "重要原則：\n"
    "1. 「沒有說」不等於「說錯」。問題沒問到的細節，答案沒提到不扣分。\n"
    "2. 一個簡短而正確的答案就是正確，不要因為它短而降級。\n"
    "3. 特別注意否定詞。條文說「仍應完成簽到」而答案說「不需要簽到」，"
    "即使兩者用字高度重疊，判定也是錯誤。\n\n"
    "請嚴格回覆下列 JSON，不要加上程式碼區塊標記或其他文字：\n"
    '{"reason": "一到兩句話說明依據", "verdict": "正確或不完整或錯誤"}\n'
    "reason 必須寫在 verdict 前面：先說明依據，再下判定。"
)

JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def _judge_messages(question: str, articles: str, answer: str) -> list[dict]:
    user = (f"問題：{question}\n\n"
            f"規章條文原文：\n{articles}\n\n"
            f"待評答案：\n{answer}")
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user}]


def grade_judge(question: str, articles: str, answer: str,
                use_cache: bool = True) -> dict:
    """用整體裁判判一個答案。回傳判定、理由原文與 token 帳。

    理由原文一定要留著印進報表：裁判判錯的時候，理由是唯一能看出
    「它為什麼錯」的東西。只印判定的評估報告，跟只印答對率一樣沒用。
    """
    messages = _judge_messages(question, articles, answer)
    text, _, _ = rag_core.chat(messages, use_cache=use_cache)
    tin, tout = count_messages(messages), count_text(text)
    verdict, reason = PARSE_ERROR, text.strip()
    match = JSON_BLOCK.search(text)
    if match:
        try:
            payload = json.loads(match.group())
            reason = str(payload.get("reason", "")).strip()
            raw = str(payload.get("verdict", "")).strip()
            verdict = _pick_label(raw, VERDICTS, ambiguous=("不正確",))
        except json.JSONDecodeError:
            pass
    return {"verdict": verdict, "reason": reason,
            "input_tokens": tin, "output_tokens": tout, "calls": 1}


# ============================================================ 尺 A 的三值包裝

def grade_facts(correct: bool) -> str:
    """把舊尺的布林判定塞進三值，才跟另外兩把比得下去。

    舊尺只有對／錯兩格，沒有「不完整」這個概念——**而 Day 17 的四個誤殺
    全部是「不完整」被當成「錯誤」。** 這個函式故意寫得這麼短，
    是為了讓這件事在程式碼裡看得見：轉換過程中沒有資訊可以還原。
    """
    return CORRECT if correct else WRONG
