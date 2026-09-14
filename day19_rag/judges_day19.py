# -*- coding: utf-8 -*-
"""Day 19：修好的 rubric，以及可以換模型的判定。

Day 18 的尺 B 誤殺 9 筆，全部死在同一條聚合規則上——某個 checkpoint
被判成「相反」，整題就變「錯誤」。9 筆誤殺只有兩個病灶：

  1. **否定極性**：`305-1`「到職滿三個月的員工還沒有特休假」是一句否定敘述，
     答案同樣否定（「沒有特休假」），判定卻回「相反」。
     嫌疑犯是 Day 18 自己寫的那句防守——「特別注意否定詞：判斷點說
     『仍應簽到』而答案說『不需要簽到』，就是相反」。為了防 308 的放水
     寫得太用力，變成看到否定詞就開槍。

  2. **例外條款**：`307-2`（bonus）「原則上應於七個工作日內發給」，
     答案講的是條文自己寫的例外（未交還財物得暫緩），判定回「相反」。
     bonus 的設計意圖是「沒說不扣分，說反了才扣」，但「講的是例外」
     既不是沒說也不是說反，Day 18 的 prompt 沒給它位置。

本檔提供兩個修法，**分開可組合**，因為一次改兩個地方就不知道哪個有效：

    F1  否定極性：「相反」判的是立場，不是字面上有沒有否定詞
    F2  例外條款：判斷點是原則、答案講例外或條件未達 → 未提及

⚠️ 兩段修法的範例一律用抽象代號（甲、乙、丙）。拿被測的那八題當 prompt
範例，修完的數字就只是背答案——那是把考題寫進參考書，不是修尺。

另外，判定要能換模型。`rag_core.chat()` 的模型與快取 key 綁死 CHAT_MODEL，
所以這裡包一層：`gpt-4o-mini` 直接走 rag_core（沿用 Day 18 的快取，
B0 重跑因此是 0 元，數字與昨天位元級相同），其他模型走本檔自己的快取。
"""
import hashlib
import json
from pathlib import Path

import judges
import rag_core

BASE_DIR = Path(__file__).parent
CACHE_PATH = BASE_DIR / "chat_cache_day19.json"

MINI = "gpt-4o-mini"
GPT4O = "gpt-4o"

# 每百萬 token 的美元單價（輸入, 輸出）。換模型只要加一列。
PRICES = {MINI: (0.15, 0.60), GPT4O: (2.50, 10.00)}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES[model]
    return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout


def cost_twd(model: str, input_tokens: int, output_tokens: int) -> float:
    return rag_core.twd(cost_usd(model, input_tokens, output_tokens))


# ---------------------------------------------------------------- 換模型的 chat

def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def chat(messages: list[dict], model: str = MINI,
         use_cache: bool = True) -> str:
    """跟 rag_core.chat 同語意，但模型可換。只回文字——token 一律離線重數。

    gpt-4o-mini 刻意轉交給 rag_core：它的快取 key 是
    sha256(json([CHAT_MODEL, messages]))，與這裡的算法同形，
    所以 Day 18 已經付過錢的那些判定今天全部命中，B0 對照組 0 元。
    """
    if model == rag_core.CHAT_MODEL:
        text, _, _ = rag_core.chat(messages, use_cache=use_cache)
        return text

    key = hashlib.sha256(json.dumps([model, messages], ensure_ascii=False,
                                    sort_keys=True).encode("utf-8")).hexdigest()[:16]
    if use_cache:
        cache = _load(CACHE_PATH)
        if key in cache:
            return cache[key]
    response = rag_core.get_client().chat.completions.create(
        model=model, messages=messages, temperature=0)
    text = response.choices[0].message.content.strip()
    if use_cache:
        cache[key] = text
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False),
                              encoding="utf-8")
    return text


# ------------------------------------------------------------ 修好的 rubric prompt

F1_RULE = (
    "\n【關於否定詞】「相反」判的是立場，不是字面上有沒有否定詞。"
    "判斷點本身就是否定敘述時（含「尚未」「還沒有」「不得」「無需」等），"
    "答案同樣否定就是「符合」；一正一反才叫「相反」。\n"
    "　判斷點「甲情形不適用本規定」＋答案「這種情況不適用」→ 符合\n"
    "　判斷點「乙情形應辦理申請」　＋答案「不必辦理申請」→ 相反"
)

F2_RULE = (
    "\n【關於例外與適用條件】判斷點寫的是一般原則時，答案陳述該原則的例外、"
    "或說明本案尚未達到該原則的適用條件，都算「未提及」——"
    "答案沒有談到這個原則，不等於否定它。"
    "只有當答案主張該原則不存在、或把原則的內容說反，才算「相反」。\n"
    "　判斷點「原則上應於期限內發給」＋答案「有丙情形者得暫緩發給」"
    "→ 未提及（答案講的是例外）\n"
    "　判斷點「年資滿一定期間者給付 N 日」＋答案「本案年資未達門檻，尚無此項給付」"
    "→ 未提及（門檻未到，判斷點在本案用不上）"
)

# B0 是 Day 18 原封不動的 prompt，當對照組用。
CHECK_SYSTEMS = {
    "B0": judges.CHECK_SYSTEM,
    "B1": judges.CHECK_SYSTEM + F1_RULE,
    "B2": judges.CHECK_SYSTEM + F2_RULE,
    "B3": judges.CHECK_SYSTEM + F1_RULE + F2_RULE,
}


def check_point(point: str, answer: str, variant: str = "B3",
                model: str = MINI, use_cache: bool = True) -> tuple[str, int, int]:
    """問一個 checkpoint。回傳 (符合／未提及／相反／parse_error, 輸入, 輸出 token)"""
    messages = [{"role": "system", "content": CHECK_SYSTEMS[variant]},
                {"role": "user", "content": f"判斷點：{point}\n\n答案：{answer}"}]
    text = chat(messages, model=model, use_cache=use_cache)
    return (judges._pick_label(text, judges.CHECK_LABELS, ambiguous=("不符合",)),
            judges.count_messages(messages), judges.count_text(text))


def grade_rubric(answer: str, checkpoints: list[dict], variant: str = "B3",
                 model: str = MINI, use_cache: bool = True) -> dict:
    """用 rubric 判一個答案。聚合規則沿用 Day 18 的 `judges.aggregate()`——
    **今天只改 prompt，不改聚合規則。** 把 bonus 的「相反」從致命改成不致命
    也修得掉 307，但那是把規則放寬到看不見真正的錯，
    而 308 那筆放水正是靠這條規則抓住的。"""
    detail, tin, tout = [], 0, 0
    for cp in checkpoints:
        label, a, b = check_point(cp["point"], answer, variant=variant,
                                  model=model, use_cache=use_cache)
        detail.append({"id": cp["id"], "kind": cp["kind"], "label": label})
        tin, tout = tin + a, tout + b
    verdict = judges.aggregate([(d["kind"], d["label"]) for d in detail])
    return {"verdict": verdict, "detail": detail, "input_tokens": tin,
            "output_tokens": tout, "calls": len(checkpoints), "model": model}


# ------------------------------------------------------- 可換模型的整體裁判

def grade_judge(question: str, articles: str, answer: str,
                model: str = MINI, use_cache: bool = True) -> dict:
    """尺 C，prompt 與 Day 18 一字不動（`judges.JUDGE_SYSTEM`），只換模型。

    prompt 不准動是刻意的：這一節要回答的是「換一個模型當裁判會怎樣」，
    順手把 prompt 也改了，就分不出功勞是誰的。
    """
    messages = judges._judge_messages(question, articles, answer)
    text = chat(messages, model=model, use_cache=use_cache)
    tin, tout = judges.count_messages(messages), judges.count_text(text)
    verdict, reason = judges.PARSE_ERROR, text.strip()
    match = judges.JSON_BLOCK.search(text)
    if match:
        try:
            payload = json.loads(match.group())
            reason = str(payload.get("reason", "")).strip()
            verdict = judges._pick_label(str(payload.get("verdict", "")).strip(),
                                         judges.VERDICTS, ambiguous=("不正確",))
        except json.JSONDecodeError:
            pass
    return {"verdict": verdict, "reason": reason, "input_tokens": tin,
            "output_tokens": tout, "calls": 1, "model": model}
