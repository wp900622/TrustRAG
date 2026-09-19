# -*- coding: utf-8 -*-
"""Day 14 的 prompt：把檢索到的條文組成 context，要求模型只依 context 回答。

刻意保持「最小可用」，只放兩條約束：
1. 只用提供的條文回答 —— 沒有這句，模型會用它預訓練時看過的勞基法常識作答，
   那就不是 RAG 了，語料換掉答案也不會變（run_experiment.py 的 no-context
   對照組就是在量這件事）
2. 答案要短 —— 輸出 token 比輸入貴 4 倍，而且長答案會把關鍵事實埋在廢話裡

「標出條號」與「找不到就明確拒答」留給 Day 15：那是可信度的題目，
今天先把管線接起來、把落差量出來。
"""

SYSTEM = (
    "你是公司人資的規章問答助理。"
    "請只依據使用者提供的「規章條文」回答，不要引用條文以外的法律知識或常識。"
    "答案用繁體中文，控制在三句話以內，直接講結論與關鍵數字。"
)

# 對照組：不給任何條文，直接問。用來量「檢索到底貢獻了多少」——
# 這一組答對的題目，代表模型本來就知道，不是你的語料的功勞。
SYSTEM_NO_CONTEXT = (
    "你是公司人資的規章問答助理。"
    "請回答使用者關於公司規章的問題，用繁體中文，控制在三句話以內。"
)


# ------------------------------------------------- Day 15：prompt 三版對照
#
# 只改 system prompt，檢索側（k=3）與題組全部凍結，落差才能歸因到這幾句話。
#
#   P1 基本版      —— Day 14 的 SYSTEM，基準線
#   P2 要求引用    —— 加一句「標出條號」。想量：引用會不會讓它更謹慎？
#                     以及引用本身正確嗎（會不會引一個沒出現在 prompt 裡的條號）
#   P3 引用＋拒答  —— 再加一句「條文未涵蓋就說規章未規定」。
#                     想量：幻覺降多少？正常題要賠掉幾題？
#
# P3 最後一句特別處理「另有規定」型條文（第 43 條國外出差就是），
# 因為那類題目的正確行為不是拒答而是轉指——不寫清楚會把 B 類陷阱題
# 逼成假拒答，量到的就不是模型的毛病，是題目沒說清楚。

CITE_RULE = (
    "每個結論後面要標出依據的條號，格式如（第 24 條）；"
    "條號只能用「規章條文」裡出現過的，不得自行推斷。"
)

REFUSE_RULE = (
    "若提供的條文未涵蓋問題，直接回答「規章未規定」，不要推測，"
    "也不要引用條文以外的法律或其他公司的常見做法。"
    "若條文只寫明另有辦法規定，就回答依該辦法規定，不要自行填入數字。"
)

SYSTEM_P2 = SYSTEM + CITE_RULE
SYSTEM_P3 = SYSTEM + CITE_RULE + REFUSE_RULE

SYSTEMS = {"P1": SYSTEM, "P2": SYSTEM_P2, "P3": SYSTEM_P3}
PROMPT_LABELS = {"P1": "P1 基本版", "P2": "P2 要求引用",
                 "P3": "P3 引用＋允許拒答"}


def build_context(hits: list[dict]) -> str:
    """把 top-k 條文編號排好。

    編號（[1]、[2]…）不是為了好看：Day 15 要模型回報「你用了第幾段」，
    以及要分析「答案藏在第幾名時模型還讀不讀」，都需要這個位置標記。
    """
    blocks = []
    for i, hit in enumerate(hits, 1):
        meta = hit["meta"]
        blocks.append(f"[{i}] 第 {meta['article_no']} 條"
                      f"（{meta['chapter']}）\n{hit['text']}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[dict],
                   version: str = "P1") -> list[dict]:
    """組出送給 LLM 的 messages（有 context 版）。

    version 預設 "P1"＝Day 14 原封不動的 SYSTEM，所以 Day 14 的
    chat 快取 key 不變、數字仍可重現。Day 15 用 P2／P3 做對照。
    """
    user = (f"規章條文：\n{build_context(hits)}\n\n"
            f"問題：{question}")
    return [{"role": "system", "content": SYSTEMS[version]},
            {"role": "user", "content": user}]


def build_messages_no_context(question: str) -> list[dict]:
    """組出送給 LLM 的 messages（無 context 對照組）"""
    return [{"role": "system", "content": SYSTEM_NO_CONTEXT},
            {"role": "user", "content": f"問題：{question}"}]
