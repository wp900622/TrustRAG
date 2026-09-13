# -*- coding: utf-8 -*-
"""Day 17 新增：LLM listwise reranker。

為什麼需要它，一句話：**向量檢索一次只回答一個問題「哪一條最像這個提問」**，
而多條文綜合題需要的是「哪幾條湊起來才答得完」。第 33 條（薪資結構）跟
「加班 3 小時怎麼算錢」這句話一點都不像，所以它在 k=3 撈不到；但少了它，
答案就沒有計算基準。

作法（listwise，不是 pointwise）：
    1. 向量檢索先撈寬一點（候選 15 條）
    2. 把 15 條編號後整批交給 gpt-4o-mini，要它挑出並排序最有用的 N 條
    3. 只把那 N 條送進生成

listwise 的理由：pointwise（逐條打分）看不到條文之間的互補性，
而互補性正是今天的主題——第 19 條與第 33 條單獨看都只有一半。

刻意不用 cross-encoder：本系列的原則是不引進需要下載 1GB 模型的依賴，
而且 rag_core.chat() 已經有快取與雙軌成本統計，重排的帳單可以跟生成分開列。
代價是每題多一次 API 呼叫——那筆帳 experiment_result_day17.md 會單獨算給你看。

已知偏誤（文章裡會講）：
- 重排用的是同一個 gpt-4o-mini。裁判與考生是同一個模型，
  它挑不出來的條文，生成時本來也用不好——這個實驗量不到「更強的重排模型
  會不會更好」。
- 回傳格式靠 prompt 約束，不是 function calling。解析失敗時
  fallback 成原順序（並在結果裡標記），不讓實驗因為一次格式抖動就中斷。
"""
import re

import rag_core

RERANK_SYSTEM = (
    "你是檢索結果的重排助理。使用者會給你一個問題與若干段編號的規章條文。"
    "請挑出「要完整回答這個問題所必須用到」的條文，依重要性排序。"
    "注意：有些問題需要多條條文互相補足（例如一條給比例、另一條給計算基準），"
    "請把互補的條文一起選進來，不要只選字面最相似的那幾條。"
    "只輸出編號，用半形逗號分隔，例如：3,7,1。不要輸出任何其他文字。"
)

INDEX_PATTERN = re.compile(r"\d+")


def build_candidate_block(hits: list[dict]) -> str:
    """把候選條文編號排好，編號 1 起算，與 prompts.build_context() 同一個慣例"""
    blocks = []
    for i, hit in enumerate(hits, 1):
        meta = hit["meta"]
        blocks.append(f"[{i}] 第 {meta['article_no']} 條"
                      f"（{meta['chapter']}）\n{hit['text']}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[dict], top_n: int) -> list[dict]:
    user = (f"問題：{question}\n\n"
            f"候選條文（共 {len(hits)} 段）：\n{build_candidate_block(hits)}\n\n"
            f"請輸出最多 {top_n} 個編號，依重要性排序。")
    return [{"role": "system", "content": RERANK_SYSTEM},
            {"role": "user", "content": user}]


def parse_order(text: str, candidate_count: int, top_n: int) -> list[int]:
    """把模型回傳的字串解析成 0 起算的索引清單。

    刻意寬鬆：抓所有數字、去重、丟掉超出範圍的。模型偶爾會回
    「3, 7, 1」以外的格式（加了句號、加了「編號」兩個字），
    為了一個標點就讓實驗中斷不划算。
    """
    order, seen = [], set()
    for token in INDEX_PATTERN.findall(text):
        idx = int(token) - 1
        if 0 <= idx < candidate_count and idx not in seen:
            seen.add(idx)
            order.append(idx)
    return order[:top_n]


def rerank(question: str, hits: list[dict], top_n: int = 3
           ) -> tuple[list[dict], dict]:
    """重排並截斷成 top_n。

    回傳 (重排後的 hits, 統計資訊)。統計資訊含 token 數與是否 fallback，
    報表要把重排的帳單跟生成的帳單分開列。
    """
    if not hits:
        return [], {"input_tokens": 0, "output_tokens": 0,
                    "fallback": False, "raw": "", "picked": []}

    text, in_tok, out_tok = rag_core.chat(build_messages(question, hits, top_n))
    order = parse_order(text, len(hits), top_n)
    fallback = not order
    if fallback:
        # 解析不出任何合法編號：退回原本的相似度序，並如實記錄，
        # 免得 fallback 混進「重排結果」裡被當成 reranker 的功勞或過錯
        order = list(range(min(top_n, len(hits))))

    reranked = [hits[i] for i in order]
    stats = {"input_tokens": in_tok, "output_tokens": out_tok,
             "fallback": fallback, "raw": text,
             "picked": [hits[i]["meta"]["article_no"] for i in order]}
    return reranked, stats


def rerank_cost_usd(stats_list: list[dict]) -> float:
    """重排的總花費：跟生成用同一組單價，但要分開列才看得出它值不值得"""
    total_in = sum(s["input_tokens"] for s in stats_list)
    total_out = sum(s["output_tokens"] for s in stats_list)
    return rag_core.chat_cost_usd(total_in, total_out)
