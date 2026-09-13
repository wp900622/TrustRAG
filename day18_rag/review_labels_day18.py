# -*- coding: utf-8 -*-
"""Day 18 附帶實驗：用第二個模型「盲標」人工標註中的 24 筆，當獨立對照。

為什麼要有這支：
  `human_labels_day18.json` 是 Day 18 整篇的 ground truth。其中 k=3、k=15
  那 16 筆是人自己讀完寫的，其餘 24 筆（k=1/5/10）是 LLM 依同一套原則補的。
  一份由模型產生的 ground truth，不能再由模型「同意／不同意」來背書——
  那只是問一個有強烈附和傾向的模型要不要按 Enter。

所以這裡做的是**盲標**：
  - 覆核者看不到既有標註，只拿到「題目＋必要條文原文＋答案＋三條標註原則」
  - 自己獨立產一個判定，跑完才由程式對帳
  - 覆核者是 gpt-4o，**不是** gpt-4o-mini。被評答案、尺 B、尺 C 全部是
    gpt-4o-mini 產的；拿同一個模型來審自己的評分基準，是這篇文章裡
    最不該犯的錯（self-preference 正是第四節自首沒量的那一項）

不一致的那幾筆不自動改，程式只負責把它們印出來給人裁決。

用法：
    python review_labels_day18.py           # 盲標 k=1/5/10 那 24 筆
    python review_labels_day18.py --all     # 連 seed 的 16 筆一起盲標
結果寫入 review_labels_day18.json。
"""
import argparse
import hashlib
import json
from pathlib import Path

import chunkers
import judges
import pipeline
import rag_core

BASE_DIR = Path(__file__).parent
QUESTIONS_PATH = BASE_DIR / "questions_day17.json"
LABELS_PATH = BASE_DIR / "human_labels_day18.json"
OUT_PATH = BASE_DIR / "review_labels_day18.json"
CACHE_PATH = BASE_DIR / "review_cache_day18.json"
NL = chr(10)

REVIEW_MODEL = "gpt-4o"                 # 刻意不同於 CHAT_MODEL
SEED_KS = ("@k3", "@k15")               # 這些是人自己標的，預設不重標
USD_PER_MILLION_IN = 2.50
USD_PER_MILLION_OUT = 10.00

SYSTEM = """你是法遵文件問答的標註員。依據規章條文原文，判斷一個答案屬於下列三類中的哪一類：

正確    規章怎麼規定，答案就怎麼說（用詞可以完全不同，不必照抄條文）
不完整  沒有講錯任何事，但漏掉問題直接問到的要件
錯誤    與規章相牴觸，或答反了

標註原則（三條，必須嚴格遵守）：
1. 「沒說」不等於「說錯」。問「會不會被記過」，答「不會，先書面提醒」就是正確，
   沒有義務背出懲處種類清單。
2. 答反了就是錯誤，不管關鍵詞多齊全、語氣多肯定。
3. 問題沒問到的細節不列入要求。問「可以不發嗎」就不必講時限。

只輸出 JSON，格式：{"verdict": "正確|不完整|錯誤", "reason": "一句話理由"}"""

USER = """【問題】
{question}

【規章條文原文】
{articles}

【待判定的答案】
{answer}

請依三條標註原則判定。只輸出 JSON。"""


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def article_text_map() -> dict:
    """條號 → 條文原文。走 pipeline.build_index()，確保跟正式實驗拿到的是
    同一份切塊，覆核者看到的條文原文才會與尺 C 看到的一字不差。"""
    _, chunks, articles, _, _ = pipeline.build_index()
    mapping = {}
    for chunk in chunks:
        covered = chunkers.articles_covering(chunk, articles)
        if covered:
            mapping[covered[0]["no"]] = chunk["text"]
    return mapping


def review(question: str, articles: str, answer: str) -> tuple[str, str, int, int]:
    """盲標一筆。快取獨立一份，因為 rag_core 的快取 key 綁死 CHAT_MODEL。"""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(
                    question=question, articles=articles, answer=answer)}]
    key = hashlib.sha256(
        json.dumps([REVIEW_MODEL, messages], ensure_ascii=False,
                   sort_keys=True).encode("utf-8")).hexdigest()[:16]
    cache = load_json(CACHE_PATH)
    if key in cache:
        raw, tin, tout = cache[key], 0, 0
    else:
        r = rag_core.get_client().chat.completions.create(
            model=REVIEW_MODEL, messages=messages, temperature=0,
            response_format={"type": "json_object"})
        raw = r.choices[0].message.content.strip()
        tin, tout = r.usage.prompt_tokens, r.usage.completion_tokens
        cache[key] = raw
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False),
                              encoding="utf-8")
    try:
        data = json.loads(raw)
        verdict = data.get("verdict", "").strip()
        if verdict not in judges.VERDICTS:
            verdict = judges.PARSE_ERROR
        return verdict, str(data.get("reason", "")).strip(), tin, tout
    except json.JSONDecodeError:
        return judges.PARSE_ERROR, raw[:120], tin, tout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="連 k=3／k=15 那 16 筆 seed 標註也一起盲標")
    args = ap.parse_args()

    questions = {q["id"]: q for q in json.loads(
        QUESTIONS_PATH.read_text(encoding="utf-8"))}
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    texts = article_text_map()

    targets = [x for x in labels
               if args.all or not any(x["key"].endswith(s) for s in SEED_KS)]
    print(f"盲標 {len(targets)} 筆（覆核者 {REVIEW_MODEL}，看不到既有標註）…")

    out, tin_sum, tout_sum, disagree = [], 0, 0, []
    for item in targets:
        q = questions[int(item["key"].split("@")[0])]
        blob = NL.join(f"第 {no} 條{NL}{texts.get(no, '（原文缺漏）')}"
                       for no in q["expected"])
        verdict, reason, tin, tout = review(q["question"], blob, item["answer"])
        tin_sum, tout_sum = tin_sum + tin, tout_sum + tout
        row = {"key": item["key"], "mine": item["verdict"],
               "review": verdict, "review_reason": reason,
               "agree": verdict == item["verdict"],
               "mine_note": item.get("note", "")}
        out.append(row)
        if not row["agree"]:
            disagree.append(row)

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    cost = rag_core.twd(tin_sum / 1e6 * USD_PER_MILLION_IN
                        + tout_sum / 1e6 * USD_PER_MILLION_OUT)

    agree_n = sum(1 for r in out if r["agree"])
    print(f"{NL}盲標一致：{agree_n}/{len(out)}"
          f"　（本次計費 in={tin_sum} out={tout_sum}，約 {cost:.4f} 元）")
    if disagree:
        print(f"{NL}不一致 {len(disagree)} 筆——**不自動改，由人裁決**：")
        for r in disagree:
            print(f"{NL}  {r['key']}　我：{r['mine']}　→　{REVIEW_MODEL}：{r['review']}")
            print(f"    我的理由　：{r['mine_note']}")
            print(f"    覆核的理由：{r['review_reason']}")
    else:
        print(f"{NL}24 筆全數一致。")
    print(f"{NL}明細已寫入 {OUT_PATH.name}")


if __name__ == "__main__":
    main()
