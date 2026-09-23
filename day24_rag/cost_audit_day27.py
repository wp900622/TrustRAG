# -*- coding: utf-8 -*-
"""Day 27：把 26 天的 API 帳單重建出來。**這支不打 API。**

為什麼需要一支專門的腳本：`rag_core.chat()` 快取命中時回 `(text, 0, 0)`，
所以每天 `experiment_result_*.md` 報的是「今天實際刷卡多少」，不是累計。
26 天下來，總帳沒有人在記。

三個難點，對應三段程式碼：

1. **輸出可以精確重數**——答案原文還在快取裡，`tiktoken` 數就是了。
   但要先拿 API 真的回報過的 usage 校驗一次（`verify_tokenizer`）。
2. **輸入不見了**——快取 key 是 `sha256([model, messages])[:16]`，prompt 沒留。
   只能考古：把已知的 prompt 空間重建一次算雜湊去比對。
   對得上的有精確輸入；對不上的用比例回推，並且標成估計。
3. **反事實不是現金**——快取擋下的呼叫本來就不會付第二次錢。
   那個數字只能說成「同樣的實驗從零重跑一次要花多少」。

防呆：進場就把 `rag_core.get_client` 換成會爆炸的版本。
這支腳本一旦不小心打到 API，會在這裡停下來，不會靜靜花錢。
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tiktoken

BASE_DIR = Path(__file__).parent
NL = chr(10)

CHAT_ENC = tiktoken.get_encoding("o200k_base")     # gpt-4o / gpt-4o-mini
EMB_ENC = tiktoken.get_encoding("cl100k_base")     # text-embedding-3-small

MINI, GPT4O = "gpt-4o-mini", "gpt-4o"
EMB_MODEL = "text-embedding-3-small"
# 每百萬 token 美元。與 rag_core.py／judges_day19.py 同一組數字。
PRICES = {MINI: (0.15, 0.60), GPT4O: (2.50, 10.00)}
USD_PER_M_EMB = 0.02
USD_TO_TWD = 31


def usd(model: str, tin: int, tout: int) -> float:
    pin, pout = PRICES[model]
    return tin / 1e6 * pin + tout / 1e6 * pout


def twd(x: float) -> float:
    return x * USD_TO_TWD


def load(name: str):
    p = BASE_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def count_messages(messages: list) -> int:
    """OpenAI chat 的輸入 token：每則 +3、整體 +3（與 judges.py 同一支算法）"""
    total = 0
    for m in messages:
        c = m.get("content") or ""
        if not isinstance(c, str):
            c = json.dumps(c, ensure_ascii=False)
        total += 3 + len(CHAT_ENC.encode(m.get("role", ""))) + len(CHAT_ENC.encode(c))
    return total + 3


def chat_key(model: str, messages: list) -> str:
    payload = json.dumps([model, messages], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def emb_key(text: str) -> str:
    return hashlib.sha256(f"{EMB_MODEL}:{text}".encode("utf-8")).hexdigest()[:16]


# --- 兩則訊息（system＋user）的雜湊快速路徑。
# 一般寫法對 30 萬組候選要做 30 萬次 json.dumps，system prompt 一兩 KB，
# 光序列化就要好幾分鐘。這裡把不變的前後綴先組好，只序列化會變的 user 字串。
# 進場時用 _assert_fastpath() 跟正規寫法對一次，確認兩者位元相同。
def pair_key_builder(model: str, system: str):
    head = ('["' + model + '", [{"content": '
            + json.dumps(system, ensure_ascii=False) + ', "role": "system"}, '
            + '{"content": ')
    tail = ', "role": "user"}]]'

    def build(user: str) -> str:
        payload = head + json.dumps(user, ensure_ascii=False) + tail
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    return build


def _assert_fastpath() -> None:
    sys_p, user_p = "系統提示 with \"quote\" 與換行\n第二行", "使用者訊息 \\ 測試"
    fast = pair_key_builder(MINI, sys_p)(user_p)
    slow = chat_key(MINI, [{"role": "system", "content": sys_p},
                           {"role": "user", "content": user_p}])
    assert fast == slow, "快速路徑與 json.dumps 不一致，考古會全部對不上"


# ------------------------------------------------------- 0. 先確認不會打 API

def disarm() -> None:
    import rag_core

    def boom(*_a, **_k):
        raise RuntimeError("Day 27 的稽核腳本不准打 API")

    rag_core.get_client = boom


# ------------------------------------------- 1. 驗 tokenizer（沒驗過不能寫）

def verify_tokenizer() -> dict:
    """拿 API 真的回報過 output_tokens 的答案回頭數，看 tiktoken 準不準。

    `runs_day21/22/23.json` 每筆都記了當初 API 回報的 output_tokens 與答案原文，
    這是唯一能校驗離線計數的地方。多步 agent 的 output_tokens 是整輪加總，排掉。
    """
    pairs = []
    for f in ("runs_day21.json", "runs_day22.json", "runs_day23.json"):
        for _, per in (load(f) or {}).items():
            for _, rec in per.items():
                if not isinstance(rec, dict):
                    continue
                ans, out = rec.get("answer"), rec.get("output_tokens")
                if ans and out and rec.get("llm_calls", 1) == 1 and not rec.get("images"):
                    pairs.append((len(CHAT_ENC.encode(ans)), out))
    if not pairs:
        return {"n": 0}
    diffs = [a - b for a, b in pairs]
    return {"n": len(pairs),
            "exact": sum(1 for d in diffs if d == 0),
            "within1": sum(1 for d in diffs if abs(d) <= 1),
            "mean_diff": round(sum(diffs) / len(diffs), 3),
            "max_abs": max(abs(d) for d in diffs)}


# ------------------------------------------------------------- 2. 考古

def all_questions() -> list:
    qs = []
    for fname in ("questions.json", "questions_trap.json", "questions_day17.json",
                  "questions_multihop.json", "questions_day23.json"):
        qs += [q["question"] for q in (load(fname) or []) if "question" in q]
    return list(dict.fromkeys(qs))


def all_answers() -> list:
    """所有「曾經被生出來、可能又被送進裁判」的答案文字。"""
    out = []
    for fname in ("human_labels_day18.json", "human_labels_day20.json",
                  "human_labels_day21.json", "human_labels_day22.json",
                  "human_labels_day23.json"):
        for row in (load(fname) or []):
            if isinstance(row, dict) and row.get("answer"):
                out.append(row["answer"])
    for fname in ("runs_day21.json", "runs_day22.json", "runs_day23.json"):
        for _, per in (load(fname) or {}).items():
            for _, rec in per.items():
                if isinstance(rec, dict) and rec.get("answer"):
                    out.append(rec["answer"])
    for fname in ("chat_cache.json", "chat_cache_day19.json"):
        out += [v for v in (load(fname) or {}).values() if isinstance(v, str)]
    return list(dict.fromkeys(out))


def _day22_collections() -> dict:
    """Day 22 的 G／T／V／V+ 四份語料。OCR 走 `ocr_cache_day22.json`，不打 API。"""
    try:
        import ingest_day22 as ig
        return {name: d["collection"] for name, d in ig.ingest(use_ocr=True).items()}
    except Exception as exc:                                   # noqa: BLE001
        print(f"      [warn] Day 22 語料重建失敗，跳過：{exc}")
        return {}


def _day23_collections() -> dict:
    """Day 23 的兩份附件語料（T＝PDF 文字層、O＝OCR）。"""
    try:
        import run_experiment_day23 as d23
        out = {}
        for name, fn in (("T", d23.annex_text_T), ("O", d23.annex_text_O)):
            md, _ = fn()
            coll, _chunks = d23.build_collection(f"day27-audit-{name}",
                                                 d23.build_corpus(md))
            out[name] = coll
        return out
    except Exception as exc:                                   # noqa: BLE001
        print(f"      [warn] Day 23 語料重建失敗，跳過：{exc}")
        return {}


def archaeology() -> tuple:
    """重建已知的 prompt 空間，回傳 (key → 描述子, 描述子 → messages 的還原器)。

    只存描述子不存 messages：三十萬組候選存成 messages 會吃掉幾百 MB，
    而真正要精算輸入 token 的只有對得上的那一千多筆。
    """
    import prompts

    keymap = {}            # key -> descriptor
    main_msgs = {}         # descriptor -> messages（主線的量小，直接存）
    questions = all_questions()

    # --- 主線：語料 → top-k → prompt
    try:
        import chroma_store
        import pipeline
        cache = load("embeddings_cache.json") or {}
        collection, _chunks, _articles, _t, _c = pipeline.build_index()
    except Exception as exc:                                   # noqa: BLE001
        print(f"      [warn] 索引重建失敗，主線考古跳過：{exc}")
        collection, cache = None, {}

    if collection is not None:
        for qi, q in enumerate(questions):
            key = emb_key(q)
            if key not in cache:
                continue
            vec = np.array(cache[key], dtype=np.float32)
            vec /= np.linalg.norm(vec)
            for k in (1, 3, 5, 10, 15):
                hits = chroma_store.retrieve(collection, vec, k)
                orders = [("", hits)]
                if k == 10:      # Day 17 的條文順序實驗
                    orders.append(("rev", list(reversed(hits))))
                for oname, hh in orders:
                    for v in ("P1", "P2", "P3"):
                        d = ("main", v, k, oname, qi)
                        msgs = prompts.build_messages(q, hh, version=v)
                        main_msgs[d] = msgs
                        keymap.setdefault(chat_key(MINI, msgs), d)
    for qi, q in enumerate(questions):
        d = ("nocontext", qi)
        msgs = prompts.build_messages_no_context(q)
        main_msgs[d] = msgs
        keymap.setdefault(chat_key(MINI, msgs), d)

    # --- Day 22 的四種語料形態、Day 23 的兩份附件語料：各自一個索引、k=3
    for tag, colls in (("day22", _day22_collections()), ("day23", _day23_collections())):
        for name, coll in colls.items():
            for qi, q in enumerate(questions):
                key = emb_key(q)
                if key not in cache:
                    continue
                vec = np.array(cache[key], dtype=np.float32)
                vec /= np.linalg.norm(vec)
                import chroma_store
                hits = chroma_store.retrieve(coll, vec, 3)
                d = (tag, name, qi)
                msgs = prompts.build_messages(q, hits, version="P1")
                main_msgs[d] = msgs
                keymap.setdefault(chat_key(MINI, msgs), d)

    # --- Day 18 的盲標覆核（gpt-4o，獨立一份快取）
    try:
        import review_labels_day18 as rv
        qs17 = {q["id"]: q for q in (load("questions_day17.json") or [])}
        texts = rv.article_text_map()
        for item in (load("human_labels_day18.json") or []):
            q = qs17.get(int(item["key"].split("@")[0]))
            if not q:
                continue
            blob = NL.join(f"第 {no} 條{NL}{texts.get(no, '（原文缺漏）')}"
                           for no in q["expected"])
            msgs = [{"role": "system", "content": rv.SYSTEM},
                    {"role": "user", "content": rv.USER.format(
                        question=q["question"], articles=blob,
                        answer=item["answer"])}]
            d = ("review", item["key"])
            main_msgs[d] = msgs
            keymap.setdefault(chat_key(GPT4O, msgs), d)
    except Exception as exc:                                   # noqa: BLE001
        print(f"      [warn] Day 18 覆核訊息重建失敗，跳過：{exc}")

    # --- Day 18／19：rubric 判定。messages 只跟（判斷點, 答案）有關
    import judges
    try:
        import judges_day19
        systems = dict(judges_day19.CHECK_SYSTEMS)
    except Exception:                                          # noqa: BLE001
        systems = {"B0": judges.CHECK_SYSTEM}

    points = []
    for item in (load("rubric_day18.json") or []):
        points += [cp["point"] for cp in item.get("checkpoints", [])]
    points = list(dict.fromkeys(points))
    answers = all_answers()
    print(f"      rubric 空間：{len(systems)} 個 prompt 變體 × {len(points)} 個判斷點 "
          f"× {len(answers)} 個答案")

    users = [f"判斷點：{p}{NL}{NL}答案：{a}" for p in points for a in answers]
    for sysname, sysprompt in systems.items():
        for model in (MINI, GPT4O):
            build = pair_key_builder(model, sysprompt)
            for ui, u in enumerate(users):
                keymap.setdefault(build(u), ("rubric", sysname, model, ui))

    def resolve(d):
        kind = d[0]
        if kind in ("main", "nocontext", "day22", "day23", "review"):
            return main_msgs[d]
        if kind == "rubric":
            _, sysname, _model, ui = d
            return [{"role": "system", "content": systems[sysname]},
                    {"role": "user", "content": users[ui]}]
        return None

    return keymap, resolve


# ------------------------------------------------------- 3. 逐一快取結帳

def audit_text_cache(name: str, model: str, keymap: dict, resolve) -> dict:
    """答案原文型的快取：輸出精確、輸入靠考古，對不上的留給後面回推。"""
    cache = load(name) or {}
    row = {"file": name, "model": model, "n": len(cache), "out_tokens": 0,
           "matched": 0, "in_tokens_exact": 0, "matched_out_tokens": 0,
           "unmatched": 0, "unmatched_out_tokens": 0,
           "tags": defaultdict(lambda: {"n": 0, "in": 0, "out": 0}),
           "unmatched_out_hist": defaultdict(int),
           "matched_by_bucket": defaultdict(lambda: {"n": 0, "in": 0})}

    def bucket(o):
        return "≤5" if o <= 5 else "6-20" if o <= 20 else "21-60" if o <= 60 else "61+"

    for key, text in cache.items():
        if not isinstance(text, str):
            continue
        o = len(CHAT_ENC.encode(text))
        row["out_tokens"] += o
        d = keymap.get(key)
        if d is not None:
            msgs = resolve(d)
            i = count_messages(msgs)
            row["matched"] += 1
            row["matched_out_tokens"] += o
            row["in_tokens_exact"] += i
            t = row["tags"][d[0] if d[0] != "rubric" else f"rubric-{d[1]}"]
            t["n"] += 1
            t["in"] += i
            t["out"] += o
            b = row["matched_by_bucket"][bucket(o)]
            b["n"] += 1
            b["in"] += i
        else:
            row["unmatched"] += 1
            row["unmatched_out_tokens"] += o
            # 判定型的回覆只有「符合／未提及／相反」那幾個字，答案型長得多。
            # 對不上的那些是哪一種，用輸出長度分粗略看得出來——
            # 而回推輸入時也照這個分組各用各的平均，不要全部用同一個數字。
            row["unmatched_out_hist"][bucket(o)] += 1
    row["tags"] = dict(row["tags"])
    row["unmatched_out_hist"] = dict(row["unmatched_out_hist"])
    row["matched_by_bucket"] = dict(row["matched_by_bucket"])

    # 回推：對不上的那些，用同一個輸出長度分組裡「對得上的」平均輸入
    est, basis = 0, {}
    allb = row["matched_by_bucket"]
    fallback = (row["in_tokens_exact"] / row["matched"]) if row["matched"] else 0
    for b, n in row["unmatched_out_hist"].items():
        mb = allb.get(b)
        mean = (mb["in"] / mb["n"]) if mb and mb["n"] else fallback
        basis[b] = {"n": n, "mean_in": round(mean, 1),
                    "from": (mb["n"] if mb else 0)}
        est += n * mean
    row["in_tokens_est"] = round(est)
    row["est_basis"] = basis
    return row


def audit_agent_cache(name: str) -> dict:
    """agent 的快取存的是 assistant 訊息物件；輸出要含 tool_calls 的 JSON。"""
    cache = load(name) or {}
    out = 0
    for _, msg in cache.items():
        if not isinstance(msg, dict):
            continue
        out += len(CHAT_ENC.encode(msg.get("content") or ""))
        for tc in (msg.get("tool_calls") or []):
            out += len(CHAT_ENC.encode(json.dumps(tc.get("function", {}),
                                                  ensure_ascii=False)))
    return {"file": name, "n": len(cache), "out_tokens": out}


def audit_vision_cache(name: str) -> dict:
    """視覺快取當初就把 usage 存下來了，唯一不用猜的一份。"""
    cache = load(name) or {}
    return {"file": name, "n": len(cache),
            "in_tokens": sum(v.get("input_tokens", 0) for v in cache.values()
                             if isinstance(v, dict)),
            "out_tokens": sum(v.get("output_tokens", 0) for v in cache.values()
                              if isinstance(v, dict))}


def audit_embeddings() -> dict:
    """350 筆向量：key 是 sha256("model:text")，原文沒留，同樣用考古認。"""
    cache = load("embeddings_cache.json") or {}
    known = {}

    def add(t):
        if isinstance(t, str) and t.strip():
            known.setdefault(emb_key(t), t)

    try:
        import chunkers
        import rag_core
        text = rag_core.load_document()
        for c in chunkers.chunk_by_structure(text):
            add(c["text"])
        for a in chunkers.parse_articles(text):
            add(a.get("text", ""))
        for fn in dir(chunkers):
            if not fn.startswith("chunk_"):
                continue
            f = getattr(chunkers, fn)
            try:
                for c in f(text):
                    add(c["text"] if isinstance(c, dict) else c)
            except Exception:                                  # noqa: BLE001
                pass
    except Exception as exc:                                   # noqa: BLE001
        print(f"      [warn] 語料切塊失敗：{exc}")

    for q in all_questions():
        add(q)
    for sub in ("corpus_day22", "corpus_day23", "documents"):
        d = BASE_DIR / sub
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() not in (".md", ".txt"):
                continue
            try:
                t = p.read_text(encoding="utf-8")
            except Exception:                                  # noqa: BLE001
                continue
            add(t)
            try:
                import chunkers
                for c in chunkers.chunk_by_structure(t):
                    add(c["text"])
            except Exception:                                  # noqa: BLE001
                pass
    # agent 自己下的查詢字串也各買過一次向量
    for fname in ("runs_day21.json", "runs_day22.json"):
        for _, per in (load(fname) or {}).items():
            for _, rec in per.items():
                if isinstance(rec, dict):
                    for q in (rec.get("queries") or []):
                        add(q)

    matched = [k for k in cache if k in known]
    return {"n": len(cache), "matched": len(matched),
            "tokens_matched": sum(len(EMB_ENC.encode(known[k])) for k in matched),
            "known_space": len(known)}


# --------------------------------------------- 4. 反事實：沒有快取要花多少

def audit_runs() -> dict:
    """`runs_*.json` 記的是每一次邏輯呼叫的 token，不管當天有沒有命中快取。

    加起來 = 「這些實驗如果每次都真的打 API」的帳單。
    """
    out = {}
    for f in ("runs_day21.json", "runs_day22.json", "runs_day23.json"):
        t = {"calls": 0, "in": 0, "out": 0, "emb": 0, "rows": 0}
        for _, per in (load(f) or {}).items():
            for _, rec in per.items():
                if not isinstance(rec, dict):
                    continue
                t["rows"] += 1
                t["calls"] += rec.get("llm_calls", 1)
                t["in"] += rec.get("input_tokens", 0)
                t["out"] += rec.get("output_tokens", 0)
                t["emb"] += rec.get("embedding_tokens", 0)
        out[f] = t
    return out


# ------------------------------------------------------------------- main

def main() -> None:
    disarm()
    _assert_fastpath()

    print("[1/5] 驗 tokenizer…")
    tv = verify_tokenizer()
    print(f"      n={tv.get('n')}　完全一致={tv.get('exact')}　"
          f"誤差≤1={tv.get('within1')}　最大誤差={tv.get('max_abs')}")

    print("[2/5] 考古：重建 prompt 空間…")
    keymap, resolve = archaeology()
    print(f"      候選 key {len(keymap):,} 組")

    print("[3/5] 逐份快取結帳…")
    main_cache = audit_text_cache("chat_cache.json", MINI, keymap, resolve)
    d19 = audit_text_cache("chat_cache_day19.json", GPT4O, keymap, resolve)
    review = audit_text_cache("review_cache_day18.json", GPT4O, keymap, resolve)
    d24 = [audit_text_cache(f"chat_cache_day24_{x}.json", MINI, keymap, resolve)
           for x in ("raw", "atomic", "lock")]
    agents = [audit_agent_cache(f) for f in ("agent_cache_day20.json",
                                             "agent_cache_day21.json")]
    visions = [audit_vision_cache(f) for f in ("vision_cache_day22.json",
                                               "vision_cache_day23.json")]
    print(f"      chat_cache.json：{main_cache['matched']}/{main_cache['n']} 認得出來")

    print("[4/5] embedding 考古…")
    emb = audit_embeddings()
    print(f"      {emb['matched']}/{emb['n']} 認得出來（候選 {emb['known_space']}）")

    print("[5/5] 反事實…")
    runs = audit_runs()

    out = {"tokenizer": tv, "candidates": len(keymap), "main": main_cache,
           "day19": d19, "review": review, "day24": d24, "agents": agents,
           "visions": visions, "emb": emb, "runs": runs}
    (BASE_DIR / "cost_audit_day27.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{NL}原始數字寫入 cost_audit_day27.json")


if __name__ == "__main__":
    main()
