# -*- coding: utf-8 -*-
"""量 SSE 的到達時間。

為什麼不用 curl：`--write-out` 那些時間欄位量的是傳輸層的事件，
而且 curl 預設會緩衝輸出。要量「第一個 token 幾點到」就得自己
在收到每一行的當下打時間戳，所以用 httpx 的 stream 模式。

用法：
    python sse_probe_day26.py                      # 單題，印出每個事件的時間
    python sse_probe_day26.py --run 28             # 28 題跑一輪，出 JSON
    python sse_probe_day26.py --run 28 --no-cache  # 真的打 API（要花錢）
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

BASE_DIR = Path(__file__).parent
URL = "http://127.0.0.1:8026/ask"


def ask_stream(client: httpx.Client, question: str, k: int = 3,
               use_cache: bool = True) -> dict:
    """送一題，回傳每個事件相對於送出時間的毫秒數。"""
    t0 = time.perf_counter()
    marks: dict = {"question": question, "tokens": 0}
    body = {"question": question, "k": k, "use_cache": use_cache, "stream": True}

    with client.stream("POST", URL, json=body, timeout=120.0) as r:
        r.raise_for_status()
        event = None
        for line in r.iter_lines():
            now = (time.perf_counter() - t0) * 1000
            if line.startswith("event: "):
                event = line[7:].strip()
            elif line.startswith("data: "):
                if event == "citations":
                    marks["citations_ms"] = round(now, 1)
                    marks["n_citations"] = len(json.loads(line[6:])["citations"])
                elif event == "token":
                    marks["tokens"] += 1
                    marks.setdefault("first_token_ms", round(now, 1))
                    marks["last_token_ms"] = round(now, 1)
                elif event == "done":
                    payload = json.loads(line[6:])
                    marks["done_ms"] = round(now, 1)
                    marks["cached"] = payload["usage"]["cached"]
                    marks["answer"] = payload["answer"]
                    marks["server_total_ms"] = payload["timing"]["total_ms"]
                    marks["server_first_token_ms"] = payload.get("first_token_ms")
    return marks


def ask_plain(client: httpx.Client, question: str, k: int = 3,
              use_cache: bool = True) -> dict:
    """非串流版，拿來對照。"""
    t0 = time.perf_counter()
    r = client.post(URL, json={"question": question, "k": k,
                               "use_cache": use_cache}, timeout=120.0)
    r.raise_for_status()
    wall = (time.perf_counter() - t0) * 1000
    d = r.json()
    return {"question": question, "wall_ms": round(wall, 1),
            "server_total_ms": d["timing"]["total_ms"],
            "cached": d["usage"]["cached"], "answer": d["answer"]}


#: Day 20 起固定的 28 題＝單條文 15 ＋ 多條文 8 ＋ 陷阱 5。
#: 沿用同一組，這篇的數字才跟 Day 25 那張延遲表比得起來。
QUESTION_FILES = ("questions.json", "questions_day17.json", "questions_trap.json")


def load_questions(n: int) -> list[str]:
    out = []
    for name in QUESTION_FILES:
        qs = json.loads((BASE_DIR / name).read_text(encoding="utf-8"))
        out += [q["question"] if isinstance(q, dict) else q for q in qs]
    return out[:n] if n else out


def p(values: list[float], q: float) -> float:
    """第 q 百分位。n 小的時候 statistics.quantiles 會很怪，自己算。"""
    if not values:
        return 0.0
    s = sorted(values)
    i = min(int(round(q / 100 * (len(s) - 1))), len(s) - 1)
    return round(s[i], 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=0, help="跑幾題")
    ap.add_argument("--no-cache", action="store_true", help="真的打 API")
    ap.add_argument("--out", default="runs_day26.json")
    args = ap.parse_args()
    use_cache = not args.no_cache

    with httpx.Client() as client:
        if not args.run:
            m = ask_stream(client, "病假請幾天要附診斷證明？", use_cache=use_cache)
            for k in ("citations_ms", "first_token_ms", "last_token_ms", "done_ms"):
                print(f"  {k:<18} {m.get(k)}")
            print(f"  {'tokens':<18} {m['tokens']}")
            print(f"  {'cached':<18} {m['cached']}")
            print(f"  答案 {m['answer'][:40]}…")
            return

        questions = load_questions(args.run)
        print(f"{len(questions)} 題，use_cache={use_cache}")
        # 暖機。第一次呼叫含 TLS 握手與 openai client 初始化，
        # 實測會多花十幾秒，量進去中位數會被一題帶歪
        print("暖機中…", end="", flush=True)
        ask_stream(client, "暖機用，不列入統計", use_cache=False)
        print(" 好")

        rows = []
        for i, q in enumerate(questions, 1):
            s = ask_stream(client, q, use_cache=use_cache)
            pl = ask_plain(client, q, use_cache=use_cache)
            s["plain_wall_ms"] = pl["wall_ms"]
            s["same_answer"] = (s["answer"] or "").strip() == (pl["answer"] or "").strip()
            rows.append(s)
            print(f"  {i:>2}. citations {s.get('citations_ms'):>7} ms │ "
                  f"first_token {s.get('first_token_ms'):>7} ms │ "
                  f"done {s['done_ms']:>7} ms │ cached={s['cached']}")

    out = BASE_DIR / args.out
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    def col(name):
        return [r[name] for r in rows if r.get(name) is not None]

    print("\n" + "=" * 62)
    print(f"{'':<22}{'中位數':>12}{'p95':>12}")
    for label, name in [("到 citations", "citations_ms"),
                        ("到第一個 token", "first_token_ms"),
                        ("到 done（總時間）", "done_ms"),
                        ("非串流總時間", "plain_wall_ms")]:
        v = col(name)
        print(f"{label:<22}{statistics.median(v):>12.1f}{p(v, 95):>12.1f}")
    print("=" * 62)
    med_cit = statistics.median(col("citations_ms"))
    med_done = statistics.median(col("done_ms"))
    med_plain = statistics.median(col("plain_wall_ms"))
    print(f"使用者看到第一個東西：非串流 {med_plain:.1f} ms → 串流 {med_cit:.1f} ms"
          f"（{med_plain / med_cit:.1f} 倍）")
    print(f"總時間：非串流 {med_plain:.1f} ms → 串流 {med_done:.1f} ms"
          f"（{(med_done / med_plain - 1) * 100:+.1f}%）")
    # 成對比較。中位數各算各的會被題目難易度洗掉，
    # 同一題自己跟自己比才看得出串流到底有沒有換到總時間
    diffs = [r["done_ms"] - r["plain_wall_ms"] for r in rows]
    faster = sum(1 for d in diffs if d < 0)
    print(f"成對差額（串流 − 非串流）：中位數 {statistics.median(diffs):+.1f} ms，"
          f"串流較快 {faster}/{len(diffs)} 題")
    print(f"命中快取：{sum(1 for r in rows if r['cached'])}/{len(rows)}")
    print(f"答案與非串流一致：{sum(1 for r in rows if r['same_answer'])}/{len(rows)}")
    print(f"寫出 {out.name}")


if __name__ == "__main__":
    main()
