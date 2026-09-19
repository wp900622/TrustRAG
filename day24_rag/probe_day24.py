# -*- coding: utf-8 -*-
"""Day 24 的兩個追查：middleware 到底要多少錢，還有那 1,306 毫秒是誰的。

起因：寫服務的時候我隨手量了一次 `POST /documents`，回 1,306 ms。
當下的結論是「BaseHTTPMiddleware 把背景任務變成前景任務」——
這個說法在網路上很常見，我也寫進註解了。

結果正式跑三種掛法，全部都是 20～28 ms，**沒有重現**。
所以這支檔案專門處理兩件事，兩件都不打 LLM，0 元：

1. `POST /documents` 連打 N 次，三種掛法並排，看那 1,306 ms 會不會回來
2. middleware 自己的成本：用 `/healthz` 量（它什麼事都不做），
   而不是用 `/ask`（它被 10.5 MB 的快取讀蓋住了，根本看不到 middleware）

用法：
    python probe_day24.py            # 合併進 runs_day24.json 並重出報表
"""
import json
import statistics
import time
from pathlib import Path

import httpx

import run_experiment_day24 as exp

BASE_DIR = Path(__file__).parent
MODES = (("asgi", "純 ASGI"), ("base", "BaseHTTPMiddleware"), ("none", "不掛"))
REPEATS = 30
ROUNDS = 3


def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(round((len(s) - 1) * p)))]


def healthz_cost(port: int, repeats: int = REPEATS) -> dict:
    """/healthz 只回一個 dict，沒有 embedding、沒有檢索、沒有 LLM。

    量 middleware 就該量這個——用 /ask 量等於拿一把公斤秤稱一根頭髮。
    連線重用（同一個 Client），所以量到的不含 TCP 握手。
    """
    with httpx.Client(timeout=30.0) as client:
        url = f"http://127.0.0.1:{port}/healthz"
        client.get(url)                                   # 暖身，不計
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            client.get(url)
            samples.append((time.perf_counter() - t0) * 1000)
    return {"n": repeats, "p50": round(pct(samples, 0.5), 3),
            "p95": round(pct(samples, 0.95), 3),
            "mean": round(statistics.fmean(samples), 3),
            "min": round(min(samples), 3)}


def documents_latency(port: int, repeats: int = 5) -> dict:
    """POST /documents 連打幾次。

    第一次與後面幾次分開記：如果 1,306 ms 只出現在第一次，
    那就不是 middleware 的事，是冷啟動。
    """
    samples, jobs = [], []
    with httpx.Client(timeout=300.0) as client:
        for _ in range(repeats):
            t0 = time.perf_counter()
            r = client.post(f"http://127.0.0.1:{port}/documents",
                            json={"path": "work_rules.md", "ocr": False})
            samples.append(round((time.perf_counter() - t0) * 1000, 1))
            jobs.append(r.json()["job_id"])
            # 等這一個 job 收工再打下一個，否則量到的是排隊
            deadline = time.perf_counter() + 120
            while time.perf_counter() < deadline:
                j = client.get(f"http://127.0.0.1:{port}/documents/{jobs[-1]}").json()
                if j["status"] in ("done", "failed"):
                    break
                time.sleep(0.05)
    return {"first": samples[0], "rest": samples[1:],
            "rest_median": round(pct(samples[1:], 0.5), 1) if samples[1:] else None,
            "all": samples}


def fresh_connection_post(port: int) -> float:
    """每次都開一條新連線打一次——這才是我當初隨手量的那個寫法"""
    t0 = time.perf_counter()
    httpx.post(f"http://127.0.0.1:{port}/documents",
               json={"path": "work_rules.md", "ocr": False}, timeout=300.0)
    return round((time.perf_counter() - t0) * 1000, 1)


def fresh_connection_healthz(port: int) -> float:
    """對照組：同樣開一條新連線，但打的是什麼事都不做的 /healthz。

    這一格是關鍵。如果它也要將近一秒，那當初那 1,306 ms 就跟攝取、
    跟 middleware、跟服務全都無關——是我的量測腳本自己的成本。
    """
    t0 = time.perf_counter()
    httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=30.0)
    return round((time.perf_counter() - t0) * 1000, 1)


def main() -> None:
    out = json.loads((BASE_DIR / "runs_day24.json").read_text(encoding="utf-8"))
    probe: dict = {"repeats_healthz": REPEATS}

    # 三輪交錯跑，不是一種掛法連跑三次：process 剛起來那一輪比較慢，
    # 照順序跑會把那個慢算在第一個掛法頭上
    rounds = {mode: [] for mode, _ in MODES}
    for rnd in range(ROUNDS):
        for i, (mode, label) in enumerate(MODES):
            port = 8040 + i + rnd * 10
            proc, boot, _health = exp.start_service(mode, port)
            try:
                row = {"boot_ms": boot,
                       "healthz": healthz_cost(port),
                       "fresh_conn_healthz": fresh_connection_healthz(port),
                       "fresh_conn_post": fresh_connection_post(port)}
                time.sleep(1.5)                 # 讓那個 job 跑完再量下一組
                row["documents"] = documents_latency(port, repeats=3)
            finally:
                exp.stop_service(proc)
            rounds[mode].append(row)
            print(f"  [{rnd + 1}/{ROUNDS}] {label:<20} "
                  f"/healthz p50 {row['healthz']['p50']:.2f} ms｜"
                  f"新連線 /healthz {row['fresh_conn_healthz']} ms｜"
                  f"新連線 POST {row['fresh_conn_post']} ms｜"
                  f"之後 {row['documents']['rest']}", flush=True)

    for mode, label in MODES:
        rs = rounds[mode]
        probe[mode] = {
            "label": label, "rounds": rs,
            "healthz": {"p50": round(pct([r["healthz"]["p50"] for r in rs], 0.5), 3),
                        "p95": round(pct([r["healthz"]["p95"] for r in rs], 0.5), 3),
                        "min": round(min(r["healthz"]["min"] for r in rs), 3),
                        "n": REPEATS * len(rs)},
            "fresh_conn_healthz": [r["fresh_conn_healthz"] for r in rs],
            "first_post_fresh_conn": [r["fresh_conn_post"] for r in rs],
            "documents_rest": [x for r in rs for x in r["documents"]["rest"]]}

    out["probe"] = probe
    (BASE_DIR / "runs_day24.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (BASE_DIR / "experiment_result_day24.md").write_text(
        exp.build_report(out), encoding="utf-8")
    print("  → runs_day24.json　→ experiment_result_day24.md")


if __name__ == "__main__":
    main()
