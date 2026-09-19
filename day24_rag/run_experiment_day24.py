# -*- coding: utf-8 -*-
"""Day 24 實驗：同一組 28 題改走 HTTP，把一次請求拆成四段。

問題很簡單：**我前 13 天一直在調的那一段（檢索），佔一次請求的幾 %？**

兩組條件，這是今天最重要的設計：

    warm   兩個快取都開   ← 前 23 天腳本的樣子
    cold   兩個快取都關   ← 上線之後每一個新問題的樣子

如果照舊只跑 warm，「LLM」那一段會量到 2 毫秒，那是快取的速度不是模型的。
cold 這組每題真的打一次 embedding API 與一次 chat API，是今天唯一花錢的地方。

另外三件事順便量：
    retriever    Chroma vs numpy，同樣 28 題，top-3 有沒有不一樣
    middleware   純 ASGI／BaseHTTPMiddleware／不掛，三種各量一次
    背景任務     POST /documents 多久回、job 實際跑多久

用法：
    python run_experiment_day24.py            # 全部（cold 那組會真的打 56 次 API）
    python run_experiment_day24.py --no-paid  # 只跑 warm，0 元
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

import rag_core
import run_experiment_day20 as d20

BASE_DIR = Path(__file__).parent
RESULT_PATH = BASE_DIR / "experiment_result_day24.md"
RUNS_PATH = BASE_DIR / "runs_day24.json"
NL = chr(10)

K = 3
PORT = 8024
STAGES = ("embed", "retrieve", "llm", "serialize")
STAGE_LABEL = {"embed": "算問題向量", "retrieve": "檢索",
               "llm": "LLM", "serialize": "序列化"}


# ------------------------------------------------------------ 起／停服務

def start_service(middleware: str = "asgi", port: int = PORT):
    """開一支 uvicorn，等到 /healthz 回話為止。

    服務的輸出導到檔案，不要用 subprocess.PIPE——沒人讀那根管子，
    Windows 上約 64 KB 就塞滿，服務會卡在 write 上不動。
    """
    env = dict(os.environ, DAY24_MIDDLEWARE=middleware)
    log_path = BASE_DIR / f"service_day24_{middleware}_{port}.log"
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "uvicorn", "day24_service.main:app",
         "--port", str(port), "--log-level", "warning"],
        cwd=str(BASE_DIR), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    proc._log, proc._log_path = log, log_path
    health = {}
    while time.perf_counter() - t0 < 180:
        if proc.poll() is not None:
            log.flush()
            raise RuntimeError("服務啟動就死了：" +
                               log_path.read_text(encoding="utf-8")[-1500:])
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=2.0)
            if r.status_code == 200:
                health = r.json()
                break
        except httpx.HTTPError:
            time.sleep(0.05)
    else:
        proc.kill()
        raise RuntimeError("等不到 /healthz")
    return proc, round((time.perf_counter() - t0) * 1000, 1), health


def stop_service(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    proc._log.close()


# ---------------------------------------------------------------- 打請求

def key_of(q: dict) -> str:
    return f"{q['group']}-{q['id']}"


def ask_all(questions, use_cache: bool, retriever: str = "chroma",
            port: int = PORT, mode: str = "pipeline") -> list[dict]:
    """序列問 28 題。延遲同時記兩個：呼叫端量到的牆上時間，與服務自報的 total。"""
    rows = []
    with httpx.Client(timeout=600.0) as client:
        for q in questions:
            t0 = time.perf_counter()
            r = client.post(f"http://127.0.0.1:{port}/ask",
                            params={"retriever": retriever},
                            json={"question": q["question"], "k": K,
                                  "use_cache": use_cache, "mode": mode})
            wall = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                rows.append({"key": key_of(q), "status": r.status_code,
                             "wall_ms": wall, "error": r.text[:300]})
                continue
            body = r.json()
            rows.append({
                "key": key_of(q), "status": 200, "wall_ms": wall,
                "answer": body["answer"], "retriever": body["retriever"],
                "cited": [c["article_no"] for c in body["citations"]],
                "timing": body["timing"], "usage": body["usage"],
                "mode": body.get("mode"), "agent": body.get("agent"),
                # header 那一份是 middleware 寫的，用來驗 contextvar 有沒有傳過去
                "header_total": r.headers.get("x-total-ms"),
                "header_llm": r.headers.get("x-llm-ms")})
    return rows


# ------------------------------------------------------------------ 統計

def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round((len(s) - 1) * p)))]


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == 200]
    totals = [r["timing"]["total_ms"] for r in ok]
    out = {"n": len(rows), "ok": len(ok),
           "p50": round(pct(totals, 0.5), 1), "p95": round(pct(totals, 0.95), 1),
           "max": round(max(totals), 1) if totals else 0.0,
           "mean": round(statistics.fmean(totals), 1) if totals else 0.0,
           "wall_p50": round(pct([r["wall_ms"] for r in ok], 0.5), 1),
           "stages": {}, "share": {}}
    median_total = out["p50"] or 1.0
    for name in STAGES + ("overhead",):
        vals = [r["timing"][f"{name}_ms"] for r in ok]
        med = round(pct(vals, 0.5), 3)
        out["stages"][name] = med
        out["share"][name] = round(med / median_total * 100, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-paid", action="store_true",
                    help="跳過 cold（會真的打 API 的那一組）")
    args = ap.parse_args()

    questions = d20.load_questions()
    out: dict = {"n": len(questions), "k": K}
    t_start = time.perf_counter()

    def step(msg: str) -> None:
        print(f"  [{time.perf_counter() - t_start:6.1f}s] {msg}", flush=True)

    print(f"[Day 24] {len(questions)} 題、k={K}", flush=True)

    # ---------------------------------------------- 主服務（純 ASGI middleware）
    proc, boot, health = start_service("asgi")
    out["boot_ms"], out["health"] = boot, health
    try:
        step("warm：28 題，兩個快取都開（chroma）…")
        warm = ask_all(questions, use_cache=True, retriever="chroma")
        out["warm"] = {"summary": summarise(warm), "rows": warm}

        step("numpy：同樣 28 題，只換 retriever…")
        numpy_rows = ask_all(questions, use_cache=True, retriever="numpy")
        out["numpy"] = {"summary": summarise(numpy_rows), "rows": numpy_rows}

        step("agent-warm：28 題走 mode=agent，快取全開…")
        aw = ask_all(questions, use_cache=True, retriever="chroma", mode="agent")
        out["agent_warm"] = {"summary": summarise(aw), "rows": aw}

        step("agent-numpy：同樣 28 題，凍結的 agent 換跑在 numpy 上…")
        an = ask_all(questions, use_cache=True, retriever="numpy", mode="agent")
        out["agent_numpy"] = {"summary": summarise(an), "rows": an}

        if not args.no_paid:
            step("cold：28 題，兩個快取都關（會真的打 56 次 API）…")
            cold = ask_all(questions, use_cache=False, retriever="chroma")
            out["cold"] = {"summary": summarise(cold), "rows": cold}

            step("agent-cold：28 題走 agent 且不走快取（今天最貴的一段）…")
            ac = ask_all(questions, use_cache=False, retriever="chroma",
                         mode="agent")
            out["agent_cold"] = {"summary": summarise(ac), "rows": ac}

        step("背景攝取：POST /documents …")
        out["ingest"] = probe_ingest()
    finally:
        stop_service(proc)

    # ---------------------------------------------- middleware 三種掛法
    out["middleware"] = {}
    for mode, port in (("asgi", PORT), ("base", PORT + 1), ("none", PORT + 2)):
        step(f"middleware={mode} …")
        proc, _boot, _health = start_service(mode, port)
        try:
            rows = ask_all(questions[:10], use_cache=True, port=port)
            ing = probe_ingest(port)
        finally:
            stop_service(proc)
        ok = [r for r in rows if r.get("status") == 200]
        out["middleware"][mode] = {
            "ask_p50": round(pct([r["timing"]["total_ms"] for r in ok], 0.5), 1),
            "header_total_present": sum(1 for r in ok if r.get("header_total")),
            "header_llm_present": sum(1 for r in ok if r.get("header_llm")),
            "post_documents_ms": ing["post_ms"],
            "job_elapsed_ms": ing["job"]["elapsed_ms"],
            "n": len(rows)}

    RUNS_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    RESULT_PATH.write_text(build_report(out), encoding="utf-8")
    step(f"→ {RUNS_PATH.name}　→ {RESULT_PATH.name}")


def probe_ingest(port: int = PORT) -> dict:
    """POST /documents 要立刻回，攝取在背景跑。量「回得多快」與「實際跑多久」。"""
    with httpx.Client(timeout=300.0) as client:
        t0 = time.perf_counter()
        r = client.post(f"http://127.0.0.1:{port}/documents",
                        json={"path": "work_rules.md", "ocr": False})
        post_ms = round((time.perf_counter() - t0) * 1000, 1)
        job = r.json()
        deadline = time.perf_counter() + 120
        while time.perf_counter() < deadline:
            job = client.get(
                f"http://127.0.0.1:{port}/documents/{job['job_id']}").json()
            if job["status"] in ("done", "failed"):
                break
            time.sleep(0.1)
    return {"status_code": r.status_code, "post_ms": post_ms, "job": job}


# ------------------------------------------------------------------ 報表

def _llm_share_excl_first(rows: list[dict]) -> float:
    """排除第一題之後，LLM 佔 p50 的百分比（第一題含建連線的一次性成本）"""
    rest = [r for r in rows[1:] if r.get("status") == 200]
    if not rest:
        return 0.0
    total = pct([r["timing"]["total_ms"] for r in rest], 0.5)
    llm = pct([r["timing"]["llm_ms"] for r in rest], 0.5)
    return llm / total * 100 if total else 0.0


def build_report(out: dict) -> str:
    L, add = [], None
    L = []
    add = L.append
    add("# Day 24 實驗結果：一次 /ask 的時間花在哪四段")
    add("")
    add(f"- 題組：{out['n']} 題（單條文 15／多條文 8／陷阱 5），k={out['k']}")
    add(f"- 服務啟動（spawn → /healthz，含建索引）：{out['boot_ms']} ms；"
        f"索引 {out['health'].get('chunks')} 塊")
    add("- 檢索與生成的邏輯未改動；變因只有「腳本呼叫」vs「服務呼叫」")
    add("")

    add("## 一、四段拆解")
    add("")
    add("中位數毫秒，以及各佔 p50 的百分比。")
    add("")
    groups = [("warm", "warm　兩個快取都開（前 23 天的樣子）"),
              ("cold", "cold　兩個快取都關（上線後的樣子）")]
    add("| 條件 | 算問題向量 | 檢索 | LLM | 序列化 | 其餘（框架） | p50 |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for key, label in groups:
        if key not in out:
            continue
        s = out[key]["summary"]
        cells = [f"{s['stages'][n]} ms" for n in STAGES]
        add(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | "
            f"{s['stages']['overhead']} ms | **{s['p50']} ms** |")
    add("")
    add("| 條件 | 算問題向量 | 檢索 | LLM | 序列化 | 其餘 |")
    add("| --- | --- | --- | --- | --- | --- |")
    for key, label in groups:
        if key not in out:
            continue
        sh = out[key]["summary"]["share"]
        add(f"| {label} | {sh['embed']}% | **{sh['retrieve']}%** | "
            f"**{sh['llm']}%** | {sh['serialize']}% | {sh['overhead']}% |")
    add("")

    add("## 二、p50 與 p95")
    add("")
    add("| 條件 | n | 成功 | p50 | p95 | 最慢 | p95/p50 |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for key, label in groups:
        if key not in out:
            continue
        s = out[key]["summary"]
        ratio = s["p95"] / s["p50"] if s["p50"] else 0
        add(f"| {label} | {s['n']} | {s['ok']} | {s['p50']} ms | {s['p95']} ms | "
            f"{s['max']} ms | {ratio:.2f}× |")
    add("")

    add("### 第一次請求另外算")
    add("")
    if "cold" in out:
        rows = out["cold"]["rows"]
        first = rows[0]["timing"]
        rest_emb = sorted(r["timing"]["embed_ms"] for r in rows[1:])
        add(f"cold 的第一題（{rows[0]['key']}）total {first['total_ms']:.0f} ms，"
            f"其中 **embed 那一段就吃掉 {first['embed_ms']:.0f} ms**；"
            f"後面 27 題的 embed 中位數是 {rest_emb[len(rest_emb) // 2]:.0f} ms。")
        add("")
        add("差的那幾秒不是在算向量，是 OpenAI client 第一次真的建 HTTPS 連線。"
            "排除第一題之後，LLM 佔 p50 的比例是 "
            f"{_llm_share_excl_first(rows):.1f}%。")
        add("")
    add("## 二之二、同一個 prompt 重打一次，答案會不會一樣")
    add("")
    if "cold" in out:
        warm_ans = {r["key"]: r.get("answer") for r in out["warm"]["rows"]}
        diff = [r["key"] for r in out["cold"]["rows"]
                if r.get("status") == 200 and r.get("answer") != warm_ans.get(r["key"])]
        add(f"`temperature=0`、prompt 逐字相同、模型相同，"
            f"cold 與 warm 的答案有 **{len(diff)}/{out['n']} 題文字不一樣**。")
        add("")
        add("不一樣的題：" + "、".join(diff))
        add("")
        add("所以「服務化沒有改答案」這句話只在快取命中的時候成立。"
            "真的重打一次，同一個問題就會拿到不同字面的答案——"
            "這是 `temperature=0` 的已知邊界，不是服務的問題，"
            "但任何拿字串比對當回歸測試的人都要先知道這件事。")
        add("")
    add("## 三、換一個 retriever")
    add("")
    if "numpy" in out:
        w, n = out["warm"], out["numpy"]
        same = sum(1 for a, b in zip(w["rows"], n["rows"])
                   if a.get("status") == 200 and b.get("status") == 200
                   and a["cited"] == b["cited"])
        same_ans = sum(1 for a, b in zip(w["rows"], n["rows"])
                       if a.get("status") == 200 and b.get("status") == 200
                       and a["answer"] == b["answer"])
        add("| retriever | 檢索那一段的中位數 | top-3 與 chroma 相同 | 答案相同 |")
        add("| --- | --- | --- | --- |")
        add(f"| chroma | {w['summary']['stages']['retrieve']} ms | — | — |")
        add(f"| numpy | {n['summary']['stages']['retrieve']} ms | "
            f"{same}/{out['n']} | {same_ans}/{out['n']} |")
    add("")

    add("## 三之二、同一支服務，兩種答法")
    add("")
    if "agent_warm" in out:
        add("`mode=pipeline` 是 Day 14 那條寫死管線，`mode=agent` 是 Day 20／21 "
            "那支自己決定查幾次的。檢索與生成的程式碼都沒有改，只是誰在呼叫它們不一樣。")
        add("")
        add("| 條件 | p50 | p95 | 算問題向量 | 檢索 | LLM | LLM 佔比 |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        rows = [("pipeline warm", "warm"), ("agent warm", "agent_warm")]
        if "cold" in out:
            rows += [("pipeline cold", "cold"), ("agent cold", "agent_cold")]
        for label, key in rows:
            if key not in out:
                continue
            s = out[key]["summary"]
            add(f"| {label} | {s['p50']} ms | {s['p95']} ms | "
                f"{s['stages']['embed']} ms | {s['stages']['retrieve']} ms | "
                f"{s['stages']['llm']} ms | **{s['share']['llm']}%** |")
        add("")
        agent_rows = [r for r in out["agent_warm"]["rows"]
                      if r.get("status") == 200 and r.get("agent")]
        searches = [r["agent"]["searches"] for r in agent_rows]
        calls = [r["agent"]["llm_calls"] for r in agent_rows]
        one = sum(1 for s in searches if s == 1)
        cited_a = [len(r.get("cited", [])) for r in agent_rows]
        cited_p = [len(r.get("cited", [])) for r in out["warm"]["rows"]
                   if r.get("status") == 200]
        add(f"- agent 一題查幾次：中位數 {pct(searches, 0.5)}、最多 {max(searches)}；"
            f"**{one}/{len(searches)} 題只查 1 次**")
        add(f"- agent 一題打幾次模型：中位數 {pct(calls, 0.5)}、最多 {max(calls)}"
            f"（pipeline 永遠是 1）")
        add(f"- 撞上 6 步上限：{sum(1 for r in agent_rows if r['agent']['hit_cap'])} 題；"
            f"重複查同一個字：{sum(r['agent']['repeats'] for r in agent_rows)} 次")
        add(f"- 回去的引用條數：agent 中位數 {pct(cited_a, 0.5)}、"
            f"pipeline 固定 {pct(cited_p, 0.5)}")
        same_ans = sum(1 for a, b in zip(out["warm"]["rows"], out["agent_warm"]["rows"])
                       if a.get("status") == 200 and b.get("status") == 200
                       and a["answer"] == b["answer"])
        add(f"- 兩種答法的答案逐字相同：{same_ans}/{out['n']} 題")
    add("")
    add("## 三之三、凍結的 agent 換一個 retriever")
    add("")
    if "agent_numpy" in out:
        a, b = out["agent_warm"], out["agent_numpy"]
        same = sum(1 for x, y in zip(a["rows"], b["rows"])
                   if x.get("status") == 200 and y.get("status") == 200
                   and x["answer"] == y["answer"])
        same_q = sum(1 for x, y in zip(a["rows"], b["rows"])
                     if x.get("status") == 200 and y.get("status") == 200
                     and (x.get("agent") or {}).get("queries")
                     == (y.get("agent") or {}).get("queries"))
        add("`agent_day21.py` 一個字都沒改，透過轉接頭跑在 numpy 上：")
        add("")
        add("| retriever | 檢索那一段 | p50 | 答案與 Chroma 相同 | 查的關鍵字相同 |")
        add("| --- | --- | --- | --- | --- |")
        add(f"| chroma | {a['summary']['stages']['retrieve']} ms | "
            f"{a['summary']['p50']} ms | — | — |")
        add(f"| numpy | {b['summary']['stages']['retrieve']} ms | "
            f"{b['summary']['p50']} ms | {same}/{out['n']} | {same_q}/{out['n']} |")
    add("")
    add("## 四、middleware 三種掛法")
    add("")
    add("| 掛法 | /ask p50 | x-total-ms 有值 | x-llm-ms 有值 | POST /documents 回應 | 攝取實際耗時 |")
    add("| --- | --- | --- | --- | --- | --- |")
    for mode, label in (("asgi", "純 ASGI"),
                        ("base", "BaseHTTPMiddleware"),
                        ("none", "不掛")):
        m = out["middleware"].get(mode)
        if not m:
            continue
        add(f"| {label} | {m['ask_p50']} ms | {m['header_total_present']}/{m['n']} | "
            f"{m['header_llm_present']}/{m['n']} | **{m['post_documents_ms']} ms** | "
            f"{m['job_elapsed_ms']} ms |")
    add("")

    add("## 四之二、追查：middleware 自己要多少錢")
    add("")
    pr = out.get("probe")
    if pr:
        add("`/ask` 量不到 middleware：它被那 468 ms 的讀檔整個蓋掉。"
            "所以改用 `/healthz`（只回一個 dict），"
            f"同一條連線連打 {pr['repeats_healthz']} 次、三種掛法交錯跑三輪。")
        add("")
        add("| 掛法 | /healthz p50 | p95 | 最快 | 比「不掛」多 |")
        add("| --- | --- | --- | --- | --- |")
        base_p50 = pr.get("none", {}).get("healthz", {}).get("p50")
        for mode in ("asgi", "base", "none"):
            r = pr.get(mode)
            if not r:
                continue
            h = r["healthz"]
            delta = ("—" if mode == "none" or not base_p50
                     else f"{h['p50'] - base_p50:+.2f} ms")
            add(f"| {r['label']} | {h['p50']} ms | {h['p95']} ms | {h['min']} ms "
                f"| {delta} |")
        add("")
        add("那 1,306 ms 到底是誰的（每一格都是「開一條新連線打一次」，三輪）：")
        add("")
        add("| 掛法 | 新連線打 POST /documents | 新連線打 /healthz（對照組） | "
            "連線重用之後的 POST |")
        add("| --- | --- | --- | --- |")
        for mode in ("asgi", "base", "none"):
            r = pr.get(mode)
            if not r:
                continue
            add(f"| {r['label']} | {r['first_post_fresh_conn']} | "
                f"**{r['fresh_conn_healthz']}** | {r['documents_rest']} |")
        add("")
    else:
        add("（尚未跑 `probe_day24.py`）")
    add("")
    add("## 五、背景攝取")
    add("")
    ing = out["ingest"]
    add(f"- `POST /documents` 回 {ing['status_code']}，耗時 **{ing['post_ms']} ms**")
    add(f"- job 狀態：{ing['job']['status']}，切出 {ing['job']['chunks']} 塊，"
        f"實際跑 {ing['job']['elapsed_ms']} ms")
    add("")

    add("## 六、錢")
    add("")
    if "cold" in out:
        ok = [r for r in out["cold"]["rows"] if r.get("status") == 200]
        emb = sum(r["usage"]["embedding_tokens"] for r in ok)
        tin = sum(r["usage"]["input_tokens"] for r in ok)
        tout = sum(r["usage"]["output_tokens"] for r in ok)
        cost = rag_core.twd(rag_core.embedding_cost_usd(emb)
                            + rag_core.chat_cost_usd(tin, tout))
        add(f"- cold 那組：{len(ok)} 題真的打了 API，embedding {emb} token、"
            f"輸入 {tin}／輸出 {tout} token，合計 **{cost:.4f} 元**")
        warm_paid = sum(1 for r in out["warm"]["rows"]
                        if r.get("status") == 200 and not r["usage"]["cached"])
        add(f"- warm 與 numpy 兩組全部命中既有快取，實際打 API：{warm_paid} 題")
        if "agent_cold" in out:
            ok = [r for r in out["agent_cold"]["rows"] if r.get("status") == 200]
            emb = sum(r["usage"]["embedding_tokens"] for r in ok)
            tin = sum(r["usage"]["input_tokens"] for r in ok)
            tout = sum(r["usage"]["output_tokens"] for r in ok)
            acost = rag_core.twd(rag_core.embedding_cost_usd(emb)
                                 + rag_core.chat_cost_usd(tin, tout))
            add(f"- agent cold：{len(ok)} 題，embedding {emb} token、"
                f"輸入 {tin}／輸出 {tout} token，合計 **{acost:.4f} 元**"
                f"（是 pipeline cold 的 {acost / cost:.1f} 倍）")
            add(f"- 今天總計 **{cost + acost:.4f} 元**")
    else:
        add("（本次以 `--no-paid` 執行，cold 那組略過）")
    return NL.join(L) + NL


if __name__ == "__main__":
    main()
