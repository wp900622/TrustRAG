# -*- coding: utf-8 -*-
"""記帳：把「這次呼叫花了多少」跟快取的 key 存在一起。

Day 27 花了一整天重建前 26 天的帳單，原因只有一個：
`rag_core.chat()` 命中快取時回 `(text, 0, 0)`，而快取的 key 是
`sha256([model, messages])[:16]`，只存答案不存 prompt。
結果是「買到的東西」留著，「收據」沒留，1,298 筆裡只認得出 686 筆。

這個檔案是那件事的解藥，兩行就講得完：

    命中的時候服務不知道這次值多少——但上一次付錢的時候知道。
    用同一把 key 把金額記起來，命中就查得到。

**不改 `rag_core.py`**（它是前 23 天每一篇的證據），也不改快取的值的格式
（動了它既有的 1,298 筆會整批失效）。帳記在旁邊一份 side-car 裡，
key 跟快取共用。

三件以前做不到、現在做得到的事：

    1. 每次回應帶這一題的成本，命中的那些帶「省下多少」
    2. 命中率是一個數字，不是一個推測
    3. Day 26 那個「客戶端走掉、模型照生、錢照付」的請求，值多少錢講得出來

**帳本裡有兩種東西必須分開列。**

第一種是 `unpriced_hits`。
那是命中了、但這一筆的單價從來沒被記過的次數——也就是 Day 27 之前買的那些。
它會隨著時間自己歸零，而在它歸零之前，「省了多少」這個數字是不完整的。
寧可讓它露在報表上，也不要偷偷用平均值把它補掉。

第二種是 `spent_est_twd`：agent 那條路。它一題會打好幾次模型，
而那幾次走的是 `agent_day21.chat_tools()`——凍結的檔案，它把 API 回報的
usage 丟掉了，只回訊息本身。所以那些 token 是離線重數的。

**但「數不到收據」不是「不記帳」的理由。** agent 是這支服務最貴的一條路
（Day 21 量過：p95 是寫死管線的 6.1 倍、錢是 5.0 倍），
一份把最貴的路排除在外的帳本，剛好在最需要它的地方失明。
所以照記，只是跟 API 回報的那一半分開累計。
"""
import json
import logging
import os
import threading
from pathlib import Path

import rag_core

from .config import settings

log = logging.getLogger("day24")

# 每百萬 token 的美元單價。mini 這兩個直接取 rag_core，
# 換模型的時候不要有第二個地方要改
PRICES = {
    rag_core.CHAT_MODEL: (rag_core.USD_PER_MILLION_CHAT_INPUT,
                          rag_core.USD_PER_MILLION_CHAT_OUTPUT),
    "gpt-4o": (2.50, 10.00),
}
USD_PER_M_EMBEDDING = rag_core.USD_PER_MILLION_EMBEDDING

_LOCK = threading.RLock()
_ENTRIES: dict[str, dict] = {}      # key -> {model, in, out, paid, hits, est}
_TOTALS = {"paid_calls": 0, "hits": 0, "unpriced_hits": 0,
           "spent_twd": 0.0, "avoided_twd": 0.0, "abandoned_twd": 0.0,
           # agent 那條路的 token 是離線重數的，不是 API 回報的。
           # 記，但跟 API 回報的分開列——混在一起，spent_twd 就不能用了
           "spent_est_twd": 0.0, "avoided_est_twd": 0.0, "est_calls": 0}
_DIRTY = 0


def path() -> Path:
    return Path(settings.ledger_path)


def cost_twd(model: str, in_tok: int, out_tok: int, emb_tok: int = 0) -> float:
    """一次呼叫多少錢。認不得的模型當成 0 並留一行 log——
    帳寧可少記也不要記一個猜的數字上去。"""
    price = PRICES.get(model)
    if price is None:
        log.warning("ledger 不認得這個模型，這筆不計價: %s", model)
        return 0.0
    usd = (in_tok / 1e6 * price[0] + out_tok / 1e6 * price[1]
           + emb_tok / 1e6 * USD_PER_M_EMBEDDING)
    return usd * rag_core.USD_TO_TWD


# ------------------------------------------------------------------ 記一筆

def record(key: str, model: str, in_tok: int, out_tok: int,
           emb_tok: int = 0, estimated: bool = False) -> float:
    """真的打了 API 的那一次。回傳這次花了多少台幣。

    `estimated=True` 用在 agent 那條路：`chat_tools()` 是凍結的檔案裡的函式，
    它把 API 回報的 usage 丟掉了，只回訊息本身。所以 token 是照 Day 21
    那支的算法離線重數的（`count_messages` ＋ `TOOLS_TOKEN_ESTIMATE`）。
    那個數字可信（Day 27 拿 158 筆驗過離線計數與 API 完全一致），
    但它終究不是收據，所以獨立累計。
    """
    twd = cost_twd(model, in_tok, out_tok, emb_tok)
    with _LOCK:
        entry = _ENTRIES.setdefault(
            key, {"model": model, "in": 0, "out": 0, "emb": 0,
                  "paid": 0, "hits": 0, "est": estimated})
        # 同一個 key 重新付費（use_cache=false）時以最後一次為準：
        # 要的是「下次命中省下多少」，那當然是最近一次的真實金額
        entry.update(model=model, est=estimated,
                     **{"in": in_tok, "out": out_tok, "emb": emb_tok})
        entry["paid"] += 1
        _TOTALS["paid_calls"] += 1
        _TOTALS["spent_est_twd" if estimated else "spent_twd"] += twd
        if estimated:
            _TOTALS["est_calls"] += 1
        _mark_dirty()
    return twd


def hit(key: str) -> float | None:
    """快取擋下來的那一次。回傳省下多少台幣；這筆沒被記過單價就回 None。"""
    with _LOCK:
        entry = _ENTRIES.get(key)
        _TOTALS["hits"] += 1
        if entry is None:
            _TOTALS["unpriced_hits"] += 1
            _mark_dirty()
            return None
        entry["hits"] += 1
        twd = cost_twd(entry["model"], entry["in"], entry["out"], entry["emb"])
        _TOTALS["avoided_est_twd" if entry.get("est") else "avoided_twd"] += twd
        _mark_dirty()
        return twd


def abandoned(key: str) -> float:
    """客戶端中途走掉。模型會把整段生完，錢照付，而沒有人在聽。

    這次的 usage 拿不到（產生器被取消了），所以用同一個 key 上次付過的金額，
    沒有就用全體平均。這是估計值，單獨記一欄，不要混進 spent_twd。
    """
    twd = estimate(key)
    with _LOCK:
        _TOTALS["abandoned_twd"] += twd
        _mark_dirty()
    return twd


def estimate(key: str | None = None) -> float:
    """這個 key 大概值多少。沒記錄就用已付費呼叫的平均。"""
    with _LOCK:
        entry = _ENTRIES.get(key) if key else None
        if entry is not None:
            return cost_twd(entry["model"], entry["in"], entry["out"], entry["emb"])
        paid = _TOTALS["paid_calls"]
        return (_TOTALS["spent_twd"] / paid) if paid else 0.0


def chat_key(messages: list[dict]) -> str:
    """跟快取共用同一把 key。這一行就是整個檔案成立的理由。"""
    return rag_core._chat_cache_key(messages)


# -------------------------------------------------------------------- 報表

def summary() -> dict:
    with _LOCK:
        total = _TOTALS["paid_calls"] + _TOTALS["hits"]
        priced = _TOTALS["hits"] - _TOTALS["unpriced_hits"]
        return {
            "calls": total,
            "paid_calls": _TOTALS["paid_calls"],
            "hits": _TOTALS["hits"],
            "hit_rate": round(_TOTALS["hits"] / total, 4) if total else 0.0,
            # 命中了、但這筆的單價是 Day 27 之前買的，沒人記過。
            # 這個數字不歸零，「省了多少」就還是不完整的
            "unpriced_hits": _TOTALS["unpriced_hits"],
            "priced_hits": priced,
            "spent_twd": round(_TOTALS["spent_twd"], 4),
            "avoided_twd": round(_TOTALS["avoided_twd"], 4),
            "abandoned_twd": round(_TOTALS["abandoned_twd"], 4),
            # agent：離線重數的那一半，分開列。加總要自己加，
            # 因為「API 說的」跟「我自己數的」不該長成同一個數字
            "spent_est_twd": round(_TOTALS["spent_est_twd"], 4),
            "avoided_est_twd": round(_TOTALS["avoided_est_twd"], 4),
            "est_calls": _TOTALS["est_calls"],
            "entries": len(_ENTRIES),
        }


# ------------------------------------------------------------------ 持久化

def _mark_dirty() -> None:
    """呼叫者已經拿著 _LOCK。累積到一定筆數才落地。

    付費那筆很稀有（真的打 API），命中那筆可能每秒好幾次，
    每次都寫檔就變成拿 I/O 換記帳。掉幾筆命中計數不會讓任何決定變壞。
    """
    global _DIRTY
    _DIRTY += 1
    if _DIRTY >= settings.ledger_flush_every:
        _flush_locked()


def flush() -> None:
    with _LOCK:
        _flush_locked()


def _flush_locked() -> None:
    global _DIRTY
    _DIRTY = 0
    p = path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
        tmp.write_text(json.dumps({"entries": _ENTRIES, "totals": _TOTALS},
                                  ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)              # 原子換檔，讀的人不會看到中間狀態
    except Exception as exc:            # noqa: BLE001
        # 記帳失敗不該讓問答失敗。這是帳本，不是主線
        log.warning("ledger 寫檔失敗，這次不落地: %s", exc)


def load() -> int:
    """啟動時讀回來。檔案壞掉就從空的開始，不要讓服務起不來。"""
    p = path()
    if not p.exists():
        return 0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:            # noqa: BLE001
        log.warning("ledger 讀檔失敗，從空的開始: %s", exc)
        return 0
    with _LOCK:
        _ENTRIES.clear()
        _ENTRIES.update(data.get("entries", {}))
        _TOTALS.update(data.get("totals", {}))
    return len(_ENTRIES)


def reset() -> None:
    """測試用。"""
    global _DIRTY
    with _LOCK:
        _ENTRIES.clear()
        _TOTALS.update({"paid_calls": 0, "hits": 0, "unpriced_hits": 0,
                        "spent_twd": 0.0, "avoided_twd": 0.0,
                        "abandoned_twd": 0.0, "spent_est_twd": 0.0,
                        "avoided_est_twd": 0.0, "est_calls": 0})
        _DIRTY = 0
