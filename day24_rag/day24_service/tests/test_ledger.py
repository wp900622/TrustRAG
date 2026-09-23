# -*- coding: utf-8 -*-
"""Day 27 的帳本。

要擋的四件事，每一件都是 Day 27 考古時被咬過的：

1. 命中要能定價——那正是重建 26 天帳單時做不到的事
2. 沒被記過單價的命中要**單獨列**，不准用平均值偷偷補掉
3. 帳本壞掉不能讓服務起不來，也不能讓問答失敗
4. `/ask` 的回應要帶得出這一次的金額
"""
import json

import pytest

from day24_service import ledger
from day24_service.config import settings


@pytest.fixture(autouse=True)
def clean():
    ledger.reset()
    yield
    ledger.reset()


def _messages(q: str) -> list[dict]:
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": q}]


def test_hit_gets_priced_from_the_paid_call():
    """命中的價錢來自同一把 key 上次真的付過的那次。"""
    key = ledger.chat_key(_messages("特休假最晚幾天前申請？"))
    paid = ledger.record(key, "gpt-4o-mini", in_tok=450, out_tok=35)
    assert paid == pytest.approx((450 * 0.15 + 35 * 0.60) / 1e6 * 31)

    avoided = ledger.hit(key)
    assert avoided == pytest.approx(paid)

    s = ledger.summary()
    assert (s["paid_calls"], s["hits"], s["unpriced_hits"]) == (1, 1, 0)
    assert s["hit_rate"] == 0.5


def test_unpriced_hit_is_reported_not_guessed():
    """Day 27 之前買的那 1,298 筆沒有金額。命中它們要老實回 None。"""
    assert ledger.hit("這個key沒被記過") is None
    s = ledger.summary()
    assert s["unpriced_hits"] == 1
    assert s["priced_hits"] == 0
    assert s["avoided_twd"] == 0.0        # 不准用平均值補上去


def test_repaid_key_uses_the_latest_amount():
    """use_cache=false 重買過，下次命中省下的是最近一次的金額。"""
    key = ledger.chat_key(_messages("婚假幾天？"))
    ledger.record(key, "gpt-4o-mini", 450, 35)
    second = ledger.record(key, "gpt-4o-mini", 900, 70)
    assert ledger.hit(key) == pytest.approx(second)


def test_abandoned_uses_last_known_then_average():
    """客戶端走掉時拿不到 usage，用同一把 key 上次的金額；沒有就用平均。"""
    key = ledger.chat_key(_messages("事假一年幾天？"))
    known = ledger.record(key, "gpt-4o-mini", 450, 35)
    assert ledger.abandoned(key) == pytest.approx(known)
    # 沒見過的 key 退回平均，而且記在自己那一欄，不混進 spent_twd
    assert ledger.abandoned("沒見過") == pytest.approx(known)
    # summary() 是報表，數字在那裡四捨五入到小數第四位
    s = ledger.summary()
    assert s["abandoned_twd"] == round(known * 2, 4)
    assert s["spent_twd"] == round(known, 4)


def test_unknown_model_is_not_priced():
    """認不得的模型寧可記 0，不要記一個猜的數字。"""
    assert ledger.cost_twd("some-future-model", 1000, 100) == 0.0


def test_broken_file_does_not_block_startup():
    settings.ledger_path.write_text("{ 這不是 json", encoding="utf-8")
    try:
        assert ledger.load() == 0          # 從空的開始，不是拋例外
    finally:
        settings.ledger_path.unlink(missing_ok=True)


def test_flush_then_load_round_trips():
    key = ledger.chat_key(_messages("加班上限幾小時？"))
    ledger.record(key, "gpt-4o-mini", 450, 35)
    ledger.flush()
    raw = json.loads(settings.ledger_path.read_text(encoding="utf-8"))
    assert raw["entries"][key]["in"] == 450

    ledger.reset()
    assert ledger.load() == 1
    assert ledger.hit(key) is not None     # 重啟之後還認得這筆
    settings.ledger_path.unlink(missing_ok=True)


def test_ask_response_carries_the_cost(client):
    """實際打一次 /ask。這 28 題的答案早就在快取裡，所以不會真的花錢。"""
    r = client.post("/ask", json={"question": "婚假總共有幾天？", "k": 3})
    assert r.status_code == 200
    usage = r.json()["usage"]
    assert "cost_twd" in usage and "avoided_twd" in usage
    if usage["cached"]:
        assert usage["cost_twd"] == 0.0
    else:
        assert usage["cost_twd"] > 0

    s = client.get("/usage").json()
    assert s["calls"] >= 1
    assert 0.0 <= s["hit_rate"] <= 1.0


def test_estimated_spend_is_tracked_separately():
    """agent 那條路的 token 是離線重數的，記，但不混進 spent_twd。"""
    api = ledger.chat_key(_messages("API 回報的那種"))
    ledger.record(api, "gpt-4o-mini", 450, 35)
    est = "agent-key-0001"
    ledger.record(est, "gpt-4o-mini", 1030, 60, estimated=True)

    s = ledger.summary()
    assert s["paid_calls"] == 2 and s["est_calls"] == 1
    assert s["spent_twd"] > 0 and s["spent_est_twd"] > 0
    # 兩個數字不准互相污染
    assert s["spent_twd"] == round(ledger.cost_twd("gpt-4o-mini", 450, 35), 4)

    # 命中一筆離線重數的，省下的錢也記在估計那一欄
    ledger.hit(est)
    s = ledger.summary()
    assert s["avoided_est_twd"] > 0
    assert s["avoided_twd"] == 0.0


def test_agent_request_is_accounted(client):
    """agent 一題會打好幾次模型，回應裡的金額要是這一題加總的。"""
    before = client.get("/usage").json()
    r = client.post("/ask", json={"question": "婚假總共有幾天？", "k": 3,
                                  "mode": "agent"})
    assert r.status_code == 200
    body = r.json()
    assert body["agent"]["llm_calls"] >= 1
    usage = body["usage"]
    # 這一題不管命中幾次、付錢幾次，兩個金額欄位都要說得出話
    assert usage["cost_twd"] >= 0
    assert usage["avoided_twd"] is None or usage["avoided_twd"] >= 0

    after = client.get("/usage").json()
    # agent 的每一次內部呼叫都進了帳本，不是整題算一筆
    assert after["calls"] - before["calls"] >= body["agent"]["llm_calls"]
