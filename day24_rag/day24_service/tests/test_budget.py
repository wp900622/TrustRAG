# -*- coding: utf-8 -*-
"""Day 28：agent 的三種上限。

全部走既有快取，不打 API。步數上限那兩個用的是 Day 28 實驗跑出來的快取
（`agent_cache_day21.json` 裡 cap=2 那一輪留下的），所以第一次跑實驗之前
它們會想打 API——那是刻意的，測試要驗的東西本來就是真的跑過的路。
"""
import pytest

import agent_day21 as agent21

from day24_service import budget

QUESTION = "病假連續請幾天以上需要附診斷證明？"
# Day 21 量到走 7 步、撞上限的那題（陷阱-102），拿它來驗步數上限
LONG_QUESTION = "公司有健身房或運動補助嗎？"


def agent_ask(client, **body):
    r = client.post("/ask", json={"question": body.pop("question", QUESTION),
                                  "mode": "agent", **body})
    assert r.status_code == 200, r.text
    return r.json()


# -------------------------------------------------------------- 不設限

def test_default_is_unlimited():
    """預設不能改變既有行為：三個欄位都沒給就是不設限。"""
    assert budget.resolve().unlimited


def test_no_limits_no_stop(client):
    body = agent_ask(client)
    assert body["agent"]["stopped_reason"] is None
    assert body["agent"]["stopped_at_step"] is None
    assert body["answer"]


# -------------------------------------------------------------- 時間上限

def test_time_cap_stops_and_returns_200(client):
    """停損不是錯誤：回的是正常的回應形狀，不是 errors.py 那套。"""
    body = agent_ask(client, max_wall_ms=0.001)
    assert body["agent"]["stopped_reason"] == "time"
    assert body["agent"]["stopped_at_step"] == 1
    assert body["answer"] is None          # 被打斷，沒有答案
    assert "error" not in body


def test_time_cap_still_reports_usage(client):
    """拿不到答案的時候，至少要拿得到帳單。"""
    body = agent_ask(client, max_wall_ms=0.001)
    assert body["usage"]["cost_twd"] == 0.0
    assert body["mode"] == "agent"
    assert body["timing"]["total_ms"] > 0


# -------------------------------------------------------------- 金額上限

def test_cost_cap_does_not_fire_on_cache_hits(client):
    """命中快取時這一題成本是 0，所以金額上限擋不下來。

    這不是 bug，是金額上限的邊界：它擋得住第一次付錢，擋不住重跑。
    寫成測試是為了讓這個性質有人看著，哪天它變了會有人知道。
    """
    body = agent_ask(client, max_twd=0.001)
    assert body["agent"]["stopped_reason"] is None
    assert body["usage"]["cost_twd"] == 0.0


def test_guard_raises_on_cost():
    tally = {"cost_twd": 0.02}
    guard = budget.Guard(budget.Budget(max_twd=0.01), tally)
    with pytest.raises(budget.BudgetExceeded) as caught:
        guard.check()
    assert caught.value.reason == "cost"
    assert caught.value.spent_twd == 0.02


# -------------------------------------------------------------- 步數上限

def test_step_cap_still_answers(client):
    """步數上限走的是凍結 agent 自己那條強制作答的路，所以仍然有答案。"""
    body = agent_ask(client, question=LONG_QUESTION, max_steps=2)
    assert body["agent"]["stopped_reason"] == "steps"
    assert body["agent"]["hit_cap"] is True
    assert body["answer"]


def test_step_cap_does_not_leak(client):
    """改的是模組常數，離開請求要還原，不然下一個人拿到別人的上限。"""
    agent_ask(client, question=LONG_QUESTION, max_steps=2)
    assert agent21.MAX_STEPS == 6
    body = agent_ask(client)
    assert body["agent"]["stopped_reason"] is None


# -------------------------------------------------------------- 兩邊取緊

def test_service_ceiling_wins_when_tighter():
    assert budget._tighter(8, 3) == 3
    assert budget._tighter(2, 3) == 2
    assert budget._tighter(None, 3) == 3
    assert budget._tighter(5, 0) == 5       # 服務沒設＝0
    assert budget._tighter(None, 0) is None


def test_contract_rejects_absurd_limits(client):
    r = client.post("/ask", json={"question": QUESTION, "mode": "agent",
                                  "max_steps": 99})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"
