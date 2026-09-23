# -*- coding: utf-8 -*-
"""agent 的停損：步數、金額、牆上時間三種上限。

`/ask?mode=agent` 從 Day 24 接進來到現在，呼叫端只能決定「要不要走 agent」，
決定不了「它可以走多遠」。走多遠是寫在 `agent_day21.MAX_STEPS` 裡的模組常數，
外面看不到也改不了，而 Day 21 量到的形狀是這樣的：

    2 步就交卷 19 題 ／ 3 步 4 題 ／ 5 步 1 題 ／ 6 步 1 題 ／ 7 步 3 題

28 題裡走 5 步以上的只有 5 題，那 5 題吃掉 61% 的帳單。
一支沒有上限的 agent 端點，平均值很好看，尾巴沒人管。

## 三種上限不是同一件事

這個檔案最重要的一句話：**停下來的方式決定了呼叫端拿到什麼。**

步數上限走的是凍結 agent 自己的路。`run_agent()` 裡本來就有一段
「走到第 MAX_STEPS 步還沒交卷就把 CAP_NUDGE 推進去，要它直接作答」，
所以把上限調小，等於提早對它說「請你現在交卷」——呼叫端拿得到答案。

金額與時間上限沒有這條路。它們是在下一次模型呼叫之前結算，超過就拋例外，
等於當場打斷。呼叫端拿不到答案，只拿得到這一題已經花掉多少、查到了哪幾條。

兩種都要，因為它們擋的是不同的東西：步數擋「繞太多圈」，
另外兩個擋「單題失控」——查詢一直變形、模型一直不交卷、上游卡住不回。

## 一個誠實的邊界

金額與時間是**累計到目前為止**的值，在下一次呼叫之前檢查。
所以實際花費會超過上限，最多超過一次呼叫的量——
要做到分毫不差，得在呼叫之前就知道那次要花多少，而那件事沒有人做得到。

而且 agent 那條路的 token 是離線重數的（`agents._agent_tokens()` 的算法
逐字照抄 Day 21），所以金額上限擋的是估計金額，不是 API 回報的金額。

## 預設一律不設限

三個欄位都留 `None`，行為跟 Day 24 接進來那天完全一樣。
上限是呼叫端或部署時選擇打開的東西，不是這個檔案偷偷改掉的行為。
服務端設了硬上限的話，兩邊取小的那個——呼叫端不能把自己的上限
設到比服務的硬上限還寬。
"""
import contextlib
import time
from dataclasses import dataclass

import agent_day21 as agent21

from .config import settings


class BudgetExceeded(Exception):
    """金額或時間用完了。帶著停在哪、已經花了多少一起往上拋。

    刻意不繼承 `ServiceError`：停損不是錯誤。handler 會把它接住，
    回一個帶著 `stopped_reason` 的正常回應，而不是一個錯誤形狀。
    """

    def __init__(self, reason: str, at_step: int, spent_twd: float,
                 elapsed_ms: float):
        super().__init__(f"agent 停損：{reason}")
        self.reason = reason
        self.at_step = at_step
        self.spent_twd = spent_twd
        self.elapsed_ms = elapsed_ms


@dataclass(frozen=True)
class Budget:
    """一次 agent 請求的三個上限。全 None ＝ 不設限。"""
    max_steps: int | None = None
    max_twd: float | None = None
    max_wall_ms: float | None = None

    @property
    def unlimited(self) -> bool:
        return (self.max_steps is None and self.max_twd is None
                and self.max_wall_ms is None)

    def describe(self) -> dict:
        return {"max_steps": self.max_steps, "max_twd": self.max_twd,
                "max_wall_ms": self.max_wall_ms}


def _tighter(asked, ceiling):
    """呼叫端要的跟服務允許的，取緊的那個。服務沒設就用呼叫端的。"""
    if ceiling in (None, 0):
        return asked
    if asked in (None, 0):
        return ceiling
    return min(asked, ceiling)


def resolve(max_steps=None, max_twd=None, max_wall_ms=None) -> Budget:
    """把請求裡的三個欄位跟 `config.py` 的硬上限併起來。

    硬上限用環境變數設（`DAY24_AGENT_MAX_STEPS` 等三個），預設都是 0＝不設限，
    所以不設任何環境變數時，這支服務的行為與 Day 24 以來完全相同。
    """
    return Budget(
        max_steps=_tighter(max_steps, settings.agent_max_steps),
        max_twd=_tighter(max_twd, settings.agent_max_twd),
        max_wall_ms=_tighter(max_wall_ms, settings.agent_max_wall_ms))


class Guard:
    """跟著一題跑的碼表與計價器。

    `check()` 掛在每次模型呼叫之前。它只讀 `tally`（`agents.run()` 那份
    逐次加總的帳）與自己的碼表，不碰凍結 agent 的任何狀態。
    """

    def __init__(self, budget: Budget, tally: dict):
        self.budget = budget
        self._tally = tally
        self._t0 = time.perf_counter()
        self.calls = 0

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    @property
    def spent_twd(self) -> float:
        return float(self._tally.get("cost_twd", 0.0))

    def check(self) -> None:
        """下一次模型呼叫之前。超過就拋，沒超過就放行。"""
        self.calls += 1
        if self.budget.max_twd is not None \
                and self.spent_twd >= self.budget.max_twd:
            raise BudgetExceeded("cost", self.calls, self.spent_twd,
                                 self.elapsed_ms)
        if self.budget.max_wall_ms is not None \
                and self.elapsed_ms >= self.budget.max_wall_ms:
            raise BudgetExceeded("time", self.calls, self.spent_twd,
                                 self.elapsed_ms)


@contextlib.contextmanager
def step_cap(max_steps: int | None):
    """暫時調小凍結 agent 的步數上限，離開時還原。

    跟 `agents.run()` 換 `SEARCH_K` 是同一個做法，理由也一樣：
    `agent_day21.py` 是 Day 21 那篇的證據，改一個字那天的數字就不可重現。

    **這是行程層級的**，跟計時與記帳的包法一樣，所以 agent 那條路的信號量
    （`DAY24_AGENT_CONCURRENCY`，預設 1）同時也在保護這個常數。
    第二個請求要嘛排隊、要嘛會拿到別人的上限。
    """
    if max_steps is None:
        yield
        return
    original = agent21.MAX_STEPS
    agent21.MAX_STEPS = max(1, int(max_steps))
    try:
        yield
    finally:
        agent21.MAX_STEPS = original
