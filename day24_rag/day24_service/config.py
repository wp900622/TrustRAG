# -*- coding: utf-8 -*-
"""服務的設定，一個地方。

前 23 天這些值散在各支腳本的模組層級常數裡，那沒問題，因為每支腳本
就是一次執行。服務不一樣：同一份程式要在我的筆電、在 CI、在正式環境
用不同的值跑起來，而改法不能是「去改原始碼」。

全部可以用環境變數覆蓋，前綴 `DAY24_`。沒設就是這裡的預設，
而預設值刻意跟前 23 天的腳本一樣，所以不設任何環境變數時，
這支服務的行為與 Day 14 以來完全相同。
"""
import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    # 檢索
    default_retriever: str = os.getenv("DAY24_RETRIEVER", "chroma")
    default_k: int = _int("DAY24_K", 3)
    max_k: int = _int("DAY24_MAX_K", 15)

    # 攝取
    documents_root: Path = Path(os.getenv("DAY24_DOCUMENTS_ROOT", str(BASE_DIR)))
    job_db: Path = Path(os.getenv("DAY24_JOB_DB", str(BASE_DIR / "day24_jobs.sqlite3")))
    job_ttl_seconds: int = _int("DAY24_JOB_TTL", 7 * 24 * 3600)

    # 快取：整包讀寫的檔案快取在並發下會壞，服務啟動時換掉它
    resident_cache: bool = _bool("DAY24_RESIDENT_CACHE", True)
    # 設了就走 Redis（多個 worker 共用一份），沒設走行程內的常駐記憶體。
    # 連不上會退回 memory 並留一行 warning，不會讓服務起不來。
    redis_url: str = os.getenv("DAY24_REDIS_URL", "")
    redis_prefix: str = os.getenv("DAY24_REDIS_PREFIX", "day24:")
    redis_timeout: float = float(os.getenv("DAY24_REDIS_TIMEOUT", "2"))

    # 觀測
    middleware: str = os.getenv("DAY24_MIDDLEWARE", "asgi").lower()
    # 記帳：把每次呼叫的金額跟快取的 key 存在一起（Day 27）。
    # flush_every 是「累積幾筆才落地」，命中可能每秒好幾次，
    # 每次都寫檔就變成拿 I/O 換記帳
    ledger_path: Path = Path(os.getenv("DAY24_LEDGER",
                                       str(BASE_DIR / "usage_ledger.json")))
    ledger_flush_every: int = _int("DAY24_LEDGER_FLUSH_EVERY", 20)

    # agent
    agent_max_concurrency: int = _int("DAY24_AGENT_CONCURRENCY", 1)
    # 停損（Day 28）。三個都是 0＝不設限，所以不設環境變數時行為與 Day 24 相同。
    # 呼叫端也可以在請求裡自己設，兩邊取緊的那個（budget.resolve）
    agent_max_steps: int = _int("DAY24_AGENT_MAX_STEPS", 0)
    agent_max_twd: float = _float("DAY24_AGENT_MAX_TWD", 0.0)
    agent_max_wall_ms: float = _float("DAY24_AGENT_MAX_WALL_MS", 0.0)

    def resolve_document(self, raw: str) -> Path:
        """把使用者給的路徑解到 documents_root 底下，並擋掉跳出去的寫法。

        `../../etc/passwd` 這種東西不該因為「反正是內部服務」就放過去。
        """
        root = self.documents_root.resolve()
        target = (root / raw).resolve() if not Path(raw).is_absolute() \
            else Path(raw).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"路徑超出 documents_root：{raw}")
        return target


settings = Settings()
