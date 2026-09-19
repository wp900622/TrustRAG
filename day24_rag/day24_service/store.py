# -*- coding: utf-8 -*-
"""攝取 job 的狀態存到 SQLite，不再放行程記憶體。

放記憶體的版本有一個很具體的壞處：`uvicorn --workers 4` 一下去，
A 收下的 `job_id`，使用者去查的時候可能打到 B，而 B 上面沒有這筆，回 404。
使用者看到的是「我剛剛上傳的東西不見了」。

SQLite 夠用，而且不必多跑一個服務：WAL 模式下多個行程可以同時讀，
寫是序列化的，而我們的寫非常少（一個 job 只寫三次：queued → running → done）。
真的要橫向擴到多台機器時再換 Postgres，那時候要改的只有這個檔案。
"""
import contextlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

_LOCAL = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id     TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    path       TEXT NOT NULL,
    ocr        INTEGER NOT NULL DEFAULT 0,
    chunks     INTEGER NOT NULL DEFAULT 0,
    elapsed_ms REAL NOT NULL DEFAULT 0,
    error      TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_created_at ON jobs (created_at);
"""


class JobStore:
    """一個 job 的存放處。每條執行緒各自一個連線（sqlite3 連線不跨執行緒）。"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(_LOCAL, "conn", None)
        if conn is None or getattr(_LOCAL, "path", None) != str(self.db_path):
            conn = sqlite3.connect(self.db_path, timeout=10,
                                   isolation_level=None)
            conn.row_factory = sqlite3.Row
            # WAL：讀不會擋寫，多個 worker 才共存得了
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            _LOCAL.conn, _LOCAL.path = conn, str(self.db_path)
        return conn

    # ------------------------------------------------------------------ 寫

    def create(self, path: str, ocr: bool) -> dict:
        job_id = uuid.uuid4().hex[:12]
        self._connect().execute(
            "INSERT INTO jobs (job_id, status, path, ocr, created_at) "
            "VALUES (?, 'queued', ?, ?, ?)",
            (job_id, path, int(ocr), time.time()))
        return self.get(job_id)

    def update(self, job_id: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._connect().execute(
            f"UPDATE jobs SET {cols} WHERE job_id = ?",
            (*fields.values(), job_id))

    def purge(self, older_than_seconds: float) -> int:
        cur = self._connect().execute(
            "DELETE FROM jobs WHERE created_at < ?",
            (time.time() - older_than_seconds,))
        return cur.rowcount or 0

    # ------------------------------------------------------------------ 讀

    def get(self, job_id: str) -> dict | None:
        row = self._connect().execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def recent(self, limit: int = 20) -> list[dict]:
        rows = self._connect().execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,)).fetchall()
        return [dict(r) for r in rows]

    @contextlib.contextmanager
    def exclusive(self, timeout: float = 60.0):
        """跨行程的互斥鎖，借 SQLite 的 BEGIN IMMEDIATE 來做。

        為什麼需要：Chroma 內嵌模式不是設計給兩個行程同時寫的。
        開兩個 worker 各自攝取的時候，一邊把 collection 砍掉重建，
        另一邊手上的 handle 就失效了。與其祈禱不要撞到，不如讓它們排隊。

        不另外跑一個 Redis，是因為這裡本來就有一顆 SQLite，而攝取很少發生。
        """
        conn = self._connect()
        deadline = time.time() + timeout
        while True:
            try:
                conn.execute("BEGIN IMMEDIATE")
                break
            except sqlite3.OperationalError:
                if time.time() > deadline:
                    raise
                time.sleep(0.05)
        try:
            yield
        finally:
            conn.execute("COMMIT")

    def close(self) -> None:
        conn = getattr(_LOCAL, "conn", None)
        if conn is not None:
            conn.close()
            _LOCAL.conn = None


def as_contract(job: dict) -> dict:
    """資料庫的列 → 回應契約要的形狀（去掉 ocr／created_at 這種內部欄位）"""
    return {"job_id": job["job_id"], "status": job["status"],
            "path": job["path"], "chunks": job["chunks"],
            "elapsed_ms": job["elapsed_ms"], "error": job["error"]}


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False)
