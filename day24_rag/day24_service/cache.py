# -*- coding: utf-8 -*-
"""把腳本時代的檔案快取換成服務能用的版本。

`rag_core` 的快取是「整包讀進來、改一個 key、整包寫回去」。
單執行緒的腳本跑了 23 天沒出過事，但服務有兩條執行緒會同時走到那一行：

- 兩邊都拿著舊的那包，後寫的把先寫的蓋掉 → 靜靜掉資料
- `write_text()` 會先把檔案截成 0 再寫，另一條執行緒剛好在那個瞬間讀，
  會拿到半個檔案（UnicodeDecodeError）或空檔（JSONDecodeError）→ 整個請求 500

**不改 `rag_core.py`**（它是前 23 天每一篇的證據），改在啟動時把
`_load_json` / `_save_json` 換掉。`rag_core` 對快取只做四件事：

    key not in cache        # __contains__
    cache[key]              # __getitem__
    cache[key] = value      # __setitem__
    _save_json(path, cache) # 整包寫回

所以只要回傳一個行為像 dict 的東西就行，底下換成什麼都可以。

兩個後端：

    memory   常駐記憶體 ＋ 一把鎖 ＋ 原子換檔（預設）
    redis    設了 DAY24_REDIS_URL 就走這個

為什麼需要 redis：memory 那個是**每個 worker 各一份**。
`--workers 3` 的時候，A 算過的向量 B 不知道，同一個問題最多付三次錢。
Redis 一份共用，這件事就消失了。
"""
import json
import logging
import os
import threading
from collections.abc import MutableMapping
from pathlib import Path

import numpy as np

import rag_core

from .config import settings

log = logging.getLogger("day24")

_LOCK = threading.RLock()
_MEMORY: dict[str, dict] = {}
_ORIGINAL: dict[str, object] = {}
_BACKEND = "memory"
_CLIENT = None


# ------------------------------------------------------------ memory 後端

def _memory_load(path: Path) -> dict:
    key = str(path)
    with _LOCK:
        if key not in _MEMORY:
            _MEMORY[key] = (json.loads(path.read_text(encoding="utf-8"))
                            if path.exists() else {})
        return _MEMORY[key]


def _memory_save(path: Path, data: dict) -> None:
    key = str(path)
    with _LOCK:
        current = _MEMORY.setdefault(key, {})
        if current is not data:
            current.update(data)        # 不要整包覆蓋掉別人剛寫的
        tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
        tmp.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)           # 原子換檔，讀的人不會看到中間狀態


# ------------------------------------------------------------- redis 後端

class RedisDict(MutableMapping):
    """行為像 dict，資料在 Redis。

    兩種值的編碼不一樣，所以 codec 由建構時決定：

        vector  1536 個 float32 直接存 bytes（6 KB）。存 JSON 會膨脹到 20 KB 以上
        text    生成的答案，存 UTF-8 bytes

    **讀寫失敗一律當成沒命中**，不要讓快取把問答整條路拖死。
    Redis 掛了，服務應該變慢變貴，不是變 500。
    """

    def __init__(self, client, prefix: str, codec: str):
        self._r = client
        self._prefix = prefix
        self._codec = codec

    # -- 編碼 ---------------------------------------------------------

    def _encode(self, value):
        if self._codec == "vector":
            return np.asarray(value, dtype=np.float32).tobytes()
        return str(value).encode("utf-8")

    def _decode(self, raw: bytes):
        if self._codec == "vector":
            return np.frombuffer(raw, dtype=np.float32).tolist()
        return raw.decode("utf-8")

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    # -- MutableMapping ------------------------------------------------

    def __contains__(self, key) -> bool:
        try:
            return bool(self._r.exists(self._key(key)))
        except Exception as exc:                     # noqa: BLE001
            log.warning("redis exists failed, 當成沒命中: %s", exc)
            return False

    def __getitem__(self, key):
        try:
            raw = self._r.get(self._key(key))
        except Exception as exc:                     # noqa: BLE001
            log.warning("redis get failed: %s", exc)
            raise KeyError(key) from exc
        if raw is None:
            raise KeyError(key)
        return self._decode(raw)

    def __setitem__(self, key, value) -> None:
        try:
            self._r.set(self._key(key), self._encode(value))
        except Exception as exc:                     # noqa: BLE001
            log.warning("redis set failed, 這筆不進快取: %s", exc)

    def __delitem__(self, key) -> None:
        try:
            self._r.delete(self._key(key))
        except Exception as exc:                     # noqa: BLE001
            log.warning("redis delete failed: %s", exc)

    def __iter__(self):
        """走訪用 SCAN。**不要在請求路徑上用它**，只有 seed 與統計會呼叫。"""
        cut = len(self._prefix)
        for raw in self._r.scan_iter(match=f"{self._prefix}*", count=500):
            yield raw.decode("utf-8")[cut:]

    def __len__(self) -> int:
        try:
            return sum(1 for _ in self)
        except Exception as exc:                     # noqa: BLE001
            log.warning("redis scan failed: %s", exc)
            return 0

    # -- seed ----------------------------------------------------------

    def seed(self, data: dict) -> int:
        """把檔案裡既有的內容灌進去，已經存在的不覆蓋。

        為什麼要做：那兩個快取檔是進版控的，讀者 clone 下來重跑要拿到
        跟文章一樣的數字。換成 Redis 之後那個保證不能斷，所以啟動時
        用 SETNX 把檔案當種子灌一次。
        """
        written = 0
        try:
            pipe = self._r.pipeline(transaction=False)
            for key, value in data.items():
                pipe.set(self._key(key), self._encode(value), nx=True)
            written = sum(1 for ok in pipe.execute() if ok)
        except Exception as exc:                     # noqa: BLE001
            log.warning("redis seed failed: %s", exc)
        return written


def _codec_for(path: Path) -> str:
    """embedding 那個檔存向量，其餘存文字"""
    return "vector" if Path(path) == Path(rag_core.CACHE_PATH) else "text"


def _redis_load(path: Path) -> RedisDict:
    key = str(path)
    with _LOCK:
        if key not in _MEMORY:
            _MEMORY[key] = RedisDict(
                _CLIENT, f"{settings.redis_prefix}{Path(path).stem}:",
                _codec_for(path))
        return _MEMORY[key]


def _redis_save(path: Path, data) -> None:
    """`__setitem__` 已經寫進去了，這裡不用做事。"""


# ------------------------------------------------------------------ 安裝

def _connect():
    import redis                                     # 只有要用才 import

    client = redis.from_url(
        settings.redis_url, decode_responses=False,
        socket_timeout=settings.redis_timeout,
        socket_connect_timeout=settings.redis_timeout,
        health_check_interval=30, retry_on_timeout=True)
    client.ping()                                    # 連不上就讓它在這裡炸
    return client


def install() -> str:
    """在 lifespan 裡呼叫一次，回傳實際用的後端名稱。

    設了 `DAY24_REDIS_URL` 就走 Redis；連不上就退回 memory 並留一行 warning。
    **連不上不該讓服務起不來**，快取是加速，不是必要條件。
    """
    global _BACKEND, _CLIENT
    with _LOCK:
        if _ORIGINAL:
            return _BACKEND
        _ORIGINAL["load"] = rag_core._load_json
        _ORIGINAL["save"] = rag_core._save_json

        if settings.redis_url:
            try:
                _CLIENT = _connect()
                rag_core._load_json, rag_core._save_json = _redis_load, _redis_save
                _BACKEND = "redis"
                log.info("cache backend=redis url=%s", _safe_url())
                return _BACKEND
            except Exception as exc:                 # noqa: BLE001
                log.warning("連不上 Redis（%s），退回 memory：%s",
                            _safe_url(), exc)
                _CLIENT = None

        rag_core._load_json, rag_core._save_json = _memory_load, _memory_save
        _BACKEND = "memory"
        log.info("cache backend=memory")
        return _BACKEND


def uninstall() -> None:
    """還原。測試用，以及關機時讓行程乾淨退出。"""
    global _BACKEND, _CLIENT
    with _LOCK:
        if not _ORIGINAL:
            return
        rag_core._load_json = _ORIGINAL.pop("load")
        rag_core._save_json = _ORIGINAL.pop("save")
        _MEMORY.clear()
        if _CLIENT is not None:
            try:
                _CLIENT.close()
            except Exception:                        # noqa: BLE001
                pass
        _CLIENT, _BACKEND = None, "memory"


def warm(*paths: Path) -> dict[str, int]:
    """啟動時先把快取準備好，不要讓第一個使用者付這筆錢。

    memory：把檔案讀進記憶體。
    redis：用檔案當種子灌一次（已存在的不覆蓋），這樣版控裡那份仍然有效。
    """
    out: dict[str, int] = {}
    for path in paths:
        path = Path(path)
        if _BACKEND == "redis":
            seeded = _redis_load(path).seed(
                json.loads(path.read_text(encoding="utf-8")) if path.exists() else {})
            out[path.name] = seeded
        else:
            out[path.name] = len(_memory_load(path))
    return out


def export(path: Path) -> int:
    """把 Redis 裡的內容寫回 JSON 檔，給「要把快取進版控」的時候用。

    平常不會呼叫；`scripts` 或手動跑。回傳寫出去幾筆。
    """
    if _BACKEND != "redis":
        return 0
    store = _redis_load(Path(path))
    data = {k: store[k] for k in store}
    Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return len(data)


def _safe_url() -> str:
    """把密碼遮掉再進 log"""
    url = settings.redis_url or ""
    if "@" in url:
        head, tail = url.split("@", 1)
        scheme = head.split("://", 1)[0] if "://" in head else ""
        return f"{scheme}://***@{tail}"
    return url


def backend() -> str:
    return _BACKEND


def stats() -> dict:
    """`/healthz` 會帶回去。redis 後端只回連線狀態，不做 SCAN。"""
    with _LOCK:
        if _BACKEND != "redis":
            return {"backend": "memory",
                    **{Path(k).name: len(v) for k, v in _MEMORY.items()}}
        try:
            info = _CLIENT.info("memory")
            return {"backend": "redis", "url": _safe_url(),
                    "used_memory_human": info.get("used_memory_human"),
                    "keys": _CLIENT.dbsize()}
        except Exception as exc:                     # noqa: BLE001
            return {"backend": "redis", "url": _safe_url(), "error": str(exc)[:120]}
