# -*- coding: utf-8 -*-
"""Redis 後端的測試。沒有設 `DAY24_REDIS_URL` 就整批跳過。

本機起一個：

    docker run -d --name trustrag-redis -p 6379:6379 \\
      -v trustrag-redis-data:/data redis:7-alpine \\
      redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru

然後：

    DAY24_REDIS_URL=redis://127.0.0.1:6379/1 pytest day24_service/tests -q

刻意用 db 1，不要污染平常在用的 db 0。
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REDIS_URL = os.getenv("DAY24_REDIS_URL")
pytestmark = pytest.mark.skipif(not REDIS_URL, reason="沒有設 DAY24_REDIS_URL")


@pytest.fixture()
def store():
    """一個乾淨的 RedisDict，用測試專屬的 prefix，跑完清掉。"""
    from day24_service.cache import RedisDict
    import redis

    client = redis.from_url(REDIS_URL, decode_responses=False)
    prefix = "day24test:"
    for key in client.scan_iter(match=f"{prefix}*"):
        client.delete(key)
    yield lambda codec: RedisDict(client, prefix, codec)
    for key in client.scan_iter(match=f"{prefix}*"):
        client.delete(key)
    client.close()


def test_text_roundtrip(store):
    s = store("text")
    assert "k" not in s
    s["k"] = "病假連續請三天以上需要附診斷證明。"
    assert "k" in s
    assert s["k"] == "病假連續請三天以上需要附診斷證明。"


def test_vector_roundtrip_is_lossless_at_float32(store):
    """向量存 float32 bytes。管線本來就把它轉成 float32，所以不該有精度差。"""
    s = store("vector")
    original = np.random.default_rng(0).normal(size=1536).astype(np.float32)
    s["v"] = original.tolist()
    back = np.asarray(s["v"], dtype=np.float32)
    assert np.array_equal(back, original)


def test_missing_key_raises_keyerror(store):
    with pytest.raises(KeyError):
        store("text")["沒有這個"]


def test_seed_does_not_overwrite(store):
    s = store("text")
    s["a"] = "後來寫的"
    written = s.seed({"a": "檔案裡的舊值", "b": "檔案裡的新值"})
    assert written == 1              # 只有 b 是新的
    assert s["a"] == "後來寫的"       # a 沒有被檔案蓋掉
    assert s["b"] == "檔案裡的新值"


def test_len_and_iter_see_everything(store):
    s = store("text")
    for i in range(5):
        s[f"k{i}"] = str(i)
    assert len(s) == 5
    assert sorted(s) == [f"k{i}" for i in range(5)]


def test_redis_down_degrades_to_miss_not_error():
    """Redis 掛掉的時候，快取要變成「沒命中」，不能把請求打成 500。"""
    from day24_service.cache import RedisDict
    import redis

    dead = redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2,
                          socket_timeout=0.2)
    s = RedisDict(dead, "day24dead:", "text")
    assert ("k" in s) is False        # 不丟例外
    s["k"] = "寫不進去也不該炸"          # 不丟例外
    with pytest.raises(KeyError):     # 讀不到是 KeyError，呼叫端當成 miss
        s["k"]


def test_install_falls_back_when_redis_is_unreachable(monkeypatch):
    """連不上 Redis 不該讓服務起不來，要退回 memory。"""
    from day24_service import cache
    from day24_service.config import settings

    import dataclasses

    cache.uninstall()
    # Settings 是 frozen dataclass，不能直接改欄位，換掉模組上那個實例
    monkeypatch.setattr(cache, "settings",
                        dataclasses.replace(settings, redis_url="redis://127.0.0.1:1/0"))
    try:
        assert cache.install() == "memory"
    finally:
        cache.uninstall()


def test_service_answers_are_identical_on_redis():
    """換快取後端不該換行為：同一題的答案要跟檔案裡那份一致。"""
    from fastapi.testclient import TestClient

    os.environ.setdefault("DAY24_JOB_DB", str(ROOT / "day24_jobs_test.sqlite3"))
    from day24_service.main import app

    question = "病假連續請幾天以上需要附診斷證明？"
    with TestClient(app) as client:
        body = client.post("/ask", json={"question": question, "k": 3}).json()
    assert body["answer"]
    corpus = (ROOT / "work_rules.md").read_text(encoding="utf-8")
    for c in body["citations"]:
        assert c["text"][:20] in corpus
