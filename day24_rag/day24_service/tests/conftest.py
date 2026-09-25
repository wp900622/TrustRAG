# -*- coding: utf-8 -*-
"""測試共用的東西。

`client` 本來長在 test_service.py 裡，Day 26 多了一支 test_streaming.py
也要用，所以搬到這裡。搬過來而不是複製一份，是因為兩邊各建一個 app
會各建一份索引，測試會變慢，而且 job db 會互相踩。

scope="session"：整輪共用一個 app。本來是 module，但兩個測試檔各起一個
app 之後，第二個要刪那顆 job db 時第一個還握著連線，Windows 上就是
PermissionError。共用一個順便省下一次建索引的幾秒。
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 測試用自己的 job db，不要污染開發時那一顆
os.environ.setdefault("DAY24_JOB_DB", str(ROOT / "day24_jobs_test.sqlite3"))
# 索引也用自己的 collection。Day 29 起服務的索引會記得攝取過什麼，
# 測試攝取的東西不能留到開發用的那一顆裡
os.environ.setdefault("DAY24_COLLECTION", "trustrag-test")

from fastapi.testclient import TestClient      # noqa: E402

import chroma_store                             # noqa: E402
from chromadb.errors import NotFoundError       # noqa: E402

from day24_service.main import app             # noqa: E402


@pytest.fixture(scope="session")
def client():
    db = Path(os.environ["DAY24_JOB_DB"])
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    # 每一輪都從「剛種好第一份語料」開始，上一輪沒清乾淨的不能帶進來
    try:
        chroma_store.get_client().delete_collection(os.environ["DAY24_COLLECTION"])
    except NotFoundError:
        pass
    with TestClient(app) as c:
        yield c
