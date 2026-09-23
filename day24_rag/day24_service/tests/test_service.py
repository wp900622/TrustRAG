# -*- coding: utf-8 -*-
"""服務的測試。一毛錢都不花：全部走既有快取，不打 API。

這批測試要擋住的是前面幾版真的踩過的東西：

- 攝取同一份文件兩次，索引從 59 塊變 118 塊（id 用 job_id 編號造成的）
- 攝取完 `?retriever=numpy` 還拿著舊矩陣
- job 狀態放行程記憶體，換一個行程就查不到
- 錯誤各長各的樣子，呼叫端沒辦法用程式判斷

跑：
    pytest day24_service/tests -q
"""
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 測試用自己的 job db，不要污染開發時那一顆
os.environ.setdefault("DAY24_JOB_DB", str(ROOT / "day24_jobs_test.sqlite3"))

from fastapi.testclient import TestClient      # noqa: E402

from day24_service.config import settings      # noqa: E402
from day24_service.ingest import chunk_id      # noqa: E402
from day24_service.main import app             # noqa: E402

QUESTION = "病假連續請幾天以上需要附診斷證明？"


# client fixture 在 conftest.py，test_streaming.py 也用它


# ------------------------------------------------------------------ 健康

def test_healthz_reports_a_built_index(client):
    body = client.get("/healthz").json()
    assert body["ready"] is True
    assert body["chunks"] >= 59
    assert set(body["retrievers"]) == {"chroma", "numpy"}


def test_readyz_is_204(client):
    assert client.get("/readyz").status_code == 204


def test_every_response_carries_a_request_id(client):
    r = client.get("/healthz")
    assert r.headers.get("x-request-id")


def test_caller_supplied_request_id_is_kept(client):
    r = client.get("/healthz", headers={"x-request-id": "trace-me"})
    assert r.headers["x-request-id"] == "trace-me"


# -------------------------------------------------------------------- 契約

def test_ask_returns_the_documented_shape(client):
    body = client.post("/ask", json={"question": QUESTION, "k": 3}).json()
    assert body["answer"]
    assert body["mode"] == "pipeline"
    assert body["agent"] is None
    assert len(body["citations"]) == 3
    for key in ("embed_ms", "retrieve_ms", "llm_ms", "serialize_ms",
                "total_ms", "overhead_ms"):
        assert key in body["timing"]
    # 四段加起來不該超過總時間
    t = body["timing"]
    known = t["embed_ms"] + t["retrieve_ms"] + t["llm_ms"] + t["serialize_ms"]
    assert known <= t["total_ms"] + 1e-6


def test_citations_come_from_retrieval_not_from_the_model(client):
    """引用必須是檢索撈到的原文，不是模型講的條號。

    Day 15 量過模型會自己編條號，所以這裡驗 citation 的 text
    真的出現在語料裡，而不是只驗有幾條。
    """
    corpus = (ROOT / "work_rules.md").read_text(encoding="utf-8")
    body = client.post("/ask", json={"question": QUESTION, "k": 3}).json()
    for c in body["citations"]:
        assert c["text"][:20] in corpus


def test_timing_headers_are_set(client):
    r = client.post("/ask", json={"question": QUESTION, "k": 3})
    assert float(r.headers["x-total-ms"]) > 0
    assert "x-retrieve-ms" in r.headers


# ------------------------------------------------------------ 兩種 retriever

def test_both_retrievers_agree(client):
    a = client.post("/ask", json={"question": QUESTION, "k": 3},
                    params={"retriever": "chroma"}).json()
    b = client.post("/ask", json={"question": QUESTION, "k": 3},
                    params={"retriever": "numpy"}).json()
    assert [c["article_no"] for c in a["citations"]] == \
           [c["article_no"] for c in b["citations"]]
    assert a["answer"] == b["answer"]


def test_unknown_retriever_is_a_typed_error(client):
    r = client.post("/ask", json={"question": QUESTION},
                    params={"retriever": "qdrant"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "unknown_retriever"


# -------------------------------------------------------------------- agent

def test_agent_mode_reports_its_trace(client):
    body = client.post("/ask", json={"question": QUESTION, "mode": "agent"}).json()
    assert body["mode"] == "agent"
    assert body["agent"]["searches"] >= 1
    assert body["agent"]["llm_calls"] >= 1
    assert body["citations"]


def test_frozen_agent_runs_on_any_retriever(client):
    """agent_day21.py 一行都沒改，透過轉接頭跑在 numpy 上結果要一樣。"""
    a = client.post("/ask", json={"question": QUESTION, "mode": "agent"},
                    params={"retriever": "chroma"}).json()
    b = client.post("/ask", json={"question": QUESTION, "mode": "agent"},
                    params={"retriever": "numpy"}).json()
    assert a["answer"] == b["answer"]
    assert a["agent"]["queries"] == b["agent"]["queries"]


# -------------------------------------------------------------------- 驗證

@pytest.mark.parametrize("payload, where", [
    ({"question": ""}, "question"),
    ({"question": QUESTION, "k": 0}, "k"),
    ({"question": QUESTION, "k": 99}, "k"),
    ({"question": QUESTION, "mode": "magic"}, "mode"),
])
def test_bad_requests_are_422_with_a_field_name(client, payload, where):
    r = client.post("/ask", json=payload)
    assert r.status_code == 422
    body = r.json()["error"]
    assert body["code"] == "invalid_request"
    assert where in body["message"]


# -------------------------------------------------------------------- 攝取

def _wait(client, job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/documents/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("job 沒有在時限內結束")


def test_ingest_returns_202_immediately(client):
    r = client.post("/documents", json={"path": "work_rules.md"})
    assert r.status_code == 202
    assert r.json()["status"] in ("queued", "running", "done")
    _wait(client, r.json()["job_id"])


def test_ingesting_the_same_document_twice_does_not_grow_the_index(client):
    """這是第一版真的踩到的 bug：59 塊會變 118 塊。"""
    before = client.get("/healthz").json()["chunks"]
    for _ in range(2):
        r = client.post("/documents", json={"path": "work_rules.md"})
        job = _wait(client, r.json()["job_id"])
        assert job["status"] == "done", job
    after = client.get("/healthz").json()["chunks"]
    assert after == before


def test_ingest_refreshes_the_numpy_retriever(client):
    """攝取完沒有重建 retriever 的話，numpy 這條路會停在舊矩陣上。"""
    r = client.post("/documents", json={"path": "work_rules.md"})
    _wait(client, r.json()["job_id"])
    a = client.post("/ask", json={"question": QUESTION},
                    params={"retriever": "numpy"}).json()
    assert [c["article_no"] for c in a["citations"]]


def test_chunk_id_is_stable_for_the_same_content():
    meta = {"article_no": 24}
    assert chunk_id("work_rules.md", meta, "abc") == \
           chunk_id("work_rules.md", meta, "abc")
    assert chunk_id("work_rules.md", meta, "abc") != \
           chunk_id("work_rules.md", meta, "abd")


def test_path_traversal_is_rejected(client):
    r = client.post("/documents", json={"path": "../../../etc/passwd"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_path"


def test_missing_document_fails_the_job_not_the_request(client):
    """檔案不存在是 job 的失敗，不是 POST 的失敗——呼叫端拿得到 job_id 才查得到原因。"""
    r = client.post("/documents", json={"path": "no_such_file.md"})
    assert r.status_code == 202
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "failed"
    assert "找不到" in job["error"]


def test_unknown_job_is_404_with_a_code(client):
    r = client.get("/documents/deadbeef")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "job_not_found"


def test_jobs_survive_in_the_store(client):
    """job 狀態在 SQLite 裡，不在行程記憶體——列得出來才代表真的落地了。"""
    r = client.post("/documents", json={"path": "work_rules.md"})
    _wait(client, r.json()["job_id"])
    jobs = client.get("/documents", params={"limit": 5}).json()
    assert any(j["job_id"] == r.json()["job_id"] for j in jobs)


def test_path_is_resolved_under_documents_root():
    assert settings.resolve_document("work_rules.md").name == "work_rules.md"
    with pytest.raises(ValueError):
        settings.resolve_document("../secret.md")
