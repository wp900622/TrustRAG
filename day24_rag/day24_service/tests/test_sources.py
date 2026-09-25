# -*- coding: utf-8 -*-
"""Day 29：索引裡不只一份文件之後的測試。

第二份文件是 `fixtures/handbook_excerpt.md`：`work_rules.md` 的前三章原封不動，
16 塊，每一塊的向量都已經在快取裡，所以一毛錢都不花。刻意挑「跟第一份重疊」的內容，
因為兩份文件都有第 1～16 條，條號撞在一起正是今天要擋的事。

整個模組共用一次攝取，結束時刪掉。索引是測試自己的 collection（conftest 設的），
但同一輪裡別的測試檔也在用它，多一份文件會改變它們撈到的東西。
"""
import time

import numpy as np
import pytest

import rag_core

BASE = "work_rules.md"
EXCERPT = "handbook_excerpt.md"
EXCERPT_PATH = "day24_service/tests/fixtures/handbook_excerpt.md"
QUESTION = "忘記打卡要怎麼補救？會有事嗎？"   # 第 11 條，第三章，兩份文件都有


def _forbidden(*_a, **_kw):
    raise AssertionError("測試不准打 API")


def _wait(client, job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/documents/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("job 沒有在時限內結束")


def _search(client, retriever, **body):
    r = client.post("/search", json={"question": QUESTION, **body},
                    params={"retriever": retriever})
    assert r.status_code == 200, r.text
    return r.json()["hits"]


def _pairs(hits):
    return [(h["source"], h["article_no"]) for h in hits]


@pytest.fixture(scope="module")
def two_docs(client):
    mp = pytest.MonkeyPatch()
    mp.setattr(rag_core, "get_client", _forbidden)
    r = client.post("/documents", json={"path": EXCERPT_PATH})
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "done", job
    yield client
    if any(s["source"] == EXCERPT for s in client.get("/sources").json()):
        client.delete(f"/sources/{EXCERPT}")
    mp.undo()


# ------------------------------------------------------------ 有幾份文件

def test_sources_lists_both_documents(two_docs):
    got = {s["source"]: s for s in two_docs.get("/sources").json()}
    assert got[EXCERPT]["chunks"] == 16
    assert got[EXCERPT]["articles"] == list(range(1, 17))
    assert got[BASE]["chunks"] == 59


def test_every_chunk_knows_its_document(two_docs):
    """開機種的那 59 塊也要帶 source，不然會出現一個叫 "" 的文件。"""
    names = [s["source"] for s in two_docs.get("/sources").json()]
    assert "" not in names


def test_restart_keeps_ingested_documents(two_docs):
    """Day 29 修掉的那一個：開機走實驗用的 open_or_build，塊數不是 59 就砍掉重建。

    直接再跑一次開機建索引那一段，等於重啟。
    """
    from day24_service import main
    main._build_index()
    got = {s["source"]: s["chunks"] for s in two_docs.get("/sources").json()}
    assert got == {BASE: 59, EXCERPT: 16}


def test_ingesting_the_base_document_again_does_not_duplicate_it(two_docs):
    """開機與攝取以前用兩種 id，同一份文件會變成 118 塊。現在只有一種。"""
    r = two_docs.post("/documents", json={"path": BASE})
    assert _wait(two_docs, r.json()["job_id"])["status"] == "done"
    assert two_docs.get("/healthz").json()["chunks"] == 59 + 16


# ------------------------------------------------------ 兩個 retriever 一致

def test_new_document_is_visible_to_both_retrievers(two_docs):
    """攝取完 numpy 那條路要看得到新文件，不是只有 Chroma 看得到。"""
    for name in ("chroma", "numpy"):
        hits = _search(two_docs, name, k=3, source=EXCERPT)
        assert hits, f"{name} 查不到剛攝取的文件"
        assert {h["source"] for h in hits} == {EXCERPT}


@pytest.mark.parametrize("where", [
    {},
    {"source": BASE},
    {"source": EXCERPT},
    {"chapter_no": 2},
    {"chapter_no": 5},
    {"source": EXCERPT, "chapter_no": 5},     # 摘錄沒有第五章：兩邊都要回空的
])
def test_both_retrievers_give_the_same_hits(two_docs, where):
    """`Retriever` 這個介面唯一有意義的檢查：同一個條件，同一組結果。

    兩份文件內容重疊，同一條文在兩份裡的相似度一模一樣，排序會打平，
    所以比的是集合，不是順序。
    """
    a = _search(two_docs, "chroma", k=6, **where)
    b = _search(two_docs, "numpy", k=6, **where)
    assert sorted(_pairs(a)) == sorted(_pairs(b))
    assert len(a) == len(b) <= 6
    if where.get("chapter_no") == 5 and where.get("source") == EXCERPT:
        assert a == []


def test_filter_is_applied_before_top_k(two_docs):
    """「第五章裡最相關的三條」要湊滿三條。

    先取全庫 top-3 再丟掉不是第五章的，這個問題（問打卡，答案在第三章）會一條都不剩。
    """
    for name in ("chroma", "numpy"):
        hits = _search(two_docs, name, k=3, chapter_no=5)
        assert len(hits) == 3
        assert all(h["article_no"] >= 21 for h in hits)     # 第五章從第 21 條起


def test_min_similarity_can_return_fewer_than_k(two_docs):
    hits = _search(two_docs, "numpy", k=10, min_similarity=0.99)
    assert len(hits) < 10


# ---------------------------------------------------- 同一個條號，兩份文件

def test_same_article_number_keeps_both_documents(two_docs):
    """兩份文件都有第 7 條。只用條號當 key 的話，後進來的會把前一份蓋掉。"""
    from day24_service.main import STATE
    index = STATE["chunk_index"]
    assert (BASE, 7) in index and (EXCERPT, 7) in index


def test_citations_say_which_document(two_docs):
    """過濾要一路帶進 /ask。

    篩的是開機那一份，不是摘錄：只篩摘錄的話撈到的三條會跟 Day 14 不同，
    prompt 就不同，快取不命中就要打 API。反過來篩開機那份，prompt 跟 Day 14 一字不差。
    摘錄裡有一模一樣的第 11 條，條件沒生效的話它會混進來。
    """
    r = two_docs.post("/ask", json={"question": QUESTION, "source": BASE})
    assert r.status_code == 200, r.text
    assert {c["source"] for c in r.json()["citations"]} == {BASE}


def test_agent_filter_is_carried_by_the_adapter(two_docs):
    """凍結的 agent 簽名裡沒有過濾，條件由轉接頭夾帶。引用要指回對的那份。"""
    r = two_docs.post("/ask", json={"question": QUESTION, "mode": "agent",
                                    "source": BASE})
    assert r.status_code == 200, r.text
    cites = r.json()["citations"]
    assert cites and {c["source"] for c in cites} == {BASE}


# ------------------------------------------------------------------ 刪除

def test_delete_removes_the_document_from_both_retrievers(two_docs):
    r = two_docs.delete(f"/sources/{EXCERPT}")
    assert r.status_code == 200, r.text
    assert r.json() == {"source": EXCERPT, "removed": 16, "chunks_left": 59}
    for name in ("chroma", "numpy"):
        assert _search(two_docs, name, k=3, source=EXCERPT) == []
    # 刪一份，另一份不能跟著少
    assert _search(two_docs, "numpy", k=3, source=BASE)


def test_delete_twice_is_a_typed_404(two_docs):
    r = two_docs.delete(f"/sources/{EXCERPT}")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "source_not_found"


def test_numpy_survives_an_empty_index():
    """全部刪光之後，空矩陣乘問題向量會炸。庫裡沒東西的正確答案是沒有結果。"""
    from day24_service.retrievers import NumpyRetriever
    empty = NumpyRetriever(np.zeros((0, 1), dtype=np.float32), [], [])
    assert empty.search(np.ones(1536, dtype=np.float32), 3) == []
