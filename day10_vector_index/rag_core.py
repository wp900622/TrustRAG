# -*- coding: utf-8 -*-
"""Day 10 共用核心：讀語料、取得 Embedding（含本機快取）、FAISS 索引建立與檢索。

自 Day 9 的 rag_core.py 沿用並修改兩處（Day 9 的檔案不動，本資料夾自給自足）：
- 向量改用 float32 矩陣並預先正規化：5 萬 × 1536 規模下，float64 記憶體加倍、
  速度減半；正規化後「餘弦相似度」等價於「內積」，才能用 FAISS 的 IP 索引
- 檢索改走 FAISS：精確的 IndexFlatIP 與近似的 IndexHNSWFlat 共用同一套
  search() 介面，實驗才能控制變因（只換索引、其他全部不變）
- Embedding 快取、成本計算邏輯不變：以內容雜湊當 key，重跑不重複計費
"""
import hashlib
import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# 從專案目錄的 .env 載入 OPENAI_API_KEY（若環境變數已存在則以環境變數優先）
load_dotenv(Path(__file__).parent / ".env")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
# text-embedding-3-small 定價：每百萬 token 0.02 美元（2026-09 官網查閱）
USD_PER_MILLION_TOKENS = 0.02
USD_TO_TWD = 31  # 粗估匯率，僅供文章內成本示意

BASE_DIR = Path(__file__).parent
DOC_PATH = BASE_DIR / "work_rules.md"
CACHE_PATH = BASE_DIR / "embeddings_cache.json"

_client = None


def get_client() -> OpenAI:
    """延遲建立 OpenAI client，避免只想讀文件時也要求必須設定 API key"""
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "找不到 OPENAI_API_KEY：請複製 .env.example 為 .env，並填入你的金鑰"
            )
        _client = OpenAI()  # 自動讀取環境變數 OPENAI_API_KEY
    return _client


def load_document(path: Path = DOC_PATH) -> str:
    """讀入整份語料原文；切法由 chunkers.py 決定"""
    return Path(path).read_text(encoding="utf-8")


def _cache_key(text: str) -> str:
    """以內容雜湊當快取 key：文件改了一個字，就會重新計算該段向量"""
    return hashlib.sha256(f"{EMBEDDING_MODEL}:{text}".encode("utf-8")).hexdigest()[:16]


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def get_embeddings(texts: list[str]) -> tuple[np.ndarray, int, int]:
    """批次取得向量；已算過的直接用本機快取。

    與 Day 9 的差異：回傳 (n, 1536) 的 float32 矩陣（已 L2 正規化），
    而不是 float64 陣列的 list——餵給 FAISS 的格式就是它。
    回傳：(向量矩陣, 本次實際消耗的 token 數, 快取命中筆數)
    """
    cache = _load_cache()
    misses = [t for t in texts if _cache_key(t) not in cache]
    used_tokens = 0

    if misses:
        response = get_client().embeddings.create(model=EMBEDDING_MODEL, input=misses)
        used_tokens = response.usage.total_tokens
        for text, item in zip(misses, response.data):
            cache[_cache_key(text)] = item.embedding
        _save_cache(cache)

    matrix = np.array([cache[_cache_key(t)] for t in texts], dtype=np.float32)
    faiss.normalize_L2(matrix)  # 單位向量之後，內積＝餘弦相似度
    return matrix, used_tokens, len(texts) - len(misses)


def build_flat(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """精確索引：內積暴力掃描（向量已正規化，內積＝餘弦相似度），結果保證正確"""
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def build_hnsw(vectors: np.ndarray, m: int = 32,
               ef_construction: int = 200) -> faiss.IndexHNSWFlat:
    """近似索引：HNSW 圖。查詢走圖導航而非全量掃描，快但不保證找到真正的最近鄰。

    m＝每個節點的連線數上限、ef_construction＝建圖時的候選清單寬度；
    查詢期的取捨旋鈕是 index.hnsw.efSearch（越大越準、越慢），由呼叫端設定。
    """
    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    return index


def search(question: str, index: faiss.Index,
           top_k: int = 3) -> tuple[np.ndarray, np.ndarray, int]:
    """把問題轉成向量，用指定索引檢索。Flat 與 HNSW 走同一條路，控制變因。

    回傳：(top_k 個向量 id, 對應相似度分數, 問題 embedding 消耗的 token 數)
    id 與建索引時 add() 的順序一致，如何對回 chunk 由呼叫端決定。
    """
    vector, used_tokens, _ = get_embeddings([question])
    scores, ids = index.search(vector, top_k)
    return ids[0], scores[0], used_tokens


def cost_usd(tokens: int) -> float:
    """token 數換算成美元成本"""
    return tokens / 1_000_000 * USD_PER_MILLION_TOKENS
