# -*- coding: utf-8 -*-
"""Day 11 共用核心：讀語料、取得 Embedding（含本機快取）與成本計算。

自 Day 10 的 rag_core.py 沿用並精簡（Day 10 的檔案不動，本資料夾自給自足）：
- 移除 FAISS：索引與檢索改由 Chroma 負責（見 chroma_store.py），
  本檔只管「文字進、向量出」
- 向量正規化改用 numpy（Day 10 借的是 faiss.normalize_L2）：Day 11 不依賴 faiss
- Embedding 快取、成本計算邏輯不變：以內容雜湊當 key，重跑不重複計費
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# 從本資料夾的 .env 載入 OPENAI_API_KEY（若環境變數已存在則以環境變數優先）
load_dotenv(Path(__file__).parent / ".env")

EMBEDDING_MODEL = "text-embedding-3-small"
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

    回傳 (n, 1536) 的 float32 矩陣（已 L2 正規化）——單位向量下
    cosine 距離的分母恆為 1，跟 Day 8~10 的餘弦相似度是同一把尺。
    回傳：(向量矩陣, 本次實際消耗的 token 數, 快取命中筆數)
    """
    cache = _load_cache()
    misses = [t for t in texts if _cache_key(t) not in cache]
    used_tokens = 0

    if misses:
        response = get_client().embeddings.create(model=EMBEDDING_MODEL, input=misses)
        used_tokens = response.usage.total_tokens
        # 以 item.index 對回原輸入，不假設回傳順序與輸入一致——
        # 錯位會把向量存到錯的雜湊 key 下，靜默污染快取
        for item in response.data:
            cache[_cache_key(misses[item.index])] = item.embedding
        _save_cache(cache)

    matrix = np.array([cache[_cache_key(t)] for t in texts], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix, used_tokens, len(texts) - len(misses)


def cost_usd(tokens: int) -> float:
    """token 數換算成美元成本"""
    return tokens / 1_000_000 * USD_PER_MILLION_TOKENS
