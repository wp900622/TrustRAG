# -*- coding: utf-8 -*-
"""Day 8 共用核心：讀規範文件、取得 Embedding（含本機快取）、計算相似度、檢索最相關條文。

設計重點：
- 這一天刻意「一條規定＝一段」，不做 Chunking（那是 Day 9 的主題）
- Embedding 結果快取在本機 JSON，重跑不重複花錢
- 回傳 token 用量，讓呼叫端能算出實際成本
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# 從專案目錄的 .env 載入 OPENAI_API_KEY（若環境變數已存在則以環境變數優先）
load_dotenv(Path(__file__).parent / ".env")

EMBEDDING_MODEL = "text-embedding-3-small"
# text-embedding-3-small 定價：每百萬 token 0.02 美元（2026-09 官網查閱）
USD_PER_MILLION_TOKENS = 0.02
USD_TO_TWD = 31  # 粗估匯率，僅供文章內成本示意

BASE_DIR = Path(__file__).parent
POLICY_PATH = BASE_DIR / "leave_policy.md"
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


def load_articles(path: Path = POLICY_PATH) -> list[dict]:
    """把規範文件依「## 第 N 條」切成一條一段。

    回傳格式：[{"title": "第 1 條（目的與適用範圍）", "text": "標題+內文"}, ...]
    """
    text = Path(path).read_text(encoding="utf-8")
    articles = []
    # 第一塊是文件標題與前言，不放進檢索範圍
    for block in text.split("\n## ")[1:]:
        lines = block.strip().splitlines()
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        articles.append({"title": title, "text": f"{title}\n{body}"})
    return articles


def _cache_key(text: str) -> str:
    """以內容雜湊當快取 key：文件改了一個字，就會重新計算該段向量"""
    return hashlib.sha256(f"{EMBEDDING_MODEL}:{text}".encode("utf-8")).hexdigest()[:16]


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def get_embeddings(texts: list[str]) -> tuple[list[np.ndarray], int, int]:
    """批次取得向量；已算過的直接用本機快取。

    回傳：(依輸入順序排列的向量列表, 本次實際消耗的 token 數, 快取命中筆數)
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

    vectors = [np.array(cache[_cache_key(t)]) for t in texts]
    return vectors, used_tokens, len(texts) - len(misses)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """餘弦相似度：兩向量夾角越小（語意越近），值越接近 1"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search(question: str, articles: list[dict],
           article_vectors: list[np.ndarray], top_k: int = 3) -> tuple[list[dict], int]:
    """把問題轉成向量，回傳相似度最高的前 top_k 條條文。

    回傳：([{"title", "text", "score"}, ...], 問題 embedding 消耗的 token 數)
    """
    (question_vector,), used_tokens, _ = get_embeddings([question])
    scored = [
        {**article, "score": cosine_similarity(question_vector, vector)}
        for article, vector in zip(articles, article_vectors)
    ]
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k], used_tokens


def cost_usd(tokens: int) -> float:
    """token 數換算成美元成本"""
    return tokens / 1_000_000 * USD_PER_MILLION_TOKENS
