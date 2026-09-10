# -*- coding: utf-8 -*-
"""Day 14 共用核心：讀語料、Embedding（含快取）、**生成**（含快取）與成本計算。

自 Day 11 的 rag_core.py 沿用並擴充（Day 11 的檔案不動，本資料夾自給自足）：
- 新增 chat()：本系列第一次把檢索結果餵給 LLM 產生答案，RAG 的「G」從今天開始
- 成本從單軌變雙軌：Day 8~13 只有 embedding 費用，今天多了生成的
  輸入／輸出 token，兩者單價差 30 倍，必須分開列
- 生成也做快取：temperature=0 且 prompt 相同時答案應該一致，
  快取讓「重跑實驗」不重複計費，也讓文章裡的數字可被讀者重現
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
CHAT_MODEL = "gpt-4o-mini"

# 定價（每百萬 token，美元）。以官網為準，換模型只要改這三個數字，
# 報表與文章的成本行會跟著變。
USD_PER_MILLION_EMBEDDING = 0.02
USD_PER_MILLION_CHAT_INPUT = 0.15
USD_PER_MILLION_CHAT_OUTPUT = 0.60
USD_TO_TWD = 31  # 粗估匯率，僅供文章內成本示意

BASE_DIR = Path(__file__).parent
DOC_PATH = BASE_DIR / "work_rules.md"
CACHE_PATH = BASE_DIR / "embeddings_cache.json"
CHAT_CACHE_PATH = BASE_DIR / "chat_cache.json"

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


# ---------------------------------------------------------------- Embedding

def _cache_key(text: str) -> str:
    """以內容雜湊當快取 key：文件改了一個字，就會重新計算該段向量"""
    return hashlib.sha256(f"{EMBEDDING_MODEL}:{text}".encode("utf-8")).hexdigest()[:16]


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def get_embeddings(texts: list[str]) -> tuple[np.ndarray, int, int]:
    """批次取得向量；已算過的直接用本機快取。

    回傳 (n, 1536) 的 float32 矩陣（已 L2 正規化）——單位向量下
    cosine 距離的分母恆為 1，跟 Day 8~11 的餘弦相似度是同一把尺。
    回傳：(向量矩陣, 本次實際消耗的 token 數, 快取命中筆數)
    """
    cache = _load_json(CACHE_PATH)
    misses = [t for t in texts if _cache_key(t) not in cache]
    used_tokens = 0

    if misses:
        response = get_client().embeddings.create(model=EMBEDDING_MODEL, input=misses)
        used_tokens = response.usage.total_tokens
        # 以 item.index 對回原輸入，不假設回傳順序與輸入一致——
        # 錯位會把向量存到錯的雜湊 key 下，靜默污染快取
        for item in response.data:
            cache[_cache_key(misses[item.index])] = item.embedding
        _save_json(CACHE_PATH, cache)

    matrix = np.array([cache[_cache_key(t)] for t in texts], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix, used_tokens, len(texts) - len(misses)


# --------------------------------------------------------------- 生成（新增）

def _chat_cache_key(messages: list[dict]) -> str:
    """快取 key 包含模型與完整訊息：換模型、改一個字的 prompt 都會重新呼叫"""
    payload = json.dumps([CHAT_MODEL, messages], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def chat(messages: list[dict]) -> tuple[str, int, int]:
    """把 messages 丟給 LLM 取回答案文字。

    temperature=0：實驗要可重複，creative 留給行銷文案。
    （注意：temperature=0 只是把隨機性壓到最低，不保證位元級完全一致。）
    回傳：(答案文字, 輸入 token 數, 輸出 token 數)；快取命中時兩個 token 數皆為 0
    """
    cache = _load_json(CHAT_CACHE_PATH)
    key = _chat_cache_key(messages)
    if key in cache:
        return cache[key], 0, 0

    response = get_client().chat.completions.create(
        model=CHAT_MODEL, messages=messages, temperature=0)
    text = response.choices[0].message.content.strip()
    cache[key] = text
    _save_json(CHAT_CACHE_PATH, cache)
    return text, response.usage.prompt_tokens, response.usage.completion_tokens


# ------------------------------------------------------------------- 成本

def embedding_cost_usd(tokens: int) -> float:
    """embedding token 數換算成美元"""
    return tokens / 1_000_000 * USD_PER_MILLION_EMBEDDING


def chat_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """生成的輸入／輸出 token 分開計價——輸出比輸入貴 4 倍，混在一起算會失真"""
    return (input_tokens / 1_000_000 * USD_PER_MILLION_CHAT_INPUT
            + output_tokens / 1_000_000 * USD_PER_MILLION_CHAT_OUTPUT)


def twd(usd: float) -> float:
    """美元換算新台幣（文章成本行用）"""
    return usd * USD_TO_TWD
