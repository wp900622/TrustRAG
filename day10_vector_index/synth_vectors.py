# -*- coding: utf-8 -*-
"""Day 10 合成干擾向量：不花一毛 API 錢，把索引從 59 塊撐到 5 萬塊。

做法（隨機質心＋擾動）：
- 先抽 N_CENTROIDS 個隨機方向當「質心」，每個質心周圍再生 PER_CENTROID 個
  擾動點——讓資料有叢集結構。純高斯亂數彼此近乎正交、沒有任何叢集，
  是 HNSW 圖導航的最壞情況，量出來的數字不能代表真實 embedding 的行為
- 固定 seed：每次重跑生成完全相同的向量，不落地存檔（307 MB 省下來），
  讀者 clone 後跑出同一組資料
- 不污染答案空間：合成向量是隨機方向，真實 embedding 集中在語意流形上，
  問題向量與真實 chunk 的相似度遠高於與隨機向量的相似度（實驗會驗證這件事）
"""
import faiss
import numpy as np

N_CENTROIDS = 500     # 質心數
PER_CENTROID = 100    # 每個質心的擾動點數；總量＝500 × 100 ＝ 50,000
NOISE = 0.35          # 擾動幅度（相對單位向量），決定叢集鬆緊
SEED = 42


def make_distractors(dim: int, n_centroids: int = N_CENTROIDS,
                     per_centroid: int = PER_CENTROID,
                     noise: float = NOISE, seed: int = SEED) -> np.ndarray:
    """生成 (n_centroids × per_centroid, dim) 的 float32 單位向量矩陣"""
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((n_centroids, dim), dtype=np.float32)
    points = np.repeat(centroids, per_centroid, axis=0)
    points += noise * rng.standard_normal((n_centroids * per_centroid, dim),
                                          dtype=np.float32)
    faiss.normalize_L2(points)
    return points
