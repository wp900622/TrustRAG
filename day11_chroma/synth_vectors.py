# -*- coding: utf-8 -*-
"""Day 11 合成干擾向量：與 Day 10 同一套「隨機質心＋擾動」，把索引撐到 5 萬塊。

- 參數與 seed 與 Day 10 完全相同，rng 生成的原始點一模一樣；
  正規化改用 numpy（Day 11 不依賴 faiss），與 faiss.normalize_L2 等價，
  浮點最後一位可能有極微差異，不影響任何結論
- 固定 seed：每次重跑生成完全相同的向量，不落地存檔（約 308 MB 省下來）
- 不污染答案空間：合成向量是隨機方向，問題向量與真實 chunk 的相似度
  遠高於與隨機向量的相似度（Day 10 已驗證：Flat top-10 中出現 0 次）
"""
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
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points
