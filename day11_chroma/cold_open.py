# -*- coding: utf-8 -*-
"""量「冷開啟」：全新行程從開庫到完成第一次查詢要多久。

run_experiment.py 會以子行程呼叫本檔——同一個行程裡量會被暖快取美化，
必須用新行程才算數。也可以自己跑：python cold_open.py work-rules-50k

查詢向量用固定 seed 的亂數單位向量：冷開啟量的是「庫的載入」不是答案品質，
用亂數就不必碰 embedding 快取與 API。輸出一行 JSON。
"""
import json
import sys
import time

import numpy as np

import chroma_store


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else chroma_store.COLLECTION_50K

    start = time.perf_counter()
    client = chroma_store.get_client()
    collection = client.get_collection(name)
    count = collection.count()
    open_ms = (time.perf_counter() - start) * 1000

    rng = np.random.default_rng(0)
    vector = rng.standard_normal(1536).astype("float32")  # 與 embedding 同維度
    vector /= np.linalg.norm(vector)
    start = time.perf_counter()
    collection.query(query_embeddings=[vector.tolist()], n_results=3)
    first_query_ms = (time.perf_counter() - start) * 1000

    print(json.dumps({"open_ms": round(open_ms, 1),
                      "first_query_ms": round(first_query_ms, 1),
                      "count": count}))


if __name__ == "__main__":
    main()
