# -*- coding: utf-8 -*-
"""TrustRAG Day 24：把前 23 天的實驗腳本收成一支服務。

    main.py        FastAPI app 與端點
    contracts.py   Pydantic 請求／回應契約（唯一需要給別人讀的檔案）
    config.py      設定，全部可以用 DAY24_* 環境變數覆蓋
    timing.py      延遲分解的 middleware，跟 RAG 無關，可以直接抄走
    errors.py      統一的錯誤形狀 ＋ 跟著請求跑的 request id
    cache.py       常駐快取：整包讀寫的檔案快取在並發下會壞，這裡換掉它
    store.py       攝取 job 的狀態存 SQLite，多個 worker 才查得到同一筆
    retrievers.py  Retriever 介面 ＋ Chroma／numpy 兩個實作
    embedder.py    算問題向量：走快取的 vs 真的打 API 的
    ingest.py      攝取走背景任務（Day 22 量過 OCR 一份要 277 秒）
    agents.py      把 Day 20／21 那支 agent 接進來（它一行都沒改）
    tests/         24 個測試，不打 API、不花錢

語料、切法、prompt、檢索、生成全部沿用上層目錄的模組，一個字都沒改。
這個套件做的是它們外面那一層：設定、快取、狀態、錯誤、觀測。
"""
