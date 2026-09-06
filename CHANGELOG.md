# Changelog

| 版本 | 日期 | 變更說明 |
|------|------|----------|
| 0.1.0 | 2026-09-04 | Day 8 內容入版控 |
| 0.2.0 | 2026-09-04 | 新增 Day 9 Chunking 實驗（語料、三種切法、15 題實驗、文章草稿） |
| 0.3.0 | 2026-09-04 | Day 9 實驗新增第二指標「命中純度」（總表加欄、明細加欄；實跑 35%／32%／100%）；成本行改為「計費 tokens」與「快取命中塊數」分開列；`day09_draft.md` 全文重寫為得獎版結構（鉤子標題、雙計分表、決策速查表）；新增根目錄 `article_template.md` 系列文章模板 |
| 0.4.0 | 2026-09-06 | 新增根目錄 `view_tracking.csv`：系列文章瀏覽數長期追蹤（long format，欄位 snapshot_date/snapshot_period/day/publish_date/views），回填 9/4、9/5、9/6 三次快照共 27 筆；Day 10 既有內容一併入版控 |
| 0.5.0 | 2026-09-06 | 新增 Day 10 FAISS 向量索引實驗（`day10_vector_index/`）：Flat（精確）vs HNSW（近似）對比，`synth_vectors.py` 以固定 seed 的「質心＋擾動」合成 50,000 個干擾向量（0 元）把索引撐到 50,059 塊；實驗指標從命中率擴充到單題延遲（中位數／p95）與 recall@k，並掃 6 檔 `efSearch` 找出 recall 歸零點（=64 時 recall 1.00 且仍快 12 倍）；`search_demo.py` 新增 `--distractors` 參數；新增相依性套件 `faiss-cpu==1.15.0`（釘死版本以確保數字可重現，五問審查結論已落檔 PRD）；審查修正：實驗成本行補「刪除 `embeddings_cache.json` 重跑即可重現無快取的完整成本」註記（與 Day 9 慣例一致）、報表改為先寫檔再 print（避免 stdout 重導向時 cp950 編碼錯誤讓報表遺失）；PRD 同步更新（範例資料夾兩個→三個、向量運算改由 faiss 索引承擔、Day 10 章節與參數／函式介面） |
