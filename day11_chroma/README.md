# Day 11：Chroma 向量資料庫——換庫對照、metadata 過濾、持久化與增量

把 Day 10 的 FAISS 索引換成 Chroma（內嵌模式）：同一份 59 條《員工工作規則》、
同一組 15 題，驗證「換庫不換答案」，再實測索引做不到的三件事——
metadata 章別過濾、持久化（重開行程資料還在）、增量新增與刪除。
最後把 Day 10 的 5 萬個合成向量灌進來，看 Chroma 的預設 HNSW 參數掉不掉題（會）。

## 檔案說明

| 檔案 | 用途 |
|------|------|
| `work_rules.md` | 虛構語料：《員工工作規則》11 章 59 條（同 Day 9/10） |
| `questions.json` | 15 題測試題（同 Day 9/10，一題沒動） |
| `rag_core.py` | Embedding＋本機快取＋成本計算（Day 11 版：不再依賴 faiss） |
| `chunkers.py` | 結構化切法＋章別解析（chunk 帶 chapter metadata） |
| `chroma_store.py` | Chroma 存取層：建庫、開庫、批次寫入（demo 與實驗共用） |
| `synth_vectors.py` | 合成干擾向量（同 Day 10 的質心＋擾動、固定 seed） |
| `build_db.py` | 建庫：`--with-50k` 可另建 59＋50,000 塊的大庫 |
| `search_demo.py` | 互動查詢 demo，`--chapter N` 做章別過濾 |
| `add_demo.py` | 增量 demo：add 一條新條文→立刻可查→delete 復原 |
| `run_experiment.py` | 四項實驗，結果寫入 `experiment_result.md` |
| `cold_open.py`／`ef_probe.py` | 子行程探針：冷開啟計時／ef_search 掃描（見下方「坑」） |
| `experiment_result.md` | 實驗實跑結果（可直接對照文章） |
| `chroma_db/` | 持久化資料庫（`build_db.py` 產物，不進版控） |

## 跑法

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 OPENAI_API_KEY（系統環境變數已有則可略過）
python build_db.py     # 建 59 條的庫（首次會計算 embedding，見成本）
python search_demo.py 特休沒休完會怎樣
python search_demo.py --chapter 5 特休沒休完會怎樣
python add_demo.py
python run_experiment.py   # 含 50k 建庫（首次約 1～2 分鐘）與 ef_search 掃描
```

**執行時間與記憶體**：50,059 塊建庫實測 72 秒（多執行緒）、磁碟約 370 MB、
峰值記憶體約 1.5 GB；建好後 `run_experiment.py` 重跑會沿用既有庫。

## 已知的坑

- **剛建完索引的行程裡 `modify()` 改 `ef_search` 不會套用到查詢**，重開行程
  才生效——所以 ef 掃描由 `ef_probe.py` 在全新子行程逐檔位量測。
- Chroma 建圖是多執行緒且有背景整理（WAL 併入索引），掉題明細重跑會在附近
  浮動，無法像 Day 10 的 FAISS 用固定 seed＋單執行緒逐題重現。
- 距離空間預設是 L2，要自己指定 cosine；回傳為距離（＝1−餘弦相似度）。

## 成本說明

使用 `text-embedding-3-small`（每百萬 token 0.02 美元）。59 條與 15 題沿用
Day 9/10 的快取（`embeddings_cache.json`，已被 .gitignore 排除、重跑不重複計費）；
本日新增「第 60 條」與寵物問題兩筆共 205 tokens ≈ 新台幣 0.00013 元。
刪除快取重跑可重現無快取的完整成本（約 13,800 tokens ≈ 新台幣 0.0086 元）。
50,000 個合成向量由本機生成，0 元。

## 資安注意

- **CVE-2026-45829（CVSS 10.0，撰文時未修補）**：chromadb 以 Python server
  模式對外服務時，未認證的攻擊者可遠端執行程式碼。本資料夾只用**內嵌模式**
  （`PersistentClient`），不開任何網路埠，不受此漏洞影響。請不要把 Chroma 的
  Python server 直接暴露在網路上；需要 server 請用官方 Docker 映像並限制來源。
- 把 metadata 過濾當**權限邊界**用時，`where` 條件必須由後端依登入者身分
  產生，不得接受前端傳入——本資料夾的 `--chapter` 是本機 demo 參數，
  照搬進 web 服務就是水平權限繞過。
- 範例程式已關閉 Chroma 的匿名遙測（`anonymized_telemetry=False`；遙測＝
  軟體在背景自動回傳使用統計給開發商，不含資料內容，但範例不該預設連網回報）。
- 把 `work_rules.md` 換成真實文件前請想清楚：**整份語料會送到 OpenAI 計算
  embedding**。本語料為虛構（見檔頭聲明）。
