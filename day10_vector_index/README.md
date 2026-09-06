# Day 10 — 五萬塊向量：精確掃描 25ms，HNSW 不到 1ms，但預設參數會掉題

2026 iThome 鐵人賽「30 天打造 AI 後端：從 LLM、RAG 到 AI Agent」Day 10 範例程式。

Day 9 結束時索引只有 59 塊，暴力掃描綽綽有餘。今天用 0 元的合成向量把索引
灌到 50,059 塊，實測 FAISS 兩種索引——`IndexFlatIP`（精確暴力掃描）與
`IndexHNSWFlat`（近似圖導航）——的查詢延遲差距，以及「近似」在同一組 15 題
上掉了哪些題、掉成什麼樣子。

## 檔案說明

| 檔案 | 用途 |
|------|------|
| `work_rules.md` | 虛構的《員工工作規則》共 11 章 59 條（沿用 Day 9） |
| `questions.json` | 15 個測試問題（沿用 Day 9，同一把尺才能對照） |
| `chunkers.py` | 結構化切法＋條文範圍解析（自 Day 9 精簡：只留三種切法中的贏家） |
| `rag_core.py` | 共用核心：取 Embedding（float32＋正規化）、FAISS 索引建立與檢索 |
| `synth_vectors.py` | 合成干擾向量：隨機質心＋擾動、固定 seed、不落地存檔，0 元 |
| `search_demo.py` | 互動 demo：FAISS Flat 檢索，`--distractors 50000` 可混入合成向量 |
| `run_experiment.py` | 實驗：Flat vs HNSW 的延遲、recall@1/3/10、Day 9 題組命中率 |

## 怎麼跑

```bash
pip install -r requirements.txt
```

`faiss-cpu` 釘在 1.15.0：本日文章的數字在這個版本實測，faiss 自 v1.14.2 起
在 PyPI 有官方 wheel，Windows / macOS / Linux 都是一行裝好。

設定 OpenAI API key：複製 `.env.example` 為 `.env`，填入你的金鑰：

```
OPENAI_API_KEY=sk-...
```

`.env` 已被 `.gitignore` 排除，不會進版本控制。若環境變數 `OPENAI_API_KEY`
已存在，則以環境變數優先。

互動查詢（預設只索引 59 條真實條文；加 `--distractors` 混入合成向量）：

```bash
python search_demo.py 特休沒休完會怎樣
python search_demo.py --distractors 50000 特休沒休完會怎樣
```

跑完整實驗（結果同時寫入 `experiment_result.md`）：

```bash
python run_experiment.py
```

**執行時間與記憶體**：HNSW 建索引**刻意用單執行緒**（多執行緒建圖每次結果
不同，recall 數字就不可重現），約需 2～4 分鐘，請耐心等。全程峰值記憶體
約 1.4 GB，8 GB 機器可跑，但建議先關掉吃記憶體的程式。

## 量測方法（與它的偏誤）

- **命中判定**沿用 Day 9 的同一把尺：top-1 chunk 的字元範圍是否與預期條文
  範圍重疊；top-1 若是合成向量直接算沒中
- **recall@k**：以 Flat（精確）的 top-k 為標準答案，量 HNSW 各檔 `efSearch`
  的重合比例
- **延遲**只計 `index.search()` 本身：問題向量事先算好，不含 embedding API
  往返；每個設定 15 題 × 20 次＝300 個樣本，取中位數與 p95
- **合成向量是隨機質心＋擾動**，不是真實文本的 embedding——所以本實驗
  只能主張「資料變多會讓精確掃描變慢、讓近似檢索掉題」，不能主張
  「資料變多會讓檢索變不準」（干擾項與真實語意無關，實驗也驗證了合成向量
  從未擠進 Flat 的 top-10）
- **量測環境**：文章數字在筆電**電池供電（省電降頻）**下量測，接電源會
  整體更快；絕對值請以你的機器為準，相對倍率與掉題結論不受影響
  （recall 表與掉題明細由固定 seed＋單執行緒建圖保證，可完整重現）

## 成本說明

使用 `text-embedding-3-small`（每百萬 token 0.02 美元）。首次執行需計算
59 條條文＋15 題問題的 embedding，實測 13,619 tokens ≈ 新台幣 0.0084 元；
50,000 個合成向量由本機生成，**0 元**。算過的向量快取在
`embeddings_cache.json`，重跑不重複計費。

## 聲明

`work_rules.md` 為技術示範用之虛構文件，非任何公司之實際規章。
