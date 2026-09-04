# Day 8 — 問「特休沒休完會怎樣」，程式怎麼知道要看規章哪一段？

2026 iThome 鐵人賽「30 天打造 AI 後端：從 LLM、RAG 到 AI Agent」Day 8 範例程式。

不用向量資料庫，用 Embedding + 餘弦相似度，讓程式從一份（虛構的）《員工請假管理規範》
裡找出跟問題最相關的條文。

## 檔案說明

| 檔案 | 用途 |
|------|------|
| `leave_policy.md` | 虛構的《員工請假管理規範》共 12 條，整個系列的共用語料 |
| `questions.json` | 10 個測試問題：5 個直白問法＋5 個口語問法，附預期命中條文 |
| `rag_core.py` | 共用核心：切條文、取 Embedding（含本機快取）、餘弦相似度、檢索 |
| `search_demo.py` | 互動 demo：輸入問題，看 top-3 相關條文與相似度分數 |
| `run_experiment.py` | 小實驗：跑 10 題算命中率，輸出 Markdown 表格到 `experiment_result.md` |

## 怎麼跑

```bash
pip install -r requirements.txt
```

設定 OpenAI API key：複製 `.env.example` 為 `.env`，填入你的金鑰：

```
OPENAI_API_KEY=sk-...
```

`.env` 已被 `.gitignore` 排除，不會進版本控制。若環境變數 `OPENAI_API_KEY`
已存在，則以環境變數優先。

互動查詢：

```bash
python search_demo.py
```

或直接帶問題：

```bash
python search_demo.py 特休沒休完會怎樣
```

跑 10 題命中率實驗（結果同時寫入 `experiment_result.md`）：

```bash
python run_experiment.py
```

## 成本說明

使用 `text-embedding-3-small`（每百萬 token 0.02 美元）。整份規範＋10 個問題全部
embedding 一次約數千 token，成本遠低於新台幣 1 元。所有算過的向量會快取在
`embeddings_cache.json`，重跑不重複計費；文件內容改動的段落會自動重算。

## 聲明

`leave_policy.md` 為技術示範用之虛構文件，非任何公司之實際規章。
