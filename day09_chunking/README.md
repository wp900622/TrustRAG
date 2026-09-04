# Day 9 — 同一份文件、同一組問題：三種切法，命中率差多少？

2026 iThome 鐵人賽「30 天打造 AI 後端：從 LLM、RAG 到 AI Agent」Day 9 範例程式。

Day 8 的語料一條剛好一段，是運氣不是方法。今天把語料換成一份 59 條的虛構
《員工工作規則》，用三種切法（固定字數／固定字數+overlap／依條文結構化）
建索引，跑同一組測試問題，實測切法對命中率的影響。

## 檔案說明

| 檔案 | 用途 |
|------|------|
| `work_rules.md` | 虛構的《員工工作規則》共 11 章 59 條；「請假管理」章沿用 Day 8 的 12 條內容重新編號 |
| `questions.json` | 15 個測試問題：10 題沿用 Day 8（5 直白＋5 口語）＋5 題針對新章節，附預期命中條文 |
| `chunkers.py` | 三種切法實作＋條文範圍解析（每個 chunk 都帶在原文中的字元範圍） |
| `rag_core.py` | 共用核心：取 Embedding（含本機快取）、餘弦相似度、檢索（自 Day 8 泛化） |
| `search_demo.py` | 互動 demo：`--chunker` 選切法，輸入問題看 top-3 chunk 與分數 |
| `run_experiment.py` | 實驗：三種切法 × 15 題的命中率對比，輸出 Markdown 表格到 `experiment_result.md` |

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

互動查詢（預設用結構化切法，`--chunker fixed` / `--chunker overlap` 可切換）：

```bash
python search_demo.py 特休沒休完會怎樣
```

跑三種切法的命中率對比實驗（結果同時寫入 `experiment_result.md`）：

```bash
python run_experiment.py
```

## 命中怎麼判定

三種切法的 chunk 邊界都不同，沒辦法直接比「條號」。每個 chunk 都帶著它在
原文中的字元範圍，判定標準是：**top-1 chunk 的範圍是否與預期條文的範圍重疊**
（從寬認定：沾到就算）。三種切法用同一把尺，數字才有可比性。

## 成本說明

使用 `text-embedding-3-small`（每百萬 token 0.02 美元）。三種切法各建一次索引
加上 15 個問題，全部 embedding 一次約數萬 token，成本仍遠低於新台幣 1 元。
所有算過的向量會快取在 `embeddings_cache.json`，重跑不重複計費；文件內容改動
的段落會自動重算。

## 聲明

`work_rules.md` 為技術示範用之虛構文件，非任何公司之實際規章。
