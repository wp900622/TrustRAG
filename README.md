# TrustRAG

2026 iThome 鐵人賽系列文 **「30 天打造 AI 後端：從 LLM、RAG 到 AI Agent」** 的範例程式。

一天一篇文章、一天一個資料夾，每個資料夾都能獨立執行。文章裡的每個數字都來自資料夾內
的實驗腳本，你可以自己跑一次對照。

## 每日導覽

| 資料夾 | 主題 | 一句話說明 |
|--------|------|-----------|
| [`day08_embedding/`](day08_embedding/) | Embedding + 餘弦相似度檢索 | 不用向量資料庫，問「特休沒休完會怎樣」，讓程式從 12 條規章裡找出該看哪一條，並用 10 題實測命中率 |
| [`day09_chunking/`](day09_chunking/) | Chunking 三種切法對比 | 同一份 59 條的規章、同一組 15 題，比較固定字數／固定字數+overlap／依條文結構化三種切法的 top-1 命中率與命中品質 |

每個資料夾內都有自己的 `README.md`，寫著該天的檔案說明、跑法與成本。

## 共通執行需求

三個步驟，每個資料夾都一樣（以 `day09_chunking/` 為例）：

```bash
cd day09_chunking
pip install -r requirements.txt
```

設定 OpenAI API key：複製 `.env.example` 為 `.env`，填入你自己的金鑰。

```
OPENAI_API_KEY=sk-...
```

- 需要 Python 3（開發與實測環境為 3.13）。
- 套件版本以各資料夾的 `requirements.txt` 為準（目前為 `openai`、`numpy`、`python-dotenv`）。
- `.env` 要放在**該天的資料夾內**，不是專案根目錄；已被 `.gitignore` 排除，不會進版本控制。
- 若系統環境變數已有 `OPENAI_API_KEY`，則以環境變數優先，不必另外建 `.env`。

跑起來：

```bash
python search_demo.py 特休沒休完會怎樣   # 互動 demo，看 top-3 結果與相似度
python run_experiment.py                # 跑完整題組，輸出命中率報表
```

## 成本

Embedding 使用 `text-embedding-3-small`。每天的實驗成本都遠低於新台幣 1 元，算過的向量會
快取在資料夾內的 `embeddings_cache.json`，重跑不重複計費。各資料夾的 README 有該天的實際
token 數與金額。

## 語料聲明

本 repo 內所有規章文件（`leave_policy.md`、`work_rules.md` 等）皆為**技術示範用之虛構
文件**，內容自行編寫，非任何公司之實際規章，不得作為任何法律或人事依據。

## 專案文件

- [`PRD.md`](PRD.md)：專案現況的需求與技術規格文件（功能說明、資料結構、函式介面、環境設定）
- [`CHANGELOG.md`](CHANGELOG.md)：版本變更紀錄
