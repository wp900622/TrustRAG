# TrustRAG — 30 天打造 AI 後端：從 LLM、RAG 到 AI Agent（範例程式）

GitHub 倉庫：https://github.com/wp900622/TrustRAG

## 專案概述

TrustRAG 是 2026 iThome 鐵人賽系列文「30 天打造 AI 後端：從 LLM、RAG 到 AI Agent」的
公開範例程式庫。每天一篇文章、對應一個獨立資料夾，讀者 clone 下來就能自己跑出文章裡
的同一組數字。

這個 repo 解決的問題是：RAG 的文章大多只給「可以動的 demo」，讀者看不到「換一個做法
結果會差多少」。這裡的每個資料夾都附一組固定的測試問題與可重跑的實驗腳本，把「準不
準」變成看得到的數字，而不是憑感覺驗收。

服務對象是想自己動手做 RAG 後端的工程師，以及日後接手維護這個 repo 的人。

## 背景與動機

- 這是單人維護的技術寫作專案，主要動機是：把系列文每天的實作留成可執行、可重現的
  程式碼，讀者不必從文章截圖裡猜細節。
- 關鍵取捨一：**每天一個資料夾，各自自給自足**（各有自己的 `README.md`、
  `requirements.txt`、`.env.example`、`rag_core.py`、語料與快取）。代價是資料夾之間
  有重複程式碼，換來的是讀者只想跑 Day 9 時不必先讀懂 Day 8，也不會因為共用模組改動
  而讓舊一天的範例跑不出文章裡的數字。
- 關鍵取捨二：**每天都設計「固定測試問題 + 命中率實驗」**，而不是只做 happy path
  demo。多花的成本是要維護標準答案；換來的是能寫出「這個做法在哪些題目上會失敗」，
  這是系列文的主要價值。
- 關鍵取捨三：**語料全部自行虛構**。用真實公司規章有法務與隱私風險，虛構語料還能刻意
  埋入交叉引用、公文體用詞等會讓檢索失敗的結構，方便做實驗。
- 不做會怎樣：文章與程式碼會脫節，讀者無法驗證結論，系列文的可信度歸零。

---

## 第一區塊：功能說明（所有人適用）

### 功能清單

專案目前有兩個範例資料夾，各自是一個可獨立執行的小工具。

#### 共通功能（兩個資料夾都有）

| 功能 | 使用者可以做什麼 |
|------|-----------------|
| 互動檢索 demo（`search_demo.py`） | 輸入一個問題，看程式從規章裡找出最相關的前 3 段，每段附相似度分數。可以直接在命令列帶問題跑一次，也可以進互動模式連續問 |
| 命中率實驗（`run_experiment.py`） | 一次跑完整組測試問題，看每題「程式找到的段落是不是預期那一段」，並輸出 Markdown 表格報表 |
| 向量本機快取 | 算過的向量存在 `embeddings_cache.json`，重跑不會重複呼叫 API、不重複計費；語料改了哪一段，只有那一段會重算 |
| 成本顯示 | 每次執行都會印出本次消耗多少 token、約等於多少美元／新台幣，讓使用者知道實驗花了多少錢 |

#### `day08_embedding/`：Embedding + 餘弦相似度檢索

- 語料是虛構的《員工請假管理規範》共 12 條（`leave_policy.md`），刻意做成「一條規定
  剛好一段」，所以這一天完全不處理切分問題。
- 測試題 10 題（`questions.json`）：5 題「直白問法」（用詞和條文幾乎一樣）＋5 題
  「口語問法」（刻意用文件裡沒出現過的講法，例如問「加班換的假放到過期會怎樣」，
  文件寫的是「補休」）。
- 命中判定：top-1 條文的標題是否等於該題標注的預期條文。
- 現況備註：`experiment_result.md` 目前**不在 repo 內**，需自行執行
  `python run_experiment.py` 產生；`day08_draft.md`（文章草稿）內仍有數處
  「【待填】」等待回填實驗數字，並保留了「卡關段 A／B 二選一」兩個版本尚未取捨。

#### `day09_chunking/`：三種 Chunking 切法對比

- 語料換成規模更接近真實的虛構《員工工作規則》：11 章、59 條、約 10,100 字
  （`work_rules.md`）。其中「第五章 請假管理」沿用 Day 8 的 12 條內容重新編號為
  第 21～32 條，讓兩天的題目可以對照。
- 三種切法：
  1. **固定字數**：每 300 字硬切一塊，完全不管句子與條文邊界（實測 34 塊、平均 298 字）
  2. **固定字數＋overlap**：每 300 字一塊，相鄰兩塊重疊 100 字（實測 51 塊、平均 297 字）
  3. **結構化（依條文）**：一條規定＝一塊，依 Markdown 的「### 第 N 條」標題切
     （實測 59 塊、平均 161 字）
- 測試題 15 題（`questions.json`）：10 題沿用 Day 8（標成 `group: 沿用`）＋5 題針對新
  增章節（出勤、加班、出差、獎懲、離職，標成 `group: 新增`）。
- 命中判定：三種切法的邊界都不同，沒辦法直接比條號，所以每個 chunk 都記著自己在原文
  中的字元範圍，判定標準是「top-1 chunk 的字元範圍是否與預期條文的範圍重疊」——從寬
  認定，沾到就算。三種切法用同一把尺，數字才有可比性。
- 實驗已實跑完成，結果在 `experiment_result.md`：固定字數 10/15、固定＋overlap 9/15、
  結構化 10/15。三種切法命中率幾乎打平，但命中的品質差很多（固定切法的「命中」常常是
  一塊 300 字橫跨五條規定沾到答案；結構化切法命中時拿到的是乾淨的一條）。overlap 反而
  是三者最低。文章草稿在 `day09_draft.md`，僅剩一處「【待填：Day 8 實驗命中率】」待
  Day 8 實驗數字回填。

### 使用者流程

讀者從文章點進 repo 之後的完整步驟：

1. 讀根目錄 `README.md`，從導覽表挑一天的資料夾（例如 `day09_chunking/`）。
2. 進入該資料夾，讀資料夾內的 `README.md`（該天的檔案說明、跑法、成本與聲明）。
3. 安裝套件：`pip install -r requirements.txt`。
4. 準備金鑰：把 `.env.example` 複製成 `.env`，填入自己的 OpenAI API key。若系統環境變數
   已有 `OPENAI_API_KEY`，則以環境變數優先，不需要建 `.env`。
5. 想玩看看 → 執行 `python search_demo.py 你的問題`，看 top-3 結果與分數。
   Day 9 可加 `--chunker fixed|overlap|structure` 切換切法比較差異。
6. 想驗證文章數字 → 執行 `python run_experiment.py`，終端機印出報表，同時寫入
   `experiment_result.md`，可與文章裡的表格對照。
7. 想改成自己的語料 → 換掉該天的語料檔與 `questions.json`（格式見「資料結構」），
   重跑實驗即可。

### 畫面說明

本案的使用者介面是**終端機命令列（CLI）**，沒有網頁前端。互動方式有兩種：

**一、單次查詢**：命令列直接帶問題，跑完就結束。

```
$ python search_demo.py 特休沒休完會怎樣
切法：結構化（依條文）｜chunk 數：59｜本次索引消耗 0 tokens（快取命中 59 塊）
  1. [0.5487] 第 22 條｜第 22 條（特休假）員工於服務滿六個月後…
  2. [0.5456] 第 26 條｜第 26 條（補休）加班得選擇換取補休…
  3. [0.5302] 第 23 條｜第 23 條（特休假之遞延與結算）…
```

畫面上會看到：本次用哪種切法、索引切成幾塊、快取命中幾塊、本次花掉多少 token，以及
前 3 名結果的「相似度分數 ｜ 涵蓋到的條文 ｜ 內容前 40 字預覽」。

**二、互動模式**：不帶問題執行，出現 `> ` 提示符後可連續輸入問題，直接按 Enter（空行）
離開。索引只建一次，後續每題都沿用同一份向量。

**三、實驗報表**（非互動，屬於輸出）：`run_experiment.py` 把結果同時印在終端機與寫入
`experiment_result.md`，內容包含三種切法的總表、各切法逐題明細（命中用 ✅／❌ 標示）、
沒命中題目的 top-3 攤開，以及本次 API 成本統計。Markdown 表格可直接貼進文章。

---

## 第二區塊：技術規格（工程師適用）

### 技術架構

| 層 | 使用技術 |
|----|---------|
| 執行環境 | Python（開發與實測環境為 3.13；程式碼使用內建泛型註解與 `str.removeprefix`，語法上需 3.9 以上） |
| Embedding 模型 | OpenAI `text-embedding-3-small`，透過 `openai` SDK（`>=1.40`）呼叫 |
| 向量運算 | `numpy`（`>=1.26`）手算餘弦相似度，刻意不引入向量資料庫（那是後續天數的主題） |
| 設定載入 | `python-dotenv`（`>=1.0`）讀取資料夾內的 `.env` |
| 前端 | 無，介面為 CLI（`argparse` / `sys.argv` + `input()`） |
| 資料庫 | 無。向量以本機 JSON 檔 `embeddings_cache.json` 快取，語料是 Markdown 檔，測試題是 JSON 檔 |

模組分工（每個資料夾各有一份，互不 import）：

| 檔案 | 職責 |
|------|------|
| `rag_core.py` | 讀語料、延遲建立 OpenAI client、批次取得 Embedding（含快取）、餘弦相似度、`search()`、token 換算成本 |
| `chunkers.py`（僅 Day 9） | 三種切法、條文範圍解析（ground truth）、範圍重疊判定、涵蓋條文標籤、切法註冊表 `CHUNKERS` |
| `search_demo.py` | CLI 互動 demo |
| `run_experiment.py` | 跑完整題組、算命中率、產出 Markdown 報表 |

檢索流程（Day 9）：讀 `work_rules.md` 全文 → 用選定的切法切成 chunk 清單 → 批次取得每
塊的向量（快取優先）→ 問題轉向量 → 對所有 chunk 算餘弦相似度並排序 → 取 top-k。索引
是純記憶體的 list，沒有 ANN 索引，每次查詢都是全量掃描。

### 資料結構

沒有資料庫，以下是程式內流通的資料格式與檔案格式。

**chunk（Day 9，`chunkers.py` 三種切法的共同輸出元素）**

| 欄位 | 型別 | 說明 |
|------|------|------|
| `text` | `str` | 這一塊的文字內容 |
| `start` | `int` | 在原文中的起始字元位置（含） |
| `end` | `int` | 在原文中的結束字元位置（不含），即半開區間 `[start, end)` |

**article（Day 9，`parse_articles()` 解析出的標準答案）**

| 欄位 | 型別 | 說明 |
|------|------|------|
| `no` | `int` | 條號，從「### 第 N 條」標題解析 |
| `title` | `str` | 條文標題，例如 `第 24 條（病假）` |
| `start` / `end` | `int` | 該條在原文中的字元範圍，用來與 chunk 範圍比對是否重疊 |

**article（Day 8，`load_articles()` 的輸出元素）**

| 欄位 | 型別 | 說明 |
|------|------|------|
| `title` | `str` | 條文標題，例如 `第 4 條（病假）` |
| `text` | `str` | 標題 + 內文，實際送去 embedding 的文字 |

**檢索結果**：`search()` 回傳的每個元素是「原 chunk／article 的所有欄位」再加上
`score`（`float`，餘弦相似度，越接近 1 越相關）。

**`questions.json`（測試題）**

| 欄位 | 型別 | 說明 |
|------|------|------|
| `id` | `int` | 題號 |
| `type` | `str` | `直白`／`口語`，用來分組統計 |
| `group` | `str` | 僅 Day 9：`沿用`（Day 8 的 10 題）／`新增`（針對新章節的 5 題） |
| `question` | `str` | 使用者問法 |
| `expected` | `str` | 預期命中的條文，格式為 `第 N 條`（實驗用 `第 (\d+) 條` 取出條號） |

**`embeddings_cache.json`（向量快取，已被 `.gitignore` 排除）**

- 格式：`{ 快取 key: [float, ...] }`。
- key 產生方式：`sha256("{模型名稱}:{文字內容}")` 取前 16 個十六進位字元。因此語料改動
  只會讓被改到的段落 cache miss，其餘照用；換模型也不會拿到舊模型的向量。

**語料 Markdown 的標題約定（結構化切法依賴這個約定）**

- Day 8 `leave_policy.md`：`# 文件標題` + `## 第 N 條（標題）`，第一塊前言不進檢索範圍。
- Day 9 `work_rules.md`：`# 文件標題` + `## 第 N 章 章名` + `### 第 N 條（標題）`。
  只有 `###` 層級的區塊會進檢索範圍，前言與章標題不進。換自己的語料時要維持這個層級
  約定，否則 `chunk_by_structure()` 與 `parse_articles()` 會抓不到東西。

### API 規格

本案**沒有自建 HTTP API endpoint**（無 web 服務、無對外埠口）。以下記錄唯一的對外 API
呼叫，以及模組間的函式介面。

**對外呼叫：OpenAI Embeddings API**

| 項目 | 內容 |
|------|------|
| 呼叫方式 | `openai` SDK：`client.embeddings.create(model=..., input=[...])` |
| endpoint | `POST https://api.openai.com/v1/embeddings`（由 SDK 處理） |
| model | `text-embedding-3-small` |
| 請求參數 | `input`：本次 cache miss 的文字陣列（一次批次送出，減少往返） |
| 使用到的回應欄位 | `data[i].embedding`（向量，依輸入順序對應）、`usage.total_tokens`（計費用量） |
| 認證 | 環境變數 `OPENAI_API_KEY`，由 SDK 自動讀取；程式不接受把金鑰寫在參數或程式碼裡 |
| 錯誤處理 | 未設定金鑰時，`get_client()` 直接丟 `RuntimeError` 並提示「複製 .env.example 為 .env」。client 為延遲建立，只讀語料、不打 API 的操作不需要金鑰 |

**`rag_core.py` 公開函式**

| 函式 | 簽章要點 | 說明 |
|------|---------|------|
| `load_document(path)` / `load_articles(path)` | → `str`（Day 9）／`list[dict]`（Day 8） | 讀語料。Day 9 回傳全文由切法處理，Day 8 直接切成一條一段 |
| `get_embeddings(texts)` | `list[str]` → `(list[np.ndarray], used_tokens, cache_hits)` | 批次取向量，快取優先；回傳本次實際消耗 token 與快取命中筆數 |
| `cosine_similarity(a, b)` | `np.ndarray, np.ndarray` → `float` | 餘弦相似度 |
| `search(question, chunks, chunk_vectors, top_k=3)` | → `(list[dict], used_tokens)` | 問題轉向量後全量比對，回傳前 top_k 筆（含 `score`） |
| `cost_usd(tokens)` | `int` → `float` | token 數換算美元 |

**`chunkers.py` 公開函式（Day 9）**

| 函式 | 說明 |
|------|------|
| `chunk_fixed(text, size=300)` | 固定字數硬切 |
| `chunk_fixed_overlap(text, size=300, overlap=100)` | 固定字數＋重疊，步進為 `size - overlap` |
| `chunk_by_structure(text)` | 依 `### 第 N 條` 標題切，一條一塊 |
| `parse_articles(text)` | 解析全部條文的條號、標題與字元範圍（實驗標準答案） |
| `spans_overlap(a_start, a_end, b_start, b_end)` | 兩個半開區間是否重疊（命中判定的核心） |
| `articles_covering(chunk, articles)` / `coverage_label(chunk, articles)` | 查出 chunk 沾到哪幾條 / 濃縮成人看的標籤，如「第 24～28 條」 |
| `CHUNKERS` | 切法註冊表：`key → (顯示名稱, 函式)`，`search_demo.py --chunker` 與實驗共用 |

**CLI 介面**

| 指令 | 參數 | 行為 |
|------|------|------|
| `python search_demo.py [問題...]` | 兩天都有；不帶問題進互動模式 | 印出 top-3 結果 |
| `python search_demo.py --chunker {fixed,overlap,structure}` | 僅 Day 9，預設 `structure` | 指定切法 |
| `python run_experiment.py` | 無參數 | 跑完整題組，報表印出並寫入 `experiment_result.md` |

### 環境設定

**環境變數**

| key | 用途 | 必填 |
|-----|------|------|
| `OPENAI_API_KEY` | 呼叫 OpenAI Embeddings API 的金鑰 | 是（除非全部向量都已在快取中） |

- 每個 day 資料夾各有一份 `.env.example` 作為範本，複製成同資料夾的 `.env` 後填值。
- `rag_core.py` 以 `load_dotenv(Path(__file__).parent / ".env")` 載入，因此 `.env` 必須放在
  該 day 資料夾內（不是專案根目錄）。系統環境變數若已設定同名變數，以環境變數優先。
- `.env` 與 `embeddings_cache.json`、`__pycache__/`、`*.pyc` 已由根目錄 `.gitignore` 排除。
  本文件與 repo 內任何檔案都不得出現實際金鑰。

**程式內可調參數（非環境變數，直接改常數）**

| 常數 | 位置 | 預設 | 用途 |
|------|------|------|------|
| `EMBEDDING_MODEL` | `rag_core.py` | `text-embedding-3-small` | 使用的 embedding 模型；改動後快取 key 會全部變化，等於重新計算 |
| `USD_PER_MILLION_TOKENS` | `rag_core.py` | `0.02` | 模型單價（2026-09 官網查閱），只用於成本顯示 |
| `USD_TO_TWD` | `rag_core.py` | `31` | 粗估匯率，只用於文章內成本示意 |
| `CHUNK_SIZE` | `chunkers.py`（Day 9） | `300` | 固定字數切法每塊字元數 |
| `OVERLAP` | `chunkers.py`（Day 9） | `100` | overlap 切法的重疊字元數 |

**成本預期**：Day 8 全部語料＋10 題約數千 token；Day 9 三種切法各建一次索引＋15 題實測
為 40,875 token，約 0.0008 美元（新台幣 0.03 元）。快取命中的部分不計費。
