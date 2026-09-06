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
- 關鍵取捨二：**每天都設計「固定測試問題 + 量化實驗」**，而不是只做 happy path
  demo。多花的成本是要維護標準答案；換來的是能寫出「這個做法在哪些題目上會失敗」，
  這是系列文的主要價值。指標會隨主題長出來（已更新）：命中率（Day 8）→ 加命中純度
  （Day 9）→ 加查詢延遲與 recall@k（Day 10）；但**題目與命中判定始終是同一把尺**，
  跨天的數字才可以直接對照。
- 關鍵取捨三：**語料全部自行虛構**。用真實公司規章有法務與隱私風險，虛構語料還能刻意
  埋入交叉引用、公文體用詞等會讓檢索失敗的結構，方便做實驗。
- 不做會怎樣：文章與程式碼會脫節，讀者無法驗證結論，系列文的可信度歸零。

---

## 第一區塊：功能說明（所有人適用）

### 功能清單

專案目前有三個範例資料夾（已更新），各自是一個可獨立執行的小工具。

#### 共通功能（三個資料夾都有；已更新）

| 功能 | 使用者可以做什麼 |
|------|-----------------|
| 互動檢索 demo（`search_demo.py`） | 輸入一個問題，看程式從規章裡找出最相關的前 3 段，每段附相似度分數。可以直接在命令列帶問題跑一次，也可以進互動模式連續問 |
| demo 的做法切換參數（Day 9 起才有；已更新） | Day 9 可加 `--chunker fixed\|overlap\|structure` 換切法；Day 10 可加 `--distractors N`（預設 0、最多 50000）把 N 個合成干擾向量灌進索引，親眼確認「5 萬個假向量混進來，真實問題的 top-3 仍然是真實條文」 |
| 實驗腳本（`run_experiment.py`；已更新） | 一次跑完整組測試問題並輸出 Markdown 表格報表。指標隨天數擴充：Day 8／9 是命中率（Day 9 另加命中純度）；**Day 10 起再加上單題查詢延遲（中位數／p95）與 recall@k**，因為 Day 10 比的是「精確 vs 近似」，光看命中率看不出速度與掉題的取捨 |
| 向量本機快取 | 算過的向量存在 `embeddings_cache.json`，重跑不會重複呼叫 API、不重複計費；語料改了哪一段，只有那一段會重算 |
| 成本顯示（已更新） | 每次執行都會印出本次消耗多少 token、約等於多少美元／新台幣。Day 9 起的實驗報表把「計費 tokens」與「快取命中塊數」拆成兩項分別列出，並附一句「刪除 `embeddings_cache.json` 重跑即可重現無快取的完整成本」——因為全部命中快取時金額會是 0 元，若不分開列，讀者會誤以為這個實驗本來就免費。Day 10 沿用同一慣例，並額外明講「50,000 個合成向量由本機生成，0 元」 |

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
- 命中判定（第一指標）：三種切法的邊界都不同，沒辦法直接比條號，所以每個 chunk 都記著
  自己在原文中的字元範圍，判定標準是「top-1 chunk 的字元範圍是否與預期條文的範圍重疊」
  ——從寬認定，沾到就算。三種切法用同一把尺，數字才有可比性。
- **純度判定（第二指標，Day 9 新增）**：只在 top-1 命中時計算，定義為「chunk 範圍內屬於
  預期條文的字元比例」，算式是「chunk 與預期條文重疊的字元數 ÷ chunk 總字元數」。設計
  理由：命中率只回答「找不找得到」，純度回答「找到的那一塊有多乾淨」。大塊容易沾到答案
  而讓命中率被高估，純度就是把這個代價量出來。報表以「命中題的平均純度」呈現，該切法
  全數未命中時顯示「—」。
- 實驗已實跑完成（已更新），結果在 `experiment_result.md`：
  - 命中率：固定字數 10/15、固定＋overlap 9/15、結構化 10/15，三者幾乎打平，overlap
    反而是三者最低。
  - 命中純度：固定字數 35%、固定＋overlap 32%、結構化 100%。
  - 原本只能用文字描述的「命中品質差很多」，現在有數字了：命中率打平、純度卻差近 3 倍
    ——這組對比是 Day 9 文章的核心論點。最極端的單題是 Q1（問病假診斷證明），固定切法
    判定為命中，但那一塊 300 字橫跨第 24～28 條，純度只有 6%；同一題結構化切法拿到的是
    第 24 條本人，純度 100%。
- 文章草稿（已更新）：`day09_draft.md` 已以鐵人賽得獎為目標全文重寫，結構為反直覺鉤子
  標題、3 行冷開場、「本篇你會帶走」、切法差異的文字圖解、雙計分表（命中率＋純度）、
  切法決策速查表、成本交代、讀者互動問題收尾。仍剩一處「【待填：Day 8 實驗命中率】」
  待 Day 8 定稿後回填。

#### `day10_vector_index/`：FAISS 向量索引 — Flat（精確）vs HNSW（近似）（Day 10 新增）

Day 8／9 的檢索都是「跟索引裡每一塊算一次相似度再排序」，59 塊當然夠快。Day 10 要回答
的問題是：**索引長到五萬塊時，暴力掃描還撐不撐得住？換成近似檢索快多少、代價是什麼？**

- **語料與題目完全沿用 Day 9**：同一份 59 條的虛構《員工工作規則》（`work_rules.md`）、
  同一組 15 題（`questions.json`）。今天比的是引擎不是切法，題目動了數字就不可對照。
- **切法只留 Day 9 的贏家**：`chunkers.py` 自 Day 9 精簡，只保留「結構化（依條文）」
  一種，固定字數與 overlap 兩種不再帶著走。
- **0 元把索引從 59 塊撐到 50,059 塊**（`synth_vectors.py`）：用「隨機質心＋擾動」合成
  50,000 個干擾向量——先抽 500 個隨機方向當質心，每個質心周圍再生 100 個擾動點
  （擾動幅度 0.35），讓資料有叢集結構。刻意不用純高斯亂數，因為高維隨機向量彼此近乎
  正交、毫無叢集，是 HNSW 圖導航的最壞情況，量出來不能代表真實 embedding 的行為。
  **固定 seed（42）且不落地存檔**：約 308 MB 不進 repo，讀者 clone 後重新生成就是同一
  組資料。合成向量不花任何 API 費用。
- **兩種索引，同一套介面**：精確的 `IndexFlatIP`（內積暴力掃描，結果保證正確）與近似的
  `IndexHNSWFlat`（M=32、efConstruction=200，圖導航）。查詢都是同一行
  `index.search()`，只換索引型別，差異才只屬於演算法。
- **判定沿用 Day 9 的同一把尺**：top-1 chunk 的字元範圍是否與預期條文範圍重疊；
  top-1 若落在合成向量上直接算沒中。第二把尺是 **recall@k**：以 Flat 的 top-k 當標準
  答案，量 HNSW 各檔 `efSearch` 的重合比例——前者回答「答案對不對」，後者回答「近似
  離精確有多遠」。
- 實驗已實跑完成，結果在 `experiment_result.md`：
  - **延遲**（單題、top-10，15 題 × 20 次＝300 個樣本）：Flat 單執行緒中位數 24.58 ms
    （p95 37.47 ms）；HNSW 依 `efSearch` 從 0.566 ms（ef=8）到 5.062 ms（ef=256），
    即**比精確掃描快 5～43 倍**。另一個實測到但未進一步驗證原因的現象：Flat 開滿
    14 執行緒反而更慢（38.58 ms、0.6x），單題查詢下多執行緒沒有帶來效益。
  - **掉題代價**：`efSearch=8` 與 16（**16 是 FAISS 預設值**）recall 只有 0.73／0.80，
    15 題掉到 7/15；ef=32 為 0.87／8 題。掉題的方式是關鍵——top-1 不是「排錯的條文」，
    而是完全無關的合成向量（例如 Q1 病假診斷證明變成「合成向量 #8000」），且掉的
    Q11、Q13 在 Flat 底下還是全場相似度最高的兩題（0.72、0.73），可見掉題與題目難度
    無關、與圖的形狀有關。
  - **歸零點**：`efSearch` 掃 8／16／32／64／128／256 六檔，**開到 64 時 recall@1/3/10
    全部回到 1.00、命中率與 Flat 完全一致（10/15），延遲 2.007 ms 仍比精確掃描快 12
    倍**——這份資料上「準」與「快」可以同時要，但歸零點的位置必須自己量，不是查文件
    查得到的。
  - **合成向量沒有污染答案**：15 題的 Flat 精確結果中，合成向量在 top-1／top-3（45 個
    位置）／top-10（150 個位置）出現次數都是 **0**；Flat 命中 10/15，與 Day 9 結構化
    切法在 59 塊索引下的結果**完全一致**。所以 HNSW 掉的每一題都只能算在「近似」頭上，
    不是干擾向量搶走答案。
  - **建索引成本**：Flat 0.17 秒；HNSW 265 秒（約 3～5 分鐘）。HNSW **刻意用單執行緒
    建圖**——多執行緒的插入順序每次不同，建出來的圖與掉的題也不同，recall 數字就不可
    重現；代價是建索引從約 40 秒變成約 2 分鐘。全程峰值記憶體約 1.4 GB（1353 MB）。
- **量測偏誤自白（寫進 README 與文章，不藏起來）**：
  1. 延遲數字在筆電**電池供電（省電降頻）**下量測，接電源會整體更快；絕對值請以自己的
     機器為準，相對倍率與掉題結論不受影響（固定 seed＋單執行緒建圖保證可完整重現）。
  2. 干擾向量是合成的、與真實語意無關，所以本實驗只能主張「資料變多會讓精確掃描變慢、
     讓近似檢索掉題」，**不能**主張「資料變多會讓檢索變不準」。
  3. 延遲只計 `index.search()` 本身，問題向量事先算好，不含 embedding API 往返——量的
     是索引，不是網路。
- 文章草稿：`day10_draft.md` 已依 `article_template.md` 的骨架完成全文（鉤子標題、本篇
  你會帶走、做法、實驗設計：同一把尺、三個發現、決策速查表、成本、小結與明日預告
  Chroma），無「【待填】」殘留。

#### 根目錄檔案（給維護者與交接者用）

| 檔案 | 用途 |
|------|------|
| `README.md` | repo 入口：每日導覽表、共通執行需求、成本、語料聲明、專案文件索引 |
| `PRD.md` | 本文件，只反映專案當下現狀（功能、資料結構、函式介面、環境設定） |
| `CHANGELOG.md` | 版本變更紀錄（表格：版本／日期／變更說明），歷史演變只寫在這裡 |
| `article_template.md` | **系列文章模板（Day 9 收尾時新增）**：從 Day 9 草稿的骨架抽出，Day 10 起每天複製成 `dayNN_主題/dayNN_draft.md` 照著填。內含固定章節（本篇你會帶走／為什麼今天做這件事／做法／實驗設計：同一把尺／結果／決策速查表／這次花了多少錢／小結／完整程式碼）、各節的 HTML 註解寫作指引（例如標題公式、結果段要有反直覺或失敗案例、成本段要把計費與快取命中分開列），以及發文前 checklist。填完後把指引註解刪掉即可發文 |
| `.gitignore` | 排除 `.env`、`embeddings_cache.json`、`__pycache__/`、`*.pyc` |

`article_template.md` 存在的理由：讓 30 天的文章長同一個骨架，讀者養成閱讀慣性，也避免
每天重新想結構時漏掉系列的招牌段（固定測試題、判定標準、實跑成本）。

### 使用者流程

讀者從文章點進 repo 之後的完整步驟：

1. 讀根目錄 `README.md`，從導覽表挑一天的資料夾（例如 `day09_chunking/`）。
2. 進入該資料夾，讀資料夾內的 `README.md`（該天的檔案說明、跑法、成本與聲明）。
3. 安裝套件：`pip install -r requirements.txt`。
4. 準備金鑰：把 `.env.example` 複製成 `.env`，填入自己的 OpenAI API key。若系統環境變數
   已有 `OPENAI_API_KEY`，則以環境變數優先，不需要建 `.env`。
5. 想玩看看 → 執行 `python search_demo.py 你的問題`，看 top-3 結果與分數。
   Day 9 可加 `--chunker fixed|overlap|structure` 切換切法比較差異；
   Day 10 可加 `--distractors 50000` 把合成干擾向量灌進索引再查（已更新）。
6. 想驗證文章數字 → 執行 `python run_experiment.py`，報表先寫入 `experiment_result.md`
   再印到終端機，可與文章裡的表格對照。**Day 10 要跑約 5 分鐘、峰值記憶體約 1.4 GB**
   （HNSW 單執行緒建圖換取可重現性），建議先關掉吃記憶體的程式（已更新）。
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

Day 10 的第一行改為顯示 FAISS 索引規模，並可用 `--distractors` 灌入合成向量（已更新）：

```
$ python search_demo.py --distractors 50000 特休沒休完會怎樣
索引：50,059 塊（真實 59＋合成 50,000）｜本次索引消耗 0 tokens（快取命中 59 塊）
  1. [0.5487] 第 22 條｜第 22 條（特休假）員工於服務滿六個月後…
```

若某一名剛好落在合成向量上，該行會直接顯示「合成向量 #12345」，讓讀者看得出檢索走丟。

**二、互動模式**：不帶問題執行，出現 `> ` 提示符後可連續輸入問題，直接按 Enter（空行）
離開。索引只建一次，後續每題都沿用同一份向量。

**三、實驗報表**（非互動，屬於輸出；本節已更新）：`run_experiment.py` 產生的報表**先寫入
`experiment_result.md`、再印到終端機**——報表含 ✅／❌ 等非 ASCII 字元，stdout 被重導向到
檔案時可能因 Windows 預設 cp950 編碼而拋錯，先寫檔可確保報表不會跟著 print 一起遺失
（Day 9／Day 10 皆為此順序）。

Day 8／Day 9 報表內容：

- 三種切法的總表：chunk 數、平均字數、分組命中率（沿用 10 題／其中直白／其中口語／
  新增 5 題／全部），以及**「命中純度」欄**（命中題的平均純度，如 `35%`）。
- 各切法逐題明細：命中用 ✅／❌ 標示，並多一個**「純度」欄**（未命中該欄顯示「—」）。
- 沒命中題目的 top-3 攤開。
- 本次 API 成本統計：**「計費 tokens」與「快取命中塊數」分開列**，例如「本次 API 計費：
  0 tokens ≈ 0.000000 美元（另有 144 塊索引向量命中本機快取、未計費；刪除
  `embeddings_cache.json` 重跑即可重現無快取的完整成本）」。

Day 10 報表內容（Day 10 新增）：

- 實驗設定：索引組成（59 條真實＋50,000 合成、1536 維 float32＝308 MB）、建索引耗時、
  延遲量測方法，以及**執行環境自動落檔**（CPU 型號、執行緒數、Python／faiss-cpu／numpy
  版本、峰值記憶體）——換機器重跑時，數字對不上可以先看是不是環境不同。
- 「合成向量有沒有污染答案」段：合成向量出現在 Flat top-1／top-3／top-10 的次數，以及
  Flat 的 Day 9 題組命中率。
- 表一「單題查詢延遲」：引擎、執行緒數、中位數、p95、相對 Flat 單執行緒的倍率。
- 表二「recall 與命中率 vs efSearch」：每檔 `efSearch` 的中位數延遲、recall@1／@3／@10、
  Day 9 題組命中數。
- 「掉題解剖」：Flat 有中、HNSW 沒中的每一題，列出 efSearch、題目、預期條文、HNSW 的
  top-1 變成什麼（全部命中時顯示「（無：所有 efSearch 檔位下…）」）。
- 逐題明細（Flat 精確檢索）：top-1 涵蓋條文、相似度、✅／❌。
- 成本統計：同 Day 9 慣例，計費 tokens 與快取命中筆數分開列，並註明合成向量 0 元與
  「刪除 `embeddings_cache.json` 重跑即可重現無快取的完整成本」。

Markdown 表格可直接貼進文章。

---

## 第二區塊：技術規格（工程師適用）

### 技術架構

| 層 | 使用技術 |
|----|---------|
| 執行環境 | Python（開發與實測環境為 3.13，實測 3.13.14；程式碼使用內建泛型註解與 `str.removeprefix`，語法上需 3.9 以上） |
| Embedding 模型 | OpenAI `text-embedding-3-small`，透過 `openai` SDK（`>=1.40`）呼叫 |
| 向量運算（已更新） | `numpy`（`>=1.26`；Day 10 實測 2.5.2）。**Day 8／9**：用 numpy 手算餘弦相似度、全量掃描，刻意不引入任何索引函式庫。**Day 10 起**：改用 `faiss-cpu`（釘死 `==1.15.0`）的索引——精確的 `IndexFlatIP` 與近似的 `IndexHNSWFlat`；向量改為 float32 並預先 L2 正規化，正規化後內積等價於餘弦相似度。faiss 是**索引函式庫，仍然不是向量資料庫**（metadata、持久化、過濾是 Day 11 起的主題） |
| 向量索引（Day 10 起新增） | `faiss-cpu==1.15.0`。釘死版本的理由：本日文章的延遲與 recall 數字都在這個版本實測，換版本數字會漂移，讀者就無法重現 |
| 設定載入 | `python-dotenv`（`>=1.0`）讀取資料夾內的 `.env` |
| 前端 | 無，介面為 CLI（`argparse` / `sys.argv` + `input()`） |
| 資料庫 | 無。向量以本機 JSON 檔 `embeddings_cache.json` 快取，語料是 Markdown 檔，測試題是 JSON 檔；FAISS 索引只存在記憶體，程式結束即消失 |

模組分工（每個資料夾各有一份，互不 import）：

| 檔案 | 職責 |
|------|------|
| `rag_core.py` | 讀語料、延遲建立 OpenAI client、批次取得 Embedding（含快取）、token 換算成本。相似度計算：Day 8／9 為 numpy 手算 `cosine_similarity()`；**Day 10 改為建立 FAISS 索引（`build_flat()`／`build_hnsw()`）並由索引執行檢索**（已更新） |
| `chunkers.py`（Day 9、Day 10；已更新） | 條文範圍解析（ground truth）、範圍重疊判定、涵蓋條文標籤。Day 9 有三種切法與切法註冊表 `CHUNKERS`；**Day 10 精簡為只留結構化切法，無 `CHUNKERS`** |
| `synth_vectors.py`（僅 Day 10；Day 10 新增） | 生成合成干擾向量（隨機質心＋擾動、固定 seed、單位向量、不落地存檔） |
| `search_demo.py` | CLI 互動 demo（Day 10 走 FAISS Flat 索引，支援 `--distractors`） |
| `run_experiment.py` | 跑完整題組、產出 Markdown 報表。指標：Day 8 命中率；Day 9 加命中純度；**Day 10 加延遲（中位數／p95）、recall@k、掉題解剖**（已更新） |

檢索流程（Day 9）：讀 `work_rules.md` 全文 → 用選定的切法切成 chunk 清單 → 批次取得每
塊的向量（快取優先）→ 問題轉向量 → 對所有 chunk 算餘弦相似度並排序 → 取 top-k。索引
是純記憶體的 list，沒有 ANN 索引，每次查詢都是全量掃描。

檢索流程（Day 10，已更新）：讀 `work_rules.md` 全文 → 結構化切法切出 59 塊 → 批次取得
float32 正規化向量（快取優先）→ 選擇性 `np.vstack` 疊上合成干擾向量（**真實 chunk 固定
佔 id 0～58**，靠 id 與長度比較即可分辨檢索結果是真實條文還是合成向量）→ `index.add()`
建立 Flat／HNSW 索引 → 問題轉向量 → `index.search()` 取 top-k。

實驗腳本的量測順序有兩處是刻意安排的，改動時要留意（否則數字會失真）：

1. **先量完 Flat 延遲，才建 HNSW**：HNSW 建圖是長時間的重運算會讓 CPU 降頻，若 Flat
   基準在那之後才量，會被灌水成近兩倍慢，倍率就不誠實。
2. **HNSW 建完 `time.sleep(10)` 再量延遲**，讓 CPU 頻率回穩；且先用
   `faiss.omp_get_max_threads()` 記下硬體執行緒數再設成 1，因為該函式回傳的是「當下
   設定值」，設成 1 之後再問只會拿到 1。

#### 相依性審查紀錄：`faiss-cpu`（Day 10 新增的套件）

引入新套件前的五問審查結論，落檔以供交接者複查：

| 審查項 | 結論 |
|--------|------|
| 用途 | 本機向量索引（`IndexFlatIP` 精確暴力掃描 / `IndexHNSWFlat` 近似圖導航），是 Day 10「精確 vs 近似」實驗的核心，無此套件則本日實驗不成立 |
| 替代方案 | `hnswlib`（只有 HNSW，做不出精確對照組）、`annoy`（已不再活躍維護）、`scann`（安裝門檻高，Windows 不友善）。選 faiss 的理由：**同時涵蓋精確與近似索引**，才能在同一套 API 下控制變因；且自 v1.14.2 起 PyPI 有官方 wheel，三大平台一行裝好，讀者重現門檻最低 |
| 維護狀態 | Meta（Facebook Research）官方維護，活躍 |
| 已知 CVE | 無已知 CVE。本專案的使用情境是純本機函式庫、不開任何網路服務、不處理外部輸入的序列化索引檔，攻擊面小 |
| 授權 | MIT |

版本策略：釘死 `==1.15.0`（不是 `>=`），理由是 recall 與延遲數字必須可重現——這是本
專案唯一釘死版本的套件，其餘套件維持 `>=` 下限即可。

### 資料結構

沒有資料庫，以下是程式內流通的資料格式與檔案格式。

**chunk（Day 9 三種切法與 Day 10 結構化切法的共同輸出元素；已更新）**

| 欄位 | 型別 | 說明 |
|------|------|------|
| `text` | `str` | 這一塊的文字內容 |
| `start` | `int` | 在原文中的起始字元位置（含） |
| `end` | `int` | 在原文中的結束字元位置（不含），即半開區間 `[start, end)` |

**article（Day 9／Day 10，`parse_articles()` 解析出的標準答案；已更新）**

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

**檢索結果（Day 8／9）**：`search()` 回傳的每個元素是「原 chunk／article 的所有欄位」
再加上 `score`（`float`，餘弦相似度，越接近 1 越相關）。

**檢索結果（Day 10，已更新）**：FAISS 回傳的是**向量 id 與分數的陣列**，不是 dict——
`search()` 回傳 `(ids: np.ndarray[int], scores: np.ndarray[float], used_tokens: int)`。
id 與建索引時 `add()` 的順序一致，**`id < len(chunks)`（即 0～58）才是真實條文**，其餘
是合成向量；如何把 id 對回 chunk 由呼叫端決定。分數為內積，因向量已 L2 正規化，數值等
同餘弦相似度。

**向量矩陣（Day 10 新增）**

| 項目 | 規格 |
|------|------|
| 型別 | `np.ndarray`，`dtype=np.float32`，形狀 `(n, 1536)` |
| 前處理 | `faiss.normalize_L2()` 就地正規化為單位向量，內積＝餘弦相似度 |
| 為何是 float32 | 5 萬 × 1536 規模下，float64 記憶體加倍、速度減半；FAISS 的輸入格式本身就是 float32 |
| 記憶體佔用 | 50,059 × 1536 × 4 bytes ≈ 308 MB（實驗全程峰值 RSS 約 1353 MB） |

**合成干擾向量（Day 10 新增，`synth_vectors.make_distractors()`）**

| 參數 | 預設 | 說明 |
|------|------|------|
| `n_centroids` | 500 | 隨機質心數 |
| `per_centroid` | 100 | 每個質心的擾動點數；總量 500 × 100 ＝ 50,000 |
| `noise` | 0.35 | 擾動幅度（相對單位向量），決定叢集鬆緊 |
| `seed` | 42 | 固定 seed，重跑生成完全相同的向量；刻意不落地存檔 |

輸出為 `(n_centroids × per_centroid, dim)` 的 float32 單位向量矩陣。

**實驗逐題結果 row（Day 9，`run_experiment.py` 內部結構；`purity` 為 Day 9 新增）**

| 欄位 | 型別 | 說明 |
|------|------|------|
| （`questions.json` 的所有欄位） | — | `id`／`type`／`group`／`question`／`expected` 原樣帶入 |
| `top1_label` | `str` | top-1 chunk 涵蓋到的條文標籤，如「第 24～28 條」 |
| `score` | `float` | top-1 的餘弦相似度 |
| `hit` | `bool` | top-1 範圍是否與預期條文範圍重疊 |
| `purity` | `float \| None` | 命中時的純度（0～1）；未命中為 `None`，報表顯示「—」 |

**實驗量測結果（Day 10 新增，`run_experiment.py` 內部結構）**

Day 10 不用 dict row，改用三組 tuple 清單直接餵報表：

| 結構 | 內容 |
|------|------|
| `latency_rows` | `(引擎名稱, 執行緒數, 中位數 ms, p95 ms)`，Flat 單執行緒與全執行緒各一列 |
| `hnsw_rows` | `(efSearch, 中位數 ms, p95 ms, recall dict, 命中題數)`；`recall` 為 `{1: float, 3: float, 10: float}` |
| `drops` | `(efSearch, questions.json 的該題 dict, HNSW top-1 的標籤)`，只收「Flat 有中、HNSW 沒中」的題 |

另外 `synth_in_top` 為 `{1: int, 3: int, 10: int}`，記錄合成向量擠進 Flat top-k 的次數
（用來證明干擾向量沒有污染答案）。

**`questions.json`（測試題）**

| 欄位 | 型別 | 說明 |
|------|------|------|
| `id` | `int` | 題號 |
| `type` | `str` | `直白`／`口語`，用來分組統計 |
| `group` | `str` | Day 9／Day 10：`沿用`（Day 8 的 10 題）／`新增`（針對新章節的 5 題）。Day 10 完全沿用 Day 9 的 15 題，一題未動（已更新） |
| `question` | `str` | 使用者問法 |
| `expected` | `str` | 預期命中的條文，格式為 `第 N 條`（實驗用 `第 (\d+) 條` 取出條號） |

**`embeddings_cache.json`（向量快取，已被 `.gitignore` 排除）**

- 格式：`{ 快取 key: [float, ...] }`。
- key 產生方式：`sha256("{模型名稱}:{文字內容}")` 取前 16 個十六進位字元。因此語料改動
  只會讓被改到的段落 cache miss，其餘照用；換模型也不會拿到舊模型的向量。

**語料 Markdown 的標題約定（結構化切法依賴這個約定）**

- Day 8 `leave_policy.md`：`# 文件標題` + `## 第 N 條（標題）`，第一塊前言不進檢索範圍。
- Day 9／Day 10 `work_rules.md`（同一份語料，各自放一份；已更新）：`# 文件標題` +
  `## 第 N 章 章名` + `### 第 N 條（標題）`。只有 `###` 層級的區塊會進檢索範圍，前言與
  章標題不進。換自己的語料時要維持這個層級約定，否則 `chunk_by_structure()` 與
  `parse_articles()` 會抓不到東西。

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

**`rag_core.py` 公開函式（Day 8／Day 9）**

| 函式 | 簽章要點 | 說明 |
|------|---------|------|
| `load_document(path)` / `load_articles(path)` | → `str`（Day 9）／`list[dict]`（Day 8） | 讀語料。Day 9 回傳全文由切法處理，Day 8 直接切成一條一段 |
| `get_embeddings(texts)` | `list[str]` → `(list[np.ndarray], used_tokens, cache_hits)` | 批次取向量，快取優先；回傳本次實際消耗 token 與快取命中筆數 |
| `cosine_similarity(a, b)` | `np.ndarray, np.ndarray` → `float` | 餘弦相似度 |
| `search(question, chunks, chunk_vectors, top_k=3)` | → `(list[dict], used_tokens)` | 問題轉向量後全量比對，回傳前 top_k 筆（含 `score`） |
| `cost_usd(tokens)` | `int` → `float` | token 數換算美元 |

**`rag_core.py` 公開函式（Day 10；介面已變更）**

Day 10 的 `rag_core.py` 自 Day 9 沿用並改兩處（Day 9 的檔案不動，各資料夾自給自足）：
向量改 float32＋預先正規化、檢索改走 FAISS。`cosine_similarity()` 已不存在，因為相似度
交給索引算。

| 函式 | 簽章要點 | 說明 |
|------|---------|------|
| `load_document(path=DOC_PATH)` | → `str` | 讀入整份語料原文 |
| `get_embeddings(texts)` | `list[str]` → `(np.ndarray, used_tokens, cache_hits)` | **回傳 `(n, 1536)` 的 float32 矩陣（已 L2 正規化）**，不再是 float64 陣列的 list；快取與成本邏輯與 Day 9 相同 |
| `build_flat(vectors)` | → `faiss.IndexFlatIP` | 精確索引：內積暴力掃描，結果保證正確 |
| `build_hnsw(vectors, m=32, ef_construction=200)` | → `faiss.IndexHNSWFlat` | 近似索引：HNSW 圖，`METRIC_INNER_PRODUCT`。查詢期旋鈕 `index.hnsw.efSearch` 由呼叫端設定（越大越準越慢） |
| `search(question, index, top_k=3)` | → `(ids, scores, used_tokens)` | 問題轉向量後交給指定索引檢索。Flat 與 HNSW 走同一條路，實驗才能控制變因 |
| `cost_usd(tokens)` | `int` → `float` | token 數換算美元 |
| `cosine_similarity()`（已移除） | — | Day 10 不再需要手算：向量已正規化，相似度由 FAISS 的內積索引計算。Day 8／9 的同名函式保留不動 |

**`chunkers.py` 公開函式（Day 9／Day 10；已更新）**

| 函式 | 說明 |
|------|------|
| `chunk_fixed(text, size=300)` | 固定字數硬切（**僅 Day 9**，Day 10 未帶入） |
| `chunk_fixed_overlap(text, size=300, overlap=100)` | 固定字數＋重疊，步進為 `size - overlap`（**僅 Day 9**，Day 10 未帶入） |
| `chunk_by_structure(text)` | 依 `### 第 N 條` 標題切，一條一塊（Day 9／Day 10 都有；Day 10 只留這一種） |
| `parse_articles(text)` | 解析全部條文的條號、標題與字元範圍（實驗標準答案） |
| `spans_overlap(a_start, a_end, b_start, b_end)` | 兩個半開區間是否重疊（命中判定的核心） |
| `articles_covering(chunk, articles)` / `coverage_label(chunk, articles)` | 查出 chunk 沾到哪幾條 / 濃縮成人看的標籤，如「第 24～28 條」 |
| `ARTICLE_NO_PATTERN` | 條號正則 `第 (\d+) 條`，Day 10 的實驗直接用它從 `expected` 取條號 |
| `CHUNKERS` | 切法註冊表：`key → (顯示名稱, 函式)`，`search_demo.py --chunker` 與實驗共用（**僅 Day 9**；Day 10 只有一種切法，無此表） |

**`synth_vectors.py` 公開函式（Day 10 新增）**

| 函式 | 簽章要點 | 說明 |
|------|---------|------|
| `make_distractors(dim, n_centroids=500, per_centroid=100, noise=0.35, seed=42)` | → `np.ndarray`（float32、已正規化） | 生成合成干擾向量矩陣。固定 seed 保證可重現，不落地存檔 |

**`run_experiment.py` 內部函式（Day 10 新增；文章與交接會用到的量測邏輯）**

| 函式 | 說明 |
|------|------|
| `bench(index, queries, repeat=20)` | 量單題延遲：先 warm-up（讓快取與執行緒池就位），再逐題計時，回傳 `(median_ms, p95_ms)` |
| `day9_hits(ids_matrix, chunks, questions, expected_by_no)` | Day 9 的同一把尺：每題 top-1 是否命中預期條文；top-1 是合成向量直接算沒中 |
| `top1_label(chunk_id, chunks, articles)` | 把 top-1 的向量 id 轉成人看的標籤：真實 chunk 給條號，合成向量顯示「合成向量 #N」 |
| `peak_rss_mb()` | 量本行程峰值實體記憶體。Windows 走 `psapi.GetProcessMemoryInfo`（64 位元下必須明宣告 handle 型別，否則 pseudo-handle 會被截斷）、其他平台走 `resource.getrusage`（Linux 單位 KB、macOS 為 bytes）；量不到回 `None`，不影響實驗 |

**CLI 介面（已更新）**

| 指令 | 參數 | 行為 |
|------|------|------|
| `python search_demo.py [問題...]` | 三天都有；不帶問題進互動模式 | 印出 top-3 結果 |
| `python search_demo.py --chunker {fixed,overlap,structure}` | 僅 Day 9，預設 `structure` | 指定切法 |
| `python search_demo.py --distractors N` | 僅 Day 10，`int`，預設 `0`；上限自動夾到 50000 | 把 N 個合成干擾向量混進索引再查，用來驗證「假向量不會搶走真實答案」 |
| `python run_experiment.py` | 無參數 | 跑完整題組，報表先寫入 `experiment_result.md` 再印出（Day 10 約需 5 分鐘） |

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

**套件需求**（每個 day 資料夾各有一份 `requirements.txt`；已更新）

| 套件 | 版本 | 適用 |
|------|------|------|
| `openai` | `>=1.40` | 全部 |
| `numpy` | `>=1.26` | 全部 |
| `python-dotenv` | `>=1.0` | 全部 |
| `faiss-cpu` | **`==1.15.0`（釘死）** | Day 10 起。釘死是為了讓 recall 與延遲數字可重現，換版本數字會漂移；faiss 自 v1.14.2 起 PyPI 有官方 wheel，Windows／macOS／Linux 都是一行裝好。審查結論見「技術架構」的相依性審查紀錄 |

**程式內可調參數（非環境變數，直接改常數）**

| 常數 | 位置 | 預設 | 用途 |
|------|------|------|------|
| `EMBEDDING_MODEL` | `rag_core.py` | `text-embedding-3-small` | 使用的 embedding 模型；改動後快取 key 會全部變化，等於重新計算 |
| `EMBEDDING_DIM` | `rag_core.py`（Day 10） | `1536` | 模型輸出維度，供索引與合成向量參照 |
| `USD_PER_MILLION_TOKENS` | `rag_core.py` | `0.02` | 模型單價（2026-09 官網查閱），只用於成本顯示 |
| `USD_TO_TWD` | `rag_core.py` | `31` | 粗估匯率，只用於文章內成本示意 |
| `CHUNK_SIZE` | `chunkers.py`（Day 9） | `300` | 固定字數切法每塊字元數 |
| `OVERLAP` | `chunkers.py`（Day 9） | `100` | overlap 切法的重疊字元數 |
| `N_CENTROIDS` / `PER_CENTROID` | `synth_vectors.py`（Day 10） | `500` / `100` | 合成干擾向量的質心數與每質心點數，相乘＝總量 50,000 |
| `NOISE` | `synth_vectors.py`（Day 10） | `0.35` | 擾動幅度，決定合成資料的叢集鬆緊 |
| `SEED` | `synth_vectors.py`（Day 10） | `42` | 合成向量的隨機種子；**改動即失去與文章數字的可對照性** |
| `HNSW_M` / `HNSW_EF_CONSTRUCTION` | `run_experiment.py`（Day 10） | `32` / `200` | HNSW 建圖參數：每節點連線數上限、建圖候選清單寬度 |
| `EF_SEARCH_VALUES` | `run_experiment.py`（Day 10） | `(8,16,32,64,128,256)` | 要掃的 `efSearch` 檔位（找 recall 歸零點用） |
| `TOP_K` / `LATENCY_REPEAT` | `run_experiment.py`（Day 10） | `10` / `20` | 檢索深度（recall@10 需要 10 筆）、每題重複次數（15 × 20 ＝ 300 個延遲樣本） |

**成本預期**（已更新）：Day 8 全部語料＋10 題約數千 token；Day 9 三種切法各建一次索引
＋15 題，首次無快取實測為 40,875 token，約 0.0008 美元（新台幣 0.03 元）；Day 10 只用
結構化切法，59 條條文＋15 題首次無快取實測 13,619 token，約 0.000272 美元（新台幣
0.0084 元），**50,000 個合成向量由本機生成，0 元**。快取命中的部分不計費，因此有快取後
重跑會顯示「計費 0 tokens」——報表會同時列出快取命中的筆數與「刪除
`embeddings_cache.json` 重跑可重現完整成本」的說明，避免 0 元被誤讀成免費。

**Day 10 的非金錢成本**（已更新）：真正的門檻不是 API 費用而是機器資源——HNSW 單執行緒
建圖約 3～5 分鐘（實測 265 秒）、全程峰值記憶體約 1.4 GB（8 GB 機器可跑，但建議先關掉
吃記憶體的程式）。
