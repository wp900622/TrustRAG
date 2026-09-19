# Day 24 計畫：把 22 天的腳本收成一支服務，然後量它慢在哪

一句話：**前 13 天我一直在調切法、k、metadata 過濾、重排，全部都在檢索那一側。
今天把 28 題改走 HTTP，把一次請求拆成 embedding、檢索、LLM、序列化四段，
問一個到現在都沒問過的問題：我調了 13 天的那一段，佔總延遲的幾 %？
順便把 Day 20／21 那支 agent 也收進同一個端點。**

昨天文末那句話是：「語料這一側到今天收工，明天開始把這 22 天的實驗腳本
變成一支真的服務。」今天照那句走。

## 一、今天做出來的東西

```
day24_service/
  main.py        FastAPI app 與四個端點
  contracts.py   Pydantic 請求／回應契約（唯一需要給別人讀的檔案）
  timing.py      延遲分解的 middleware，跟 RAG 無關，可以直接抄走
  retrievers.py  Retriever 介面 ＋ Chroma／numpy 兩個實作
  embedder.py    算問題向量：走快取的 vs 真的打 API 的
  ingest.py      攝取走背景任務
```

四個端點：

```
POST /ask              問一題，回答案＋引用＋這一次請求的四段耗時
POST /documents        攝取一份文件，立刻回 202 與 job_id
GET  /documents/{id}   查攝取進度
GET  /healthz          索引塊數、retriever 是哪一個、啟動耗時
```

### 為什麼 Search 一定要走背景任務

Day 22 量過：一份 20 頁、沒有文字層的 PDF 走離線 OCR 要 **277 秒**
（`experiment_result_day22.md` 的 V 路徑）。沒有一個 HTTP client 會等你 277 秒，
中間的反向代理也不會。所以 `POST /documents` 只登記一個 job 就回。

### 為什麼 retriever 要抽成介面

前 13 天每次動檢索都是直接改 `chroma_store.py`，那沒問題，因為只有一個呼叫者。
變成服務之後 `/ask` 不該知道底下是 Chroma 還是別的東西，它只需要
「給我一個問題向量，還我 k 條帶 metadata 的原文」。介面刻意只有一個方法。

多寫一個 `NumpyRetriever`（59 條的暴力矩陣乘法）不是為了推薦它，
是為了讓「檢索很慢」這句話有個對照組。

## 二、兩組條件，因為前 23 天的數字都是快取量出來的

這是今天最重要的設計。`chat_cache.json` 裡這 28 題早就全命中，
所以如果照舊跑，「LLM」那一段會量到 2 毫秒——那是快取的速度，不是模型的。

```
warm   兩個快取都開   ← 前 23 天腳本的樣子
cold   兩個快取都關   ← 上線之後每一個新問題的樣子
```

cold 這組每一題都真的打一次 embedding API 與一次 chat API。
**今天唯一花錢的就是這一段。**

## 三、凍結清單

59 條原文、k=3、P1 prompt、`temperature=0`、切法、embedding 模型。
`prompts.build_messages()` 收到的 hits 連 metadata 都原封不動，
所以 warm 那組的 prompt 與 Day 14 以來逐字相同，快取才命中得了。

**今天的變因只有一個：這段程式是腳本在呼叫，還是一支服務在呼叫。**

## 四、要量的東西

```
四段拆解      embed／retrieve／llm／serialize，warm 與 cold 各一組
那一塊剩下的  total 減掉四段＝框架、驗證、ASGI 的來回，也要列出來
p50 / p95     尾巴有多長
retriever     Chroma vs numpy，同樣 28 題，top-3 有沒有不一樣
middleware    三種掛法：純 ASGI／BaseHTTPMiddleware／不掛
背景任務      POST /documents 多久回、job 實際跑多久
錢            cold 那 56 次呼叫
```

## 五、先寫下來，免得跑完自圓其說

1. **cold 那組，`llm` 佔 p50 的 90% 以上。**
2. **檢索佔不到 2%。** 我調了 13 天的那一段，在一次請求裡幾乎看不到。
3. **embedding（真的打 API）比檢索大，但不到 10%。**
4. **序列化 < 1 ms，佔不到 0.5%。**
5. **p95 是 p50 的 2 倍以上**：LLM 的尾巴很長，而且只要有一題長，p95 就會被拉走。
6. **numpy 比 Chroma 快**，因為 59 塊根本不需要索引結構；兩邊的 top-3 完全一樣。
7. **warm 那組，最大的一段會是 `embed`**——不是因為算向量，是因為要把
   10.5 MB 的 `embeddings_cache.json` 整包讀進來。也就是說前 23 天量到的「快」是假的。
8. **`POST /documents` 掛 BaseHTTPMiddleware 會等到攝取跑完才回**（>500 ms），
   換成純 ASGI 會 < 50 ms。這一條是我寫到一半撞到的，先寫死再量。
9. **middleware 本身的成本 < 1 ms**（掛 vs 不掛的 p50 差）。

## 六、agent 也要收進來

系列名的最後一段是「到 AI Agent」。Day 20／21 那支（自己決定查幾次、
交卷前自驗）一直留在腳本裡，今天一起收進 `/ask`：`mode=agent`。

兩個限制決定了作法：`agent_day21.py` 是 Day 21 那篇的證據，**一個字都不能改**；
而它直接吃 Chroma 的 collection，服務這一側卻已經把檢索抽成介面了。
解法是一個轉接頭，把任何 Retriever 包成 collection 的樣子。
**介面能不能用，看的是舊程式能不能不改就跑進來。**

### 再寫六條預測

10. **agent 的 p50 至少是 pipeline 的 3 倍**，因為它一題要打好幾次模型。
11. **agent 的 llm 佔比會比 pipeline 更高，超過 90%。**
12. **agent 的檢索總時間是 pipeline 的 2 倍以上**（查好幾次），但佔比仍然不到 3%。
13. **agent cold 的錢是 pipeline cold 的 5 倍以上**（Day 21 量過 ×8.8）。
14. **28 題裡至少 20 題只查 1 次**：Day 20 量過它很少主動多查。
15. **凍結的 agent 跑在 numpy 上，28 題答案與跑在 Chroma 上逐字相同。**

Day 20 三條全錯、Day 21 錯一條半、Day 22 七條錯三條、Day 23 七條錯四條。
這十五條也會錯，錯的照寫。
