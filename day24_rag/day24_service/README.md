# day24_service

把前 23 天散落的實驗腳本收成一支服務。檢索與生成的邏輯一個字都沒改，
沿用上層目錄的 `chunkers` / `prompts` / `rag_core` / `chroma_store`。

```
day24_service/
  main.py        FastAPI app 與四個端點
  contracts.py   Pydantic 請求／回應契約（唯一需要給別人讀的檔案）
  timing.py      延遲分解的 middleware，跟 RAG 無關，可以直接抄走
  retrievers.py  Retriever 介面 ＋ Chroma／numpy 兩個實作
  embedder.py    算問題向量：走快取的 vs 真的打 API 的
  ingest.py      攝取走背景任務
  agents.py      把 Day 20／21 那支 agent 接進來（它一行都沒改）
  config.py      設定，全部可以用 DAY24_* 環境變數覆蓋
  cache.py       常駐快取：把 rag_core 的整包讀寫換成上鎖＋原子換檔
  store.py       攝取 job 的狀態，存 SQLite（多個 worker 才查得到）
  errors.py      統一的錯誤形狀 ＋ 跟著請求跑的 request id
  tests/         24 個測試，不打 API、不花錢
```

## 測試

```bash
pytest day24_service/tests -q
```

全部走既有快取，一毛錢都不花。這批測試擋的是前幾版真的踩過的東西：
同一份文件攝取兩次讓索引從 59 塊長到 118 塊、攝取完 numpy 那條路還拿著舊矩陣、
job 狀態放行程記憶體換個 worker 就查不到、錯誤各長各的樣子。

## 跑起來

```bash
uvicorn day24_service.main:app --port 8024
```

環境變數只有一個，用來切 middleware 的掛法：

```
DAY24_MIDDLEWARE=asgi   # 預設。純 ASGI，正確的那個
DAY24_MIDDLEWARE=base   # BaseHTTPMiddleware，會把背景任務變成前景任務
DAY24_MIDDLEWARE=none   # 不掛，用來量 middleware 本身的成本
```

## 問一題

```bash
curl -s -X POST http://127.0.0.1:8024/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"病假連續請幾天以上需要附診斷證明？","k":3}'
```

回應（節錄）：

```json
{
  "answer": "病假連續請三天（含）以上需要附診斷證明。",
  "citations": [
    {"article_no": 24, "title": "第 24 條（病假）", "similarity": 0.6205,
     "text": "第 24 條（病假）\n員工因普通傷害、疾病…"}
  ],
  "usage": {"embedding_tokens": 0, "input_tokens": 0,
            "output_tokens": 0, "cached": true},
  "timing": {"embed_ms": 282.9, "retrieve_ms": 4.7, "llm_ms": 5.6,
             "serialize_ms": 0.1, "total_ms": 293.6, "overhead_ms": 0.3},
  "retriever": "chroma"
}
```

兩個欄位值得單獨講：

- **`citations` 是檢索實際撈到的那幾條，不是模型講的。**
  Day 15 量過模型自己標的條號會幻覺，所以引用不能讓它自己說。
- **`timing` 是回應的一部分，不是 log。** 要回答「慢在哪」就不該需要翻 log。

`use_cache: false` 會繞過 embedding 與生成的快取，真的打 API——
上線之後每一個沒看過的問題都走這條路，量真實延遲要用它。

`?retriever=numpy` 換成 59 條的暴力矩陣乘法，其餘完全不變。

## 換一種答法

```bash
curl -s -X POST http://127.0.0.1:8024/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"我平日加班 3 小時，加班費要怎麼算？","mode":"agent"}'
```

`mode=agent` 走 Day 20／21 那支：手上有 search／check／answer 三個工具，
自己決定查幾次。回應會多一個 `agent` 區塊：

```json
"agent": {"steps": 2, "searches": 1, "llm_calls": 2,
          "queries": ["加班費"], "hit_cap": false, "n_checks": 0}
```

`agent_day21.py` 一個字都沒改。它原本直接吃 Chroma 的 collection，
`agents.py` 裡的 `RetrieverCollection` 把任何 Retriever 包成 collection 的樣子，
所以它照樣跑得動 `?retriever=numpy`，28 題的答案與查詢關鍵字都一樣。

`self_check=true` 是 Day 21 的 C2：沒自驗過想交卷會被退件。預設關，
因為 Day 21 量過它救回 1 題、改壞 1 題，成本 ×2.0。

## 攝取一份文件

慢的事情丟背景。Day 22 量過，一份 20 頁、沒有文字層的 PDF 走離線 OCR 要 277 秒，
沒有哪個用戶端會等，中間的反向代理也不會。

```bash
curl -s -X POST http://127.0.0.1:8024/documents \
  -H 'Content-Type: application/json' \
  -d '{"path":"work_rules.md","ocr":false}'
# → 202 {"job_id":"a23b9c87dd9b","status":"queued",...}

curl -s http://127.0.0.1:8024/documents/a23b9c87dd9b
# → {"job_id":"a23b9c87dd9b","status":"done","chunks":59,"elapsed_ms":611.5}
```

## 設定

全部可以用環境變數覆蓋，前綴 `DAY24_`。不設就跟前 23 天的腳本一樣。

| 變數 | 預設 | 做什麼 |
| --- | --- | --- |
| `DAY24_RETRIEVER` | `chroma` | 預設用哪個 retriever |
| `DAY24_K` | `3` | 預設取幾條 |
| `DAY24_DOCUMENTS_ROOT` | 專案目錄 | 攝取只能讀這底下的檔案 |
| `DAY24_JOB_DB` | `day24_jobs.sqlite3` | job 狀態存哪 |
| `DAY24_RESIDENT_CACHE` | `1` | 快取常駐記憶體（關掉會退回會壞的版本） |
| `DAY24_AGENT_CONCURRENCY` | `1` | 同時能跑幾個 agent 請求 |
| `DAY24_MIDDLEWARE` | `asgi` | 計時 middleware 的掛法 |

## 多個 worker

```bash
uvicorn day24_service.main:app --port 8024 --workers 3
```

三個 worker 下實測：連續丟 9 個攝取 job，9 個都 done，索引維持 59 塊，
任何一個 worker 都查得到任何一個 `job_id`。三件事讓它成立：

- job 狀態在 SQLite（WAL），不在行程記憶體
- chunk 的 id 由「檔名 + 條號 + 內容雜湊」算出來，重複攝取是覆蓋不是追加
- 攝取寫索引時用 `store.exclusive()` 跨行程互斥，`ChromaRetriever` 也不把
  collection handle 抓在手上——內嵌模式下別的 worker 重建之後，舊 handle 會失效

## 還沒做的

- **攝取收的是路徑，不是上傳的檔案。** 換成 `UploadFile` 的話，job 那一段不用改。
- **`agents.py` 用替換模組函式的方式計時**，那是行程層級的，所以 agent 請求目前
  被序列化成一次一個（`DAY24_AGENT_CONCURRENCY`）。正解是把 retriever 與 embedder
  注入 agent，但那要改 `agent_day21.py`，而它是 Day 21 那篇的證據。
- **索引在每個 worker 各建一份。** 語料大起來之後應該共用一份，或改用外部向量庫。
