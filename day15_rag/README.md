# Day 15：可信度——幻覺、引用與拒答

Day 14 把檢索接上生成，量到答案正確率 14/15（k=3），也在文末承認那把尺有兩個洞：
模型說「條文未提及」一律算錯，而**語料根本沒寫的問題它照樣掰得出答案，尺完全
看不到**。Day 15 補這兩個洞。

檢索側凍結在 Day 14 量到的甜蜜點 **k=3**，語料與 15 題原封不動，全程只改
system prompt——落差才能乾淨歸因到那幾句話。

## 這一天新增什麼

- **五題陷阱題**（`questions_trap.json`）：語料裡沒有答案的問題，分兩類
  - A 類（3 題）語料完全沒有這條規定 → 正確行為只有「說不知道」
  - B 類（2 題）語料寫明「依另一辦法規定」→ 正確行為是轉指，不是編數字
- **Prompt 三版對照**（`prompts.py`）：P1 基本版／P2 要求引用／P3 引用＋允許拒答
- **四個可信度指標**（`grader.py`）：幻覺率、正確處理率、過度拒答率、引用正確率

## 檔案說明

| 檔案 | 用途 |
|------|------|
| `work_rules.md` | 虛構語料：《員工工作規則》11 章 59 條（同 Day 9~14） |
| `questions.json` | 正常 15 題（沿用 Day 9~14，一字未改）＋ `facts`／`forbid` 判定欄位 |
| `questions_trap.json` | **本日新增**：五題陷阱題，含逐題的 `proper_signals` 與 `hallucination_markers` |
| `chunkers.py` | 按條文結構切塊（同 Day 9~11） |
| `rag_core.py` | 讀語料、Embedding、生成，兩層快取與成本計算 |
| `chroma_store.py` | Chroma collection 的建立與檢索（同 Day 11） |
| `prompts.py` | context 組裝與 **三版 system prompt** |
| `pipeline.py` | 一次完整 RAG：問題 → 向量 → top-k → prompt → 生成 |
| `grader.py` | 判定尺：關鍵事實比對（Day 14）＋ **兩段式拒答判定與引用判定**（Day 15） |
| `build_db.py` | 建立 Chroma 持久化資料庫 |
| `run_experiment.py` | Day 14 主實驗（四組 k 值對照），輸出 `experiment_result.md` |
| `run_experiment_day15.py` | **本日主實驗**（三版 prompt × 20 題），輸出 `experiment_result_day15.md` |
| `ask_demo.py` | 單題問答的互動示範 |

## 怎麼跑

```bash
pip install -r requirements.txt
cp .env.example .env          # 填入你自己的 OPENAI_API_KEY
python build_db.py            # 建立 Chroma 資料庫（59 條）
python run_experiment_day15.py
```

首跑成本約新台幣 0.17 元（3 版 × 20 題 = 60 次生成）。`chat_cache.json` 以
「模型＋完整 messages」雜湊為 key，`temperature=0` 下重跑走快取、計費 0 元——
**改判定邏輯不需要重新付錢生成**，這是本日改了三版判定還沒超支的原因。

## 結果摘要

| prompt 版本 | 正常題答對 | 過度拒答 | 陷阱題幻覺 | 正確處理 | 引用正確 |
|---|---|---|---|---|---|
| P1 基本版 | 14/15 | 0/15 | 1/5（假拒答 1） | 4/5 | 未要求 |
| P2 要求引用 | 13/15 | 0/15 | 0/5 | 5/5 | 14/15 |
| P3 引用＋允許拒答 | 13/15 | 1/15 | 0/5 | 4/5 | 14/15 |

逐題明細與答案原文在 `experiment_result_day15.md`。三個值得單獨看的現象：

1. **假拒答**：P1 唯一的幻覺是「依國外出差辦法另行規定……國內出差住宿費上限為
   每人每晚新臺幣三千元」——它同時是拒答也是幻覺。只看拒答關鍵詞的判定會給它
   一個勾，所以 `grader.refusal_verdict()` 收緊成兩段判定。
2. **過度拒答率要跟 recall@k 一起讀**：P3 那一題「過度拒答」是第 9 題，
   而該題的預期條文從來沒撈進 top-k——在它拿到的 context 裡，拒答是正確行為。
3. **尺看不到的錯**：第 105 題三版都把第 19 條的加成比例抄反了（前 2 小時應為
   加給三分之一），而判定只問「有沒有編出不存在的金額」，三版都過。
   分數再高都要抽樣人工複核，而且優先複核答對的題。

## 已知限制

- 20 題規模小，單一模型（`gpt-4o-mini`）、單一語料，結論只在這個規模成立
- 判定尺是關鍵詞與正規表示式，不是 LLM 裁判：可重現、零成本，但會漏。
  `grader.py` 裡記錄了本日實際踩到的三個誤判
- 多條文綜合題（第 105 題需要第 19 條＋第 33 條）尚未處理
- Prompt injection（使用者在問題裡寫「忽略上面的條文」）尚未處理
