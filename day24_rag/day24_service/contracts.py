# -*- coding: utf-8 -*-
"""請求與回應的契約。

前 23 天的「契約」是一個 dict 加上我自己記得欄位叫什麼。變成服務之後，
這份檔案是唯一一個別人（前端、另一個服務、三個月後的我）需要讀的東西。

三個決定寫在這裡，不寫在程式裡：

1. **`citations` 由服務從檢索結果直接吐，不經過模型的嘴。**
   Day 15 量過模型自己標的條號會幻覺，所以引用不能讓它自己說。
2. **`timing` 是回應的一部分，不是 log。** 今天整篇文章都在問「時間花在哪」，
   那這個答案就應該每一次請求都回得出來，而不是要我去翻 log。
3. **`answer` 允許是 null。** 攝取還沒跑完、或檢索一條都沒撈到的時候，
   回一個空字串假裝有答案是最糟的選擇。
"""
from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500,
                          description="使用者的問題，繁體中文")
    k: int = Field(default=3, ge=1, le=15,
                   description="檢索取幾條；Day 14 起的預設是 3")
    use_cache: bool = Field(
        default=True,
        description="關掉就真的打 embedding 與 LLM 的 API。"
                    "今天的實驗要靠它才量得到真實延遲")
    mode: Literal["pipeline", "agent"] = Field(
        default="pipeline",
        description="pipeline＝Day 14 那條寫死管線（查一次、答一次）；"
                    "agent＝Day 20／21 那支自己決定查幾次的")
    self_check: bool = Field(
        default=False,
        description="只對 agent 有效。開了就是 Day 21 的 C2：沒自驗過想交卷會被退件。"
                    "預設關，因為 Day 21 量過它救回 1 題、改壞 1 題，成本 ×2.0")
    source: str | None = Field(
        default=None, max_length=200,
        description="只查這一份文件。索引裡不只一份的時候，"
                    "不給就是整個索引一起查")
    chapter_no: int | None = Field(
        default=None, ge=1, le=99,
        description="只查這一章。跟 source 可以一起給，兩個都符合才算")
    max_steps: int | None = Field(
        default=None, ge=1, le=12,
        description="只對 agent 有效。它最多可以走幾步；走到就被要求直接作答。"
                    "不給就照服務的設定，服務也沒設就是凍結 agent 自己的 6 步")
    max_twd: float | None = Field(
        default=None, gt=0, le=10,
        description="只對 agent 有效。這一題最多花多少台幣，超過就停手。"
                    "金額是離線重數的估計值，不是 API 回報的")
    max_wall_ms: float | None = Field(
        default=None, gt=0, le=600_000,
        description="只對 agent 有效。這一題最多跑多久，超過就停手")
    stream: bool = Field(
        default=False,
        description="開了就走 SSE：出處先送，答案一段一段送。"
                    "總時間不會變短，變短的是第一個字出現的時間。"
                    "只對 mode=pipeline 有效")


class Citation(BaseModel):
    """一條被檢索到的規章條文。similarity 是餘弦相似度，不是機率。"""
    article_no: int
    title: str
    similarity: float
    text: str
    # Day 29 加的。索引裡不只一份文件之後，「第 3 條」這個引用自己是不完整的
    source: str = ""


class Timing(BaseModel):
    """一次請求的四段拆解，單位毫秒。

    embed ＋ retrieve ＋ llm ＋ serialize 不會剛好等於 total：
    差額是框架、驗證與網路，那一塊也要看得見，所以 overhead 單獨列。
    """
    embed_ms: float
    retrieve_ms: float
    llm_ms: float
    serialize_ms: float
    total_ms: float
    overhead_ms: float


class Usage(BaseModel):
    embedding_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    # Day 27 加的兩個。`cost_twd` 是這次真的花的錢，命中時是 0。
    # `avoided_twd` 只有命中才有值：同一把 key 上次付過多少，這次就省了多少。
    # 命中、但那一筆是 Day 27 之前買的（沒人記過金額）時是 null，
    # 不要用平均值把它補成一個看起來有根據的數字。
    cost_twd: float = 0.0
    avoided_twd: float | None = None


class AgentTrace(BaseModel):
    """agent 這條路才有的東西。

    `searches` 與 `llm_calls` 是 pipeline 永遠等於 1 的兩個數字，
    放在回應裡，呼叫端才知道這一題到底繞了多遠。
    """
    steps: int
    searches: int
    llm_calls: int
    queries: list[str]
    repeats: int = 0
    hit_cap: bool = False          # 撞上 6 步上限，被要求直接作答
    n_checks: int = 0
    flagged: bool = False          # 自驗說有問題
    revised: bool = False          # 自驗之後答案真的動了
    # Day 28：這一題是自己收尾的，還是被上限攔下來的。
    # "steps" 仍然有答案（走的是凍結 agent 自己那條強制作答的路），
    # "cost" 與 "time" 沒有，它們是直接打斷，answer 會是 null
    stopped_reason: str | None = None
    stopped_at_step: int | None = None


class AskResponse(BaseModel):
    answer: str | None
    citations: list[Citation]
    usage: Usage
    timing: Timing
    retriever: str = Field(description="這次用的是哪一個 retriever 實作")
    mode: Literal["pipeline", "agent"] = "pipeline"
    agent: AgentTrace | None = None


class SearchRequest(BaseModel):
    """只檢索，不問模型。

    Day 24 到 Day 28，這支服務只有一個入口是 `/ask`，而 `/ask` 一定會叫模型。
    想知道「這個問題撈得到哪幾條」的時候，得付一次生成的錢才看得到答案。
    檢索本來就是一個獨立的能力，這裡把它單獨開出來。
    """
    question: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=50,
                   description="取幾條。上限比 /ask 寬，因為這裡不進 prompt")
    source: str | None = Field(default=None, max_length=200)
    chapter_no: int | None = Field(default=None, ge=1, le=99)
    min_similarity: float | None = Field(
        default=None, ge=-1.0, le=1.0,
        description="低於這個相似度的不要。過濾發生在檢索之後，"
                    "所以回傳的條數可能少於 k")
    use_cache: bool = Field(default=True,
                            description="關掉就真的打一次 embedding API")


class SearchResponse(BaseModel):
    hits: list[Citation]
    retriever: str
    k: int
    filtered: bool = Field(description="這次有沒有帶條件")
    embedding_tokens: int = 0
    elapsed_ms: float = 0.0


class SourceSummary(BaseModel):
    """索引裡的一份文件。"""
    source: str
    chunks: int
    articles: list[int] = Field(description="這份文件裡有哪些條號")


class DeleteResult(BaseModel):
    source: str
    removed: int
    chunks_left: int


class IngestRequest(BaseModel):
    """攝取一份文件。

    刻意收「路徑」而不是收檔案內容：今天的重點是背景任務的形狀，
    不是上傳。真的要收檔案的話換成 UploadFile，job 的部分一個字都不用改。
    """
    path: str = Field(description="要攝取的檔案路徑（.md 或 .pdf）")
    ocr: bool = Field(default=False,
                      description="PDF 沒有文字層時才要；Day 22 量過一份 20 頁的要 277 秒")


class IngestJob(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    path: str
    chunks: int = 0
    elapsed_ms: float = 0.0
    error: str | None = None
