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
    mode: Literal["pipeline", "agent"] = Field(
        default="pipeline",
        description="pipeline＝Day 14 那條寫死管線（查一次、答一次）；"
                    "agent＝Day 20／21 那支自己決定查幾次的")
    self_check: bool = Field(
        default=False,
        description="只對 agent 有效。開了就是 Day 21 的 C2：沒自驗過想交卷會被退件。"
                    "預設關，因為 Day 21 量過它救回 1 題、改壞 1 題，成本 ×2.0")


class Citation(BaseModel):
    """一條被檢索到的規章條文。similarity 是餘弦相似度，不是機率。"""
    article_no: int
    title: str
    similarity: float
    text: str


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


class AskResponse(BaseModel):
    answer: str | None
    citations: list[Citation]
    usage: Usage
    timing: Timing
    retriever: str = Field(description="這次用的是哪一個 retriever 實作")
    mode: Literal["pipeline", "agent"] = "pipeline"
    agent: AgentTrace | None = None
    mode: Literal["pipeline", "agent"] = "pipeline"
    agent: AgentTrace | None = None


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
