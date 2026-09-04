# -*- coding: utf-8 -*-
"""Day 9 三種切法實作：固定字數、固定字數＋overlap、依標題結構化。

設計重點：
- 每個 chunk 都帶 (start, end)＝它在原文中的字元範圍。三種切法的邊界都不同，
  沒辦法直接比「條號」，改用「top-1 chunk 的範圍是否與預期條文範圍重疊」判定命中，
  三種切法才是同一把尺
- 條文範圍（ground truth）由 parse_articles() 從「### 第 N 條」標題解析
"""
import re

CHUNK_SIZE = 300   # 固定字數切法：每塊的字元數
OVERLAP = 100      # overlap 切法：相鄰兩塊的重疊字元數

# 匹配「## 第一章」與「### 第 N 條」兩個層級的 Markdown 標題
HEADING_PATTERN = re.compile(r"^(#{2,3}) (.+)$", re.MULTILINE)
ARTICLE_NO_PATTERN = re.compile(r"第 (\d+) 條")


def chunk_fixed(text: str, size: int = CHUNK_SIZE) -> list[dict]:
    """切法一：固定字數硬切，完全不管句子、段落或條文邊界"""
    chunks = []
    for start in range(0, len(text), size):
        end = min(start + size, len(text))
        chunks.append({"text": text[start:end], "start": start, "end": end})
    return chunks


def chunk_fixed_overlap(text: str, size: int = CHUNK_SIZE,
                        overlap: int = OVERLAP) -> list[dict]:
    """切法二：固定字數＋前後重疊。句子被腰斬時，重疊帶讓相鄰塊各保有一份完整上下文"""
    chunks = []
    step = size - overlap
    for start in range(0, len(text), step):
        end = min(start + size, len(text))
        chunks.append({"text": text[start:end], "start": start, "end": end})
        if end == len(text):
            break
    return chunks


def _blocks(text: str):
    """依標題位置切區段：每個標題的範圍到下一個任意層級標題（或文末）為止"""
    matches = list(HEADING_PATTERN.finditer(text))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        yield match, match.start(), end


def chunk_by_structure(text: str) -> list[dict]:
    """切法三：依 Markdown 標題結構切，一條規定＝一塊（Day 8 做法的一般化）。

    只保留「### 第 N 條」層級的區塊；文件前言與章標題不含答案，不進檢索範圍。
    """
    chunks = []
    for match, start, end in _blocks(text):
        if match.group(1) != "###":
            continue
        block_text = text[start:end].strip().removeprefix("### ")
        chunks.append({"text": block_text, "start": start, "end": end})
    return chunks


def parse_articles(text: str) -> list[dict]:
    """解析每一條條文的條號、標題與字元範圍，作為實驗判定命中的標準答案"""
    articles = []
    for match, start, end in _blocks(text):
        if match.group(1) != "###":
            continue
        title = match.group(2).strip()
        no_match = ARTICLE_NO_PATTERN.search(title)
        if no_match is None:
            continue
        articles.append({"no": int(no_match.group(1)), "title": title,
                         "start": start, "end": end})
    return articles


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """兩個半開區間 [start, end) 是否重疊"""
    return a_start < b_end and b_start < a_end


def articles_covering(chunk: dict, articles: list[dict]) -> list[dict]:
    """回傳與 chunk 範圍重疊的條文清單（報表用：看得出這個 chunk 切到了哪幾條）"""
    return [a for a in articles
            if spans_overlap(chunk["start"], chunk["end"], a["start"], a["end"])]


def coverage_label(chunk: dict, articles: list[dict]) -> str:
    """把 chunk 涵蓋的條文濃縮成人看的標籤，例如「第 24～25 條」或「前言」"""
    covered = articles_covering(chunk, articles)
    if not covered:
        return "（前言／章標題）"
    if len(covered) == 1:
        return f"第 {covered[0]['no']} 條"
    return f"第 {covered[0]['no']}～{covered[-1]['no']} 條"


# 實驗與 demo 共用的切法註冊表：key → (顯示名稱, 切法函式)
CHUNKERS = {
    "fixed": ("固定字數", chunk_fixed),
    "overlap": ("固定字數+overlap", chunk_fixed_overlap),
    "structure": ("結構化（依條文）", chunk_by_structure),
}
