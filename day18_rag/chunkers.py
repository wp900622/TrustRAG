# -*- coding: utf-8 -*-
"""Day 11 切法：沿用 Day 9/10 的贏家「依條文結構化」，新增章別解析——
每個 chunk 帶上所屬的章（chapter_no／chapter），作為 Chroma 的 metadata，
支撐本日的 metadata 過濾實驗。

- 每個 chunk 仍帶 (start, end)＝它在原文中的字元範圍，命中判定沿用「範圍重疊」
- 條文範圍（ground truth）由 parse_articles() 從「### 第 N 條」標題解析
- 章由 parse_chapters() 從「## 第X章」標題解析；章號依出現順序編（1 起算），
  刻意不解析中文數字——讀者換自己的語料時不必遷就編號寫法
"""
import re

# 匹配「## 第一章」與「### 第 N 條」兩個層級的 Markdown 標題
HEADING_PATTERN = re.compile(r"^(#{2,3}) (.+)$", re.MULTILINE)
ARTICLE_NO_PATTERN = re.compile(r"第 (\d+) 條")


def _blocks(text: str):
    """依標題位置切區段：每個標題的範圍到下一個任意層級標題（或文末）為止"""
    matches = list(HEADING_PATTERN.finditer(text))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        yield match, match.start(), end


def parse_chapters(text: str) -> list[dict]:
    """解析每一章的序號、標題與字元範圍（章的範圍涵蓋到下一個「##」標題為止）"""
    chapters = []
    for match, start, _ in _blocks(text):
        if match.group(1) != "##":
            continue
        chapters.append({"no": len(chapters) + 1,
                         "title": match.group(2).strip(), "start": start})
    for i, chapter in enumerate(chapters):
        chapter["end"] = (chapters[i + 1]["start"]
                          if i + 1 < len(chapters) else len(text))
    return chapters


def chapter_of(pos: int, chapters: list[dict]) -> dict | None:
    """回傳涵蓋字元位置 pos 的章；語料開頭的前言不屬於任何章，回 None"""
    for chapter in chapters:
        if chapter["start"] <= pos < chapter["end"]:
            return chapter
    return None


def chunk_by_structure(text: str) -> list[dict]:
    """依 Markdown 標題結構切，一條規定＝一塊（Day 9 三種切法的贏家）。

    只保留「### 第 N 條」層級的區塊；文件前言與章標題不含答案，不進檢索範圍。
    Day 11 起每塊多帶 chapter_no／chapter 兩個欄位，進資料庫時成為 metadata。
    """
    chapters = parse_chapters(text)
    chunks = []
    for match, start, end in _blocks(text):
        if match.group(1) != "###":
            continue
        block_text = text[start:end].strip().removeprefix("### ")
        chapter = chapter_of(start, chapters)
        chunks.append({"text": block_text, "start": start, "end": end,
                       "chapter_no": chapter["no"] if chapter else 0,
                       "chapter": chapter["title"] if chapter else "（無章）"})
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
