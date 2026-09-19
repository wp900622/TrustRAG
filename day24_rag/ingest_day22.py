# -*- coding: utf-8 -*-
"""Day 22（二）：四條攝取路徑，同一份內容。

    G   乾淨 markdown（前 21 天的原檔）
    T   PDF 的文字層（PyMuPDF 抽）
    V   150dpi 的頁面圖 → 離線 OCR（RapidOCR／PaddleOCR 的 ONNX 版）
    V+  退化過的掃描圖 → 同一個 OCR

⚠️ V 原本要用視覺模型（gpt-4o／gpt-4o-mini）當 OCR，**它拒絕了**。
四種問法、兩個模型、裁不裁掉「本文件僅供內部使用」的頁尾，一律回
「我無法逐字轉錄，但可以幫你總結」——而「總結」正好是 RAG 最不能接受的。
證據留在 `vision_cache_day22.json`（20 次呼叫、1.72 元、一個字都沒拿到），
`day22_plan.md` 的補記寫了這個改動。

## 一個刻意的設計：三條路徑共用同一個「還原」函式

`chunkers.chunk_by_structure` 靠的是 `### 第 N 條` 這個 markdown 標題，
而 PDF 與圖片裡沒有 `###`。所以 T／V／V+ 都要先把純文字**還原**成
有標題結構的 markdown——`reconstruct()` 做這件事，三條路徑用同一個函式。

**還原步驟本身就是一個會壞的地方**，而它在真實專案裡通常沒有人量。
今天它被量：每一條的字元相似度、數字有沒有被抄錯，逐條列出來。

## 頁首頁尾：三條路徑用同一把剪刀

PDF 的每一頁都有「○○股份有限公司　工作規則」與「第 N 頁／共 10 頁」。
T 抽得到它們，V 也會把它們逐字轉錄出來。
**所以清理的規則對三條路徑完全一樣**——不然量到的就不是「文字怎麼來的」，
而是「我幫誰清得比較乾淨」。

## OCR 的成本單位不是錢，是時間

離線 OCR 不用付 API 費用，但每頁要跑約 37 秒。所以攝取那一欄今天有兩個單位，
兩個都要報：**錢**（T 幾乎 0、OCR 0）與**時間**（T 0.2 秒、OCR 十幾分鐘）。
只報錢會讓人以為 OCR 是免費的。

用法：
    python ingest_day22.py            # 建四個 collection，印攝取品質報表
    python ingest_day22.py --no-ocr   # 只跑 G 與 T（不跑那十幾分鐘的 OCR）
"""
import argparse
import base64
import difflib
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

import fitz

import chroma_store
import chunkers
import judges
import rag_core

BASE_DIR = Path(__file__).parent
CORPUS_DIR = BASE_DIR / "corpus_day22"
PDF_PATH = CORPUS_DIR / "work_rules.pdf"
CLEAN_DIR = CORPUS_DIR / "clean"
SCAN_DIR = CORPUS_DIR / "scan"
CACHE_PATH = BASE_DIR / "vision_cache_day22.json"      # 視覺模型那次失敗的證據
OCR_CACHE_PATH = BASE_DIR / "ocr_cache_day22.json"
TEXT_DIR = CORPUS_DIR / "extracted"

VISION_MODEL = "gpt-4o-mini"
# 每百萬 token 的美元單價（輸入, 輸出），與 judges_day19 同一張表
VISION_PRICE = (0.15, 0.60)

PATHS = ("G", "T", "V", "V+")
PATH_LABEL = {
    "G": "G　乾淨 markdown（前 21 天的原檔）",
    "T": "T　PDF 的文字層",
    "V": "V　150dpi 頁面圖 → 離線 OCR",
    "V+": "V+　退化掃描圖 → 離線 OCR",
}
COLLECTION = {"G": "day22-g", "T": "day22-t", "V": "day22-v", "V+": "day22-vplus"}

# 頁首頁尾：三條路徑共用同一把剪刀
HEADER_RE = re.compile(r"^[○○〇oO0]{0,3}\s*股份有限公司.*工作規則\s*$")
# Day 23 修：附件的頁尾是「附件第 2 頁／共 2 頁」，前面多了兩個字，
# 原本 ^第 開頭的比對認不出來，於是頁尾混進了附圖一那一段。
# 三條路徑共用同一把剪刀，所以修在這裡。
FOOTER_RE = re.compile(r"^\S{0,4}?第\s*\d+\s*頁\s*[／/]\s*共\s*\d+\s*頁")
CHAPTER_RE = re.compile(r"^第\s*([一二三四五六七八九十]+)\s*章\s*(.+)$")
ARTICLE_RE = re.compile(r"^第\s*(\d+)\s*條\s*[（(](.+?)[）)]\s*(.*)$")
TITLE_RE = re.compile(r"^[○○〇]{0,3}.*工作規則$")

VISION_PROMPT = (
    "請逐字轉錄這一頁文件的全部文字內容，包含頁首與頁尾。"
    "保留原本的分行與段落，不要摘要、不要改寫、不要補充任何說明文字，"
    "也不要加上 markdown 標記。只輸出轉錄結果。")


# ------------------------------------------------------------------ 共用清理

NORM_FORM = "NFC"        # 見下方 docstring：選 NFC 還是 NFKC，差 59 條


def clean_lines(lines: list[str],
                form: str | None = NORM_FORM) -> tuple[list[str], int, int]:
    """丟掉頁首、頁尾與文件標題，並做 Unicode 正規化。

    回傳（留下的行, 丟掉幾行, 被正規化改動的字數）。

    ## 這一段是跑完第一版才發現的

    T 路徑抽到的「勞」「契」「六」落在 **CJK 相容表意文字區**（U+F9xx），
    不是常用區的那個碼位。字形一模一樣、肉眼看不出來，
    但對程式來說是不同的字——連「第六章」都因此比對不到，
    被當成內文併進上一條。這不是 PDF 的 bug，是字型 cmap 的正常行為。

    ## 選哪一種正規化，差 59 條

        不正規化   59 條的字元相似度平均 0.938，**完整還原 0 條**
        NFC        平均 1.000，**完整還原 59 條**——與原檔一字不差
        NFKC       平均 0.952，**完整還原 0 條**

    NFC 做的是「同一個字的不同碼位合併」，正好是這裡的病。
    NFKC 多做了「相容字元展開」——它會把全形括號 `（）` 換成半形 `()`，
    於是修好一個問題的同時，製造了另一個。
    **兩者差一個字母，結果差 59 條。** 三條路徑共用同一把剪刀。
    """
    kept, dropped, changed = [], 0, 0
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if form:
            norm = unicodedata.normalize(form, line)
            changed += sum(1 for a, b in zip(line, norm) if a != b)
            line = norm
        if HEADER_RE.match(line) or FOOTER_RE.match(line) or TITLE_RE.match(line):
            dropped += 1
            continue
        kept.append(line)
    return kept, dropped, changed


def reconstruct(lines: list[str]) -> str:
    """純文字 → 帶 markdown 標題的語料。三條路徑共用這一個函式。

    規則只有三條：
      「第X章 …」      → `## 第X章 …`
      「第 N 條（…）」  → `### 第 N 條（…）`
      其他             → 併進目前這一條的內文（PDF 會在句子中間斷行，所以不補空白）
    """
    out, buf = [], []

    def flush():
        if buf:
            out.append("".join(buf))
            buf.clear()

    for line in lines:
        m_chapter = CHAPTER_RE.match(line)
        if m_chapter:
            flush()
            out.append(f"## 第{m_chapter.group(1)}章 {m_chapter.group(2).strip()}")
            continue
        m_article = ARTICLE_RE.match(line)
        if m_article:
            flush()
            out.append(f"### 第 {int(m_article.group(1))} 條"
                       f"（{m_article.group(2).strip()}）")
            rest = m_article.group(3).strip()
            if rest:
                buf.append(rest)
            continue
        buf.append(line)
    flush()
    return ("# 工作規則" + chr(10) * 2
            + (chr(10) * 2).join(out) + chr(10))


# ------------------------------------------------------------------ 四條路徑

def text_G() -> str:
    return rag_core.load_document()


def raw_lines_T() -> list[str]:
    doc = fitz.open(str(PDF_PATH))
    lines = []
    for page in doc:
        lines += page.get_text().splitlines()
    doc.close()
    return lines


def text_T(form: str | None = NORM_FORM) -> tuple[str, int, int]:
    kept, dropped, changed = clean_lines(raw_lines_T(), form=form)
    return reconstruct(kept), dropped, changed


def _load_cache() -> dict:
    return json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}


def transcribe(image: Path) -> dict:
    """一頁圖 → 逐字轉錄。快取以圖片內容 hash 為 key，重跑不重複付錢。"""
    raw = image.read_bytes()
    key = hashlib.sha256(raw + VISION_PROMPT.encode()).hexdigest()[:16]
    cache = _load_cache()
    if key in cache:
        return cache[key]

    mime = "image/png" if image.suffix == ".png" else "image/jpeg"
    b64 = base64.b64encode(raw).decode()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": VISION_PROMPT},
        {"type": "image_url",
         "image_url": {"url": f"data:{mime};base64,{b64}"}}]}]

    # 一張 150dpi 的 A4 頁很吃 token，連續送十張會撞 TPM 上限（每分鐘 20 萬）。
    # 退避重試，並且每一頁成功就立刻寫快取——中途被打斷不必重付已經付過的錢。
    response = None
    for attempt in range(6):
        try:
            response = rag_core.get_client().chat.completions.create(
                model=VISION_MODEL, temperature=0, messages=messages)
            break
        except Exception as exc:
            if "rate_limit" not in str(exc).lower() or attempt == 5:
                raise
            time.sleep(8 * (attempt + 1))
    if response is None:
        raise RuntimeError("視覺模型重試六次仍失敗")
    out = {"text": response.choices[0].message.content.strip(),
           # 圖片 token 沒辦法離線重數（OpenAI 的切塊規則沒有公開），
           # 所以這一項**只能用 API 回報的 usage**，並在報表裡標明。
           "input_tokens": response.usage.prompt_tokens,
           "output_tokens": response.usage.completion_tokens}
    cache[key] = out
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return out


_OCR = None


def _ocr_engine():
    """延遲載入：模型第一次用的時候才載，`--no-ocr` 就完全不碰它。"""
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


def ocr_page(image: Path) -> dict:
    """一頁圖 → OCR 出來的行。快取以圖片內容 hash 為 key。

    OCR 不花錢，但每頁約 37 秒——**時間也是成本**，所以一起記進快取。
    """
    raw = image.read_bytes()
    key = hashlib.sha256(raw).hexdigest()[:16]
    cache = json.loads(OCR_CACHE_PATH.read_text(encoding="utf-8"))         if OCR_CACHE_PATH.exists() else {}
    if key in cache:
        return cache[key]

    start = time.time()
    result, _ = _ocr_engine()(str(image))
    out = {"lines": [r[1] for r in (result or [])],
           "seconds": round(time.time() - start, 1)}
    cache[key] = out
    OCR_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False),
                              encoding="utf-8")
    return out


def text_V(folder: Path, pattern: str,
           form: str | None = NORM_FORM) -> tuple[str, int, int, dict]:
    pages = sorted(folder.glob(pattern))
    lines, bill = [], {"in": 0, "out": 0, "pages": len(pages), "seconds": 0.0}
    for p in pages:
        res = ocr_page(p)
        bill["seconds"] += res["seconds"]
        lines += res["lines"]
    kept, dropped, changed = clean_lines(lines, form=form)
    return reconstruct(kept), dropped, changed, bill


# ------------------------------------------------------------------ 攝取品質

def article_map(text: str) -> dict[int, str]:
    """條號 → 該條的內文（不含標題），用來跟原檔逐條比對。"""
    out = {}
    for chunk in chunkers.chunk_by_structure(text):
        m = chunkers.ARTICLE_NO_PATTERN.search(chunk["text"].splitlines()[0])
        if m:
            body = chunk["text"].split(chr(10), 1)
            out[int(m.group(1))] = "".join(body[1].split()) if len(body) > 1 else ""
    return out


NUM_RE = re.compile(r"[零一二三四五六七八九十百千萬兩0-9]+分之[零一二三四五六七八九十百千萬兩0-9]+"
                    r"|[零一二三四五六七八九十百千萬兩0-9]+")


def numbers(text: str) -> list[str]:
    return NUM_RE.findall(text)


def fidelity(gold: dict[int, str], got: dict[int, str]) -> list[dict]:
    """逐條比對：字元相似度、數字有沒有被抄錯或漏掉。"""
    rows = []
    for no, gold_text in sorted(gold.items()):
        mine = got.get(no)
        if mine is None:
            rows.append({"no": no, "ratio": 0.0, "missing": True,
                         "num_lost": len(numbers(gold_text)), "num_added": 0})
            continue
        gold_nums, my_nums = numbers(gold_text), numbers(mine)
        lost = list(gold_nums)
        for n in my_nums:
            if n in lost:
                lost.remove(n)
        added = list(my_nums)
        for n in gold_nums:
            if n in added:
                added.remove(n)
        rows.append({
            "no": no, "missing": False,
            "ratio": difflib.SequenceMatcher(None, gold_text, mine).ratio(),
            "num_lost": len(lost), "num_added": len(added),
            "lost_examples": lost[:4], "added_examples": added[:4]})
    return rows


# ------------------------------------------------------------------ 建索引

def build_collection(name: str, text: str):
    """一條路徑一個 collection。不動 `chroma_store.COLLECTION`（前 21 天的庫）。"""
    chunks = chunkers.chunk_by_structure(text)
    articles = chunkers.parse_articles(text)
    vectors, tokens, cached = rag_core.get_embeddings([c["text"] for c in chunks])
    client = chroma_store.get_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(
        name, configuration={"hnsw": {"space": "cosine"}})
    metadatas = [chroma_store.article_metadata(c, articles) for c in chunks]
    collection.add(ids=[f"art-{m['article_no']}-{i}"
                        for i, m in enumerate(metadatas)],
                   embeddings=vectors.tolist(),
                   documents=[c["text"] for c in chunks],
                   metadatas=metadatas)
    return collection, chunks, articles, tokens


def ingest(use_ocr: bool = True) -> dict:
    """跑四條路徑，回傳每條路徑的語料、索引與帳。"""
    out = {}
    gold_text = text_G()
    gold = article_map(gold_text)

    sources = [("G", lambda: (gold_text, 0, 0, None)),
               ("T", lambda: (*text_T(), None))]
    if use_ocr:
        sources += [("V", lambda: text_V(CLEAN_DIR, "*.png")),
                    ("V+", lambda: text_V(SCAN_DIR, "*.jpg"))]

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in sources:
        text, dropped, changed, bill = fn()
        (TEXT_DIR / f"{name.replace('+', 'plus')}.md").write_text(
            text, encoding="utf-8")
        collection, chunks, articles, emb_tokens = build_collection(
            COLLECTION[name], text)
        rows = fidelity(gold, article_map(text))
        out[name] = {
            "text": text, "collection": collection, "chunks": chunks,
            "articles": articles, "dropped_lines": dropped,
            "nfkc_changed": changed, "bill": bill,
            "fidelity": rows, "emb_tokens": emb_tokens,
            "n_articles": len(articles), "chars": len(text),
        }
    return out


def vision_cost_twd(bill: dict) -> float:
    """視覺模型那次失敗花的錢（V 改成離線 OCR 之後，這裡固定是 0）。"""
    if not bill or not bill.get("in"):
        return 0.0
    usd = bill["in"] / 1e6 * VISION_PRICE[0] + bill["out"] / 1e6 * VISION_PRICE[1]
    return rag_core.twd(usd)


def refusal_bill() -> tuple[int, float, int]:
    """數一數視覺模型那 20 次拒絕的帳：幾次、多少錢、幾次真的拿到文字。"""
    if not CACHE_PATH.exists():
        return 0, 0.0, 0
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    usd = sum(v["input_tokens"] / 1e6 * VISION_PRICE[0]
              + v["output_tokens"] / 1e6 * VISION_PRICE[1] for v in cache.values())
    got = sum(1 for v in cache.values() if len(v["text"]) > 200)
    return len(cache), rag_core.twd(usd), got


def main() -> None:
    ap = argparse.ArgumentParser(description="Day 22：四條攝取路徑")
    ap.add_argument("--no-ocr", action="store_true", help="只跑 G 與 T")
    args = ap.parse_args()

    data = ingest(use_ocr=not args.no_ocr)
    nl = chr(10)
    print(f"{nl}{'路徑':<4}{'條數':>5}{'字數':>8}{'頁首頁尾':>10}"
          f"{'平均相似度':>12}{'完整還原':>10}{'數字漏':>8}{'攝取秒數':>10}")
    for name in [p for p in PATHS if p in data]:
        d = data[name]
        rows = d["fidelity"]
        avg = sum(r["ratio"] for r in rows) / len(rows)
        perfect = sum(1 for r in rows if r["ratio"] >= 0.999)
        lost = sum(r["num_lost"] for r in rows)
        secs = (d["bill"] or {}).get("seconds", 0.0)
        print(f"{name:<4}{d['n_articles']:>5}{d['chars']:>8}"
              f"{d['dropped_lines']:>10}{avg:>12.3f}"
              f"{perfect:>8}/{len(rows)}{lost:>8}{secs:>10.0f}")

    n, twd, got = refusal_bill()
    if n:
        print(f"{nl}視覺模型那次失敗：{n} 次呼叫、{twd:.2f} 元、"
              f"拿到文字的 {got} 次")

    print(f"{nl}最不像原文的條（各路徑前 5）：")
    for name in [p for p in PATHS if p in data and p != "G"]:
        worst = sorted(data[name]["fidelity"], key=lambda r: r["ratio"])[:5]
        print(f"  {name}: " + "、".join(
            f"第 {r['no']} 條 {r['ratio']:.2f}" for r in worst))


if __name__ == "__main__":
    main()
