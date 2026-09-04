# -*- coding: utf-8 -*-
"""Day 8 互動 demo：輸入一個問題，看程式從《員工請假管理規範》裡找出最相關的三條條文。

用法：
    python search_demo.py                    # 互動模式，輸入空行結束
    python search_demo.py 特休沒休完會怎樣    # 直接帶一個問題
"""
import sys

import rag_core


def build_index() -> tuple[list[dict], list]:
    """讀文件並建立全部條文的向量索引，順便印出這一步花了多少錢"""
    articles = rag_core.load_articles()
    vectors, used_tokens, cache_hits = rag_core.get_embeddings(
        [article["text"] for article in articles]
    )
    print(f"已載入 {len(articles)} 條條文"
          f"（快取命中 {cache_hits} 條，本次 API 消耗 {used_tokens} tokens"
          f" ≈ {rag_core.cost_usd(used_tokens):.6f} 美元）")
    return articles, vectors


def answer(question: str, articles: list[dict], vectors: list) -> None:
    """對單一問題檢索並印出 top-3 結果"""
    results, _ = rag_core.search(question, articles, vectors, top_k=3)
    print(f"\n問題：{question}")
    for rank, item in enumerate(results, start=1):
        preview = item["text"].splitlines()[1][:40] if "\n" in item["text"] else ""
        print(f"  {rank}. [{item['score']:.4f}] {item['title']}　{preview}…")


def main() -> None:
    articles, vectors = build_index()

    # 命令列直接帶問題：跑一次就結束
    if len(sys.argv) > 1:
        answer(" ".join(sys.argv[1:]), articles, vectors)
        return

    # 互動模式：輸入空行結束
    print("輸入問題開始檢索（直接按 Enter 結束）：")
    while True:
        question = input("> ").strip()
        if not question:
            break
        answer(question, articles, vectors)


if __name__ == "__main__":
    main()
