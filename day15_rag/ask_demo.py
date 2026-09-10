# -*- coding: utf-8 -*-
"""互動 demo：問一題，看完整的 RAG 過程。

    python ask_demo.py 特休沒休完會怎樣
    python ask_demo.py --k 5 忘記打卡要怎麼補救
    python ask_demo.py --no-context 特休沒休完會怎樣   # 對照組：不給條文直接問

刻意把中間過程全印出來（檢索到哪幾條、context 多長、答案、帳單）——
RAG 出錯時要能一眼分辨是「沒找到」還是「找到了但答錯」。
"""
import argparse

import pipeline
import rag_core


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 14 RAG 問答 demo")
    parser.add_argument("question", nargs="+", help="要問的問題")
    parser.add_argument("--k", type=int, default=3, help="餵給模型幾條條文（預設 3）")
    parser.add_argument("--no-context", action="store_true",
                        help="不做檢索，直接問模型（對照組）")
    args = parser.parse_args()
    question = " ".join(args.question)

    collection = None
    if not args.no_context:
        collection, chunks, _, tokens, cached = pipeline.build_index()
        print(f"索引：{len(chunks)} 塊條文"
              f"（本次計費 {tokens} tokens、快取命中 {cached} 塊）\n")

    result = pipeline.answer(collection, question, k=args.k,
                             with_context=not args.no_context)

    if result.hits:
        print(f"檢索 top-{args.k}：")
        for i, hit in enumerate(result.hits, 1):
            meta = hit["meta"]
            print(f"  [{i}] 第 {meta['article_no']:>2} 條 {meta['title']}"
                  f"　相似度 {hit['similarity']:.3f}　{meta['chapter']}")
        print(f"\ncontext：{result.context_chars} 字\n")
    else:
        print("（對照組：沒有給任何條文）\n")

    print("答案：")
    print(result.answer)

    usd = rag_core.chat_cost_usd(result.input_tokens, result.output_tokens)
    usd += rag_core.embedding_cost_usd(result.embedding_tokens)
    billed = result.input_tokens + result.output_tokens
    print(f"\n帳單：輸入 {result.input_tokens}、輸出 {result.output_tokens} tokens"
          f"（{'快取命中，本次 0 元' if billed == 0 else f'≈ 新台幣 {rag_core.twd(usd):.5f} 元'}）")


if __name__ == "__main__":
    main()
