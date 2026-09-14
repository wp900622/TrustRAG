# -*- coding: utf-8 -*-
"""Day 19 的補充診斷。全部走快取，0 元。

正式報表（`experiment_result_day19.md`）只列「B0 與 B3 判定不同」的答案，
所以 305 那五筆——從頭到尾都判錯的那五筆——一個字都沒出現。
而它正是今天最重要的一筆：**我昨天對它的診斷是錯的。**

這支腳本補三張表，結果寫進 `experiment_result_day19_detail.md`：

  一、305 的 checkpoint 逐項 × 五個變體
      看「相反」那個標籤怎麼隨著 prompt 改版在三個 checkpoint 之間跳。
  二、兩個裁判（mini／4o）彼此的一致率
      換模型買到了什麼，逐筆列出來。
  三、B3 與 C 的錯是不是同一批
      如果不是，那「兩把尺不一致就叫人看」是可行的；
      這一節就是在算那個作法的覆蓋率與天花板。
"""
from pathlib import Path

import judges_day19 as j19
import pipeline
import run_experiment_day18 as d18
import run_experiment_day19 as d19

OUT_PATH = Path(__file__).parent / "experiment_result_day19_detail.md"
NL = chr(10)


def main() -> None:
    questions = d19.load_json(d19.QUESTIONS_PATH)
    rubrics = d19.load_json(d19.RUBRIC_PATH)
    qmap = {q["id"]: q for q in questions}
    rmap = {r["id"]: r["checkpoints"] for r in rubrics}
    collection, chunks, articles, _, _ = pipeline.build_index()
    texts = d18.article_text_map(chunks, articles)
    rows, _, _ = d18.rebuild_day17_answers(collection, questions)
    labels = d18.load_labels(rows)

    variants = ["B0", "B1", "B2", "B3", "B3o"]
    out = ["# Day 19 補充診斷（正式報表沒列到的三張表）", "",
           "全部走快取重算，0 元。數字與 `experiment_result_day19.md` 同一次跑的結果。", ""]

    # ------------------------------------------------------------------ 一
    out += ["## 一、305 的 checkpoint 逐項：「相反」在三個判斷點之間跳", "",
            "305 的五筆誤殺，B2 與 B3 一筆都沒救回。原因不是某一個 checkpoint "
            "修不好，是**每換一次 prompt，被判「相反」的就換成另一個 checkpoint**，"
            "而聚合規則是「任一相反即錯誤」。修 prompt 只是把氣泡推到隔壁。", ""]
    for r in [x for x in rows if x["id"] == 305]:
        per = {v: d19.grade_one(v, qmap[305], r["answer"], rmap[305], texts)
               for v in variants}
        out += [f"### {r['key']}　基準：**{labels[r['key']]}**", "",
                "> " + r["answer"].replace(NL, " "), "",
                "判定：" + "｜".join(f"{v}={per[v]['verdict']}" for v in variants), "",
                "| checkpoint | 種類 | " + " | ".join(variants) + " |",
                "|---" * (2 + len(variants)) + "|"]
        for i, cp in enumerate(rmap[305]):
            out.append(f"| `{cp['id']}` {cp['point']} | {cp['kind']} | "
                       + " | ".join(per[v]["detail"][i]["label"] for v in variants)
                       + " |")
        out.append("")

    # ------------------------------------------------------------------ 二
    same, flips = 0, []
    for r in rows:
        q = qmap[r["id"]]
        blob = d18.articles_blob(q, texts)
        c = j19.grade_judge(q["question"], blob, r["answer"], model=j19.MINI)
        co = j19.grade_judge(q["question"], blob, r["answer"], model=j19.GPT4O)
        if c["verdict"] == co["verdict"]:
            same += 1
        else:
            flips.append((r["key"], labels[r["key"]], c["verdict"],
                          co["verdict"], co["reason"]))
    out += ["## 二、兩個裁判彼此的一致率（尺 C vs C′，40 個答案）", "",
            f"**{same}/{len(rows)} 判定相同。** 換一個貴 20 倍的模型，"
            f"改變了 {len(flips)} 筆判定。", "",
            "| 答案 | 基準 | C（mini） | C′（4o） | 誰對 | C′ 的理由 |",
            "|---|---|---|---|---|---|"]
    for key, human, c, co, reason in flips:
        who = ("C′" if co == human else "C" if c == human else "都不對")
        out.append(f"| {key} | {human} | {c} | {co} | {who} | "
                   f"{reason.replace(NL, ' ')[:70]} |")
    out.append("")

    # ------------------------------------------------------------------ 三
    b3 = {r["key"]: d19.grade_one("B3", qmap[r["id"]], r["answer"],
                                  rmap[r["id"]], texts)["verdict"] for r in rows}
    c_v = {r["key"]: d19.grade_one("C", qmap[r["id"]], r["answer"],
                                   rmap[r["id"]], texts)["verdict"] for r in rows}
    b3_bad = {k for k, v in b3.items() if v != labels[k]}
    c_bad = {k for k, v in c_v.items() if v != labels[k]}
    dis = sorted([k for k in b3 if b3[k] != c_v[k]],
                 key=lambda x: (int(x.split("@")[0]), int(x.split("k")[1])))
    agree_keys = [k for k in b3 if b3[k] == c_v[k]]
    agree_bad = [k for k in agree_keys if b3[k] != labels[k]]
    bad_union = b3_bad | c_bad
    caught = [k for k in bad_union if k in dis]

    out += ["## 三、B3 與 C 的錯是不是同一批", "",
            f"- B3 判錯 **{len(b3_bad)}** 筆（偏誤殺）、C 判錯 **{len(c_bad)}** 筆（偏放水）",
            f"- 兩把尺都判錯：**{len(b3_bad & c_bad)}** 筆"
            f"（{'、'.join(sorted(b3_bad & c_bad))}）",
            f"- 兩把尺都判對：**{len(rows) - len(bad_union)}/{len(rows)}**", "",
            "### 如果「兩把尺不一致就叫人看」", "",
            f"- 兩把尺判定不同：**{len(dis)}/{len(rows)}**"
            f"（{len(dis) / len(rows):.0%}）——這是要人看的量",
            f"- 兩把尺判定相同：{len(agree_keys)} 筆，其中**一起錯 {len(agree_bad)} 筆**"
            f"（{'、'.join(agree_bad) or '無'}）",
            f"- 至少一把判錯的答案共 **{len(bad_union)}** 筆，"
            f"其中 **{len(caught)}** 筆落在那 {len(dis)} 筆不一致裡（覆蓋率 "
            f"{len(caught) / len(bad_union):.0%}）", "",
            f"**天花板就是那 {len(agree_bad)} 筆一起錯的**："
            "兩把尺達成共識而且共識是錯的，任何「看不一致」的機制都攔不到它。"
            "**共識不等於正確。**", "",
            "⚠️ 這是在同一批 40 筆上算出來的描述，沒有用第二批答案驗證過。"
            "它是這批資料的性質，不是對下一批的預測。", "",
            "| 答案 | 基準 | B3 | C | 至少一把對 |", "|---|---|---|---|---|"]
    for k in dis:
        ok = "✅" if (b3[k] == labels[k] or c_v[k] == labels[k]) else "❌ 都不對"
        out.append(f"| {k} | {labels[k]} | {b3[k]} | {c_v[k]} | {ok} |")
    out.append("")

    OUT_PATH.write_text(NL.join(out), encoding="utf-8")
    print(f"補充診斷已寫入 {OUT_PATH.name}")
    print(f"  兩個裁判一致：{same}/{len(rows)}")
    print(f"  B3 與 C 判定不同：{len(dis)}/{len(rows)}"
          f"｜至少一把錯的 {len(bad_union)} 筆中，{len(caught)} 筆落在不一致裡")
    print(f"  兩把尺一起錯：{len(agree_bad)} 筆 {agree_bad}")


if __name__ == "__main__":
    main()
