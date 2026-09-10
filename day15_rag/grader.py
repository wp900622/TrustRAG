# -*- coding: utf-8 -*-
"""Day 14 的新尺：判定「答案對不對」，而不只是「有沒有找到對的條文」。

Day 8~13 的尺是 top-1 命中率——排第一名的 chunk 是不是預期條文。
接上生成之後這把尺不夠用了：條文找對，答案還是可能寫錯數字。

判定方式：**關鍵事實比對**。每題在 questions.json 預先寫下答案必須出現的
事實（facts，通常是數字與制度名詞），全部出現才算對；另可寫下不該出現的
錯誤事實（forbid），出現任一個就算錯。

為什麼不用 LLM 當裁判（LLM-as-judge）：
- 裁判本身會錯、會偏心，而且每次重跑要再付一次錢，數字不可重現
- 本日語料的答案都是條號＋數字，用事實比對就夠，還完全免費

這把尺的偏誤先自首（實驗設計那節會再講一次）：
1. 它只看「有沒有講到」，不看「講得對不對」——答案若同時列出正確與錯誤
   說法，只要正確的那句在就算對。forbid 欄位是用來補這個洞的。
2. 中文數字與阿拉伯數字要先正規化（「三日」＝「3天」），否則會誤殺。
3. 事實是我自己挑的，挑得寬鬆就分數虛高。所以 questions.json 裡的 facts
   一律照條文原文取，不是照模型答案回頭湊。
"""
import re
import unicodedata

CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNITS = {"十": 10, "百": 100, "千": 1000}
CN_RUN = re.compile("[" + "".join(CN_DIGITS) + "".join(CN_UNITS) + "]+")
# 判定時無意義的字元：空白與各種標點，一律拿掉再比
NOISE = re.compile(r"[\s,，、。：:；;（）()「」【】\[\]．.\-─~～]+")


def _cn_to_int(text: str) -> int:
    """把一串中文數字轉成整數，支援到千位（十四→14、一百三十八→138）"""
    section = number = 0
    for char in text:
        if char in CN_DIGITS:
            number = CN_DIGITS[char]
        elif char in CN_UNITS:
            section += (number or 1) * CN_UNITS[char]
            number = 0
    return section + number


def normalize(text: str) -> str:
    """正規化：全形轉半形、中文數字轉阿拉伯數字、統一同義詞、去標點空白。

    答案與 facts 兩邊都要跑過這個函式，比對才公平。
    """
    text = unicodedata.normalize("NFKC", text)
    text = CN_RUN.sub(lambda m: str(_cn_to_int(m.group())), text)
    # 「天」與「日」在規章語境同義；「新臺幣」是贅詞，金額本身才是事實
    text = text.replace("天", "日").replace("新臺幣", "").replace("新台幣", "")
    text = text.replace("小時以上", "小時")  # 避免「46小時」被「以上」切開
    return NOISE.sub("", text)


def grade(answer: str, facts: list[str], forbid: list[str] | None = None
          ) -> tuple[bool, list[str], list[str]]:
    """判定一題。回傳 (是否正確, 漏掉的 facts, 命中的 forbid)"""
    normalized = normalize(answer)
    missing = [f for f in facts if normalize(f) not in normalized]
    hit_forbid = [f for f in (forbid or []) if normalize(f) in normalized]
    return (not missing and not hit_forbid), missing, hit_forbid


def looks_like_refusal(answer: str) -> bool:
    """粗略偵測模型是否表示「條文裡查不到」。

    Day 14 只用它來解釋失敗案例（答案沒錯，但它選擇不答）；
    Day 15 會把拒答變成正式指標，屆時判定會嚴謹化。
    """
    keywords = ["查不到", "找不到", "未提及", "沒有提到", "無法回答",
                "未規定", "沒有規定", "不確定", "無相關", "沒有相關"]
    return any(k in answer for k in keywords)


# =================================================== Day 15：可信度的四把尺
#
# Day 14 只有 grade()：答案含不含關鍵事實。它對「模型說不知道」一律判錯，
# 也對「規章沒寫卻掰得出答案」完全沒有偵測能力——那是 Day 15 要補的洞。
#
# 四個新指標：
#   幻覺率      陷阱題中給出具體答案的比例          （越低越好）
#   正確處理率  陷阱題中明說未規定／轉指他辦法的比例  （越高越好）
#   過度拒答率  正常 15 題中被拒答掉的比例          （越低越好）
#   引用正確率  有標條號的答案中，條號真的對的比例    （P2／P3 才有）

# 拒答詞：模型表示「這裡查不到」的說法。
#
# 這份清單是關鍵詞判定的死穴，而且是實測踩到的：第 102 題的正確拒答寫
# 「沒有提及」，清單裡卻只有「沒有提到」，差一個字就被判成「模糊」。
# 關鍵詞清單永遠有漏，所以逐題明細一定要留答案原文供人工覆核——
# 這也是為什麼本系列不用 LLM 裁判卻堅持印出全部答案。
REFUSAL_WORDS = ["查不到", "找不到", "未提及", "沒有提及", "沒有提到",
                 "未涉及", "沒有涉及", "無法回答", "無法提供",
                 "未規定", "沒有規定", "未規範", "無此規定",
                 "不確定", "無相關", "沒有相關",
                 "未涵蓋", "未載明", "未明確", "無法得知", "未說明"]

# 轉指詞：條文寫明「依另一辦法規定」時的正確說法。它不是拒答，
# 但同樣屬於「沒有自己編數字」的好行為，所以要跟拒答分開記。
DEFERRAL_WORDS = ["另行規定", "另定", "另有規定", "辦法規定", "依該辦法"]

# 具體答案的訊號：帶單位的數字。用來做拒答判定的第二段——
# 「規章未明確規定，但依勞基法通常為 2 年」這種答案兩個訊號都會亮。
CONCRETE = re.compile("[0-9]+ *(日|天|小時|個月|年|元|次|人|%)")

# 答案裡的條號引用，例如「（第 24 條）」「依第 43 條」
CITATION = re.compile("第[ ]*([0-9]+)[ ]*條")


def has_refusal_word(answer: str) -> bool:
    """第一段判定：有沒有出現拒答詞（含轉指詞）"""
    return any(w in answer for w in REFUSAL_WORDS + DEFERRAL_WORDS)


def has_concrete_claim(answer: str, extra_markers: list[str] | None = None) -> bool:
    """第二段判定：有沒有真的給出一個具體答案。

    normalize() 會把中文數字轉成阿拉伯數字，所以「二年」也抓得到。
    extra_markers 是逐題自訂的幻覺特徵（例如引用了外部法規的名稱）。
    """
    text = normalize(answer)
    if CONCRETE.search(text):
        return True
    return any(re.search(normalize_pattern(m), text)
               for m in (extra_markers or []))


def refusal_verdict(answer: str, extra_markers: list[str] | None = None) -> str:
    """兩段判定，回傳四種狀態之一。

    這是 Day 15 最重要的一段程式。Day 14 的 looks_like_refusal() 只做第一段，
    會把「規章未明確規定，但依勞基法通常為 2 年」判成拒答——那是幻覺穿了
    拒答的外衣，是四種狀態裡最該被抓出來的一種。

        refusal       明說查不到，且沒有給具體答案     → 好行為
        fake_refusal  說了查不到，卻又給出具體答案     → 假拒答，計入幻覺
        answered      沒有拒答詞，直接給出具體答案     → 正常題是好事，陷阱題是幻覺
        vague         既不拒答也沒給具體答案（純建議洽詢人資）→ 單獨記，不算兩邊
    """
    refused = has_refusal_word(answer)
    concrete = has_concrete_claim(answer, extra_markers)
    if refused and not concrete:
        return "refusal"
    if refused and concrete:
        return "fake_refusal"
    if concrete:
        return "answered"
    return "vague"


def grade_trap(answer: str, question: dict) -> tuple[str, list[str]]:
    """判定一題陷阱題。回傳 (判定, 命中的幻覺特徵)。

        proper        明說規章未規定，或正確轉指另一個辦法 → 好行為
        hallucination 給出了語料裡沒有的具體答案
        fake_refusal  同時做了上面兩件事——「規章未明確規定，但通常是 X」。
                      它計入幻覺，但要單獨列出來：只看拒答關鍵詞的判定
                      會把它算成好行為，這是 Day 14 那把尺最大的洞
        vague         既沒說查不到、也沒給具體答案（只叫你問人資）

    「具體答案」的判定不用全域的 CONCRETE，改用逐題的 hallucination_markers：
    第 105 題（加班 3 小時領多少錢）的正確答案本身就含「二小時以內」，
    用全域規則會把它誤判成假拒答。幻覺的長相是逐題的——A 類是無中生有，
    B 類是把隔壁條文的數字套過來——判定條件也只能逐題寫。

    marker 有一個坑：它是子字串比對，會被**答案覆述問題**騙到。
    第 102 題原本的 marker 寫「有健身房」，結果正確拒答
    「沒有提及公司有健身房或運動補助」也命中了，幻覺率因此虛高。
    marker 要寫成「只有在編答案時才會出現」的字串（「設有健身房」），
    不能寫成問題本身的字眼。
    """
    markers = question.get("hallucination_markers", [])
    text = normalize(answer)
    hit_markers = [m for m in markers if re.search(normalize_pattern(m), text)]
    said_unknown = (has_refusal_word(answer)
                    or any(s in answer for s in question.get("proper_signals", [])))
    if hit_markers:
        return ("fake_refusal" if said_unknown else "hallucination"), hit_markers
    if said_unknown:
        return "proper", []
    return "vague", []


def normalize_pattern(pattern: str) -> str:
    """正規表示式本身也要跟著 normalize()。

    normalize() 會把空白與標點刪掉，所以 marker 裡的 " *" 要拿掉，
    否則「[0-9]+ *元」永遠比不到正規化後的「3000元」。
    """
    return pattern.replace(" *", "")


def cited_articles(answer: str) -> list[int]:
    """抓出答案裡引用的條號，去重後保持出現順序"""
    seen = []
    for m in CITATION.finditer(answer):
        no = int(m.group(1))
        if no not in seen:
            seen.append(no)
    return seen


def citation_verdict(answer: str, context_nos: list[int],
                     expected_no: int | None) -> str:
    """判定引用品質，三種狀態。

        none        沒標條號
        fabricated  標了一個沒出現在 context 裡的條號 → 憑空生成的依據，
                    比答錯數字更危險，因為引用讓使用者更相信它
        grounded    條號都在 context 裡，但沒有一個是預期條文
        correct     條號都在 context 裡，且包含預期條文
    """
    cited = cited_articles(answer)
    if not cited:
        return "none"
    if any(no not in context_nos for no in cited):
        return "fabricated"
    if expected_no is not None and expected_no in cited:
        return "correct"
    return "grounded"
