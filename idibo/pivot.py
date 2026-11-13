"""Cross-lingual retrieval: translate a Yorùbá/Hausa/Igbo question into English before searching.

The knowledge base (INEC pages, Electoral Act 2026) is English. Searching with the native question found the right
source for only 7/20 Yorùbá and 4/20 Hausa fact questions vs 14/20 in English; searching with the native question
plus the model's own English translation found 12/20 and 16/20 (scripts/retrieval_eval.py).
The few-shot examples below are not benchmark items. Pidgin is skipped: it already retrieves like English.
"""
import re

PIVOT_LANGS = {"yo": "Yorùbá", "ha": "Hausa", "ig": "Igbo"}

_SHOTS = [
    ("Hausa", "Ina zan je in karɓi katin zaɓe na?", "Where do I go to collect my voter card?"),
    ("Yorùbá", "Ṣé mo lè dìbò tí mi ò bá ní káàdì ìdìbò?", "Can I vote if I don't have a voter card?"),
    ("Igbo", "Kedu afọ ole m ga-eju tupu m nwee ike ịtụ vootu?", "How old must I be before I can vote?"),
]


def pivot_messages(question: str, lang: str) -> list[dict]:
    msgs = [{"role": "system", "content": "You translate Nigerian-language questions into plain English. "
                                          "Output only the English translation, never the original."}]
    for name, q, en in _SHOTS:
        msgs += [{"role": "user", "content": f"{name}: {q}"}, {"role": "assistant", "content": en}]
    msgs.append({"role": "user", "content": f"{PIVOT_LANGS[lang]}: {question}"})
    return msgs


def clean_pivot(text: str) -> str:
    """First line only; empty if the model echoed the original instead of translating."""
    line = (text or "").strip().split("\n")[0][:300]
    if sum(line.count(c) for c in "ẹọṣɓɗƙịụṅ") >= 2:
        return ""
    return line


def search_query(question: str, english: str) -> str:
    return f"{question} {english}".strip() if english else question
