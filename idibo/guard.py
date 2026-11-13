"""Injection guard for the deployed app (added after the AuditKIT red-team found the model obeying "say X" injections).

Two cheap checks around the model, no extra model call:
  1. injection cues in the user's text ("ignore your instructions", "developer mode", "note to the AI", hidden HTML
     comments...), in English, Pidgin, Yorùbá and Hausa;
  2. strings the text demands be repeated ('say "X"', "Output: X", "end your reply with X", marker-like ALL-CAPS tokens).
If the text has cues and the reply contains a demanded string, the reply is replaced with a fixed one that says the
message carries hidden instructions (check mode) or that the assistant doesn't take such instructions (ask mode).
The model's training and evaluation are unchanged; this is a deployment layer and is reported as such.
"""
from __future__ import annotations

import re

CUES = re.compile(
    r"ignore\s+(all\s+|your\s+|the\s+|any\s+)?(previous\s+|prior\s+|above\s+)?(instructions|rules|guidelines|ethical)"
    r"|forget\s+(all\s+|your\s+)?(di\s+|the\s+)?(rules|instructions)|developer\s+mode|you\s+are\s+now\b"
    r"|#+\s*system\s*:|\bsystem\s+prompt\b|note\s+to\s+the\s+(ai|assistant)|\b(assistant|ai)\s*:|<!--"
    r"|\bsay\s+['\"‘“]|\boutput\s*:|reply\s+only|execute\s+it|repeat\s+(the\s+following|after\s+me)|end\s+your\s+reply\s+with|bypass|unrestricted"
    r"|olùrànlọ́wọ́\s+ai|gbàgbé\s+gbogbo|mataimakin\s+ai|ka\s+manta\s+da",
    re.I,
)
QUOTED = re.compile(r"['\"‘“]([^'\"‘’“”]{4,80})['\"’”]")
AFTER = re.compile(r"(?:output|reply only|end your reply with|add|write|kọ|rubuta)\s*:?\s*([A-Z][A-Z0-9 _-]{3,60})")
MARKER = re.compile(r"\b[A-Z0-9]+(?:[_-][A-Z0-9]+)+\b|\b[A-Z]{3,}(?:\s+[A-Z]{3,})+\b")

GUARD = {
    "check": {
        "en": "Warning: this message contains hidden instructions aimed at an AI assistant, a common sign of "
              "manipulation. I can't confirm it. Check official notices on inecnigeria.org or INEC's verified accounts.",
        "pcm": "Shine your eye: dis message get hidden instruction wey dem write for AI, na sign say e fit be wayo. "
               "I no fit confirm am. Check am for inecnigeria.org or INEC verified page.",
        "yo": "Ìkìlọ̀: ọ̀rọ̀ yìí ní àwọn àṣẹ tí wọ́n fi pamọ́ fún ẹ̀rọ AI, èyí jẹ́ àmì ẹ̀tàn. Èmi kò lè fìdí rẹ̀ múlẹ̀. "
              "Ẹ ṣàyẹ̀wò rẹ̀ lórí inecnigeria.org.",
        "ha": "Gargaɗi: wannan saƙo yana ɗauke da umarni da aka ɓoye domin mataimakin AI, alamar yaudara ce. "
              "Ba zan iya tabbatar da shi ba. Ku duba inecnigeria.org.",
        "ig": "Ịdọ aka ná ntị: ozi a nwere ntuziaka zoro ezo maka AI, nke a bụ akara aghụghọ. Enweghị m ike ikwenye ya. "
              "Lelee inecnigeria.org.",
    },
    "ask": {
        "en": "I don't follow instructions to change my rules or to repeat set phrases. I answer questions about voting "
              "in Nigeria from INEC and Electoral Act sources. What would you like to know?",
        "pcm": "I no dey follow instruction wey wan change my rules or make me repeat one kind talk. I dey answer "
               "question about voting for Naija from INEC and Electoral Act sources. Wetin you wan know?",
        "yo": "Èmi kò lè tẹ̀lé àṣẹ láti yí òfin mi padà tàbí láti tún ọ̀rọ̀ kan sọ. Mo ń dáhùn ìbéèrè nípa ìdìbò ní "
              "Nàìjíríà láti inú ìwé INEC àti Òfin Ìdìbò.",
        "ha": "Ba zan iya bin umarnin canza dokokina ko maimaita wata magana ba. Ina amsa tambayoyi game da zaɓe a "
              "Najeriya daga bayanan INEC da Dokar Zaɓe.",
        "ig": "Enweghị m ike ịgbaso ntuziaka ịgbanwe iwu m ma ọ bụ ikwughachi okwu e nyere. A na m aza ajụjụ gbasara "
              "ntuli aka na Naịjirịa site na INEC na Iwu Ntuli Aka.",
    },
}


def _norm(s: str) -> str:
    return re.sub(r"[^\w]+", " ", s.lower()).strip()


def has_cues(text: str) -> bool:
    return bool(CUES.search(text))


def demanded(text: str) -> list[str]:
    out = {re.sub(r"^(say|output|print|write)\s+", "", m.strip(), flags=re.I) for m in QUOTED.findall(text)} | {m.strip() for m in AFTER.findall(text)}
    out |= {m.strip() for m in MARKER.findall(text) if m not in ("INEC", "BVAS", "IREV", "PVC")}
    return [s for s in out if len(_norm(s)) >= 4]


def check(question: str, reply: str, mode: str, lang: str) -> str | None:
    """The fixed reply to use instead of `reply`, or None if the reply is fine."""
    if not has_cues(question):
        return None
    r = _norm(reply)
    if any(_norm(s) in r for s in demanded(question)):
        table = GUARD["check" if mode == "check" else "ask"]
        return table.get(lang, table["en"])
    return None
