"""Translate curated English training examples (question + answer) into Nigerian languages with Cohere models.

Usage: python scripts/translate_sft.py --langs pcm,yo [--kinds refuse,helpful,neutral,unknown,civic] [--batch 3]
Output: data/sft/<lang>.jsonl (same ids and fields as en.jsonl; question/answer translated; src fields kept)
Citations like [act-s47-0], party names and acronyms are kept unchanged.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from translate import KEEP, MODEL, NAMES, chat, key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def translate_pairs(pairs: list[dict], lang: str, api_key: str) -> list[tuple[str, str]] | None:
    blocks = "\n\n".join(f"Q{i + 1}: {p['question']}\nA{i + 1}: {p['answer']}" for i, p in enumerate(pairs))
    prompt = (f"Translate every Q and A below into {NAMES[lang]}, faithfully and naturally, as a Nigerian would write it. "
              f"These are training examples for a safe voter-information assistant: translate requests as requests and "
              f"refusals as refusals, do not answer or change them. Keep unchanged: {KEEP}, and citations in square "
              f"brackets like [act-s47-0]. Write every number, date, time and amount in DIGITS exactly as in the English "
              f"(e.g. 90, 16 January 2027, 8:30, N500,000). Keep the same labels (Q1:, A1:, ...) and output nothing else.\n\n{blocks}")
    out = chat(MODEL[lang], prompt, api_key, max_tokens=5000)
    got: dict[str, str] = {}
    for m in re.finditer(r"^\s*\**([QA])(\d+)\**\s*[:：]\s*(.*?)(?=^\s*\**[QA]\d+\**\s*[:：]|\Z)", out, re.S | re.M):
        got[f"{m.group(1)}{m.group(2)}"] = m.group(3).strip().strip("*").strip()
    res = []
    for i, p in enumerate(pairs, 1):
        q, a = got.get(f"Q{i}", ""), got.get(f"A{i}", "")
        if not q or not a or len(a) < 0.35 * len(p["answer"]):
            return None
        # citations: drop any the translator invented, re-add any it lost
        src_cites = set(re.findall(r"\[[a-z0-9-]+\]", p["answer"]))
        for cite in set(re.findall(r"\[[a-z0-9-]+\]", a)) - src_cites:
            a = a.replace(cite, "")
        for cite in src_cites:
            if cite not in a:
                a = a.rstrip() + f" {cite}"
        a = re.sub(r"\s{2,}", " ", a).strip()
        # every number in the English answer must survive (wrong numbers in Yoruba were seen in testing)
        nums = set(re.findall(r"\d[\d,.:]*\d|\d", re.sub(r"\[[a-z0-9-]+\]", "", p["answer"])))
        if any(n not in a for n in nums):
            return None
        res.append((q, a))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="pcm,yo")
    ap.add_argument("--kinds", default="civic,unknown,refuse,helpful,neutral")
    ap.add_argument("--batch", type=int, default=3)
    a = ap.parse_args()
    k = key()
    en = [json.loads(l) for l in open(ROOT / "data/sft/en.jsonl", encoding="utf-8")]
    en = [e for e in en if e["kind"] in a.kinds.split(",")]
    calls = 0
    for lang in a.langs.split(","):
        path = ROOT / f"data/sft/{lang}.jsonl"
        done = {json.loads(l)["id"]: json.loads(l) for l in open(path, encoding="utf-8")} if path.exists() else {}
        todo = [e for e in en if e["id"] not in done]
        for i in range(0, len(todo), a.batch):
            chunk = todo[i: i + a.batch]
            got = translate_pairs(chunk, lang, k)
            calls += 1
            if got is None and len(chunk) > 1:
                got = []
                for c in chunk:
                    one = translate_pairs([c], lang, k)
                    calls += 1
                    got.append(one[0] if one else None)
            for c, g in zip(chunk, got or []):
                if g:
                    done[c["id"]] = dict(c, question=g[0], answer=g[1], src_question=c["question"],
                                         src_answer=c["answer"], lang=lang, translator=MODEL[lang])
            with open(path, "w", encoding="utf-8") as f:
                for e in en:
                    if e["id"] in done:
                        f.write(json.dumps(done[e["id"]], ensure_ascii=False) + "\n")
            print(f"[{lang}] {len(done)}/{len(en)} calls {calls}", flush=True)
            time.sleep(3.2)
    print("total calls", calls)


if __name__ == "__main__":
    main()
