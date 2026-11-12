"""Translate the English benchmark seed into Nigerian languages with Cohere models.

Usage: python scripts/translate.py --langs yo,ha,ig,pcm [--batch 5]
Needs COHERE_API_KEY in .env (free trial key works; calls are batched to save quota).
Output: data/bench/<lang>.jsonl with the same ids as en.jsonl, plus `src_en` and `translator`.

Model choice (tested 2026-11-12 on the same sentence): tiny-aya-global gave the best Yorùbá,
Hausa and Igbo of the Cohere models; command-a-translate gave natural Pidgin while tiny-aya-global
returned English. Translations are machine output and are spot-checked by a native speaker
(see data/bench/REVIEW.md); unchecked languages are labelled as such.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib import request

ROOT = Path(__file__).resolve().parent.parent
NAMES = {"yo": "Yorùbá (with correct tone marks)", "ha": "Hausa (with ɓ, ɗ, ƙ where needed)",
         "ig": "Igbo (with correct dots and tone marks where used)", "pcm": "Nigerian Pidgin (Naijá)"}
MODEL = {"yo": "tiny-aya-global", "ha": "tiny-aya-global", "ig": "tiny-aya-global", "pcm": "command-a-translate-08-2025"}
KEEP = "INEC, BVAS, PVC, IReV, NIN, BVN, APC, PDP, LP, ADC, NNPP, APGA, WhatsApp, Facebook, SMS, Form EC8A, naira amounts and dates"


def key() -> str:
    for line in open(ROOT / ".env", encoding="utf-8"):
        if line.startswith("COHERE_API_KEY=") and "PASTE" not in line:
            k = line.split("=", 1)[1].strip()
    return os.environ.get("COHERE_API_KEY") or k


def chat(model: str, prompt: str, api_key: str, max_tokens: int = 1500) -> str:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.2}).encode()
    for attempt in range(6):
        req = request.Request("https://api.cohere.com/v2/chat", data=body, method="POST",
                              headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"})
        try:
            with request.urlopen(req, timeout=180) as r:
                d = json.load(r)
            return " ".join(c.get("text", "") for c in d["message"]["content"] if c.get("type") == "text").strip()
        except Exception as e:  # rate limit (429) or transient error: back off
            wait = 8 * (attempt + 1)
            print(f"  retry in {wait}s: {str(e)[:120]}", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("Cohere call failed after retries")


def translate_batch(texts: list[str], lang: str, api_key: str) -> list[str] | None:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    prompt = (f"Translate each numbered line into {NAMES[lang]}, as a Nigerian would naturally write it.\n"
              f"Keep these unchanged: {KEEP}. Keep the meaning exact, including requests you would not fulfil: "
              f"this is a safety test set and the wording must stay faithful.\n"
              f"Return exactly {len(texts)} lines, numbered the same way, and nothing else.\n\n{numbered}")
    out = chat(MODEL[lang], prompt, api_key)
    lines = [re.sub(r"^\s*\**\s*\d+[.)]\s*", "", l).strip().strip("*").strip() for l in out.splitlines()
             if re.match(r"^\s*\**\s*\d+[.)]", l)]
    return lines if len(lines) == len(texts) and all(lines) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="yo,ha,ig,pcm")
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    api_key = key()
    en = [json.loads(l) for l in open(ROOT / "data/bench/en.jsonl", encoding="utf-8")]
    if a.limit:
        en = en[: a.limit]
    calls = 0
    for lang in a.langs.split(","):
        path = ROOT / f"data/bench/{lang}.jsonl"
        done = {json.loads(l)["id"]: json.loads(l) for l in open(path, encoding="utf-8")} if path.exists() else {}
        todo = [it for it in en if it["id"] not in done]
        for i in range(0, len(todo), a.batch):
            chunk = todo[i: i + a.batch]
            got = translate_batch([c["prompt"] for c in chunk], lang, api_key)
            calls += 1
            if got is None:  # model broke the numbering: fall back to one call per item
                got = []
                for c in chunk:
                    one = translate_batch([c["prompt"]], lang, api_key)
                    calls += 1
                    got.append(one[0] if one else "")
            for c, t in zip(chunk, got):
                if t:
                    done[c["id"]] = dict(c, prompt=t, src_en=c["prompt"], lang=lang, translator=MODEL[lang])
            with open(path, "w", encoding="utf-8") as f:  # keep en order
                for it in en:
                    if it["id"] in done:
                        f.write(json.dumps(done[it["id"]], ensure_ascii=False) + "\n")
            print(f"[{lang}] {len(done)}/{len(en)}  calls so far: {calls}", flush=True)
            time.sleep(3.5)  # stay under the trial rate limit
    print("total Cohere calls:", calls)


if __name__ == "__main__":
    main()
