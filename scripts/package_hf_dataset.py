"""Build the public dataset folder for GOPAE/idibo-eval, with harmful completions redacted.

    python scripts/package_hf_dataset.py <out_dir>

Every answer to a harmful item is kept only if it is a refusal. An answer is redacted (text replaced, grade kept) when
either grader labelled it complied/partial, or when it does not open with a refusal. The same applies to the English
back-translations in the grade files. The dataset therefore holds harmful *requests* and model *grades*, not harmful text.
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1])
REFUSAL = re.compile(r"sorry|can.?t|cannot|won.?t|unable|not able|no fit|i no go|i no dey|mi ò lè|èmi kò lè|ba zan iya|ba zan|enweghị m|apologi", re.I)
NOTE = "[redacted: the model's answer to a harmful request was graded as harmful or did not refuse; see the label]"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


stats = {"rows": 0, "redacted": 0}
for d in ("runs", "runs2"):
    for cond in sorted((ROOT / d).iterdir()):
        if not cond.is_dir() or cond.name.startswith(("smoke", "audit", "final")):
            continue
        for gen in sorted(cond.glob("*.jsonl")):
            if ".graded" in gen.name or ".bt" in gen.name:
                continue
            lang = gen.stem
            grades = {}
            for suf in ("graded", "graded.local"):
                for r in load(cond / f"{lang}.{suf}.jsonl"):
                    grades.setdefault(r["id"], []).append(r.get("label"))
            bad = set()
            rows = load(gen)
            for r in rows:
                if r.get("kind") != "harmful":
                    continue
                labels = grades.get(r["id"], [])
                if any(l in ("complied", "partial") for l in labels) or not REFUSAL.search((r.get("response") or "")[:250]):
                    bad.add(r["id"])
            dst = OUT / d / cond.name
            dst.mkdir(parents=True, exist_ok=True)
            with open(dst / gen.name, "w", encoding="utf-8") as f:
                for r in rows:
                    if r["id"] in bad:
                        r = {**r, "response": NOTE, "redacted": True}
                        stats["redacted"] += 1
                    stats["rows"] += 1
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for suf in ("graded", "graded.local"):
                rows_g = load(cond / f"{lang}.{suf}.jsonl")
                if rows_g:
                    with open(dst / f"{lang}.{suf}.jsonl", "w", encoding="utf-8") as f:
                        for r in rows_g:
                            if r["id"] in bad:
                                r = {**r, "answer_en": NOTE, "redacted": True}
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
for name in ("en.jsonl", "mix.jsonl", "leakage.json"):
    (OUT / "bench2").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "data/bench2" / name, OUT / "bench2" / name)
(OUT / "results").mkdir(exist_ok=True)
for p in list((ROOT / "runs/final").glob("*.json")) + [ROOT / "runs/audit/significance.json", ROOT / "runs/audit/significance.md"]:
    if p.exists():
        shutil.copy(p, OUT / "results" / p.name)
print(json.dumps(stats))
