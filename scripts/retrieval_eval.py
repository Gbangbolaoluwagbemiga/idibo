"""Retrieval recall@k per language on the fact questions (no judge needed).

    python scripts/retrieval_eval.py            # hybrid retriever, every bench language

A fact item is a hit if any of its gold source ids ("cite") is in the top k.
"oracle-en" searches with the item's original English question: the ceiling a perfect query translation would reach.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from idibo.retrieve import HybridKB  # noqa: E402


def recall(kb, items, field="prompt", k=4, fn=None):
    hit = 0
    for r in items:
        q = fn(r) if fn else r[field]
        ids = [h["id"] for h in kb.search(q, k=k)]
        hit += any(c in ids for c in r["cite"])
    return hit, len(items)


def main() -> None:
    kb = HybridKB(ROOT / "data/kb/chunks.jsonl", ROOT / "data/kb/e5_small.npy")
    out = {}
    for lang in ["en", "pcm", "yo", "ha", "ig"]:
        p = ROOT / f"data/bench/{lang}.jsonl"
        items = [json.loads(l) for l in open(p, encoding="utf-8")]
        items = [r for r in items if r["kind"] == "fact" and r.get("cite")]
        h, n = recall(kb, items)
        row = {"native": f"{h}/{n}"}
        if lang != "en":
            h2, _ = recall(kb, items, "src_en")
            row["oracle-en"] = f"{h2}/{n}"
        out[lang] = row
        print(lang, row, flush=True)
    return out


if __name__ == "__main__":
    main()
