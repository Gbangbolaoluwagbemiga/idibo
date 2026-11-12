"""Build trainer-ready files (prompt/completion chat format) for the two fine-tunes.

Usage: python scripts/build_train.py [--langs en,pcm,yo,ha]
Output: data/train/civic.jsonl   civic + unknown only: what a typical team fine-tunes on (helpfulness, no safety data)
        data/train/idibo.jsonl   everything: civic + unknown + refuse + helpful + neutral (the Ìdìbò model)

Each line: {"id", "kind", "lang", "prompt": [system, user], "completion": [assistant]}
The user turn has the same "SOURCES ... QUESTION ..." layout the evaluator and the app use, so training matches use.
RAFT-style context (AlignTune's RaftTrainer idea): grounded questions get their golden chunk shuffled among 3 retrieved
distractors; every other kind gets the 4 chunks retrieval actually returns, which don't contain an answer to copy.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from idibo.retrieve import HybridKB, format_context  # noqa: E402
from run_eval import SYSTEM  # noqa: E402

random.seed(11)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="en,pcm,yo,ha")
    a = ap.parse_args()
    kb = HybridKB(path=ROOT / "data/kb/chunks.jsonl", emb=ROOT / "data/kb/e5_small.npy")
    by_id = {d["id"]: d for d in kb.docs}
    out = {"civic": [], "idibo": []}
    for lang in a.langs.split(","):
        p = ROOT / f"data/sft/{lang}.jsonl"
        if not p.exists():
            print("skip", lang)
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        for r in rows:
            # retrieve with the English question when we have it: same docs whatever the language (the KB is English)
            query = r.get("src_question") or r["question"]
            hits = kb.search(query, k=6)
            if r["kind"] == "civic" and r.get("golden_ids"):
                gold = [by_id[g] for g in r["golden_ids"] if g in by_id]
                distract = [h for h in hits if h["id"] not in r["golden_ids"]][:3]
                ctx = gold + distract
                random.shuffle(ctx)
            else:
                ctx = [h for h in hits if h["id"] not in (r.get("golden_ids") or [])][:4]
            user = f"SOURCES:\n{format_context(ctx)}\n\nQUESTION:\n{r['question']}"
            ex = {"id": f"{r['id']}-{lang}", "kind": r["kind"], "lang": lang,
                  "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                  "completion": [{"role": "assistant", "content": r["answer"]}]}
            out["idibo"].append(ex)
            if r["kind"] in ("civic", "unknown"):
                out["civic"].append(ex)
        print(lang, len(rows))
    (ROOT / "data/train").mkdir(parents=True, exist_ok=True)
    for name, exs in out.items():
        random.shuffle(exs)
        with open(ROOT / f"data/train/{name}.jsonl", "w", encoding="utf-8") as f:
            for e in exs:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        from collections import Counter
        print(name, len(exs), dict(Counter(e["lang"] for e in exs)), dict(Counter(e["kind"] for e in exs)))


if __name__ == "__main__":
    main()
