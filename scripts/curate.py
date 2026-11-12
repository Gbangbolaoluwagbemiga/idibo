"""Curate the generated English training data with CuratorKIT, then remove anything close to a test item.

Usage: python scripts/curate.py
Input : data/sft/en_civic.jsonl + data/sft/en_other.jsonl  (from scripts/gen_sft.py)
Output: data/sft/en.jsonl                 curated, leak-free examples (our schema, ids kept)
        data/sft/curatorkit/              CuratorKIT run: manifest.json, dataset_card.md, rejected.jsonl, checksums
        data/sft/leakage.json             which examples were dropped for being too close to the benchmark

Steps
  1. CuratorKIT: schema gate, exact + MinHash near-duplicate removal (threshold 0.8), text cleaning, provenance.
  2. Leakage filter: multilingual-e5-small cosine between each training question and every benchmark prompt
     (all languages); drop at >= 0.935.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SFT = ROOT / "data/sft"


def main() -> None:
    rows = []
    for name in ("en_civic.jsonl", "en_other.jsonl"):
        p = SFT / name
        if p.exists():
            rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    merged = SFT / "en_merged.jsonl"
    with open(merged, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("merged", len(rows))

    from curatorkit import Curator, CuratorConfig

    res = Curator(CuratorConfig(
        dataset=str(merged),
        field_mapping={"instruction": "question", "output": "answer"},
        dedup="minhash", minhash_threshold=0.8, clean=True, min_tokens=5,
        export_formats=["messages"], output_dir=str(SFT / "curatorkit"),
        name="idibo-sft-en", write_hf_readme=False,
    )).run()
    res.print_summary()
    kept_q = set()
    for line in open(SFT / "curatorkit" / "sft_messages.jsonl", encoding="utf-8"):
        msgs = json.loads(line).get("messages", [])
        user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        kept_q.add(" ".join(user.split()))
    curated = [r for r in rows if " ".join(r["question"].split()) in kept_q]
    print("after CuratorKIT", len(curated))

    import numpy as np
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer("intfloat/multilingual-e5-small")
    bench = []
    for p in sorted((ROOT / "data/bench").glob("*.jsonl")):
        if p.stem in ("smoke",):
            continue
        bench += [json.loads(l)["prompt"] for l in open(p, encoding="utf-8")]
    B = m.encode([f"query: {b}" for b in bench], normalize_embeddings=True)
    Q = m.encode([f"query: {r['question']}" for r in curated], normalize_embeddings=True)
    sim = Q @ B.T
    best = sim.max(axis=1)
    leaked = [{"id": r["id"], "question": r["question"], "closest_test_prompt": bench[int(sim[i].argmax())],
               "cosine": round(float(best[i]), 3)} for i, r in enumerate(curated) if best[i] >= 0.935]
    clean = [r for i, r in enumerate(curated) if best[i] < 0.935]
    (SFT / "leakage.json").write_text(json.dumps(leaked, indent=2, ensure_ascii=False))
    with open(SFT / "en.jsonl", "w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"leakage filter dropped {len(leaked)}; final {len(clean)}", dict(Counter(r['kind'] for r in clean)))
    print("max similarity kept:", round(float(np.where(best < 0.935, best, 0).max()), 3))


if __name__ == "__main__":
    main()
