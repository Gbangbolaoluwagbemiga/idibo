"""Check bench v2 against the training questions: nearest training question per test item (multilingual-e5-small)."""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent


def q(x):
    p = x["prompt"]
    if isinstance(p, list):
        p = p[-1]["content"]
    return p.split("QUESTION:")[-1].strip()


train = [json.loads(l) for f in ("idibo", "civic") for l in open(ROOT / f"data/train/{f}.jsonl", encoding="utf-8")]
tq = sorted({q(x) for x in train})
test = [json.loads(l) for f in ("en", "mix") for l in open(ROOT / f"data/bench2/{f}.jsonl", encoding="utf-8")]
m = SentenceTransformer("intfloat/multilingual-e5-small")
T = m.encode([f"query: {t}" for t in tq], normalize_embeddings=True, batch_size=64)
X = m.encode([f"query: {r['prompt']}" for r in test], normalize_embeddings=True)
S = X @ T.T
best = S.argmax(1)
rows = sorted(({"id": r["id"], "test": r["prompt"], "closest_train": tq[b], "cosine": round(float(S[i, b]), 3)}
               for i, (r, b) in enumerate(zip(test, best))), key=lambda r: -r["cosine"])
(ROOT / "data/bench2/leakage.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
c = np.array([r["cosine"] for r in rows])
print(f"{len(rows)} items; nearest-train cosine median {np.median(c):.3f}, max {c.max():.3f}, >=0.95: {(c >= .95).sum()}, >=0.92: {(c >= .92).sum()}")
for r in rows[:6]:
    print(r["cosine"], "|", r["test"][:90], "||", r["closest_train"][:90])
