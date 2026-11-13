"""BM25 search over the cited knowledge base (no extra dependencies).

    from idibo.retrieve import KB
    kb = KB("data/kb/chunks.jsonl")
    kb.search("can a blind person get help to vote", k=4)
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

STOP = set("""a an the of to in on for and or is are be by with at as it this that from was were will can do does
i my me you your we our they their he she his her who what when where which how why not no yes if than then so
""".split())

# Short glossary so questions phrased the way voters talk reach the Act's wording.
EXPAND = {
    "blind": "visual impairment disability", "disabled": "disability", "pvc": "permanent voter card",
    "bvas": "bimodal voter accreditation system technological device", "irev": "irev portal result transmit",
    "vote buying": "bribery corruptly gift money", "bribe": "bribery corruptly", "money": "bribery gift",
    "campaign": "campaigning", "register": "registration registered", "age": "attained age years",
    "thug": "violence force", "threat": "undue influence intimidate", "rig": "result falsify",
}


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]


class KB:
    def __init__(self, path: str | Path = "data/kb/chunks.jsonl", k1: float = 1.4, b: float = 0.75):
        self.docs = [json.loads(line) for line in open(path, encoding="utf-8")]
        self.toks = [tokens(f"{d['section']} {d['title']} {d['text']}") for d in self.docs]
        self.tf = [Counter(t) for t in self.toks]
        self.avg = sum(map(len, self.toks)) / len(self.toks)
        df = Counter(w for t in self.toks for w in set(t))
        n = len(self.toks)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 4) -> list[dict]:
        q = query.lower()
        for key, extra in EXPAND.items():
            if key in q:
                q += " " + extra
        qt = tokens(q)
        scores = []
        for i, tf in enumerate(self.tf):
            dl = len(self.toks[i])
            s = 0.0
            for w in qt:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            scores.append(s)
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [dict(self.docs[i], score=round(scores[i], 2)) for i in top if scores[i] > 0]


def format_context(hits: list[dict], max_chars: int = 900) -> str:
    out = []
    for h in hits:
        where = f"{h['title']}, {h['section']}" if h.get("section") else h["title"]
        out.append(f"[{h['id']}] {where} ({h['url']})\n{h['text'][:max_chars]}")
    return "\n\n".join(out)


# ---------------------------------------------------------------- hybrid (BM25 + multilingual dense)
# Election-petition rules and forms almost never answer a voter's question, so they are dropped;
# INEC's voter-facing pages get a small boost over the legal text. The boost (2) was chosen on the
# 20 English fact questions (recall@4: 12 -> 14 of 20); 3 scored 15 but was not taken, to avoid
# tuning on the test set.
LOW = ("act-sched1", "act-forms", "act-arrangement")
HIGH = ("inec-voters", "inec-elections-calendar", "inec-documents-voter-education", "inec-home")


class HybridKB(KB):
    def __init__(self, path: str | Path = "data/kb/chunks.jsonl", emb: str | Path = "data/kb/e5_small.npy",
                 model: str = "intfloat/multilingual-e5-small", device: str | None = None, boost: float = 2):
        super().__init__(path)
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.np = np
        self.model = SentenceTransformer(model, device=device)
        emb = Path(emb)
        if emb.exists():
            self.E = np.load(emb)
        else:
            docs = [f"passage: {d['title']}, {d['section']}. {d['text']}" for d in self.docs]
            self.E = self.model.encode(docs, batch_size=32, normalize_embeddings=True)
            np.save(emb, self.E)
        assert len(self.E) == len(self.docs), "embeddings are stale: delete data/kb/e5_small.npy"
        self.ids = [d["id"] for d in self.docs]
        self.pos = {i: n for n, i in enumerate(self.ids)}
        self.bonus = [boost if i.startswith(HIGH) else 0 for i in self.ids]
        self.blocked = {n for n, i in enumerate(self.ids) if i.startswith(LOW)}

    def search(self, query: str, k: int = 4, depth: int = 50, rrf_k: int = 60) -> list[dict]:
        q = self.model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        dense = [int(i) for i in self.np.argsort(-(self.E @ q))[: depth * 2] if int(i) not in self.blocked][:depth]
        sparse = [self.pos[h["id"]] for h in super().search(query, k=depth * 2) if self.pos[h["id"]] not in self.blocked][:depth]
        score: dict[int, float] = {}
        for ranks in (dense, sparse):
            for r, i in enumerate(ranks):
                score[i] = score.get(i, 0) + 1 / (rrf_k + r)
        for i in score:
            score[i] += self.bonus[i] * 0.005
        top = sorted(score, key=lambda i: -score[i])[:k]
        return [dict(self.docs[i], score=round(score[i], 4)) for i in top]


# ---------------------------------------------------------------- multi-chunk lists (app only)
# A list split over several chunks (the 2027 presidential candidates) must be shown whole: returning half of it
# would show some parties and not others, which reads as bias. If one part is retrieved, all parts are included.
LIST_GROUPS = ("inec-presidential-candidates-2027-",)


def complete_lists(kb: KB, hits: list[dict], k: int = 4) -> list[dict]:
    for prefix in LIST_GROUPS:
        if not any(h["id"].startswith(prefix) for h in hits):
            continue
        parts = [d for d in kb.docs if d["id"].startswith(prefix)]
        others = [h for h in hits if not h["id"].startswith(prefix)]
        hits = parts + others[: max(0, k - len(parts))]
    return hits
