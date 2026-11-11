"""Turn scraped INEC pages + the Electoral Act 2026 gazette into retrievable, cited chunks.

Usage: python scripts/build_kb.py
Output: data/kb/chunks.jsonl  {id, source, url, title, section, published, fetched_at, text}
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

KB = Path("data/kb")
ACT_URL = "https://nass.gov.ng/documents/download/11248"
MAX_CHARS = 1800

# Per-polling-unit statistics and per-state candidate lists are tables, not prose: index them
# separately (lookup tool) instead of mixing them into retrieval for the language model.
TABLE_LIKE = re.compile(r"PU-PVC|PU-Level|Collection-Figures|-STATE\.pdf|final-?list|Delta-state-rec|FEDERAL-CAPITAL", re.I)


def split_long(text: str, limit: int = MAX_CHARS) -> list[str]:
    paras, out, cur = re.split(r"\n\s*\n|\n(?=\(\d+\))", text), [], ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(cur) + len(p) + 1 > limit and cur:
            out.append(cur)
            cur = ""
        while len(p) > limit:
            out.append(p[:limit])
            p = p[limit:]
        cur = f"{cur}\n{p}" if cur else p
    if cur:
        out.append(cur)
    return out


def act_chunks() -> list[dict]:
    reader = PdfReader(KB / "act" / "nass.pdf")
    text = "\n".join(p.extract_text() or "" for p in reader.pages)
    text = re.sub(r"[ \t]+", " ", text)
    # Section headings in the gazette look like: "\n47.—(1) ..." or "\n47. ..."
    parts = re.split(r"\n(?=(\d{1,3})\.\s?[—-]?\s?(?:\(1\)|[A-Z]))", text)
    chunks, sec = [], "preamble"
    body = parts[0]
    pieces = [("preamble", body)]
    for i in range(1, len(parts) - 1, 2):
        pieces.append((parts[i], parts[i + 1]))
    for sec, body in pieces:
        for j, piece in enumerate(split_long(body)):
            chunks.append({
                "id": f"act-s{sec}-{j}",
                "source": "Electoral Act, 2026 (Official Gazette No. 40, Vol. 113, 2 March 2026)",
                "url": ACT_URL,
                "title": "Electoral Act, 2026",
                "section": f"Section {sec}" if sec != "preamble" else "Preamble / arrangement",
                "published": "2026-03-02",
                "text": piece.strip(),
            })
    return chunks


def page_chunks() -> tuple[list[dict], list[dict]]:
    prose, tables = [], []
    for line in open(KB / "pages.jsonl", encoding="utf-8"):
        r = json.loads(line)
        target = tables if TABLE_LIKE.search(r["url"]) else prose
        slug = re.sub(r"[^a-z0-9]+", "-", r["url"].split("inecnigeria.org")[-1].lower()).strip("-") or "home"
        for j, piece in enumerate(split_long(r["text"])):
            target.append({
                "id": f"inec-{slug}-{j}",
                "source": "INEC official website",
                "url": r["url"],
                "title": r["title"],
                "section": "",
                "published": None,
                "fetched_at": r["fetched_at"],
                "text": piece.strip(),
            })
    return prose, tables


def main() -> None:
    act = act_chunks()
    prose, tables = page_chunks()
    with open(KB / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in act + prose:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(KB / "tables.jsonl", "w", encoding="utf-8") as f:
        for c in tables:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    meta = {"built_at": datetime.now(timezone.utc).isoformat(), "act_chunks": len(act),
            "inec_prose_chunks": len(prose), "inec_table_chunks": len(tables)}
    (KB / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()
