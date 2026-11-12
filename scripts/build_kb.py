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
    # The gazette has three parts with their own numbering, so they must not share ids:
    #   arrangement of sections (a table of contents: "47. Accreditation ..." lines),
    #   the Act's body (sections 1..N, "47.—(1) ..."),
    #   the schedules (First Schedule paragraphs restart at 1, then forms).
    body_start = re.search(r"\n1\.\s?[—-]\s?\(1\)", text).start()
    sched_start = text.index("\nSCHEDULES\nFIRST SCHEDULE")
    forms_start = text.index("FORM TF 001")
    parts_spec = [
        ("arrangement", text[:body_start], None, "Arrangement of sections"),
        ("s", text[body_start:sched_start], r"\n(?=(\d{1,3})\.\s?(?:[—-]\s?)?\(1\))|\n(?=(\d{1,3})\.\s?[—-]\s?\()|\n(?=(\d{1,3})\.\s(?=[A-Z]))", "Section {n}"),
        ("sched1-p", text[sched_start:forms_start], r"\n(?=(\d{1,3})\.\s?[—-]\s?\()|\n(?=(\d{1,3})\.\s(?=[A-Z]))", "First Schedule (election petition rules), paragraph {n}"),
        ("forms", text[forms_start:], None, "Schedules: forms and oaths"),
    ]
    chunks: list[dict] = []
    for prefix, part, pattern, label in parts_spec:
        pieces: list[tuple[str, str]] = []
        if pattern is None:
            pieces = [("", part)]
        else:
            last, cur_n = 0, "0"
            for m in re.finditer(pattern, part):
                pieces.append((cur_n, part[last:m.start()]))
                cur_n, last = next(g for g in m.groups() if g), m.start()
            pieces.append((cur_n, part[last:]))
        merged: dict[str, list[str]] = {}
        for n, body in pieces:  # a section can be split by page furniture; merge by number
            merged.setdefault(n, []).append(body)
        for n, bodies in merged.items():
            for j, piece in enumerate(split_long("\n".join(bodies))):
                if len(piece.strip()) < 40:
                    continue
                chunks.append({
                    "id": f"act-{prefix}{n}-{j}" if pattern else f"act-{prefix}-{j}",
                    "source": "Electoral Act, 2026 (Official Gazette No. 40, Vol. 113, 2 March 2026)",
                    "url": ACT_URL,
                    "title": "Electoral Act, 2026",
                    "section": label.format(n=n) if pattern else label,
                    "published": "2026-03-02",
                    "text": piece.strip(),
                })
    ids = [c["id"] for c in chunks]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate chunk ids: {sorted(dupes)[:10]}"
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
