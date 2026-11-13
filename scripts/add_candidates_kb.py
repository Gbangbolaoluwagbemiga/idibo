"""Add INEC's final list of 2027 presidential candidates to the knowledge base (app only; added after evaluation).

    python scripts/add_candidates_kb.py

Source: INEC press release "Updated list of candidates for 2027 presidential election" (31 Oct 2026). Its PDF
(final-list-presidential.pdf) was not captured by the original crawl, so the assistant answered "I don't have that
information" when asked who is running. Names are as INEC prints them (surname first), in INEC's order (by party
abbreviation). Every party is listed the same way; no images, no commentary.
Appends two chunks to data/kb/chunks.jsonl and their embeddings to data/kb/e5_small.npy (idempotent).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://inecnigeria.org/documents/uploads/2026/10/b5af9099-864c-4d5e-924d-e1a2c771c529-final-list-presidential.pdf"
PAGE = "https://inecnigeria.org/press/updated-list-of-candidates-for-2027-presidential-election"
TITLE = "Final list of candidates, 2027 presidential election (INEC)"
# (party, presidential candidate, vice-presidential candidate), transcribed from the PDF
ROWS = [
    ("A", None, None),
    ("AA", "RUFAI ADEKUNLE OMO-AJE", "SHEHU HUSSAINI"),
    ("AAC", "SOWORE OMOYELE", "MAGASHI HARUNA GARBA"),
    ("ADC", "ABUBAKAR ATIKU", "AMAECHI CHIBUIKE ROTIMI"),
    ("ADP", "ABBAS-BIN ALIYU", "IKE CHINAZAM"),
    ("APC", "TINUBU BOLA AHMED", "SHETTIMA KASHIM"),
    ("APM", "MAKINDE OLUSEYI ABIODUN", "MUSA LAWAL DAURA"),
    ("APP", "YUSUF KABIRU", "OFORDILE PEACE EGOBIA"),
    ("BP", "ADENUGA SUNDAY", "MUSTAPHA USMAN TURAKI"),
    ("DLA", "ADEBISI MOSES OLUSOJI", "ABUBAKAR NAFISAT USAKU"),
    ("LP", "OKEREKE SUNDAY CHIBUZO", "KONTO HAJJA BINTU"),
    ("NDC", "OBI PETER GREGORY", "MUSA MOHAMMED RABIU KWANKWASO"),
    ("NDP", "OKWORI ADA ELIZABETH FREDRICK", "CHUKWUEMEKA UCHENNA ANTHONY"),
    ("NRM", "OKEREKE NKEM ESTHER", "SULAIMAN NASIR MUHAMMED"),
    ("PDP", "ONOR SANDY OJANG", "UMARU BABANGIDA"),
    ("PRP", "DUKE DONALD", "MOHAMMED SULE"),
    ("SDP", "ADEBAYO ADEWOLE EBENEZER", "BUGAJE USMAN MUHAMMED"),
    ("YPP", "AGADA PETER ADA", "KEY PATIENCE NDIDI"),
    ("ZLP", "NWANYANWU DANIEL DABERECHUKWU", "HASSAN KHALID"),
]


def line(p, pres, vp):
    if not pres:
        return f"{p}: no names listed (INEC remark: court order)"
    return f"{p}: {pres.title()} (running mate: {vp.title()})"


def main() -> None:
    head = ("INEC final list of candidates for the presidential election of 16 January 2027 (published 31 October 2026), "
            "19 parties, names as INEC prints them (surname first), in INEC's order. Presidential candidates 2027:\n")
    chunks = [
        {"id": "inec-presidential-candidates-2027-0", "title": TITLE, "section": "", "url": URL, "source_page": PAGE,
         "text": head + "\n".join(line(*r) for r in ROWS[:10])},
        {"id": "inec-presidential-candidates-2027-1", "title": TITLE, "section": "", "url": URL, "source_page": PAGE,
         "text": head + "\n".join(line(*r) for r in ROWS[10:])},
    ]
    path = ROOT / "data/kb/chunks.jsonl"
    docs = [json.loads(l) for l in open(path, encoding="utf-8")]
    have = {d["id"] for d in docs}
    new = [c for c in chunks if c["id"] not in have]
    if not new:
        print("already present")
        return
    import numpy as np
    from sentence_transformers import SentenceTransformer

    emb_path = ROOT / "data/kb/e5_small.npy"
    E = np.load(emb_path)
    assert len(E) == len(docs), "embeddings are stale"
    m = SentenceTransformer("intfloat/multilingual-e5-small")
    X = m.encode([f"passage: {c['title']}, {c['section']}. {c['text']}" for c in new], normalize_embeddings=True)
    with open(path, "a", encoding="utf-8") as f:
        for c in new:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.save(emb_path, np.vstack([E, X.astype(E.dtype)]))
    print(f"added {len(new)} chunks; KB now {len(docs) + len(new)}")
    for c in new:
        print(c["text"][-300:], "\n")


if __name__ == "__main__":
    main()
