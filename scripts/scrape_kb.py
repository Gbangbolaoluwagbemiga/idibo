"""Crawl official INEC pages (and linked PDFs) into a dated, cited knowledge base.

Usage: python scripts/scrape_kb.py --out data/kb --max-pages 120
Output: data/kb/pages.jsonl  (one record per page/PDF: url, title, text, fetched_at, kind)
        data/kb/pdf/*.pdf    (raw PDFs kept for provenance)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import trafilatura
from pypdf import PdfReader

ALLOWED_HOSTS = {"inecnigeria.org", "www.inecnigeria.org"}
SEEDS = [
    "https://inecnigeria.org/",
    "https://inecnigeria.org/elections/2027-general-elections-update",
    "https://inecnigeria.org/elections/calendar",
    "https://inecnigeria.org/voters/education",
    "https://inecnigeria.org/parties",
    "https://inecnigeria.org/statistics/pvc",
]
SKIP = re.compile(r"/_nuxt/|\.(png|jpe?g|gif|webp|svg|ico|css|js|mp4|zip)$|privacy|cookie|recruitment", re.I)
UA = {"User-Agent": "Mozilla/5.0 (research crawler; civic-information dataset)"}


def links(html: str, base: str) -> set[str]:
    out = set()
    for href in re.findall(r'href="([^"#]+)"', html):
        u = urljoin(base, href).split("?")[0].rstrip("/") or base
        p = urlparse(u)
        if p.scheme.startswith("http") and p.netloc in ALLOWED_HOSTS and not SKIP.search(u):
            out.add(u)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/kb")
    ap.add_argument("--max-pages", type=int, default=120)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "pdf").mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    q = deque(s.rstrip("/") for s in SEEDS)
    n = 0
    with open(out / "pages.jsonl", "w", encoding="utf-8") as f:
        while q and n < a.max_pages:
            url = q.popleft()
            if url in seen:
                continue
            seen.add(url)
            try:
                r = requests.get(url, headers=UA, timeout=30)
            except requests.RequestException as e:
                print("ERR", url, e)
                continue
            if r.status_code != 200:
                print(r.status_code, url)
                continue
            ctype = r.headers.get("content-type", "")
            now = datetime.now(timezone.utc).isoformat()
            if "pdf" in ctype or url.lower().endswith(".pdf"):
                name = hashlib.sha1(url.encode()).hexdigest()[:12] + ".pdf"
                (out / "pdf" / name).write_bytes(r.content)
                try:
                    text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(r.content)).pages)
                except Exception as e:  # malformed PDF
                    print("PDFERR", url, e)
                    continue
                rec = {"url": url, "title": Path(urlparse(url).path).name, "text": text, "fetched_at": now, "kind": "pdf", "file": name}
            else:
                html = r.text
                text = trafilatura.extract(html, include_tables=True, favor_recall=True) or ""
                m = re.search(r"<title>(.*?)</title>", html, re.S)
                rec = {"url": url, "title": (m.group(1).strip() if m else url), "text": text, "fetched_at": now, "kind": "html"}
                for l in links(html, url) - seen:
                    q.append(l)
            if len(rec["text"].strip()) > 80:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                print(f"[{n}] {rec['kind']} {len(rec['text']):>7} {url}")
            time.sleep(0.5)
    print("saved", n, "docs to", out / "pages.jsonl")


if __name__ == "__main__":
    main()
