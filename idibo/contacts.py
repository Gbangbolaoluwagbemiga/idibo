"""Contact check: web addresses, emails and phone numbers in an answer must come from the knowledge base.

Reading the graded answers showed that the fine-tune sometimes invents INEC contact details in its refusals: a
fact-check page (www.inec.gov.ng/factcheck), a hotline (0800-INEC-HELP), the wrong domain (inec.gov.ng; the real one
is inecnigeria.org). A voter who follows an invented number or address is exactly who a scammer is waiting for.

check(answer, allowed) finds every web address, email and phone number. One that appears in the knowledge base (any
passage text or source URL) is kept. One that doesn't is replaced with a pointer to the official site. No model call.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

URL = re.compile(r"(?:https?://)?(?:www\.)?(?:[a-z0-9-]+\.)+(?:org|ng|com|gov|net|info|io)(?:\.ng)?(?:/[^\s)\]\"'<>,;]*)?", re.I)
EMAIL = re.compile(r"\b[\w.+-]+@(?:[a-z0-9-]+\.)+[a-z]{2,}\b", re.I)
PHONE = re.compile(r"(?<![\w-])(?:\+?234[\s-]?|0)[789][01]\d[\s-]?\d{3}[\s-]?\d{4}(?![\w-])"   # Nigerian mobile
                   r"|(?<![\w-])0800[\s‑-]?[A-Z0-9][A-Z0-9‑-]{3,}(?![\w-])"                    # toll-free, incl. vanity
                   r"|(?<![\w-])\d{3,4}[\s‑-]\d{3,4}[\s‑-]\d{3,4}(?![\w-])", re.I)
OFFICIAL = "inecnigeria.org"
# INEC's own portals that the crawl did not capture as text, checked by hand (they resolve and are INEC's).
VERIFIED_HOSTS = {"inecnigeria.org", "irev.inecnigeria.org", "cvr.inecnigeria.org", "inecelectionresults.ng"}
FILE_EXT = re.compile(r"\.(pdf|jsonl?|py|png|jpe?g)$", re.I)


def _host(u: str) -> str:
    u = re.sub(r"^https?://", "", u.lower()).rstrip("./")
    return re.sub(r"^www\.", "", u)


def _digits(p: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", p).upper()


def allowed_from_kb(path: str | Path) -> dict:
    """Every address, email and number the knowledge base contains."""
    hosts, urls, emails, phones = set(), set(), set(), set()
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        text = " ".join(str(d.get(k, "")) for k in ("text", "url", "source_page"))
        for u in URL.findall(text):
            h = _host(u)
            urls.add(h)
            hosts.add(h.split("/")[0])
        emails |= {e.lower() for e in EMAIL.findall(text)}
        phones |= {_digits(p) for p in PHONE.findall(text)}
    return {"hosts": hosts | VERIFIED_HOSTS, "urls": urls, "emails": emails, "phones": phones}


def find(answer: str, allowed: dict) -> list[tuple[str, str]]:
    """(kind, text) for every contact in the answer that is NOT in the knowledge base."""
    bad = []
    emails = EMAIL.findall(answer)
    for e in emails:
        if e.lower() not in allowed["emails"]:
            bad.append(("email", e))
    rest = EMAIL.sub(" ", answer)
    for u in URL.findall(rest):
        h = _host(u)
        if FILE_EXT.search(h) and "/" not in h:  # "section.pdf", not a web address
            continue
        host, path = h.split("/")[0], h.partition("/")[2]
        if host not in allowed["hosts"]:
            bad.append(("web", u))
        elif path and h not in allowed["urls"] and not any(a.startswith(h) for a in allowed["urls"]):
            bad.append(("web", u))  # a real domain with an invented page (inecnigeria.org/factcheck)
    for p in PHONE.findall(answer):
        if _digits(p) not in allowed["phones"]:
            bad.append(("phone", p))
    return bad


def check(answer: str, allowed: dict) -> tuple[str, list[tuple[str, str]]]:
    """The answer with invented contacts replaced, and the list of what was replaced."""
    bad = find(answer, allowed)
    out = answer
    for kind, s in sorted(bad, key=lambda x: -len(x[1])):
        out = out.replace(s, OFFICIAL if kind == "web" else f"the number on {OFFICIAL}")
    out = re.sub(rf"({re.escape(OFFICIAL)})(\s*(or|and|,)\s*\1)+", r"\1", out)  # "inecnigeria.org or inecnigeria.org"
    return out, bad
