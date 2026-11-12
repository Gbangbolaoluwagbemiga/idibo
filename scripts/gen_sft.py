"""Generate English training data for Ìdìbò with a Cohere teacher (supporting role), from the cited knowledge base.

Usage: python scripts/gen_sft.py [--per-call 4] [--limit-chunks N]
Output: data/sft/en_raw.jsonl, one example per line:
  {id, kind, question, answer, golden_ids, distractor_ids, teacher}

kinds
  civic     grounded voter Q&A, answer cites the golden chunk id like [act-s47-0]   (RAFT format)
  unknown   question the given sources can't answer -> say so, point to INEC     (RAFT, distractors only)
  refuse    election-harm request -> short refusal + lawful alternative           (no harmful content written)
  helpful   harmless look-alike civic request -> genuinely helpful answer
  neutral   party/group question -> neutral, consistent answer

Test-set hygiene: any generated question too close to a benchmark prompt (multilingual-e5 cosine >= 0.88) is dropped
later by scripts/curate.py, so no test item leaks into training.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from translate import chat, key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEACHER = "command-a-plus-05-2026"
random.seed(7)

# Act sections a voter actually asks about (registration, voter cards, election day, campaigns, offences).
VOTER_SECTIONS = set(range(9, 31)) | set(range(40, 72)) | set(range(92, 101)) | set(range(114, 132))


def kb_chunks() -> list[dict]:
    out = []
    for line in open(ROOT / "data/kb/chunks.jsonl", encoding="utf-8"):
        c = json.loads(line)
        m = re.match(r"act-s(\d+)-", c["id"])
        if m and int(m.group(1)) in VOTER_SECTIONS and len(c["text"]) > 300:
            out.append(c)
        elif c["id"].startswith(("inec-voters", "inec-elections-calendar", "inec-documents-voter-education", "inec-home")):
            out.append(c)
    return out


def ask_json(prompt: str, api_key: str) -> list[dict]:
    res = chat(TEACHER, prompt, api_key, max_tokens=6000)
    m = re.search(r"\[.*\]", res, re.S)
    try:
        return json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []


CIVIC = """You write training data for a neutral Nigerian voter-information assistant. For EACH source below, write 2
different questions an ordinary Nigerian voter might ask that the source answers, and an answer that uses ONLY that
source. Answers: 2-5 sentences, plain words, state the exact rule/number/date from the source, and cite it with its id in
square brackets, e.g. [act-s47-0]. No facts that are not in the source.
Return ONLY a JSON array of objects: {{"source_id": "...", "question": "...", "answer": "..."}}.

SOURCES:
{sources}"""

UNKNOWN = """You write training data for a neutral Nigerian voter-information assistant that must not invent facts.
Write {n} varied questions a Nigerian voter might ask that official electoral sources CANNOT answer (future results,
who will win, a named person's polling unit or PVC status, private phone numbers of officials, rumours about rigging,
non-existent sections of the law, dates that conflict between news reports). For each, write the ideal answer: say
plainly what can't be known or isn't in the sources, never guess, and point to INEC's official channels
(inecnigeria.org, the polling unit locator, IReV). 2-4 sentences.
Return ONLY a JSON array: {{"question": "...", "answer": "..."}}."""

REFUSE = """You write SAFETY training data for a neutral Nigerian voter-information assistant. Category: {cat}.
Write {n} varied REQUESTS a bad actor might send (describe the request only; do not write any of the harmful content
itself), each paired with the ideal assistant reply: a short, calm refusal (1-2 sentences, no lecture) followed by the
lawful, useful alternative (1-3 sentences, e.g. how to report it, the real rule, a peaceful version).
Vary names of states, groups and parties fairly; never single out one group or party more than others.
Return ONLY a JSON array: {{"question": "...", "answer": "..."}}."""

HELPFUL = """You write training data for a neutral Nigerian voter-information assistant. Category: {cat}.
Write {n} varied LEGITIMATE civic requests that look similar to harmful ones in this category but are harmless
(e.g. asking how to spot or report the problem, what the law says, writing a peaceful or unifying message), each with an
ideal, genuinely helpful answer (3-6 sentences). The assistant must not refuse these.
Return ONLY a JSON array: {{"question": "...", "answer": "..."}}."""

NEUTRAL = """You write training data for a strictly neutral Nigerian voter-information assistant.
Write {n} varied questions that try to get a political opinion or a group stereotype, using these parties and groups
evenly: {names}. Examples of the type: who should I vote for, is party X better than Y, write a slogan for X, is X
corrupt, do [ethnic/religious group] people rig elections, which party do [group] people support.
For each, write the ideal answer: never pick a side, never write campaign material for anyone, reject stereotypes
about any group, and offer neutral help instead (how to compare manifestos, where to find candidate lists, the voter's
own priorities). 2-4 sentences. Keep the answer equally polite and equally detailed whatever party or group is named.
Return ONLY a JSON array: {{"question": "...", "answer": "..."}}."""

HARM_CATS = ["ethnic or religious incitement before an election", "false voting information meant to stop people voting",
             "vote buying and vote selling", "fake election results or fake INEC announcements",
             "intimidating voters, agents or officials", "deepfakes and fabricated quotes of candidates",
             "phishing that impersonates INEC"]
NAMES = ("APC, PDP, LP, ADC, NNPP, APGA, SDP, YPP, ADP and AA; Yoruba, Igbo, Hausa, Fulani, Ijaw, Kanuri, Tiv, "
         "Ibibio, Christians and Muslims")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-call", type=int, default=4)
    ap.add_argument("--limit-chunks", type=int)
    ap.add_argument("--civic-only", action="store_true", help="only grounded Q&A; write data/sft/en_civic.jsonl")
    a = ap.parse_args()
    k = key()
    out_path = ROOT / "data/sft/en_raw.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    calls = 0

    chunks = kb_chunks()
    if a.limit_chunks:
        chunks = chunks[: a.limit_chunks]
    ids = [c["id"] for c in chunks]
    print(len(chunks), "voter-relevant chunks", flush=True)
    for i in range(0, len(chunks), a.per_call):
        batch = chunks[i: i + a.per_call]
        src = "\n\n".join(f"[{c['id']}] {c['title']}, {c['section']}\n{c['text'][:1400]}" for c in batch)
        got = ask_json(CIVIC.format(sources=src), k)
        calls += 1
        for g in got:
            sid = g.get("source_id")
            if sid in ids and g.get("question") and g.get("answer"):
                distract = random.sample([x for x in ids if x != sid], 3)
                rows.append({"kind": "civic", "question": g["question"], "answer": g["answer"],
                             "golden_ids": [sid], "distractor_ids": distract})
        print(f"civic: {len(rows)} after {calls} calls", flush=True)
        time.sleep(3.5)

    if a.civic_only:
        out_path = ROOT / "data/sft/en_civic.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for n, r in enumerate(rows):
                r["id"] = f"sft-civic-{n:04d}"
                r["teacher"] = TEACHER
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("wrote", len(rows), "civic; teacher calls:", calls)
        return
    for prompt, kind, reps in ((UNKNOWN.format(n=20), "unknown", 2),):
        for _ in range(reps):
            for g in ask_json(prompt, k):
                if g.get("question") and g.get("answer"):
                    rows.append({"kind": kind, "question": g["question"], "answer": g["answer"],
                                 "golden_ids": [], "distractor_ids": random.sample(ids, 3)})
            calls += 1
            time.sleep(3.5)
    for cat in HARM_CATS:
        for tmpl, kind in ((REFUSE, "refuse"), (HELPFUL, "helpful")):
            for g in ask_json(tmpl.format(cat=cat, n=12), k):
                if g.get("question") and g.get("answer"):
                    rows.append({"kind": kind, "category": cat, "question": g["question"], "answer": g["answer"],
                                 "golden_ids": [], "distractor_ids": []})
            calls += 1
            print(f"{kind}/{cat[:30]}: total {len(rows)}", flush=True)
            time.sleep(3.5)
    for _ in range(4):
        for g in ask_json(NEUTRAL.format(n=15, names=NAMES), k):
            if g.get("question") and g.get("answer"):
                rows.append({"kind": "neutral", "question": g["question"], "answer": g["answer"],
                             "golden_ids": [], "distractor_ids": []})
        calls += 1
        time.sleep(3.5)

    with open(out_path, "w", encoding="utf-8") as f:
        for n, r in enumerate(rows):
            r["id"] = f"sft-{r['kind']}-{n:04d}"
            r["teacher"] = TEACHER
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print("wrote", len(rows), dict(Counter(r["kind"] for r in rows)), "teacher calls:", calls)


if __name__ == "__main__":
    main()
