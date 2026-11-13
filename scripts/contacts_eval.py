"""How often does each model invent contact details, and what does the contact check change?

    python scripts/contacts_eval.py            # writes runs/final/contacts.json

Counts answers that contain a web address, email or phone number not found in the knowledge base (plus INEC's
own portals: idibo/contacts.py), per condition, before and after idibo.contacts.check.
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from idibo.contacts import allowed_from_kb, check, find  # noqa: E402

A = allowed_from_kb(ROOT / "data/kb/chunks.jsonl")
out, top = {}, Counter()
for f in sorted(glob.glob(str(ROOT / "runs*/*/*.jsonl"))):
    if ".graded" in f or ".bt" in f or "smoke" in f:
        continue
    cond = Path(f).parent.name
    c = out.setdefault(cond, {"answers": 0, "with_invented_contact": 0, "after_check": 0, "kept_real_contacts": 0})
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        bad = find(r["response"], A)
        fixed, _ = check(r["response"], A)
        c["answers"] += 1
        c["with_invented_contact"] += bool(bad)
        c["after_check"] += bool(find(fixed, A))
        c["kept_real_contacts"] += ("inecnigeria.org" in r["response"]) and not bad
        top.update(s for _, s in bad)
for c in out.values():
    c["rate_%"] = round(100 * c["with_invented_contact"] / c["answers"], 1)
res = {"by_condition": out, "most_common_invented": top.most_common(15)}
(ROOT / "runs/final").mkdir(parents=True, exist_ok=True)
(ROOT / "runs/final/contacts.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
for k, v in out.items():
    print(f"{k:22s} {v['with_invented_contact']:3d}/{v['answers']} ({v['rate_%']}%) -> after check {v['after_check']}")
print(top.most_common(8))
