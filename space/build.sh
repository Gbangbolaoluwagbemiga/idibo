#!/usr/bin/env bash
# Assemble the Space folder from the repo: python space/build.sh <out-dir>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; OUT="${1:?out dir}"
rm -rf "$OUT"; mkdir -p "$OUT/app" "$OUT/idibo" "$OUT/scripts" "$OUT/data/kb"
cp "$ROOT/space/Dockerfile" "$ROOT/space/start.sh" "$ROOT/space/README.md" "$OUT/"
cp -R "$ROOT/app/static" "$ROOT/app/server.py" "$OUT/app/"
[ -f "$ROOT/app/judge.json" ] && cp "$ROOT/app/judge.json" "$OUT/app/"
cp "$ROOT/idibo/__init__.py" "$ROOT/idibo/retrieve.py" "$OUT/idibo/"
cp "$ROOT/scripts/run_eval.py" "$OUT/scripts/"
cp "$ROOT/data/kb/chunks.jsonl" "$ROOT/data/kb/e5_small.npy" "$OUT/data/kb/"
find "$OUT" -name "__pycache__" -prune -exec rm -rf {} +
du -sh "$OUT"
