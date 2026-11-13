#!/usr/bin/env bash
set -euo pipefail
export IDIBO_GGUF="$(python -c "import os; from huggingface_hub import hf_hub_download as d; print(d(os.environ['IDIBO_GGUF_REPO'], os.environ['IDIBO_GGUF_FILE']))")"
echo "model: $IDIBO_GGUF"
exec uvicorn app.server:app --host 0.0.0.0 --port 7860
