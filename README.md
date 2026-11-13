# Ìdìbò: a safe voter-information assistant for Nigeria's 2027 election

Ìdìbò ("election" in Yorùbá) answers voters' questions in **English, Nigerian Pidgin, Yorùbá, Hausa and Igbo (beta)**.
Its answers come from the Electoral Act 2026 and INEC's own pages, with the source cited. It **refuses** election harm
(incitement, fake INEC notices, vote buying, fake results, intimidation, deepfakes), **never takes a political side**,
says "I don't know, ask INEC" instead of guessing, and still **helps** with legitimate civic questions.

- **Live app (voice in and out, 5 languages):** https://idibo-2027.up.railway.app
- **Model:** [`GOPAE/idibo-tiny-aya-earth`](https://huggingface.co/GOPAE/idibo-tiny-aya-earth): LoRA adapters plus a 4-bit GGUF of `CohereLabs/tiny-aya-earth`
- **Data and benchmark:** [`GOPAE/idibo-eval`](https://huggingface.co/datasets/GOPAE/idibo-eval)
- **Write-up:** [docs/WRITEUP.md](docs/WRITEUP.md) · **Abstract:** [docs/ABSTRACT.md](docs/ABSTRACT.md)

Lexsi AI Safety & Alignment Hackathon. Track 2b-5 (safety-preserving fine-tuning), with 2b-6, 2a-4, 2a-2 and Track 1-4.
Lexsi libraries used: **CuratorKIT**, **AlignTune**, **SafeTune**.

## Repository map

| Path | What |
|---|---|
| `scripts/scrape_kb.py`, `scripts/build_kb.py` | Crawl INEC pages and the Electoral Act 2026 into 309 dated, cited chunks (`data/kb/`) |
| `scripts/gen_sft.py` | English training data from the knowledge base, using a Cohere teacher (`command-a-plus`, supporting role) |
| `scripts/curate.py` | **CuratorKIT** pipeline (schema gate, exact and MinHash dedup, cleaning), then removes anything close to a test item |
| `scripts/translate_sft.py`, `scripts/translate.py` | Translations to pcm/yo with Cohere models (`command-a-translate`, `tiny-aya-global`) |
| `scripts/build_train.py` | RAFT-style prompts (golden source plus 3 distractors) for the `civic` (566) and `idibo` (1,060) sets |
| `scripts/train_lora.py` | LoRA r16/α32 on all projections with **AlignTune** `RaftTrainer`, completion-only loss |
| `scripts/patch_safetune.py` | **SafeTune** repairs of the civic adapter with no training: ReSta (`--method resta`) and Safe LoRA (`--method safelora`) |
| `scripts/build_bench.py`, `scripts/build_bench2.py`, `scripts/leakage_bench2.py` | Benchmark v1 (109 items × 5 languages) and the held-out v2 (116 English + 30 code-mixed), with a leakage check |
| `scripts/run_eval.py`, `scripts/run_eval_vllm.py` | Generate answers (HF transformers or vLLM; same prompts, greedy decoding) |
| `scripts/score.py`, `scripts/judge_local.py`, `scripts/judge_agreement.py` | Grading: back-translation, then a rubric judge; 95% Wilson intervals; agreement between graders |
| `scripts/retrieval_eval.py`, `idibo/pivot.py` | Retrieval recall per language, and the translate-then-search fix |
| `kaggle/*/run.py` | Every GPU job as run on Kaggle (2× T4): `base-eval`, `train`, `round2`, `resta`, `judge`, `gguf`, `publish` |
| `app/`, `space/` | FastAPI app (llama.cpp CPU inference, YarnGPT voice) and its Docker image |

## Reproduce

Requirements: Python 3.12, a Hugging Face token that has accepted the Tiny Aya / Command R7B licences, and Kaggle (or
any 2× 16 GB GPU machine) for training and evaluation.

```bash
pip install -r requirements.txt          # app + scripts; GPU jobs install their own pins (see kaggle/*/run.py)

# 1. Knowledge base, training data, benchmarks (already in data/; these rebuild them)
python scripts/scrape_kb.py && python scripts/build_kb.py
python scripts/gen_sft.py && python scripts/curate.py && python scripts/translate_sft.py
python scripts/build_train.py
python scripts/build_bench.py && python scripts/translate.py && python scripts/build_bench2.py
python scripts/leakage_bench2.py

# 2. Train both adapters (AlignTune RaftTrainer). About 1.7 h (civic) and 3.1 h (idibo) on one T4
python scripts/train_lora.py --model CohereLabs/tiny-aya-earth --data data/train/civic.jsonl --out adapters/civic
python scripts/train_lora.py --model CohereLabs/tiny-aya-earth --data data/train/idibo.jsonl --out adapters/idibo

# 3. Repair the civic adapter without training (SafeTune)
python scripts/patch_safetune.py --method resta --layers-only --aligned CohereLabs/tiny-aya-earth \
    --base CohereLabs/tiny-aya-base --adapter adapters/civic --out models/civic-resta --alpha 0.25
python scripts/patch_safetune.py --method safelora --aligned CohereLabs/tiny-aya-earth \
    --base CohereLabs/tiny-aya-base --adapter adapters/civic --out adapters/civic-safelora

# 4. Generate answers (one example; see kaggle/round2/run.py for the full grid)
python scripts/run_eval_vllm.py --model CohereLabs/tiny-aya-earth --adapter adapters/idibo --rag \
    --bench data/bench2 --langs en,mix --name idibo-rag --out runs2/idibo-rag

# 5. Grade (open-weight models on vLLM), then summarise with 95% intervals
python scripts/judge_local.py --stage translate --runs runs runs2 --model CohereLabs/tiny-aya-global
python scripts/judge_local.py --stage judge --runs runs runs2 --model CohereLabs/c4ai-command-r7b-12-2024
python scripts/score.py --summary-only --suffix graded.local runs/* runs2/*
python scripts/judge_agreement.py runs/*
python scripts/retrieval_eval.py
```

Run the app locally (Metal on a Mac, CPU elsewhere):

```bash
IDIBO_BACKEND=llamacpp IDIBO_GGUF=models/idibo-idibo-q4_k_m.gguf uvicorn app.server:app --port 8000
# optional: YARNGPT_API_KEY in .env for Nigerian voices
```

The exact runs behind the reported numbers are the Kaggle scripts in `kaggle/`. Their outputs (generations, grades)
are in the `idibo-eval` dataset.

## Safety notes

- The benchmarks contain harmful *requests* only, never harmful content.
- Ìdìbò endorses no party. Party, ethnic and religious names appear only in symmetric swap tests.
- It is not official INEC guidance. Every answer cites its source and points to inecnigeria.org.

## Licence

Code: Apache-2.0. The model and adapters derive from Tiny Aya (CC-BY-NC-4.0) and inherit its non-commercial terms;
data: CC-BY-NC-4.0.
