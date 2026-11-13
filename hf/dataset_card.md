---
license: cc-by-nc-4.0
language: [en, pcm, yo, ha, ig]
tags: [safety, elections, nigeria, benchmark, red-teaming, code-mixed, lexsi-hackathon]
pretty_name: Ìdìbò election-safety benchmark and data
---

# Ìdìbò: election-safety benchmark, training data and results (Nigeria 2027)

Everything behind [`GOPAE/idibo-tiny-aya-earth`](https://huggingface.co/GOPAE/idibo-tiny-aya-earth).

## Contents

| Path | What |
|---|---|
| `bench/{en,pcm,yo,ha,ig}.jsonl` | **Benchmark v1**: 109 items per language. 20 fact (with gold answer and source id), 8 abstain (unknowable or conflicting), 18 harmful requests each paired with 18 benign look-alikes, 45 neutrality swaps (6 parties, 5 ethnic groups, 2 religions). Non-English: machine-translated (Cohere models), `src_en` holds the original |
| `bench2/en.jsonl` | **Benchmark v2 (held out, written after training)**: 40 harmful/benign pairs over 10 harm categories, 16 jailbreak-wrapped harmful requests (role-play, fiction, research, translation, hypothetical, persona split, JSON, continuation, emotional appeal, …) with 8 benign wrappers, 12 religion swaps |
| `bench2/mix.jsonl` | 30 **code-mixed** prompts written by hand (Pidgin-English, Yorùbá-English, Hausa-English), with `src_en` |
| `bench2/leakage.json` | Nearest training question for each v2 item (max cosine 0.948; none ≥ 0.95) |
| `kb/chunks.jsonl` | 309 cited passages: Electoral Act 2026 sections and INEC voter pages, with URL and date |
| `train/{idibo,civic}.jsonl` | Training sets (prompt with golden plus distractor passages → completion) |
| `sft/` | English SFT data, the CuratorKIT run (manifest, dataset card, rejected rows) and translations |
| `generations/`, `runs/`, `runs2/` | Every model answer, plus grades from command-a-plus (round 1) and from tiny-aya-global + c4ai-command-r7b (all rounds) |

Item fields: `id, kind (fact|abstain|harmful|benign|neutral), category, prompt, expected, gold?, cite?, pair?, swap?, src_en?`.

## Intended use

Measuring whether an assistant for elections, especially in Nigerian languages:
- helps with election harm, or over-refuses look-alike civic questions;
- takes political sides;
- invents facts.

Harmful items are **requests only**; the dataset contains no harmful completions.

## Known issues

- Machine translations are unreviewed in places. Example: one Yorùbá fact item asks "where" instead of "when".
- We tried to machine-translate v2 with a safety-tuned model (Tiny Aya Earth). It answered or refused many harmful
  prompts instead of translating them, so v2 is English plus hand-written code-mixed only.
- Small: read the 95% intervals the scorer reports.
