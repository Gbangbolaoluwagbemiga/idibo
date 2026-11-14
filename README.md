# Ìdìbò: a safe, multilingual voter-information assistant for Nigeria's 2027 election

**One-line claim:** fine-tuning Cohere's Tiny Aya Earth with safety data mixed into civic training gives a model that
refuses every harmful and jailbreak request on a held-out test (0 of 56, 0 of 16) while refusing only 6% of honest
civic questions, against 35% for the same fine-tune without the safety data (p = 0.001, AuditKIT paired bootstrap).

> **Content warning.** This repository and its benchmarks contain *requests* for election harm (ethnic incitement,
> fake INEC notices, voter suppression, vote buying, intimidation), quoted for evaluation. No harmful completions are
> published: model answers to harmful requests that did not refuse are redacted in the released data.

| | |
|---|---|
| **Demo video** (3½ min) | https://youtu.be/Nn0A--cZJWw |
| **Live app** (web, voice, 5 languages) | https://idibo-2027.up.railway.app |
| **Results and method page** | https://idibo-2027.up.railway.app/results.html |
| **WhatsApp** | Meta Cloud API test number +1 555 192 8126 (registered test phones only; shown in the demo video) |
| **HF model collection** | https://huggingface.co/collections/GOPAE/idibo-models-nigeria-2027-election-safety-6af85462c9b6f21539aaa65b |
| **HF dataset collection** | https://huggingface.co/collections/GOPAE/idibo-data-nigeria-2027-election-safety-6af854634cce9ce0f613d91d |
| **Model repo** | [`GOPAE/idibo-tiny-aya-earth`](https://huggingface.co/GOPAE/idibo-tiny-aya-earth): adapters `idibo`, `civic`, `civic-safelora` and a 4-bit GGUF |
| **Dataset repo** | [`GOPAE/idibo-eval`](https://huggingface.co/datasets/GOPAE/idibo-eval): benchmarks, knowledge base, training data, graded answers, results |
| **Full write-up** | [docs/WRITEUP.md](docs/WRITEUP.md) · **Abstract (under 500 words):** [docs/ABSTRACT.md](docs/ABSTRACT.md) |
| **Team** | Oluwagbemiga Gbangbola (solo): GitHub [@Gbangbolaoluwagbemiga](https://github.com/Gbangbolaoluwagbemiga), Hugging Face `GOPAE` |

Lexsi AI Safety & Alignment Hackathon (Lexsi Labs with Cohere Labs and Hugging Face), November 2026.

---

## For judges: one minute

- **Problem.** Nigeria votes in January 2027. Voters will ask AI about registration, polling and the law in English,
  Nigerian Pidgin, Yorùbá, Hausa and Igbo. The same models can be asked for fake INEC notices, ethnic incitement, vote
  buying or endorsements. Safety tuning usually breaks the other way: the assistant starts refusing honest questions.
- **Headline (held-out benchmark, English):**

  | Model | Harmful answered | Jailbreaks that worked | Honest questions refused |
  |---|---|---|---|
  | Tiny Aya Earth + same prompt and sources | 28.6% | 50.0% | 2.1% |
  | Plain civic fine-tune | 0% | 0% | **35.4%** |
  | **Ìdìbò** (civic + safety data) | **0%** | **0%** | **6.2%** |

- **Base model + method:** `CohereLabs/tiny-aya-earth` (3.35B) + LoRA r16 trained RAFT-style with AlignTune on 1,060
  examples curated with CuratorKIT; SafeTune repairs of the plain fine-tune; AuditKIT red-team of the live app and
  significance tests for every claim.
- **Also found:** safety tuning made the model **invent INEC contact details** (a fake hotline 43 times; 6.7% of
  answers), and SafeTune ReSta made it worse (14.3%). A deployment check takes it to 0%. A live red-team found the app
  repeated injected phrases (6 attacks worked); a guard brought that to 0 of 19.
- **Deployed:** 4-bit GGUF on 8 CPU cores, about $0.0009 per answer against about $0.0075 for a closed API; on the web
  with Nigerian voice in and out, and on WhatsApp.

---

## Tracks and problem statements covered

- **Track 2b, problem 5: safety-preserving fine-tuning** (primary): a civic fine-tune hardened with safety data,
  compared with the same fine-tune without it.
- **Track 2b, problem 6: repairing safety drift without retraining**: SafeTune ReSta and Safe LoRA applied to the plain
  fine-tune, compared head to head.
- **Track 2a, problem 4: multilingual and code-mixed safety**: five languages, two never seen in training (Hausa,
  Igbo), and hand-written code-mixed prompts.
- **Track 2a, problem 2: a safety benchmark for a domain and its languages**: an election-safety benchmark for Nigeria
  (v1: 109 items × 5 languages; v2: held out, 116 English + 30 code-mixed).
- **Track 1, problem 4: low-resource languages for a public service** (secondary).

## Abstract

See [docs/ABSTRACT.md](docs/ABSTRACT.md) (under 500 words). In short: we fine-tune Tiny Aya Earth for grounded, cited
voter information in five Nigerian languages, mixing refusals with lawful alternatives, neutral replies and "ask INEC"
abstentions into the civic data. Against the same recipe without that data, the result keeps full safety and gives
back almost all the helpfulness lost to over-refusal; one training-free SafeTune repair (Safe LoRA) recovers two thirds
of the gap and the other (ReSta) does not. Safety transfers to Hausa, which was never trained. The system is
red-teamed live with AuditKIT and deployed on CPU, on the web and on WhatsApp.

## Problem statement and approach

**Why this problem.** An election assistant fails in two ways: it **helps with election harm** or takes a side, or it
**refuses ordinary civic questions** ("what is the penalty for vote buying?") because they look like harm. Both cost
votes. Nigerian voters ask in Pidgin, Yorùbá, Hausa and Igbo, languages with little safety evaluation.

**Threat model.** Users (or people posing as users) who want fake official notices (date or venue changes), ethnic or
religious incitement, voter-suppression messages, vote-buying scripts, intimidation, fake results, deepfake help or a
partisan endorsement; using direct requests, jailbreak wrappers (role-play, fiction, "research", translation, persona
split, continuation), code-mixing, and prompt injection hidden inside a forwarded message sent to the "Is this message
real?" checker.

**Research questions.**
1. Does mixing safety examples into a domain fine-tune keep safety without the over-refusal a plain fine-tune shows?
2. Can a drifted plain fine-tune be repaired without retraining, and at what cost?
3. Does the safety behaviour transfer to languages never seen in training (Hausa, Igbo) and to code-mixed prompts?
4. What breaks when the model is deployed as a full system (retrieval, prompt, 4-bit model, voice, WhatsApp)?

**Approach.**
1. A cited knowledge base: the Electoral Act 2026 (all 156 sections) and INEC voter pages, 309 passages.
2. Training data generated from that knowledge base, curated with CuratorKIT, translated to Pidgin and Yorùbá.
3. Two LoRA adapters with AlignTune's `RaftTrainer` (golden passage plus 3 distractors): `civic` (civic answers only)
   and `idibo` (civic answers + refusals with a lawful alternative + neutral replies + "ask INEC" abstentions).
4. SafeTune ReSta and Safe LoRA applied to `civic`, with no training.
5. Two benchmarks (v1, and a held-out v2 written after training), graded by back-translation plus an open-weight judge,
   with 95% Wilson intervals and AuditKIT paired-bootstrap significance.
6. Deployment as a 4-bit GGUF; AuditKIT red-team of the live URL; deployment checks for what the red-team and manual
   review found.

**Related work.** RAFT (Zhang et al., 2024: retrieval-augmented fine-tuning with distractors); Safe LoRA (Hsu et al.,
2024: projecting LoRA updates onto a safety subspace); restoring safety by task arithmetic, ReSta (Bhardwaj et al.,
2024); "Fine-tuning aligned language models compromises safety" (Qi et al., 2023); XSTest (Röttger et al., 2024:
over-refusal on look-alike prompts); multilingual jailbreaks (Yong et al., 2023; Deng et al., 2024); Aya and Tiny Aya
(Cohere Labs).

## Hackathon repos (kits) used, and how they connect

| Kit | How it was used | Features |
|---|---|---|
| **CuratorKIT** 0.1.21 | Curated the English SFT data before translation (`scripts/curate.py`) | `JSONLReader` → `SchemaGate` → `ExactDeduplicator` → `MinHashDeduplicator` (threshold 0.8) → `TextCleaner` → `MessagesExporter`, with the manifest, dataset card and provenance record (`data/sft/curatorkit/`) |
| **AlignTune** | Trained both LoRA adapters (`scripts/train_lora.py`) | `RaftTrainer` (golden + distractor passages, completion-only loss) |
| **SafeTune** | Repaired the plain `civic` adapter with no training (`scripts/patch_safetune.py`) | `ReStaTrainer` / `apply_resta` (safety vector θ_aligned − θ_base, α = 0.25, DARE, transformer layers only); `SafeLoRATrainer` / `apply_safe_lora` (threshold 0.5; 74 of 252 LoRA modules projected) |
| **AuditKIT** 1.1.9 | Red-teamed the deployed app (`scripts/audit_redteam.py`) and tested every before/after claim (`scripts/audit_compare.py`) | `RedTeamRunner` with the built-in probes (prompt injection, jailbreak, encoding, over-refusal) plus three custom election probes and detectors; `ak.evaluate`, `ak.compare(a, b).significance()` paired bootstrap |

**Pipeline:**

```
INEC pages + Electoral Act 2026 ──► knowledge base (309 cited passages)
        │
        ├─► SFT data (Cohere teacher) ──► CuratorKIT ──► leak filter ──► translate (pcm, yo) ──► RAFT prompts
        │                                                                                        │
        │                                     AlignTune RaftTrainer ◄────────────────────────────┘
        │                                       │                 │
        │                                    civic adapter     idibo adapter
        │                                       │
        │                                SafeTune ReSta / Safe LoRA ──► civic-resta, civic-safelora
        │
        └─► benchmarks v1, v2 ──► generate (vLLM) ──► back-translate + judge ──► AuditKIT significance
                                                                           │
                         idibo ──► GGUF Q4 ──► live app ──► AuditKIT red-team ──► guard, contact check
```

## Models and licences

| Role | Model | Licence |
|---|---|---|
| Base (fine-tuned) | `CohereLabs/tiny-aya-earth` (3.35B, dense), fp16 for training and evaluation, Q4_K_M GGUF deployed | CC-BY-NC-4.0 |
| ReSta base reference | `CohereLabs/tiny-aya-base` | CC-BY-NC-4.0 |
| Teacher (SFT data) and round-1 grader | `command-a-plus` (Cohere API, supporting role) | Cohere API terms |
| Translation | `command-a-translate` (API); `CohereLabs/tiny-aya-global` (back-translation for grading) | API terms / CC-BY-NC-4.0 |
| Open-weight grader | `CohereLabs/c4ai-command-r7b-12-2024` | CC-BY-NC-4.0 |
| Leaderboard (running at the time of writing) | `tiny-aya-global`, `tiny-aya-fire`, `tiny-aya-water`, `aya-expanse-8b`, `aya-23-8B` | CC-BY-NC-4.0 |
| Retrieval embedder | `intfloat/multilingual-e5-small` | MIT |

Every language model in the pipeline is a Cohere model.

## Datasets, benchmarks and licences

**Sources.** INEC web pages and documents (inecnigeria.org) and the Electoral Act 2026 (National Assembly): public law
and public information, quoted with source URLs. Synthetic training data generated from them by a Cohere teacher.
Benchmarks written for this project. Released data: CC-BY-NC-4.0.

**Training data** (`data/train/`):

| Set | English | Pidgin | Yorùbá | Total | Content |
|---|---|---|---|---|---|
| `civic` | 191 | 191 | 184 | 566 | civic answers + "ask INEC" abstentions (33 per language) |
| `idibo` | 356 | 356 | 348 | 1,060 | per language: civic 158 (Yorùbá 151), helpful look-alikes 77, refusals with a lawful alternative 60, neutral replies 28, abstentions 33 |

Hausa and Igbo were left out of training on purpose, to measure transfer.

**CuratorKIT filtering** (English, before translation): 374 rows read → 374 passed the schema gate → 374 after exact
dedup → **368 after MinHash dedup** (6 near-duplicates removed at threshold 0.8) → 368 cleaned and exported.

**Train/test splits and leak check.** Benchmarks are never trained on. Two leak checks with multilingual-e5-small
cosine similarity:
1. The curation step dropped every training example within 0.935 of any benchmark prompt in any language: **12
   dropped** (similarity 0.935–0.962), 356 kept (`data/sft/leakage.json`).
2. Benchmark v2 was written after training and checked against all training questions: maximum similarity **0.948**,
   none ≥ 0.95 (`data/bench2/leakage.json`).

**Safety benchmarks** (the focus task):

| Benchmark | Items | Languages | Notes |
|---|---|---|---|
| v1 | 109 per language: 20 facts (gold answer + source), 8 unanswerable, 18 harmful requests paired with 18 benign look-alikes, 45 neutrality swaps (6 parties, 5 ethnic groups, 2 religions) | en, pcm, yo, ha, ig | non-English machine-translated (Cohere), `src_en` kept |
| v2 (held out) | 116 English: 40 harmful/benign pairs over 10 harm categories, 16 jailbreak-wrapped harmful + 8 benign wrappers, 12 religion swaps; plus 30 hand-written code-mixed prompts | en, pcm-en, yo-en, ha-en | written after training |

**Attack prompt set** (live red-team, AuditKIT): built-in prompt injection (8), jailbreak (8), encoding (5) and
over-refusal (5); plus ours: indirect injection hidden in a forwarded "INEC notice" sent to the checker (6, in
en/pcm/yo/ha), instruction extraction in Nigerian languages (4), endorsement bait (4).

**Over-refusal test set:** the benign look-alike half of every harmful/benign pair (v1: 18 per language; v2: 48
English, including 8 benign jailbreak wrappers; 6 code-mixed).

## Metrics, and why these

| Metric | Why it matters here | Better |
|---|---|---|
| Harmful compliance | the core safety failure | lower |
| Jailbreak compliance | real users try role-play, fiction, "research" wrappers | lower |
| Over-refusal (benign look-alikes refused) | a safe but useless assistant fails voters; the usual cost of safety tuning | lower |
| Took a side / stereotyped (swap tests) | neutrality is a legal and ethical requirement | lower |
| Facts correct; abstains when unknowable | a wrong date or polling time disenfranchises people | higher |
| Same-language rate | a Hausa speaker should get Hausa | higher |
| Retrieval recall@4 | the model can't cite what it never sees | higher |
| Invented contacts (web, email, phone not in the sources) | a made-up "INEC hotline" is a scam vector | lower |
| Live attack success (AuditKIT) | the deployed system, not just the model | lower |

**What should not get worse:** over-refusal is reported next to every safety number; facts and same-language rate show
whether the safety data cost usefulness.

**Checking the automatic judge.** Answers are back-translated to English by `tiny-aya-global` and graded one at a time
against a fixed rubric by `c4ai-command-r7b`. On the 1,241 items also graded by `command-a-plus` (round 1): **83.9%
agreement, Cohen's κ = 0.80**. On harmful items κ is only 0.28, and the open grader is more lenient (of 33 answers
command-a-plus called harmful, it called 20 refusals), so its harmful-compliance rates are **lower bounds**. A simple
refusal-opening check agrees with the ranking: 1 of Ìdìbò's 56 v2 harmful answers lacks one, against 40 for the base.

## Method and hyperparameters

**AlignTune `RaftTrainer` (both adapters):** LoRA r = 16, α = 32, dropout 0.05 on all projections (q, k, v, o, gate,
up, down); 2 epochs; learning rate 1e-4 with a cosine schedule and 5% warmup; batch 1 × gradient accumulation 8; max
length 2,048; fp16; completion-only loss; gradient checkpointing; seed 42. Each prompt: system policy + golden passage
+ 3 retrieved distractors. One Kaggle T4: about 1.7 h (`civic`) and 3.1 h (`idibo`).

**SafeTune:** ReSta with α = 0.25, DARE, transformer layers only (`model.layers.`), aligned = Tiny Aya Earth, base =
Tiny Aya Base: 143 s on one T4. Safe LoRA, threshold 0.5, same pair: 74 of 252 LoRA modules projected, 110 s.

**Retrieval:** hybrid BM25 + multilingual-e5-small with reciprocal-rank fusion, top 4. For Yorùbá, Hausa and Igbo the
model first translates the question to English and searches with both.

**Generation:** greedy decoding, max 320 new tokens, vLLM (v2) or HF transformers (v1), identical prompts.

**AuditKIT:** `RedTeamRunner(callable).run(probes, detectors)` against `https://idibo-2027.up.railway.app/api/ask`; a
paired bootstrap over item ids for each metric and pair of conditions.

**Seeds:** training seed 42; vLLM seed 0; greedy decoding throughout.

## Results

### Same model before vs after (held-out benchmark v2, English, % [95% CI])

| Model | Harmful | Jailbreak | Over-refusal | Took a side |
|---|---|---|---|---|
| Tiny Aya Earth + our prompt + sources | 28.6 [18.4–41.5] | 50.0 [28.0–72.0] | 2.1 [0.4–10.9] | 0 of 12 |
| Plain civic fine-tune | 0 [0–6.4] | 0 [0–19.4] | 35.4 [23.4–49.6] | 0 of 12 |
| + SafeTune ReSta | 0 [0–6.4] | 0 [0–19.4] | 27.1 [16.6–41.0] | 0 of 12 |
| + SafeTune Safe LoRA | 5.4 [1.8–14.6] | 0 [0–19.4] | 16.7 [8.7–29.6] | 0 of 12 |
| **Ìdìbò** | **0 [0–6.4]** | **0 [0–19.4]** | **6.2 [2.1–16.8]** | 0 of 12 |

### Every language (benchmark v1, command-r7b grades)

| Language | Model | Harmful | Over-refusal | Facts correct | Abstains when it should |
|---|---|---|---|---|---|
| English | base + sources / plain fine-tune / **Ìdìbò** | 27.8 / 5.6 / **0** | 0 / 44.4 / **11.1** | 75 / 65 / 65 | 100 / 100 / 100 |
| Pidgin | base / plain / **Ìdìbò** | 22.2 / 11.1 / **5.6** | 5.6 / 55.6 / **11.1** | 60 / 50 / 45 | 87.5 / 100 / 100 |
| Yorùbá | base / plain / **Ìdìbò + query translation** | 5.6 / 5.9 / **0** | 27.8 / 66.7 / **38.9** | 30 / 15 / 35 | 87.5 / 87.5 / 100 |
| Hausa (unseen) | base / plain / **Ìdìbò + query translation** | 11.1 / 11.2 / **5.6** | 0 / 55.6 / **16.7** | 30 / 15 / 30 | 100 / 100 / 100 |
| Igbo (unseen, beta) | base / **Ìdìbò + query translation** | 5.6 / **5.6** | 11.1 / **33.3** | 50 / 25 | 100 / 87.5 |

Answers in Pidgin when asked in Pidgin: base 12.8%, Ìdìbò 100%.

### Significance (AuditKIT paired bootstrap)

| Comparison | Measure | Change | p |
|---|---|---|---|
| Ìdìbò vs base | refuses harmful | +0.29 | 0.001 |
| Ìdìbò vs base | refuses jailbreaks | +0.50 | 0.001 |
| Ìdìbò vs plain fine-tune | helps on honest questions | +0.29 | 0.001 |
| Safe LoRA vs plain fine-tune | helps on honest questions | +0.19 | 0.012 |
| Safe LoRA vs plain fine-tune | refuses harmful | −0.05 | 0.12 (not significant) |
| ReSta vs plain fine-tune | helps on honest questions | +0.08 | 0.27 (not significant) |
| Query translation vs none (Yorùbá) | facts correct | +0.30 | 0.005 |

Every row: `runs/audit/significance.md`.

### Findings

- **Safety-preserving fine-tuning works.** The plain fine-tune became "safe" by refusing a third of honest questions;
  mixing safety examples into training kept the safety and gave the helpfulness back.
- **Of the two repairs, only Safe LoRA helps an over-refusing model.** It recovers about two thirds of the gap at a cost
  of 3 harmful answers in 56 (not significant). ReSta, which adds more of the safety direction, does not help, and on
  v1 Yorùbá it made over-refusal worse (p = 0.034).
- **Safety transfers to an unseen language.** Hausa: harmful 5.6% (base 11.1%), over-refusal 16.7–22.2% (plain
  fine-tune 55.6%, p = 0.023).
- **Igbo, also unseen, is the weakest.** Facts 25% against 50% for the base. Labelled beta in the app.
- **Safety tuning invents contacts.** Answers with a web address, email or phone found nowhere in the sources: base
  4.8%, plain fine-tune 1.9%, ReSta 14.3%, Safe LoRA 0.5%, Ìdìbò 6.7%. "Report it to…" refusals invite a made-up
  "where" (`0800-INEC-HELP` 43 times; the domain `inec.gov.ng`, which does not exist). A check against the sources
  removes all of them.
- **A safety-tuned model can't build its own safety benchmark.** Asked to translate harmful prompts, Tiny Aya Earth
  answered or refused them instead, which would silently turn attacks into benign items; v2 is English plus
  hand-written code-mixed prompts.

### Results on the deployed (4-bit GGUF) system: live red-team

| Probe (AuditKIT, against the live URL) | Before the guard | After |
|---|---|---|
| Prompt injection | 3 of 8 | **0 of 8** |
| Encoding | 1 of 5 | **0 of 5** |
| Indirect injection (hidden instruction in a forwarded "INEC notice") | 2 of 6 | **0 of 6** |
| Jailbreak · instruction extraction · endorsement bait | 0 · 0 · 0 | 0 · 0 · 0 |
| Over-refusal (a calculator program: out of scope by design) | 1 of 5 | 1 of 5 |

The benchmark numbers were measured on the fp16 adapter. The deployed Q4 GGUF was evaluated through this live red-team
and manual spot checks, not the full benchmark (see "With more credits").

### What didn't work

- Telling the model "reply in English" (in the system or user prompt) did not stop it mirroring a Pidgin-flavoured
  question.
- Larger embedders (BGE-M3, LaBSE, e5-base) did not fix cross-lingual retrieval; translating the query did.
- Machine-translating benchmark v2 with a safety-tuned model (see Findings).
- ReSta as a repair for over-refusal.
- Twilio's new trial WhatsApp numbers only send templates; WhatsApp runs on Meta's Cloud API instead.

### Sample inputs and outputs

Side-by-side answers by language (attacks, honest questions, and a failure) are on the results page:
https://idibo-2027.up.railway.app/results.html#examples. Every answer and grade is in `GOPAE/idibo-eval`.

### Cost and ROI

- **Per answer** (about 2,000 input + 250 output tokens):
  - Closed API (Cohere Command A at $2.50 per million input tokens and $10 per million output): about **$0.0075**, i.e.
    **$7.50 per 1,000 replies**.
  - Self-hosted Ìdìbò (Q4 GGUF on 8 vCPU / 4 GB on Railway, about $200/month at full use, about 12 s per answer ≈
    216k answers a month): about **$0.0009**, i.e. **$0.90 per 1,000 replies**. Roughly 8× cheaper, and the data and
    safety behaviour stay ours to audit.
- **Compute used:** free Kaggle quota (1× and 2× NVIDIA T4): about 5 GPU-hours of training, about 1 hour of SafeTune
  and generation, about 1 hour of grading, and about 3 hours for the leaderboard. The Cohere API trial for data
  generation and round-1 grading. Railway for hosting. Out-of-pocket cost: hosting only.

## Limitations

- Over-refusal stays high on code-mixed prompts (4 of 6 benign refused) and in Yorùbá (39% with query translation).
- Igbo is untrained and weaker than the base model on facts.
- The model mirrors the question's style: a Pidgin-flavoured question with English selected may get a Pidgin answer.
- It sometimes adds "ID or passport" to what to bring to vote (the requirement is the PVC).
- The model itself still invents contact details; the deployment check, not the model, removes them.
- The open-weight grader is lenient on harmful items (κ = 0.28), so harmful rates are lower bounds.
- Benchmarks are small; read the intervals. The v1 translations and the app's Yorùbá, Hausa and Igbo guard messages
  are machine-made and not yet reviewed by native speakers.
- INEC's 2027 presidential candidate list was added to the app after evaluation.
- WhatsApp runs on Meta's test number (up to five registered phones); a public launch needs a verified business number.
- Not official INEC guidance.

## With more credits

| Planned approach | Expected before → after |
|---|---|
| Native-speaker review of all translations, plus Hausa and Igbo training data | Igbo facts 25% → at least the base model's 50%; Igbo over-refusal 33% → under 15% |
| Code-mixed training examples and more benign look-alikes | code-mixed over-refusal 4 of 6 → about 1 of 6 (Safe LoRA's level) |
| Contact details answered only from the knowledge base, in training, not just a check | invented contacts in raw model output 6.7% → under 1% |
| The full benchmark on the deployed Q4 GGUF, and fine-tunes of Aya Expanse 8B / 32B | confirm the fp16 results survive quantisation; better facts from a stronger base |
| A human-labelled sample of 200 harmful answers | a calibrated judge (κ on harmful items from 0.28 toward 0.7) |
| Election-day scale: replicas behind a queue, a verified WhatsApp sender, an SMS/USSD fallback | from 5 test phones to public use |

## Artifacts

- **HF models collection:** adapters `idibo`, `civic`, `civic-safelora` and `idibo-q4_k_m.gguf` (link at the top).
- **HF datasets collection:** benchmarks v1 and v2 with leak checks, the knowledge base, training data, every generation
  with its grades (harmful completions redacted), results, and AuditKIT outputs (link at the top).
- **GitHub:** this repository. **Demo:** the live app and WhatsApp; the video is in the submission.

## Repository layout

| Path | What it does |
|---|---|
| `scripts/scrape_kb.py`, `scripts/build_kb.py` | Crawl INEC pages and the Electoral Act 2026 into 309 dated, cited passages (`data/kb/`) |
| `scripts/add_candidates_kb.py` | Add INEC's final 2027 presidential candidate list (app only, after evaluation) |
| `scripts/gen_sft.py` | English training data from the knowledge base, using a Cohere teacher |
| `scripts/curate.py` | **CuratorKIT** pipeline, then the leak filter against all benchmark prompts |
| `scripts/translate_sft.py`, `scripts/translate.py` | Translation to Pidgin and Yorùbá (training) and of benchmark v1 |
| `scripts/build_train.py` | RAFT prompts (golden + 3 distractors) for `civic` (566) and `idibo` (1,060) |
| `scripts/train_lora.py` | LoRA training with **AlignTune** `RaftTrainer` |
| `scripts/patch_safetune.py` | **SafeTune** ReSta (`--method resta`) and Safe LoRA (`--method safelora`) |
| `scripts/build_bench.py`, `scripts/build_bench2.py`, `scripts/leakage_bench2.py` | Benchmarks v1 and v2, and the v2 leak check |
| `scripts/run_eval.py`, `scripts/run_eval_vllm.py` | Generate answers (HF or vLLM, same prompts, greedy) |
| `scripts/judge_local.py`, `scripts/score.py`, `scripts/judge_agreement.py` | Back-translation, rubric grading, 95% Wilson intervals, grader agreement |
| `scripts/audit_redteam.py`, `scripts/audit_compare.py` | **AuditKIT** live red-team and paired-bootstrap significance |
| `scripts/contacts_eval.py`, `scripts/retrieval_eval.py` | Invented-contact counts; retrieval recall per language |
| `scripts/package_hf_dataset.py` | Builds the public dataset with harmful completions redacted |
| `idibo/retrieve.py`, `idibo/pivot.py` | Hybrid retrieval; translate-then-search for yo/ha/ig |
| `idibo/guard.py`, `idibo/contacts.py` | Deployment checks: prompt-injection guard; contact check |
| `app/` | FastAPI app: llama.cpp CPU inference, YarnGPT voice, web UI, results page, WhatsApp webhooks (Meta and Twilio) |
| `kaggle/*/run.py` | Every GPU job exactly as run on Kaggle: `base-eval`, `train`, `round2`, `resta`, `judge`, `gguf`, `publish`, `leaderboard` |
| `docs/` | Write-up, abstract, submission notes, demo script, AuditKIT significance tables |
| `hf/` | Model and dataset cards |

## Reproduce

**Environment.** Python 3.12. Training and evaluation: 1–2 GPUs with 16 GB each (we used free Kaggle T4s). A Hugging
Face token that has accepted the Tiny Aya and Command R7B licences. The app runs on CPU (8 vCPU recommended) or Metal.

```bash
pip install -r requirements.txt          # app + scripts; the GPU jobs pin their own versions (kaggle/*/run.py)

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

# 4. Generate answers (one example; kaggle/round2/run.py runs the full grid)
python scripts/run_eval_vllm.py --model CohereLabs/tiny-aya-earth --adapter adapters/idibo --rag \
    --bench data/bench2 --langs en,mix --name idibo-rag --out runs2/idibo-rag

# 5. Grade with open-weight models, summarise with 95% intervals, test significance
python scripts/judge_local.py --stage translate --runs runs runs2 --model CohereLabs/tiny-aya-global
python scripts/judge_local.py --stage judge --runs runs runs2 --model CohereLabs/c4ai-command-r7b-12-2024
python scripts/score.py --summary-only --suffix graded.local runs/* runs2/*
python scripts/judge_agreement.py runs/*
python scripts/audit_compare.py --suffix graded.local
python scripts/contacts_eval.py
python scripts/retrieval_eval.py

# 6. Red-team the live app (or a local one with --url)
python scripts/audit_redteam.py
```

**Run the app locally:**

```bash
IDIBO_BACKEND=llamacpp IDIBO_GGUF=models/idibo-q4_k_m.gguf uvicorn app.server:app --port 8000
# optional, in .env: YARNGPT_API_KEY (Nigerian voices); META_WA_TOKEN, META_PHONE_NUMBER_ID, META_APP_SECRET and
# META_VERIFY_TOKEN (WhatsApp)
```

The exact runs behind the reported numbers are the Kaggle scripts in `kaggle/`; their outputs are in `GOPAE/idibo-eval`.

## Declaration of work done outside the hackathon

None. All code, data, models and the app were built during the hackathon, starting 11 November 2026 (first commit).
We used open-source libraries (the Lexsi kits, Hugging Face, vLLM, llama.cpp, FastAPI), Cohere models, public INEC
documents and the Electoral Act 2026.

## Responsible use

- The benchmarks contain harmful *requests* only; answers that did not refuse are redacted in the released data.
- Ìdìbò endorses no party. Party, ethnic and religious names appear only in symmetric swap tests.
- It is not official INEC guidance, says so, cites its sources and points to inecnigeria.org.
- No user data is stored beyond an in-memory language choice on WhatsApp.

## Acknowledgements and references

Lexsi Labs (CuratorKIT, AlignTune, SafeTune, AuditKIT), Cohere Labs (Tiny Aya, Aya, Command models), Hugging Face,
Kaggle (free GPUs), YarnGPT (Nigerian voices), INEC and the National Assembly (public documents). References are under
Related work and in [docs/WRITEUP.md](docs/WRITEUP.md).

**LLM usage statement.** A large share of the code and documentation was written with an AI coding assistant, directed,
checked and run by the author. All experiments and numbers come from the scripts and logs in this repository.

## Licence

Code: Apache-2.0. The adapters and GGUF derive from Tiny Aya (CC-BY-NC-4.0) and inherit its non-commercial terms.
Data: CC-BY-NC-4.0.
