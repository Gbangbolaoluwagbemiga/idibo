# Ìdìbò: a safe, multilingual voter-information assistant for Nigeria's 2027 election

*Lexsi AI Safety & Alignment Hackathon. Primary track: **2b, problem 5 (safety-preserving fine-tuning)**. Also: **2b-6**
(repairing safety drift without retraining), **2a-4** (multilingual and code-mixed safety) and **2a-2** (a safety
benchmark for a domain and its languages). Secondary: **Track 1, problem 4** (low-resource languages for a public service).*

> Status of this draft: numbers marked ⟨…⟩ are filled in from the final grading run (`kaggle/judge`).

## 1. Why this problem

Nigeria votes in early 2027. Voters will ask chat assistants how to register, where to vote and what the law says, in
English, Nigerian Pidgin, Yorùbá, Hausa and Igbo. Open models that speak these languages now exist (Tiny Aya). The same
models can be asked to write ethnic incitement, fake INEC notices that move the election date, vote-buying scripts or
partisan endorsements. A civic assistant fails in two ways:
- it **helps with election harm**, or takes a political side;
- it **refuses ordinary civic questions** ("what is the penalty for vote buying?") because they look like harm.

Ìdìbò is built to avoid both, in the languages people actually use, with answers grounded in INEC pages and the
Electoral Act 2026.

## 2. What we built

| Piece | What it is | Lexsi repo |
|---|---|---|
| Knowledge base | 309 cited passages: Electoral Act 2026 sections and INEC voter pages | CuratorKIT (cleaning and dedup of the SFT data) |
| Ìdìbò adapter | LoRA (r16) on `CohereLabs/tiny-aya-earth`, trained RAFT-style: golden source plus 3 distractor passages, loss on the answer only. 1,060 examples: civic Q&A, refusals with a lawful alternative, neutral answers to partisan prompts, "I don't know, ask INEC" answers | AlignTune (`RaftTrainer`) |
| Civic adapter (control) | The same recipe trained on civic Q&A only (566 examples): the "plain fine-tune" | AlignTune |
| Repairs, no training | ReSta (safety vector `θ_aligned − θ_base`, α = 0.25, DARE) and Safe LoRA (projects LoRA updates onto the safety subspace) applied to the civic adapter | **SafeTune** |
| Benchmarks | v1: 109 items × 4 languages; v2 (held out, written after training): 116 English items plus 30 hand-written code-mixed prompts | — |
| Grader | Answers are back-translated by `tiny-aya-global` and graded by `c4ai-command-r7b` on our own GPUs. Agreement with the earlier `command-a-plus` API grades is reported (§5) | — |
| App | https://idibo-2027.up.railway.app: the 4-bit GGUF on CPU, cited answers, Nigerian voice in and out (YarnGPT), five languages | — |

Base model: `CohereLabs/tiny-aya-earth` (3.35B, dense, Cohere). Every model in the pipeline is a Cohere model.

## 3. Metrics, and why these

| Metric | Why it matters for an election assistant | Better |
|---|---|---|
| **Harmful compliance**: share of harmful requests where the model produces the harmful content | the core safety failure | lower |
| **Jailbreak compliance**: the same, under role-play, fiction, "research", translation, "hypothetically", persona-split and continuation wrappers | real users do try this | lower |
| **Over-refusal**: share of benign look-alike requests (same topic as a harmful one) that the model refuses | a safe but useless assistant fails voters; this is the usual cost of safety tuning | lower |
| **Partisanship**: took a side, or stereotyped a group, on party, ethnicity and religion swap tests | neutrality is a legal and ethical requirement | lower |
| **Grounded facts**: correct on answerable questions; abstains on unknowable ones | wrong polling hours or dates disenfranchise people | higher |
| **Same-language rate**: answers in the user's language | a Hausa speaker should get Hausa | higher |
| **Retrieval recall@4**: the right source in the top 4 | a model can't cite what it never sees | higher |

Every rate is reported with a 95% Wilson interval. Version 1 had only 18 harmful prompts, so one answer moved the rate by
5.6 points. Version 2 was written so that the claims stand on more than a handful of items.

## 4. Results

### 4.1 Safety-preserving fine-tuning (2b-5): plain fine-tune vs Ìdìbò

Benchmark v2, English (held out). Rates are % ⟨95% CI⟩.

| Model | Harmful compliance | Jailbreak compliance | Over-refusal | Took side |
|---|---|---|---|---|
| Tiny Aya Earth + our prompt + sources | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| Plain civic fine-tune | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| **Ìdìbò** | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

Benchmark v1 in four languages (command-a-plus grades, round 1):

| Pidgin | Base + sources | Plain fine-tune | Ìdìbò |
|---|---|---|---|
| Harmful compliance | 55.5% | 5.6% | **0%** |
| Over-refusal | 11.1% | 61.1% | **16.7%** |
| Took a side | 24.4% | 6.7% | **0%** |
| Answers in Pidgin | 12.8% | 100% | **100%** |

The plain fine-tune becomes "safe" by refusing almost everything, 44–61% of benign civic questions. Ìdìbò keeps the
safety and gives back most of the helpfulness.

### 4.2 Repairing a drifted fine-tune without retraining (2b-6)

The civic adapter is repaired with SafeTune and no gradient step: ReSta took 143 s on one T4, and Safe LoRA took 110 s
(it projected 74 of 252 LoRA modules).

| Model | Harmful | Jailbreak | Over-refusal |
|---|---|---|---|
| Plain civic fine-tune | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| + ReSta (SafeTune) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| + Safe LoRA (SafeTune) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| Ìdìbò (safety in the training data) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

### 4.3 Multilingual and code-mixed safety (2a-4)

- **Hausa was never in the training data** (training covered en/pcm/yo). Hausa results therefore measure zero-shot transfer of the safety behaviour: ⟨…⟩.
- **Code-mixed prompts** (Pidgin-English, Yorùbá-English, Hausa-English, written by hand): ⟨…⟩.
- **Igbo** was neither trained nor (until now) evaluated: ⟨…⟩. The app labels it "Beta".

### 4.4 Cross-lingual retrieval

The knowledge base is English. Retrieval recall@4 on the 20 fact questions:

| | en | pcm | yo | ha | ig |
|---|---|---|---|---|---|
| Hybrid search, native question | 14 | 13 | 7 | 4 | 8 |
| Larger embedders, dense only (best of e5-base, LaBSE, BGE-M3) | 13 | 15 | 5 | 13 | 8 |
| **Native question + the model's own English translation** | 14 | 14 | **12** | **16** | **10** |

The translation step costs about 0.5 s on a laptop. It is live in the app for yo/ha/ig. Its effect on answer quality: ⟨…⟩.

## 5. How we measured, and its limits

- **Held-out test set.** Benchmark v2 was written after training. Its nearest training question has a cosine similarity
  of at most 0.948 (multilingual-e5); there are no near-duplicates (`data/bench2/leakage.json`).
- **Grader.** The answer is back-translated into English, then graded against a fixed rubric, one item at a time.
  Round 1 used `command-a-plus` over the API. When the trial quota ran out we moved to open weights on our own GPUs.
  Agreement between the two graders on the same items: ⟨κ…⟩.
- **Why v2 is English plus code-mixed only.** We first machine-translated v2 with Tiny Aya Earth. The safety-tuned
  translator **answered or refused** many harmful prompts instead of translating them, which would silently turn
  harmful items into benign ones. Finding: a safety-tuned model cannot be trusted to build its own multilingual safety
  benchmark. The v1 translations (Cohere API) are kept, but are also machine-made: one Yorùbá item turned "when" into
  "where". Native-speaker review sheets are in `review/`.
- Generation is greedy. The v2 conditions were generated with vLLM and the v1 conditions with HF transformers, using the
  same prompts.

## 6. Cost: closed API vs self-hosted (estimate)

Per answer: about 2,000 input tokens (instructions and 4 sources) and about 250 output tokens.
- **Closed API** (Cohere Command A list price, $2.50 / M input, $10 / M output): about **$0.0075 per answer**, so
  **about $7,500 per million questions**.
- **Self-hosted Ìdìbò** (4-bit, 8 vCPU / 4 GB on Railway at $20 / vCPU-month + $10 / GB-month, about $200/month fully
  used): about 12 s per answer, about 216k answers per month per replica, so **about $0.0009 per answer, or about
  $900 per million questions**. That is roughly 8× cheaper, the data stays in our deployment, and the safety behaviour
  is ours to audit and repair.

## 7. Reproduce

See the repository README: `scripts/` (data, training, evaluation, grading), `kaggle/` (each GPU job as a script) and
`space/` (the app image).

## 8. Responsible-use notes

- The benchmark contains harmful *requests* only, never harmful content.
- Ìdìbò endorses no party. Party names appear only in symmetric swap tests.
- It is not official INEC guidance, and it says so.
- Dates come from INEC's published calendar in the knowledge base (presidential election: 16 January 2027). Earlier
  reports said 20 February; the benchmark checks that the model corrects that rumour from the source instead of
  agreeing with it. Every answer cites its source and points to inecnigeria.org for changes.
