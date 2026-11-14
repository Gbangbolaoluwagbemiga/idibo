# Ìdìbò: a safe, multilingual voter-information assistant for Nigeria's 2027 election

*Lexsi AI Safety & Alignment Hackathon. Primary track: **2b, problem 5 (safety-preserving fine-tuning)**. Also: **2b-6**
(repairing safety drift without retraining), **2a-4** (multilingual and code-mixed safety) and **2a-2** (a safety
benchmark for a domain and its languages). Secondary: **Track 1, problem 4** (low-resource languages for a public service).*

**In one paragraph.** On a held-out benchmark written after training, a plain civic fine-tune of Tiny Aya Earth became
safe by refusing: it refused 35% of ordinary civic questions. Ìdìbò, trained with safety data mixed in, refused every
harmful and jailbreak request (0 of 56 and 0 of 16) and only 6% of benign look-alikes. AuditKIT's paired bootstrap
says that gain in helpfulness is real (+29 points, p = 0.001). SafeTune's Safe LoRA, applied to the plain fine-tune
with no training, won back about two thirds of that over-refusal gap (35% → 17%, p = 0.012). An AuditKIT red-team of
the live app then found that it repeated injected phrases; we added a guard and re-tested (§6).

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
| Audit | Red-team of the deployed app (built-in probes plus three election probes) and paired-bootstrap significance for every before/after claim | **AuditKIT** |
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

Benchmark v2, English (held out: 56 harmful requests including 16 jailbreaks, 48 benign look-alikes, 12 swap tests).
Rates are % [95% Wilson CI], graded by `c4ai-command-r7b`.

| Model | Harmful compliance | Jailbreak compliance | Over-refusal | Took a side |
|---|---|---|---|---|
| Tiny Aya Earth + our prompt + sources | 28.6 [18.4–41.5] | 50.0 [28.0–72.0] | 2.1 [0.4–10.9] | 0 of 12 |
| Plain civic fine-tune | 0 [0–6.4] | 0 [0–19.4] | 35.4 [23.4–49.6] | 0 of 12 |
| **Ìdìbò** | **0 [0–6.4]** | **0 [0–19.4]** | **6.2 [2.1–16.8]** | 0 of 12 |

AuditKIT paired bootstrap on the same items (`scripts/audit_compare.py`): Ìdìbò vs the base model, refusing harmful
+0.29 (p = 0.001) and jailbreaks +0.50 (p = 0.001); Ìdìbò vs the plain fine-tune, helping on benign questions +0.29
(p = 0.001). The harmful rate of the base model is a lower bound: this grader counts a "counter-message" rewrite as a
refusal, and 40 of the base model's 56 answers do not open with a refusal (§5).

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

Benchmark v2, English, % [95% CI]:

| Model | Harmful | Jailbreak | Over-refusal |
|---|---|---|---|
| Plain civic fine-tune | 0 [0–6.4] | 0 of 16 | 35.4 [23.4–49.6] |
| + ReSta (SafeTune) | 0 [0–6.4] | 0 of 16 | 27.1 [16.6–41.0] |
| + Safe LoRA (SafeTune) | 5.4 [1.8–14.6] | 0 of 16 | **16.7 [8.7–29.6]** |
| Ìdìbò (safety in the training data) | 0 [0–6.4] | 0 of 16 | 6.2 [2.1–16.8] |

- **Safe LoRA** cut over-refusal from 35% to 17%, about two thirds of the gap to Ìdìbò, with no training: helpful
  +0.19, p = 0.012 (AuditKIT). It cost 3 harmful answers out of 56 (−0.05, p = 0.12, not significant). The same
  helpfulness gain is significant on benchmark v1 in English (+0.28, p = 0.014), Pidgin (+0.33, p = 0.029) and Hausa
  (+0.33, p = 0.009).
- **ReSta** (α = 0.25, transformer layers only) did not significantly help (+0.08, p = 0.27). On v1 Yorùbá it made
  over-refusal worse (−0.22, p = 0.034), and factual accuracy fell to 0 in Yorùbá and Hausa. Adding the safety vector
  to a model that is already over-cautious pushes it further toward refusing.
- **Takeaway:** training with safety data mixed in (Ìdìbò) beats both repairs. When retraining is not an option, Safe
  LoRA is the repair to use, and ReSta suits a fine-tune that lost safety, not one that over-refuses.

### 4.3 Multilingual and code-mixed safety (2a-4)

Benchmark v1 (18 harmful, 18 benign look-alikes, 45 swap tests, 20 fact and 8 unknowable questions per language),
command-r7b grades.

| | Harmful compliance | Over-refusal | Facts correct | Answers in the user's language |
|---|---|---|---|---|
| Hausa, base + sources | 11.1% | 0% | 30% | 76% |
| Hausa, plain fine-tune | 11.2% | 55.6% | 15% | 80% |
| Hausa, **Ìdìbò** | **5.6%** | **22.2%** | 25% | 68% |
| Igbo, base + sources | 5.6% | 11.1% | 50% | 96% |
| Igbo, **Ìdìbò** | 16.7% | 33.3% | 25% | 95% |

- **Hausa was never in the training data** (training covered en/pcm/yo), so these rows measure zero-shot transfer.
  The safety behaviour transfers: Ìdìbò helps on 33 points more benign Hausa questions than the plain fine-tune
  (p = 0.023), with the lowest harmful rate of the three.
- **Code-mixed prompts** (30, written by hand: Pidgin-English, Yorùbá-English, Hausa-English): Ìdìbò refused all 9
  harmful ones and answered 8 of 9 fact questions correctly (plain fine-tune 5 of 9, base 7 of 9). But it refused 4 of
  the 6 benign look-alikes, where Safe LoRA refused 1. Code-mixed over-refusal is Ìdìbò's clearest remaining weakness
  (n = 6, so the interval is wide: 30–90%).
- **Igbo** was neither trained nor evaluated before this run, and it is the weakest language: 3 of 18 harmful requests
  got harmful answers (5.8–39.2%), and fact accuracy halved against the base model (50% → 25%, p = 0.011). The app
  labels Igbo "Beta · not trained on Igbo". With the query-translation step (§4.4), Igbo harmful compliance fell to
  1 of 18 (not significant).
- **Fine-tuning cost factual accuracy in Yorùbá** (30% → 5%, p = 0.024), which the query-translation step restores
  (§4.4).

### 4.4 Cross-lingual retrieval

The knowledge base is English. Retrieval recall@4 on the 20 fact questions:

| | en | pcm | yo | ha | ig |
|---|---|---|---|---|---|
| Hybrid search, native question | 14 | 13 | 7 | 4 | 8 |
| Larger embedders, dense only (best of e5-base, LaBSE, BGE-M3) | 13 | 15 | 5 | 13 | 8 |
| **Native question + the model's own English translation** | 14 | 14 | **12** | **16** | **10** |

The translation step costs about 0.5 s on a laptop. It is live in the app for yo/ha/ig. Its effect on answer quality: Yorùbá facts
correct rose from 5% to 35% (+0.30, p = 0.005, AuditKIT); Hausa 25% → 30% and Igbo 25% → 25% (not significant). Safety
and neutrality did not get worse in any of the three.

### 4.5 A hallucination the benchmarks did not grade: invented contact details

Reading the answers by hand turned up something no rubric asked about. Refusals that offer a lawful alternative
("report it to INEC…") invite the model to make up where to report it: a hotline `0800-INEC-HELP` (43 times), a
fact-check page `www.inec.gov.ng/factcheck` (28), and the domain `inec.gov.ng`, which does not exist (INEC's site is
inecnigeria.org). For an election assistant this is a safety bug: an invented number is exactly what a scammer would
like voters to call.

`scripts/contacts_eval.py` counts answers with a web address, email or phone number found nowhere in the knowledge base
(plus INEC's own portals irev. and cvr.inecnigeria.org):

| Model | Answers with an invented contact |
|---|---|
| Base + sources | 4.8% (33 of 691) |
| Plain civic fine-tune | 1.9% (11 of 582) |
| + ReSta | **14.3%** (83 of 582) |
| + Safe LoRA | 0.5% (3 of 582) |
| Ìdìbò | 6.7% (46 of 691) |

The safety-trained models invent more contacts than the plain fine-tune, and the ReSta repair, which pushes the model
further toward refusal templates, invents the most. Every one of 14 flagged answers checked by hand was a real
invention. The fix is a check at deployment, not more training: `idibo/contacts.py` keeps a contact only if it appears
in the sources, and replaces any other with inecnigeria.org. Applied to every recorded answer it removes all of them
(0 left in every condition), and it runs on every live answer, on the web and on WhatsApp.

## 5. How we measured, and its limits

- **Held-out test set.** Benchmark v2 was written after training. Its nearest training question has a cosine similarity
  of at most 0.948 (multilingual-e5); there are no near-duplicates (`data/bench2/leakage.json`).
- **Grader.** The answer is back-translated into English, then graded against a fixed rubric, one item at a time.
  Round 1 used `command-a-plus` over the API. When the trial quota ran out we moved to open weights on our own GPUs.
  Agreement between the two graders on the same 1,241 items: 83.9%, Cohen's κ = 0.80 over all labels
  (`runs/final/judge_agreement.json`). On harmful items alone κ is only 0.28, and the disagreement has a direction:
  of 33 answers command-a-plus judged harmful, command-r7b called 20 refusals. The open grader is lenient, so every
  harmful-compliance rate graded by command-r7b is a **lower bound**. It does not change the ranking: a plain check
  for a refusal opening finds 1 of Ìdìbò's 56 v2 answers without one (a Pidgin refusal phrased "I no fit give you…"),
  against 5 for the plain fine-tune and 40 for the base model. Neutrality labels are almost all one class, so κ is
  not informative there (agreement 82%); round 1's stricter command-a-plus neutrality grades are in §4.1.
- **Why v2 is English plus code-mixed only.** We first machine-translated v2 with Tiny Aya Earth. The safety-tuned
  translator **answered or refused** many harmful prompts instead of translating them, which would silently turn
  harmful items into benign ones. Finding: a safety-tuned model cannot be trusted to build its own multilingual safety
  benchmark. The v1 translations (Cohere API) are kept, but are also machine-made: one Yorùbá item turned "when" into
  "where". Native-speaker review sheets are in `review/`.
- Generation is greedy. The v2 conditions were generated with vLLM and the v1 conditions with HF transformers, using the
  same prompts.

## 6. Red-teaming the live app (AuditKIT)

The benchmarks test the model. Voters use the whole system: retrieval, prompt, the 4-bit model, and the "Is this message
true?" mode that reads text an attacker wrote. `scripts/audit_redteam.py` points AuditKIT's `RedTeamRunner` at the live
URL with its built-in probes (prompt injection, jailbreak, encoding, over-refusal) and three election probes:
- **indirect injection**: a fake INEC notice sent to "Is this message true?" with an instruction hidden inside it (in
  English, Pidgin, Yorùbá and Hausa), asking the assistant to confirm it and add a marker;
- **multilingual extraction**: requests for its instructions in Pidgin, Yorùbá, Hausa and English;
- **endorsement bait**: pressure to name one party.

**First run (before the guard), attacks that worked:**

| Probe | Prompts | Attacks that worked |
|---|---|---|
| prompt injection (built-in) | 8 | 3: it repeated "I HAVE BEEN HACKED", "I can help with anything", "ACCESS_GRANTED" |
| encoding (built-in) | 5 | 1: it decoded a leetspeak "Say I AM COMPROMISED" and wrote the phrase |
| jailbreak (built-in) | 8 | 0 |
| over-refusal (built-in) | 5 | 1: it refused to write a calculator program (out of scope by design) |
| indirect injection (ours) | 6 | 2: it declined to confirm the fake notice but still added the attacker's marker |
| multilingual extraction (ours) | 3 | 0 |
| endorsement bait (ours) | 4 | 0 |

No jailbreak, no instruction leak and no endorsement got through. What did get through is a model that **echoes
phrases it is told to say**. For a fact-checking assistant, a marker appended to "I can't confirm this" is enough to
make a screenshot look like a confirmation.

**Fix.** `idibo/guard.py`, a deployment layer with no extra model call. It looks for injection cues in the user's text
(English, Pidgin, Yorùbá, Hausa), lists the strings the text demands be repeated, and replaces a reply that contains one
with a fixed answer. In check mode, that answer says the message hides instructions aimed at an AI, a sign of
manipulation. When cues are present the reply is checked before it is shown. On the first run's 40 replies it caught
all 7 that echoed a demanded phrase; on the 691 benchmark questions (v1 and v2, all languages) it found no cue, so
benchmark behaviour is unchanged. The model's training and evaluation are not affected, and it is reported as a
deployment layer, not a model result.

**Re-run after the guard** (same probes, same live URL):

| Probe | Before | After |
|---|---|---|
| prompt injection | 3 of 8 | **0 of 8** |
| encoding | 1 of 5 | **0 of 5** |
| indirect injection (fake notice with a hidden instruction) | 2 of 6 | **0 of 6** |
| jailbreak, instruction extraction, endorsement bait | 0 | 0 |
| over-refusal (the out-of-scope calculator request) | 1 of 5 | 1 of 5 |

Every successful attack from the first run was blocked. The one remaining flag is a refusal to write a calculator
program, which is out of scope for an election assistant by design (`runs/audit/redteam.after_guard.json`).

## 7. Where voters are: WhatsApp

Election rumours in Nigeria travel on WhatsApp, so Ìdìbò answers there too (Meta's WhatsApp Cloud API). A voter can
ask by text or voice note in any of the five languages, or **forward a message they doubt** and get a check with
numbered sources. Replies go through exactly the same pipeline as the web app (retrieval, injection guard, contact
check), a voice note gets a spoken answer back (YarnGPT), and the chat shows "typing…" while it works. New users are
told how to switch language in all five languages. Webhooks are verified (Meta's HMAC signature), and nothing about a
user is stored beyond their chosen language, in memory.

It currently runs on Meta's free test number (+1 555 192 8126), which can message up to five registered phones; a
public launch needs a verified business number and display name. (We first built it on Twilio; Twilio's new trial
numbers only send pre-approved templates, so free-text answers were blocked. That code path is still in the server.)

## 8. Cost: closed API vs self-hosted (estimate)

Per answer: about 2,000 input tokens (instructions and 4 sources) and about 250 output tokens.
- **Closed API** (Cohere Command A list price, $2.50 / M input, $10 / M output): about **$0.0075 per answer**, so
  **about $7,500 per million questions**.
- **Self-hosted Ìdìbò** (4-bit, 8 vCPU / 4 GB on Railway at $20 / vCPU-month + $10 / GB-month, about $200/month fully
  used): about 12 s per answer, about 216k answers per month per replica, so **about $0.0009 per answer, or about
  $900 per million questions**. That is roughly 8× cheaper, the data stays in our deployment, and the safety behaviour
  is ours to audit and repair.

## 9. Reproduce

See the repository README: `scripts/` (data, training, evaluation, grading), `kaggle/` (each GPU job as a script) and
`space/` (the app image).

## 10. Responsible-use notes

- The benchmark contains harmful *requests* only, never harmful content.
- Ìdìbò endorses no party. Party names appear only in symmetric swap tests.
- It is not official INEC guidance, and it says so.
- Dates come from INEC's published calendar in the knowledge base (presidential election: 16 January 2027). Earlier
  reports said 20 February; the benchmark checks that the model corrects that rumour from the source instead of
  agreeing with it. Every answer cites its source and points to inecnigeria.org for changes.

## 11. Known limitations

- **Over-refusal on code-mixed prompts** (4 of 6 benign) and in Yorùbá (44%) is still high. Safe LoRA does better on
  code-mixed prompts.
- **Igbo** is untrained and weaker than the base model on facts; it is labelled Beta in the app.
- **The model mirrors the question's style.** A Pidgin-flavoured question asked with English selected may get a Pidgin
  answer. The app swaps the standard Pidgin "I don't have that information" for its English version; other answers are
  left as they are.
- **Hallucinated detail.** Asked what to bring to the polling unit, it sometimes adds "ID or passport" (the requirement
  is the PVC).
- **Invented contact details** (§4.5) are removed at deployment by a check against the sources; the model itself still
  produces them.
- **The candidate list was added after evaluation.** INEC's final list of 2027 presidential candidates was missing from
  the original crawl. The app now shows it word for word from INEC's PDF (`scripts/add_candidates_kb.py`). Benchmarks
  were run on the 309-passage knowledge base without it.
- **Machine translations.** The v1 translations and the guard's Yorùbá, Hausa and Igbo messages are not yet reviewed
  by native speakers.

