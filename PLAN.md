# Ìdìbò: a safe election assistant for Nigeria's 2027 vote (Lexsi AI Safety & Alignment Hackathon)

Working title. Ìdìbò means "election" in Yoruba.
Deadline: Nov 15 23:59 AoE = **Mon Nov 16, 12:59 WAT**. Finalist video or slides are due Nov 22.

## The problem
Nigeria votes in early 2027. The date is reported as 16 Jan 2027 after the Electoral Act 2026; earlier reports said 20 Feb, so we confirm it with INEC.
- 66.5% of Nigerians fear AI-driven election misinformation (SBM Intelligence, 2026).
- 52% get political news mainly from WhatsApp and social media.
- Election rumours often break in Hausa, Yoruba or Igbo before English.
- Open models that speak these languages now exist (Tiny Aya). They can answer voters' questions, and they can also be asked to produce incitement or voter-suppression messages.

**Gap (searched 2026-11-11):** no election-safety benchmark or aligned model exists for Hausa, Yoruba, Igbo or Pidgin. General work exists in English: Polistemics, and HalluBias-Election (IJCAI 2026).

## What we build
A voter-information assistant on `CohereLabs/tiny-aya-earth` that answers in en / pcm / yo / ha / ig. It is trained to do 5 things:
1. **Stay grounded.** It answers procedural questions (PVC, BVAS, IReV, polling units, timetable, voter rights) from a dated, cited set of official INEC and Electoral Act documents.
2. **Abstain or flag conflicts** when it doesn't know or the sources disagree, instead of making things up. The election date is the demo case.
3. **Refuse to produce harmful election content:** ethnic or religious incitement, voter-suppression disinformation ("tell Igbo voters the election moved to Sunday"), vote-buying scripts, impersonating INEC.
4. **Stay neutral.** It won't tell anyone who to vote for, and it treats every party, candidate and ethnic group the same way.
5. **Not over-refuse.** It still explains the law, helps people report fake messages, summarises what manifestos say, and explains how voting works.

## Pipeline (5 Lexsi repos)
1. **CuratorKIT** builds the data:
   - **knowledge base:** INEC and Electoral Act text with date and source attached;
   - **SFT set:** grounded Q&A in 5 languages. Aya Expanse 32B is a Cohere model, so it can draft and translate, and a native speaker spot-checks the result;
   - **safety set:** harmful requests (requests only; we never write the harmful content) paired with harmless look-alikes;
   - **neutrality pairs:** the same prompt with the party, candidate or ethnic group swapped.
2. **AlignTune** trains the model: LoRA SFT, then DPO with preferred = grounded, safe and neutral answers and rejected = hallucinated, harmful or partisan ones.
3. **SafeTune** hardens it and restores safety after fine-tuning. It includes one research experiment carried over from "LIC": calibrating the repair in English versus in the target language.
4. **AgentTune** (optional, if time allows) trains a retrieval tool call (`search_inec_docs`) so answers cite their sources.
5. **AuditKIT** runs every evaluation, reported per language.

## Metrics: base model vs SFT vs SFT+DPO vs +SafeTune, per language
- Harmful-request compliance (lower is better)
- Over-refusal on harmless look-alikes (lower is better)
- Grounded accuracy on answerable questions (higher is better)
- Hallucination rate on questions that can't be answered or where sources conflict (lower is better); abstention is the correct answer
- Neutrality asymmetry: the gap in refusal and tone across swapped pairs (lower is better)
- General ability: a small Global-MMLU slice (no big drop)
- ROI: self-hosting a 3.35B model vs paying per token for a closed API at election-scale traffic

## Deliverables
- HF model collection (adapters and merged model) and HF dataset collection (KB, SFT set, safety and neutrality benchmark, eval logs)
- GitHub repo with reproducibility notes, HF article write-up, abstract under 500 words
- A Gradio Space: ask in any of the 5 languages, see a cited answer, and compare the base model with Ìdìbò on attack prompts
- Optional: a WhatsApp front end (the Handova bot code shows how)

## Safety of the project itself
- The safety prompts are requests, never finished harmful content. Rejected DPO answers are kept short and non-operational.
- We don't endorse any candidate or party. The neutrality tests use real party names only in swapped pairs.
- We never claim the model is "certified" or official INEC guidance.

## Schedule (WAT)
- **Wed 11 eve:** environment setup, KB scrape, data schema, baseline evals of the base model
- **Thu 12:** generate SFT, safety and neutrality data; user spot-checks Yoruba and Pidgin; SFT run
- **Fri 13:** DPO, SafeTune, full eval grid
- **Sat 14:** Space, write-up, HF collections
- **Sun 15:** buffer and polish; the user submits the form

## Compute
- M1 Pro 16 GB for development and inference checks.
- Training and the eval grid run on free Kaggle T4s or the $20 of HF credits.
