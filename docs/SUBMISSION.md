# Submission form: what goes in each field

| Field | Value |
|---|---|
| Project | Ìdìbò: a safe, multilingual voter-information assistant for Nigeria's 2027 election |
| Registrant email | ⟨your registered email⟩ |
| Abstract | docs/ABSTRACT.md (under 500 words) |
| HF model collection | https://huggingface.co/collections/GOPAE/idibo-models-nigeria-2027-election-safety-6af85462c9b6f21539aaa65b : `GOPAE/idibo-tiny-aya-earth` (adapters `idibo`, `civic`, `civic-safelora`; GGUF) |
| HF dataset collection | https://huggingface.co/collections/GOPAE/idibo-data-nigeria-2027-election-safety-6af854634cce9ce0f613d91d : `GOPAE/idibo-eval` (benchmarks v1 and v2, KB, training data, generations, grades) |
| Write-up | https://github.com/Gbangbolaoluwagbemiga/idibo/blob/main/docs/WRITEUP.md (results page: https://idibo-2027.up.railway.app/results.html) |
| GitHub repo (reproducibility) | https://github.com/Gbangbolaoluwagbemiga/idibo |
| Public deployment | https://idibo-2027.up.railway.app (web, voice, 5 languages) · WhatsApp: Meta test number +1 555 192 8126 (registered phones; shown in the demo video) |

**Tracks and problem statements**
- Primary: **Track 2b, problem 5: safety-preserving fine-tuning** (a domain task with safety hardening, compared with a plain fine-tune)
- **Track 2b, problem 6**: repairing safety drift without retraining (SafeTune ReSta and Safe LoRA on the plain fine-tune)
- **Track 2a, problem 4**: multilingual and code-mixed safety (en, pcm, yo, ha, ig and code-mixed prompts; Hausa as an unseen language)
- **Track 2a, problem 2**: a safety benchmark for a domain and its languages (election safety, Nigerian languages)
- Secondary: **Track 1, problem 4**: low-resource languages for a public service

**Lexsi repositories used**
- **CuratorKIT**: curation of the SFT data (schema gate, exact and MinHash dedup, cleaning, provenance): `scripts/curate.py`
- **AlignTune**: `RaftTrainer` LoRA fine-tuning of both adapters: `scripts/train_lora.py`
- **SafeTune**: `ReStaTrainer` / `apply_resta` and `SafeLoRATrainer` repairs: `scripts/patch_safetune.py`
- **AuditKIT**: `RedTeamRunner` red-team of the live app with three custom election probes (`scripts/audit_redteam.py`),
  and `ak.compare(...).significance()` paired bootstrap for every before/after claim (`scripts/audit_compare.py`)

**Models (all Cohere)**
- Fine-tuned: `CohereLabs/tiny-aya-earth`; ReSta reference: `CohereLabs/tiny-aya-base`
- Supporting roles: `command-a-plus` (data generation, round-1 grading), `command-a-translate` and `tiny-aya-global` (translation and back-translation), `c4ai-command-r7b-12-2024` (open-weight grader)
