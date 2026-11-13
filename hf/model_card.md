---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-earth
library_name: peft
language: [en, pcm, yo, ha, ig]
tags: [safety, elections, nigeria, rag, lora, gguf, aligntune, safetune, lexsi-hackathon]
datasets: [GOPAE/idibo-eval]
pipeline_tag: text-generation
---

# Ìdìbò: Tiny Aya Earth for safe voter information (Nigeria 2027)

A LoRA fine-tune of [`CohereLabs/tiny-aya-earth`](https://huggingface.co/CohereLabs/tiny-aya-earth) (3.35B) that answers
voters' questions in English, Nigerian Pidgin, Yorùbá and Hausa (Igbo: beta, untrained). It answers from retrieved
INEC and Electoral Act 2026 passages, with citations. It refuses election harm, stays politically neutral, abstains
when it doesn't know, and still helps with legitimate civic questions.

**Try it:** https://idibo-2027.up.railway.app · **Code:** ⟨GitHub URL⟩ · **Write-up:** ⟨link⟩

## Files

| File | What |
|---|---|
| `adapters/idibo/` | **Ìdìbò**: LoRA r16/α32 on all projections; 1,060 RAFT-style examples (civic Q&A, refusals with a lawful alternative, neutral answers, abstentions) in en/pcm/yo |
| `adapters/civic/` | Control: the same recipe on civic Q&A only (566 examples), the "plain fine-tune" |
| `adapters/civic-safelora/` | The civic adapter repaired with **SafeTune Safe LoRA**: no training; 74 of 252 LoRA modules projected |
| `idibo-q4_k_m.gguf` | Ìdìbò merged and quantised (Q4_K_M, 2.1 GB) for llama.cpp: about 10–14 s per answer on 8 vCPUs |

Training: **AlignTune** `RaftTrainer`, 2 epochs, lr 1e-4 (cosine), fp16, completion-only loss, each prompt holding the
golden passage plus 3 retrieved distractors. Data curated with **CuratorKIT**. Trained on Kaggle (one T4, 3.1 h).

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-earth")
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-earth"),
                                  "GOPAE/idibo-tiny-aya-earth", subfolder="adapters/idibo")
```

Use it with the system prompt and retrieval from the repository (`scripts/run_eval.py`, `idibo/retrieve.py`). It was
trained to answer from `SOURCES:` passages.

## Results (summary; full tables with 95% intervals are in the write-up)

⟨headline table filled from the final grading⟩

Round 1, Pidgin, graded by command-a-plus:

| | Base + sources | Plain fine-tune | Ìdìbò |
|---|---|---|---|
| Harmful compliance | 55.5% | 5.6% | **0%** |
| Over-refusal | 11.1% | 61.1% | **16.7%** |
| Took a political side | 24.4% | 6.7% | **0%** |
| Answers in Pidgin | 12.8% | 100% | **100%** |

## Limitations

- Yorùbá and Igbo factual accuracy is limited by cross-lingual retrieval. The app translates the question to English
  before searching, which raised recall@4 from 7→12/20 (yo) and 4→16/20 (ha).
- Translations in training and evaluation are machine-made; native-speaker review was partial.
- The benchmarks are small (v2: 56 harmful, 48 benign, 30 code-mixed). Read the intervals, not just the point estimates.
- Not official INEC guidance. Election details change: always check inecnigeria.org.
- Non-commercial licence, inherited from Tiny Aya.
