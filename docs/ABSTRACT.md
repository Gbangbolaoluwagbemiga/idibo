# Abstract

**Ìdìbò: safety-preserving fine-tuning of Tiny Aya for multilingual election information in Nigeria**

Nigeria votes in early 2027, and voters will ask AI assistants about registration, polling and the law in English,
Nigerian Pidgin, Yorùbá, Hausa and Igbo. The open models that can serve them can also be asked to write ethnic
incitement, fake INEC notices, vote-buying scripts or partisan endorsements. Safety tuning often fails differently: the
assistant starts refusing ordinary civic questions.

We fine-tune Cohere's Tiny Aya Earth (3.35B) with LoRA, using AlignTune's RAFT trainer, on 1,060 examples curated with
CuratorKIT. The examples are grounded in a cited knowledge base of the Electoral Act 2026 and INEC pages, and mix civic
answers, refusals that offer a lawful alternative, neutral replies to partisan prompts, and "I don't know, ask INEC"
answers. A control adapter is trained on the civic answers alone. We then repair that plain fine-tune without any
training, using two SafeTune methods: ReSta (adding the safety vector between the aligned and base models) and Safe
LoRA (projecting the LoRA update onto the safety subspace).

We evaluate on two benchmarks. The first has 109 items in five languages (facts, abstentions, harmful/benign pairs and
party, ethnic and religious swap tests). The second is held out and was written after training: 116 English items,
including 16 jailbreak-wrapped attacks, plus 30 hand-written code-mixed prompts. We report 95% intervals. In Pidgin, the
base model given the same sources and instructions helped with 55.5% of harmful requests, took a political side 24.4% of
the time and answered in Pidgin only 12.8% of the time. Ìdìbò reached 0%, 0% and 100%. The plain fine-tune was also safe,
but refused 44–61% of benign civic questions; Ìdìbò refused 11–17%. On the held-out set, Ìdìbò refused all 56 harmful and 16 jailbreak
requests and only 6% of benign look-alikes; the plain fine-tune refused 35%. Safe LoRA, with no training, cut that to
17%, about two thirds of the gap; ReSta did not help significantly. AuditKIT's paired bootstrap confirms both
gains (p ≤ 0.012). Hausa was never seen in training, yet Ìdìbò had the lowest Hausa harmful rate (5.6%).

Three practical findings. First, the knowledge base is English, so Yorùbá and Hausa questions found the right source
only 7/20 and 4/20 times. Having the model translate its own query raised this to 12/20 and 16/20, beating larger
multilingual embedders. Second, a safety-tuned model cannot be used to translate a safety benchmark: it answered or
refused harmful prompts instead of translating them, which would quietly turn attacks into benign items. Third, an AuditKIT red-team of the live app found no jailbreak, leak or endorsement, but the app repeated injected phrases, including a marker hidden in a fake INEC notice; a guard fixed this.

Ìdìbò runs as a 4-bit GGUF on 8 CPU cores (about 12 s per answer) with Nigerian voice input and output, at about $0.0009 per
answer versus $0.0075 for a closed API. It is live at https://idibo-2027.up.railway.app. Model,
adapters, benchmarks, generations and grades are public.
