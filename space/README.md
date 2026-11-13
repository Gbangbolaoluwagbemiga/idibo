---
title: Ìdìbò
emoji: 🗳️
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
license: cc-by-nc-4.0
short_description: Safe voter information for Nigeria's 2027 election, by voice
models:
  - GOPAE/idibo-tiny-aya-earth
  - CohereLabs/tiny-aya-earth
---

# Ìdìbò

A voice-first, picture-first voter-information assistant for Nigeria's 2027 general election, in English, Naijá
(Pidgin), Yorùbá, Hausa and Igbo. Answers come only from INEC's website and the Electoral Act 2026, with the source
one tap away. It says "I don't know" instead of guessing, refuses incitement, vote buying, fake results and voter
suppression, and never tells anyone who to vote for.

The model is [GOPAE/idibo-tiny-aya-earth](https://huggingface.co/GOPAE/idibo-tiny-aya-earth): CohereLabs/tiny-aya-earth
fine-tuned with LoRA (Lexsi AlignTune's RaftTrainer), served here as a 4-bit GGUF on a free CPU, so an answer takes
20-40 seconds and streams word by word.

**Not an INEC app.** Unofficial civic tool built for the Lexsi AI Safety hackathon.
