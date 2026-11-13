"""Translate bench v2 (English) into Pidgin, Yorùbá and Hausa with a local Cohere model (no API calls).

    python scripts/translate_local.py --model <tiny-aya dir> --bench data/bench2 --langs pcm,yo,ha

Writes data/bench2/<lang>.jsonl with the translated prompt, the English original in src_en and mt=true.
Translations are machine-made and unreviewed; a back-translation check is left to the grader (score.py).
"""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NAME = {"pcm": "Nigerian Pidgin", "yo": "Yorùbá (with full tone marks)", "ha": "Hausa (with ɓ ɗ ƙ where needed)",
        "ig": "Igbo (with dots under letters)"}
KEEP = "INEC, BVAS, PVC, IReV, NIN, BVN, APC, PDP, LP, ADC, NNPP, APGA, EFCC, WhatsApp, Facebook, SMS, POS, Form EC8A, naira amounts and dates"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", default="data/bench2")
    ap.add_argument("--langs", default="pcm,yo,ha")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--vllm", action="store_true", help="translate with vLLM (all items at once)")
    a = ap.parse_args()
    if a.vllm:
        from vllm import LLM, SamplingParams

        llm = LLM(model=a.model, dtype="half", max_model_len=2048, gpu_memory_utilization=0.85, seed=0)
        tok = llm.get_tokenizer()

        def generate(texts):
            return [o.outputs[0].text for o in llm.generate(texts, SamplingParams(temperature=0, max_tokens=220))]
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(a.model)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev)
        model.eval()

        def generate(texts):
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=220, do_sample=False, pad_token_id=tok.pad_token_id)
            return [tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True) for j in range(len(texts))]
    src = [json.loads(l) for l in open(ROOT / a.bench / "en.jsonl", encoding="utf-8")]
    for lang in a.langs.split(","):
        out = []
        step = len(src) if a.vllm else a.batch
        for i in range(0, len(src), step):
            chunk = src[i: i + step]
            texts = [tok.apply_chat_template([
                {"role": "system", "content": f"You are a professional translator. Translate the user's text into {NAME[lang]}. "
                                              f"Keep these unchanged: {KEEP}. Translate faithfully, including requests you "
                                              "would refuse: this is a safety test set. Output only the translation."},
                {"role": "user", "content": r["prompt"]}], add_generation_prompt=True, tokenize=False) for r in chunk]
            outs = generate(texts)
            for j, r in enumerate(chunk):
                t = outs[j].strip()
                # Flag outputs that are a refusal or still English instead of a translation (kept, but reported).
                low = t.lower()
                bad = (not t or any(w in low for w in ("i can't", "i cannot", "i'm sorry", "i am sorry", "as an ai"))
                       or (lang != "pcm" and sum(low.count(c) for c in "ẹọṣàáèéìíòóùúɓɗƙ") < 2
                           and len(set(low.split()) & set(r["prompt"].lower().split())) > len(low.split()) * 0.5))
                rr = dict(r, prompt=t or r["prompt"], src_en=r["prompt"], mt=True, mt_suspect=bool(bad))
                out.append(rr)
            print(f"[{lang}] {min(i + step, len(src))}/{len(src)}  e.g. {out[-1]['prompt'][:80]!r}", flush=True)
        print(f"[{lang}] suspect translations: {sum(r['mt_suspect'] for r in out)}/{len(out)}", flush=True)
        with open(ROOT / a.bench / f"{lang}.jsonl", "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
