"""Same as run_eval.py (same prompts, retrieval, greedy decoding, output format), but on vLLM.

Every item of every language is submitted at once and vLLM batches them continuously, which is far faster
than padded HF batches on a T4 (run_eval.py averaged ~20 s per answer on Kaggle).

    python scripts/run_eval_vllm.py --model <dir> [--adapter <lora>] --name idibo-rag --rag --bench data/bench2 \
        --langs en,mix,pcm,yo,ha --out runs2/idibo-rag
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_eval import SYSTEM, SYSTEM_PLAIN  # noqa: E402  (same policy text as the HF runner)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--name", required=True)
    ap.add_argument("--langs", default="en")
    ap.add_argument("--rag", action="store_true")
    ap.add_argument("--policy", choices=["idibo", "plain"], default="idibo")
    ap.add_argument("--bench", default="data/bench")
    ap.add_argument("--pivot", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs (2 for 8B models on T4s)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from idibo.pivot import PIVOT_LANGS, clean_pivot, pivot_messages, search_query

    t0 = time.time()
    llm = LLM(model=a.model, dtype="half", max_model_len=4096, gpu_memory_utilization=0.85, enable_lora=bool(a.adapter),
              max_lora_rank=16, enforce_eager=False, seed=0, tensor_parallel_size=a.tp)
    lora = LoRARequest("adapter", 1, a.adapter) if a.adapter else None
    tok = llm.get_tokenizer()

    def chat(msg_lists, max_tokens):
        prompts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msg_lists]
        outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=max_tokens), lora_request=lora)
        return [o.outputs[0].text.strip() for o in outs]

    kb = None
    if a.rag:
        from idibo.retrieve import HybridKB, format_context

        kb = HybridKB(path=ROOT / "data/kb/chunks.jsonl", emb=ROOT / "data/kb/e5_small.npy", device="cpu")
    system = SYSTEM if a.policy == "idibo" else SYSTEM_PLAIN
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for lang in a.langs.split(","):
        path = out_dir / f"{lang}.jsonl"
        bench = [json.loads(l) for l in open(ROOT / a.bench / f"{lang}.jsonl", encoding="utf-8")]
        done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")} if path.exists() else set()
        todo = [it for it in bench if it["id"] not in done]
        if not todo:
            continue
        english = [""] * len(todo)
        if kb and a.pivot and lang in PIVOT_LANGS:
            english = [clean_pivot(x) for x in chat([pivot_messages(it["prompt"], lang) for it in todo], 60)]
        msgs, ctxs = [], []
        for it, en in zip(todo, english):
            user, ctx_ids = it["prompt"], []
            if kb:
                hits = kb.search(search_query(it["prompt"], en), k=4)
                ctx_ids = [h["id"] for h in hits]
                user = f"SOURCES:\n{format_context(hits)}\n\nQUESTION:\n{it['prompt']}"
            msgs.append([{"role": "system", "content": system}, {"role": "user", "content": user}])
            ctxs.append(ctx_ids)
        t = time.time()
        resps = chat(msgs, a.max_new_tokens)
        secs = round((time.time() - t) / len(todo), 2)
        with open(path, "a", encoding="utf-8") as f:
            for it, resp, ctx, en in zip(todo, resps, ctxs, english):
                rec = {k: it.get(k) for k in ("id", "kind", "category", "pair", "swap")}
                rec.update(lang=lang, prompt=it["prompt"], response=resp, context_ids=ctx, model=a.model,
                           adapter=a.adapter, rag=a.rag, policy=a.policy, seconds=secs, bench=a.bench,
                           pivot_en=en or None, engine="vllm")
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"== {a.name}/{lang}: {len(todo)} answers, {secs}s each (amortised)", flush=True)
    print(f"== {a.name} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
