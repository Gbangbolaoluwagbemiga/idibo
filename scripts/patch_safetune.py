"""Repair the plain "civic" fine-tune WITHOUT retraining, with SafeTune (Track 2b, problem 6).

    python scripts/patch_safetune.py --method resta    --aligned <tiny-aya-earth> --base <tiny-aya-base> \
        --adapter adapters/civic --out /kaggle/tmp/civic-resta
    python scripts/patch_safetune.py --method safelora --aligned <tiny-aya-earth> --base <tiny-aya-base> \
        --adapter adapters/civic --out adapters/civic-safelora

Both methods use the safety direction  v = theta_aligned (Tiny Aya Earth, instruction/safety tuned)
                                       - theta_base    (Tiny Aya Base, pretrained only):
  - ReSta (ReStaTrainer): theta_civic + alpha * v, with DARE on the fine-tuning delta. alpha = 0.25, the value
    SafeTune documents for Tiny Aya (alpha >= 0.5 breaks it). Output: a full merged model.
  - Safe LoRA (SafeLoRATrainer): projects each LoRA update onto the safety subspace of v where the update and its
    projection disagree (cosine <= threshold). Output: a repaired LoRA adapter.
No gradient step is taken; nothing is trained.
"""
import argparse
import json
import time

import torch
from peft import PeftModel
from safetune.runner import recover
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["resta", "safelora"], required=True)
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--num-proj", type=int, default=25,
                    help="Safe LoRA fallback: if no module falls under the threshold, project the N least-aligned ones")
    a = ap.parse_args()
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_dtype = torch.float16 if dev == "cuda" else torch.float32
    try:
        tok = AutoTokenizer.from_pretrained(a.aligned)
    except Exception:  # only for the tiny dry-run model, which has no tokenizer
        tok = None
    # Reference models stay on CPU in fp16 (3.35B each): the safety vector is formed one tensor at a time.
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float16)
    aligned = AutoModelForCausalLM.from_pretrained(a.aligned, dtype=torch.float16)
    ft = AutoModelForCausalLM.from_pretrained(a.aligned, dtype=gpu_dtype).to(dev)
    ft = PeftModel.from_pretrained(ft, a.adapter)
    info = {"method": a.method, "adapter": a.adapter}
    if a.method == "resta":
        ft = ft.merge_and_unload()  # theta_civic as full weights
        recover.ReStaTrainer(ft, base_model=base, aligned_model=aligned, alpha=a.alpha, device=dev).apply()
        ft.save_pretrained(a.out, safe_serialization=True)
        if tok:
            tok.save_pretrained(a.out)
        info.update(alpha=a.alpha, dare=True, output="full model")
    else:
        before = {n: p.detach().clone() for n, p in ft.named_parameters() if "lora_B" in n}
        recover.SafeLoRATrainer(ft, base_model=base, aligned_model=aligned, threshold=a.threshold).apply()
        changed = sum(not torch.equal(before[n], p.detach()) for n, p in ft.named_parameters() if n in before)
        mode = f"threshold {a.threshold}"
        if changed == 0:  # nothing fell under the threshold: use the paper's fixed-number selection instead
            from safetune.interventions.recover.safe_lora import apply_safe_lora

            apply_safe_lora(ft, aligned_state_dict=aligned.state_dict(), base_state_dict=base.state_dict(),
                            select_layers_type="number", num_proj_layers=a.num_proj)
            changed = sum(not torch.equal(before[n], p.detach()) for n, p in ft.named_parameters() if n in before)
            mode = f"number {a.num_proj} (threshold {a.threshold} selected none)"
        ft.save_pretrained(a.out)
        if tok:
            tok.save_pretrained(a.out)
        info.update(selection=mode, lora_modules=len(before), lora_modules_projected=changed, output="adapter")
    info["seconds"] = round(time.time() - t0)
    json.dump(info, open(f"{a.out}/safetune_patch.json", "w"), indent=1)
    print("== patched:", info, flush=True)


if __name__ == "__main__":
    main()
