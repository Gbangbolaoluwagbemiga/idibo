"""Train one Ìdìbò LoRA adapter on Tiny Aya Earth with AlignTune's RaftTrainer.

    python scripts/train_lora.py --model <path> --data data/train/idibo.jsonl --out adapters/idibo

Runs in its own process so the GPU memory is fully released when it exits (evaluation follows in another process).
RAFT-style data: each prompt holds the golden source chunk plus retrieved distractors (scripts/build_train.py);
loss is on the completion only.
"""
import argparse
import json
import time

import torch
from aligntune.backends.trl.raft.raft_trainer import RaftTrainer, RaftTrainerConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=-1)
    a = ap.parse_args()

    t0 = time.time()
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])
    print(f"== {a.data}: {len(ds)} examples", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16 if cuda else torch.float32)
    if cuda:
        model = model.to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    args = SFTConfig(
        output_dir=f"{a.out}-ckpt", num_train_epochs=a.epochs, max_steps=a.max_steps, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=0.05,
        logging_steps=10, save_strategy="no", fp16=cuda, max_length=2048, completion_only_loss=True,
        report_to="none", seed=42, gradient_checkpointing=True, use_cpu=not cuda,
    )
    trainer = RaftTrainer(model=model, processing_class=tok, args=args, train_dataset=ds,
                          raft_config=RaftTrainerConfig(max_golden_docs=1, max_distractor_docs=3))
    trainer.train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    json.dump(trainer.state.log_history, open(f"{a.out}/log_history.json", "w"), indent=1)
    print(f"== trained {a.out} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
