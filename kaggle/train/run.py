# Ìdìbò: LoRA fine-tunes of Tiny Aya Earth with AlignTune's RaftTrainer, then evaluation, on Kaggle (2x T4).
# Inputs: dataset crypticdev/idibo-data (code + data/train/*.jsonl) and the output of crypticdev/idibo-base-eval
# (which holds a copy of the gated model, so this notebook needs no HF token and can be started from the CLI).
#
# GPU 0 trains "civic" then "idibo"; GPU 1 meanwhile evaluates the base model in the other languages.
import glob, json, os, shutil, subprocess, sys, tarfile, time


def sh(cmd, env=None):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, env=env)


sh("pip install -q -U 'transformers>=4.51.3' accelerate sentence-transformers peft 'trl>=1.0' datasets")
# --no-deps: AlignTune pins torch<2.14, and letting pip downgrade Kaggle's torch would cost a big reinstall.
sh("pip install -q --no-deps aligntune && pip install -q typer rich pyyaml psutil")
import importlib
importlib.import_module("aligntune.backends.trl.raft.raft_trainer")  # fail fast if an import is missing

W = "/kaggle/working/idibo"
tars = glob.glob("/kaggle/input/**/idibo.tar.gz", recursive=True)
if tars:
    os.makedirs(W, exist_ok=True)
    tarfile.open(tars[0]).extractall(W)
else:
    found = glob.glob("/kaggle/input/**/scripts/run_eval.py", recursive=True)
    shutil.copytree(os.path.dirname(os.path.dirname(found[0])), W, dirs_exist_ok=True)
os.chdir(W)
MODEL = glob.glob("/kaggle/input/**/model/tiny-aya-earth/config.json", recursive=True)
assert MODEL, "attach the output of crypticdev/idibo-base-eval (it holds the model copy)"
MODEL = os.path.dirname(MODEL[0])
print("model:", MODEL, flush=True)

VARIANTS = os.environ.get("VARIANTS", "civic,idibo").split(",")
EVAL_LANGS = os.environ.get("EVAL_LANGS", "en,pcm,yo,ha")
OUT = "/kaggle/working"

# GPU 1: base model in the non-English languages, both conditions, while GPU 0 trains.
env1 = dict(os.environ, CUDA_VISIBLE_DEVICES="1")
base_langs = ",".join(l for l in EVAL_LANGS.split(",") if l != "en")
bg = subprocess.Popen(
    f"python scripts/run_eval.py --model {MODEL} --name base-plain --policy plain --langs {base_langs} "
    f"--out {OUT}/runs/base-plain && python scripts/run_eval.py --model {MODEL} --name base-idibo-rag --policy idibo "
    f"--rag --langs {base_langs} --out {OUT}/runs/base-idibo-rag",
    shell=True, env=env1)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig
from aligntune.backends.trl.raft.raft_trainer import RaftTrainer, RaftTrainerConfig

for variant in VARIANTS:
    t0 = time.time()
    rows = [json.loads(l) for l in open(f"data/train/{variant}.jsonl", encoding="utf-8")]
    ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])
    print(f"== {variant}: {len(ds)} examples", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    args = SFTConfig(
        output_dir=f"{OUT}/ckpt/{variant}", num_train_epochs=2, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=10, save_strategy="no", fp16=True, max_length=2048, completion_only_loss=True,
        report_to="none", seed=42, gradient_checkpointing=True,
    )
    trainer = RaftTrainer(model=model, processing_class=tok, args=args, train_dataset=ds,
                          raft_config=RaftTrainerConfig(max_golden_docs=1, max_distractor_docs=3))
    trainer.train()
    adir = f"{OUT}/adapters/{variant}"
    model.save_pretrained(adir)
    tok.save_pretrained(adir)
    json.dump(trainer.state.log_history, open(f"{adir}/log_history.json", "w"), indent=1)
    print(f"== {variant} trained in {(time.time() - t0) / 60:.1f} min", flush=True)
    del trainer, model
    torch.cuda.empty_cache()
    sh(f"python scripts/run_eval.py --model {MODEL} --adapter {adir} --name {variant}-rag --policy idibo --rag "
       f"--langs {EVAL_LANGS} --out {OUT}/runs/{variant}-rag")

bg.wait()
print("all done", flush=True)
