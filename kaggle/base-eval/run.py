# Ìdìbò: baseline evaluation of CohereLabs/tiny-aya-earth on Kaggle (GPU).
# Needs: dataset crypticdev/idibo-data attached, internet on, and a Kaggle Secret named HF_TOKEN.
import os, subprocess, sys, tarfile, glob, time

def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

sh("pip install -q -U 'transformers>=4.51.3' accelerate sentence-transformers peft")

from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

# Kaggle unpacks uploaded archives, so accept either the .tar.gz or the unpacked folders.
import shutil
W = "/kaggle/working/idibo"
tars = glob.glob("/kaggle/input/**/idibo.tar.gz", recursive=True)
if tars:
    os.makedirs(W, exist_ok=True)
    tarfile.open(tars[0]).extractall(W)
else:
    found = glob.glob("/kaggle/input/**/scripts/run_eval.py", recursive=True)
    print("inputs:", glob.glob("/kaggle/input/*/*"), flush=True)
    root = os.path.dirname(os.path.dirname(found[0]))
    shutil.copytree(root, W, dirs_exist_ok=True)
os.chdir(W)
print("project files:", sorted(os.listdir(W)), flush=True)

# Save a copy of the gated model in this run's output, so later notebooks (started from the CLI, where
# Kaggle doesn't pass secrets) can attach this output as input and never need the HF token.
from huggingface_hub import snapshot_download
MODEL_DIR = "/kaggle/working/model/tiny-aya-earth"
snapshot_download("CohereLabs/tiny-aya-earth", local_dir=MODEL_DIR, token=os.environ["HF_TOKEN"],
                  allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
print("model saved:", sorted(os.listdir(MODEL_DIR)), flush=True)
# The pretrained base too: SafeTune's ReSta repair needs theta_aligned (earth) - theta_base (tiny-aya-base).
try:
    snapshot_download("CohereLabs/tiny-aya-base", local_dir="/kaggle/working/model/tiny-aya-base",
                      token=os.environ["HF_TOKEN"], allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
    print("base saved:", sorted(os.listdir("/kaggle/working/model/tiny-aya-base")), flush=True)
except Exception as e:  # licence not accepted yet: carry on with the evaluation
    print("could not fetch tiny-aya-base:", str(e)[:200], flush=True)

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", flush=True)

LANGS = os.environ.get("LANGS", "en")
t = time.time()
# 1) the model as a community downloads it: plain prompt, no sources
sh(f"python scripts/run_eval.py --model {MODEL_DIR} --name base-plain --policy plain --langs {LANGS} --out /kaggle/working/runs/base-plain")
# 2) prompt-only Ìdìbò: our system prompt + retrieved official sources, no training
sh(f"python scripts/run_eval.py --model {MODEL_DIR} --name base-idibo-rag --policy idibo --rag --langs {LANGS} --out /kaggle/working/runs/base-idibo-rag")
print("done in", round((time.time() - t) / 60, 1), "min", flush=True)
