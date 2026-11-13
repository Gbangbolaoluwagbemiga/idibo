# Publish Ìdìbò to the Hugging Face Hub from Kaggle (datacenter to datacenter, so the 2 GB GGUF never crosses the
# builder's home connection). Needs the Kaggle secret HF_WRITE_TOKEN, so start it from the browser (Save Version):
# secrets are not passed to runs pushed from the CLI.
#   model   GOPAE/idibo-tiny-aya-earth : idibo-q4_k_m.gguf + LoRA adapters (idibo, civic)
#   dataset GOPAE/idibo-eval           : benchmark (5 languages), training data, knowledge base, model generations
import glob, os, shutil, subprocess, tarfile

subprocess.run("pip install -q -U huggingface_hub", shell=True, check=True)
from huggingface_hub import HfApi
from kaggle_secrets import UserSecretsClient

token = UserSecretsClient().get_secret("HF_WRITE_TOKEN")
api = HfApi(token=token)
user = api.whoami()["name"]
MODEL, DATA = f"{user}/idibo-tiny-aya-earth", f"{user}/idibo-eval"
print("publishing as", user, flush=True)

def one(pattern):
    hits = glob.glob(pattern, recursive=True)
    assert hits, f"missing input: {pattern}"
    return hits[0]

# ---------------- model repo
api.create_repo(MODEL, repo_type="model", exist_ok=True)
gguf = one("/kaggle/input/**/idibo-idibo-q4_k_m.gguf")
api.upload_file(path_or_fileobj=gguf, path_in_repo="idibo-q4_k_m.gguf", repo_id=MODEL,
                commit_message="Ìdìbò Q4_K_M GGUF (LoRA merged into Tiny Aya Earth)")
for v in ("idibo", "civic"):
    a = os.path.dirname(one(f"/kaggle/input/**/adapters/{v}/adapter_config.json"))
    api.upload_folder(folder_path=a, path_in_repo=f"adapters/{v}", repo_id=MODEL,
                      allow_patterns=["adapter_config.json", "adapter_model.safetensors", "log_history.json"],
                      commit_message=f"LoRA adapter: {v}")
print("model uploaded:", f"https://huggingface.co/{MODEL}", flush=True)

# ---------------- dataset repo
W = "/kaggle/tmp/idibo"
tars = glob.glob("/kaggle/input/**/idibo.tar.gz", recursive=True)
if tars:
    os.makedirs(W, exist_ok=True); tarfile.open(tars[0]).extractall(W)
else:
    src = [f for f in glob.glob("/kaggle/input/**/scripts/run_eval.py", recursive=True)
           if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(f)), "data/train/idibo.jsonl"))][0]
    shutil.copytree(os.path.dirname(os.path.dirname(src)), W, dirs_exist_ok=True)
stage = "/kaggle/tmp/dataset"
os.makedirs(stage, exist_ok=True)
for sub in ("bench", "train", "sft"):
    if os.path.isdir(f"{W}/data/{sub}"):
        shutil.copytree(f"{W}/data/{sub}", f"{stage}/{sub}", dirs_exist_ok=True)
os.makedirs(f"{stage}/kb", exist_ok=True)
shutil.copy(f"{W}/data/kb/chunks.jsonl", f"{stage}/kb/chunks.jsonl")
runs = os.path.dirname(os.path.dirname(one("/kaggle/input/**/runs/idibo-rag/en.jsonl")))
shutil.copytree(runs, f"{stage}/generations", dirs_exist_ok=True)
for junk in glob.glob(f"{stage}/**/._*", recursive=True):
    os.remove(junk)
api.create_repo(DATA, repo_type="dataset", exist_ok=True)
api.upload_folder(folder_path=stage, repo_id=DATA, repo_type="dataset",
                  commit_message="Ìdìbò benchmark, training data, knowledge base and generations")
print("dataset uploaded:", f"https://huggingface.co/datasets/{DATA}", flush=True)
