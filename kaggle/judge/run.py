# Ìdìbò: grade every answer with open-weight Cohere models on Kaggle (2x T4, vLLM). No API calls.
# Needs the Kaggle secret HF_TOKEN (gated models), so start it from the browser: "Save Version".
#
# Inputs: dataset crypticdev/idibo-data (code), dataset crypticdev/idibo-runs (round-1 generations and their
# command-a-plus grades), output of crypticdev/idibo-round2 (new generations + the translated bench v2).
#   1. back-translate non-English answers: CohereLabs/tiny-aya-global (falls back to tiny-aya-earth)
#   2. judge every answer: CohereLabs/c4ai-command-r7b-12-2024 (falls back to aya-expanse-8b)
#   3. summaries with 95% intervals + agreement with command-a-plus on the round-1 items it graded
import glob, os, shutil, subprocess, sys, tarfile, time

OUT = "/kaggle/working"
W = f"{OUT}/idibo"
T0 = time.time()


def sh(cmd, check=True):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check, cwd=W if os.path.isdir(W) else None)


def log(m):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)


tars = glob.glob("/kaggle/input/**/idibo.tar.gz", recursive=True)
os.makedirs(W, exist_ok=True)
if tars:
    tarfile.open(tars[0]).extractall(W)
else:
    # Prefer the idibo-data dataset: the round-2 output also holds an older copy of the code.
    found = sorted(glob.glob("/kaggle/input/**/scripts/judge_local.py", recursive=True),
                   key=lambda f: ("idibo-data" not in f, len(f)))
    assert found, "attach crypticdev/idibo-data (round-2 version)"
    shutil.copytree(os.path.dirname(os.path.dirname(found[0])), W, dirs_exist_ok=True)

# generations: round 1 (with command-a-plus grades) and round 2
for t in glob.glob("/kaggle/input/**/runs.tar.gz", recursive=True)[:1]:
    tarfile.open(t).extractall(W)
# Merge every runs/ and runs2/ folder found in the inputs (round 1 dataset + round 2 output).
for d in sorted(glob.glob("/kaggle/input/*/runs") + glob.glob("/kaggle/input/*/*/runs")
                + glob.glob("/kaggle/input/*/runs2") + glob.glob("/kaggle/input/*/*/runs2")):
    if os.path.isdir(d) and "/idibo/" not in d:
        log(f"merge {d}")
        shutil.copytree(d, f"{W}/{os.path.basename(d)}", dirs_exist_ok=True)
for d in ("runs", "runs2"):
    if os.path.isdir(f"{W}/{d}"):
        shutil.rmtree(f"{W}/{d}/smoke-batch", ignore_errors=True)
        log(f"{d}: {sorted(os.listdir(f'{W}/{d}'))}")
# Keep .bt / .graded.local files from an earlier judge version so a re-run resumes.
for f in glob.glob("/kaggle/input/**/*.graded.local.jsonl", recursive=True) + glob.glob("/kaggle/input/**/*.bt.jsonl", recursive=True):
    rel = f.split("/idibo/", 1)[-1] if "/idibo/" in f else None
    if rel and rel.startswith(("runs/", "runs2/")) and not os.path.exists(f"{W}/{rel}"):
        os.makedirs(os.path.dirname(f"{W}/{rel}"), exist_ok=True)
        shutil.copy(f, f"{W}/{rel}")

from kaggle_secrets import UserSecretsClient  # noqa: E402
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
sh("pip install -q uv && uv venv -q /kaggle/tmp/vv && uv pip install -q --python /kaggle/tmp/vv/bin/python vllm sentence-transformers huggingface_hub")
VPY = "/kaggle/tmp/vv/bin/python"


def fetch(repo):
    dst = f"/kaggle/tmp/{repo.split('/')[1]}"
    r = subprocess.run(f"{VPY} -c \"from huggingface_hub import snapshot_download as s; "
                       f"s('{repo}', local_dir='{dst}', allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja'])\"",
                       shell=True)
    return dst if r.returncode == 0 and os.path.exists(f"{dst}/config.json") else None


BT = fetch("CohereLabs/tiny-aya-global")
if not BT:
    hit = glob.glob("/kaggle/input/**/model/tiny-aya-earth/config.json", recursive=True)
    BT = os.path.dirname(hit[0]) if hit else fetch("CohereLabs/tiny-aya-earth")
JUDGE = fetch("CohereLabs/c4ai-command-r7b-12-2024") or fetch("CohereLabs/aya-expanse-8b")
assert BT and JUDGE, "could not download the models: accept their terms on Hugging Face"
log(f"back-translation: {BT}; judge: {JUDGE}")

RUNS = " ".join(d for d in ("runs", "runs2") if os.path.isdir(f"{W}/{d}"))
sh(f"{VPY} scripts/judge_local.py --stage translate --runs {RUNS} --model {BT} --tp 2")
log("back-translation done")
sh(f"{VPY} scripts/judge_local.py --stage judge --runs {RUNS} --model {JUDGE} --tp 2 --name {os.path.basename(JUDGE)}")
log("judging done")

r1 = " ".join(sorted(glob.glob(f"{W}/runs/*/")))
r2 = " ".join(sorted(glob.glob(f"{W}/runs2/*/")))
sh(f"{sys.executable} scripts/score.py --summary-only --suffix graded.local {r1} --out {OUT}/summary_bench1.json", check=False)
if r2:
    sh(f"{sys.executable} scripts/score.py --summary-only --suffix graded.local {r2} --out {OUT}/summary_bench2.json", check=False)
sh(f"{sys.executable} scripts/judge_agreement.py {r1}", check=False)
for d in ("runs", "runs2"):
    if os.path.isdir(f"{W}/{d}"):
        shutil.copytree(f"{W}/{d}", f"{OUT}/{d}", dirs_exist_ok=True)
if os.path.exists(f"{W}/runs/judge_agreement.json"):
    shutil.copy(f"{W}/runs/judge_agreement.json", OUT)
shutil.rmtree(W, ignore_errors=True)
log("all done")
