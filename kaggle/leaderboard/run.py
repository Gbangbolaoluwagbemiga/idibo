# Ìdìbò election-safety leaderboard: the same benchmarks, prompt and sources, run on other open Cohere models.
# Needs the Kaggle secret HF_TOKEN (gated models), so start it from the browser: "Save Version".
#
# Every model gets Ìdìbò's system prompt and the same 4 retrieved sources (309-passage evaluation KB), greedy decoding,
# on benchmark v1 (en, pcm, yo, ha, ig) and v2 (en held out, code-mixed). Then the same grading pipeline as the main
# results: back-translation by tiny-aya-global, grading by c4ai-command-r7b (so command-r7b is not a contestant).
import glob, os, shutil, subprocess, sys, tarfile, time

OUT = "/kaggle/working"
W = f"{OUT}/idibo"
T0 = time.time()
MODELS = [  # (repo, tensor-parallel GPUs)
    ("CohereLabs/tiny-aya-global", 1),
    ("CohereLabs/tiny-aya-fire", 1),
    ("CohereLabs/tiny-aya-water", 1),
    ("CohereLabs/aya-expanse-8b", 2),
    ("CohereLabs/aya-23-8B", 2),
]


def sh(cmd, check=True):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check, cwd=W)


def log(m):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)


os.makedirs(W, exist_ok=True)
tars = glob.glob("/kaggle/input/**/idibo.tar.gz", recursive=True)
assert tars, "attach dataset crypticdev/idibo-data"
tarfile.open(tars[0]).extractall(W)
# resume: earlier outputs of this kernel, if attached
for d in glob.glob("/kaggle/input/**/lb1", recursive=True) + glob.glob("/kaggle/input/**/lb2", recursive=True):
    shutil.copytree(d, f"{W}/{os.path.basename(d)}", dirs_exist_ok=True)

from kaggle_secrets import UserSecretsClient  # noqa: E402
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
sh("pip install -q uv && uv venv -q /kaggle/tmp/vv && uv pip install -q --python /kaggle/tmp/vv/bin/python vllm sentence-transformers huggingface_hub rank_bm25")
VPY = "/kaggle/tmp/vv/bin/python"


def fetch(repo):
    dst = f"/kaggle/tmp/{repo.split('/')[1]}"
    r = subprocess.run(f"{VPY} -c \"from huggingface_hub import snapshot_download as s; "
                       f"s('{repo}', local_dir='{dst}', allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja'])\"",
                       shell=True)
    return dst if r.returncode == 0 and os.path.exists(f"{dst}/config.json") else None


ran = []
for repo, tp in MODELS:
    name = repo.split("/")[1]
    path = fetch(repo)
    if not path:
        log(f"SKIP {repo}: download failed (accept its terms on Hugging Face)")
        continue
    for bench, langs, root in (("data/bench", "en,pcm,yo,ha,ig", "lb1"), ("data/bench2", "en,mix", "lb2")):
        cmd = (f"{VPY} scripts/run_eval_vllm.py --model {path} --name {name} --rag --policy idibo --bench {bench} "
               f"--langs {langs} --out {root}/{name} --tp {tp}")
        if sh(cmd, check=False).returncode != 0 and tp == 1:
            log(f"{name}: retry with 2 GPUs")
            sh(cmd.replace("--tp 1", "--tp 2"), check=False)
    shutil.rmtree(path, ignore_errors=True)  # disk
    ran.append(name)
    log(f"generated {name}")
    for d in ("lb1", "lb2"):
        if os.path.isdir(f"{W}/{d}"):
            shutil.copytree(f"{W}/{d}", f"{OUT}/{d}", dirs_exist_ok=True)  # keep partial results if the session ends

BT = fetch("CohereLabs/tiny-aya-global")
JUDGE = fetch("CohereLabs/c4ai-command-r7b-12-2024")
assert BT and JUDGE


def stage(args):
    if sh(f"{VPY} scripts/judge_local.py {args} --tp 2", check=False).returncode != 0:
        log("tensor-parallel run failed; retrying on one GPU")
        sh(f"{VPY} scripts/judge_local.py {args} --tp 1")


RUNS = " ".join(d for d in ("lb1", "lb2") if os.path.isdir(f"{W}/{d}"))
stage(f"--stage translate --runs {RUNS} --model {BT}")
log("back-translation done")
stage(f"--stage judge --runs {RUNS} --model {JUDGE} --name c4ai-command-r7b-12-2024")
log("judging done")
for d, out in (("lb1", "leaderboard_bench1.json"), ("lb2", "leaderboard_bench2.json")):
    dirs = " ".join(sorted(glob.glob(f"{W}/{d}/*/")))
    if dirs:
        sh(f"{sys.executable} scripts/score.py --summary-only --suffix graded.local {dirs} --out {OUT}/{out}", check=False)
    if os.path.isdir(f"{W}/{d}"):
        shutil.copytree(f"{W}/{d}", f"{OUT}/{d}", dirs_exist_ok=True)
shutil.rmtree(W, ignore_errors=True)
log(f"all done: {ran}")
