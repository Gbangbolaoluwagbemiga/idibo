# Ìdìbò round 2 on Kaggle (2x T4): SafeTune repairs + the held-out bench v2 + cross-lingual retrieval + Igbo.
# Needs no HF token (the models come from the idibo-base-eval output), so it can be started from the CLI.
#
# Inputs: dataset crypticdev/idibo-data (code + data, incl. data/bench2), outputs of crypticdev/idibo-base-eval
# (tiny-aya-earth + tiny-aya-base copies) and crypticdev/idibo-train (civic + idibo LoRA adapters).
#
# A two-GPU job queue: each GPU takes the next job whose inputs exist. Generation runs on vLLM in its own venv
# (falls back to the HF runner if vLLM can't be installed). Every job is a separate process (GPU memory is freed).
import glob, json, os, shutil, subprocess, sys, tarfile, threading, time

OUT = "/kaggle/working"
T0 = time.time()


def sh(cmd, check=True):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


# ---------------------------------------------------------------- code + models
W = "/kaggle/working/idibo"
tars = glob.glob("/kaggle/input/**/idibo.tar.gz", recursive=True)
if tars:
    os.makedirs(W, exist_ok=True)
    tarfile.open(tars[0]).extractall(W)
else:
    found = [f for f in glob.glob("/kaggle/input/**/scripts/run_eval_vllm.py", recursive=True)]
    assert found, f"idibo-data (v-round2) not found: {glob.glob('/kaggle/input/*/*')}"
    shutil.copytree(os.path.dirname(os.path.dirname(found[0])), W, dirs_exist_ok=True)
os.chdir(W)
assert os.path.exists("data/bench2/en.jsonl"), "this idibo-data version has no bench2"


def local_copy(name):
    hit = glob.glob(f"/kaggle/input/**/model/{name}/config.json", recursive=True)
    assert hit, f"{name} not found: attach the idibo-base-eval output"
    dst = f"/kaggle/tmp/{name}"
    if not os.path.exists(f"{dst}/config.json"):
        shutil.copytree(os.path.dirname(hit[0]), dst, dirs_exist_ok=True)  # local disk: loads far faster
    return dst


EARTH, BASE = local_copy("tiny-aya-earth"), local_copy("tiny-aya-base")
ADP = {}
for v in ("civic", "idibo"):
    hit = glob.glob(f"/kaggle/input/**/adapters/{v}/adapter_config.json", recursive=True)
    assert hit, f"adapter {v} not found: attach the idibo-train output"
    ADP[v] = f"{OUT}/adapters/{v}"
    shutil.copytree(os.path.dirname(hit[0]), ADP[v], dirs_exist_ok=True)
log(f"models ready: {EARTH}, {BASE}, adapters {ADP}")

# ---------------------------------------------------------------- environments
sh("pip install -q -U 'transformers>=5.15,<6' accelerate sentence-transformers 'peft<1' datasets")
sh("pip install -q 'safetune @ git+https://github.com/Lexsi-Labs/SafeTune.git'")
sh("pip uninstall -y -q torchao || true")  # Kaggle's torchao 0.10 breaks PEFT's LoRA dispatch
sh(f"{sys.executable} -c 'import safetune.runner.recover as r; print(\"safetune\", r.ReStaTrainer)'")

VPY = "/kaggle/tmp/vv/bin/python"
use_vllm = False
try:
    # Kaggle's python has no ensurepip, so `python -m venv` fails (v1): build the venv with uv instead.
    sh("pip install -q uv && uv venv -q /kaggle/tmp/vv && uv pip install -q --python /kaggle/tmp/vv/bin/python vllm sentence-transformers")
    sh(f"cd {W} && {VPY} -c 'import vllm, sys; sys.path.insert(0, \"scripts\"); print(\"vllm\", vllm.__version__)'")
    use_vllm = True
except subprocess.CalledProcessError as e:
    log(f"vLLM unavailable ({e}); falling back to the HF runner")
log(f"engine: {'vllm' if use_vllm else 'hf'}")

# ---------------------------------------------------------------- jobs
PY = sys.executable
R2 = f"{OUT}/runs2"   # bench v2 generations
R1 = f"{OUT}/runs"    # bench v1 generations (new conditions only)
RESTA = "/kaggle/tmp/civic-resta"
SAFEL = f"{OUT}/adapters/civic-safelora"


def gen(name, bench, langs, model=EARTH, adapter=None, rag=True, pivot=False, root=R2):
    out = f"{root}/{name}"
    if use_vllm:
        cmd = (f"{VPY} scripts/run_eval_vllm.py --model {model} --name {name} --langs {langs} --bench {bench} --out {out}"
               + (f" --adapter {adapter}" if adapter else "") + (" --rag" if rag else "") + (" --pivot" if pivot else ""))
    else:
        cmd = (f"{PY} scripts/run_eval.py --model {model} --name {name} --langs {langs} --bench {bench} --out {out} --batch 4"
               + (f" --adapter {adapter}" if adapter else "") + (" --rag" if rag else "") + (" --pivot" if pivot else ""))
    return cmd + " --policy idibo"


# v1 machine-translated bench v2 with tiny-aya-earth: the safety-tuned translator answered or refused many harmful
# prompts instead of translating them, so v2 runs in English plus the 30 hand-written code-mixed prompts only.
B2 = "en,mix"
# (name, command, files that must exist first, file that marks it done). Listed in priority order.
PATCHES = [
    ("patch-resta", f"{PY} scripts/patch_safetune.py --method resta --aligned {EARTH} --base {BASE} "
                    f"--adapter {ADP['civic']} --out {RESTA} --alpha 0.25", [], f"{RESTA}/safetune_patch.json"),
    ("patch-safelora", f"{PY} scripts/patch_safetune.py --method safelora --aligned {EARTH} --base {BASE} "
                       f"--adapter {ADP['civic']} --out {SAFEL}", [], f"{SAFEL}/safetune_patch.json"),
]
JOBS = [
    ("b2-idibo-rag", gen("idibo-rag", "data/bench2", B2, adapter=ADP["idibo"]), [], None),
    ("b2-civic-rag", gen("civic-rag", "data/bench2", B2, adapter=ADP["civic"]), [], None),
    ("b2-base-idibo-rag", gen("base-idibo-rag", "data/bench2", B2), [], None),
    ("b2-civic-resta-rag", gen("civic-resta-rag", "data/bench2", B2, model=RESTA), [f"{RESTA}/safetune_patch.json"], None),
    ("b2-civic-safelora-rag", gen("civic-safelora-rag", "data/bench2", B2, adapter=SAFEL), [f"{SAFEL}/safetune_patch.json"], None),
    ("b1-civic-resta-rag", gen("civic-resta-rag", "data/bench", "en,pcm,yo,ha", model=RESTA, root=R1), [f"{RESTA}/safetune_patch.json"], None),
    ("b1-civic-safelora-rag", gen("civic-safelora-rag", "data/bench", "en,pcm,yo,ha", adapter=SAFEL, root=R1), [f"{SAFEL}/safetune_patch.json"], None),
    ("b1-idibo-rag-xq", gen("idibo-rag-xq", "data/bench", "yo,ha,ig", adapter=ADP["idibo"], pivot=True, root=R1), [], None),
    ("b1-idibo-rag-ig", gen("idibo-rag", "data/bench", "ig", adapter=ADP["idibo"], root=R1), [], None),
    ("b1-base-idibo-rag-ig", gen("base-idibo-rag", "data/bench", "ig", root=R1), [], None),
]
state = {name: "todo" for name, *_ in PATCHES + JOBS}
lock = threading.Lock()


def gpu_free_mb(gpu):
    q = subprocess.run(f"nvidia-smi -i {gpu} --query-gpu=memory.free --format=csv,noheader,nounits", shell=True,
                       capture_output=True, text=True).stdout.strip()
    return int(q) if q.isdigit() else 0


def run_job(gpu, name, cmd, marker, attempt=1):
    """One job in its own process group; the whole group is killed afterwards (a killed vLLM parent left its
    EngineCore child holding GPU memory in v2, and every later job on that GPU failed to start)."""
    for _ in range(40):  # wait for the GPU to be really free
        if gpu_free_mb(gpu) > 13000:
            break
        subprocess.run(f"nvidia-smi -i {gpu} --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9",
                       shell=True)
        time.sleep(15)
    log(f"GPU{gpu} start {name} (attempt {attempt}, free {gpu_free_mb(gpu)} MB)")
    with open(f"{OUT}/logs/{name}.log", "a") as lf:
        p = subprocess.Popen(cmd, shell=True, cwd=W, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True,
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), VLLM_WORKER_MULTIPROC_METHOD="spawn"))
        rc = p.wait()
        try:
            os.killpg(p.pid, 9)
        except ProcessLookupError:
            pass
    ok = rc == 0 and (marker is None or os.path.exists(marker))
    if not ok and attempt == 1:
        log(f"GPU{gpu} {name} failed (rc {rc}); retrying once")
        return run_job(gpu, name, cmd, marker, attempt=2)
    return ok, rc


def worker(gpu):
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    while True:
        with lock:
            pending = [j for j in JOBS if state[j[0]] == "todo"]
            if not pending:
                return
            ready = [j for j in pending if all(os.path.exists(p) for p in j[2])]
            # a job whose inputs can never appear (a failed dependency) is skipped
            dead = [j for j in pending if any(state.get(n) == "failed" for n, _, _, m in PATCHES + JOBS if m in j[2])]
            for j in dead:
                state[j[0]] = "failed"
                log(f"skip {j[0]}: a dependency failed")
            job = next((j for j in ready if j not in dead), None)
            if job:
                state[job[0]] = "running"
        if not job:
            time.sleep(30)
            continue
        name, cmd, _, marker = job
        ok, rc = run_job(gpu, name, cmd, marker)
        with lock:
            state[name] = "done" if ok else "failed"
        log(f"GPU{gpu} {'done' if ok else 'FAILED'} {name} (rc {rc})" + ("" if ok else ":\n" + "".join(open(f"{OUT}/logs/{name}.log").readlines()[-25:])))


NGPU = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout.count("GPU ")
assert NGPU >= 1, "no GPU attached: set Accelerator to GPU T4 x2"
# Phase 1: the two SafeTune repairs, one at a time and alone (ReSta holds three 3.35B models in CPU RAM; in v2 it was
# killed for memory while vLLM jobs ran beside it).
os.makedirs(f"{OUT}/logs", exist_ok=True)
for name, cmd, _, marker in PATCHES:
    ok, rc = run_job(0, name, cmd, marker)
    state[name] = "done" if ok else "failed"
    log(f"{name}: {'done' if ok else 'FAILED'} (rc {rc})" + ("" if ok else ":\n" + "".join(open(f"{OUT}/logs/{name}.log").readlines()[-25:])))
    if ok:
        log(open(marker).read())
# Phase 2: generation on both GPUs.
threads = [threading.Thread(target=worker, args=(g,)) for g in range(NGPU)]
for t in threads:
    t.start()
while any(t.is_alive() for t in threads):
    time.sleep(300)
    counts = {f.split("/kaggle/working/")[1]: sum(1 for _ in open(f)) for f in glob.glob(f"{OUT}/runs*/*/*.jsonl")}
    log(f"{state}\n{counts}")
for t in threads:
    t.join()

for d in glob.glob(f"{OUT}/adapters/*-ckpt"):
    shutil.rmtree(d, ignore_errors=True)
for v in ("civic", "idibo"):  # inputs, not outputs
    shutil.rmtree(ADP[v], ignore_errors=True)
json.dump(state, open(f"{OUT}/round2_state.json", "w"), indent=1)
log(f"finished: {state}")
assert all(v == "done" for v in state.values()), "some jobs failed; see logs/"
