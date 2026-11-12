# Ìdìbò: LoRA fine-tunes of Tiny Aya Earth with AlignTune's RaftTrainer, then evaluation, on Kaggle (2x T4).
# Inputs: dataset crypticdev/idibo-data (code + data/train/*.jsonl) and the output of crypticdev/idibo-base-eval
# (which holds a copy of the gated model, so this notebook needs no HF token and can be started from the CLI).
#
# Every training and evaluation step is its own process, so its GPU memory is freed when it exits (v4 trained
# "civic" fine, then its evaluation ran out of memory because the training process still held 7 GB).
# Results from earlier runs (dataset crypticdev/idibo-artifacts: the civic adapter, finished eval files) are
# copied in first; run_eval.py resumes per item, so nothing finished is redone.
#   GPU 0: train civic (if missing) -> train idibo -> evaluate idibo-rag in en, pcm
#   GPU 1: evaluate civic-rag (4 langs) -> finish base evals -> wait for the idibo adapter -> idibo-rag in yo, ha
import glob, json, os, shutil, subprocess, sys, tarfile, time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # this process trains on GPU 0; the base evals get GPU 1 via env1


def sh(cmd, env=None):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, env=env)


sh("pip install -q -U 'transformers>=4.51.3' accelerate sentence-transformers peft 'trl>=1.0' datasets")
# --no-deps: AlignTune pins torch<2.14, and letting pip downgrade Kaggle's torch would cost a big reinstall.
sh("pip install -q --no-deps aligntune && pip install -q typer rich pyyaml psutil")
# Kaggle ships torchao 0.10, which PEFT's LoRA dispatcher rejects (needs >0.16); we don't use it.
sh("pip uninstall -y -q torchao || true")
import importlib
importlib.import_module("aligntune.backends.trl.raft.raft_trainer")  # fail fast if an import is missing

W = "/kaggle/working/idibo"
tars = glob.glob("/kaggle/input/**/idibo.tar.gz", recursive=True)
if tars:
    os.makedirs(W, exist_ok=True)
    tarfile.open(tars[0]).extractall(W)
else:
    # Only the dataset copy: the base-eval output also has an idibo/ folder, but without data/train.
    found = [f for f in glob.glob("/kaggle/input/**/scripts/run_eval.py", recursive=True)
             if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(f)), "data/train/civic.jsonl"))]
    assert found, f"training data not found in inputs: {glob.glob('/kaggle/input/*/*')}"
    shutil.copytree(os.path.dirname(os.path.dirname(found[0])), W, dirs_exist_ok=True)
os.chdir(W)
# Count GPUs without importing torch: initialising CUDA here would make the later CUDA_VISIBLE_DEVICES=0
# ineffective, and the Trainer would then spread over both GPUs and collide with the GPU-1 base evals.
NGPU = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout.count("GPU ")
assert NGPU >= 1, "no GPU attached: set Accelerator to GPU T4 x2"
print("gpus:", NGPU, flush=True)
MODEL = glob.glob("/kaggle/input/**/model/tiny-aya-earth/config.json", recursive=True)
assert MODEL, "attach the output of crypticdev/idibo-base-eval (it holds the model copy)"
MODEL = os.path.dirname(MODEL[0])
# Reading weights lazily from /kaggle/input took 17 min; a local copy loads in seconds.
if not os.path.exists("/kaggle/tmp/tiny-aya-earth/config.json"):
    shutil.copytree(MODEL, "/kaggle/tmp/tiny-aya-earth", dirs_exist_ok=True)
MODEL = "/kaggle/tmp/tiny-aya-earth"
print("model:", MODEL, flush=True)


OUT = "/kaggle/working"
LANGS = os.environ.get("EVAL_LANGS", "en,pcm,yo,ha")
# Results of earlier versions (dataset crypticdev/idibo-artifacts, one artifacts.tar.gz that Kaggle may or may not
# unpack); found by content, whatever the mount path.
for t in glob.glob("/kaggle/input/**/artifacts.tar.gz", recursive=True)[:1]:
    tarfile.open(t).extractall("/kaggle/tmp/artifacts")
for marker in (glob.glob("/kaggle/input/**/adapters/civic/adapter_config.json", recursive=True)
               + glob.glob("/kaggle/tmp/artifacts/adapters/civic/adapter_config.json"))[:1]:
    root = marker.rsplit("/adapters/", 1)[0]
    print("artifacts:", root, flush=True)
    for kind in ("adapters", "runs"):
        if os.path.isdir(f"{root}/{kind}"):
            shutil.copytree(f"{root}/{kind}", f"{OUT}/{kind}", dirs_exist_ok=True)
for f in sorted(glob.glob(f"{OUT}/runs/*/*.jsonl")):
    print("resume:", f, sum(1 for _ in open(f)), flush=True)

PY = sys.executable
ev = lambda name, langs, extra: (f"{PY} scripts/run_eval.py --model {MODEL} {extra} --name {name} "
                                 f"--langs {langs} --out {OUT}/runs/{name}")
train = lambda v: (f"[ -f {OUT}/adapters/{v}/adapter_config.json ] || "
                   f"{PY} scripts/train_lora.py --model {MODEL} --data data/train/{v}.jsonl --out {OUT}/adapters/{v}")
wait_idibo = f"while [ ! -f {OUT}/adapters/idibo/adapter_config.json ]; do sleep 60; done"
rag = "--policy idibo --rag"

gpu0 = " && ".join([
    train("civic"), train("idibo"),
    ev("idibo-rag", "en,pcm", f"--adapter {OUT}/adapters/idibo {rag}"),
])
gpu1 = " && ".join([
    f"while [ ! -f {OUT}/adapters/civic/adapter_config.json ]; do sleep 60; done",
    ev("civic-rag", LANGS, f"--adapter {OUT}/adapters/civic {rag}"),
    ev("base-idibo-rag", "pcm,yo,ha", rag),
    ev("base-plain", "pcm,yo,ha", "--policy plain"),
    wait_idibo,
    ev("idibo-rag", "yo,ha", f"--adapter {OUT}/adapters/idibo {rag}"),
])
procs = []
for gpu, cmd in ((0, gpu0), (1, gpu1)):
    print(f"+ [GPU {gpu}] {cmd}", flush=True)
    log = open(f"{OUT}/gpu{gpu}.log", "w")
    procs.append(subprocess.Popen(cmd, shell=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                  stdout=log, stderr=subprocess.STDOUT))
t0 = time.time()
while any(p.poll() is None for p in procs):
    time.sleep(300)
    done = {f.split("/runs/")[1]: sum(1 for _ in open(f)) for f in glob.glob(f"{OUT}/runs/*/*.jsonl")}
    print(f"[{(time.time() - t0) / 60:.0f} min] {done}", flush=True)
for gpu, p in enumerate(procs):
    print(f"GPU {gpu} exit {p.returncode}; last lines:", flush=True)
    print("".join(open(f"{OUT}/gpu{gpu}.log").readlines()[-15:]), flush=True)
shutil.rmtree(f"{OUT}/adapters/idibo-ckpt", ignore_errors=True)
shutil.rmtree(f"{OUT}/adapters/civic-ckpt", ignore_errors=True)
assert all(p.returncode == 0 for p in procs), "a GPU chain failed; see gpu*.log"
print("all done", flush=True)
