# Ìdìbò → GGUF for CPU serving (the HF Space and local tests): merge the LoRA adapter into Tiny Aya Earth,
# convert with llama.cpp, quantise to Q4_K_M. CPU-only notebook; inputs are the base-eval output (model copy)
# and the train output (adapters).
import glob, os, shutil, subprocess

def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

sh("pip install -q -U 'transformers>=4.51.3' peft accelerate sentencepiece gguf protobuf")
sh("pip uninstall -y -q torchao || true")
sh("git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /kaggle/tmp/llama.cpp")
sh("cmake -S /kaggle/tmp/llama.cpp -B /kaggle/tmp/llama.cpp/build -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF > /dev/null")
sh("cmake --build /kaggle/tmp/llama.cpp/build --target llama-quantize -j 4 > /dev/null")

base = os.path.dirname(glob.glob("/kaggle/input/**/model/tiny-aya-earth/config.json", recursive=True)[0])
OUT = "/kaggle/working"
for variant in os.environ.get("VARIANTS", "idibo").split(","):
    adapter = os.path.dirname(glob.glob(f"/kaggle/input/**/adapters/{variant}/adapter_config.json", recursive=True)[0])
    print("base:", base, "adapter:", adapter, flush=True)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    merged = f"/kaggle/tmp/merged-{variant}"
    m.save_pretrained(merged, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(merged)
    del m
    f16 = f"/kaggle/tmp/{variant}-f16.gguf"
    sh(f"python /kaggle/tmp/llama.cpp/convert_hf_to_gguf.py {merged} --outtype f16 --outfile {f16}")
    sh(f"/kaggle/tmp/llama.cpp/build/bin/llama-quantize {f16} {OUT}/idibo-{variant}-q4_k_m.gguf Q4_K_M")
    os.remove(f16)
    shutil.rmtree(merged)
sh(f"ls -la {OUT}")
