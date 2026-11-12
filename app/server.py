"""Ìdìbò web app: a voice-first, picture-first voter assistant (FastAPI + a static front end).

Run locally:   IDIBO_BACKEND=cohere uvicorn app.server:app --reload --port 7860
On HF Spaces:  Docker Space, port 7860 (see app/Dockerfile).

Backends (IDIBO_BACKEND)
  llamacpp  production: our fine-tuned Ìdìbò model as GGUF on CPU (IDIBO_GGUF=path)       <- the Space
  hf        transformers + optional LoRA adapter (IDIBO_MODEL, IDIBO_ADAPTER), GPU/MPS for local tests
  cohere    development only: base tiny-aya-earth via Cohere's API with our system prompt + sources.
            Every answer is labelled "dev backend" in the UI so it can never be mistaken for the fine-tune.

Voice
  /api/tts  YarnGPT (Nigerian voices; en/pcm/yo/ha/ig). Key from YARNGPT_API_KEY (a Space secret). Cached on disk.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from idibo.retrieve import HybridKB, format_context  # noqa: E402
from run_eval import SYSTEM  # noqa: E402  (same policy the model was trained and evaluated with)

BACKEND = os.environ.get("IDIBO_BACKEND", "cohere")
STATIC = Path(__file__).parent / "static"
CACHE = Path(os.environ.get("IDIBO_CACHE", "/tmp/idibo-cache"))
CACHE.mkdir(parents=True, exist_ok=True)
LANGS = {"en": "English", "pcm": "Naijá", "yo": "Yorùbá", "ha": "Hausa", "ig": "Igbo"}
VOICE = {"en": "femi", "pcm": "femi", "yo": "idera", "ha": "zainab", "ig": "chinenye"}
YARN_LANG = {"en": "english", "pcm": "pidgin", "yo": "yoruba", "ha": "hausa", "ig": "igbo"}

app = FastAPI(title="Ìdìbò")
_kb: HybridKB | None = None


def kb() -> HybridKB:
    global _kb
    if _kb is None:
        _kb = HybridKB(path=ROOT / "data/kb/chunks.jsonl", emb=ROOT / "data/kb/e5_small.npy", device="cpu")
    return _kb


def env_key(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    for p in (ROOT / ".env", ROOT.parent.parent.parent / ".env"):
        if p.exists():
            for line in open(p, encoding="utf-8"):
                if line.startswith(f"{name}=") and "PASTE" not in line:
                    return line.split("=", 1)[1].strip()
    return None


# ------------------------------------------------------------------ model backends
class Generator:
    label = ""

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:  # pragma: no cover
        yield ""


class CohereDev(Generator):
    label = "dev backend: base tiny-aya-earth via Cohere API (not the fine-tuned model)"

    async def stream(self, messages):
        key = env_key("COHERE_API_KEY")
        body = {"model": "tiny-aya-earth", "messages": messages, "max_tokens": 500, "temperature": 0.2, "stream": True}
        async with httpx.AsyncClient(timeout=120) as c:
            async with c.stream("POST", "https://api.cohere.com/v2/chat", json=body,
                                headers={"Authorization": f"Bearer {key}"}) as r:
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        ev = json.loads(line[5:])
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") == "content-delta":
                        yield ev["delta"]["message"]["content"]["text"]


class LlamaCpp(Generator):
    label = "Ìdìbò (fine-tuned Tiny Aya Earth, running on CPU)"

    def __init__(self):
        from llama_cpp import Llama

        self.llm = Llama(model_path=os.environ["IDIBO_GGUF"], n_ctx=4096, n_threads=os.cpu_count() or 2, verbose=False)

    async def stream(self, messages):
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def run():
            for part in self.llm.create_chat_completion(messages=messages, max_tokens=420, temperature=0.1, stream=True):
                delta = part["choices"][0]["delta"].get("content")
                if delta:
                    loop.call_soon_threadsafe(q.put_nowait, delta)
            loop.call_soon_threadsafe(q.put_nowait, None)

        loop.run_in_executor(None, run)
        while (tok := await q.get()) is not None:
            yield tok


class HFLocal(Generator):
    label = "Ìdìbò (local transformers)"

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        mid = os.environ.get("IDIBO_MODEL", "CohereLabs/tiny-aya-earth")
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(mid)
        self.model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16 if dev != "cpu" else torch.float32).to(dev)
        if os.environ.get("IDIBO_ADAPTER"):
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, os.environ["IDIBO_ADAPTER"]).to(dev)
        self.dev = dev

    async def stream(self, messages):
        from threading import Thread

        from transformers import TextIteratorStreamer

        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to(self.dev) for k, v in enc.items()}
        st = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        Thread(target=self.model.generate, kwargs=dict(**enc, max_new_tokens=420, do_sample=False, streamer=st)).start()
        for piece in st:
            yield piece
            await asyncio.sleep(0)


_gen: Generator | None = None


def gen() -> Generator:
    global _gen
    if _gen is None:
        _gen = {"cohere": CohereDev, "llamacpp": LlamaCpp, "hf": HFLocal}[BACKEND]()
    return _gen


# ------------------------------------------------------------------ API
class Ask(BaseModel):
    question: str
    lang: str = "en"
    mode: str = "ask"  # "ask" | "check" (is this message real?)


CHECK_PREFIX = {
    "en": "Someone sent me this message. Is it true according to INEC and the Electoral Act? Message: ",
    "pcm": "Person send me this message. E true according to INEC and the Electoral Act? Message: ",
    "yo": "Ẹnìkan fi ọ̀rọ̀ yìí ránṣẹ́ sí mi. Ṣé òótọ́ ni gẹ́gẹ́ bí INEC àti Òfin Ìdìbò ṣe sọ? Ọ̀rọ̀ náà: ",
    "ha": "Wani ya aiko mini da wannan saƙon. Gaskiya ne bisa ga INEC da Dokar Zaɓe? Saƙon: ",
    "ig": "Mmadụ zitere m ozi a. Ọ bụ eziokwu dịka INEC na Iwu Ntuli Aka si kwuo? Ozi ahụ: ",
}


@app.post("/api/ask")
async def ask(req: Ask):
    q = req.question.strip()[:1200]
    if not q:
        raise HTTPException(400, "empty question")
    if req.mode == "check":
        q = CHECK_PREFIX.get(req.lang, CHECK_PREFIX["en"]) + q
    hits = kb().search(q, k=4)
    user = f"SOURCES:\n{format_context(hits)}\n\nQUESTION:\n{q}"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    sources = [{"id": h["id"], "title": h["title"], "section": h.get("section", ""), "url": h["url"]} for h in hits]

    async def events():
        yield f"data: {json.dumps({'type': 'meta', 'backend': gen().label, 'sources': sources})}\n\n"
        full = ""
        t = time.time()
        try:
            async for piece in gen().stream(messages):
                full += piece
                yield f"data: {json.dumps({'type': 'token', 'text': piece})}\n\n"
        except Exception as e:  # keep the UI alive and honest
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)[:200]})}\n\n"
        cited = sorted(set(re.findall(r"\[([a-z0-9-]+)\]", full)) & {s["id"] for s in sources})
        yield f"data: {json.dumps({'type': 'done', 'cited': cited, 'seconds': round(time.time() - t, 1)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/api/source/{sid}")
def source(sid: str):
    for d in kb().docs:
        if d["id"] == sid:
            return {k: d.get(k) for k in ("id", "title", "section", "url", "text", "published")}
    raise HTTPException(404)


class TTS(BaseModel):
    text: str
    lang: str = "en"


@app.post("/api/tts")
async def tts(req: TTS):
    key = env_key("YARNGPT_API_KEY")
    if not key:
        raise HTTPException(503, "voice is not configured")
    text = re.sub(r"\[[a-z0-9-]+\]", "", req.text)  # don't read citation ids aloud
    text = re.sub(r"[*#_`]", "", text).strip()[:1500]
    voice = VOICE.get(req.lang, "femi")
    h = hashlib.sha256(f"{voice}|{req.lang}|{text}".encode()).hexdigest()[:24]
    path = CACHE / f"{h}.mp3"
    if not path.exists():
        hdr = {"Authorization": f"Bearer {key}", "Idempotency-Key": f"idibo-{h}"}
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post("https://api.yarngpt.ai/api/v1/tts", headers=hdr,
                             json={"text": text, "voice": voice, "language": YARN_LANG.get(req.lang, "english"),
                                   "output_format": "mp3"})
            if r.status_code >= 400:
                raise HTTPException(502, "voice service error")
            job = r.json()
            for _ in range(60):
                s = (await c.get(f"https://api.yarngpt.ai/api/v1/status/{job['job_id']}", headers=hdr)).json()
                if s.get("status") in ("completed", "failed"):
                    break
                await asyncio.sleep(1.0)
            if s.get("status") != "completed":
                raise HTTPException(502, "voice not ready")
            audio = await c.get(s["audio_url"])
            path.write_bytes(audio.content)
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/api/judge")
def judge_view():
    p = ROOT / "app" / "judge.json"
    return JSONResponse(json.loads(p.read_text()) if p.exists() else {"examples": [], "summary": {}})


@app.get("/api/health")
def health():
    return {"ok": True, "backend": BACKEND, "langs": LANGS}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
