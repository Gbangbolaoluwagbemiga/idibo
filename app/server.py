"""Ìdìbò web app: a voice-first, picture-first voter assistant (FastAPI + a static front end).

Run locally:   IDIBO_BACKEND=cohere uvicorn app.server:app --reload --port 7860
On HF Spaces:  Docker Space, port 7860 (see app/Dockerfile).

Backends (IDIBO_BACKEND)
  llamacpp  production: our fine-tuned Ìdìbò model as GGUF on CPU (IDIBO_GGUF=path)       <- the Space
  hf        transformers + optional LoRA adapter (IDIBO_MODEL, IDIBO_ADAPTER), GPU/MPS for local tests
  cohere    development only: base tiny-aya-earth via Cohere's API with our system prompt + sources.
            Every answer is labelled "dev backend" in the UI so it can never be mistaken for the fine-tune.

Voice
  /api/tts  YarnGPT, voice "umar" in all five languages. Key from YARNGPT_API_KEY (a Space secret). Cached on disk.
  /api/stt  recorded question -> text: YarnGPT ASR, falling back to local faster-whisper (no Igbo in Whisper).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from idibo import contacts, guard  # noqa: E402
from idibo.pivot import PIVOT_LANGS, clean_pivot, pivot_messages, search_query  # noqa: E402
from idibo.retrieve import HybridKB, complete_lists, format_context  # noqa: E402
from run_eval import SYSTEM  # noqa: E402  (same policy the model was trained and evaluated with)

BACKEND = os.environ.get("IDIBO_BACKEND", "cohere")
STATIC = Path(__file__).parent / "static"
CACHE = Path(os.environ.get("IDIBO_CACHE", "/tmp/idibo-cache"))
CACHE.mkdir(parents=True, exist_ok=True)
LANGS = {"en": "English", "pcm": "Naijá", "yo": "Yorùbá", "ha": "Hausa", "ig": "Igbo"}
VOICE = {"en": "umar", "pcm": "umar", "yo": "umar", "ha": "umar", "ig": "umar"}  # one voice, all five languages
YARN = "https://api.yarngpt.ai/api/v1"

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


async def complete(messages: list[dict], max_tokens: int = 60) -> str:
    g = gen()
    if hasattr(g, "complete"):
        return await g.complete(messages, max_tokens)
    out = ""
    async for piece in g.stream(messages):
        out += piece
        if len(out) > 400:
            break
    return out


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
    label = "Ìdìbò (fine-tuned Tiny Aya Earth, 4-bit)"

    def __init__(self):
        from llama_cpp import Llama

        from llama_cpp import LlamaRAMCache

        # Containers often report the host's cores, not the quota: too many threads makes CPU inference crawl.
        self.threads = int(os.environ.get("IDIBO_THREADS") or min(os.cpu_count() or 2, 8))
        self.llm = Llama(model_path=os.environ["IDIBO_GGUF"], n_ctx=4096, n_threads=self.threads,
                         n_threads_batch=self.threads, n_batch=512,
                         n_gpu_layers=int(os.environ.get("IDIBO_GPU_LAYERS", "-1")), verbose=False)  # Metal on a Mac; CPU on a server
        self.llm.set_cache(LlamaRAMCache(capacity_bytes=1 << 30))  # reuse the shared system-prompt prefix
        self.lock = asyncio.Lock()
        self.tlock = threading.Lock()

    async def complete(self, messages, max_tokens=60):
        def run():
            with self.tlock:
                return self.llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0)

        async with self.lock:
            r = await asyncio.to_thread(run)
        return r["choices"][0]["message"]["content"]

    async def stream(self, messages):
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        stop = threading.Event()

        def run():
            # The thread holds tlock for its whole life: if the client disconnects, the async lock is released but this
            # thread may still be generating; two generations on one llama.cpp context corrupt its state (garbled text).
            with self.tlock:
                try:
                    for part in self.llm.create_chat_completion(messages=messages, max_tokens=420, temperature=0.1,
                                                                repeat_penalty=1.12, stream=True):  # 4-bit looped on Pidgin
                        if stop.is_set():
                            break
                        delta = part["choices"][0]["delta"].get("content")
                        if delta:
                            loop.call_soon_threadsafe(q.put_nowait, delta)
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, None)

        async with self.lock:  # one llama.cpp context: never run two generations at once
            loop.run_in_executor(None, run)
            try:
                while (tok := await q.get()) is not None:
                    yield tok
            finally:
                stop.set()  # client gone: stop generating now


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


# ------------------------------------------------------------------ official lists, shown verbatim
# "Who are the candidates?" is answered with INEC's own list, word for word, in INEC's order: no generated names, no
# party left out. (The fine-tune learned to answer "who will win"-style questions with "I don't know, ask INEC": it
# declined list requests and answered APC's candidate but not LP's. Verbatim lines treat every party the same.)
LIST_PREFIX = "inec-presidential-candidates-2027-"
LIST_CUE = re.compile(r"candidat|running for|run for|contest|on the ballot|aspirant|takara|olùdíje|oludije|na-azọ|azo ọkwa|"
                      r"azọ ọkwa|who dey run|who go run|who de run", re.I)
LIST_HEAD = {
    "en": "INEC's final list of presidential candidates for the 16 January 2027 election (19 parties, in INEC's order, "
          "names as INEC prints them, surname first). Ìdìbò does not share photos; see the official list at the source below.",
    "pcm": "Na INEC final list of presidential candidates for the 16 January 2027 election be dis (19 parties, as INEC "
           "arrange am, surname first). Ìdìbò no dey share photo; check the official list for the source below.",
    "yo": "Àkójọ ìkẹyìn INEC ti àwọn olùdíje ipò ààrẹ fún ìdìbò 16 Oṣù Kínní 2027 (ẹgbẹ́ 19, ní ìlànà INEC, orúkọ ìdílé "
          "ṣáájú). Ìdìbò kì í pín àwòrán; wo àkójọ INEC níbi orísun ìsàlẹ̀.",
    "ha": "Jerin ƙarshe na INEC na 'yan takarar shugaban ƙasa na zaɓen 16 ga Janairu 2027 (jam'iyyu 19, bisa tsarin INEC, "
          "sunan iyali farko). Ìdìbò ba ya raba hotuna; duba jerin INEC a tushen da ke ƙasa.",
    "ig": "Ndepụta ikpeazụ INEC nke ndị na-azọ ọkwa onye isi ala maka ntuli aka nke 16 Jenụwarị 2027 (otu 19, n'usoro "
          "INEC, aha nna mbụ). Ìdìbò anaghị ekesa foto; lee ndepụta INEC n'isi mmalite dị n'okpuru.",
}
ADVICE = re.compile(r"\bshould\b|vote for|better|best|trust|support|recommend|endorse|good|bad|win|yẹ kí|ya kamata|kwesịrị|"
                    r"suppose vote|make I vote", re.I)  # advice and predictions go to the model, which stays neutral
MATE = {"en": "running mate", "pcm": "running mate", "yo": "igbákejì", "ha": "mataimaki", "ig": "osote"}


def official_list(question: str, english: str, raw_hits: list[dict], lang: str) -> str | None:
    text = f"{question} {english}"
    if not any(h["id"].startswith(LIST_PREFIX) for h in raw_hits) or not LIST_CUE.search(text) or ADVICE.search(text):
        return None
    parts = [d for d in kb().docs if d["id"].startswith(LIST_PREFIX)]
    lines, where = [], {}
    for d in parts:
        for l in d["text"].split("\n")[1:]:
            lines.append(l)
            where[l] = d["id"]
    parties = {l.split(":")[0] for l in lines}
    words = set(re.findall(r"[A-Za-z]+", f"{question} {english}"))
    asked = parties & {w.upper() for w in words if w.upper() in parties and (w.isupper() or len(w) > 3)}
    if asked:  # about specific parties: the same verbatim lines, so every party is answered the same way
        lines = [l for l in lines if l.split(":")[0] in asked]
    body = "\n".join("- " + l.replace("running mate", MATE.get(lang, "running mate")) for l in lines)
    head = LIST_HEAD.get(lang, LIST_HEAD["en"])
    if asked:
        head = head.split(" (")[0] + ":"
    cites = " ".join(f"[{i}]" for i in dict.fromkeys(where[l] for l in lines))
    return f"{head}\n\n{body}\n\n{cites}"


PIDGIN = re.compile(r"\b(dey|wetin|una|wey|abeg|no get|make you|e go|dem|don|fit)\b", re.I)


# The fine-tune mirrors the question's style, so a Pidgin-flavoured question gets its Pidgin "I don't know" even when
# English is selected; it also won't translate Pidgin to English (it copies it). Its standard abstention is swapped for
# the English one; other Pidgin answers are left as they are (a known limitation).
ABSTAIN_EN = ("I don't have that information. Official election data, including future results, detailed vote counts, "
              "and personal PVC status, are only available from INEC. Please visit inecnigeria.org, use the polling "
              "unit locator, or check the IReV portal for the latest information.")


def looks_pidgin(text: str) -> bool:
    return len(PIDGIN.findall(text)) >= 2


_ALLOWED = None


def allowed_contacts() -> dict:
    global _ALLOWED
    if _ALLOWED is None:
        _ALLOWED = contacts.allowed_from_kb(ROOT / "data/kb/chunks.jsonl")
    return _ALLOWED


@app.post("/api/ask")
async def ask(req: Ask):
    q = req.question.strip()[:1200]
    if not q:
        raise HTTPException(400, "empty question")
    if req.mode == "check":
        q = CHECK_PREFIX.get(req.lang, CHECK_PREFIX["en"]) + q
    english = ""
    if req.lang in PIVOT_LANGS and BACKEND != "cohere":  # the sources are English: search with a translation too
        try:
            english = clean_pivot(await complete(pivot_messages(req.question.strip()[:600], req.lang)))
        except Exception:
            english = ""
    probe = kb().search(search_query(q, english), k=8)  # the list may rank just below the top 4 (e.g. after the pivot)
    raw_hits = probe[:4]
    verbatim = official_list(req.question, english, probe, req.lang) if req.mode == "ask" else None
    if verbatim:
        listed = [d for d in kb().docs if d["id"].startswith(LIST_PREFIX)]
        hits = listed + [h for h in raw_hits if not h["id"].startswith(LIST_PREFIX)][: 4 - len(listed)]
    else:
        hits = complete_lists(kb(), raw_hits)
    user = f"SOURCES:\n{format_context(hits)}\n\nQUESTION:\n{q}"  # the prompt format evaluated in the benchmarks
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    sources = [{"id": h["id"], "title": h["title"], "section": h.get("section", ""), "url": h["url"]} for h in hits]

    async def events():
        yield f"data: {json.dumps({'type': 'meta', 'backend': gen().label, 'sources': sources})}\n\n"
        full = ""
        t = time.time()
        try:
            if verbatim:  # an official list: shown as published, not generated
                full = verbatim
                yield f"data: {json.dumps({'type': 'token', 'text': verbatim})}\n\n"
            else:
                held = guard.has_cues(req.question)  # possible injection: check the reply before showing it
                async for piece in gen().stream(messages):
                    full += piece
                    if not held:
                        yield f"data: {json.dumps({'type': 'token', 'text': piece})}\n\n"
                fixed = guard.check(req.question, full, req.mode, req.lang)
                if fixed:
                    full = fixed
                full, invented = contacts.check(full, allowed_contacts())  # web addresses / numbers not in the sources
                if held:
                    yield f"data: {json.dumps({'type': 'token', 'text': full})}\n\n"
                elif fixed or invented:
                    yield f"data: {json.dumps({'type': 'replace', 'text': full})}\n\n"
        except Exception as e:  # keep the UI alive and honest
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)[:200]})}\n\n"
        if req.lang == "en" and not verbatim and full.strip().lower().startswith("i no get that information"):
            full = ABSTAIN_EN
            yield f"data: {json.dumps({'type': 'replace', 'text': full})}\n\n"
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
    if len(text) < 20:  # YarnGPT's minimum is 20 characters; trailing pause marks are silent
        text = text.ljust(19) + "......"
    voice = VOICE.get(req.lang, "femi")
    h = hashlib.sha256(f"{voice}|{req.lang}|{text}".encode()).hexdigest()[:24]
    path = CACHE / f"{h}.mp3"
    if not path.exists():
        hdr = {"Authorization": f"Bearer {key}", "Idempotency-Key": f"idibo-{h}"}
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{YARN}/tts", headers=hdr,
                             json={"text": text, "voice": voice, "output_format": "mp3"})  # YarnGPT infers the language
            if r.status_code >= 400:
                raise HTTPException(502, "voice service error")
            job = r.json()
            for _ in range(150):  # poll often: most short chunks are ready in 1-3 s
                s = (await c.get(f"{YARN}/status/{job['job_id']}", headers=hdr)).json()
                if s.get("status") in ("completed", "failed"):
                    break
                await asyncio.sleep(0.4)
            if s.get("status") != "completed":
                raise HTTPException(502, "voice not ready")
            audio = await c.get(s["audio_url"])
            path.write_bytes(audio.content)
    return FileResponse(path, media_type="audio/mpeg")


# ------------------------------------------------------------------ voice in
# Local fallback only where it works: Whisper turned a Yorùbá test clip into Telugu-script gibberish, and it has no
# Igbo. For yo/ha/ig, voice input needs YarnGPT's Nigerian ASR; otherwise we say so instead of guessing.
WHISPER_LANG = {"en": "en", "pcm": "en"}
_whisper = None


def whisper_text(path: str, lang: str) -> str:
    """Local fallback: faster-whisper (multilingual 'small', int8 on CPU)."""
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel

        # local_files_only: never download inside a request (fetch the model ahead, e.g. in the Dockerfile)
        _whisper = WhisperModel(os.environ.get("IDIBO_WHISPER", "small"), device="cpu", compute_type="int8",
                                local_files_only=True)
    # Decode with ffmpeg (16 kHz mono float32): avoids PyAV, whose current releases break faster-whisper's loader.
    import subprocess

    import numpy as np

    pcm = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, check=True).stdout
    audio = np.frombuffer(pcm, dtype=np.float32)
    segs, _ = _whisper.transcribe(audio, language=WHISPER_LANG[lang], beam_size=1, vad_filter=True)
    return " ".join(s.text.strip() for s in segs).strip()


def _find_text(j) -> str | None:
    if isinstance(j, dict):
        for k in ("text", "transcript", "transcription"):
            if isinstance(j.get(k), str):
                return j[k]
        for v in j.values():
            if (t := _find_text(v)) is not None:
                return t
    return None


@app.post("/api/stt")
async def stt(file: UploadFile = File(...), lang: str = Form("en")):
    data = await file.read()
    if not data or len(data) > 8_000_000:
        raise HTTPException(400, "empty or too long")
    return await transcribe(data, file.filename or "q.webm", file.content_type or "audio/webm", lang)


async def transcribe(data: bytes, filename: str, content_type: str, lang: str) -> dict:
    h = hashlib.sha256(data).hexdigest()[:24]
    key = env_key("YARNGPT_API_KEY")
    if key:  # YarnGPT ASR first (Nigerian languages); any failure (e.g. no credits) falls back to Whisper
        try:
            hdr = {"Authorization": f"Bearer {key}", "Idempotency-Key": f"idibo-asr-{h}"}
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post(f"{YARN}/asr", headers=hdr, files={"file": (filename, data, content_type)})
                j = r.json() if r.status_code < 400 else None
                if j and _find_text(j) is None and j.get("job_id"):
                    for _ in range(60):  # ASR jobs report at /asr/{id} (not /status/{id}, which is TTS-only)
                        await asyncio.sleep(0.5)
                        j = (await c.get(f"{YARN}/asr/{j['job_id']}", headers=hdr)).json()
                        if j.get("status") in ("completed", "failed"):
                            break
                if j and (t := _find_text(j)):
                    return {"text": t.strip(), "engine": "yarngpt"}
        except Exception:
            pass
    if lang not in WHISPER_LANG:
        raise HTTPException(503, "speech recognition for this language needs the YarnGPT service")
    suffix = Path(filename).suffix or ".webm"
    tmp = CACHE / f"stt-{h}{suffix}"
    tmp.write_bytes(data)
    try:
        text = await asyncio.get_running_loop().run_in_executor(None, whisper_text, str(tmp), lang)
    except Exception:
        raise HTTPException(503, "speech recognition is not available right now")
    finally:
        tmp.unlink(missing_ok=True)
    return {"text": text, "engine": "whisper"}


# ------------------------------------------------------------------ WhatsApp (Twilio)
# Election rumours in Nigeria travel on WhatsApp, so Ìdìbò answers there too: send a question (text or voice note) or
# forward a message to check it. The reply goes through the same pipeline as the web app (sources, injection guard,
# contact check), with numbered sources and, when the user sent a voice note, a spoken answer.
# Twilio waits at most 15 s for a webhook, and an answer takes longer, so the webhook returns at once and the reply is
# sent with Twilio's REST API.
PUBLIC_URL = os.environ.get("IDIBO_PUBLIC_URL", "https://idibo-2027.up.railway.app")
WA_LANG: dict[str, str] = {}  # sender -> chosen language (memory only; nothing about users is stored on disk)
LANG_WORDS = {"en": ("english", "en"), "pcm": ("pidgin", "naija", "pcm"), "yo": ("yoruba", "yorùbá", "yo"),
              "ha": ("hausa", "ha"), "ig": ("igbo", "ig")}
WA_HELP = ("Ìdìbò: voter information for Nigeria's 2027 election, from INEC and the Electoral Act 2026.\n\n"
           "• Ask a question, by text or voice note.\n• Forward a message you doubt, and I'll check it.\n"
           "• Change language: send english, pidgin, yoruba, hausa or igbo.\n\n"
           "Not official INEC guidance. I never tell anyone who to vote for.")


def detect_lang(text: str) -> str:
    t = text.lower()
    if re.search(r"[ƙɗɓ]", t) or re.search(r"\b(ina|yaya|zaɓe|zabe|shugaban|kuri'?a|ƙuri)\b", t):
        return "ha"
    if re.search(r"\b(nke|bụ|ntuli|ọ bụ|kedu|gị|m na)\b", t):
        return "ig"
    if re.search(r"[ẹọṣ]|[àèìòù]", t) or re.search(r"\b(ìdìbò|idibo ni|nígbà|níbo|báwo|kí ni)\b", t):
        return "yo"
    if looks_pidgin(t):
        return "pcm"
    return "en"


def twilio_ok(sig: str, params: dict, path: str = "/api/whatsapp") -> bool:
    import base64
    import hmac

    token = env_key("TWILIO_AUTH_TOKEN")
    if not token:
        return False
    data = PUBLIC_URL + path + "".join(k + params[k] for k in sorted(params))
    want = base64.b64encode(hmac.new(token.encode(), data.encode(), hashlib.sha1).digest()).decode()
    return hmac.compare_digest(want, sig or "")


async def wa_send(to: str, frm: str, body: str = "", media: str | None = None) -> None:
    sid, token = env_key("TWILIO_ACCOUNT_SID"), env_key("TWILIO_AUTH_TOKEN")
    data = {"To": to, "From": frm, "Body": body[:1550]}
    if media:
        data["MediaUrl"] = media
    async with httpx.AsyncClient(timeout=30, auth=(sid, token)) as c:
        r = await c.post(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json", data=data)
        if r.status_code >= 400:
            print("twilio send failed", r.status_code, r.text[:200], flush=True)


async def answer_text(question: str, lang: str, mode: str) -> tuple[str, list[dict]]:
    """Run the web app's /api/ask pipeline and collect the final answer and its sources."""
    resp = await ask(Ask(question=question, lang=lang, mode=mode))
    full, sources = "", []
    async for chunk in resp.body_iterator:
        for line in (chunk if isinstance(chunk, str) else chunk.decode()).splitlines():
            if not line.startswith("data: "):
                continue
            e = json.loads(line[6:])
            if e["type"] == "meta":
                sources = e["sources"]
            elif e["type"] == "token":
                full += e["text"]
            elif e["type"] == "replace":
                full = e["text"]
    return full.strip(), sources


def wa_format(answer: str, sources: list[dict]) -> str:
    ids = {s["id"]: s for s in sources}
    order: list[str] = []

    def num(m):
        sid = m.group(1)
        if sid not in ids:
            return ""
        if sid not in order:
            order.append(sid)
        return f"[{order.index(sid) + 1}]"

    text = re.sub(r"\s*\[([a-z0-9-]+)\]", lambda m: " " + num(m) if num(m) else "", answer)
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)  # WhatsApp bold
    lines = [text]
    if order:
        lines.append("\n" + "\n".join(f"[{i + 1}] {ids[s]['title']}" + (f", {ids[s]['section']}" if ids[s].get('section') else "")
                                       + f"\n{ids[s]['url']}" for i, s in enumerate(order)))
    lines.append("\n_Ìdìbò · not official INEC guidance · inecnigeria.org_")
    return "\n".join(lines)


async def wa_reply(form: dict, audio: bytes | None = None) -> tuple[str, str | None]:
    """The reply to one incoming WhatsApp message: (text, optional voice-note URL)."""
    user = form.get("From", "")
    text = (form.get("Body") or "").strip()
    voice_in = audio is not None or (int(form.get("NumMedia") or 0) > 0 and (form.get("MediaContentType0") or "").startswith("audio"))
    lang = WA_LANG.get(user)
    low = text.lower().strip(" .!")
    for code, words in LANG_WORDS.items():
        if low in words:
            WA_LANG[user] = code
            return f"OK: {LANGS[code]}. " + I18N_OK.get(code, ""), None
    if low in ("hi", "hello", "help", "start", "menu", "") and not voice_in:
        return WA_HELP, None
    if voice_in:
        if audio is None:
            sid, token = env_key("TWILIO_ACCOUNT_SID"), env_key("TWILIO_AUTH_TOKEN")
            async with httpx.AsyncClient(timeout=60, auth=(sid, token), follow_redirects=True) as c:
                audio = (await c.get(form["MediaUrl0"])).content
        text = (await transcribe(audio, "voice.ogg", form.get("MediaContentType0") or "audio/ogg", lang or "en"))["text"]
    lang = lang or detect_lang(text)
    # A long or forwarded message is something to check; a short one is a question.
    mode = "check" if len(text) > 160 or re.match(r"(?i)(fwd|forwarded|check|is this (true|real))\b", text) else "ask"
    answer, sources = await answer_text(text, lang, mode)
    media = None
    if voice_in and answer:
        try:
            await tts(TTS(text=answer, lang=lang))
            clean = re.sub(r"[*#_`]", "", re.sub(r"\[[a-z0-9-]+\]", "", answer)).strip()[:1500]
            if len(clean) < 20:
                clean = clean.ljust(19) + "......"
            media = f"{PUBLIC_URL}/api/voice/" + hashlib.sha256(f"{VOICE.get(lang, 'femi')}|{lang}|{clean}".encode()).hexdigest()[:24] + ".mp3"
        except Exception as e:
            print("whatsapp voice failed", repr(e)[:200], flush=True)
    return wa_format(answer, sources), media


I18N_OK = {"en": "Ask your question.", "pcm": "Ask your question.", "yo": "Béèrè ìbéèrè rẹ.",
           "ha": "Yi tambayarka.", "ig": "Jụọ ajụjụ gị."}
# Replies go back in the webhook response (TwiML): Twilio's trial WhatsApp numbers refuse free-text messages sent
# through the REST API, but accept a reply to an incoming message. Twilio waits 15 s per request and an answer can take
# longer, so after 12 s the server answers <Redirect>; Twilio calls back and gets the answer when it is ready.
WA_JOBS: dict[str, asyncio.Task] = {}
WA_WAIT, WA_HOPS = 12.0, 8


def twiml(body: str = "", media: str | None = None, redirect: str | None = None) -> Response:
    from xml.sax.saxutils import escape

    if redirect:
        x = f'<Response><Redirect method="POST">{escape(redirect)}</Redirect></Response>'
    else:
        m = f"<Media>{escape(media)}</Media>" if media else ""
        x = f"<Response><Message><Body>{escape(body[:1590])}</Body>{m}</Message></Response>"
    # text/xml, and no-transform so Railway's edge does not gzip it (Twilio rejected the compressed reply: error 12300)
    return Response('<?xml version="1.0" encoding="UTF-8"?>' + x, media_type="text/xml; charset=utf-8",
                    headers={"Cache-Control": "no-transform"})


async def wa_step(sid: str, hop: int) -> Response:
    task = WA_JOBS.get(sid)
    if task is None:
        return twiml("Sorry, I lost track of that message. Please send it again.")
    try:
        text, media = await asyncio.wait_for(asyncio.shield(task), WA_WAIT)
    except asyncio.TimeoutError:
        if hop >= WA_HOPS:
            WA_JOBS.pop(sid, None)
            return twiml("Sorry, that is taking too long. Please try again in a minute.")
        return twiml(redirect=f"{PUBLIC_URL}/api/whatsapp/wait?sid={sid}&hop={hop + 1}")
    except Exception as e:
        print("whatsapp error", repr(e)[:300], flush=True)
        WA_JOBS.pop(sid, None)
        return twiml("Sorry, I couldn't answer that just now. Please try again in a minute.")
    WA_JOBS.pop(sid, None)
    return twiml(text, media)


@app.post("/api/whatsapp")
async def whatsapp(request: Request):
    form = {k: str(v) for k, v in (await request.form()).items()}
    if not twilio_ok(request.headers.get("X-Twilio-Signature", ""), form, "/api/whatsapp"):
        raise HTTPException(403, "bad signature")
    sid = form.get("MessageSid") or hashlib.sha256(json.dumps(form, sort_keys=True).encode()).hexdigest()[:20]
    if sid not in WA_JOBS:
        WA_JOBS[sid] = asyncio.create_task(wa_reply(form))
    return await wa_step(sid, 0)


@app.post("/api/whatsapp/wait")
async def whatsapp_wait(request: Request, sid: str, hop: int = 1):
    form = {k: str(v) for k, v in (await request.form()).items()}
    if not twilio_ok(request.headers.get("X-Twilio-Signature", ""), form, f"/api/whatsapp/wait?{request.url.query}"):
        raise HTTPException(403, "bad signature")
    return await wa_step(sid, hop)


# ------------------------------------------------------------------ WhatsApp (Meta Cloud API)
# The same bot on Meta's WhatsApp Cloud API: Meta calls /api/meta/whatsapp with each message; replies are sent with the
# Graph API (free text is allowed within 24 h of the user's last message). Needs META_WA_TOKEN, META_PHONE_NUMBER_ID,
# META_APP_SECRET (to check Meta's signature) and META_VERIFY_TOKEN (any string, also typed into Meta's webhook form).
GRAPH = "https://graph.facebook.com/v25.0"


async def meta_send(to: str, body: str = "", audio_url: str | None = None) -> None:
    token, pid = env_key("META_WA_TOKEN"), env_key("META_PHONE_NUMBER_ID")
    msg = ({"type": "audio", "audio": {"link": audio_url}} if audio_url
           else {"type": "text", "text": {"body": body[:4000], "preview_url": False}})
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{GRAPH}/{pid}/messages", headers={"Authorization": f"Bearer {token}"},
                         json={"messaging_product": "whatsapp", "to": to, **msg})
        if r.status_code >= 400:
            print("meta send failed", r.status_code, r.text[:300], flush=True)


async def meta_handle(m: dict) -> None:
    user = m.get("from", "")
    try:
        audio = None
        if m.get("type") == "audio":
            token = env_key("META_WA_TOKEN")
            async with httpx.AsyncClient(timeout=60, headers={"Authorization": f"Bearer {token}"}) as c:
                info = (await c.get(f"{GRAPH}/{m['audio']['id']}")).json()
                audio = (await c.get(info["url"])).content
        elif m.get("type") != "text":
            await meta_send(user, "Please send text or a voice note.")
            return
        body = (m.get("text") or {}).get("body", "")
        text, media = await wa_reply({"From": "meta:" + user, "Body": body}, audio)
        await meta_send(user, text)
        if media:
            await meta_send(user, audio_url=media)
    except Exception as e:
        print("meta whatsapp error", repr(e)[:300], flush=True)
        await meta_send(user, "Sorry, I couldn't answer that just now. Please try again in a minute.")


@app.get("/api/meta/whatsapp")
def meta_verify(request: Request):
    q = request.query_params
    want = env_key("META_VERIFY_TOKEN")
    if want and q.get("hub.mode") == "subscribe" and q.get("hub.verify_token") == want:
        return Response(q.get("hub.challenge", ""), media_type="text/plain")
    raise HTTPException(403, "verification failed")


@app.post("/api/meta/whatsapp")
async def meta_webhook(request: Request):
    import hmac

    raw = await request.body()
    secret = env_key("META_APP_SECRET")
    want = "sha256=" + hmac.new((secret or "").encode(), raw, hashlib.sha256).hexdigest()
    if not secret or not hmac.compare_digest(want, request.headers.get("X-Hub-Signature-256", "")):
        raise HTTPException(403, "bad signature")
    data = json.loads(raw or b"{}")
    for entry in data.get("entry", []):
        for ch in entry.get("changes", []):
            for m in ch.get("value", {}).get("messages", []) or []:
                if m.get("id") not in META_SEEN:  # Meta can deliver the same message twice
                    META_SEEN.add(m.get("id"))
                    asyncio.create_task(meta_handle(m))
    return {"ok": True}


META_SEEN: set = set()


@app.get("/api/voice/{name}")
def voice_file(name: str):
    if not re.fullmatch(r"[0-9a-f]{24}\.mp3", name) or not (CACHE / name).exists():
        raise HTTPException(404, "not found")
    return FileResponse(CACHE / name, media_type="audio/mpeg")


@app.get("/api/judge")
def judge_view():
    p = ROOT / "app" / "judge.json"
    return JSONResponse(json.loads(p.read_text()) if p.exists() else {"examples": [], "summary": {}})


@app.get("/api/health")
def health():
    info = {"ok": True, "backend": BACKEND, "langs": LANGS, "cpus": os.cpu_count()}
    if _gen is not None and hasattr(_gen, "threads"):
        info["threads"] = _gen.threads
    return info


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
