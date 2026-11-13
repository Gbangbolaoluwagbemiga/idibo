# Ìdìbò on a free CPU Hugging Face Space: the fine-tuned Tiny Aya Earth as Q4_K_M GGUF (llama.cpp), hybrid
# retrieval over INEC pages + the Electoral Act 2026, YarnGPT voice, Whisper fallback for English/Pidgin speech.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg build-essential cmake && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Built from source: the GGUF was converted with a current llama.cpp, so the runtime must be current too.
RUN CMAKE_ARGS="-DGGML_NATIVE=OFF" pip install --no-cache-dir "llama-cpp-python>=0.3.36"
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" httpx pydantic python-multipart numpy \
      sentence-transformers faster-whisper huggingface_hub
USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface PYTHONUNBUFFERED=1 \
    IDIBO_BACKEND=llamacpp IDIBO_GGUF_REPO=GOPAE/idibo-tiny-aya-earth IDIBO_GGUF_FILE=idibo-q4_k_m.gguf \
    IDIBO_CACHE=/tmp/idibo-cache
# Fetch the small models at build time (fast inside HF's network); the GGUF is fetched at start (start.sh).
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('intfloat/multilingual-e5-small'); \
from faster_whisper import WhisperModel as W; W('small', device='cpu', compute_type='int8')"
COPY --chown=user . .
EXPOSE 7860
CMD ["bash", "start.sh"]
