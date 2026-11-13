"""Create or update the Ìdìbò Hugging Face Space from this repo.

    python space/deploy.py            # reads HF_WRITE_TOKEN and YARNGPT_API_KEY from .env (never printed)

Uploads only the small app files (about 1 MB); the Space downloads the GGUF from the model repo when it starts.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent


def env(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    for line in open(ROOT / ".env", encoding="utf-8"):
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip() or None
    return None


def main() -> None:
    token = env("HF_WRITE_TOKEN")
    if not token:
        sys.exit("Add HF_WRITE_TOKEN=... (a Hugging Face *write* token) to .env first.")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    space = f"{user}/idibo"
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    if env("YARNGPT_API_KEY"):
        api.add_space_secret(space, "YARNGPT_API_KEY", env("YARNGPT_API_KEY"))
    api.add_space_variable(space, "IDIBO_GGUF_REPO", f"{user}/idibo-tiny-aya-earth")
    with tempfile.TemporaryDirectory() as out:
        subprocess.run(["bash", str(ROOT / "space/build.sh"), out], check=True)
        api.upload_folder(folder_path=out, repo_id=space, repo_type="space", commit_message="Deploy Ìdìbò")
    print(f"https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main()
