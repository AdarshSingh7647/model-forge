#!/usr/bin/env python3
"""
Usage:
    python download_models.py --models Qwen/Qwen3-8B Qwen/Qwen3-4B --cache_dir "$HF_CACHE_DIR" --load_check
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
from huggingface_hub import login

load_dotenv()  # loads variables from .env
hf_key = os.getenv("HF_KEY")
if hf_key:
    login(token=hf_key)

DEFAULT_CACHE_DIR = os.getenv(
    "HF_CACHE_DIR", os.path.expanduser("~/.cache/huggingface/hub")
)
DEFAULT_MODELS = ["Qwen/Qwen3-8B", "Qwen/Qwen3-4B"]


def parse_args():
    p = argparse.ArgumentParser(description="Download/cache Qwen3 models from HF Hub.")
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="HF model repo ids to download.",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Directory to use as HF_HOME/hub cache (snapshot_download target).",
    )
    p.add_argument(
        "--load_check",
        action="store_true",
        help="After downloading, do a quick tokenizer+config load to sanity check.",
    )
    p.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional specific revision/commit/branch to pin for all models.",
    )
    return p.parse_args()


def resolve_local_dir(cache_dir: str, repo_id: str) -> str:
    """
    Mirrors the layout LLaMA-Factory / transformers expects when you pass
    a local path as model_name_or_path: <cache_dir>/<org>__<name>
    (snapshot_download with local_dir gives a flat, directly-loadable folder,
    which is more convenient for LLaMA-Factory configs than the nested
    models--org--name/snapshots/<hash>/ blob layout used by the default cache.)
    """
    safe_name = repo_id.replace("/", "__")
    return os.path.join(cache_dir, safe_name)


def download_one(repo_id: str, cache_dir: str, revision: str | None) -> str:
    from huggingface_hub import snapshot_download

    local_dir = resolve_local_dir(cache_dir, repo_id)
    os.makedirs(local_dir, exist_ok=True)

    print(f"\n=== Downloading {repo_id} -> {local_dir} ===")
    snapshot_download(
        repo_id=repo_id,
        # revision=revision,
        local_dir=local_dir,
        # Avoid symlink farms on shared /scratch filesystems (some clusters
        # don't like symlinks across filesystems, and it keeps the dir
        # self-contained / easy to point LLaMA-Factory at directly).
        local_dir_use_symlinks=False,
        max_workers=8,
        # Skip files we don't need for training (e.g. .gguf, .onnx variants)
        # if the repo happens to ship them; keeps download lean.
        ignore_patterns=["*.gguf", "*.onnx", "*.msgpack", "*.h5", "*.tflite"],
    )
    print(f"=== Done: {repo_id} ===")
    return local_dir


def sanity_load_check(local_dir: str, repo_id: str):
    from transformers import AutoConfig, AutoTokenizer

    print(f"--- Sanity check: loading config+tokenizer for {repo_id} from {local_dir} ---")
    cfg = AutoConfig.from_pretrained(local_dir)
    tok = AutoTokenizer.from_pretrained(local_dir)
    print(f"    model_type={cfg.model_type}, hidden_size={getattr(cfg, 'hidden_size', 'NA')}, "
          f"num_hidden_layers={getattr(cfg, 'num_hidden_layers', 'NA')}, vocab_size={len(tok)}")


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    # Point HF caches at the scratch dir too, in case any auxiliary files
    # (datasets, etc.) get pulled implicitly during load checks.
    os.environ.setdefault("HF_HOME", args.cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", args.cache_dir)

    local_dirs = {}
    for repo_id in args.models:
        local_dir = resolve_local_dir(args.cache_dir, repo_id)
        marker = os.path.join(local_dir, "config.json")
        if os.path.exists(marker):
            print(f"[skip] {repo_id} already present at {local_dir}")
        else:
            local_dir = download_one(repo_id, args.cache_dir, args.revision)
        local_dirs[repo_id] = local_dir

        if args.load_check:
            sanity_load_check(local_dir, repo_id)

    print("\n=== Summary: local model paths (use these in LLaMA-Factory configs) ===")
    for repo_id, path in local_dirs.items():
        print(f"  {repo_id:20s} -> {path}")

    # Also dump a small json map other scripts (e.g. config generator) can read.
    import json
    map_path = os.path.join(args.cache_dir, "model_paths.json")
    with open(map_path, "w") as f:
        json.dump(local_dirs, f, indent=2)
    print(f"\nWrote model path map to {map_path}")


if __name__ == "__main__":
    sys.exit(main())