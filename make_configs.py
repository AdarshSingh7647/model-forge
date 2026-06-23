#!/usr/bin/env python3
"""
Generate LLaMA-Factory LoRA SFT YAML configs for all (setup x model)
combinations: 4 setups x {Qwen3-8B, Qwen3-4B} = 8 configs.

Assumes:
  - build_sft_data.py has already been run with --llamafactory_data_dir
    pointing at LLaMA-Factory's data/ directory, so dataset_info.json
    contains entries named exactly as in SETUP_DATASETS below.
  - download_models.py has been run, and model_paths.json (in the HF cache
    dir) maps repo_id -> local checkpoint dir. We read that map here so
    configs always point at the correct local path.

Usage:
    python make_configs.py \
        --model_cache_dir "$MODEL_CACHE_DIR" \
        --out_dir ./configs \
        --output_root ./lora_runs
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # pick up paths from .env if present

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR") or os.getenv(
    "HF_CACHE_DIR", os.path.expanduser("~/.cache/huggingface/hub")
)
DEFAULT_OUTPUT_ROOT = str(PROJECT_ROOT / "lora_runs")

# Map setup name (matches dataset names registered by build_sft_data.py)
# to whether it needs mask_history, and a short human label for run naming.
SETUP_DATASETS = {
    "setup1_full_loss": {
        "label": "full_loss",
        "mask_history": False,
        "description": "Loss on thinking+final_answer",
    },
    "setup2_answer_only": {
        "label": "answer_only",
        "mask_history": True,
        "description": "Loss on final_answer only (thinking turn masked)",
    },
    "setup3_1_hint_answer_only": {
        "label": "hint_answer_only",
        "mask_history": False,
        "description": "Thinking given as hint in prompt; loss on final_answer only",
    },
    "setup3_2_hint_full_loss": {
        "label": "hint_full_loss",
        "mask_history": False,
        "description": "Thinking given as hint in prompt; loss on thinking+final_answer",
    },
}

MODEL_REPOS = {
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_4b": "Qwen/Qwen3-4B",
}

# Per-model defaults; 8B needs a slightly smaller per-device batch / more
# grad accumulation on a single A100-80GB than 4B does, to stay safely
# within memory with LoRA + bf16 + reasonable sequence length.
MODEL_TRAIN_DEFAULTS = {
    "qwen3_8b": {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "cutoff_len": 4096,
    },
    "qwen3_4b": {
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "cutoff_len": 4096,
    },
}

LORA_DEFAULTS = {
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target": "all",
}


def build_config(
    model_tag: str,
    model_path: str,
    setup_name: str,
    output_root: str,
    num_gpus: int,
) -> dict:
    setup_meta = SETUP_DATASETS[setup_name]
    train_defaults = MODEL_TRAIN_DEFAULTS[model_tag]
    run_name = f"{model_tag}_{setup_meta['label']}"
    output_dir = os.path.join(output_root, run_name)

    cfg = {
        "### model": None,
        "model_name_or_path": model_path,
        "trust_remote_code": True,

        "### method": None,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        **LORA_DEFAULTS,

        "### dataset": None,
        "dataset": f"{setup_name}_train",
        # only set eval dataset if a _val split exists; build_sft_data.py
        # only writes one when val_fraction>0 and there are enough rows.
        # We reference by convention; if missing, LLaMA-Factory will error,
        # so we leave a commented hint via the report script instead of
        # forcing it here (see generate_run_script for handling).
        "template": "qwen3",
        "cutoff_len": train_defaults["cutoff_len"],
        "max_samples": None,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "dataloader_num_workers": 4,

        "### mask_history": None,
        "mask_history": setup_meta["mask_history"],

        "### output": None,
        "output_dir": output_dir,
        "logging_steps": 10,
        "save_steps": 200,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "report_to": "none",

        "### train": None,
        "per_device_train_batch_size": train_defaults["per_device_train_batch_size"],
        "gradient_accumulation_steps": train_defaults["gradient_accumulation_steps"],
        "learning_rate": 1.0e-4,
        "num_train_epochs": 3.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": True,
        "ddp_timeout": 180000000,

        "### eval": None,
        "val_size": 0.0,  # we pre-split val files ourselves; set per-run below if val exists
        "per_device_eval_batch_size": train_defaults["per_device_train_batch_size"],
        "eval_strategy": "no",
    }
    return cfg


def write_yaml(cfg: dict, path: str):
    """
    Hand-roll YAML writing (no external pyyaml dependency required) since
    LLaMA-Factory's config schema is flat key:value with occasional
    "### comment" section markers (None-valued keys -> rendered as comments).
    """
    lines = []
    for k, v in cfg.items():
        if k.startswith("###"):
            lines.append(f"\n# {k.replace('###', '').strip()}")
            continue
        if v is None:
            continue
        if isinstance(v, bool):
            v_str = "true" if v else "false"
        elif isinstance(v, str):
            v_str = v
        else:
            v_str = str(v)
        lines.append(f"{k}: {v_str}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Generate LLaMA-Factory LoRA configs for 4 setups x 2 models.")
    p.add_argument("--model_cache_dir", default=DEFAULT_MODEL_CACHE_DIR,
                    help="Where download_models.py wrote model_paths.json")
    p.add_argument("--out_dir", default=str(PROJECT_ROOT / "configs"))
    p.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT,
                    help="Root dir where each run's adapter checkpoints get written.")
    p.add_argument("--num_gpus", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model_paths_file = os.path.join(args.model_cache_dir, "model_paths.json")
    if os.path.exists(model_paths_file):
        with open(model_paths_file) as f:
            repo_to_path = json.load(f)
    else:
        print(f"[warn] {model_paths_file} not found -- did you run download_models.py? "
              f"Falling back to constructing expected paths.")
        repo_to_path = {
            repo: os.path.join(args.model_cache_dir, repo.replace("/", "__"))
            for repo in MODEL_REPOS.values()
        }

    written_paths = []
    for model_tag, repo_id in MODEL_REPOS.items():
        model_path = repo_to_path.get(repo_id)
        if model_path is None:
            print(f"[error] No local path found for {repo_id} in {model_paths_file}", file=sys.stderr)
            sys.exit(1)

        for setup_name in SETUP_DATASETS:
            cfg = build_config(model_tag, model_path, setup_name, args.output_root, args.num_gpus)
            fname = f"{model_tag}_{SETUP_DATASETS[setup_name]['label']}.yaml"
            fpath = os.path.join(args.out_dir, fname)
            write_yaml(cfg, fpath)
            written_paths.append(fpath)
            print(f"[config] wrote {fpath}")

    # Also emit a single shell script that runs every config sequentially
    # (safe default for shared multi-GPU nodes where you don't want 8
    # concurrent torchrun jobs fighting for memory/IO).
    run_all_path = os.path.join(args.out_dir, "run_all.sh")
    with open(run_all_path, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        f.write("# Sequentially launches all 8 LoRA SFT runs (4 setups x 2 models).\n")
        f.write("# Edit NUM_GPUS / launcher below if you want multi-GPU per run.\n\n")
        f.write("SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n\n")
        for fpath in written_paths:
            fname = os.path.basename(fpath)
            f.write(f'echo "=== Running {fname} ==="\n')
            f.write(f'llamafactory-cli train "$SCRIPT_DIR/{fname}"\n\n')
    os.chmod(run_all_path, 0o755)
    print(f"\n[script] wrote {run_all_path} (runs all 8 configs sequentially)")

    print(f"\nTotal configs generated: {len(written_paths)}")


if __name__ == "__main__":
    sys.exit(main())