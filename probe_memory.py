#!/usr/bin/env python3
"""One-off probe: measure real peak GPU memory for one LoRA SFT training
step at cutoff_len=32768, matching configs/qwen3_*.yaml settings, with no
flash-attn package installed (falls back to PyTorch sdpa kernel)."""

import argparse
import gc
import os

import torch
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

load_dotenv()

# Model cache dir comes from the environment so no machine-specific path is
# baked in. download_models.py stores models as <cache>/<org>__<name>.
_CACHE = os.getenv("MODEL_CACHE_DIR") or os.getenv(
    "HF_CACHE_DIR", os.path.expanduser("~/.cache/huggingface/hub")
)
MODELS = {
    "8b": (os.path.join(_CACHE, "Qwen__Qwen3-8B"), 2),
    "4b": (os.path.join(_CACHE, "Qwen__Qwen3-4B"), 4),
}


def probe(model_path, batch_size, seq_len):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    model.train()

    after_load_mb = torch.cuda.max_memory_allocated() / 1024**2

    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device="cuda")
    labels = input_ids.clone()

    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    reserved_mb = torch.cuda.max_memory_reserved() / 1024**2

    print(f"  after model+LoRA load: {after_load_mb/1024:.1f} GB")
    print(f"  peak allocated (fwd+bwd, batch={batch_size}, seq_len={seq_len}): {peak_mb/1024:.1f} GB")
    print(f"  peak reserved: {reserved_mb/1024:.1f} GB")

    del model, out, input_ids, labels
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=list(MODELS.keys()))
    p.add_argument("--seq_len", type=int, default=32768)
    args = p.parse_args()

    path, bs = MODELS[args.model]
    print(f"=== {args.model} (batch_size={bs}, seq_len={args.seq_len}) ===")
    probe(path, bs, args.seq_len)
