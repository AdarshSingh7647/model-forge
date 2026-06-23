#!/usr/bin/env python3
"""
Live status of all 8 LoRA SFT runs (2 models x 4 setups) launched by
configs/run_all.sh. Reads trainer_state.json (written by LLaMA-Factory/HF
Trainer every `save_steps`, plus a live in-progress one in the output dir
root) from each run's output_dir and prints current/total step, epoch,
and latest loss.

Usage: python training_status.py [--watch SECONDS]
"""

import argparse
import glob
import json
import os
import time

import yaml

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")


def discover_runs():
    runs = []
    for cfg_path in sorted(glob.glob(os.path.join(CONFIGS_DIR, "qwen3_*.yaml"))):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        name = os.path.splitext(os.path.basename(cfg_path))[0]
        runs.append({"name": name, "output_dir": cfg["output_dir"]})
    return runs


def read_state(output_dir):
    state_path = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        return None
    with open(state_path) as f:
        return json.load(f)


def format_row(run):
    state = read_state(run["output_dir"])
    if state is None:
        return f"{run['name']:35s}  not started"

    step = state.get("global_step", 0)
    max_steps = state.get("max_steps", 0)
    epoch = state.get("epoch", 0.0)
    log_history = state.get("log_history", [])
    last_loss = next((e["loss"] for e in reversed(log_history) if "loss" in e), None)
    pct = (100 * step / max_steps) if max_steps else 0.0
    loss_str = f"{last_loss:.4f}" if last_loss is not None else "n/a"
    done = step >= max_steps and max_steps > 0
    status = "DONE" if done else "running"
    return (f"{run['name']:35s}  step {step:5d}/{max_steps:<5d} ({pct:5.1f}%)  "
            f"epoch {epoch:5.2f}  loss {loss_str:8s}  {status}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--watch", type=int, default=0, help="Refresh every N seconds instead of printing once.")
    args = p.parse_args()

    runs = discover_runs()
    while True:
        if args.watch:
            os.system("clear")
        print(f"=== Training status ({len(runs)} runs) ===")
        for run in runs:
            print(format_row(run))
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
