#!/usr/bin/env bash
set -euo pipefail

# Sequentially launch LoRA SFT runs, each on BOTH GPUs via torchrun/DDP.
# Because every run uses both GPUs, runs cannot overlap -- they go one after
# another. Each run auto-resumes from its latest checkpoint (the configs set
# overwrite_output_dir: false), so re-running this script after a crash picks
# up where it stopped.
#
# Usage:
#   bash run_all.sh                       # run the default set (all 8) in order
#   bash run_all.sh qwen3_8b_answer_only qwen3_4b_full_loss   # only these
#   nohup bash run_all.sh > logs/run_all_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# IMPORTANT: do NOT start this while another training job is using the GPUs
# (e.g. the manually-launched qwen3_8b_full_loss run). The first config here is
# qwen3_8b_full_loss -- if that one is already training/finished, either wait
# for the GPUs to free up or pass an explicit list that omits it.

# Repo root = parent of this script's dir (this script lives in configs/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LF_DIR="$ROOT/LLaMA-Factory"          # must run from here so LLaMA-Factory finds data/dataset_info.json
LOG_DIR="$ROOT/logs"

# Load optional env (TRAINER_BIN, etc.) from the repo's .env if present.
set -a; [ -f "$ROOT/.env" ] && . "$ROOT/.env"; set +a

# Put the trainer venv first so `torchrun` and `llamafactory-cli` resolve to it
# (only if TRAINER_BIN is set; otherwise rely on whatever is on PATH).
if [ -n "${TRAINER_BIN:-}" ]; then export PATH="$TRAINER_BIN:$PATH"; fi
# FORCE_TORCHRUN=1 makes llamafactory-cli launch under torchrun: one worker
# process per visible GPU (data-parallel / DDP) instead of a single process.
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(
    qwen3_8b_full_loss
    qwen3_8b_answer_only
    qwen3_8b_hint_answer_only
    qwen3_8b_hint_full_loss
    qwen3_4b_full_loss
    qwen3_4b_answer_only
    qwen3_4b_hint_answer_only
    qwen3_4b_hint_full_loss
  )
fi

mkdir -p "$LOG_DIR"
cd "$LF_DIR"

echo "GPUs=$CUDA_VISIBLE_DEVICES  FORCE_TORCHRUN=$FORCE_TORCHRUN  runs=${CONFIGS[*]}"
for name in "${CONFIGS[@]}"; do
  cfg="$ROOT/configs/${name}.yaml"
  [ -f "$cfg" ] || { echo "!! config not found: $cfg" >&2; exit 1; }
  ts="$(date +%Y%m%d_%H%M%S)"
  log="$LOG_DIR/${name}_${ts}.log"
  echo "=== $(date '+%F %T')  training ${name} on GPUs ${CUDA_VISIBLE_DEVICES} -> ${log} ==="
  llamafactory-cli train "$cfg" 2>&1 | tee "$log"
done
echo "=== $(date '+%F %T')  all runs complete ==="
