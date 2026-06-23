#!/usr/bin/env bash
#
# Rewrite machine-specific ABSOLUTE paths in the generated / git-ignored files
# so a moved-or-fresh checkout never fails on stale paths. Idempotent — safe to
# run any number of times; it always rewrites to THIS checkout + the model cache
# from .env.
#
# Targets:
#   LLaMA-Factory/data/dataset_info.json   (dataset file_name paths)
#   configs/*.yaml                         (model_name_or_path, output_dir)
#   eval_runs/*/predict.yaml               (model/adapter/output paths)
#
# Anchored rewrites (prefix before each anchor is replaced):
#   .../training_data/  .../lora_runs/  .../eval_runs/   -> $REPO_DIR/<anchor>/
#   .../Qwen__<name>                                     -> $MODEL_CACHE_DIR/Qwen__<name>
#
# Run standalone:  bash fix_paths.sh
# (also invoked automatically at the top of run_all.sh and eval/run_eval.sh)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; [ -f "$REPO_DIR/.env" ] && . "$REPO_DIR/.env"; set +a
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${HF_CACHE_DIR:-}}"

rewrite_file() {
  local f="$1"
  [ -f "$f" ] || return 0
  # in-repo dirs: replace whatever absolute prefix precedes the anchor
  sed -i -E 's#[^"[:space:]]*/(training_data|lora_runs|eval_runs)/#'"$REPO_DIR"'/\1/#g' "$f"
  # base-model dir: replace prefix before Qwen__<name> with the model cache
  if [ -n "$MODEL_CACHE_DIR" ]; then
    sed -i -E 's#[^"[:space:]]*/(Qwen__[A-Za-z0-9._-]+)#'"$MODEL_CACHE_DIR"'/\1#g' "$f"
  fi
}

shopt -s nullglob
n=0
for f in \
  "$REPO_DIR/LLaMA-Factory/data/dataset_info.json" \
  "$REPO_DIR/configs/"*.yaml \
  "$REPO_DIR/eval_runs/"*/predict.yaml
do
  [ -f "$f" ] || continue
  rewrite_file "$f"
  n=$((n + 1))
done

echo "[fix_paths] repo=$REPO_DIR  model_cache=${MODEL_CACHE_DIR:-<unset>}  files_fixed=$n"
