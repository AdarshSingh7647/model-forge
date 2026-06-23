#!/usr/bin/env bash
# Generate predictions on a setup's held-out *_val set with a trained LoRA
# adapter, then score them (EM / RelaxedEM / F1 / Top1 / KendallTau).
#
# Everything is derived from the matching TRAINING config so this never drifts:
#   - base model      <- model_name_or_path
#   - val dataset     <- dataset (the train name with _train -> _val)
#   - adapter dir     <- output_dir   (override with $2)
#
# Usage:
#   eval/run_eval.sh <config_name> [adapter_path] [max_new_tokens]
# Examples:
#   eval/run_eval.sh qwen3_8b_full_loss
#   eval/run_eval.sh qwen3_8b_answer_only lora_runs/qwen3_8b_answer_only/checkpoint-200
#
# NOTE: generation here uses HuggingFace (no vLLM in this env) and is SLOW for
# the *thinking* setups (full chain-of-thought per example). The answer-only
# setups are fast. Runs fully independently of training -- safe to run on GPU 1
# while training occupies GPU 0:  CUDA_VISIBLE_DEVICES=1 eval/run_eval.sh ...
set -euo pipefail

# Repo root = parent of this script's dir (this script lives in eval/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LF_DIR="$ROOT/LLaMA-Factory"

# Load optional env (TRAINER_BIN) from the repo's .env if present, then resolve
# the python / llamafactory-cli binaries from it (fall back to PATH).
set -a; [ -f "$ROOT/.env" ] && . "$ROOT/.env"; set +a
if [ -n "${TRAINER_BIN:-}" ]; then
  PY="$TRAINER_BIN/python"; CLI="$TRAINER_BIN/llamafactory-cli"
else
  PY="python"; CLI="llamafactory-cli"
fi

# Pin to ONE GPU so this stays single-process (no DDP) and does not collide with
# a training job. Defaults to GPU 1 (training uses GPU 0); override by exporting
# CUDA_VISIBLE_DEVICES before calling.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export FORCE_TORCHRUN=0

CONFIG_NAME="${1:?usage: run_eval.sh <config_name> [adapter_path] [max_new_tokens]}"
CONFIG="$ROOT/configs/${CONFIG_NAME}.yaml"
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }

# --- derive fields from the training config (no extra yaml libs needed) ---
read -r BASE_MODEL TRAIN_DS OUT_DIR <<EOF
$($PY - "$CONFIG" <<'PYEOF'
import sys, re
cfg = open(sys.argv[1]).read()
def get(k):
    m = re.search(rf'^{k}:\s*(.+?)\s*(?:#.*)?$', cfg, flags=re.M)
    return m.group(1).strip() if m else ""
print(get("model_name_or_path"), get("dataset"), get("output_dir"))
PYEOF
)
EOF

VAL_DS="${TRAIN_DS/_train/_val}"
ADAPTER="${2:-$OUT_DIR}"
# thinking setups emit a full CoT -> need many tokens; answer-only setups are short
if [[ "$CONFIG_NAME" == *answer_only* && "$CONFIG_NAME" != *hint_full* ]]; then
  DEFAULT_MAXNEW=128
else
  DEFAULT_MAXNEW=4096
fi
MAXNEW="${3:-$DEFAULT_MAXNEW}"

PRED_DIR="$ROOT/eval_runs/${CONFIG_NAME}"
mkdir -p "$PRED_DIR"
PRED_YAML="$PRED_DIR/predict.yaml"

echo "config      : $CONFIG_NAME"
echo "base model  : $BASE_MODEL"
echo "adapter     : $ADAPTER"
echo "val dataset : $VAL_DS"
echo "max_new_tok : $MAXNEW"
echo "out dir     : $PRED_DIR"

if [ ! -f "$ADAPTER/adapter_config.json" ] && [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
  echo "ERROR: no LoRA adapter weights in: $ADAPTER" >&2
  echo "(training has not saved an adapter there yet -- train first, or pass a" >&2
  echo " checkpoint-XX dir as arg 2, e.g. $OUT_DIR/checkpoint-20)" >&2
  exit 1
fi
echo "GPU(s)      : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cat > "$PRED_YAML" <<EOF
### model
model_name_or_path: $BASE_MODEL
adapter_name_or_path: $ADAPTER
trust_remote_code: true
flash_attn: fa2

### method
stage: sft
do_predict: true
finetuning_type: lora

### dataset
eval_dataset: $VAL_DS
template: qwen3
cutoff_len: 32768
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: $PRED_DIR
overwrite_output_dir: true
report_to: none

### eval / generation
per_device_eval_batch_size: 1
predict_with_generate: true
max_new_tokens: $MAXNEW
temperature: 0.0
do_sample: false
ddp_timeout: 180000000
EOF

echo "=== generating predictions (this can take a while) ==="
( cd "$LF_DIR" && "$CLI" train "$PRED_YAML" )

PREDS="$PRED_DIR/generated_predictions.jsonl"
[ -f "$PREDS" ] || { echo "ERROR: predictions not produced at $PREDS" >&2; exit 1; }

echo "=== scoring ==="
"$PY" "$ROOT/eval/ranking_metrics.py" \
  --pred_file "$PREDS" \
  --out_file "$PRED_DIR/metrics.json" \
  --dump_examples "$PRED_DIR/per_example.jsonl"

echo
echo "metrics  -> $PRED_DIR/metrics.json"
echo "per-ex   -> $PRED_DIR/per_example.jsonl"
echo "rawpreds -> $PREDS"
