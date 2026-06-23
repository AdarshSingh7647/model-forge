#!/usr/bin/env bash
#
# One-command, deterministic rebuild of the `trainer` conda env.
#
#   bash env/setup_env.sh
#
# Target: Linux x86_64, NVIDIA H200 (Hopper / sm_90), system CUDA 12.6,
# driver >= 560. (torch's cu124 wheels bundle their own CUDA 12.4 runtime and
# run fine on a CUDA-12.6 host — only the driver must be new enough, which an
# H200/12.6 box always is.)
#
# Everything is pinned, so the result is reproducible. flash-attn installs from
# the official prebuilt wheel (fast, multi-arch incl. Hopper); if that wheel is
# ever unavailable it builds from source pinned to sm_90. Either path yields the
# same import-compatible result.
#
# Override knobs (optional env vars):
#   ENV_NAME=trainer   TORCH_INDEX=https://download.pytorch.org/whl/cu124
#   MAX_JOBS=4         (parallelism if flash-attn falls back to a source build)
set -euo pipefail

ENV_NAME="${ENV_NAME:-trainer}"
PY_VER="3.11"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$REPO_DIR/env/requirements-lock.txt"

# --- pinned versions (single source of truth) ---
TORCH_PKGS="torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
FA_VER="2.8.3.post1"
FA_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

echo ">>> [1/6] creating conda env '$ENV_NAME' (python $PY_VER)"
conda create -y -n "$ENV_NAME" "python=$PY_VER"

# activate it for the rest of the script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
python -m pip install --upgrade pip

echo ">>> [2/6] installing pinned torch from $TORCH_INDEX"
pip install $TORCH_PKGS --index-url "$TORCH_INDEX"

echo ">>> [3/6] installing the rest of the locked deps (excluding torch* + flash-attn)"
# torch is already installed from the CUDA index; flash-attn is handled below.
grep -viE '^(torch|torchvision|torchaudio|flash[_-]attn)([=<>!~ ]|$)' "$LOCK" \
  | grep -vE '^\s*(#|$)' \
  | pip install -r /dev/stdin

echo ">>> [4/6] installing flash-attn $FA_VER (prebuilt wheel, sm_90-capable)"
if ! pip install "$FA_WHEEL_URL"; then
  echo "    prebuilt wheel unavailable -> building from source for Hopper (sm_90)"
  TORCH_CUDA_ARCH_LIST="9.0+PTX" MAX_JOBS="${MAX_JOBS:-4}" \
    pip install "flash-attn==${FA_VER}" --no-build-isolation
fi

echo ">>> [5/6] installing vendored LLaMA-Factory (editable)"
pip install -e "$REPO_DIR/LLaMA-Factory" --no-deps

echo ">>> [6/6] verifying"
python - <<'PY'
import torch, flash_attn, transformers, peft, accelerate, datasets, trl
print("torch       :", torch.__version__, "| bundled cuda", torch.version.cuda)
print("arch_list   :", torch.cuda.get_arch_list())  # must include 'sm_90' for H200
print("flash_attn  :", flash_attn.__version__)
print("transformers:", transformers.__version__, "| peft", peft.__version__,
      "| accelerate", accelerate.__version__)
print("cuda avail  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu         :", torch.cuda.get_device_name(0),
          torch.cuda.get_device_capability(0))
PY

echo ""
echo "DONE. Activate with:  conda activate $ENV_NAME"
echo "Expected: torch 2.6.0+cu124, flash_attn 2.8.3.post1, arch_list contains sm_90."
