# Recreating the `trainer` conda env on a new system (no Docker)

This env is a **conda env**, Python **3.11.15**, **torch 2.6.0+cu124**.

## Short answer: do you have to rebuild wheels?

**Mostly no.** Everything except one package installs as a prebuilt wheel:

- **torch / torchvision / torchaudio / triton / all `nvidia-*` CUDA libs** →
  prebuilt wheels. They bundle their own CUDA 12.4 runtime, so the new machine
  does **not** need a system CUDA toolkit installed — only an NVIDIA **driver
  new enough for CUDA 12.4** (driver **≥ 550.54**). `nvcc` version on the host
  is irrelevant unless you compile flash-attn yourself.
- **transformers, peft, accelerate, datasets, trl, deepspeed-if-present, etc.**
  → pure-python / prebuilt wheels, no build.
- **flash-attn 2.8.3.post1** → this is the *only* package that was compiled from
  source here. You do **not** have to recompile it — install the matching
  **prebuilt wheel** (see below). Recompiling is only needed if no matching
  wheel exists for your torch/python/arch.

## Requirements for "runs identically"

1. **Linux x86_64** (same as here).
2. **Python 3.11** (use 3.11 to reuse all wheels, incl. flash-attn `cp311`).
3. **torch 2.6.0+cu124** exactly (other pins assume this).
4. **NVIDIA driver ≥ 550.54** (CUDA 12.4 capable).
5. **Same GPU architecture family** ideally (this ran on A100 = `sm_80`).
   flash-attn 2.8.x supports Ampere/Ada/Hopper; check your GPU is covered.

---

## Path A — `conda-pack` (recommended: zero rebuild, byte-for-byte same)

Best if the new machine is the same Linux/arch with a compatible driver. This
ships the *entire* env including the compiled flash-attn, so nothing builds.

On **this** machine:
```bash
conda install -n base conda-pack -y
conda pack -p /scratch/asing725/IP/envs/trainer -o trainer-env.tar.gz
# move trainer-env.tar.gz via Drive (it is multi-GB; do NOT commit it)
```
On the **new** machine:
```bash
mkdir -p ~/envs/trainer && tar -xzf trainer-env.tar.gz -C ~/envs/trainer
source ~/envs/trainer/bin/activate
conda-unpack          # rewrites absolute paths baked into the env
python -c "import torch, flash_attn; print(torch.__version__, flash_attn.__version__)"
```
Then re-point the editable trainer (paths changed):
```bash
pip install -e /path/to/model-forge/LLaMA-Factory --no-deps
```

## Path B — recreate from lockfiles (clean rebuild from spec)

```bash
conda create -n trainer python=3.11 -y
conda activate trainer

# 1) torch first, from the cu124 index (pinned build)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

# 2) the rest of the locked deps
pip install -r requirements-lock.txt

# 3) flash-attn: install the PREBUILT wheel (no compile). On the flash-attention
#    GitHub releases page (github.com/Dao-AILab/flash-attention/releases) grab the
#    asset matching THIS env exactly:
#       flash_attn-2.8.3.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
#    (cu12 + torch2.6 + cxx11abiFALSE + cp311). Then:
pip install "<paste the matching wheel URL here>"

# 4) the vendored trainer, editable
pip install -e /path/to/model-forge/LLaMA-Factory --no-deps
```

> If you ever must compile flash-attn (no matching wheel): you need a CUDA
> toolkit with `nvcc` (12.x), `ninja`, and lots of RAM/time:
> `MAX_JOBS=4 pip install flash-attn==2.8.3.post1 --no-build-isolation`

## Path C — exact conda layer (same OS/arch only)

`conda-explicit.txt` pins conda-level packages by URL+hash for an identical
platform:
```bash
conda create -n trainer --file conda-explicit.txt
conda activate trainer
pip install -r requirements-lock.txt   # then steps 3-4 above
```

---

## Verify it runs the same

```bash
python - <<'PY'
import torch, flash_attn, transformers, peft, accelerate
print("torch", torch.__version__, "| cuda", torch.version.cuda)
print("flash_attn", flash_attn.__version__)
print("transformers", transformers.__version__, "| peft", peft.__version__)
print("cuda avail:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
llamafactory-cli version
```
Expect: torch `2.6.0+cu124`, flash_attn `2.8.3.post1`, transformers `5.6.0`,
peft `0.18.1`, CUDA available True.

## Files

- `requirements-lock.txt` — exact pip versions (`pip freeze`).
- `environment.yml` — portable conda spec (no build strings).
- `conda-explicit.txt` — exact same-platform conda spec (URLs+hashes).
