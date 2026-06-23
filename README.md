# The Forge 🔨

*A playground for forging, compressing, and stress-testing small LLMs.*

## Theme

**A controlled empirical study of how post-training supervision design and
post-hoc compression shape a small LLM's table-reranking ability.** Starting
from Qwen3 base models (4B and 8B), we LoRA-fine-tune under a matrix of
reasoning-supervision strategies — full chain-of-thought loss vs. answer-only
loss, with and without hints — then apply quantization and related efficiency
techniques. Every variant (base, fine-tuned, and quantized) is benchmarked on
the downstream table-ranking task and against one another, in order to quantify
the **merits and trade-offs (task accuracy vs. compute, latency, and memory)**
of each training and quantization method, and to identify which combination is
Pareto-best.

## What gets compared

- **Models:** Qwen3-8B, Qwen3-4B (LoRA fine-tuning via LLaMA-Factory).
- **Supervision setups** (4):
  | Setup | Conversation | Loss target |
  |---|---|---|
  | `setup1_full_loss` | prompt → thinking + answer | thinking + answer |
  | `setup2_answer_only` | prompt → thinking → cue → answer | answer only (`mask_history`) |
  | `setup3_1_hint_answer_only` | prompt + hint → answer | answer only |
  | `setup3_2_hint_full_loss` | prompt + hint → thinking + answer | thinking + answer |
- **End task:** rank candidate tables for a query; scored with EM / RelaxedEM /
  F1 / Top1 / KendallTau (`eval/ranking_metrics.py`).
- **Compression:** quantization + related techniques applied to trained adapters
  (analysis layer on top of the trained variants).

## Layout

```
The Forge (model-forge/)
├── download_models.py      # fetch Qwen3 base models -> model_paths.json
├── build_sft_data.py       # build the 4 sharegpt SFT setups (train = sharded folders)
├── shard_data.py           # split a JSON-array file into <100 MB shards
├── make_configs.py         # generate the 8 LLaMA-Factory LoRA configs
├── probe_memory.py         # measure peak GPU memory per (model, seq_len)
├── training_status.py      # live status across all runs
├── configs/run_all.sh      # sequentially launch the runs
├── eval/                   # run_eval.sh + ranking_metrics.py
├── training_data/          # (git-ignored) SFT data; move out-of-band
├── lora_runs/              # (git-ignored) adapter checkpoints
└── LLaMA-Factory/          # (git-ignored) vendored trainer; install separately
```

## Setup

The full training stack (conda + torch + flash-attn + LLaMA-Factory) is built by
one pinned command — see [env/SETUP.md](env/SETUP.md) for details and the
H200 / CUDA-12.6 notes:

```bash
# 0. place the vendored trainer in the repo first (git-ignored; move via Drive
#    or clone the pinned version) so ./LLaMA-Factory exists.
bash env/setup_env.sh        # creates the "trainer" conda env, deterministically
conda activate trainer
cp .env.example .env         # then fill in HF_KEY + paths
```

`requirements.txt` holds only the lightweight deps for the helper scripts here;
the heavy trainer stack is installed by `env/setup_env.sh`. All machine-specific
paths and the HF token come from `.env` (never committed).

## Pipeline

```bash
python download_models.py --load_check        # 1. base models
python build_sft_data.py --llamafactory_data_dir LLaMA-Factory/data   # 2. data
python make_configs.py                        # 3. 8 LoRA configs
bash configs/run_all.sh                        # 4. train (auto-resumes from latest checkpoint)
eval/run_eval.sh qwen3_8b_full_loss            # 5. predict + score
```

## Data sharding

Train splits are large, so `build_sft_data.py` writes each one as a **directory
of `<100 MB` JSON shards** (`part-00000.json`, …) via `shard_data.py`.
LLaMA-Factory loads every file in a directory as one dataset, so configs
reference the folder by name and read all shards transparently. To shard any
existing JSON-array file:

```bash
python shard_data.py --in big.json --out_dir big/ --replace
```

## Note on large artifacts

Model weights (`lora_runs/`), datasets (`training_data/`), and the vendored
`LLaMA-Factory/` are **git-ignored** and moved out-of-band (e.g. Google Drive).
The repo holds code + the mechanism to regenerate everything.

To resume a training run on another machine, copy the full checkpoint folder
(including `optimizer.pt`), the base model, and the dataset; match the trainer
env and GPU count; update the paths in the config; then launch with
`overwrite_output_dir: false` to auto-resume.
