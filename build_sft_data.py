#!/usr/bin/env python3
"""
Build SFT training data (sharegpt format, for LLaMA-Factory) for the
table-ranking task, in 4 supervision setups, from a raw examples file.

Raw file (CoTReranker/sft_examples.json) is a list of dicts shaped like:
{
  "qid": "...",
  "dataset": "...",
  "prompt": "<full instruction + candidate tables + format request>",
  "thinking": "<chain-of-thought reasoning text>",
  "answer": "\n\n{\"ranked_tables\": [...]}",   # final answer, may have leading whitespace
  "ranked_table_ids": [...],
  "candidate_ids": [...]
}

We check whether the already-processed file
  <PROCESSED_DIR>/sft_deepseekR1_table_ranking.json
exists. If yes, we treat that as the canonical filtered/cleaned single-setup
source and use it as the basis for generating the 4 setup-specific sharegpt
files. If it does NOT exist, we build it first by filtering the raw
sft_examples.json (dropping malformed/incomplete rows), then proceed.

Output (per --model_tag, since prompts can optionally differ per model,
though by default the same data is reused for both models since the data
itself is model-agnostic; the model_tag only affects file naming so
LLaMA-Factory configs can stay simple and explicit):

  <out_dir>/setup1_full_loss_{model_tag}.json        -- thinking+answer, full loss
  <out_dir>/setup2_answer_only_{model_tag}.json      -- thinking(masked)+answer(loss), mask_history
  <out_dir>/setup3_1_hint_answer_only_{model_tag}.json   -- hint-in-prompt, loss on answer only
  <out_dir>/setup3_2_hint_full_loss_{model_tag}.json     -- hint-in-prompt, loss on thinking+answer

Each file is sharegpt-format: a list of
  {"conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}, ...],
   "system": <optional>}
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from shard_data import shard_json_array

load_dotenv()  # pick up paths/secrets from .env if present

# Repo root = directory this script lives in. All in-repo defaults derive from
# it so the project is portable; external inputs come from environment vars.
PROJECT_ROOT = Path(__file__).resolve().parent

RAW_PATH_DEFAULT = os.getenv("RAW_SFT_PATH", "")
PROCESSED_DIR_DEFAULT = os.getenv(
    "PROCESSED_DIR", str(PROJECT_ROOT / "training_data" / "table_retrieval")
)
PROCESSED_FILENAME = "sft_deepseekR1_table_ranking.json"

# Whatever script originally produced sft_examples.json from these metadata
# files only ever read the "total_thinking" key, never falling back to
# "reasoning_text". As a result ~75% of NQ-Tables rows ended up with an
# empty "thinking" field even though the real CoT text exists in these raw
# files, keyed by qid. We re-derive it from here before filtering.
METADATA_DIR_DEFAULT = os.getenv("METADATA_DIR", "")
DEEPSEEK_METADATA_SPLITS = ["train", "dev", "test"]

# Priority order of keys that hold the real chain-of-thought text in the
# metadata files. "thinking_in_text" is deliberately excluded: across all
# rows that have it, it is always an empty string (a dead field from
# whichever export pass produced that batch) -- "total_thinking" holds the
# real text for those same rows instead.
THINKING_KEY_PRIORITY = ["reasoning_text", "total_thinking"]

SYSTEM_PROMPT = (
    "You are a careful retrieval assistant that ranks candidate tables by "
    "their relevance to a user's question."
)

HINT_INSTRUCTION = (
    "Think step-by-step before answering; here is an example of the kind of "
    "reasoning to learn from:\n"
)

REQUIRED_RAW_FIELDS = ["qid", "prompt", "thinking", "answer"]


# --------------------------------------------------------------------------- #
# Step 1: locate / build the canonical processed (filtered) file
# --------------------------------------------------------------------------- #

def is_valid_example(ex: Dict[str, Any]) -> bool:
    """Drop rows missing required fields or with empty/garbage content."""
    for field in REQUIRED_RAW_FIELDS:
        if field not in ex or ex[field] is None:
            return False
        if isinstance(ex[field], str) and len(ex[field].strip()) == 0:
            return False

    # answer must contain a parseable JSON object with "ranked_tables"
    answer_json = extract_answer_json(ex["answer"])
    if answer_json is None or "ranked_tables" not in answer_json:
        return False
    if not isinstance(answer_json["ranked_tables"], list) or len(answer_json["ranked_tables"]) == 0:
        return False

    return True


def extract_answer_json(answer_str: str) -> Optional[Dict[str, Any]]:
    """
    The raw 'answer' field looks like '\n\n{"ranked_tables": [4, 2, 1, 3]}'
    (leading whitespace/newlines, otherwise a clean JSON object). Find the
    first {...} block and parse it.
    """
    match = re.search(r"\{.*\}", answer_str, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def normalize_answer_str(answer_str: str) -> str:
    """Clean answer field down to just the compact JSON object string."""
    parsed = extract_answer_json(answer_str)
    if parsed is None:
        return answer_str.strip()
    return json.dumps(parsed)


def load_metadata_index(metadata_dir: str) -> Dict[str, Dict[str, Any]]:
    """Index deepseek_listwise_metadata_{train,dev,test}.json by qid, so we
    can backfill 'thinking' for rows where sft_examples.json left it empty."""
    index: Dict[str, Dict[str, Any]] = {}
    for split in DEEPSEEK_METADATA_SPLITS:
        path = os.path.join(metadata_dir, f"deepseek_listwise_metadata_{split}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            index.update(json.load(f))
    return index


def recover_thinking(qid: Any, metadata_idx: Dict[str, Dict[str, Any]]) -> Optional[str]:
    meta = metadata_idx.get(qid)
    if meta is None:
        return None
    for key in THINKING_KEY_PRIORITY:
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def build_processed_file(raw_path: str, processed_path: str, metadata_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    print(f"[build] Processed file not found. Filtering raw data:\n"
          f"        raw       = {raw_path}\n"
          f"        processed -> {processed_path}")

    if not os.path.exists(raw_path):
        print(f"[error] Raw file not found at {raw_path}", file=sys.stderr)
        sys.exit(1)

    with open(raw_path, "r") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        # tolerate a top-level {"data": [...]} wrapper, just in case
        raw_data = raw_data.get("data", raw_data.get("examples", []))

    print(f"[build] Loaded {len(raw_data)} raw examples.")

    metadata_idx: Dict[str, Dict[str, Any]] = {}
    if metadata_dir:
        print(f"[build] Indexing raw metadata files in {metadata_dir} for 'thinking' recovery...")
        metadata_idx = load_metadata_index(metadata_dir)
        print(f"[build] Indexed {len(metadata_idx)} metadata rows by qid.")

    filtered = []
    dropped = 0
    recovered = 0
    seen_qids = set()
    for ex in raw_data:
        if metadata_idx and isinstance(ex.get("thinking"), str) and ex["thinking"].strip() == "":
            recovered_text = recover_thinking(ex.get("qid"), metadata_idx)
            if recovered_text:
                ex["thinking"] = recovered_text
                recovered += 1

        if not is_valid_example(ex):
            dropped += 1
            continue
        qid = ex.get("qid")
        if qid is not None and qid in seen_qids:
            dropped += 1  # drop exact-duplicate qids
            continue
        if qid is not None:
            seen_qids.add(qid)

        clean_ex = {
            "qid": ex["qid"],
            "dataset": ex.get("dataset", "unknown"),
            "prompt": ex["prompt"].strip(),
            "thinking": ex["thinking"].strip(),
            "answer": normalize_answer_str(ex["answer"]),
            "ranked_table_ids": ex.get("ranked_table_ids", []),
            "candidate_ids": ex.get("candidate_ids", []),
        }
        filtered.append(clean_ex)

    print(f"[build] Recovered 'thinking' text for {recovered} rows via metadata cross-reference.")
    print(f"[build] Kept {len(filtered)} examples, dropped {dropped} malformed/duplicate rows.")

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    with open(processed_path, "w") as f:
        json.dump(filtered, f, indent=2)
    print(f"[build] Wrote processed file to {processed_path}")

    return filtered


def load_or_build_processed(raw_path: str, processed_dir: str, metadata_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    processed_path = os.path.join(processed_dir, PROCESSED_FILENAME)
    if os.path.exists(processed_path):
        print(f"[check] Found existing processed file: {processed_path}")
        with open(processed_path, "r") as f:
            data = json.load(f)
        print(f"[check] Loaded {len(data)} processed examples.")
        return data
    return build_processed_file(raw_path, processed_path, metadata_dir)


# --------------------------------------------------------------------------- #
# Step 2: convert canonical examples into the 4 sharegpt-format setups
# --------------------------------------------------------------------------- #

def make_conversation(turns: List[Dict[str, str]], system: str = SYSTEM_PROMPT) -> Dict[str, Any]:
    return {"system": system, "conversations": turns}


def setup1_full_loss(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Single turn: human=prompt, gpt=thinking+answer. Full loss on gpt turn."""
    response = f"{ex['thinking']}\n\n{ex['answer']}"
    return make_conversation([
        {"from": "human", "value": ex["prompt"]},
        {"from": "gpt", "value": response},
    ])


def setup2_answer_only(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Multi-turn: human=prompt, gpt=thinking (masked), human=continuation cue,
    gpt=answer (loss target). Requires mask_history=true in LLaMA-Factory
    config so only the LAST gpt turn contributes to the loss.
    """
    return make_conversation([
        {"from": "human", "value": ex["prompt"]},
        {"from": "gpt", "value": ex["thinking"]},
        {"from": "human", "value": "Based on that reasoning, give the final ranking."},
        {"from": "gpt", "value": ex["answer"]},
    ])


def setup3_prompt_with_hint(ex: Dict[str, Any]) -> str:
    """Build the augmented prompt: original prompt + 1-line instruction + thinking-as-hint."""
    return (
        f"{ex['prompt']}\n\n"
        f"{HINT_INSTRUCTION}"
        f"{ex['thinking']}"
    )


def setup3_1_hint_answer_only(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Single turn: human=prompt+hint, gpt=answer only. Loss naturally on answer only."""
    return make_conversation([
        {"from": "human", "value": setup3_prompt_with_hint(ex)},
        {"from": "gpt", "value": ex["answer"]},
    ])


def setup3_2_hint_full_loss(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Single turn: human=prompt+hint, gpt=thinking+answer (reused thinking). Full loss."""
    response = f"{ex['thinking']}\n\n{ex['answer']}"
    return make_conversation([
        {"from": "human", "value": setup3_prompt_with_hint(ex)},
        {"from": "gpt", "value": response},
    ])


SETUP_BUILDERS = {
    "setup1_full_loss": setup1_full_loss,
    "setup2_answer_only": setup2_answer_only,
    "setup3_1_hint_answer_only": setup3_1_hint_answer_only,
    "setup3_2_hint_full_loss": setup3_2_hint_full_loss,
}

# Which setups require mask_history=true in the LLaMA-Factory dataset config
SETUPS_REQUIRING_MASK_HISTORY = {"setup2_answer_only"}


def build_all_setups(
    examples: List[Dict[str, Any]],
    out_dir: str,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Data content is model-agnostic (Qwen3-8B and Qwen3-4B are trained on the
    identical text), so we write ONE copy per setup -- not duplicated per
    model tag. Model choice is purely a config-file concern (model_name_or_path
    in the LLaMA-Factory YAML), not a dataset concern. This avoids doubling
    disk usage for no benefit.
    """
    import random

    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    n_val = max(1, int(len(examples) * val_fraction)) if len(examples) > 20 else 0
    val_idx = set(indices[:n_val])

    written: Dict[str, str] = {}

    for setup_name, builder in SETUP_BUILDERS.items():
        train_rows, val_rows = [], []
        for i, ex in enumerate(examples):
            row = builder(ex)
            (val_rows if i in val_idx else train_rows).append(row)

        # Train splits are large, so write them as a *directory* of <100 MB
        # shards. LLaMA-Factory loads every file in the directory as one
        # dataset, so configs reference it by name exactly as before.
        train_dir = os.path.join(out_dir, f"{setup_name}_train")
        shards = shard_json_array(train_rows, train_dir)
        written[f"{setup_name}_train"] = train_dir

        # Val splits are small; a single file is fine.
        n_val_shards = 0
        if val_rows:
            val_path = os.path.join(out_dir, f"{setup_name}_val.json")
            with open(val_path, "w") as f:
                json.dump(val_rows, f, indent=2)
            written[f"{setup_name}_val"] = val_path

        print(f"[setup] {setup_name}: {len(train_rows)} train ({len(shards)} shards) / "
              f"{len(val_rows)} val "
              f"(mask_history={'true' if setup_name in SETUPS_REQUIRING_MASK_HISTORY else 'false'})")

    return written


# --------------------------------------------------------------------------- #
# Step 3: register datasets in LLaMA-Factory's dataset_info.json
# --------------------------------------------------------------------------- #

def update_dataset_info(llamafactory_data_dir: str, written_files: Dict[str, str]):
    """
    LLaMA-Factory discovers local datasets via data/dataset_info.json in its
    repo's data dir. We add/overwrite entries pointing at our absolute file
    paths so configs can reference them by name regardless of where this
    script's out_dir is.
    """
    info_path = os.path.join(llamafactory_data_dir, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
    else:
        info = {}
        os.makedirs(llamafactory_data_dir, exist_ok=True)

    for key, path in written_files.items():
        info[key] = {
            "file_name": path,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[register] Updated {info_path} with {len(written_files)} dataset entries.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Build sharegpt SFT data for table ranking, 4 setups.")
    p.add_argument("--raw_path", default=RAW_PATH_DEFAULT)
    p.add_argument("--metadata_dir", default=METADATA_DIR_DEFAULT,
                    help="Dir containing deepseek_listwise_metadata_{train,dev,test}.json, "
                         "used to backfill 'thinking' text that's empty in --raw_path. "
                         "Pass '' to disable recovery.")
    p.add_argument("--processed_dir", default=PROCESSED_DIR_DEFAULT)
    p.add_argument("--out_dir", default=os.path.join(PROCESSED_DIR_DEFAULT, "sharegpt_setups"))
    p.add_argument("--llamafactory_data_dir", default=None,
                    help="Path to LLaMA-Factory's data/ dir, to register dataset_info.json entries. "
                         "If omitted, registration step is skipped.")
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--force_rebuild_processed", action="store_true",
                   help="Rebuild the processed canonical file from raw even if it already exists.")
    return p.parse_args()


def main():
    args = parse_args()

    processed_path = os.path.join(args.processed_dir, PROCESSED_FILENAME)
    if args.force_rebuild_processed and os.path.exists(processed_path):
        print(f"[force] Removing existing processed file to rebuild: {processed_path}")
        os.remove(processed_path)

    examples = load_or_build_processed(args.raw_path, args.processed_dir, args.metadata_dir or None)

    written = build_all_setups(
        examples,
        out_dir=args.out_dir,
        val_fraction=args.val_fraction,
    )

    if args.llamafactory_data_dir:
        update_dataset_info(args.llamafactory_data_dir, written)
    else:
        print("[skip] --llamafactory_data_dir not provided; skipping dataset_info.json registration. "
              "Run again with that flag, or merge manually -- see written paths below.")

    print("\n=== Written files ===")
    for k, v in written.items():
        print(f"  {k:45s} -> {v}")


if __name__ == "__main__":
    sys.exit(main())