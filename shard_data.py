#!/usr/bin/env python3
"""
Shard a large JSON-array dataset file into many smaller (<100 MB) JSON files
so the data stays under hosting/file-size limits (e.g. GitHub's 100 MB cap)
and can be version-controlled.

LLaMA-Factory natively loads a *directory* of same-extension files as one
dataset (it lists every file in the folder and concatenates them), so the
training code does not need to change: point `file_name` in dataset_info.json
at the shard directory instead of a single file and every shard is read.

Each shard is itself a valid JSON array, written compactly to save space.

Use as a library:
    from shard_data import shard_json_array
    shard_json_array(records, out_dir, max_bytes=90 * 1024 * 1024)

Or from the CLI to shard an existing file in place:
    python shard_data.py --in big.json --out_dir big_shards/
    python shard_data.py --in big.json --out_dir big/ --replace   # remove big.json after
"""

import argparse
import json
import os
import shutil
import sys
from typing import Any, List

DEFAULT_MAX_BYTES = 90 * 1024 * 1024  # keep comfortably under the 100 MB limit
SHARD_PREFIX = "part-"


def shard_json_array(
    records: List[Any],
    out_dir: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> List[str]:
    """Write `records` into out_dir as part-00000.json, part-00001.json, ...
    where each shard is a JSON array kept under `max_bytes`. Returns the list
    of shard paths written. Clears any pre-existing shards in out_dir first."""
    # Start clean so re-running doesn't leave stale shards behind.
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            if name.startswith(SHARD_PREFIX) and name.endswith(".json"):
                os.remove(os.path.join(out_dir, name))
    os.makedirs(out_dir, exist_ok=True)

    written: List[str] = []
    shard_idx = 0
    cur: List[Any] = []
    # 2 accounts for the enclosing "[" and "]"; we grow with ",<record>".
    cur_bytes = 2

    def flush() -> None:
        nonlocal shard_idx, cur, cur_bytes
        if not cur:
            return
        path = os.path.join(out_dir, f"{SHARD_PREFIX}{shard_idx:05d}.json")
        with open(path, "w") as f:
            json.dump(cur, f, ensure_ascii=False, separators=(",", ":"))
        written.append(path)
        shard_idx += 1
        cur = []
        cur_bytes = 2

    for rec in records:
        rec_bytes = len(json.dumps(rec, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 1
        # A single record larger than the budget still gets its own shard.
        if cur and cur_bytes + rec_bytes > max_bytes:
            flush()
        cur.append(rec)
        cur_bytes += rec_bytes
    flush()

    return written


def main() -> int:
    p = argparse.ArgumentParser(description="Shard a JSON-array file into <100 MB parts.")
    p.add_argument("--in", dest="in_path", required=True, help="Input JSON-array file.")
    p.add_argument("--out_dir", required=True, help="Directory to write shards into.")
    p.add_argument("--max_bytes", type=int, default=DEFAULT_MAX_BYTES,
                   help=f"Max bytes per shard (default {DEFAULT_MAX_BYTES}).")
    p.add_argument("--replace", action="store_true",
                   help="Delete the input file after successfully writing shards.")
    args = p.parse_args()

    if not os.path.isfile(args.in_path):
        print(f"[error] input not found: {args.in_path}", file=sys.stderr)
        return 1

    print(f"[shard] loading {args.in_path} ...")
    with open(args.in_path) as f:
        records = json.load(f)
    if not isinstance(records, list):
        print(f"[error] expected a top-level JSON array, got {type(records).__name__}", file=sys.stderr)
        return 1

    shards = shard_json_array(records, args.out_dir, args.max_bytes)
    total = sum(os.path.getsize(s) for s in shards)
    print(f"[shard] wrote {len(shards)} shard(s), {total / 1024**2:.1f} MB total -> {args.out_dir}")

    if args.replace:
        os.remove(args.in_path)
        print(f"[shard] removed original {args.in_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
