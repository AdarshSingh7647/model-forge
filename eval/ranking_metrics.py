#!/usr/bin/env python3
"""
Compute table-ranking metrics from a LLaMA-Factory `generated_predictions.jsonl`
(produced by `do_predict` + `predict_with_generate`), or from any jsonl whose
rows carry a model output and a gold reference.

The task: given a question + candidate tables, the model emits a JSON object
    {"ranked_tables": [8, 1, 2, 3, 7, 4, 6, 5]}
(optionally preceded by chain-of-thought text). We extract that list from both
the prediction and the gold label and score it.

Metrics (all macro-averaged over examples; an unparseable prediction scores 0):
  EM          exact match  -- predicted list == gold list (order AND membership)
  RelaxedEM   set match    -- same set of table ids, order ignored
  F1          set-overlap F1 over table ids (precision/recall on the id sets)
  Top1        is the predicted #1 table the gold #1 table (the retrieval win)
  KendallTau  rank-correlation over the shared ids (ordering quality), averaged
              over examples where it is defined (>=2 shared ids)

Counts reported separately:
  n_total        examples scored
  n_invalid      predictions we could not parse a ranked_tables list from
  n_wrong_em     predictions that are not an exact match (includes invalid)
  n_wrong_set    predictions whose id set differs from gold (includes invalid)

Run on a predictions file:
    python ranking_metrics.py --pred_file .../generated_predictions.jsonl
Self-test (no data needed):
    python ranking_metrics.py --self_test
"""

import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional

# field names emitted by LLaMA-Factory's predict_with_generate
PRED_KEYS = ("predict", "prediction", "output")
GOLD_KEYS = ("label", "labels", "reference", "gold")


def parse_ranked_tables(text: Any) -> Optional[List[int]]:
    """
    Pull the ranked-tables list out of a model/gold string. Robust to leading
    chain-of-thought: scans every {...} block and returns the ids from the LAST
    one that parses and contains a non-empty "ranked_tables" list. Falls back to
    a regex over a bare `"ranked_tables": [ ... ]` if no JSON object parses.
    Returns a list[int] (ids coerced to int where possible) or None.
    """
    if text is None:
        return None
    if isinstance(text, (list, tuple)):  # already a list of ids
        return _coerce_int_list(text)
    if not isinstance(text, str):
        text = str(text)

    found: Optional[List[Any]] = None
    for m in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("ranked_tables"), list) and obj["ranked_tables"]:
            found = obj["ranked_tables"]  # keep last valid one

    if found is None:
        # greedy single-object attempt (handles nested braces the lazy regex split)
        m = re.search(r'"ranked_tables"\s*:\s*\[([^\]]*)\]', text, flags=re.DOTALL)
        if m:
            raw = [p.strip().strip('"\'') for p in m.group(1).split(",") if p.strip() != ""]
            found = raw

    if not found:
        return None
    return _coerce_int_list(found)


def _coerce_int_list(items: List[Any]) -> Optional[List[int]]:
    out: List[int] = []
    for x in items:
        try:
            out.append(int(x))
        except (ValueError, TypeError):
            # non-int id (e.g. a string table name) -- keep hashable as-is via fallback
            return _as_str_list(items)
    return out


def _as_str_list(items: List[Any]) -> List[Any]:
    return [str(x).strip() for x in items]


def set_f1(pred: List[Any], gold: List[Any]) -> float:
    ps, gs = set(pred), set(gold)
    if not ps and not gs:
        return 1.0
    if not ps or not gs:
        return 0.0
    inter = len(ps & gs)
    if inter == 0:
        return 0.0
    precision = inter / len(ps)
    recall = inter / len(gs)
    return 2 * precision * recall / (precision + recall)


def kendall_tau(pred: List[Any], gold: List[Any]) -> Optional[float]:
    """Kendall tau over the ids present in BOTH lists (ordering agreement).
    None if fewer than 2 shared ids."""
    shared = [x for x in gold if x in set(pred)]
    if len(shared) < 2:
        return None
    pred_rank = {x: i for i, x in enumerate(pred)}
    gold_rank = {x: i for i, x in enumerate(gold)}
    concordant = discordant = 0
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            a, b = shared[i], shared[j]
            sign_g = gold_rank[a] - gold_rank[b]
            sign_p = pred_rank[a] - pred_rank[b]
            if sign_g * sign_p > 0:
                concordant += 1
            elif sign_g * sign_p < 0:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return None
    return (concordant - discordant) / denom


def score_example(pred_text: Any, gold_text: Any) -> Dict[str, Any]:
    gold = parse_ranked_tables(gold_text)
    pred = parse_ranked_tables(pred_text)
    invalid = pred is None
    if gold is None:
        # gold itself unparseable: skip from aggregates by flagging
        return {"skip": True}
    if invalid:
        return {"skip": False, "invalid": True, "em": 0, "relaxed_em": 0,
                "f1": 0.0, "top1": 0, "tau": None, "pred": pred, "gold": gold}
    em = int(pred == gold)
    relaxed = int(set(pred) == set(gold))
    f1 = set_f1(pred, gold)
    top1 = int(len(pred) > 0 and len(gold) > 0 and pred[0] == gold[0])
    tau = kendall_tau(pred, gold)
    return {"skip": False, "invalid": False, "em": em, "relaxed_em": relaxed,
            "f1": f1, "top1": top1, "tau": tau, "pred": pred, "gold": gold}


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in rows if not r.get("skip")]
    n = len(scored)
    if n == 0:
        return {"n_total": 0, "note": "no scorable rows (no parseable gold)"}
    n_invalid = sum(int(r["invalid"]) for r in scored)
    taus = [r["tau"] for r in scored if r["tau"] is not None]
    return {
        "n_total": n,
        "n_invalid": n_invalid,
        "n_wrong_em": sum(1 for r in scored if r["em"] == 0),
        "n_wrong_set": sum(1 for r in scored if r["relaxed_em"] == 0),
        "EM": round(sum(r["em"] for r in scored) / n, 4),
        "RelaxedEM": round(sum(r["relaxed_em"] for r in scored) / n, 4),
        "F1": round(sum(r["f1"] for r in scored) / n, 4),
        "Top1": round(sum(r["top1"] for r in scored) / n, 4),
        "KendallTau": round(sum(taus) / len(taus), 4) if taus else None,
        "invalid_rate": round(n_invalid / n, 4),
    }


def load_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pick(row: Dict[str, Any], keys) -> Any:
    for k in keys:
        if k in row:
            return row[k]
    return None


def evaluate_file(pred_file: str, out_file: Optional[str] = None,
                  dump_examples: Optional[str] = None) -> Dict[str, Any]:
    rows = load_rows(pred_file)
    scored = []
    per_example = []
    for i, row in enumerate(rows):
        pred_text = _pick(row, PRED_KEYS)
        gold_text = _pick(row, GOLD_KEYS)
        s = score_example(pred_text, gold_text)
        scored.append(s)
        if not s.get("skip"):
            per_example.append({"idx": i, "em": s["em"], "relaxed_em": s["relaxed_em"],
                                "f1": round(s["f1"], 4), "top1": s["top1"],
                                "invalid": s["invalid"], "pred": s["pred"], "gold": s["gold"]})
    metrics = aggregate(scored)
    metrics["pred_file"] = pred_file
    if out_file:
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=2)
    if dump_examples:
        with open(dump_examples, "w") as f:
            for e in per_example:
                f.write(json.dumps(e) + "\n")
    return metrics


def _self_test() -> int:
    cases = [
        # (pred, gold, expect em, relaxed, top1)
        ('{"ranked_tables": [3,1,2]}', '{"ranked_tables": [3,1,2]}', 1, 1, 1),
        ('blah\n{"ranked_tables": [1,3,2]}', '{"ranked_tables": [3,1,2]}', 0, 1, 0),
        ('CoT...\n\n{"ranked_tables": [3,1]}', '{"ranked_tables": [3,1,2]}', 0, 0, 1),
        ('no json here', '{"ranked_tables": [3,1,2]}', 0, 0, 0),  # invalid
        ('reason {"x":1} then {"ranked_tables":[5,4]}', '{"ranked_tables":[5,4]}', 1, 1, 1),
    ]
    rows = [score_example(p, g) for p, g, *_ in cases]
    ok = True
    for (p, g, em, rel, t1), r in zip(cases, rows):
        got = (r["em"], r["relaxed_em"], r["top1"])
        exp = (em, rel, t1)
        status = "OK " if got == exp else "FAIL"
        if got != exp:
            ok = False
        print(f"[{status}] pred={p[:32]!r:35} -> em/rel/top1 got={got} exp={exp}")
    agg = aggregate(rows)
    print("aggregate:", json.dumps(agg, indent=2))
    assert agg["n_invalid"] == 1, "expected exactly one invalid"
    print("\nSELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_file", help="generated_predictions.jsonl to score")
    ap.add_argument("--out_file", default=None, help="write metrics json here")
    ap.add_argument("--dump_examples", default=None, help="write per-example scores jsonl here")
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if not args.pred_file:
        ap.error("provide --pred_file or --self_test")
    metrics = evaluate_file(args.pred_file, args.out_file, args.dump_examples)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
