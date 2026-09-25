#!/usr/bin/env python3
"""Eval-once via the frozen scorer.

Runs the frozen winner (``classical_logreg_v1``) on one human split and
records it with the frozen ``scorer.evaluate`` / ``write_results_row`` —
unchanged. Writes go to this repo's ``outputs/`` only:

- ``outputs/results.csv`` — scoreboard (ships with the repo; recreated if
  deleted). The private human labels are never written.
- ``outputs/predictions/classical_logreg_v1_{train,eval}.jsonl`` — per-day
  details.

Usage (eval runs ONCE per finalist, after the A2 sign-off — re-running the
same command overwrites the same row, it does not create a new look)::

    python3 run_eval.py --system classical_logreg_v1 --split eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent

from data_io import ROOT as REPO_ROOT  # noqa: E402
from data_io import load_labeled_records  # noqa: E402
from predict import SYSTEM_NAME, make_predictor  # noqa: E402
from scorer import evaluate, write_details, write_results_row  # noqa: E402

assert REPO_ROOT == HERE, "vendored data_io ROOT drift"

OUTPUTS = HERE / "outputs"
RESULTS = OUTPUTS / "results.csv"


def ensure_scoreboard() -> Path:
    """Ensure the scoreboard exists (ships with the repo; the scorer creates
    a fresh file on first write if it was deleted)."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    return RESULTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=(SYSTEM_NAME,), default=SYSTEM_NAME)
    parser.add_argument("--split", choices=("train", "eval"), default="eval")
    args = parser.parse_args()

    predictor = make_predictor()
    records = load_labeled_records(split=args.split)
    summary, details = evaluate(args.system, records, predictor, repeats=1)
    summary["split"] = args.split

    results_path = ensure_scoreboard()
    details_path = OUTPUTS / "predictions" / f"{args.system}_{args.split}.jsonl"
    write_results_row(results_path, summary)
    write_details(details_path, details)

    print(f"system: {args.system}")
    print(f"split: {args.split}, n={summary['n']}")
    print(f"schema validity: {summary['schema_validity']:.1%}")
    print(f"accuracy: {summary['accuracy']:.1%}")
    print(f"macro F1: {summary['macro_f1']:.3f}")
    print(f"decline rate: {summary['decline_rate']:.1%}")
    print(f"decline precision: {summary['decline_precision']:.1%}")
    print(f"consistency: {summary['consistency']:.1%}")
    print(f"p50 latency: {summary['latency_p50_s'] * 1000:.2f} ms/day")
    print(f"wrote {results_path}")
    print(f"wrote {details_path}")


if __name__ == "__main__":
    main()
