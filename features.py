#!/usr/bin/env python3
"""DayEvidence -> flat feature row. Closed contract, no raw floats.

Allowed input surface (per implementation plan section 2): only the fields of
``DayEvidence`` from the vendored ``evidence.py``, plus ``is_weekend`` derived
from the record date. No raw CSV peeking, no labels.

None handling in this module: emit ``nan`` and let A2 fit median imputation on
train folds only. A1 never fits anything.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from data_io import DayRecord, group_meter_rows, load_labeled_records, meter_period_p95

REPO_ROOT = Path(__file__).resolve().parent

LABELS = ("active", "standby", "off", "unsure")

# Closed feature list (implementation plan section 2). Order is the column order.
FEATURES = [
    "missing_pct",
    "n_missing",
    "n_valid",
    "zero_pct_valid",
    "zero_count",
    "max_zero_run",
    "trailing_zero_run",
    "positive_p10_kwh",
    "positive_median_kwh",
    "positive_p95_kwh",
    "positive_max_kwh",
    "positive_min_kwh",
    "flatness_ratio",
    "floor_to_ref_ratio",
    "low_positive_share_valid",
    "low_weekday_night_share_d",
    "low_weekday_day_share_d",
    "low_weekend_share_d",
    "weekday_night_median_kwh",
    "weekday_day_median_kwh",
    "weekend_median_kwh",
    "night_to_day_ratio",
    "meter_reference_kwh",
    "low_threshold_kwh",
    "is_borderline_floor",
    "is_weekend",
]


def _num(value: float | int | None) -> float:
    return float("nan") if value is None else float(value)


def row_from_record(record) -> list[float]:
    """Build one ordered feature row from a DayRecord (via its .evidence)."""
    ev = record.evidence
    is_weekend = 1.0 if datetime.fromisoformat(record.date).weekday() >= 5 else 0.0
    return [
        _num(ev.missing_pct),
        _num(ev.n_missing),
        _num(ev.n_valid),
        _num(ev.zero_pct_valid),
        _num(ev.zero_count),
        _num(ev.max_zero_run),
        _num(ev.trailing_zero_run),
        _num(ev.positive_p10_kwh),
        _num(ev.positive_median_kwh),
        _num(ev.positive_p95_kwh),
        _num(ev.positive_max_kwh),
        _num(ev.positive_min_kwh),
        _num(ev.flatness_ratio),
        _num(ev.floor_to_ref_ratio),
        _num(ev.low_positive_share_valid),
        _num(ev.low_weekday_night_share_d),
        _num(ev.low_weekday_day_share_d),
        _num(ev.low_weekend_share_d),
        _num(ev.weekday_night_median_kwh),
        _num(ev.weekday_day_median_kwh),
        _num(ev.weekend_median_kwh),
        _num(ev.night_to_day_ratio),
        _num(ev.meter_reference_kwh),
        _num(ev.low_threshold_kwh),
        1.0 if ev.is_borderline_floor else 0.0,
        is_weekend,
    ]


def build_matrix(records) -> tuple[np.ndarray, list[str], list[str]]:
    """Return (X, y, ids) for DayRecords. X holds nan for missing evidence."""
    records = list(records)
    assert len(FEATURES) == len(row_from_record(records[0])), "feature/row length drift"
    x = np.array([row_from_record(r) for r in records], dtype=float)
    y = [r.gold_label for r in records]
    ids = [r.day_id for r in records]
    assert all(label in LABELS for label in y), f"unexpected label in {set(y)}"
    return x, y, ids


DEFAULT_SYNTH_ROOT = Path(__file__).resolve().parent / "synth"


def load_synth_records(synth_root: Path | None = None):
    """Build DayRecords for every synth day. Decontaminated by construction.

    Reads ``labels_synth38_v1.json`` + ``data/raw/s38_m*.csv`` under the synth
    root, reusing the frozen evidence contract (grouping, p95 reference).
    Asserts zero day_id overlap with the frozen labels_v1 (defense in depth:
    the generator already guarantees this via distinct meter IDs).
    gold_label carries the WEAK label — never human gold.
    """
    synth_root = Path(synth_root) if synth_root else DEFAULT_SYNTH_ROOT
    labels_doc = json.loads((synth_root / "labels_synth38_v1.json").read_text())
    assert labels_doc.get("version") == "synth38_v1", "not the synth38_v1 contract"
    # Defense in depth (the generator already guarantees this via distinct
    # meter IDs): skip when the private human labels are absent — synth day
    # IDs live in their own s38_* namespace by construction.
    human_labels = REPO_ROOT / "data" / "labels_v1.json"
    frozen_ids: set[str] = set()
    if human_labels.exists():
        frozen_ids = {
            e["day_id"]
            for e in json.loads(human_labels.read_text())["labels"]
        }
    else:
        print("warning: data/labels_v1.json absent (private) — "
              "decontamination check skipped, see data/README.md")
    p95_cache: dict[str, float] = {}
    records = []
    for entry in labels_doc["days"]:
        assert entry["day_id"] not in frozen_ids, f"CONTAMINATED: {entry['day_id']}"
        meter_id = entry["meter_id"]
        if meter_id not in p95_cache:
            p95_cache[meter_id] = meter_period_p95(meter_id, root=synth_root)
        rows = group_meter_rows(meter_id, root=synth_root).get(entry["date"])
        if rows is None:
            raise ValueError(f"{entry['day_id']} absent from synth CSVs")
        records.append(
            DayRecord(
                day_id=entry["day_id"],
                meter_id=meter_id,
                date=entry["date"],
                split="synth",
                values=tuple(v for _, v in rows),
                meter_p95_kwh=p95_cache[meter_id],
                gold_label=entry["weak_label"],
                gold_note=entry.get("reason", ""),
                timestamps=tuple(s for s, _ in rows),
            )
        )
    if not records:
        raise ValueError("no synth records found")
    return records


def check_splits() -> None:
    frozen = REPO_ROOT
    labels_path = frozen / "data" / "labels_v1.json"
    selection_path = frozen / "outputs" / "label_pack" / "selection.json"
    for path in (labels_path, selection_path):
        if not path.exists():
            raise SystemExit(
                f"check-splits needs the private human data at {path} — "
                "see data/README.md. It is not shipped with this repo."
            )
    labels_doc = json.loads(labels_path.read_text())
    assert labels_doc.get("frozen") and labels_doc.get("version") == "labels_v1", (
        "not the frozen labels_v1 contract"
    )
    label_split = {e["day_id"]: e["split"] for e in labels_doc["labels"]}
    selection = json.loads((frozen / "outputs" / "label_pack" / "selection.json").read_text())
    sel_train = {d["day_id"] for d in selection["splits"]["train"]}
    sel_eval = {d["day_id"] for d in selection["splits"]["eval"]}

    train = load_labeled_records(split="train")
    eval_ = load_labeled_records(split="eval")
    train_ids = [r.day_id for r in train]
    eval_ids = [r.day_id for r in eval_]

    assert len(train) == 30 and len(eval_) == 30, (len(train), len(eval_))
    assert set(train_ids) == sel_train, "train IDs differ from selection.json"
    assert set(eval_ids) == sel_eval, "eval IDs differ from selection.json"
    assert all(label_split[i] == "train" for i in train_ids), "labels_v1 split mismatch (train)"
    assert all(label_split[i] == "eval" for i in eval_ids), "labels_v1 split mismatch (eval)"
    assert not (set(train_ids) & set(eval_ids)), "train/eval overlap"

    x_train, y_train, _ = build_matrix(train)
    x_eval, y_eval, _ = build_matrix(eval_)
    assert x_train.shape == (30, len(FEATURES)), x_train.shape
    assert x_eval.shape == (30, len(FEATURES)), x_eval.shape

    from collections import Counter

    print(f"features: {len(FEATURES)}")
    print(f"train: n=30 {dict(Counter(y_train))}")
    print(f"eval:  n=30 {dict(Counter(y_eval))}")
    nan_train = int(np.isnan(x_train).sum())
    nan_eval = int(np.isnan(x_eval).sum())
    print(f"NaN cells: train={nan_train} eval={nan_eval} (expected: sparse, bucket medians only)")
    cols_with_nan = [
        FEATURES[j]
        for j in range(len(FEATURES))
        if bool(np.isnan(x_train[:, j]).any() or np.isnan(x_eval[:, j]).any())
    ]
    print(f"columns with NaN: {cols_with_nan}")
    print("check-splits: OK — 60 rows, 30/30 IDs match selection.json + labels_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-splits", action="store_true")
    parser.add_argument("--build-synth", action="store_true")
    parser.add_argument("--synth-root", type=Path, default=None)
    args = parser.parse_args()
    if args.check_splits:
        check_splits()
    elif args.build_synth:
        from collections import Counter

        records = load_synth_records(args.synth_root)
        x, y, _ = build_matrix(records)
        print(f"synth: n={len(records)} {dict(Counter(y))} shape={x.shape}")
        print("build-synth: OK — decontaminated, matrix builds")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
