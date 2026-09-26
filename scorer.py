# Vendored from MbitAI's shared meter-day harness on 2026-09-25.
# Logic is unchanged from the frozen original; only this header is new.
# Here ROOT resolves to meter-day-classifier, which mirrors the same
# data/ + outputs/label_pack/ layout (human labels are private, not shipped).
"""Metrics and run harness for the frozen meter-day evaluation."""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

LABELS = ("active", "standby", "off", "unsure")


def validate_prediction(prediction: object) -> tuple[bool, str]:
    if not isinstance(prediction, dict):
        return False, "prediction is not an object"
    label = prediction.get("label")
    reason = prediction.get("reason")
    baseline = prediction.get("baseline")
    if label not in LABELS:
        return False, f"label must be one of {LABELS}, got {label!r}"
    if not isinstance(reason, str) or not reason.strip():
        return False, "reason must be a non-empty string"
    if not isinstance(baseline, str) or not baseline.strip():
        return False, "baseline must be a non-empty string"
    return True, ""


def _f1(gold: list[str], predicted: list[str], label: str) -> dict[str, float]:
    tp = sum(g == label and p == label for g, p in zip(gold, predicted))
    fp = sum(g != label and p == label for g, p in zip(gold, predicted))
    fn = sum(g == label and p != label for g, p in zip(gold, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def classification_metrics(
    gold: list[str],
    predicted: list[str],
    valid: list[bool] | None = None,
) -> dict[str, object]:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted lengths differ")
    valid = valid or [prediction in LABELS for prediction in predicted]
    if len(valid) != len(gold):
        raise ValueError("valid mask length differs")

    valid_gold = [g for g, ok in zip(gold, valid) if ok]
    valid_predicted = [p for p, ok in zip(predicted, valid) if ok]
    per_class = {
        label: _f1(valid_gold, valid_predicted, label)
        for label in LABELS
    }
    macro_f1 = statistics.mean(item["f1"] for item in per_class.values())
    predicted_declines = sum(p == "unsure" for p in valid_predicted)
    true_declines = sum(g == "unsure" for g in valid_gold)
    correct_declines = sum(
        g == "unsure" and p == "unsure"
        for g, p in zip(valid_gold, valid_predicted)
    )
    return {
        "n": len(gold),
        "n_valid": len(valid_gold),
        "schema_validity": sum(valid) / len(valid) if valid else 0.0,
        "accuracy": (
            sum(g == p for g, p in zip(valid_gold, valid_predicted))
            / len(valid_gold)
            if valid_gold else 0.0
        ),
        "macro_f1": macro_f1,
        "per_class": per_class,
        "decline_rate": predicted_declines / len(valid_predicted)
        if valid_predicted else 0.0,
        "decline_precision": correct_declines / predicted_declines
        if predicted_declines else 0.0,
        "gold_decline_rate": true_declines / len(valid_gold)
        if valid_gold else 0.0,
    }


def _modal_rate(labels: list[str]) -> float:
    if not labels:
        return 0.0
    return Counter(labels).most_common(1)[0][1] / len(labels)


def evaluate(
    system_name: str,
    records: Iterable[object],
    predictor: Callable[[object], dict],
    repeats: int = 1,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Run a predictor and return one summary plus per-day evidence.

    The predictor may return a private ``_meta`` dictionary containing
    ``wall_s``, ``prompt_tokens`` and ``completion_tokens``.  Those values are
    recorded but never affect the classification metrics.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")

    records = list(records)
    gold = [record.gold_label for record in records]
    all_predictions: list[str] = []
    valid: list[bool] = []
    details = []
    walls = []
    prompt_tokens = []
    completion_tokens = []

    for record in records:
        repeated = []
        repeated_valid = []
        repeated_reasons = []
        repeated_meta = []
        for _ in range(repeats):
            started = time.perf_counter()
            try:
                raw = predictor(record)
                elapsed = time.perf_counter() - started
            except Exception as exc:  # keep failures visible in the results
                raw = {
                    "label": None,
                    "reason": f"predictor error: {exc}",
                    "baseline": system_name,
                    "_meta": {"error": repr(exc)},
                }
                elapsed = time.perf_counter() - started
            raw = dict(raw)
            meta = dict(raw.pop("_meta", {}))
            meta.setdefault("wall_s", elapsed)
            ok, error = validate_prediction(raw)
            repeated.append(raw.get("label"))
            repeated_valid.append(ok)
            repeated_reasons.append(raw.get("reason", ""))
            repeated_meta.append(meta)
            walls.append(float(meta["wall_s"]))
            if meta.get("prompt_tokens") is not None:
                prompt_tokens.append(int(meta["prompt_tokens"]))
            if meta.get("completion_tokens") is not None:
                completion_tokens.append(int(meta["completion_tokens"]))

        # Schema validity is measured on the scored first call.  A malformed
        # later repeat affects consistency, but must not retroactively remove
        # the first call from the headline accuracy denominator.
        valid.append(repeated_valid[0])
        # First response is the scored response. Repeats measure consistency,
        # rather than changing the headline answer after seeing later calls.
        all_predictions.append(repeated[0])
        details.append({
            "day_id": record.day_id,
            "meter_id": record.meter_id,
            "date": record.date,
            "gold_label": record.gold_label,
            "predicted_label": repeated[0],
            "schema_valid": repeated_valid[0],
            "validation_error": ""
            if repeated_valid[0]
            else validate_prediction({
                "label": repeated[0],
                "reason": repeated_reasons[0],
                "baseline": system_name,
            })[1],
            "reason": repeated_reasons[0],
            "repeat_labels": repeated,
            "consistency": _modal_rate(repeated),
            "metadata": repeated_meta[0],
        })

    metrics = classification_metrics(gold, all_predictions, valid)
    metrics["consistency"] = statistics.mean(
        detail["consistency"] for detail in details
    )
    metrics["latency_p50_s"] = statistics.median(walls) if walls else 0.0
    metrics["latency_n"] = len(walls)
    metrics["prompt_tokens_mean"] = (
        statistics.mean(prompt_tokens) if prompt_tokens else None
    )
    metrics["completion_tokens_mean"] = (
        statistics.mean(completion_tokens) if completion_tokens else None
    )
    metrics["cost_per_1000_local"] = None
    summary = {"system": system_name, **metrics}
    return summary, details


def write_results_row(path: Path, summary: dict[str, object]) -> None:
    """Upsert one (system, split) summary while preserving other rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    row = {
        key: value
        for key, value in summary.items()
        if key not in {"per_class"}
    }
    for label, values in summary["per_class"].items():
        for metric, value in values.items():
            row[f"{label}_{metric}"] = value
    row = {key: json.dumps(value) if isinstance(value, (dict, list)) else value
           for key, value in row.items()}
    key = (summary["system"], summary.get("split"))
    rows = [r for r in rows if (r.get("system"), r.get("split")) != key]
    rows.append(row)
    fieldnames = sorted({key for old in rows for key in old})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_details(path: Path, details: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(detail, sort_keys=True)
                               for detail in details) + "\n")
