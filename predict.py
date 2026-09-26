#!/usr/bin/env python3
"""Frozen-model predictor + template reason (displayed, never scored).

A3 wiring. The model and operating point were frozen in A2
(`artifacts/model.joblib` + `operating_point.json`); this module loads
them read-only and applies the exact combined decline rule from
``train.decline_curve``: predict ``unsure`` when the argmax is ``unsure`` OR
the top1-top2 probability margin is below the frozen threshold (t=0.0, i.e.
argmax-only in v1).

Reason contract (implementation plan section 5): cite printed evidence numbers
and the meter baseline. No generator call in v1.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from features import FEATURES, row_from_record

HERE = Path(__file__).resolve().parent

LABELS = ("active", "standby", "off", "unsure")

SYSTEM_NAME = "classical_logreg_v1"

ARTIFACTS = HERE / "artifacts"


def _fmt(value: float | None, decimals: int = 4) -> str:
    return "not available" if value is None else f"{value:.{decimals}f}"


def _fmt_share(value: float | None) -> str:
    """Format a 0..1 share as a percent (A1 nit fix: was a bare fraction)."""
    return "not available" if value is None else f"{value * 100:.0f}%"


def build_reason(evidence, label: str) -> str:
    """Build the one-line display reason for a predicted label."""
    assert label in LABELS, f"bad label {label!r}"
    night_med = _fmt(evidence.weekday_night_median_kwh)
    ref = _fmt(evidence.meter_reference_kwh)
    low_share = _fmt_share(evidence.low_positive_share_valid)
    return (
        f"label is {label} because night median {night_med} kWh vs meter "
        f"reference {ref} kWh; low-band share {low_share} of valid intervals; "
        f"max zero run {evidence.max_zero_run}; baseline p95 {ref} kWh."
    )


def load_artifacts():
    """Load the frozen A2 model + operating point (read-only, no fitting)."""
    model = joblib.load(ARTIFACTS / "model.joblib")
    operating = json.loads((ARTIFACTS / "operating_point.json").read_text())
    assert operating["features"] == FEATURES, "artifact/model feature drift"
    assert operating["threshold_rule"].startswith("max decline-F1"), (
        "operating point is not the frozen A2 calibration"
    )
    return model, float(operating["threshold"])


def make_predictor():
    """Return a ``scorer.evaluate``-compatible ``predict(record)`` closure."""
    model, threshold = load_artifacts()
    classes = list(model.classes_)
    assert set(classes) == set(LABELS), f"unexpected classes {classes}"

    def predict(record) -> dict:
        started = time.perf_counter()
        row = np.array(row_from_record(record), dtype=float).reshape(1, -1)
        proba = model.predict_proba(row)[0]
        top2 = np.sort(proba)[-2:]
        margin = float(top2[1] - top2[0])
        argmax = classes[int(np.argmax(proba))]
        # Same combined rule as train.decline_curve: at t=0.0 the margin
        # clause is a no-op (margins are >= 0) and unsure is argmax-only.
        label = "unsure" if (argmax == "unsure" or margin < threshold) else argmax
        return {
            "label": label,
            "reason": build_reason(record.evidence, label),
            "baseline": SYSTEM_NAME,
            "_meta": {"wall_s": time.perf_counter() - started},
        }

    return predict
