# Vendored from Project 37 (gepa-meter-sorter/baselines/rules_floor.py) on 2026-09-25.
# Logic is unchanged from the frozen original; only this header is new.
# Ships as the reference floor for generate_synth_meters.py --calibrate
# (no predictor in this repo uses it since the hybrid was removed).
"""
Deterministic per-meter rules floor for meter-day state classification.

DESCRIPTION:
- Classify one meter-day as active, standby, off, or unsure using only the
  current day and a reference percentile from that meter's whole period.
- Follow the frozen label guide's decision order: missing, zeros, flat band,
  borderline floor, standby, active, and finally a close-call fallback.
- Abstain with unsure when more than 25% of intervals are missing, when one
  flat band sits near the meter running level, when the day floor is
  borderline (28-55% of the meter reference), or when nothing above resolves
  the day.
- Call off when zeros dominate (>= 60% of valid intervals) or when a zero
  flatline covers the whole 08:00-17:00 work shift.
- Read every threshold statistic from the shared evidence contract so each
  reason can be re-audited against the same numbers a human or model sees.
- Never read truth.json or human labels; the rules stay fully label-blind.

PREREQUISITES:
1. Python 3.8+ must be installed
2. uv package manager must be installed
3. numpy must be installed for the shared evidence encoder

SETUP FOR NEW USERS:
1. Install uv: `pip install uv` (or follow official instructions)
2. Navigate to this script directory
3. Install dependencies: `uv pip install -r requirements.txt`

FILE REQUIREMENTS:
- None directly. The rules operate on `DayRecord` objects built by `data_io`
  or by unit tests; callers resolve the raw CSV files.

USAGE:
    uv run python -m unittest -v
    uv run python evaluate.py --system rules --split eval

OUTPUT:
- `predict()` returns a dict with the keys `label`, `reason`, `baseline`.
- No files are written by this module.

DEPENDENCIES (automatically installed by uv):
- numpy>=1.24.0

TROUBLESHOOTING:
- All thresholds compare interval energy against the meter p95, so the units
  cancel out and no kW conversion is needed.
- The work-shift zero check uses interval-start clock hours taken from the
  shared evidence timestamps, so the spring-forward day keeps local-clock
  alignment.
"""

from __future__ import annotations

from datetime import datetime

from data_io import INTERVAL, DayRecord
from evidence import LOW_BAND_FRACTION

BASELINE_NAME = "rules_floor_v1"

# Decision anchors mirror the frozen label guide.  The borderline floor window
# itself lives in `evidence.py` so rules and evidence share one calibration.
HEAVY_MISSING_PCT = 25.0
DOMINANT_ZERO_PCT = 60.0
FLAT_MIN_RATIO = 0.75
FLAT_MIN_NIGHT_TO_DAY = 0.85
FLAT_MIN_REFERENCE_SHARE = 0.70
STANDBY_MIN_LOW_SHARE = 0.70
ACTIVE_MIN_RUNNING_SHARE = 0.33
WORK_SHIFT_START_HOUR = 8
WORK_SHIFT_END_HOUR = 17
INTERVALS_PER_HOUR = 4
WORK_SHIFT_MIN_INTERVALS = (
    WORK_SHIFT_END_HOUR - WORK_SHIFT_START_HOUR
) * INTERVALS_PER_HOUR


def _result(label: str, reason: str) -> dict[str, str]:
    return {"label": label, "reason": reason, "baseline": BASELINE_NAME}


def _zero_shift_interval_count(
    values: tuple[float | None, ...],
    timestamps: tuple[datetime, ...],
) -> int:
    """Return the work-shift interval count when all of them are exact zero."""
    shift_values = [
        value
        for value, stamp in zip(values, timestamps)
        if WORK_SHIFT_START_HOUR <= (stamp - INTERVAL).hour < WORK_SHIFT_END_HOUR
    ]
    if len(shift_values) < WORK_SHIFT_MIN_INTERVALS:
        return 0
    if not all(value == 0.0 for value in shift_values):
        return 0
    return len(shift_values)


def predict(record: DayRecord) -> dict[str, str]:
    """Classify one meter-day in the frozen label guide's decision order."""
    evidence = record.evidence
    reference = record.meter_p95_kwh

    # Step 1: heavy missing.
    if evidence.n_valid == 0:
        return _result("unsure", "no valid intervals to classify")
    if evidence.missing_pct > HEAVY_MISSING_PCT:
        return _result(
            "unsure",
            f"{evidence.missing_pct:.1f}% of intervals are missing; the day "
            "is not reliable enough to classify",
        )

    # Step 2: dominant zeros, or a zero flatline across the work shift.
    if evidence.zero_pct_valid >= DOMINANT_ZERO_PCT:
        return _result(
            "off",
            f"{evidence.zero_pct_valid:.1f}% of valid intervals are exact zero",
        )
    shift_zeros = _zero_shift_interval_count(record.values, evidence.timestamps)
    if shift_zeros:
        return _result(
            "off",
            "a zero flatline covers the whole 08:00-17:00 work shift "
            f"({shift_zeros} intervals)",
        )

    day_p95 = evidence.positive_p95_kwh
    low_share = evidence.low_positive_share

    # Step 3: one flat band at this meter's usual running level.
    flatness = evidence.flatness_ratio
    night_day = evidence.night_to_day_ratio
    if (
        flatness is not None
        and flatness >= FLAT_MIN_RATIO
        and (night_day is None or night_day >= FLAT_MIN_NIGHT_TO_DAY)
        and day_p95 is not None
        and day_p95 >= FLAT_MIN_REFERENCE_SHARE * reference
    ):
        return _result(
            "unsure",
            f"positive p50/p95 is {flatness:.2f} and the flat band sits at "
            f"{day_p95 / reference:.0%} of the meter reference; nothing "
            "separates idle from running",
        )

    # Step 4: borderline floor inside the shared calibrated window.
    floor_ratio = evidence.floor_to_ref_ratio
    if evidence.is_borderline_floor and floor_ratio is not None:
        return _result(
            "unsure",
            f"the borderline floor sits at {floor_ratio:.0%} of the meter "
            "reference; splitting idle from running would be a guess",
        )

    # Step 5: clear standby, either a low p95 or a floor band all day.
    if day_p95 is not None and day_p95 <= LOW_BAND_FRACTION * reference:
        return _result(
            "standby",
            f"positive p95 is {day_p95 / reference:.1%} of the meter "
            f"reference, clearly below the {LOW_BAND_FRACTION:.0%} running "
            "threshold",
        )
    if low_share >= STANDBY_MIN_LOW_SHARE:
        return _result(
            "standby",
            f"{low_share:.1%} of valid intervals sit in the clear low band "
            f"below {LOW_BAND_FRACTION:.0%} of the meter reference",
        )

    # Step 6: clear running day above the low band; the floor already passed
    # the flat and borderline separation checks above.
    running_count = sum(
        1
        for value in record.values
        if value is not None and value > LOW_BAND_FRACTION * reference
    )
    running_share = running_count / evidence.n_valid
    if running_share >= ACTIVE_MIN_RUNNING_SHARE:
        return _result(
            "active",
            f"{running_share:.1%} of valid intervals run above the "
            f"{LOW_BAND_FRACTION:.0%} threshold with a clearly lower floor",
        )

    # Step 7: close call becomes an explicit abstention.
    return _result(
        "unsure",
        f"running share {running_share:.1%} and low-band share "
        f"{low_share:.1%} do not resolve the day",
    )
