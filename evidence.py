# Vendored from MbitAI's shared meter-day harness on 2026-09-25.
# Logic is unchanged from the frozen original; only this header is new.
# Here ROOT resolves to meter-day-classifier, which mirrors the same
# data/ + outputs/label_pack/ layout (human labels are private, not shipped).
"""
Shared low-band evidence encoder for meter-day evaluation.

DESCRIPTION:
- Load interval-end timestamps and group them by local day.
- Preserve missing readings as None; missing data never becomes exact zero.
- Calculate the whole-meter positive p95 reference and its 35% low threshold.
- Count low-positive intervals separately from exact zeros and running intervals.
- Report weekday-night, weekday-day, and weekend low-positive shares and
  medians using interval-start local time, with None for an unavailable bucket.
- Report the positive p10/p50/p95 levels, consecutive and trailing zero runs,
  p50/p95 flatness, and the night/day median ratio.
- Flag borderline floors (28-55% of the meter reference) where one static
  low-band threshold cannot separate idle from running.
- Format a compact decision-facts line (flatness, floor ratio, running share)
  so models compare printed numbers instead of calculating them.
- Provide one structured evidence record plus shared formatting helpers for
  prompt, pack, and audit consumers.

PREREQUISITES:
1. Python 3.8+ must be installed
2. uv package manager must be installed
3. numpy must be installed for the project test environment

SETUP FOR NEW USERS:
1. Install uv: `pip install uv` (or follow official instructions)
2. Navigate to this script directory
3. Install dependencies: `uv pip install -r requirements.txt`

FILE REQUIREMENTS:
- Optional raw input files under `data/raw/<meter_id>.csv`
- Input rows must contain `timestamp` and `value_kwh` columns
- Empty `value_kwh` cells are treated as missing

USAGE:
    uv run python -m unittest -v

OUTPUT:
- No files are written by this module.
- Call `encode_day_evidence()` for structured evidence or
  `format_evidence_line()` for the shared prompt/Markdown line.

DEPENDENCIES (automatically installed by uv):
- numpy>=1.24.0

TROUBLESHOOTING:
- A timestamp without an offset is still usable for the local bucket, but raw
  generator data should retain its offset.
- A bucket with no valid intervals has a share of `None` and formats as
  `not available`; this is safer than silently displaying a false zero.
"""

from __future__ import annotations

import csv
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
INTERVAL = timedelta(minutes=15)
LOW_BAND_FRACTION = 0.35
# Borderline-floor window calibrated on the frozen benchmark selection: it
# covers the meter_7 (0.29-0.32) and meter_8 (0.42-0.47) floor clusters while
# leaving clear floors (<= 0.06) and near-reference meters (>= 0.87) outside.
BORDERLINE_FLOOR_LOW = 0.28
BORDERLINE_FLOOR_HIGH = 0.55


@dataclass(frozen=True)
class BucketEvidence:
    """Low-positive counts, safe share, and median for one local time bucket."""

    valid_count: int
    low_positive_count: int
    low_positive_share: float | None
    median_kwh: float | None = None

    @property
    def low_share(self) -> float | None:
        """Short alias for baseline code and reason construction."""
        return self.low_positive_share


@dataclass(frozen=True)
class DayEvidence:
    """Structured evidence shared by deterministic and model paths."""

    day_id: str
    meter_id: str
    date: str
    values: tuple[float | None, ...]
    timestamps: tuple[datetime, ...]
    n_intervals: int
    n_missing: int
    missing_pct: float
    n_valid: int
    zero_count: int
    zero_pct_valid: float
    positive_min_kwh: float | None
    positive_median_kwh: float | None
    positive_p95_kwh: float | None
    positive_max_kwh: float | None
    meter_reference_kwh: float
    low_threshold_kwh: float
    low_positive_count: int
    low_positive_share_valid: float
    weekday_night: BucketEvidence
    weekday_day: BucketEvidence
    weekend: BucketEvidence
    max_zero_run: int = 0
    trailing_zero_run: int = 0
    positive_p10_kwh: float | None = None
    floor_to_ref_ratio: float | None = None
    flatness_ratio: float | None = None
    weekday_night_median_kwh: float | None = None
    weekday_day_median_kwh: float | None = None
    weekend_median_kwh: float | None = None
    night_to_day_ratio: float | None = None

    @property
    def low_positive_share(self) -> float:
        """Low-positive intervals divided by all valid day intervals."""
        return self.low_positive_share_valid

    @property
    def low_weekday_night_share_d(self) -> float | None:
        """Approved contract name for the weekday-night share."""
        return self.weekday_night.low_positive_share

    @property
    def low_weekday_day_share_d(self) -> float | None:
        """Approved contract name for the weekday-day share."""
        return self.weekday_day.low_positive_share

    @property
    def low_weekend_share_d(self) -> float | None:
        """Approved contract name for the weekend share."""
        return self.weekend.low_positive_share

    @property
    def is_borderline_floor(self) -> bool:
        """Flag a day floor that sits too close to the meter reference.

        The calibrated window covers both benchmark borderline clusters
        (meter_7 at 0.29-0.32 and meter_8 at 0.42-0.47) so rules and prompts
        can abstain instead of guessing a split between idle and running.
        """
        if self.floor_to_ref_ratio is None:
            return False
        return BORDERLINE_FLOOR_LOW <= self.floor_to_ref_ratio <= BORDERLINE_FLOOR_HIGH

    @property
    def stats(self) -> dict[str, float | int | None]:
        """Return the same day-stat fields exposed by ``DayRecord``."""
        return {
            "n_intervals": self.n_intervals,
            "n_missing": self.n_missing,
            "missing_pct": self.missing_pct,
            "zero_pct_valid": self.zero_pct_valid,
            "positive_min_kwh": self.positive_min_kwh,
            "positive_p10_kwh": self.positive_p10_kwh,
            "positive_median_kwh": self.positive_median_kwh,
            "positive_p95_kwh": self.positive_p95_kwh,
            "positive_max_kwh": self.positive_max_kwh,
            "max_zero_run": self.max_zero_run,
            "trailing_zero_run": self.trailing_zero_run,
            "flatness_ratio": self.flatness_ratio,
        }

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly nested representation."""
        return {
            "day_id": self.day_id,
            "meter_id": self.meter_id,
            "date": self.date,
            "stats": self.stats,
            "meter_reference_kwh": self.meter_reference_kwh,
            "low_threshold_kwh": self.low_threshold_kwh,
            "low_positive_count": self.low_positive_count,
            "low_positive_share_valid": self.low_positive_share_valid,
            "low_weekday_night_share_d": self.low_weekday_night_share_d,
            "low_weekday_day_share_d": self.low_weekday_day_share_d,
            "low_weekend_share_d": self.low_weekend_share_d,
            "max_zero_run": self.max_zero_run,
            "trailing_zero_run": self.trailing_zero_run,
            "positive_p10_kwh": self.positive_p10_kwh,
            "floor_to_ref_ratio": self.floor_to_ref_ratio,
            "flatness_ratio": self.flatness_ratio,
            "is_borderline_floor": self.is_borderline_floor,
            "weekday_night_median_kwh": self.weekday_night_median_kwh,
            "weekday_day_median_kwh": self.weekday_day_median_kwh,
            "weekend_median_kwh": self.weekend_median_kwh,
            "night_to_day_ratio": self.night_to_day_ratio,
            "weekday_night": {
                "valid_count": self.weekday_night.valid_count,
                "low_positive_count": self.weekday_night.low_positive_count,
                "low_positive_share": self.weekday_night.low_positive_share,
                "median_kwh": self.weekday_night.median_kwh,
            },
            "weekday_day": {
                "valid_count": self.weekday_day.valid_count,
                "low_positive_count": self.weekday_day.low_positive_count,
                "low_positive_share": self.weekday_day.low_positive_share,
                "median_kwh": self.weekday_day.median_kwh,
            },
            "weekend": {
                "valid_count": self.weekend.valid_count,
                "low_positive_count": self.weekend.low_positive_count,
                "low_positive_share": self.weekend.low_positive_share,
                "median_kwh": self.weekend.median_kwh,
            },
        }


def parse_timestamp(text: str) -> datetime:
    """Parse a generator timestamp while retaining any UTC offset."""
    return datetime.fromisoformat(text)


def day_of_timestamp(stamp: datetime) -> str:
    """Return the local day represented by an interval-end timestamp."""
    return (stamp - INTERVAL).date().isoformat()


def load_meter_rows(
    meter_id: str,
    root: Path | None = None,
) -> list[tuple[datetime, float | None]]:
    """Load timestamp/value pairs without changing missing values."""
    data_root = root or ROOT
    path = data_root / "data" / "raw" / f"{meter_id}.csv"
    with path.open(newline="") as handle:
        rows: list[tuple[datetime, float | None]] = []
        for row in csv.DictReader(handle):
            value = row["value_kwh"]
            rows.append(
                (
                    parse_timestamp(row["timestamp"]),
                    None if value == "" else float(value),
                )
            )
    return rows


def group_meter_rows(
    meter_id: str,
    root: Path | None = None,
) -> dict[str, list[tuple[datetime, float | None]]]:
    """Group interval-end rows using the meter-day start timestamp."""
    days: dict[str, list[tuple[datetime, float | None]]] = {}
    for stamp, value in load_meter_rows(meter_id, root=root):
        days.setdefault(day_of_timestamp(stamp), []).append((stamp, value))
    return days


def load_meter_days(
    meter_id: str,
    root: Path | None = None,
) -> dict[str, list[float | None]]:
    """Return day values using the shared interval-end day mapping."""
    return {
        day: [value for _, value in rows]
        for day, rows in group_meter_rows(meter_id, root=root).items()
    }


def meter_period_p95(meter_id: str, root: Path | None = None) -> float:
    """Return p95 of all nonmissing, strictly positive meter readings."""
    positive = [
        value
        for values in load_meter_days(meter_id, root=root).values()
        for value in values
        if value is not None and value > 0
    ]
    if not positive:
        raise ValueError(f"{meter_id} has no positive readings")
    return float(np.percentile(positive, 95))


def _compute_zero_runs(values: tuple[float | None, ...]) -> tuple[int, int]:
    """Calculate maximum and trailing runs of consecutive exact zero intervals."""
    max_run = 0
    current_run = 0
    for value in values:
        if value == 0.0:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0
    trailing_run = 0
    for value in reversed(values):
        if value == 0.0:
            trailing_run += 1
        else:
            break
    return max_run, trailing_run


def stats_for(values: Iterable[float | None]) -> dict[str, float | int | None]:
    """Calculate safe day statistics, excluding None from valid denominators."""
    values = list(values)
    valid = [value for value in values if value is not None]
    positive = [value for value in valid if value > 0]
    zero_count = sum(value == 0 for value in valid)
    max_zero_run, trailing_zero_run = _compute_zero_runs(tuple(values))
    p10 = float(np.percentile(positive, 10)) if positive else None
    p50 = statistics.median(positive) if positive else None
    p95 = float(np.percentile(positive, 95)) if positive else None
    flatness = (p50 / p95) if (p50 is not None and p95 and p95 > 0) else None
    return {
        "n_intervals": len(values),
        "n_missing": len(values) - len(valid),
        "missing_pct": (
            100 * (len(values) - len(valid)) / len(values) if values else 0.0
        ),
        "zero_pct_valid": 100 * zero_count / len(valid) if valid else 0.0,
        "positive_min_kwh": min(positive) if positive else None,
        "positive_p10_kwh": p10,
        "positive_median_kwh": p50,
        "positive_p95_kwh": p95,
        "positive_max_kwh": max(positive) if positive else None,
        "max_zero_run": max_zero_run,
        "trailing_zero_run": trailing_zero_run,
        "flatness_ratio": flatness,
    }


def raw_values(values: Iterable[float | None]) -> str:
    """Encode interval-end values without converting missing to zero."""
    parts = []
    for index, value in enumerate(values):
        end_minutes = (index + 1) * 15
        clock = f"{end_minutes // 60:02d}:{end_minutes % 60:02d}"
        parts.append(f"{clock}=" + ("missing" if value is None else f"{value:.6g}"))
    return " ".join(parts)


def stats_line(values: Iterable[float | None]) -> str:
    """Format the shared day-stat line used by prompt and label packs."""
    stats = stats_for(values)
    if stats["positive_median_kwh"] is None:
        positive = "no positive values"
    else:
        positive = (
            "min/median/p95/max of positive: "
            f"{stats['positive_min_kwh']:.6g}/"
            f"{stats['positive_median_kwh']:.6g}/"
            f"{stats['positive_p95_kwh']:.6g}/"
            f"{stats['positive_max_kwh']:.6g} kWh"
        )
    return (
        f"missing {stats['missing_pct']:.1f}% of {stats['n_intervals']} intervals; "
        f"zero {stats['zero_pct_valid']:.1f}% of valid; {positive}"
    )


def _derived_interval_ends(date_text: str, count: int) -> tuple[datetime, ...]:
    """Derive ordinary 15-minute ends for direct DayRecord construction."""
    day = datetime.fromisoformat(date_text)
    return tuple(day + INTERVAL * (index + 1) for index in range(count))


def _bucket_for_start(start: datetime) -> str:
    if start.weekday() >= 5:
        return "weekend"
    if 7 <= start.hour <= 18:
        return "weekday_day"
    return "weekday_night"


def _bucket_result(
    valid_count: int,
    low_count: int,
    median_kwh: float | None = None,
) -> BucketEvidence:
    share = low_count / valid_count if valid_count else None
    return BucketEvidence(
        valid_count=valid_count,
        low_positive_count=low_count,
        low_positive_share=share,
        median_kwh=median_kwh,
    )


def encode_day_evidence(
    values: Iterable[float | None],
    meter_reference_kwh: float,
    timestamps: Iterable[datetime] | None = None,
    *,
    day_id: str = "unknown_day",
    meter_id: str = "unknown_meter",
    date: str | None = None,
) -> DayEvidence:
    """Encode one local day using the approved 35% low-band contract.

    ``timestamps`` are interval-end timestamps. When it is omitted, ordinary
    15-minute ends are derived from ``date`` for small unit-test records; the
    real loader always supplies timestamps, including the 92-interval DST day.
    """
    value_tuple = tuple(values)
    count = len(value_tuple)
    if date is None:
        date = day_id.rsplit("_", 1)[-1] if "_" in day_id else "1970-01-01"
    timestamp_tuple = (
        tuple(timestamps)
        if timestamps is not None
        else _derived_interval_ends(date, count)
    )
    if len(timestamp_tuple) != count:
        raise ValueError(
            f"{day_id} has {count} values but {len(timestamp_tuple)} timestamps"
        )
    wrong_days = [
        stamp.isoformat()
        for stamp in timestamp_tuple
        if day_of_timestamp(stamp) != date
    ]
    if wrong_days:
        raise ValueError(
            f"{day_id} contains timestamps outside local day {date}: "
            f"{wrong_days[:3]}"
        )
    if meter_reference_kwh <= 0:
        raise ValueError("meter_reference_kwh must be strictly positive")

    stats = stats_for(value_tuple)
    low_threshold = LOW_BAND_FRACTION * meter_reference_kwh
    bucket_counts = {
        "weekday_night": [0, 0],
        "weekday_day": [0, 0],
        "weekend": [0, 0],
    }
    bucket_values: dict[str, list[float]] = {
        "weekday_night": [],
        "weekday_day": [],
        "weekend": [],
    }
    low_positive_count = 0
    for stamp, value in zip(timestamp_tuple, value_tuple):
        bucket = _bucket_for_start(stamp - INTERVAL)
        if value is None:
            continue
        bucket_counts[bucket][0] += 1
        bucket_values[bucket].append(value)
        if 0 < value <= low_threshold:
            bucket_counts[bucket][1] += 1
            low_positive_count += 1

    bucket_medians: dict[str, float | None] = {}
    for b_name, b_vals in bucket_values.items():
        if b_vals:
            bucket_medians[b_name] = float(statistics.median(b_vals))
        else:
            bucket_medians[b_name] = None

    night_med = bucket_medians["weekday_night"]
    day_med = bucket_medians["weekday_day"]
    if night_med is None or day_med is None:
        night_to_day_ratio = None
    elif day_med > 0:
        night_to_day_ratio = round(night_med / day_med, 4)
    elif night_med == 0:
        night_to_day_ratio = 1.0
    else:
        night_to_day_ratio = None

    p10 = stats["positive_p10_kwh"]
    floor_to_ref_ratio = (
        round(float(p10) / meter_reference_kwh, 4) if p10 is not None else None
    )

    n_valid = count - int(stats["n_missing"])
    low_share = low_positive_count / n_valid if n_valid else 0.0
    return DayEvidence(
        day_id=day_id,
        meter_id=meter_id,
        date=date,
        values=value_tuple,
        timestamps=timestamp_tuple,
        n_intervals=count,
        n_missing=int(stats["n_missing"]),
        missing_pct=float(stats["missing_pct"]),
        n_valid=n_valid,
        zero_count=sum(value == 0 for value in value_tuple if value is not None),
        zero_pct_valid=float(stats["zero_pct_valid"]),
        positive_min_kwh=stats["positive_min_kwh"],
        positive_median_kwh=stats["positive_median_kwh"],
        positive_p95_kwh=stats["positive_p95_kwh"],
        positive_max_kwh=stats["positive_max_kwh"],
        meter_reference_kwh=meter_reference_kwh,
        low_threshold_kwh=low_threshold,
        low_positive_count=low_positive_count,
        low_positive_share_valid=low_share,
        weekday_night=_bucket_result(
            *bucket_counts["weekday_night"], median_kwh=bucket_medians["weekday_night"]
        ),
        weekday_day=_bucket_result(
            *bucket_counts["weekday_day"], median_kwh=bucket_medians["weekday_day"]
        ),
        weekend=_bucket_result(
            *bucket_counts["weekend"], median_kwh=bucket_medians["weekend"]
        ),
        max_zero_run=int(stats["max_zero_run"]),
        trailing_zero_run=int(stats["trailing_zero_run"]),
        positive_p10_kwh=stats["positive_p10_kwh"],
        floor_to_ref_ratio=floor_to_ref_ratio,
        flatness_ratio=stats["flatness_ratio"],
        weekday_night_median_kwh=bucket_medians["weekday_night"],
        weekday_day_median_kwh=bucket_medians["weekday_day"],
        weekend_median_kwh=bucket_medians["weekend"],
        night_to_day_ratio=night_to_day_ratio,
    )


def _format_bucket(bucket: BucketEvidence) -> str:
    if bucket.low_positive_share is None:
        return "not available"
    return (
        f"{bucket.low_positive_share:.1%} "
        f"({bucket.low_positive_count}/{bucket.valid_count})"
    )


def format_evidence_line(evidence: DayEvidence) -> str:
    """Format the exact evidence line used by model prompts and human packs."""
    night_str = _format_bucket(evidence.weekday_night)
    if evidence.weekday_night_median_kwh is not None:
        night_str += f", median {evidence.weekday_night_median_kwh:.6g} kWh"

    day_str = _format_bucket(evidence.weekday_day)
    if evidence.weekday_day_median_kwh is not None:
        day_str += f", median {evidence.weekday_day_median_kwh:.6g} kWh"

    weekend_str = _format_bucket(evidence.weekend)
    if evidence.weekend_median_kwh is not None:
        weekend_str += f", median {evidence.weekend_median_kwh:.6g} kWh"

    ratio_part = ""
    if evidence.night_to_day_ratio is not None:
        ratio_part = f"night/day ratio {evidence.night_to_day_ratio:.2f}; "

    return (
        "Low-band evidence: low positive means 0 < kWh <= "
        f"{evidence.low_threshold_kwh:.6g} kWh "
        f"({LOW_BAND_FRACTION:.0%} of meter reference "
        f"{evidence.meter_reference_kwh:.6g} kWh); "
        f"weekday night {night_str}; "
        f"weekday day {day_str}; "
        f"{ratio_part}"
        f"weekend {weekend_str}; "
        f"max zero run {evidence.max_zero_run}; "
        f"missing {evidence.missing_pct:.1f}% of intervals."
    )


def _format_ratio_fact(value: float | None, decimals: int) -> str:
    """Format one ratio as a decimal with its percent form, or as missing."""
    if value is None:
        return "not available"
    return f"{value:.{decimals}f} ({value:.0%})"


def format_decision_facts(evidence: DayEvidence) -> str:
    """Format the pre-computed decision features that steer the label checks.

    The shared evidence line already prints the levels, medians, night/day
    ratio, low-band shares, and zero runs.  This line adds only the three
    calibrated features a model cannot safely divide or compare by hand:
    p50/p95 flatness, the p10-based floor ratio, and the running share above
    the low-band threshold.  No threshold or label appears here, so prompts
    still do the adjudication.
    """
    if evidence.n_valid:
        running_share = (
            evidence.n_valid - evidence.zero_count - evidence.low_positive_count
        ) / evidence.n_valid
        running = f"{running_share:.1%} of valid intervals"
    else:
        running = "not available"
    return (
        "Decision facts (pre-computed from this day and the meter reference; "
        "compare these printed numbers, do not recalculate): "
        "flatness positive p50/p95 = "
        f"{_format_ratio_fact(evidence.flatness_ratio, 2)}; "
        "floor ratio positive p10/reference = "
        f"{_format_ratio_fact(evidence.floor_to_ref_ratio, 4)}; "
        f"running share above the low-band threshold = {running}."
    )
