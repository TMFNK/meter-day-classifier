# Vendored from MbitAI's shared meter-day harness on 2026-09-25.
# Logic is unchanged from the frozen original; only this header is new.
# Here ROOT resolves to meter-day-classifier, which mirrors the same
# data/ + outputs/label_pack/ layout (human labels are private, not shipped).
"""
Shared, label-blind loading and evidence formatting for the meter evaluation.

DESCRIPTION:
- Reuse the shared interval-end timestamp and day grouping implementation.
- Preserve missing values as None in raw data, stats, and evidence.
- Keep meter p95, the approved low-band evidence, and the pre-computed
  decision facts attached to each prompt representation.
- Provide one prompt representation for deterministic and model callers.

PREREQUISITES:
1. Python 3.8+ must be installed
2. uv package manager must be installed
3. numpy must be installed

SETUP FOR NEW USERS:
1. Install uv: `pip install uv` (or follow official instructions)
2. Navigate to this script directory
3. Install dependencies: `uv pip install -r requirements.txt`

FILE REQUIREMENTS:
- Meter CSVs under `data/raw/<meter_id>.csv` when loading real generated data

USAGE:
    uv run python -m unittest -v

OUTPUT:
- No files are written by this module.

DEPENDENCIES (automatically installed by uv):
- numpy>=1.24.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from evidence import (
    INTERVAL,
    DayEvidence,
    day_of_timestamp,
    encode_day_evidence,
    format_decision_facts,
    format_evidence_line,
    group_meter_rows,
    load_meter_days,
    load_meter_rows,
    meter_period_p95,
    parse_timestamp,
    raw_values,
    stats_for,
    stats_line,
)

ROOT = Path(__file__).resolve().parent

__all__ = [
    "INTERVAL",
    "DayEvidence",
    "day_of_timestamp",
    "load_meter_days",
    "load_meter_rows",
    "parse_timestamp",
]


@dataclass(frozen=True)
class DayRecord:
    day_id: str
    meter_id: str
    date: str
    split: str
    values: tuple[float | None, ...]
    meter_p95_kwh: float
    gold_label: str
    gold_note: str
    timestamps: tuple[datetime, ...] = ()

    @property
    def stats(self) -> dict[str, float | int | None]:
        return stats_for(self.values)

    @property
    def evidence(self) -> DayEvidence:
        """Return the shared structured low-band evidence for this day."""
        return encode_day_evidence(
            self.values,
            self.meter_p95_kwh,
            self.timestamps or None,
            day_id=self.day_id,
            meter_id=self.meter_id,
            date=self.date,
        )

    @property
    def prompt_input(self) -> str:
        return (
            f"Day (15-minute interval energy, kWh):\n{raw_values(self.values)}\n\n"
            f"Stats: {stats_line(self.values)}.\n"
            f"Meter reference: whole-period positive p95 = "
            f"{self.meter_p95_kwh:.6g} kWh per interval.\n"
            f"{format_evidence_line(self.evidence)}\n"
            f"{format_decision_facts(self.evidence)}"
        )


def load_labeled_records(
    labels_path: Path | None = None,
    split: str = "eval",
) -> list[DayRecord]:
    labels_path = labels_path or ROOT / "data" / "labels_v1.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"missing frozen labels at {labels_path}; fill data/labels_worksheet.csv "
            "and run freeze_labels.py first"
        )
    document = json.loads(labels_path.read_text())
    if not document.get("frozen") or document.get("version") != "labels_v1":
        raise ValueError("labels file is not the frozen labels_v1 contract")

    records = []
    day_cache: dict[str, dict[str, list[tuple[datetime, float | None]]]] = {}
    p95_cache: dict[str, float] = {}
    for entry in document["labels"]:
        if entry["split"] != split:
            continue
        meter_id = entry["meter_id"]
        if meter_id not in day_cache:
            day_cache[meter_id] = group_meter_rows(meter_id)
            p95_cache[meter_id] = meter_period_p95(meter_id)
        rows = day_cache[meter_id].get(entry["date"])
        if rows is None:
            raise ValueError(f"{entry['day_id']} is absent from meter CSV")
        values = [value for _, value in rows]
        timestamps = tuple(stamp for stamp, _ in rows)
        records.append(
            DayRecord(
                day_id=entry["day_id"],
                meter_id=meter_id,
                date=entry["date"],
                split=entry["split"],
                values=tuple(values),
                meter_p95_kwh=p95_cache[meter_id],
                gold_label=entry["label"],
                gold_note=entry.get("note", ""),
                timestamps=timestamps,
            )
        )
    if not records:
        raise ValueError(f"no frozen {split} records found")
    return records
