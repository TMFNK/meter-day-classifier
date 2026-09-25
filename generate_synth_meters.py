#!/usr/bin/env python3
"""Synthetic meter generator for Project 38 (mbitai Meter Classifier Alt).

What this is:
- Mimics the ecoplanet case-study CSV shape ONLY: 15-minute interval rows
  with ``timestamp`` + ``value_kwh`` columns, grouped into meter-days.
  Structure-only mimicry. No private interview rows, parameters, or plots are
  reused — every level is drawn from the approved synthetic ranges below with
  pinned seeds, so reruns are byte-identical.
- Target scale: ~100,000 interval rows (~1,000+ meter-days at 96/day).
- Output day labels are CALIBRATED WEAK labels: the Eddie-approved amended
  weak-label rule from Project 37 (unit fix + >=30% running -> active +
  borderline-floor -> unsure), byte-identical logic, plus a ``--calibrate``
  step that measures the new days against the frozen human gold (train split
  only — eval stays sealed) and against the deterministic rules floor.

Provenance: state machine, value model, spike/missing models, and weak_label()
were vendored from Project 37's ``generate_meters.py`` (frozen, read-only).
Only the meter plan, scale, seeds, IDs, and calibration are new. The
``--calibrate`` step uses this repo's vendored ``data_io`` / ``rules_floor``
and reads the private human labels from ``--data-root`` (default: this repo
root) — report-only, it writes nothing outside ``--root``.

Outputs under --root (default: ``synth/`` next to this script):
- ``data/raw/s38_m<nn>.csv`` — timestamp,value_kwh per meter (the "100k rows").
- ``labels_synth38_v1.json`` — ~1,000 calibrated day labels (the "1000 rows").
- ``generator_stats.csv`` — per-meter summary.

Usage:
  python3 generate_synth_meters.py --root synth [--meters 12] [--days 90]
  python3 generate_synth_meters.py --root synth --calibrate [--data-root .]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from data_io import DayRecord, group_meter_rows, meter_period_p95
from rules_floor import predict as rules_predict

INTERVAL = timedelta(minutes=15)
PERIOD_START = date(2026, 1, 5)  # fixed calendar so DST day is always covered
SPRING_FORWARD = date(2026, 3, 29)  # 92-interval day, exercises the short-day path
OFFSET_SWITCH = datetime(2026, 3, 29, 2, 0)
SEED_BASE = 38000
LABEL_ORDER = ["active", "standby", "off", "unsure"]

# Meter plan: 12 meters x 90 days x 96 intervals ~= 103k rows. Archetype mix is
# weighted toward standby/off so the weak-label mix is trainable (the original
# 8-meter plan skewed active/unsure). missing_rate is drawn per meter.
METER_PLAN = [
    ("night_standby", 0.00, 0.02),
    ("night_standby", 0.03, 0.07),
    ("night_standby", 0.00, 0.03),
    ("night_standby", 0.05, 0.10),
    ("mostly_off", 0.00, 0.01),
    ("mostly_off", 0.00, 0.02),
    ("mostly_off", 0.02, 0.05),
    ("weekend_line", 0.05, 0.12),
    ("weekend_line", 0.08, 0.15),
    ("mixed", 0.02, 0.06),
    ("mixed", 0.04, 0.08),
    ("night_standby", 0.01, 0.04),
]


def offset_for(stamp):
    return "+01:00" if stamp < OFFSET_SWITCH else "+02:00"


def stamp_str(stamp):
    return stamp.strftime("%Y-%m-%d %H:%M") + offset_for(stamp)


def day_of(stamp):
    return (stamp - INTERVAL).date()


def day_stamps(day):
    base = datetime(day.year, day.month, day.day)
    out = []
    if day == SPRING_FORWARD:
        t = base + INTERVAL
        limit = base + timedelta(hours=1, minutes=45)
        while t <= limit:
            out.append(t)
            t += INTERVAL
        t = base + timedelta(hours=3)
        while t <= base + timedelta(days=1):
            out.append(t)
            t += INTERVAL
    else:
        t = base + INTERVAL
        while t <= base + timedelta(days=1):
            out.append(t)
            t += INTERVAL
    return out


def build_rows(meter_start, n_days):
    rows = []
    for k in range(n_days):
        for stamp in day_stamps(meter_start + timedelta(days=k)):
            rows.append({
                "stamp": stamp,
                "day": day_of(stamp),
                "hour": stamp.hour,
                "text": stamp_str(stamp),
            })
    return rows


def draw_params(archetype, rng):
    p = {"active_kw": float(np.exp(rng.uniform(np.log(0.4), np.log(400.0))))}
    p["run_sigma"] = float(rng.uniform(0.03, 0.12))
    p["floor_sigma_frac"] = float(rng.uniform(0.10, 0.30))
    if archetype in ("night_standby", "weekend_line"):
        p["floor_kw"] = p["active_kw"] * float(rng.uniform(0.015, 0.08))
    if archetype == "night_standby":
        p["idle_kw"] = p["floor_kw"] * float(rng.uniform(0.4, 0.8))
        p["zero_p"] = float(rng.uniform(0.0, 0.10))
    elif archetype == "mostly_off":
        p["zero_target"] = float(rng.uniform(0.62, 0.70))
    elif archetype == "always_on":
        p["flat_level_kw"] = p["active_kw"] * float(rng.uniform(0.9, 1.1))
        p["noise_sigma"] = float(rng.uniform(0.01, 0.05))
    elif archetype == "weekend_line":
        p["day_level_kw"] = p["active_kw"] * float(rng.uniform(0.5, 0.8))
        p["zero_p"] = float(rng.uniform(0.0, 0.10))
    elif archetype == "mixed":
        p["floor_ratio"] = float(rng.uniform(0.4, 0.6))
        p["floor_kw"] = p["active_kw"] * p["floor_ratio"]
    return p


def day_groups(rows):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["day"]].append(i)
    return dict(sorted(groups.items()))


def assign_states(archetype, p, rows, rng):
    states = ["ZERO"] * len(rows)
    levels = [0.0] * len(rows)
    groups = day_groups(rows)

    if archetype == "night_standby":
        for i, r in enumerate(rows):
            weekday = r["day"].weekday() < 5
            if weekday and 7 <= r["hour"] <= 17:
                states[i], levels[i] = "RUN", p["active_kw"]
            elif weekday:
                states[i], levels[i] = "FLOOR", p["floor_kw"]
            else:
                states[i], levels[i] = "IDLE", p["idle_kw"]
        for i in range(len(rows)):
            if states[i] in ("FLOOR", "IDLE") and rng.random() < p["zero_p"]:
                states[i], levels[i] = "ZERO", 0.0

    elif archetype == "mostly_off":
        for _, idxs in groups.items():
            n_day = len(idxs)
            run_steps = int(round(n_day * (1.0 - p["zero_target"])))
            n_blocks = int(rng.integers(1, 4))
            if run_steps < n_blocks:
                n_blocks = max(1, run_steps)
            base = run_steps // n_blocks
            rem = run_steps % n_blocks
            parts = [base + (1 if k < rem else 0) for k in range(n_blocks)]
            parts = [x for x in parts if x >= 1]
            mask = np.zeros(n_day, dtype=bool)
            for length in parts:
                if n_day - length + 1 <= 0:
                    continue
                for s in rng.permutation(n_day - length + 1):
                    if not mask[s:s + length].any():
                        mask[s:s + length] = True
                        break
            for j, i in enumerate(idxs):
                if mask[j]:
                    states[i], levels[i] = "RUN", p["active_kw"]
                else:
                    states[i], levels[i] = "ZERO", 0.0

    elif archetype == "always_on":
        for i in range(len(rows)):
            states[i], levels[i] = "RUN", p["flat_level_kw"]

    elif archetype == "weekend_line":
        for i, r in enumerate(rows):
            if r["hour"] >= 19 or r["hour"] <= 5:
                states[i], levels[i] = "RUN", p["active_kw"]
            elif r["day"].weekday() < 5:
                if rng.random() < 0.4:
                    states[i], levels[i] = "RUN", p["day_level_kw"]
                else:
                    states[i], levels[i] = "FLOOR", p["floor_kw"]
            else:
                states[i], levels[i] = "FLOOR", p["floor_kw"]
        for i in range(len(rows)):
            if states[i] == "FLOOR" and rng.random() < p["zero_p"]:
                states[i], levels[i] = "ZERO", 0.0

    elif archetype == "mixed":
        for i in range(len(rows)):
            states[i], levels[i] = "FLOOR", p["floor_kw"]
        for _, idxs in groups.items():
            slots = list(range(0, 24, 3))
            rng.shuffle(slots)
            n_blocks = int(rng.integers(2, 4))
            starts = sorted(slots[:n_blocks])
            use_run = bool(rng.random() < 0.5)
            cursor = -1
            for st in starts:
                st_eff = max(st, cursor)
                if st_eff >= 24:
                    break
                en = min(24, st_eff + int(rng.integers(3, 7)))
                if en - st_eff < 1:
                    continue
                for i in idxs:
                    h = rows[i]["hour"]
                    if st_eff <= h < en:
                        if use_run:
                            states[i], levels[i] = "RUN", p["active_kw"]
                        else:
                            states[i], levels[i] = "FLOOR", p["floor_kw"]
                cursor = en
                use_run = not use_run

    return states, levels


def make_values(states, levels, p, archetype, rng):
    vals = np.zeros(len(states))
    for i, st in enumerate(states):
        if st == "ZERO":
            vals[i] = 0.0
        elif st == "RUN":
            sig = p["noise_sigma"] if archetype == "always_on" else p["run_sigma"]
            vals[i] = max((levels[i] / 4.0) * (1.0 + rng.normal(0.0, sig)), 1e-6)
        else:
            vals[i] = max((levels[i] / 4.0) * (1.0 + rng.normal(0.0, p["floor_sigma_frac"])), 1e-6)
    return vals


def apply_spikes(vals, rows, rng):
    weeks = defaultdict(list)
    for i, r in enumerate(rows):
        iso = r["day"].isocalendar()
        weeks[(iso[0], iso[1])].append(i)
    for key in sorted(weeks):
        idxs = weeks[key]
        for _ in range(int(rng.integers(0, 4))):
            pos = int(rng.integers(0, len(idxs)))
            length = int(rng.integers(1, 3))
            mult = float(rng.uniform(3.0, 10.0))
            for j in range(pos, min(pos + length, len(idxs))):
                vals[idxs[j]] *= mult


def apply_missing(n_rows, rate, rng):
    missing = np.zeros(n_rows, dtype=bool)
    n_missing = int(round(rate * n_rows))
    if n_missing <= 0:
        return missing
    if n_missing <= 4:
        chosen = rng.choice(n_rows, size=n_missing, replace=False)
        missing[chosen] = True
        return missing
    block_total_target = int(round(0.3 * n_missing))
    n_blocks = int(rng.integers(1, 5))
    lengths = [int(rng.integers(4, 25)) for _ in range(n_blocks)]
    total = sum(lengths)
    while total > block_total_target:
        reduced = False
        for k in range(n_blocks):
            if lengths[k] > 4 and total > block_total_target:
                lengths[k] -= 1
                total -= 1
                reduced = True
        if not reduced:
            break
    block_placed = 0
    for length in lengths:
        if n_rows - length + 1 <= 0:
            continue
        for s in rng.permutation(n_rows - length + 1):
            if not missing[s:s + length].any():
                missing[s:s + length] = True
                block_placed += length
                break
    scattered_target = n_missing - block_placed
    if scattered_target > 0:
        free = np.where(~missing)[0]
        scattered_target = min(scattered_target, len(free))
        chosen = rng.choice(free, size=scattered_target, replace=False)
        missing[chosen] = True
    return missing


def weak_label(states, vals, missing, p):
    """Amended weak-label rule, byte-identical logic to Project 37 v1."""
    active_kw = p["active_kw"]
    total = len(states)
    n_missing = int(missing.sum())
    if total == 0 or n_missing / total > 0.25:
        return "unsure", "heavy_missing"
    valid = ~missing
    v = vals[valid]
    s = [states[i] for i in range(total) if valid[i]]
    if len(v) == 0:
        return "unsure", "heavy_missing"
    zero_share = sum(1 for x in s if x == "ZERO") / len(s)
    if zero_share >= 0.6:
        return "off", "zeros_dominate"
    p50, p95 = np.percentile(v, [50, 95])
    p95_kw = p95 * 4.0
    if p95 > 0 and p50 / p95 >= 0.75 and p95_kw >= 0.7 * active_kw:
        return "unsure", "flat_no_distinction"
    run_share = sum(1 for x in s if x == "RUN") / len(s)
    floor_share = sum(1 for x in s if x in ("FLOOR", "IDLE")) / len(s)
    ratio = p.get("floor_ratio")
    if ratio is not None and 0.35 <= ratio <= 0.65 and run_share > 0 and floor_share > 0:
        return "unsure", "borderline_floor"
    if p95_kw <= 0.35 * active_kw:
        return "standby", "low_day_floor"
    if run_share >= 0.30:
        return "active", "running_block_present"
    if floor_share >= 0.60:
        return "standby", "floor_dominates"
    return "unsure", "mixed_day"


def write_csv(path, rows, vals, missing):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "value_kwh"])
        for r, v, m in zip(rows, vals, missing):
            w.writerow([r["text"], "" if m else f"{v:.6f}"])


def generate(root, n_meters=None, n_days=90):
    plan = METER_PLAN if n_meters is None else (METER_PLAN * ((n_meters // len(METER_PLAN)) + 1))[:n_meters]
    days_out: list[dict] = []
    stats_rows = []
    total_rows = 0
    for idx, (archetype, miss_lo, miss_hi) in enumerate(plan):
        meter_id = f"s38_m{idx + 1:02d}"
        rng = np.random.default_rng(SEED_BASE + idx)
        missing_rate = float(rng.uniform(miss_lo, miss_hi))
        rows = build_rows(PERIOD_START, n_days)
        p = draw_params(archetype, rng)
        states, levels = assign_states(archetype, p, rows, rng)
        vals = make_values(states, levels, p, archetype, rng)
        apply_spikes(vals, rows, rng)
        missing = apply_missing(len(rows), missing_rate, rng)

        for d, idxs in day_groups(rows).items():
            sub_states = [states[i] for i in idxs]
            label, reason = weak_label(sub_states, vals[idxs], missing[idxs], p)
            days_out.append({
                "day_id": f"{meter_id}_{d.isoformat()}",
                "meter_id": meter_id,
                "date": d.isoformat(),
                "weak_label": label,
                "reason": reason,
                "n_intervals": len(idxs),
                "n_missing": int(missing[idxs].sum()),
            })

        write_csv(root / "data" / "raw" / f"{meter_id}.csv", rows, vals, missing)
        counts = Counter(d["weak_label"] for d in days_out if d["meter_id"] == meter_id)
        total_rows += len(rows)
        stats_rows.append({
            "meter_id": meter_id, "archetype": archetype, "seed": SEED_BASE + idx,
            "rows": len(rows), "missing_pct": round(100.0 * missing.sum() / len(rows), 2),
            "active_kw": round(p["active_kw"], 4),
            **{f"weak_{lab}": counts.get(lab, 0) for lab in LABEL_ORDER},
        })
        print(meter_id, archetype, "rows", len(rows),
              "missing%", stats_rows[-1]["missing_pct"], dict(counts))

    labels_doc = {
        "version": "synth38_v1",
        "frozen": False,
        "generator": Path(__file__).name,
        "seed_base": SEED_BASE,
        "period_start": PERIOD_START.isoformat(),
        "n_days_per_meter": n_days,
        "n_interval_rows": total_rows,
        "weak_label_counts": dict(Counter(d["weak_label"] for d in days_out)),
        "days": days_out,
    }
    labels_path = root / "labels_synth38_v1.json"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w") as f:
        json.dump(labels_doc, f, indent=2, sort_keys=True)
        f.write("\n")

    stats_path = root / "generator_stats.csv"
    with stats_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)

    digest = hashlib.sha256(labels_path.read_bytes()).hexdigest()[:16]
    print(f"days: {len(days_out)} interval rows: {total_rows} labels sha: {digest}")
    print("wrote", labels_path)
    print("wrote", stats_path)
    return labels_doc


def calibrate(root, data_root):
    """Report-only calibration. Reads private human gold, writes nothing there."""
    labels_path = data_root / "data" / "labels_v1.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"calibration needs the private human labels at {labels_path} — "
            "see data/README.md. Generation (--root without --calibrate) "
            "needs no private data."
        )
    labels_doc = json.loads((root / "labels_synth38_v1.json").read_text())
    frozen_ids = {e["day_id"] for e in
                  json.loads(labels_path.read_text())["labels"]}
    synth_ids = {d["day_id"] for d in labels_doc["days"]}
    overlap = frozen_ids & synth_ids
    print(f"decontamination: frozen={len(frozen_ids)} synth={len(synth_ids)} overlap={len(overlap)}")
    assert not overlap, f"CONTAMINATED: {sorted(overlap)[:5]}"

    # Human side needs gold labels only — read the labels file directly so
    # no human meter CSVs are ever loaded through the vendored ROOT default.
    human_mix = Counter(e["label"] for e in json.loads(labels_path.read_text())["labels"]
                        if e["split"] == "train")
    synth_mix = Counter(d["weak_label"] for d in labels_doc["days"])
    print(f"human-train-30 mix: {dict(human_mix)}")
    print(f"synth weak mix:     {dict(synth_mix)}")

    # weak <-> rules agreement on every synth day (quantifies rule-distillation overlap)
    p95_cache: dict[str, float] = {}
    agree = 0
    total = 0
    rule_mix: Counter = Counter()
    for meter_file in sorted((root / "data" / "raw").glob("s38_m*.csv")):
        meter_id = meter_file.stem
        groups = group_meter_rows(meter_id, root=root)
        p95_cache[meter_id] = meter_period_p95(meter_id, root=root)
    by_day = {d["day_id"]: d for d in labels_doc["days"]}
    for day_id, d in by_day.items():
        rows = group_meter_rows(d["meter_id"], root=root).get(d["date"])
        values = tuple(v for _, v in rows)
        stamps = tuple(s for s, _ in rows)
        rec = DayRecord(day_id=day_id, meter_id=d["meter_id"], date=d["date"], split="synth",
                        values=values, meter_p95_kwh=p95_cache[d["meter_id"]],
                        gold_label=d["weak_label"], gold_note=d["reason"], timestamps=stamps)
        rule_label = rules_predict(rec)["label"]
        rule_mix[rule_label] += 1
        agree += rule_label == d["weak_label"]
        total += 1
    print(f"rules mix on synth: {dict(rule_mix)}")
    print(f"weak<->rules agreement: {agree}/{total} = {agree / total:.1%}")
    print("NOTE: high agreement means a classifier trained here distills the rules family; "
          "it can approach but not discover beyond the 86.7% floor.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--meters", type=int, default=None)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--data-root", default=None,
                    help="repo root holding the private data/labels_v1.json "
                         "(calibration only; default: this script's directory)")
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    root = Path(args.root) if args.root else here / "synth"
    if args.calibrate:
        data_root = Path(args.data_root) if args.data_root else here
        calibrate(root, data_root)
    else:
        generate(root, n_meters=args.meters, n_days=args.days)


if __name__ == "__main__":
    main()
