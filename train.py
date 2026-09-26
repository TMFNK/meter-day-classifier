#!/usr/bin/env python3
"""A2: weak-train CV + human-30 threshold calibration. Human eval never loaded.

Data roles (implementation plan section 10):
- FIT: synth 1080 weak days (stratified 5-fold CV, macro-F1 selection).
- CALIBRATE: human-train-30 only — the unsure-threshold operating point.
- SEALED: human-eval-30. This script refuses to load it (no --split eval path
  exists; load_labeled_records is called with split="train" only).
- Extra row: human-only-30 variant (3-fold CV on human-train) measures what
  weak pretraining buys. Fitting on human-train is allowed — it is train data.

Deviation from plan section 3, recorded: with 1080 rows for 26 features no
explicit feature selection is run in v1 (42 rows/feature is healthy);
selection is revisited only if CV shows overfit symptoms.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer  # noqa: F401 (kept for A3 reuse)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, build_matrix, load_synth_records

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
RANDOM_STATE = 38


def make_estimators():
    return {
        "logreg": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=5000,
                                       random_state=7)),
        ]),
        "rf": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                           random_state=7, n_jobs=-1)),
        ]),
    }


PARAM_GRIDS = {
    "logreg": {"clf__C": [0.05, 0.1, 0.5, 1.0, 5.0]},
    "rf": {"clf__max_depth": [2, 3, 4], "clf__min_samples_leaf": [5, 10]},
}


def cv_table(name, base, X, y, param_grid, n_splits):
    """Manual grid CV (small grids; keeps per-fold macro-F1 + accuracy)."""
    from itertools import product

    keys = list(param_grid)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for values in product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, values))
        est = base.set_params(**params)
        scores = cross_validate(est, X, y, cv=cv,
                                scoring={"macro_f1": "f1_macro", "acc": "accuracy"})
        rows.append({
            "model": name, **params,
            "macro_f1_mean": round(float(scores["test_macro_f1"].mean()), 4),
            "macro_f1_std": round(float(scores["test_macro_f1"].std()), 4),
            "acc_mean": round(float(scores["test_acc"].mean()), 4),
        })
    rows.sort(key=lambda r: r["macro_f1_mean"], reverse=True)
    return rows


def decline_curve(proba, classes, gold, label_order):
    """Combined decline rule: predict unsure when argmax is unsure OR the
    top1-top2 margin < t. At t=0 this is argmax-only; higher t adds
    low-confidence declines.

    Pick rule (pre-registered): threshold maximizing decline-F1 on the
    calibration set, where decline = predicting unsure and the positive
    class is gold unsure. Ties broken toward the smaller threshold
    (fewer forced declines).
    """
    order = {label: i for i, label in enumerate(label_order)}
    top2 = np.sort(proba, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    pred_top = classes[np.argmax(proba, axis=1)]
    argmax_unsure = pred_top == "unsure"
    gold_is_unsure = np.array([g == "unsure" for g in gold])
    curve = []
    for t in [round(v, 2) for v in np.arange(0.0, 0.55, 0.05)]:
        declined = argmax_unsure | (margin < t)
        pred = np.where(declined, "unsure", pred_top)
        tp = int((declined & gold_is_unsure).sum())
        fp = int((declined & ~gold_is_unsure).sum())
        fn = int((~declined & gold_is_unsure).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        curve.append({"t": t, "rate": round(float(declined.mean()), 3),
                      "prec": round(prec, 3), "rec": round(rec, 3),
                      "f1": round(f1, 3)})
    best = sorted(curve, key=lambda r: (r["f1"], -r["t"]), reverse=True)[0]
    return curve, best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("synth", "human30", "both"), default="both")
    parser.add_argument("--synth-root", type=Path, default=None)
    args = parser.parse_args()

    from features import load_labeled_records  # frozen train split only

    label_order = ["active", "standby", "off", "unsure"]
    cv_rows_all = {}

    if args.variant in ("synth", "both"):
        records = load_synth_records(args.synth_root)
        X, y, _ = build_matrix(records)
        print(f"synth train: n={len(y)} {dict(Counter(y))}")
        estimators = make_estimators()
        for name, base in estimators.items():
            rows = cv_table(name, base, np.array(X), np.array(y),
                            PARAM_GRIDS[name], n_splits=5)
            cv_rows_all[f"synth_{name}"] = rows
            for r in rows:
                print(r)

    if args.variant in ("human30", "both"):
        records = load_labeled_records(split="train")
        X, y, _ = build_matrix(records)
        print(f"human30 train: n={len(y)} {dict(Counter(y))}")
        estimators = make_estimators()
        for name, base in estimators.items():
            rows = cv_table(name, base, np.array(X), np.array(y),
                            PARAM_GRIDS[name], n_splits=3)
            cv_rows_all[f"human30_{name}"] = rows
            for r in rows:
                print(r)

    # Winner = best synth macro-F1 (weak pretraining is the primary row).
    synth_rows = [r for k, v in cv_rows_all.items()
                  if k.startswith("synth_") for r in v]
    winner = max(synth_rows, key=lambda r: r["macro_f1_mean"]) if synth_rows else None
    operating = {"cv_tables": cv_rows_all}

    if winner:
        print(f"winner: {winner}")
        estimators = make_estimators()
        est = estimators[winner["model"]].set_params(**{
            k: v for k, v in winner.items()
            if k.startswith("clf__")})
        records = load_synth_records(args.synth_root)
        X, y, _ = build_matrix(records)
        est.fit(np.array(X), np.array(y))

        # Calibrate the unsure threshold on HUMAN-TRAIN-30 (train split: allowed).
        # The human labels are private (see data/README.md): without them the
        # CV tables above are the whole public output and nothing is frozen.
        from features import load_labeled_records as load_train

        try:
            cal_records = load_train(split="train")
        except FileNotFoundError:
            print("calibration skipped: data/labels_v1.json is private and "
                  "absent — see data/README.md. CV tables above stand; no "
                  "artifacts written.")
            return
        Xc, yc, _ = build_matrix(cal_records)
        proba = est.predict_proba(np.array(Xc))
        curve, best = decline_curve(proba, est.classes_, yc, label_order)
        print("decline curve on human-train-30 (calibration):")
        for r in curve:
            print(r)
        print(f"frozen operating point: {best}")

        medians = est.named_steps["impute"].statistics_.tolist()
        operating.update({
            "winner": winner,
            "threshold": best["t"],
            "threshold_rule": "max decline-F1 on human-train-30, ties to smaller t",
            "features": FEATURES,
            "imputer_medians": medians,
            "label_order": label_order,
        })
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(est, ARTIFACTS / "model.joblib")
        with (ARTIFACTS / "operating_point.json").open("w") as f:
            json.dump(operating, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {ARTIFACTS / 'model.joblib'}")
        print(f"wrote {ARTIFACTS / 'operating_point.json'}")
    else:
        print("no synth variant run — nothing frozen")


if __name__ == "__main__":
    main()
