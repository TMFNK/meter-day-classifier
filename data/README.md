# Private human data (not shipped)

This directory holds the 60 hand-labeled meter-days that training and
evaluation need. They are private and deliberately absent from git
(see `.gitignore`). To reproduce calibration (`train.py`), the contract
test (`features.py --check-splits`), or scoring (`run_eval.py`), place
these three paths here:

- `data/labels_v1.json` — the 60 frozen labels (`version: labels_v1`,
  `split: train` × 30 + `split: eval` × 30)
- `data/raw/meter_<n>.csv` — raw 15-minute `timestamp,value_kwh` rows per meter
- `outputs/label_pack/selection.json` — the frozen 30/30 day-ID split

Everything else works without them:

- `python3 generate_synth_meters.py --root synth --meters 2 --days 5`
  regenerates synthetic training data from pinned seeds (no private input).
- `python3 features.py --build-synth` builds the weak-label matrix
  (the human-label decontamination check is skipped with a warning).
- `python3 train.py --variant synth` runs the full CV + refit on weak
  labels only (writes `synth/artifacts/` — keep the shipped files unless
  you mean to retrain).
- The shipped `synth/artifacts/model.joblib` + `operating_point.json`
  and `outputs/` predictions reproduce the published 90.0% number as-is.
