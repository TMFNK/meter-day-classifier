#!/usr/bin/env bash
# Reproduce meter-day-classifier: public pipeline, no private data needed.
# Steps touching the private human labels fail cleanly with a pointer to
# data/README.md (see README "Reproduce").
set -euo pipefail
cd "$(dirname "$0")"

python3 -m py_compile ./*.py
python3 features.py --build-synth
python3 train.py --variant synth

SMOKE_DIR="$(mktemp -d)"
trap 'rm -rf "$SMOKE_DIR"' EXIT
python3 generate_synth_meters.py --root "$SMOKE_DIR" --meters 2 --days 5
python3 -c "
from features import load_synth_records, build_matrix
from predict import make_predictor
from pathlib import Path
recs = load_synth_records(Path('$SMOKE_DIR'))
X, y, ids = build_matrix(recs)
assert X.shape[1] == 26
labels = [make_predictor()(r)['label'] for r in recs[:5]]
print('smoke inference:', labels)
"
