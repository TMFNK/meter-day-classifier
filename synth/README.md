# Generated synthetic training data (committed on purpose)

Everything here is the output of one command:

```bash
python3 generate_synth_meters.py --root synth
```

Regenerating is **byte-identical** to the committed files (pinned per-meter
seeds `38000 + meter_index`), so you can verify a rerun produced no drift.

| Path                     | What it is                                                          |
| ------------------------ | ------------------------------------------------------------------- |
| `data/raw/s38_m01.csv` … | 12 meter CSVs, 15-minute `timestamp,value_kwh` rows (~103,632 total) |
| `labels_synth38_v1.json` | 1,080 weak (rule-generated) day labels — the training targets        |
| `generator_stats.csv`    | Per-meter summary: archetype, seed, rows, missing %, weak-label mix  |

The meter IDs use an `s38_*` namespace so a synthetic day can never
collide with a private human-labeled day; `features.py --build-synth`
asserts zero overlap whenever the private labels are present.

**Not here:** the trained model lives in `../artifacts/`, and the private
60 hand-labeled gold days live in `../data/` (never committed). Keeping the
model out of this folder means regenerating the data cannot overwrite it.
