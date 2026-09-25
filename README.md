# meter-day-classifier

Classical-ML meter-day classifier: one label per meter-day (`active` /
`standby` / `off` / `unsure`) from 15-minute kWh intervals. A non-prompt
alternative to GEPA prompt optimization on the same frozen task — no LLM
calls, no prompt search. Result: **90.0% eval accuracy vs an 86.7%
deterministic rules floor**, at 0.77 ms/day and zero inference cost.

Self-contained: evidence contract, loader, scorer, and the synthetic-data
generator all live in this repo. The only thing not shipped is the 60
hand-labeled days (private — see `data/README.md`).

Keywords: energy analytics, meter data, classification, offline,
reproducibility, scikit-learn.

## For business readers (MbitAI solution)

For the business presentation start here: [`docs/SOLUTION.md`](docs/SOLUTION.md)
(buyers, ROI, engagement shapes).

## How it works

The label is treated as statistics, the reason as language — split apart:

1. **Pretrain on weak labels.** 1,080 synthetic meter-days carry
   rule-generated labels (`synth/`). A `LogisticRegression(C=1.0)` learns
   26 precomputed evidence features (5-fold CV macro-F1 0.9919 on weak
   labels — rule distillation, not a success claim).
2. **Calibrate abstention on 30 human train days.** `unsure` is a
   first-class model output; the decline threshold froze at t=0.0
   (argmax-only) with 11/11 correct declines and zero false declines.
3. **Evaluate once on 30 sealed human days.** One run, recorded below.

Reasons are templates citing printed evidence numbers (never scored, no
generator call).

## Synthetic training data

Thirty hand-labeled days cannot fit 26 features — model selection on 30
rows is noise. Since human labels are capped at 60, volume comes from weak
(rule-generated) labels instead.

**Generator (`generate_synth_meters.py`, stdlib + numpy, pinned seeds).**
Mimics the meter CSV shape only: 15-minute `timestamp,value_kwh` rows
grouped into meter-days. No real consumption data is reused — every level
is drawn from fixed synthetic ranges, so reruns are byte-identical
(sha-verified against the shipped files). Scale: 12 meters × 90 days × 96
intervals = **103,632 interval rows** with **1,080 day labels**. The meter
mix is weighted toward standby/off (5 night-standby, 3 mostly-off, 2
weekend-line, 2 mixed) so the label mix stays trainable; the spring-forward
short day (2026-03-29, 92 intervals) is covered.

**Weak labels.** Each synthetic day is labeled by the amended
weak-label rule (unit fix, ≥30% running share → active, borderline floor
→ unsure). Weak labels agree with the deterministic rules floor on 97.5%
of the 1,080 days — stated plainly, training here is **rule distillation**:
the classifier approaches but cannot discover beyond the floor. Its value
is a portable sub-millisecond model plus a tunable abstention curve, not
new judgment. The sealed human eval is the only judge that matters.

**How the data is used (three roles, never mixed).**

| Role | Data | Used for |
|---|---|---|
| FIT | synth 1,080 weak days | 5-fold CV, winner pick, refit |
| CALIBRATE | human train 30 (private) | unsure-threshold operating point |
| SEALED | human eval 30 (private) | one eval run per model, nothing else |

Decontamination is enforced by construction (synthetic IDs live in their
own `s38_*` namespace) and asserted whenever the private labels are
present: synth-vs-human day-ID overlap is 0. Without the labels the check
is skipped with a warning.

Regenerate the full set (byte-identical) or a smoke-size one:

```bash
python3 generate_synth_meters.py --root synth
python3 generate_synth_meters.py --root /tmp/smoke --meters 2 --days 5
python3 generate_synth_meters.py --root synth --calibrate --data-root .  # needs private labels
```

## Results (frozen human-eval-30)

| System | Accuracy | Macro-F1 | Decline rate | Decline precision |
|---|---|---|---|---|
| `rules_floor_v1` (deterministic floor) | 86.7% | 0.881 | 40.0% | 0.667 |
| `zero_shot_v1` (local 2B LLM) | 43.8% | 0.279 | 50.0% | 0.375 |
| `classical_logreg_v1` (this repo) | **90.0%** | **0.910** | 36.7% | **0.727** |

`classical_logreg_v1`: validity 1.0, consistency 1.0 (deterministic),
per-class 7/9 active, 6/7 standby, 6/6 off, 8/8 unsure. All 3 errors are
over-abstentions (never a confident wrong label). A rules-first hybrid
variant was evaluated and removed: it predicted identically to the pure
classifier on all 30 eval days, so only the simpler model ships.

## Layout

| Path | Role |
|---|---|
| `evidence.py` | Vendored evidence contract (`DayEvidence`, grouping, p95) — logic unchanged, see header |
| `data_io.py` | Vendored loader (`DayRecord`, label/split loading) — logic unchanged |
| `scorer.py` | Vendored metrics + run harness — logic unchanged |
| `rules_floor.py` | Vendored deterministic floor, reference for `--calibrate` only |
| `generate_synth_meters.py` | Pinned-seed synthetic meter generator (stdlib + numpy) |
| `features.py` | Closed 26-feature contract from `DayEvidence` only; `--check-splits`, `--build-synth` |
| `train.py` | Weak-label CV + human-train calibration; has no eval-loading path by design |
| `predict.py` | Frozen-model predictor + template reason |
| `run_eval.py` | Eval-once via the scorer; writes to `outputs/` only |
| `synth/` | Generated training data (12 meters × 90 days), weak labels, frozen artifacts (`model.joblib`, `operating_point.json`) |
| `data/` | Placeholder only — private human labels live here locally, never in git |
| `outputs/` | Scoreboard (`results.csv`) + per-day predictions (`.jsonl`) |

## Setup

Requires Python 3.10+ with:

```bash
pip install -r requirements.txt   # numpy, scikit-learn==1.4.0
```

No sibling checkout, no server, no GPU — everything runs offline from
this directory.

## Reproduce (no private data needed)

```bash
python3 generate_synth_meters.py --root synth --meters 2 --days 5  # tiny regen smoke test
python3 features.py --build-synth    # 1080-row weak matrix check
python3 train.py --variant synth     # CV on weak labels (artifacts need private labels)
```

Regenerating the full `synth/` (12 × 90) is byte-identical to the shipped
files (sha-verified). Steps needing the private human labels —
`--check-splits`, threshold calibration inside `train.py`, `--calibrate`,
`run_eval.py` — fail with a clean pointer to `data/README.md` when the
labels are absent. With the labels placed, the published 90.0% row
reproduces from the shipped `synth/artifacts/` model.

## License

Apache-2.0, copyright 2026 MbitAI. See LICENSE and NOTICE.

Need this applied to your own meters? [MbitAI](https://www.mbitai.com)
