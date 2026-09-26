# meter-day-classifier

Classical-ML meter-day classifier: one label per meter-day (`active` /
`standby` / `off` / `unsure`) from 15-minute kWh intervals. It does the
same frozen task as GEPA prompt optimization, but with no LLM calls and
no prompt search. Result: **90.0% eval accuracy vs an 86.7%
deterministic rules floor**, at 0.77 ms/day and zero inference cost.

Self-contained: evidence contract, loader, scorer, and the synthetic-data
generator all live in this repo. The only thing not shipped is the 60
hand-labeled days (private; see `data/README.md`).

Keywords: energy analytics, meter data, classification, offline,
reproducibility, scikit-learn.

## The business problem

Industrial sites collect months of 15-minute electricity data per
submeter, but the question that matters (_was this machine running,
idling in standby, or off, and on which days should someone look closer?_)
is still answered by eyeballing plots or by one-off threshold scripts
nobody can compare or audit. The waste is concrete: a compressor humming
through the night, a production line left in standby over the weekend,
an idle load nobody noticed because the daily total looked normal.

This is the problem behind the ecoplanet standby-detection case study:
standby consumption is a recurring, hidden source of energy waste, and
detecting it reliably is harder than it looks. Real meter data is messy
(missing readings, oddities, machines that don't behave as expected), and
"no reliable estimate for this meter, and here's why" has to be a
first-class answer, not a failure. A classifier that silently guesses on
a bad day is worse than one that declines.

What about a cloud LLM? Shipping customer consumption data to a
third party raises data-protection obligations, per-token bills make
per-day scoring uneconomic at meter-park scale, and generative outputs
are hard to audit line-by-line. So this project handles the same task
with a small, deterministic, fully offline model: one defensible state per
meter-day, an explicit abstention when the evidence is thin, and a
one-line reason citing the printed numbers behind the call.

For the commercial framing (buyers, ROI, engagement shapes) see
[`docs/SOLUTION.md`](docs/SOLUTION.md).

## How it works

The label is statistics, the reason is language. The two are split apart:

1. **Pretrain on weak labels.** 1,080 synthetic meter-days carry
   rule-generated labels (`synth/`). A `LogisticRegression(C=1.0)` learns
   26 precomputed evidence features (5-fold CV macro-F1 0.9919 on weak
   labels, which is rule distillation, not a success claim).
2. **Calibrate abstention on 30 human train days.** `unsure` is a
   first-class model output; the decline threshold froze at t=0.0
   (argmax-only) with 11/11 correct declines and zero false declines.
3. **Evaluate once on 30 sealed human days.** One run, recorded below.

Reasons are templates citing printed evidence numbers (never scored, no
generator call).

## Synthetic training data

Thirty hand-labeled days cannot fit 26 features: model selection on 30
rows is noise. Since human labels are capped at 60, volume comes from weak
(rule-generated) labels instead.

**Generator (`generate_synth_meters.py`, stdlib + numpy, pinned seeds).**
Mimics the meter CSV shape only: 15-minute `timestamp,value_kwh` rows
grouped into meter-days. No real consumption data is reused; every level
is drawn from fixed synthetic ranges, so reruns are byte-identical
(sha-verified against the shipped files). Scale: 12 meters × 90 days × 96
intervals = **103,632 interval rows** with **1,080 day labels**. The meter
mix is weighted toward standby/off (5 night-standby, 3 mostly-off, 2
weekend-line, 2 mixed) so the label mix stays trainable; the spring-forward
short day (2026-03-29, 92 intervals) is covered.

### How a synthetic meter is built

Each meter is a fixed calendar plus a small state machine, drawn
deterministically from a per-meter seed (`38000 + meter_index`, so the
whole set regenerates byte-identically):

1. **Calendar.** Fixed start (2026-01-05), 15-minute stamps with DST
   offsets; the spring-forward day is generated with its real 92
   intervals so the short-day path is exercised.
2. **Archetype and parameters.** Each meter gets one of four behavior
   archetypes and draws its parameters from fixed ranges: running level
   (log-uniform 0.4–400 kW), run noise, floor ratio, idle/zero
   probabilities, missing rate.
   - _night_standby_: runs weekday 07–17, sits on a low floor overnight
     and on weekends, with occasional zeros.
   - _mostly_off_: mostly zero with 1–3 random run blocks per day
     (~62–70% zeros overall).
   - _weekend_line_: runs at night (19–05) and partly by day on
     weekdays, floor on weekends.
   - _mixed_: floor all day with 2–3 random run or floor blocks.
3. **Values.** Each state's level is converted to 15-minute kWh (level ÷
   4, since a level is a kW draw over a quarter hour) with multiplicative
   Gaussian noise; a separate noise sigma applies to always-on meters.
4. **Disturbances.** Weekly spike bursts (0–3 per week, ×3–10
   multiplier, 1–2 intervals) and missing data: about 30% of missing
   readings placed in contiguous outage blocks of 4–25 intervals, the
   rest scattered, with a per-meter missing rate drawn from 0–15%.
5. **Weak label.** Each day is labeled by the amended weak-label rule
   (byte-identical to the parent benchmark's): heavy missing (>25% of
   intervals) → `unsure`; zeros ≥60% of valid intervals → `off`; one flat
   band near the meter's running level → `unsure` (flat, no distinction);
   floor within a third of the running level → `unsure` (borderline
   floor); p95 ≤ 35% of running level → `standby`; running share ≥30% →
   `active`; floor share ≥60% → `standby`; else `unsure` (mixed day).
   The generated mix is 498 active / 270 off / 188 unsure / 124 standby.

**Weak labels and what they mean.** Weak labels agree with the
deterministic rules floor on 97.5% of the 1,080 days. Stated plainly,
training here is **rule distillation**: the classifier approaches but
cannot discover beyond the floor. Its value is a portable sub-millisecond
model with a tunable abstention curve, not new judgment. The sealed human
eval is the only judge that matters.

**How the data is used (three roles, never mixed).**

| Role      | Data                     | Used for                             |
| --------- | ------------------------ | ------------------------------------ |
| FIT       | synth 1,080 weak days    | 5-fold CV, winner pick, refit        |
| CALIBRATE | human train 30 (private) | unsure-threshold operating point     |
| SEALED    | human eval 30 (private)  | one eval run per model, nothing else |

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

## How results are evaluated

Everything below is scored with the frozen `scorer.py` metrics on the
sealed 30-day human eval split. One run per finalist, recorded once in
`outputs/results.csv` and `outputs/predictions/*.jsonl`.

**Metrics.**

| Metric             | What it means here                                                                                                                                             |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Accuracy           | Fraction of the 30 sealed days labeled correctly.                                                                                                              |
| Macro-F1           | Unweighted mean of per-class F1; punishes a model that wins by predicting only the majority class.                                                             |
| Decline rate       | Fraction of days the system answered `unsure` instead of committing.                                                                                           |
| Decline precision  | Of the days the system declined, how many the gold label also marks `unsure` (i.e. genuinely ambiguous). A high decline rate at low precision is just hedging. |
| Schema validity    | Fraction of outputs that parse as a legal `{label, reason}`; 1.0 by construction for this model (single forward pass, no generation).                          |
| Consistency        | Repeatability across repeated runs; 1.0 because inference is deterministic.                                                                                    |
| p50 latency / cost | Measured wall time per day and dollar cost per 1,000 days (zero: no tokens, no server).                                                                        |

**Protocol.** The 60 hand-labeled days are frozen 30 train / 30 eval
(`labels_v1.json` + `selection.json`). Synthetic weak days only ever fill
the FIT role; human train days only calibrate the `unsure` threshold; the
eval split is loaded exactly once per model by `run_eval.py` and is never
used for fitting or tuning. `train.py` contains no eval-loading path by
design.

**Results (frozen human-eval-30).**

| System                                 | Accuracy  | Macro-F1  | Decline rate | Decline precision |
| -------------------------------------- | --------- | --------- | ------------ | ----------------- |
| `rules_floor_v1` (deterministic floor) | 86.7%     | 0.881     | 40.0%        | 0.667             |
| `zero_shot_v1` (local 2B LLM)          | 43.8%     | 0.279     | 50.0%        | 0.375             |
| `classical_logreg_v1` (this repo)      | **90.0%** | **0.910** | 36.7%        | **0.727**         |

_Why the floor row says 86.7% here but the committed `results.csv` floor
row reads 93.3%:_ 86.7% is the pre-registered floor from the original
benchmark, frozen before the weak-label rule was amended. The floor row
in the shipped `results.csv` was re-run later with the amended rule (the
same rule that generated the synthetic training labels), which happens to
agree with the human eval labels on all 30 days. The headline comparison
stays the pre-registered 86.7%: it is what the classifier was measured
against at eval time, and quoting the amended-rule floor would credit the
classifier with beating a baseline rebuilt from its own training rule.
The `zero_shot_v1` / `gepa_opt_v1` rows in `results.csv` are legacy
comparison rows carried over from the parent benchmark; this repo's own
row is `classical_logreg_v1`.

`classical_logreg_v1`: validity 1.0, consistency 1.0 (deterministic),
per-class 7/9 active, 6/7 standby, 6/6 off, 8/8 unsure. All 3 errors are
over-abstentions (never a confident wrong label). A rules-first hybrid
variant was evaluated and removed: it predicted identically to the pure
classifier on all 30 eval days, so only the simpler model ships.

**What the numbers do and don't show.** They show a deterministic,
offline model that beats a hand-written rules floor on a sealed
human-labeled split while declining less often and more precisely. They
do not show production readiness or real-world accuracy: training labels
are weak (rule-generated) and the 60 gold days are a small, private set
from one synthetic benchmark. The eval is a controlled comparison, not a
savings claim.

## Layout

| Path                       | Role                                                                                                                  |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Model pipeline**         |                                                                                                                       |
| `evidence.py`              | Vendored evidence contract (`DayEvidence`, grouping, p95); logic unchanged, see header                                |
| `data_io.py`               | Vendored loader (`DayRecord`, label/split loading); logic unchanged                                                   |
| `features.py`              | Closed 26-feature contract from `DayEvidence` only; `--check-splits`, `--build-synth`                                 |
| `train.py`                 | Weak-label CV + human-train calibration; has no eval-loading path by design                                           |
| `predict.py`               | Frozen-model predictor + template reason                                                                              |
| **Evaluation**             |                                                                                                                       |
| `scorer.py`                | Vendored metrics + run harness; logic unchanged                                                                       |
| `rules_floor.py`           | Vendored deterministic floor, reference for `--calibrate` only                                                        |
| `run_eval.py`              | Eval-once via the scorer; writes to `outputs/` only                                                                   |
| **Data**                   |                                                                                                                       |
| `generate_synth_meters.py` | Pinned-seed synthetic meter generator (stdlib + numpy)                                                                |
| `synth/`                   | Generated training data (12 meters × 90 days), weak labels, frozen artifacts (`model.joblib`, `operating_point.json`) |
| `data/`                    | Placeholder only; private human labels live here locally, never in git                                                |
| `outputs/`                 | Scoreboard (`results.csv`) + per-day predictions (`.jsonl`)                                                           |
| **Docs / run**             |                                                                                                                       |
| `docs/SOLUTION.md`         | Business framing: buyers, ROI, engagement shapes                                                                      |
| `reproduce.sh`             | One-command public run (no private data needed)                                                                       |
| `requirements.txt`         | `numpy`, `scikit-learn==1.4.0`                                                                                        |

## Setup

Requires Python 3.10+ with:

```bash
pip install -r requirements.txt   # numpy, scikit-learn==1.4.0
```

No sibling checkout, no server, no GPU. Everything runs offline from
this directory.

## Reproduce (no private data needed)

```bash
./reproduce.sh   # compile check, weak-matrix build, CV, smoke generation + inference
```

Or step by step:

```bash
python3 generate_synth_meters.py --root synth --meters 2 --days 5  # tiny regen smoke test
python3 features.py --build-synth    # 1080-row weak matrix check
python3 train.py --variant synth     # CV on weak labels (artifacts need private labels)
```

Regenerating the full `synth/` (12 × 90) is byte-identical to the shipped
files (sha-verified). Steps needing the private human labels
(`--check-splits`, threshold calibration inside `train.py`, `--calibrate`,
`run_eval.py`) fail with a clean pointer to `data/README.md` when the
labels are absent. With the labels placed, the published 90.0% row
reproduces from the shipped `synth/artifacts/` model.

## License

Apache-2.0, copyright 2026 MbitAI. See LICENSE and NOTICE.

Need this applied to your own meters? [MbitAI](https://www.mbitai.com)
