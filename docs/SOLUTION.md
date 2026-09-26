# Meter-day states as a business solution (MbitAI)

Who this is for, what it buys you, and how it is delivered. The technical
detail lives in `README.md` (method, scores, layout) and `data/README.md`
(private-data boundary).

## The problem in one paragraph

Every site with submeters collects 15-minute interval data, yet the
day-to-day question (was this meter running, idling, or off, and on which
days should someone look closer?) is still answered by eyeballing plots or
by brittle threshold rules nobody can compare. Cloud-model classifiers fix
the comparison at the price of shipping customer consumption data to a
third party, per-token bills, and answers nobody can audit. This project
exists so you do not have to choose between a number you can defend and a
pipeline you can afford.

## What the classifier delivers

- **One state per meter-day, with a reason.** `active` / `standby` / `off`
  plus an explicit `unsure` instead of a forced guess. Every label ships a
  one-line reason citing printed evidence numbers against the meter's own
  baseline; this is the artifact energy reviewers and customer
  questionnaires ask for.
- **Abstention you can tune.** `unsure` is a first-class output with a
  measured decline curve (36.7% rate at 0.727 precision on the eval set),
  rather than a silent default. Uncertain days queue for human review;
  confident days never misfire (zero confident errors on eval).
- **Deterministic, pinned, reproducible.** Pinned seeds end to end: the
  synthetic training data regenerates byte-identically, the model is a
  fixed logistic regression, inference is sub-millisecond. A state
  delivered to a customer re-runs identically months later.
- **Offline and data-sovereign.** Training and inference need no network,
  no GPU, no model server: scikit-learn on commodity hardware. Nothing
  metered leaves the machine. No cloud account, no per-token bill, and no
  data-protection addendum to negotiate.
- **Honest scoring.** 90.0% eval accuracy against a pinned 86.7%
  deterministic rules floor, with the per-class table and every per-day
  prediction committed in `outputs/`.

## Who buys it

| Buyer | Pain | Classifier answer |
| --- | --- | --- |
| Facility / energy manager | Idle load hiding in 15-min data; no ranked list of days worth a site visit | Day states + reasons per meter; `unsure` queue is the visit list |
| Utility / metering MSP | Customer-specific consumption patterns, per-site tuning cost | One feature contract, per-meter baseline; synthetic-data onboarding per site |
| ESCO / energy consultant | Savings claims need a defensible before/after day count | Pinned model + committed predictions; states re-run bit-identically for audits |
| Compliance / CISO | Consumption data in scope for GDPR / sectoral rules; cloud LLMs need DPAs | Fully offline pipeline; private labels never leave the premises |

## Engagement shapes (MbitAI)

1. **Assessment (fixed scope).** Your meter sample in, day states +
   reasons + scorecard out, under NDA, on your hardware or ours. Go/no-go
   on numbers, not slides.
2. **Pilot (your meters).** The classifier states a live feed next to your
   current rules; rescued days and decline-precision deltas are measured on
   your data with the same scorecard.
3. **Production + handover.** Pinned model for your meter park, abstention
   threshold set from your labeled days, runbooks for re-state and review,
   team training on the reason format.

Contact: [https://www.mbitai.com](https://www.mbitai.com)

## Costs to budget (measured, not estimated)

| Item | Observed |
| --- | --- |
| Synthetic training data (12 meters × 90 days) | Seconds, numpy only, byte-identical reruns |
| Classifier CV + refit (1,080 days, 26 features) | Seconds on CPU, scikit-learn |
| Inference | 0.77 ms/day, $0 per 1,000 (no tokens, no server) |
| License | Apache-2.0 (keeps your tree license-clean) |

No per-seat, per-meter, or per-token metering exists anywhere in the pipeline.

## Limits, stated plainly

- Training labels are weak (rule-generated): the model distills the rules
  family and cannot discover beyond it. Its value is portable inference
  plus a tunable abstention curve, not new judgment.
- The remaining errors are abstentions on near-noise-floor meters (whole
  meter scale ~0.1 kWh/interval); the model declines rather than commits.
- Each meter needs its own whole-period p95 reference; a meter with no
  history cannot be stated.
- Threshold calibration needs ~30 hand-labeled days per deployment; the 60
  reference labels behind the published number are private.

These are documented here because enterprise buyers should find limits
before the pilot does, and because each one is observable in the
committed predictions rather than hidden in model weights.
