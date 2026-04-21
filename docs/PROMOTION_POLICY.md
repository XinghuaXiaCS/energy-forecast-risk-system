# Model Promotion Policy

This document explains how models move from *candidate* to *champion* in this system.
It exists so that "champion–challenger" is a governance process, not just a name in code.

## Roles of each model

| Role | Purpose |
|---|---|
| **Baseline** | Simple, interpretable reference (logistic regression). Must never be worse than random. Used as a sanity check and as a safety-net model if more complex candidates regress. |
| **Candidates** | Richer models trained every training run (e.g. random forest, shallow tree). They compete to be promoted. |
| **Champion** | The single model currently used for production risk scoring and the one whose probabilities are isotonic-calibrated on the validation set. |
| **Challenger** | Any non-champion candidate that continues to be trained and scored in *shadow* (scored but not action-generating) so that the team can evaluate whether it should replace the champion next cycle. |
| **Fallback** | Rule-based score used when input data is too degraded for any model to be trusted. |

## Promotion criteria

A challenger candidate is promoted to champion only when **all** of the following hold:

1. **Utility beats the current champion** on the validation window, where utility is
   `AUC + 0.6 * PR-AUC - 0.4 * Brier`.
2. **Brier score does not regress** by more than 0.01 vs the current champion.
3. **Calibration is non-worse** after isotonic calibration on the validation window.
4. Improvement holds for **two consecutive training windows**, not a single run.
5. **Alert precision on High/Critical band** does not regress by more than 2 absolute
   percentage points on the last batch period.
6. **No adverse shift on a monitored subgroup** (region × horizon). No subgroup may
   regress in AUC by more than 0.03.

## Rollback triggers

Rollback to the previously promoted champion or to the fallback rule is triggered when:

- rolling 4-week AUC < `retraining.auc_floor`
- rolling 4-week PR-AUC < `retraining.pr_auc_floor`
- any monitored feature PSI > `retraining.max_population_stability_index`
- reviewer-reported false-alert rate exceeds agreed threshold for two consecutive weeks

## Cadence

- Candidate training: every training run
- Formal champion review: monthly (on `retraining.monthly_review_day`)
- Emergency review: triggered by any rollback condition above

## Current demo behaviour

In this repo, the first training run promotes whichever candidate has the highest
utility (since there is no prior champion). In a production deployment the candidate
would have to out-perform the deployed champion under the rules above before being
promoted, and the previous champion would be retained as the rollback target.
