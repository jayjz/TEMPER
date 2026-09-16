# Evaluation Protocol

## Objective

Prevent benchmark theater, calibration leakage, and cost claims that do not survive reproducible measurement.

## Dataset partitions

Use at minimum:

```text
train
validation
calibration
test
```

Where practical, add:

```text
OOD / temporal-shift test
adversarial test
```

Rules:

- `train`: parameter fitting.
- `validation`: architecture/hyperparameter/early-stopping choices.
- `calibration`: temperature or other post-hoc calibration fitting and threshold selection.
- `test`: touched once for final evaluation of a frozen configuration.
- `OOD`: tests generalization outside the fitting distribution.

For small datasets, use nested or repeated cross-validation rather than leaking test information.

## Baselines

Every experiment must include the simplest credible baselines.

Typical baseline set:

1. majority / frequency baseline
2. logistic regression or linear classifier on fixed embeddings
3. small encoder with cross-entropy
4. calibrated small encoder
5. generative SLM prompted for classification
6. frontier model, when cost allows
7. cascade system

Do not compare a tuned proposed method only against weak defaults.

## Metrics

### Predictive quality

- accuracy
- precision / recall / F1 where class imbalance matters
- AUROC / AUPRC when appropriate
- task-specific expected utility

### Probabilistic quality

Use multiple measures:

- negative log likelihood / log loss
- Brier score
- reliability diagrams
- ECE as a descriptive summary, not sole proof
- classwise calibration
- calibration slope/intercept where useful

### Selective prediction

For thresholded automation:

- coverage
- selective risk
- risk-coverage curve
- coverage at fixed risk targets
- risk at fixed coverage targets
- area under risk-coverage curve

Thresholds must be chosen on validation/calibration data, never the final test set.

### Systems

- median / p95 / p99 latency
- throughput
- batch size
- peak RAM / VRAM
- model load time
- input preprocessing time
- energy if measurable
- external API cost
- amortized training cost when making economic claims

## North-star systems metric

Define:

```text
CPVCD = total system cost / number of verified correct decisions
```

"Total system cost" must declare included components:

- specialist inference
- verifier inference
- frontier escalation
- external API calls
- training amortization
- monitoring / retraining if included

Never quote CPVCD without its accounting boundary.

## Calibration

Calibration means predicted probabilities correspond to empirical frequencies under a declared distribution.

Do not equate:

- high accuracy with calibration;
- low ECE with universal calibration;
- temperature scaling with guaranteed calibration;
- conformal coverage with calibrated probabilities.

Temperature scaling must be fit on held-out calibration data.

## Conformal prediction

If used:

- state the conformity score;
- state the exchangeability / distribution assumptions;
- report marginal versus class-conditional coverage;
- report prediction-set size;
- test coverage under shift.

Conformal prediction provides coverage properties under assumptions; it does not make arbitrary model probabilities calibrated.

## Repeated runs

For stochastic training:

- predeclare seeds;
- report mean and dispersion;
- preserve every run;
- do not select the best seed as the headline result.

For major claims, target at least 3-5 seeds if compute permits.

## Statistical analysis

Depending on the experiment:

- paired bootstrap confidence intervals
- bootstrap intervals for metric differences
- McNemar test for paired classifier errors
- permutation tests where suitable
- effect sizes, not p-values alone

Predeclare primary comparisons before inspecting final test results.

## Distribution shift

At least one research track must use:

- temporal split, or
- domain split, or
- synthetic controlled shift.

Measure both predictive degradation and calibration degradation.

## Error analysis

Every substantive experiment must inspect:

- high-confidence errors
- low-confidence correct cases
- systematic class confusions
- OOD cases
- escalation failures
- verifier/model disagreement
- teacher/student disagreement if distillation is used

High-confidence errors are priority evidence.

## Reproducibility

Every result must resolve to:

```text
experiment manifest
  -> git commit
  -> dataset hash/version
  -> model revision
  -> environment
  -> raw predictions
  -> metrics
  -> report
```

Summary tables alone are insufficient evidence.
