# EXP-0001 — Conventional Decision Baseline on CLINC150

Status: COMPLETE

## Research Question

How well can conventional non-generative classifiers solve a bounded semantic decision task before calibration, abstention, or frontier-model escalation is introduced?

## Hypothesis

A conventional text classifier will establish a strong and inexpensive baseline on CLINC150, but raw predictive confidence will not be sufficient evidence for reliable selective automation.

## Null Hypothesis

A simple baseline performs too poorly or produces confidence signals too weak for later calibration and selective-prediction experiments to be meaningful.

## Scope

EXP-0001 evaluates conventional classification only.

It does not test:

- temperature scaling;
- Brier-oriented model training;
- OOD scoring methods;
- frontier-model escalation;
- parallel constrained decoding;
- reinforcement learning;
- dynamic label spaces.

Those require later experiments.

## Dataset

CLINC150 full dataset.

Primary source:

Larson et al.,
"An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction,"
EMNLP-IJCNLP 2019.

## Dataset Use

The official test partition remains untouched until the baseline configuration is frozen.

The official validation split will be deterministically partitioned into:

- model-validation subset;
- calibration subset.

EXP-0001 itself will not fit calibration parameters.

The calibration partition is reserved for later experiments.

## Initial Models

### B0 — Majority Baseline

Always predict the most frequent in-scope intent.

Purpose:

establish the trivial lower bound.

### B1 — TF-IDF + Logistic Regression

Features:

- word-level TF-IDF;
- default configuration documented in the manifest;
- multinomial logistic regression.

Purpose:

establish a strong inexpensive classical baseline.

### B2 — Small Bidirectional Encoder

Candidate architecture:

ModernBERT or another explicitly documented bidirectional encoder.

Exact model revision must be frozen before evaluation.

Purpose:

establish the neural specialist baseline.

The neural baseline is not required for the first executable pass if heavyweight ML dependencies have not yet been admitted.

## Primary Outcome

Macro F1 on in-scope intent classification.

## Secondary Outcomes

- accuracy;
- negative log likelihood;
- multiclass Brier score;
- raw maximum-softmax confidence;
- inference latency;
- throughput;
- model size.

## OOS Analysis

OOS examples will be evaluated separately.

EXP-0001 will record raw confidence behavior on OOS examples but will not claim an OOD detection method.

## Calibration

EXP-0001 does not fit temperature scaling or another calibration transformation.

Raw probabilities are preserved for EXP-0002.

## Selective Prediction

Risk-coverage curves may be generated descriptively from raw confidence.

Thresholds must not be optimized against the final test set.

Formal abstention experiments belong to P3.

## Seeds

Classical deterministic baselines use a declared random state where applicable.

Neural headline comparisons require multiple predeclared seeds.

## Evidence Requirements

Preserve:

- experiment manifest;
- dataset provenance;
- split indices;
- raw predictions;
- probabilities;
- metrics;
- environment information;
- model revision;
- preprocessing configuration.

## Interpretation Criteria

EXP-0001 establishes a baseline.

It does not establish that:

- specialist models beat frontier models;
- calibration is solved;
- OOS detection is solved;
- TEMPER lowers system cost;
- the TEMPER thesis is supported.

Those require later controlled experiments.

## Completion Criteria

EXP-0001 is complete when:

1. dataset acquisition is reproducible;
2. splits are frozen;
3. majority baseline is evaluated;
4. TF-IDF + logistic regression is evaluated;
5. raw predictions and probability outputs are preserved;
6. metrics are recorded;
7. no final-test tuning occurred;
8. limitations and errors are documented.

## Result Record

See [EXP-0001 B2 Validation Result](../../docs/results/EXP-0001-B2-validation-2026-09-17.md).

- B0, B1, and B2 validation completed.
- B2 completed all three frozen seeds: 13, 21, and 37.
- The calibration partition was unused; no calibration fit or threshold selection occurred.
- The final-test partition remains untouched.
- This result establishes a conventional validation baseline only; it makes no OOD, calibration, abstention, systems-efficiency, or TEMPER systems-thesis claim.
