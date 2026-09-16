# Exec Plan 0001 — Research Foundation

Status: COMPLETE

Completed: 2026-09-16

## Objective

Establish TEMPER as a reproducible research repository before substantive model training.

## P0.1 — Repository Contract

Completed.

The following research contracts are present and aligned:

- README;
- thesis;
- roadmap;
- evaluation protocol;
- evidence policy;
- research questions;
- ADRs;
- AGENTS instructions.

## P0.2 — Python Environment

Completed.

- Python 3.12;
- uv environment;
- locked dependencies;
- reproducible pyproject configuration.

## P0.3 — Package Skeleton

Completed.

Current package boundaries include:

- calibration;
- contracts;
- evaluation;
- evidence;
- selective prediction.

No agent framework, serving stack, distributed system, or database was admitted.

## P0.4 — Experiment Contract

Completed.

The Pydantic experiment manifest records:

- experiment identity and status;
- dataset provenance;
- model identity and revision;
- tokenizer revision where applicable;
- hardware;
- split definition;
- preprocessing;
- objective;
- seed;
- hyperparameters;
- software environment;
- calibration method;
- threshold-selection method;
- metrics;
- artifacts;
- limitations;
- conclusion.

## P0.5 — Metric Kernel

Completed.

Implemented and tested:

- multiclass Brier score;
- negative log likelihood;
- expected calibration error;
- coverage;
- selective risk;
- risk-coverage curve construction.

ECE remains a descriptive summary rather than sole evidence of calibration quality.

## P0.6 — Dataset Selection

Completed.

ADR 0003 selects CLINC150 for EXP-0001.

The official final test data is reserved for frozen final evaluation.

Validation and calibration serve separate experimental roles.

## P0.7 — External Baseline Record

Completed.

The external-baseline record covers:

- probability calibration;
- selective prediction;
- ModernBERT-class encoder baselines;
- cascade economics;
- TypeSafe Jev and RLCD public claims;
- parallel constrained decoding.

External claims are explicitly separated from TEMPER reproductions.

## P0.8 — Security Controls

Completed for repository foundation.

Controls include:

- security-focused .gitignore;
- repomix generated-output exclusion;
- detect-secrets;
- Bandit;
- pip-audit;
- no committed model weights or private datasets.

## P0 Exit Criteria

Satisfied:

- repository contracts internally aligned;
- environment resolves reproducibly;
- experiment manifest validates;
- metric kernel has deterministic tests;
- CLINC150 formally selected;
- split policy documented;
- EXP-0001 protocol predeclared;
- final test results have not been used for model tuning;
- external research claims are separated from local evidence;
- security checks are operational.

## Next Phase

P1 begins with EXP-0001.

Initial comparisons:

- majority baseline;
- TF-IDF plus logistic regression;
- small bidirectional encoder.

P1 must preserve raw predictions and must not use final-test results for model or threshold selection.