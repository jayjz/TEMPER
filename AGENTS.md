# AGENTS.md

Instructions for coding and research agents working in this repository.

## Mission

Study whether specialized probabilistic decision models, combined with calibrated abstention, independent verification, and selective escalation, can reduce cost per verified correct decision relative to general-purpose generative inference.

## Non-negotiable research rules

- Do not optimize for confirming the thesis.
- Never turn a hypothesis into a fact.
- Preserve negative and null results.
- Do not silently change datasets, splits, metrics, seeds, thresholds, or baselines.
- Do not use test data for model selection, threshold selection, calibration, or early stopping.
- Calibration data must remain distinct from final test data.
- Report confidence intervals or repeated-run variation where meaningful.
- Do not report only the best seed.
- Do not compare latency across materially different hardware without labeling the comparison.
- Do not compare API cost and local inference cost without stating the accounting assumptions.
- Do not call pseudo-labels "ground truth."
- Do not call confidence thresholds "safety guarantees."
- Do not claim conformal coverage outside its assumptions.
- Do not infer unpublished Jev/RLCD architecture details.

## Required workflow for substantive changes

1. Read `README.md`.
2. Read `docs/THESIS.md`.
3. Read `docs/EVALUATION_PROTOCOL.md`.
4. Read the relevant active execution plan.
5. Inspect existing code and experiment records before changing anything.
6. State the hypothesis being tested.
7. Make the smallest change capable of testing it.
8. Run focused tests.
9. Run required static checks.
10. Record the experiment, including failures.
11. Update documentation only for claims established by evidence.

## Evidence hierarchy

Prefer, in order:

1. observed real-world outcome
2. deterministic or formal verifier
3. curated expert label
4. independently reproduced benchmark label
5. multi-teacher consensus / weak supervision
6. single-model pseudo-label
7. model self-evaluation

Lower levels may be useful but must be labeled.

## Experiment identity

Every experiment must record:

- immutable experiment ID
- git commit
- dataset version/hash
- split definition
- model identifier and revision
- preprocessing configuration
- hyperparameters
- random seed(s)
- hardware
- software environment
- metrics
- calibration method
- threshold-selection procedure
- artifacts
- conclusion
- limitations

## Prohibited shortcuts

Do not:

- build an agent framework during P0-P3
- add RL before supervised/calibration baselines exist
- add a vector database without a measured need
- add distributed training before single-device experiments establish value
- use LLM-generated labels as the only evaluation ground truth
- tune on the test set
- use a single metric as evidence of calibration
- bury regressions by changing the benchmark
- rewrite experiment history

## Code quality

Prefer:

- typed Python
- explicit dataclasses/Pydantic at system boundaries
- deterministic seeds where supported
- pure metric functions
- immutable experiment manifests
- small modules
- tests for metric calculations and split leakage
- machine-readable results alongside human-readable reports

Avoid premature abstractions.

## Agent handoff

Before ending a session, record:

- what changed
- what was verified
- what remains unverified
- exact commands run
- active hypothesis
- next bounded action

Do not leave a vague "continue research" instruction.
