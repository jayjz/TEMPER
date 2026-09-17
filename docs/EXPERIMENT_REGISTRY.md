# Experiment Registry

The registry is the durable index of TEMPER experiments.

Raw experiment artifacts belong under `experiments/`.

## Experiments

| ID | Phase | Question | Status | Result |
|---|---|---|---|---|
| EXP-0001 | P1 | CLINC150 conventional decision baseline | RUNNING | - |
| EXP-0002 | P2 | Temperature scaling vs raw probabilities | PLANNED | - |
| EXP-0003 | P2 | Cross-entropy vs Brier-oriented training | PLANNED | - |
| EXP-0004 | P3 | Risk-coverage and abstention | PLANNED | - |
| EXP-0005 | P4 | Specialist-to-frontier cascade | PLANNED | - |
| EXP-0006 | P5 | Parallel constrained decoding reproduction | PROPOSED | - |

## Status Vocabulary

### PLANNED

Protocol exists but execution has not begun.

### RUNNING

Experiment is currently being executed.

### COMPLETE

Protocol completed and artifacts are available.

### REPLICATED

A result was independently repeated under its declared replication conditions.

### NEGATIVE

A valid experiment did not support its research hypothesis.

### INCONCLUSIVE

Evidence does not distinguish competing hypotheses sufficiently.

### INVALIDATED

Methodological or implementation failure prevents scientific interpretation.

## Rules

Never:

- overwrite a completed experiment;
- silently replace raw predictions;
- change a completed experiment's dataset;
- reuse an experiment ID for a new run;
- promote a best seed while hiding other seeds.

Corrections require either:

- a new experiment; or
- an explicit amendment preserving the original artifact.