# Research Roadmap

The roadmap is stage-gated. Later phases are not automatically admitted.

## P0 — Research foundation

**Goal:** establish experimental discipline before model development.

Deliverables:

- thesis and null hypotheses
- evaluation protocol
- evidence policy
- experiment manifest format
- dataset split policy
- metric implementations with tests
- reproducible environment
- one selected benchmark task

Exit criteria:

- metric tests pass
- no test/calibration leakage
- experiment can be reproduced from a manifest
- benchmark license/provenance recorded

## P1 — Single-task baseline

**Question:** what does a conventional small encoder actually achieve?

Build:

- linear/fixed-embedding baseline
- ModernBERT-class encoder baseline
- cross-entropy training
- raw probability evaluation

Do not add:

- RL
- agents
- dynamic schemas
- synthetic teacher data

Exit criteria:

- repeated baseline runs recorded
- raw predictions preserved
- test set untouched until configuration frozen

## P2 — Calibration study

Compare:

- uncalibrated cross-entropy
- temperature scaling
- Brier-oriented training
- selected hybrid/alternative methods

Exit criteria:

- calibration metrics and diagrams
- repeated runs
- class-conditional analysis
- high-confidence error analysis
- explicit conclusion including null/negative outcome

## P3 — Selective prediction

Add abstention.

Research:

- threshold selection
- risk-coverage
- fixed-risk operating points
- OOD behavior

Exit criteria:

- risk-coverage curves
- thresholds selected without test leakage
- performance under at least one shift condition

## P4 — Cascade economics

Compare:

```text
specialist only
frontier only
specialist -> frontier
specialist -> verifier -> frontier
```

Exit criteria:

- declared cost model
- latency distribution
- CPVCD
- quality at matched cost and cost at matched quality
- sensitivity analysis for API pricing / local hardware assumptions

## P5 — Dynamic decision spaces

Investigate runtime-defined choices.

Candidate architectures:

- cross-encoder NLI scorer
- bi-encoder state/choice scoring
- shared state encoding with batched choice scoring

Exit criteria:

- held-out labels unseen during task-specific training
- calibration analysis as choice-set size changes
- latency scaling with number of choices
- comparison against fixed-head classifier

## P6 — Parallel multi-decision inference

One state, several questions.

Test:

- repeated independent inference
- shared encoder + multiple heads
- shared encoder + dynamic questions

Exit criteria:

- measured amortization benefit
- interference analysis
- per-head calibration
- memory/latency profile

## P7 — Distillation and weak supervision

Introduce teacher models only now.

Compare:

- real labels only
- teacher pseudo-labels
- multi-teacher consensus
- mixed real + weak labels

Exit criteria:

- teacher/student disagreement analysis
- external ground-truth evaluation
- no use of teacher agreement as sole success criterion

## P8 — Intelligence crystallization

Given historical escalations and verified outcomes:

1. detect recurring expensive decision regions;
2. estimate whether specialization is economically justified;
3. train candidate specialist;
4. calibrate;
5. shadow deploy;
6. compare against existing path;
7. admit only if predefined criteria pass;
8. monitor drift and retire when necessary.

Exit criteria:

- at least one end-to-end crystallization cycle
- measured net economic gain/loss
- regression controls
- rollback/retirement policy

## P9 — Decision compiler/runtime

Only after model behavior is understood.

Prototype:

```python
class Decision(BaseModel):
    severity: Literal["low", "medium", "high"]
    escalate: bool
```

Compiler responsibilities:

- validate decision semantics
- map schema to model query/heads
- batch compatible decisions
- attach calibration metadata
- enforce thresholds
- emit evidence records

Exit criteria:

- compiler cannot silently invent missing semantics
- schema/version provenance
- benchmarked overhead
- end-to-end evidence trace

## Long-term research directions

- continual specialization under drift
- multi-modal decision primitives
- causal rather than correlational decision models
- policy-aware expected utility
- hardware-aware model compilation
- specialist discovery
- learned escalation policies
- formal guarantees for bounded subdomains
