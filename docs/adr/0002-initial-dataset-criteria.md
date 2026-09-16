# ADR 0002 — Initial Dataset Selection Criteria

Status: Accepted; fulfilled by ADR 0003

## Context

TEMPER's first experiment must test probability quality and selective prediction without introducing avoidable ambiguity from weak labels, subjective annotation, or frontier-model pseudo-labels.

The first dataset therefore needs strong provenance.

## Decision Criteria

The initial dataset should satisfy:

1. public or otherwise legally usable for research;
2. objective or independently documented labels;
3. enough examples for distinct train, validation, calibration, and test partitions;
4. bounded decision output;
5. no requirement for generative reasoning;
6. practical local preprocessing and training;
7. no sensitive personal data;
8. known class distribution;
9. reproducible acquisition;
10. preferably a domain or shift dimension for later robustness evaluation.

## Preferred Task Families

Initial preference order:

1. software or system event classification;
2. public intent classification;
3. public support-ticket routing;
4. public security-event classification with strong label provenance.

## Excluded Initial Domains

Do not begin with:

- healthcare diagnosis;
- credit decisions;
- financial trading;
- criminal-risk scoring;
- synthetic LLM-generated labels;
- subjective quality judgments.

These introduce unnecessary safety, provenance, or ground-truth ambiguity before the methodology is established.

## Evaluation Requirements

The chosen dataset must support four distinct roles:

- train;
- validation;
- calibration;
- final test.

The final test partition must not be used for:

- early stopping;
- hyperparameter selection;
- temperature fitting;
- threshold selection;
- architecture selection.

## Resolution

ADR 0003 selects CLINC150 for EXP-0001 and records its provenance, license, structure, split policy, and limitations.