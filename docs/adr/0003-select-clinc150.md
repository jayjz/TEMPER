# ADR 0003 — Select CLINC150 for EXP-0001

Status: Accepted

## Decision

TEMPER will use the CLINC150 full dataset as the initial P1 benchmark.

## Why CLINC150

CLINC150 provides:

- 150 in-scope intent classes;
- 10 domains;
- explicit out-of-scope examples;
- fixed train, validation, and test partitions;
- bounded single-intent outputs;
- public research provenance;
- small enough size for rapid local experimentation.

The dataset was created specifically to evaluate intent classification in the presence of out-of-scope queries.

This aligns directly with TEMPER's research questions around:

- bounded decision inference;
- calibrated confidence;
- selective prediction;
- abstention;
- out-of-scope detection;
- escalation.

## Source

Primary source:

Larson et al.,
"An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction,"
EMNLP-IJCNLP 2019.

Repository:

clinc/oos-eval

UCI dataset identifier:

570

## License

The UCI distribution is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

## Dataset Structure

For the full dataset:

- each of 150 in-scope intents has:
  - 100 training examples;
  - 20 validation examples;
  - 30 test examples;

- out-of-scope data contains:
  - 100 training examples;
  - 100 validation examples;
  - 1,000 test examples.

All examples are English and single-intent.

## TEMPER Split Policy

TEMPER will preserve the official test set untouched.

For initial experiments:

- official training data:
  used for fitting model parameters;

- official validation data:
  divided deterministically into:
  - validation subset;
  - calibration subset;

- official test data:
  final evaluation only.

The OOS examples will not be silently mixed with in-scope training data.

Experiments must explicitly state whether OOS examples were used during training.

## Known Limitations

CLINC150 is not production traffic.

It was crowdsourced using seed phrases and scenario prompts.

Therefore:

- natural production frequency distributions are not represented;
- temporal drift is absent;
- annotation language may be cleaner than real user traffic;
- single-intent assumptions simplify real interactions.

These limitations prevent generalizing EXP-0001 directly to production systems.

## Why Not Banking77 First

Banking77 is a strong fine-grained intent benchmark but lacks CLINC150's explicit out-of-scope evaluation design.

It remains a candidate secondary benchmark.

## Why Not HDFS First

HDFS provides valuable system-log anomaly labels but is substantially larger and introduces log-specific preprocessing.

It is better suited to a later cross-domain replication.

## Consequence

EXP-0001 tests TEMPER's methodology on a manageable benchmark with explicit support for both in-scope classification and abstention/OOS research.

Success on CLINC150 will not establish the TEMPER thesis generally.

Cross-domain replication will still be required.