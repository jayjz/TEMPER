# TEMPER

[![Research Status](https://img.shields.io/badge/research-P1%20active-blue)](#current-status)
[![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Environment](https://img.shields.io/badge/environment-uv-DE5FE9)](https://docs.astral.sh/uv/)
[![Research](https://img.shields.io/badge/type-experimental%20research-6f42c1)](#research-scope)
[![Reproducibility](https://img.shields.io/badge/evidence-reproducibility%20first-blue)](#evidence-and-reproducibility)

**TEMPER** is an experimental research program studying whether recurring bounded decisions currently handled by expensive general-purpose models can be progressively transferred into smaller, calibrated, verifiable decision models.

The project focuses on the system around the model—not benchmark accuracy alone.

Its central question is whether **specialization + calibrated abstention + independent verification + selective escalation** can lower cost and latency without sacrificing verified system-level reliability.

>> **Research status:** P0 research foundation complete; P1 single-task baseline complete; P2 calibration study next.
> **Current experiment:** EXP-0001 - conventional decision baselines on CLINC150. B0/B1/B2 validation completed; B2 used frozen seeds 13, 21, and 37.
> **Thesis status:** hypothesis under investigation; not established. No TEMPER systems-thesis claim is supported yet.
> **Observed validation evidence:** B0 majority baseline and B1 TF-IDF + logistic regression have been executed on the frozen seed-42 validation partition.



---

## Research Thesis

> **Expensive general-purpose reasoning can, for recurring bounded decisions, be progressively compiled into smaller probabilistic decision models that deliver lower cost and latency while preserving system-level reliability through calibrated abstention, independent verification, and selective escalation.**

TEMPER treats this as a falsifiable hypothesis rather than an architectural assumption.

The stronger systems formulation asks whether a composed system:

```text
Specialist + Verifier + Escalation Policy + General-Purpose Model

can outperform direct use of a general-purpose model at a declared operating point with respect to:

verified decision quality;
inference cost;
latency;
throughput;
compute requirements;
calibration;
observability;
operational controllability.
Core Research Question

Under what conditions does a specialist + verifier + escalation system outperform direct frontier-model inference on cost per verified correct decision?

TEMPER is specifically interested in repeated, bounded decision problems where:

outputs can be explicitly enumerated or constrained;
decision quality can be independently evaluated;
uncertainty matters operationally;
incorrect automation has measurable cost;
repeated frontier-model reasoning may eventually become unnecessary.
System Hypothesis

The intended research question is not whether a small model can classify something.

It is whether the entire decision system can reduce dependence on expensive reasoning while keeping risk measurable and bounded.

Intelligence Crystallization

TEMPER uses intelligence crystallization as a working term for:

novel case
→ expensive reasoning
→ verified outcome
→ accumulated evidence
→ recurring decision region
→ specialist training
→ calibration
→ shadow evaluation
→ selective deployment
→ fewer expensive calls

The research question is whether this produces a real economic and reliability advantage after accounting for training, verification, calibration, escalation, monitoring, retraining, drift, and maintenance complexity.

North-Star Metric

TEMPER's primary systems metric is:

Cost Per Verified Correct Decision — CPVCD
CPVCD =
    total declared system cost
    --------------------------------
    number of verified correct decisions

The accounting boundary must always be declared.

Depending on the experiment, total system cost may include:

specialist inference;
verification;
frontier escalation;
external API calls;
local compute;
training amortization;
monitoring;
retraining.

Lower inference cost alone does not establish lower CPVCD.

Evaluation Model

TEMPER explicitly separates:

valid output
    !=
correct prediction
    !=
normalized probability
    !=
calibrated probability
    !=
safe automated action
Predictive quality
accuracy;
precision / recall / F1;
task-specific utility;
class-level error analysis.
Probabilistic quality
negative log likelihood;
multiclass Brier score;
reliability diagrams;
ECE as a descriptive summary;
class-conditional calibration;
high-confidence error analysis.
Selective prediction
coverage;
selective risk;
risk-coverage curves;
coverage at fixed risk;
risk at fixed coverage;
high-confidence failure rate.
Systems performance
median / p95 / p99 latency;
throughput;
model size;
RAM / VRAM;
model load time;
preprocessing time;
API cost;
compute cost.
Robustness
out-of-scope behavior;
domain shift;
temporal shift where available;
adversarial or controlled perturbations;
calibration degradation under shift.
Calibration and Abstention Are Separate Problems

TEMPER does not assume that the probability estimator with the best calibration also provides the best selective-routing signal.

Later experiments will compare confidence signals such as:

raw maximum softmax probability;
calibrated maximum probability;
entropy;
class margin;
logit-derived confidence;
model-specific OOD scores where appropriate.

Probability calibration asks whether confidence values mean what they claim. Selective prediction asks whether those values rank safe and unsafe decisions well enough to automate selectively.

Evidence Hierarchy

TEMPER prefers evidence in this order:

observed real-world outcome;
deterministic or formal verifier;
independently curated expert label;
established benchmark label with documented provenance;
multi-source weak supervision;
single-model pseudo-label;
model self-evaluation.

Lower-quality evidence may be useful for training.

It must not silently become higher-quality evaluation evidence.

Evidence and Reproducibility

Every substantive result should resolve through:

claim
→ experiment
→ metrics
→ raw predictions
→ manifest
→ model revision
→ dataset version/hash
→ label provenance
→ verification procedure

Failed and negative experiments are evidence and should be preserved.

Current Status
P0 — Research Foundation

Status: COMPLETE

P0 established the research and evidence contract before substantive model development.

Completed work includes:

research thesis and null hypotheses;
evaluation protocol;
evidence policy;
experiment registry;
research-question registry;
Pydantic experiment manifest;
reproducible Python 3.12 / uv environment;
security-oriented repository hygiene;
calibration and selective-prediction metric kernel;
deterministic unit tests;
benchmark-selection criteria;
CLINC150 selection;
external research baseline record;
predeclared EXP-0001 protocol.
Current Experiment
EXP-0001 — Conventional Decision Baseline on CLINC150

Status: COMPLETE

EXP-0001 asks:

How well can conventional non-generative classifiers solve a bounded semantic decision task before calibration, abstention, or frontier-model escalation is introduced?

The current stage is intentionally simple:

CLINC150
   │
   ├── B0 majority baseline
   ├── B1 TF-IDF + logistic regression
   └── B2 small bidirectional encoder

B0 and B1 validation execution is complete.

B2 is the next implementation target.

The final test partition has not been used for model selection, calibration, threshold selection, or the reported validation results.

Dataset Provenance

Dataset:

CLINC150 full

Source:

UCI 570 / clinc/oos-eval

Observed UCI archive SHA-256:

0d8ecc3e1edd7b25cabde0177544ce536ddf773844bc80ef1a75f36e7f030ea2

Canonical normalized dataset SHA-256:

fb3217519e3c601c7a9b019dfd6744bed8f2564833e2b7eac4a406cacb462489

The acquisition implementation handles the current UCI ZIP layout by:

candidate archive members
        ↓
UTF-8 / JSON decoding
        ↓
full CLINC150 schema validation
        ↓
canonical-content identity
        ↓
0 valid       → reject
1 unique      → accept
>1 distinct   → reject

This prevents AppleDouble/archive lookalikes from being treated as dataset payloads while retaining fail-closed behavior for ambiguous valid content.

Frozen Split

EXP-0001 uses the committed seed-42 split definition:

experiments/EXP-0001/splits/clinc150-full-seed-42.json

Partition sizes:

Partition	Examples
training	15,000
validation	1,500
calibration	1,500
final test	4,500

The official CLINC150 validation partition was deterministically divided so every one of the 150 intents contributes:

10 validation examples
10 calibration examples

The calibration partition is reserved for later work.

The final test partition remains reserved for frozen final evaluation.

First Observed Results

These measurements were produced on the frozen 1,500-example validation partition using producer revision:

d4cf036244607ab4f88a7c880628f1a046bf8f5f
B0 — Majority Baseline

The majority baseline always predicts one class.

Metric	Validation result
Accuracy	0.006667
Macro F1	0.0000883
NLL	27.4468
Multiclass Brier	1.98667
Fit time	~0.00016 s
Inference time / 1,500 examples	~0.00024 s

The approximately 1/150 accuracy is the expected floor for a balanced 150-class task.

Its poor NLL and Brier score reflect its degenerate, maximally confident one-class probability distribution.

B1 — TF-IDF + Logistic Regression

Configuration:

word TF-IDF
lowercase = true
ngram_range = (1, 1)
norm = l2

LogisticRegression
C = 1.0
solver = lbfgs
max_iter = 1000
random_state = 42

Observed validation results:

Metric	Validation result
Accuracy	0.88733
Macro F1	0.88640
NLL	1.07091
Multiclass Brier	0.36105
Fit time	~2.44 s
Inference time / 1,500 examples	~0.0091 s

Prediction artifact shape:

labels:        (1500,)
predictions:   (1500,)
probabilities: (1500, 150)
class labels:  (150,)

Probability rows were verified to sum to 1.0.

What EXP-0001 Has Established So Far

Observed evidence currently supports only the following narrow statements:

TEMPER can reproducibly acquire and canonicalize the selected CLINC150 source.
The seed-42 experiment split can be reproduced exactly.
The B0 and B1 evidence pipeline produces reconstructable predictions, probabilities, metrics, provenance, and runtime metadata.
A conventional TF-IDF + logistic-regression classifier substantially outperforms the trivial majority baseline on the frozen CLINC150 validation partition.
B1 therefore constitutes a credible inexpensive classical baseline for the next comparison.

These results do not establish:

calibrated probabilities;
reliable abstention;
OOD detection;
robustness under distribution shift;
superiority to an encoder model;
superiority to a frontier model;
lower cost per verified correct decision;
support for TEMPER's systems thesis.
Next Research Step
B2 — Small Bidirectional Encoder

The next EXP-0001 step is to freeze and evaluate a small bidirectional encoder using the same:

canonical dataset;
training partition;
seed-42 validation partition;
label mapping;
evidence contract;
primary and secondary metrics.

The exact model and revision must be frozen before evaluation.

B2 must not introduce:

calibration fitting;
threshold optimization;
OOD scoring;
frontier escalation;
final-test model selection;
reinforcement learning;
dynamic label spaces.

After B2 validation evidence is independently reviewed, EXP-0001 can determine whether the baseline configuration is sufficiently frozen for final-test evaluation.

Research Roadmap
Phase	Status
P0 — Research Foundation	Complete
P1 — Single-Task Baseline	Complete
P2 — Calibration Study	Planned
P3 — Selective Prediction	Planned
P4 — Cascade Economics	Planned
P5 — Dynamic Decision Spaces	Planned
P6 — Parallel Multi-Decision Inference	Planned
P7 — Distillation / Weak Supervision	Planned
P8 — Intelligence Crystallization	Planned
P9 — Decision Compiler / Runtime	Conditional

Later phases remain stage-gated.

Null Hypotheses

TEMPER is explicitly designed to permit failure.

H0.1

Specialized models do not materially improve cost per verified correct decision after verification, retraining, monitoring, and escalation costs are included.

H0.2

Calibration degrades too severely under distribution shift for confidence-based automation to remain useful.

H0.3

Dynamic decision spaces remove enough specialization advantage that general-purpose generative models remain operationally superior.

H0.4

Teacher-distilled specialists inherit teacher errors and cannot safely expand coverage without expensive human or external verification.

H0.5

Maintenance complexity of a decision fabric exceeds the compute savings it creates.

A result supporting any of these null hypotheses is useful.

Research Boundaries

TEMPER does not currently attempt to build:

AGI;
unrestricted recursive self-improvement;
autonomous multi-agent systems;
a general-purpose agent framework;
a chatbot;
a foundation model from scratch;
a clone of TypeSafe Jev;
an implementation based on undocumented RLCD internals;
a production serving platform.
Repository Structure
TEMPER/
├── docs/
│   ├── adr/
│   ├── agent-log/
│   ├── exec-plans/
│   ├── research/
│   ├── templates/
│   ├── EVALUATION_PROTOCOL.md
│   ├── EVIDENCE_POLICY.md
│   ├── EXPERIMENT_REGISTRY.md
│   ├── RESEARCH_QUESTIONS.md
│   ├── ROADMAP.md
│   └── THESIS.md
│
├── experiments/
│   └── EXP-0001/
│       ├── protocol.md
│       ├── prepare.py
│       ├── run_baselines.py
│       └── splits/
│           └── clinc150-full-seed-42.json
│
├── src/
│   └── temper/
│       ├── baselines/
│       ├── calibration/
│       ├── contracts/
│       ├── datasets/
│       ├── evaluation/
│       ├── evidence/
│       └── selective/
│
├── tests/
│   └── unit/
│
├── .python-version
├── .secrets.baseline
├── AGENTS.md
├── pyproject.toml
├── README.md
└── uv.lock
Agent Handoff Log

AI-assisted engineering sessions use:

docs/agent-log/AGENT_HANDOFF_LOG.md

The log is append-only and exists to preserve operational handoffs between Codex, Grok, and human review.

Each entry should record:

timestamp
agent/model
branch
base/head commit
objective
observed evidence
files changed
tests/checks
limitations/blockers
next handoff

The agent log is not a research source of truth.

Research claims remain governed by the experiment manifests, immutable artifacts, Git revisions, experiment protocol, evaluation protocol, and evidence policy.

Installation

TEMPER targets Python 3.12 and uses uv.

git clone https://github.com/jayjz/TEMPER.git
cd TEMPER

uv sync --group dev --group security

Run tests:

uv run pytest

Quality gates:

uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run bandit -r src
uv run pip-audit
git diff --check
Research Discipline

Changes affecting experimental interpretation should answer:

What hypothesis is being tested?
What evidence would falsify it?
Which dataset partition is being used?
Was the final test set touched?
Is the result predictive, probabilistic, selective, or systems-level?
What baseline is being compared?
Are raw outputs preserved?
What assumptions limit the claim?
Is the result local evidence or an external claim?
Can another person reproduce it?
Source-of-Truth Hierarchy

For TEMPER research state:

Git revision
    ↓
experiment protocol
    ↓
frozen dataset / split identity
    ↓
experiment manifest
    ↓
raw prediction artifact
    ↓
recomputed metrics
    ↓
README / agent notes / project summaries

README text and agent logs must never silently supersede experiment artifacts or research contracts.a