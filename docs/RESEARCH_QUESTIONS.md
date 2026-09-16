# Research Questions

TEMPER is organized around falsifiable research questions rather than feature milestones.

## RQ1 — Decision Specialization

Under what task characteristics can a specialized decision model outperform direct general-purpose generative inference on cost per verified correct decision?

Variables include decision frequency, output-space size, input complexity, distribution stability, verification cost, error asymmetry, model size, latency requirements, and escalation cost.

## RQ2 — Probability Calibration

Which training and post-hoc methods produce probabilities that best correspond to empirical outcome frequencies?

Initial comparisons include:

- cross-entropy;
- Brier-oriented training;
- temperature scaling;
- selected hybrid objectives.

Measurements include:

- log loss;
- Brier score;
- reliability diagrams;
- ECE as a descriptive statistic;
- class-conditional calibration;
- calibration under distribution shift.

Low ECE alone is not sufficient evidence of reliable uncertainty.

## RQ3 — Selective Prediction

How much workload can a specialist accept while remaining below a declared error target?

Primary measurements include:

- coverage;
- selective risk;
- risk-coverage curves;
- coverage at fixed risk;
- risk at fixed coverage;
- high-confidence error rate.

## RQ4 — Cascade Economics

Can a specialist followed by an uncertainty gate and selective frontier escalation preserve or improve system quality while materially reducing expensive inference?

Controls should include:

- specialist only;
- frontier only;
- confidence-only cascade;
- specialist plus verifier;
- specialist plus frontier;
- specialist plus verifier plus frontier.

## RQ5 — Parallel Constrained Decoding

For bounded structured decisions, how much autoregressive generation cost comes from output serialization rather than semantic reasoning?

Compare ordinary autoregressive structured generation against shared-prefix or parallel constrained candidate scoring.

Measure:

- latency;
- throughput;
- forward-pass count;
- memory;
- schema validity;
- task accuracy;
- probability calibration.

External implementations may be used as baselines, but reported performance must be independently reproduced before TEMPER adopts the claim.

## RQ6 — Softmax Is Not Calibration

Does candidate-restricted softmax produce empirically meaningful probabilities without calibration training or post-hoc correction?

Working hypothesis:

normalized candidate logits are not automatically empirically calibrated probabilities.

## RQ7 — Dynamic Decision Spaces

Can runtime-defined semantic choices be handled without fixed task-specific classifier heads?

Candidate approaches include:

- bi-encoder compatibility;
- cross-encoder or NLI scoring;
- shared-state encoding with batched choice scoring;
- constrained decoder candidate scoring.

Important problems include new labels, changing choice-set cardinality, semantic overlap, probability normalization, and calibration as choices change.

## RQ8 — Multi-Decision Inference

Can one representation of a state support many decisions without repeatedly paying full inference cost?

Measure:

- state-encoding amortization;
- latency scaling;
- memory;
- task interference;
- per-decision calibration.

## RQ9 — Calibration-Aware Training

Does directly optimizing probability quality produce a superior selective-risk frontier compared with ordinary cross-entropy plus post-hoc calibration?

Initial methods should remain simple.

Later work may evaluate reinforcement-learning objectives informed by proper scoring rules.

## RQ10 — Distillation

When can expensive model behavior be transferred into a smaller specialist?

Label evidence preference:

1. observed outcome;
2. deterministic verifier;
3. expert label;
4. established benchmark;
5. multi-teacher weak supervision;
6. single-teacher pseudo-label;
7. self-evaluation.

Teacher outputs are not automatically ground truth.

## RQ11 — Intelligence Crystallization

Given repeated frontier-model decisions and independently verified outcomes, can TEMPER identify stable subproblems worth compiling into specialists?

Questions include:

- when is a pattern repetitive enough;
- when does training cost become justified;
- how much coverage can move to the specialist;
- when does drift invalidate the specialist;
- when should the specialist be retired.

## RQ12 — Decision Compilation

Can typed software contracts become declarations for learned inference?

A schema specifies output structure but not sufficient semantics.

A valid compiler must detect missing semantics rather than silently inventing them.

## RQ13 — Failure Boundaries

Where should this architecture not be used?

Test explicitly:

- open-ended creative tasks;
- novel multi-step reasoning;
- ambiguous policies;
- changing labels;
- concept drift;
- rare classes;
- adversarial input;
- tasks with unavailable verification;
- tasks whose uncertainty cannot be reliably estimated.

A useful theory must explain its own boundary conditions.
## RQ14 — Historical Replay and Policy Evaluation

Can preserved decision trajectories serve as an offline replay substrate for evaluating alternative abstention, verification, and escalation policies without repeating expensive model calls?

The key distinction is between model improvement and policy improvement.

A replay experiment should hold historical model outputs and observed outcomes fixed while varying only system policy, such as:

- confidence thresholds;
- abstention rules;
- verification gates;
- escalation policies;
- stopping rules;
- batching or scheduling policy.

Candidate measurements include:

- verified decision quality;
- coverage;
- selective risk;
- frontier-call rate;
- verification rate;
- latency;
- total declared cost;
- cost per verified correct decision.

Historical replay must not be treated as valid counterfactual evidence when the proposed policy would require observations or model outputs that were not recorded in the original trajectory.

Replay therefore provides evidence only within the support of the preserved history.

A future research question is whether offline replay can reduce the cost of evaluating policy changes while preserving enough fidelity to predict online system behavior.