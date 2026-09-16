# ADR 0001 — Research Scope and North-Star Metric

Status: Accepted

## Context

TEMPER studies decision-oriented model systems. Many useful AI workloads are bounded decisions rather than open-ended natural-language generation.

Several tempting directions would weaken the research:

- reproducing a proprietary model from incomplete public information;
- training another niche chatbot;
- building an agent framework before establishing the core phenomenon;
- optimizing latency before establishing task validity;
- treating model confidence as calibrated probability;
- building a decision compiler before understanding the underlying decision models.

These directions would make causal attribution difficult and risk converting the project into product engineering before its scientific question is established.

## Decision

TEMPER will study learned probabilistic decision primitives embedded inside systems with explicit abstention, verification, and escalation.

Primary systems question:

> Under what conditions can specialized decision inference reduce cost and latency while preserving verified system-level reliability?

## Primary Thesis

For recurring bounded decisions, expensive general-purpose reasoning may be progressively replaced by smaller specialized models when those models are combined with:

- calibrated uncertainty;
- abstention;
- independent verification;
- selective escalation.

This is a hypothesis, not a premise.

## North-Star Metric

Cost Per Verified Correct Decision (CPVCD).

Supporting measurements include:

- task utility;
- calibration;
- selective risk;
- coverage;
- latency;
- throughput;
- memory;
- escalation rate;
- robustness under distribution shift.

The accounting boundary for CPVCD must always be declared.

## Baseline Requirement

Novel methods must be compared against credible simple approaches.

Initial baseline families include:

- majority predictor;
- linear classifier;
- encoder classifier;
- calibrated encoder;
- autoregressive small language model;
- parallel constrained decoder;
- frontier model;
- cascade.

## Why Parallel Constrained Decoding Is Included

Recent public implementations suggest that bounded structured decisions can sometimes avoid serializing an entire structured response token by token.

This creates an attribution question:

> How much decision-system efficiency comes from specialized training, and how much comes from changing inference?

TEMPER must separate these effects before attributing gains to new training objectives.

## Consequences

Benefits:

- falsifiable;
- measurable;
- resistant to hype;
- separates training gains from inference gains;
- compatible with modest hardware;
- supports positive and negative results.

Costs:

- slower path to impressive demos;
- additional experimental bookkeeping;
- additional baselines;
- some promising ideas may fail.

Those costs are accepted.

## Rejected Alternative — Clone Jev

Rejected.

Public evidence does not currently expose enough of Jev's architecture or RLCD methodology for faithful scientific reproduction.

## Rejected Alternative — Runtime First

Rejected.

The runtime should emerge from measured decision workloads rather than assumptions.

## Rejected Alternative — RL First

Rejected.

Reinforcement learning is unnecessary until supervised and post-hoc calibration baselines establish what problem remains unsolved.

## Revisit Conditions

Revisit this ADR when:

- TypeSafe publishes sufficient RLCD technical detail;
- P1-P4 establish meaningful evidence for decision specialization;
- a simpler architecture falsifies the central premise;
- new empirical results materially alter the research question.