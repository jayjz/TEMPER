# Research Thesis

## Primary thesis

> **Expensive general-purpose reasoning can, for recurring bounded decisions, be progressively compiled into smaller probabilistic decision models that deliver lower cost and latency while preserving system-level reliability through calibrated abstention, independent verification, and selective escalation.**

This is a hypothesis, not a premise.

## Stronger formulation

Let:

- `G` be a general-purpose frontier model,
- `S` be a specialized decision model,
- `V` be an independent verifier,
- `E` be an escalation policy,
- `D` be a task distribution.

We investigate whether a composed system:

```text
S + V + E + G
```

can Pareto-dominate direct use of `G` on parts of `D` with respect to:

- verified decision quality,
- inference cost,
- latency,
- throughput,
- energy / compute,
- observability,
- operational controllability.

"Pareto-dominate" means improving at least one objective without degrading the others under a declared operating point.

## Crystallization hypothesis

Repeated expensive reasoning may create data that enables future specialization:

```text
novel case
  -> frontier reasoning
  -> verified outcome
  -> accumulated evidence
  -> specialist training
  -> calibrated deployment
  -> increased specialist coverage
  -> fewer frontier calls
```

We call this process **intelligence crystallization** as a working research term.

A key question is whether this loop actually reduces frontier dependence under realistic distribution drift, or merely overfits historical traffic.

## Null hypotheses

The research must make failure possible.

### H0.1
Specialized decision models do not materially improve cost per verified correct decision after the cost of verification, retraining, monitoring, and escalation is included.

### H0.2
Calibration degrades too severely under distribution shift for confidence-based automation to remain useful.

### H0.3
Dynamic decision spaces remove enough specialization advantage that general-purpose generative models remain operationally superior.

### H0.4
Teacher-distilled specialists inherit teacher errors and cannot safely expand coverage without expensive human or external verification.

### H0.5
Maintenance complexity of a decision fabric exceeds the compute savings it creates.

A result supporting any of these nulls is valuable.

## Research boundaries

This project does **not** initially study:

- AGI
- recursive self-improvement in the unrestricted sense
- open-ended autonomous agents
- foundation-model pretraining from scratch
- general conversational quality
- chain-of-thought faithfulness
- claims about TypeSafe's undisclosed internals

## Research contribution targets

A meaningful contribution could be any one of:

1. a rigorous benchmark for cost per verified correct decision;
2. a reproducible result showing when selective specialist cascades outperform direct frontier inference;
3. a negative result identifying where calibration/abstention fails;
4. a dynamic-label decision architecture with strong efficiency/calibration characteristics;
5. a reproducible intelligence-crystallization protocol;
6. a decision-schema compiler that maps typed software contracts into learned inference;
7. an evidence protocol for auditing learned decision primitives in production.
