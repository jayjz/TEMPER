# External Baselines and Research Anchors

Status: P0 literature record

## Purpose

This document separates external evidence from claims TEMPER has independently reproduced.

No result listed here should be described as a TEMPER result unless a corresponding experiment exists in the experiment registry.

## Calibration

### Guo et al. — On Calibration of Modern Neural Networks

Source:
https://proceedings.mlr.press/v70/guo17a.html

External finding:

Modern neural networks may be poorly calibrated even when predictive performance is strong. The paper evaluates post-hoc calibration methods and reports temperature scaling as a strong simple baseline across its experiments.

TEMPER use:

- motivates held-out probability-calibration evaluation;
- motivates temperature scaling as a baseline;
- does not establish that temperature scaling guarantees calibration.

TEMPER reproduction:

Not yet.

## Selective Prediction

### Geifman and El-Yaniv — SelectiveNet

Source:
https://proceedings.mlr.press/v97/geifman19a.html

External finding:

Selective prediction studies systems that may reject examples rather than force predictions over the entire input distribution. Risk-coverage behavior is a central evaluation concept.

TEMPER use:

- motivates coverage and selective-risk evaluation;
- motivates explicit abstention experiments;
- does not imply calibrated probabilities are automatically the best rejection score.

TEMPER reproduction:

Not yet.

## Encoder Baseline

### Warner et al. — ModernBERT

Source:
https://arxiv.org/abs/2412.13663

External finding:

ModernBERT is a modern bidirectional encoder designed for efficient classification, retrieval, and related workloads.

TEMPER use:

A ModernBERT-class model is a candidate neural specialist baseline for P1.

Exact model and revision must be frozen before evaluation.

TEMPER reproduction:

Not yet.

## Cascade Economics

### Chen, Zaharia, and Zou — FrugalGPT

Source:
https://arxiv.org/abs/2305.05176

External finding:

Model cascades can trade model selection, accuracy, and inference cost rather than sending every query to one expensive model.

TEMPER use:

- motivates cascade economics as a serious systems baseline;
- does not establish that TEMPER's specialist-verifier-escalation architecture will be economical.

TEMPER reproduction:

Not yet.

## TypeSafe Jev and RLCD

Primary public source:
https://typesafe.ai/blog/introducing-system-one-models-and-jev

External claims from TypeSafe include:

- Jev is described as a System One model for fast structured decisions;
- TypeSafe describes a new model architecture;
- TypeSafe describes a parallel sampler;
- TypeSafe names its training method Reinforcement Learning for Calibrated Decisions (RLCD);
- TypeSafe reports substantial efficiency improvements.

TEMPER position:

These are external company claims.

Public implementation detail is currently insufficient for TEMPER to infer the exact Jev architecture or reproduce RLCD faithfully.

TEMPER therefore does not assume:

- RLCD is equivalent to Brier-loss optimization;
- Jev is a conventional encoder classifier;
- structured output alone establishes correctness or calibration.

TEMPER reproduction:

Not yet.

## Parallel Constrained Decoding

Candidate external artifact:
https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD

TEMPER position:

Public decoder-side implementations that reuse shared state and score bounded candidate choices are valuable inference controls.

They should be treated separately from claims about specialized calibration-aware training.

TEMPER reproduction:

Not yet.

## Evidence Rule

For each future external comparison, record separately:

1. what the external authors report;
2. what implementation TEMPER actually reproduced;
3. differences in model, hardware, data, and protocol;
4. whether the result replicated;
5. limitations preventing direct comparison.