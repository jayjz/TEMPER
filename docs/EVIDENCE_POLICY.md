# Evidence Policy

## Purpose

TEMPER must make it difficult to accidentally convert an interesting observation into an unsupported research claim.

## Claim States

Every material claim should be classifiable as one of:

- PROPOSED — hypothesis or design idea.
- OBSERVED — directly measured in a recorded experiment.
- REPLICATED — reproduced under declared repeated conditions.
- ROBUST — survived predefined shift or stress testing.
- EXTERNAL — reported by an external source.
- UNVERIFIED — plausible but not established.

## Evidence Hierarchy

Prefer evidence in this order:

1. observed real-world outcome;
2. deterministic or formal verifier;
3. independently curated expert label;
4. established benchmark label with documented provenance;
5. multi-source weak supervision;
6. single-model pseudo-label;
7. model self-evaluation.

Lower levels can be useful for training.

They must not silently become higher-quality evaluation evidence.

## External Research

For every external result distinguish between:

- what the authors report;
- what TEMPER has independently reproduced.

An external result remains EXTERNAL until reproduction occurs.

## TypeSafe and Jev

TypeSafe publicly describes Jev as a System One model using a new architecture, a parallel sampler, and a training method called Reinforcement Learning for Calibrated Decisions (RLCD).

TEMPER does not currently have enough public implementation detail to reproduce RLCD faithfully.

TEMPER therefore must not treat statements such as the following as established facts:

- RLCD is Brier loss;
- Jev is ModernBERT with task heads;
- Jev is an encoder-only Transformer.

## Parallel-Constrained Decoding Baselines

Public implementations that reuse shared state, score bounded candidates, avoid full structured autoregressive serialization, or assemble output programmatically are treated as inference baselines.

They are not automatically evidence of specialized training or RLCD reproduction.

Reported latency improvements remain external claims until independently reproduced.

## Calibration Claims

These concepts are distinct:

1. valid output schema;
2. correct prediction;
3. normalized softmax score;
4. calibrated probability;
5. safe action.

None automatically implies the next.

Softmax normalization is not proof of empirical calibration.

Calibration must be evaluated against observed outcomes on held-out data.

## Experiment Provenance

Every substantive experiment must record, where applicable:

- experiment ID;
- git commit;
- dataset name, version, and hash;
- label source;
- split definition;
- model identifier;
- model revision;
- tokenizer revision;
- preprocessing configuration;
- objective or loss;
- hyperparameters;
- seed;
- hardware;
- software environment;
- calibration method;
- threshold-selection procedure;
- raw predictions;
- metrics;
- limitations;
- conclusion or outcome state.

## Failed Experiments

Failures are evidence.

Record:

- hypothesis;
- procedure;
- observed failure;
- known or suspected cause;
- whether the experiment is valid;
- whether reproduction is warranted.

Do not delete failed runs because a later experiment succeeds.

## Invalid Experiments

Use INVALIDATED when an experiment cannot support interpretation because of:

- train/test leakage;
- calibration/test leakage;
- corrupted data;
- incorrect labels;
- implementation error;
- unrecorded configuration;
- irreproducible environment.

INVALIDATED and NEGATIVE are not equivalent.

## Performance Claims

Do not publish claims such as:

- 5x faster;
- 99 percent reliable;
- 10x cheaper;
- cannot hallucinate;
- guaranteed safe;
- calibrated;

without reproducible evidence and a declared scope.

## Security-Sensitive Evidence

Never store in Git:

- live credentials;
- API keys;
- private customer data;
- sensitive packet captures;
- exploit loot;
- raw secret-scanner findings containing actual secrets;
- proprietary model weights without authorization.

Store hashes, metadata, sanitized fixtures, and reproducibility instructions instead.

## North-Star Evidence Chain

A mature TEMPER systems claim should ideally resolve through:

claim -> experiment report -> metrics -> raw predictions -> manifest -> model revision -> dataset version -> label provenance -> verification procedure

If that chain is broken, the claim must state the limitation.