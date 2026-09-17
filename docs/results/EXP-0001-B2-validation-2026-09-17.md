# EXP-0001 B2 Validation Result

Status: VALIDATION COMPLETE

Date: 2026-09-17

## Scope

This record documents the frozen EXP-0001 B2 bidirectional-encoder validation runs.

It does not report final-test performance, calibrated probabilities, OOD performance, selective automation, frontier escalation, CPVCD, or support for the overall TEMPER systems thesis.

## Research role

B2 is the conventional neural specialist baseline for EXP-0001.

The experiment asks how far a standard bidirectional encoder with a linear classification head and cross-entropy training can go on the frozen in-scope CLINC150 task before calibration, abstention, OOD methods, or escalation are introduced.

## Frozen configuration

- model: `google-bert/bert-base-uncased`
- model revision: `86b5e0934494bd15c9632b12f734a8a67f723594`
- tokenizer revision: same immutable revision
- architecture: `BertForSequenceClassification`
- labels: 150
- objective: multiclass cross-entropy
- max sequence length: 128
- train batch size: 16
- learning rate: `2e-5`
- weight decay: `0.01`
- epochs: 3
- optimizer: AdamW
- scheduler: linear decay
- warmup ratio: `0.1`
- gradient clipping: `1.0`
- CUDA precision: float16
- checkpoint selection: final epoch only
- early stopping: none
- seeds: `13, 21, 37`

No configuration was changed after observing B2 results.

## Data boundary

Dataset: CLINC150 full.

Frozen dataset identity:

- archive SHA-256: `0d8ecc3e1edd7b25cabde0177544ce536ddf773844bc80ef1a75f36e7f030ea2`
- canonical SHA-256: `fb3217519e3c601c7a9b019dfd6744bed8f2564833e2b7eac4a406cacb462489`

Partitions:

- train: 15,000
- validation: 1,500
- calibration: 1,500, unused
- final test: 4,500, untouched

The calibration and final-test partitions were not used by B2.

## Producer provenance

All three B2 seed manifests record producer commit:

`aa0f85cf6031fa5e6a93cb0c6138bfb91feb2772`

All runs used:

- Windows 11
- NVIDIA GeForce RTX 4060
- CUDA execution
- PyTorch `2.14.0+cu130`
- Transformers `5.17.0`
- Tokenizers `0.23.2`
- Python `3.12.7`

The manifests report `headline_hardware_qualification=UNVERIFIED`.

## Predictive and probabilistic results

| Seed | Accuracy | Macro F1 | NLL | Multiclass Brier |
| ---: | ---: | ---: | ---: | ---: |
| 13 | 0.9593333333 | 0.9589545169 | 0.5849050980 | 0.1907247021 |
| 21 | 0.9473333333 | 0.9469664575 | 0.6439254014 | 0.2177154842 |
| 37 | 0.9560000000 | 0.9556948392 | 0.6187840524 | 0.2022728046 |
| **Mean** | **0.9542222222** | **0.9538719379** | **0.6158715173** | **0.2035709970** |
| **Sample std** | **0.0061943822** | **0.0061984369** | **0.0296177514** | **0.0135421401** |

Primary outcome:

- B2 mean validation macro F1: `0.9538719379`
- sample standard deviation: `0.0061984369`

For context, the previously observed B1 TF-IDF + logistic-regression validation macro F1 was `0.8863992084`.

This establishes B2 as a materially stronger validation baseline than B1 under the frozen EXP-0001 configuration.

It does not establish statistical or production superiority outside this benchmark.

## Systems observations

Per-run manifest measurements:

| Seed | Fit seconds | Inference seconds | Throughput examples/s | Peak GPU memory bytes |
| ---: | ---: | ---: | ---: | ---: |
| 13 | 309.5339334 | 2.2050834 | 680.2463798 | 2453924864 |
| 21 | 306.1442903 | 2.1960569 | 683.0424112 | 2453924864 |
| 37 | 315.7849729 | 2.3327883 | 643.0073402 | 2453924864 |

These systems measurements are not treated as a clean headline hardware comparison.

Operator conditions were not perfectly controlled across every run. In particular, background GPU state and later power-management precautions changed during manual execution. Seed 21 was followed by an unexpected system reboot after its evidence artifacts had been written.

Predictive results remain preserved, but timing, throughput, power, and thermal comparisons should be repeated under a formally controlled hardware protocol before making systems-efficiency claims.

## Evidence integrity

For every seed, the preserved evidence includes:

- experiment manifest
- manifest SHA-256 sidecar
- raw labels
- raw predictions
- full probability matrix
- ordered class-label mapping
- stored metrics
- prediction-artifact SHA-256
- model and tokenizer revisions
- producer Git commit
- dataset identity
- frozen hyperparameters
- runtime metadata

Uploaded artifact and manifest hashes were independently checked against the supplied sidecars before this result record was written.

Official evidence root:

`C:\Users\jcoul\Desktop\Projects\TEMPER-EVIDENCE\EXP-0001`

The evidence root intentionally lives outside the TEMPER source repository.

## Interpretation

EXP-0001 now establishes three useful reference points on the same bounded validation task:

- B0: trivial majority lower bound
- B1: inexpensive classical TF-IDF + logistic-regression baseline
- B2: conventional neural specialist baseline

B2 produced a mean validation macro F1 of approximately `0.9539` across the three predeclared seeds, with sample standard deviation approximately `0.0062`.

The result supports continuing to calibration and selective-prediction experiments because the specialist is sufficiently accurate for uncertainty quality, abstention, and risk-coverage behavior to become meaningful research questions.

The result does not show that:

- raw B2 probabilities are calibrated;
- maximum-softmax confidence is a safe automation signal;
- OOD detection works;
- abstention preserves reliability;
- specialist-plus-escalation is cheaper than direct general-model inference;
- Qwen or another general-purpose model can be replaced;
- TEMPER's central systems thesis is supported.

Those require later controlled experiments.

## Known limitations

- CLINC150 is a crowdsourced benchmark rather than production traffic.
- The task is English and single-intent.
- Temporal drift is absent.
- Three seeds provide only a coarse estimate of run-to-run variation.
- Raw softmax probabilities are not calibrated.
- Final-test performance has not been measured.
- OOS behavior has not yet been evaluated in this result.
- Hardware qualification remains `UNVERIFIED`.
- RAM and GPU-driver fields were not populated in the manifests.
- Systems timing conditions were not sufficiently controlled for a headline cost/latency claim.
- The archive SHA-256 is freeze-declared; the original ZIP was not re-hashed during each run.

## Next research step

Do not tune B2 from these validation results.

The next phase should isolate probability calibration and selective prediction:

1. fit post-hoc calibration using only the reserved calibration partition;
2. compare raw versus calibrated probability quality;
3. evaluate risk-coverage and abstention behavior;
4. predeclare operating thresholds before final-test evaluation;
5. later compare specialist-only and specialist-plus-escalation systems against a general-purpose model under a declared CPVCD accounting boundary.

EXP-0001 should remain a conventional-baseline experiment rather than absorbing these later questions.