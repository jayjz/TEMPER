# EXP-0001 B2 Design Freeze

Status: FROZEN — planning only; not executed.

This document freezes scientific choices for the EXP-0001 B2 neural baseline so a later implementation run cannot introduce new design decisions.

It is not a result. It contains no measured B2 metrics.

# Objective

Freeze the smallest scientifically defensible bidirectional encoder baseline that provides a clean neural comparison against B1 (TF-IDF + logistic regression) on the same frozen CLINC150 EXP-0001 experiment.

The goal is to minimize confounding and maximize interpretability, not to maximize expected benchmark score.

# Scientific role of B2

B2 is the conventional neural specialist baseline for EXP-0001.

It answers: how far does a standard bidirectional encoder with a linear classification head and cross-entropy training get on the frozen in-scope CLINC150 task, using raw softmax probabilities, before calibration, abstention, OOD methods, or escalation are introduced.

B2 is not:

- a ModernBERT architecture study;
- a distillation study;
- a generative classifier;
- a calibrated probability model;
- an OOD detector;
- a selective-automation policy;
- a CPVCD or thesis test.

Headline comparison is against B0 and B1 on the same frozen validation partition, same class mapping, same primary metric (macro F1), with every neural seed preserved.

# Selected encoder

Selected model identifier:

`google-bert/bert-base-uncased`

This is BERT-base uncased: a 110M-parameter bidirectional Transformer encoder pretrained with masked language modeling and next-sentence prediction (Devlin et al., 2019). Sequence-classification use is the documented intended fine-tune setting.

Why this model, not a higher-scoring or newer encoder:

- It is a genuine bidirectional encoder, not a decoder-only LLM.
- It is the conventional public encoder for sentence classification, which keeps the B1-versus-B2 contrast as "bag-of-words linear model versus standard encoder," not "TF-IDF versus a new architecture."
- Protocol language allows "ModernBERT or another explicitly documented bidirectional encoder." BERT-base is the more conservative reading for a first neural baseline.
- DistilBERT / MiniLM would add distillation as a confound.
- ModernBERT would add architecture, positional-encoding, and dependency novelty as confounds.
- Decoder-only, prompted, API, teacher-label, and RL classifiers are out of scope.

License: Apache 2.0. Not gated.

Hardware class: BERT-base (110M) with max length 128 and batch size 16 is expected to fit an RTX 4060-class 8 GB GPU under fp16. If it does not fit the actual device, implementation must stop rather than silently change this freeze.

Rejected alternatives (not to be substituted at implementation time):

- `answerdotai/ModernBERT-base` — protocol candidate, deferred to a later named comparison if needed;
- `distilbert-base-uncased` — distillation confound;
- any Qwen/Llama/decoder-only or API classifier.

# Immutable model revision

Pin weights, config, and tokenizer files from one Hugging Face revision. Do not follow `main`.

- Hub id: `google-bert/bert-base-uncased`
- Immutable revision SHA: `86b5e0934494bd15c9632b12f734a8a67f723594`
- Observed on 2026-09-17 from `https://huggingface.co/api/models/google-bert/bert-base-uncased`
- `config.json` at this revision: `model_type=bert`, 12 layers, hidden size 768, 12 heads, vocab 30522, `max_position_embeddings=512`
- Weight files present at this revision include `model.safetensors` and `pytorch_model.bin`
- Prefer loading `model.safetensors` from this revision

Load with `revision="86b5e0934494bd15c9632b12f734a8a67f723594"` and do not fall back to another commit.

# Tokenizer + revision

Tokenizer is the tokenizer shipped in the same repository at the same revision.

- Tokenizer identifier: `google-bert/bert-base-uncased`
- Tokenizer revision SHA: `86b5e0934494bd15c9632b12f734a8a67f723594`
- Files: `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`
- `tokenizer_config.json` at this revision: `do_lower_case=true`, `model_max_length=512`

Tokenization for B2:

- `truncation=true`
- `padding="max_length"`
- frozen `max_length=128` (below the tokenizer maximum of 512; CLINC150 utterances are short)
- special tokens handled by this tokenizer
- do not add a custom vocabulary

# Frozen training configuration

These values are frozen. Implementation must not sweep them.

1. Hugging Face model identifier: `google-bert/bert-base-uncased`
2. Immutable model revision: `86b5e0934494bd15c9632b12f734a8a67f723594`
3. Tokenizer identifier: `google-bert/bert-base-uncased`
4. Tokenizer revision: `86b5e0934494bd15c9632b12f734a8a67f723594`
5. Classification head: Hugging Face `AutoModelForSequenceClassification` / `BertForSequenceClassification` with `num_labels=150`. Default `[CLS]` linear head. No extra MLP, CRF, or prompt head.
6. Training objective / loss: standard multiclass cross-entropy over the 150 in-scope classes. No class weights, label smoothing, Brier loss, or temperature in the training loop.
7. Maximum sequence length: 128
8. Batch size: 16 (per-step train batch; no gradient accumulation)
9. Learning rate: `2e-5`
10. Weight decay: `0.01`
11. Epochs: exactly 3
12. Optimizer: AdamW (`torch.optim.AdamW`), `betas=(0.9, 0.999)`, `eps=1e-8`
13. Scheduler: linear decay to zero
14. Warmup policy: `warmup_ratio=0.1` of total training steps; no other warmup
15. Gradient clipping: global `max_grad_norm=1.0`
16. Mixed-precision policy: CUDA `float16` autocast when CUDA is used; `float32` on CPU. Do not use bfloat16 in this freeze.
17. Device policy: use CUDA when available; otherwise CPU. Record the actual device. A CPU run is allowed only as a recorded non-headline execution. Do not silently change batch size or sequence length to chase memory.
18. Deterministic settings: seed Python, NumPy, and PyTorch (CPU and CUDA); `torch.backends.cudnn.deterministic=True`; `torch.backends.cudnn.benchmark=False`; attempt `torch.use_deterministic_algorithms(True)` and record whether it succeeded. Do not claim bit-identical GPU runs.
19. Seed set: `{13, 21, 37}` (see below)
20. Checkpoint-selection rule: no validation-based checkpoint picking. Train all 3 epochs and use the final-epoch weights.
21. Early-stopping rule: none. Early stopping is explicitly absent.

Additional frozen training details:

- Train only on official in-scope `train` (15,000 examples).
- Shuffle the train set each epoch with a generator seeded from the run seed.
- Evaluation during/after training uses the frozen seed-42 validation indices only.
- Per-epoch validation metrics may be logged as evidence; they must not change the checkpoint.
- Class-id mapping: sorted unique in-scope labels from official train, identical in spirit to B1 (`sorted({record[1] for record in payload["train"]})`). Persist the ordered mapping in the artifact.
- Dropout: leave BERT config defaults unchanged (`hidden_dropout_prob=0.1`, `attention_probs_dropout_prob=0.1`).
- Training loop: explicit PyTorch loop. Do not require Hugging Face `Trainer` / `accelerate` for this freeze.
- Do not mix OOS examples into training, validation-for-selection, or loss.
- Do not use the calibration partition for any B2 fit, logging-based selection, or early stopping.
- Do not use the final-test partition.

# Predeclared seed set

Frozen seed set, declared before any B2 result is observed:

```text
13, 21, 37
```

Justification:

- EXP-0001 protocol requires multiple predeclared seeds for neural headline comparisons.
- Evaluation protocol: predeclare seeds, report mean and dispersion, preserve every run, do not headline the best seed; target at least 3–5 seeds if compute permits.
- Three seeds is the smallest set that can support a mean and a dispersion without becoming an architecture sweep.
- Seeds are distinct from the frozen split seed `42` so model stochasticity is not aliased with split construction.

Headline reporting later must use all three runs (mean and dispersion). Implementation must not drop a seed after seeing its score.

# Data boundaries

Use the existing frozen split file. Do not modify it.

- Split file: `experiments/EXP-0001/splits/clinc150-full-seed-42.json`
- Split seed: 42
- Training: official CLINC150 `train`, 15,000 examples
- Development validation: frozen seed-42 `validation_indices`, 1,500 examples
- Calibration: 1,500 examples — **do not use**
- Final test: 4,500 examples — **do not use**

Dataset hashes already observed for EXP-0001 (not re-derived here):

- `archive_sha256`: `0d8ecc3e1edd7b25cabde0177544ce536ddf773844bc80ef1a75f36e7f030ea2`
- `canonical_sha256`: `fb3217519e3c601c7a9b019dfd6744bed8f2564833e2b7eac4a406cacb462489`

Implementation must verify the canonical hash before fitting, matching B0/B1.

# Metrics

Compute the same in-scope validation metrics as B0/B1:

- primary: macro F1
- secondary: accuracy, negative log likelihood, multiclass Brier
- systems: training time, validation inference time, throughput, model size
- also preserve raw maximum-softmax confidence in the probability matrix

Do not report these as calibrated probabilities. Do not optimize thresholds. Do not claim OOD performance. Descriptive OOS confidence, if ever recorded after this freeze, must not affect B2 selection.

# Hardware/runtime evidence contract

Implementation must record, per B2 run, at least:

- CPU identity
- GPU identity, or explicit `none`
- RAM
- CUDA version / GPU driver / runtime identity where applicable
- framework versions: Python, `torch`, `transformers`, `tokenizers`, `numpy`
- device (`cuda` / `cpu` and index)
- dtype (`float16` or `float32`)
- batch size (frozen 16)
- max sequence length (frozen 128)
- seed (one of 13, 21, 37)
- model identifier
- immutable model revision
- tokenizer revision
- training time
- validation inference time
- throughput
- model size (parameter count and on-disk checkpoint size if a checkpoint is written)
- peak GPU memory if `torch.cuda.max_memory_allocated` is practical

Do not invent measured values in this document. If `HardwareRef` cannot currently hold every field, store extras in manifest `runtime` / `software_environment` rather than silently dropping them.

# Artifact requirements

Before a B2 result may be treated as reconstructable, preserve at least:

- experiment manifest (JSON, fail-closed overwrite protection)
- raw labels
- raw predictions
- raw probability matrix
- ordered class-label mapping
- metrics
- model/config provenance (id, revision, tokenizer revision, hyperparameters, seed, git commit, dataset hashes)
- hashes of the prediction artifact and of the manifest

Reuse the existing prediction-artifact schema (`labels`, `predictions`, `probabilities`, `class_labels`, `metrics`) unless a later authorized change says otherwise.

Size note (planning estimate, not a measurement):

- validation probability matrix is about 1,500 × 150 values; compressed `.npz` plus manifest per seed is likely a few megabytes;
- three seeds remain small;
- fine-tuned BERT-base weights are hundreds of megabytes per seed.

Recommendation, not an established policy:

- prediction artifacts and manifests are likely small enough for direct Git inclusion once an artifact-durability policy exists;
- fine-tuned weights/checkpoints should stay outside Git;
- do not upload or commit generated B2 outputs in this planning turn.

The existing B0/B1 durability gap remains unresolved. B2 must not inherit "README numbers only" as sufficient evidence.

# Dependency implications

Current `pyproject.toml` has no `torch` or `transformers`. This freeze does **not** add them.

A later implementation run will need to admit at least:

- `torch` (CUDA build on the operator GPU machine)
- `transformers`

Transitive packages will likely include `huggingface_hub`, `safetensors`, and `tokenizers`.

Avoid admitting `accelerate`, `datasets`, or Trainer extras unless a later authorized change requires them. The frozen training loop is plain PyTorch.

This is a heavyweight-ML admission relative to the current scikit-learn stack. It is an implementation-turn dependency change, not this turn.

# Reproducibility controls

- Pin Hub revision SHA, not `main`.
- Verify dataset `canonical_sha256` before fit.
- Load the frozen split and refuse any seed other than 42 for split metadata.
- Refuse test and calibration partitions in the B2 training/validation path.
- Write artifacts atomically and refuse overwrite.
- Record git commit, seeds, hardware, dtype, and package versions in the manifest.
- Preserve all three seeds; do not delete worse runs.
- Do not tune on observed B2 validation scores after this freeze. If the frozen config fails to run, stop and report; do not invent a new config.

# Stop conditions

Implementation must stop and report instead of improvising if:

- revision `86b5e0934494bd15c9632b12f734a8a67f723594` cannot be fetched or hash-verified;
- the loaded model is not BERT / not bidirectional;
- CUDA memory requires a silent batch-size or sequence-length change;
- a dependency change larger than torch+transformers becomes necessary for a hidden reason;
- the frozen split would need to change;
- calibration or final-test access appears required to "make B2 work";
- a seed fails and there is a temptation to replace it after seeing scores.

# Known limitations

- BERT-base is English uncased pretrained data, not CLINC150 in-domain pretraining.
- Three seeds estimate variation; they are not a full uncertainty analysis.
- GPU training is not guaranteed bit-identical across hardware.
- Raw softmax is not a calibration result.
- RTX 4060-class fit is an engineering expectation from model size and sequence length, not a measured profile from this machine.
- Fine-tuned weights are expected to live outside Git unless a later policy says otherwise.
- This freeze does not complete EXP-0001 by itself.

# EXP-0001 completion-contract observation

Inspected `experiments/EXP-0001/protocol.md` completion criteria:

1. dataset acquisition is reproducible;
2. splits are frozen;
3. majority baseline is evaluated;
4. TF-IDF + logistic regression is evaluated;
5. raw predictions and probability outputs are preserved;
6. metrics are recorded;
7. no final-test tuning occurred;
8. limitations and errors are documented.

B2 is **not** named in those eight criteria.

Separately, the B2 section says the neural baseline "is not required for the first executable pass if heavyweight ML dependencies have not yet been admitted." Those dependencies are not in `pyproject.toml` today.

Roadmap P1 still lists a ModernBERT-class encoder baseline as a P1 build item.

Finding: under the current EXP-0001 protocol, B2 is **optional** for declaring EXP-0001 complete. It is **not** forbidden. There is a residual ambiguity between EXP-0001 completion (B0+B1 sufficient) and P1 phase exit (encoder baseline listed).

Smallest human-authorized amendment, if desired later, would add an explicit completion bullet that B2 validation has been executed under this freeze. Do not make that amendment in this turn.

# Exact next implementation handoff

Next bounded action: B2 implementation planning/execution against this freeze, on `feat/exp-0001-b2-encoder` or a follow-on implementation branch.

That later run may:

- admit `torch` and `transformers`;
- implement the frozen training/eval path;
- run seeds 13, 21, and 37 on validation only;
- write manifests and prediction artifacts.

That later run may not:

- change model, revision, tokenizer, hyperparameters, seeds, splits, or metrics;
- use calibration or test data;
- sweep architectures;
- headline the best seed;
- claim calibrated probabilities or thesis support.
