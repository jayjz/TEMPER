# Agent Handoff Log

This file is an operational handoff log for agents working in TEMPER.

It is not a research source of truth.

Agents must append new entries. Never rewrite, delete, or silently edit old entries.

Source-of-truth evidence remains Git, experiment protocols, experiment manifests, and raw artifacts.

---

## 2026-09-17T00:08:37Z

- agent = Grok
- branch = `feat/exp-0001-b2-encoder`
- base commit = `d4cf036244607ab4f88a7c880628f1a046bf8f5f`
- objective = Prepare TEMPER for the B2 encoder phase by recording already-observed EXP-0001 B0/B1 validation evidence in the public README and establishing this append-only handoff log. Do not implement B2 in this turn.

### Observed evidence

Recorded from the operator-supplied EXP-0001 observation for this handoff. B0 and B1 were not rerun. Gitignored result artifacts are not present in this workspace.

Dataset:

- `archive_sha256` = `0d8ecc3e1edd7b25cabde0177544ce536ddf773844bc80ef1a75f36e7f030ea2`
- `canonical_sha256` = `fb3217519e3c601c7a9b019dfd6744bed8f2564833e2b7eac4a406cacb462489`
- frozen split = `experiments/EXP-0001/splits/clinc150-full-seed-42.json`
- seed = 42
- train = 15000
- validation = 1500
- calibration = 1500
- test = 4500

The committed frozen-split file was independently inspected. Top-level fields match the sizes and seed above.

B0 validation:

- accuracy = 0.006666666666666667
- macro_f1 = 8.830022075055188e-05
- nll = 27.446814308489024
- multiclass_brier = 1.9866666666666666

B1 validation:

- accuracy = 0.8873333333333333
- macro_f1 = 0.8863992084286425
- nll = 1.07091089590222
- multiclass_brier = 0.36105027523628147

B1 runtime:

- fit_seconds = 2.442327100000057
- inference_seconds = 0.009108400000059191
- n_examples = 1500

Both B0/B1 manifests report producer revision `d4cf036244607ab4f88a7c880628f1a046bf8f5f`.

Also observed:

- no calibration was fitted
- no threshold was selected
- no B2 evaluation has occurred
- no final-test metrics have been used
- no TEMPER systems-thesis claim is supported

### Files changed

- `README.md`
- `docs/agent-log/AGENT_HANDOFF_LOG.md`

### Checks performed

- `git status --short --branch`
- `git diff --check`
- inspected the complete branch diff against `d4cf036244607ab4f88a7c880628f1a046bf8f5f`
- verified frozen split metadata in `experiments/EXP-0001/splits/clinc150-full-seed-42.json`
- no repository Markdown formatter or Markdown CI check is configured; none was introduced

### Limitations

- B0/B1 metrics were transcribed from the operator-supplied observation, not recomputed.
- `experiments/EXP-0001/protocol.md` and `docs/EXPERIMENT_REGISTRY.md` still say `PLANNED`. Those files are research source-of-truth documents and were left untouched. Public README status is `RUNNING` based on the observed B0/B1 validation execution.
- This workspace clone does not contain gitignored `experiments/EXP-0001/data/` or `experiments/EXP-0001/output/` artifacts, so manifests and raw predictions were not reopened here.
- B2 is not implemented.
- README validation numbers are not test results.

### Next handoff

B2 implementation planning.

---

## 2026-09-17T00:18:33Z

- agent/model = Grok
- branch = `feat/exp-0001-b2-encoder`
- base commit = `d4cf036244607ab4f88a7c880628f1a046bf8f5f`
- head at start of this turn = `afe3ce8f1b4f4fd65e0d57c4668e3e233ae9fc27`
- objective = Reconcile stale EXP-0001 status metadata (`PLANNED` -> `RUNNING`) and record evidence-durability plus B2 pre-flight constraints. Cleanup/documentation/provenance only. Do not implement B2.

This cleanup run produced no new model evidence.

### Status reconciliation performed

Human-authorized lifecycle/status transition only:

- `docs/EXPERIMENT_REGISTRY.md`: EXP-0001 status `PLANNED` -> `RUNNING`. Question unchanged. Result column remains `-`. Other experiment statuses unchanged.
- `experiments/EXP-0001/protocol.md`: `Status: PLANNED` -> `Status: RUNNING`. No other protocol content modified.
- `README.md`: no edit. It already stated P0 COMPLETE, P1 RUNNING, EXP-0001 RUNNING.

### Files changed

- `docs/EXPERIMENT_REGISTRY.md`
- `experiments/EXP-0001/protocol.md`
- `docs/agent-log/AGENT_HANDOFF_LOG.md`

### Evidence durability observations

Inspected `.gitignore`, `docs/EVIDENCE_POLICY.md`, `AGENTS.md`, `README.md`, this log, `git ls-files experiments`, and `git check-ignore` on candidate artifact paths.

Verified in this workspace and on GitHub-tracked files:

- Committed under `experiments/EXP-0001/`: `prepare.py`, `protocol.md`, `run_baselines.py`, and `splits/clinc150-full-seed-42.json` only.
- No B0/B1 manifests are committed. `git ls-files` has no `experiments/**/output/**` and no experiment manifest JSON besides the frozen split.
- No raw prediction artifacts (`.npz` or `predictions.parquet`) are committed.
- Local tree has no `experiments/EXP-0001/data/` and no `experiments/EXP-0001/output/`.
- `.gitignore` ignores `experiments/**/data/`, `experiments/**/artifacts/`, `experiments/**/checkpoints/`, `experiments/**/logs/`, and `experiments/**/predictions.parquet`.
- `.gitignore` does **not** ignore `experiments/**/output/`, `output/results/`, or `output/manifests/`. `run_baselines.py` writes manifests to `--output/manifests/` and prediction `.npz` files to `--output/results/`. Those paths are therefore committable if present, but they are currently absent from Git and from this clone.
- README B0/B1 validation metrics therefore cannot be independently reconstructed from GitHub alone. GitHub currently holds transcribed numbers, frozen-split metadata, and acquisition hashes, not the run manifests or raw prediction arrays.
- What remains local/operator-observed: B0/B1 validation metrics, B1 runtime, and the claim that both manifests report producer revision `d4cf036244607ab4f88a7c880628f1a046bf8f5f`. Those artifacts were not opened in this workspace.

Could not verify from this clone: whether manifests or `.npz` files exist on the operator machine at `C:\Users\jcoul\Desktop\Projects\TEMPER`.

A durable experiment-artifact policy must be decided before B2 evidence becomes important. This turn does not invent that policy, select storage infrastructure, upload artifacts, or commit generated model outputs.

### B2 pre-flight constraints

For the next agent, before any B2 evaluation:

- exact encoder identifier must be frozen before evaluation
- immutable model revision must be pinned
- neural seed set must be predeclared before viewing B2 results
- protocol currently requires multiple neural seeds
- hardware identity must be recorded for B2
- at minimum record CPU/GPU/RAM/device/dtype/batch size/max sequence length
- no final-test access
- no calibration access
- no model-selection decisions based on final test
- no claim that raw softmax probabilities are calibrated

This turn does not select the B2 model, choose the seed set, or change dependencies.

### Checks run

- `git status --short --branch`
- inspected the complete branch diff
- `git diff --check`
- confirmed only intended files changed
- confirmed source-of-truth substantive content is unchanged except the two authorized `PLANNED` -> `RUNNING` transitions
- confirmed no experiment output files were created or modified

### Limitations

- No B0/B1/B2 execution.
- No final-test or calibration access.
- No new metrics.
- Manifest and prediction durability is reported from Git tracking and this workspace only.
- Previous handoff entry still records that registry/protocol said `PLANNED`; that historical entry was not rewritten.

### Next handoff

B2 model/config/seed freeze and implementation planning.

Not B2 execution yet.

---

## 2026-09-17T00:29:01Z

- agent/model = Grok
- branch = `feat/exp-0001-b2-encoder`
- base commit = `d4cf036244607ab4f88a7c880628f1a046bf8f5f`
- head at start of this turn = `a1146124fa0941c46316730f4185a04c74b036e8`
- objective = Freeze the EXP-0001 B2 neural-baseline design so a later implementation run cannot make new scientific choices. Planning/config-freeze only.

This planning run produced no new model evidence.

### Freeze summary

- encoder = `google-bert/bert-base-uncased`
- immutable revision = `86b5e0934494bd15c9632b12f734a8a67f723594`
- tokenizer = same id and revision
- objective = multiclass cross-entropy, 150 in-scope classes, `[CLS]` linear head
- max_length = 128; batch size = 16; lr = 2e-5; weight decay = 0.01; epochs = 3
- optimizer = AdamW; scheduler = linear; warmup_ratio = 0.1; max_grad_norm = 1.0
- no early stopping; use final-epoch weights
- predeclared seeds = 13, 21, 37
- data = official train 15000 + frozen seed-42 validation 1500; calibration and test unused

### Status reconciliation performed

None. Registry/protocol already RUNNING. README not edited.

### Files changed

- `docs/exec-plans/active/EXP-0001-B2-FREEZE.md` (created)
- `docs/agent-log/AGENT_HANDOFF_LOG.md` (appended)

### Evidence durability observations

Unchanged from the previous entry: B0/B1 manifests and raw predictions are still not in Git. This freeze specifies the minimum B2 artifact set and recommends (not as policy) that manifests/prediction `.npz` are small enough for later Git inclusion while fine-tuned weights stay outside Git. No artifacts were uploaded or committed.

### B2 pre-flight constraints

Recorded in the freeze document, including encoder pin, seeds, hardware metadata contract, no calibration, no final-test, no best-seed headline, and no claim that raw softmax is calibrated.

### EXP-0001 completion-contract observation

B2 is optional for EXP-0001 completion under the current protocol (completion criteria name B0 and B1, not B2; neural baseline is not required for the first executable pass if heavyweight ML dependencies have not been admitted). Residual ambiguity with P1 roadmap encoder-baseline language. No protocol amendment made.

### Checks run

- `git status --short --branch`
- `git diff --check`
- inspected the complete diff
- confirmed no source code, dependency, experiment-output, or source-of-truth research-contract changes

### Limitations

- No B0/B1/B2 execution.
- No calibration or final-test access.
- RTX 4060 fit is an engineering expectation, not a measured profile from this machine.
- Hugging Face revision SHA was fetched from the Hub API on 2026-09-17; implementation must pin that SHA rather than `main`.
- Could not inspect operator-local GPU identity.

### Next handoff

B2 implementation against `docs/exec-plans/active/EXP-0001-B2-FREEZE.md`.

Not a new scientific choice. Not B2 execution in this turn.

---

## 2026-09-17T00:54:39Z

- agent/model = Grok
- branch = `feat/exp-0001-b2-encoder`
- starting commit = `af379c53d1e8c78a278e9302f48cd996026ae5a9`
- objective = Implement EXP-0001 B2 exactly as frozen. Make B2 runnable and evidence-producing without new scientific choices. Execute validation-only training only if hardware satisfies the freeze without modification.

This turn is an implementation record. It is not a B2 research result.

### Status labels

- IMPLEMENTED: frozen B2 training/eval path, CLI, unit tests, torch+transformers admission.
- OBSERVED: this sandbox has no CUDA, 3.838 GB RAM, 2 CPUs; `torch.cuda.is_available() is False`.
- UNVERIFIED: B2 validation metrics; Hub fetch of the pinned BERT revision on the operator GPU machine; whether batch 16 / max_length 128 / fp16 fits the operator 4060-class GPU; operator-local B0/B1 artifacts.

### Implementation summary

Added an explicit PyTorch loop for `google-bert/bert-base-uncased` revision `86b5e0934494bd15c9632b12f734a8a67f723594`. No Hugging Face Trainer, accelerate, datasets, or Lightning.

Frozen values encoded as constants: max_length 128, batch 16, lr 2e-5, weight decay 0.01, epochs 3, AdamW, linear warmup 0.1, grad clip 1.0, seeds `{13, 21, 37}`, validation partition only. CUDA uses fp16 autocast; CPU uses fp32. OOM fails closed and does not change batch/sequence/precision.

`run_b2.py` verifies the canonical dataset hash, validates the frozen seed-42 split, trains on official train (15000), evaluates frozen validation (1500), and writes seed-specific npz+json. There is no `--partition` flag. Calibration and official test examples are not bound into B2 tensors. Split-contract validation still checks index disjointness, including calibration/test index identity, without using those examples for fit or metrics.

Overwrite of an existing seed artifact or manifest is refused before dataset load. Fine-tuned weights are not written.

### Dependencies admitted

Direct (`pyproject.toml`):

- `torch>=2.4` (lock: `torch==2.14.0`)
- `transformers>=4.45` (lock: `transformers==5.17.0`)

Transitive, not requested: `huggingface-hub==1.31.0`, `tokenizers==0.23.2`, `safetensors==0.8.0`, plus Linux-marker NVIDIA CUDA wheel libs from the torch CUDA build. `accelerate`, `datasets`, and Lightning remain absent.

### Files changed

- `src/temper/baselines/encoder.py` (created)
- `src/temper/baselines/__init__.py`
- `src/temper/datasets/splits.py` (`validate_exp0001_splits` extracted)
- `src/temper/datasets/__init__.py`
- `experiments/EXP-0001/run_b2.py` (created)
- `experiments/EXP-0001/run_baselines.py` (thin wrapper over shared split validator)
- `tests/unit/test_b2_encoder.py` (created)
- `pyproject.toml`
- `uv.lock`
- `README.md` (factual: B2 implemented; validation execution pending freeze-satisfying hardware; `run_b2.py` command)
- `docs/agent-log/AGENT_HANDOFF_LOG.md` (appended)

Unchanged source-of-truth:

- `docs/THESIS.md`, `docs/ROADMAP.md`, `docs/EVALUATION_PROTOCOL.md`, `docs/EVIDENCE_POLICY.md`, `docs/RESEARCH_QUESTIONS.md`, `docs/EXPERIMENT_REGISTRY.md`, `docs/adr/*`
- `experiments/EXP-0001/protocol.md`
- `docs/exec-plans/active/EXP-0001-B2-FREEZE.md`
- `experiments/EXP-0001/splits/clinc150-full-seed-42.json`

### Tests added

`tests/unit/test_b2_encoder.py` (15 tests). BERT is not downloaded. Coverage: pinned revision, allowed seeds, rejected calibration/test partitions, canonical class order, probability-column identity, seed-specific artifact names, overwrite refusal, frozen config serialization, hardware snapshot serialization, aggregate mean/std without best-seed selection, tiny fake-encoder loop, CLI has no test-partition option, shared split validator.

### Quality gates

All passed:

- `uv run pytest` — 54 passed
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy src`
- `uv run bandit -r src` — no issues
- `uv run pip-audit` — no known vulnerabilities
- `git diff --check`

### Hardware observed

- platform: Linux-6.12.8+-x86_64-with-glibc2.36
- CPU: Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz (2 cores)
- GPU: none (`nvidia-smi` unavailable)
- RAM: 3.838 GB
- `torch==2.14.0+cu130`, `torch.cuda.is_available() is False`
- Python 3.12.14

This machine does not satisfy the freeze's 4060-class 8 GB CUDA assumption. Frozen batch/seq/precision were not modified.

### Whether execution occurred

No. B2 validation training/evaluation was not attempted. Seeds 13, 21, and 37 have no observed metrics.

### Artifact paths

None generated. No `experiments/EXP-0001/data/` or `experiments/EXP-0001/output/` in this workspace. Intended paths after a later authorized run:

- `output/results/EXP-0001-B2-validation-seed-{13,21,37}.npz`
- `output/manifests/EXP-0001-B2-validation-seed-{13,21,37}.json`
- `output/manifests/EXP-0001-B2-validation-aggregate.json`

`.gitignore` still does not ignore `experiments/**/output/`. Fine-tuned weights stay outside Git (`*.pt`, `*.pth`, `*.safetensors`, `checkpoints/`). Artifact-durability policy remains unresolved; this turn does not commit generated evidence.

### Limitations

- Implementation is not a measured B2 result.
- CPU execution, if later forced, is not a headline GPU comparison under the freeze.
- GPU runs are not claimed bit-identical.
- Raw softmax is not calibrated.
- B0/B1 GitHub durability gap is unchanged.
- Linux torch lock pulls CUDA NVIDIA transitives; that does not mean CUDA is available here.

### Next handoff

Run frozen B2 validation on hardware that satisfies the freeze without changing batch size, max length, precision, model, seeds, or splits. Preserve all three seeds. Do not access calibration or final test. Then independent Codex review of this implementation. Do not merge.

---

## 2026-09-17T01:31:39Z

- agent/model = Grok
- branch = `feat/exp-0001-b2-encoder`
- starting commit = `1ff090491af3b81d194fcccc2727bb8675eea22b`
- objective = Close EXP-0001 B2 provenance and evidence-integrity defects before official GPU runs. No B2 training. No scientific-freeze change.

This turn is an implementation-hardening record. It is not a B2 research result.

### Status labels

- IMPLEMENTED: frozen dataset identity as code-controlled; B2 aggregate provenance verification; atomic prediction/manifest writes; sidecar digest contract; explicit `headline_hardware_qualification=UNVERIFIED`; dirty-git fail-closed for B2 execution; re-seed after model load.
- OBSERVED: quality gates passed (69 tests); this sandbox still has no CUDA and was not used for B2 training.
- UNVERIFIED: B2 validation metrics; Hub fetch of pinned BERT; whether the operator GPU fits the frozen batch/seq/fp16 config; operator-local B0/B1 artifacts.

### Defects confirmed

- P0-1 CONFIRMED: B2 (and B0/B1) treated `--canonical-sha256` as authoritative, so a modified dataset plus hash(modified) could pass.
- P0-2 CONFIRMED: B2 `--aggregate` trusted seed `.npz` files and stored metrics without manifests, revision, dataset identity, or artifact hashes.
- P1 durability CONFIRMED: `write_prediction_artifact` claimed atomicity but wrote in place; manifests had no SHA-256; freeze requires hashes of both artifact and manifest.
- P1 hardware CONFIRMED: runtime did not distinguish headline qualification from "CUDA happened to be available."

### Adjacent issues inspected

- Test/calibration leakage in B2 command: not a defect. B2 still has no `--partition` and does not bind `payload["test"]` or `calibration_indices`.
- Model init consuming RNG: addressed by seeding before and after `from_pretrained`.
- Scheduler first-step LR=0 from PyTorch 2.x LambdaLR init: HuggingFace-compatible; not changed.
- Dirty git HEAD: B2 execution now fails closed. Aggregate of already-written seeds does not require a clean tree.
- B0/B1 still expose `--partition test` (pre-existing; out of B2 scope). B0/B1 still do not sidecar-digest or refuse dirty git.

### Files changed

- `src/temper/datasets/exp0001.py` (created): frozen archive/canonical SHA-256 and verifier
- `src/temper/evidence/integrity.py` (created): hashing, atomic writes, sidecar digest, git cleanliness
- `src/temper/baselines/b2_evidence.py` (created): fail-closed seed-bundle verification and aggregate
- `src/temper/datasets/__init__.py`, `src/temper/evidence/__init__.py`
- `src/temper/evaluation/baseline.py`: prediction artifacts written via temp+replace
- `src/temper/baselines/encoder.py`: hardware qualification fields; no training-config change
- `experiments/EXP-0001/run_b2.py`: remove caller hash flags; verify frozen canonical; sidecar; clean git
- `experiments/EXP-0001/run_baselines.py`: caller hashes must match frozen identity; file bytes compared to frozen canonical
- `tests/unit/test_b2_evidence.py` (created)
- `tests/unit/test_b2_encoder.py`, `tests/unit/test_run_baselines_provenance.py`
- `README.md` (CLI: no caller dataset hash)
- `docs/agent-log/AGENT_HANDOFF_LOG.md` (appended)

Unchanged source-of-truth: freeze, protocol, registry, thesis, roadmap, evaluation protocol, evidence policy, research questions, ADRs, frozen split JSON.

### Evidence contract after this turn

- Canonical dataset identity is `fb3217519e3c601c7a9b019dfd6744bed8f2564833e2b7eac4a406cacb462489`.
- Archive identity `0d8ecc3e1edd7b25cabde0177544ce536ddf773844bc80ef1a75f36e7f030ea2` is freeze-declared; B2 does not re-hash a zip.
- Prediction `.npz` SHA-256 lives in `runtime.artifact_sha256` (covers the artifact, not the manifest).
- Manifest SHA-256 lives in sibling `<manifest>.sha256.json` (`temper.evidence.sidecar.v1`). The sidecar is not self-hashed.
- Aggregate admits only seeds `{13,21,37}`, matching frozen BERT revision, frozen canonical hash, matching sidecar, and recomputed metrics. No best-seed headline.
- `headline_hardware_qualification` is always `UNVERIFIED` in this freeze. CUDA is recorded factually and is not treated as RTX 4060-equivalent.

### Quality gates

- `uv run pytest` — 69 passed
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy src`
- `uv run bandit -r src` — no issues
- `uv run pip-audit` — no known vulnerabilities
- `git diff --check`

No B2 training. No calibration or final-test access.

### Next handoff

Official B2 seeds 13, 21, 37 on freeze-satisfying GPU hardware, from a clean git tree, using the real canonical `data_full.json`. Do not change the freeze. Preserve every seed. Then independent Codex review. Do not merge.
