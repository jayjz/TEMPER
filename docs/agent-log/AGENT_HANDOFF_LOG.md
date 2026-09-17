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
