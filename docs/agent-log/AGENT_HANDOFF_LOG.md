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
