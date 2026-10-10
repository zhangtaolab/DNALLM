---
phase: 261006-uq7
plan: 01
subsystem: tooling
tags: [deps, ruff, dependabot, lint, mamba-abi]
requires:
  - "dependabot PR #41 (dev-group ruff 0.16.9 -> 0.16.10) published and CI-green"
provides:
  - "ruff 0.16.10 pinned in pyproject.toml dev extra and installed in .venv, repo-wide gates proven green at that version"
affects: []
tech-stack:
  added: []
  patterns:
    - "baseline-first version bump: capture old-version gate outputs before touching the venv, then prove the same gates at the new version"
key-files:
  created:
    - .planning/quick/261006-uq7-apply-dependabot-ruff-bump-on-phs-with-m/261006-uq7-SUMMARY.md
  modified:
    - pyproject.toml
decisions:
  - "PR #42 (torch ceiling <2.12 -> <2.15) deliberately NOT applied — held-open-as-record per prior owner decision (ABI claim for the exact-pinned native kernels); recorded durably in the commit body"
metrics:
  duration: 6 min
  completed: 2026-10-06
status: complete
actuals:
  tokens: 1800
  tasks: 2
  commits: 1
  plan_head_before: 859187ac21088e2312bbdf2a92fed5d39d5d4ab0
  plan_head_after: 7cf8458552f8803a14f72d715be463a12b538374
---

# Quick Task 261006-uq7: Apply dependabot ruff bump on phs Summary

One-line: bumped the dev-extra ruff pin 0.16.9 -> 0.16.10 (dependabot PR #41 equivalent,
direct branch edit) with baseline-first gates proven green at both versions, mamba/torch red
line byte-untouched, one pushed commit; PR #42 stays held-open-as-record.

## What Was Done

- **Task 1** — Captured the 0.16.9 baseline while the venv still held 0.16.9 (both
  repo-wide gates clean), made the scoped one-line edit to `pyproject.toml` line 83
  (`"ruff==0.16.9",` -> `"ruff==0.16.10",`), installed 0.16.10 into `.venv` with the
  owner-specified targeted uv command (resolver touched only ruff), re-ran both gates at
  0.16.10 (green, zero findings, zero formatter drift — no mechanical reformat set), and
  ran the spot-check pytest pair (20 passed).
- **Task 2** — Pre-staging audit a)-d) all pass (details below), committed `pyproject.toml`
  alone by explicit pathspec, pushed `phs` to origin on the first attempt, and verified
  `origin/phs == HEAD`.

## Gate Outputs: Baseline (0.16.9) vs New Version (0.16.10)

Baseline captured BEFORE the bump, while `.venv` still held 0.16.9:

```
$ .venv/bin/ruff --version
ruff 0.16.9
$ .venv/bin/ruff format --check .
285 files already formatted          # exit 0
$ .venv/bin/ruff check . --statistics
                                     # no output, exit 0 — zero findings
```

After the bump and targeted install:

```
$ .venv/bin/ruff --version
ruff 0.16.10
$ uv pip install --python .venv/bin/python ruff==0.16.10
 - ruff==0.16.9
 + ruff==0.16.10                     # resolver touched only ruff
$ .venv/bin/ruff format --check .
285 files already formatted          # exit 0 — NO formatter drift at 0.16.10
$ .venv/bin/ruff check . --statistics
                                     # no output, exit 0 — zero findings
```

STOP-and-report outcome: **not triggered** — 0.16.10 introduced zero new lint findings and
zero formatter drift, so no noqa, no suppression, no `[tool.ruff]` edits, and the
conditional mechanical-reformat set is **empty** (nothing to enumerate).

## Spot Check

```
$ .venv/bin/python -m pytest tests/test_runner_infra_contracts.py tests/test_models_lock_contracts.py -q
20 passed in 0.43s
```

(Re-run after the commit as part of the Task 1 verify chain: 20 passed in 0.56s.)

## Task 2 Audit Evidence (pre-staging)

- PRE (pre-task commit): `859187ac21088e2312bbdf2a92fed5d39d5d4ab0`
- **a)** `git diff -- pyproject.toml` contained exactly 2 content lines
  (`grep -cE '^[+-][^+-]'` == 2): removed `    "ruff==0.16.9",`, added
  `    "ruff==0.16.10",` — nothing else in the file moved.
- **b)** Red-line pins byte-present post-state: `"torch>=2.4.0,<2.12"` (pyproject.toml
  lines 64/145/151), `"causal_conv1d==1.7.0"` (line 166), `"mamba-ssm==2.3.2.post1"`
  (line 167), `"mambapy>=1.2.0"` (line 41). The PR #42 `<2.15` widening is absent.
- **c)** `git diff --quiet -- .github/` → quiet (dependabot.yml ignore rules and all
  workflows byte-unchanged).
- **d)** `git status --porcelain` showed ONLY the pre-existing dirty set
  (`.planning/config.json`, deleted 09-phase `.continue-here.md`, untracked
  `.planning/graphs/`, `.planning/state.json`, `.planning/tmp/`,
  `.planning/quick/261002-inq-*`, plus this task's own untracked planning dir) and
  `pyproject.toml` — no format-run rewrites (none existed), nothing else swept in.

## Commit and Push

- Commit: `7cf8458552f8803a14f72d715be463a12b538374` (short `7cf8458`) on `phs`
  - Subject: `deps(quick-261006-uq7): bump ruff 0.16.9 -> 0.16.10 (PR #41 equivalent, phs direct apply)`
  - Body records: owner directive timestamp, both gates green at 0.16.10 before commit,
    zero formatter drift, the PR #42 held-open-as-record disposition with the full ABI
    rationale, and the no-attribution-trailers rule. Verified: no trailers present
    (`co-authored`/`generated-with`/`claude`/`signed-off` grep → none).
  - Diffstat: `1 file changed, 1 insertion(+), 1 deletion(-)` — pyproject.toml only.
- Push: `git push origin phs` succeeded on attempt 1 (`859187a..7cf8458  phs -> phs`).
- End state: `HEAD == origin/phs == 7cf8458552f8803a14f72d715be463a12b538374`.

Task 2 automated verify gate output: `RED-LINE-INTACT-AND-PUSHED` (single-line pyproject
delta pair + all four pins + `.github/` byte-quiet across the commit + origin == HEAD).

## PR #42 Held-Open-as-Record (restated)

Dependabot PR #42 (torch ceiling `<2.12` -> `<2.15`) was deliberately NOT applied, per the
prior owner decision: the ceiling is an ABI claim — the exact-pinned native kernels
(`mamba-ssm==2.3.2.post1`, `causal-conv1d==1.7.0`, compiled against the installed torch
2.11.0+cu130) break on re-resolution — and `.github/dependabot.yml`'s torch ignore rule
keeps guarding it. No GitHub PR operations of any kind were performed (no merge, close,
or comment on #41 or #42).

## Deviations from Plan

None — plan executed exactly as written. The conditional mechanical-reformat set is empty
(formatter reported zero drift between 0.16.9 and 0.16.10), the uv targeted install
succeeded on the first attempt (pip fallback never needed), and the push succeeded on the
first attempt (retries never needed).

## Self-Check: PASSED

- `.venv/bin/ruff --version` → `ruff 0.16.10` (FOUND)
- pyproject.toml dev extra pins `ruff==0.16.10` (FOUND, line 83)
- Commit `7cf8458` is ancestor/equal of HEAD: `git rev-parse HEAD` == `7cf8458...` (FOUND)
- `origin/phs` == HEAD (FOUND)
- Spot-check tests green (FOUND)
