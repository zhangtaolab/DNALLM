---
phase: 27-transformers-4-5-model-compatibility-census
fixed_at: 2026-10-28T09:55:08Z
review_path: .planning/phases/27-transformers-4-5-model-compatibility-census/27-REVIEW.md
iteration: 1
findings_in_scope: 2
fixed: 2
skipped: 0
status: all_fixed
---

# Phase 27: Code Review Fix Report

**Fixed at:** 2026-10-28T09:55:08Z
**Source review:** `.planning/phases/27-transformers-4-5-model-compatibility-census/27-REVIEW.md` (incremental #3661, reviewed 2026-10-28T02:19:44Z)
**Iteration:** 1 (this cycle; overwrites the prior cycle's report for the earlier REVIEW iteration)
**Fix scope:** critical_warning (CR-01, WR-01; IN-01, IN-02, IN-03 are out of scope this pass)

**Summary:**
- Findings in scope: 2
- Fixed: 2
- Skipped: 0

Both fix commits are on `v1.4` and pushed to `origin/v1.4`.

## Fixed Issues

### CR-01: Census lane's seeded `cp` targets a directory deleted by checkout's default clean — the lane is permanently red on every live dispatch

**Files modified:** `.github/workflows/census_compat.yml`
**Commit:** `53d0017`
**Applied fix:** Added `mkdir -p census-artifacts` to the top of the census step's run block,
after `source .venv/bin/activate` and before the seeded `cp` — exactly the reviewer's
suggested block, mirroring the proven feasibility.yml:96 post-checkout-mkdir pattern. The
pre-checkout `mkdir` in the GPU-check step is kept: it serves the GPU-absent marker path,
where checkout is skipped so nothing removes it. A comment block above the run line records
the rationale (persistent shared workspace → repo always present at dispatch →
actions/checkout's default `git clean -ffdx` removes the untracked directory), so a future
editor does not "de-duplicate" the two mkdirs and re-create the permanently-red lane.
No other line of the workflow changed — the dispatch-only posture (triggers, permissions,
runner, guard, timeout) is untouched byte-for-byte.

### WR-01: The no-masking contract pin misses `continue-on-error` — a step-level flag that produces exactly the "silent red-to-green lie" the pin forbids

**Files modified:** `tests/test_runner_infra_contracts.py`
**Commit:** `44b0cf6`
**Applied fix:** Added `test_no_continue_on_error_anywhere` to `TestCensusLaneInstallSequence`
(immediately after the sibling `test_no_masked_failure_in_the_lane`), pinning that neither
the `census-compat` job nor any of its steps carries a `continue-on-error` key in the parsed
workflow YAML — the reviewer's snippet, with one adaptation: the job-level assert's message
is wrapped in the file's parenthesized style because the suggested one-line form is 103
characters (over the ruff format limit of 100). `ruff format --check` confirms the file is
format-clean. The class docstring's "or masking the kernel build" claim is now backed by an
actual pin against this masking vector, not just `|| true`.

## Verification

**Where gates ran:** main checkout (`workflow.use_worktrees=false` — no isolated worktree;
both commits landed directly on `v1.4`). Numbers are reproducible from the current tree.

Per-commit (before each commit):
- CR-01: `yaml.safe_load` parse of `.github/workflows/census_compat.yml` — ok; census step
  run block verified to contain the post-checkout `mkdir` ahead of the `cp`;
  `pytest tests/test_runner_infra_contracts.py -q` — **12 passed** (workflow edit broke no pin)
- WR-01: `python -c "import ast; ast.parse(...)"` — ok; `ruff format --check` — "1 file
  already formatted"; `ruff check` — "All checks passed!";
  `pytest tests/test_runner_infra_contracts.py -q` — **13 passed** (12 + the new pin)
- Negative check on the WR-01 detection expression: injecting a synthetic
  `continue-on-error` step (and job key) into the parsed workflow trips both assertions
  (`flagged == ['masked-build']`, job key detected) — the pin fails red when the masking
  vector is actually present.

Post-fix state: working tree clean apart from the pre-existing untracked
`.planning/ui-reviews/`; both commits carry no attribution trailers per owner rule.

**Not verified here:** a live dispatch of the `census_compat` workflow (needs the
self-hosted GPU runner and network; dispatch is deferred until the v1.4→main merge per
27-05-SUMMARY). The CR-01 fix is deterministic per actions/checkout v4.2.2's
`prepareExistingDirectory` behavior and mirrors the feasibility.yml pattern already proven
on the same box; the owner should dispatch the lane once after merge to confirm end-to-end
green, as with the prior cycle.

## Skipped Issues

None — both in-scope findings were fixed. IN-01, IN-02, IN-03 are out of scope for this
pass (critical_warning scope) and remain open for a future cycle if the owner chooses.

---

_Fixed: 2026-10-28T09:55:08Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
