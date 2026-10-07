---
phase: 05-execution-harness-honest-gates-runner-feasibility
fixed_at: 2026-10-01T20:10:00Z
review_path: .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW.md
iteration: 1
findings_in_scope: 7
fixed: 7
skipped: 0
status: all_fixed
---

# Phase 5: Code Review Fix Report

**Fixed at:** 2026-10-01T20:10:00Z
**Source review:** .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 7 (CR-01 + WR-01..06; IN-01..04 out of scope for `fix_scope: critical_warning`)
- Fixed: 7
- Skipped: 0

## Fixed Issues

### CR-01: Unused `noqa` in spike runner fails `ruff check .` — CI lint gate is red on every push

**Files modified:** `scripts/feasibility/spike_families.py`
**Commit:** 89b2f63
**Applied fix:** Deleted the inert `# noqa: ANN202` directive from the `_fromstring`
shim definition. Acceptance proof re-run live: `.venv/bin/python -m ruff check .
--statistics` exits 0 with no output (previously exit 1 with exactly one RUF100) —
the repo-wide lint gate is green again.

### WR-01: `assert_tree_clean` passes silently when the `git status` call itself fails

**Files modified:** `tests/examples/_execution.py`
**Commit:** f8e0f9f
**Applied fix:** Added `assert result.returncode == 0, f"git status failed
(rc=...): stderr"` ahead of the stdout interpretation, with a comment explaining
the silent-tripwire-disable failure mode. Behavior verified live: REPO_ROOT pointed
at a non-git dir now raises (previously passed silently); clean repo tree still
passes; injected untracked file under `example/` still fails.

### WR-02: `check_docs_sync.py` claims byte-identity but uses shallow (stat-signature) comparison

**Files modified:** `scripts/check_docs_sync.py`
**Commit:** 9876a0b
**Applied fix:** Replaced the `dcmp.diff_files` loop in `check_sync` with
`filecmp.cmpfiles(dcmp.left, dcmp.right, dcmp.common_files, shallow=False)` —
always reads bytes. Chose the `cmpfiles` route over `dircmp(..., shallow=False)`
because the constructor kwarg is Python 3.13+ only and the project supports 3.10+.
Strictness is a superset of the old check (shallow `cmp()` already reads bytes when
the stat signature differs, so `diff_files` never contained byte-identical files);
un readable files now surface as `UNREADABLE:` errors instead of passing. Injected-
drift semantics verified live: same-size+same-mtime content divergence is now
CAUGHT (the old blind spot), size/mtime-visible drift still caught, identical trees
pass, and the current repo tree still exits 0 (`OK: docs/example/ is in sync`).

### WR-03: New test module imports `nbclient` at module scope, but `nbclient` lives only in the `notebook` extra

**Files modified:** `pyproject.toml`
**Commit:** 5c30083
**Applied fix:** Added `"nbclient>=0.10"` to the `test` extra (mirrors the existing
entry in `notebook`; installed 0.11.0 satisfies it). The `importorskip` alternative
was rejected per review guidance — CI is meant to run these tests. README needed no
change: its install line (`uv pip install -e '.[test,cpu]'`) never instructed
installing anything extra for the notebook-execution tests, and that documented
path now works as written.

### WR-04: `|| true` masks the spike runner's exit code, hiding infrastructure crashes as a green step

**Files modified:** `scripts/feasibility/spike_families.py`, `.github/workflows/feasibility.yml`
**Commit:** 9de82cd
**Applied fix:** `main()` now separates evidence completeness from verdict: exit 0
when every requested family emitted a complete evidence block (OK or FAIL — FAIL is
matrix evidence), exit 1 when any family crashed before emitting (each crash gets a
`family=/result=CRASH/failure_text=` block and the loop continues so remaining
families still produce evidence), exit 2 for the GPU guard (unchanged). Stale
`_gpu_guard` docstring exit-code cross-reference updated. Workflow: dropped
`|| true` (kept `set -o pipefail` + `tee`), kept upload `if: always()`, and
rewrote the step comment to document the new contract. Semantics verified live via
monkeypatched `_run_family`: all-FAIL run exits 0 with `failed_families=` listed;
mid-run crash exits 1 with `crashed_families=` and the remaining families still
evidenced; all-OK exits 0. **Note for human verification:** the runner executes
only on the self-hosted GPU box; the workflow-side behavior (red step on crash,
artifact still uploaded) should be confirmed on the next dispatch.

### WR-05: GPU-absent path reports a green job with zero evidence produced

**Files modified:** `.github/workflows/feasibility.yml`
**Commit:** 898d325
**Applied fix:** Applied the reviewer's primary fix (stronger than the marker-only
minimum): on GPU absence the gpu-check step now `mkdir -p`s the spike-logs dir,
writes `gpu_absent=<UTC timestamp> nvidia-smi unavailable` into
`spike_runner_all.log` (picked up by the `if: always()` upload — the artifact
carries the reason), and `exit 1` so an evidence-only deliverable can no longer
green with nothing. Step-failure semantics verified against GitHub Actions rules:
subsequent steps' explicit `if:` conditions get an implicit `success() &&` so they
are skipped, while the upload's `if: always()` still runs. Marker-writing script
simulated locally and produces the expected file/content. This is a logic change
in CI control flow — flagged for human verification on the next dispatch.

### WR-06: `feasibility.yml` omits the least-privilege `permissions:` block the repo convention mandates

**Files modified:** `.github/workflows/feasibility.yml`
**Commit:** 824b901
**Applied fix:** Added a file-level `permissions: contents: read` block directly
after `on: workflow_dispatch`, matching the ci.yml convention, with a comment
noting this workflow executes repo code on the self-hosted GPU box. YAML
re-parsed and the permissions mapping asserted programmatically.

## Verification

All verification ran in the **main checkout** at `/home/forrest/Github/DNALLM`
(`workflow.use_worktrees` is `false` in `.planning/config.json`, so no isolated
worktree was created; edits, commits, and gates all ran in the main working tree —
results are reproducible directly from this tree).

- `.venv/bin/python -m ruff check . --statistics` → exit 0, no findings (CR-01
  acceptance proof, run after all fixes)
- `.venv/bin/python -m ruff format --check` on all three touched Python files
  (`scripts/feasibility/spike_families.py`, `scripts/check_docs_sync.py`,
  `tests/examples/_execution.py`) → all already formatted
- `.venv/bin/python scripts/check_docs_sync.py` → exit 0, mirror in sync
- Fast leg: `.venv/bin/python -m pytest tests/examples/ tests/configuration/
  -m "not slow"` → **192 passed, 1 skipped, 2 deselected in 7.91s**
  (consistent with the review's pre-fix baseline of 94 passed / 1 skipped /
  2 deselected for tests/examples plus the configuration suite)
- Per-fix behavioral checks documented in each entry above (non-git-dir guard
  raise, byte-level drift injection, exit-semantics scenarios, YAML parses)

## Commits

| Finding | Commit | Files |
|---------|--------|-------|
| CR-01 | 89b2f63 | `scripts/feasibility/spike_families.py` |
| WR-01 | f8e0f9f | `tests/examples/_execution.py` |
| WR-02 | 9876a0b | `scripts/check_docs_sync.py` |
| WR-03 | 5c30083 | `pyproject.toml` |
| WR-04 | 9de82cd | `scripts/feasibility/spike_families.py`, `.github/workflows/feasibility.yml` |
| WR-05 | 898d325 | `.github/workflows/feasibility.yml` |
| WR-06 | 824b901 | `.github/workflows/feasibility.yml` |

Milestone constraint honored: commits are local only (MANUAL-PUSH-ONLY); nothing
was pushed.

---

_Fixed: 2026-10-01T20:10:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
