---
phase: 01-harness-integrity-measured-baseline
fixed_at: 2026-09-29T19:16:06Z
review_path: .planning/phases/01-harness-integrity-measured-baseline/01-REVIEW.md
iteration: 2
findings_in_scope: 1
fixed: 1
skipped: 0
status: all_fixed
---

# Phase 01: Code Review Fix Report

**Fixed at:** 2026-09-29T19:16:06Z
**Source review:** .planning/phases/01-harness-integrity-measured-baseline/01-REVIEW.md (iteration 2)
**Iteration:** 2

**Summary (this iteration):**
- Findings in scope: 1 (0 Critical, 1 Warning — WR-07; fix_scope = critical_warning, so IN-01..IN-09
  were not attempted)
- Fixed: 1
- Skipped: 0
- Status: all_fixed

## Fixed Issues

### WR-07: `ci_checks.sh` installs uv but never adds it to PATH — auto-setup aborts on fresh hosts

**Files modified:** `scripts/ci_checks.sh`
**Commit:** 10476ff
**Applied fix:** Exactly the review's one-line remedy, adapted only in comment placement. Inside the
`if ! command -v uv` bootstrap branch, immediately after `curl -LsSf https://astral.sh/uv/install.sh | sh`,
the script now runs `export PATH="$HOME/.local/bin:$PATH"` (with a three-line comment explaining that
the installer is a child process that cannot mutate the parent shell's PATH, so without the export
the subsequent bare `uv venv` / `uv pip install` calls abort with "command not found" under
`set -euo pipefail` on fresh hosts). The export is scoped to the install branch only — when uv is
already resolvable nothing changes. The CI jobs are unaffected (they were already healthy: uv's
installer exports the path via `$GITHUB_PATH` under Actions).
**Verification:** Tier 1 re-read of the edited block (lines 59-67) confirmed the fix is present and
surrounding code intact; Tier 2 `bash -n scripts/ci_checks.sh` passed ("SYNTAX OK"); additionally a
behavioral subshell check on this host — with a PATH stripped of `~/.local/bin`, `command -v uv`
fails (confirming the install branch would be taken on a fresh host), and after applying the same
`export PATH="$HOME/.local/bin:$PATH"`, `command -v uv` succeeds ("MECHANISM VERIFIED"). This is a
functional-environment fix, not a logic change, so no extra human-verification flag is needed.

## Skipped Issues

None this iteration — the single in-scope finding (WR-07) was fixed.

## Out of Scope (not attempted, per fix_scope and owner decisions)

- **IN-01 .. IN-09 (Info tier):** out of scope for `fix_scope = critical_warning`.
- **Deferred to owner / Phase-4 CI-gate phase** (re-affirmed by the iteration-2 review, which did not
  re-count them): WR-02 (`test-mamba`/`test-cuda` structural no-op on GPU-less runners), WR-05
  (broad `.github/workflows/README.md` staleness), WR-06 (unpinned `curl | sh` uv installer —
  distinct from WR-07, which fixed only the functional PATH breakage, not the supply-chain pin).

## Iteration-1 Ledger (preserved)

Fixed in iteration 1 and **verified fixed by the iteration-2 review** (checked against the working
tree, not this report):

- **WR-01 — least-privilege workflow permissions** — `.github/workflows/ci.yml` — commit `b8926d5`.
  Top-level `permissions` narrowed to `contents: read`; only `deploy` overrides with
  `contents: write` (needed for `mkdocs gh-deploy --force`). Adaptation: kept a top-level read block
  so future jobs inherit read-only. Verified via PyYAML parse + per-job assertions.
- **WR-03 — mamba failure-artifact upload reachable** — `.github/workflows/ci.yml` — commit `9b916e3`.
  Test step got `id: mamba-tests` + `continue-on-error: true` + `2>&1 | tee pytest.log` +
  `set -o pipefail`; upload step keys on `if: always() && steps.mamba-tests.outcome == 'failure'`.
  Verified structurally; needs a GPU runner to observe live.
- **WR-04 — stale `pytest.ini` docs removed** — `tests/TESTING.md`, `CONTRIBUTING.md` — commit `839b3ef`.
  Replaced the deleted-ini documentation with a pointer to `[tool.pytest.ini_options]` in
  `pyproject.toml` plus a "never recreate" warning (HARN-01 regression guard).

Skipped in iteration 1 (deliberate scope discipline; still open as owner/Phase-4 decisions):

- **WR-02 — `test-mamba` structural no-op that `deploy` treats as passing** — deferred: both fix
  options (GPU self-hosted runner vs deleting the job + `needs` entry) are infrastructure decisions
  belonging to the Phase-4 CI-gate restructuring.
- **WR-05 — `.github/workflows/README.md` materially stale** — deferred: broad doc rewrite coupled to
  the Phase-4 workflow changes; the targeted `README.md:185` correction already landed.
- **WR-06 — unpinned `curl | sh` uv installer in four CI jobs + local script** — deferred: pin
  strategy (pinned URL vs `astral-sh/setup-uv`) is an owner toolchain decision with CI-break risk.

---

**Verification note (where verification ran):** This iteration worked directly in the main checkout
at `/home/forrest/Github/DNALLM` (sequential dispatch — no isolated worktree, per orchestrator
instruction), so all verification results above are reproducible from the main working tree at commit
`10476ff`. Verification was: `bash -n` syntax check of `scripts/ci_checks.sh`, plus a subshell
simulation of the fresh-host PATH mechanism. No test suite or live CI run was executed (out of scope
per fix-loop strategy; the verifier phase covers it). Markdown edits have no syntax checker (Tier 3
fallback, re-read only). Commit `10476ff` was pushed to `origin/dev`.

_Fixed: 2026-09-29T19:16:06Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 2_
