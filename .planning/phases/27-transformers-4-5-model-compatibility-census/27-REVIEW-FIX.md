---
phase: 27-transformers-4-5-model-compatibility-census
fixed_at: 2026-10-27T12:05:00Z
review_path: .planning/phases/27-transformers-4-5-model-compatibility-census/27-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 27: Code Review Fix Report

**Fixed at:** 2026-10-27T12:05:00Z
**Source review:** `.planning/phases/27-transformers-4-5-model-compatibility-census/27-REVIEW.md`
**Iteration:** 1
**Fix scope:** critical_warning (CR-01, WR-01, WR-02, WR-03; the six Info findings are out of scope)

**Summary:**
- Findings in scope: 4
- Fixed: 4
- Skipped: 0

**Application order note:** WR-01 was applied before CR-01 (not severity order). CR-01's
chosen fix is the reviewer's option (a) — carry dispositions through `merge_records` and
seed the lane artifact from the committed file — which the REVIEW itself notes depends on
the WR-01 carry ("see WR-01"). Applying CR-01 first would have forced option (b), a
lane-side duplicate of the frozen 27-02 judgments in code.

## Fixed Issues

### CR-01: `--run --verify` can never exit 0 — the dispatch lane is permanently red

**Files modified:** `scripts/census_compat.py`, `.github/workflows/census_compat.yml`, `tests/models/test_census_compat.py`
**Commit:** `16363d4`
**Status:** fixed: requires human verification (exit-code composition logic; verified offline end-to-end below, but the live lane has not been dispatched)
**Applied fix:** Reviewer option (a) + honest exit composition:
- `run_census` no longer returns 1 for unreachable-on-both-hubs entries. They stay loud
  `census-blocked` records (per-record evidence + `CENSUS BLOCKED` report); the verdict
  gate decides their significance — committed-known unreachable matches its committed
  `census-blocked` field, a newly unreachable repo disagrees with its committed verdict
  and fails the diff. Fail-closed posture preserved: never a silent pass, never conflated
  with infrastructure failure. Hard errors (artifact IO) still return 1; usage errors 2.
- `main()` composes `--run --verify` as `return run_rc or verify_rc` (was `run_rc if
  run_rc != 0 else verify_rc`, which discarded `verify_rc` behind the always-1
  unreachable signal). A hard census error and a verdict disagreement now each surface
  independently.
- The workflow seeds `census-artifacts/census-fresh.json` from the committed
  `tests/models/compat_census.json` before the run, so the frozen 27-02 dispositions ride
  `merge_records` onto unchanged verdicts and `compat_verdict` can map every record; the
  stale "an unreachable repo ... exits non-zero" workflow comment is replaced with the
  actual seeded/drift semantics.
- `compat_verdict` itself is unchanged (no lane-side duplicate mapping) and
  `--emit-docs` regeneration is untouched — verified byte-identical after the fixes.

### WR-01: `merge_records` silently drops `disposition` and `previous_status`

**Files modified:** `scripts/census_compat.py`, `tests/models/test_census_compat.py`
**Commit:** `61d128c`
**Applied fix:** On an unchanged verdict, `merge_records` now carries `disposition` and
`previous_status` forward — restructured to run independently of the load carry, because
the review's literal snippet nested the carry inside the `record["load"] is None` guard:
a `--with-loads` re-run re-probes failed loads (fresh non-None load dict), which would
have dropped the skip dispositions again. A verdict change still drops the stale
disposition and stamps the immediate prior status. `TestDispositionCompleteness` stays
green across a default-artifact re-run.

### WR-02: Loading the census script in tests sets `HF_ENDPOINT` process-wide at collection

**Files modified:** `tests/models/test_census_compat.py`
**Commit:** `10f7ad9` (test-only change)
**Applied fix:** `_load_census_module` snapshots and restores the environment around
`exec_module` (pop before, restore/remove in a `finally`), so collection-time loading no
longer reroutes the whole pytest process through `hf-mirror.com` or clobbers a user-/CI-
provided endpoint. The script's CLI behavior (mirror at import) and the generated child
programs (endpoint set in their own source) are unchanged.

### WR-03: Probe subprocesses inherit the full runner environment, including `GITHUB_TOKEN`

**Files modified:** `scripts/census_compat.py`, `tests/models/test_census_compat.py`
**Commit:** `6187404`
**Applied fix:** New `sanitized_probe_env()` (module constant `BLOCKED_PROBE_ENV_PREFIXES
= ("GITHUB_", "RUNNER_", "ACTIONS_")`, matching the reviewer's snippet) is passed as
`env=` to both `run_import_probe` and `run_load_probe` subprocess calls. Blocklist chosen
over the prose allowlist deliberately: it is the reviewer's operative code suggestion and
cannot break interpreter/venv/hub-cache/CUDA plumbing on the self-hosted GPU box in ways
offline tests cannot catch. The children keep everything else, including `HF_HOME`,
`PATH`, `HOME`, and CUDA variables.

## Verification

**Where gates ran:** main checkout (`workflow.use_worktrees=false` — no isolated
worktree; all four commits landed directly on `v1.4`). Numbers are reproducible from the
current tree.

Per-commit (before each commit):
- `pytest tests/models/test_census_compat.py` — 72 / 78 / 80 / 83 passed at the
  respective commits (68 baseline + 15 new tests)
- `ruff check` + `ruff format --check` on both touched files — clean at every commit
- `python -c "import ast; ast.parse(...)"` on touched Python files — ok
- Workflow YAML parse check (`yaml.safe_load`) after the CR-01 workflow edit — ok

Post-fix (after `6187404`):
- `pytest tests/models/ tests/utils/` — **933 passed** (918 baseline + 15 new; no regressions)
- `python scripts/census_compat.py --verify --artifact tests/models/compat_census.json` (dry) — exit 0
- `python scripts/census_compat.py --emit-docs docs/user_guide/model_compatibility.md` —
  "docs page already current", byte-identical (`DOCS_DATA_DATE` untouched; determinism held)
- New CR-01 end-to-end offline regression (`TestSeededFreshRunVerification`): committed
  artifact → fresh-shaped records (dispositions/previous_status stripped, green loads
  parked) → `merge_records` → `verify_registry` against the real roster = 0. This is the
  composition that structurally could not exit 0 before the fixes.

**Not verified here:** a live dispatch of the `census_compat` workflow (needs the
self-hosted GPU runner and network). The offline seeded simulation covers the same code
path; the owner should dispatch the lane once to confirm end-to-end.

## Skipped Issues

None — all four in-scope findings were fixed.

---

_Fixed: 2026-10-27T12:05:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
