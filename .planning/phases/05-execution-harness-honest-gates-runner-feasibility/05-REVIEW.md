---
phase: 05-execution-harness-honest-gates-runner-feasibility
reviewed: 2026-10-01T20:27:41Z
depth: standard
iteration: 3
files_reviewed: 10
files_reviewed_list:
  - .github/workflows/docs-validation.yml
  - .github/workflows/feasibility.yml
  - .github/workflows/README.md
  - pyproject.toml
  - README.md
  - scripts/check_docs_sync.py
  - scripts/feasibility/spike_families.py
  - tests/examples/_execution.py
  - tests/examples/test_notebook_execution.py
  - tests/expected_skips.yaml
findings:
  critical: 0
  warning: 1
  info: 5
  total: 6
status: issues_found
---

# Phase 5: Code Review Report (Iteration 3 — Final Convergence Check)

**Reviewed:** 2026-10-01T20:27:41Z
**Depth:** standard
**Files Reviewed:** 10 (narrowed per convergence context: the source/config files the fix passes touched; the `docs/example/**` mirrors were byte-verified twice already and are out of this pass's scope)
**Status:** issues_found

## Summary

Iteration-3 convergence re-review after two fix passes (`89b2f63..824b901`,
then `a5f38ac` + `8fe451b`). Two goals per the convergence criteria: verify the
WR-07/WR-08 doc fixes hold on current text, and confirm the doc edits
introduced no new inaccuracies.

### Convergence verification — all 9 prior findings hold

Re-verified independently against current source (not the fix reports). Since
`824b901` only two commits landed (`a5f38ac`, `8fe451b`), touching only
`.github/workflows/README.md` and `.github/workflows/feasibility.yml` — every
Python/config file is byte-identical to its verified fix commit, and the
strongest gates were re-run live this pass:

- **CR-01:** live `ruff check .` → "All checks passed!", exit 0 repo-wide. **VERIFIED.**
- **WR-01:** returncode guard present (`tests/examples/_execution.py:196-198`); file unchanged since `f8e0f9f`. **VERIFIED.**
- **WR-02:** `filecmp.cmpfiles(..., shallow=False)` at `scripts/check_docs_sync.py:62-64`; live run exits 0 (`OK: docs/example/ is in sync`). **VERIFIED.**
- **WR-03:** `nbclient>=0.10` in the `test` extra (`pyproject.toml:99`); `pytest tests/examples/test_notebook_execution.py --collect-only` collects 2 items. **VERIFIED.**
- **WR-04:** no `|| true` (`feasibility.yml:91-95`, `set -o pipefail` + `tee` kept); `main()` exits 2 on GPU-guard failure, 1 only when a family crashed pre-evidence, 0 otherwise (`spike_families.py:841-870`). **VERIFIED.**
- **WR-05:** GPU-absent path writes the `gpu_absent` marker into the artifact path then `exit 1` (`feasibility.yml:43-49`); the `if: always()` upload (line 100) still delivers it. **VERIFIED.**
- **WR-06:** file-level `permissions: contents: read` (`feasibility.yml:18-19`). **VERIFIED.**
- **WR-07 (this pass's target):** `.github/workflows/README.md:121` now reads
  "**GPU Check** (runs BEFORE checkout): … writes a `gpu_absent` marker into
  the artifact path and FAILS the job — unlike `test-mamba`'s green fail-safe
  no-op …", with **Code Checkout** documented as following, gated on
  `has_gpu`. Cross-checked against `feasibility.yml:33-54` (check is step 1,
  checkout gated at line 53) and against `ci.yml:268-294` — `test-mamba`
  really is a green no-op (`has_gpu=false`, no `exit 1`), so the contrast
  claim is accurate. **VERIFIED — fix holds.**
- **WR-08 (this pass's target):** step renamed to "Run the committed spike
  runner (D-06 fallback variants)" (`feasibility.yml:73`) and the comment
  (`feasibility.yml:75-78`) now states fallback-variant-only semantics. Code
  truth re-confirmed at `spike_families.py:792-805`: `--fallback` REPLACES —
  evo1 → `evo-1-8k-base`, evo2 → notebook id + `noFA-noFP8` override,
  megadna → notebook id + pinned clone; pybigwig/marimo ignore the flag. The
  "committed local evidence" claim is real: `git ls-files` shows all 8
  `spike-logs/*.log` files tracked (including the notebook-variant runs).
  README §7 step 4 (`.github/workflows/README.md:124`) matches. **VERIFIED —
  fix holds.**

### Doc-edit diff attribution

`git show 8fe451b` / `a5f38ac`: the edits touched ONLY the step name, the
first paragraph of the run-step comment, and README §7 step 1 — all three are
accurate against the code. **The edits introduced no new inaccuracies in the
text they wrote.** One residual inaccuracy survives in the *untouched*
adjacent comment text (WR-09 below), which the corrected sentences now make
internally contradictory.

### Checked and cleared this pass (potential findings that did not survive verification)

- `docs-validation.yml:44` cites "masked-outcome steps removed per WR-08" —
  this is the **milestone-level** WR-08 (v1.1 scoping: the docs-validation
  `continue-on-error` false-green; see `REQUIREMENTS.md` CI-01, `STATE.md:99`),
  not this review's iteration-2 WR-08. Cross-reference is correct; no
  `continue-on-error` remains in the file. Not a finding.
- The committed `spike_megadna_fallback.log` shows `disk_human=8.0K` for a
  ~582MB model; live re-run of `_snapshot_disk_gb("lingxusb/megaDNA_updated")`
  on the current tree returns **0.5824 GB / 582.4MB** (evo2: 2.70 GB) — the
  measurement code is correct; the old 8.0K line was an artifact of that
  throwaway-venv run's cache resolution. This also confirms the workflow
  comment's "582MB" figure. Not a finding.
- Spike-only packages (`evo-model`/`stripedhyena`, `MEGABYTE_pytorch`,
  `pyBigWig`) confirmed absent from every `pyproject.toml` dependency group
  (only a mypy ignore-list mention), `marimo` present in `notebook` — so the
  "On this venv" per-family outcome predictions (evo/evo2/megadna/pybigwig
  FAIL, marimo runs for real) are all correct.

### Live gates re-run

`ruff check .` → 0; `scripts/check_docs_sync.py` → 0; fast leg
`pytest tests/examples/ -m "not slow"` → **94 passed / 1 skipped /
2 deselected** (matches the pre-fix baseline exactly).

### New finding this iteration

WR-09: two pre-existing clauses in the run-step comment block of
`feasibility.yml` (echoed in README §7 step 2) contradict the corrected
WR-08 text seven lines above them and the committed spike evidence. One
Warning; the five known-deferred Info findings (IN-01..05) are re-confirmed
still present and remain Info.

## Warnings

### WR-09: Two residual clauses in the run-step comment contradict the corrected fallback semantics and the committed evidence

**File:** `.github/workflows/feasibility.yml:79-84` (and `.github/workflows/README.md:122`)
**Issue:** The WR-08 fix corrected the first paragraph of the comment block,
but the untouched text below it now contradicts both the corrected sentences
and the committed spike evidence, in two clauses:

1. **"megadna downloads its 582MB checkpoint and fails the unpickle without
   the pinned clone + MEGABYTE_pytorch"** (`feasibility.yml:82-84`). Under the
   dispatched `--family all --fallback` command the runner itself PERFORMS the
   pinned clone (`spike_megadna(..., pinned_clone=fallback)` → clone +
   hash-verify + `sys.path.insert`, `spike_families.py:517-522,799`), so the
   clone is not absent. The committed `spike-logs/spike_megadna_fallback.log`
   (attempt 2a) records the actual mechanism: with the clone present, the
   unpickle advances past `No module named megaDNA` and fails at
   `ModuleNotFoundError: No module named 'MEGABYTE_pytorch'`
   (`megaDNA/megadna.py:9` imports it; the package is in no dependency group).
   The predicted outcome (FAIL as environment evidence, 582MB download —
   figure confirmed live at 582.4MB) is right; the causal clause is wrong,
   and it directly contradicts the just-fixed "pinned megaDNA clone" fallback
   description at lines 76-77. An operator reading the artifact's
   `failure_text` (which mentions only `MEGABYTE_pytorch`) against a comment
   that blames a missing clone could conclude the fallback leg malfunctioned.
2. **"the spike-only packages stay inside this ephemeral job venv"**
   (`feasibility.yml:62-63`, echoed in README §7 step 2's parenthetical).
   Nothing spike-only is installed into the job venv — the install step runs
   only `uv pip install -e ".[base]"`, and lines 80-81 of the same file state
   the opposite: "the spike-only packages are intentionally absent". Same
   term, two contradictory claims about the same venv within one file.

Both clauses pre-date the fix passes (the edits did not introduce them), but
the WR-08 correction makes the first one internally contradictory within a
single comment block, and this phase's own standard (WR-07/WR-08) prices
runbook misdescriptions of runner evidence at Warning.

**Fix:** One surgical edit to each clause:

```yaml
        # Single fallback pass per the plan: --fallback runs each family's
        # D-06 fallback variant (evo-1-8k-base / evo2 noFA-noFP8 config /
        # pinned megaDNA clone), NOT the notebook variant — notebook-variant
        # verdicts come from the committed local evidence (spike-logs/).
        # On this venv the
        # evo/evo2 families fail fast at their handler ImportErrors (the
        # spike-only packages are intentionally absent from this venv), megadna
        # clones the pinned repo (the fallback itself) and downloads its 582MB
        # checkpoint, but the unpickle still FAILS: the clone provides the
        # megaDNA package, not the MEGABYTE_pytorch pip package its model file
        # imports. pybigwig fails its import, and marimo executes for real on
        # the warm ModelScope cache — every failure text is matrix evidence,
        # and the runner exits 0 whenever each family EMITTED its evidence
        # block (the matrix, not this exit code, is the verdict carrier). A
        # red step therefore means an infrastructure crash with missing
        # evidence (import error, OOM kill, disk full) — no `|| true` mask;
        # the upload below still runs via its if: always().
```

and change the venv-install comment (`feasibility.yml:62-63`) plus README §7
step 2's parenthetical to e.g. "the spike-only packages stay OUT of this
ephemeral job venv — their absence is part of the evidence" (README: drop
"stay inside this ephemeral job venv" for "their absence from this venv is
deliberate evidence").

## Info

### IN-01 (carried, iterations 1-2 — still present): `test_timeout` / `extra_inputs` spec keys are dead config

**File:** `tests/examples/_execution.py:54-60` and `tests/examples/test_notebook_execution.py:53,78`
**Issue:** Unchanged: spec dict documents `test_timeout`/`extra_inputs`; test layer hardcodes `@pytest.mark.timeout(1800)` and calls `seed_sandbox(pilot_dir, tmp_path)` without `extra_inputs`. Both coincide at 1800/empty today. Known-deferred to Phase 8.
**Fix:** wire the spec keys when generalizing the fixture, or drop them.

### IN-02 (carried, iterations 1-2 — still present): `assert_tree_clean` fails on pre-existing developer WIP under `example/`

**File:** `tests/examples/_execution.py:173-201`
**Issue:** Teardown asserts absolute cleanliness of `example`/`docs/example` rather than a pre-run baseline/delta comparison. Known-deferred.
**Fix:** snapshot `git status --porcelain --` in fixture setup and assert no new lines in teardown.

### IN-03 (carried, iterations 1-2 — still present): hardcoded developer home path in the mirrored NER dataset generator

**File:** `example/notebooks/finetune_NER_task/generate_bpe_dataset.py:14` (byte-identical mirror at `docs/example/.../generate_bpe_dataset.py:14`, re-confirmed by grep this pass)
**Issue:** `sys.path.insert(0, "/home/forrest/Github/DNALLM")` still present; mirror fidelity intact, so per D-03 the content repair is deferred to the Phase 8 per-notebook loop.
**Fix (Phase 8):** drop the line or derive `Path(__file__).resolve().parents[2]`.

### IN-04 (carried, iterations 1-2 — still present): pinned megaDNA clone uses a fixed shared `/tmp` path

**File:** `scripts/feasibility/spike_families.py:518`
**Issue:** `/tmp/megadna-pinned-clone` remains a predictable shared location; hash verification checks `HEAD` only, not untracked on-disk content. Spike-runner blast radius only. Known-deferred.
**Fix:** `clone_dir = Path(tempfile.mkdtemp(prefix="megadna-pinned-"))`.

### IN-05 (carried, iteration 2 — still present): `# ruff: ignore[rule-name]` comments are inert — not a ruff directive

**File:** `tests/examples/_execution.py:35,185`; `tests/examples/test_notebook_execution.py:13,67`; `scripts/feasibility/spike_families.py:7,259,265,271,635,675`
**Issue:** All twelve invented-syntax comments still present (re-confirmed on current source this pass); they suppress nothing (ruff uses `# noqa:` / `# ruff: noqa:`). Harmless while the project config does not fire those codes — `ruff check .` exits 0. Known-deferred.
**Fix:** delete the comments or replace with real `# noqa: S603, S607`-style directives where suppression is genuinely wanted.

---

_Reviewed: 2026-10-01T20:27:41Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
_Iteration: 3 (final convergence check; CR-01 + WR-01..08 all verified holding; WR-09 new; IN-01..05 carried as known-deferred)_
