---
phase: 27-transformers-4-5-model-compatibility-census
reviewed: 2026-10-28T02:19:44Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - docs/user_guide/model_compatibility.md
  - .github/workflows/census_compat.yml
  - scripts/census_compat.py
  - tests/models/test_compat_docs_consistency.py
  - tests/test_runner_infra_contracts.py
findings:
  critical: 1
  warning: 1
  info: 3
  total: 5
status: issues_found
---

# Phase 27: Code Review Report (incremental #3661 — plan 27-05 gap closure)

**Reviewed:** 2026-10-28T02:19:44Z
**Depth:** standard (diff-aware against base ef0cc2f2)
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Incremental review of phase 27's changes since the prior 27-REVIEW.md commit: the census lane's mamba-kernel install step, the generator's plain-language failure-cell leads with bounded technical suffixes, the regenerated docs page, the drift-gate lockstep extension, and the runner-infra contract pins.

The prior cycle's findings (CR-01 composed-exit, WR-01 disposition carry, WR-03 probe env sanitization) were re-checked in the current state and remain correctly fixed — not re-raised. Verified this cycle: ruff lint/format clean on all five files; 80/80 tests pass in the two test files; `--emit-docs` regeneration is byte-identical to the committed page; the new class map assigns the correct lead class to all six committed broken evidence strings with no ordering hazard; the page's summary counts (19/21/1/6/2 = 49) match the registry; `install_mamba.sh` uses `-m pip` so the lane's pip bootstrap is correct.

One blocker: the census lane's seeded-artifact `cp` writes into a directory that `actions/checkout@v4`'s default clean deletes on the persistent self-hosted workspace, so the lane — the deliverable of UAT G-27-1's fix — is deterministically red on every live dispatch. The green evidence recorded in 27-05-SUMMARY was a manual replica run that bypassed the workflow's step ordering, and the live dispatch is deferred until the v1.4→main merge, so this path has never executed on the real runner.

## Critical Issues

### CR-01: Census lane's seeded `cp` targets a directory deleted by checkout's default clean — the lane is permanently red on every live dispatch

**File:** `.github/workflows/census_compat.yml:45` (mkdir), `:58` (checkout), `:122` (cp)
**Issue:** The only `mkdir -p census-artifacts` runs in the GPU-check step, which executes BEFORE `actions/checkout@v4` (line 58, invoked with defaults, i.e. `clean: true`). On the `[self-hosted, dnallm-nightly]` box the workspace (`_work/DNALLM/DNALLM`) is persistent and shared across all workflows of this repo — the nightly ci.yml lanes check out into the same directory, so a git repo is present at every census dispatch. With a repo present, checkout's `prepareExistingDirectory` runs `git clean -ffdx` before checking out (verified against actions/checkout v4.2.2 source, `src/git-directory-helper.ts:91-100`; if the clean fails it deletes the directory contents outright, which is no better), removing the untracked `census-artifacts/` directory created in step 1. The census step then fails at:

```
cp: cannot create regular file 'census-artifacts/census-fresh.json': No such file or directory
```

so the step exits 1 and the lane can never go green — the exact permanently-red-lane symptom UAT G-27-1 was fixed to eliminate, with a different root cause. The sibling `feasibility.yml` pattern this lane copies does its happy-path `mkdir -p` AFTER checkout (feasibility.yml:96); only its GPU-absent marker path relies on pre-checkout creation, and the workflows README (§Feasibility, line 161) documents exactly that rationale: "the check must precede checkout so the marker survives the **skipped** checkout." The GPU-absent path of this lane is correct for that reason; the happy path is not. The lane's green evidence (`.scratch/p27-uat/lane-sim/census-green.log`, 49 matched / exit 0) was produced by manually pre-seeding `census-artifacts/` on the box and running the census command directly — the mkdir/checkout/cp ordering was never exercised, and live dispatch is deferred (27-05-SUMMARY: GitHub registers dispatchable workflows from the default branch only), so nothing has caught this.
**Fix:** Re-create the directory after checkout, at the top of the census step's run block (mirroring feasibility.yml:96), keeping the pre-checkout mkdir for the GPU-absent marker path:

```yaml
        run: |
          source .venv/bin/activate
          mkdir -p census-artifacts
          cp tests/models/compat_census.json census-artifacts/census-fresh.json
          python scripts/census_compat.py --run --with-loads --verify --artifact census-artifacts/census-fresh.json
```

(Alternative: write the fresh artifact under `$RUNNER_TEMP`, which checkout never touches, and point the upload at the absolute path.)

## Warnings

### WR-01: The no-masking contract pin misses `continue-on-error` — a step-level flag that produces exactly the "silent red-to-green lie" the pin forbids

**File:** `tests/test_runner_infra_contracts.py:318-331`
**Issue:** `test_no_masked_failure_in_the_lane` only greps executed shell lines for `|| true`. Adding `continue-on-error: true` to the mamba-install step or the census-run step (or the job) would green the lane while the kernel build or census fails, re-creating the UAT G-27-1 failure mode, and none of the four pins in `TestCensusLaneInstallSequence` would trip — the class docstring claims it holds against "masking the kernel build," which is broader than what the test checks. The house posture is explicit elsewhere: the example-nightly README section documents "NO step carries `continue-on-error` — a forever-green job is prohibited."
**Fix:** The sibling test already parses the workflow with `yaml.safe_load`; add an assertion that neither the job nor any step carries a `continue-on-error` key:

```python
    def test_no_continue_on_error_anywhere(self) -> None:
        """No job- or step-level continue-on-error: a failed build must redden the lane."""
        doc = yaml.safe_load(_load_census_workflow_text())
        job = doc["jobs"]["census-compat"]
        steps = job.get("steps", [])
        assert "continue-on-error" not in job, "job-level continue-on-error would green a failing lane"
        flagged = [s.get("name", s.get("uses")) for s in steps if "continue-on-error" in s]
        assert not flagged, f"steps carrying continue-on-error: {flagged}"
```

## Info

### IN-01: Bounded technical fragments can end mid-clause with unbalanced parentheses

**File:** `scripts/census_compat.py:1732-1750` (`docs_bound_technical_fragment`); visible at `docs/user_guide/model_compatibility.md:40`
**Issue:** The word-boundary truncation can cut inside a nested parenthetical, leaving the committed DNABERT-2 cell ending "(rebuild_alibi_tensor line 399:…)" — the inner "(" opened by "(rebuild_alibi_tensor" is never closed. Markdown renders it fine, but it reads as a broken sentence on the page whose stated audience is biologists (the plain-language directive this change implements).
**Fix:** Before appending the ellipsis, strip trailing unclosed punctuation (`(`, `[`, `:`, `,`) or cut at the last closing-paren/comma boundary, e.g. `cut = cut.rstrip("(:[,; ")` before `return cut + "…"`. Apply in lockstep (generator + gate's `_bound_fragment`).

### IN-02: The page's summary-sentence counts are not pinned by the drift gate

**File:** `tests/models/test_compat_docs_consistency.py:260-291` (where the pin belongs); generator computes them at `scripts/census_compat.py:1853-1877`
**Issue:** The gate re-derives every table row, the footnotes, and the ordering, but no test re-derives the prose summary "Of the 49 preset models, the team found 19 working as-is, 21 ..., 1 ..., 6 ..., and 2 ..." from the registry. A hand-edited or drifted summary sentence would pass the gate even though it is the first thing a reader sees. Counts verified correct today (19/21/1/6/2 = 49 against `Counter` over the registry).
**Fix:** Add a `TestPageContract` test asserting the summary sentence re-derived from `Counter(entry.tf_compat_status ...)` over `_pretrained()` appears verbatim in the page.

### IN-03: New pure helpers lack direct unit tests for their fail-closed and boundary edge cases

**File:** `scripts/census_compat.py:1706-1729` (`docs_broken_lead_class`), `:1732-1750` (`docs_bound_technical_fragment`)
**Issue:** `tests/models/test_census_compat.py` unit-tests the script's other pure helpers (`TestClassifyConfig`, `TestFailingImportExtraction`, ...) but carries zero tests for the two new ones (grep: no references). Their behavior is pinned only indirectly via the drift gate over the committed-registry snapshot, so the fail-closed `ValueError` on an unmapped evidence shape and the 100/101-character boundary of the bound are never exercised by a test that constructs such input.
**Fix:** Add to `tests/models/test_census_compat.py`: unmapped evidence string raises `ValueError`; empty evidence raises; fragment of exactly 100 chars returned unchanged; 101 chars / no-space fragment bounded with the ellipsis.

---

_Reviewed: 2026-10-28T02:19:44Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
