---
phase: 09-ci-wiring-census-verification
reviewed: 2026-10-06T13:17:16Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - .github/workflows/README.md
  - .github/workflows/ci.yml
  - docs/user_guide/continuous_integration.md
  - mkdocs.yml
  - pyproject.toml
  - tests/TESTING.md
  - tests/examples/test_notebook_execution.py
  - tests/test_models_lock_contracts.py
  - tests/test_runner_infra_contracts.py
findings:
  critical: 1
  warning: 3
  info: 6
  total: 10
status: issues_found
---

# Phase 9: Code Review Report

**Reviewed:** 2026-10-06T13:17:16Z
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Summary

Reviewed the phase-9 CI wiring surface: the rewritten `ci.yml` topology (cron-string gates, example-nightly staged job, D-13 hygiene floors, D-14 failure scene, D-12 budget comments), the CI-08 models.lock guard, the runner-infra and notebook-execution contract tests, and the CI-09 coverage docs. Cross-referenced every claim I could execute locally: the stage-0.5 census triple reproduces exactly (`193/202 tests collected (9 deselected)`), all 20 lock/infra contract tests and the 41 fast-lane notebook-contract tests pass, the runner unit pins exist in `scripts/runner/ollama.service`, and the timeout ladder (outer mark strictly above `cell_timeout`) holds for every gated and active notebook spec.

One blocker: `ruff check .` fails on the current tree (`S105` on `OLD_MODEL_TOKEN` in `tests/test_runner_infra_contracts.py`, introduced with the 261006-lhm model-swap commit). Every push/PR leg (`test`, `test-windows`, `coverage-gate`) runs `ruff check . --statistics` and will go red at the lint step; it escaped notice because all recent runs were `workflow_dispatch` on `phs`, and the nightly legs do not run ruff. Two further warnings are fail-open/fail-soft holes in this phase's new hard gates (the D-13 memory floor and the D-08 stage-4 summary).

In-flight owner decisions were treated as ground truth and are NOT flagged: the deferred num_ctx cut (2026-10-06 00:52), the qwen3.5:4b model swap (15:27), and the exactly-two advisory mypy steps (D-09 boundary).

## Structural Findings (fallow)

No `<structural_findings>` block was provided for this review.

## Narrative Findings (AI reviewer)

### Critical Issues

#### CR-01: Lint gate broken — `OLD_MODEL_TOKEN` trips S105, `ruff check .` fails repo-wide

**File:** `tests/test_runner_infra_contracts.py:154`
**Issue:** `OLD_MODEL_TOKEN = "qwen3.8"` fires ruff `S105` (`hardcoded-password-string`) because the variable name matches the token/password name pattern. Verified with the project's own pinned ruff (0.16.9, repo config): `ruff check . --statistics` reports exactly this one error. The per-file ignores for `tests/**/*.py` exempt `hardcoded-password-func-arg` (S106) but not `hardcoded-password-string` (S105) — only `tests/conftest.py` carries that exemption (`pyproject.toml:377`). The CI jobs `test` (ci.yml:94), `test-windows` (ci.yml:173), and `coverage-gate` (ci.yml:436 area — ruff steps at lines 88-94/167-173) run `ruff check . --statistics` on every push to main/master/dev and every PR, so the next PR or watched-branch push fails all three legs at the lint step. The breakage is invisible so far because recent CI activity was `workflow_dispatch` on `phs` (confirmed via `gh run list`), which runs only the nightly legs — none of which runs ruff. Introduced by the 261006-lhm swap commits (dc44856/b91f2a6) despite their "GREEN" labels.
**Fix:** Rename the constant so it no longer matches the secret-name pattern (it names a model, not a token) and update its three use sites (lines 204, 229, 248):

```python
OLD_MODEL_NAME = "qwen3.8"
NEW_MODEL = "qwen3.5:4b"
```

Alternatively add `# noqa: S105` on the assignment — renaming is cleaner than a suppression or widening the tests per-file-ignores.

## Warnings

### WR-01: Stage-4 summary grep cannot see the "junit missing" failure ledger lines (D-08 forever-green hole)

**File:** `.github/workflows/ci.yml:1041` (writer) and `.github/workflows/ci.yml:1071` (matcher)
**Issue:** The stage-4 audit records a missing junit as `echo "stage4-audit-$junit=1 (junit missing: pytest never wrote it)" >> stage-results.txt` — a line ending in `)`. The fail-soft summary decides redness with `grep -qE '=[1-9][0-9]*$' stage-results.txt`, anchored to end-of-line digits, so this recorded failure value (`=1`) is unreachable by the matcher. The summary can then print "OK: every recorded stage item exited 0" despite a recorded non-zero outcome, contradicting the D-08 contract stated in the job comments ("the stage-4 summary exits non-zero when ANY recorded outcome failed … a forever-green job is prohibited"). The companion failure path (server never ready → `stageN-...=1`) is matched correctly, which is why this has not surfaced; the unmatched shape is any stage-4 audit whose junit file is absent while every invocation-level rc recorded 0.
**Fix:** Record the bare code on the ledger line and keep the reason on stdout only:

```bash
else
  echo "stage4-audit-$junit=1 (junit missing: pytest never wrote it)"
  echo "stage4-audit-$junit=1" >> stage-results.txt
fi
```

(or drop the `$` anchor: `grep -qE '=[1-9][0-9]*[^0-9]*' stage-results.txt` — but the two-line form keeps the ledger machine-parseable).

### WR-02: D-13 memory-floor hard gates fail open when the `free` parse yields an empty value

**File:** `.github/workflows/ci.yml:904-910` (stage 1.5) and `.github/workflows/ci.yml:988-994` (stage 2.5)
**Issue:** `AVAIL_GI=$(LC_ALL=C free -g | awk '/^Mem:/{print $7}')` followed by `if [ "${AVAIL_GI}" -lt 35 ]; then … exit 1; fi`. If the awk extraction ever yields an empty string (a `free` build/variant without the seventh "available" column, an unexpected output format, or a localized row label despite `LC_ALL=C` in a nested-su environment), `[ "" -lt 35 ]` emits "integer expression expected" and returns exit status 2 — which the `if` treats as false, so the floor PASSES and the step goes green while having measured nothing. That is exactly the "floor that measures nothing" failure mode the step's own comment (Pitfall 1) says it exists to prevent, and these are HARD gates protecting server-binding stages on an exhausted box. The echoed `after: ${AVAIL_GI}Gi available` would show the empty value, but nothing fails.
**Fix:** Fail closed on an unparseable value, in both hygiene steps:

```bash
AVAIL_GI=$(LC_ALL=C free -g | awk '/^Mem:/{print $7}')
if [ -z "${AVAIL_GI}" ]; then
  echo "FAIL: could not parse the free -g available column (floor unchecked — refusing to continue)"
  exit 1
fi
```

### WR-03: `tests/TESTING.md` coverage guidance contradicts the enforced gate the phase just documented

**File:** `tests/TESTING.md:160-164` (also 9, 172-184, 206)
**Issue:** The phase's CI-09/D-15 deliverable established the coverage-expectation story (90 floor enforced on every `--cov` invocation, no codecov upload in CI) in the new user-guide page and the workflows README — but `tests/TESTING.md`, the doc CI links to for local census commands, still teaches the opposite: "Coverage Targets — Overall Coverage: Aim for >80%" (line 162) vs the enforced `fail_under = 90`; a "CI/CD Integration" example showing a `codecov/codecov-action@v3` upload (lines 174-184) vs the workflows README's explicit "no XML coverage report or codecov upload is produced in CI"; a stale self-reference calling this file `README.md` in the structure tree (line 9); and a "Missing Dependencies" tip recommending `pytest-xdist` (line 206), which is not a dependency anywhere in `pyproject.toml`. A contributor following this page will aim at the wrong floor and expect a codecov gate that does not exist.
**Fix:** Update the Coverage Targets section to state the 90 floor (ratchet, suite at ~96.3%), replace the codecov example with the actual CI invocation (`pytest -m "not slow" --cov --junitxml=...` + `scripts/audit_skips.py`), fix the tree's file name, and drop the pytest-xdist mention.

## Info

### IN-01: Deploy job uses deprecated `actions/cache@v3` and a run-number key that never exact-hits

**File:** `.github/workflows/ci.yml:1100-1106`
**Issue:** Every other cache step in the file uses `actions/cache@v4`; the mkdocs cache alone pins `@v3` (deprecated major, aging out of support). Additionally `key: mkdocs-material-${{ github.run_number }}` guarantees an exact-key miss every run (the key strictly increases), so the step always restores via the `mkdocs-material-` prefix fallback and post-job saves a brand-new entry each run — constant cache churn against the 10 GB quota for a small docs cache.
**Fix:** Bump to `actions/cache@v4` and key on something stable (e.g. `hashFiles('pyproject.toml')`-based, or the docs extra pins) unless the always-refresh behavior is intentional — if it is, say so in a comment.

### IN-02: `test-mamba` failure artifact lists a log no step ever writes

**File:** `.github/workflows/ci.yml:361`
**Issue:** The `Upload mamba test logs on failure` step includes `/tmp/mamba-build.log` in its path list, but no step in the job writes that file (the kernel build happens inside `uv pip install -e ".[mamba]"` with no tee). On failure the artifact silently contains only `pytest.log`; the build-failure evidence the path implies is never captured.
**Fix:** Either tee the install to that file (`uv pip install -e ".[mamba]" --no-cache-dir --no-build-isolation 2>&1 | tee /tmp/mamba-build.log` with `set -o pipefail`) or drop the stale path.

### IN-03: `models.lock` header still documents the cache layer this phase deleted (cross-file, D-11)

**File:** `models.lock:2-3` (cross-file observation surfaced by reviewing the D-11 deletions in ci.yml)
**Issue:** The lock header says "Keys the gated CI job's model cache (actions/cache hashFiles)" — but D-11 (owner decision 2026-10-05) removed the models.lock-keyed hub cache restore from both nightly jobs, and the header's edit-rotation instruction ("Edit an entry to rotate the cache key") now references a nonexistent mechanism. Not in the reviewed file list, but the staleness was created by this phase's ci.yml change.
**Fix:** Reword the header to describe the lock's current role (provenance contract audited by `tests/test_models_lock_contracts.py`) and drop the cache-key sentence.

### IN-04: Lock parser accepts duplicate/absent `dataset:` rows despite the "single row" fail-closed claim

**File:** `tests/test_models_lock_contracts.py:196-202`
**Issue:** `_parse_lock_rows` docstring promises fail-closed parsing of "the single `dataset:` row", but a lock with two `dataset:` rows silently keeps the last one, and a lock with no dataset row at all parses cleanly (returning `""`). The live-tree test pins the exact dataset id, so real drift is still caught downstream — this is a parser-contract gap, not a live hole.
**Fix:** Track whether a dataset row was already seen and raise on the second, and/or add a `dataset_id is empty` check to the live-tree test.

### IN-05: Redundant disjunct in the README rationale assertion

**File:** `tests/test_runner_infra_contracts.py:105`
**Issue:** `assert "num_ctx 8192" in text or ("num_ctx" in text and "8192" in text)` — the left disjunct is strictly subsumed by the right, so the condition reduces to the right side and reads as if it expressed three distinct acceptance shapes.
**Fix:** Simplify to `assert "num_ctx" in text and "8192" in text`.

### IN-06: Unpinned `latest` micromamba binary fetched and executed on the self-hosted GPU box

**File:** `.github/workflows/ci.yml:567-568` and `.github/workflows/ci.yml:765-766`
**Issue:** Both bedtools rootless steps fetch `https://micro.mamba.pm/api/micromamba/linux-$(uname -m)/latest` — a floating "latest" binary pulled over TLS with no version or checksum pin, then executed to install packages from conda-forge/bioconda into the runner environment. On the single-purpose `dnallm-nightly` box this is a nondeterministic toolchain and an upstream-supply-chain surface; the same code exists in both jobs, so a pin would be defined once.
**Fix:** Pin a micromamba release version in the URL (e.g. `.../api/micromamba/linux-$(uname -m)/<version>`) and optionally verify a checksum before use; bump deliberately like any other toolchain pin.

---

### Verification evidence (commands run during review)

- `.venv/bin/ruff check . --no-cache --statistics` → `1 S105 hardcoded-password-string` (tests/test_runner_infra_contracts.py:154) — the basis of CR-01.
- `.venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants" -k "not mcp_example"` → `193/202 tests collected (9 deselected)` — the stage-0.5 pinned triple matches the live tree.
- `.venv/bin/python -m pytest tests/test_models_lock_contracts.py tests/test_runner_infra_contracts.py` → 20 passed; `pytest tests/examples/test_notebook_execution.py -m "not slow"` → 41 passed, 23 deselected; `pytest tests/configuration/test_yaml_load.py` → 21 passed.
- `grep OLLAMA_ scripts/runner/ollama.service` → both D-06/D-12 Environment pins present (contract tests are not vacuous).
- Timeout-ladder audit of `tests/examples/_execution.py` specs: every gated notebook at `cell_timeout: 3600` carries the 7200s outer override (`_TIMEOUT_7200_GATED`); all others stay strictly below their class marks — no strictly-above violation.
- `mkdocs.yml` and `docs/user_guide/continuous_integration.md` were checked against the implemented topology (cron strings, gates, job names, stage table, coverage numbers) — consistent; no findings.

_Reviewed: 2026-10-06T13:17:16Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
