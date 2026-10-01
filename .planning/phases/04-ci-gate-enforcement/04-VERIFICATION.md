---
phase: 04-ci-gate-enforcement
verified: 2026-09-30T18:58:00Z
status: passed
previous_status: human_needed
closed: 2026-10-01T05:00:00Z
score: 10/10 must-haves verified
covered_files:
  - .planning/phases/04-ci-gate-enforcement/04-01-PLAN.md
  - .planning/phases/04-ci-gate-enforcement/04-02-PLAN.md
  - .planning/phases/04-ci-gate-enforcement/04-03-PLAN.md
  - .planning/phases/04-ci-gate-enforcement/04-01-SUMMARY.md
  - .planning/phases/04-ci-gate-enforcement/04-02-SUMMARY.md
  - .planning/phases/04-ci-gate-enforcement/04-03-SUMMARY.md
  - pyproject.toml
  - models.lock
  - .github/workflows/ci.yml
  - .github/workflows/README.md
  - tests/finetune/test_trainer_real_model.py
  - tests/inference/test_inference.py
covered_digest: "v2:sha256:a7b4a1a93b29e3509c339d183b8e16a80acb07776faf04e81bae1aec0b306392"
behavior_unverified: 0 # sole deferred item (nightly green completion) closed 2026-10-01 — see Deferred Close Addendum
overrides_applied: 0
behavior_unverified_items: []
human_verification_resolved:
  - test: "Observe nightly census to green completion"
    resolution: "RESOLVED 2026-10-01 — run 36811033498 (workflow_dispatch dev, self-hosted runner dnallm-nightly, post model-swap 95c9ba0): conclusion success; 1656 passed / 7 allowlisted skips / 0 failed in 905.93s; Total coverage 96.30% >= 90; skip audit OK. Job wall 1h21m57s incl. one-time models-cache re-save (models.lock key changed). Supersedes hosted calibration run 36747594207 which died at the 360-min platform cap — census moved to the self-hosted runner per the recorded GATE-02 amendment"
  - test: "Owner configures branch protection making coverage-gate a required check on dev and main"
    resolution: "RESOLVED 2026-10-01 — gh api .../branches/{dev,main}/protection returns required context 'coverage-gate (py3.12, fast leg)' on both branches (non-strict)"
  - test: "Optionally cancel the hung test-cuda (3.11, 12.4) job on probe run 36749810723"
    resolution: "RESOLVED — run settled to completed/failure on its own (2026-09-30T19:27:08Z); coverage-gate FAILURE evidence unaffected"
---

# Phase 4: CI Gate Enforcement Verification Report

**Phase Goal:** Coverage cannot regress — the gate goes live green and provably fails CI when coverage drops
**Verified:** 2026-09-30T18:58:00Z
**Status:** passed (human_needed items resolved 2026-10-01T05:00Z — see Deferred Close Addendum)
**Re-verification:** No — initial verification; deferred observational item closed additively

**GATE-02 scope note:** Requirement GATE-02 was AMENDED by owner decision 2026-09-30 (04-CONTEXT.md): the gated PR job runs the fast leg (`-m "not slow"`); the slow-inclusive suite runs as a scheduled nightly job under the same `fail_under=90`. Verification below applies the amended shape.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `fail_under = 90` active in `[tool.coverage.report]`, enforced through the pytest exit code; identical command locally and in CI; no local/CI threshold fork (GATE-01) | ✓ VERIFIED | `tomllib` read of `pyproject.toml`: `fail_under: 90` (keys: fail_under, show_missing). grep of `ci.yml` for `fail_under\|fail-under\|--cov-fail\|codecov`: zero hits — the workflow carries no threshold constant; both CI census steps run bare `--cov` against the same pyproject config. Behavioral: local gated run and CI run (truths 2-3) |
| 2 | The enforced fast-leg command passes locally at current HEAD (post-summary fix rounds included) | ✓ VERIFIED | Own run: `pytest -m "not slow" -ra --durations=0 --junitxml=... --cov -p no:cacheprovider -p no:progress` → **rc=0**, `Required test coverage of 90.0% reached. Total coverage: 96.29%`, 1635 passed / 1 skipped / 27 deselected in 109.86s — at HEAD b126e88 |
| 3 | The gate is live and green on dev (push) | ✓ VERIFIED | `gh run view 36745734429` (push, dev, 2026-09-30T16:38:56Z): conclusion **success**; coverage-gate (py3.12, fast leg) success at 96.27%; all six `test` matrix legs + test-cuda x2 + test-mamba success; coverage-nightly + deploy skipped — event guards proven at runtime |
| 4 | Amended two-job CI shape exists structurally: coverage-gate (push/PR, fast leg, py3.12, numpy 2.2.0) + coverage-nightly (cron `0 3 * * *` + workflow_dispatch, full slow-inclusive census, same `--cov`, models.lock-keyed cache, timeout backstop); matrix/cuda/mamba/deploy event-guarded; deploy keeps push-to-main/master-only (GATE-02 amended) | ✓ VERIFIED | PyYAML parse of `ci.yml`: triggers push+pull_request on [main, master, dev] + schedule + workflow_dispatch; jobs `[test, test-cuda, test-mamba, coverage-gate, coverage-nightly, deploy]`; coverage-gate `if: push\|\|pull_request`, timeout-minutes 90, step runs `-m "not slow" ... --cov`; coverage-nightly `if: schedule\|\|workflow_dispatch`, timeout-minutes **900** (post-summary WR-05 fix, bab8827), full census (no marker filter), `actions/cache@v4` over `~/.cache/huggingface/hub` + `~/.cache/modelscope/hub` keyed `hashFiles('models.lock')`; guards on test/test-cuda/test-mamba = push/PR-only; deploy = push to main/master; permissions `contents: read` |
| 5 | The nightly job completes a green slow-inclusive census end to end | ✓ VERIFIED | **Deferred item closed 2026-10-01**: run 36811033498 (workflow_dispatch on dev, self-hosted runner `dnallm-nightly`, post model-swap 95c9ba0) concluded **success** — log contains verbatim `Required test coverage of 90.0% reached. Total coverage: 96.30%` and `1656 passed, 7 skipped, 8 warnings in 905.93s (0:15:05)`; skip audit step `OK: every skip in pytest-junit-nightly.xml matches the allowlist`; census 03:36:53Z→03:51:59Z. Job wall 1h21m57s (03:33:15Z→04:55:12Z); the hour after census is the one-time post-job models-cache re-save (models.lock key changed with the dnagpt swap) — steady-state nightly cost is dominated by the 15-min census |
| 6 | A real coverage-dropping PR to dev produces a FAILED coverage-gate check with the verbatim fail-under line in the CI log (GATE-04, end-to-end exercise of the Phase-1 exit-code fix) | ✓ VERIFIED | PR #39 (base dev, head gate-regression-probe): `gh run view 36749810723` → coverage-gate job conclusion **failure**; job 110005226448 log fetched via API contains verbatim: `ERROR: Coverage failure: total of 79 is less than fail-under=90`, `FAIL Required test coverage of 90.0% not reached. Total coverage: 78.91%` (TOTAL 7405/1562), and `==== 1261 passed, 1 skipped, 25 deselected ====` — all tests green, only the floor made the job red |
| 7 | Zero probe residue | ✓ VERIFIED | PR #39: state CLOSED, mergedAt null (never merged); `git ls-remote origin 'refs/heads/*probe*'` empty; no local probe branch; `tests/models/` restored on dev (test_head, test_losses, test_model, test_special, test_tokenizer) |
| 8 | PRs to both dev and main trigger the gated coverage job (GATE-05) | ✓ VERIFIED | dev: live — PR #39 (base dev) triggered coverage-gate (the failed check of truth 6). main: structural — pull_request trigger block is `[main, master, dev]` (PyYAML), inherited by coverage-gate; making the check *required* is the documented owner follow-up (04-02 FA-GATE-05), out of executor scope |
| 9 | Codecov uploader step and orphaned coverage.xml export removed; no dead/failing reporting step; permissions stay contents: read (GATE-03) | ✓ VERIFIED | `'codecov' in ci.yml`: False (case-insensitive); `'coverage.xml' in ci.yml`: False; permissions: `{contents: read}`. Reporting is terminal + junit only; both census jobs run `scripts/audit_skips.py` on their own junit (fail-closed skip audit wired in both jobs) |
| 10 | Workflows README documents both new jobs, the enforced 90 floor, and scoped-run `--no-cov` guidance in existing style (WR-05 minimal-touch) | ✓ VERIFIED | `.github/workflows/README.md`: sections for `coverage-gate` (line 89) and `coverage-nightly` (line 104) naming jobs exactly as their `name:` fields; enforced-floor explanation (line 136: "identical everywhere"); nightly 03:00 UTC schedule (line 15); `--no-cov` guidance (lines 213-217); flipped slow-test claims corrected (lines 130, 244) |

**Score:** 10/10 truths verified (deferred truth 5 closed 2026-10-01 — see Deferred Close Addendum)

### Wave-1 Input Truths (04-01 must-haves)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| W1 | Local full census exits 0 above the 90 floor | ✓ VERIFIED | 04-01-SUMMARY evidence corroborated by structure (fail_under present) and by the verifier's own fast-leg green run at HEAD (96.29%); the full-suite command of record is unchanged in CI nightly |
| W2 | Synthetic drop exits 1 with the fail-under line locally | ✓ VERIFIED | Reproduced end-to-end in real CI at truth 6 (78.91% red vs local rehearsal 78.92%) — the stronger, live version of this proof |
| W3 | models.lock has exactly 9 provenance-commented entries (2 hf + 6 ms + 1 dataset) | ✓ VERIFIED | `grep -c` on models.lock: 9; inspection: 2 `hf` + 6 `ms` + 1 `dataset` line, each with test-call-site provenance comments |
| W4 | 7 per-test timeout marks (3×7200 + 4×3600) override the global 300s; addopts `--timeout=300` and markers list untouched | ✓ VERIFIED | Counts: trainer_real_model 3×`timeout(7200)` + 3×`timeout(3600)`, inference 1×`timeout(3600)` = 7 marks / 4 at 3600; pyproject addopts still `--timeout=300`. Post-summary WR-02 round (e8c8053) added the remaining cold-download marks; WR-03/WR-05 raised job timeout to 900 min at HEAD |

### Prohibitions (04-03 must_haves.prohibitions — judgment-tier)

| Prohibition | Status | Evidence |
|-------------|--------|----------|
| `fail_under` leaves the phase at exactly 90 — never raised/lowered | ✓ UPHELD | tomllib: `fail_under: 90` at HEAD; no other threshold anywhere (workflow grep clean) |
| No test deleted/skipped/excluded on any landing branch to make a gated run green; probe branch never merged | ✓ UPHELD | PR #39 mergedAt null; dev tree at HEAD has tests/models intact (7 entries); probe run's only red was the floor itself (1261 passed); `expected_skips.yaml` allowlist unchanged from Phase 2/3 |
| No local/CI gate fork — no CI-only threshold flags, no duplicated constants, no custom comparison scripts | ✓ UPHELD | Both CI census steps are bare `--cov` pytest invocations; zero threshold literals in ci.yml; single enforcement source is pyproject `[tool.coverage.report]` |

Prohibition verdicts are LLM-judgment (judgment-tier, no test-tier consumer); recorded as verified-by-inspection, human review optional.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pyproject.toml` | `fail_under = 90` in `[tool.coverage.report]` | ✓ VERIFIED | tomllib read confirms; report keys otherwise unchanged (show_missing) |
| `models.lock` | 9-entry manifest keyed by hashFiles | ✓ VERIFIED | 9 entries incl. `microsoft/DialoGPT-small`; consumed by nightly cache key `hashFiles('models.lock')` (wired) |
| `.github/workflows/ci.yml` | coverage-gate + coverage-nightly jobs, triggers, guards, uploader removal | ✓ VERIFIED | PyYAML-validated; both jobs present and live-proven (truths 3-4, 6) |
| `.github/workflows/README.md` | documents both jobs + 90 floor + --no-cov | ✓ VERIFIED | Truth 10 |
| `tests/finetune/test_trainer_real_model.py` | per-test timeout marks + CR-01 fail-closed | ✓ VERIFIED | 6 timeout marks; `pytest.fail(f"Configuration file not found: ...")` replaces `return False` (8eed59c at HEAD) |
| `tests/inference/test_inference.py` | timeout mark + WR-06 fail-closed | ✓ VERIFIED | `timeout(3600)` on test_real_model_integration; `self.fail(f"Real-model integration workflow failed: {e}")` replaces skipTest (6a270e5 at HEAD) |
| `04-03-SUMMARY.md` | probe evidence + owner hand-off | ✓ VERIFIED | Contains verbatim fail-under line, PR/run URLs, hand-off sections |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| pyproject fail_under | coverage-gate / coverage-nightly pytest steps | both run bare `--cov` against the same config | ✓ WIRED | Step commands inspected (lines 283, 366); zero threshold in workflow — enforcement by construction; live-proven both green (96.27%) and red (78.91%) |
| models.lock | coverage-nightly actions/cache | `hashFiles('models.lock')` over both hub roots | ✓ WIRED | Cache step paths `~/.cache/huggingface/hub` + `~/.cache/modelscope/hub`, key on models.lock hash |
| gate/nightly junit | scripts/audit_skips.py | fail-closed skip audit per job | ✓ WIRED | Both jobs run `audit_skips.py <own junit> tests/expected_skips.yaml` |
| timeout marks | pytest-timeout plugin | marker precedence over `--timeout=300` | ✓ WIRED | 7 marks present; pytest-timeout in deps (`>=2.3.1,<2.5`); 04-01 verified live marker acceptance via `--collect-only` rc=0 |
| README | ci.yml job names | documentation matches `name:` fields | ✓ WIRED | README names `coverage-gate`/`coverage-nightly` exactly as the workflow's name: fields (required-check context match for the owner) |

### Data-Flow Trace (Level 4)

Not a data-rendering phase — Level 4 maps to exit-code flow instead: measured coverage total → pytest-cov `fail_under` comparison → pytest rc → GitHub step/job conclusion. Traced end-to-end in both directions with live artifacts (green run 36745734429 success at 96.27%; red run 36749810723 coverage-gate FAILURE with `Coverage failure: total of 79 is less than fail-under=90` in the job log; 1261 tests passed in the red run — the floor, not a test failure, produced the red conclusion). Status: ✓ FLOWING.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Local gated fast leg green at HEAD | `.venv/bin/python -m pytest -m "not slow" -q --cov ... -p no:progress` | rc=0, 96.29%, 1635 passed / 109.86s | ✓ PASS |
| CI gate green (push, dev) | `gh run view 36745734429` | conclusion success; coverage-gate + 6 matrix legs success; nightly/deploy skipped | ✓ PASS |
| CI gate red on coverage drop (GATE-04) | `gh api .../jobs/110005226448/logs` | verbatim `Coverage failure: total of 79 is less than fail-under=90`; coverage-gate conclusion failure | ✓ PASS |
| Probe zero residue | `gh pr view 39` + `git ls-remote origin 'refs/heads/*probe*'` | CLOSED / mergedAt null; branch absent | ✓ PASS |
| fail_under value | tomllib parse | `90` | ✓ PASS |

### Probe Execution

No `scripts/*/tests/probe-*.sh` convention in this repo; the phase's probe is the GATE-04 ephemeral PR, executed and verified live above (truths 6-7). PASS.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| GATE-01 | 04-01 | `fail_under = 90` enforced through pytest exit code, identical local/CI | ✓ SATISFIED | Truths 1-3 |
| GATE-02 | 04-01, 04-02 | Dedicated slow-inclusive coverage CI job — **amended**: gated fast-leg PR job + nightly slow census, same floor, models.lock cache, timeout marks/backstop | ✓ SATISFIED (amended) | Truths 4, 5, W3, W4 — nightly green completion verified 2026-10-01 (run 36811033498) |
| GATE-03 | 04-02 | Dead codecov v3 step fixed or removed | ✓ SATISFIED | Truth 9 (removed; owner disposition) |
| GATE-04 | 04-03 | Synthetic regression provably fails CI | ✓ SATISFIED | Truths 6-7 |
| GATE-05 | 04-02, 04-03 | Gate triggers on PRs to dev and main | ✓ SATISFIED | Truth 8 (dev live; main structural) |

Orphaned requirements: none — REQUIREMENTS.md maps exactly GATE-01..05 to Phase 4, all claimed by plans.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | No TBD/FIXME/XXX/TODO markers in any phase-modified file | - | Clean |

ℹ️ Info: probe run 36749810723 run-level status remains `in_progress` because one hosted test-cuda (3.11, 12.4) job hung after branch deletion (empty conclusion). The coverage-gate job itself completed with FAILURE — GATE-04 evidence unaffected. Optional owner cleanup listed under Human Verification.

### Human Verification Required

### 1. Nightly calibration run terminal state (feeds owner runtime decision)

**Test:** Watch run 36747594207 to completion (`gh run view 36747594207 --repo zhangtaolab/DNALLM`), or await the first 03:00 UTC cron run.
**Expected:** coverage-nightly concludes success with total ≥ 90 (structure and dispatch isolation already proven; census started healthy at 04-02 close). If it fails, capture the exact failure mode (exit 137 / job-terminated / coverage floor) — that evidence is the owner's runtime-menu input (720-min dispatch-time window vs 900-min at HEAD).
**Why human:** Live CI outcome still in flight at verification (2h02m elapsed as of 2026-09-30T18:56:52Z); this is the plan's own deferred phase-close evidence (04-02 SUMMARY: "should be allowed to finish; its conclusion ... is the phase-close evidence for the owner's runtime decision").

### 2. Branch protection: make coverage-gate a required check

**Test:** In repo settings, add the `coverage-gate (py3.12, fast leg)` check context as required on dev and main (payload drafted in 04-03-SUMMARY owner hand-off).
**Expected:** PRs cannot merge with a red coverage-gate check.
**Why human:** Explicitly scoped to the owner by 04-02 FA-GATE-05 and 04-03 hand-off — a repo-settings action, not a code artifact; neither branch is currently protected.

### 3. Optional: cancel hung CUDA job on probe run

**Test:** Cancel the stuck test-cuda (3.11, 12.4) job on run 36749810723 via the Actions UI.
**Expected:** Run-level status resolves; coverage-gate FAILURE evidence already recorded and unaffected.
**Why human:** GitHub-hosted runner hang after probe branch deletion — infrastructure cleanup outside the repo.

### Gaps Summary

No failed must-haves. All five requirements (GATE-01..05, with GATE-02 under the recorded owner amendment) are satisfied with live, independently re-observed evidence: the gate is green on dev in real CI (run 36745734429, 96.27%), provably bites on a real coverage-dropping PR (PR #39 / run 36749810723, verbatim `Coverage failure: total of 79 is less than fail-under=90` with 1261 tests passing), leaves zero residue, removes the dead codecov step, and passes locally at HEAD (96.29%, rc=0) including all post-summary code-review fix rounds (CR-01/CR-02 fail-closed tests, WR-06 fail-closed, timeout 900). The single item open at initial verification was observational, not structural — the nightly slow census had not yet completed a run — and is now closed green (see Deferred Close Addendum). Status: passed.

## Deferred Close Addendum (2026-10-01T05:00Z)

The three human/deferred items were resolved after initial verification:

1. **Nightly green completion — CLOSED PASS.** Successor run **36811033498** (workflow_dispatch on dev, self-hosted runner `dnallm-nightly`, dispatched 2026-10-01T03:33Z after the open_chromatin mamba→dnagpt swap in 95c9ba0) concluded **success**: `1656 passed, 7 skipped, 8 warnings in 905.93s (0:15:05)`; `Required test coverage of 90.0% reached. Total coverage: 96.30%`; skip audit OK. This supersedes the hosted calibration run 36747594207, which died at the GitHub-hosted 360-min platform cap (6h00m33s census) — the census moved to the self-hosted runner under the recorded GATE-02 amendment, which removes the platform cap. Owner runtime decision input: steady-state census ≈ 15 min; the 1h21m57s job wall on this run includes the one-time post-job models-cache re-save forced by the models.lock key change.
2. **Branch protection — APPLIED.** `gh api .../branches/{dev,main}/protection` returns `required_status_checks.contexts = ["coverage-gate (py3.12, fast leg)"]` on both dev and main (non-strict enforcement).
3. **Hung probe job — SETTLED.** Run 36749810723 resolved to `completed`/`failure` on its own (2026-09-30T19:27:08Z); the coverage-gate FAILURE evidence is unaffected.

En route to the green run, CR-02's fail-closed asserts surfaced a real latent defect: the `plant-dnamamba-BPE-open_chromatin` test model (ModelScope mamba remote code) never loaded under transformers 5.x (`MambaCache` removed) and had been false-passing; fixed by swapping to cached non-mamba `plant-dnagpt-BPE-promoter` (95c9ba0) and reverting the spurious `transformers<5.18` bound to `<6` (MambaCache absence exists in 5.17 too; risk documented in dependabot.yml, bfe4ee8).

---

_Verified: 2026-09-30T18:58:00Z_
_Verifier: Claude (gsd-verifier)_
