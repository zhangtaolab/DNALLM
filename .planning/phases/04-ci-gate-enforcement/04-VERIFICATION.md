---
phase: 04-ci-gate-enforcement
verified: 2026-10-01T05:15:27Z
status: passed
score: 10/10 must-haves verified
covered_files:
  - .github/dependabot.yml
  - .github/workflows/README.md
  - .github/workflows/ci.yml
  - .planning/phases/04-ci-gate-enforcement/04-01-PLAN.md
  - .planning/phases/04-ci-gate-enforcement/04-02-PLAN.md
  - .planning/phases/04-ci-gate-enforcement/04-03-PLAN.md
  - .planning/phases/04-ci-gate-enforcement/04-01-SUMMARY.md
  - .planning/phases/04-ci-gate-enforcement/04-02-SUMMARY.md
  - .planning/phases/04-ci-gate-enforcement/04-03-SUMMARY.md
  - dnallm/mcp/tests/configs/open_chromatin_inference_config.yaml
  - dnallm/mcp/tests/test_mcp_functionality.py
  - models.lock
  - pyproject.toml
  - tests/finetune/test_trainer_real_model.py
  - tests/inference/test_inference.py
covered_digest: "v2:sha256:6c8a7561aa20f940001bc374b8dadd27768010b61756a04df8c6e68647d224ad"
behavior_unverified: 0 # all behavior-dependent truths carry live CI evidence in both directions (green 36811033498 / red 36749810723), re-observed this pass
overrides_applied: 0
re_verification:
  previous_status: passed
  previous_score: 10/10
  trigger: stale covered_digest — post-verification source changes (95c9ba0 model swap + transformers bound revert, bfe4ee8 dependabot docs, ci.yml self-hosted runner + free-disk guards)
  gaps_closed:
    - "Nightly slow-inclusive census green end-to-end — run 36811033498 re-confirmed live this pass: conclusion success at headSha 95c9ba0, verbatim log lines re-extracted from job 110205918141"
    - "Branch protection required check — re-confirmed live this pass via gh api on dev AND main"
    - "Probe run 36749810723 settled completed/failure — re-confirmed live this pass"
  gaps_remaining: []
  regressions: []
---

# Phase 4: CI Gate Enforcement Verification Report

**Phase Goal:** Coverage cannot regress — the gate goes live green and provably fails CI when coverage drops
**Verified:** 2026-10-01T05:15:27Z
**Status:** passed
**Re-verification:** Yes — re-run after the stale-digest gate trip (prior pass 2026-09-30T18:58Z scored 10/10; this pass re-verified every truth against current HEAD 3c60673 and independently re-observed all live CI evidence)

**GATE-02 scope note:** GATE-02 was AMENDED by owner decision 2026-09-30 (04-CONTEXT.md): the gated PR job runs the fast leg (`-m "not slow"`); the slow-inclusive census runs as a scheduled nightly under the same `fail_under=90`, now on the self-hosted `dnallm-nightly` runner (hosted 360-min platform cap killed calibration run 36747594207). Verification applies the amended shape.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence (re-verified at HEAD 3c60673 / live CI) |
|---|-------|--------|--------------------------------------------------|
| 1 | `fail_under = 90` active in `[tool.coverage.report]`, enforced through the pytest exit code; identical command locally and in CI; no local/CI threshold fork (GATE-01) | ✓ VERIFIED | tomllib read at HEAD: `fail_under: 90`, report keys exactly `{fail_under, show_missing}`, run.omit 7 entries, addopts `--timeout=300` intact. `grep -nE "fail_under\|fail-under\|--cov-fail" ci.yml` → zero hits; both census steps run bare `--cov` against the same pyproject config. Behavioral both directions live (truths 5-6) |
| 2 | The enforced fast-leg command passes at current source state | ✓ VERIFIED | Live CI, not just local: push runs 36811033050 (95c9ba0), 36811124461 (bfe4ee8), 36812230468 (4a9672a) on dev all conclude **success**. The 3 commits local-only past origin/dev (566a0e1/84484aa/3c60673) are `.planning/`-docs-only — verified by per-commit `git show --stat` — so the green runs cover the current source state |
| 3 | The gate is live and green on dev (push) | ✓ VERIFIED | `gh run list --workflow CI --branch dev --event push`: three most recent pushes all `success` (above); each carries the coverage-gate job per the workflow structure |
| 4 | Amended two-job CI shape exists structurally: coverage-gate (push/PR, fast leg, py3.12, numpy 2.2.0, timeout 90) + coverage-nightly (cron `0 3 * * *` + workflow_dispatch, full census, same `--cov`, models.lock-keyed dual-hub cache, timeout 900, self-hosted `[self-hosted, dnallm-nightly]`, continue-on-error false); test/test-cuda/test-mamba/deploy event-guarded; deploy push-to-main/master-only with needs unchanged | ✓ VERIFIED | PyYAML parse at HEAD: triggers push+PR `[main, master, dev]` + workflow_dispatch + schedule; jobs `[test, test-cuda, test-mamba, coverage-gate, coverage-nightly, deploy]`; both census commands verbatim (`-m "not slow"` gate line; nightly full-census line); both skip audits on own junit; cache key `${{ runner.os }}-models-${{ hashFiles('models.lock') }}` over BOTH hub roots; py3.12 both coverage jobs; permissions `contents: read`; 4× `runner.environment == 'github-hosted'` free-disk guards (lines 37/135/251/325); nightly `continue-on-error: False` |
| 5 | The nightly job completes a green slow-inclusive census end to end | ✓ VERIFIED | **Live, independently re-observed this pass.** `gh run view 36811033498`: `completed/success`, event workflow_dispatch, headSha **95c9ba0** (post model-swap). Run's job list: exactly ONE non-skipped job — `coverage-nightly` success, all five others skipped (event isolation live on the self-hosted runner). Verbatim lines re-extracted from job 110205918141's log via the API: `Required test coverage of 90.0% reached. Total coverage: 96.30%`, `1656 passed, 7 skipped, 8 warnings in 905.93s (0:15:05)`, `OK: every skip in pytest-junit-nightly.xml matches the allowlist` |
| 6 | A real coverage-dropping PR to dev produces a FAILED coverage-gate check with the verbatim fail-under line in the CI log (GATE-04, end-to-end exercise of the Phase-1 exit-code fix) | ✓ VERIFIED | **Live, independently re-observed this pass.** Job 110005226448 log re-fetched via API; verbatim lines present: `ERROR: Coverage failure: total of 79 is less than fail-under=90`, `FAIL Required test coverage of 90.0% not reached. Total coverage: 78.91%`, `1261 passed, 1 skipped, 25 deselected` — all tests green, only the floor made the job red. Run 36749810723 now `completed/failure` (the hung cuda leg settled 2026-09-30T19:27:08Z) |
| 7 | Zero probe residue | ✓ VERIFIED | Re-checked this pass: `git ls-remote origin 'refs/heads/*probe*'` → empty; `tests/models/` intact on dev at HEAD (test_head, test_losses, test_model, test_special/, test_tokenizer); `tests/expected_skips.yaml` unchanged since Phase 2 (last commit 8656018) |
| 8 | PRs to both dev and main trigger the gated coverage job (GATE-05) | ✓ VERIFIED (upgraded) | dev: live via probe PR #39 (truth 6). main: structural — `pull_request` block `[main, master, dev]` inherited by coverage-gate — AND now enforced: `gh api .../branches/{dev,main}/protection` (re-run this pass) returns `required_status_checks.contexts = ["coverage-gate (py3.12, fast leg)"]` on BOTH branches. The prior pass's owner follow-up (04-02 FA-GATE-05) is done: a red coverage-gate check now blocks merges on dev and main |
| 9 | Codecov uploader step and orphaned coverage.xml export removed; no dead/failing reporting step; permissions stay contents: read (GATE-03) | ✓ VERIFIED | `grep -ci codecov ci.yml` → 0; `grep -c "coverage xml -o" ci.yml` → 0; permissions `{contents: read}`. Reporting is terminal + junit; both census jobs run the fail-closed skip audit |
| 10 | Workflows README documents both new jobs, the enforced 90 floor, and scoped-run `--no-cov` guidance (WR-05 minimal-touch) | ✓ VERIFIED | README at HEAD: `### 5. Coverage Gate Job (coverage-gate)` (line 89) and `### 6. Nightly Coverage Job (coverage-nightly)` (line 104) naming jobs exactly as their `name:` fields; enforced-floor statement (lines 39, 91, 136: "identical everywhere"); nightly 03:00 UTC (line 15); `--no-cov` guidance (lines 213-214); slow-test claims corrected (lines 130, 244). The one "codecov" mention (line 135) documents its ABSENCE ("no XML coverage report or codecov upload is produced in CI") — the WR-04 fix (871e616), accurate, not stale |

**Score:** 10/10 truths verified. All behavior-dependent truths (5, 6, 8) carry live end-to-end CI evidence re-observed this pass — zero PRESENT_BEHAVIOR_UNVERIFIED.

### Wave-1 Input Truths (04-01 must-haves)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| W1 | Local full census exits 0 above the 90 floor | ✓ VERIFIED | Superseded by the stronger live form: nightly CI runs the census command of record (slow included) green at post-swap headSha 95c9ba0 — 96.30%, 1656 passed (truth 5) |
| W2 | Synthetic drop exits 1 with the fail-under line | ✓ VERIFIED | Superseded by the stronger live form: real CI red at 78.91% with the verbatim lines (truth 6) |
| W3 | models.lock has exactly 9 provenance-commented entries (2 hf + 6 ms + 1 dataset) | ✓ VERIFIED — with ⚠️ warning | awk counts at HEAD: 9 total / 2 hf / 6 ms / 1 dataset; anchors present; H3K27 absent. **Warning:** one entry is now stale — see Anti-Patterns table (models.lock row) |
| W4 | 7 per-test timeout marks (3×7200 + 4×3600) override the global 300s; addopts `--timeout=300` and markers list untouched | ✓ VERIFIED | trainer file: 6 marks (3×7200 lines 54/313/703, 3×3600 lines 406/485/560); inference file: 1×3600 (line 460); addopts `--timeout=300` intact; markers list has no timeout entry. Suite-wide marks total 3×7200 + 5×3600 + 2×900 + 1×1800 — the extra marks beyond the 7 planned are the documented WR-02 cold-download fix round (e8c8053); 04-REVIEW.md independently re-derived the arithmetic (840 min sum < 900 min nightly kill). Accurate |

### Prohibitions (04-03 must_haves.prohibitions — judgment-tier)

| Prohibition | Status | Evidence |
|-------------|--------|----------|
| `fail_under` leaves the phase at exactly 90 — never raised/lowered | ✓ UPHELD | Deterministic: tomllib `fail_under: 90` at HEAD; no other threshold constant anywhere (workflow grep zero) |
| No test deleted/skipped/excluded on any landing branch to make a gated run green; probe branch never merged | ✓ UPHELD | Deterministic components: PR #39 CLOSED/mergedAt null (probe run re-checked, truth 6-7); `tests/models/` restored on dev; skip allowlist unchanged since Phase 2 (8656018) and the fail-closed skip audit ran green in BOTH live census jobs this pass (7 allowlisted skips, 0 unexpected). The "solely to make green" intent-reading is LLM judgment — non-authoritative, human review optional (no test on any landing branch changed exclusions in this phase's commits) |
| No local/CI gate fork — no CI-only threshold flags, no duplicated constants, no custom comparison scripts | ✓ UPHELD | Deterministic: zero threshold literals in ci.yml; both census steps are bare `--cov` pytest invocations; single enforcement source is pyproject `[tool.coverage.report]` |

Prohibition verdicts: the observable core of each is machine-evidenced (tomllib / grep / gh + live skip audit); the residual intent-reading is LLM judgment (non-authoritative). No `unverified-prohibition` requires blocking disposition — the prior human checkpoint (04-UAT.md, closed complete 2026-10-01, 3/3 pass) covered phase close.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pyproject.toml` | `fail_under = 90` in `[tool.coverage.report]`; transformers bound `<6` (95c9ba0 revert) | ✓ VERIFIED | tomllib: `fail_under: 90`, keys `{fail_under, show_missing}`; omit 7; `--timeout=300`; `transformers>=4.49.0,<6` |
| `models.lock` | 9-entry manifest (2/6/1) keyed by hashFiles | ✓ VERIFIED (⚠️ one stale entry) | Counts 9/2/6/1; consumed by nightly cache key (live-proven: cache restored+saved on green run 36811033498). Stale-entry warning below |
| `.github/workflows/ci.yml` | coverage-gate + coverage-nightly jobs, triggers, guards, uploader removal, self-hosted nightly | ✓ VERIFIED | PyYAML-validated at HEAD; both jobs live-proven (truths 4-6) |
| `.github/workflows/README.md` | documents both jobs + 90 floor + --no-cov | ✓ VERIFIED | Truth 10 |
| `.github/dependabot.yml` | pin rationale for mamba-ssm/causal-conv1d + transformers MambaCache risk (bfe4ee8) | ✓ VERIFIED | Comment block present (lines 16-24); factual accuracy independently re-derived by 04-REVIEW.md against installed transformers 5.17.0 |
| `tests/finetune/test_trainer_real_model.py` | per-test timeout marks + CR-01 fail-closed | ✓ VERIFIED | 6 marks (3×7200/3×3600); included in the green nightly census |
| `tests/inference/test_inference.py` | timeout mark + WR-06 fail-closed | ✓ VERIFIED | `timeout(3600)` line 460; included in the green nightly census |
| `dnallm/mcp/tests/configs/open_chromatin_inference_config.yaml` + `test_mcp_functionality.py` | model swap mamba→dnagpt, binary/num_labels 2 (95c9ba0) | ✓ VERIFIED | Config reads `zhangtaolab/plant-dnagpt-BPE-promoter`, task binary, num_labels 2, swap NOTE in header; exercised green in nightly 36811033498; incremental code review (04-REVIEW.md, 2026-10-01): 0 critical, 0 warning on these files |
| `04-03-SUMMARY.md` | probe evidence + owner hand-off | ✓ VERIFIED | Verbatim fail-under line, PR/run URLs, hand-off sections present |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| pyproject fail_under | coverage-gate / coverage-nightly pytest steps | both run bare `--cov` against the same config | ✓ WIRED | Zero threshold literals in workflow; live-proven both directions (96.30% green / 78.91% red) |
| models.lock | coverage-nightly actions/cache | `hashFiles('models.lock')` over both hub roots | ✓ WIRED | Cache key + paths verified structurally; live-proven on run 36811033498 (first self-hosted run cold-missed, saved post-job) |
| gate/nightly junit | scripts/audit_skips.py | fail-closed skip audit per job | ✓ WIRED | Both audit steps verified; nightly audit emitted `OK: every skip ... matches the allowlist` (live log) |
| timeout marks | pytest-timeout plugin | marker precedence over `--timeout=300` | ✓ WIRED | 3×7200 + 5×3600 + 2×900 + 1×1800 across suite; nightly ran 905.93s census with slow tests surviving past 300s (live proof the marks bind) |
| README | ci.yml job names | documentation matches `name:` fields | ✓ WIRED | Exact-name match incl. the required-check context now configured on branch protection |

### Data-Flow Trace (Level 4)

Not a data-rendering phase — Level 4 maps to exit-code flow: measured coverage total → pytest-cov `fail_under` comparison → pytest rc → step/job conclusion → PR check. Traced end-to-end in BOTH directions with live artifacts re-fetched this pass: green (run 36811033498, `Required test coverage of 90.0% reached. Total coverage: 96.30%`, job success) and red (job 110005226448, `Coverage failure: total of 79 is less than fail-under=90` with 1261 tests passing — the floor, not a test failure, produced the red conclusion). Plus the enforcement terminal: branch protection now consumes the check context on dev and main. Status: ✓ FLOWING.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Nightly census green (post-swap source) | `gh run view 36811033498 --json status,conclusion,event,headSha` | `completed/success`, workflow_dispatch, headSha 95c9ba0 | ✓ PASS |
| Nightly verbatim coverage lines | `gh api .../jobs/110205918141/logs` + grep | `Required test coverage of 90.0% reached. Total coverage: 96.30%`; `1656 passed, 7 skipped ... 905.93s`; skip audit `OK` | ✓ PASS |
| Nightly event isolation | run 36811033498 jobs JSON | exactly 1 non-skipped job (coverage-nightly success); 5 skipped | ✓ PASS |
| Gate red on coverage drop (GATE-04) | `gh api .../jobs/110005226448/logs` + grep | verbatim `Coverage failure: total of 79 is less than fail-under=90`; `78.91%`; `1261 passed` | ✓ PASS |
| Gate green on recent dev pushes | `gh run list --workflow CI --branch dev --event push` | 95c9ba0 / bfe4ee8 / 4a9672a runs all success | ✓ PASS |
| Branch protection (GATE-05 enforcement terminal) | `gh api .../branches/{dev,main}/protection` | required context `coverage-gate (py3.12, fast leg)` on BOTH branches | ✓ PASS |
| fail_under value + config purity | tomllib parse of pyproject.toml | `90`; omit 7; addopts `--timeout=300`; no markers-list entry | ✓ PASS |
| Probe residue | `git ls-remote origin 'refs/heads/*probe*'` | empty; tests/models intact (5 entries) | ✓ PASS |

### Probe Execution

No `scripts/*/tests/probe-*.sh` convention in this repo (verified: `find scripts -path '*/tests/probe-*.sh'` → 0). The phase's probe is the GATE-04 ephemeral PR — executed and live-verified above (truths 6-7). PASS.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| GATE-01 | 04-01 | `fail_under = 90` enforced through pytest exit code, identical local/CI | ✓ SATISFIED | Truths 1-3 |
| GATE-02 | 04-01, 04-02 | Dedicated slow-inclusive coverage CI job — **amended**: gated fast-leg PR job + nightly slow census (self-hosted runner), same floor, models.lock cache, timeout marks/backstop | ✓ SATISFIED (amended) | Truths 4-5, W3, W4 — nightly green re-observed live |
| GATE-03 | 04-02 | Dead codecov v3 step fixed or removed | ✓ SATISFIED | Truth 9 (removed; owner disposition) |
| GATE-04 | 04-03 | Synthetic regression provably fails CI | ✓ SATISFIED | Truths 6-7 — red evidence re-fetched verbatim this pass |
| GATE-05 | 04-02, 04-03 | Gate triggers on PRs to dev and main | ✓ SATISFIED (enforced) | Truth 8 — dev live; main structural + required check now configured on both branches |

Orphaned requirements: none — REQUIREMENTS.md maps exactly GATE-01..05 to Phase 4, all claimed by plans (04-01: GATE-01/02; 04-02: GATE-02/03/05; 04-03: GATE-04/05), all marked Complete.

### Decision Coverage

`check.decision-coverage-verify`: skipped — no trackable `<decisions>` entries in 04-CONTEXT.md. Manual check of the pre-locked decisions: ratchet at 90 not 96 (tomllib), slow tests under the same ratchet in the nightly lane (live), codecov removed not bumped (grep), models.lock manifest created (artifact), WR-02/05/06 surfaced with recorded rationale (04-03-SUMMARY owner hand-off §c). All honored.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `models.lock` | 8 | Stale entry: `ms zhangtaolab/plant-dnamamba-BPE-open_chromatin` provenance names `open_chromatin_inference_config.yaml`, but 95c9ba0 swapped that config to `plant-dnagpt-BPE-promoter` without updating models.lock (last lock commit: 599dd46) | ⚠️ Warning | Data hygiene only — no functional impact: the lock is a hash input, not a download list; the dnagpt-promoter artifact is already covered by two other entries; the post-swap nightly ran green, proving the cache covers the fetched set. Suggested fix: replace the stale entry (or repoint its comment) in a one-line follow-up |
| `04-VERIFICATION.md` addendum / `04-UAT.md` | - | Factual attribution error: "models.lock key changed with the dnagpt swap" — models.lock was never modified; the ~1h post-census cache re-save on run 36811033498 is explained by the NEW self-hosted runner's first successful cache write (hosted calibration run died before ever saving) | ℹ️ Info | Observable outcome (green run) unaffected; corrects the record for the maintainer reading the closeout docs |
| `.github/workflows/ci.yml` | 410 | deploy job pins deprecated `actions/cache@v3` (all other uses @v4) — pre-existing, outside this cycle's delta (IN-07, open in 04-REVIEW-DISPOSITION.md) | ℹ️ Info | Latent: deploy runs only on push to main/master; bump to @v4 when convenient |

Debt markers (TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER): zero hits across all phase-modified files. `pragma: no cover` count in `dnallm/`: exactly 3 (transformers_compat.py) — Phase-3 baseline held.

Open info-tier code-review findings IN-01..IN-07 are tracked in 04-REVIEW-DISPOSITION.md (7 open, all info severity; 0 critical / 0 warning outstanding after the fresh incremental review of the post-verification delta, 2026-10-01). None block the phase goal.

### Advisory (New Scope, Unevidenced)

None — no unevidenced new-scope findings this pass. (The two documentation/data-hygiene findings above carry deterministic evidence and are recorded as Warning/Info, not advisory.)

### Human Verification Required

N/A — infrastructure/foundation phase (CI enforcement) with no user-facing elements. All three items deferred at the initial pass are resolved and recorded in 04-UAT.md (status: complete, 3/3 pass), and each was independently re-confirmed live during this re-verification:

1. **Nightly green completion** — run 36811033498 success (re-observed this pass; verbatim log lines re-extracted).
2. **Branch protection** — required check live on dev AND main (re-queried via API this pass).
3. **Hung probe job** — run 36749810723 settled completed/failure (re-queried this pass).

No ⚠️ PRESENT_BEHAVIOR_UNVERIFIED truths — every behavior-dependent truth has live end-to-end CI evidence in both directions.

### Gaps Summary

No failed must-haves; no gaps. The phase goal — coverage cannot regress, gate live green, provably red on a drop — holds at current HEAD with independently re-observed live evidence: the gate enforced 96.30% green on the post-model-swap source (run 36811033498 at headSha 95c9ba0, full slow census, skip audit clean) and 78.91% red on the probe PR with the verbatim `Coverage failure: total of 79 is less than fail-under=90` line re-fetched from the job log this pass. The stale-digest source changes (model swap, transformers bound revert, dependabot docs, self-hosted nightly runner) were re-verified structurally and passed an incremental code review with 0 critical / 0 warning; the only follow-ups are info/warning-tier hygiene items recorded above (stale models.lock entry, closeout-doc attribution error, IN-01..IN-07) — none affect enforcement. Branch protection now makes the gate a required check on both dev and main, completing the last owner follow-up. Status: passed.

---

_Verified: 2026-10-01T05:15:27Z_
_Verifier: Claude (gsd-verifier)_
_Re-verification of: 2026-09-30T18:58:00Z pass (trigger: stale covered_digest after post-verification source changes)_
