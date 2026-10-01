---
phase: 04-ci-gate-enforcement
verified: 2026-10-01T12:40:00Z
status: passed
score: 10/10 must-haves verified
covered_files:
  - .planning/phases/04-ci-gate-enforcement/04-01-PLAN.md
  - .planning/phases/04-ci-gate-enforcement/04-02-PLAN.md
  - .planning/phases/04-ci-gate-enforcement/04-03-PLAN.md
  - .planning/phases/04-ci-gate-enforcement/04-01-SUMMARY.md
  - .planning/phases/04-ci-gate-enforcement/04-02-SUMMARY.md
  - .planning/phases/04-ci-gate-enforcement/04-03-SUMMARY.md
  - .github/dependabot.yml
  - .github/workflows/README.md
  - .github/workflows/ci.yml
  - dnallm/mcp/tests/configs/open_chromatin_inference_config.yaml
  - dnallm/mcp/tests/test_mcp_functionality.py
  - models.lock
  - pyproject.toml
  - tests/finetune/test_trainer_real_model.py
  - tests/inference/test_inference.py
covered_digest: "v2:sha256:14bd231748ae2869edd68f3342618884dd44d2e342fff4bc7fc731cf773aad42"
behavior_unverified: 0 # all behavior-dependent truths carry live CI evidence in both directions (green re-extracted at 36821471332 this pass / red re-fetched verbatim from job 110005226448 this pass), plus a fresh local census at source-identical HEAD exiting 0 under the same fail_under
overrides_applied: 0
re_verification:
  previous_status: passed
  previous_score: 10/10
  trigger: stale covered_digest — post-verification changes to covered files (3797794 windows leg, 0d5a831+8151c09 test-mamba moved to nightly self-hosted lane with continue-on-error removed, de4b5cc CR-03 .[base] install fix, README accuracy round; plus phase-01/03 fix-round source changes 882d211/42ada4f/2dde7c5 covered by this phase's pyproject/census scope)
  gaps_closed: []
  gaps_remaining: []
  regressions: []
---

# Phase 4: CI Gate Enforcement Verification Report

**Phase Goal:** Coverage cannot regress — the gate goes live green and provably fails CI when coverage drops
**Verified:** 2026-10-01T12:40:00Z
**Status:** passed
**Re-verification:** Yes — stale-digest refresh at HEAD f58afc6 (prior pass 2026-10-01T05:15:27Z scored 10/10 at HEAD 3c60673, pre-dating the test-mamba nightly-lane move, the windows leg, WR-01, and CR-03's fix)

**GATE-02 scope note:** GATE-02 was AMENDED by owner decision 2026-09-30 (04-CONTEXT.md): the gated PR job runs the fast leg (`-m "not slow"`); the slow-inclusive census runs as a scheduled nightly under the same `fail_under=90` on the self-hosted `dnallm-nightly` runner. The shape evolved FURTHER after the prior pass, all owner-directed and code-reviewed in 04-REVIEW.md: a `test-windows` (py3.12, push/PR) fast leg was added (3797794), and `test-mamba` moved from the push/PR matrix to the nightly lane on the same self-hosted GPU box (0d5a831), with its `continue-on-error` mask removed (8151c09, WR-01) and its install fixed to `.[base]` (de4b5cc, CR-03). `deploy.needs` correctly dropped test-mamba (dependents of a skipped needs job would silently stop push deploys — assessed "Verified sound" in 04-REVIEW.md). Verification applies the amended, current shape.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence (re-verified at HEAD f58afc6 / live CI, all re-fetched this pass) |
|---|-------|--------|--------------------------------------------------------------------------|
| 1 | `fail_under = 90` active in `[tool.coverage.report]`, enforced through the pytest exit code; identical command locally and in CI; no local/CI threshold fork (GATE-01) | ✓ VERIFIED | tomllib read at HEAD: `fail_under: 90`, report keys exactly `{fail_under, show_missing}`, run.omit 7 entries, addopts `--timeout=300` intact, no timeout entry in markers. `grep -nE "fail_under\|fail-under\|--cov-fail\|codecov\|coverage xml" ci.yml` → zero hits; both census steps run bare `--cov` (commands re-read verbatim: gate `-m "not slow" ... --junitxml=pytest-junit-gate.xml --cov`; nightly `-ra --durations=0 --junitxml=pytest-junit-nightly.xml --cov`). Behavioral both directions live (truths 5-6) |
| 2 | The enforced fast-leg command passes at current source state | ✓ VERIFIED | Live CI: push run **36847288136** (34037a4) concludes **success** — coverage-gate success, all 6 matrix legs success, test-cuda 2 legs success, **test-windows (py3.12) success**, nightly/mamba/deploy skipped (event isolation). Per-commit `git show --stat` over 34037a4..HEAD (14 commits): 13 are `.planning/`-docs-only; the one exception (de4b5cc) touches only the test-mamba install step + one README line — **coverage-gate/coverage-nightly job definitions and the entire `dnallm/`+`tests/` source are byte-identical between 34037a4 and HEAD** (verified by git diff), so the green run covers the current source state. Corroboration: local fast-leg regression green tonight (1636 passed / 1 skipped / 27 deselected, 89s, three runs) and this pass's own collection census (1637/1664 collected, 27 deselected) matches that arithmetic exactly |
| 3 | The gate is live and green on dev (push) | ✓ VERIFIED | Run 36847288136 (push, dev, 34037a4) success with the coverage-gate job green; the branch-protection required check (truth 8) consumes exactly this job's check context |
| 4 | Amended two-job CI shape exists structurally at HEAD: coverage-gate (push/PR `[main, master, dev]`, fast leg, py3.12, numpy 2.2.0, timeout 90, ubuntu-latest) + coverage-nightly (cron `0 3 * * *` + workflow_dispatch, full census, same bare `--cov`, models.lock-keyed dual-hub cache, timeout 900, self-hosted `[self-hosted, dnallm-nightly]`, `continue-on-error: False`); test/test-cuda/test-windows push/PR-guarded; test-mamba schedule/dispatch-only on the self-hosted box (evolved nightly leg); deploy push-to-main/master-only with needs `[test, test-cuda]` | ✓ VERIFIED | PyYAML parse at HEAD: triggers push+PR `[main, master, dev]` + workflow_dispatch + cron `0 3 * * *`; jobs `[test, test-windows, test-cuda, test-mamba, coverage-gate, coverage-nightly, deploy]`; both census commands verbatim (above); cache key `${{ runner.os }}-models-${{ hashFiles('models.lock') }}` over BOTH hub roots (`~/.cache/huggingface/hub`, `~/.cache/modelscope/hub`); numpy `2.2.0` pinned in both coverage jobs (lines 391/475); permissions `contents: read`; 4× `runner.environment == 'github-hosted'` free-disk guards (lines 37/214/355/429); nightly `continue-on-error: False`; event guards as stated. Live event isolation observed on both sides this pass: push run 36847288136 (nightly/mamba/deploy skipped) and dispatch run 36821471332 (test/test-cuda/test-windows/coverage-gate/deploy skipped; coverage-nightly + test-mamba ran) |
| 5 | The nightly job completes a green slow-inclusive census end to end | ✓ VERIFIED | **Live, re-extracted this pass** from dispatch run **36821471332** (0d5a831, self-hosted) coverage-nightly job log: `Required test coverage of 90.0% reached. Total coverage: 96.30%`, `1656 passed, 7 skipped, 8 warnings in 962.22s`, `OK: every skip in pytest-junit-nightly.xml matches the allowlist`. The coverage-nightly job definition is unchanged 0d5a831→HEAD (de4b5cc touched only test-mamba; the only other ci.yml delta is the added test-windows job). Currency of the census at current source: the phase-03 verifier's fresh local census at HEAD (source-identical for the suite — cf90de9..f58afc6 is docs + test-mamba-leg-only) ran the identical census command of record with `--cov` under the same pyproject `fail_under=90` and **exited 0** — 1657 passed / 7 skipped / 96.30% (7,133/7,407) in 926.87s; the +1 test vs CI is the multilabel regression test added by fix 42ada4f/2dde7c5. Both legs' arithmetic cross-checks against this pass's own collection census (1664 total; fast leg 1637 collected / 27 deselected) |
| 6 | A real coverage-dropping PR to dev produces a FAILED coverage-gate check with the verbatim fail-under line in the CI log (GATE-04, end-to-end exercise of the Phase-1 exit-code fix) | ✓ VERIFIED | **Live, re-fetched verbatim this pass** via `gh run view --job 110005226448 --log-failed` (coverage-gate job of probe run 36749810723, conclusion failure): `ERROR: Coverage failure: total of 79 is less than fail-under=90`, `FAIL Required test coverage of 90.0% not reached. Total coverage: 78.91%`, `1261 passed, 1 skipped, 25 deselected` — all tests green, only the floor made the job red |
| 7 | Zero probe residue | ✓ VERIFIED | Re-checked this pass: `git ls-remote origin 'refs/heads/*probe*'` → empty; PR #39 `state: CLOSED`, `mergedAt: null`, base dev; `tests/models/` intact on dev at HEAD (test_head, test_losses, test_model, test_special/, test_tokenizer); `tests/expected_skips.yaml` unchanged since Phase 2 (last commit 8656018); skip audits green in both live census runs |
| 8 | PRs to both dev and main trigger the gated coverage job (GATE-05) | ✓ VERIFIED | dev: live via probe PR #39 (truth 6). main: structural — `pull_request` block `[main, master, dev]` inherited by coverage-gate — AND enforced: `gh api .../branches/{dev,main}/protection` re-queried live this pass returns `required_status_checks.contexts = ["coverage-gate (py3.12, fast leg)"]` on BOTH branches |
| 9 | Codecov uploader step and orphaned coverage.xml export removed; no dead/failing reporting step; permissions stay contents: read (GATE-03) | ✓ VERIFIED | `grep -ci codecov ci.yml` → 0; `grep -c "coverage xml -o" ci.yml` → 0; permissions `{contents: read}`. Reporting is terminal + junit; both census jobs run the fail-closed skip audit; the one README "codecov" mention (line 143) documents its ABSENCE |
| 10 | Workflows README documents the gate/nightly jobs, the enforced 90 floor, and scoped-run `--no-cov` guidance (WR-05 minimal-touch) | ✓ VERIFIED | README at HEAD: `### 5. Coverage Gate Job (coverage-gate)` and `### 6. Nightly Coverage Job (coverage-nightly)` naming jobs exactly as their `name:` fields; ALSO now documents the evolved shape accurately — windows leg (§2), test-mamba as nightly self-hosted lane with the `.[base]` install rationale (§4 line 78, updated by de4b5cc), deploy `needs: [test, test-cuda]` with the test-mamba exclusion rationale (line 116), 03:00 UTC schedule triggering nightly census + mamba kernel leg (line 15), 27/21/6 census scope (line 100). Enforced-floor statement lines 37, 83, 93, 144 ("identical everywhere"); `--no-cov` guidance lines 221-223 |

**Score:** 10/10 truths verified. All behavior-dependent truths (2, 3, 5, 6, 8) carry live end-to-end CI evidence re-observed this pass — zero PRESENT_BEHAVIOR_UNVERIFIED.

### Wave-1 Input Truths (04-01 must-haves)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| W1 | Local full census exits 0 above the 90 floor | ✓ VERIFIED | Superseded by the stronger forms: CI full census green at 0d5a831 (96.30%, verbatim lines this pass) + fresh local census at source-identical HEAD exiting 0 at 96.30% (phase-03 verifier, command of record) |
| W2 | Synthetic drop exits 1 with the fail-under line | ✓ VERIFIED | Superseded by the stronger live form: real CI red at 78.91% with the verbatim lines re-fetched this pass (truth 6) |
| W3 | models.lock has exactly 9 provenance-commented entries (2 hf + 6 ms + 1 dataset) | ✓ VERIFIED — with ⚠️ warning | awk counts at HEAD: 9 total / 2 hf / 6 ms / 1 dataset; unchanged since 599dd46; consumed by the nightly cache key. **Warning:** the `ms plant-dnamamba-BPE-open_chromatin` entry's provenance comment is stale (the open_chromatin config was swapped to `plant-dnagpt-BPE-promoter` at 95c9ba0) — see Anti-Patterns table |
| W4 | 7 per-test timeout marks (3×7200 + 4×3600) override the global 300s; addopts `--timeout=300` and markers list untouched | ✓ VERIFIED | Re-grepped at HEAD: trainer file 6 marks (3×7200 lines 54/313/703, 3×3600 lines 406/485/560) + inference 1×3600 (line 460); suite-wide 3×7200 + 5×3600 + 2×900 + 1×1800 (the extras are the documented WR-02 cold-download round); addopts `--timeout=300` intact; markers list has no timeout entry. New tests since the marks landed added none |

### Prohibitions (04-03 must_haves.prohibitions — judgment-tier)

| Prohibition | Status | Evidence (re-derived at HEAD this pass) |
|-------------|--------|------------------------------------------|
| `fail_under` leaves the phase at exactly 90 — never raised/lowered | ✓ UPHELD | Deterministic: tomllib `fail_under: 90`; zero threshold literals anywhere in ci.yml; no second threshold constant exists |
| No test deleted/skipped/excluded on any landing branch to make a gated run green; probe branch never merged | ✓ UPHELD | Deterministic components: PR #39 CLOSED/mergedAt null (re-queried); `tests/models/` restored on dev; skip allowlist unchanged since 8656018; fail-closed skip audit green in both live census runs (verbatim `OK:` line re-extracted this pass); post-phase fix rounds only ADDED tests (1656→1657). The "solely to make green" intent-reading is LLM judgment — non-authoritative; the phase's human checkpoint (04-UAT.md, closed complete, 3/3 pass) covered phase close |
| No local/CI gate fork — no CI-only threshold flags, no duplicated constants, no custom comparison scripts | ✓ UPHELD | Deterministic: zero threshold literals in ci.yml; both census steps are bare `--cov` pytest invocations; single enforcement source is pyproject `[tool.coverage.report]` |

Prohibition verdicts: the observable core of each is machine-evidenced (tomllib / grep / gh + live skip audit); the residual intent-reading is LLM judgment (non-authoritative). No `unverified-prohibition` requires blocking disposition.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pyproject.toml` | `fail_under = 90` in `[tool.coverage.report]`; transformers `<6`; `base` extras superset | ✓ VERIFIED | tomllib: `fail_under: 90`, keys `{fail_under, show_missing}`, omit 7; `transformers>=4.49.0,<6`; `base = ["dnallm[dev,test,notebook,mcp]", ...]` — the CR-03 derivation's superset claim confirmed structurally |
| `models.lock` | 9-entry manifest (2/6/1) keyed by hashFiles | ✓ VERIFIED (⚠️ one stale entry) | Counts 9/2/6/1 at HEAD; cache key consumes it (nightly live-proven green post-swap at 0d5a831 — cache covers the fetched set) |
| `.github/workflows/ci.yml` | coverage-gate + coverage-nightly jobs, triggers, guards, uploader removal, self-hosted nightly | ✓ VERIFIED | PyYAML-validated at HEAD; both jobs live-proven (truths 4-6); evolved shape documented in README |
| `.github/workflows/README.md` | documents gate + nightly jobs, 90 floor, --no-cov | ✓ VERIFIED | Truth 10 — includes the evolved windows/mamba/deploy shape accurately |
| `.github/dependabot.yml` | pin rationale (bfe4ee8) | ✓ VERIFIED | Unchanged this delta; accuracy re-derived by 04-REVIEW.md against installed transformers 5.17.0 |
| `tests/finetune/test_trainer_real_model.py` | per-test timeout marks | ✓ VERIFIED | 6 marks (3×7200/3×3600); included in green census (local HEAD 1657-pass run) |
| `tests/inference/test_inference.py` | timeout mark | ✓ VERIFIED | `timeout(3600)` line 460; included in green census |
| `dnallm/mcp/tests/configs/open_chromatin_inference_config.yaml` + `test_mcp_functionality.py` | model swap mamba→dnagpt (95c9ba0) | ✓ VERIFIED | Config reads `zhangtaolab/plant-dnagpt-BPE-promoter` with swap NOTE; exercised green in nightlies at 95c9ba0 and 0d5a831; 04-REVIEW.md incremental review: 0 critical / 0 warning on these files |
| `04-03-SUMMARY.md` | probe evidence + owner hand-off | ✓ VERIFIED | Verbatim fail-under line, PR/run URLs, hand-off sections present; probe evidence independently re-confirmed live (truth 6) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| pyproject fail_under | coverage-gate / coverage-nightly pytest steps | both run bare `--cov` against the same config | ✓ WIRED | Zero threshold literals in workflow; live-proven both directions re-fetched this pass (96.30% green at 0d5a831 / 78.91% red at probe) |
| models.lock | coverage-nightly actions/cache | `hashFiles('models.lock')` over both hub roots | ✓ WIRED | Key + paths verified at HEAD; nightly green post-swap proves the cache covers the fetched set |
| gate/nightly junit | scripts/audit_skips.py | fail-closed skip audit per job | ✓ WIRED | Both audit steps present; nightly audit `OK:` line re-extracted live this pass |
| timeout marks | pytest-timeout plugin | marker precedence over `--timeout=300` | ✓ WIRED | Marks re-counted at HEAD; slow census ran 962.22s in CI with slow tests surviving far past 300s |
| README | ci.yml job names | documentation matches `name:` fields | ✓ WIRED | Exact-name match including the required-check context configured on branch protection |

### Data-Flow Trace (Level 4)

Not a data-rendering phase — Level 4 maps to exit-code flow: measured coverage total → pytest-cov `fail_under` comparison → pytest rc → step/job conclusion → PR check → branch protection. Traced end-to-end in BOTH directions with live artifacts re-fetched this pass: green (run 36821471332, `Required test coverage of 90.0% reached. Total coverage: 96.30%`, job success) and red (job 110005226448, `Coverage failure: total of 79 is less than fail-under=90` with 1261 tests passing — the floor, not a test failure, produced the red conclusion), terminating in the required check on dev and main. Status: ✓ FLOWING.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Latest dev push all-green incl. windows leg | `gh run view 36847288136` | completed/success; coverage-gate + test-windows + 6 matrix legs + 2 cuda legs green; nightly/mamba/deploy skipped | ✓ PASS |
| Nightly census green (slow included, self-hosted) | `gh run view --job <nightly> --log` on 36821471332 | `96.30%`, `1656 passed, 7 skipped`, skip audit `OK` (verbatim) | ✓ PASS |
| Dispatch event isolation (evolved shape) | 36821471332 jobs JSON | coverage-nightly + test-mamba ran; test/test-cuda/test-windows/coverage-gate/deploy all skipped | ✓ PASS |
| Gate red on coverage drop (GATE-04) | `gh run view --job 110005226448 --log-failed` | verbatim `Coverage failure: total of 79 is less than fail-under=90`; `78.91%`; `1261 passed` | ✓ PASS |
| Branch protection (GATE-05 terminal) | `gh api .../branches/{dev,main}/protection` | required context `coverage-gate (py3.12, fast leg)` on BOTH branches | ✓ PASS |
| fail_under value + config purity | tomllib parse of pyproject.toml | `90`; omit 7; `--timeout=300`; no markers-list entry | ✓ PASS |
| Probe residue | `git ls-remote origin 'refs/heads/*probe*'` + `gh pr view 39` | empty; PR CLOSED/mergedAt null; tests/models intact | ✓ PASS |
| Census arithmetic consistency (this pass's own run) | `pytest --collect-only -q` (+ `-m "not slow"`) | 1664 total; fast leg 1637 collected / 27 deselected — matches live CI fast leg (1636+1+27) and HEAD census (1657+7) exactly | ✓ PASS |
| CR-03 masked-red basis | `gh run view --job 110237767158 --log` | `3 failed, 1580 passed, 1 skipped, 20 deselected` with notebook-import failures — confirms the finding the de4b5cc fix addresses | ✓ PASS |
| CR-03 install-chain derivation | tomllib extras + installed dist metadata | `base = dnallm[dev,test,notebook,mcp]` (strict superset of test,dev + mcp extra); `fastmcp-slim`/`pydantic-ai-slim` require `exceptiongroup`; `tests/mcp/test_client_sdk.py` hard-imports it | ✓ PASS |

### Probe Execution

No `scripts/*/tests/probe-*.sh` convention in this repo. The phase's probe is the GATE-04 ephemeral PR — executed, and its red evidence re-fetched verbatim this pass (truths 6-7). PASS.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| GATE-01 | 04-01 | `fail_under = 90` enforced through pytest exit code, identical local/CI | ✓ SATISFIED | Truths 1-3 |
| GATE-02 | 04-01, 04-02 | Dedicated slow-inclusive coverage CI job — **amended** (fast gated PR leg + nightly slow census on self-hosted runner), same floor, models.lock cache, timeout marks/backstop | ✓ SATISFIED (amended) | Truths 4-5, W3, W4 |
| GATE-03 | 04-02 | Dead codecov v3 step fixed or removed | ✓ SATISFIED | Truth 9 (removed; owner disposition) |
| GATE-04 | 04-03 | Synthetic regression provably fails CI | ✓ SATISFIED | Truths 6-7 — red evidence re-fetched verbatim this pass |
| GATE-05 | 04-02, 04-03 | Gate triggers on PRs to dev and main | ✓ SATISFIED (enforced) | Truth 8 — dev live; main structural + required check live on both branches |

Orphaned requirements: none — REQUIREMENTS.md maps exactly GATE-01..05 to Phase 4, all claimed by plans (04-01: GATE-01/02; 04-02: GATE-02/03/05; 04-03: GATE-04/05), all rows marked Complete.

### Decision Coverage

No trackable `<decisions>` entries in 04-CONTEXT.md. Manual re-check of the pre-locked decisions at HEAD: ratchet at 90 not 96 (tomllib), slow tests under the same ratchet in the nightly lane (live), codecov removed not bumped (grep), models.lock manifest created (artifact), WR-02/05/06 dispositions recorded (04-03-SUMMARY owner hand-off). All honored.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `.github/workflows/ci.yml` (test-mamba job) | ~308-325 | **Pending live rehearsal**: the CR-03 fix (de4b5cc, install `.[base]` instead of `.[test,dev]`) is structurally derived and the masked-red basis is live-logged (job 110237767158: 3 failed / 1580 passed on missing mcp-extra + exceptiongroup deps), but no CI run has executed the fixed install yet — de4b5cc is unpushed (origin/dev at 34037a4) and the next 03:00 UTC schedule is the first unsupervised exercise | ⚠️ Warning | Not a gate lane: the mamba step runs `pytest tests/ -m "not slow"` with NO `--cov`, no required check references it, and its failure cannot affect coverage-nightly's independent caches. Install-chain derivation verified structurally this pass (`base = dnallm[dev,test,notebook,mcp]` superset; `fastmcp-slim`/`pydantic-ai-slim` → `exceptiongroup` carrier confirmed in installed metadata; the failing imports are exactly the mcp-extra set). Also note the same dispatch proved `coverage-nightly` resolves `.[base]` green on this exact box. Owner follow-up recorded in 04-REVIEW-FIX.md ("the one outstanding follow-up the fixer cannot perform") — tonight's schedule closes it |
| `models.lock` | 8 | Stale entry (carried from prior pass, unchanged since 599dd46): `ms plant-dnamamba-BPE-open_chromatin` provenance names the open_chromatin config that 95c9ba0 swapped to `plant-dnagpt-BPE-promoter` | ⚠️ Warning | Data hygiene only — the lock is a hash input, not a download list; the dnagpt-promoter artifact is covered by two other entries; post-swap nightlies ran green. One-line follow-up when convenient |
| `tests/expected_skips.yaml` | SONAME entry | Comment says "non-Linux legs only (CI is linux)" — stale now that a Windows CI leg exists (entry itself matches; audit passes) | ℹ️ Info | IN-10, open in 04-REVIEW-DISPOSITION.md |
| `.github/workflows/ci.yml` | deploy job | `actions/cache@v3` pin (pre-existing) | ℹ️ Info | IN-07, open; deploy runs only on push to main/master |

Debt markers (TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER): zero hits across all covered files (re-scanned this pass). `pragma: no cover` count in `dnallm/`: exactly 3 (transformers_compat.py) — Phase-3 baseline held.

Open info-tier code-review findings IN-01..IN-10 are tracked in 04-REVIEW-DISPOSITION.md (10 open, all info severity; CR-03 fixed with the live-rehearsal follow-up noted above). None block the phase goal.

### Advisory (New Scope, Unevidenced)

None — the one new-scope concern (test-mamba nightly-lane rehearsal) carries deterministic evidence (live masked-red log + structural install-chain derivation verified against installed metadata) and is recorded as a Warning above, not an advisory.

### Human Verification Required

N/A — infrastructure phase with no user-facing elements. The three items deferred at the initial pass are resolved and recorded in 04-UAT.md (status: complete, 3/3 pass), and each terminal fact was independently re-queried live during this re-verification (nightly green census re-extracted; branch protection re-queried on both branches; probe run terminal state re-queried). No ⚠️ PRESENT_BEHAVIOR_UNVERIFIED truths.

### Gaps Summary

No failed must-haves; no gaps. The phase goal — coverage cannot regress, gate live green, provably red on a drop — holds at HEAD f58afc6 with independently re-observed live evidence in both directions: green full census at 96.30% (CI at 0d5a831, verbatim lines re-extracted this pass, job definition unchanged since; plus a fresh local census at source-identical HEAD exiting 0 under the same `fail_under=90` — 1657 passed / 7 skipped / 96.30%) and red at 78.91% on the probe PR (verbatim `Coverage failure: total of 79 is less than fail-under=90` re-fetched from the job log this pass, 1261 tests passing), terminating in the required check on both dev and main. The latest dev push run (36847288136 at 34037a4) is all-green including the new windows leg, and the 14 unpushed commits alter nothing the gate consumes (docs-only, plus the test-mamba install step). The stale-digest deltas were re-verified structurally and passed the incremental code review (CR-03 fixed, 0 critical / 0 warning outstanding). Follow-ups, all non-blocking and honestly recorded: the nightly test-mamba leg's first live rehearsal of the `.[base]` install at tonight's 03:00 UTC schedule (Warning — non-gate lane, derivation verified), the stale models.lock entry, and IN-01..IN-10 info items. Status: passed.

---

_Verified: 2026-10-01T12:40:00Z_
_Verifier: Claude (gsd-verifier)_
_Re-verification of: 2026-10-01T05:15:27Z pass (trigger: stale covered_digest after post-verification covered-file changes — windows leg, test-mamba nightly-lane move + WR-01 + CR-03 fix, README round, phase-01/03 fix-round source changes)_
