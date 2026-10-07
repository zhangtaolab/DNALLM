---
phase: 09-ci-wiring-census-verification
verified: 2026-10-07T00:08:10Z
status: gaps_found
score: 16/18 must-haves verified
covered_files:
  - .github/workflows/README.md
  - .github/workflows/ci.yml
  - .planning/phases/09-ci-wiring-census-verification/09-01-PLAN.md
  - .planning/phases/09-ci-wiring-census-verification/09-01-SUMMARY.md
  - .planning/phases/09-ci-wiring-census-verification/09-02-PLAN.md
  - .planning/phases/09-ci-wiring-census-verification/09-02-SUMMARY.md
  - .planning/phases/09-ci-wiring-census-verification/09-03-PLAN.md
  - .planning/phases/09-ci-wiring-census-verification/09-03-SUMMARY.md
  - .planning/phases/09-ci-wiring-census-verification/09-04-PLAN.md
  - .planning/phases/09-ci-wiring-census-verification/09-04-SUMMARY.md
  - .planning/phases/09-ci-wiring-census-verification/09-CENSUS-ROLLUP.md
  - docs/user_guide/continuous_integration.md
  - mkdocs.yml
  - pyproject.toml
  - scripts/runner/README.md
  - scripts/runner/ollama.service
  - tests/TESTING.md
  - tests/examples/_execution.py
  - tests/examples/test_notebook_execution.py
  - tests/test_models_lock_contracts.py
  - tests/test_runner_infra_contracts.py
covered_digest: "v3:sha256:2b82439c48c6789fb6ebd7d81b6f7b999d14f5882f98c936798f07f1f526e86a"
behavior_unverified: 0
overrides_applied: 2
overrides:
  - must_have: "The live service state after the owner re-apply serves qwen3.8:latest at num_ctx 8192 on loopback only (09-02, D-06 live half)"
    reason: "Owner decisions supersede plan text (recorded in STATE.md 2026-10-06): the num_ctx cut was DEFERRED entirely at 00:52 CST ('ollama 配置先不改了', superseding the 2026-10-05 A/A decision), and the mcp agent model was swapped qwen3.8:latest -> qwen3.5:4b at 15:27 CST (quick task 261006-lhm, commits dc44856/b91f2a6/0a5ca0d/170e86f). The in-repo OLLAMA_CONTEXT_LENGTH=8192 unit pin stays committed (machine-verified) but inert; D-12 comments and 09-CENSUS-ROLLUP.md deviation 0 / D-06 row record the un-cut measured reality (442-449s pre-swap, 87s post-swap). The open sudo re-apply (which alone restores the drifted live OLLAMA_HOST=0.0.0.0 bind, T-09-03) is tracked open-but-not-required-now in 09-USER-SETUP.md. Post-close repairs (000c743, a072f0d) aligned the README/unit narrative to this same standing decision — the override remains accurate at HEAD."
    accepted_by: "owner (Tao Zhang, STATE.md 2026-10-06 entries 00:52 / 15:27 CST)"
    accepted_at: "2026-10-06T07:52:00Z"
  - must_have: "D-02: one baseline census run with ALL cuts applied (giants exit + epochs 1 + num_ctx 8k) rebuilds the authoritative count (09-04)"
    reason: "Owner decision 2026-10-06 00:52 CST voided the num_ctx component BEFORE the baseline dispatch (rollup deviation 0 records the voided Task-2 precondition); the baseline run 37345067326 ran with giants exit + epochs-1 live and num_ctx un-cut by owner direction. The census counts, D-12 measured budgets, and green-gate evidence (37432001711) remain the authoritative record; the num_ctx deferral is the recorded revisit-with-data item."
    accepted_by: "owner (Tao Zhang, STATE.md 2026-10-06 00:52 CST entry)"
    accepted_at: "2026-10-06T07:52:00Z"
re_verification:
  previous_status: passed
  previous_score: 18/18
  gaps_closed: []
  gaps_remaining:
    - "D-03 census ratchet re-pin: stage-0.5 literal 193/202 vs measured 197/206 at HEAD"
  regressions:
    - "Truth 7 (D-03 pin) and the collection clause of Truth 1 (SC1): the phase-05..08 post-close repair rounds added 4 fast tests to tests/examples/test_notebook_execution.py (374e8e6 IN-01, 9d44cbf IN-02, 5dc18c6 CR-01 x2) without re-pinning the census triple; the nightly CI wiring itself is unchanged and the new tests pass, but the next example-nightly dispatch hard-fails at stage 0.5"
gaps:
  - truth: "D-03: the pre-stage-1 HARD census assertion pins the exact selector triple and the pinned literal equals a fresh local measurement byte-for-byte (09-01) — and SC1's 'nightly census run collects and passes' clause at HEAD"
    status: failed
    reason: "Post-close repair commits added 4 fast tests to tests/examples/test_notebook_execution.py without re-pinning the census ratchet. Fresh measurement at HEAD c5ad8b5: '197/206 tests collected (9 deselected)'; ci.yml L862 greps for '193/202 tests collected (9 deselected)' — deterministic mismatch, so the next example-nightly dispatch (cron 30 5 * * *) fails at stage 0.5 (echo FAIL + exit 1) before any test executes. The tests themselves are green (20 contract + 6 seam + 3 new all pass locally; the phase-08 verifier's project-wide fast run covered them)."
    artifacts:
      - path: ".github/workflows/ci.yml"
        issue: "Stage 0.5 D-03 pin (L862-863) expects '193/202 tests collected (9 deselected)'; the test tree now measures '197/206 tests collected (9 deselected)' (4 fast additions: IN-01 non-mapping override, IN-02 venv-probe timeout, 2 TestEvoNotebookContentContracts evo-1 pins)"
    missing:
      - "Re-pin the D-03 grep literal and FAIL echo at ci.yml L862-863 from '193/202 tests collected (9 deselected)' to '197/206 tests collected (9 deselected)'"
      - "Append a post-close addendum to 09-CENSUS-ROLLUP.md recording the new authoritative triple (197/206, 9 deselected) and attributing the +4 to the 05-08 repair commits"
      - "Re-dispatch example-nightly (workflow_dispatch) at the re-pinned HEAD and record the green run id (D-17-style confirmation)"
---

# Phase 9: CI Wiring & Census Verification — Verification Report

**Phase Goal:** The nightly census formally gates the finished execution-test layer — verified end to end on the real runner for collection, skip audit, runtime budget, hygiene steps, lock consistency, and the documented coverage expectation
**Verified:** 2026-10-07T00:08:10Z
**Status:** gaps_found
**Re-verification:** Yes — REGENERATION run (convergence round): the prior report (2026-10-06T14:17:17Z, passed 18/18, digest 111f703d, measured at 7cf8458/fff1c7e) went stale solely because the phase-05/06/07/08 post-close code-review repair rounds legitimately touched files in this phase's covered set. Not a gap-closure re-run — the prior report had no gaps. All evidence re-measured at current HEAD `c5ad8b5` (branch `phs`; one docs-only planning commit landed mid-verification: c5ad8b5 regenerates phase-07's VERIFICATION.md — outside this phase's covered inputs, no measurement affected).

## Delta Since the Stale Report

The covered-input delta is exactly the four later phases' code-review repair rounds (planning docs excluded — inert per #4623):

| File | Commits | Change | Verified how |
|------|---------|--------|--------------|
| `.github/workflows/ci.yml` | c3d74a6 (08 IN-05) | ONE line: docs-deploy mkdocs cache `actions/cache@v3` → `@v4` (L1121). No example-nightly/test-mamba/coverage-nightly step touched | diff read (whole ci.yml delta = 1 line); nightly wiring re-verified structurally below |
| `scripts/runner/README.md` | 000c743, a072f0d (08 WR-01) | Narrative only: standing num_ctx-deferral status block, retired-model sweep, qwen3.5:4b Modelfile re-probe caveat | diff read; re-apply op + drift notice intact |
| `scripts/runner/ollama.service` | 000c743 | Comments only: deferral/inert status, re-probe caveat | diff read; BOTH Environment pins byte-preserved (L45 host, L53 ctx) |
| `tests/examples/_execution.py` | 374e8e6 (05 IN-01), 9d44cbf (05 IN-02), 000c743 | `seed_sandbox` yaml_overrides hardened to fail-closed on non-mapping sections (ValueError); TimeoutExpired in megadna/evo venv probes → typed-skip evidence; comment alignment | diff read; 6/6 seam contracts re-run green at HEAD |
| `tests/examples/test_notebook_execution.py` | d03ab4d (05 WR-02), 374e8e6, 9d44cbf, 5dc18c6 (08 CR-01) | ACTIVE-lane `notebook_sandbox` now forwards spec `yaml_patch` (still sandbox-only); **+4 fast tests** (IN-01 non-mapping, IN-02 timeout contract, 2 evo-1 content pins); evo notebook evo-1 load restored (5dc18c6/ceea260 — dnallm-side, outside covered set) | diff read; 3 new tests re-run green; **collection counts re-measured — see Gap 1** |

Non-covered repair files (phase-05..08 scope: `dnallm/models/*`, `tests/models/*`, `tests/utils/*`, `scripts/check_docs_sync.py`, `models.lock` header, docs/example pages, `publish.yml`) are those phases' covered inputs, verified by their own regeneration reports; none is in this phase's must-have set except via the census collection count below.

**The delta is repair-class as documented — but one repair consequence was NOT handled: the +4 tests moved the census collection triple.** The nightly wiring is unchanged; the D-03 ratchet literal is not.

## Goal Achievement

### Observable Truths

Roadmap Success Criteria (the contract) are rows 1-5; PLAN must-have truths follow. Same 18-truth structure as the prior report (its must-have decomposition remains valid); every row re-verified against the codebase at HEAD `c5ad8b5`.

| # | Truth | Status | Evidence (re-measured at HEAD c5ad8b5) |
|---|-------|--------|----------|
| 1 | SC1: nightly census collects and passes all slow execution tests; `audit_skips.py` green with the typed categories registered; fast leg zero new skips | ✗ FAILED | Skip-audit substance HOLDS: `tests/expected_skips.yaml` typed prefixes present (`network-unavailable:` L29, `environment-unavailable:` L35, `optional-dep:` L40); `audit_skips.py` wired at 5 CI points incl. the stage-4 per-junit loop; `git log e9056c2..HEAD -- tests/expected_skips.yaml` empty; deselected counts unchanged (9 in tests/examples, 53 repo-wide — zero new skips). BUT the "collects and passes" clause fails at HEAD: the census's own stage-0.5 gate now deterministically rejects the (correct) collection measurement — see Gap 1 (root cause shared with truth 7). Run of record 37432001711 (green, pre-repair headSha 170e86f) remains the last end-to-end green census |
| 2 | SC2: measured runtime budgets recorded; separate example-execution nightly job split out | ✓ VERIFIED | example-nightly split-out job with `timeout-minutes: 2700` (L682); D-12 run-id citations 37345067326 / 37377004230 / 37432001711 appear at 11 ci.yml lines and 27 rollup lines; rollup reproduces every measured figure; final run measured 2:09:55 |
| 3 | SC3: hygiene steps observable — kernel pkill + VRAM assertion, sum-of-ceilings review, `if: always()` uploads | ✓ VERIFIED | Both `Stage 1.5/2.5` bodies re-read at HEAD: `pkill -f ipykernel_launcher` (L902/L994/L1043), before/after `LC_ALL=C free -g` logging, WR-02 fail-closed empty-parse guard + hard `-lt 35` + `exit 1` floor in BOTH bodies (L901-915, L993-1007); `if: always()` uploads at L1047/L1071/L1083 with all 5 path members (junit glob, mcp-server logs, stage-results.txt, stage*.log, census-collect.txt); sum-of-ceilings review + KEEP-900 decision recorded in the rollup (unchanged) |
| 4 | SC4: fast-leg models.lock consistency guard fails on drift | ✓ VERIFIED | `tests/test_models_lock_contracts.py` re-run at HEAD: 12 passed (in the 20-passed contract run, 0.43s) incl. the drift-injection exact-set test; file untouched by the repair delta |
| 5 | SC5: coverage expectation documented (kernel subprocesses, does not move the 96.30% gate) | ✓ VERIFIED | `docs/user_guide/continuous_integration.md` L26 (`fail_under = 90`, 96.30% landing), L30 (ratchet semantics), L32 (AUDIT-04 note: kernel subprocesses, "does not move the 96.30% coverage gate"); nav registered mkdocs.yml L66; file untouched by the delta |
| 6 | D-01: giants exit surgical — exactly 1 test marked, fast evo contract tests unmarked | ✓ VERIFIED | `giants:` marker registered in pyproject L521; `_GIANTS_GATED` frozenset (L1371) + `_gated_test_param` (L1376-1418) intact; fresh `-m giants` collection at HEAD: **1/206 collected (205 deselected)** — still exactly one giants-marked test; TestSpecEnvOverrides / TestEvoIsolatedLane unmarked |
| 7 | D-03: pre-stage-1 HARD census assertion pins the exact selector triple; pinned literal equals a fresh measurement byte-for-byte | ✗ FAILED | ci.yml L862 greps `'^=* *193/202 tests collected \(9 deselected\) in '`; fresh measurement at HEAD: `=============== 197/206 tests collected (9 deselected) in 0.91s` — MISMATCH. The 05-08 repair commits added 4 fast tests to tests/examples/test_notebook_execution.py without re-pinning. Next nightly dispatch fails stage 0.5 (FAIL echo + exit 1) before any test runs. **Gap 1** |
| 8 | D-04/D-11: zero evo provisioning and zero models.lock-keyed cache steps remain; neighbors intact | ✓ VERIFIED | Token scan over the example-nightly job (L607-1096): zero hits for `evo-venv`/`wheelhouse-flashattn`/`models-giants`/`evo_torch`/`flash_attn`/`stripedhyena`/`hashFiles('models.lock')`/`Restore model caches`; keeps present (`wheelhouse-mamba`, `megadna-venvs`, `MEGABYTE_pytorch`) |
| 9 | D-19: each nightly job fires on exactly one cron via schedule-string gates; dispatch disjunct kept; push/PR excluded | ✓ VERIFIED | L282 (test-mamba) and L448 (coverage-nightly) gate on `0 3 * * *` + `workflow_dispatch`; L607 (example-nightly) gates on `30 5 * * *` + `workflow_dispatch`; root schedule entries unchanged (L20-21); run 37432001711 observed firing exactly this way |
| 10 | Marker registration atomic — no invocation ever sees an unknown-marker error | ✓ VERIFIED | Fresh at HEAD: repo-wide `pytest tests/ -m "not slow" --collect-only -q` exits 0 — **1882/1935 collected (53 deselected)** (+41 vs the prior report's 1841/1894 = the repair rounds' project-wide test additions; error-free under `--strict-markers` is the claim, and it holds) |
| 11 | D-05: epochs cut is sandbox-only; committed notebook/YAML byte-identical | ✓ VERIFIED | `seed_sandbox(..., yaml_overrides=None)` seam intact and HARDENED by IN-01 (non-mapping section now raises documented ValueError); spec `yaml_patch` key on the finetune_custom_head entry; committed `example/notebooks/finetune_custom_head/finetune_config.yaml` L38 still reads `num_train_epochs: 3`; `git log e9056c2..HEAD --grep="09-0" -- example/` empty; 05-WR-02's ACTIVE-lane yaml_patch forwarding (d03ab4d) remains sandbox-only with `assert_tree_clean()` — strengthens, not weakens, D-05 |
| 12 | D-07: both cut seams proven by kernel-free contract tests | ✓ VERIFIED | Re-ran at HEAD: `TestSeedSandboxYamlOverrides` **6 passed** (5 prior + new IN-01 non-mapping test); `tests/test_runner_infra_contracts.py` 8 green in the 20-passed run (unit pins, README re-apply, SSE-probe http_code shape) |
| 13 | D-06: in-repo ollama unit pins OLLAMA_CONTEXT_LENGTH=8192 beside the byte-preserved loopback pin | ✓ VERIFIED | `scripts/runner/ollama.service` L45 `Environment="OLLAMA_HOST=127.0.0.1:11434"` + L53 `Environment="OLLAMA_CONTEXT_LENGTH=8192"` (line numbers shifted by comment growth only); contract tests pin both (green); 08-WR-01 comment rewrite records the deferral/inert status accurately |
| 14 | D-06 live half: live service serves qwen3.8:latest at num_ctx 8192 on loopback only | ✓ PASSED (override) | Override carried forward unchanged (owner num_ctx deferral STATE.md L180 00:52 CST + model swap L181 15:27 CST — both re-read at HEAD); the repair delta's README/unit narrative rewrite (000c743/a072f0d) documents this same standing decision, so the override text remains accurate; open sudo re-apply tracked in 09-USER-SETUP.md |
| 15 | D-02: one baseline census run with ALL cuts applied rebuilds the authoritative count | ✓ PASSED (override) | Override carried forward unchanged: num_ctx component voided by owner before baseline 37345067326 (rollup deviation 0); giants + epochs-1 were live at baseline; census of record + D-12 budgets remain authoritative — accepted by owner 2026-10-06 |
| 16 | D-17: one complete example-nightly dispatch GREEN under the final wiring | ✓ VERIFIED | gh-corroborated again this pass: run 37432001711 `conclusion: success`, `headBranch: phs`, `headSha: 170e86f91cb...`, completed 2026-10-06T12:52:09Z; example-nightly green in 2h9m55s. Caveat at HEAD: this run predates the repair delta; the nightly wiring it exercised is unchanged (only docs-deploy cache-v4 + comments differ), but a NEW dispatch at HEAD would fail stage 0.5 — Gap 1 |
| 17 | D-18: test-mamba and coverage-nightly re-dispatched green in-phase, durations recorded | ✓ VERIFIED | Same run 37432001711: test-mamba green 10m59s, coverage-nightly green 1h51m6s at 96.42% (above fail_under=90); durations recorded in the rollup and ci.yml comments; STATE.md L185 records the close |
| 18 | D-08/D-09/D-10: ty advisory standalone on coverage-gate; exactly 2 mypy steps; no static check enters pytest | ✓ VERIFIED | `uvx ty@0.0.84 check dnallm/` with `\|\| true` advisory shape at L436; `mypy dnallm/` count in ci.yml = exactly 2; stage 1-3 pytest invocations are pure pytest (re-read at L1020-1079 for stage 3); WR-01 missing-junit ledger line (`stage4-audit-$junit=1` bare-code) present in the stage-4 branch |

**Score:** 16/18 truths verified (14 VERIFIED + 2 PASSED (override); 2 FAILED — truths 1 and 7, one shared root cause, one gap; 0 present-behavior-unverified)

### Deferred Items

None — Phase 9 is the final phase of milestone v1.1; no later-phase coverage exists to defer to. The census re-pin is a real gap (Step 9b conservative rule: no matching later phase).

### Advisory (New Scope, Unevidenced)

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| — | none | — | The regeneration pass ran the Step 7 scan at full scope over all covered files (zero debt markers). The only open findings anywhere remain the five OPEN Info-severity review items (IN-02..IN-06) already recorded in 09-REVIEW-DISPOSITION.md — carried prior findings with a disposition ledger, not new-scope unevidenced blockers. No new-scope finding without deterministic evidence arose. (Note: `check decision-coverage-verify` skipped this pass — the tool reports "CONTEXT.md missing" for `09-CONTEXT.md`; the prior pass verified 20/20 honored, and the repair delta touches no decision-honoring artifact: decisions are CI-wiring, and the ci.yml delta is one docs-deploy cache-version line.) |

### Required Artifacts

All plan-declared artifacts re-checked (exists / substantive / wired) at HEAD. Level 4 data-flow: N/A — CI configuration, contract tests, and docs; no rendered dynamic data.

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pyproject.toml` | giants marker registration | ✓ VERIFIED | L521 markers entry; untouched by the delta (ruff pin still 0.16.10 from a13b50e) |
| `tests/examples/test_notebook_execution.py` | `_GIANTS_GATED` + composable marks + TestSeedSandboxYamlOverrides | ✓ VERIFIED | L1371-1418; contract class now 6 tests (IN-01 addition); 6/6 + 3/3 new tests green at HEAD |
| `.github/workflows/ci.yml` | deselect, D-03 gate, deletions, D-19 gates, hygiene, upload, ty, D-12 budgets | ⚠️ VERIFIED STRUCTURE / STALE PIN | All structural elements re-verified at HEAD incl. WR-01/WR-02 hardening; the delta is one docs-deploy cache line. **The D-03 pin literal is stale against the current test tree — Gap 1** (the substance the pin protects — exact collection accounting — is exactly what it is failing on; the guard works, the literal needs the 197/206 re-pin). gsd-tools `verify.artifacts` "Missing pattern: OLLAMA-free hygiene floor literal 35" remains an authoring-prose artifact, as before |
| `tests/TESTING.md` | giants marker documented | ✓ VERIFIED | Untouched by the delta; both listings intact |
| `.github/workflows/README.md` | post-surgery topology | ✓ VERIFIED | Untouched by the delta; dual cron + num_ctx deferral + qwen3.5:4b language intact |
| `tests/examples/_execution.py` | yaml_overrides + yaml_patch spec key + cell_timeout seam | ✓ VERIFIED | Seam hardened (IN-01 fail-closed non-mapping, IN-02 typed-skip timeouts); spec key + 3600s cell_timeout intact; 6/6 contracts green |
| `scripts/runner/ollama.service` | both Environment pins | ✓ VERIFIED | L45 + L53 byte-preserved; comment block accurately reports deferral/inert status |
| `scripts/runner/README.md` | re-apply op + Why num_ctx + drift notice | ✓ VERIFIED | Re-apply steps + `systemctl show` verify + 0.0.0.0 drift notice ("never document 0.0.0.0") intact under the new deferral narrative |
| `tests/test_runner_infra_contracts.py` | fast pins for unit + probe + swap | ✓ VERIFIED | Untouched by the delta; 8/8 green at HEAD |
| `tests/test_models_lock_contracts.py` | CI-08 guard with drift-injection proof | ✓ VERIFIED | Untouched by the delta; 12/12 green at HEAD |
| `docs/user_guide/continuous_integration.md` | AUDIT-04 note + topology + ratchet | ✓ VERIFIED | Untouched by the delta; L26/L30/L32 verified |
| `mkdocs.yml` | one nav entry | ✓ VERIFIED | L66, single entry |
| `09-CENSUS-ROLLUP.md` | authoritative census record | ✓ VERIFIED (as history) | Run-id lines + deviations + D-20 record intact; records the 193/202 baseline truthfully — needs a 197/206 post-close addendum when Gap 1 is closed |

### Key Link Verification

`gsd-tools verify.key-links` resolves the same 4/10 as the prior report (6 `from:` fields remain descriptive prose — plan-authoring format limitation, not missing wiring; identical to both prior reports). Manual re-verification with direct codebase evidence at HEAD:

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| pyproject markers list | gated parametrize | giants registration + spec-derived application | ✓ WIRED | `giants:` L521; `_gated_test_param` L1376; repo-wide collection error-free (truth 10) |
| test file giants mark | ci.yml stage-1 invocation | `-m "not giants"` AND `-k "not mcp_example"` | ✓ WIRED | Both flags in the stage-1 line; fresh measurement deselects exactly 9 = 8 mcp + 1 giants |
| ci.yml D-03 literal | pytest collect-only triple | grep of the tee'd census-collect.txt | ✗ NOT_WIRED (stale literal) | Pin 193/202 ≠ measured 197/206 — the capture→grep link now rejects a correct measurement. **Gap 1** |
| NOTEBOOK_EXEC_SPECS yaml_patch | seed_sandbox yaml_overrides | spec forwarding | ✓ WIRED | Now forwarded in BOTH lanes (gated L1418 + ACTIVE fixture, d03ab4d); seam applied fail-closed; 6/6 contracts green |
| ollama.service | README + contract test | re-apply docs + fast pins | ✓ WIRED | README op + both pin tests green |
| models.lock rows | example content scanner | org-name extraction + membership assert | ✓ WIRED | 12/12 green incl. drift injection |
| docs page | mkdocs nav + cross-links | nav registration | ✓ WIRED | nav L66; cross-references present |
| D-12 budget comments | rollup measured records | identical run ids | ✓ WIRED | 3 ids in both ci.yml (11 lines) and rollup (27 lines) |
| census-collect.txt capture | D-14 upload path | assertion output in always-upload | ✓ WIRED | In the `if: always()` path list (re-read L1075-1083) |
| D-06 unit pin + owner re-apply | D-02 baseline dispatch | live-cut precondition | ✓ WIRED (superseded form) | Voided by owner deferral; trace = STATE.md L180 + rollup deviation 0 (see override 2) |

### Data-Flow Trace (Level 4)

Not applicable — no artifact renders dynamic data from a query/store. The CI workflow consumes its own step outputs (census-collect.txt, stage-results.txt, junit), each traced to its producing step.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Stage-1 triple matches the D-03 pin | `.venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants" -k "not mcp_example"` | `197/206 tests collected (9 deselected) in 0.91s` vs pin `193/202 tests collected (9 deselected)` | ✗ FAIL — **Gap 1** |
| giants-only collection is exactly 1 | `.venv/bin/python -m pytest tests/examples --collect-only -q -m giants` | `1/206 tests collected (205 deselected) in 0.86s` | ✓ PASS |
| Contract files green | `pytest tests/test_runner_infra_contracts.py tests/test_models_lock_contracts.py -q` | `20 passed in 0.43s` | ✓ PASS |
| D-05/D-07 seam contracts | `pytest tests/examples/test_notebook_execution.py -k "YamlOverride or yaml_patch" -q` | `6 passed, 62 deselected in 0.91s` | ✓ PASS |
| Tests added by tonight's repairs | `pytest tests/examples/test_notebook_execution.py -k "TestEvoNotebookContentContracts or TestVenvProbeTimeoutContract" -q` | `3 passed, 65 deselected in 0.93s` | ✓ PASS |
| Unknown-marker-free repo-wide collection | `pytest tests/ -m "not slow" --collect-only -q` | `1882/1935 tests collected (53 deselected) in 4.16s`, exit 0 | ✓ PASS |
| Run of record still green | `gh run view 37432001711 --json conclusion,headBranch,headSha,updatedAt` | `conclusion: success`, phs, `170e86f...`, 2026-10-06T12:52:09Z | ✓ PASS |

Session ground truth corroborating the new tests' health beyond this pass: the phase-08 verifier's project-wide fast run at the post-repair HEAD (1881 passed) covered the 4 additions. The 197 collected tests are not broken — only the ratchet literal is stale.

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| scripts/*/tests/probe-*.sh | — | none exist in this phase | N/A — probes are CI-internal steps; the equivalent end-to-end evidence is the green runner run corroborated above |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| CI-03 | 09-01, 09-02, 09-04 | Census integrity, zero new skips, typed categories, audits green | ✗ PARTIAL → Gap 1 | Zero new skips ✓, typed categories ✓, audits green in the run of record ✓, expected_skips.yaml zero commits ✓ — but census integrity at HEAD is broken by the stale D-03 pin (truths 1, 7): the next census run deterministically fails before auditing anything |
| CI-06 | 09-02, 09-04 | Measured budgets; split-out nightly job | ✓ SATISFIED | Truths 2, 15; D-12 measured comments + KEEP-900; 2700-min split-out job |
| CI-07 | 09-04 | Hygiene steps, ceilings review, always-uploads | ✓ SATISFIED | Truth 3; both step bodies re-read incl. WR-02 guards |
| CI-08 | 09-03 | models.lock guard fails on drift | ✓ SATISFIED | Truth 4; 12/12 green at HEAD |
| CI-09 | 09-03 | Coverage expectation documented | ✓ SATISFIED | Truth 5; page + nav verified |

**Orphaned requirements:** none — REQUIREMENTS.md maps exactly {CI-03, CI-06, CI-07, CI-08, CI-09} to Phase 9 (L109-115, all marked Complete); the union of plan `requirements:` fields (09-01: CI-03; 09-02: CI-03/CI-06; 09-03: CI-08/CI-09; 09-04: CI-03/CI-06/CI-07) is the same set.

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| tests/test_models_lock_contracts.py | CI-08 | 12 | 0 | 0 | Value (exact-set equality on drift) | OK |
| tests/test_runner_infra_contracts.py | CI-03/CI-06 | 8 | 0 | 0 | Value (literal pins, http_code shape) | OK |
| tests/examples/test_notebook_execution.py (requirement-linked classes) | CI-03/CI-06 | 6 + 3 new | 0 | 0 | Value (sandbox-vs-source, JSON cell contracts, monkeypatched timeout) | OK — the 3 additions are independent-oracle contracts, not system-output echoes |

Disabled-test scan at HEAD: zero `skip`/`xfail` markers in the requirement-linked files.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | none | — | Zero TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER markers across all covered files (grep exit 0-count), including the 5 files changed by the repair delta |

Prohibition checks at HEAD: expected_skips.yaml zero commits in the phase range; `example/` touched by no 09-0 commit; no cache deletion/pruning in ci.yml; no `continue-on-error` in example-nightly; mypy surface exactly 2 steps; docs page states the coverage non-movement correctly; TESTING.md's no-codecov/no-XML note intact.

### Human Verification Required

N/A — Infrastructure/CI phase; the one failure is deterministic and mechanical (grep literal vs measured line), not a human-judgment item. No truth is present-behavior-unverified.

**Open owner item (advisory, not a phase must-have):** the `09-USER-SETUP.md` sudo re-apply remains open-but-not-required-now; it alone restores the drifted live `OLLAMA_HOST=0.0.0.0` bind (T-09-03, real LAN exposure of an unauthenticated model server). Tracked in 09-USER-SETUP.md, STATE.md, and the rollup; outside this phase's success criteria.

### Gaps Summary

One gap, one root cause, two affected truths. The phase-05..08 post-close repair rounds — each legitimate, each with same-change tests — added 4 fast tests to `tests/examples/test_notebook_execution.py` (374e8e6, 9d44cbf, 5dc18c6) without re-pinning the census ratchet those phases' own green nightly depends on. Everything else the phase delivered survives the repair delta intact and re-verified green at HEAD `c5ad8b5`: the skip-audit machinery, typed categories and zero-new-skips, the split-out job and measured budgets, both hygiene guards, the lock guard (12/12), the docs/nav story, the giants surgical exit (still exactly 1), marker atomicity, the hardened sandbox seam (6/6), both ollama pins, the cron gates, the ty/mypy shape, and the green run of record (gh-corroborated). The repair-class narrative in the delta files themselves (deferral status, re-probe caveats, evo-1 restore) is accurate and consistent with the two carried owner overrides. The fix is mechanical and small: re-pin `193/202` → `197/206` at ci.yml L862-863, append the post-close census addendum to the rollup, and re-dispatch example-nightly to confirm green — structured in the frontmatter for `/gsd-plan-phase --gaps`. Until then, the next cron fire (`30 5 * * *`, tonight) goes red at stage 0.5 by design: the ratchet is doing its job on an un-re-pinned tree.

---

_Verified: 2026-10-07T00:08:10Z_
_Verifier: Claude (gsd-verifier) — regeneration run, convergence round at final HEAD c5ad8b5_
