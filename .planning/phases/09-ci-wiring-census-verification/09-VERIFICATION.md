---
phase: 09-ci-wiring-census-verification
verified: 2026-10-06T14:17:17Z
status: passed
score: 18/18 must-haves verified
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
covered_digest: "v3:sha256:111f703d958b00a889956b5607caaab4a5b38bf0be5ce4f413661aff0efc6355"
behavior_unverified: 0
overrides_applied: 2
overrides:
  - must_have: "The live service state after the owner re-apply serves qwen3.8:latest at num_ctx 8192 on loopback only (09-02, D-06 live half)"
    reason: "Owner decisions supersede plan text (recorded in STATE.md 2026-10-06): the num_ctx cut was DEFERRED entirely at 00:52 CST ('ollama 配置先不改了', superseding the 2026-10-05 A/A decision), and the mcp agent model was swapped qwen3.8:latest -> qwen3.5:4b at 15:27 CST (quick task 261006-lhm, commits dc44856/b91f2a6/0a5ca0d/170e86f). The in-repo OLLAMA_CONTEXT_LENGTH=8192 unit pin stays committed (machine-verified) but inert; D-12 comments and 09-CENSUS-ROLLUP.md deviation 0 / D-06 row record the un-cut measured reality (442-449s pre-swap, 87s post-swap). The open sudo re-apply (which alone restores the drifted live OLLAMA_HOST=0.0.0.0 bind, T-09-03) is tracked open-but-not-required-now in 09-USER-SETUP.md."
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
  gaps_remaining: []
  regressions: []
---

# Phase 9: CI Wiring & Census Verification — Verification Report

**Phase Goal:** The nightly census formally gates the finished execution-test layer — verified end to end on the real runner for collection, skip audit, runtime budget, hygiene steps, lock consistency, and the documented coverage expectation
**Verified:** 2026-10-06T14:17:17Z
**Status:** passed
**Re-verification:** Yes — REGENERATION run (#4682): the prior report (2026-10-06T13:28:26Z, passed 18/18) went stale because covered source changed after that verifier last ran. Not a gap-closure re-run — the prior report had no gaps. All evidence below re-measured at current HEAD `7cf8458` (branch `phs`; the review-fix HEAD `859187a` plus one post-verification dev-dep commit, assessed in the delta table).

## Delta Since the Stale Report (all verified)

The covered-input delta is exactly the post-verification review-fix round, one concurrent dev-dep commit, plus planning docs:

| Commit | Files | Change | Verified how |
|--------|-------|--------|--------------|
| acc8c88 (WR-01) | ci.yml | Stage-4 missing-junit branch now writes the bare `stage4-audit-$junit=1` code to the ledger (end-anchored summary grep can see it); human-readable reason to stdout only | diff read; ledger line asserted in the structural suite; simulation recorded in 09-REVIEW-FIX.md |
| f5066b6 (WR-02) | ci.yml | Both D-13 hygiene steps fail CLOSED on an empty `free -g` parse (`[ -z "${AVAIL_GI}" ]` → FAIL + exit 1) before the `-lt 35` compare | diff read; guard asserted in both step bodies (structural suite); both-path shell simulation in 09-REVIEW-FIX.md |
| 313fd7d (WR-03) | tests/TESTING.md | Coverage guidance aligned to the enforced gate: `fail_under = 90` (not ">80%"), 96.30%/96.42% suite reality, codecov example replaced with the real coverage-gate invocation, "no codecov upload and no XML coverage report in CI" note, stale refs fixed | File read at HEAD: L162/L165-166/L176/L193 carry the aligned story; giants marker docs intact (L99, L124-127) |
| f407fcc (IN-01) | tests/TESTING.md | Retitled the stale "XML coverage report for CI" comment to "local XML coverage report (CI uploads none)" | diff read |
| 1b1ddd4/278ea3f/8d10417/859187a | .planning only | Review-fix report, disposition ledger, incremental re-review (0C/0W/1I, the 1 Info fixed) | 09-REVIEW-DISPOSITION.md read: CR-01/WR-01..03 + IN-01 fixed; IN-02..06 open Info |
| 7cf8458 (quick-261006-uq7) | pyproject.toml (1 line) | Dev-dep pin bump `ruff==0.16.9` -> `ruff==0.16.10` — landed DURING this verification (concurrent owner quick task in the same worktree) | `git show --stat`: 1 file, 1 insertion, 1 deletion, pin line only; giants marker at pyproject L521 re-checked intact; both collection triples re-measured after the commit — identical (1/202 and 193/202, 9 deselected); no pytest/ci.yml/test surface touched, so no must-have moves |
| e499966/1a57f5c | .planning only | The stale VERIFICATION.md itself, phase close-out, ROADMAP/STATE/PROJECT | planning-root docs — inert for the fingerprint (#4623) |

CR-01 (8a405fe, `OLD_MODEL_TOKEN`→`OLD_MODEL_NAME`) and the model-swap quick-task commits (dc44856/b91f2a6/0a5ca0d/170e86f family) predate the stale report and were already covered there; re-checked at HEAD: `grep -rn OLD_MODEL_TOKEN tests/ dnallm/` → zero hits. The incremental re-review of the fix deltas returned 0 Critical / 0 Warning / 1 Info, the Info fixed at f407fcc.

## Goal Achievement

### Observable Truths

Roadmap Success Criteria (the contract) are rows 1-5; PLAN must-have truths follow. Same 18-truth structure as the prior report (its must-have decomposition remains valid); every row re-verified against the codebase at HEAD `7cf8458`.

| # | Truth | Status | Evidence (re-measured at HEAD 7cf8458) |
|---|-------|--------|----------|
| 1 | SC1: nightly census collects and passes all slow execution tests; `audit_skips.py` green with the typed categories registered; fast leg zero new skips | ✓ VERIFIED | `tests/expected_skips.yaml` carries all three typed prefixes (`network-unavailable:` L29, `environment-unavailable:` L35, `optional-dep:` L40); `audit_skips.py` wired at 5 CI points incl. the example-nightly stage-4 per-junit loop (L1046, now with the WR-01 bare-code ledger line); `git log e9056c2..HEAD -- tests/expected_skips.yaml` empty; run 37432001711 example-nightly green = stage-4 audits green; fresh local collection 193/202 (9 deselected) matches the runner-measured census-collect line in the rollup |
| 2 | SC2: measured runtime budgets recorded; separate example-execution nightly job split out | ✓ VERIFIED | example-nightly is the split-out job (`timeout-minutes: 2700`); D-12 comments are MEASURED actuals citing run ids 37345067326 / 37377004230 / 37432001711 (ci.yml L449-455 and L641-680; all three ids appear in both ci.yml and the rollup — counts 6/4/4 and 10/13/8); rollup reproduces every figure; final run measured 2:09:55 with the first wheelhouse cache hit |
| 3 | SC3: hygiene steps observable — kernel pkill + VRAM assertion, sum-of-ceilings review, `if: always()` uploads | ✓ VERIFIED | Named `Stage 1.5/2.5: hygiene (D-13)` steps (L889, L986): `pkill -f ipykernel_launcher`, before/after `LC_ALL=C free -g` logging, hard `-lt 35` + `exit 1` floor, nvidia-smi telemetry-only — plus the WR-02 fail-closed empty-parse guard in BOTH step bodies; VRAM assertion delivered as the GB10-operative free-g floor per the rollup's CI-07 equivalence note; sum-of-ceilings review = recomputed paper ceilings (~3920min all-marks / ~640min bind-set) with the recorded KEEP-900 decision; `if: always()` upload carries all 5 path members (stage-results.txt, stage*.log, census-collect.txt, pytest-junit-*.xml, mcp-server-*.log) |
| 4 | SC4: fast-leg models.lock consistency guard fails on drift | ✓ VERIFIED | `tests/test_models_lock_contracts.py` (12 `def test_`); re-ran at HEAD: 12 passed (in the 20-passed contract run, 0.42s) incl. `TestDriftInjection::test_unlocked_synthetic_model_id_is_reported_exactly` (asserts reported set EQUALS injected id on synthetic fixtures); fail-closed parse; unmarked/kernel-free/network-free |
| 5 | SC5: coverage expectation documented (kernel subprocesses, does not move the 96.30% gate) | ✓ VERIFIED | `docs/user_guide/continuous_integration.md` L26/L30/L32 states the AUDIT-04 note verbatim in substance ("run their notebooks in *kernel subprocesses*... **does not move the 96.30% coverage gate**"), quotes `fail_under = 90`; nav registered in mkdocs.yml L66; WR-03 additionally aligned tests/TESTING.md to the same story (no contradiction remains) |
| 6 | D-01: giants exit surgical — exactly 1 test marked, 4 fast evo contract tests unmarked | ✓ VERIFIED | `giants:` registered in pyproject markers (L521, re-checked after the ruff-bump commit); `_GIANTS_GATED` (L1251) + `_gated_test_param` spec-derived marks (L1256-1298); fresh `-m giants` collection at HEAD: **1/202 collected (201 deselected) in 0.87s**; TestSpecEnvOverrides / TestEvoIsolatedLane untouched by the frozenset |
| 7 | D-03: pre-stage-1 HARD census assertion pins the exact selector triple | ✓ VERIFIED | Stage 0.5 step (ci.yml L849): exact stage-1 flags (`-m "not giants"` AND `-k "not mcp_example"`), tee to census-collect.txt, `grep -qE` on `193/202 tests collected (9 deselected)`, explicit FAIL echo + `exit 1`, no fail-soft; pinned literal equals a fresh local measurement byte-for-byte; observed green in runs 37345067326/37377004230/37432001711 |
| 8 | D-04/D-11: zero evo provisioning and zero models.lock-keyed cache steps remain; neighbors intact | ✓ VERIFIED | yaml token-absence scan in the structural suite: no `evo-venvs`/`wheelhouse-flashattn`/`models-giants`/`evo_torch`/`flash_attn`/`stripedhyena`/`hashFiles('models.lock')`/`Restore model caches` in example-nightly; keeps present (`wheelhouse-mamba`, `megadna-venvs`, `MEGABYTE_pytorch`); uv caches retained |
| 9 | D-19: each nightly job fires on exactly one cron via schedule-string gates; dispatch disjunct kept; push/PR excluded | ✓ VERIFIED | yaml assertion: exact cron literal per job (`0 3 * * *` test-mamba + coverage-nightly, `30 5 * * *` example-nightly) + `workflow_dispatch` disjunct + no push/pull_request disjunct in any of the three gates; root schedule entries unchanged (`0 3 * * *`, `30 5 * * *`); observed live: dispatch run 37432001711 fired all three nightly legs and skipped the push/PR jobs |
| 10 | Marker registration atomic — no invocation ever sees an unknown-marker error | ✓ VERIFIED | Fresh at HEAD: repo-wide `pytest tests/ -m "not slow" --collect-only -q` exits 0 — 1841/1894 collected (53 deselected); `--strict-markers` in addopts would fail collection otherwise |
| 11 | D-05: epochs cut is sandbox-only; committed notebook/YAML byte-identical | ✓ VERIFIED | `seed_sandbox(..., yaml_overrides=None)` seam (signature L361, applied L436, fail-closed ValueError L440); spec `yaml_patch` key on the finetune_custom_head entry only (L200); committed `example/notebooks/finetune_custom_head/finetune_config.yaml` L38 still reads `num_train_epochs: 3`; `git log e9056c2..HEAD --grep="09-0" -- example/` empty (only the owner model-swap quick-task commit b91f2a6 touches example/, covered by override 1); TestSeedSandboxYamlOverrides re-run 5/5 green |
| 12 | D-07: both cut seams proven by kernel-free contract tests | ✓ VERIFIED | Re-ran both at HEAD: `TestSeedSandboxYamlOverrides` 5 passed; `tests/test_runner_infra_contracts.py` pins both unit Environment lines (L72-74) + README re-apply + SSE-probe http_code shape — 8 tests green in the 20-passed run |
| 13 | D-06: in-repo ollama unit pins OLLAMA_CONTEXT_LENGTH=8192 beside the byte-preserved loopback pin | ✓ VERIFIED | `scripts/runner/ollama.service` L38 `Environment="OLLAMA_HOST=127.0.0.1:11434"` + L44 `Environment="OLLAMA_CONTEXT_LENGTH=8192"` with D-06/D-12 header comments (L11, L17); contract tests pin both |
| 14 | D-06 live half: live service serves qwen3.8:latest at num_ctx 8192 on loopback only | ✓ PASSED (override) | Override carried forward unchanged: owner deferred the num_ctx cut (STATE.md 2026-10-06 00:52 CST, L180) and swapped the agent model to qwen3.5:4b (15:27 CST, L181); in-repo pin committed but inert by owner direction; un-cut measured reality recorded in D-12 + rollup deviation 0; open sudo re-apply tracked in 09-USER-SETUP.md — accepted by owner (Tao Zhang) on 2026-10-06. STATE.md traces re-read at HEAD |
| 15 | D-02: one baseline census run with ALL cuts applied rebuilds the authoritative count | ✓ PASSED (override) | Override carried forward unchanged: num_ctx component voided by owner before the baseline (rollup deviation 0); giants + epochs-1 WERE live at baseline 37345067326 (stage-0.5 pin green 193/202; stage-1 192P/1S/0F in 1:59:36; epochs cut measured 566.3s); census of record + budgets remain authoritative — accepted by owner (Tao Zhang) on 2026-10-06 |
| 16 | D-17: one complete example-nightly dispatch GREEN under the final wiring | ✓ VERIFIED | Re-corroborated via gh at verification time: run 37432001711 `conclusion: success`, `head_branch: phs`, `head_sha: 170e86f91cb3a1207ac4712f30ab31c584316a80` (matches the rollup record exactly, completed 2026-10-06T12:52:09Z); example-nightly green in 2h9m55s; stage-4 "OK: every recorded stage item exited 0" per rollup backed by the run's own junit |
| 17 | D-18: test-mamba and coverage-nightly re-dispatched green in-phase, durations recorded | ✓ VERIFIED | Same run 37432001711 (workflow-level conclusion success covers all three legs): test-mamba green 10m59s (1840P/1S/53 deselected), coverage-nightly green 1h51m6s (1938P/15S/0F at 96.42%, above fail_under=90); both legs' measured durations in the rollup and ci.yml comments |
| 18 | D-08/D-09/D-10: ty advisory standalone on coverage-gate; exactly 2 mypy steps; no static check enters pytest | ✓ VERIFIED | `uvx ty@0.0.84 check dnallm/` with `\|\| true` advisory shape on coverage-gate (structural assert); `ci.yml` count of `mypy dnallm/` = exactly 2; no static-check invocation inside any pytest step; WR-01 hardened the D-08 summary grep so a missing-junit failure now turns the job red (bare ledger code line asserted at HEAD) |

**Score:** 18/18 truths verified (16 VERIFIED + 2 PASSED (override); 0 present-behavior-unverified)

### Deferred Items

None — Phase 9 is the final phase of milestone v1.1; no later-phase coverage exists. The owner-deferred num_ctx item is an override (above), not a deferred gap.

### Advisory (New Scope, Unevidenced)

None. The regeneration pass ran the Step 7 scan at full scope over all phase-modified files; the only findings anywhere are the five OPEN Info-severity review items (IN-02..IN-06) already recorded in 09-REVIEW-DISPOSITION.md — carried prior findings with a disposition ledger, not new-scope unevidenced blockers. No new-scope finding without deterministic evidence arose.

### Required Artifacts

All plan-declared artifacts re-checked at three levels (exists / substantive / wired) at HEAD. Level 4 data-flow: N/A — CI configuration, contract tests, and docs; no rendered dynamic data.

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pyproject.toml` | giants marker registration | ✓ VERIFIED | L521 markers entry, slow-line style; addopts/--strict-markers unchanged; the only later commit (7cf8458) touches the ruff pin line only |
| `tests/examples/test_notebook_execution.py` | `_GIANTS_GATED` + composable marks + TestSeedSandboxYamlOverrides | ✓ VERIFIED | L1251-1298; contract class L362; 5/5 tests pass at HEAD |
| `.github/workflows/ci.yml` | deselect, D-03 gate, deletions, D-19 gates, hygiene, upload, ty, D-12 budgets | ✓ VERIFIED | Full structural assertion suite passes at HEAD incl. the WR-01/WR-02 fix elements. Note: gsd-tools `verify.artifacts` reports "Missing pattern: OLLAMA-free hygiene floor literal 35" — the plan's `contains:` value is authoring prose, not a file literal; the substance (`-lt 35`, floor 35Gi) is present and asserted |
| `tests/TESTING.md` | giants marker documented | ✓ VERIFIED | Both listings (L99 pyproject summary, L124-127 Test Markers) with dispatch/manual lane; WR-03/IN-01 edits preserved both and aligned the coverage guidance with the enforced gate |
| `.github/workflows/README.md` | post-surgery topology | ✓ VERIFIED | Dual cron + cron literals, D-11 bullet, example-nightly section with measured totals, num_ctx deferral + qwen3.5:4b language |
| `tests/examples/_execution.py` | yaml_overrides + yaml_patch spec key + cell_timeout seam | ✓ VERIFIED | Signature L361; application L436-447; spec key L200; mcp pair cell_timeout 3600 |
| `scripts/runner/ollama.service` | both Environment pins | ✓ VERIFIED | L38 + L44 with rationale headers (L11/L17) |
| `scripts/runner/README.md` | re-apply op + Why num_ctx + drift notice | ✓ VERIFIED | daemon-reload/restart steps (L18/L33), `systemctl show` verify (L36), 0.0.0.0 drift notice with "never document 0.0.0.0" (L41-45) |
| `tests/test_runner_infra_contracts.py` | fast pins for unit + probe + swap | ✓ VERIFIED | 4 test classes incl. TestExampleNightlySseProbe and TestMcpExampleModelSwap; 8/8 green at HEAD; CR-01 rename residual zero |
| `tests/test_models_lock_contracts.py` | CI-08 guard with drift-injection proof | ✓ VERIFIED | 12 tests green at HEAD; `_parse_lock_rows` fail-closed; `_NON_MODEL_ALLOWLIST` reasoned; route alignment covered-set pinned |
| `docs/user_guide/continuous_integration.md` | AUDIT-04 note + topology + ratchet | ✓ VERIFIED | L26/L30/L32 content greps pass; explicitly states example lane does NOT move the gate (prohibition honored) |
| `mkdocs.yml` | one nav entry | ✓ VERIFIED | L66, User Guide block, single entry |
| `09-CENSUS-ROLLUP.md` | authoritative census record | ✓ VERIFIED | Baseline/Green-gate/re-dispatch run-id lines present; final triple with deselected count; deviations 0-5; D-20 record |

### Key Link Verification

`gsd-tools verify.key-links` resolved 4 of 10 declared links; 6 fail with "Source file not found (from: must be a relative file path)" because the plans' `from:` fields are descriptive prose — a plan-authoring format limitation, **not** missing wiring (same result as the prior report). Every link was re-verified manually with direct codebase evidence at HEAD:

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| pyproject markers list | gated parametrize | giants registration + spec-derived application under --strict-markers | ✓ WIRED | `giants:` L521; `pytest.mark.giants` via `_gated_test_param` L1273; repo-wide collection error-free (truth 10) |
| test file giants mark | ci.yml stage-1 invocation | `-m "not giants"` ANDed with `-k "not mcp_example"` | ✓ WIRED | Both flags in the stage-1 pytest line; fresh measurement deselects exactly 9 = 8 mcp + 1 giants |
| ci.yml D-03 literal | pytest collect-only triple | grep of the triple captured to census-collect.txt | ✓ WIRED | Pinned `193/202 tests collected (9 deselected)` == fresh measurement; runner confirmed in 3 green runs |
| NOTEBOOK_EXEC_SPECS yaml_patch | seed_sandbox yaml_overrides | fixture forwards `spec.get("yaml_patch")` | ✓ WIRED | Spec key L200; fixture forwarding L1293; seam applied L436; 5/5 contract tests green |
| ollama.service | README + contract test | re-apply docs + fast pins | ✓ WIRED | Tool-verified (pattern found); README op + 2 pin tests green |
| models.lock rows | example content scanner | quoted/unquoted org-name extraction, membership assert | ✓ WIRED | Tool-verified path; 24 hf/ms rows + dataset row parse; live-tree membership green; drift injection reports exactly |
| docs page | mkdocs nav + cross-links | nav registration + references | ✓ WIRED | Tool-verified; nav L66; TESTING/README cross-references present |
| D-12 budget comments | rollup measured records | identical run ids in both | ✓ WIRED | 37345067326 / 37377004230 / 37432001711 appear in both ci.yml (L449, L641+) and the rollup |
| census-collect.txt capture | D-14 upload path | assertion output joins the always-uploaded scene | ✓ WIRED | In the `if: always()` upload path list (structural assert) |
| D-06 unit pin + owner re-apply | D-02 baseline dispatch | live-cut precondition | ✓ WIRED (superseded form) | Precondition voided by owner deferral; trace = STATE.md 00:52 entry + rollup deviation 0 + D-12 un-cut records (see override) |

### Data-Flow Trace (Level 4)

Not applicable — no artifact renders dynamic data from a query/store. The CI workflow consumes its own step outputs (census-collect.txt, stage-results.txt, junit), each traced above to its producing step.

### Behavioral Spot-Checks

All re-run fresh during this verification (spot-checks 1-8 measured at review-fix HEAD `859187a`; 1-2 re-measured after the concurrent ruff-bump commit `7cf8458` landed mid-verification, identical results — the bump touches only the dev-dep pin line):

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| giants-only collection is exactly 1 | `.venv/bin/python -m pytest tests/examples --collect-only -q -m giants` | `1/202 tests collected (201 deselected) in 0.87s` (re-measured post-7cf8458) | ✓ PASS |
| Stage-1 triple matches the D-03 pin | `pytest tests/examples --collect-only -q -m "not giants" -k "not mcp_example"` | `193/202 tests collected (9 deselected) in 0.90s` — byte-identical to the ci.yml pin (re-measured post-7cf8458) | ✓ PASS |
| Contract files green | `pytest tests/test_runner_infra_contracts.py tests/test_models_lock_contracts.py -q` | `20 passed in 0.42s` | ✓ PASS |
| D-05/D-07 seam contracts | `pytest tests/examples/test_notebook_execution.py -k "YamlOverride or yaml_patch" -q` | `5 passed, 59 deselected in 0.86s` | ✓ PASS |
| Unknown-marker-free repo-wide collection | `pytest tests/ -m "not slow" --collect-only -q` | `1841/1894 tests collected (53 deselected) in 4.06s`, exit 0 | ✓ PASS |
| Full structural workflow assertion (hygiene incl. WR-02 guards, upload, tee, ty advisory, mypy=2, no continue-on-error, D-03 pin, D-19 gates, D-04/D-11 absence, WR-01 ledger line, D-12 run ids) | python yaml assertion suite over ci.yml | `ALL STRUCTURAL ASSERTIONS PASS` | ✓ PASS |
| Repo-wide lint (CR-01 residual + WR fixes) | `.venv/bin/ruff check . --no-cache --statistics` | exit 0, zero violations | ✓ PASS |
| Final green run of record | `gh run view 37432001711 --json ...` (read-only) | `conclusion: success`, headBranch phs, headSha `170e86f91c...`, completed 2026-10-06T12:52:09Z | ✓ PASS |

Session ground truth (orchestrator, at review-fix HEAD `859187a`): prior-phase regression gate re-run over the 19 prior-phase test files — 592 passed / 1 skipped / 0 failed. The only commit since (7cf8458) changes the ruff dev-dep pin and cannot affect pytest outcomes; the two collection spot-checks re-run after it confirm. The WR-01/WR-02 error-path semantics additionally carry the recorded shell simulations in 09-REVIEW-FIX.md (unguarded floor passes — bug confirmed; guarded — FAIL + exit 1; new ledger line matches the end-anchored grep, old line does not).

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| scripts/*/tests/probe-*.sh | — | none exist in this phase | N/A — probes are CI-internal steps; the equivalent end-to-end evidence is the green runner run corroborated above |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| CI-03 | 09-01, 09-02, 09-04 | Census integrity, zero new skips, typed categories, audits green | ✓ SATISFIED | Truths 1, 6, 7, 10, 11; expected_skips.yaml zero commits in `e9056c2..HEAD`; green run audits |
| CI-06 | 09-02, 09-04 | Measured budgets; split-out nightly job | ✓ SATISFIED | Truths 2, 15; D-12 measured comments + KEEP-900 decision; 2700-min split-out job |
| CI-07 | 09-04 | Hygiene steps, ceilings review, always-uploads | ✓ SATISFIED | Truth 3; structural yaml assertions incl. the WR-02 fail-closed guards + green-run observation |
| CI-08 | 09-03 | models.lock guard fails on drift | ✓ SATISFIED | Truth 4; drift-injection proof run green at HEAD (12/12) |
| CI-09 | 09-03 | Coverage expectation documented | ✓ SATISFIED | Truth 5; page + nav verified; WR-03 removed the last internal contradiction in TESTING.md |

**Orphaned requirements:** none — REQUIREMENTS.md maps exactly {CI-03, CI-06, CI-07, CI-08, CI-09} to Phase 9 (all marked Complete, L109-115); the union of plan `requirements:` fields is the same set.

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| tests/test_models_lock_contracts.py | CI-08 | 12 | 0 | 0 | Value (exact-set equality on drift; live-tree zero-unlocked) | OK — failure mode proven by injection on synthetic fixtures reusing the production checker |
| tests/test_runner_infra_contracts.py | CI-03/CI-06 | 8 | 0 | 0 | Value (literal pins, http_code shape) | OK |
| TestSeedSandboxYamlOverrides | CI-06 | 5 | 0 | 0 | Value (sandbox 1 / source 3) | OK |

Disabled-test scan at HEAD: zero `skip`/`xfail` markers in the three requirement-linked files (grep exit 1). Expected-value provenance: the drift fixtures are synthetic independent oracles, not system output — no circularity. No requirement-linked test file changed since the prior pass (the review-fix delta touched only ci.yml and TESTING.md; the ruff bump touched only the pyproject pin line).

### Decision Coverage

`check.decision-coverage-verify`: 20/20 trackable CONTEXT.md decisions honored by shipped artifacts (skipped: false, blocking: false).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | none | — | Zero TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER markers across all 12 phase-modified files (grep exit 1), including the two files changed by the review-fix delta |

Prohibition checks (all four plans' must-not blocks) at HEAD: expected_skips.yaml zero commits in the phase range; example/ touched only by the owner's model-swap quick task (ground-truth owner decision, outside plan commits — `git log --grep="09-0" -- example/` empty); no cache deletion/pruning command anywhere in ci.yml; no `continue-on-error` in example-nightly; mypy surface untouched (exactly 2 steps); docs page states the opposite of the coverage misconception; TESTING.md now explicitly states there is no codecov upload and no XML coverage report in CI (WR-03).

### Human Verification Required

N/A — Infrastructure/CI phase with no user-facing elements (per the infrastructure-phase scoping gate). All acceptance criteria are verifiable programmatically and were verified at HEAD: structural assertions + contract tests + directly observed green runner run 37432001711 (gh-corroborated again this pass). No truth is present-behavior-unverified: the two plan items tagged as needing recorded dispatch evidence (CI-06 boundary, CI-03 concurrency) have it (rollup measured numbers + the completed green run sequence), and the two post-run review fixes (WR-01/WR-02) are error-path hardenings with recorded two-path shell simulations plus structural assertion at HEAD — the exercised happy path is byte-identical to what ran green on the runner.

**Open owner item (advisory, not a phase must-have):** the `09-USER-SETUP.md` sudo re-apply remains open-but-not-required-now; it is the only path that restores the drifted live `OLLAMA_HOST=0.0.0.0` bind on the runner host (T-09-03, a real LAN exposure of an unauthenticated model server). Tracked in 09-USER-SETUP.md, STATE.md, and the rollup; outside this phase's success criteria.

### Gaps Summary

None. Every roadmap success criterion and every plan must-have is re-verified against the codebase at HEAD `7cf8458` and the real runner evidence; the stale-dating delta (WR-01/WR-02 ci.yml hardening, WR-03/IN-01 TESTING.md alignment) strengthens the verified truths and introduces no regression — corroborated independently by the fresh spot-checks above, the orchestrator's 592P/1S/0F prior-phase regression gate at the review-fix HEAD (the only later commit being the ruff dev-dep pin bump, with both collection spot-checks re-measured identical after it), repo-wide ruff green, and the incremental re-review's 0C/0W verdict. The two plan-text deviations remain owner decisions recorded in STATE.md and handled as documented overrides (num_ctx deferral; model swap), each with verified traces in the rollup, ci.yml comments, and README. The 5 open Info findings (IN-02..IN-06) in 09-REVIEW-DISPOSITION.md are review-level advisories (none fails a must-have).

---

_Verified: 2026-10-06T14:17:17Z_
_Verifier: Claude (gsd-verifier) — regeneration run #4682_
