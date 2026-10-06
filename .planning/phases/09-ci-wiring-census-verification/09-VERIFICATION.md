---
phase: 09-ci-wiring-census-verification
verified: 2026-10-06T13:28:26Z
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
covered_digest: "v3:sha256:a1812c2d8d3d618d23f7269a1317b6eb14aae19f435ee2af993387c08a9d6d07"
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
---

# Phase 9: CI Wiring & Census Verification — Verification Report

**Phase Goal:** The nightly census formally gates the finished execution-test layer — verified end to end on the real runner for collection, skip audit, runtime budget, hygiene steps, lock consistency, and the documented coverage expectation
**Verified:** 2026-10-06T13:28:26Z
**Status:** passed
**Re-verification:** No — initial verification (no prior VERIFICATION.md existed)

## Goal Achievement

### Observable Truths

Roadmap Success Criteria (the contract) are rows 1-5; PLAN must-have truths follow. Every row verified against the actual codebase and, where behavior-dependent, against the real runner run of record.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SC1: nightly census collects and passes all slow execution tests; `audit_skips.py` green with the typed categories registered; fast leg zero new skips | ✓ VERIFIED | `tests/expected_skips.yaml` carries all three typed prefixes (`network-unavailable:` L29, `environment-unavailable:` L35, `optional-dep:` L40); audit wired at 5 CI points incl. the example-nightly stage-4 per-junit audit loop (ci.yml L1038); zero commits to expected_skips.yaml in the phase range (git log e9056c2..HEAD empty); run 37432001711 example-nightly green = stage-4 audits green (hard steps); fresh local collection 193/202 (9 deselected) matches the runner-measured census-collect line in the rollup |
| 2 | SC2: measured runtime budgets recorded; separate example-execution nightly job split out | ✓ VERIFIED | example-nightly is the split-out job (`timeout-minutes: 2700`); D-12 comments in ci.yml are MEASURED actuals citing run ids 37345067326 / 37377004230 / 37432001711 (ci.yml L449-480, L638-680); rollup reproduces every figure; final run measured 2:09:55 with the first wheelhouse cache hit |
| 3 | SC3: hygiene steps observable — kernel pkill + VRAM assertion, sum-of-ceilings review, `if: always()` uploads | ✓ VERIFIED | Named `Stage 1.5/2.5: hygiene (D-13)` steps: `pkill -f ipykernel_launcher`, before/after `LC_ALL=C free -g` logging, hard `-lt 35` + `exit 1` floor, nvidia-smi telemetry-only (step bodies read in full); VRAM assertion delivered as the GB10-operative free-g floor per the rollup's CI-07 equivalence note; sum-of-ceilings review = recomputed paper ceilings (~3920min all-marks / ~640min bind-set) with the recorded KEEP-900 decision; `if: always()` upload carries all 5 path members (stage-results.txt, stage*.log, census-collect.txt, pytest-junit-*.xml, mcp-server-*.log) |
| 4 | SC4: fast-leg models.lock consistency guard fails on drift | ✓ VERIFIED | `tests/test_models_lock_contracts.py` (24.7KB, 12 tests, 5 classes); ran it: 12 passed in 0.43s incl. `TestDriftInjection::test_unlocked_synthetic_model_id_is_reported_exactly` (asserts reported set EQUALS injected id on synthetic fixtures — the failure-mode proof, read in source) and `TestRouteAlignment` mismatch reporting; fail-closed parse per the audit_skips discipline; unmarked/kernel-free/network-free |
| 5 | SC5: coverage expectation documented (kernel subprocesses, does not move the 96.30% gate) | ✓ VERIFIED | `docs/user_guide/continuous_integration.md` states the AUDIT-04 note verbatim in substance ("run their notebooks in *kernel subprocesses*... **does not move the 96.30% coverage gate**"), quotes `fail_under = 90` ratchet; nav registered in mkdocs.yml L66 (User Guide block) |
| 6 | D-01: giants exit surgical — exactly 1 test marked, 4 fast evo contract tests unmarked | ✓ VERIFIED | `giants:` registered in pyproject markers (L521); `_GIANTS_GATED` holds exactly one id + `_gated_test_param` applies marks spec-derived (test_notebook_execution.py L1251-1298); fresh `-m giants` collection: **1/202 collected (201 deselected)**; TestSpecEnvOverrides (L670) / TestEvoIsolatedLane (L822) untouched by the frozenset |
| 7 | D-03: pre-stage-1 HARD census assertion pins the exact selector triple | ✓ VERIFIED | Stage 0.5 step (ci.yml L849-866): exact stage-1 flags (`-m "not giants" -k "not mcp_example"`), tee to census-collect.txt, grep -qE on `193/202 tests collected (9 deselected)`, explicit FAIL echo + `exit 1`, no fail-soft; pinned literal equals a fresh local measurement byte-for-byte; observed green in runs 37345067326/37377004230/37432001711 |
| 8 | D-04/D-11: zero evo provisioning and zero models.lock-keyed cache steps remain; neighbors intact | ✓ VERIFIED | yaml token-absence scan: no `evo-venvs`/`wheelhouse-flashattn`/`models-giants`/`evo_torch`/`flash_attn`/`stripedhyena` in example-nightly; keeps present (`wheelhouse-mamba`, `megadna-venvs`, `MEGABYTE_pytorch`); no `models-${{ hashFiles('models.lock') }}` key or `Restore model caches` in either nightly job; uv caches retained in both |
| 9 | D-19: each nightly job fires on exactly one cron via schedule-string gates; dispatch disjunct kept; push/PR excluded | ✓ VERIFIED | yaml assertion: exact cron literal per job (`0 3 * * *` x2, `30 5 * * *` x1) + `workflow_dispatch` disjunct + no push/pull_request in all three gates; root schedule entries unchanged; observed live: dispatch run 37432001711 fired all three nightly legs and skipped the push/PR jobs |
| 10 | Marker registration atomic — no invocation ever sees an unknown-marker error | ✓ VERIFIED | Repo-wide `pytest tests/ -m "not slow" --collect-only -q` exits 0: 1841/1894 collected (53 deselected); `--strict-markers` in addopts would fail collection otherwise |
| 11 | D-05: epochs cut is sandbox-only; committed notebook/YAML byte-identical | ✓ VERIFIED | `seed_sandbox(..., yaml_overrides=None)` seam (L357-361, applied L436-447, fail-closed ValueError); spec `yaml_patch` key on the finetune_custom_head entry only (L200); committed `example/notebooks/finetune_custom_head/finetune_config.yaml` L38 still reads `num_train_epochs: 3`; git log shows zero phase-plan commits under example/; TestSeedSandboxYamlOverrides 5/5 green (source-still-reads-3 honesty pin included) |
| 12 | D-07: both cut seams proven by kernel-free contract tests | ✓ VERIFIED | Ran both files: `TestSeedSandboxYamlOverrides` 5 passed; `tests/test_runner_infra_contracts.py` pins both unit Environment lines + README re-apply + SSE-probe http_code shape (8 tests green in the 20-passed run) |
| 13 | D-06: in-repo ollama unit pins OLLAMA_CONTEXT_LENGTH=8192 beside the byte-preserved loopback pin | ✓ VERIFIED | `scripts/runner/ollama.service` L38 `Environment="OLLAMA_HOST=127.0.0.1:11434"` + L44 `Environment="OLLAMA_CONTEXT_LENGTH=8192"` with D-06/precedence header comments; contract tests pin both |
| 14 | D-06 live half: live service serves qwen3.8:latest at num_ctx 8192 on loopback only | ✓ PASSED (override) | Override: owner deferred the num_ctx cut (STATE.md 2026-10-06 00:52 CST) and swapped the agent model to qwen3.5:4b (15:27 CST); in-repo pin committed but inert by owner direction; un-cut measured reality recorded in D-12 + rollup deviation 0; open sudo re-apply tracked in 09-USER-SETUP.md — accepted by owner (Tao Zhang) on 2026-10-06 |
| 15 | D-02: one baseline census run with ALL cuts applied rebuilds the authoritative count | ✓ PASSED (override) | Override: num_ctx component voided by owner before the baseline (rollup deviation 0); giants + epochs-1 WERE live at baseline 37345067326 (stage-0.5 pin green 193/202; stage-1 192P/1S/0F in 1:59:36; epochs cut measured 566.3s); census of record + budgets remain authoritative — accepted by owner (Tao Zhang) on 2026-10-06 |
| 16 | D-17: one complete example-nightly dispatch GREEN under the final wiring | ✓ VERIFIED | Independently corroborated via gh: run 37432001711 `conclusion: success`, `head_branch: phs`, `head_sha: 170e86f...` (matches the rollup record exactly); example-nightly green in 2h9m55s; stage-4 "OK: every recorded stage item exited 0" per rollup backed by the run's own junit |
| 17 | D-18: test-mamba and coverage-nightly re-dispatched green in-phase, durations recorded | ✓ VERIFIED | Same run 37432001711: test-mamba green 10m59s (1840P/1S/53 deselected), coverage-nightly green 1h51m6s (1938P/15S/0F at 96.42%, above fail_under=90); both legs' measured durations in the rollup and ci.yml comments |
| 18 | D-08/D-09/D-10: ty advisory standalone on coverage-gate; exactly 2 mypy steps; no static check enters pytest | ✓ VERIFIED | `uvx ty@0.0.84 check dnallm/` with `\|\| true` advisory shape on coverage-gate; `grep -c 'mypy dnallm/'` = exactly 2; no static-check invocation inside any pytest step |

**Score:** 18/18 truths verified (16 VERIFIED + 2 PASSED (override); 0 present-behavior-unverified)

### Deferred Items

None — Phase 9 is the final phase of milestone v1.1; no later-phase coverage exists. The owner-deferred num_ctx item is an override (above), not a deferred gap.

### Required Artifacts

All plan-declared artifacts checked at three levels (exists / substantive / wired). Level 4 data-flow: N/A — CI configuration, contract tests, and docs; no rendered dynamic data.

| Artifact | Expected | Status | Details |
|----------|----------|--------|--------|
| `pyproject.toml` | giants marker registration | ✓ VERIFIED | L521 markers entry, slow-line style; addopts/--strict-markers unchanged otherwise |
| `tests/examples/test_notebook_execution.py` | `_GIANTS_GATED` + composable marks + TestSeedSandboxYamlOverrides | ✓ VERIFIED | L1251-1298; contract class L362; mcp pair in `_TIMEOUT_7200_GATED` (decision B); 5/5 tests pass |
| `.github/workflows/ci.yml` | deselect, D-03 gate, deletions, D-19 gates, hygiene, upload, ty, D-12 budgets | ✓ VERIFIED | All structural assertions pass (see truths 2,3,7,8,9,18). Note: gsd-tools `verify.artifacts` reported "Missing pattern: OLLAMA-free hygiene floor literal 35" — the plan's `contains:` value is authoring prose, not a file literal; the substance (`-lt 35`, "floor 35Gi") is present and asserted |
| `tests/TESTING.md` | giants marker documented | ✓ VERIFIED | Both listings (L99 pyproject summary, L124 Test Markers) with dispatch/manual lane |
| `.github/workflows/README.md` | post-surgery topology | ✓ VERIFIED | Dual cron + cron literals (L16-17), D-11 bullet, example-nightly section with measured totals, num_ctx deferral + qwen3.5:4b language |
| `tests/examples/_execution.py` | yaml_overrides + yaml_patch spec key + cell_timeout seam | ✓ VERIFIED | Signature L357-361; application L436-447; spec key L200; mcp pair cell_timeout 3600 |
| `scripts/runner/ollama.service` | both Environment pins | ✓ VERIFIED | L38 + L44 with rationale headers |
| `scripts/runner/README.md` | re-apply op + Why num_ctx + drift notice | ✓ VERIFIED | daemon-reload/restart steps, `systemctl show` verify, 0.0.0.0 drift notice (never suggested as an example) |
| `tests/test_runner_infra_contracts.py` | fast pins for unit + probe + swap | ✓ VERIFIED | 4 test classes incl. TestExampleNightlySseProbe (http_code shape) and TestMcpExampleModelSwap; 8/8 green |
| `tests/test_models_lock_contracts.py` | CI-08 guard with drift-injection proof | ✓ VERIFIED | 12 tests green; `_parse_lock_rows` fail-closed; `_NON_MODEL_ALLOWLIST` reasoned + liveness-checked; route alignment covered-set pinned |
| `docs/user_guide/continuous_integration.md` | AUDIT-04 note + topology + ratchet | ✓ VERIFIED | All content greps pass; explicitly states example lane does NOT move the gate (prohibition honored) |
| `mkdocs.yml` | one nav entry | ✓ VERIFIED | L66, User Guide block, single entry |
| `09-CENSUS-ROLLUP.md` | authoritative census record | ✓ VERIFIED | Baseline/Green-gate/re-dispatch run-id lines present; final triple with deselected count; deviations 0-5; D-20 record |

### Key Link Verification

`gsd-tools verify.key-links` could not resolve 6 of 10 declared links because the plans' `from:` fields are descriptive prose ("pyproject.toml markers list"), not bare relative file paths — a plan-authoring format limitation, **not** missing wiring. Every link was therefore verified manually with direct codebase evidence:

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| pyproject markers list | gated parametrize | giants registration + spec-derived application under --strict-markers | ✓ WIRED | `giants:` L521; `pytest.mark.giants` applied via `_gated_test_param` L1273; repo-wide collection error-free (truth 10) |
| test file giants mark | ci.yml stage-1 invocation | `-m "not giants"` ANDed with `-k "not mcp_example"` | ✓ WIRED | ci.yml L880 carries both flags; fresh measurement deselects exactly 9 = 8 mcp + 1 giants |
| ci.yml D-03 literal | pytest collect-only triple | grep of the triple captured to census-collect.txt | ✓ WIRED | Pinned `193/202 tests collected (9 deselected)` == fresh measurement; runner confirmed in 3 green runs |
| NOTEBOOK_EXEC_SPECS yaml_patch | seed_sandbox yaml_overrides | fixture forwards `spec.get("yaml_patch")` | ✓ WIRED | Spec key L200; fixture forwarding; seam applied L436-447; 5/5 contract tests green |
| ollama.service | README + contract test | re-apply docs + fast pins | ✓ WIRED | Tool-verified (pattern found); README op + 2 pin tests green |
| models.lock rows | example content scanner | quoted/unquoted org-name extraction, membership assert | ✓ WIRED | 24 hf/ms rows + dataset row parse; live-tree membership green; drift injection reports exactly |
| docs page | mkdocs nav + cross-links | nav registration + references | ✓ WIRED | Tool-verified; nav L66; TESTING/README cross-references present |
| D-12 budget comments | rollup measured records | identical run ids in both | ✓ WIRED | 37345067326 / 37377004230 / 37432001711 appear in both ci.yml comments and the rollup |
| census-collect.txt capture | D-14 upload path | assertion output joins the always-uploaded scene | ✓ WIRED | ci.yml L1060 in the `if: always()` path list |
| D-06 unit pin + owner re-apply | D-02 baseline dispatch | live-cut precondition | ✓ WIRED (superseded form) | Precondition voided by owner deferral; trace = STATE.md 00:52 entry + rollup deviation 0 + D-12 un-cut records (see override) |

### Data-Flow Trace (Level 4)

Not applicable — no artifact renders dynamic data from a query/store. The CI workflow consumes its own step outputs (census-collect.txt, stage-results.txt, junit), each traced above to its producing step.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| giants-only collection is exactly 1 | `.venv/bin/python -m pytest tests/examples --collect-only -q -m giants` | `1/202 tests collected (201 deselected)` | ✓ PASS |
| Stage-1 triple matches the D-03 pin | `pytest tests/examples --collect-only -q -m "not giants" -k "not mcp_example"` | `193/202 tests collected (9 deselected)` — byte-identical to ci.yml L862-863 | ✓ PASS |
| Contract files green | `pytest tests/test_runner_infra_contracts.py tests/test_models_lock_contracts.py -q` | `20 passed in 0.43s` | ✓ PASS |
| D-05/D-07 seam contracts | `pytest tests/examples/test_notebook_execution.py -k "YamlOverride or yaml_patch" -q` | `5 passed, 59 deselected in 0.91s` | ✓ PASS |
| Unknown-marker-free repo-wide collection | `pytest tests/ -m "not slow" --collect-only -q` | `1841/1894 tests collected (53 deselected)`, exit 0 | ✓ PASS |
| Final green run of record | `gh api .../runs/37432001711` (read-only) | `conclusion: success`, head_sha `170e86f` @ phs; test-mamba 10m59s / example-nightly 2h9m55s / coverage-nightly 1h51m6s all green | ✓ PASS |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| scripts/*/tests/probe-*.sh | — | none exist in this phase | N/A — probes are CI-internal steps; the equivalent end-to-end evidence is the green runner run corroborated above |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| CI-03 | 09-01, 09-02, 09-04 | Census integrity, zero new skips, typed categories, audits green | ✓ SATISFIED | Truths 1, 6, 7, 10, 11; expected_skips.yaml untouched; green run audits |
| CI-06 | 09-02, 09-04 | Measured budgets; split-out nightly job | ✓ SATISFIED | Truths 2, 15; D-12 measured comments + KEEP-900 decision; 2700-min split-out job |
| CI-07 | 09-04 | Hygiene steps, ceilings review, always-uploads | ✓ SATISFIED | Truth 3; structural yaml assertions + green-run observation |
| CI-08 | 09-03 | models.lock guard fails on drift | ✓ SATISFIED | Truth 4; drift-injection proof read in source and run green |
| CI-09 | 09-03 | Coverage expectation documented | ✓ SATISFIED | Truth 5; page + nav verified |

**Orphaned requirements:** none — REQUIREMENTS.md maps exactly {CI-03, CI-06, CI-07, CI-08, CI-09} to Phase 9; the union of plan `requirements:` fields is the same set.

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| tests/test_models_lock_contracts.py | CI-08 | 12 | 0 | 0 | Value (exact-set equality on drift; live-tree zero-unlocked) | OK — failure mode proven by injection on synthetic fixtures reusing the production checker |
| tests/test_runner_infra_contracts.py | CI-03/CI-06 | 8 | 0 | 0 | Value (literal pins, http_code shape) | OK |
| TestSeedSandboxYamlOverrides | CI-06 | 5 | 0 | 0 | Value (sandbox 1 / source 3) | OK |

Disabled-test scan: zero `skip`/`xfail` markers in the three requirement-linked files. Expected-value provenance: the drift fixtures are synthetic independent oracles, not system output — no circularity.

### Decision Coverage

`check.decision-coverage-verify`: 20/20 trackable CONTEXT.md decisions honored by shipped artifacts (skipped: false, blocking: false).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | none | — | Zero TBD/FIXME/XXX/TODO/HACK/placeholder markers across all 12 phase-modified files |

Prohibition checks (all four plans' must-not blocks): expected_skips.yaml zero commits in phase range; example/ touched only by the owner's model-swap quick task (ground-truth owner decision, outside plan commits); no cache deletion/pruning command anywhere in ci.yml; no continue-on-error in example-nightly; mypy surface untouched; docs page states the opposite of the coverage misconception.

### Human Verification Required

N/A — Infrastructure/CI phase with no user-facing elements. All acceptance criteria are verifiable programmatically and were verified (structural assertions + contract tests + the directly observed green runner run 37432001711). No truth is left present-behavior-unverified; the two backstop-tagged plan items (CI-06 boundary, CI-03 concurrency) required recorded dispatch evidence and have it (rollup measured numbers + the completed green run sequence), so neither abstains.

**Open owner item (advisory, not a phase must-have):** the `09-USER-SETUP.md` sudo re-apply remains open-but-not-required-now; it is the only path that restores the drifted live `OLLAMA_HOST=0.0.0.0` bind on the runner host (T-09-03, a real LAN exposure of an unauthenticated model server). Tracked in 09-USER-SETUP.md, STATE.md, and the rollup; outside this phase's success criteria.

### Gaps Summary

None. Every roadmap success criterion and every plan must-have is verified against the codebase and the real runner evidence; the only two plan-text deviations are owner decisions recorded in STATE.md and handled as documented overrides (num_ctx deferral; model swap), each with verified traces in the rollup, ci.yml comments, and README. The 9 open advisory findings in 09-REVIEW-DISPOSITION.md are review-level (none fails a must-have per the reviewer's own verification) and CR-01 is fixed at 8a405fe.

---

_Verified: 2026-10-06T13:28:26Z_
_Verifier: Claude (gsd-verifier)_
