---
phase: 09-ci-wiring-census-verification
verified: 2026-10-07T03:15:46Z
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
covered_digest: "v3:sha256:4dd4b502a90aee060e3ac6ac1781c15c26ddbe388494a829bc8380b43033c8d3"
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
  previous_status: gaps_found
  previous_score: 16/18
  gaps_closed:
    - "D-03 census ratchet re-pin: stage-0.5 literal 193/202 vs measured 197/206 at HEAD — closed by 86c1fe8 (both ci.yml Stage 0.5 carriers re-pinned to 197/206 + PROJECT.md narrative + rollup post-close addendum) and confirmed green by example-nightly dispatch run 37550730293 (workflow_dispatch @ 86c1fe8, all steps success incl. Stage 0.5 on the re-pinned triple), run id backfilled by 43a47a0"
  gaps_remaining: []
  regressions: []
---

# Phase 9: CI Wiring & Census Verification — Verification Report

**Phase Goal:** The nightly census formally gates the finished execution-test layer — verified end to end on the real runner for collection, skip audit, runtime budget, hygiene steps, lock consistency, and the documented coverage expectation
**Verified:** 2026-10-07T03:15:46Z
**Status:** passed
**Re-verification:** Yes — GAP-CLOSURE re-run: the prior report (2026-10-07T00:08:10Z, gaps_found 16/18, digest 2b82439c, measured at c5ad8b5) had exactly one gap — the D-03 census ratchet literal was stale (193/202 pinned vs 197/206 measured) after the phase-05..08 post-close repair rounds added 4 fast tests. The gap is now closed at HEAD `43a47a0` (branch `phs`). All evidence re-measured at HEAD; the previously-failed truths (1 collection clause, 7 D-03 pin) received full 3-level re-verification; the 16 previously-green truths received quick regression checks (existence + basic sanity + the cheap behavioral spot-checks re-run).

## Gap Closure Verification (the one prior gap)

The prior gap's three `missing:` items, each verified closed:

| Missing item | Closure evidence at HEAD 43a47a0 | Status |
|---|---|---|
| Re-pin ci.yml L862-863 grep literal + FAIL echo 193/202 → 197/206 | Commit 86c1fe8 diff read: both Stage 0.5 carriers re-pinned — L862 `grep -qE '^=* *197/206 tests collected \(9 deselected\) in '`, L863 FAIL echo "expected triple: 197/206 tests collected (9 deselected)". `grep -n "193/202" ci.yml` → zero hits; the only census-triple literals in ci.yml are the two re-pinned 197/206 lines | ✓ CLOSED |
| Rollup post-close addendum recording 197/206 and attributing +4 | 09-CENSUS-ROLLUP.md "Post-close census addendum (2026-10-07...)": growth table attributing the 4 tests to 374e8e6 (05 IN-01), 9d44cbf (05 IN-02), 5dc18c6 (08 CR-01 x2); fresh measurement "197/206 tests collected (9 deselected) — was 193/202"; deselected unchanged (8 mcp + 1 giants); both carriers re-pinned in the same commit (landed in 86c1fe8, +23 lines) | ✓ CLOSED |
| Re-dispatch example-nightly at the re-pin HEAD, record green run id | **Run 37550730293** (`workflow_dispatch`, headBranch `phs`, headSha `86c1fe8b6fe...` — the re-pin commit itself): example-nightly job `conclusion: success`, every step success including "Stage 0.5: census collection assertion (D-03) — hard gate, census integrity" and "Stage 1: torch-heavy example execution". Run id recorded in the rollup by 43a47a0 | ✓ CLOSED |

**Independent re-measurement (not trusting the commit message):** the exact stage-0.5 command re-run locally at HEAD — `.venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants" -k "not mcp_example"` → `=============== 197/206 tests collected (9 deselected) in 2.38s ================`; simulating the gate's exact grep against this output → **PASS** (byte-for-byte match), `MEASURED=[197/206 tests collected (9 deselected)]`.

**Runner-side proof (fetched from the job log of 37550730293, job 112565306501):**

- Stage 0.5 output: `=============== 197/206 tests collected (9 deselected) in 1.37s ================` then `census collection measured: 197/206 tests collected (9 deselected)` — the runner measured the re-pinned triple and the hard gate passed.
- Stage 1 result: `===== 196 passed, 1 skipped, 9 deselected, 1 warning in 7370.53s (2:02:50) =====` — all 197 collected tests accounted for (196 pass + the one allowlisted typed skip).
- stage-results.txt as printed by the Stage 4 summary: `stage1-examples=0 stage1-yaml=0 stage2-mcp-probes-streamable=0 stage2-mcp-probes-sse=0 stage3-mcp-pair=0` + all five `stage4-audit-*=0`, then `OK: every recorded stage item exited 0`. The summary step carries a hard gate (`if grep -qE '=[1-9][0-9]*$' stage-results.txt; then ... exit 1`) — job success genuinely proves every stage exited 0; the job cannot be forever-green.
- Skip audits: `1 skipped test(s) in pytest-junit-example-stage1.xml` (the known typed skip), 0 in every other junit.
- Stage 3 mcp pair: `2 passed, 66 deselected in 91.01s` — the post-swap latency behavior holds on the re-pin HEAD.
- test-mamba (in_progress) and coverage-nightly (queued) legs of the same dispatch were still executing at verification time — corroborating only, not gap-critical (the gap's missing item was the census ratchet + example-nightly; D-18's in-phase green evidence remains run 37432001711).

## Delta Since the Prior Report

`git log c5ad8b5..HEAD` = 5 commits; the covered-input delta is exactly three files (2ed4bc0/2be8a20/acb675c are planning docs of other phases plus the prior 09 report itself — inert per #4623):

| File | Commit | Change | Verified how |
|------|--------|--------|--------------|
| `.github/workflows/ci.yml` | 86c1fe8 | Exactly 2 lines: Stage 0.5 grep literal + FAIL echo re-pinned 193/202 → 197/206 (L862-863). No other step touched | diff read (whole ci.yml delta = 4 ++/--); gate simulation PASS locally; runner green in 37550730293 |
| `.planning/PROJECT.md` | 86c1fe8 | 1 line: shipped-phase narrative updated to "197/206 collected, 9 deselected; re-pinned 2026-10-07 after post-close repair growth" | diff read; consistent with the rollup addendum |
| `09-CENSUS-ROLLUP.md` | 86c1fe8, 43a47a0 | +23-line post-close addendum (growth table, fresh measurement, both-carriers note) + 1-line run-id backfill (37550730293 example-nightly success) | full addendum re-read; run corroborated live via gh |

## Goal Achievement

### Observable Truths

Same 18-truth structure as the prior reports (the must-have decomposition remains valid); every row re-verified against the codebase at HEAD `43a47a0`.

| # | Truth | Status | Evidence (re-measured at HEAD 43a47a0) |
|---|-------|--------|----------|
| 1 | SC1: nightly census collects and passes all slow execution tests; `audit_skips.py` green with the typed categories registered; fast leg zero new skips | ✓ VERIFIED | **Collects and passes now proven at HEAD by dispatch 37550730293**: Stage 0.5 measured `197/206 tests collected (9 deselected)` and passed the hard gate; Stage 1 `196 passed, 1 skipped, 9 deselected in 2:02:50`; all stage-results items =0 under the hard non-zero exit gate. Typed prefixes present (`network-unavailable:` L29, `environment-unavailable:` L35, `optional-dep:` L40 in tests/expected_skips.yaml); `git log e9056c2..HEAD -- tests/expected_skips.yaml` empty; deselected counts unchanged (9 in tests/examples, 53 repo-wide — zero new skips); runner skip audits green (1 known typed skip) |
| 2 | SC2: measured runtime budgets recorded; separate example-execution nightly job split out | ✓ VERIFIED | example-nightly job `timeout-minutes: 2700` (L682); D-12 run-id co-citations re-counted: 37345067326 (ci.yml 6 / rollup 10), 37377004230 (4/13), 37432001711 (4/8), plus 37550730293 in the rollup addendum; new dispatch's Stage 1 measured 2:02:50, consistent with the recorded 2:09:55 envelope |
| 3 | SC3: hygiene steps observable — kernel pkill + VRAM assertion, sum-of-ceilings review, `if: always()` uploads | ✓ VERIFIED | `pkill -f ipykernel_launcher` at L902/994/1043; fail-closed parse guard + hard `-lt 35` + `exit 1` in BOTH hygiene bodies (L915, L1007); `if: always()` uploads at L1047/1071/1083 with census-collect.txt an upload path member (L1080); sum-of-ceilings review + KEEP-900 in the rollup; runner log shows Stage 1.5/2.5 steps success in 37550730293 |
| 4 | SC4: fast-leg models.lock consistency guard fails on drift | ✓ VERIFIED | Re-run at HEAD: contract files `20 passed in 1.09s` including the 12 lock-contract tests with the drift-injection exact-set test; file untouched since the prior pass |
| 5 | SC5: coverage expectation documented (kernel subprocesses, does not move the 96.30% gate) | ✓ VERIFIED | docs page re-read: `fail_under = 90` ratchet floor, 96.30% landing, AUDIT-04 note verbatim ("does not move the 96.30% coverage gate"); nav registered mkdocs.yml L66 |
| 6 | D-01: giants exit surgical — exactly 1 test marked, fast evo contract tests unmarked | ✓ VERIFIED | `giants:` marker pyproject L521; fresh `-m giants` collection: **1/206 collected (205 deselected)** — still exactly one |
| 7 | D-03: pre-stage-1 HARD census assertion pins the exact selector triple; pinned literal equals a fresh measurement byte-for-byte | ✓ VERIFIED | ci.yml L862 pins `197/206 tests collected (9 deselected)`; fresh local measurement with the exact stage-1 selector flags: `197/206 tests collected (9 deselected) in 2.38s`; exact-grep simulation PASS. Zero `193/202` remnants in ci.yml. Runner-corroborated: Stage 0.5 success in 37550730293 on the identical triple |
| 8 | D-04/D-11: zero evo provisioning and zero models.lock-keyed cache steps remain; neighbors intact | ✓ VERIFIED | Token scan over the example-nightly job body: zero hits for evo-venv/wheelhouse-flashattn/models-giants/evo_torch/flash_attn/stripedhyena/`hashFiles('models.lock')`; wheelhouse-mamba / megadna-venvs / MEGABYTE_pytorch neighbors present |
| 9 | D-19: each nightly job fires on exactly one cron via schedule-string gates; dispatch disjunct kept; push/PR excluded | ✓ VERIFIED | L282/L448 gate `workflow_dispatch \|\| schedule == '0 3 * * *'`; L607 gates on `'30 5 * * *'`; root schedule entries unchanged; run 37550730293 observed firing via workflow_dispatch with exactly the three nightly legs + skipped fast/deploy legs |
| 10 | Marker registration atomic — no invocation ever sees an unknown-marker error | ✓ VERIFIED | Fresh repo-wide `pytest tests/ -m "not slow" --collect-only -q`: **1882/1935 collected (53 deselected)**, exit 0 — identical to the prior pass (no further test growth) |
| 11 | D-05: epochs cut is sandbox-only; committed notebook/YAML byte-identical | ✓ VERIFIED | Committed `example/notebooks/finetune_custom_head/finetune_config.yaml` L38 still `num_train_epochs: 3`; `git log e9056c2..HEAD --grep="09-0" -- example/` empty; seam contracts green (truth 12) |
| 12 | D-07: both cut seams proven by kernel-free contract tests | ✓ VERIFIED | Re-run: `TestSeedSandboxYamlOverrides`-scope `-k "YamlOverride or yaml_patch"` → **6 passed, 62 deselected**; infra contracts 8/8 green inside the 20-passed run |
| 13 | D-06: in-repo ollama unit pins OLLAMA_CONTEXT_LENGTH=8192 beside the byte-preserved loopback pin | ✓ VERIFIED | `scripts/runner/ollama.service` L45 `OLLAMA_HOST=127.0.0.1:11434` + L53 `OLLAMA_CONTEXT_LENGTH=8192`; contract tests pin both (green) |
| 14 | D-06 live half: live service serves qwen3.8:latest at num_ctx 8192 on loopback only | ✓ PASSED (override) | Override carried forward unchanged; STATE.md owner entries re-read at HEAD (00:52 deferral, 15:27 model swap); the gap-closure delta touches none of this; Stage 3 pair ran green post-swap (91s) in the new dispatch |
| 15 | D-02: one baseline census run with ALL cuts applied rebuilds the authoritative count | ✓ PASSED (override) | Override carried forward unchanged: num_ctx component voided by owner before baseline 37345067326 (rollup deviation 0); the post-close addendum correctly frames the 197/206 re-measure as the successor authoritative triple |
| 16 | D-17: one complete example-nightly dispatch GREEN under the final wiring | ✓ VERIFIED | Strengthened since the prior report: the prior caveat ("a NEW dispatch at HEAD would fail stage 0.5") is resolved — run **37550730293** IS that new dispatch, at the re-pin commit 86c1fe8, and the example-nightly job is `success` with all steps green incl. Stage 0.5 (gh-corroborated live this pass; job log fetched and inspected). Historical run of record 37432001711 remains green |
| 17 | D-18: test-mamba and coverage-nightly re-dispatched green in-phase, durations recorded | ✓ VERIFIED | In-phase evidence stands: run 37432001711 legs green (10m59s / 1h51m6s @ 96.42%), durations in the rollup and ci.yml comments; the new dispatch's two legs were in_progress/queued at verification time — corroborating, not required |
| 18 | D-08/D-09/D-10: ty advisory standalone on coverage-gate; exactly 2 mypy steps; no static check enters pytest | ✓ VERIFIED | `uvx ty@0.0.84 check dnallm/ \|\| true` advisory at L436; `mypy dnallm/` count in ci.yml = exactly 2 (L127, L206); stage 1-3 invocations pure pytest (stage-1 block re-read at L876-884) |

**Score:** 18/18 truths verified (16 VERIFIED + 2 PASSED (override); 0 present-behavior-unverified)

### Deferred Items

None — Phase 9 is the final phase of milestone v1.1; the prior gap is closed, not deferred.

### Advisory (New Scope, Unevidenced)

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| — | none | — | Re-verification ran the Step 7 scan at full scope over all covered files (zero debt markers; the single `continue-on-error` grep hit inside the example-nightly job body is the negative-assertion comment "no step in this job carries continue-on-error" at L679, not a directive). No new-scope finding without deterministic evidence arose; nothing to downgrade. Two gsd-tools pattern mismatches against PLAN prose are recorded in Required Artifacts below — they are tool-literal vs plan-prose artifacts with deterministic counter-evidence, not Step 7 findings. (Decision-coverage gate ran clean this pass: 20/20 honored — see Decision Coverage.) |

### Required Artifacts

All plan-declared artifacts re-checked (exists / substantive / wired) at HEAD. Level 4 data-flow: N/A — CI configuration, contract tests, and docs; no rendered dynamic data.

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pyproject.toml` | giants marker registration | ✓ VERIFIED | L521 markers entry; untouched by the gap-closure delta |
| `tests/examples/test_notebook_execution.py` | `_GIANTS_GATED` + composable marks + TestSeedSandboxYamlOverrides | ✓ VERIFIED | giants-only collection still exactly 1; seam contracts 6/6 green at HEAD; untouched by the gap-closure delta |
| `.github/workflows/ci.yml` | deselect, D-03 gate, deletions, D-19 gates, hygiene, upload, ty, D-12 budgets | ✓ VERIFIED | All structural elements re-verified at HEAD; the D-03 pin literal now equals the fresh measurement byte-for-byte (truth 7) and is runner-green. gsd-tools `verify.artifacts` reports two authoring-prose mismatches, both deterministic non-defects: (a) 09-02 pattern "193/202" — the PLAN text was authored at the pre-re-pin triple; the artifact's intent (hard-asserted census triple) is satisfied by the 197/206 literal the bump-point design requires (plan prose cannot be edited post-hoc without rewriting history; the rollup addendum is the authoritative record of the bump); (b) 09-04 pattern "OLLAMA-free hygiene floor literal 35" — same class as both prior reports; the `-lt 35` floors exist at L915/L1007 |
| `tests/TESTING.md` | giants marker documented | ✓ VERIFIED | Untouched; listings intact incl. the no-codecov/no-XML note (L193) |
| `.github/workflows/README.md` | post-surgery topology | ✓ VERIFIED | Untouched by the delta; dual cron + num_ctx deferral + qwen3.5:4b language intact |
| `tests/examples/_execution.py` | yaml_overrides + yaml_patch spec key + cell_timeout seam | ✓ VERIFIED | Untouched; 6/6 contracts green at HEAD |
| `scripts/runner/ollama.service` | both Environment pins | ✓ VERIFIED | L45 + L53 byte-preserved |
| `scripts/runner/README.md` | re-apply op + Why num_ctx + drift notice | ✓ VERIFIED | Untouched; re-apply steps + drift notice intact |
| `tests/test_runner_infra_contracts.py` | fast pins for unit + probe + swap | ✓ VERIFIED | Untouched; green at HEAD (inside the 20-passed run) |
| `tests/test_models_lock_contracts.py` | CI-08 guard with drift-injection proof | ✓ VERIFIED | Untouched; 12/12 green at HEAD |
| `docs/user_guide/continuous_integration.md` | AUDIT-04 note + topology + ratchet | ✓ VERIFIED | Untouched; all three clauses verified |
| `mkdocs.yml` | one nav entry | ✓ VERIFIED | L66, single entry |
| `09-CENSUS-ROLLUP.md` | authoritative census record | ✓ VERIFIED | Run-id lines + deviations + D-20 record intact; the new post-close addendum records the 197/206 growth attribution, the fresh measurement, the same-commit both-carrier re-pin, and the green re-dispatch run id — exactly the items the prior gap required |

### Key Link Verification

`gsd-tools verify.key-links` resolves 2/10 (09-02: 1/2, 09-03: 1/2) — every NOT is the same parse class as both prior reports (`from: must be a relative file path` — descriptive prose in plan-authored `from:` fields, a plan-authoring format limitation, not missing wiring; the prior report's 4/10 vs this pass's 2/10 differs only in how many prose fields the matcher accepted, no link gained or lost substance). Manual verification with direct codebase evidence at HEAD:

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| pyproject markers list | gated parametrize | giants registration + spec-derived application | ✓ WIRED | `giants:` L521; giants-only collection = exactly 1; repo-wide collection error-free (truth 10) |
| test file giants mark | ci.yml stage-1 invocation | `-m "not giants"` AND `-k "not mcp_example"` | ✓ WIRED | Both flags in the stage-1 line (L880); fresh measurement deselects exactly 9 |
| ci.yml D-03 literal | pytest collect-only triple | grep of the tee'd census-collect.txt | ✓ WIRED | **Restored this pass**: pin 197/206 = measured 197/206 byte-for-byte; gate simulation PASS locally; Stage 0.5 success on the runner (37550730293) |
| NOTEBOOK_EXEC_SPECS yaml_patch | seed_sandbox yaml_overrides | spec forwarding | ✓ WIRED | 6/6 seam contracts green at HEAD |
| ollama.service | README + contract test | re-apply docs + fast pins | ✓ WIRED | Tool-verified + 20 contract tests green |
| models.lock rows | example content scanner | org-name extraction + membership assert | ✓ WIRED | 12/12 green incl. drift injection |
| docs page | mkdocs nav + cross-links | nav registration | ✓ WIRED | Tool-verified; nav L66 |
| D-12 budget comments | rollup measured records | identical run ids | ✓ WIRED | 3 budget run ids co-cited in ci.yml (14 lines) and rollup (31 lines); 37550730293 in the rollup addendum |
| census-collect.txt capture | D-14 upload path | assertion output in always-upload | ✓ WIRED | Path member at L1080 under `if: always()` |
| D-06 unit pin + owner re-apply | D-02 baseline dispatch | live-cut precondition | ✓ WIRED (superseded form) | Voided by owner deferral; trace = STATE.md + rollup deviation 0 (override 2) |

### Data-Flow Trace (Level 4)

Not applicable — no artifact renders dynamic data from a query/store. The CI workflow consumes its own step outputs (census-collect.txt, stage-results.txt, junit), each traced to its producing step; this pass additionally traced the consumed values end-to-end on the runner (job log of 37550730293) rather than by inspection alone.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Stage-1 triple matches the D-03 pin | `.venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants" -k "not mcp_example"` | `197/206 tests collected (9 deselected) in 2.38s` = pin at L862 | ✓ PASS (was FAIL in prior report) |
| Exact stage-0.5 grep gate against fresh output | simulate `tail -1 \| grep -qE '^=* *197/206 tests collected \(9 deselected\) in '` | matches; `MEASURED=[197/206 tests collected (9 deselected)]` | ✓ PASS |
| Census collects AND passes on the real runner | job log of run 37550730293 (example-nightly, @ 86c1fe8) | Stage 0.5 measured 197/206 green; Stage 1 `196 passed, 1 skipped, 9 deselected in 2:02:50`; all stage-results =0; hard summary gate "OK" | ✓ PASS |
| giants-only collection is exactly 1 | `pytest tests/examples --collect-only -q -m giants` | `1/206 tests collected (205 deselected) in 0.84s` | ✓ PASS |
| Contract files green | `pytest tests/test_runner_infra_contracts.py tests/test_models_lock_contracts.py -q` | `20 passed in 1.09s` | ✓ PASS |
| D-05/D-07 seam contracts | `pytest tests/examples/test_notebook_execution.py -k "YamlOverride or yaml_patch" -q` | `6 passed, 62 deselected in 0.84s` | ✓ PASS |
| Unknown-marker-free repo-wide collection | `pytest tests/ -m "not slow" --collect-only -q` | `1882/1935 tests collected (53 deselected) in 4.61s`, exit 0 | ✓ PASS |
| example-nightly dispatch green at re-pin HEAD | `gh run view 37550730293` + job-log fetch | job `success`, all steps success, headSha 86c1fe8, Stage 0.5 + Stage 1 green | ✓ PASS |

### Probe Execution

| Probe | Command | Result | Status |
|-------|-------|--------|--------|
| scripts/*/tests/probe-*.sh | — | none exist in this phase | N/A — probes are CI-internal steps; the equivalent end-to-end evidence is the green dispatch corroborated above (log-inspected, not just conclusion-checked) |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| CI-03 | 09-01, 09-02, 09-04 | Census integrity, zero new skips, typed categories, audits green | ✓ SATISFIED | Upgraded from PARTIAL: census integrity restored at HEAD — re-pinned gate green locally and on the runner (37550730293); zero new skips; typed categories registered; runner skip audits green |
| CI-06 | 09-02, 09-04 | Measured budgets; split-out nightly job | ✓ SATISFIED | Truths 2, 15; D-12 measured comments + KEEP-900; 2700-min split-out job; new Stage 1 measurement 2:02:50 inside the envelope |
| CI-07 | 09-04 | Hygiene steps, ceilings review, always-uploads | ✓ SATISFIED | Truth 3; both step bodies + runner steps green |
| CI-08 | 09-03 | models.lock guard fails on drift | ✓ SATISFIED | Truth 4; 12/12 green at HEAD |
| CI-09 | 09-03 | Coverage expectation documented | ✓ SATISFIED | Truth 5; page + nav verified |

**Orphaned requirements:** none — REQUIREMENTS.md maps exactly {CI-03, CI-06, CI-07, CI-08, CI-09} to Phase 9 (L109-115, all marked Complete); the union of plan `requirements:` fields (09-01: CI-03; 09-02: CI-03/CI-06; 09-03: CI-08/CI-09; 09-04: CI-03/CI-06/CI-07) is the same set.

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| tests/test_models_lock_contracts.py | CI-08 | 12 | 0 | 0 | Value (exact-set equality on drift) | OK |
| tests/test_runner_infra_contracts.py | CI-03/CI-06 | 8 | 0 | 0 | Value (literal pins, http_code shape) | OK |
| tests/examples/test_notebook_execution.py (requirement-linked classes) | CI-03/CI-06 | 6 + 3 repair-added | 0 | 0 | Value (sandbox-vs-source, JSON cell contracts, monkeypatched timeout) | OK — independent-oracle contracts, not system-output echoes |

Disabled-test scan at HEAD: zero skip/xfail markers in the requirement-linked files.

### Decision Coverage

Gate ran clean this pass (prior pass reported a tool-side skip): `check.decision-coverage-verify` → `total: 20, honored: 20, not_honored: []` — "All trackable CONTEXT.md decisions are honored by shipped artifacts." Non-blocking by design; recorded for drift review.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | none | — | Zero TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER markers across all covered files, including the three delta files |

Prohibition checks at HEAD: expected_skips.yaml zero commits in the phase range; `example/` touched by no 09-0 commit; no cache deletion/pruning tokens in ci.yml; zero actual `continue-on-error` directives in the example-nightly job (the L679 hit is the negative-assertion comment); mypy surface exactly 2 steps; ty advisory standalone with `|| true`; docs page states the coverage non-movement correctly; TESTING.md's no-codecov/no-XML note intact.

### Human Verification Required

N/A — Infrastructure/CI phase with no user-facing elements. The one prior failure was deterministic and mechanical and is now closed with runner-grade behavioral evidence (the census transition is exercised by a real green dispatch, stronger than a local test — no truth is present-behavior-unverified).

**Open owner item (advisory, not a phase must-have):** the `09-USER-SETUP.md` sudo re-apply remains open-but-not-required-now; it alone restores the drifted live `OLLAMA_HOST=0.0.0.0` bind (T-09-03, real LAN exposure of an unauthenticated model server). Tracked in 09-USER-SETUP.md, STATE.md, and the rollup; outside this phase's success criteria.

### Gaps Summary

None. The single prior gap — the D-03 census ratchet stale after the phase-05..08 post-close repair growth — is closed completely and verifiably: both ci.yml Stage 0.5 carriers re-pinned to the measured `197/206 tests collected (9 deselected)` in the same commit (86c1fe8) as the rollup's post-close addendum attributing the +4 growth to the exact repair commits; the re-pin was then proven on the real runner by dispatch 37550730293 (example-nightly job success at the re-pin headSha, Stage 0.5 green on the re-pinned triple, Stage 1 census 196P/1S/0F with the hard summary gate reporting every stage item exit 0), with the run id recorded in the rollup (43a47a0). The fresh local measurement at HEAD reproduces the pinned triple byte-for-byte, and the exact-grep gate simulation passes. All 16 previously-green truths re-verified without regression (covered-input delta since the prior report is exactly the three gap-closure files; every cheap behavioral spot-check re-run green with values identical to the prior pass). The two carried owner overrides (num_ctx deferral / model swap; D-02 baseline form) remain accurate at HEAD. The nightly cron (`30 5 * * *`) will now stay green at this tree.

---

_Verified: 2026-10-07T03:15:46Z_
_Verifier: Claude (gsd-verifier) — gap-closure re-verification at final HEAD 43a47a0_
