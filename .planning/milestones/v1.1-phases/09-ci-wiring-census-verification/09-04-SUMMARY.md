---
phase: 09-ci-wiring-census-verification
plan: 04
subsystem: ci-testing
tags: [census, d-02-baseline, d-12-budgets, d-17-green-gate, d-18-redispatch, sse-probe, bedtools, timeout-bump, num-ctx-deferred, model-swap]

requires:
  - phase: 09-01
    provides: giants exit topology + the D-03 census pin this plan measures against
  - phase: 09-02
    provides: D-05 epochs cut (live) + the D-06 unit pin (deferred-not-live per owner) + the 193/202 triple
  - phase: 09-03
    provides: lock-consistency guards (untouched here)
provides:
  - 09-CENSUS-ROLLUP.md — the authoritative Phase 9 census record: D-02 baseline counts (192P/1S/9-deselected stage-1 census, pin 193/202 green), measured budgets for all three nightly legs, the full dispatch ledger (37345067326 / 37377004230 / 37406829738 / 37432001711 green-gate + two cancelled non-evidence runs), deviations 0-5, and the CI-07 free-g-floor equivalence note
  - D-12 measured budget comments in ci.yml (both nightly jobs) + workflows README refresh — every number reproducible from cited run ids
  - SSE-correct readiness probe for example-nightly stage 2 (5dfadda) + TestExampleNightlySseProbe contract tests (2) pinning the http_code shape
  - coverage-nightly bedtools rootless provisioning (decd2cc) — closes its full-tree census environment gap
  - mcp-pair cell timeout 3600s + 7200s outer mark (aa4a6e7, owner decision B) at the run_notebook NotebookClient(timeout=) seam
  - green-phase-close evidence: final run 37432001711 ALL THREE nightly legs green (example ledger zero / test-mamba 1840P / coverage 1938P @ 96.42%)
affects: [v1.1 ship (nightly topology now proven green end-to-end; num_ctx deferral + qwen3.5:4b swap recorded for the owner's revisit)]

actuals:
  tokens: 39247    # chars/4 over git diff 08f4d59..HEAD — range INCLUDES the two owner-dispatched interleaved quick tasks (261006-cum plot fix, 261006-lhm model swap); plan-owned commits alone are 7 of the 14
  tasks: 3
  commits: 14      # MEASURED: git rev-list --count 08f4d59..HEAD (7 plan commits + 7 quick-task commits interleaved mid-flight)
plan_head_before: 08f4d59b9e4f46ca12ed9a67d95cdc8f915a61c5
plan_head_after: 0f25afa6348ed8288d195420fe5f2aa280d2b2ee

tech-stack:
  added: []        # comments, workflow wiring, test/spec edits only — no new dependencies (T-09-SC held)
  patterns:
    - "SSE readiness = received HTTP status, never curl exit code: %{http_code} prints once headers arrive even when -m cuts the infinite event-stream body; 000 is the only not-ready signal (pinned by same-change contract tests)"
    - "Per-leg budget ceilings live at the spec seam the executor actually hits (NOTEBOOK_EXEC_SPECS cell_timeout -> NotebookClient(timeout=...)) with the outer pytest mark kept strictly ABOVE the cell budget via _TIMEOUT_7200_GATED"
    - "One workflow_dispatch measures all three D-19-gated nightly legs at one commit — per-leg dispatch intent collapses into single-run evidence"

key-files:
  created:
    - .planning/phases/09-ci-wiring-census-verification/09-CENSUS-ROLLUP.md
  modified:
    - .github/workflows/ci.yml                  # D-12 measured comments; bedtools steps; budget citations (job shape itself landed in Task 1 / 9fb853c)
    - .github/workflows/README.md               # measured totals + bedtools step + deferral language
    - tests/test_runner_infra_contracts.py      # TestExampleNightlySseProbe (2 tests)
    - tests/examples/_execution.py              # mcp pair cell_timeout 3600 + decision-B comments
    - tests/examples/test_notebook_execution.py # pair joins _TIMEOUT_7200_GATED

key-decisions:
  - "One dispatch = all three legs: every workflow_dispatch fires all three D-19-gated nightly jobs, so the plan's per-leg dispatches collapsed into single runs measuring every leg at one commit — recorded, not worked around"
  - "coverage-nightly OQ3 decision (with measurement in hand): KEEP the 900-min kill as a documented override below both recomputed paper sums (~640min bind-set / ~3920min all-marks) — per-test marks are the primary protection and the measured green-path total is ~1:42-1:58"
  - "num_ctx cut recorded as DEFERRED everywhere it was promised (owner decision 2026-10-06 00:52 CST): D-12 comments state the un-cut measured reality; the USER-SETUP re-apply stays open-but-not-required-now"
  - "Stage-3 timeout raised at the real seam (owner decision B): the spec cell_timeout IS NotebookClient(timeout=) — the trait behind the traitlets 1800s error — and the pair joined _TIMEOUT_7200_GATED so the outer mark stays strictly above the cell budget (the harness contract)"
  - "Cancelled dispatches (37335797121, 37427527257) are bookkeeping, never evidence (T-09-10); the green-gate line names the final-wiring run 37432001711 with 37377004230 kept as the first-green superseded shape"

patterns-established:
  - "Readiness probes for streaming endpoints must assert the received HTTP status; an exit-code probe is structurally ungreenable against a healthy SSE server (17 minutes of blind polling proved it live)"
  - "A leg that 'died transiently at checkout' deserves a full-census triage once it finally runs — behind run 1's coverage-nightly sat a real environment gap (bedtools) plus the known plot failure, not egress luck"
  - "Nightly budget comments cite dispatch run ids whose logs reproduce every figure — no proportional estimation anywhere (Pitfall 7 held)"

requirements-completed: [CI-03, CI-06, CI-07]

coverage:
  - id: D1
    description: "Task 1 job shape (9fb853c, predecessor): named D-13 hygiene steps with the >=35Gi free -g floor, complete if:always() D-14 failure scene, per-stage tee'd logs, exact-pinned advisory ty step, exactly 2 mypy steps"
    requirement: CI-07
    verification:
      - kind: structural
        ref: "Task 1 yaml assertion (predecessor run): 2 hard hygiene steps, 5-member upload path under always(), >=4 tee'd invocations, uvx ty@0.0.84 advisory, mypy count 2"
        status: pass
    human_judgment: false
  - id: D2
    description: "D-02 baseline census of record + D-03 pin integrity: stage-0.5 193/202 (9 deselected) green on the runner; stage-1 192 passed / 1 skipped / 0 failed; zero unexpected skips (audits green)"
    requirement: CI-03
    verification:
      - kind: run-evidence
        ref: "run 37345067326 census-collect.txt + pytest-junit-example-stage1.xml (+ reproduced exactly in 37377004230 and 37432001711); Task 2 verify BASELINE-RECORDED"
        status: pass
    human_judgment: false
  - id: D3
    description: "D-12 measured budget rewrite: both nightly comments carry measured actuals citing run ids; coverage-nightly raise-vs-keep decided (KEEP 900, documented override); README refreshed with the same numbers"
    requirement: CI-06
    verification:
      - kind: structural
        ref: "Task 2 verify BASELINE-RECORDED (run id cross-check rollup<->ci.yml, triple grep -qF, measured language); numbers reproducible from run logs"
        status: pass
    human_judgment: false
  - id: D4
    description: "D-17 green-run gate: one complete example-nightly dispatch green under the final wiring"
    requirement: CI-03
    verification:
      - kind: run-evidence
        ref: "Green-gate dispatch run id: 37432001711 — 'OK: every recorded stage item exited 0', every junit audit green, census 192P/1S, hygiene floors green, sse probes green, pair green"
        status: pass
    human_judgment: false
  - id: D5
    description: "D-18 transient-leg closure: test-mamba and coverage-nightly green in-phase at the final head"
    requirement: CI-06
    verification:
      - kind: run-evidence
        ref: "37432001711: test-mamba 1840P/0F/1S in 1:53; coverage-nightly 1938P/0F/15S in 1:42:18 at 96.42% coverage (floor 90)"
        status: pass
    human_judgment: false
  - id: D6
    description: "num_ctx deferral recorded honestly (owner decision 2026-10-06): un-cut measured reality in D-12 (qwen3.8-era pair 442-449s + the >30-min tail that red 37406829738; post-swap pair 87s), deferral language in rollup/comments/README, USER-SETUP item referenced as open"
    requirement: CI-06
    verification:
      - kind: structural
        ref: "rollup deviation 0 + deviation 5; ci.yml stage-3 comment lines; README example-nightly paragraph"
        status: pass
    human_judgment: false

duration: "~21h wall (four full runner cycles: 37345067326, 37377004230, 37406829738, 37432001711 + two cancelled dispatches)"
completed: 2026-10-06
status: complete
---

# Phase 9 Plan 04: Nightly Census Verification & Phase Closure Summary

**The D-02 baseline census, D-12 measured budgets, and the D-17/D-18 green gates all landed on real runner evidence — four dispatch cycles measured every nightly leg (192P/1S stage-1 census at a green 193/202 pin; budgets 2:10-2:50 example / 1:42-1:58 coverage / 10:49 mamba), surfaced and repaired three real defects (SSE exit-code probe structurally ungreenable, coverage-nightly bedtools gap, an 1800s cell line below the un-cut qwen3.8 latency tail), inherited two owner mid-flight decisions (plot fix 550d311, qwen3.5:4b swap 170e86f) plus owner decision B's timeout bump, and closed with run 37432001711 green on ALL THREE legs (example ledger zero, test-mamba 1840P, coverage 1938P @ 96.42%).**

## Performance

- **Duration:** ~21h wall (dominated by the queue-serialized runner; executor active time was a small fraction)
- **Tasks:** 3/3 (Task 1 by the predecessor agent at 9fb853c)
- **Files modified:** 5 source/doc files + 1 new phase artifact (rollup)

## Accomplishments

- **Task 1 (9fb853c, predecessor):** the example-nightly job-shape completion — named Stage 1.5/2.5 hygiene hard gates with the GB10-safe `free -g` >=35Gi floor, the complete `if: always()` failure scene (five path members), per-stage tee'd logs, and the exact-pinned advisory `uvx ty@0.0.84` step on coverage-gate (mypy boundary held at exactly 2).
- **Task 2:** baseline dispatch 37345067326 at final-Task-1 HEAD measured the census of record — stage-0.5 pin green (193/202, 9 deselected), stage-1 **192 passed / 1 skipped / 0 failed in 1:59:36**, D-05 epochs cut measured (gated finetune_custom_head **566.3s** vs the ~31min 3-epoch dev-box figure), zero unexpected skips. The D-12 rewrite replaced both nightly budget comments' stale recomputations with measured actuals citing run ids; the coverage-nightly OQ3 decision landed (KEEP 900, documented override below both paper sums); 09-CENSUS-ROLLUP.md scaffolded and filled as the authoritative record.
- **Task 3:** the green gates. Run 37377004230 delivered the first complete-green example-nightly (post SSE repair) and proved the bedtools fix on coverage-nightly; run 37406829738 went green on test-mamba + coverage post-plot-fix and exposed the stage-3 1800s cell timeout (owner decision B → aa4a6e7 at the `NotebookClient(timeout=cell_timeout)` seam with the `_TIMEOUT_7200_GATED` outer-mark contract); the final run **37432001711** closed everything green on all three legs. D-20 recorded; CI-07's VRAM-equivalence note written; every cancelled dispatch recorded as bookkeeping.
- **Three real defects found and fixed with same-change tests:** the SSE readiness probe (5dfadda + 2 contract tests — an exit-code probe can NEVER succeed against a healthy SSE endpoint; 17:58 of blind polling over a fully-serving server proved it), the coverage-nightly bedtools gap (decd2cc — 3 of its 4 first-live failures), and the timeout line (aa4a6e7).

## Task Commits

1. **Task 1: job shape** - `9fb853c` (ci, predecessor agent; pushed)
2. **Rule 3 fix: coverage-nightly bedtools** - `decd2cc` (fix)
3. **Rule 1 fix: SSE-correct readiness probe + contract tests** - `5dfadda` (fix)
4. **Task 2: D-12 measured rewrite + rollup** - `df987b0` (ci)
5. **Task 3: closure records** - `8774151` (ci)
6. **Owner decision B: timeout bump** - `aa4a6e7` (fix)
7. **Final all-green records** - `0f25afa` (ci)

**Interleaved owner-dispatched quick tasks (inherited, not plan work):** 550d311/0aae609 (261006-cum plot fix) and febb627..170e86f (261006-lhm qwen3.5:4b model swap).

**Plan metadata:** (this commit) (docs: complete plan)

## Files Created/Modified

- `.github/workflows/ci.yml` — D-12 measured budget comments (both nightly jobs, citing 37345067326/37377004230/37432001711); coverage-nightly bedtools steps; SSE-correct stage-2 probe
- `.github/workflows/README.md` — measured totals, bedtools step entry, num_ctx deferral + model-swap language
- `.planning/phases/09-ci-wiring-census-verification/09-CENSUS-ROLLUP.md` — NEW: the authoritative Phase 9 census record
- `tests/test_runner_infra_contracts.py` — TestExampleNightlySseProbe (2 tests; contract file 5/5 green)
- `tests/examples/_execution.py` + `tests/examples/test_notebook_execution.py` — mcp-pair cell_timeout 3600 + `_TIMEOUT_7200_GATED` membership (decision B)

## Decisions Made

- Single-dispatch evidence model (one workflow_dispatch = all three D-19-gated legs) adopted and recorded instead of fighting it with per-leg re-dispatches.
- coverage-nightly timeout decision: KEEP 900 with the measurement in hand (marks primary; measured green path ~1:42-1:58; raising buys nothing the marks do not provide).
- The D-02 census baseline stayed run 37345067326 (the plan's "final HEAD" at dispatch time) with runs 2-4 layered as the repair/validation chain — every number in D-12 names its run.

## Deviations from Plan

**0. [Owner decision 2026-10-06 deferral] Task-2 precondition (D-06 live re-apply) VOIDED** — baseline measured the UN-CUT mcp-pair behavior; every num_ctx promise replaced by deferral language (rollup deviation 0; D-12 comments; README). The USER-SETUP sudo item stays open (it alone restores the drifted `OLLAMA_HOST=0.0.0.0` live bind, T-09-03).

**1. [Rule 3 - blocking environment gap] coverage-nightly bedtools** — 3 of its 4 first-live failures; fixed at decd2cc with the proven example-nightly rootless steps; proven by 37377004230 (1933P) and green-closed by 37432001711 (1938P). The plan's "transient" classification of this leg was true only of its checkout-death point.

**2. [Rule 1 - wiring bug] SSE readiness probe** — structurally ungreenable exit-code probe (run 37345067326: healthy server polled blind 17:58); fixed at 5dfadda (http_code gate) with 2 same-change contract tests; live-proven in every later run (sse 3 passed in 4.75s).

**3. [Fenced pre-existing failure - resolved by owner mid-flight] `test_plot_for_regression`** — diagnosed read-only (plot.py:234 `astype(float)` over a dict-valued metric column) and left per the fence; the owner dispatched quick task 261006-cum (550d311) while the runner cycles ran; green on the runner from 37406829738 onward.

**4. [Owner decision B] stage-3 cell timeout 1800→3600s** — run 37406829738's only example-nightly failure (CellTimeoutError at exactly 1800s on the un-cut qwen3.8 tail); raised at the real seam with the outer 7200s mark (harness contract held); superseded in practice by the qwen3.5:4b swap (pair 87s) but kept as headroom.

**5. [Inherited owner decision] qwen3.8 → qwen3.5:4b model swap** (261006-lhm, febb627..170e86f) — 11-file sweep landing mid-phase; the timeout bump survives it by owner direction (the 4.2B brain still defaults to a 262144 ctx); stage 3 measured minutes-scale post-swap.

**Total deviations:** 5 recorded + 1 voided precondition — all auto-fixed or owner-resolved inline; every acceptance criterion PASS at close.

## Issues Encountered

- **Egress flakiness:** every push/dispatch succeeded with 1-attempt retries except none needed beyond attempt 1 after the first dispatch; `gh run view --log` gates on run completion (worked around via the per-job logs API with `--allow-escape-sequences`).
- **Cache store at threshold:** the mamba wheelhouse missed on runs 1-2 (23-24 min rebuilds each) before finally hitting on the final run — recorded in D-12 as a measured cost; the store posture stays owner discretion (OQ4).
- **Run 3's coverage-nightly was misreported as red in circulation** — verified GREEN from its own logs (1935P in 1:58:18); the red coverage was run 2's pre-fix state. Corrected in the rollup.

## Authentication Gates

None.

## User Setup Required

- **Deferred, not blocking (owner decision 2026-10-06):** the `09-USER-SETUP.md` ollama re-apply (num_ctx 8192 default + the live loopback-pin restore, T-09-03) remains open-but-not-required-now. With the qwen3.5:4b swap, the urgency is lower still; revisit alongside any num_ctx un-deferral.

## Known Stubs

None — no stubs, placeholders, or unwired data paths were introduced.

## Next Phase Readiness

- Phase 09 is COMPLETE: criteria 1 (green complete dispatch, D-17 via 37432001711), 2 (measured budgets, D-12), and 3 (observable hygiene, D-13/D-14 shape + the free -g equivalence note) all hold on runner evidence.
- The nightly topology is proven green end-to-end at phs @ 0f25afa's ancestry (final run at 170e86f; subsequent commits are records only). Post-merge, the 03:00/05:30 UTC crons take over from the default branch with the D-19 gates.
- Open owner items carried forward: num_ctx deferral revisit-with-data (rollup records both models' measurements), the cache-store posture (OQ4), PR #42 as record (D-20), and the USER-SETUP re-apply above.

## Self-Check: PASSED

- Files: rollup + SUMMARY present; all 5 modified files carry their changes
- Commits: 9fb853c, decd2cc, 5dfadda, df987b0, 8774151, aa4a6e7, 0f25afa — all ancestors of HEAD; both plan verifies (BASELINE-RECORDED, PHASE-GATE-RECORDED) green at the final head
- Fences held: zero cache cleanup anywhere; evo exit remains deselect-only (`-m "not giants"`, never a typed skip); exactly 2 mypy steps; no num_ctx/ollama state touched by this plan

---
*Phase: 09-ci-wiring-census-verification*
*Completed: 2026-10-06*
