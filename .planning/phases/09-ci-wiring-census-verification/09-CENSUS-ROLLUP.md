# 09-CENSUS-ROLLUP — D-02 Baseline Census & Nightly Budget Record (09-04)

**Purpose.** The authoritative Phase 9 census record: the post-wiring baseline census
(D-02) measured on the real runner, the D-12 measured nightly budgets, the D-17/D-18
dispatch evidence, and the deviations between plan-era expectations and the measured
reality. Supersedes the 08-CENSUS-ROLLUP steady-state numbers as the budget record.

**Execution environment.** Self-hosted `dnallm-nightly` runner (GB10, queue-serialized —
one job at a time by design), ref `phs`, final phase HEAD. One `workflow_dispatch`
fires all three nightly jobs together (each D-19 gate keeps the dispatch disjunct),
so the single baseline run measures all three legs at the same commit.

## Decisions applied (the wiring this census measures)

| decision | wiring state at baseline |
| --- | --- |
| D-01 giants exit | `giants` pytest marker registered; the one evo execution test marked spec-derived (`_GIANTS_GATED`); stage 1 runs `-m "not giants"` — a deselect, never a typed skip (owner policy: the runner environment works, a skip would be dishonest). The 4 fast evo contract tests stay on every fast leg (OQ2) |
| D-03 census pin | Stage 0.5 hard gate: collect-only triple for the exact stage-1 selector set, pinned at **193/202 tests collected (9 deselected)** (09-01 pinned 188/197; 09-02 deliberately bumped to 193/202 in the same commit that grew the census by its 5 contract tests). Deselect split unchanged: 8 mcp + 1 giants |
| D-04/D-11 removals | evo provisioning (isolated venv, flash-attn wheelhouse, giant prefetch) and the models.lock-keyed hub cache restores deleted — cold pulls by design; the box's local `$HOME` hub caches stay the warm path, never cleaned (owner rule). The flash-attn build-isolation failure of run 37278002681 dissolved with the deletion |
| D-05 epochs cut (LIVE) | `finetune_custom_head` runs sandbox-only `num_train_epochs: 1` via the spec `yaml_patch` key consumed by `seed_sandbox` (committed notebook content byte-identical, still reads 3). First live runner exercise measured in this baseline. Dev-box projection ~31 -> ~11 min for that notebook |
| D-06 num_ctx cut (DEFERRED) | **NOT LIVE — deferred by owner decision 2026-10-06 00:52 CST ("ollama 配置先不改了"), superseding the 2026-10-05 A/A cut decision and the 2026-10-06 00:42 per-request rework idea.** The in-repo `OLLAMA_CONTEXT_LENGTH=8192` unit pin (commit e85eb73) stays committed but inert — the live ollama service is untouched, so the mcp_example pair measured here runs at its existing (~256k ctx) kv-cache behavior. The latency/VRAM cost is accepted by the owner for now; the D-12 measured numbers below are the data the owner asked for to revisit the decision later. The `09-USER-SETUP.md` sudo re-apply item stays open-but-not-required-now (it is also the only thing that restores the drifted live `OLLAMA_HOST=0.0.0.0` bind to the loopback pin, T-09-03) |
| D-08/D-09/D-10 ty/mypy | ty runs advisory-only on the coverage-gate fast leg (`uvx ty@0.0.84 check dnallm/ \|\| true`); exactly 2 mypy steps remain; no static check enters pytest. The hard flip + mypy retirement are ONE later atomic change (D-09) |
| D-13 hygiene | Stage 1.5 / Stage 2.5 named hard gates: kernel pkill, before/after `LC_ALL=C free -g` available-value logging, `>=35Gi` floor assertion, nvidia-smi telemetry-only (GB10 reports no memory through it) |
| D-14 failure scene | `if: always()` upload carries every junit, every per-stage tee'd log, census-collect.txt, stage-results.txt, and the MCP server logs — green and red alike |
| D-19 cron gates | all three nightly jobs carry cron-string gates (`0 3 * * *` for coverage-nightly/test-mamba, `30 5 * * *` for example-nightly) closing the 05:30 double-trigger before phs merges to main |

## Dispatch record (run ids)

| run id | ref/commit | event | outcome |
| --- | --- | --- | --- |
| 37327398343 | phs @ e9056c2 | workflow_dispatch (09-01 wiring) | all three legs red at old wiring — example-nightly died at the since-deleted Stage-0 evo provisioning step (flash-attn build isolation); coverage-nightly and test-mamba died at "Checkout code" (transient egress). The latter two are exactly D-18's re-dispatch targets; neither failure was code-caused |
| 37333370370 | stale remote phs @ e9056c2 | workflow_dispatch (09-02 D-16) | cancelled — dispatched before push, ran 12-commits-stale code exercising nothing from the phase |
| 37335797121 | phs @ e85eb73b | workflow_dispatch (09-02 D-16) | cancelled by the orchestrator 2026-10-06 (owner default) — it could serve as neither the D-02 baseline (predates Task-1 job shape 9fb853c) nor D-17 evidence; verified completed/cancelled before the baseline below was dispatched |
| **37345067326** | phs @ 9fb853c (final phase HEAD) | workflow_dispatch (09-04 D-02) | **the baseline census run of record — measured below** |
| 37427527257 | phs @ aa4a6e7 | workflow_dispatch (09-04, post-timeout-bump) | CANCELLED by the orchestrator while still queued (superseded by the model swap) — NOT evidence of anything; the cancellation is bookkeeping, not a failure |
| **37432001711** | phs @ 170e86f | workflow_dispatch (orchestrator, 15:48 CST) | **the FINAL verification run of record** — carries the plot fix (550d311) + the timeout bump (aa4a6e7) + the qwen3.5:4b model swap (261006-lhm); D-17/D-18 closure evidence |

Baseline dispatch run id: 37345067326

## Authoritative census counts (D-02)

**Run 37345067326 (phs @ 9fb853c), example-nightly leg — the baseline census of record.**

- Stage 0.5 pin held green on the runner, measured: stage-0.5 census collection: 193/202 tests collected (9 deselected) in 1.28s
  (identical to the 09-01-pinned / 09-02-bumped literal in ci.yml — zero pin drift; deselect split unchanged at 8 mcp + 1 giants; run 37377004230 reproduced it in 4s).
- **Stage 1: 192 passed / 1 skipped / 0 failed / 9 deselected in 7176.79s (1:59:36)**
  (junit `pytest-junit-example-stage1.xml`, 193 items). The single skip is the benign
  no-imports entry (predict_data) — the permanent baseline skip; **zero unexpected
  skips** (both stage-4 audits green on the stage-1 junit). Census arithmetic: the
  08-09 full-census 196P/1S grew to 201P/1S with 09-02's 5 contract tests, minus the
  8 mcp deselected (stage 3 owns them) minus the 1 giants-deselected evo execution
  test (D-01) = **192P/1S** — measured, matching the plan's ~195-expectation family
  (the plan's ≈190 baseline predated the 09-01/09-02 census growth arithmetic).
- Stage 1 YAML leg: green (`tests/configuration/test_yaml_load.py`, junit recorded).
- Stage 2 streamable probes: **3 passed in 4.86s**. Stage 2 sse: FAILED on probe
  shape (see Deviations — the server was healthy; fixed at 5dfadda, exercised in
  run 37377004230). Stage 3 mcp pair (ollama at its EXISTING ~256k ctx, D-06
  deferred): **2 passed in 442.31s (7:22)**.
- test-mamba leg, same run: **1833 passed / 1 failed / 1 skipped / 53 deselected in
  114.21s (1:54)** — kernels compiled and installed green; the single failure is the
  fenced pre-existing `test_plot_for_regression` (see Deviations), byte-identical to
  the fast-lane state measured at 09-02 plan time.
- coverage-nightly leg, same run: **1928 passed / 4 failed / 15 skipped in
  5234.50s (1:27:14)** — 3 of the 4 failures are the missing-bedtools gap (fixed at
  decd2cc), the 4th the same fenced pre-existing failure. The 15 skips are the
  allowlisted typed set (6 MCP localhost probes, gated optional-dep entries in this
  leg's `.[base,fla]` venv — OQ1 posture — and the benign no-imports entry).

**Final census statement (the authoritative post-wiring triple under the final
selectors `-m "not giants"` -k `"not mcp_example"`): 193/202 tests collected
(9 deselected); 192 passed + 1 benign skip at execution; 0 failed.**

**D-05 epochs-cut measurement (first live runner exercise):** the
`finetune_custom_head` gated execution measured **566.3s (9:26)** at sandbox
`num_train_epochs: 1` — consistent with (slightly better than) the ~11 min
projection from the 09-02 A/A decision; no runner-side 3-epoch A/B exists (the
~31 min figure was dev-box), and the notebook's committed content still reads
`num_train_epochs: 3` (the sandbox-patch honesty contract held — junit green,
`test_finetune_custom_head_spec_pins_the_epochs_cut` passed in the same run).

**D-06 deferral measurement (the owner's revisit data):** the mcp pair at its
EXISTING ~256k ctx behavior measured **442.31s (7:22)** on the clean runner
(115 Gi available entering stage 3; the ~36GB kv-cache concern did not
materialize as a latency or memory-floor problem on the 128GB GB10 in this
run — the 08-08 dev-box ~25 min figure was measured under VRAM contention,
not comparable). The num_ctx cut stays DEFERRED; if latency/VRAM ever
regresses on a contended box, this is the baseline to compare against.

## Measured budgets (D-12)

All figures below are MEASURED wall actuals (whole minutes/seconds from run logs
and the jobs API), never proportional estimates. The ci.yml budget comments cite
these run ids; every number is reproducible from the run's own logs/artifacts.

### example-nightly (per-stage; run 37345067326 @ 9fb853c = baseline, run 37377004230 @ 5dfadda = green gate at the fixed wiring)

| stage | run 37345067326 | run 37377004230 | note |
| --- | --- | --- | --- |
| stage 0 (venv + extras + bedtools + inventory) | ~44s | ~50s | bedtools from cached prefix ~15s; uv warm |
| stage 0 mamba kernel wheelhouse | 22:28 | 23:52 | cache MISSED both runs (store at 10GB threshold throttles saves — the build is a recurring measured cost, not amortized; owner disposition = OQ4) |
| stage 0 megaDNA provisioning | 7s | 7s | pinned clone + venvs from warm local caches |
| stage 0.5 census assertion | 4s | 4s | collect 1.28s; pin green both runs |
| stage 1 examples + YAML | 1:59:46 | 1:57:24 | 192P/1S/9 deselected both runs (1:59:36 / 1:57:14 pytest) |
| stage 1.5 hygiene | 6s | 5s | ~115 Gi available, floor pass |
| stage 2 streamable probes | 37s | 38s | server load ~25s + 3P in 4.86/4.94s |
| stage 2 sse probes | 17:58 (FAILED — pre-fix probe) | **41s (3 passed in 4.75s)** | the SSE-probe repair (5dfadda) proven live: 17:58 of blind polling → streamable-scale cost with the 3 sse probes actually EXECUTING (they never ran before the fix) |
| stage 2.5 hygiene | 5s | 6s | floor pass |
| stage 3 mcp pair | 7:54 | 8:01 | 2P in 442.31/449.32s at un-cut ~256k ctx (D-06 deferred) |
| stage 4 audits + upload + summary | 3s | 3s | all audits 0 in run 2 — the complete green ledger |
| **job total (step pipeline)** | **2:49:53** | **2:42:46** (+ ~11 min post-job cache save; a ~33-min trailing runner-side log upload extended the job record to 3:15:34 — stages are the budget) | one-time/defect costs inside run 1: 22:28 wheelhouse + 17:58 broken sse wait |

### coverage-nightly (runs 37345067326 + 37377004230)

- Run 1 (bedtools gap — NER/CRE/script failed fast): install ~4 min (warm uv);
  census (full 1947-item tree, slow included) **1:27:14** (5234.50s); job total
  **≈ 88 min** (16:59:49 → ~18:28).
- Run 2 (re-dispatch, bedtools restored from the shared cache prefix in ~20s):
  census **1:40:35** (6035.89s; the bedtools trio now executes: 1933P/1F/15S);
  job total **1:49:02** (01:03:23 → 02:52:25). The longer census is the honest
  cost of those tests passing instead of failing fast.
- Paper-ceiling recomputation from the current test files (marks above the 300s
  global default, per-item multiplicities from a fresh full-tree collection): the
  full-tree sum is ≈ 3920 min — dominated by the tests/examples execution marks
  (13×7200s class + 8 gated 3600/7200s + 3×7200s marimo + 4×3600s script +
  showcase 2400+5400+2400s), most of which typed-skip at gate time in THIS job's
  `.[base,fla]` venv (no megaDNA/mamba/giants prereqs) and are example-nightly's
  lane; the bind-here set (real-model 5×1800s + remote-code 1800s + smokes
  2×1800s + trainer 2×7200s+3×3600s + 7200s fn + inference 3600s + model 2×900s
  + mcp 3600s) sums to ≈ 640 min. Both figures sit below the marks' own
  per-test protection logic — the D-12 relationship (per-test marks PRIMARY,
  job kill BACKSTOP) is unchanged; the decision is recorded in the ci.yml
  comment.

### test-mamba (runs 37345067326 + 37377004230)

- Run 1 job total **10:49** (21:18:05 → 21:28:54): install incl. the recurring
  kernel source build (`--no-cache-dir --no-build-isolation`) **8:29**, census
  (`not slow`) **2:00** (114.21s; 1833P/1F/1S).
- Run 2 (the D-18 re-dispatch at 5dfadda): census **1:53** (113.73s;
  **1835P/1F/1S** — the +2 passed are 09-04's SSE contract tests; the 1 failure
  is the fenced pre-existing `test_plot_for_regression`, unchanged).
- The recurring kernel build is the long pole exactly as the ci.yml comment
  states; the 180-min timeout is measured comfortable (build ~8.5 min + census
  ~2 min on this box — the 180 figure keeps worst-case cold-compile headroom).

## Requirement-equivalence note (CI-07 VRAM assertion)

CI-07's VRAM assertion is **DELIVERED as the Stage 1.5/Stage 2.5 `free -g`
available-memory floor with the `>=35Gi` literal** — the GB10-operative equivalent.
The runner inventory (08-09 dispatch 2) proved GB10 reports no memory through
`nvidia-smi` (`NVIDIA GB10, [N/A]`), so an nvidia-smi-based VRAM assertion would be a
gate that measures nothing; the `free -g` available column is the metric the box
actually exposes, asserted hard (`exit 1` below 35) with before/after value logging,
while nvidia-smi stays telemetry-only. Literal phase verification should read the
floor as the VRAM assertion's delivered form.

## Deviations

**0. [Owner decision 2026-10-06 deferral] Task-2 precondition (D-06 owner re-apply
live) VOIDED**
- The plan's Task-2 precondition required live `systemctl show ollama -p Environment`
  evidence of `OLLAMA_CONTEXT_LENGTH=8192` before the baseline dispatch. The owner
  deferred the num_ctx cut entirely (2026-10-06 00:52 CST, superseding the 00:42
  per-request rework idea; recorded in STATE.md), so the precondition is void, the
  live service was NOT touched, no per-request rework landed (quick task 261006-114
  stopped pre-plan), and the baseline measured the UN-CUT mcp-pair behavior. Every
  plan-text promise of num_ctx savings is superseded by this record: the D-12
  comments state the un-cut measured reality and mark the cut DEFERRED
  (revisit-with-data). The `09-USER-SETUP.md` sudo re-apply item stays open but
  not-required-now (it also remains the only path to restoring the drifted live
  `OLLAMA_HOST=0.0.0.0` bind, T-09-03).

**1. [Rule 3 - blocking environment gap] coverage-nightly had no bedtools — 3 of
its 4 first-live failures**
- **Found during:** run 37345067326 (Task 2 baseline dispatch; the coverage-nightly
  leg's first live execution past checkout in the phase-8/9 era).
- **Issue:** the full-tree census executes the NER data_generation notebook
  (`intersectBed` disabled), the CRE showcase bands test (`shutil.which("bedtools")
  is None`), and the script lane (same intersectBed class) — coverage-nightly never
  carried the bedtools provisioning that example-nightly got in 08-09 (commit
  ea1bfd6); every prior dispatch of this leg died at checkout/evo before reaching
  the census, so the gap was invisible until now.
- **Fix:** decd2cc — the proven rootless steps (cached `.bedtools-env` prefix +
  micromamba/bioconda fallback onto `GITHUB_PATH`, shared cache key) mirrored into
  coverage-nightly between the numpy install and the census.
- **Verified by:** run 37377004230 (the fixed wiring: those 3 failures gone).
- **Note:** this was NOT a transient-egress failure as the plan's D-18 framing
  assumed — the "transient" classification (died at Checkout code) was true only of
  the death point; behind it sat a real environment gap.

**2. [Rule 1 - wiring bug] the stage-2 SSE readiness probe could never succeed**
- **Found during:** run 37345067326 — `stage2-mcp-probes-sse=1`, the sole
  example-nightly failure.
- **Issue:** the probe gated on curl's EXIT CODE (`curl -m 2 ... && break`), but an
  SSE endpoint answers 200 + `text/event-stream` and holds the body open forever —
  curl always dies at the 2s cap (exit 28) against a perfectly healthy server. The
  D-14 failure scene proves it: `mcp-server-sse.log` shows the server fully up at
  04:52:17 (3 models loaded from warm cache, `Uvicorn running on :8000`, zero
  errors) while the probe polled blind for 17:58 and recorded failure. The stage
  had never executed on the runner before (all prior dispatches died upstream), so
  the bug was latent since the 08-09 wiring.
- **Fix:** 5dfadda — readiness now asserts the RECEIVED HTTP status
  (`-w "%{http_code}"`, gate on `200`; `000` when nothing answered), with
  same-change contract tests (`TestExampleNightlySseProbe`, 2 tests pinning the
  http_code shape and forbidding the always-fail shapes).
- **Verified by:** run 37377004230 (stage-2 sse green at streamable-scale cost).

**3. [Fenced pre-existing failure - owner triage pending] `test_plot_for_regression`
is the sole remaining red on test-mamba and coverage-nightly**
- **Found during:** run 37345067326 (both legs), identical to the fast-lane state
  proven pre-existing at 09-02 plan time (WINDOWS.md id 16, `unmet-truth`).
- **Diagnosis (read-only, not fixed per the phase fence):**
  `tests/benchmark/test_benchmark.py:454` → `dnallm/inference/benchmark.py:617` →
  `dnallm/inference/plot.py:234` — `dbar[metric].astype(float)` over a
  dict-valued metric column (the quick-task 13/14 Mapping-config fallout family);
  `TypeError: float() argument must be a string or a real number, not 'dict'`.
- **Disposition: RESOLVED mid-phase by owner-dispatched quick task 261006-cum**
  (commit 550d311, 2026-10-06 01:32Z — the test retasks through the engine-owned
  config; test-side only, `tests/benchmark/` 31/31 green locally; WINDOWS.md
  id 16 flipped fixed). Recorded here per the no-silent-drop rule; the
  final-dispatch confirmation (run 37406829738) proves it census-wide on the
  runner. It never touched example-nightly (the D-17 green gate was unaffected).

**4. [Owner decision B, 2026-10-06 - timeout line matches the un-cut budget]
stage-3 mcp-pair cell timeout raised 1800 -> 3600s**
- **Found during:** run 37406829738 (the post-plot-fix dispatch @ 8774151) —
  example-nightly red with `stage3-mcp-pair=1` as its ONLY non-zero entry:
  the pydantic_ai notebook's analysis cell hit `CellTimeoutError` after
  exactly 1800s (`traitlets client.py:845 Timeout waiting for execute reply`),
  `1 failed, 1 passed in 2099.56s`. Everything else green (census 192P/1S,
  both probe batches, all audits) — the same un-cut ~256k ctx lane that
  measured ~8 min in runs 1/2 showed a >30-min latency tail this run.
- **Root contributor:** the deferred num_ctx cut (owner 2026-10-06 00:52) —
  qwen3.8 serves the pair at 256k ctx; the fixed 1800s cell line sat below
  the un-cut tail's variance.
- **Fix (owner decision B, "timeout line matches the un-cut budget reality
  recorded in D-12"):** aa4a6e7 — the seam is `run_notebook`'s
  `NotebookClient(timeout=cell_timeout)` (the exact trait behind the
  traitlets error); both mcp_example spec entries raised to 3600s, and the
  pair joined `_TIMEOUT_7200_GATED` so the outer pytest-timeout mark stays
  strictly ABOVE the cell budget (the harness contract: an outer kill at the
  cell budget would preempt nbclient's clean CellTimeoutError handling and
  the partial-failure artifact capture). No contract test pinned the old
  value; the census triple is unaffected (marks do not change collection).
  Revisit when/if the num_ctx cut is un-deferred.
- **Verified by:** the final verification run 37432001711 (the interim
  post-fix dispatch 37427527257 was cancelled queued by the orchestrator —
  superseded by the model swap below — and is not evidence).

**5. [Inherited owner decision 2026-10-06 15:27 CST - model swap] mcp_example
pair agent model qwen3.8:latest -> qwen3.5:4b (quick task 261006-lhm)**
- Landed mid-phase (commits febb627..170e86f, pushed) while 09-04's runner
  cycles ran: an 11-file sweep (both mcp notebooks + docs mirrors + runner
  README + one ci.yml comment line + the spec/test comment layer), RED-GREEN
  contract battery 231 passed, docs sync gates 24/24, and a live capability
  probe PASSED 15:24 CST (3-turn tool-calling: 34.8s cold / 6.1s / 5.1s warm).
- The timeout bump (deviation 4) SURVIVES the swap by owner direction: the
  4.2B Q4_K_M brain (~3.3GB) still defaults to a 262144 context — the
  ~256k-class un-cut reality persists, so `cell_timeout 3600` + the 7200s
  outer override stay as headroom; stage 3 is expected minutes-scale now
  (measured by the final run).
- D-11 re-decision recorded in STATE.md by the quick task; the num_ctx
  deferral (deviation 0) is unchanged and still the revisit-with-data item.

**Run-2 confirmations (37377004230, phs @ 5dfadda):** example-nightly GREEN
end-to-end — every stage-results entry 0, both audits green on every junit,
stage-2 sse 3 passed in 4.75s (the repair proven live), census reproduced
exactly (192P/1S/9 deselected, 1:57:14). test-mamba re-dispatched: 1835P/1F/1S
— red by exactly the fenced item. coverage-nightly re-dispatched: 1933P/1F/15S
in 1:40:35 (job 1:49:02) — the bedtools trio now executes green (restored
from the shared cache prefix in ~20s, "bedtools v2.31.1"), red by exactly the
fenced item.

**Run-3 record (37406829738, phs @ 8774151 — post-plot-fix dispatch):**
test-mamba **GREEN** (1837 passed / 1 skipped in 1:52, job 10:48 — the first
fully-green mamba leg; 1837 = 1835 + the quick task's 2 pin tests).
coverage-nightly **GREEN** (1935 passed / 15 skipped in 1:58:18 — the plot
fix held census-wide; the "red coverage" in circulation described run 2's
pre-fix state). example-nightly red by ONLY the stage-3 cell timeout
(deviation 4 above; census and probes all green again).

## FINAL verification run (37432001711, phs @ 170e86f) — ALL THREE LEGS GREEN

- **test-mamba: SUCCESS** — 1840 passed / 1 skipped / 53 deselected in 1:53
  (1840 = 1837 + the model-swap quick task's 3 pins).
- **example-nightly: SUCCESS** — the complete green ledger ("OK: every
  recorded stage item exited 0"; every junit audit green). Census pin green
  (193/202, 9 deselected); stage 1 **192 passed / 1 skipped in 1:56:53**;
  YAML 21 passed; both probe batches green (sse 3 passed — the repaired
  probe's second consecutive green); **stage 3 mcp pair 2 passed in 87.10s
  (1:27) at qwen3.5:4b** (the model swap collapsed the qwen3.8-era ~8 min /
  >30-min-tail lane to minutes-scale, inside the 3600s cell budget with
  huge headroom); both hygiene floors green. Job 08:51:03 → 11:00:58 =
  **2:09:55 total**, and the mamba wheelhouse cache HIT for the first time
  (build step 1s vs the 23-24 min misses of runs 1-2) — the measured
  steady state is now ~2:10 with the build amortized.
- **coverage-nightly: SUCCESS** — **1938 passed / 15 skipped / 0 failed in
  1:42:18; required coverage 96.42%** (floor 90). Job 11:01:02 → 12:52:08
  = 1:51:06.

**D-17: MET at the final wiring (green complete example-nightly dispatch,
run 37432001711). D-18: MET — both transient legs green in-phase at the
final head (test-mamba 1840P/0F; coverage-nightly 1938P/0F @ 96.42%).**

## Phase closure records (09-04 Task 3)

Green-gate dispatch run id: 37432001711

- **D-17 green-run gate: MET — at the FINAL wiring.** Run 37432001711's
  example-nightly (phs @ 170e86f: plot fix + SSE-probe repair + bedtools +
  timeout bump + qwen3.5:4b swap) completed green end-to-end: stage-4 summary
  "OK: every recorded stage item exited 0", every junit audit green, census
  192P/1S/9-deselected, both D-13 hygiene floors passed, sse probes green,
  stage-3 pair green. (The FIRST complete green dispatch was run 37377004230
  @ 5dfadda — a superseded wiring shape, kept in the ledger below.)
- **D-18 transient-leg closure: MET — both legs GREEN in-phase at the final
  head.**
  - The re-dispatch chain: run 37377004230 proved the legs COMPLETE past
    their "transient" checkout-death facade and exposed the real defects
    behind it (bedtools gap decd2cc; plot failure — owner-fixed at 550d311;
    stage-3 timeout — owner decision B, aa4a6e7; model swap 261006-lhm).
  - coverage-nightly re-dispatch run id: 37377004230 — 1933P/1F/15S in
    1:40:35 (the bedtools-fix proof run; its single failure was then the
    unfenced-again plot test, fixed the same day). **GREEN closure: run
    37432001711 — 1938 passed / 15 skipped / 0 failed in 1:42:18 at 96.42%
    coverage.**
  - test-mamba re-dispatch run id: 37377004230 — 1835P/1F/1S (same single
    fenced failure). **GREEN closure: run 37432001711 — 1840 passed /
    1 skipped / 0 failed in 1:53.**
- **D-20 bookkeeping (no repo action):** the dependabot torch-ignore rules
  landed earlier at commit 16a9ffb; PR #42 stays open as a record, not
  merged.
- **D-16/D-17 dispatch evidence integrity (T-09-10):** every run id above is
  backed by the cited run's own logs/junit (stage-4 ledger, census-collect
  line, step timings from the jobs API); no outcome is asserted without its
  run. The cancelled dispatches (37335797121, 37427527257) are recorded as
  bookkeeping, never as evidence.

## Open dispositions / deferrals

- **num_ctx cut deferred** (owner decision, see the D-06 row above). Revisit with the
  stage-3 measured numbers below in hand; if re-decided, either the owner runs the
  `09-USER-SETUP.md` re-apply (restoring the loopback pin, T-09-03) or a per-request
  injection quick task lands (nothing for it was merged — quick task 261006-114 was
  stopped pre-plan).
- **OQ1 (coverage-nightly purity)**: coverage-nightly keeps its unfiltered full-suite
  run this phase — the evo gated test typed-skips there (cold venv, allowlisted
  `optional-dep:`), which stays honest. Census purity via one `-m "not giants"` flag
  remains a possible later change.
- **OQ4 (cache-store posture)**: out of scope — the two `Linux-uv-cuda-*` store
  entries (10.37GB, at the eviction threshold) are owner discretion; no deletion or
  pruning happens from this phase (owner rule).
- **D-20 bookkeeping (no repo action)**: dependabot torch-ignore rules landed at
  16a9ffb; PR #42 stays open as a record, not merged.
- **Pre-existing fast-lane failure** `test_plot_for_regression` (WINDOWS.md id 16,
  `unmet-truth`): proven pre-existing at 09-02 plan-start HEAD; NOT this phase's to
  fix; it does not touch the example-nightly lane.
