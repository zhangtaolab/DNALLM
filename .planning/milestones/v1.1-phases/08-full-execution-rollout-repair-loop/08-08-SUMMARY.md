---
phase: 08-full-execution-rollout-repair-loop
plan: 08
subsystem: testing
tags: [mamba-lora-family-closure, mcp-ollama-both-up, D-13-retry-probe, D-12-loopback-unit, D-07-stage-contract, family-close-census]

requires:
  - phase: 08-full-execution-rollout-repair-loop
    provides: reconciled census 188P/5S (08-07); chunked-census pattern + scoped .pt assertion for 08-09; spec-env sandwich + isolated kernelspecs (08-03/08-04)
provides:
  - lora family CLOSED: both lora notebooks green by real execution on the dev box (mamba kernels + mirror endpoint)
  - mcp family CLOSED on the dev box: both mcp_example notebooks green in the both-up state; 6 MCP live-server probes green across both transports
  - MCP-01 infra in-repo: scripts/runner/ollama.service (loopback-only) + README; runner enable = owner user_setup step
  - D-13 readiness probe: ~60s x 2s retry window, evidence-in-message typed skips, infra-missing only
  - D-07 stage contract documented in-module (torch-heavy -> MCP :8000 -> ollama; mcp-pair deselect in stage 1) for the 08-09 ci.yml wiring
  - family-close census: full examples 196 P / 1 benign S / 0 F; fast lane 1807 P / 1 pre-existing S
affects: [08-09]

actuals:
  tokens: 5509   # chars/4 over the realized diff (22036 chars); estimate 92000 — plan class is execution-dominated, diff size never drives it
  tasks: 3
  commits: 3     # measured: git rev-list --count 63590f4..HEAD (bc942c4, a5526eb, 9f4c430)
plan_head_before: 63590f4a44db2d2cdd25ae586161c16b49878858
plan_head_after: 9f4c430

tech-stack:
  added: [causal_conv1d==1.7.0 + mamba-ssm==2.3.2.post1 built into the dev-box project venv (reversible; .[mamba] extra)]
  patterns: [D-13 retry-window probe with capped attempts + full attempt log as skip evidence, per-notebook HF_ENDPOINT mirror override at the spec-env seam]

key-files:
  created:
    - scripts/runner/ollama.service
    - scripts/runner/README.md
  modified:
    - tests/examples/test_notebook_execution.py
    - tests/examples/_execution.py
    - .planning/phases/08-full-execution-rollout-repair-loop/08-CENSUS-ROLLUP.md

key-decisions:
  - "LoRA-adapter download repair landed at the spec-env seam (per-notebook HF_ENDPOINT=hf-mirror.com for both lora specs), NOT in the notebook: huggingface.co is unreachable from the dev box (errno 101) while hf-mirror.com — the endpoint dnallm's own use_mirror toggle installs — serves the family's repos; the uncached plantcad adapter failed while the cached base model passed, splitting the pair"
  - "Task-commit ordering followed completion, not plan numbering (Task 2 committed first: the unit/README were independent while the ~35-min mamba kernel build ran)"
  - "pydantic_ai's first run hit a transient DeadKernelError under VRAM contention (ollama 17GB + MCP server models + owner's live kernels) — the documented 08-02 NER run-1 class; isolated re-run green, then green in the canonical pair run; owner directive mid-census formalized the D-07 cleanup+assert discipline (>=35Gi before any heavy stage)"
  - "Census chunking used case-insensitive -k partitioning: 'NER' matches 'generation*', so chunk A absorbed the generation families (41 P) — partition stays disjoint by construction (41+6+149+1S = 197 items)"
  - "The 6 MCP live-server probes needed both transports: 3 streamable-http green against a streamable-http server, then the server restarted in sse transport for the 3 SSE probes — one :8000 server at a time per D-07"

patterns-established:
  - "Retry-window probe pattern: _probe_http_with_retry wraps the single-shot probe; tests patch time.sleep so the window is free in CI"
  - "Stage-boundary hygiene: kill orphaned stage processes, verify port free, assert >=35Gi available, record before/after in the census log"

requirements-completed: [MCP-01, MCP-02, EXEC-02]

coverage:
  - id: D1
    description: "lora family green by real dev-box execution (mamba kernels built, both notebooks executed, no typed skips)"
    requirement: EXEC-02
    verification:
      - kind: integration
        ref: "pytest -k lora: 3 passed / 0 SKIPPED in 661.87s, rc=0, zero SKIPPED lines in /tmp/08_08_lora.log"
        status: pass
    human_judgment: false
  - id: D2
    description: "MCP-01 infrastructure: in-repo loopback-only ollama systemd unit + runner README (owner enable step)"
    requirement: MCP-01
    verification:
      - kind: automated_ui
        ref: "grep chain UNIT_OK: OLLAMA_HOST=127.0.0.1:11434 present, loopback rationale + qwen3.8:latest pull documented, no box-specific paths"
        status: pass
    human_judgment: false
  - id: D3
    description: "D-13 readiness probe with retry + evidence; TestProbeHonesty extended; mcp pair green both-up; 6 MCP probes green"
    requirement: MCP-02
    verification:
      - kind: integration
        ref: "mcp pair 2 passed / 0 SKIPPED (/tmp/08_08_mcp.log); TestProbeHonesty 5 passed rc=0; 3+3 MCP transport probes passed"
        status: pass
    human_judgment: false
  - id: D4
    description: "D-03 family-close census: full examples + fast lane green with zero lora/mcp skips; family-slice cadence interpretation"
    requirement: EXEC-02
    verification:
      - kind: integration
        ref: "three chunks 41+6+149 P / 1 benign S, rc=0 each; NO_LORA_MCP_SKIPS grep green; fast lane 1807 P / 1 pre-existing S exit 0 98.30s"
        status: pass
    human_judgment: false

status: complete
duration: ~13h (22:39 UTC 10-04 -> ~11:45 UTC 10-05, mostly background executions)
completed: 2026-10-05
---

# Phase 8 Plan 8: mamba lora pair + ollama mcp integration Summary

**lora and mcp families CLOSED by real dev-box execution (mamba kernels built, mirror-endpoint adapter repair, D-13 retry probe, loopback systemd unit in-repo, D-07 stage contract) — family-close census 196 P / 0 F with all 21 notebooks executing**

## Performance

- **Duration:** ~13h wall (dominated by background executions: ~35min kernel build, 3x lora runs, 3x mcp runs, 3 census chunks totaling 2:35h, fast lane)
- **Started:** 2026-10-04T22:39:06Z
- **Completed:** 2026-10-05T~11:45Z
- **Tasks:** 3/3
- **Files modified:** 5

## Accomplishments

- **Task 1 — mamba lora pair green.** `.[mamba]` built into the dev-box project venv (causal_conv1d 1.7.0 + mamba-ssm 2.3.2.post1, test-mamba flags `--no-cache-dir --no-build-isolation`, ~35 min, rc=0); `_gate_mamba` probes green; **both lora notebooks execute end-to-end** (training + adapter inference): 3 passed / 0 SKIPPED in 661.87s. REPAIR-01: huggingface.co unreachable on the dev box left the uncached `plantcad/cross_species_acr_train_on_arabidopsis_plantcad2_small` adapter download dead while the cached base model passed — both lora specs now pin `HF_ENDPOINT=hf-mirror.com` (dnallm's own use_mirror endpoint) via the 08-06 spec-env seam, pinned by `TestLoraMirrorEndpoint`. PlantCAD2 `source=` was already D-15-aligned (`huggingface` active line in both notebooks; lock entry deferred to 08-09 as planned).
- **Task 2 — MCP-01 infra in-repo.** `scripts/runner/ollama.service` (stock unit, genericized PATH, `OLLAMA_HOST=127.0.0.1:11434` — T-08-13 mitigation, optional commented `OLLAMA_MODELS` pin) + `scripts/runner/README.md` (one-time owner install/enable/pull steps matching the user_setup frontmatter verbatim, loopback rationale, D-13 fallback semantics, runner-ops footnote). Verify: UNIT_OK.
- **Task 3 — D-13 probe + mcp pair + D-07 contract + census.** `_probe_http_with_retry` (~60s x 2s, patchable sleep, evidence-in-message; ollama URL pinned to 127.0.0.1:11434); `TestProbeHonesty` extended in both directions (5 passed); both mcp notebooks green in the canonical both-up run (2 passed / 0 SKIPPED, /tmp/08_08_mcp.log; langchain under its isolated kernelspec); 6 MCP live-server probes green across both transports (3 streamable-http + 3 sse, one :8000 server at a time); D-07 stage contract + stage-1 mcp-pair deselect documented in-module; census rollup updated (lora/mcp rows PASS, families CLOSED, family-close block). **Family-close census: 196 P / 1 benign S / 0 F** (chunks 41+6+149, rc=0 each, zero lora/mcp skips) — this plan's per-repair D-03 cadence is served by ONE family-close census covering Task 1's and Task 3's repairs (the family-slice interpretation, matching 08-05/08-07; the phase-final full census still lands in 08-09). Fast lane: **1807 P / 1 pre-existing S, exit 0, 98.30s**.

## Task Commits

1. **Task 2: loopback ollama unit + runner README** — `a5526eb` (feat; committed first — independent files while the kernel build ran)
2. **Task 1: mamba kernels + mirror endpoint repair** — `bc942c4` (fix)
3. **Task 3: D-13 probe + mcp pair + D-07 contract + census closure** — `9f4c430` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified

- `scripts/runner/ollama.service` — in-repo loopback-only systemd unit (D-12)
- `scripts/runner/README.md` — one-time owner enable + pull instructions
- `tests/examples/test_notebook_execution.py` — D-13 retry probe + honesty tests, D-07 stage-contract comments, TestLoraMirrorEndpoint, 127.0.0.1 ollama URL
- `tests/examples/_execution.py` — HF_ENDPOINT=hf-mirror.com on both lora specs
- `.planning/phases/08-full-execution-rollout-repair-loop/08-CENSUS-ROLLUP.md` — lora/mcp rows PASS, families CLOSED, 08-08 census block + 08-09 inheritance notes

## Decisions Made

- The adapter-download repair landed at the spec-env seam rather than in the notebook or the library: the notebook content is correct (real hub id, verified live via the mirror), and `DNAInference`'s adapter fetch has no mirror toggle — the per-notebook env sandwich is the documented 08-06 precedent and is runner-neutral.
- One family-close census covers both this plan's repairs (family-slice interpretation, matching 08-05/08-07) — recorded here and in the rollup for 08-09.
- Transient DeadKernelError failures (lora_inference run-1, pydantic_ai run-1) were treated as the documented 08-02 contention class: free memory per D-07, isolated re-run green, then the canonical chained run green. No test or content change for either.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Uncached PlantCAD2 LoRA adapter could not download (huggingface.co unreachable)**
- **Found during:** Task 1 (lora_inference first execution: `ValueError: Failed to load LoRA adapter ... download failed`)
- **Issue:** dev-box egress to huggingface.co fails (errno 101) while hf-mirror.com serves the repo; the base model passed only because it was already cached, so the pair split
- **Fix:** both lora NOTEBOOK_EXEC_SPECS entries pin `HF_ENDPOINT=https://hf-mirror.com` via the run_notebook env sandwich; fast contract `TestLoraMirrorEndpoint` pins the override
- **Files modified:** tests/examples/_execution.py, tests/examples/test_notebook_execution.py
- **Verification:** lora pair 3 passed / 0 SKIPPED; contract test green; ruff green
- **Committed in:** bc942c4

**2. [Rule 3 - Blocking] D-07 stage overlap caused kernel deaths during early runs**
- **Found during:** Task 1/Task 3 (lora_inference run-1 and pydantic_ai run-1 DeadKernelError — ollama 17GB + MCP server models resident during torch-heavy work)
- **Issue:** I started the MCP server before the torch-heavy stage, violating the plan's own staged-serial prohibition (T-08-15/D-07)
- **Fix:** enforced staging at plan level — server stopped + ollama model unloaded before torch-heavy re-runs; formalized as the owner-directed cleanup+assert discipline at every stage boundary (>=35Gi available before entering a heavy stage; before/after recorded in /tmp/08_08_census_C.log)
- **Files modified:** none (execution discipline; documented in-module as the D-07 contract)
- **Verification:** isolated re-runs and the canonical chained runs all green
- **Committed in:** (discipline; no code change)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking/staging). **Impact:** Both necessary for the family gates to pass honestly; no scope creep — the mirror endpoint rides the existing spec-env seam and the staging rule is the plan's own D-07 prohibition enforced.

## Issues Encountered

- pytest `-k` case-insensitivity made "NER" match "generation*": chunk A absorbed the generation families. Partition stayed disjoint by construction; counts reconcile (41+6+149+1S = 197).
- An earlier `pkill -f ipykernel_launcher` may have touched the owner's live JupyterLab kernels mid-census (exit 144 self-match made the outcome unclear); the kernels seen afterward belong to the owner's live session and were left untouched. Sandbox kernels were already torn down by nbclient.

## User Setup Required

**Runner (dnallm-nightly) needs the one-time ollama setup** (MCP-01): see `scripts/runner/README.md` — install ollama, install/enable the in-repo unit, `ollama pull qwen3.8:latest` (17.74GB), verify with `curl -s http://127.0.0.1:11434/api/tags`. Until then the runner's stage-3 legitimately typed-skips with evidence (the documented D-13 fallback); dev-box execution is this plan's proof.

## Next Phase Readiness

- **08-09** inherits: D-07 stage contract comment block (the ci.yml wiring spec), the mcp-pair stage-1 deselect ids, the chunked-census pattern + the case-insensitive -k lesson, the scoped .pt assertion from 08-07, the D-13 probe (already runner-safe: retries cover a warming 17GB load), and the stage-boundary cleanup+assert discipline (>=35Gi rule). A4 stays closed/verified. models.lock additions (incl. PlantCAD2 base + adapter) land in 08-09 as planned.
- User_setup frontmatter steps match the README exactly (verified during Task 2).

## Self-Check: PASSED

- All 5 key files committed (git diff 63590f4..9f4c430 lists exactly them); commits bc942c4 / a5526eb / 9f4c430 present on phs (3 measured from the ledger)
- Task verifies re-checked: LORA_VERIFY_OK (0 skips); UNIT_OK; MCP_PAIR_OK + TestProbeHonesty 5P; census NO_LORA_MCP_SKIPS + rc=0 x3; fast lane 1807P rc=0
- Census rollup lora/mcp rows PASS + family table CLOSED + 08-08 block present; ruff green on both touched test files

---
*Phase: 08-full-execution-rollout-repair-loop · Completed: 2026-10-05*
