---
phase: 08-full-execution-rollout-repair-loop
plan: "09"
subsystem: infra
tags: [github-actions, models-lock, cache-quota, nightly-ci, mcp-server, ollama, bedtools, wheel-cache]

requires:
  - phase: 08-full-execution-rollout-repair-loop (plans 01-08)
    provides: example-nightly skeleton, all families green on the dev box, census baselines
provides:
  - Revision-pinned, prefix-aligned models.lock covering every real-executed model id (24 rows)
  - Completed example-nightly job (prereq steps incl. bedtools rootless, wheel caches, giants prefetch, stages 2/3)
  - First-dispatch consumption record + cache-quota measurement + final D-03 census (196P/1S/0F)
affects: [09-ci-hardening-and-docs, phase-9 CI-06/CI-07 runtime budgeting]

actuals:
  tokens: 8700        # chars/4 over the realized diff (34,740 chars across models.lock/ci.yml/rollup)
  tasks: 3
  commits: 3          # MEASURED: git rev-list --count a98da4e..HEAD at SUMMARY time
  plan_head_before: a98da4eddc9b16a132095fc09c8934de95b8024f
  plan_head_after: 6384805

tech-stack:
  added: []            # no new libraries — job-step installs only (micromamba bedtools, pinned evo/megaDNA stacks)
  patterns:
    - "Cached-wheelhouse source builds (flash-attn/mamba) keyed on version+arch+torch — nightly compile tax paid once"
    - "Rootless system deps via micromamba/bioconda prefix onto GITHUB_PATH (no sudo/apt on the runner)"
    - "One :8000 MCP server at a time across stages (transport restart contract); readiness window sized for eager model loads"

key-files:
  created: []
  modified:
    - models.lock
    - .github/workflows/ci.yml
    - .planning/phases/08-full-execution-rollout-repair-loop/08-CENSUS-ROLLUP.md

key-decisions:
  - "ms revision shas sourced via modelscope git ls-remote (the research ASSUMED /repo/revisions endpoint 404s) — all 14 rows pinned, A5 contingency never fired"
  - "Job-level HF_ENDPOINT=hf-mirror.com on example-nightly: huggingface.co is unreachable from the runner box, so cold hf fetches (giants prefetch included) would dead-wait without it (08-08 lora-repair finding generalized to the job level)"
  - "Stage 2/3 server topology per the D-07 architecture: streamable-http -> 3 probes -> sse restart -> 3 probes -> stop; stage 3 starts a FRESH streamable-http server (the mcp pair's gate probes :8000) and stops it after the pair"
  - "Cache-quota: measured, not decided — clean lock-only cache ~=15.2GiB > 10GB quota (evo2 exclusion still over at 12.5GiB); owner decision request recorded with three options; giants stay outside regardless"
  - "Owner's live JupyterLab kernels left intact during the census (idle, auto-respawned); memory trough 39Gi never approached the 35Gi floor"

patterns-established:
  - "First-dispatch consumption discipline: every runner-side failure classified (class + evidence) and dispositioned (repair commit / owner action / D-04 deferral) in the rollup — nothing silently dropped"
  - "Wheel caches restore BEFORE their build/install consumer steps (a cache step placed after its consumer never helps)"

requirements-completed: [CI-04, CI-05, EXEC-02, REPAIR-01]

coverage:
  - id: D1
    description: "models.lock growth: all 14 census-executed ids, revision-pinned (@sha), prefix-aligned, giants comment on evo-1 row"
    requirement: CI-04
    verification:
      - kind: other
        ref: "python3 lock-structure check (Task 1 <verify>): 'lock rows=24 pinned=14 all-14-ids-present giants-comment OK'"
        status: pass
    human_judgment: false
  - id: D2
    description: "Completed example-nightly job: bedtools rootless step, gated-family prereqs (evo/megaDNA/mamba pinned), wheel caches, giants prefetch, stages 2/3 wired, fail-soft summary, no continue-on-error"
    requirement: CI-05
    verification:
      - kind: other
        ref: "yaml token+ordering check (Task 2 <verify>): 'example-nightly completion OK (+ ordering checks)'"
        status: pass
    human_judgment: false
  - id: D3
    description: "First-dispatch outcomes consumed: both dispatches classified, all 6 failures repaired or dispositioned, 7 skips classified, runner inventory recorded"
    requirement: REPAIR-01
    verification:
      - kind: other
        ref: "junit artifact pytest-junit-example-stage1.xml + gh run view --log 37185365961; rollup 'First-Dispatch Consumption' section (grep-asserted in Task 3 <verify>)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Final D-03 reconciliation census: 196 passed / 1 benign skip / 0 failed in 2:59:34 (zero family skips); fast lane 1807P/1S in 97.55s"
    requirement: EXEC-02
    verification:
      - kind: integration
        ref: ".venv/bin/python -m pytest tests/examples -q -rs (/tmp/08_09_examples.log, rc=0) + skip-absence grep + pytest tests/ -m 'not slow' -q (/tmp/08_09_fastlane.log, rc=0)"
        status: pass
    human_judgment: false
  - id: D5
    description: "Cache-quota measurement recorded and owner decision request surfaced (the decision itself is open)"
    verification:
      - kind: other
        ref: "rollup 'Cache-quota measurement + decision record' section (quota grep count >=1 asserted in Task 3 <verify>); GitHub actions/caches API + per-model du evidence"
        status: pass
    human_judgment: true
    rationale: "The measurement is recorded and grep-asserted, but the disposition (pay-as-you-go vs drop the models-cache layer vs partial) is an owner pricing/billing decision — explicitly handed back open, never auto-taken."

duration: 3h 22min
completed: 2026-10-05
status: complete
---

# Phase 8 Plan 09: Full Execution Rollout — Lock, Nightly Completion, Final Census Summary

**Revision-pinned 24-row models.lock, the completed example-nightly job (bedtools rootless + wheel-cached gated-family prereqs + giants prefetch + stages 2/3), first-dispatch consumption with every failure dispositioned, and the final D-03 census (196P/1S/0F) — with the cache-quota decision handed back open to the owner.**

## Performance

- **Duration:** 3h 22min (2026-10-05 04:08–07:30 UTC; ~3h of it the final census)
- **Started:** 2026-10-05T04:08:13Z
- **Completed:** 2026-10-05T07:30:00Z
- **Tasks:** 3/3
- **Files modified:** 3

## Accomplishments

- **CI-04 closed:** models.lock grew to 24 `^(hf|ms)` rows — 9 ms zhangtaolab + 5 hf ids, every new row `@sha`-pinned with shas verified live at write time (ms via `git ls-remote` against modelscope — the research-assumed `/repo/revisions` endpoint 404s; hf via the mirror API, matching research exactly). Prefix-vs-source alignment checked across all 14 ids (benchmark's NT-100m rides `benchmark_config.yaml: source: modelscope`); no notebook edits were required. The evo-1 row carries the giants-tier comment.
- **CI-05 / D-07 / D-10 closed:** the example-nightly job is complete — bedtools rootless probe + micromamba/bioconda fallback onto GITHUB_PATH (cached prefix, no sudo/apt), evo isolated venv (stripedhyena 0.2.2 pinned `--no-deps`, evo-model 0.5, evo2==0.3.0 + vtx 1.1.0, TE-absence asserted), flash-attn 2.8.3.post1 sm_120 wheel cached on version+arch+torch, mamba kernels via cached-wheelhouse into the project venv, megaDNA pinned clone cb2f5ab4 + MEGABYTE_pytorch==0.2.1 into both venvs, safetensors-only giants prefetch into ~/models-giants (outside every cached path), stage 2 both transports (3+3 probes), stage 2.5 cleanup, stage 3 mcp pair with fresh streamable-http server, stage 4 audits all five junits + uploads server logs. Zero continue-on-error; fail-soft summary unchanged.
- **First dispatch consumed (REPAIR-01):** dispatch 1 (37184990854) = stage-0 LDFLAGS infra failure (repaired 08-01); dispatch 2 (37185365961, 66 min, 6F/147P/7S/8D) — all six failures classified and dispositioned (3× ipython backend2gui → 439370f; combined sibling gap → c5f0916; 2× bedtools → this plan's Task 2), all seven skips allowlisted (6 gated-prereq skips now un-gated by Task 2). Runner inventory recorded: bedtools MISSING (A10 confirmed), GB10, hf-mirror 200, ollama UP (runner shares the dev box's owner-enabled service).
- **Quota measured:** GitHub cache store = 2 uv entries (10.37GB, at the threshold), NO models cache ever saved (source dirs exceed the 10GB entry cap); clean lock-only cache ≈15.2GiB > quota. Decision handed back open with options (a) pay-as-you-go, (b) drop the models-cache layer (de-facto, proven green at 65-min cold stage 1), (c) partial ms-only. Owner cleanup action recorded: the 28GB evo-1 full-repo leftover inside `~/.cache/huggingface/hub` (executor removal denied by permission policy — correct per the project-root pin).
- **Final D-03 census (EXEC-02):** 196 passed / 1 benign skip / 0 failed in 2:59:34 with both endpoints up — zero lora/mcp/megaDNA/evo skips, all 21 notebooks execute for real; fast lane 1807P/1S in 97.55s (identical to the 08-08 baseline, zero new skips). Stage discipline recorded: 115Gi → trough 39Gi → 115Gi, port 8000 freed after teardown.
- **Hand-off dispatch fired:** the completed job's first real dispatch — run **37278002681** (workflow_dispatch on phs, 2026-10-05T07:29:14Z) — outcome lands per the Phase-5 D-04 post-merge boundary.

## Task Commits

1. **Task 1: models.lock growth** — `475d896` (feat)
2. **Task 2: example-nightly job completion** — `ea1bfd6` (feat)
3. **Task 3: first-dispatch consumption + quota + final census** — `6384805` (docs)

**Plan metadata:** `<this commit>` (docs: complete plan)

## Files Created/Modified

- `models.lock` — 14 pinned rows + pin-format header note; rotates the actions/cache key by design
- `.github/workflows/ci.yml` — example-nightly completion (+267/−22 lines: prereq steps, caches, giants prefetch, stages 2/2.5/3, stage-4 audit extension, job-level HF_ENDPOINT, timeout-comment actuals)
- `.planning/phases/08-full-execution-rollout-repair-loop/08-CENSUS-ROLLUP.md` — First-Dispatch Consumption + cache-quota record + final census + Phase-9 runtime hand-off

## Decisions Made

- ms sha source switched to `git ls-remote https://www.modelscope.cn/<id>.git` (the research's ASSUMED endpoint 404s) — A5 contingency never fired, every row pinned.
- The 6 MCP probes are the two probe files (3 streamable + 3 sse) per the research/coverage-nightly definition — NOT the whole `dnallm/mcp/tests` dir (test_server_integration instantiates DNALLMMCPServer directly; whole-dir would risk :8000 collisions).
- Bedtools wheelhouse-style cache restored BEFORE its install step (a cache after its consumer never helps).
- Owner's live JupyterLab kernels left intact during the census — idle, and memory never approached the floor (trough 39Gi vs 35Gi floor).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Job-level HF_ENDPOINT=hf-mirror.com on example-nightly**
- **Found during:** Task 2
- **Issue:** The plan pins the mirror only in the lora specs; huggingface.co is unreachable from the runner box, so cold hf fetches (giants prefetch, megaDNA/InstaDeepAI/PlantCAD2/evo2 on a cold cache) would dead-wait on the origin.
- **Fix:** Job-level `env: HF_ENDPOINT: https://hf-mirror.com` with the network-reality comment.
- **Files modified:** .github/workflows/ci.yml
- **Verification:** yaml structure check green; mirror verified HTTP 200 from the runner (dispatch-2 inventory).
- **Committed in:** ea1bfd6

**2. [Rule 1 - Bug] Bedtools cache-restore ordering**
- **Found during:** Task 2 (self-caught at verify time)
- **Issue:** First draft placed the bedtools cache step AFTER the install step — the restored prefix would never be seen and the cache would never help.
- **Fix:** Cache step moved before the install step; ordering asserted in the verify (extended token check).
- **Files modified:** .github/workflows/ci.yml
- **Verification:** ordering assertions pass.
- **Committed in:** ea1bfd6

**3. [Rule 3 - Blocking] Stage-3 MCP server restart**
- **Found during:** Task 2
- **Issue:** The plan's literal stage text stops the server after stage 2's probes, but the mcp pair's gate probes `:8000/mcp` — a stopped server would typed-skip the pair in stage 3.
- **Fix:** Implemented per the D-07 architecture diagram + module contract (one :8000 server at a time): fresh streamable-http instance for stage 3, stopped after the pair.
- **Files modified:** .github/workflows/ci.yml
- **Verification:** 'dnallm-mcp-server' + stage tokens present; dev-box census proved the both-up pair green under exactly this server shape.
- **Committed in:** ea1bfd6

**4. [Rule 1 - Interpretation] "6 MCP probes" scoped to the two probe files**
- **Found during:** Task 2
- **Issue:** The plan's parenthetical (`pytest dnallm/mcp/tests`) would also run test_server_integration, which instantiates DNALLMMCPServer directly — port-collision risk against the staged background server.
- **Fix:** Stage 2 runs `test_streamable_http_client.py` then `test_sse_client.py` (3+3 = the 6 live-server probes per the coverage-nightly definition).
- **Files modified:** .github/workflows/ci.yml
- **Verification:** structure check green.
- **Committed in:** ea1bfd6

---

**Total deviations:** 4 auto-fixed (2 blocking, 2 bug/interpretation)
**Impact on plan:** All four were required for the job to actually work on this runner/network; no scope creep.

## Issues Encountered

- The 28GB evo-1 full-repo leftover inside the quota-cache path could not be removed by the executor (permission policy correctly denies irreversible deletion outside the pinned project tree) — recorded as an OWNER ACTION in the rollup with the exact path.
- Operator slip during Task 3 verify: a sanity command accidentally re-launched the full 3-hour census; killed within ~2 minutes, no repo impact, the original census log untouched (the verify's grep read the original file).
- One background completion watcher hit its 2h limit mid-census (the census itself is nohup-detached and unaffected); re-armed once.

## Open Owner Items (handed back, NOT decided by the executor)

1. **Cache-quota decision** (THE open item): options (a) pay-as-you-go, (b) drop the models-cache layer, (c) partial ms-only — numbers and prerequisites in the rollup's quota section. Note the models cache is currently inert and the nightly is green regardless (proven cold-download stage 1 in 65 min).
2. **Box-side $HOME cleanup action:** `rm -rf ~/.cache/huggingface/hub/models--togethercomputer--evo-1-8k-base` (28GB, re-downloadable never needed — giants tier holds the sanctioned copy); the 19GB Qwen and 38GB legacy-blob trees are flagged for the same pass.
3. **Hand-off dispatch 37278002681** outcome lands per the Phase-5 D-04 post-merge runner-confirmation boundary.

## User Setup Required

None — no external service configuration introduced by this plan (the ollama systemd unit enablement was already satisfied; the runner shares this box's owner-enabled service).

## Next Phase Readiness

- Phase 8 execution is complete (9/9 plans); phase regression gate → verifier → transition are the orchestrator's next steps.
- Phase 9 (CI-06/CI-07) reads the measured runtime-budget hand-off in the rollup: dispatch + census actuals, one-time first-run costs (flash-attn/mamba builds, giants prefetch), steady-state ~3.5–4h estimate, and the quota caveat.
- The owner's pre-Phase-9 todo stands: `/gsd-map-codebase` + `/gsd-graphify` refresh.

---
*Phase: 08-full-execution-rollout-repair-loop*
*Completed: 2026-10-05*

## Self-Check: PASSED

- models.lock, ci.yml, 08-CENSUS-ROLLUP.md, 08-09-SUMMARY.md all exist on disk
- Task commits 475d896 / ea1bfd6 / 6384805 verified in git log
- All plan <verification> checks re-run green (lock structure, job structure+ordering, census rc=0 + skip-absence, fast lane rc=0, quota + First-Dispatch Consumption greps)
