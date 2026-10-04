---
phase: 08-full-execution-rollout-repair-loop
plan: 06
subsystem: testing
tags: [evo-1-8k, evo2-noFP8, giants-tier, isolated-kernelspec, HF_HUB_OFFLINE, hf-mirror, provenance-stamp, notebook-repair]

requires:
  - phase: 08-full-execution-rollout-repair-loop
    provides: allow_patterns passthrough + evo-1 safetensors patterns + spec env override + np.fromstring shim (08-03); isolated-kernelspec lane pattern + D-21 stamps (08-04/08-05)
provides:
  - Dev-box giants tier (~12.9GB evo-1-8k safetensors-only at pinned revision + evo2 2.7GB, both with refs/main) outside every cached path
  - Isolated dnallm-evo-kernel kernelspec (.scratch/evo-venvs/evo) with FEASIBILITY-locked evo stack; project venv provably untouched
  - Repaired evo notebook (8k reference, noFP8 override cell, D-21 stamp) with synced mirrors — first real execution green, ZERO new transformers-5.17/5.18 rungs (A7 resolved: no OPEN-RUNG ledger)
  - Hermetic evo lane: spec env HF_HUB_CACHE + HF_HUB_OFFLINE against the prefetched tier
affects: [08-07, 08-08, 08-09]

actuals:
  tokens: 9666   # chars/4 over the realized diff (38662 chars); estimate 72000 overshot the sibling-plan way — execution/campaign time dominates, diff size never does
  tasks: 2
  commits: 3     # measured: git rev-list --count 2b6aef6..HEAD (84c32f5, a7ed221, 4dbfe6b)
plan_head_before: 2b6aef6d4dafc5bae8681a1046aec1d673e6acc6
plan_head_after: 4dbfe6b69a9a7d27633ecc0bb937cb90f552b261

tech-stack:
  added: []   # evo-model 0.5 + stripedhyena 0.2.2 + evo2 0.3.0 + flash-attn 2.8.3.post1 live ONLY in the gitignored throwaway venv (spike build reused, never rebuilt)
  patterns: [hermetic offline lane against a prefetched giants tier (HF_HUB_OFFLINE + refs/main via real mirror fetches, no hand-edited refs), spike-build reuse across venvs (flash_attn package + top-level flash_attn_2_cuda .so), FEASIBILITY-documented capability override cell (is_fp8_capable -> False) as visible notebook content]

key-files:
  created: []   # venv/kernelspec/giants are runtime artifacts, intentionally uncommitted
  modified:
    - example/notebooks/generation_evo_models/inference.ipynb
    - docs/example/notebooks/generation_evo_models/inference.ipynb
    - docs/example/notebooks/inference_evo_models.md
    - tests/examples/_execution.py
    - tests/examples/test_notebook_execution.py

key-decisions:
  - "Giants tier served hermetically: the spec env gains HF_HUB_OFFLINE=1 on top of 08-03's HF_HUB_CACHE — the handler resolves revision 'main' for the undotted evo-1-8k id, which would head-call the Hub on every run (and hang on this box, where huggingface.co is unreachable and only hf-mirror.com answers); refs/main for both models was created by REAL mirror fetches, never hand-edited"
  - "Prefetches ran through HF_ENDPOINT=https://hf-mirror.com (the repo's own mirror toggle, model.py:380 precedent) because direct huggingface.co is network-unreachable (errno 101) from the dev box; the committed harness carries no mirror dependency"
  - "flash-attn reused from the surviving /tmp/feas-venv spike build (flash_attn package + flash_attn_2_cuda.cpython-313 .so — the top-level .so is load-bearing; copying the package alone breaks import); no 50-min rebuild needed"
  - "evo2's 2.7GB full fetch landed in the giants dir on the dev box (consequence of the spec env override); research sizes it quota-cache-friendly, so the runner layout decision stays with 08-09"
  - "The evo2 noFP8 path is a visible notebook cell (documented FEASIBILITY deviation: is_fp8_capable patched False) rather than a library behavior change — evo.py is outside this plan's file list and the deviation belongs in the tutorial's face"
  - "Zero OPEN-RUNG lines: the first real execution surfaced no new transformers-5.17 rungs — the venv resolved transformers 5.18.0, so A7 is resolved one minor BEYOND the budgeted version"

patterns-established:
  - "Hermetic giants lane: prefetch (pinned revision + refs/main via a reachable endpoint) + HF_HUB_OFFLINE in the spec env = revision resolution never head-calls the Hub and any re-fetch fails loudly instead of hanging"
  - "Spike-build reuse: compiled extension packages can move across venvs when python ABI + torch build match; copy every top-level artifact (the .so beside the package), not just the package dir"

requirements-completed: [EXEC-02, CI-05, REPAIR-01]

coverage:
  - id: D1
    description: "Dev-box giants tier: safetensors-only evo-1-8k snapshot (~12.9GB) at pinned revision a9be7b6, zero .pt files, outside cached paths; evo2 full snapshot + refs/main for both"
    requirement: CI-05
    verification:
      - kind: automated_ui
        ref: "Task 1 verify exit-code chain: du 12316 MiB in [10000,15000] window, find *.pt == 0, kernelspec file present (KERNELSPEC_OK echoed); snapshot dir == a9be7b66485080893399ade87c7d34f81ad3e249"
        status: pass
    human_judgment: false
  - id: D2
    description: "Isolated dnallm-evo kernelspec + lane wiring: VIRTUAL_ENV pinned throwaway venv, gate probes THAT venv, ensure_evo_kernel registers without ever installing; project venv untouched"
    requirement: EXEC-02
    verification:
      - kind: unit
        ref: "tests/examples/test_notebook_execution.py#TestEvoIsolatedLane (3 tests) + fast module 32 passed; kernel.json env VIRTUAL_ENV=.scratch/evo-venvs/evo asserted live; project .venv pip list shows no evo/stripedhyena/flash-attn/vtx"
        status: pass
    human_judgment: false
  - id: D3
    description: "Repaired evo notebook (8k reference + noFP8 override + D-21 stamp) with same-commit synced mirrors, executed green under dnallm-evo-kernel — zero skips, zero new rungs"
    requirement: REPAIR-01
    verification:
      - kind: integration
        ref: "pytest -k evo: 5 passed / 0 SKIPPED in 49s (gated leg 48.43s re-run with --durations); check_notebook_md_sync 24/24 + check_docs_sync both exit 0"
        status: pass
    human_judgment: false

status: complete
duration: 20min
completed: 2026-10-04
---

# Phase 8 Plan 6: evo Giants Tracer — Tier, Kernel, Repair, First Real Execution Summary

**evo-1-8k giants tier prefetched safetensors-only (12.3GB, zero .pt) into ~/models-giants with an isolated dnallm-evo kernel, the notebook repaired to the 8k + noFP8 variants (D-21 stamped, mirrors synced), and the first real execution green in 49s under transformers 5.18 — zero new remote-code rungs, so no OPEN-RUNG ledger for 08-07**

## Performance

- **Duration:** ~20 min (19:36–19:56 UTC; ~11 min of it the 12.9GB+2.7GB mirror downloads and venv provisioning, run in parallel with the code wiring)
- **Started:** 2026-10-04T19:36:58Z
- **Completed:** 2026-10-04T19:56:00Z
- **Tasks:** 2/2
- **Files modified:** 5 (notebook, 2 mirror files, 2 harness/test files)

## Accomplishments

- **Giants tier proven on the dev box (CI-05).** `~/models-giants/hub` holds the evo-1-8k safetensors-only snapshot (12,316 MiB, inside the 10,000–15,000 MiB window) at pinned revision `a9be7b66485080893399ade87c7d34f81ad3e249`, zero `.pt` files anywhere under it, plus the evo2 full snapshot — both with `refs/main` created by real mirror fetches. Nothing in the giants dir is reachable from any cached path.
- **Isolated dnallm-evo lane landed.** Throwaway venv `.scratch/evo-venvs/evo` (gitignored): dnallm `-e .[cuda130]`, `evo-model==0.5`, `stripedhyena==0.2.2 --no-deps`, `evo2==0.3.0` (the empty transformer-engine meta package its resolver pulls is uninstalled immediately — TE stays absent per the FEASIBILITY lock), and flash-attn 2.8.3.post1 reused from the surviving Phase-5 spike build (no 50-min rebuild). `dnallm-evo-kernel` kernelspec registered with `VIRTUAL_ENV` pinned; the project venv imports none of the stack.
- **Harness wiring with same-change tests.** `evo_prerequisites_installed()` probes the kernelspec venv's interpreter (stripedhyena + evo2 + flash_attn); `ensure_evo_kernel()` registers/repairs the spec idempotently and NEVER installs (out-of-band provisioning only); `_gate_evo` now reflects the execution environment; `TestEvoIsolatedLane` pins the contracts (3 fast tests). The spec env carries `HF_HUB_CACHE` (08-03) + `HF_HUB_OFFLINE=1` — a hermetic lane where any re-fetch fails loudly instead of hanging on an unreachable Hub.
- **Notebook repaired (D-02/D-06/D-21).** evo-1 reference 131k → `togethercomputer/evo-1-8k-base` (source huggingface, per the spike-proven route); the evo2 leg forces the noFP8 config path via a visible override cell (`is_fp8_capable → False`, the documented FEASIBILITY deviation); prerequisite comments updated to the FEASIBILITY-locked pins; a D-21 stamp cell prints transformers/torch/flash_attn versions; stale 131k-era outputs cleared on edited cells. Mirrors re-exported same-commit (D-20): wrapper `docs/example/notebooks/inference_evo_models.md` (8k excerpt + noFP8 block + prose) and the byte-identical raw copy.
- **First real execution: GREEN, zero rungs (A7 resolved).** `pytest -k evo`: **5 passed / 0 SKIPPED in 49s** (gated leg 48.43s — both model legs load from the giants tier offline, generate + score green). No `AttributeError`/`ImportError` naming stripedhyena or remote positional embeddings ever fired: the 08-03 np.fromstring shim plus the existing closed map carried transformers **5.18.0** (the venv resolved a minor NEWER than the project's 5.17). Because the run exited 0, the OPEN-RUNG ledger is legitimately empty.
- **A4 flagged assumption VERIFIED.** `from_pretrained` loaded the safetensors+index+configs snapshot with `HF_HUB_OFFLINE=1` set — a re-fetch of the 16.81GB `.pt` was impossible by construction and the load succeeded, closing the "loads without re-fetching" assumption with the strongest possible evidence.

## Task Commits

1. **Task 1: giants tier + isolated dnallm-evo kernelspec lane** — `84c32f5` (test)
2. **Task 2: notebook repair + first real execution** — `a7ed221` (fix)
3. **Style: ruff format the touched test module** — `4dbfe6b` (style; pre-existing 08-05 drift surfaced when my commit touched the file)

**Plan metadata:** (this commit)

## Files Created/Modified

- `example/notebooks/generation_evo_models/inference.ipynb` — 8k reference, noFP8 override cell, D-21 stamp, pinned-prereq comments, stale outputs cleared
- `docs/example/notebooks/inference_evo_models.md` — wrapper: 8k excerpt + noFP8 block + stamp prose (AST sync green)
- `docs/example/notebooks/generation_evo_models/inference.ipynb` — byte-identical raw copy (docs sync green)
- `tests/examples/_execution.py` — EVO_KERNEL_NAME/EVO_VENV_DIR, evo_prerequisites_installed, ensure_evo_kernel, spec kernel_name + offline env
- `tests/examples/test_notebook_execution.py` — venv-probing _gate_evo, evo kernel branch in the gated lane, TestEvoIsolatedLane (3 tests)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] huggingface.co unreachable from the dev box — revision resolution would hang the lane**
- **Found during:** Task 1 (first prefetch died with `httpx.ConnectError: [Errno 101] Network is unreachable`; a direct probe of revision `main` timed out at 60s)
- **Issue:** The handler resolves revision `main` for the undotted `evo-1-8k-base` id, so every lane run head-calls the Hub; on this box only `hf-mirror.com` answers (200 in 0.47s)
- **Fix:** Prefetches rerun through `HF_ENDPOINT=https://hf-mirror.com` (the repo's own mirror-toggle precedent, model.py:380) creating `refs/main` for both models by REAL fetches; the committed lane runs hermetically via `HF_HUB_OFFLINE=1` in the spec env — no mirror dependency enters the harness, no cache hand-edits
- **Files modified:** tests/examples/_execution.py (+ contract test)
- **Verification:** gated leg green offline; any re-fetch would have failed loudly under HF_HUB_OFFLINE=1
- **Committed in:** 84c32f5, a7ed221

**2. [Rule 1 - Bug] flash-attn copy missed the top-level `flash_attn_2_cuda` .so**
- **Found during:** Task 1 venv provisioning (first copy attempt: `ModuleNotFoundError: No module named 'flash_attn_2_cuda'`)
- **Issue:** The spike build installs the compiled kernel as a top-level `.so` NEXT TO the package, not inside it
- **Fix:** Copy `flash_attn` + dist-info + `flash_attn_2_cuda.cpython-313-aarch64-linux-gnu.so`; probes then green (same py3.13 + torch 2.11.0+cu130 ABI)
- **Verification:** `import flash_attn, flash_attn_2_cuda` green in the venv
- **Committed in:** (runtime artifact, no commit)

**3. [Rule 2 - Missing wiring] Task 1's files list named only the test module, but the lane needs _execution.py too**
- **Found during:** Task 1 (constants, venv probe, and ensure helper live beside the megadna/langchain precedents)
- **Issue:** Same 08-03 shape: spec keys and helpers cannot reach the kernel from the test module alone
- **Fix:** Wiring split exactly along the megadna precedent; same-change fast tests
- **Files modified:** tests/examples/_execution.py
- **Committed in:** 84c32f5

**4. [Rule 3 - Blocking] Plan's mirror path was approximate**
- **Found during:** Task 2 (no `docs/example/notebooks/generation_evo_models/inference.md` exists)
- **Issue:** The evo mirror is the wrapper `docs/example/notebooks/inference_evo_models.md` + raw ipynb copy — the sync scripts are the authority
- **Fix:** Updated both real mirror artifacts; both sync gates green
- **Committed in:** a7ed221

---

**Total deviations:** 4 auto-fixed (1 bug, 1 missing-wiring, 2 blocking). **Impact:** All within the plan's envelope; the offline-lane decision (Deviation 1) is load-bearing for runner wiring — see Next Phase Readiness.

## Issues Encountered

- evo2's resolver pulls the empty `transformer-engine` meta package (its RuntimeError escapes vortex's ImportError guard); uninstalled immediately after install per the FEASIBILITY lock — `from evo2 import Evo2` green.

## Reversible Install Record (T-08-12 / T-08-SC)

- **Installed into the throwaway venv only** (`.scratch/evo-venvs/evo`, gitignored): `evo-model==0.5`, `stripedhyena==0.2.2` (--no-deps), `evo2==0.3.0` (+vtx 1.1.0, −transformer-engine), `flash-attn 2.8.3.post1` (spike-build copy), dnallm `-e .[cuda130]`, ipykernel
- **Project .venv:** untouched — `pip list | grep -iE 'stripedhyena|^evo|flash-attn|vtx'` matches nothing (nvidia-nvtx is unrelated)
- **Reverse:** `rm -rf .scratch/evo-venvs/evo ~/.local/share/jupyter/kernels/dnallm-evo-kernel`; giants tier removal is a plain `rm -rf ~/models-giants`

## User Setup Required

None - no external service configuration required. (Boxes without the venv keep getting the honest `optional-dep:` typed skip from `_gate_evo` with the venv probe evidence.)

## Next Phase Readiness

- **A7 closed with zero rungs**: 08-07 inherits no OPEN-RUNG ledger (the literal marker is absent because the run exited 0 — the exit-code-enforced contract's green path). Its residual-sweep scope should re-confirm under the 5.17 project venv only if the runner pins one.
- **08-08/08-09 runner wiring must reproduce**: (a) the giants prefetch step (pinned revision + `refs/main` for BOTH models — offline resolution needs the ref, not just blobs), (b) the venv stack exactly as recorded above (TE uninstalled after evo2), (c) `HF_HUB_OFFLINE=1` only if the runner pre-warms the tier; a network-capable runner may instead resolve live, in which case drop the offline key there — never carry a mirror endpoint into CI.
- **Network note**: the dev box cannot reach huggingface.co directly (only hf-mirror.com); any local re-verification of hub-touching lanes must export `HF_ENDPOINT=https://hf-mirror.com` or run offline against the prefetched tier.
- D-03 census: the evo row flips in 08-07/08-09's reconciliation, per the phase's plan split.

## Self-Check: PASSED

- All 5 key-files committed (git diff 2b6aef6..4dbfe6b lists exactly them)
- Commits 84c32f5 / a7ed221 / 4dbfe6b present on phs (3 measured from the 2b6aef6 ledger)
- Task 1 verify re-run: KERNELSPEC_OK (12,316 MiB window, 0 .pt, pinned snapshot dir); Task 2 verify re-run: 5 passed / 0 SKIPPED rc=0, both sync gates exit 0; fast module 32 passed; ruff check + format green on touched files

---
*Phase: 08-full-execution-rollout-repair-loop · Completed: 2026-10-04*
