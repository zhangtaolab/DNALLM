---
phase: 27-transformers-4-5-model-compatibility-census
plan: 02
subsystem: testing
tags: [transformers, compatibility, shim-rungs, census, remote-code, tdd, dispositions]

# Dependency graph
requires:
  - phase: 27-01
    provides: "tests/models/compat_census.json — the 49-record pre-fix evidence base; scripts/census_compat.py"
provides:
  - "Two census-driven compat rungs in dnallm/utils/transformers_compat.py, registered in apply_patches() (rung 12 _patch_apply_chunking_to_forward, rung 13 _patch_tied_weights_bookkeeping_on_load)"
  - "tests/utils/test_transformers_compat_census.py — 19 per-rung contract tests (RED-first, machine-validated RED_EVIDENCE_OK)"
  - "Post-shim census artifact: 27 import-ok (was 25), every shim-restored entry load+forward measured, 8 disposition records covering every failure"
  - "Census tool upgrades: task-appropriate Auto* load routing, previous_status stamping, shim-pass load selection (13 new tests)"
affects: [27-03, 27-04]

# Actuals (#2632) — chars/4 over the realized diff; commits measured from the ledger.
actuals:
  tokens: 22645
  tasks: 3
  commits: 5
  plan_head_before: "91e51572ddff38929dfbe5473bd9489602a4270b"
  plan_head_after: "135eeac4a9922fc73934bcec43135c71d405a0d2"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Alias-first rung: when transformers 5.x still ships an implementation (pytorch_utils), re-attach the LIVE object; vendoring (v4.49.0-cited) is only the fallback for a future removal"
    - "Load-boundary protocol bridging: remote __init__ paths that skip post_init entirely get the 5.x all_tied_weights_keys computation AND the tie_weights arity bridge inside a wrapped _finalize_model_loading (shape-adaptive static/classmethod wrapper)"
    - "Task-appropriate Auto* dispatch in the census load layer mirrors dnallm's own task-type dispatch, honoring task-keyed auto_map entries"

key-files:
  created:
    - tests/utils/test_transformers_compat_census.py
  modified:
    - dnallm/utils/transformers_compat.py
    - scripts/census_compat.py
    - tests/models/compat_census.json
    - tests/models/test_census_compat.py
    - tests/utils/test_transformers_compat.py

key-decisions:
  - "Census-driven rung scope: the artifact's measured failures (apply_chunking_to_forward gone from modeling_utils; skipped-post_init bookkeeping gap) built the rung set — the research-predicted classes (is_torch_fx_available, from_legacy_cache, ROPE_INIT_FUNCTIONS) never surfaced as census failures and got no rungs (D-06: the census is the truth)."
  - "_patch_apply_chunking_to_forward aliases the live transformers.pytorch_utils implementation transformers 5.17 still ships (zero drift by construction; the owner-endorsed alias-to-5.x fix shape), with the v4.49.0-vendored copy as fallback only."
  - "The 6mer and PlantCAD2 load failures were census-tool misroutes (generic AutoModel loading native architectures with remote configs for checkpoints whose auto_map carries only task keys), not library breakage — fixed in the tool by task-appropriate Auto* routing; their fresh green loads replace the failures, so by design they carry no disposition."
  - "Rung-recovered imports that still fail at load/forward carry skip dispositions whose reason names both the structural cause AND the restoring rung (gena-base: checkpoint token_type_ids buffer garbage 3.9e12 vs type_vocab_size 2 — version-independent data defect; gena-bigbird: external deepspeed dependency)."
  - "_finalize_model_loading is a STATIC method at transformers 5.17 — the rung's wrapper is signature-adaptive (probes the captured original's first parameter) so it survives class- and static-shaped variants."

patterns-established:
  - "Disposition vocabulary in the census artifact: every failure record (import-fail now/previously, census-blocked, or failed load/forward) carries exactly one of {fix: <_patch_* rung>} or {skip: <structural reason>}; passing records carry no key"
  - "previous_status stamping on every verdict change keeps pre-fix evidence addressable through the merge"

requirements-completed: ["TBD"]

# Coverage metadata (#1602)
coverage:
  - id: D1
    description: "Two census-driven compat rungs: apply_chunking_to_forward restored on modeling_utils (alias + v4.49.0 vendored fallback); tied-weights bookkeeping + tie_weights arity materialized at the from_pretrained boundary — both registered in apply_patches(), absence-gated, sentinel-idempotent, no-op on 4.x, closed-map"
    verification:
      - kind: unit
        ref: "python -m pytest tests/utils/test_transformers_compat_census.py -q (19 passed; RED first: 16 failed on planned assertions, machine gate RED_EVIDENCE_OK)"
        status: pass
      - kind: integration
        ref: "live probes: kuleshov-group/PlantCaduceus_l32 bare-backbone load+forward (1,64,2048) with rung; zhangtaolab/plant-dnabert-6mer native load through wrapped finalize (1,32,4107)"
        status: pass
      - kind: other
        ref: "python -m ruff check dnallm/utils/transformers_compat.py tests/utils/ (clean)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Census tool fixes: task-appropriate Auto* load routing, previous_status preservation, shim-pass load selection (recovered imports load; green loads carry; failed loads re-probe; still-failing park final)"
    verification:
      - kind: unit
        ref: "python -m pytest tests/models/test_census_compat.py -q (53 passed, incl. 13 new routing/merge/selection tests)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Post-shim census artifact: 2 imports recovered with D-02a load+forward evidence, 3 load failures restored green, 17 prior green loads carried, zero regressions, 49 records with provenance"
    verification:
      - kind: integration
        ref: "Task-2 plan assertion (2 recovered entries load-verified) + integrity sweep (exact failing_import kept, no status regressions)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Disposition completeness: every failure record carries fix-rung or concrete structural reason; artifact frozen as 27-03's write-back input"
    verification:
      - kind: unit
        ref: "TestDispositionCompleteness (offline, standard lanes) + Task-3 plan assertion 'dispositions complete'"
        status: pass
    human_judgment: false

# Metrics
duration: 38min
completed: 2026-10-27
status: complete
---

# Phase 27 Plan 02: Shim Rungs from Census Evidence Summary

**Two census-driven compat rungs (chunking alias + load-boundary bookkeeping) recovered 2 imports and restored 3 load failures; every remaining failure now carries a fix-rung or structural-reason disposition**

## Performance

- **Duration:** 38 min
- **Started:** 2026-10-27T08:13:27Z
- **Completed:** 2026-10-27T08:52:03Z
- **Tasks:** 3
- **Files modified:** 6 (1 new test module, 5 modified)

## Accomplishments

- **Rung 12 — `_patch_apply_chunking_to_forward`**: transformers 5.x removed the name from `modeling_utils` (the import site of both gena-lm checkpoints' remote code); 5.17 still ships the implementation in `pytorch_utils`, so the rung re-attaches the LIVE object (zero drift) with a v4.49.0-vendored fallback (`src/transformers/pytorch_utils.py` cited verbatim, two documented deviations: S101 assert re-raise, B905 explicit `strict=False`).
- **Rung 13 — `_patch_tied_weights_bookkeeping_on_load`**: remote 4.x classes whose `__init__` ends with neither `post_init()` nor `init_weights()` (the bare `Caduceus` backbone) crash at `_move_missing_keys_from_meta_to_device` reading `all_tied_weights_keys`. The rung wraps `_finalize_model_loading` (static at 5.17; wrapper is shape-adaptive) to materialize exactly post_init's computation — plus the legacy `tie_weights` arity bridge, which the same skipped entry never installed. Verified live: PlantCaduceus_l32 loads and forwards `(1, 32, 2048)`.
- **Census re-run (49 entries, transformers 5.17.0, CUDA)**: 25→27 import-ok (gena ×2 recovered, `previous_status` stamped); load verdicts restored green for plant-nucleotide-transformer-6mer `(1,2)`, PlantCAD2-Small `(1,2)` (task-appropriate Auto* routing) and PlantCaduceus_l32 (rung 13); DNABERT-2 and MutBERT re-fail honestly; gpn/Omni-DNA keep exact failing-import evidence; 17 previously-green representative loads carried forward; zero regressions.
- **Dispositions complete (8 records)**: every failure carries exactly one of `fix`/`skip` with concrete reasons naming WHAT is structurally unfixable — gena-base's checkpoint `token_type_ids` buffer garbage (3.9e12 vs type_vocab_size 2, version-independent), gena-bigbird's external `deepspeed` dependency, gpn's removed native architecture, Omni-DNA's external `olmo` import, DNABERT-2's eager meta-device ALiBi construction, MutBERT's own `input_shape[:-1]` repo bug, GENERanno's absent repo id, CrossDNA's missing root config.json.
- **19 new rung-contract tests** (RED-first, machine-validated `RED_EVIDENCE_OK`, then green) + 13 census-tool tests + 3 disposition tests; `EXPECTED_PATCH_INSTALLERS` roster consciously extended 12→15.

## Task Commits

Each task was committed atomically (TDD RED → GREEN for Task 1):

1. **Task 1 (RED): failing rung tests** - `7d4a97e` (test)
2. **Task 1 (GREEN): two compat rungs** - `759018a` (feat)
3. **Task 2: census tool routing/merge/selection fixes** - `89a2342` (fix)
4. **Task 2: post-shim census evidence artifact** - `18a3f38` (test)
5. **Task 3: dispositions + completeness test** - `135eeac` (test)

**Plan metadata:** committed separately (docs: complete plan).

## Files Created/Modified

- `dnallm/utils/transformers_compat.py` — rungs 12-13 (+295 lines: vendored impl, attach helper, materializer, shape-adaptive wrapper, apply_patches registration).
- `tests/utils/test_transformers_compat_census.py` — 19 per-rung contract tests (resolution, alias identity, idempotency, absence-gate, vendor fallback, chunking semantics, materialization, delegation, closed-map).
- `scripts/census_compat.py` — `resolve_load_class_name`, `previous_status` stamping, `apply_loads(previous=...)` selection.
- `tests/models/compat_census.json` — post-shim artifact: 27/18/2/2 statuses, 2 recovered imports with load evidence, 8 dispositions; frozen input for 27-03.
- `tests/models/test_census_compat.py` — routing/merge/selection/disposition suites (53 total).
- `tests/utils/test_transformers_compat.py` — installer roster pin extended.

## Decisions Made

- Census-driven rung scope over research predictions (D-06): only measured failure classes got rungs; `is_torch_fx_available`, `from_legacy_cache`, and the RoPE-init family never appeared as census failures (the import census already ran with the 11 existing rungs armed).
- Alias-to-live-implementation preferred over vendoring when transformers 5 still ships the code (owner-endorsed alias shape; zero drift).
- Load-recovered-but-status-unchanged entries (6mer, PlantCAD2) carry no disposition: their recovery evidence is the fresh green load; the tooling fix lives in git history and this SUMMARY, and 27-03 computes registry status from the evidence.
- gena dispositions use `skip` (not `fix`) because the D-03 verdict is what the registry consumes; the restoring rung is named inside each skip reason and visible via `previous_status`.
- The bookkeeping rung bridges at `_finalize_model_loading` (the boundary transformers itself calls between weight loading and the first consumer) rather than patching each consumer — one hook, all consumers covered.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Census load layer misrouted task-keyed remote checkpoints through AutoModel**
- **Found during:** Task 1 investigation (pre-build measurement)
- **Issue:** the 27-01 load program dispatched only ForCausalLM/ForMaskedLM suffixes; checkpoints whose auto_map carries only task keys (plant-nucleotide-transformer-6mer, PlantCAD2-Small) resolved NATIVE architectures with remote configs — the 6mer's `rope_theta` failure and PlantCAD2's `all_tied_weights_keys` failure were both measured through the wrong route.
- **Fix:** `resolve_load_class_name` routes ForSequenceClassification/ForTokenClassification through their Auto* classes (mirroring dnallm's task-type dispatch); bare backbones (Caduceus, BertForPreTraining) keep the AutoModel route their auto_map intends.
- **Files modified:** scripts/census_compat.py, tests/models/test_census_compat.py
- **Verification:** both entries load+forward green in the re-run; 13 new tests.
- **Committed in:** `89a2342`

**2. [Rule 2 - Missing critical detail] tie_weights arity bridge needed at the load boundary too**
- **Found during:** Task 1 GREEN (live Caduceus verification)
- **Issue:** the existing arity bridge installs inside the `init_weights` wrapper — a remote `__init__` that skips post_init entirely never installs it, so from_pretrained's `model.tie_weights(missing_keys=...)` died with TypeError right after the bookkeeping fix.
- **Fix:** the `_finalize_model_loading` wrapper also calls `_bridge_legacy_tie_weights(receiver)` (self-guarding per signature); failing test written first.
- **Files modified:** dnallm/utils/transformers_compat.py, tests/utils/test_transformers_compat_census.py
- **Verification:** Caduceus bare backbone loads+forwards `(1,64,2048)`; bridge test green.
- **Committed in:** `759018a`

---

**Total deviations:** 2 auto-fixed (1 Rule 1, 1 Rule 2)
**Impact on plan:** Both deviations were evidence-driven (D-06) and kept the plan's scope intact: the first corrected the measurement instrument, the second completed the same rung's semantic gap found by running it live.

## Issues Encountered

- gena-lm-bert-base's restored import exposed a version-independent checkpoint defect (garbage `token_type_ids` buffer): diagnosed on CPU (CUDA device-side asserts are opaque), dispositioned as skip with the exact buffer value; no library fix is possible without changing checkpoint data.
- Mid-GREEN wrapper arity bug (assumed classmethod; 5.17 ships `_finalize_model_loading` as static) was caught by the native-load regression probe before any commit and fixed with a shape-adaptive wrapper plus corrected delegation tests.

## Authentication Gates

None — no authentication surfaces were touched.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- 27-03 consumes `tests/models/compat_census.json` directly: statuses (27 import-ok / 18 arch-ok / 2 import-fail / 2 census-blocked), `previous_status` for the 2 shim-recovered entries, per-record `disposition` for every failure, `load` evidence with `forward_ok` bools, and family/load_note fields — all present and validated by offline tests.
- The registry write-back map is unambiguous: green evidence → `verified-tf5`/`shim-required-tf5` per CONTEXT; skip dispositions → `remote-code-broken` with the exact failing import/reason.

## Self-Check: PASSED

- `tests/utils/test_transformers_compat_census.py` — FOUND
- `dnallm/utils/transformers_compat.py` (rungs registered) — FOUND
- `tests/models/compat_census.json` (dispositions) — FOUND
- Commits `7d4a97e`, `759018a`, `89a2342`, `18a3f38`, `135eeac` — all ancestors of HEAD (verified below).
