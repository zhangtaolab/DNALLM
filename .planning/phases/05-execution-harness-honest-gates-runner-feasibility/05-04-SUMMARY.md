---
phase: 05-execution-harness-honest-gates-runner-feasibility
plan: 04
subsystem: transformers-5.x compatibility shim (remote-code pruning helpers)
tags: [transformers-compat, trust-remote-code, pruning-helpers, nucleotide-transformer, gap-closure, typed-skip]

requires:
  - phase: 05-execution-harness-honest-gates-runner-feasibility
    provides: GAP-1 reopen evidence (05-VERIFICATION.md addendum), D-07 locked fix direction, typed-skip allowlist (environment-unavailable: prefix)
provides:
  - "Vendored v4.49.0 pruning helpers (_find_pruneable_heads_and_indices, _prune_linear_layer) attached to transformers.modeling_utils when absent (no-op on 4.x), wired into apply_patches()"
  - "tests/models/test_model_remote_code.py — live self-healing NT v2 promoter smoke: real load+forward attempt, typed environment-unavailable skip carrying the exact deeper-breakage traceback until the environment gap closes"
  - "tests/utils/test_transformers_compat.py::TestRemoteCodePruningHelpers — 6 fast network-free behavior-contract tests (helper arithmetic, prune_linear_layer shapes/values/bias slicing, attachment idempotence, version-agnostic identity)"
  - "Owner hand-off evidence: native-ESM (trust_remote_code=False) route probed and refuted — FFN weight-shape mismatch (ckpt 4096x512 vs native 2048x512); benchmark notebook flagged as census FAIL row"
affects: [05-gap-closure census (05-05/05-06), Phase 8 repair/rescoping, benchmark notebook example/notebooks/benchmark/benchmark.ipynb]

actuals:
  tokens: 4160
  tasks: 2
  commits: 5
  plan_head_before: 8fa9e05c89f6aee24c2c1bf00a69c16b00e4a879
  plan_head_after: e8013fb566cf08028c056afe7b965c84f51db35b

tech-stack:
  added: []  # zero installs; vendoring from the already-installed upstream reference (T-05-SC)
  patterns:
    - "Absence-gated module-attribute attach: setattr-equivalent assignment under upstream names, guarded by hasattr (4.x no-op) + _dnallm_remote_code_pruning_patch sentinel, mirroring _dnallm_quant_key_patch"
    - "Self-healing typed skip: slow smoke keeps attempting the real load; only the documented structural marker triggers environment_unavailable_skip, anything else fails loudly"

key-files:
  created:
    - tests/models/test_model_remote_code.py
  modified:
    - dnallm/utils/transformers_compat.py
    - tests/utils/test_transformers_compat.py

key-decisions:
  - "Fallback ladder terminated at its designed rung: after the import fix, the remote modeling_esm.py needs config.is_decoder/config.add_cross_attention (transformers-4.x PretrainedConfig defaults removed in 5.x, absent from the checkpoint's config.json) — a config-attribute dependency, not a vendored pure helper, so per D-07 the shim was NOT extended; smoke converted to the typed skip with exact traceback, benchmark notebook flagged as census FAIL"
  - "Native-ESM route probed and refuted (scratch diagnostic, never wired into dnallm): trust_remote_code=False load dies on intermediate.dense.weight mismatch (ckpt 4096x512 vs 2048x512 from config.json) — the remote esm_config.py derives dims config.json does not carry, so the remote config code is load-bearing"
  - "Version-agnostic identity test inverts instead of skipping on 4.x (5.x: attached == vendored; 4.x: upstream's own helpers untouched) — zero new skips on any CI leg"

patterns-established:
  - "Ladder-terminal typed skip: evidence-backed environment_unavailable_skip promoted automatically to a real green smoke when the environment gap closes"

requirements-completed: [EXEC-01, REPAIR-03]  # REPAIR-03 claimed PARTIAL per plan objective; Phase 7-9 overlap flagged to owner per D-09

coverage:
  - id: D1
    description: "Gated pruning-helper shim vendored verbatim from transformers v4.49.0 and wired into apply_patches() (active from dnallm import; 4.x no-op)"
    requirement: EXEC-01
    verification:
      - kind: unit
        ref: "tests/utils/test_transformers_compat.py::TestRemoteCodePruningHelpers (6 tests, 26 passed total)"
        status: pass
      - kind: command
        ref: ".venv/bin/python -c \"import transformers, dnallm; from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer; print(transformers.__version__)\" -> 5.17.0, exit 0"
        status: pass
      - kind: command
        ref: "grep -c 'def _patch_remote_code_pruning_helpers' == 1; total-name count >= 2 (apply_patches wiring); absence gate + sentinel present"
        status: pass
    human_judgment: false
  - id: D2
    description: "Red-first real-model smoke for zhangtaolab/nucleotide-transformer-v2-100m-promoter: born red reproducing GAP-1 exactly, landed as the ladder-terminal typed environment-unavailable skip carrying the exact deeper-breakage traceback"
    requirement: EXEC-01
    verification:
      - kind: command
        ref: "RED: .scratch/05-04/red-run-smoke.log + red-evidence.json (check tdd-red-evidence -> RED_EVIDENCE_OK; failure chain modeling_esm.py:36 ImportError -> model.py:888 ValueError)"
        status: pass
      - kind: command
        ref: "POST-SHIM: .venv/bin/python -m pytest tests/models/test_model_remote_code.py -q -> 1 skipped (typed), exit 0; attach check exit 0"
        status: pass
      - kind: command
        ref: "tracer feedback gate: both <verify> commands re-run end-to-end, exit 0/exit 0"
        status: pass
    human_judgment: false
  - id: D3
    description: "Owner hand-off: benchmark notebook census FAIL row (third registry model not executable on transformers 5.17 via any sanctioned route) + Phase 7-9 overlap rescoping decision per D-09"
    verification: []
    human_judgment: true
    rationale: "Resolution requires an owner decision among non-plan options (PretrainedConfig legacy-defaults patch, transformers pin for this consumer, or checkpoint re-export); D-09 explicitly defers roadmap rescoping to the owner at hand-off, not inside this plan"

duration: 27min
completed: 2026-10-02
status: complete
---

# Phase 5 Plan 04: NT x transformers-5.17 pruning-helper shim Summary

**Vendored the transformers-v4.49.0 pruning helpers behind an absence-gated attach (closes the GAP-1 import crash), then terminated D-07's fallback ladder at its designed rung — the remote config also needs removed 4.x PretrainedConfig defaults — landing the smoke as an evidence-backed typed skip with the benchmark notebook flagged as a census FAIL row for the owner.**

## Performance
- **Duration:** 27 min
- **Started:** 2026-10-02T04:47:57Z
- **Completed:** 2026-10-02T05:15:03Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- GAP-1's import failure is closed: importing dnallm attaches `find_pruneable_heads_and_indices`/`prune_linear_layer` (v4.49.0 semantics) to `transformers.modeling_utils` on 5.x; the remote `modeling_esm.py` now imports and the load proceeds deep into model construction.
- The smoke was born red through the exact documented chain (`modeling_esm.py:36` ImportError → `model.py:888` ValueError), RED evidence machine-verified (`RED_EVIDENCE_OK`), and the tracer's verify legs re-ran green end-to-end before expansion.
- The deeper breakage was diagnosed precisely (remote reads `config.is_decoder` at EsmSelfAttention line 335 with `config.add_cross_attention` right behind; neither in config.json) and handled by the ladder's sanctioned terminal rung, not by silently widening the shim.
- Fast behavior-contract tests pin the vendored semantics (head-pruning arithmetic incl. already-pruned offset, linear-prune shapes/values/bias slicing, attachment idempotence, version-agnostic identity) — 26 passed, fast leg 1590 passed with zero new skips.

## Task Commits
1. **Task 1 (tracer, RED): failing NT v2 promoter remote-code smoke** - `ebdf482` (test)
2. **Task 1 (tracer, GREEN): gated pruning-helper shim + ladder-terminal typed skip** - `0329d63` (feat)
3. **Task 1 follow-up (mypy soundness rename, behavior unchanged)** - `42f3df8` (refactor)
4. **Task 2: fast behavior-contract tests for the vendored helpers** - `50e32ba` (test)
5. **Post-task checker-agnostic attachment (ty/pyright + mypy clean, behavior unchanged)** - `e8013fb` (refactor)

## Files Created/Modified
- `dnallm/utils/transformers_compat.py` - vendored `_find_pruneable_heads_and_indices`/`_prune_linear_layer` (source tag named in comment) + `_patch_remote_code_pruning_helpers` (absence gate, `_dnallm_remote_code_pruning_patch` sentinel) wired into `apply_patches()`
- `tests/models/test_model_remote_code.py` - new slow-marked `TestRemoteCodeCheckpointCompat` live smoke; typed skip only on the documented structural marker
- `tests/utils/test_transformers_compat.py` - new `TestRemoteCodePruningHelpers` (6 tests); all 20 pre-existing tests untouched

## Decisions Made
- Ladder termination (see key-decisions): structural config-default dependency is outside "vendored pure helpers" — shim kept for what it fixed, typed skip recorded, notebook flagged. The typed skip is a live probe, not a dead skip: a future environment fix turns it into the real green (1, 2)-logits smoke automatically.
- Owner coverage directive honored: the dnallm/ shim change ships in the same plan as its pytest coverage (Task 2 class + integration smoke); fast lane on touched areas + plan verify commands all run and recorded below.

## Deviations from Plan
**[Rule 1 - Plan-spec bug] already_pruned_heads dict literal TypeErrors** — Found during: Task 2 | Issue: the behavior contract specified `already_pruned_heads={0: 0}` (a dict), but `set - dict` raises TypeError inside the mandated-verbatim v4.49.0 subtraction; the remote EsmAttention passes its `pruned_heads` **set** | Fix: test uses `{0}` (set), preserving the exact contract semantics (heads == {1}; shifted mask row) | Files: tests/utils/test_transformers_compat.py | Verification: 26 passed | Commit: 50e32ba

**[Rule 1 - Annotation soundness] upstream heads rebinding trips mypy** — Found during: Task 2 acceptance (mypy) | Issue: v4.49.0 rebinds the `heads` param (list→set); mypy flags the incompatible reassignment and return narrowing | Fix: renamed the post-subtraction set (`pruned_heads`); returned set and mask arithmetic identical | Files: dnallm/utils/transformers_compat.py | Verification: 26 passed; diagnostic mypy (--python-version 3.13) reports 0 errors in the file | Commit: 42f3df8

**[Rule 1 - Checker dialect] ty/pyright do not honor mypy `# type: ignore[attr-defined]`** — Found during: post-task review (owner IDE runs Astral ty) | Issue: the three direct module-attribute assignments in `_patch_remote_code_pruning_helpers` produced 3 user-visible ty Errors (unresolved attributes on transformers.modeling_utils) | Fix: literal-name `setattr` calls (invisible to static attribute resolution — mypy, ty and pyright all clean with zero ignore comments), ruff `set-attr-with-constant` silenced via the project's `ruff: ignore[name]` dialect; hasattr gate and getattr sentinel untouched | Files: dnallm/utils/transformers_compat.py | Verification: ruff check/format clean, mypy diagnostic 0 errors, 26 passed + 1 typed skip, attach identity re-asserted | Commit: e8013fb

**Total deviations:** 3 auto-fixed (3x Rule 1). **Impact:** none on runtime behavior — all three are annotation/suppression-dialect repairs; the pinned patch properties (attach-when-absent, sentinel, apply_patches wiring) are unchanged and test-pinned.

## TDD Gate Compliance
- Task 1 (tracer) followed the full RED→GREEN contract: RED commit `ebdf482` precedes feat `0329d63`; RED evidence persisted and machine-verified (`check tdd-red-evidence` → `RED_EVIDENCE_OK`, target test failed through the documented chain).
- Task 2 is `tdd="true"` but by plan design its tests are characterization tests against the already-landed Task-1 shim (its precondition is "Task 1 landed"), so they land green-on-arrival; the RED half of the cycle is carried by Task 1's red run of the same feature. Gate commits present: `test(05-04)` (ebdf482, 50e32ba) and `feat(05-04)` (0329d63).

## Issues Encountered
- **Ladder structural rung (live risk realized, exactly as D-07 anticipated):** after the import fix the load fails at `'EsmConfig' object has no attribute 'is_decoder'` (remote `modeling_esm.py:335`, raised via transformers 5.17 `heterogeneity/configuration_utils.py:312`). Not expressible as vendored pure helpers → shim not extended; exact traceback embedded in the typed skip evidence; benchmark notebook = census FAIL row (owner hand-off, D-09).
- **Native-ESM route refuted (scratch probe, never committed):** `trust_remote_code=False` load fails on `intermediate.dense.weight` mismatch (ckpt 4096x512 vs native 2048x512 from config.json) — the remote `esm_config.py` derives dims config.json lacks. Log: `.scratch/05-04/probe-native-esm.log` (gitignored, session-scoped; key lines quoted above).
- **Pre-existing mypy environment failure:** project-config mypy (`python_version = 3.10`) aborts on the installed numpy 2.x stubs' PEP-695 `type` statements ("Type statement is only supported in Python 3.12 and greater") for every file, before and after this plan — identical on untouched files. A `--python-version 3.13` diagnostic run proves 0 errors in `transformers_compat.py`.
- **check_docs_sync.py exits 1 on the owner's dirty baseline, not on plan damage:** the three DIFFER paths are exactly the three owner-session notebooks the dispatch note marks as not-mine-to-touch (execution-output churn); this plan's commits touch zero mirrored paths (`git diff --name-only 8fa9e05..HEAD -- example/ docs/example/` → empty). Verification item recorded as environment-blocked; repair belongs to the owner's notebook-commit decision.
- **Tracer feedback gate:** re-ran both tracer `<verify>` commands end-to-end (exit 0 + exit 0) — `⚡ Tracer verified end-to-end — expanding` — before Task 2.

## Known Stubs
None. The typed skip in `tests/models/test_model_remote_code.py` is not a stub: it is the plan's sanctioned terminal deliverable (D-07 ladder), carries the exact traceback as evidence, is registered against the pre-existing `environment-unavailable:` allowlist prefix, and self-heals into the real green smoke when the environment gap closes. Recorded in the broken-windows ledger as an open census item for ship-time visibility.

## User Setup Required
None - no external service configuration required. (Owner actions are hand-off decisions, not setup: census FAIL row disposition for the benchmark notebook and Phase 7-9 rescoping per D-09.)

## Next Phase Readiness
- GAP-1's import leg is closed; the census plans (05-05/05-06) inherit a precise FAIL row for `example/notebooks/benchmark/benchmark.ipynb` (third model) with three owner options documented: structural PretrainedConfig default patch (rejected by this plan's ladder), transformers pin for that consumer, or checkpoint re-export.
- Fast leg green (1590 passed, 1 pre-existing skip), ruff clean on all three touched files, zero new skips anywhere.
- SCRATCH evidence (red log, red-evidence.json, probe log, TAP converter) lives in gitignored `.scratch/05-04/` per the owner standing rule; key lines are inlined in this SUMMARY.

## Self-Check: PASSED
- tests/models/test_model_remote_code.py — FOUND (defines TestRemoteCodeCheckpointCompat::test_nt_v2_promoter_loads_and_forwards, slow + timeout(1800))
- tests/utils/test_transformers_compat.py — FOUND (TestRemoteCodePruningHelpers, 6 tests; 20 pre-existing untouched)
- Commits ebdf482 / 0329d63 / 42f3df8 / 50e32ba / e8013fb — all FOUND on phs
- Post-check verification: 26 passed + 1 typed skip (exit 0); fast leg 1590 passed, 1 pre-existing skip, 0 new; attach check exit 0

---
*Phase: 05-execution-harness-honest-gates-runner-feasibility*
*Completed: 2026-10-02*
