---
phase: 28-optional-extras-for-the-two-external-package-models-deepspee
plan: "02"
subsystem: models
tags: [extra-required-tf5, census-disposition, crossdna, flash-attn, gena-lm-bigbird, deepspeed, support-vocabulary, registry-write-back, generated-docs]

requires:
  - phase: 28-optional-extras-for-the-two-external-package-models-deepspee
    provides: curated deepspeed/flash-attn extras, sha256-pinned installer, measured extras-replica evidence (.scratch/p28-extras/compat.json), owner ruling A
provides:
  - extra-required-tf5 verdict accepted by all six fail-closed vocabulary surfaces and the ordered disposition.extra branch in compat_verdict (D-01, D-02)
  - CrossDNA registry flip to extra-required-tf5 on the Plan-01 measured green record (D-03, D-15)
  - BigBird honest split evidence under owner ruling A - skip disposition kept, load-green/forward-red named (D-03/D-04 ruling half)
  - Shared support words for the extra-required status and CrossDNA's exact extra (D-14)
  - Regenerated 49-row model_compatibility.md, byte-identical on re-emission (D-15)
  - Actionable missing-extra loader guidance at both load-chain interception points (D-17)
affects: [28-03 optional_dependencies rewrite, 28-04 extras lane, TUI/MCP/preflight support columns]

actuals:
  tokens: 17805
  tasks: 2
  commits: 4

tech-stack:
  added: []
  patterns:
    - ordered disposition branch (extra before skip AND green) so one judgment maps both the extras-green seed and default-env failure shapes to one verdict
    - guidance-marker attribute (_dnallm_missing_extra_guidance) letting a specific ImportError survive a broad except-Exception wrap unchanged

key-files:
  created: []
  modified:
    - dnallm/models/model.py
    - dnallm/models/special/crossdna.py
    - dnallm/models/registry.py
    - dnallm/models/support.py
    - scripts/census_compat.py
    - dnallm/models/model_info.yaml
    - tests/models/compat_census.json
    - docs/user_guide/model_compatibility.md
    - tests/models/test_special/test_family_handlers.py
    - tests/models/test_census_compat.py
    - tests/models/test_model_info_registry.py
    - tests/models/test_support.py
    - tests/models/test_compat_docs_consistency.py
    - tests/models/test_registry.py

key-decisions:
  - "OWNER RULING A applied throughout: CrossDNA-only extra-required-tf5 flip; BigBird keeps its committed skip disposition with evidence naming the load-green/forward-red split - no new status vocabulary, BigBird never flips"
  - "BigBird's support word stays 'needs deepspeed' (plan-literal 'works with the deepspeed extra' withheld): the forward is measured structurally dead even with the extra, so a works-word would overclaim; CrossDNA alone gets 'works with the flash-attn extra'"
  - "BigBird's docs lead class moves external-package -> library-change via a new lockstep needle ('upstream-dead Triton API', generator + gate): the residual blocker after the extra greens the load is the removed triton superblock API, not a package gap installing something could fix"
  - "Seed asymmetry is deliberate: CrossDNA carries the Plan-01 measured green load record (in-artifact flip evidence, 27.2 SHIM_RESTORED precedent); BigBird keeps its default-env failure record - the split story lives in the disposition text, the artifact stays a default-install census"
  - "TestDispositionCompleteness vocabulary grows to {fix, skip, extra} and a green record may carry exactly one extra judgment - the companion edit the plan's seed shape requires"
  - "DOCS_DATA_DATE stays 2026-10-29 (Plan-01 probe date); never a wall-clock stamp"

patterns-established:
  - "Guidance-marker exception attribute: specific user-facing ImportErrors set _dnallm_missing_extra_guidance; generic wraps re-raise marked errors unchanged instead of flattening them into ValueError"

requirements-completed: [D-01, D-02, D-03, D-04, D-14, D-15, D-17]

coverage:
  - id: DEL-1
    description: "Missing-extra loader guidance: deepspeed ModuleNotFoundError (direct/nested/cyclic) and fla's Flash Attention ImportError become plain-language ImportErrors with exact pip commands, cause retained; both CrossDNA paths and both generic wrap sites intercepted; unrelated exceptions keep prior behavior (D-17)"
    requirement: D-17
    verification:
      - kind: unit
        ref: "tests/models/test_special/test_family_handlers.py::TestMissingExtraGuidance (17 tests) + tests/models/test_model.py generic-wrap pin unchanged (426 passed incl. test_model.py + test_crossdna.py)"
        status: pass
    human_judgment: false
  - id: DEL-2
    description: "extra-required-tf5 accepted by all six fail-closed surfaces (RegistryEntry Literal, census TF_COMPAT_STATUSES, both independent TF_COMPAT_VOCABULARY mirrors, SUPPORT_WORDS, DOCS_STATUS_CELLS) with the ordered extra-before-skip-and-green branch (D-01, D-02)"
    requirement: D-01
    verification:
      - kind: unit
        ref: "tests/models/test_census_compat.py::TestCompatVerdictMapping extra tests + tests/models/test_registry.py round-trip + tests/models/test_model_info_registry.py mirror/vocabulary"
        status: pass
    human_judgment: false
  - id: DEL-3
    description: "CrossDNA flips to extra-required-tf5 through the sanctioned writer on the Plan-01 measured green record; exactly one extra-required row with full enabler evidence (D-03, D-15)"
    requirement: D-03
    verification:
      - kind: unit
        ref: "tests/models/test_model_info_registry.py::test_extra_required_evidence_pins_the_measured_extra_verdicts + mirror test; writer dry run: 2 planned replacements / 47 already; --apply: 2 replacements; --verify: 49 matched OK"
        status: pass
    human_judgment: false
  - id: DEL-4
    description: "BigBird split evidence under owner ruling A: skip disposition kept, evidence names deepspeed 0.19.6, _patch_apply_chunking_to_forward, and the load-green/forward-red split; GPN/Omni-DNA unchanged (D-03/D-04 ruling half)"
    requirement: D-04
    verification:
      - kind: unit
        ref: "tests/models/test_model_info_registry.py::test_broken_evidence_pins_unchanged_under_ruling_a + re-scoped resolve assertions (exit 0)"
        status: pass
    human_judgment: false
  - id: DEL-5
    description: "Support vocabulary: works with an optional extra status word, Works-after-installing docs cell, CrossDNA exact override; BigBird keeps needs deepspeed per ruling A (D-14)"
    requirement: D-14
    verification:
      - kind: unit
        ref: "tests/models/test_support.py (six statuses, four overrides, four cells, all-real-row sweep) + tests/mcp/test_server_streaming.py 104 passed (consumer surfaces)"
        status: pass
    human_judgment: false
  - id: DEL-6
    description: "Regenerated model_compatibility.md with the seven-number summary sentence and both changed rows; re-emission byte-identical (D-15)"
    requirement: D-15
    verification:
      - kind: automated
        ref: "census_compat.py --emit-docs twice + cmp identical; tests/models/test_compat_docs_consistency.py green"
        status: pass
    human_judgment: false

duration: 16min
completed: 2026-10-29
status: complete
commits: 4
plan_head_before: f69754df3f03d4c63dde32a27e60e234c16218da
plan_head_after: 3997652810ebcda44c3d55f31104ee7e1d072814
---

# Phase 28 Plan 02: Optional extras measured flips Summary

**CrossDNA flipped to the new `extra-required-tf5` verdict on Plan-01's measured load+forward green record while BigBird kept its committed skip disposition with honest split evidence per owner ruling A; both missing-extra failures now raise plain-language install guidance at the load chain, and the regenerated compatibility page re-emits byte-identically at 19/21/5/1-extra/3-broken/0-blocked.**

## Performance

- **Duration:** 16 min (2026-10-29T15:50:51Z → 16:07:04Z UTC)
- **Tasks:** 2 (both TDD: RED + GREEN commits each)
- **Files modified:** 14 tracked
- **Estimate vs actual:** 62,000 tokens estimated / 17,805 actual (chars/4 over the realized diff) — the ruling-A re-scope (one flip instead of two) shrank the writer/docs surface

## Writer Receipts (plan output spec)

| Fact | Value |
|---|---|
| Writer dry run | **2 planned replacement(s), 47 already written** (exactly BigBird + CrossDNA) |
| Writer apply | **wrote 0 insertion(s) and 2 replacement(s)** into `dnallm/models/model_info.yaml` |
| Post-apply idempotence | 0 planned / 49 already written |
| `--verify` | **49 entry/entries compared — all matched — "OK: census matches committed registry fields"** (no UNEXPECTED, no VERSION DRIFT, no MISSING) |
| Final distribution | **19 verified / 21 import-verified / 5 shim-required / 1 extra-required / 3 remote-code-broken / 0 census-blocked = 49** |
| DOCS_DATA_DATE | **2026-10-29** (Plan-01 probe date; unchanged — never a wall-clock stamp) |
| Docs byte-identity | fresh `--emit-docs` to /tmp, `cmp` identical; committed page "already current" on re-run |
| Registry diff | exactly 3 lines in `model_info.yaml`: CrossDNA status, CrossDNA evidence, BigBird evidence |

## Accomplishments

- D-17 landed first as a tracer slice: `_raise_missing_extra_import_error` (cycle-safe `__cause__`/`__context__` traversal, direct + nested deepspeed shapes, CrossDNA guidance re-raised unchanged) at both generic wrap sites, and `_from_pretrained_with_flash_attn_guidance` on both CrossDNA `from_pretrained` paths — 17 new offline tests, no network.
- The `extra` disposition maps symmetrically: the extras-green seed AND a default-env failed load both map to `extra-required-tf5` through one branch placed before skip and before the green-load branches, so no green record can ever claim works-as-is and no default re-probe can go unclassifiable.
- Every fail-closed gate (registry Literal, census statuses, two independent vocabulary mirrors, support words, docs cells, docs-consistency rebuild) accepts the new value; the sanctioned writer changed exactly two rows.
- BigBird's public story is now the honest ruling-A split: docs lead reclassified external-package → library-change (new `upstream-dead Triton API` needle in generator + gate lockstep), technical fragment reads "load works with the deepspeed extra but forward is blocked by an upstream-dead Triton API".

## Task Commits

1. **Task 1 RED: failing TestMissingExtraGuidance** — `76b0235` (test)
2. **Task 1 GREEN: load-chain guidance both models** — `78d2a21` (feat)
3. **Task 2 RED: fail-closed pins for the ruling-A flip** — `a15151f` (test)
4. **Task 2 GREEN: CrossDNA flip through every surface** — `3997652` (feat)

TDD gate compliance: both RED runs validated (`RED_EVIDENCE_OK`, records at `.scratch/tdd-red-28-02-task1.json` 16 failed / `.scratch/tdd-red-28-02-task2.json` 13 failed) before their GREEN commits; every `test(28-02)` commit precedes its `feat(28-02)` commit. Tracer feedback gate: the tracer's automated-only `<verify>` re-ran end-to-end green (74/74) before expansion into Task 2 (`HUMAN_VERIFY_MODE=end-of-phase`, no checkpoint).

## Files Created/Modified

- `dnallm/models/model.py` — `_raise_missing_extra_import_error` + invocation at both generic wrap sites (before the ValueError wrap)
- `dnallm/models/special/crossdna.py` — `_from_pretrained_with_flash_attn_guidance` + `_DNALLM_MISSING_EXTRA_GUIDANCE_ATTR`; both `from_pretrained` calls routed
- `dnallm/models/registry.py` — `extra-required-tf5` in the `tf_compat_status` Literal + docstring
- `dnallm/models/support.py` — extra status word, CrossDNA override, docs cell, four-repo/four-string docstrings
- `scripts/census_compat.py` — six-value `TF_COMPAT_STATUSES`, ordered extra branch in `compat_verdict`, seven-number summary sentence, `upstream-dead Triton API` lead needle
- `dnallm/models/model_info.yaml` — writer-produced: CrossDNA flip + BigBird evidence (3 lines)
- `tests/models/compat_census.json` — CrossDNA seeded from the Plan-01 measured green record + extra disposition; BigBird skip text updated (disposition type unchanged)
- `docs/user_guide/model_compatibility.md` — regenerated (summary sentence, CrossDNA cell, BigBird lead+fragment)
- Test gates: `test_family_handlers.py` (+17), `test_census_compat.py` (+7, disposition vocabulary extended), `test_model_info_registry.py` (counts 4→3 broken +1 extra, mirror order, evidence pins), `test_support.py` (six/four/four literals), `test_compat_docs_consistency.py` (six cells, needle, sentence), `test_registry.py` (vocabulary mirror, companion)

## Decisions Made

Recorded in key-decisions frontmatter. The load-bearing one: **owner ruling A governs every surface** — the plan text (written pre-ruling) still says "2 extra-required" in places; the binding ruling (2026-10-29, 28-01-SUMMARY + STATE decision log) re-scopes to CrossDNA-only, and each dependent surface (verify command, support word, seed shape, counts) was re-scoped to match rather than executed literally.

## Deviations from Plan

### Ruling-Driven Re-scopes (owner directives, not auto-fixes)

**1. [Owner Ruling A] BigBird does not flip to extra-required-tf5**
- **Found during:** Task 2 precondition + verify-3
- **Issue:** the plan's verify-3 asserts both BigBird and CrossDNA resolve to `extra-required-tf5`, and its behavior pins two seeded extra rows / a 2/2 broken-extra split — all written before the ruling
- **Fix:** re-scoped exactly as directed: CrossDNA-only flip; BigBird keeps remote-code-broken with split evidence; distribution 19/21/5/1/3/0; exactly one extra-required row; the ruling's quoted split sentence ("load works with the deepspeed extra; forward blocked by an upstream-dead Triton API") rides in the evidence with the measured detail (deepspeed 0.19.6 sparse LUT → triton libtriton.superblock absent in 3.6.0 and 3.8.0, no installable resolution under torch>=2.4)
- **Files modified:** tests/models/compat_census.json, tests/models/test_model_info_registry.py, dnallm/models/model_info.yaml, docs/user_guide/model_compatibility.md
- **Verification:** re-scoped resolve assertions exit 0; --verify 49 matched; all gates green
- **Committed in:** a15151f, 3997652

**2. [Owner Ruling A] BigBird's support word stays "needs deepspeed"**
- **Found during:** Task 2 (support.py edit)
- **Issue:** plan-literal "replace BigBird's override with exactly works with the deepspeed extra" would publish a works-claim for a model whose forward is measured structurally dead — contradicting the ruling's honesty requirement; it would also break the out-of-plan pins (tests/mcp/test_server_streaming.py:926) and pre-date the D-16 guide rewrite
- **Fix:** override unchanged ("needs deepspeed", still true); CrossDNA alone gains "works with the flash-attn extra"; the load-green/forward-red split is carried by the census evidence, docs cell, and the Task-1 loader guidance
- **Files modified:** dnallm/models/support.py (CrossDNA override added only), tests/models/test_support.py (override table gained CrossDNA, BigBird row pinned unchanged with a ruling comment)
- **Verification:** test_support + test_server_streaming (104) green
- **Committed in:** a15151f, 3997652

### Auto-fixed Issues

**3. [Rule 3 - Companion] test_registry.py vocabulary mirror extended**
- **Found during:** Task 2 RED
- **Issue:** `tests/models/test_registry.py` holds a second independent `TF_COMPAT_VOCABULARY` mirror (not in the plan's file list); leaving it at five values leaves the loader-side round-trip gate not covering the new status
- **Fix:** added `extra-required-tf5` to the mirror and its docstrings (five-value → six-value)
- **Files modified:** tests/models/test_registry.py
- **Verification:** 328 tests green across the five gates
- **Committed in:** a15151f

**4. [Rule 2 - Companion] TestDispositionCompleteness learned the extra judgment**
- **Found during:** Task 2 seed design
- **Issue:** the plan's seed shape (green load record + `{"extra": ...}` disposition) violates the gate's committed invariants as written (`keys <= {fix, skip}` and "passing records carry no disposition")
- **Fix:** allowed keys grow to `{fix, skip, extra}` (value rules mirror skip); a non-failure record may carry exactly one extra judgment (never skip/fix on green); `merge_records` untouched as instructed — the generic carry already preserves extra dispositions (green-drop drops only skip), pinned by two new carry-forward tests
- **Files modified:** tests/models/test_census_compat.py
- **Verification:** TestDispositionCompleteness + TestDispositionCarryForward green over the committed 49-record artifact
- **Committed in:** a15151f

---

**Total deviations:** 4 (2 owner-ruling re-scopes applied as directed, 1 Rule 3, 1 Rule 2)
**Impact on plan:** all six fail-closed surfaces, the writer, and the docs page hold one internally consistent ruling-A story; no gate disagrees.

## Issues Encountered

- First RED-evidence record was rejected by the classifier (`INVALID_RED: invalid_record` — plain console tail, unparseable format); re-recorded with `--junitxml` + report metadata, both tasks then passed `RED_EVIDENCE_OK`.
- ruff 0.17.1 caught two test-style issues during GREEN (assert-in-except, composite assertion) and two formatting nits — all fixed before commit; final repo-wide `check` clean, all ten touched Python files format-clean.
- Precondition note: `.scratch/p28-extras/compat.json` BigBird record carries `load.ok=true, forward_ok=false` — the precondition's literal "both forward_ok=true" was superseded by owner ruling A for BigBird's half (the ruling exists precisely because of this measurement); CrossDNA's half is met literally.

## Authentication Gates

None.

## User Setup Required

None - no external service configuration required.

## Known Stubs

None - every shipped surface is wired to measured evidence.

## Next Phase Readiness

- Plan 03 (optional_dependencies.md rewrite, D-16) can proceed: CrossDNA's install guide writes itself from the measured route (`pip install --no-build-isolation 'dnallm[flash-attn]'`, Linux + CUDA); BigBird's section should carry the ruling-A split wording and the deepspeed `<0.19.7` caveat (loader guidance already links the guide).
- Plan 04 (extras lane): the length-64 probe fact rides forward — a lane probing BigBird at `LOAD_SEQUENCE_LENGTH=32` fails on divisibility before the kernel verdict; CrossDNA verifies against its committed `extra-required-tf5` row through the same `--verify` drift gate (the symmetric default-failure mapping is what makes the committed row the lane's expected value).
- The committed `census_compat.py` tool is untouched (probe length, merge_records, SHIM_RESTORED_LOADS, census_compat.yml all unchanged, as instructed).

## Self-Check: PASSED

- All 4 task commits (76b0235, 78d2a21, a15151f, 3997652) are ancestors of HEAD 3997652.
- All modified files exist; both RED evidence records exist on disk; `.scratch/p28-extras/compat.json` intact.
- Final battery re-run green on the committed tree: 328 (five gates) + 104 (MCP/preflight consumers) + 74 (family handlers) passed; writer idempotent (0/49); --verify 49 matched OK; docs re-emission byte-identical; repo-wide ruff 0.17.1 clean.

---
*Phase: 28-optional-extras-for-the-two-external-package-models-deepspee*
*Completed: 2026-10-29*
