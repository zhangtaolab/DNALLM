---
phase: 22-registry-foundation-runtime-loader-enrichment
plan: "01"
subsystem: models/registry
tags: [pydantic, pyyaml, importlib-resources, lru-cache, registry, read-only]

requires:
  - phase: none
    provides: first phase of v1.4 (root dependency)
provides:
  - dnallm.models.registry — cached single-parse typed loader, sole parser of model_info.yaml (REG-01)
  - resolve / try_resolve by exact entry name or repo id; get_by_task_type / get_by_base_model / all_entries
  - is_lab_model + sort_key_lab_first (189/47 partition, PFC-07 anchor)
  - effective_max_length / effective_tokenizer_type with base_model-chain inheritance and cycle guard (D-02 loader half)
  - REG-03 loader-side guards: sole-parser mentioner allowlist, no-write-API verb scan, no-emitter/open-mode source scan, frozen entries
affects: [23-pre-flight-core, 24-preset-loading-rewire, 26-mcp-consistency, 22-02-enrichment]

actuals:
  tokens: 10898   # chars/4 over the realized dnallm/+tests/ diff (43,591 chars / 4)
  tasks: 3
  commits: 6      # MEASURED: git rev-list --count d7805c9..HEAD

tech-stack:
  added: []        # zero new dependencies (pydantic + pyyaml + importlib.resources, all core)
  patterns:
    - "lru_cache(maxsize=1) single-parse loader with reload() cache-clear hook"
    - "frozen Pydantic entry models with hard validation at first access (D-01b)"
    - "structure-guard tests policing read-only invariants (mentioner allowlist, verb scan, AST mode scan)"

key-files:
  created:
    - dnallm/models/registry.py
    - tests/models/test_registry.py
  modified:
    - dnallm/models/__init__.py

key-decisions:
  - "TaskInfo.label_names accepts the packaged regression-entry bare-string form (36 entries) via a field_validator wrapping to a one-element list — keeps the published contract list[str] | None and matches TaskConfig's regression label_names convention"
  - "Sole-mentioner allowlist pinned to {dnallm/models/registry.py, dnallm/mcp/server.py} (the stale :821 integration comment, Phase 26 rewire target)"
  - "Local coverage measured via `coverage run` + filtered report — pytest-cov is broken in the dev venv by numpy 2.5.3/py3.13 (known deferred item; CI matrix legs unaffected)"

patterns-established:
  - "Sole-parse-owner loader: one module owns a packaged YAML; grep-guard tests make second parsers CI-red"
  - "Deliberate-violation probe discipline for guard tests: violate locally, observe red, revert, then commit"

requirements-completed: [REG-01, REG-03]

coverage:
  - id: D1
    description: "Cached single-parse typed loader end-to-end: resolve/try_resolve by exact name or repo id, frozen entries, loud validation (schema violations naming the offender, duplicates, missing sections, resource/parse failures as ValueError), reload() re-parse"
    requirement: REG-01
    verification:
      - kind: unit
        ref: "tests/models/test_registry.py#test_resolve_by_entry_name_returns_typed_entry"
        status: pass
      - kind: unit
        ref: "tests/models/test_registry.py#test_malformed_entry_fails_loudly_naming_offender"
        status: pass
      - kind: unit
        ref: "tests/models/test_registry.py#test_reload_forces_reparse"
        status: pass
      - kind: other
        ref: "command: .venv/bin/python -c \"from dnallm.models.registry import resolve; e = resolve('Plant DNABERT BPE promoter'); assert e.model == 'zhangtaolab/plant-dnabert-BPE-promoter' and e.task.num_labels == 2 and e.kind == 'finetuned'\" -> ok"
        status: pass
    human_judgment: false
  - id: D2
    description: "Query surface across all 236 entries: task_type/base_model lookups spanning both sections, file-order all_entries, is_lab_model 189/47 partition, sort_key_lab_first lab-first name-stable ordering, effective-value inheritance (own-override wins, base chain, None propagation, unresolvable base, cycle guard)"
    requirement: REG-01
    verification:
      - kind: unit
        ref: "tests/models/test_registry.py#test_get_by_task_type_spans_both_sections"
        status: pass
      - kind: unit
        ref: "tests/models/test_registry.py#test_is_lab_model_and_sort_key_partition_and_order"
        status: pass
      - kind: unit
        ref: "tests/models/test_registry.py#test_effective_value_cycle_returns_none"
        status: pass
      - kind: other
        ref: "command: from dnallm.models import resolve, all_entries, is_lab_model, sort_key_lab_first; len(all_entries()) == 236; lab count == 189; lab-first key ordering -> ok"
        status: pass
    human_judgment: false
  - id: D3
    description: "REG-03 loader-side read-only guarantees permanently enforced: sole-parser mentioner allowlist (exactly registry.py + the stale server.py:821 comment), no write/serialize verb on the module's public names, no pyyaml emitter token and no open mode beyond 'r' in the source, frozen entries"
    requirement: REG-03
    verification:
      - kind: unit
        ref: "tests/models/test_registry.py#test_registry_module_is_sole_parser_of_model_info_yaml"
        status: pass
      - kind: unit
        ref: "tests/models/test_registry.py#test_registry_module_exposes_no_write_api"
        status: pass
      - kind: unit
        ref: "tests/models/test_registry.py#test_registry_module_source_has_no_yaml_emitter"
        status: pass
      - kind: other
        ref: "command: grep -rl 'model_info.yaml' dnallm --include='*.py' minus allowlist | wc -l == 0 -> CLEAN"
        status: pass
    human_judgment: false
  - id: D4
    description: "Eleven-name facade re-export from dnallm.models with import-time laziness (no YAML parse at import)"
    requirement: REG-01
    verification:
      - kind: unit
        ref: "tests/models/test_registry.py#test_subpackage_facade_exports_registry_api"
        status: pass
      - kind: other
        ref: "command: import dnallm.models.registry; _load_registry.cache_info().currsize == 0 after import -> lazy ok"
        status: pass
    human_judgment: false

duration: 26 min
completed: 2026-10-20
status: complete
plan_head_before: d7805c92ab087c1799af8a094c86bc78c5c77a9d
plan_head_after: 431d1b7019be410bc9643674207676a039ac2a06
---

# Phase 22 Plan 01: Runtime Loader & Enrichment — Registry Foundation Summary

**Cached read-only typed registry loader (sole parser of model_info.yaml, 236 entries) with exact name/repo/task_type/base_model lookups, lab-first helpers, and base_model-chain inheritance — born read-only behind four probe-verified structure guards.**

## Performance

- **Duration:** 26 min (10:33:30Z – 11:00:15Z, plus close-out)
- **Tasks:** 3 (all tdd="true"; Task 1 tracer)
- **Commits:** 6 (2 RED test, 2 GREEN feat, 2 test: guards + boundary proof)
- **Files:** 3 (2 created, 1 modified), 1,024 insertions
- **Coverage:** dnallm/models/registry.py 119/119 statements, **100%** (floor 96%)

## Accomplishments

- REG-01 fully delivered: `dnallm/models/registry.py` parses the packaged YAML exactly once per process (`lru_cache` + `reload()`), validates hard through frozen Pydantic models (D-01b), and serves exact-match lookups by entry name and repo id; every lookup flavor works across all 236 entries (`get_by_task_type`, `get_by_base_model`, `all_entries`).
- D-02 loader half: `_effective_value` resolves `max_length` / `tokenizer_type` along the `base_model` chain at query time — own override wins, visited-set cycle guard, unresolvable bases surface as None (no runtime probing).
- Lab-first helpers measured and pinned: `is_lab_model` partitions 189 lab / 47 third-party; `sort_key_lab_first` ranks `zhangtaolab/*` first with stable name order (PFC-07 contract anchor).
- REG-03 loader-side guards landed as structure tests: sole-parser mentioner allowlist `{registry.py, mcp/server.py}`, no-write-API verb scan, no-emitter/no-write-mode source scan, frozen entries — each guard probe-verified red under deliberate violation.
- Facade: eleven public names re-exported from `dnallm.models`; importing stays lazy (no file open at import, proven via cache_info).
- Existing 9 kwt integrity tests pass unchanged; the full `tests/models/` lane is 514 passed / 0 failed.

## Task Commits

Each task was committed atomically (TDD: RED then GREEN per task):

1. **Task 1 (tracer): end-to-end typed lookup** — RED `d6f0589` (test) → GREEN `94c9819` (feat, amended for ruff-clean suppression form)
2. **Task 2: query surface + inheritance** — RED `5b7e991` (test) → GREEN `1e6a5e7` (feat)
3. **Task 3: REG-03 read-only guards** — `9031398` (test; guards pass by design against Task 1/2 output — RED proof via four deliberate-violation probes, below) + `431d1b7` (boundary proof closing must-have truth #10)

**Tracer feedback gate:** verify re-run end-to-end after Task 1 (pytest 18/18 + the `-c` resolve command + facade import) — passed, expanded to Tasks 2–3 (end-of-phase mode, automated-only verify).

## TDD Record

- **Task 1 RED:** `from dnallm.models import registry` → ImportError, 1 collection error, exit 2 — every named target test absent-behavior red (module does not exist). **GREEN:** 18 passed.
- **Task 2 RED:** 13 new tests failed in-test (AttributeError on the missing query/inheritance functions; ImportError on the 11-name facade) while the 18 Task-1 tests stayed green — the exact planned surface absent. **GREEN:** 31 passed.
- **Task 3:** guard tests pass against the just-built module by design (plan-directed); RED proof = deliberate local violation of each guard (bogus mentioner file, `save_everything` verb, `dump` emitter comment, `frozen=True` removed) — each turned the suite red, then reverted to green.
- REFACTOR: none needed — no cleanup commit (Task 2's `try_resolve` was written with explicit None checks mirroring `resolve` from the start).
- Evidence records: `.planning/phases/22-registry-foundation-runtime-loader-enrichment/tdd-records/` (real command, exit code, unmodified output, expected/actual, semantic assessment per task).
- **Classifier caveat (honesty note):** `gsd_run check tdd-red-evidence` returned INVALID_RED/`invalid_record` — its report adapters (TAP, Maven-shaped JUnit, unittest text, swift, go) do not parse pytest's `testsuites` JUnit XML, and the machine gate is off for this plan (`workflow.tdd_mode: false`, plan `type: execute`). The RED evidence above is the real command output with semantic assessment, per the plan's own "observe RED (kwt precedent)" instruction.

## Files Created/Modified

- `dnallm/models/registry.py` — the loader: TaskInfo/RegistryEntry (frozen, extra=forbid), `_build_registry` loud validation, `_load_registry` (CSafeLoader w/ SafeLoader fallback, UTF-8, lru_cache), `resolve`/`try_resolve`/`reload`, `get_by_task_type`/`get_by_base_model`/`all_entries`, `is_lab_model`/`sort_key_lab_first`, `_effective_value`/`effective_max_length`/`effective_tokenizer_type`
- `tests/models/test_registry.py` — 36 tests: end-to-end resolve, cache/reload, loud validation, boundary proof, query surface, lab-first, synthetic inheritance matrix, facade import, four REG-03 guards
- `dnallm/models/__init__.py` — eleven registry names re-exported and added to `__all__`

## Decisions Made

- **label_names bare-string acceptance (Rule 1, see Deviations):** 36 packaged regression entries store a free-text target description in `task.label_names` as a plain string; the schema wraps it into a one-element list at validation (matches `TaskConfig`'s regression convention, `configs.py:153`).
- **extra="forbid" on both models:** typo'd future fields (e.g. `max_lenght`) fail loudly at first access instead of being silently ignored — the strongest reading of D-01b.
- **Suppression comment for S506:** the plan-mandated `getattr(yaml, "CSafeLoader", yaml.SafeLoader)` is statically unresolvable for ruff's unsafe-yaml-load rule; suppressed inline with justification (both loaders are safe).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] label_names schema rejected 36 real regression entries**
- **Found during:** Task 1 (schema dry-run against the real YAML before writing the loader)
- **Issue:** The plan's literal schema (`label_names: list[str] | None`) makes `_build_registry` reject the packaged file — 36 promoter-strength regression entries carry `label_names: "promoter strength in tobacco leaves"` (a bare string). Plan truth #1 (end-to-end resolve over the real file) and truth #6 (loud failure on malformed data) contradicted each other under the literal schema.
- **Fix:** `TaskInfo._wrap_bare_string_label_names` field_validator (mode="before") wraps a bare string into a one-element list. Consumers always see `list[str] | None`; the packaged file is untouched.
- **Files modified:** dnallm/models/registry.py, tests/models/test_registry.py (test_regression_bare_string_label_names_wrapped_to_list)
- **Verification:** resolve of `zhangtaolab/plant-dnabert-BPE-promoter_strength_leaf` returns `label_names == ["promoter strength in tobacco leaves"]`; all 236 entries validate.
- **Committed in:** 94c9819

**2. [Rule 1 - Bug] ruff unsafe-yaml-load fired on the plan-mandated CSafeLoader call**
- **Found during:** Task 1 GREEN commit (ruff check)
- **Issue:** S506 cannot resolve `Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)` and rejects any `yaml.load` call — CI's `ruff check` gate would fail on the plan's required fast-loader form.
- **Fix:** inline `# ruff: ignore[unsafe-yaml-load]` with a comment naming both loaders as safe (CSafeLoader = C implementation of SafeLoader; measured 12x faster on this file per STACK.md).
- **Files modified:** dnallm/models/registry.py
- **Verification:** `ruff check` clean on all touched files; parse path unchanged.
- **Committed in:** 94c9819 (amended before push)

---

**Total deviations:** 2 auto-fixed (2x Rule 1)
**Impact on plan:** Both fixes were required for the plan's own success criteria to hold against the real packaged data and the real lint gate. No scope creep; no production behavior beyond the plan's contract.

## Issues Encountered

- **pytest-cov unusable in the local dev venv:** `--cov` activates a tracer under which numpy 2.5.3's C modules fail with `ImportError: cannot load module more than once per process` on py3.13 — the already-documented deferred item ("numpy 2.5.x cannot be instrumented by coverage on py3.13"). Coverage was measured with plain `coverage run -m pytest` + `coverage report --include='*/dnallm/models/registry.py'` (100%). CI matrix legs (numpy 1.26.4 / 2.2.0) are unaffected.
- **GREEN commit initially landed with a ruff S506 error** (hooks are not installed in this clone, and my ruff pre-check ran with `;` chaining so the commit proceeded). Fixed immediately and `--amend`ed before any push; every subsequent commit was ruff-verified with `&&` chaining.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- REG-01 complete; REG-03 loader-side complete (the data-side half — provenance-presence test + enrichment invariants — lands with 22-02, which also declares REG-03).
- Zero consumers rewired, as planned: no production module outside `dnallm/models/__init__.py` imports registry. Phases 23/24/26 consume the facade.
- 22-02 note: `test_effective_value_live_registry_pre_enrichment_is_none` pins the pre-enrichment None contract and is expected to flip (values, not seam) when enrichment lands.
- `tests/models/test_registry.py::test_registry_module_is_sole_parser_of_model_info_yaml` will go red if 22-02's probe script lives under `dnallm/` and mentions the file — keep it in `.scratch/`/`scripts/` per CONTEXT.md discretion.

## Self-Check: PASSED

- Files exist: dnallm/models/registry.py, tests/models/test_registry.py, dnallm/models/__init__.py — all FOUND
- Commits d6f0589, 94c9819, 5b7e991, 1e6a5e7, 9031398, 431d1b7 — all ancestors of HEAD
- TDD gate commits present: `test(22-01)` and `feat(22-01)` both in git log
- evaluation-scope (commits-only, plan 22-01): resolved, 6 commits listed
- Plan verification block: 44/44 registry+kwt suites, 514/514 models lane, coverage 100% ≥ 96, mentioner gate CLEAN, facade import lazy with 236 entries

---
*Phase: 22-registry-foundation-runtime-loader-enrichment*
*Completed: 2026-10-20*
