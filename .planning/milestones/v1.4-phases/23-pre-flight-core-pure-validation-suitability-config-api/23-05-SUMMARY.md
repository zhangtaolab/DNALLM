---
phase: 23-pre-flight-core-pure-validation-suitability-config-api
plan: "05"
subsystem: testing
tags: [preflight, facade, integration-testing, pydantic, registry, tdd]

# Dependency graph
requires:
  - phase: 23-pre-flight-core-pure-validation-suitability-config-api (plan 01)
    provides: validate_data + DataReport, check_compatibility, report inventory, structure guards
  - phase: 23-pre-flight-core-pure-validation-suitability-config-api (plan 02)
    provides: three-dimension suitability engine, intent-routed lab-first listings, prober shortlist seam
  - phase: 23-pre-flight-core-pure-validation-suitability-config-api (plan 03)
    provides: probe_availability / effective_endpoints / clear_probe_memo (the prober seam default)
  - phase: 23-pre-flight-core-pure-validation-suitability-config-api (plan 04)
    provides: generate_config / edit_config over the shared validate_sections path
provides:
  - The complete dnallm.preflight facade — the stable shell-facing contract for Phase 24/25/26 (validate_data, check_compatibility, probe_availability, generate_config, edit_config, effective_endpoints, clear_probe_memo + every report type)
  - The cross-module user-story proof on the happy path and the severed-network path (facade-only imports)
  - The docstring-facade drift guard (structure test) keeping documentation and __all__ in lockstep
  - Phase-23 closeout evidence: full lane green, every preflight module >=96% covered with zero omit entries, adjacent lanes green, ruff clean, security spot checks pinned
affects: [24 load_preset_model rewire, 25 TUI preflight panels, 26 MCP finding/tool conventions]

# Actuals (#2632) — same scale as the plan estimate (chars/4 over the realized diff)
actuals:
  tokens: 3662
  tasks: 2
  commits: 3

tech-stack:
  added: []  # zero new dependencies (facade re-exports only)
  patterns:
    - "Facade-only consumption proof: the integration suite imports only dnallm.preflight — if a shell cannot drive the story through the facade, the facade is incomplete"
    - "Docstring-facade drift guard: every backtick-documented public name that resolves on the package must sit in __all__"

key-files:
  created:
    - tests/preflight/test_integration.py
  modified:
    - dnallm/preflight/__init__.py
    - tests/preflight/test_structure.py

key-decisions:
  - "23-05: the story's edit_config input is the RAW mapping derived from GeneratedConfig.config (model_dump per section) — the generated report holds validated section objects, and the round-trip editor's documented input is a parsed mapping; passing the section objects directly yields mapping errors (verified live), so the test documents the serialization step a real shell performs"
  - "23-05: the integration suite accesses the new facade names via the module object (import dnallm.preflight as preflight) with from-imports carrying the Wave-1 names and report types — RED therefore fails on planned assertions (missing-names list, AttributeError) instead of a collection-time ImportError"
  - "23-05: the drift guard reads backtick-quoted bare identifiers from the package docstring that resolve as public module attributes — documented-but-unexported names fail, while prose references to submodules/paths are filtered out"

patterns-established:
  - "Probe-verified structure guards extended, not forked: the docstring drift guard was proven red under a deliberate __all__ omission before being committed green"

requirements-completed: [PFC-01, PFC-02, PFC-03, PFC-04, PFC-05, PFC-06, PFC-07]

coverage:
  - id: D1
    description: "Complete facade: every documented public name (validate_data, check_compatibility, probe_availability, generate_config, edit_config, effective_endpoints, clear_probe_memo + report types) resolves via the package and sits in __all__ (D-01 cross-shell contract)"
    verification:
      - kind: unit
        ref: "tests/preflight/test_integration.py#test_facade_exports_complete_public_api"
        status: pass
      - kind: other
        ref: "command: python -c facade gate (all six function names hasattr + in __all__) — prints 'facade ok'"
        status: pass
    human_judgment: false
  - id: D2
    description: "Happy-path user story end-to-end through the facade: tmp csv (mixed lengths, one >512 row, N rows) -> validate_data (5 rows counted) -> check_compatibility(predict, fake remote-confirmed prober, shortlist 3, lab-first entry 0, 3 probed entries) -> generate_config (zero errors, task prefilled) -> edit_config num_labels relabel (zero errors)"
    verification:
      - kind: integration
        ref: "tests/preflight/test_integration.py#test_user_story_happy_path_end_to_end"
        status: pass
    human_judgment: false
  - id: D3
    description: "Severed-network story (SC-5/PFC-06): raising ConnectionError prober -> availability unknown/PROBE_ERROR on the shortlist, zero exceptions, generate/edit unaffected, every step returns a report object"
    requirement: PFC-06
    verification:
      - kind: integration
        ref: "tests/preflight/test_integration.py#test_user_story_severed_network_still_completes"
        status: pass
    human_judgment: false
  - id: D4
    description: "Intent routing in the story: finetune branch lists pretrained entries only (disjoint from the predict branch's finetuned entries, num_labels None) with the lab-first prefix property on both listings (PFC-05/PFC-07)"
    requirement: PFC-05
    verification:
      - kind: integration
        ref: "tests/preflight/test_integration.py#test_finetune_intent_lists_pretrained_entries_lab_first"
        status: pass
    human_judgment: false
  - id: D5
    description: "Phase closeout: full preflight lane green (189 passed, 1 slow deselected), every dnallm/preflight/*.py module >=96% covered with zero omit entries (__init__/compatibility/config_gen/report/validation 100%, availability 99%), registry + configuration lanes green (189), ruff clean, security spot checks pinned (pkl refusal, unsafe-YAML ConstructorError with no payload executed, no write without output_path)"
    verification:
      - kind: other
        ref: "command: COVERAGE_FILE=.coverage-p2305 coverage run -m pytest tests/preflight/ -m 'not slow' && coverage report --include='*/dnallm/preflight/*' --fail-under=96 (6 modules, total 99%)"
        status: pass
      - kind: other
        ref: "command: pytest tests/models/test_registry.py tests/configuration -q (189 passed) + ruff check dnallm/preflight/ tests/preflight/ dnallm/configuration/configs.py (clean)"
        status: pass
      - kind: unit
        ref: "tests/preflight/test_structure.py#test_docstring_documented_names_are_exported + omit-list and root-facade guards (in-lane)"
        status: pass
    human_judgment: false

# Metrics
duration: 6 min
completed: 2026-10-20
status: complete
plan_head_before: 426a2938b3ae98143a9827b53bda606ddbc3c755
plan_head_after: 5e92885d5477aa8fe73a92dc19c2fc020b4af943
commits: 3
---

# Phase 23 Plan 05: Facade Completion + Phase Closeout Summary

**Complete dnallm.preflight facade (four user-story functions + endpoint reporting + report types) proven by a facade-only cross-module user story on happy and severed-network paths, closed out at 99% package coverage with all six modules >=96%**

## Performance

- **Duration:** 6 min
- **Started:** 2026-10-20T17:30:37Z
- **Completed:** 2026-10-20T17:36:58Z
- **Tasks:** 2/2
- **Files modified:** 3 (1 created, 2 modified; 278 insertions)

## TDD Gate Compliance

Task 1 carried `tdd="true"`; it follows the test -> feat commit-scope contract (executor log validation):

| Task | RED | GREEN | REFACTOR |
|------|-----|-------|----------|
| 1 (facade + user story) | 5c4ebeb (3 failed on the planned missing facade names — 1 assertion failure listing the missing names, 2 AttributeError on the absent generate_config; 1 prior-green routing test) | 5b4bb4a | not needed (re-export-only facade born clean, ruff-formatted) |
| 2 (closeout, not tdd) | — | 5e92885 (drift guard probe-verified red under a deliberate `__all__` omission before commit) | — |

RED evidence note: this is a `type: execute` plan (not `type: tdd`), so the formal `tdd-red-evidence` classifier gate does not apply; the commit-scope contract (test -> feat, RED fails on planned behavior, GREEN passes) was followed and is reflected above.

## Accomplishments

- Facade complete: `__all__` extended with `probe_availability`, `effective_endpoints`, `clear_probe_memo` (from availability) and `generate_config`, `edit_config` (from config_gen); the report types (`AvailabilityReport`, `SectionError`, `GeneratedConfig`, ...) were already exported in Wave 1. The docstring now documents the complete API including the availability network boundary and the write-only-when-given output contract. Facade stays re-export-only; still not a root `dnallm` facade member.
- The phase's user story proven end-to-end through the public facade only (zero submodule imports — grep-pinned): tmp csv with mixed lengths / one >512-char row / N-containing rows -> `validate_data` (5 rows counted) -> `check_compatibility` predict listing (fake remote-confirmed prober, shortlist 3, entry 0 lab, 3 entries probed) -> `generate_config` on the top lab listing entry (zero errors, task prefilled from the registry) -> `edit_config` num_labels relabel (zero errors).
- Severed-network story (SC-5): the same flow with a ConnectionError-raising prober — shortlist availability degrades to unknown/PROBE_ERROR, no exception anywhere, generate/edit unaffected, every step returns a report object.
- Intent routing pinned inside the story: the finetune branch lists pretrained entries only (disjoint from predict's finetuned entries; all num_labels None), both listings carry the lab-first prefix property (PFC-07).
- Phase closeout: full preflight lane 189 passed (1 slow deselected); every module >=96% (five at 100%, availability 99% — the 2 uncovered lines are the racing-delete du guard documented in 23-03); pyproject omit list gained no preflight entry; registry + configuration lanes 189 green; ruff clean and formatted; the three security spot checks pinned as one-off commands (pkl -> UNSUPPORTED_FORMAT_PKL never unpickled; python/object YAML -> ConstructorError with no payload artifact executed; generate_config writes nothing without output_path).
- New structure guard: every public name documented in the package docstring must sit in `__all__` (drift guard for future shells), probe-verified red under a deliberate omission.

## Task Commits

1. **Task 1 (facade + integration story, tdd)** — RED `5c4ebeb` + GREEN `5b4bb4a`
2. **Task 2 (closeout: drift guard + verification battery)** — `5e92885`

**Plan metadata:** recorded in the completion report.

## Files Created/Modified

- `dnallm/preflight/__init__.py` — facade re-exports completed (availability + config_gen names), docstring documents the full API, `__all__` final
- `tests/preflight/test_integration.py` — 4 facade-only user-story tests (facade completeness, happy path, severed network, finetune-intent routing)
- `tests/preflight/test_structure.py` — docstring-facade drift guard added (guard #6)

## Decisions Made

See key-decisions in frontmatter. All plan premises were verified live against the current facade BEFORE the tests were pinned (the 23-04 lesson): predict listing 115 entries lab-first with `zhangtaolab/agront-1b-H3K27ac` at entry 0; finetune listing 49 entries all num_labels None and disjoint from predict; `generate_config` on the entry yields zero errors with the registry-prefilled task section; the num_labels-3 relabel revalidates with zero errors (TaskConfig's binary branch imposes no num_labels/label_names length constraint).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 23 is complete: all seven PFC requirements verifiable from committed tests (PFC-01 23-01; PFC-02/05/07 23-02; PFC-03/06 23-03 + the severed-network story here; PFC-04 23-04; integration here). This is the last declaring plan — the shared-ID gate releases PFC-01..PFC-07 for marking complete.
- The facade is the frozen shell contract for Phase 24 (loader rewire), Phase 25 (TUI panels import from `dnallm.preflight` only), and Phase 26 (MCP finding conventions).
- Carried flag for Phase 26 review (from 23-03): D-11's source-first hub order is implemented via the `is_lab_model()` proxy — `RegistryEntry` has no `source` field, so ModelScope is probed first for every entry and `source_hub` reports modelscope when both hubs carry the repo.
- Also carried for later docs (from 23-04): the validated, runtime-consumed registry length channel is `inference.max_length`; `finetune.max_length` is a packaged-example key that `TrainingConfig` ignores.

## Self-Check: PASSED

- tests/preflight/test_integration.py, dnallm/preflight/__init__.py, tests/preflight/test_structure.py, 23-05-SUMMARY.md — FOUND
- Commits 5c4ebeb, 5b4bb4a, 5e92885 — all ancestors of HEAD (evaluation-scope: 3 commits on this branch)

---
*Phase: 23-pre-flight-core-pure-validation-suitability-config-api*
*Completed: 2026-10-20*
