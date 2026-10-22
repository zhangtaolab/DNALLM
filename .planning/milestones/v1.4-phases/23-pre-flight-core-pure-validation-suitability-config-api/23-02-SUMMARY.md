---
phase: 23-pre-flight-core-pure-validation-suitability-config-api
plan: "02"
subsystem: testing
tags: [pydantic, registry, preflight, compatibility, verdicts, listings, tdd]

# Dependency graph
requires:
  - phase: 23-pre-flight-core-pure-validation-suitability-config-api (plan 01)
    provides: validate_data + DataReport census (length_counts, charset_census), frozen report inventory (DimensionVerdict/ModelVerdict/ListingEntry/AvailabilityReport), check_compatibility length-only tracer
provides:
  - Full three-dimension suitability engine (task x length x charset) with worst-of overall
  - Tokenizer budget classifier (kmer sliding-window inversion, BPE conservative ratio, char 1:1, unknown) neutralizing the k-mer trap
  - Charset stance table (strict char-level classes / BPE never-fail / unknown never-fail) with per-model isolation
  - Intent-routed listings (finetune=presfiltered bases, predict=head-shipping finetuned) lab-first sorted
  - Injectable prober shortlist seam (default cap 20, labs probed first, raising prober -> unknown PROBE_ERROR)
affects: [23-03, 23-04, 23-05, 25 TUI preflight panels, 26 MCP finding conventions]

# Actuals (#2632) — same scale as the plan estimate (chars/4 over the realized diff)
actuals:
  tokens: 15523
  tasks: 3
  commits: 5

tech-stack:
  added: []  # zero new dependencies (zero-install phase honored)
  patterns:
    - "Name-first budget classification: the k-mer size comes from the entry-name regex, never from the tokenizer class"
    - "Worst-of dimension aggregation: fail > warn > unknown > ok, computed only over the dimensions that exist"
    - "Injectable prober shortlist: network capability enters as a callable parameter, bounded at min(N, len(listing))"

key-files:
  created:
    - tests/preflight/test_compatibility.py
  modified:
    - dnallm/preflight/compatibility.py
    - tests/preflight/test_validation.py

key-decisions:
  - "ListingEntry carries no per-dimension verdicts: the frozen 23-01 type has no dimensions field and report.py is locked by the interface contract — the dimension functions are applied via the finetune listing's warn-or-ok task filter instead of forking the type (recorded deviation, contract-sanctioned)"
  - "Unknown tokenizer kind never fails charset: a tokenizer gap cannot prove hard incompatibility (D-05), so rejected chars warn CHARSET_NONSTANDARD_CHARS on unknown stances exactly like BPE classes"
  - "Default intent is finetune when only task_type is given: the plan routes listings on (intent or task_type), and the fine-tune journey is the flagship flow; predict without a task degrades to an empty listing, never an error"
  - "Tracer budget pins superseded as planned: 23-01's 1:1 512-char promoter budget became 1536 (BPE x3) and the end-to-end fixture's N now yields a charset warn — three test_validation.py assertions re-pinned to the new semantics"

patterns-established:
  - "Budget facts tuple (kind, k, budget_chars) resolved once per entry; _budget_kind/_char_budget are the plan-named thin readers over it"
  - "Probing only ever touches the first min(cap, len(listing)) entries of a lab-first sorted listing — labs get probed first (owner directive)"

requirements-completed: [PFC-02, PFC-05, PFC-07]

coverage:
  - id: D1
    description: "Three-dimension verdict engine (task/length/charset) with pinned boundary semantics: kmer 517 / BPE 1536 / char 512 budgets, truncation warn-never-fail at any fraction, strict charset fail isolated per model, U as T-equivalent warn"
    requirement: PFC-02
    verification:
      - kind: unit
        ref: "tests/preflight/test_compatibility.py#budget classifier pins, boundary 0.01 fraction, 100%-truncation warn, strict/BPE isolation, degradation NO_DATA/EMPTY_DATASET_DIM"
        status: pass
      - kind: other
        ref: "command: plan Task 1 verify (validate_data(['ACGT'*200]) -> promoter dims length ok, overall ok/warn) prints 'dims ok'"
        status: pass
    human_judgment: false
  - id: D2
    description: "Intent-routed listings over the base<->finetuned axis with lab-first ordering and the injectable prober shortlist seam (network-free default)"
    requirement: PFC-05
    verification:
      - kind: unit
        ref: "tests/preflight/test_compatibility.py#finetune/predict routing, sort_key_lab_first ordering equality, Mock prober call-count pins (min(N,len), default 20, zero without injection)"
        status: pass
      - kind: other
        ref: "command: plan Task 2 verify (Mock prober + intent predict) prints 'listing ok'"
        status: pass
    human_judgment: false
  - id: D3
    description: "Degradation, isolation, and full-registry proofs plus the module coverage gate: tokenizer-gap census, registry-miss isolation, empty no-arg report, worst-mitigable-case warn-only, 236-entry smoke, compatibility.py at 100%"
    requirement: PFC-07
    verification:
      - kind: unit
        ref: "tests/preflight/test_compatibility.py#tokenizer-gap census, miss isolation, worst mitigable case, full-registry smoke, raising prober"
        status: pass
      - kind: other
        ref: "command: coverage run pytest (test_compatibility + test_validation) -> compatibility.py 100% >= 96 gate"
        status: pass
    human_judgment: false

# Metrics
duration: 11 min
completed: 2026-10-20
status: complete
plan_head_before: f2e3159cefa6d59e34f1310a0c6faaf4b1aa37b2
plan_head_after: 841e8dfbcd50dc742b74f2a2500139f2b74beb53
commits: 5
---

# Phase 23 Plan 02: Suitability Engine Summary

**Three-dimension verdict engine (task x length x charset) with name-first k-mer budget classification (6mer->517, BPE->1536), strict/BPE charset stances, intent-routed lab-first listings, and an injectable prober shortlist — 100% module coverage**

## Performance

- **Duration:** 11 min
- **Started:** 2026-10-20T16:22:26Z
- **Completed:** 2026-10-20T16:33:08Z
- **Tasks:** 3/3
- **Files modified:** 3 (1,122 insertions, 80 deletions)

## TDD Gate Compliance

Tasks 1 and 2 carried `tdd="true"`; both follow the test -> feat commit-scope contract (executor log validation):

| Task | RED | GREEN | REFACTOR |
|------|-----|-------|----------|
| 1 (three dimensions) | b8a603d (23 failing on missing task/charset dims + wrong budgets; 1 invariant already green) | 13b3d0c | not needed (ruff-formatted at GREEN) |
| 2 (listings + prober) | 99567a8 (12 failing: missing intent/prober kwargs, empty listings, no ValueError) | 8262036 | not needed |

RED evidence note: this is a `type: execute` plan (not `type: tdd`), so the formal `tdd-red-evidence` classifier gate does not apply; every RED run failed on planned assertions (no collection errors, no fixture faults) and each GREEN run turned its task's failures fully green before committing.

## Accomplishments

- Three verdict dimensions per model with worst-of overall (fail > warn > unknown > ok): task (D-08: finetuned hard-match fail, base ok for supervised, mask/generation warn HEAD_SUPPORT_UNCERTAIN), length (D-06: exact truncated counts against the classified char budget, warn-never-fail), charset (D-07: strict fail on rejected chars isolated per model, BPE/unknown warn-only, ambiguity + U-equivalent warns with counts)
- The k-mer trap neutralized: budget classification reads k from the entry-name regex FIRST (all 54 kmer-named entries share one char-level tokenizer class that carries no k), pinned live at 6mer->517 (512+k-1) and BPE->1536 (512x3 conservative); tokenizer gaps (17 entries) degrade the length dimension to unknown TOKENIZER_UNKNOWN
- Intent-routed listings (PFC-05): finetune lists pretrained bases (warn-or-ok task filtered when task_type given), predict lists finetuned entries shipping heads with num_labels/label_names; every listing sorted by `registry.sort_key_lab_first` (PFC-07: zhangtaolab first — mixed listings pinned)
- Prober shortlist seam (D-10): network-free by default (availability None), first min(N or 20, len(listing)) entries probed in lab-first order, per-probe try/except degrading to unknown PROBE_ERROR (PFC-06)
- Quality: 44 behavior tests in the new suite (109 preflight lane total) + 41 registry tests green; ruff clean and formatted; compatibility.py coverage 100% (gate ≥96%); zero new dependencies; zero network in the lane

## Task Commits

1. **Task 1 (three dimensions, tdd)** — RED `b8a603d` + GREEN `13b3d0c`
2. **Task 2 (listings + prober, tdd)** — RED `99567a8` + GREEN `8262036`
3. **Task 3 (degradation/isolation/smoke + coverage gate)** — `841e8df`

**Plan metadata:** (recorded in the completion report)

## Files Created/Modified

- `dnallm/preflight/compatibility.py` — full engine: `_budget_facts`/`_budget_kind`/`_char_budget`/`_k_from_name`, `_charset_stance`, `_task_verdict`, `_length_verdict`, `_charset_verdict`, `_intent_listing`, `_build_listing`, `_attach_probes`; `check_compatibility` extended with `intent`/`prober`/`probe_shortlist` keyword-only parameters
- `tests/preflight/test_compatibility.py` — 44 behavior tests: classifier pins over the live registry, boundary/truncation semantics, task/charset verdicts, listings, prober seam, degradation/isolation/full-registry smoke
- `tests/preflight/test_validation.py` — three superseded tracer pins re-based (budget 512->1536, truncation fixture 600->1600, end-to-end overall ok->warn with charset detail)

## Decisions Made

See key-decisions in frontmatter. Additional rationale: the plan's supervised-typed-base ok-branch cannot be pinned over the live registry (every pretrained entry is mask or generation today), so that rule is pinned through `_task_verdict` directly on a synthetic `RegistryEntry` — the rule must hold as the registry grows, and the synthetic pin keeps it honest without touching registry data.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Superseded tracer pins in test_validation.py (file outside the plan's files_modified list)**
- **Found during:** Task 1 (GREEN — full-lane verification)
- **Issue:** The plan's verification requires the whole preflight lane green, but three 23-01 tracer tests pinned the superseded semantics: PROMOTER_BUDGET 512 (now BPE 1536), a truncation fixture of 600 chars (no longer over the 1536 budget), and an end-to-end overall-ok assertion (the fixture's N now yields a charset warn)
- **Fix:** re-pinned all three to the new budget/classifier semantics, preserving each test's original intent (exact-count truncation warn, end-to-end flow)
- **Files modified:** tests/preflight/test_validation.py
- **Verification:** 109-test preflight lane green
- **Committed in:** 13b3d0c (Task 1 GREEN)

**2. [Rule 1 - Bug] Self-inflicted GREEN defects caught by the RED tests**
- **Found during:** Task 1 (GREEN)
- **Issue:** (a) `_combined_length_histogram` ran before the data-None guard (AttributeError on `check_compatibility(None, ...)`); (b) a leftover draft loop in `check_compatibility` and a function-local `re` import
- **Fix:** guard reordered; draft loop removed; `re` moved to module level
- **Files modified:** dnallm/preflight/compatibility.py
- **Verification:** all 24 Task 1 tests green including the NO_DATA degradation test
- **Committed in:** 13b3d0c (Task 1 GREEN)

**3. [Contract instruction] ListingEntry carries no per-dimension verdicts**
- **Found during:** Task 2 (implementation)
- **Issue:** The Task 2 action says listing entries "get per-dimension verdicts when data is provided", but `ListingEntry` (frozen in 23-01's type inventory) has no dimensions field, and the interface contract forbids editing report.py — "if a field is genuinely missing, stop and record a deviation instead of forking the type"
- **Fix:** recorded as instructed; the dimension functions are still exercised on the listing path via the finetune intent's warn-or-ok task filter; no report.py edit, no forked type
- **Files modified:** none (contract-sanctioned interpretation)
- **Verification:** all Task 2 behavior tests pass — no test in the plan asserts dimensions on listing entries
- **Committed in:** n/a (scope note)

---

**Total deviations:** 2 auto-fixed (1 Rule 3 blocking, 1 Rule 1 bug) + 1 contract-sanctioned scope note
**Impact on plan:** The auto-fixes keep the plan's own lane-green verification true and were confined to test pins the plan explicitly superseded. No scope creep; report.py untouched.

## Issues Encountered

None - all three tasks ran clean; both plan `<verify>` commands printed their expected markers ('dims ok', 'listing ok').

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `check_compatibility` is complete for 23-03 (availability): the prober seam signature `Callable[[str], AvailabilityReport]` and `probe_shortlist` are stable; 23-03's adapters attach through `_attach_probes` unchanged
- 23-04 (config generation) consumes `effective_max_length` and the budget kinds the same way `_length_verdict` does; reason-code vocabulary is final
- 23-05 completes the facade: `check_compatibility` is already re-exported; the new keyword-only parameters require no facade change
- PFC-02/PFC-05/PFC-07 are shared with 23-05 — the requirements ready-ids gate will hold them Complete until 23-05's summary exists (expected)

## Self-Check: PASSED

- dnallm/preflight/compatibility.py, tests/preflight/test_compatibility.py — FOUND
- Commits b8a603d, 13b3d0c, 99567a8, 8262036, 841e8df — all ancestors of HEAD

---
*Phase: 23-pre-flight-core-pure-validation-suitability-config-api*
*Completed: 2026-10-20*
