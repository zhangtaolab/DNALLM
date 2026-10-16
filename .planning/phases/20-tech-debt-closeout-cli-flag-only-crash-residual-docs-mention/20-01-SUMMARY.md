---
phase: 20-tech-debt-closeout-cli-flag-only-crash-residual-docs-mention
plan: 01
subsystem: cli
tags: [click, cli, usage-error, regression-test, docs-accuracy, extras-guard, models-lock, census]

# Dependency graph
requires:
  - phase: 19-docs-accuracy-sweep-milestone-closeout
    provides: deferred-items #1/#2 (the CLI KeyError defect record and the task_guides hf-route flag), the 19-03/19-04 lock-alignment standard, the docs surface that already stated --config-required
provides:
  - "--config-required click.UsageError exit (code 2) on all four train/inference CLI entry points, raised before the lazy core imports — the KeyError 'task' flag-only paths are deleted"
  - "tests/cli/test_config_required.py — the four-entry-point regression contract (bare + full-flags, core-never-constructed, no-escaped-traceback, byte-identical message)"
  - "faq legacy-extras line marked not-current with a page-relative GPU Support Matrix link, SANCTIONED_CUDA121_ROCM_RESIDUE re-pinned verbatim"
  - "task_guides.md seven runnable-example route sites flipped to source=\"modelscope\" per models.lock rows 21/29"
  - "usage.md --config-required note matching the shipped code"
affects: [v1.3 milestone closeout, v1.3-MILESTONE-AUDIT option-B closeout, future CLI wording changes (message substring contract)]

# Actuals (#2632) — pairs with the plan's estimate (54000 tokens / 3 tasks).
actuals:
  tokens: 6081     # chars/4 over changed (+/-) lines of the realized diff
  tasks: 3
  commits: 3       # MEASURED: git rev-list --count e08ed6d..HEAD
plan_head_before: e08ed6dbdb17e7b877ebcd45af9a2de920eadbd6
plan_head_after: 0655411bff314cad000f22d41266bd5e69b7b416

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Flag-only CLI paths raise click.UsageError with one static shared message BEFORE the lazy core imports (import-light error path, no argv interpolation — T-20-01/T-20-02 mitigations)"
    - "Message-substring contract: '--config is required' pinned by tests at all four sites; any future wording change must keep the substring or change the tests same-commit"
    - "extras-guard sanctioned-line edits land same-commit with their byte-identical docs line (deletion-creep coupling held via AST-verified byte comparison)"

key-files:
  created:
    - tests/cli/test_config_required.py
  modified:
    - dnallm/cli/train.py
    - dnallm/cli/cli.py
    - dnallm/cli/inference.py
    - tests/cli/test_cli.py
    - tests/test_extras_guard.py
    - docs/user_guide/cli/usage.md
    - docs/faq/models_troubleshooting.md
    - docs/user_guide/fine_tuning/task_guides.md

key-decisions:
  - "One generic static UsageError message reused byte-identically at all four sites (wording executor-discretion per D-01): names dnallm-model-config-generator and docs/user_guide/cli/usage.md, f-string-free"
  - "No-traceback pin adapted to click >=8.2 CliRunner semantics: result.exception is None OR isinstance(SystemExit) — a literal exception-is-None assertion is impossible for any nonzero exit on the installed/pinned click 8.5.0 (see Deviations)"
  - "Task 2 acceptance grep conflict resolved in favor of the plan's scope-fence prohibition: benchmark/mutagenesis guard text retained untouched (see Deviations)"
  - "No ci.yml edit: census measured UNCHANGED at 216/217 (1 deselected) after every test-edit step, exactly as the planning-time probe predicted"

patterns-established:
  - "tests/cli regression-test conventions extended: patch at origin package attributes, CliRunner + tmp_path exists=True files, per-entry-point bare/full-flags test pairs plus a message-identity consistency pin"

requirements-completed: []  # debt-closeout phase — plan frontmatter carries requirements: []

# Coverage metadata (#1602)
coverage:
  - id: D1
    description: "All four train/inference CLI entry points (dnallm-train/dnallm-inference console scripts + dnallm train/dnallm inference subcommands) exit 2 with the identical static '--config is required' message on any flag-only invocation; the KeyError paths are deleted"
    verification:
      - kind: unit
        ref: tests/cli/test_config_required.py (10 tests across 5 classes, all passing)
        status: pass
      - kind: other
        ref: "four-sites CliRunner probe: exit_code==2 + substring for (cli ['train'], cli ['inference'], inf.main [], tr.main []) -> ALL-FOUR-SITES-GREEN"
        status: pass
    human_judgment: false
  - id: D2
    description: "Regression coverage pinning core-never-constructed, no-escaped-traceback, and byte-identical message across the four entry points (D-02)"
    verification:
      - kind: unit
        ref: "tests/cli/test_config_required.py::TestConfigRequiredMessageConsistency::test_all_four_entry_points_emit_identical_message"
        status: pass
    human_judgment: false
  - id: D3
    description: "Census discipline held: stage-0.5 measurement unchanged at 216/217 (1 deselected) in the same changes as the test edits; expected_skips.yaml untouched; no ci.yml edit needed"
    verification:
      - kind: other
        ref: ".venv/bin/python -m pytest tests/examples --collect-only -q -m 'not giants' -> CENSUS-STABLE (run after Task 1 and Task 2 edits)"
        status: pass
      - kind: other
        ref: "git diff e08ed6d..HEAD -- tests/expected_skips.yaml -> empty"
        status: pass
    human_judgment: false
  - id: D4
    description: "faq legacy-extras line: v0.5-era [cuda]/[rocm] marked not-current with a page-relative GPU Support Matrix pointer; extras-guard sanctioned entry re-pinned byte-identical in the same commit; guard at 11 passed"
    verification:
      - kind: unit
        ref: "tests/test_extras_guard.py -> 11 passed"
        status: pass
      - kind: other
        ref: "AST byte-comparison of docs/faq/models_troubleshooting.md line 13 vs SANCTIONED_CUDA121_ROCM_RESIDUE entry -> True; pattern-matching lines in faq == [13]"
        status: pass
      - kind: other
        ref: "mkdocs build --strict -> exit 0, 0 WARNING lines (faq link resolves)"
        status: pass
    human_judgment: false
  - id: D5
    description: "task_guides.md route alignment: exactly seven runnable Example Implementation sites at lines 65/147/222/296/369/445/522 flipped to source=\"modelscope\" (six plant-dnabert-BPE per lock row 21, one plant-dnagpt-BPE per lock row 29); zero hf-route sites remain"
    verification:
      - kind: other
        ref: "grep -n 'source=\"modelscope\"' task_guides.md -> exactly 65/147/222/296/369/445/522; grep -c 'source=\"huggingface\"' -> 0"
        status: pass
      - kind: other
        ref: "python scripts/check_docs_sync.py && python scripts/check_notebook_md_sync.py -> both exit 0"
        status: pass
    human_judgment: false
  - id: D6
    description: "usage.md note now describes the shipped --config-required usage-error exit and the generator; the 'currently broken'/minimal-dict claims are gone (D-01 docs side)"
    verification:
      - kind: other
        ref: "grep -c 'currently broken' docs/user_guide/cli/usage.md -> 0; grep -c 'minimal dict' -> 0"
        status: pass
    human_judgment: false

# Metrics
duration: 10 min
completed: 2026-10-16
status: complete
---

# Phase 20 Plan 01: Tech-debt closeout — CLI flag-only crash + residual docs mentions Summary

**--config-required usage error (exit 2, one static message) replaces the KeyError 'task' crash on all four train/inference CLI entry points, with a 10-test regression contract, held census, and the three docs residues (usage note, faq legacy extras + guard re-pin, seven modelscope route flips) closed**

## Performance

- **Duration:** 10 min
- **Started:** 2026-10-16T00:01:18Z
- **Completed:** 2026-10-16T00:11:17Z
- **Tasks:** 3/3
- **Files modified:** 9 (1 created, 8 modified)

## Accomplishments

- **CLI fix (D-01/D-02, Tasks 1-2):** `dnallm/cli/train.py`, `dnallm/cli/cli.py` (train + inference commands), and `dnallm/cli/inference.py` now raise `click.UsageError` with a single static message ("--config is required: the task section and other core settings come only from a configuration file, so a flag-only invocation is not supported. Generate a config with `dnallm-model-config-generator --output config.yaml` and rerun with --config. See the CLI usage docs (docs/user_guide/cli/usage.md).") raised BEFORE the lazy core imports; the entire flag-only else branches (partial guards, inline dict construction, trainer invocations) are deleted. TDD: RED observed (2 failures Task 1, 7 failures Task 2) before each GREEN.
- **Regression coverage (D-02):** `tests/cli/test_config_required.py` — final test names: TestStandaloneTrainConfigRequired::{test_bare_invocation_is_a_usage_error, test_full_flags_still_requires_config, test_config_path_still_loads_and_trains}; TestGroupTrainConfigRequired::{test_bare_invocation_is_a_usage_error, test_full_flags_still_requires_config}; TestGroupInferenceConfigRequired::{test_bare_invocation_is_a_usage_error, test_full_flags_still_requires_config}; TestStandaloneInferenceConfigRequired::{test_bare_invocation_is_a_usage_error, test_full_flags_still_requires_config}; TestConfigRequiredMessageConsistency::test_all_four_entry_points_emit_identical_message.
- **test_cli.py rewrites:** ten superseded flag-only tests renamed to the exit-2 contract (two standalone-train in Task 1; TestTrainCommand pair, TestInferenceCommand trio, TestStandaloneInferenceModule trio in Task 2). Benchmark/mutagenesis/generator/mcp-server tests and every -c config-path test untouched.
- **Census held (D-02):** stage-0.5 measurement UNCHANGED at `216/217 tests collected (1 deselected)` after every test-edit step (probe prediction confirmed); ci.yml untouched; expected_skips.yaml untouched.
- **Docs residue (Task 3):** usage.md note rewritten to the shipped behavior; faq:13 single-line edit marks the v0.5-era `[cuda]`/`[rocm]` extras as legacy-not-current and links the GPU Support Matrix (`../getting_started/gpu_support.md`), with `SANCTIONED_CUDA121_ROCM_RESIDUE` re-pinned byte-identical in the same commit; task_guides.md seven sites flipped to `source="modelscope"` at unchanged line numbers 65/147/222/296/369/445/522.

## Task Commits

Each task was committed atomically:

1. **Task 1: Tracer — dnallm-train --config-required usage error** - `c3c65cd` (fix)
2. **Task 2: Expansion — remaining three sites + four-entry-point coverage** - `48f9905` (fix)
3. **Task 3: Docs residue — usage note, faq + guard re-pin, task_guides routes** - `0655411` (docs)

**Plan metadata:** this SUMMARY commit (docs).

## Files Created/Modified

- `tests/cli/test_config_required.py` (new) - four-entry-point --config-required regression contract (10 tests)
- `dnallm/cli/train.py` - standalone dnallm-train: UsageError before lazy imports; flag-only branch deleted; `import sys` removed
- `dnallm/cli/cli.py` - `train`/`inference` subcommands: same shape; flag-only branches deleted (`import sys` retained — benchmark/mutagenesis still use it)
- `dnallm/cli/inference.py` - standalone dnallm-inference: same shape; flag-only branch deleted; `import sys` removed
- `tests/cli/test_cli.py` - ten superseded flag-only tests rewritten to the exit-2 contract
- `tests/test_extras_guard.py` - faq SANCTIONED_CUDA121_ROCM_RESIDUE entry re-pinned verbatim (same commit as the faq edit)
- `docs/user_guide/cli/usage.md` - "currently broken" note replaced with the --config-required exit + generator guidance
- `docs/faq/models_troubleshooting.md` - line 13: legacy extras marked not-current + GPU Support Matrix page-relative link (still single-line, still line 13)
- `docs/user_guide/fine_tuning/task_guides.md` - seven runnable-example route values flipped to modelscope

## Decisions Made

- **Message wording (executor discretion per CONTEXT):** one generic static string reused byte-identically at all four sites; contains the pinned substring `--config is required`, names `dnallm-model-config-generator`, references `docs/user_guide/cli/usage.md`; f-string-free (T-20-01 mitigation); identity enforced by the consistency test rather than a shared constant (avoids `dnallm/cli/__init__.py` import-order coupling).
- **Census:** measured unchanged after Tasks 1 and 2 — no ci.yml re-pin (the sanctioned move was conditional on a measured deviation; none occurred).
- **Windows ledger:** no entries appended — no stubs, no skipped tests, no unrun verifies (all `<verify>` and `<acceptance_criteria>` checks ran green; see Deviations for the two documented adaptations).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Plan's no-traceback pin is unsatisfiable on the installed click**
- **Found during:** Task 1 (RED/GREEN)
- **Issue:** The plan's behavior Test 3 and its TRACER-GREEN `<verify>` one-liner assert `result.exception is None`. On click >= 8.2 (installed 8.5.0; pyproject pins `click<8.5.1`, so CI resolves the same), `CliRunner.invoke` records the standalone-mode `SystemExit` in `Result.exception` for EVERY nonzero exit (verified against click 8.5.0 `testing.py` source: `if e_code != 0: exception = e`). A handled UsageError exit therefore surfaces as `SystemExit(2)`, never None — the literal assertion can never pass in this environment.
- **Fix:** Adapted the pin in `tests/cli/test_config_required.py` (`_assert_no_traceback`) and the tracer one-liner to `result.exception is None or isinstance(result.exception, SystemExit)`. This still proves exactly what the plan's parenthetical states ("the old behavior surfaced a KeyError as the exception") — any escaped traceback (KeyError or otherwise) fails the pin — and it holds on BOTH old (None) and new (SystemExit) click semantics, so the suite is version-robust.
- **Files modified:** tests/cli/test_config_required.py
- **Verification:** TRACER-GREEN printed with the adapted assertion; all 71 tests/cli tests green.
- **Committed in:** c3c65cd (test helper), 48f9905 (extended tests)

**2. [Rule 3 - Blocking] Task 2 acceptance-criterion grep conflicts with the plan's own scope-fence prohibition**
- **Found during:** Task 2 (acceptance-criteria gate)
- **Issue:** AC3 requires `grep -c "are required when not using --config" dnallm/cli/cli.py ... returns 0 for all three`. In cli.py the pattern also matches the benchmark guard (L138: "--model and --data are required...") and the mutagenesis guard (L182: "--model and --sequence are required...") — commands the plan's must-have prohibition ("The benchmark and mutagenesis flag-only paths and their tests are untouched") and Task 2 action text ("Leave TestBenchmarkCommand, TestMutagenesisSubcommand... untouched") explicitly forbid changing. The criterion is unreachable without violating the stronger contract clause; the planner overlooked that the shared sentence suffix appears in the out-of-scope commands.
- **Fix:** Resolved in favor of the prohibition (zero fix attempts made deliberately — each would be a scope-fence violation). The criterion's intent — "superseded guard text gone" — is satisfied: all train/inference guard occurrences are deleted (0 in train.py/inference.py; 0 train/inference variants in cli.py); only the two benchmark/mutagenesis guards retain the suffix, matching their untouched tests (all remaining exit-1 assertions in test_cli.py map to TestBenchmarkCommand/TestMutagenesisSubcommand/TestStandaloneMutagenesModule/TestModelConfigGeneratorCommand/TestMcpServerCommand).
- **Files modified:** none beyond plan scope (this is a documented non-change)
- **Verification:** `grep -n 'are required when not using --config' dnallm/cli/cli.py` -> exactly L138 (benchmark) + L182 (mutagenesis); class-mapping of all `exit_code == 1` assertions -> boundary classes only.
- **Committed in:** n/a (no additional change; Task 2 commit 48f9905)

---

**Total deviations:** 2 auto-fixed (1 bug-class check adaptation, 1 acceptance-criteria/scope-fence conflict resolved per the stronger clause)
**Impact on plan:** Both deviations preserve the plan's intent exactly; no scope creep, no behavior beyond the three debt items.

## Issues Encountered

None beyond the two documented deviations. RED->GREEN order honored on both TDD tasks with failing runs captured before each fix (Task 1: 2 failed/1 passed; Task 2: 7 failed/3 passed).

## Authentication Gates

None — no authenticated surfaces touched.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All three v1.3 audit debt items (option B) are closed; code matches docs (usage.md), docs match lock (task_guides/models.lock), faq presents no retired extras as current, and the census triple holds at 216/217 (1 deselected).
- Phase 20 has a single plan — this SUMMARY completes the phase's in-repo work; milestone closeout (option-B ledger, owner human-gates #25/#26/#27, the temp `- v1.3` push-filter removal) remains with the owner as recorded in the plan's scope fence.
- Standing note for any future CLI wording change: keep the `--config is required` substring or update tests/cli/test_config_required.py in the same change.

## Self-Check: PASSED

- tests/cli/test_config_required.py exists on disk: FOUND
- Commits c3c65cd, 48f9905, 0655411 are ancestors of HEAD: FOUND (all three verified)
- Phase-level verification block: 82 passed (71 tests/cli + 11 extras guard), CENSUS-STABLE, DOCS-SYNC-GREEN, MKDOCS-STRICT-CLEAN (exit 0 / 0 warnings), ruff green, expected_skips.yaml delta empty: FOUND

---
*Phase: 20-tech-debt-closeout-cli-flag-only-crash-residual-docs-mention*
*Completed: 2026-10-16*
