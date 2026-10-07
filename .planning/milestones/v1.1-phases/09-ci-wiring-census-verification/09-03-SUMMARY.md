---
phase: 09-ci-wiring-census-verification
plan: 03
subsystem: testing
tags: [models-lock, contract-test, ci-gate, mkdocs, coverage-expectation, AUDIT-04]

requires:
  - phase: 08-full-execution-rollout-repair-loop
    provides: models.lock pinned-row format (D-15 prefix alignment), the fast contract-test idioms (_code_cells/_active_lines), the audit_skips fail-closed parse discipline
provides:
  - CI-08 fast-leg guard — tests/test_models_lock_contracts.py: fail-closed models.lock parser + example-content scanner + membership guard, with the drift failure mode proven by injection
  - Best-effort route-alignment guard (single-id/single-route notebooks: ACTIVE source= route must equal the lock prefix; ms <=> modelscope, hf <=> huggingface)
  - CI-09/D-15 public docs page — docs/user_guide/continuous_integration.md carrying the AUDIT-04 design note (kernel subprocesses unmeasured; example lane does not move the 96.30% gate), registered in the mkdocs User Guide nav
affects: [09-ci-wiring-census-verification (09-04 green-dispatch gate), docs site, future census-growth PRs (lock additions are now guarded)]

actuals:
  tokens: 7926
  tasks: 2
  commits: 3
  plan_head_before: 45de22b162dda1a9f0efae54309967d4fd58c8ba
  plan_head_after: 1a8e6f8497fd46d67fc712c044a16e37d200c10b

tech-stack:
  added: []  # stdlib only: json, re, pathlib, collections.abc (+pytest) — no installs (T-09-SC)
  patterns:
    - "Fail-closed lock parsing (audit_skips discipline): any non-comment row that is not a well-formed hf/ms model row or the dataset: row raises ValueError naming the line"
    - "Drift-injection proof as a permanent test: synthetic mini-lock + mini-notebook fixtures assert the checker REPORTS known-present drift (exact-set equality kills over-reporting and vacuous passes, T-09-05)"
    - "Reasoned exclusion record: _NON_MODEL_ALLOWLIST entries are (token, one-line reason) pairs, and a test enforces both the reason and liveness (stale entries fail, T-09-08)"
    - "Dual-position candidate extraction: quoted org/name literals for code text, plus unquoted YAML scalar-value positions for configs"

key-files:
  created:
    - tests/test_models_lock_contracts.py
    - docs/user_guide/continuous_integration.md
  modified:
    - mkdocs.yml

key-decisions:
  - "No models.lock additions: the first full scan found zero fetched-but-unlocked models — all non-lock candidates were non-model strings (MIME dict keys, data-file paths, one local LoRA adapter directory), so nothing entered the lock and the allowlist carries exactly one documented entry"
  - "YAML configs need value-position scanning, not just quoted tokens: benchmark_config.yaml carries UNQUOTED `path:` model ids — a quoted-only extractor (the plan's literal sketch) would have missed them and violated the CI-08 must-have; the extractor applies both patterns to .yaml files"
  - "Route-alignment covered set pinned at 15 live files with a >=10 non-vacuity floor; ambiguous multi-id/multi-route notebooks, the raw-AutoModel embedding_attention, marimo runtime-dropdown apps, and YAML configs are the documented skipped set"
  - "Docs links to out-of-docs targets (workflows README, tests/TESTING.md, pyproject) use GitHub blob URLs — the repo convention; literal relative paths to .github/ would 404 on the rendered Pages site"
  - "RED/GREEN via the 06-02 NotImplementedError-stub precedent: the full test file committed first (12 failed / exit 1 / zero collection errors), implementation second (12 passed)"

patterns-established:
  - "Lock-membership guard pattern: parse-triage-report over in-repo trusted content, parameterized by (lock_path, example_root) so synthetic drift fixtures reuse the production checker"
  - "Coverage-expectation docs pattern: public page quotes the pyproject ratchet verbatim and states the measurement boundary by design-note name (AUDIT-04) instead of restating numbers loosely"

requirements-completed: [CI-08, CI-09]

coverage:
  - id: D1
    description: "CI-08 models.lock consistency guard on the fast leg — fails on drift (proven by injection), live tree green with documented exclusions"
    requirement: "CI-08"
    verification:
      - kind: unit
        ref: "tests/test_models_lock_contracts.py — 12 tests, `.venv/bin/python -m pytest tests/test_models_lock_contracts.py -q` → 12 passed in 0.43s"
        status: pass
      - kind: unit
        ref: "RED evidence: commit 0aa67b2 → 12 failed / exit 1 / zero collection errors; GREEN: commit 3870efb → 12 passed"
        status: pass
      - kind: integration
        ref: "plan verify block: collect-only grep OK; example/ unmodified; models.lock unmodified (GUARD-OK); no markers/skips in the file; full-tree collect 1939 tests"
        status: pass
    human_judgment: false
  - id: D2
    description: "CI-09/D-15 coverage-expectation docs page with the AUDIT-04 kernel-subprocess note, reachable from the mkdocs nav"
    requirement: "CI-09"
    verification:
      - kind: integration
        ref: "plan verify block: page exists; greps for 96.30 / AUDIT-04 / kernel subprocesses / fail_under pass; nav registered in mkdocs.yml; `python3 scripts/check_docs_sync.py` → OK; `python3 scripts/check_notebook_md_sync.py` → 24/24 in sync"
        status: pass
    human_judgment: false

duration: 15min
completed: 2026-10-05
status: complete
---

# Phase 9 Plan 03: CI-08 Lock Guard + CI-09 Coverage-Expectation Docs Summary

**A fast-leg contract test now fails on any example-referenced remote model id missing from models.lock (proven by drift injection), and the AUDIT-04 coverage expectation — example execution runs in kernel subprocesses and by design does not move the 96.30% gate — is published on the docs site and reachable from the nav.**

## Performance

- **Duration:** 15 min
- **Started:** 2026-10-05T14:53:42Z
- **Completed:** 2026-10-05T15:08:32Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- CI-08: `tests/test_models_lock_contracts.py` — fail-closed lock parser, example-tree scanner (notebook code cells, marimo apps, YAML configs incl. unquoted value positions, the NER helper script), membership guard with a reasoned one-entry allowlist, and the route-alignment extension covering 15 live single-id/single-route files. Unmarked, kernel-free, network-free, zero skips — collected on every fast leg under `-m "not slow"`.
- CI-09/D-15: `docs/user_guide/continuous_integration.md` — nightly topology summary (03:00 coverage-nightly + test-mamba, 05:30 staged example-nightly, cron-string gates, workflow_dispatch), the AUDIT-04 kernel-subprocess coverage note with the fail_under=90 ratchet quoted verbatim, the giants policy pointer, and the typed-skip audit paragraph; one nav line in the mkdocs User Guide block.

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): CI-08 guard skeleton** - `0aa67b2` (test) — full behavior classes against NotImplementedError stubs; 12 failed / exit 1 / zero collection errors (test-level RED, #3770 discipline)
2. **Task 1 (GREEN): CI-08 guard implementation** - `3870efb` (test) — parser/scanner/guards implemented; 12 passed
3. **Task 2: CI-09/D-15 docs page + nav** - `1a8e6f8` (docs)

**Plan metadata:** committed after this SUMMARY (docs: complete plan)

## Files Created/Modified
- `tests/test_models_lock_contracts.py` — the CI-08 guard: `_parse_lock_rows` (fail-closed), `_example_model_id_candidates` (rglob scan + dual-position extraction + URL/.git/extension/MIME filters), `_find_unlocked_ids`, `_route_alignment_violations`, `_NON_MODEL_ALLOWLIST` (reasoned, liveness-checked), 12 tests across 5 behavior classes
- `docs/user_guide/continuous_integration.md` — the CI/testing page (topology summary, AUDIT-04 note, ratchet, giants pointer, skip-audit pointer)
- `mkdocs.yml` — one nav entry: `Continuous Integration: user_guide/continuous_integration.md` (User Guide block, after Troubleshooting)

## Triage Record (Task 1 first scan — CI-08 acceptance)

The first full scan of `example/` produced 21 distinct candidates; every non-lock candidate was honestly triaged:
- **19 lock-covered ids** (18 model rows + the dataset row) — membership green.
- **`image/png`, `text/plain`** — MIME display-dict keys in the showcase notebooks → generic MIME-top-level filter (not allowlist).
- **`data/TAIR10_*.gff|.gff3|.gtf`, `data/chr1_*.fas`** — local data-file references → extension filter (extended with the observed bio-data extensions).
- **`plantcad/cross_species_acr_train_on_arabidopsis_plantcad2_small`** — local LoRA adapter directory name (`lora_finetune` saves under `./outputs`; `lora_inference` passes it as `lora_adapter=`) → the single `_NON_MODEL_ALLOWLIST` entry with its reason.
- **models.lock: unchanged** — no genuinely fetched-but-unlocked model was found, so no lock rows were added (lock additions are deliberate review acts; the guard itself never writes the lock).

## RED/GREEN Evidence (acceptance criterion 1)

- **RED** (`0aa67b2`): `.venv/bin/python -m pytest tests/test_models_lock_contracts.py -q` → `12 failed in 0.42s`, `PYTEST_EXIT=1`, `12 FAILED / 0 ERROR` — test-level failures via the NotImplementedError stubs, no collection errors (06-02 precedent).
- **GREEN** (`3870efb`): same command → `12 passed in 0.43s`, exit 0 — including `TestDriftInjection::test_unlocked_synthetic_model_id_is_reported_exactly` (asserts the reported set EQUALS the injected id) and `TestRouteAlignment::test_synthetic_route_prefix_mismatch_is_reported` (synthetic hf-prefix + modelscope-route mismatch is reported, with an aligned positive control).

## Decisions Made
- See key-decisions in frontmatter (triage outcome, YAML value-position scanning, covered-set pin + non-vacuity floor, blob-URL links, RED/GREEN staging).

## Deviations from Plan

**1. [Rule 3 - Blocking] YAML unquoted-value scanning added to the extractor**
- **Found during:** Task 1 implementation
- **Issue:** The plan's action sketches a "quoted-token regex"; the example YAML configs (benchmark_config.yaml) reference model ids as UNQUOTED scalars (`path: zhangtaolab/plant-dnabert-BPE-promoter`), so a quoted-only extractor would miss them and fail the CI-08 must-have ("every remote model-id literal referenced by ... example YAML configs").
- **Fix:** `.yaml` files additionally match a value-position regex (after `key:` or `- `, with per-line comment stripping); verified the three benchmark ids are captured (dedicated test: `test_yaml_unquoted_value_positions_are_scanned`).
- **Files modified:** tests/test_models_lock_contracts.py
- **Verification:** 12/12 green; live scan finds the 3 benchmark_config.yaml ids.
- **Commit:** 3870efb

**2. [Rule 3 - Blocking] Docs links use GitHub blob URLs instead of literal relative paths**
- **Found during:** Task 2 authoring
- **Issue:** The plan says "relative link" to the workflows README / TESTING.md, but both live outside `docs/` — a literal relative path 404s on the rendered mkdocs Pages site (mkdocs serves only docs/).
- **Fix:** Blob URLs (`https://github.com/zhangtaolab/DNALLM/blob/main/...`), the established repo convention (docs/example/*.md precedent); the in-repo path is also spelled out in the link text.
- **Files modified:** docs/user_guide/continuous_integration.md
- **Verification:** page greps + both sync gates green.
- **Commit:** 1a8e6f8

**3. [Rule 2 - Missing critical, minor] Extractor filters extended beyond the plan's illustrative lists**
- **Found during:** Task 1 first scan
- **Issue:** The plan's extension list (`.yaml/.csv/.py/...`) and filter set did not cover the observed bio-data extensions (`.gff/.gff3/.gtf/.fas`) or MIME-type dict keys (`image/png`, `text/plain`) — without generic filters these would have landed in the allowlist as string-specific entries.
- **Fix:** Extension tuple extended with the observed data-file extensions; a MIME-top-level filter added. Generic filtering keeps the allowlist reserved for genuinely id-shaped non-model strings (one entry).
- **Files modified:** tests/test_models_lock_contracts.py
- **Verification:** `TestScanFilters` proves URL/.git/path/extension/MIME tokens produce no candidates.
- **Commit:** 3870efb

**Total deviations:** 3 auto-fixed (2x Rule 3 blocking-correctness, 1x Rule 2 minor)
**Impact on plan:** None on scope or must-haves — each deviation was required to satisfy a stated must-have (YAML coverage, working public links) or to keep the allowlist discipline honest; all verifies green.

## Issues Encountered
None — no auth gates, no package installs (stdlib only), no hook retries beyond one ruff composite-assertion fix folded into the GREEN commit.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- This plan was wave 1 (with 09-01, already merged); 09-02 and 09-04 remain. The new guard is additive to the fast leg (12 tests at tests/ root — outside the tests/examples census triple, so no Stage 0.5 bump needed: full-tree collect now 1939).
- 09-04's green-dispatch gate re-verifies the documented nightly topology end to end; the docs page's topology claims match the ci.yml state landed by 09-01 (cron-string gates, census assertion, deleted evo/cache steps).
- No blockers.

## Self-Check: PASSED

All created files exist (tests/test_models_lock_contracts.py, docs/user_guide/continuous_integration.md, 09-03-SUMMARY.md); all three task commits (0aa67b2, 3870efb, 1a8e6f8) are ancestors of HEAD; mkdocs.yml carries exactly one Continuous Integration nav entry.

---
*Phase: 09-ci-wiring-census-verification*
*Completed: 2026-10-05*
