---
phase: 27-transformers-4-5-model-compatibility-census
plan: 04
subsystem: docs
tags: [transformers, compatibility, docs, generated-page, drift-gate, workflow-dispatch, fail-closed, ci-gate]

# Dependency graph
requires:
  - phase: 27-03
    provides: "tf_compat_status / tf_compat_evidence / tf_compat_tested on all 49 pretrained registry rows, served through the typed dnallm.models.registry loader"
provides:
  - "scripts/census_compat.py --emit-docs [PATH] - deterministic plain-language compatibility-page generator reading the typed registry only (zero yaml parsing)"
  - "docs/user_guide/model_compatibility.md - generated 49-row page (GENERATED header, D-05 honest labels, both provenance footnotes verbatim incl. Chinese originals)"
  - "tests/models/test_compat_docs_consistency.py - offline standard-lane drift gate: 54 tests re-deriving all rows independently of the generator"
  - "scripts/census_compat.py --verify - fail-closed fresh-vs-committed diff (matched/UNEXPECTED trail, roster-gap detection, VERSION DRIFT report)"
  - ".github/workflows/census_compat.yml - dispatch-only re-verification lane on [self-hosted, dnallm-nightly] (no push/PR/schedule triggers)"
  - "mkdocs.yml Model Compatibility nav entry; dependabot KNOWN RISK pointer naming the lane re-run command"
affects: [docs site, standard CI lanes, self-hosted runner lanes]

# Actuals (#2632) - chars/4 over the realized diff; commits measured from the ledger.
actuals:
  tokens: 11569
  tasks: 3
  commits: 3
  plan_head_before: "d96743421b0f2bbbfcc5eb70a265a849a315107a"
  plan_head_after: "7ee66ec23c349b7816cd16d248cbdbdc293629dd"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Deterministic generated-docs pattern: a wall-clock regeneration stamp is incompatible with the byte-identical regeneration gate, so the page header carries a pinned data-date constant (DOCS_DATA_DATE) bumped alongside census re-runs - determinism by construction"
    - "Checker-independent drift gate over a generated page: the gate re-implements the status-cell derivation rules (same five pinned wordings, same evidence-cause parsing) without importing the generator, so generator drift cannot green itself (Phase 26 pattern)"
    - "Dispatch-only lane posture copied wholesale from feasibility.yml: workflow_dispatch + event-name guard + contents: read + loud GPU-absent marker + unmasked fail-closed run + if: always() evidence upload"

key-files:
  created:
    - docs/user_guide/model_compatibility.md
    - tests/models/test_compat_docs_consistency.py
    - .github/workflows/census_compat.yml
  modified:
    - scripts/census_compat.py
    - mkdocs.yml
    - .github/dependabot.yml

key-decisions:
  - "Regeneration date is a pinned data-date constant, not a wall-clock stamp: the plan demands both a regeneration date line AND byte-identical regeneration from an unchanged registry - a wall clock breaks the second on any later day, so DOCS_DATA_DATE (2026-10-27) is the honest deterministic resolution, bumped alongside census re-runs."
  - "The remote-code-broken cause rule prefers the explanatory clause for bare-symbol failing imports: a cell reading 'fails at olmo' says nothing to a biologist, so a single-symbol failing import takes the following evidence clause up to its first ' - ' boundary ('fails at remote configuration_olmo.py line 8 imports olmo.config.ModelConfig') - a fixed deterministic rule implemented identically by generator and gate."
  - "census-blocked reasons are mapped, never raw: 'unreachable on both hubs' and 'cannot read fetched config.json' map to plain-language repository reasons, so the owner's local cache paths embedded in the raw evidence never reach the public page."
  - "Footnote 1 interpolates the committed tf_compat_tested value (single distinct value enforced) rather than hardcoding '5.17.0': a future census re-run under a newer transformers re-stamps the page footnote automatically, same registry -> same bytes today."

requirements-completed: ["TBD"]

# Coverage metadata (#1602)
coverage:
  - id: D1
    description: "--emit-docs generator + the generated page + mkdocs nav: deterministic plain-language 49-row table derived from typed registry accessors only, five pinned status wordings, both provenance footnotes verbatim (incl. Chinese originals), no unqualified 4-and-5 claim"
    verification:
      - kind: unit
        ref: "python -I scripts/census_compat.py --emit-docs /tmp/p27-page.md && cmp -s /tmp/p27-page.md docs/user_guide/model_compatibility.md (byte-identical regeneration)"
        status: pass
      - kind: other
        ref: "49-cell pinned-wording count assertion over the committed page (exact plan verify command) + grep: mkdocs nav line 68 + zero yaml.load/safe_load in the script"
        status: pass
    human_judgment: false
  - id: D2
    description: "Offline CI drift gate: tests/models/test_compat_docs_consistency.py re-derives all 49 expected rows from the registry independently of the generator; fail-closed on missing page/header/table; pins header, footnotes, row count, lab-first order, per-row status/details, and the unqualified-claim prohibition"
    verification:
      - kind: unit
        ref: "python -m pytest tests/models/test_compat_docs_consistency.py -q (54 passed)"
        status: pass
      - kind: other
        ref: "deliberate-red demonstrated twice: status-cell drift -> exactly the drifted row failed; injected 'compatible with transformers 4 and 5' claim -> claim test + row failed; python -m pytest tests/models/ -q -> 682 passed"
        status: pass
    human_judgment: false
  - id: D3
    description: "Dispatch-only re-verification lane: --verify fail-closed diff (matched/UNEXPECTED trail, roster gaps, absent-artifact failure, VERSION DRIFT) + census_compat.yml posture (dispatch-only, contents: read, self-hosted runner, event guard, 240min, loud GPU check, unmasked run, always() upload) + dependabot pointer"
    verification:
      - kind: unit
        ref: "dry --verify on the committed artifact: all 49 matched, exit 0; offline probes: drift-injected artifact -> UNEXPECTED + exit 1, absent artifact -> ERROR + exit 1, simulated transformers 5.19.0 -> VERSION DRIFT + exit 1"
        status: pass
      - kind: other
        ref: "plan's workflow-structure assertion (dispatch-only triggers, permissions, runner, guard) + posture checks (timeout 240, no '|| true', always() upload, gpu_absent marker + exit 1, no default-hub URL) + ruff check/format clean on both code files"
        status: pass
    human_judgment: false

# Metrics
duration: 19min
completed: 2026-10-27
status: complete
---

# Phase 27 Plan 04: Docs and Enforcement Summary

**Generated plain-language compatibility page for biologists (49 honest rows, D-05 two-sided labels) plus two independent locks: an offline CI drift gate that re-derives every row from the registry (deliberate-red proven both ways) and a dispatch-only fail-closed re-verification lane whose --verify diff turns red on world drift - including a transformers version bump (VERSION DRIFT)**

## Performance

- **Duration:** 19 min
- **Started / Completed:** 2026-10-27T09:32:38Z / 2026-10-27T09:52:00Z
- **Tasks:** 3
- **Files:** 6 (3 created, 3 modified)

## Accomplishments

- **Task 1 - generator + page + nav (`156bc27`):** `--emit-docs [PATH]` on `scripts/census_compat.py` derives the entire page from `dnallm.models.registry` accessors (zero yaml parsing; fail-closed on missing verdicts, unknown statuses, mixed tested stamps, or table-breaking cell values). The committed `docs/user_guide/model_compatibility.md` carries the exact GENERATED header, a plain-language intro (transformers introduced, never assumed; no census/taxonomy vocabulary), a computed count sentence (19 working as-is / 21 code-loading verified / 1 via built-in fixes / 6 not working / 2 unchecked), the 49-row lab-first table (Model / What we found / Details; repo id in Details, family representative named for import-verified rows), and both provenance footnotes verbatim including the Chinese originals. `mkdocs.yml` gained the Model Compatibility nav entry under User Guide, after Preset Model Loading.
- **Task 2 - offline drift gate (`fe63c0c`):** `tests/models/test_compat_docs_consistency.py` (54 tests) re-implements the expected-row derivation independently (no import from the generator - Phase 26 checker-independence): GENERATED header pinned as line 1, table parsed fail-closed (missing header/separator/data row or wrong cell count = pytest.fail, never a skip), per-row status cells matched against the derived wordings, Details non-empty + representative named, row count == registry count, page order == lab-first order, both footnotes verbatim in order under the table, and the literal `compatible with transformers 4 and 5` prohibited in the table region. Deliberate-red demonstrated twice (status-cell drift -> 1 row failed; injected unqualified claim -> claim test + row failed), then reverted; full `tests/models/` lane 682 passed.
- **Task 3 - re-verification lane (`7ee66ec`):** `--verify` maps artifact verdicts through `compat_verdict` and diffs per entry against the committed `tf_compat_status` (typed loader), printing a `matched/UNEXPECTED` trail; roster gaps in either direction, absent/unparseable artifacts, unmappable records, or a transformers version mismatch (loud `VERSION DRIFT` - installed vs committed `tf_compat_tested`) all exit 1. A dry offline run on the committed artifact passes (49 matched, exit 0). `.github/workflows/census_compat.yml` copies the feasibility.yml posture wholesale: top-of-file WHY-dispatch-only comment, `on: workflow_dispatch` only, `contents: read`, `census-compat` on `[self-hosted, dnallm-nightly]` behind the event-name guard, 240-minute timeout, loud GPU-absent marker + exit 1, `uv venv` + `uv pip install -e ".[full]"`, an unmasked `--run --with-loads --verify` census step (no `|| true`), and an `if: always()` evidence upload. `.github/dependabot.yml`'s transformers KNOWN RISK comment now names the lane re-run command as the bump-merge duty.

## Task Commits

1. **Task 1: --emit-docs generator + generated compatibility page + mkdocs nav** - `156bc27` (feat)
2. **Task 2: offline CI drift gate over the committed compatibility page** - `fe63c0c` (test)
3. **Task 3: dispatch-only census re-verification lane (--verify fail-closed)** - `7ee66ec` (feat)

**Plan metadata:** committed after this SUMMARY (docs: complete plan).

## Files Created/Modified

- `scripts/census_compat.py` - `--emit-docs` generator section (deterministic page rendering, five pinned wordings, cause/reason derivation, pipe-guard) + `--verify` section (fail-closed world-vs-registry diff, VERSION DRIFT) + CLI wiring.
- `docs/user_guide/model_compatibility.md` - the generated public page (49 rows + footnotes).
- `tests/models/test_compat_docs_consistency.py` - the offline D-07 gate (54 tests).
- `.github/workflows/census_compat.yml` - the dispatch-only D-08 lane.
- `mkdocs.yml` - nav entry.
- `.github/dependabot.yml` - KNOWN RISK re-run pointer.

## Decisions Made

- **Regeneration date = pinned data-date constant** (see frontmatter key-decisions): reconciles the plan's two competing requirements (a regeneration date line vs byte-identical regeneration) deterministically.
- **Bare-symbol failing imports prefer the explanatory evidence clause** so no cell reads "fails at olmo"; **census-blocked reasons are mapped to plain-language repository reasons** so raw local cache paths never reach the public page.
- **Footnote 1 interpolates the committed tested stamp** (single distinct value enforced) instead of hardcoding "5.17.0" - future census re-runs re-stamp the page automatically.
- **`--verify` composes with `--run` and runs dry on the committed artifact** - the dry form is the offline unit-testable path (passes today because 27-03's consistency gate holds), and the workflow's composed form is the full world-drift oracle.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking detail] regeneration-date line vs byte-identical regeneration conflict**
- **Found during:** Task 1
- **Issue:** the plan requires a regeneration date line in the header AND byte-identical regeneration from an unchanged registry; a wall-clock date makes every later regeneration differ, permanently breaking the drift gate the same plan mandates.
- **Fix:** pinned `DOCS_DATA_DATE` constant (2026-10-27) in the script with the rationale comment; bumped alongside census re-runs. Byte-identity verified by cmp immediately and re-verified after every subsequent edit.
- **Files modified:** scripts/census_compat.py
- **Verification:** `--emit-docs /tmp/p27-page.md && cmp -s` identical across four runs spanning Tasks 1-3.
- **Committed in:** `156bc27`

**2. [Rule 1 - Bug] ruff RUF001 on the verbatim fullwidth Chinese parentheses**
- **Found during:** Tasks 1 and 2 (first commit attempts)
- **Issue:** the repo's preview ruff config flags the fullwidth `（）/` characters in the owner-approved D-05 label text as ambiguous-unicode; CI's repo-wide ruff gate would go red.
- **Fix:** sanctioned suppression comment on the pinned strings (`# ruff: ignore[ambiguous-unicode-character-string]`, rule NAME per this config's suppression policy, with a comment stating the characters are verbatim label text); Task 2's commit was amended before anything downstream depended on it.
- **Files modified:** scripts/census_compat.py, tests/models/test_compat_docs_consistency.py
- **Verification:** `ruff check` + `format --check` clean on both files.
- **Committed in:** `156bc27`, `fe63c0c`

---

**Total deviations:** 2 auto-fixed (1 Rule 3, 1 Rule 1)
**Impact on plan:** none on scope or honesty; both were determinism/lint-compliance resolutions the plan's own gates required.

## Issues Encountered

- A workflow comment initially mentioned the default hub hostname while explaining the no-hub-URL rule; reworded so the file carries no literal default-hub string (self-caught by the posture check).
- The first Task-2 commit briefly landed with the RUF001 violation because the guard chain short-circuited before ruff; fixed and amended within the same task (no downstream consumers; final hash `fe63c0c`).

## Authentication Gates

None - no authentication surfaces were touched.

## User Setup Required

None - no external service configuration required. The dispatch lane is owner-triggered post-phase (`gh workflow run census_compat.yml --ref v1.4`) - operational, not setup.

## Next Phase Readiness

Phase 27 complete, ready for verification. The census deliverable is public and enforced from both directions: the docs page cannot drift from the registry (offline gate on standard lanes, 54 tests) and the committed fields cannot drift from the world (dispatch-only lane: verdict disagreement, roster gap, or a transformers bump all fail red). Every model's transformers compatibility is now measured, fixed where fixable, recorded in the registry, published for biologists, and locked against drift.

## Self-Check: PASSED

- `docs/user_guide/model_compatibility.md` - present in HEAD (`156bc27`)
- `tests/models/test_compat_docs_consistency.py` - present in HEAD (`fe63c0c`)
- `.github/workflows/census_compat.yml` - present in HEAD (`7ee66ec`)
- Commits `156bc27`, `fe63c0c`, `7ee66ec` - all ancestors of HEAD; ledger count 3/3
