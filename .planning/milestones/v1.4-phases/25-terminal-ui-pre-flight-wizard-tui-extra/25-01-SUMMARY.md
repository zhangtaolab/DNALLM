---
phase: 25-terminal-ui-pre-flight-wizard-tui-extra
plan: 01
subsystem: packaging
tags: [extras, pyproject, textual, platformdirs, pkg-01, install-docs, c-i-legs]

# Dependency graph
requires:
  - phase: 24-preset-loading-load-preset-model-registry-rewire
    provides: the v1.3-close commitment that the one-cycle base alias is removed at v1.4 (PKG-01)
provides:
  - "tui extra (textual>=8.2,<9, platformdirs>=3.6,<5) installable in the dev venv and on every pytest CI leg"
  - "test = [dnallm[tui], ...] so every .[full...] leg resolves textual transitively"
  - "all includes tui; full remains byte-identical (dnallm[dev,test,notebook,mcp] + isort)"
  - "extras table carries no base key; the residue gate sanctions exactly one post-removal honesty row"
  - "TestTuiExtraMembers + test_base_alias_is_removed as the executable TUI-01/PKG-01 contract"
affects: [25-02-structure-tests-and-seam, 25-10-phase-end-verification]

# Actuals (#2632) — pairs with the plan's `estimate` (36000 tokens / 2 tasks / low confidence).
actuals:
  tokens: 5739   # chars/4 over the realized diff (22956 diff chars, .planning excluded)
  tasks: 2
  commits: 2
  plan_head_before: 255c201011fc77afe8fba06ff27b5c9fa7f320f9
  plan_head_after: 12d41026f3cb712895ed1634e0db25d3ad0ae5fe

# Tech tracking
tech-stack:
  added:
    - "textual>=8.2,<9 (resolved 8.2.8 in the dev venv) — terminal wizard layer only"
    - "platformdirs>=3.6,<5 (4.12.1 already present; band mirrors textual's own constraint)"
  patterns:
    - "exact-string extras guards asserting membership in both directions, re-parsed from pyproject.toml with tomllib on every run"
    - "bidirectional doc-residue gate: an extra bracket-spelling line fails, a deleted sanctioned line fails"

key-files:
  created: []
  modified:
    - pyproject.toml
    - tests/test_extras_guard.py
    - README.md
    - docs/getting_started/installation.md
    - docs/getting_started/install_honesty.md

key-decisions:
  - "tui is declared after mcp and before full, and is NOT added directly to full — full stays byte-identical; textual reaches every CI leg through full -> dev -> test -> dnallm[tui]"
  - "dnallm[tui] is the FIRST member of the test extra, matching dev's dnallm[test,notebook] convention"
  - "PKG-01's removal target is the base alias (base = [dnallm[full]]), not full; the REQUIREMENTS 'full alias' wording is a mislabel confirmed before dispatch"
  - "The residue gate shrinks to exactly one sanctioned pair: the post-removal dnallm[base] row of the cross-installer honesty matrix"
  - "Both dependency-group tables in README.md and installation.md gained a tui row and the all row gained + tui (Rule 2 doc-accuracy fix)"

patterns-established:
  - "New extras are guarded by exact-member set equality plus an explicit transitive-ride assertion (test rides tui, all includes tui)"
  - "Alias retirement is proven by absence, not by shape; deletion creep is caught by the residue gate's missing-line arm"

requirements-completed: [TUI-01, PKG-01]

coverage:
  - id: D1
    description: "tui extra declared with the two exact owner-pinned ranges, dnallm[tui] first in test, tui added to all, full byte-identical"
    requirement: TUI-01
    verification:
      - kind: unit
        ref: "tests/test_extras_guard.py#TestTuiExtraMembers::test_tui_extra_members_exact"
        status: pass
      - kind: unit
        ref: "tests/test_extras_guard.py#TestTuiExtraMembers::test_test_extra_rides_tui"
        status: pass
      - kind: unit
        ref: "tests/test_extras_guard.py#TestTuiExtraMembers::test_all_extra_includes_tui"
        status: pass
      - kind: integration
        ref: ".venv/bin/uv pip install -e '.[tui]' + importlib.metadata probe (textual 8.2.8, platformdirs 4.12.1)"
        status: pass
    human_judgment: false
  - id: D2
    description: "deprecated base extras alias removed; full preserved; residue gate reduced to the single post-removal honesty row and kept bidirectional"
    requirement: PKG-01
    verification:
      - kind: unit
        ref: "tests/test_extras_guard.py#TestFullExtraAndBaseAlias::test_base_alias_is_removed"
        status: pass
      - kind: unit
        ref: "tests/test_extras_guard.py#TestFullExtraAndBaseAlias::test_full_extra_members_preserved"
        status: pass
      - kind: unit
        ref: "tests/test_extras_guard.py#TestExtrasSurgeryResidueGates::test_base_bracket_residue_equals_sanctioned_list"
        status: pass
      - kind: other
        ref: "tomllib assertion: 'base' not in extras and full == ['dnallm[dev,test,notebook,mcp]', 'isort>=6.0.1'] -> PKG01_OK"
        status: pass
    human_judgment: false
  - id: D3
    description: "install docs brought in line: README/installation deprecation notes deleted, honesty page reads as current state, dependency-group tables list tui"
    requirement: PKG-01
    verification:
      - kind: unit
        ref: "tests/test_extras_guard.py#TestExtrasSurgeryResidueGates::test_base_bracket_residue_equals_sanctioned_list (sanctioned set is exactly the rewritten honesty row)"
        status: pass
      - kind: other
        ref: "grep -rn 'Deprecation note (v1.3)' README.md docs/getting_started/installation.md -> no hits"
        status: pass
    human_judgment: false
  - id: D4
    description: "owner approval of the two new PyPI packages against their audited upstream repositories"
    verification: []
    human_judgment: true
    rationale: "Dependency legitimacy is a human trust decision; the plan schedules it at the phase-end check (25-10) per human_verify_mode=end-of-phase. This plan only proves the packages exist and resolve."

# Metrics
duration: 3min
completed: 2026-10-21
status: complete
---

# Phase 25 Plan 01: `tui` Extra + PKG-01 Alias Retirement Summary

**Declared the `tui` extra (textual 8.2.8 + platformdirs 4.12.1 resolved) so every `.[full]` CI leg now installs Textual, and retired the deprecated `base` extras alias in the same install-line change.**

## Performance

- **Duration:** 3 min (07:35:27Z → 07:37:59Z)
- **Started:** 2026-10-21T07:35:27Z
- **Completed:** 2026-10-21T07:37:59Z
- **Tasks:** 2 (1 tracer, 1 auto)
- **Files modified:** 5

## Accomplishments

- `tui = ["textual>=8.2,<9", "platformdirs>=3.6,<5"]` declared after `mcp`, before `full`, with a one-line rationale comment; installed into the dev venv and verified (`textual 8.2.8`, `platformdirs 4.12.1`, `import textual` OK, `rich` unchanged at 15.0.0 — Pitfall 9 held).
- `dnallm[tui]` is now the first member of `test`, so `full → dev → test → dnallm[tui]` installs Textual on every `.[full...]` CI leg without touching `.github/workflows/ci.yml` (verified: all nine install lines are `.[full...]` variants).
- `all` now reads `dnallm[full,dev,test,notebook,docs,ui,mcp,fla,tui]`; `full` is byte-identical to before.
- The `base = ["dnallm[full]"]` alias and its four-line deprecation comment are gone; `test_base_alias_is_pure_self_reference` became `test_base_alias_is_removed` (absence assertion citing PKG-01 and the v1.4 cycle).
- `SANCTIONED_BASE_BRACKET_RESIDUE` shrank from four pairs to exactly one — the post-removal `dnallm[base]` honesty-matrix row — with the gate still failing in both directions.
- README/installation v1.3 deprecation blockquotes deleted; the honesty page's surviving row now reads as the live state, and both dependency-group tables document `tui`.

## Task Commits

1. **Task 1: Declare the `tui` extra and ride it on `test` + `all`** — `182a3c8` (feat)
2. **Task 2: PKG-01 — retire the deprecated alias and rewrite its guards** — `12d4102` (refactor)

**Plan metadata:** committed separately by the orchestrator (this executor does not write STATE.md/ROADMAP.md).

## Files Created/Modified

- `pyproject.toml` — added `tui`; `test` gained `dnallm[tui]`; `all` gained `tui`; `base` block and comment deleted.
- `tests/test_extras_guard.py` — `EXPECTED_TUI_MEMBERS`, `TestTuiExtraMembers` (3 tests); `EXPECTED_BASE_ALIAS` deleted; `test_base_alias_is_removed`; residue set reduced to one entry; class docstring rewritten for retirement.
- `README.md` — deprecation blockquote deleted; stale "deprecated `base` alias rows" reference corrected; `tui` row added; `all` row lists `+ tui`.
- `docs/getting_started/installation.md` — deprecation blockquote deleted; `tui` row added; `all` row lists `+ tui`.
- `docs/getting_started/install_honesty.md` — during-cycle row deleted; surviving row reworded to "since the v1.4 alias removal"; "Reading the rows honestly" and probe-provenance prose updated to the post-removal state.

## Decisions Made

- `tui` rides `full` transitively (via `dev → test`) rather than being named directly in `full` — adding it directly would fail the exact-set `test_full_extra_members_preserved`.
- `plan_test_rides_tui` asserts the literal `dnallm[tui]` member, because that single string is what makes Textual available on every CI pytest leg.
- PKG-01 removed `base`, not `full` — the REQUIREMENTS "`full` extras alias" wording is a mislabel; `full` is the real extra and stayed byte-identical.
- Docs prose avoids the literal `dnallm[base]` bracket spelling outside the one sanctioned matrix row; the residue gate is the reason.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Documented the new `tui` group and corrected a stale cross-reference**
- **Found during:** Task 2 (docs bring-in-line)
- **Issue:** Task 1 added a public install surface, but both dependency-group tables (README.md, `docs/getting_started/installation.md`) still enumerated `all` without `tui` and had no `tui` row; README also still pointed readers at "the deprecated `base` alias rows" on the honesty matrix, which now holds exactly one post-removal row.
- **Fix:** added a `tui` row to both tables, added `+ tui` to both `all` rows, and reworded the README pointer to "the post-removal `base` row".
- **Files modified:** README.md, docs/getting_started/installation.md
- **Verification:** `tests/test_extras_guard.py` green (15 tests across the two plan-level suites); no deprecation-note grep hits remain.
- **Committed in:** `12d4102` (Task 2 commit)

**2. [Rule 2 - Missing Critical] Brought the honesty page's prose in line with its surviving row**
- **Found during:** Task 2 (install_honesty.md edit)
- **Issue:** Deleting the during-cycle row left three passages describing the alias as live/future ("During the v1.3 alias cycle…", "the POST-removal state — pre-explained here so the v1.4 removal surprises nobody", and probe-provenance row numbering that no longer matched the table).
- **Fix:** rewrote the two "Reading the rows honestly" bullets and the probe-provenance paragraph to describe the post-removal state as current, and renumbered the surviving rows.
- **Files modified:** docs/getting_started/install_honesty.md
- **Verification:** residue gate green against the single sanctioned row; doc reads as current state.
- **Committed in:** `12d4102` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 2 — documentation accuracy directly caused by the packaging change)
**Impact on plan:** No scope creep beyond the two docs files the plan already owned. No source, CI, or coverage-config change.

## Issues Encountered

- The first draft of the rewritten honesty prose used the literal `dnallm[base]` bracket spelling in two explanatory lines; the bidirectional residue gate flagged both as unsanctioned residue (exactly its job). Rewrote the prose to name the extra without the bracket form, keeping the gate's sanctioned set at one row. No code change needed.

## Authentication Gates

None.

## Known Stubs

None.

## User Setup Required

None — no external service configuration required.

## Threat Flags

None beyond the plan's threat model. T-25-01 mitigation evidence: both packages are present on PyPI (`textual` 8.2.0–8.2.8, `platformdirs` 4.9.x/4.12.1), resolved under the owner-locked ranges, and `textual` imports cleanly. Owner approval against the audited upstream repositories is still scheduled for the phase-end human check (25-10).

## Next Phase Readiness

- 25-02 can create `dnallm/tui/` and `tests/tui/test_structure.py`: the extra already resolves, and `test` carries `dnallm[tui]`, so structure tests import nothing new.
- `full` remains the exact two-member set the rest of the phase depends on; no CI workflow edit was needed or made.
- TUI-01's structure-test half (zero textual imports outside `dnallm/tui/`, zero torch/transformers inside it) is NOT delivered here and lands in 25-02 — the requirement stays open until that plan's SUMMARY exists.

## Self-Check: PASSED

- Commits verified on HEAD: `182a3c8` (FOUND), `12d4102` (FOUND).
- No files created by this plan (all five deliverables are modifications); all five modified paths exist.
- Final verification re-run after the last commit: `tests/test_extras_guard.py` + `tests/test_version_contract.py` → 15 passed; `import dnallm` OK; tomllib PKG-01 assertion printed `PKG01_OK`.
- `commits:` measured from the persisted ledger: `git rev-list --count 255c201..12d4102` = 2.
