---
phase: 28-optional-extras-for-the-two-external-package-models-deepspee
plan: "05"
subsystem: docs
tags: [fla-retirement, base-dependency, readme, installation-guides, install-honesty, windows-kda, troubleshooting, d-09, d-13]

requires:
  - phase: 28-optional-extras-for-the-two-external-package-models-deepspee
    provides: fla promoted to base with triton-windows marker (28-01), canonical optional_dependencies guide wording (28-03)
provides:
  - Every user-facing installation surface outside the optional-dependency guide and notebooks tells the base-provided fla story (D-09)
  - The all aggregate no longer lists fla anywhere; both primary guides state the base/built-in contract (D-13)
  - Windows and troubleshooting guidance runs on base installs: honesty row, FAQ fix, and both KDA runbook editable-install commands
affects: [v1.4 close-out, README/docs consistency surfaces, future install-guide edits]

actuals:
  tokens: 5466     # chars/4 over the realized diff (21,864 bytes); plan estimate 28,000 (confidence low)
  tasks: 2
  commits: 2       # MEASURED: git rev-list --count 461b163..c1df6fa

plan_head_before: 461b1633ec3d7af2777299cececbecc311fba922
plan_head_after: c1df6fa78b5f92990a400be86864f15a03879bdf

tech-stack:
  added: []
  patterns:
    - "Residue-gate companion: when a sanctioned-residue pin (test_extras_guard.py) holds the exact physical line a plan requires rewording, the sanctioned entry moves in the same commit as the reworded line (28-01 gate-move precedent, documented in the gate's own comment)"

key-files:
  created: []
  modified:
    - README.md
    - docs/getting_started/installation.md
    - docs/getting_started/quick_start.md
    - docs/getting_started/install_honesty.md
    - docs/faq/models_troubleshooting.md
    - WINDOWS-KDA-VERIFY.md
    - tests/test_extras_guard.py

key-decisions:
  - "Retirement is named, not silenced: every touched surface says the former fla extra has been retired and a normal (re)install provides the package, matching the canonical optional_dependencies.md phrasing, so upgraders recognize why the old command stopped working"
  - "Residue-gate entries for README:260 and models_troubleshooting.md track the reworded lines inside the same task commits that reword them — the gate is the mechanism that records sanctioned text movement"
  - "WINDOWS-KDA-VERIFY.md keeps its Chinese ops-doc voice; the KDA probe body and measured evidence stay byte-identical — only the two install commands, the background block, the Step-1/2 triage pointer, and a dated revision note change"
  - "quick_start.md drops only fla from the group list; its historical base entry stays untouched (plan forbids unrelated edits)"

patterns-established:
  - "Single-physical-line paragraphs that residue gates pin: reword in place on one line, then move the pin — never reflow into multiple lines"

requirements-completed: [D-09, D-13]

coverage:
  - id: DEL-1
    description: "Primary installation surfaces aligned with base-provided fla: README extras table (fla row deleted, all composition slimmed, built-in-kernels statement added), PlantHelixSeek section rewritten around a normal editable install with warning and bare/backend-extra guidance retained, PlantHelixSeek/FungiHelixSeek cells marked built-in; installation.md all row + base-dependency note; quick_start.md group list (D-09, D-13)"
    requirement: D-09
    verification:
      - kind: automated
        ref: "python -I six-surface probe (task-1 leg): no .[fla] / dnallm[fla] / `fla` | forms; 'base dependenc' + 'built-in flash-linear-attention' present in README; old all-composition string absent"
        status: pass
      - kind: automated
        ref: "mkdocs build --strict -> exit 0, 'Documentation built in 4.77 seconds', zero warnings"
        status: pass
      - kind: unit
        ref: "tests/test_extras_guard.py 20 passed (residue gate accepts reworded README line) + tests/models/test_compat_docs_consistency.py::TestGuideDiscovery 3 passed"
        status: pass
    human_judgment: true
    rationale: "New user-visible sentences on the main install journey (built-in-kernels statement, rewritten PlantHelixSeek lead) are subject to the 2026-10-21 owner plain-language directive; automated legs prove retired-form absence, required wording presence, and build validity, not prose adequacy for biologists (27.2-04 / 28-03 DEL-1 precedent)"
  - id: DEL-2
    description: "Windows and troubleshooting guidance moved onto the base install: install_honesty dnallm[fla] row replaced by a base-install row (marker honored by both installers, [project] dependencies location, community-fork explanation and probe provenance preserved); FAQ fix installs/reinstalls standard DNALLM with fallback warning and version-band advice retained; WINDOWS-KDA-VERIFY both editable commands -> -e ., background/triage point at [project] dependencies, probe code and measured evidence byte-stable (D-09)"
    requirement: D-09
    verification:
      - kind: automated
        ref: "python -I six-surface probe (task-2 leg): no .[fla] / dnallm[fla] / 'fla extra' forms; '-e .' count 4 (>=2); '[project] dependencies' present; 'base dependenc' + 'triton-windows' in install_honesty.md"
        status: pass
      - kind: automated
        ref: "git diff WINDOWS-KDA-VERIFY.md contains zero probe-code lines (no chunk_kda/randn/softplus/maxdiff/assert changes) — KDA probe body and expected numerical evidence unchanged"
        status: pass
      - kind: unit
        ref: "tests/test_extras_guard.py 20 passed (reworded troubleshooting line sanctioned)"
        status: pass
    human_judgment: false

duration: 8min
completed: 2026-10-29
status: complete
---

# Phase 28 Plan 05: Optional extras: deepspeed & flash-attn model support Summary

**All six user-facing installation surfaces outside the optional-dependency guide now tell the base-provided fla story — the README extras table, both getting-started guides, the install-honesty matrix, the PlantHelixSeek FAQ fix, and the Windows KDA runbook dropped every retired fla-extra command, with the sanctioned-residue gate tracking the two reworded lines in the same commits.**

## Performance

- **Duration:** 8 min (2026-10-29T16:27:18Z → 16:39Z UTC)
- **Tasks:** 2 (Task 1 tracer with automated-only verify; Task 2 auto)
- **Files modified:** 7 tracked (6 planned + 1 companion test gate)
- **Estimate vs actual:** 28,000 tokens estimated / 5,466 actual (chars/4 over the realized diff) — another scoped docs-consistency plan landing far under a low-confidence estimate

## Verification Receipts (plan output spec)

| Verify | Result |
|---|---|
| Six-document source assertions (task-1 leg: README / installation / quick_start) | **PASS** — no `.[fla]`, `dnallm[fla]`, or `` `fla` | `` table-row forms; README carries `base dependenc…` and `built-in flash-linear-attention`; the old `all` composition string absent |
| Six-document source assertions (task-2 leg: install_honesty / models_troubleshooting / WINDOWS-KDA-VERIFY) | **PASS** — no retired forms incl. contiguous `fla extra`; WINDOWS-KDA-VERIFY has 4× `-e .` (≥2) and `[project] dependencies`; install_honesty carries `base dependenc…` + `triton-windows` |
| Strict MkDocs build | **PASS** — exit 0, "Documentation built in 4.79 seconds", zero warnings/errors (mkdocs 1.6.1 via `.venv/bin/mkdocs`) |
| KDA probe body byte-stability | **PASS** — the WINDOWS-KDA-VERIFY.md diff touches zero probe-code lines; Step 5 python block, fused-gate contract, and the maxdiff 0.000788 evidence are unchanged |
| Residue-gate companion | **20 passed** (`tests/test_extras_guard.py`, both reworded lines sanctioned) |
| Guide discovery + consistency lanes | **107 passed** (extras guard 20 + compat docs consistency 87, incl. TestGuideDiscovery README pins) |
| Repo-wide ruff | **All checks passed** |
| Tracer feedback gate | Task 1's automated-only `<verify>` re-ran end-to-end green on the committed tree before expansion into Task 2 (`HUMAN_VERIFY_MODE=end-of-phase`, auto mode off → no checkpoint); tracer verified end-to-end — expanding |

The Windows KDA probe itself was not re-executed on this Linux box (per the plan's verification note); its code and measured expected values remain byte-stable while only its installation route changed.

## Accomplishments

- **README:** the `fla` optional-group row is deleted and `all` no longer names it; a concise **Built-in kernels** statement after the table says flash-linear-attention 0.5.x and its Windows `triton-windows` marker are base dependencies; the PlantHelixSeek section now opens with "No extra is needed" around a normal editable install (silent non-KDA-fallback warning and bare-package/backend-extra/0.5.x-band guidance retained); the PlantHelixSeek and FungiHelixSeek special-model cells read "Built-in flash-linear-attention (base dependency)".
- **installation.md:** the `all` composition drops `fla`; a new note states the base-dependency contract (installs on every lane, Windows companion marker, retirement) without touching the unrelated flash-attn source-compile instructions.
- **install_honesty.md:** the `dnallm[fla]` Windows row is now a base-`dnallm`-install row — same marker, both installers, declaration moved to `[project]` dependencies at v1.4; the community-fork explanation and Row-4 probe provenance survive with location wording updated.
- **models_troubleshooting.md:** the fix installs or reinstalls standard DNALLM (`-e .` in a clone, `pip install --upgrade dnallm` for wheel users); the KDA fallback warning, measured separation evidence, and version-band advice are intact.
- **WINDOWS-KDA-VERIFY.md:** both editable-install commands are the base `-e .` form; the background block and the Step-1/2 failure-triage pointer name `[project] dependencies` instead of the deleted extra; a dated revision note records the route change with the probe/evidence explicitly unchanged.

## Task Commits

1. **Task 1 (tracer): align the primary installation surfaces with base-provided fla (D-09, D-13)** — `a57c2f5` (docs)
2. **Task 2: move Windows and troubleshooting guidance onto the base install (D-09)** — `c1df6fa` (docs)

Plan metadata commit follows this file.

## Files Created/Modified

- `README.md` — extras table (all composition, fla row deleted), built-in-kernels statement, PlantHelixSeek section, special-models cells
- `docs/getting_started/installation.md` — `all` row + base-dependency note
- `docs/getting_started/quick_start.md` — dependency-group list drops `fla`
- `docs/getting_started/install_honesty.md` — Windows row + Row-4 provenance
- `docs/faq/models_troubleshooting.md` — PlantHelixSeek fix block + guidance paragraph
- `WINDOWS-KDA-VERIFY.md` — background, two install commands, triage pointer, revision note
- `tests/test_extras_guard.py` — sanctioned cuda121/rocm residue entries track the two reworded lines (companion)

## Decisions Made

Recorded in key-decisions frontmatter. Load-bearing: **retirement is named everywhere** — each surface says the former `fla` extra has been retired and a normal install provides the package (the canonical `optional_dependencies.md` phrasing from 28-03), so a user who memorized `dnallm[fla]` learns why it stopped resolving rather than hitting a bare missing-extra warning.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking companion] Sanctioned-residue gate pinned the exact lines the plan required rewriting**
- **Found during:** Task 1 read-first pass (pre-edit scan of gates)
- **Issue:** `tests/test_extras_guard.py::TestExtrasSurgeryResidueGates::test_cuda121_rocm_residue_equals_sanctioned_list` pins README's "Three install notes…" line and models_troubleshooting.md's guidance paragraph as exact physical lines (the gate that already tracked the 28-01 pyproject-line move); the plan's mandated reword of both lines would have failed the gate's `missing` assertion
- **Fix:** each sanctioned entry was updated to the new exact line inside the same commit that rewords the line (README entry in the Task 1 commit, troubleshooting entry in the Task 2 commit), keeping every commit green; the gate's comment block records the 28-05 move per the 28-01 precedent
- **Files modified:** tests/test_extras_guard.py
- **Verification:** 20/20 extras-guard tests green after each commit; repo-wide ruff clean
- **Committed in:** a57c2f5 (README entry), c1df6fa (troubleshooting entry)

---

**Total deviations:** 1 auto-fixed (Rule 3 companion)
**Impact on plan:** zero scope growth beyond the gate companion; no user-facing wording differs from the plan's action spec.

## Issues Encountered

None. `mkdocs` is not on the shell PATH in this dev venv — both strict builds ran via `.venv/bin/mkdocs` (1.6.1), the same route 28-03 recorded.

## Authentication Gates

None.

## User Setup Required

None - no external service configuration required.

## Known Stubs

None - every changed statement traces to the 28-01 packaging facts (base `flash-linear-attention>=0.5.2,<0.6` + `triton-windows` marker in `[project]` dependencies).

## Next Phase Readiness

- Phase 28 is fully executed (plans 01–05 complete); D-09 closes with this plan (28-03 deliberately held it via the shared-ID gate) and D-13's aggregate-extra honesty is restored across every table.
- The Windows KDA runbook now installs exactly what the packaging declares; a future Windows verification run exercises the base install path end to end (probe body and expected values untouched).

## Self-Check: PASSED

- All 6 planned files + the companion test file exist on disk and are modified in the two task commits; both task commits (a57c2f5, c1df6fa) are ancestors of HEAD c1df6fa.
- Measured plan commit count over the persisted ledger: 2 (nonzero — no uncommitted production changes); `git status` clean of untracked files.
- Every plan verify re-ran green on the committed tree (both six-surface probes, strict MkDocs, 107 companion tests, repo-wide ruff).

---
*Phase: 28-optional-extras-for-the-two-external-package-models-deepspee*
*Completed: 2026-10-29*
