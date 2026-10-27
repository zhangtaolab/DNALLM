---
phase: 27-transformers-4-5-model-compatibility-census
plan: 05
subsystem: docs, ci, testing
tags:
  - transformers
  - census-lane
  - mamba-kernels
  - contract-test
  - plain-language
  - drift-gate
  - gap-closure

# Dependency graph
requires:
  - phase: 27-04
    provides: "docs page + offline drift gate + dispatch-only census lane (--verify) whose red run exposed UAT G-27-1"
  - phase: 27-02
    provides: "committed skip dispositions that ride merge_records onto unchanged verdicts (CR-01 seeded-run semantics)"
provides:
  - ".github/workflows/census_compat.yml - mamba-kernel install (pip bootstrap + sha256-pinned install_mamba.sh, TUNA) between .[full] and the census run"
  - "tests/test_runner_infra_contracts.py - TestCensusLaneInstallSequence (ordering, exact installer invocation, no masked lines, D-08 posture) - revert-proof offline"
  - "scripts/census_compat.py - four per-class plain-language leads for remote-code-broken rows (fail-closed on unmapped evidence) + 100-char word-boundary technical suffix"
  - "tests/models/test_compat_docs_consistency.py - independent lead map, suffix bound, jargon blocklist, 300-char cap (68 tests)"
  - "docs/user_guide/model_compatibility.md - regenerated page: six failure rows lead plain, technical cause one parenthetical away"
  - ".scratch/p27-uat/lane-sim/census-green.log - replica green-run evidence (49 matched / exit 0 / 302 s; scratch, not committed)"
affects: [census lane re-verification, docs site, future transformers bumps]

# Actuals (#2632) - chars/4 over the realized diff; commits measured from the ledger.
actuals:
  tokens: 7144
  tasks: 3
  commits: 3
  plan_head_before: "df28b98f1c5f9470b9c3568cd0a9e669eafadec4"
  plan_head_after: "450421e"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Committed-workflow content pins with deliberate-red proof: the contract class asserts install ordering, the exact sanctioned installer invocation, no masking on executed lines, and the D-08 posture against the committed YAML text - run against the pre-change text, the fix pins fail in seconds"
    - "Plain-language lead layering over a generated page: a per-class lead map keyed on distinctive registry-evidence substrings (most-specific-first, fail-closed ValueError on unmapped shapes) plus a word-boundary-bounded technical suffix; generator and drift gate carry independent copies (checker independence)"
    - "Runner-box replica validation of a dispatch-deferred lane: the lane's exact command is executed on the same hardware/network/interpreter shape and the exit code is captured by redirection (never a pipe), giving the gap's evidence while main-branch registration blocks the live dispatch"

key-files:
  created: []
  modified:
    - .github/workflows/census_compat.yml
    - tests/test_runner_infra_contracts.py
    - scripts/census_compat.py
    - tests/models/test_compat_docs_consistency.py
    - docs/user_guide/model_compatibility.md

key-decisions:
  - "The census lane installs the mamba kernels through the sanctioned sha256-pinned scripts/install_mamba.sh pipeline (pip bootstrap first, TUNA index), never the plain .[mamba] extra: the uv source build of that extra fails on this box (verified), while the script route is proven here - and the committed contract test makes an editorial revert fail the fast lane in seconds."
  - "Broken-row leads are per class, honest per class: checkpoint-data-defect -> damaged files; forward-contract -> own defect; dependency-set -> external package; ALiBi/meta-construction and library removals -> library change. A version-independent model defect is never worded as a library removal."
  - "The technical cause is demoted, not deleted: bounded at a 100-char word boundary with a single trailing ellipsis, so every status cell stays under 300 characters while the engineering detail survives one parenthetical away."
  - "DOCS_DATA_DATE stays 2026-10-27: this is a presentation-only change, not a census re-run; the data date moves only alongside --write-registry."
  - "The masking pin checks executed lines only: the lane's own comments name the forbidden '|| true' shape as prose (pre-existing), which is documentation, not masking."

requirements-completed: ["TBD"]

# Coverage metadata (#1602)
coverage:
  - id: D1
    description: "Census lane installs the mamba kernels via the sanctioned pipeline, pinned offline by TestCensusLaneInstallSequence (ordering, exact invocation, no masking, D-08 posture)"
    verification:
      - kind: unit
        ref: ".venv/bin/python -m pytest tests/test_runner_infra_contracts.py -q (12 passed; baseline 8)"
        status: pass
      - kind: other
        ref: "deliberate-red: the new ordering + installer pins run against git show HEAD~1:census_compat.yml fail (bootstrap/installer lines absent pre-change); posture assertion re-run via yaml.safe_load on the committed workflow"
        status: pass
    human_judgment: false
  - id: D2
    description: "Plain-language failure cells: four per-class leads + bounded technical suffixes in the generator, regenerated page, independent drift-gate derivation (68 tests) incl. jargon blocklist and 300-char cap"
    verification:
      - kind: unit
        ref: ".venv/bin/python -m pytest tests/models/test_compat_docs_consistency.py -q (68 passed; baseline 54); deliberate-red: one altered lead sentence -> exactly the GENA-LM row tests red"
        status: pass
      - kind: other
        ref: "python -I scripts/census_compat.py --emit-docs /tmp/mc-recheck.md && cmp ... -> BYTE-IDENTICAL; ruff check + format --check clean on all three touched Python files"
        status: pass
    human_judgment: false
  - id: D3
    description: "G-27-1 replica green run: the lane's exact command against the replica venv with the sanctioned kernels reports full agreement and exit 0"
    verification:
      - kind: integration
        ref: ".scratch/p27-uat/lane-sim/census-green.log: grep -c '^  matched ' = 49, grep -c UNEXPECTED = 0, 'OK: census matches committed registry fields' present, 'LANE-EXIT-CODE: 0' present; all five mamba-family entries matched"
        status: pass
    human_judgment: false

# Metrics
duration: 21min
completed: 2026-10-27
status: complete
---

# Phase 27 Plan 05: Gap Closure - Census Lane Green + Plain-Language Failure Cells Summary

**The census lane can now exit 0 (its install step carries the sanctioned sha256-pinned mamba kernels, revert-proofed by an offline contract test, and the lane's exact command proven green on the runner-box replica: 49 matched / exit 0), and the six "not working" documentation rows now lead with honest plain-language verdicts a biologist can act on, with the engineering cause demoted to a bounded parenthetical**

## Performance

- **Duration:** 21 min
- **Started / Completed:** 2026-10-27T12:12:19Z / 2026-10-27T12:33:00Z
- **Tasks:** 3
- **Files:** 5 (0 created, 5 modified; plus scratch evidence outside git)

## Accomplishments

- **Task 1 - census-lane mamba install + contract pins (`a0c7d5f`):** `.github/workflows/census_compat.yml` now bootstraps pip into the uv venv and runs `bash scripts/install_mamba.sh --python .venv/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple` between the `.[full]` install and the census run, with the rationale comment (torch C++20 floor, sha256 pin, TUNA route, census-prerequisite semantics). Nothing else changed (diff: +20 lines, 0 deletions). `tests/test_runner_infra_contracts.py` gained `TestCensusLaneInstallSequence` (4 tests): install ordering, the exact sanctioned invocation, no `|| true` masking on executed lines, and the D-08 posture (dispatch-only trigger, contents: read, runner labels, event guard, 240-minute timeout). Deliberate-red: run against the pre-change workflow text, the ordering and installer pins fail (the new lines are absent) — an editorial revert re-creates the permanently-red lane from UAT G-27-1 and fails in seconds.
- **Task 2 - plain-language failure cells (`5c1c377` RED, `450421e` GREEN):** the generator now derives each remote-code-broken cell as `Not usable with the current transformers - <per-class lead> (technical: <bounded cause>)`. Four honest leads: damaged-files (GENA-LM), own-defect (MutBERT), external-package (GENA-LM-BigBird, Omni-DNA), library-change (DNABERT-2's meta-device construction, GPN's removed native architecture). Class map keys are grounded in the six committed evidence details, checked most-specific-first, with a fail-closed ValueError on unmapped shapes. The technical cause is truncated at a word boundary to 100 characters with a single trailing ellipsis. The page was regenerated via `--emit-docs` only: exactly the six Not-usable rows changed, every status cell is under 300 characters, DOCS_DATA_DATE stayed 2026-10-27. The drift gate was extended in lockstep with its own independent copies (class map, leads, suffix bound, jargon blocklist, 300-char cap): 68 passed (was 54). Deliberate-red: altering one lead sentence turned exactly the GENA-LM row tests red; restoring returned 68 green.
- **Task 3 - replica green run (scratch evidence):** re-seeded `census-artifacts/census-fresh.json` from the committed census and ran the lane's exact command (`--run --with-loads --verify --artifact`) against the replica venv (`torch 2.14.1+cu130`, `transformers 5.17.0`, CUDA available, `mamba_ssm 2.3.2.post1` + `causal_conv1d 1.7.0` importable). Result: **49 matched / 0 UNEXPECTED / "OK: census matches committed registry fields" / exit 0**, wall-clock **302 s** for the census step (the sanctioned mamba install itself had completed earlier: `mamba-install.log`). All five previously-disagreeing mamba-family entries matched (caduceus-ph/ps, PlantCaduceus_l32 -> shim-required-tf5 via the seeded disposition riding merge_records, PlantCAD2, PlantBiMoE). Running 49 model fetches + loads in ~5 minutes leaves the 240-minute lane budget comfortably intact.

## Task Commits

Each task was committed atomically (Task 2 followed the RED -> GREEN TDD gate):

1. **Task 1: census lane installs mamba kernels via sanctioned pipeline + contract pins** - `a0c7d5f` (feat)
2. **Task 2 RED: failing plain-language lead gate for broken rows** - `5c1c377` (test)
3. **Task 2 GREEN: plain-language failure-cell leads in the docs generator + page** - `450421e` (feat)

**Plan metadata:** committed after this SUMMARY (docs: complete plan).

## Files Created/Modified

- `.github/workflows/census_compat.yml` - pip bootstrap + sanctioned mamba install inside the existing install step; posture untouched.
- `tests/test_runner_infra_contracts.py` - `TestCensusLaneInstallSequence` (4 new tests; file now 12 passed).
- `scripts/census_compat.py` - `DOCS_BROKEN_LEADS`, `DOCS_BROKEN_LEAD_CLASSES`, `DOCS_TECHNICAL_FRAGMENT_MAX_CHARS`, `docs_broken_lead_class()`, `docs_bound_technical_fragment()`, new remote-code-broken cell shape.
- `tests/models/test_compat_docs_consistency.py` - independent lead map + suffix bound + jargon blocklist + 300-char cap (68 tests, was 54).
- `docs/user_guide/model_compatibility.md` - regenerated (six failure rows changed; 43 rows, header, footnotes, data date byte-stable).
- `.scratch/p27-uat/lane-sim/census-green.log` - green-run evidence (scratch, gitignored, not committed).

## Decisions Made

- Per-class honest leads (see frontmatter key-decisions): damaged-files / own-defect / external-package / library-change, fail-closed on unmapped evidence.
- Technical suffix bounded at 100 chars (word boundary + single ellipsis) so the 300-char cell cap holds with margin while the engineering cause survives.
- DOCS_DATA_DATE untouched (presentation-only change).
- Masking pin scoped to executed lines (the lane's own prose comment names the forbidden shape; that is documentation, not masking).
- Deferred UI-REVIEW warning (b) — inline footnote markers — deliberately NOT implemented (owner-deferred in 27-UAT.md).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking detail] Masking pin matched the lane's own prose comment**
- **Found during:** Task 1 (first contract-test run)
- **Issue:** the plan asked the pin to assert "the workflow text contains no `|| true` masking", but the committed workflow's own comment says "NO `|| true` masking — a red step means real drift" — a literal whole-text check fails on documentation about the rule.
- **Fix:** the pin checks executed (non-comment) lines only, with a comment stating why; a masked command still fails it.
- **Files modified:** tests/test_runner_infra_contracts.py
- **Verification:** 12 passed; deliberate-red vs pre-change text.
- **Committed in:** `a0c7d5f`

**2. [Rule 1 - Bug] YAML trigger pin iterated a bare-scalar string character-wise**
- **Found during:** Task 1 (first contract-test run)
- **Issue:** `on: workflow_dispatch` parses (via pyyaml's `True` key gotcha) to the plain string `"workflow_dispatch"`; `set(triggers)` therefore produced a character set and the posture pin failed spuriously.
- **Fix:** normalize the trigger value (string -> single-element set; dict/list -> key/element set) before asserting `{"workflow_dispatch"}`.
- **Files modified:** tests/test_runner_infra_contracts.py
- **Verification:** 12 passed; the plan's own `python -I` posture command also passes.
- **Committed in:** `a0c7d5f`

---

**Total deviations:** 2 auto-fixed (1 Rule 3, 1 Rule 1)
**Impact on plan:** none on scope or intent; both were test-authoring correctness fixes found before the first Task 1 commit, and both pins are stricter in spirit (a real masking command and any real posture change still fail).

## Issues Encountered

- The first contract-test run failed on both pins above; both were fixed within Task 1 before its commit (no downstream consumers).
- Two ruff findings during authoring (bad-quotes-inline-string in the test file; a formatter join in the gate file) were fixed with the project's own ruff format/check before each commit.

## Deliberate-Red Demonstrations (phase convention)

1. **Contract class vs pre-change workflow:** re-ran the new assertions against `git show HEAD~1:.github/workflows/census_compat.yml` — the install-ordering pin fails (bootstrap/installer lines absent) and the installer-shape pin fails; the masking and posture pins hold on both sides by design (they are standing guards, not fix pins).
2. **Drift gate vs an altered lead:** temporarily changed one lead sentence ("damaged" -> "broken") in the page; `pytest tests/models/test_compat_docs_consistency.py` reported exactly the two GENA-LM row tests red (2 failed / 66 passed); the pristine copy was restored and 68 passed returned (restored file `cmp`-identical to the pre-edit page).

## Authentication Gates

None - no authentication surfaces were touched.

## User Setup Required

None - no external service configuration required. The LIVE dispatch (`gh workflow run census_compat.yml --ref <branch>`) remains a deferred operational act until the v1.4 -> main merge (GitHub registers dispatchable workflows from the default branch only; main is frozen for this cycle) — a structural constraint recorded in the plan, not a gap.

## Threat Surface Scan

No new security-relevant surface. The plan's register is implemented as written: T-27-14 (editorial revert / masking re-creates the red lane) mitigated by `TestCensusLaneInstallSequence`; T-27-15 (mislabelled failure class) mitigated by per-class evidence-substring mapping, fail-closed ValueError, and the gate's independent re-derivation + jargon blocklist; T-27-SC (supply chain) unchanged — the lane installs only the pre-existing sha256-pinned `install_mamba.sh` versions (mamba-ssm 2.3.2.post1 + causal_conv1d 1.7.0), no new package enters the dependency set.

## Next Phase Readiness

Phase 27 gap closure complete: G-27-1 and G-27-2 both carry this SUMMARY so `/gsd-verify-work` can reconcile them as resolved. The census lane is structurally green-capable and revert-proof; the docs page's failure rows now speak to the target audience. Remaining operational act after the v1.4 -> main merge: one live `gh workflow run census_compat.yml --ref v1.4` to close the standing human-verification item (same hardware, network, and command as this replica proof).

## Self-Check: PASSED

- `.github/workflows/census_compat.yml` - modified and committed in `a0c7d5f`
- `tests/test_runner_infra_contracts.py` - modified and committed in `a0c7d5f`
- `scripts/census_compat.py`, `docs/user_guide/model_compatibility.md` - committed in `450421e`
- `tests/models/test_compat_docs_consistency.py` - committed in `5c1c377` (RED, formatting applied pre-commit)
- `.scratch/p27-uat/lane-sim/census-green.log` - present on disk (49 matched / 0 UNEXPECTED / exit 0); scratch by design
- Commits `a0c7d5f`, `5c1c377`, `450421e` - all ancestors of HEAD; task-commit ledger count 3/3 (the plan-metadata commit adds one more, excluded from `commits:` per the 27-04 convention)

---
*Phase: 27-transformers-4-5-model-compatibility-census*
*Completed: 2026-10-27*
