---
phase: 261026-aij
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - .github/workflows/ci.yml
  - .github/workflows/README.md
  - tests/test_dispatch_direction.py
  - .planning/milestones/v1.1-phases/09-ci-wiring-census-verification/09-CENSUS-ROLLUP.md
autonomous: true
must_haves:
  truths:
    - "The D-03 Stage 0.5 gate matches a live collection on v1.4 tip: running the exact gate selector (.venv/bin/python -m pytest tests/examples --collect-only -q -m \"not giants\") prints 222/223 tests collected (1 deselected) — measured at planning time on tip 6b055e5 (0.90s collect) — and the re-pinned grep -qE pattern passes against that tail line"
    - "Both Stage 0.5 carriers in ci.yml (the grep -qE pattern and the FAIL echo's expected-triple message) carry 222/223, and the retired 216/217 literal appears nowhere in .github/workflows/ci.yml, .github/workflows/README.md, or tests/test_dispatch_direction.py"
    - "The census growth is attributed in the dated 09 rollup note: +6 collected items are the six rice-genome warm-or-skip guard tests of 0885416 (2026-10-23, the nightly-flakiness rice fix class); the Phase-16 example commits d05a3b9/87b0c13/628926a (2026-10-13, CRE/NER platform-dispatch work) modified existing tests without changing the collected count; the deselect split is unchanged at 1 (the giants-marked evo execution test, D-01)"
    - "The ci.yml D-12 comment's authoritative-record pointer resolves again — it names .planning/milestones/v1.1-phases/09-ci-wiring-census-verification/09-CENSUS-ROLLUP.md (the archived location; the old .planning/phases/ path is an empty directory)"
    - "The 2026-10-07 post-close addendum and every prior rollup section are byte-untouched (append-only discipline), and the historical 216/217 mentions in .planning/WINDOWS.md, .planning/MILESTONES.md, .planning/PROJECT.md are left as dated history"
  artifacts:
    - .github/workflows/ci.yml (Stage 0.5 grep -qE pattern + FAIL echo literal re-pinned to 222/223; D-12 comment record-pointer refreshed to the archived rollup path)
    - .github/workflows/README.md (stage-0.5 table row cites the current pin 222/223)
    - tests/test_dispatch_direction.py (root-placement docstring cites the current census triple)
    - .planning/milestones/v1.1-phases/09-ci-wiring-census-verification/09-CENSUS-ROLLUP.md (dated v1.4 recalibration section appended after line 380)
  key_links:
    - "one number on four surfaces in one commit: ci.yml gate grep -qE literal == live pytest collection tail line == README stage-0.5 row pin == rollup note's measured figure"
---

<objective>
Recalibrate the example-nightly D-03 census collection assertion for the v1.4
line. The Stage 0.5 hard gate in .github/workflows/ci.yml is pinned at a triple
measured on 2026-10-11 (Phase 13 D-04, commit ac4115f): 216/217 tests collected
(1 deselected). The v1.4 example commits grew the census by 6 items, so every
nightly dispatch now fails the gate before stage 1 — the guard working exactly
as designed, waiting for its deliberate bump-point re-pin.

Measured evidence (all live, planning time 2026-10-26):
- Pin commit ac4115f, detached worktree, exact gate selector: 216/217 tests
  collected (1 deselected) in 1.24s — the pin was honest when landed.
- v1.4 tip 6b055e5, same selector: **222/223 tests collected (1 deselected) in
  0.90s** — the new triple.
- +6 arithmetic closes exactly: 0885416 (2026-10-23, warm-or-skip guard for
  rice genome inputs) adds the six new test functions; d05a3b9/87b0c13/628926a
  (2026-10-13, Phase 16 CRE/NER example work) add zero new `def test_`
  functions (verified per-commit) — they modified existing tests in the same
  window. 216 + 6 = 222, deselect count unchanged.

Live carriers of the retired literal (all must flip; the historical mentions
in .planning/WINDOWS.md, MILESTONES.md, PROJECT.md stay as dated history):
1. .github/workflows/ci.yml:910 — the gate's `grep -qE '^=* *216/217 tests
   collected \(1 deselected\) in '` pattern
2. .github/workflows/ci.yml:911 — the FAIL echo's `expected triple:` message
3. .github/workflows/README.md:131 — the stage-0.5 table row ("pinned at
   `216/217 tests collected (1 deselected)` in the workflow")
4. tests/test_dispatch_direction.py:31 — the root-placement docstring line
   "Root placement keeps the examples census triple (216/217) untouched."

Plus one adjacent one-line correction in the same commit: the ci.yml D-12
comment (line ~675) points at `.planning/phases/09-ci-wiring-census-verification/`
for the authoritative record — that directory is empty since milestone
archiving; the rollup lives at `.planning/milestones/v1.1-phases/09-ci-wiring-census-verification/09-CENSUS-ROLLUP.md`.

Purpose: the D-03 discipline (one census-moving edit per change, re-pinned
same-change with the record appended) — the same shape as the rollup's
2026-10-07 post-close addendum.
Output: re-pinned gate + README + docstring citation, dated recalibration note
appended to the 09 census rollup, all in one commit.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md

@.github/workflows/ci.yml
@.github/workflows/README.md
@tests/test_dispatch_direction.py
@.planning/milestones/v1.1-phases/09-ci-wiring-census-verification/09-CENSUS-ROLLUP.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Re-pin the D-03 triple on every live carrier (ci.yml gate + README row + docstring citation)</name>
  <files>.github/workflows/ci.yml, .github/workflows/README.md, tests/test_dispatch_direction.py</files>
  <action>
    RE-MEASURE FIRST (mutable-scope discipline — do not pin a planned number):
    run `.venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants" 2>/dev/null | tail -1`.
    Expected at v1.4 tip: `222/223 tests collected (1 deselected)`. If the fresh
    measurement differs (more example tests landed since planning), pin the
    MEASURED triple everywhere below and correct the attribution facts in
    Task 2's note accordingly — never ship a pin that disagrees with the box.

    Then four literal edits, nothing else:
    (1) ci.yml Stage 0.5 gate (line ~910): inside the grep -qE pattern, replace
    the old `216/217 tests collected \(1 deselected\)` substring with the new
    triple in the same escaped shape — the pattern must remain
    `'^=* *<NEW> tests collected \(<N> deselected\) in '`. The selector flags
    on the pytest line above stay EXACTLY `-m "not giants"`; the fail-loud
    shape (grep -qE, cat census-collect.txt, exit 1) is untouched.
    (2) ci.yml FAIL echo (line ~911): replace `expected triple: 216/217 tests
    collected (1 deselected)` with the new triple, plain parens, same sentence.
    (3) .github/workflows/README.md stage-0.5 table row (line ~131): replace
    `pinned at \`216/217 tests collected (1 deselected)\` in the workflow`
    with the new triple. Row prose otherwise unchanged.
    (4) tests/test_dispatch_direction.py root-placement docstring (line ~31):
    the parenthesized census citation becomes `(222/223)` (or the measured
    triple). Docstring-only — zero code changes in the file.

    Adjacent one-liner in the same file as (1)/(2): the D-12 measured-actuals
    comment block (line ~675) says `authoritative record:
    .planning/phases/09-ci-wiring-census-verification/` — update only that
    path to `.planning/milestones/v1.1-phases/09-ci-wiring-census-verification/`
    (the directory was archived at milestone close; the run-id text and the
    `09-CENSUS-ROLLUP.md` filename on the surrounding comment lines stay).

    Do NOT add any new comment or doc text citing the retired old triple — the
    acceptance gate below requires it absent from all three files. Do NOT touch
    .planning/WINDOWS.md, .planning/MILESTONES.md, or .planning/PROJECT.md
    (dated historical records). No commit yet — Task 2 lands in the same commit.
  </action>
  <verify>
    <automated>.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('ci.yml parses')" && [ "$(grep -c '222/223' .github/workflows/ci.yml)" = "2" ] && [ "$(grep -c '216/217' .github/workflows/ci.yml)" = "0" ] && [ "$(grep -c '216/217' .github/workflows/README.md)" = "0" ] && grep -q '222/223 tests collected (1 deselected)' .github/workflows/README.md && [ "$(grep -c '216/217' tests/test_dispatch_direction.py)" = "0" ] && grep -q '(222/223)' tests/test_dispatch_direction.py && .venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants" 2>/dev/null | tail -1 | grep -qE '^=* *222/223 tests collected \(1 deselected\) in ' && echo CENSUS-GATE-GREEN && .venv/bin/pytest tests/test_dispatch_direction.py -q && grep -n 'milestones/v1.1-phases/09-ci-wiring-census-verification' .github/workflows/ci.yml</automated>
  </verify>
  <done>CENSUS-GATE-GREEN echoed (the re-pinned grep pattern matches a live collection tail line — the exact gate logic proven locally); ci.yml parses as YAML; the new triple appears exactly twice in ci.yml (both carriers) and the old literal is absent from ci.yml, the workflows README, and the dispatch-direction docstring; README stage-0.5 row and the docstring cite the new triple; tests/test_dispatch_direction.py still 19 passed (baselined green on tip); the D-12 comment's record pointer names the archived milestones path.</done>
</task>

<task type="auto">
  <name>Task 2: Append the dated recalibration note to the 09 census rollup</name>
  <files>.planning/milestones/v1.1-phases/09-ci-wiring-census-verification/09-CENSUS-ROLLUP.md</files>
  <action>
    Append a new section at the END of the rollup (it currently ends at line
    380 with the 2026-10-07 post-close addendum's run 37550730293 record) —
    append-only, byte-untouched above. Heading:

    `## v1.4-line recalibration (2026-10-26, quick 261026-aij)`

    Content (prose + facts, mirroring the addendum's shape; use the measured
    figures, correcting attribution only if Task 1's fresh measurement
    differed):
    - Pin lineage in one line: 193/202 (9 deselected) at phase-9 close →
      197/206 (9 deselected) at the 2026-10-07 addendum → the Phase 13 D-04
      re-pin at ac4115f (2026-10-11) that left the mcp_example pair collected
      and collapsed the deselect set 9→1 → 222/223 (1 deselected) today.
    - Measurement evidence: the pin reproduced live at ac4115f in a detached
      worktree (216/217, 1.24s) and the v1.4 tip 6b055e5 measures 222/223
      (1 deselected, 0.90s) with the exact stage-0.5 selector — both at
      planning time 2026-10-26; cite the selector flags verbatim.
    - Delta attribution: the +6 collected items are the six warm-or-skip guard
      tests for rice genome inputs added by 0885416 (2026-10-23 — the
      nightly-flakiness rice fix class from the v1.4 triage); commits
      d05a3b9, 87b0c13, 628926a (2026-10-13, Phase 16 CRE/NER
      platform-dispatch work) modified existing example tests without
      changing the collected count (verified per-commit: zero new test
      functions). Deselect split unchanged: 1 = the giants-marked evo
      execution test (D-01).
    - Carriers re-pinned in the same commit as this note (the designed
      bump-point, triggered on the v1.4 line): both ci.yml Stage 0.5
      carriers (the grep -qE gate pattern and the FAIL echo), the workflows
      README stage-0.5 row, and the tests/test_dispatch_direction.py
      root-placement docstring citation; plus the D-12 comment's
      authoritative-record pointer refreshed to this file's archived
      milestones/v1.1-phases location.
    - Runner proof pending, addendum precedent cited: the 2026-10-07 re-pin
      was proven by example-nightly run 37550730293 at the re-pin HEAD; the
      next example-nightly dispatch at this re-pin HEAD records the green
      gate for 222/223 (05:30 UTC schedule or owner dispatch — no
      auto-dispatch from the executor, same self-hosted queue discipline as
      Phase 13 D-04).

    Then make the single commit carrying ALL of this quick task (ci.yml +
    README + test docstring + this note — the D-03 same-change discipline),
    message: `ci(261026-aij): D-03 census recalibration 216/217 -> 222/223 on
    v1.4 (+6 from 0885416) + dated 09 rollup note`. No Co-Authored-By
    trailers (repo convention).
  </action>
  <verify>
    <automated>[ "$(grep -c '^## v1.4-line recalibration (2026-10-26' .planning/milestones/v1.1-phases/09-ci-wiring-census-verification/09-CENSUS-ROLLUP.md)" = "1" ] && grep -q '222/223 tests collected (1 deselected)' .planning/milestones/v1.1-phases/09-ci-wiring-census-verification/09-CENSUS-ROLLUP.md && grep -q '0885416' .planning/milestones/v1.1-phases/09-ci-wiring-census-verification/09-CENSUS-ROLLUP.md && grep -q 'ac4115f' .planning/milestones/v1.1-phases/09-ci-wiring-census-verification/09-CENSUS-ROLLUP.md && [ "$(grep -c 'Post-close census addendum' .planning/milestones/v1.1-phases/09-ci-wiring-census-verification/09-CENSUS-ROLLUP.md)" = "1" ] && tail -40 .planning/milestones/v1.1-phases/09-ci-wiring-census-verification/09-CENSUS-ROLLUP.md | grep -q 'v1.4-line recalibration' && LASTSUBJ=$(git log -1 --format='%h %s') && [ -n "$LASTSUBJ" ] && echo "$LASTSUBJ" | grep -q '261026-aij'</automated>
  </verify>
  <done>The dated section exists exactly once at the rollup's end and carries the measured 222/223 (1 deselected) figure, the 0885416 +6 attribution with ac4115f pin lineage, the same-commit carrier list, and the pending-runner-proof clause citing the run-37550730293 precedent; the 2026-10-07 addendum heading still present exactly once (prior sections untouched); one commit titled ci(261026-aij) carries all four files.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| workflow author -> CI gate logic | editing .github/workflows/ci.yml edits the job's own hard-gate logic — the change surface IS the guard |

No new trust boundaries: no package installs, no external API integration, no
secrets, no untrusted input handling — a literal recalibration of an existing
collect-only assertion plus documentation.

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Severity | Mitigation Plan |
|-----------|----------|-----------|-------------|----------|-----------------|
| T-aij-01 | Tampering | ci.yml Stage 0.5 gate | mitigate | low | The edit changes only the pinned triple literals; the fail-loud gate shape (grep -qE against the collection tail, cat of census-collect.txt, exit 1 on mismatch) and the exact selector flags stay untouched, and the verify block re-proves the gate logic against a live collection before commit — a mis-pinned or gutted gate cannot land green. |
| T-aij-02 | Repudiation | 09 census rollup record | mitigate | low | The recalibration lands as a dated append-only section with run-verifiable attribution (commit ids 0885416/ac4115f, measured figures, pending-runner-proof clause) following the established addendum precedent — no history is rewritten. |
| T-aij-SC | Tampering | package installs | accept | low | No package-manager installs anywhere in this task (existing venv, stdlib yaml parse check only); the package-legitimacy gate is not triggered. |
</threat_model>

<verification>
- Live gate proof: the Task 1 verify runs the exact stage-0.5 selector locally and matches the re-pinned grep -qE pattern against the collection tail line — the gate is proven, not assumed (222/223 tests collected, 1 deselected, at v1.4 tip 6b055e5; baselined at planning time in a detached worktree at the pin commit ac4115f reproducing 216/217).
- Literal hygiene: the new triple appears exactly twice in ci.yml (both Stage 0.5 carriers) and once each in the README stage-0.5 row and the dispatch-direction docstring; the old literal is absent from all three code/doc surfaces (baselines at planning time: 2 / 1 / 1 occurrences respectively). Historical planning records (WINDOWS.md, MILESTONES.md, PROJECT.md) intentionally keep their dated mentions.
- YAML integrity: ci.yml parses via yaml.safe_load after the edit (baselined parsing on tip).
- No behavior change: tests/test_dispatch_direction.py still 19 passed in 0.57s (docstring-only edit); no dnallm/ package code touched, so the pytest-with-code-change owner rule is not triggered; example collection deselect split unchanged at 1 (giants, D-01).
- Record integrity: the 09 rollup gains exactly one dated section; the 2026-10-07 addendum heading count stays 1; a single ci(261026-aij) commit carries all four files per the D-03 same-change discipline.
</verification>

<success_criteria>
- The D-03 Stage 0.5 gate would pass a nightly dispatch at the re-pin HEAD: its grep pattern matches the live collection triple 222/223 (1 deselected) on v1.4 tip (or the executor's fresher measured triple, with Task 2's attribution corrected to match — never a stale number).
- All four live carriers agree on one number in one commit: ci.yml gate pattern, ci.yml FAIL echo, workflows README stage-0.5 row, dispatch-direction docstring; the ci.yml D-12 comment points at the rollup's real archived path.
- The 09 census rollup ends with the dated v1.4 recalibration section carrying the measured figures, the +6 attribution (six rice warm-or-skip guard tests from 0885416; Phase-16 commits count-neutral), the ac4115f pin lineage, and the pending-runner-proof clause citing the run 37550730293 precedent.
- Prior rollup sections and all historical planning records byte-untouched; tests/test_dispatch_direction.py green; ci.yml parses.
</success_criteria>

<output>
Create `.planning/quick/261026-aij-recalibrate-the-example-nightly-d-03-cen/261026-aij-SUMMARY.md` when done
</output>
