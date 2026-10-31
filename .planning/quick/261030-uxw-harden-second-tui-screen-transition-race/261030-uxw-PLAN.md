---
phase: 261030-uxw
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - tests/tui/test_screens.py
  - .planning/WINDOWS.md
autonomous: true
requirements:
  - QUICK-UXW
estimate:
  tokens: 18000
  raw_tokens: 18000
  tasks: 2
  confidence: low
must_haves:
  truths:
    - "The edit-existing-config test waits for ConfigScreen to become active through the existing bounded _wait_until helper before settling the screen and asserting the recorded edit_config call"
    - "The targeted test file and complete tests/tui lane pass without any production TUI source change or unrelated test sweep"
    - "WINDOWS ledger row 42 records the CI evidence and screen-transition family note, carries kind flake (fixed) and status fixed, and cites the actual test-fix commit"
  artifacts:
    - "tests/tui/test_screens.py — a bounded ConfigScreen-active poll in test_edit_an_existing_config_opens_it_through_the_core using the file's existing _wait_until helper"
    - ".planning/WINDOWS.md — one new table-only row 42 immediately after row 41, with the failure evidence, family note, resolution, and timestamps"
  key_links:
    - "entry-from-config click -> bounded _wait_until poll -> ConfigScreen active -> _settle_config -> unchanged fake.edit_calls assertions"
    - "test-fix commit -> row-42 reason -> CI run 41065033616 failure -> rows 41-42 family note -> future instances hardened only on CI evidence"
---

<objective>
Make the second CI-observed Windows screen-transition assertion resilient with a
minimal test-only change, prove it on the focused TUI lanes, and register ledger
row 42 with the reusable family diagnosis and traceable fix evidence.

Purpose: clicking the entry-screen config action can leave the test observing
EntryScreen before the asynchronous transition to ConfigScreen completes. A
single pause is not a readiness guarantee on a slow runner. Polling the desired
screen type through the existing bounded helper removes that race without
changing production behavior.

Output: one atomic test-fix commit and one ledger commit, with no changes under
dnallm/tui and no co-author trailers.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@tests/tui/test_screens.py
@.planning/WINDOWS.md

Planning facts on branch v1.4: the target test clicks #entry-from-config, pauses
once, and immediately snapshots isinstance(app.screen, ConfigScreen) into active
before calling _settle_config. The file already provides _wait_until, which polls
for up to 500 iterations with a 0.01-second asyncio sleep (approximately five
seconds). Commit 166dda2 established the boolean-poll idiom for the row-41
HeaderTitle race. Here the same helper waits directly for the required screen
type; the existing _settle_config call and all three fake.edit_calls assertions
remain after transition confirmation.

The neighboring empty-path test is intentionally unchanged. It asserts that an
empty path leaves EntryScreen active and produces a prompt rather than opening
ConfigScreen. Polling for the already-active EntryScreen would return immediately
and would not settle a transition, so it is not the CI-evidenced race shape and
rewriting it would add no protection.

Ledger rows 40-41 are table-only triage entries; the embedded JSON ledger ends at
id 39. Preserve that layout: insert only row 42 after row 41, do not backfill JSON,
renumber entries, alter row 41, or recalculate unrelated frontmatter counts.

Source audit: GOAL items map to Task 1 (transition-safe test, focused verification,
atomic test commit) and Task 2 (new row 42 with complete CI/family evidence and the
actual fix hash). REQ QUICK-UXW is the local identifier for the orchestrator-
supplied quick requirement; no ROADMAP requirement ID was supplied. RESEARCH is
absent; this is Level-0 work using the existing _wait_until pattern. CONTEXT locks
a test-only minimal diff, no dnallm/tui changes, no sweep beyond the evidenced
instance, targeted lanes only, at most two commits, conventional English commit
messages, and no co-author trailers. No deferred idea or other source item is
included; no source item is unplanned.
</context>

<tasks>

<task type="tracer">
  <name>Task 1: Wait for ConfigScreen after the edit-existing-config click</name>
  <files>tests/tui/test_screens.py</files>
  <action>
    Edit only test_edit_an_existing_config_opens_it_through_the_core in
    tests/tui/test_screens.py. Immediately after the existing pilot.pause(),
    replace the one-time active snapshot with active = await _wait_until(lambda:
    isinstance(app.screen, ConfigScreen)). Keep the subsequent _settle_config
    call, the existing assert active, and all three fake.edit_calls assertions
    unchanged. This mirrors commit 166dda2's bounded boolean-poll idiom, yields
    to the event loop during the asynchronous transition, and fails honestly if
    ConfigScreen never becomes active within the existing approximately-five-
    second bound.

    Leave test_edit_an_existing_config_with_an_empty_path_prompts byte-for-byte
    unchanged for the reason recorded in context. Do not alter _wait_until or
    _settle_config, add a fixed sleep, broaden the timeout, add timing hooks, or
    modify any other test. Do not touch anything under dnallm/tui; production
    code is reference-only for this task.

    Run the single target, the complete test_screens.py file, and the complete
    tests/tui lane from the checkout root. No repo-wide pytest lane is permitted.
    Run Ruff check and format check only on the touched test file, plus the
    whitespace and no-production-diff gates below. Commit this file alone,
    suggested subject `test(tui): wait for config screen transition`, with no
    co-author trailer.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/tui/test_screens.py::test_edit_an_existing_config_opens_it_through_the_core -q && uv run --no-sync pytest tests/tui/test_screens.py -q && uv run --no-sync pytest tests/tui -q && uv run --no-sync ruff check tests/tui/test_screens.py && uv run --no-sync ruff format --check tests/tui/test_screens.py && git diff --check -- tests/tui/test_screens.py && git diff --exit-code -- dnallm/tui</automated>
  </verify>
  <done>The target polls for ConfigScreen through the existing bounded helper before _settle_config; the active assertion and all three edit-call assertions remain; the neighboring empty-path test and every other test stay unchanged; the target, test file, and complete TUI lane pass; Ruff and whitespace checks pass; dnallm/tui has no diff; and the test change is one atomic conventional commit.</done>
  <precondition>Run from the checkout root on branch v1.4 with the existing uv environment; no other process is modifying the scoped test file or TUI sources during the lane.</precondition>
</task>

<task type="auto">
  <name>Task 2: Add WINDOWS row 42 with the fix and family evidence</name>
  <files>.planning/WINDOWS.md</files>
  <action>
    Confirm Task 1 is committed and its focused lanes are green, then obtain the
    actual short fix hash with git log -1 --format=%h --
    tests/tui/test_screens.py; never invent a hash or use the later ledger
    commit hash. Capture the current UTC timestamp for the new row's recorded_at
    and resolved_at cells.

    In .planning/WINDOWS.md, insert exactly one new pipe-table row with id 42
    immediately after row 41. Use phase ci-triage-261030, kind flake (fixed), file
    tests/tui/test_screens.py, blank line cell, and status fixed. In the
    description, record that Windows-leg CI run 41065033616 (push b7d217d,
    2026-10-30 13:39Z) failed test_edit_an_existing_config_opens_it_through_the_core
    1/3606 with bare assert False at test_screens.py:1670 because the ConfigScreen
    snapshot followed the click and only one pause. Include the prior family data:
    run 41059324819 failed the row-41 skip-path test, fixed by 166dda2, while
    identical code passed run 41061141375 and the 41059324819 rerun.

    End the description with this family note, using ASCII arrows inside the
    table: click -> single pause -> immediate screen-state assertion is
    timing-fragile on slow runners; two instances CI-evidenced (rows 41-42), both
    hardened with bounded _wait_until; other unaudited instances may exist and
    get hardened on first CI evidence.

    Populate reason with the actual Task-1 short hash and commit subject, state
    that the target now waits for ConfigScreen through bounded _wait_until before
    _settle_config while preserving the edit-call assertions, and cite the green
    target, full test_screens.py, and complete tests/tui lanes. Keep every other
    row, the embedded JSON, and frontmatter counts unchanged. Commit the ledger
    separately, suggested subject `docs(windows): record config screen transition
    flake fixed`, in English with no co-author trailer.
  </action>
  <verify>
    <automated>uv run --no-sync python -I -c 'import subprocess; from pathlib import Path; path=Path(".planning/WINDOWS.md"); after=path.read_text(encoding="utf-8").splitlines(); rows=[line for line in after if line.startswith("| 42 |")]; assert len(rows)==1; cells=[cell.strip() for cell in rows[0].strip("|").split("|")]; assert len(cells)==10; fix=subprocess.check_output(["git", "log", "-1", "--format=%h", "--", "tests/tui/test_screens.py"], text=True).strip(); assert cells[0]=="42"; assert cells[1]=="ci-triage-261030"; assert cells[2]=="flake (fixed)"; assert cells[3]=="tests/tui/test_screens.py"; assert cells[6]=="fixed"; assert fix in cells[7]; assert all(token in cells[5] for token in ("41065033616", "b7d217d", "1/3606", "41059324819", "41061141375", "rows 41-42", "_wait_until", "first CI evidence")); assert all(token in cells[7] for token in ("_wait_until", "tests/tui")); assert cells[8] and cells[9]; before=subprocess.check_output(["git", "show", "HEAD:.planning/WINDOWS.md"], text=True).splitlines(); index=after.index(rows[0]); assert after[index-1].startswith("| 41 |"); assert len(after)==len(before)+1; assert after[:index]==before[:index]; assert after[index+1:]==before[index:]; print(rows[0])' && git diff --check -- .planning/WINDOWS.md && git diff --exit-code -- tests/tui/test_screens.py</automated>
  </verify>
  <done>Row 42 alone is inserted immediately after row 41; it carries flake (fixed), fixed status, nonempty timestamps, the full CI evidence chain and required family note, plus the real test-fix hash and focused-lane evidence; all prior rows, JSON entries, and frontmatter counts remain intact; and the ledger is committed separately from the test fix.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| asynchronous Textual transition -> test assertion | a test can observe the previous screen before ConfigScreen becomes active |
| local fix evidence -> persistent ledger | an incorrect hash or incomplete CI/family chain would make the resolution unauditable |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-uxw-01 | Denial of Service | ConfigScreen readiness wait | low | mitigate | Reuse the existing approximately-five-second _wait_until bound and assert its boolean result; never add an unbounded wait or fixed sleep. |
| T-uxw-02 | Tampering | WINDOWS row-42 registration | low | mitigate | Derive the short hash from committed test-file history and gate the sole inserted row on position, kind, status, hash, CI run IDs, family wording, timestamps, and byte-preservation of every prior line. |
| T-uxw-SC | Tampering | npm/pip/cargo installs | low | accept | Reserved supply-chain row: this plan installs no packages and adds no dependency, using only the existing project environment. |
</threat_model>

<verification>
Task 1 is the end-to-end tracer through the real entry-from-config action,
ConfigScreen transition, subsequent screen settling, and recorded core-call
assertions. Its automated gate runs the single regression target, complete test
file, full targeted TUI lane, scoped Ruff checks, whitespace validation, and a
no-production-TUI-diff check. A locally passing pre-change test is expected for
a nondeterministic flake; CI run 41065033616 is the RED evidence and must not be
recreated through production timing changes.

Task 2 parses the single row-42 table entry and verifies its position, fixed
fields, actual Task-1 short hash, complete CI and family evidence, timestamps,
and insertion-only diff against HEAD. The two work products remain separate
atomic commits. No network access, model download, dependency install cache
cleanup, repo-wide test lane, unrelated test hardening, or ledger migration is
permitted.
</verification>

<success_criteria>
- The observed edit-existing-config assertion tolerates the asynchronous ConfigScreen transition through the existing bounded polling helper and still proves the original edit_config arguments.
- The neighboring empty-path test and all other unevidenced instances remain unchanged; no file under dnallm/tui changes.
- tests/tui/test_screens.py and the complete tests/tui lane pass; scoped Ruff check and format check pass.
- WINDOWS row 42 alone is added after row 41 with flake (fixed) / fixed, the real fix hash, timestamps, the full 41065033616 evidence and prior-run context, and the required rows 41-42 family note.
- Work lands as exactly two atomic conventional commits, one for the test and one for the ledger, with no co-author trailers.
</success_criteria>

<output>
Create `.planning/quick/261030-uxw-harden-second-tui-screen-transition-race/261030-uxw-SUMMARY.md` when done. Record the single-target, test-file, and full-TUI lane results, scoped Ruff/format results, confirmation that dnallm/tui and all other tests stayed unchanged, both actual commit hashes, and the final row-42 text with its CI evidence and family note.
</output>
