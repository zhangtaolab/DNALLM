---
phase: 261030-uck
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - tests/tui/test_screens.py
  - .planning/WINDOWS.md
autonomous: true
requirements:
  - QUICK-UCK
estimate:
  tokens: 18000
  raw_tokens: 18000
  tasks: 2
  confidence: low
must_haves:
  truths:
    - "The download-skip test waits for the summary screen's single HeaderTitle to mount before collecting Static content, and its no-download and path-order assertions remain unchanged"
    - "The targeted test file and complete tests/tui lane pass without any production TUI source change"
    - "WINDOWS ledger row 41 records both kind flake (fixed) and status fixed, cites the actual test-fix commit, and preserves the complete CI evidence chain"
  artifacts:
    - "tests/tui/test_screens.py — a bounded mount-ready poll in test_skipping_the_download_reaches_the_summary_without_downloading using the file's existing _wait_until helper"
    - ".planning/WINDOWS.md — only the row-41 triage entry flipped to fixed with dated resolution evidence"
  key_links:
    - "download-skip transition -> SummaryScreen -> bounded _wait_until poll -> exactly one HeaderTitle mounted -> _rendered_text(summary)"
    - "test-fix commit -> row-41 reason -> 41059324819 first-run failure -> 41061141375 pass -> 41059324819 rerun success -> mount-safe fix"
---

<objective>
Make the observed Windows-only HeaderTitle mount-timing flake resilient with a
minimal test-only change, prove it on the focused TUI lanes, and close ledger
row 41 with traceable commit and CI evidence.

Purpose: the summary transition can expose Header before its asynchronous
HeaderTitle child has mounted. An immediate render/query can therefore raise
NoMatches on a slow runner even though the screen is correct. Waiting on the
existing bounded poll removes that race without changing production behavior.

Output: one atomic test-fix commit and one ledger-resolution commit, with no
changes under dnallm/tui and no co-author trailers.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@tests/tui/test_screens.py
@.planning/WINDOWS.md

Planning facts on branch v1.4: the target test currently clicks #download-skip,
pauses once, confirms SummaryScreen, and immediately calls _rendered_text. The
file already provides _wait_until, which polls for up to 500 iterations with a
0.01-second asyncio sleep (approximately five seconds). Installed Textual has
neither Pilot.wait_for nor Pilot.wait_for_screens_and_animation; DOMQuery does
support len(). Header composes HeaderTitle asynchronously and later resolves it
with query_one(HeaderTitle). No explicit HeaderTitle query exists elsewhere in
this test file, so no neighboring rewrite is warranted.

Baseline verification at b7d217d: the single target test passed in 0.87 seconds
and the complete targeted tests/tui lane passed 205 tests in 48.04 seconds. The
nondeterministic CI failure is the RED evidence; do not add artificial timing
hooks or production edits to force a local failure.

Ledger row 41 is a table-only triage entry added with row 40 in b7d217d. The
embedded JSON ledger ends at id 39, so the normal windows-fixed command cannot
find id 41. Preserve that existing triage layout: edit only the row-41 table
entry, and do not backfill JSON, renumber entries, or recalculate unrelated
frontmatter counts in this quick task.

Source audit: GOAL items map to Task 1 (mount-safe test, focused verification,
atomic test commit) and Task 2 (row-41 fixed resolution, actual commit/evidence
chain, separate docs commit). REQ QUICK-UCK is the local identifier for the
orchestrator-supplied quick requirement; no ROADMAP requirement ID was supplied.
RESEARCH is absent; this is Level-0 work using the existing _wait_until pattern.
CONTEXT locks a test-only minimal diff, no dnallm/tui changes, targeted lanes
only, at most two commits, conventional English commit messages, and no
co-author trailers. No deferred idea or other source item is included; no source
item is unplanned.
</context>

<tasks>

<task type="tracer">
  <name>Task 1: Wait for HeaderTitle before rendering the skip-path summary</name>
  <files>tests/tui/test_screens.py</files>
  <action>
    Edit only test_skipping_the_download_reaches_the_summary_without_downloading
    in tests/tui/test_screens.py. After the existing SummaryScreen type assertion
    and before text = _rendered_text(summary), use the file's existing
    _wait_until helper to poll until len(summary.query("HeaderTitle")) == 1.
    Store the boolean as header_mounted and assert it with a concise timeout
    message before calling _rendered_text. This reuses the established bounded
    approximately-five-second poll, yields to the event loop during the async
    mount window, and fails honestly if the header never appears.

    Preserve the existing pilot click and pause, downloader-calls assertion,
    skipped-model text, config-path text, and both fixed-order assertions
    byte-for-byte. Do not alter _rendered_text or _wait_until, add a HeaderTitle
    import, introduce a fixed sleep, broaden the helper timeout, or modify any
    other test. Do not touch anything under dnallm/tui; the production screen is
    reference-only for this task.

    Run the target test, the complete test_screens.py file, and the complete
    tests/tui lane from the checkout root. The broader lane is in scope because
    its measured baseline is approximately 48 seconds; no repo-wide pytest lane
    is permitted. Run Ruff check and format check only on the touched test file,
    plus the whitespace and no-production-diff gates below. Commit this file
    alone, suggested subject `test(tui): wait for summary header mount`, with no
    co-author trailer.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/tui/test_screens.py::test_skipping_the_download_reaches_the_summary_without_downloading -q && uv run --no-sync pytest tests/tui/test_screens.py -q && uv run --no-sync pytest tests/tui -q && uv run --no-sync ruff check tests/tui/test_screens.py && uv run --no-sync ruff format --check tests/tui/test_screens.py && git diff --check -- tests/tui/test_screens.py && git diff --exit-code -- dnallm/tui</automated>
  </verify>
  <done>The skip-path test polls for exactly one mounted HeaderTitle through the existing bounded helper before rendering; every original behavioral assertion remains; the target, test file, and 205-test TUI lane pass; Ruff and whitespace checks pass; dnallm/tui has no diff; and the test change is one atomic conventional commit.</done>
  <precondition>Run from the checkout root on branch v1.4 with the existing uv environment; no other process is modifying the scoped test file or TUI sources during the lane.</precondition>
</task>

<task type="auto">
  <name>Task 2: Close WINDOWS row 41 with the measured fix evidence</name>
  <files>.planning/WINDOWS.md</files>
  <action>
    Confirm Task 1 is committed and its focused lanes are green, then obtain the
    actual short fix hash with git log -1 --format=%h --
    tests/tui/test_screens.py; never invent a hash or use the later ledger
    commit hash. Capture the current UTC timestamp for resolved_at.

    In .planning/WINDOWS.md, edit only the pipe-table row whose id is 41. Keep
    its phase, file, description, and recorded_at cells unchanged. Change kind
    from flake (open) to flake (fixed), status from open to fixed, and populate
    resolved_at with the captured UTC timestamp. Replace the reason cell with a
    concise resolution that names the actual fix hash and commit subject, states
    that the test now uses bounded HeaderTitle mount polling, and includes this
    exact evidence order: CI run 41059324819 first-run failure -> CI run
    41061141375 pass -> CI run 41059324819 rerun success -> this fix. Also cite
    the passing test_screens.py and complete tests/tui lanes from Task 1.

    Because row 41 is table-only, do not run the windows-fixed CLI and do not
    add a new JSON object, change another row, or alter frontmatter counts as
    part of this scoped closure. Run the row-integrity check below before
    committing. Commit the ledger separately, suggested subject `docs(windows):
    mark HeaderTitle mount flake fixed`, in English with no co-author trailer.
  </action>
  <verify>
    <automated>uv run --no-sync python -I -c 'import subprocess; from pathlib import Path; rows=[line for line in Path(".planning/WINDOWS.md").read_text(encoding="utf-8").splitlines() if line.startswith("| 41 |")]; assert len(rows)==1; cells=[cell.strip() for cell in rows[0].strip("|").split("|")]; fix=subprocess.check_output(["git", "log", "-1", "--format=%h", "--", "tests/tui/test_screens.py"], text=True).strip(); assert cells[1]=="ci-triage-261030"; assert cells[2]=="flake (fixed)"; assert cells[3]=="tests/tui/test_screens.py"; assert cells[6]=="fixed"; assert fix in cells[7]; assert all(token in cells[7] for token in ("41059324819", "41061141375", "first-run failure", "rerun success", "tests/tui")); assert cells[8]=="2026-10-30T13:36:00Z"; assert cells[9]; print(rows[0])' && git diff --check -- .planning/WINDOWS.md && git diff --exit-code -- tests/tui/test_screens.py</automated>
  </verify>
  <done>Only row 41 changes; it carries flake (fixed), fixed status, a nonempty resolved_at, the real test-fix hash, the ordered CI failure/pass/rerun/fix chain, and focused-lane evidence; its historical description and recorded_at remain intact; and the ledger is committed separately from the test fix.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| asynchronous Textual mount -> test assertion | a test can observe a parent screen before its HeaderTitle child exists |
| local fix evidence -> persistent ledger | an incorrect hash or incomplete CI chain would make the resolution unauditable |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-uck-01 | Denial of Service | HeaderTitle readiness wait | low | mitigate | Reuse the existing approximately-five-second _wait_until bound and assert its boolean result; never add an unbounded wait or fixed sleep. |
| T-uck-02 | Tampering | WINDOWS row-41 resolution | low | mitigate | Derive the short hash from the committed test-file history and gate the exact row on kind, status, hash, CI run IDs, rerun wording, and unchanged recorded_at. |
| T-uck-SC | Tampering | npm/pip/cargo installs | low | accept | Reserved supply-chain row: this plan installs no packages and adds no dependency, using only the existing project environment. |
</threat_model>

<verification>
Task 1 is the end-to-end tracer through the real download-skip transition,
SummaryScreen mounting, HeaderTitle readiness, and rendered-content assertions.
Its automated gate runs the single regression target, the complete test file,
the full targeted TUI lane, scoped Ruff checks, whitespace validation, and a
no-production-TUI-diff check. A locally passing pre-change test is expected for
a nondeterministic flake; the CI failure and subsequent pass/rerun evidence are
already recorded and must not be recreated by changing production timing.

Task 2 parses the single row-41 table entry and verifies both fixed fields, the
actual Task-1 short hash, every required CI evidence element, the unchanged
recorded_at, and a populated resolved_at. The two work products remain separate
atomic commits. No network access, model download, dependency install, cache
cleanup, repo-wide test lane, or unrelated ledger migration is permitted.
</verification>

<success_criteria>
- The observed skip-path query tolerates the asynchronous HeaderTitle mount through the existing bounded polling helper and still proves no download plus the original summary path order.
- tests/tui/test_screens.py and all 205 tests under tests/tui pass; scoped Ruff check and format check pass.
- No file under dnallm/tui changes and no other test is refactored.
- WINDOWS row 41 alone flips to flake (fixed) / fixed with a real fix hash, resolution timestamp, the full 41059324819 -> 41061141375 -> rerun -> fix chain, and local verification evidence.
- Work lands as exactly two atomic conventional commits, one for the test and one for the ledger, with no co-author trailers.
</success_criteria>

<output>
Create `.planning/quick/261030-uck-fix-tui-headertitle-mount-timing-flake-w/261030-uck-SUMMARY.md` when done. Record the single-target, test-file, and full-TUI lane results, scoped Ruff/format results, confirmation that dnallm/tui and all other tests stayed unchanged, both actual commit hashes, and the final row-41 resolution text with its CI evidence chain.
</output>
