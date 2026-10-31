---
phase: 261030-sga
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - tests/conftest.py
  - tests/utils/test_logger.py
  - tests/mcp/test_timeout.py
  - pyproject.toml
  - .planning/milestones/v1-phases/03-test-authoring-to-90-coverage/deferred-items.md
  - .planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md
autonomous: true
requirements:
  - QUICK-SGA
must_haves:
  truths:
    - "Pytest collection imports dnallm while its launch cwd is unchanged afterward, and the default DNALLM file sink lives under a pytest-owned temporary directory rather than the checkout's logs directory"
    - "A pre-existing logs/dnallm.log is never deleted, truncated, renamed, or written by the targeted pytest lane; its directory existence, size, mtime, and SHA-256 remain unchanged"
    - "All existing logger behavior tests still pass, including fresh loggers creating their normal cwd-relative sink under tmp_path"
    - "The two MCP timeout tests preserve their timeout-error and response-structure assertions but complete in short-timeout form, with no approximately 30-second duration"
    - "The base numpy requirement is exactly numpy>=2.0.0,<2.6, the existing 1.x-retirement comment survives, and an adjacent English comment records the coverage incompatibility and lifting condition"
    - "Only the phase-03 logger and timeout bullets and the phase-12 numpy bullet are resolved, each with dated verification evidence"
  artifacts:
    - "tests/conftest.py — import-time temporary-cwd logger priming plus session-autouse cleanup of only the redirected handlers and temporary directory"
    - "tests/utils/test_logger.py — regression assertion that the import-time default dnallm.log handler is outside the checkout"
    - "tests/mcp/test_timeout.py — both long waits converted to the established 0.1-second timeout pattern with matching assertions"
    - "pyproject.toml — numpy<2.6 ceiling and coverage-compatibility rationale"
    - ".planning/milestones/v1-phases/03-test-authoring-to-90-coverage/deferred-items.md — dated logger and timeout resolutions only"
    - ".planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md — dated numpy resolution only"
  key_links:
    - "root conftest import -> temporary cwd -> first dnallm import and get_logger singleton creation -> immediate cwd restoration -> collection-time module imports reuse the redirected singleton"
    - "session-autouse teardown -> close/remove only FileHandlers under the pytest temporary directory -> TemporaryDirectory cleanup; no checkout logs cleanup"
    - "configured 0.1-second server timeout -> existing _with_timeout_wrapper -> timeout error payload carrying timeout_seconds=0.1"
    - "numpy requirement + CI 2.2.0 pin + coverage rationale -> dependency-policy deferred-item closure"
---

<objective>
Eliminate pytest's checkout-local DNALLM log side effect without changing library
behavior, remove the two fixed 30-second MCP timeout waits, and add the approved
numpy ceiling. Close exactly the three corresponding archived deferred items
after their focused checks pass.

Purpose: improve suite hygiene and feedback time while making fresh dependency
resolves avoid the observed numpy/coverage failure. The logger intervention must
run before test-module imports and must preserve any pre-existing checkout logs.

Output: three focused code/config changes, targeted regression evidence, and
three dated deferred-item resolutions in atomic conventional commits with no
co-author trailers.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@dnallm/utils/logger.py
@tests/conftest.py
@tests/utils/test_logger.py
@tests/mcp/test_timeout.py
@pyproject.toml
@.planning/milestones/v1-phases/03-test-authoring-to-90-coverage/deferred-items.md
@.planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md

Planning facts on branch v1.4: `dnallm/utils/logger.py:57-60` creates the
cwd-relative directory and file only while constructing the first handler-free
logger; `get_logger` then reuses its singleton. The root conftest is imported
before test modules and currently has no module-level dnallm import, so it can
control the cwd of that first import. Logger tests that intentionally construct
fresh loggers already chdir to tmp_path, and no other test resets
`_logger_instance`. The checkout already has `logs/dnallm.log`; absence is not a
valid gate and no cleanup of that file or directory is permitted.

Source audit: GOAL items map to Task 1 (test-side import-time logger isolation),
Task 2 (two short-timeout rewrites), and Task 3 (numpy ceiling plus exactly three
deferred closures). REQ QUICK-SGA is the local identifier for the
orchestrator-supplied quick requirement; no ROADMAP requirement ID was supplied.
RESEARCH is absent; this is Level-0 work using the existing singleton, pytest
session fixture, and `test_timeout_configurable` patterns. CONTEXT contains the
locked owner decisions of 2026-10-30: no dnallm library edit, no server edit, no
lockfile regeneration, no pre-existing logs deletion, targeted lanes only, and
no repo-wide pytest. The phase-03 reverse-complement bullet, phase-12 mkdocs
bullet, and phase-12 HBG1 bullet are explicit exclusions and remain untouched.
No source item is unplanned.
</context>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: Prime the import-time logger in a pytest-owned temporary cwd</name>
  <files>tests/conftest.py, tests/utils/test_logger.py</files>
  <behavior>
    - Before the conftest intervention, the new regression test run from the checkout root fails because the default dnallm.log FileHandler resolves to the checkout's logs/dnallm.log.
    - After the intervention, collection-time dnallm imports reuse a default singleton whose dnallm.log FileHandler is outside the checkout, while the process cwd is restored before any test module is collected.
    - Existing fresh-logger tests still observe the library's normal cwd-relative logs/dnallm.log behavior under their own tmp_path.
    - Session teardown closes and removes only FileHandlers whose resolved baseFilename is under the pytest temporary directory, then cleans that directory; it never removes or truncates checkout logs.
  </behavior>
  <action>
    Start RED in tests/utils/test_logger.py. Add a pathlib Path import and, at
    the start of TestGetLogger before the existing singleton test, add
    `test_import_time_file_sink_is_outside_checkout`. It must call the default
    `get_logger()`, select its FileHandlers whose baseFilename basename is
    `dnallm.log`, require at least one such handler (so removing the file sink
    cannot make the test vacuously green), and assert none resolves to
    `Path(__file__).resolve().parents[2] / "logs" / "dnallm.log"`. Run this one
    test from the checkout root before changing conftest and record the intended
    RED failure. Do not reset the singleton in this regression test.

    In tests/conftest.py, implement import-time priming before any dnallm import
    and before test collection. Add the small stdlib imports needed for logging,
    cwd management, paths, and a TemporaryDirectory. Keep the TemporaryDirectory
    and the actual primed stdlib logger object (`get_logger().logger`) in
    module-global references for the whole pytest process. Save `Path.cwd()`,
    chdir into that temporary directory, import `dnallm.utils.logger` through a
    function-local import or importlib (avoiding an E402 module-level late
    import), call `get_logger()` to force construction of the default singleton,
    save its `.logger` object, and restore the saved cwd in `finally`. The
    temporary cwd must cover both the import and explicit get_logger call because
    dnallm's eager CLI import can construct the singleton during the import
    itself; that also means the actual logger name can be `dnallm.cli`, so do not
    assume it is `dnallm`. Do not monkeypatch pathlib, logging, DNALLMLogger, or
    any library module, and do not edit dnallm/utils/logger.py.

    Convert the existing session-scoped autouse `global_cleanup` fixture from an
    immediate return to yield-based teardown. After the yield, iterate over a
    copy of the saved primed stdlib logger's handlers; for each FileHandler whose
    resolved baseFilename is inside the module-global TemporaryDirectory, close
    it and remove it from that same logger object. Then call
    TemporaryDirectory.cleanup(). This explicit close is required for Windows as
    well as POSIX. Do not look up a logger by a hard-coded name, close unrelated
    handlers, chdir at teardown, or recursively inspect or remove the checkout
    logs path. Add concise English comments explaining that priming must precede
    test module imports and why the launch cwd is restored immediately.

    Run the regression test and then the complete tests/utils/test_logger.py
    lane to GREEN. Before the final full-lane run, snapshot whether logs exists
    and, if logs/dnallm.log exists, its size, mtime, and SHA-256; afterward use
    the exact automated comparison below. The snapshot is evidence, not a
    cleanup license: if it differs, stop and diagnose rather than deleting or
    restoring user data. Run Ruff only on the two touched Python files. Commit
    both files together, suggested subject `test(logging): isolate pytest log
    sink`, with no co-author trailer.
  </action>
  <verify>
    <automated>before="$(if [ -f logs/dnallm.log ]; then stat -c 'present:%s:%Y' logs/dnallm.log; sha256sum logs/dnallm.log; elif [ -d logs ]; then printf 'dir-without-file\n'; else printf 'absent\n'; fi)"; .venv/bin/python -m pytest tests/utils/test_logger.py -q --no-cov && after="$(if [ -f logs/dnallm.log ]; then stat -c 'present:%s:%Y' logs/dnallm.log; sha256sum logs/dnallm.log; elif [ -d logs ]; then printf 'dir-without-file\n'; else printf 'absent\n'; fi)" && test "$before" = "$after" && printf 'checkout log snapshot unchanged:\n%s\n' "$after" && .venv/bin/ruff check tests/conftest.py tests/utils/test_logger.py && .venv/bin/ruff format --check tests/conftest.py tests/utils/test_logger.py && git diff --check -- tests/conftest.py tests/utils/test_logger.py && git diff --exit-code -- dnallm/utils/logger.py</automated>
  </verify>
  <done>The new regression fails before and passes after the conftest change; the complete logger lane passes; the checkout logs snapshot is byte- and mtime/size-identical (or remains absent); cwd is restored during collection; only pytest-owned handlers/temp files are cleaned; Ruff, whitespace, and no-library-diff checks pass; and the mechanism plus regression test are committed atomically.</done>
  <precondition>Run from the checkout root on branch v1.4 with the existing .venv; no other process is writing the scoped test files or checkout logs during the before/after verification lane.</precondition>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Convert the two full-timeout tests to the established short-timeout pattern</name>
  <files>tests/mcp/test_timeout.py</files>
  <behavior>
    - test_tool_timeout_returns_error still returns isError=True, error_type=timeout, tool_name=slow_tool, a suggestion, and timeout text, but the configured and reported timeout is 0.1 seconds.
    - test_timeout_error_structure still checks every required field, content item shape, suggestion, and tool name, but accepts the configured numeric timeout value and requires timeout_seconds=0.1.
    - Both slow coroutines still sleep longer than their configured timeout, while the complete timeout lane finishes within a hard 20-second bound and reports no approximately 30-second call.
  </behavior>
  <action>
    Edit only the two named tests in tests/mcp/test_timeout.py. In
    `test_tool_timeout_returns_error`, set `mock_server._tool_timeout_seconds =
    0.1` before wrapping the coroutine, reduce `asyncio.sleep(100)` to
    `asyncio.sleep(1)`, and update the timeout_seconds assertion and timeout
    text expectation from 30 to 0.1. Preserve every other assertion and the
    wrapper call.

    In `test_timeout_error_structure`, set the same instance timeout to 0.1
    before wrapping, reduce `asyncio.sleep(1000)` to `asyncio.sleep(1)`, replace
    the integer-only type assertion with a numeric int-or-float assertion, and
    require timeout_seconds == 0.1. Preserve the complete required-field,
    boolean/error type, tool name, suggestion, content-list, text-item, and
    `Timeout after` assertions. Do not change the shared mock fixture,
    test_timeout_configurable, streaming tests, or any file under dnallm/; the
    server implementation is reference-only.

    Run the entire tests/mcp/test_timeout.py lane with `--durations=0` under the
    hard 20-second process bound below. Inspect the durations output: neither
    rewritten test may retain an approximately 30-second call. Record the actual
    lane result and slowest durations for the later deferred-item evidence. Run
    Ruff on this file only and commit separately, suggested subject `test(mcp):
    shorten timeout test waits`, with no co-author trailer.
  </action>
  <verify>
    <automated>timeout 20s .venv/bin/python -m pytest tests/mcp/test_timeout.py -q --no-cov --durations=0 && .venv/bin/ruff check tests/mcp/test_timeout.py && .venv/bin/ruff format --check tests/mcp/test_timeout.py && git diff --check -- tests/mcp/test_timeout.py && git diff --exit-code -- dnallm/mcp/server.py</automated>
  </verify>
  <done>The complete timeout lane passes within 20 seconds, durations show no approximately 30-second item, both tests retain their original assertion intent with timeout_seconds=0.1, server code is unchanged, and the focused test edit is committed separately.</done>
</task>

<task type="auto">
  <name>Task 3: Cap numpy and close only the three verified deferred items</name>
  <files>pyproject.toml, .planning/milestones/v1-phases/03-test-authoring-to-90-coverage/deferred-items.md, .planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md</files>
  <action>
    In pyproject.toml, keep the existing comment `# numpy 1.x retired
    2026-10-10; 1.x users stay on the 0.8.x series` unchanged and change only
    the base numpy requirement to `numpy>=2.0.0,<2.6`. Add an adjacent English
    comment recording all of these facts: numpy 2.5.x plus coverage 7.16.x on
    Python 3.13 fails every instrumented run at conftest import with
    `ImportError: cannot load module more than once per process` from numpy's
    double-init guard; the CI matrix pins numpy 2.2.0 and the 1.26.4 leg was
    retired in 2026-10; lift the <2.6 ceiling only after coverage compatibility
    with numpy 2.6+ is verified. Do not alter other dependency bounds, create
    or regenerate a lockfile, install packages, or run a resolver/network lane.
    Parse pyproject and verify the 2.2.0 CI pin is accepted while 2.6.0 is not,
    then commit this file alone, suggested subject `chore(deps): cap numpy below
    2.6`, with no co-author trailer.

    After Tasks 1 and 2 are green and committed and the numpy commit exists,
    collect their actual short hashes with `git log -3 --format='%h %s'`; never
    invent or reuse a hash. In the phase-03 deferred-items file, edit only the
    logger-sink and test_timeout bullets. For each, change lowercase
    `status: acknowledged` to `status: resolved` and bold `**Status:** open` to
    `**Status:** resolved`; retain the original **What:** as historical context
    and append a dated **Resolved:** entry. Logger evidence must cite the Task-1
    hash, the new regression test, temporary-cwd import-time priming with
    session-autouse cleanup, the passing logger lane, and the unchanged checkout
    log snapshot (include its observed SHA-256 when the file pre-exists).
    Timeout evidence must cite the Task-2 hash, the two 0.1-second conversions,
    the passing full timeout-file lane, the 20-second bound, and the measured
    durations showing no approximately 30-second item.

    In the phase-12 deferred-items file, edit only the numpy bullet: change its
    lowercase status to resolved and append a dated resolution citing the actual
    numpy commit hash, the exact `numpy>=2.0.0,<2.6` requirement, the adjacent
    coverage/double-init comment, and the parse/specifier evidence that numpy
    2.2.0 remains allowed. Preserve all existing bullets and their formatting;
    do not introduce tables. Explicitly leave the phase-03 reverse-complement
    bullet, phase-12 mkdocs resolution, and phase-12 HBG1 owner deferral
    byte-untouched.

    Run the targeted documentation checks below, then commit the two deferred
    files together separately, suggested subject `docs(quick-261030-sga): close
    verified deferred items`, with no co-author trailer.
  </action>
  <verify>
    <automated>.venv/bin/python -c 'import tomllib; from packaging.specifiers import SpecifierSet; data=tomllib.load(open("pyproject.toml", "rb")); req=next(item for item in data["project"]["dependencies"] if item.startswith("numpy")); assert req == "numpy>=2.0.0,<2.6"; spec=SpecifierSet(req.removeprefix("numpy")); assert spec.contains("2.2.0"); assert not spec.contains("2.6.0")' && grep -F '# numpy 1.x retired 2026-10-10; 1.x users stay on the 0.8.x series' pyproject.toml && grep -F 'coverage compatibility' pyproject.toml && grep -A 18 -F "dnallm's import-time file sink" .planning/milestones/v1-phases/03-test-authoring-to-90-coverage/deferred-items.md | grep -F 'status: resolved' && grep -A 18 -F "dnallm's import-time file sink" .planning/milestones/v1-phases/03-test-authoring-to-90-coverage/deferred-items.md | grep -F '**Resolved:** 2026-10-30' && grep -A 18 -F 'test_timeout.py spends' .planning/milestones/v1-phases/03-test-authoring-to-90-coverage/deferred-items.md | grep -F 'status: resolved' && grep -A 18 -F 'test_timeout.py spends' .planning/milestones/v1-phases/03-test-authoring-to-90-coverage/deferred-items.md | grep -F '**Resolved:** 2026-10-30' && grep -A 12 -F 'DNADataset.raw_reverse_complement is a no-op' .planning/milestones/v1-phases/03-test-authoring-to-90-coverage/deferred-items.md | grep -F '5c95dc5' && grep -A 26 -F 'numpy 2.5.x cannot be instrumented' .planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md | grep -F 'status: resolved' && grep -A 26 -F 'numpy 2.5.x cannot be instrumented' .planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md | grep -F '2026-10-30' && grep -A 24 -F 'HBG1/BCL11A golden fixture awaits manuscript inputs' .planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md | grep -F 'status: acknowledged' && grep -A 10 -F 'Fresh verification at HEAD 8b224ea' .planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md | grep -F 'zero WARNING lines' && git diff --check -- pyproject.toml .planning/milestones/v1-phases/03-test-authoring-to-90-coverage/deferred-items.md .planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md</automated>
  </verify>
  <done>The exact numpy ceiling and complete rationale are committed without a lockfile or resolver run; all three targeted bullets alone carry dated resolved statuses and real code-commit/verification evidence; reverse-complement, mkdocs, and HBG1 content remains intact; and the deferred-item updates are in a separate docs commit.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| pytest process -> checkout and temporary filesystem | test initialization controls where persistent log artifacts are created and which handlers are closed |
| dependency declaration -> fresh environment resolver | the numpy requirement determines which third-party version enters instrumented test runs |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-sga-01 | Tampering | conftest logger priming and session cleanup | medium | mitigate | Restrict redirection to a pytest-owned TemporaryDirectory, restore cwd in finally, close only handlers beneath that directory, and compare the checkout log's existence, size, mtime, and SHA-256 before and after the lane. |
| T-sga-02 | Denial of Service | MCP timeout tests | low | mitigate | Preserve real timeout behavior with a 0.1-second configured limit and enforce a 20-second outer bound on the complete focused lane. |
| T-sga-03 | Tampering | numpy dependency resolution | medium | mitigate | Exclude numpy 2.6+ until coverage compatibility is verified, retain the CI-compatible 2.2.0 pin, and test the declared specifier locally without a network resolve. |
| T-sga-SC | Tampering | npm/pip/cargo installs | low | accept | Reserved supply-chain row: this plan installs no packages and adds no dependency, using only the existing project environment. |
</threat_model>

<verification>
Task 1 is the end-to-end tracer through root-conftest loading, dnallm import,
first get_logger construction, handler placement, collection-time singleton
reuse, and session cleanup. Its RED assertion must fail for the checkout-sink
reason before the conftest edit. The final logger-lane check additionally proves
a pre-existing checkout log remains unchanged rather than wrongly requiring
absence.

Task 2 runs the complete current timeout test file with durations and a hard
20-second bound. Task 3 validates TOML and the numpy specifier locally, then
checks exact deferred-item status/evidence anchors and preservation anchors for
the three excluded bullets. Ruff and whitespace checks are limited to touched
Python files as directed.

No repo-wide pytest lane, network access, dependency installation, lockfile
operation, model download, cache cleanup, checkout logs cleanup, or unrelated
refactor is permitted.
</verification>

<success_criteria>
- No pytest collection-time default DNALLM file handler targets the checkout logs/dnallm.log, and any pre-existing checkout log is unchanged in existence, content, size, and mtime.
- The launch cwd is restored immediately after priming; all existing logger behavior tests pass and only pytest-owned temporary handlers/files are cleaned.
- Both long MCP tests use the existing 0.1-second pattern with equivalent assertions; the full timeout lane passes under 20 seconds with no approximately 30-second duration.
- pyproject declares exactly numpy>=2.0.0,<2.6, retains the 1.x-retirement comment, and documents the observed coverage failure, CI 2.2.0 compatibility, and lifting condition.
- Only the logger, timeout, and numpy deferred bullets flip to resolved with dated commit and verification evidence; all explicitly excluded bullets remain untouched.
- No dnallm library or MCP server file changes, no repo-wide tests, no new dependencies or lockfile, and no co-author trailers.
- Work lands as four atomic conventional commits: logger mechanism, timeout tests, numpy policy, then deferred-item evidence.
</success_criteria>

<output>
Create `.planning/quick/261030-sga-test-hygiene-plus-numpy-policy-autouse-c/261030-sga-SUMMARY.md` when done. Record the logger RED/GREEN evidence, before/after checkout-log snapshot and SHA-256, logger and timeout lane results with durations, numpy parse/specifier result, scoped Ruff/format results, all four commit hashes, and confirmation that no dnallm source, pre-existing logs, lockfile, or excluded deferred bullet was changed.
</output>
