---
phase: 261028-vgn
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/tui/workers.py
  - tests/tui/test_workers.py
autonomous: true
must_haves:
  truths:
    - "The isolated pinned ty@0.0.84 check of dnallm/tui/workers.py reports Found 0 diagnostics (planning baseline: 18 — 17 unresolved post_message attributes plus one unsupported tqdm base)"
    - "Every pre-flight worker still delivers its typed messages through the real Textual App/MessagePump route; no worker body, cancellation guard, error copy, or download daemon-thread behavior changes"
    - "The byte-progress reporter still emits increasing done/total/label updates and is bound only when the downloader explicitly accepts tqdm_class"
    - "Both source fixes ship with red-to-green pytest contract coverage in the same change and use no suppression pragmas, severity overrides, or ty exclusion configuration"
  artifacts:
    - "dnallm/tui/workers.py — explicit MessagePump host contract and concrete tqdm.std base"
    - "tests/tui/test_workers.py — host-contract and concrete-progress-base regression tests alongside the existing functional seam tests"
  key_links:
    - "PreflightWorkers -> MessagePump.post_message -> DNALLMPreflightApp/App -> typed TUI messages"
    - "tqdm.std.tqdm -> _ProgressReporter.update -> _bind_progress -> hub byte callback -> DownloadProgress"
---

<objective>
Clear all 18 pinned ty@0.0.84 diagnostics in `dnallm/tui/workers.py` with two
root-cause fixes and no behavior-changing rewrite.

Purpose: restore an honest-green ty hard gate without suppressing useful type
checks. The live planning baseline is exactly 18 diagnostics: 17
`error[unresolved-attribute]` sites for the Textual message host method and one
`warning[unsupported-base]` for the dynamic tqdm base. (The item text says 16
attribute errors; the authoritative `/tmp/ty-diagnostics.txt` and isolated rerun
both contain 17, so completion is judged by zero diagnostics, not by that count.)

Output: a typed worker mixin, a concrete tqdm progress base, and focused TUI
regression tests proving the existing runtime contracts remain intact.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@dnallm/tui/workers.py
@dnallm/tui/app.py
@tests/tui/test_workers.py
@tests/tui/test_structure.py

Authoritative diagnostic inventory: `/tmp/ty-diagnostics.txt`.
Pinned type environment: `/tmp/ty-stub-env` (already built by the coordinator).
Scope is only the two files in `files_modified`; sibling quick items own every
other diagnostic path. No dependency on those siblings is required.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Declare the Textual message-host contract and pin it with a regression test</name>
  <files>dnallm/tui/workers.py, tests/tui/test_workers.py</files>
  <behavior>
    - RED first: a new host-contract test asserts `PreflightWorkers` is a `MessagePump` subclass; it fails on the current mixin, which relies on the later App base implicitly.
    - GREEN after the source fix: the same assertion passes, and the existing `_Harness(PreflightWorkers, App[None])` round-trip, failure, cancellation, and download tests still deliver their typed messages.
    - Method resolution remains `PreflightWorkers -> App -> DOMNode -> MessagePump`; the mixin adds no initializer, state, or replacement message implementation.
  </behavior>
  <action>
    In `tests/tui/test_workers.py`, import `MessagePump` from
    `textual.message_pump` and add a behavior-named synchronous test beside the
    `_Harness` definition asserting that `PreflightWorkers` satisfies the
    MessagePump host contract. Run that single test before editing source and
    record its failure as the RED proof.

    In `dnallm/tui/workers.py`, import `MessagePump` beside the existing Textual
    imports and change the `PreflightWorkers` declaration to inherit
    `MessagePump`. Add one short class-docstring sentence explaining that this is
    the explicit host interface supplied by the eventual Textual App; do not add
    a forwarding method, a protocol stub, a runtime conditional class, or any
    suppression comment. Leave every worker body and all message payloads
    byte-identical. This inheritance shape was MRO-probed at planning time with
    the pinned Textual runtime and resolves to the real MessagePump method.

    Run the new test and the complete worker seam file. The existing functional
    tests are the behavior guard for all 17 formerly diagnosed call sites,
    including the download opener and nested daemon-thread post helper.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/tui/test_workers.py -q</automated>
  </verify>
  <done>The new host-contract test has recorded RED-to-GREEN evidence; all existing worker seam tests pass; `PreflightWorkers` now exposes the real Textual message interface without changing any worker body.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Use the concrete tqdm base, prove progress behavior, and run the type/lint gates</name>
  <files>dnallm/tui/workers.py, tests/tui/test_workers.py</files>
  <behavior>
    - RED first: a new progress-reporter contract test requires the generated reporter's direct base to be `tqdm.std.tqdm`; the current `tqdm.auto` selection resolves to `tqdm_asyncio` in the pinned environment and fails this assertion.
    - GREEN after the source fix: the reporter directly subclasses the concrete std implementation, while the existing byte-reporting test still observes done values `[1, 2]`, total `4`, and label `bytes`.
    - The isolated pinned ty check of `dnallm/tui/workers.py` finishes with `Found 0 diagnostics` and exit status 0.
  </behavior>
  <action>
    In `tests/tui/test_workers.py`, import the private
    `_make_progress_reporter` helper and `tqdm.std.tqdm` under a distinguishing
    alias. Add a synchronous test that builds a reporter with a no-op emit
    callback, asserts it was created, and asserts its direct base tuple is
    exactly the concrete std tqdm class. Run it before editing source and record
    the RED result. Keep the existing end-to-end byte progress test unchanged;
    it supplies the runtime behavior assertion rather than replacing it with an
    implementation-only check.

    In `_make_progress_reporter`, change only the function-local tqdm import
    from `tqdm.auto` to `tqdm.std`, retaining the local import, lazy optional
    fallback, reporter class name, update logic, and return contract. Extend the
    helper docstring with one sentence: the TUI forwards counts rather than
    selecting a notebook/async frontend, so the concrete std implementation is
    the intentional base. Do not add a cast, `Any` base, ignore pragma, or new
    dependency.

    Run the focused TUI lanes, the isolated pinned ty check, and ruff over both
    touched files. Do not run repo-wide pytest lanes or modify any sibling-owned
    source file. Finally run the authoritative whole-tree ty command from
    `<verification>`; before sibling merges it may still report their paths,
    but it must contain no entry for `dnallm/tui/workers.py`.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/tui/test_workers.py tests/tui/test_structure.py -q && CPATH=/home/linuxbrew/.linuxbrew/include LIBRARY_PATH=/home/linuxbrew/.linuxbrew/lib uvx ty@0.0.84 check dnallm/tui/workers.py --python /tmp/ty-stub-env --output-format concise && .venv/bin/ruff format --check dnallm/tui/workers.py tests/tui/test_workers.py && .venv/bin/ruff check dnallm/tui/workers.py tests/tui/test_workers.py</automated>
  </verify>
  <done>Both new regression tests have RED-to-GREEN evidence; worker and structure lanes pass; isolated ty reports zero diagnostics; ruff format/check pass; the whole-tree ty output contains no workers.py diagnostic.</done>
  <precondition>The coordinator-provided `/tmp/ty-stub-env` and project `.venv` exist and contain the pinned type/test dependencies; halt rather than silently substituting a different Python or ty version.</precondition>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| worker/daemon thread -> Textual event loop | typed messages cross through the host's real MessagePump implementation |
| hub tqdm callback -> TUI progress message | third-party byte counts enter the reporter and are forwarded as typed progress payloads |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-vgn-01 | Tampering | PreflightWorkers/MessagePump MRO | low | mitigate | Add no replacement message method or initializer; retain the real App/MessagePump route and prove round-trip, failure, cancellation, and download delivery with the existing worker seam suite plus the new host-contract test. |
| T-vgn-02 | Denial of Service | _ProgressReporter.update | low | mitigate | Keep update and callback logic unchanged; select only the concrete std base and retain the functional increasing-progress test so type cleanup cannot disable progress reporting or introduce a new frontend path. |
| T-vgn-SC | Tampering | package installs | low | accept | Reserved row: this plan installs no npm/pip/cargo packages and adds no dependencies; it uses the coordinator's existing pinned environment. |
</threat_model>

<verification>
Item acceptance command (must exit 0 and print `Found 0 diagnostics`):

`CPATH=/home/linuxbrew/.linuxbrew/include LIBRARY_PATH=/home/linuxbrew/.linuxbrew/lib uvx ty@0.0.84 check dnallm/tui/workers.py --python /tmp/ty-stub-env --output-format concise`

Authoritative batch/CI command (run verbatim):

`CPATH=/home/linuxbrew/.linuxbrew/include LIBRARY_PATH=/home/linuxbrew/.linuxbrew/lib uvx ty@0.0.84 check dnallm/ --python /tmp/ty-stub-env`

For this isolated item, success is zero workers.py entries in that whole-tree
output; sibling items own the remaining paths. After all sibling fixes merge,
the authoritative command must print `Found 0 diagnostics`. Also retain green
results for the two targeted pytest files and ruff format/check on both touched
files. No repo-wide test lane is part of executor verification.
</verification>

<success_criteria>
- All 18 original workers.py diagnostics are gone under ty@0.0.84 with the pinned stub environment.
- The fix uses real type declarations only: no line suppression, severity override, `[tool.ty]` exclusion, or configuration change.
- Textual message delivery, cancellation/failure handling, daemon download semantics, and byte/attempt progress behavior remain covered and green.
- `tests/tui/test_workers.py` changes land with the source changes, satisfying the owner's same-change pytest rule.
- Only `dnallm/tui/workers.py` and `tests/tui/test_workers.py` are modified.
</success_criteria>

<output>
Create `.planning/quick/261028-vgn-fix-all-18-ty-0-0-84-diagnostics-in-dnallm-tui-workers-py-16/261028-vgn-SUMMARY.md` when done, recording the RED-to-GREEN tests, isolated zero-diagnostic result, targeted pytest/ruff results, and any remaining whole-tree diagnostics by sibling owner.
</output>
