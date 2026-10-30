---
phase: 261028-vgq
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/tui/screens/download.py
autonomous: true
must_haves:
  truths:
    - "The isolated pinned ty@0.0.84 check of dnallm/tui/screens/download.py reports Found 0 diagnostics (planning baseline: exactly one invalid-assignment error at line 291)"
    - "DownloadScreen._worker declares the actual DownloadOperation | None contract while start, cooperative cancellation, cleanup, retry, skip, and quit behavior remain unchanged"
    - "The correction is type-level only: DownloadOperation is imported under TYPE_CHECKING with postponed annotations, adding no runtime import, cast, Any fallback, suppression, visual change, or configuration change"
    - "The existing tests/tui/test_screens.py lane remains green in the same change and ruff format/check pass on the touched source file"
  artifacts:
    - "dnallm/tui/screens/download.py — _worker is annotated with the real DownloadOperation handle returned by the app seam"
  key_links:
    - "DNALLMPreflightApp.download_worker -> DownloadOperation -> DownloadScreen._worker -> cooperative cancel()/None cleanup"
    - "TYPE_CHECKING DownloadOperation import + postponed annotations -> static screen/seam contract without a new runtime import"
    - "existing 101-test screen lane -> progress, cancellation, retry, skip, quit, and summary behavior guards"
---

<objective>
Clear the single pinned ty@0.0.84 `error[invalid-assignment]` in
`dnallm/tui/screens/download.py` by declaring the handle type that the download
screen already stores.

Purpose: make the screen's static contract match the working runtime contract.
`DNALLMPreflightApp.download_worker` returns `DownloadOperation`, not Textual's
`Worker[Any]`; the screen only checks for `None`, calls `cancel()`, and clears
the handle, all of which `DownloadOperation | None` expresses directly.

Output: a corrected type-only annotation and import cleanup, with the existing
TUI screen regression lane proving behavior is unchanged.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@dnallm/tui/screens/download.py
@dnallm/tui/workers.py
@tests/tui/test_screens.py

Authoritative diagnostic inventory: `/tmp/ty-diagnostics.txt` (line 9).
Pinned type environment: `/tmp/ty-stub-env` (already built by the coordinator).
Planning baseline: the isolated pinned ty command reported exactly one
diagnostic, and `.venv/bin/python -m pytest tests/tui/test_screens.py -q`
passed all 101 tests in 38.23 seconds.

`DownloadOperation` is defined in `dnallm/tui/workers.py` and exposes the
`cancel()` method used by the screen. That file is read-only context for this
item: sibling `261028-vgn` owns its edits, so this plan has no sibling
dependency and must not modify it.

Scope is only `dnallm/tui/screens/download.py`; sibling quick items own every
other diagnostic path. Keep the change type-level with no visual or copy edits.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Declare the real download handle and run the focused static/runtime gates</name>
  <files>dnallm/tui/screens/download.py</files>
  <behavior>
    - Static RED baseline: the isolated pinned ty command reports exactly one invalid-assignment diagnostic where download_worker returns DownloadOperation to the Worker-typed attribute.
    - Runtime characterization: the existing 101-test screen lane passes before the edit, including progress, cancellation between attempts, retry, skip, prompt quit, and summary transitions.
    - Static GREEN: _worker accepts the actual DownloadOperation | None handle and the isolated check reports Found 0 diagnostics.
    - Runtime behavior remains unchanged: no executable statement, widget, message, cancellation call, timer, thread, or user-facing copy changes.
  </behavior>
  <action>
    From the repository root, first run the isolated ty command and focused
    screen lane from `<verification>` to record the static RED and runtime
    characterization baselines. Do not manufacture a failing runtime test for
    this annotation-only defect; the pinned type check is the RED/GREEN oracle,
    while the existing screen tests are the behavior guard required with this
    source change.

    In `dnallm/tui/screens/download.py`, remove `Any` from the `typing` import
    and remove the type-only `textual.worker.Worker` import because neither is
    used after the correction. Inside the existing `TYPE_CHECKING` block, import
    `DownloadOperation` from `..workers`. Change only the `_worker` declaration
    in `DownloadScreen.__init__` from the Textual worker union to
    `DownloadOperation | None`, relying on the file's existing postponed
    annotation evaluation. Do not add a runtime workers import, `cast`, `Any`,
    `# ty: ignore`, protocol, wrapper, or alias.

    Leave the `download_worker(...)` assignment, `_cancel`, `_stop_running`, and
    every other executable statement byte-identical. In particular, preserve the
    screen's private attribute name and its cooperative `cancel()` call; do not
    substitute a Textual worker or alter daemon-thread behavior. Do not edit
    `dnallm/tui/workers.py`, tests, configuration, UI copy, or sibling-owned
    files. No new runtime test is needed because this edit adds no branch or
    behavior; retain the existing 101-test screen lane unchanged as same-change
    pytest coverage.

    Run the focused screen lane, isolated pinned ty check, and ruff
    format/check over the touched source file. Then run the authoritative
    whole-tree ty command verbatim. Before sibling fixes merge it may still
    report their paths, but its output must contain no
    `dnallm/tui/screens/download.py` entry. Do not fix sibling-owned
    diagnostics, run repo-wide pytest lanes, or alter type/lint configuration.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/tui/test_screens.py -q && CPATH=/home/linuxbrew/.linuxbrew/include LIBRARY_PATH=/home/linuxbrew/.linuxbrew/lib uvx ty@0.0.84 check dnallm/tui/screens/download.py --python /tmp/ty-stub-env --output-format concise && .venv/bin/ruff format --check dnallm/tui/screens/download.py && .venv/bin/ruff check dnallm/tui/screens/download.py</automated>
  </verify>
  <done>All 101 existing screen tests pass; isolated ty exits 0 with Found 0 diagnostics; ruff format/check pass; the diff is limited to type-only imports and the _worker annotation; the whole-tree ty output has no download.py diagnostic.</done>
  <precondition>Run from the repository root with the existing project `.venv` and coordinator-provided `/tmp/ty-stub-env`; halt rather than substituting another Python environment or ty version.</precondition>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| app download seam -> download screen state | the operation handle crosses from the worker mixin into screen-managed cancellation state |
| download screen -> daemon download operation | a user cancellation request crosses into the cooperatively stopped transfer |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-vgq-01 | Tampering | DownloadScreen._worker | low | mitigate | Change only the declared handle type; preserve the exact seam assignment, None guard, cancel call, and cleanup, and run all 101 existing screen tests covering the affected flows. |
| T-vgq-02 | Denial of Service | download cancellation and import-time behavior | low | mitigate | Keep DownloadOperation under TYPE_CHECKING and make no thread or cancellation logic edits; retain the prompt-quit and cancellation-between-attempts tests so the type correction cannot introduce a runtime import or blocking worker path. |
| T-vgq-SC | Tampering | package installs | low | accept | Reserved row: this plan installs no npm/pip/cargo packages and adds no dependencies; it uses the coordinator's existing pinned environment. |
</threat_model>

<verification>
Item acceptance command (must exit 0 and print `Found 0 diagnostics`):

`CPATH=/home/linuxbrew/.linuxbrew/include LIBRARY_PATH=/home/linuxbrew/.linuxbrew/lib uvx ty@0.0.84 check dnallm/tui/screens/download.py --python /tmp/ty-stub-env --output-format concise`

Focused runtime lane:

`.venv/bin/python -m pytest tests/tui/test_screens.py -q`

Touched-file lint gates:

`.venv/bin/ruff format --check dnallm/tui/screens/download.py`

`.venv/bin/ruff check dnallm/tui/screens/download.py`

Authoritative batch/CI command (run verbatim):

`CPATH=/home/linuxbrew/.linuxbrew/include LIBRARY_PATH=/home/linuxbrew/.linuxbrew/lib uvx ty@0.0.84 check dnallm/ --python /tmp/ty-stub-env`

For this isolated item, success is zero download.py entries in that whole-tree
output; sibling items own the remaining paths. After all sibling fixes merge,
the authoritative command must print `Found 0 diagnostics`. No repo-wide test
lane is part of executor verification.
</verification>

<success_criteria>
- The original download.py invalid-assignment diagnostic is gone under ty@0.0.84 with the pinned stub environment, and the isolated file check reports zero diagnostics.
- `_worker` declares `DownloadOperation | None` through a TYPE_CHECKING import; the obsolete `Any` and `Worker` imports are removed.
- No executable behavior changes: download start, cancellation, cleanup, retry, skip, prompt quit, progress, and summary behavior remain intact.
- The existing 101-test screen lane passes unchanged, satisfying the runtime regression guard for this type-only source change.
- Ruff format/check pass on the touched file; no suppression, severity override, `[tool.ty]` exclusion, dependency, visual/copy, or configuration change is introduced.
- Only `dnallm/tui/screens/download.py` is modified.
</success_criteria>

<output>
Create `.planning/quick/261028-vgq-fix-1-ty-0-0-84-error-invalid-assignment-at-dnallm-tui-scree/261028-vgq-SUMMARY.md` when done, recording the static RED-to-GREEN result, unchanged 101-test runtime result, touched-file ruff results, and any remaining whole-tree diagnostics by sibling owner.
</output>
