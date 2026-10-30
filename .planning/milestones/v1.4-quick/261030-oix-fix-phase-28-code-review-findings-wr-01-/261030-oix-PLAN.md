---
phase: 261030-oix
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/census_compat.py
  - tests/models/test_census_compat.py
autonomous: true
requirements:
  - WR-01
  - IN-01
must_haves:
  truths:
    - "A rendered status cell longer than 300 characters raises ValueError before --emit-docs creates or modifies its destination"
    - "A resolved lead class missing from DOCS_BROKEN_LEADS raises a descriptive ValueError, so --emit-docs returns its designed clean exit 1 rather than escaping KeyError"
    - "The current registry still regenerates docs/user_guide/model_compatibility.md byte-for-byte, with DOCS_DATA_DATE and all census data unchanged"
  artifacts:
    - "scripts/census_compat.py — shared 300-character status-cell guard and missing-lead-class guard"
    - "tests/models/test_census_compat.py — two emit-path regression tests proving fail-closed behavior and no destination write"
  key_links:
    - "docs_status_cell over-cap ValueError -> emit_docs existing ValueError handler -> ERROR output, return 1, no page write"
    - "docs_broken_lead_class missing-wording ValueError -> emit_docs existing ValueError handler -> clean exit instead of traceback"
---

<objective>
Close Phase-28 review findings WR-01 and IN-01 in the compatibility-docs
generator: make the 300-character status-cell invariant fail closed at emit time
and turn lead-map drift into the same clean ValueError path used by other
unrenderable docs states.

Purpose: the generator must never write a page its independent consistency gate
rejects, and a future wording-map mistake must not produce a raw traceback. This
is a presentation-only guard change; no census re-run or generated-page content
change is intended.

Output: two minimal stdlib-only generator guards, two targeted pytest cases, and
byte-identical regeneration evidence.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-REVIEW.md
@scripts/census_compat.py
@tests/models/test_census_compat.py
@tests/models/test_compat_docs_consistency.py

Planning baseline observed live on branch v1.4: all 229 tests across the two
focused lane files pass in 3.02s; re-emitting to a temporary path compares
byte-identically with the committed compatibility page; repo-wide Ruff 0.16.10
check and targeted format checks pass. The knowledge graph is 599 hours old and
returned no relevant nodes, so current source and review evidence govern.

Source audit: GOAL (WR-01 cap guard, IN-01 lead-map guard, targeted tests,
byte-identical page) is covered by Task 1. REQ IDs WR-01 and IN-01 are both
implemented by Task 1. No RESEARCH.md or phase CONTEXT.md is present for this
quick item; the orchestrator-supplied owner constraints are covered by Task 1.
There are no deferred or unplanned source items.
</context>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: Prove both emit failures, then add the two fail-closed guards</name>
  <files>scripts/census_compat.py, tests/models/test_census_compat.py</files>
  <behavior>
    - An over-cap cell reaches emit_docs through a synthetic remote-code-broken registry entry; after the fix emit_docs returns 1, prints an ERROR naming the model and 300-character cap, and leaves the destination absent.
    - A resolved but unmapped lead class reaches emit_docs through the same public path; after the fix it returns 1 with an ERROR naming the missing class and DOCS_BROKEN_LEADS, leaves the destination absent, and no KeyError escapes.
    - Existing in-cap cells and every committed registry row render exactly as before.
  </behavior>
  <action>
    Start with RED tests in tests/models/test_census_compat.py. Add one class
    named TestDocsFailClosedGuards containing exactly two focused tests. Reuse
    the file's census module harness, SimpleNamespace, monkeypatch, tmp_path,
    and capsys; do not import the generator into the independent consistency
    gate and do not weaken its checker independence.

    For both tests, construct a synthetic pretrained entry with name, model,
    kind, tf_compat_status, tf_compat_tested, and tf_compat_evidence populated.
    Monkeypatch dnallm.models.registry.all_entries to return only that entry and
    sort_key_lab_first to a deterministic key, because docs_page_text imports
    both accessors at call time. Call census.emit_docs on a destination beneath
    tmp_path so each test exercises the actual docs_status_cell ->
    docs_page_text -> emit_docs path rather than only a private helper.

    The WR-01 test uses remote-code-broken evidence whose first cause clause is
    120 non-space characters and whose later semicolon clause contains the
    existing upstream-dead Triton API needle. This selects the existing longest
    lead while docs_bound_technical_fragment produces its maximum 101-character
    suffix, making the current generator emit an over-cap cell. Assert return
    code 1, no destination file, and ERROR output naming entry.model and the
    300-character limit. Before the production edit, confirm this test fails
    because the current code returns 0 and writes the page.

    The IN-01 test monkeypatches census.DOCS_BROKEN_LEAD_CLASSES to resolve a
    synthetic evidence detail to a class absent from DOCS_BROKEN_LEADS. Assert
    return code 1, no destination file, and ERROR output naming both the missing
    class and DOCS_BROKEN_LEADS. Before the production edit, confirm this test
    fails with the escaping KeyError; after the fix no exception may escape.

    Implement WR-01 in scripts/census_compat.py by adding DOCS_CELL_MAX_CHARS
    with value 300 beside the existing docs length constants. In
    docs_status_cell, preserve the existing wording derivation for shared,
    broken, and blocked statuses but send every successful branch through one
    final length check before returning. If the rendered cell is over the cap,
    raise ValueError naming entry.model, the cap, and the actual length. Cells
    at exactly 300 characters remain valid. Update the local docstring Raises
    text only as needed; do not change the independent gate's CELL_MAX_CHARS or
    any wording.

    Implement IN-01 inside docs_broken_lead_class at the matched-class return:
    before returning lead_class, require that it exists in DOCS_BROKEN_LEADS and
    otherwise raise ValueError naming the missing class and the map. Leave the
    existing DOCS_BROKEN_LEADS lookup in docs_status_cell and the existing
    emit_docs ValueError handler in place; do not substitute a broad
    KeyError catch. Keep both guards stdlib-pure and preserve the lazy registry
    and support imports.

    Run the new class to GREEN, then run all verification commands. Do not
    modify DOCS_DATA_DATE, registry or census artifacts, generated docs, lead
    wording, fragment bounds, or any other file. Do not run a census, install
    packages, clean caches, or broaden this into refactoring.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/models/test_census_compat.py::TestDocsFailClosedGuards -q && .venv/bin/python -m pytest tests/models/test_census_compat.py tests/models/test_compat_docs_consistency.py -q && tmp="$(mktemp)" && trap 'rm -f "$tmp"' EXIT && .venv/bin/python scripts/census_compat.py --emit-docs "$tmp" && cmp -s "$tmp" docs/user_guide/model_compatibility.md && .venv/bin/ruff check . && .venv/bin/ruff format --check scripts/census_compat.py tests/models/test_census_compat.py && git diff --check -- scripts/census_compat.py tests/models/test_census_compat.py && git diff --quiet -- docs/user_guide/model_compatibility.md dnallm/models/model_info.yaml tests/models/compat_census.json</automated>
  </verify>
  <done>Both new emit-path tests pass, the focused lane now passes 231 tests, temporary regeneration is byte-identical to the committed page, repo-wide Ruff and targeted formatting pass, and the generated page, registry, and census artifact have no diff.</done>
  <precondition>Run from the checkout root on branch v1.4 with the existing .venv; the two scoped source files start without uncommitted edits and the committed page matches the pre-change generator.</precondition>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| registry evidence -> docs generator -> public markdown | stored evidence and internal wording maps become user-facing table content |
| emit_docs -> destination filesystem | generated text crosses into a persistent docs artifact only after validation |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-oix-01 | Denial of Service | docs_status_cell and generated table layout | medium | mitigate | Enforce the independent gate's 300-character cap before page writing; test the over-cap emit path and assert no destination is created. |
| T-oix-02 | Tampering | DOCS_BROKEN_LEAD_CLASSES to DOCS_BROKEN_LEADS mapping | low | mitigate | Fail closed with a descriptive ValueError on missing wording; test clean exit 1 with no traceback or partial page. |
| T-oix-SC | Tampering | npm/pip/cargo installs | low | accept | Reserved supply-chain row: this plan installs no packages and adds no dependency, using only the existing project environment. |
</threat_model>

<verification>
The two new tests are the end-to-end tracer through synthetic registry input,
cell derivation, page rendering, emit error handling, and the filesystem
no-write guarantee. Their pre-fix failures must be observed for the intended
reasons before changing production code.

The final gate combines the new tests, both established focused lane files,
byte-for-byte temporary regeneration, repo-wide Ruff, targeted formatting, and
scoped diff checks. No repo-wide pytest lane, live model download, census run,
or workflow dispatch is required.
</verification>

<success_criteria>
- WR-01: every generated status cell over 300 characters fails before writing, with a ValueError naming the model, cap, and actual length.
- IN-01: an unmapped lead class produces clean ERROR output and exit 1 through the existing ValueError handler, never an escaping KeyError.
- Exactly two targeted regression tests are added and the focused lane passes 231 tests.
- The compatibility page remains byte-identical; DOCS_DATA_DATE, registry data, census evidence, wording, and fragment bounds are unchanged.
- Repo-wide Ruff, targeted format, whitespace, and untouched-data diff checks pass.
- Work remains on v1.4 with no new dependencies, cache cleanup, census re-run, or unrelated refactor.
</success_criteria>

<output>
Create `.planning/quick/261030-oix-fix-phase-28-code-review-findings-wr-01-/261030-oix-SUMMARY.md` when done. Record the RED failure evidence for both tests, the final 231-test result, temporary-path cmp result, Ruff/format results, and confirmation that no census run or generated-page change occurred.
</output>
