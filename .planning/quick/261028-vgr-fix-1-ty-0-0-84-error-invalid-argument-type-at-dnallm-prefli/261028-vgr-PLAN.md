---
phase: 261028-vgr
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/preflight/availability.py
autonomous: true
must_haves:
  truths:
    - "The isolated pinned ty@0.0.84 check of dnallm/preflight/availability.py reports Found 0 diagnostics (planning baseline: exactly one invalid-argument-type error at line 485)"
    - "_hf_adapter still sums every available Hugging Face sibling size exactly, skips siblings whose size metadata is None, and preserves the confirmed report revision and source hub"
    - "The fix narrows the already-guarded metadata value before int(); it adds no suppression, cast, fallback verdict, probe-semantics change, or type-configuration change"
    - "The existing tests/preflight/test_availability.py lane remains green in the same change and ruff format/check pass on the touched source file"
  artifacts:
    - "dnallm/preflight/availability.py — _hf_adapter passes a narrowed non-None sibling size to int()"
  key_links:
    - "HfApi.model_info(files_metadata=True) -> sibling.size | None -> existing None guard -> narrowed local size -> exact size_bytes sum"
    - "test_remote_hf_adapter_body_exact_size_and_revision -> integer sizes 100 and 23 plus None-sized sibling -> REMOTE_CONFIRMED size_bytes=123"
---

<objective>
Clear the single pinned ty@0.0.84 `error[invalid-argument-type]` in
`dnallm/preflight/availability.py` by narrowing the Hugging Face sibling size
that the existing generator already guards against `None`.

Purpose: make the static type contract match the working runtime contract. The
current filter excludes `None`, but the generator passes `sibling.size` back to
`int()` and ty cannot carry narrowing through that repeated attribute access.
Binding the guarded value to a local variable preserves the exact size-sum
behavior while giving `int()` a non-`None` argument.

Output: a two-expression correction inside `_hf_adapter`, with the existing
availability regression lane proving remote probing behavior is unchanged.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@dnallm/preflight/availability.py
@tests/preflight/test_availability.py

Authoritative diagnostic inventory: `/tmp/ty-diagnostics.txt` (line 1).
Pinned type environment: `/tmp/ty-stub-env` (already built by the coordinator).
Planning baseline: the isolated pinned ty command reported exactly the target
line-485 diagnostic, and `.venv/bin/python -m pytest
tests/preflight/test_availability.py -q` passed all 45 tests in 1.32 seconds.

`test_remote_hf_adapter_body_exact_size_and_revision` already exercises the
changed generator with integer sizes 100 and 23 and a third sibling whose size
is `None`; it requires an exact `size_bytes == 123`, `info.sha` propagation, and
a `REMOTE_CONFIRMED` Hugging Face report. Reuse that coverage unchanged rather
than adding a duplicate test for the same branch.

Scope is only `dnallm/preflight/availability.py`; sibling quick items own every
other diagnostic path. Keep the Phase 23 probe contract intact: missing size
metadata is skipped, and this edit must not turn incomplete metadata into a
probe failure, an unknown verdict, or a fabricated size.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Narrow the guarded HF sibling size and run focused static/runtime gates</name>
  <files>dnallm/preflight/availability.py</files>
  <behavior>
    - Static RED baseline: the isolated pinned ty command reports exactly one invalid-argument-type diagnostic at availability.py:485 because int() receives int | None.
    - Runtime characterization: all 45 existing availability tests pass before the edit, including exact HF size summation with a None-sized sibling.
    - Static GREEN: the generator binds the guarded sibling size once, narrows it with the existing is-not-None condition, and passes that narrowed value to int(); isolated ty reports Found 0 diagnostics.
    - Runtime behavior remains unchanged: present integer sizes are summed exactly, None sizes are skipped, empty or absent sibling lists sum to zero, and revision/source-hub/report states are untouched.
  </behavior>
  <action>
    From the repository root, first run the isolated ty command and focused
    availability lane from `<verification>` to record the static RED and runtime
    characterization baselines. The pinned type check is the RED/GREEN oracle;
    the existing test containing both integer and `None` sibling sizes is the
    same-change pytest behavior guard. Do not manufacture a failing runtime
    test for this static narrowing defect.

    In `_hf_adapter`, change only the `size_bytes` generator so its filter binds
    `size` with an assignment expression from `getattr(sibling, "size", None)`,
    retains the existing `is not None` guard on that local, and changes the sum
    expression from `int(sibling.size)` to `int(size)`. Keep `(info.siblings or
    [])`, `sum(...)`, and all surrounding statements unchanged. This makes the
    value checked by the guard identical to the value consumed by `int()` and
    avoids a second attribute lookup.

    Do not replace missing metadata with a numeric fallback, broaden exception
    handling, alter `AvailabilityReport`, or change local/remote probe order,
    timeout handling, hub fallthrough, or verdict classification. Do not add
    `cast`, `Any`, `# ty: ignore`, a helper, a wrapper, a new test duplicating
    existing coverage, severity overrides, or `[tool.ty]` exclusions. Do not
    edit tests, configuration, or sibling-owned files.

    Run the focused availability lane, isolated pinned ty check, and ruff
    format/check over the touched source file. Then run the authoritative
    whole-tree ty command verbatim. Before sibling fixes merge it may still
    report their paths, but its output must contain no
    `dnallm/preflight/availability.py` entry. Do not fix sibling-owned
    diagnostics, run repo-wide pytest lanes, or alter type/lint configuration.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/preflight/test_availability.py -q && CPATH=/home/linuxbrew/.linuxbrew/include LIBRARY_PATH=/home/linuxbrew/.linuxbrew/lib uvx ty@0.0.84 check dnallm/preflight/availability.py --python /tmp/ty-stub-env --output-format concise && .venv/bin/ruff format --check dnallm/preflight/availability.py && .venv/bin/ruff check dnallm/preflight/availability.py</automated>
  </verify>
  <done>All 45 existing availability tests pass; isolated ty exits 0 with Found 0 diagnostics; ruff format/check pass; the diff is limited to binding and consuming the narrowed size in _hf_adapter; the whole-tree ty output has no availability.py diagnostic.</done>
  <precondition>Run from the repository root with the existing project `.venv` and coordinator-provided `/tmp/ty-stub-env`; halt rather than substituting another Python environment or ty version.</precondition>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Hugging Face API -> _hf_adapter | remote repository metadata, including optional sibling sizes, crosses into local report construction |
| _hf_adapter -> availability report consumers | the derived byte total and confirmed verdict feed pre-flight display and downstream decisions |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-vgr-01 | Tampering | _hf_adapter size aggregation | low | mitigate | Preserve exact summation of present sizes and skipping of None metadata; run the existing adapter-body test that requires size_bytes=123 and unchanged revision/source hub. |
| T-vgr-02 | Denial of Service | _hf_adapter None handling | low | mitigate | Keep the is-not-None guard on the exact local passed to int(), preventing int(None) without introducing a new exception path or changing timeout/verdict behavior. |
| T-vgr-SC | Tampering | package installs | low | accept | Reserved row: this plan installs no npm/pip/cargo packages and adds no dependencies; it uses the coordinator's existing pinned environment. |
</threat_model>

<verification>
Item acceptance command (must exit 0 and print `Found 0 diagnostics`):

`CPATH=/home/linuxbrew/.linuxbrew/include LIBRARY_PATH=/home/linuxbrew/.linuxbrew/lib uvx ty@0.0.84 check dnallm/preflight/availability.py --python /tmp/ty-stub-env --output-format concise`

Focused runtime lane:

`.venv/bin/python -m pytest tests/preflight/test_availability.py -q`

Touched-file lint gates:

`.venv/bin/ruff format --check dnallm/preflight/availability.py`

`.venv/bin/ruff check dnallm/preflight/availability.py`

Authoritative batch/CI command (run verbatim):

`CPATH=/home/linuxbrew/.linuxbrew/include LIBRARY_PATH=/home/linuxbrew/.linuxbrew/lib uvx ty@0.0.84 check dnallm/ --python /tmp/ty-stub-env`

For this isolated item, success is zero availability.py entries in that
whole-tree output; sibling items own the remaining paths. After all sibling
fixes merge, the authoritative command must print `Found 0 diagnostics`. No
repo-wide test lane is part of executor verification.
</verification>

<success_criteria>
- The original availability.py invalid-argument-type diagnostic is gone under ty@0.0.84 with the pinned stub environment, and the isolated file check reports zero diagnostics.
- `_hf_adapter` binds the guarded size to a local and passes that narrowed non-None value to `int()`.
- Present sizes still sum exactly and None sizes remain skipped; revision, source hub, report state, timeout handling, and hub fallthrough are unchanged.
- All 45 existing availability tests pass unchanged, satisfying the same-change pytest regression guard for this behavior-preserving source correction.
- Ruff format/check pass on the touched file; no suppression, cast, severity override, `[tool.ty]` exclusion, dependency, or configuration change is introduced.
- Only `dnallm/preflight/availability.py` is modified.
</success_criteria>

<output>
Create `.planning/quick/261028-vgr-fix-1-ty-0-0-84-error-invalid-argument-type-at-dnallm-prefli/261028-vgr-SUMMARY.md` when done, recording the static RED-to-GREEN result, unchanged 45-test runtime result, touched-file ruff results, and any remaining whole-tree diagnostics by sibling owner.
</output>
