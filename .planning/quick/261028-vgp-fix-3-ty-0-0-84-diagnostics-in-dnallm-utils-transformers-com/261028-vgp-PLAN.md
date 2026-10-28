---
phase: 261028-vgp
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/utils/transformers_compat.py
autonomous: true
must_haves:
  truths:
    - "The isolated pinned ty@0.0.84 check of dnallm/utils/transformers_compat.py reports Found 0 diagnostics (planning baseline: exactly three diagnostics at lines 1597-1601)"
    - "The tied-weights materializer remains absence-gated and still merges the receiver's expanded map with prefixed child maps without reinitializing weights or changing the 27.1-02 IN-04 load-boundary receiver selection"
    - "The fix is type-level only: PreTrainedModel is resolved through the existing TYPE_CHECKING import and postponed annotations, so no new runtime import, cast, Any, protocol, suppression, or configuration change is introduced"
    - "The existing tests/utils/test_transformers_compat_census.py lane remains green in the same change and ruff format/check pass on the touched source file"
  artifacts:
    - "dnallm/utils/transformers_compat.py — _materialize_tied_weights_bookkeeping declares its real PreTrainedModel receiver and carries no stale ty ignore directives"
  key_links:
    - "TYPE_CHECKING PreTrainedModel import + postponed annotations -> typed model receiver -> named_children/get_expanded_tied_weights_keys/all_tied_weights_keys static surface"
    - "_patch_tied_weights_bookkeeping_on_load receiver selection -> _materialize_tied_weights_bookkeeping -> unchanged absence gate and post_init-equivalent mapping"
    - "existing census regression lane -> materialization, compliant-receiver, kwargs-receiver, and delegation behavior guards"
---

<objective>
Clear the three pinned ty@0.0.84 diagnostics in
`dnallm/utils/transformers_compat.py` by declaring the real model receiver type
instead of allowing `hasattr` to narrow `object` to a one-member structural
protocol.

Purpose: make the 27.1-02 IN-04 load-boundary compatibility code statically
honest while keeping its runtime behavior identical. The current `object`
annotation loses the `PreTrainedModel`/`nn.Module` attribute surface after the
`all_tied_weights_keys` absence check, producing the `named_children` error,
the incompatible mapping assignment error, and the stale-ignore warning.

Output: a `PreTrainedModel` parameter annotation and removal of the two ty
ignore directives that are unnecessary once the real receiver type is known.
No executable statement or test file changes.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@dnallm/utils/transformers_compat.py
@tests/utils/test_transformers_compat_census.py

Authoritative diagnostic inventory: `/tmp/ty-diagnostics.txt` (lines 28-30).
Pinned type environment: `/tmp/ty-stub-env` (already built by the coordinator).
Planning baseline: the isolated pinned ty command reported exactly three
diagnostics, and `.venv/bin/python -m pytest
tests/utils/test_transformers_compat_census.py -q` passed all 22 tests.

`PreTrainedModel` is already imported under `TYPE_CHECKING` at the top of the
source file, and `from __future__ import annotations` is already enabled. Use
those existing facilities; do not add a runtime transformers import. The
current ignore on the `get_expanded_tied_weights_keys` line is not one of the
three initial diagnostics, but it becomes unused after the annotation fix and
must be removed in the same edit so the result is genuinely zero-diagnostic.

Scope is only `dnallm/utils/transformers_compat.py`; sibling quick items own
every other diagnostic path. No dependency on those siblings is required.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Restore the real receiver type and remove obsolete suppressions</name>
  <files>dnallm/utils/transformers_compat.py</files>
  <behavior>
    - Static RED baseline: the isolated pinned ty command reports exactly three diagnostics: unresolved named_children, invalid all_tied_weights_keys assignment, and an unused ignore at the assignment.
    - Runtime characterization: the existing census lane passes before the edit and proves post_init-equivalent mapping, absence-gated passthrough, delegation, and kwargs-first IN-04 receiver behavior.
    - Static GREEN: declaring PreTrainedModel restores the full model attribute surface; removing both obsolete ignore directives leaves the isolated check at Found 0 diagnostics.
    - Runtime behavior remains unchanged: no executable statement, mapping key/value, child iteration, absence gate, wrapper, or receiver-selection expression changes.
  </behavior>
  <action>
    From the repository root, first run the isolated ty command and the focused
    pytest file from `<verification>` to record the static RED and runtime
    characterization baselines. Do not manufacture a failing runtime test for
    this annotation-only defect; the pinned type check is the RED/GREEN oracle,
    while the existing 22 tests are the behavior guard required with this
    source change.

    In `dnallm/utils/transformers_compat.py`, change only the
    `_materialize_tied_weights_bookkeeping` parameter annotation from `object`
    to `PreTrainedModel`. Reuse the existing `TYPE_CHECKING` import and
    postponed annotation evaluation; add no import. `PreTrainedModel` is the
    appropriate real base here because the function deliberately calls both
    its `get_expanded_tied_weights_keys` API and its inherited `nn.Module`
    `named_children` API.

    Remove the `# ty: ignore[unresolved-attribute]` directive from the existing
    `get_expanded_tied_weights_keys` expression and from the final
    `model.all_tied_weights_keys` assignment. The latter is already stale; the
    former would become stale after the annotation correction. Keep the short
    explanatory dynamic-receiver/instance-patch comments where useful, but do
    not replace either directive with another suppression. Do not use `cast`,
    `Any`, a new Protocol, a severity override, or a `[tool.ty]` exclusion.

    Leave every executable statement byte-identical, including the initial
    `hasattr` absence gate, `all_submodels=False`, child-map prefixing and
    copying, and the final mapping assignment. Do not touch
    `_patch_tied_weights_bookkeeping_on_load` or its kwargs `model`/`self`
    receiver precedence: that is shipped 27.1-02 IN-04 behavior. No new test is
    needed because this edit adds no runtime branch or behavior; retain the
    existing census lane unchanged as same-change pytest coverage.

    Run the focused pytest file, isolated pinned ty check, and ruff
    format/check over the touched source file. Then run the authoritative
    whole-tree ty command verbatim. Before sibling fixes merge it may still
    report their paths, but its output must contain no
    `dnallm/utils/transformers_compat.py` entry. Do not fix sibling-owned
    diagnostics, run repo-wide pytest lanes, or alter type/lint configuration.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/utils/test_transformers_compat_census.py -q && CPATH=/home/linuxbrew/.linuxbrew/include LIBRARY_PATH=/home/linuxbrew/.linuxbrew/lib uvx ty@0.0.84 check dnallm/utils/transformers_compat.py --python /tmp/ty-stub-env --output-format concise && .venv/bin/ruff format --check dnallm/utils/transformers_compat.py && .venv/bin/ruff check dnallm/utils/transformers_compat.py</automated>
  </verify>
  <done>All 22 focused census tests pass; isolated ty exits 0 with Found 0 diagnostics; ruff format/check pass; the diff is limited to the model annotation and removal/cleanup of the two obsolete ignore comments; the whole-tree ty output has no transformers_compat.py diagnostic.</done>
  <precondition>Run from the repository root with the existing project `.venv` and coordinator-provided `/tmp/ty-stub-env`; halt rather than substituting another Python environment or ty version.</precondition>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| transformers from_pretrained load boundary -> `_materialize_tied_weights_bookkeeping` | third-party and remote-code model receivers supply tied-weight maps to the compatibility rung |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-vgp-01 | Tampering | tied-weights materialization | medium | mitigate | Change only the static receiver annotation and stale comments; preserve the absence gate and exact own/child mapping computation, and run all 22 existing census tests covering materialization, compliant receivers, load delegation, and IN-04 kwargs receiver selection. |
| T-vgp-02 | Denial of Service | module import / optional transformers compatibility | low | mitigate | Reuse the existing TYPE_CHECKING import and postponed annotations so the type fix cannot add import-time transformers work or a new runtime dependency path; keep the targeted runtime lane green. |
| T-vgp-SC | Tampering | package installs | low | accept | Reserved row: this plan installs no npm/pip/cargo packages and adds no dependencies; it uses the coordinator's existing pinned environment. |
</threat_model>

<verification>
Item acceptance command (must exit 0 and print `Found 0 diagnostics`):

`CPATH=/home/linuxbrew/.linuxbrew/include LIBRARY_PATH=/home/linuxbrew/.linuxbrew/lib uvx ty@0.0.84 check dnallm/utils/transformers_compat.py --python /tmp/ty-stub-env --output-format concise`

Focused runtime lane:

`.venv/bin/python -m pytest tests/utils/test_transformers_compat_census.py -q`

Touched-file lint gates:

`.venv/bin/ruff format --check dnallm/utils/transformers_compat.py`

`.venv/bin/ruff check dnallm/utils/transformers_compat.py`

Authoritative batch/CI command (run verbatim):

`CPATH=/home/linuxbrew/.linuxbrew/include LIBRARY_PATH=/home/linuxbrew/.linuxbrew/lib uvx ty@0.0.84 check dnallm/ --python /tmp/ty-stub-env`

For this isolated item, success is zero transformers_compat.py entries in that
whole-tree output; sibling items own the remaining paths. After all sibling
fixes merge, the authoritative command must print `Found 0 diagnostics`. No
repo-wide test lane is part of executor verification.
</verification>

<success_criteria>
- All three original transformers_compat.py diagnostics are gone under ty@0.0.84 with the pinned stub environment, and the isolated file check reports zero diagnostics.
- `_materialize_tied_weights_bookkeeping` uses the existing type-only `PreTrainedModel` import; both obsolete ignore directives are removed without replacement suppressions.
- No executable behavior changes: absence gating, expanded/child mapping semantics, weight initialization behavior, and 27.1-02 IN-04 receiver selection remain intact.
- The existing 22-test census lane passes unchanged, satisfying the runtime regression guard for this type-only source change.
- Ruff format/check pass on the touched file; no severity override, `[tool.ty]` exclusion, dependency, or configuration change is introduced.
- Only `dnallm/utils/transformers_compat.py` is modified.
</success_criteria>

<output>
Create `.planning/quick/261028-vgp-fix-3-ty-0-0-84-diagnostics-in-dnallm-utils-transformers-com/261028-vgp-SUMMARY.md` when done, recording the static RED-to-GREEN result, unchanged 22-test runtime result, touched-file ruff results, and any remaining whole-tree diagnostics by sibling owner.
</output>
