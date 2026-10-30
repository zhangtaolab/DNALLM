---
phase: 261030-qxo
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - .planning/phases/27.1-address-tech-debt-in-01-in-06-census-lane-live-dispatch/deferred-items.md
  - .planning/milestones/v1.1-phases/05-execution-harness-honest-gates-runner-feasibility/deferred-items.md
  - .planning/milestones/v1.1-phases/09-ci-wiring-census-verification/deferred-items.md
autonomous: true
requirements:
  - UAT-27.1-PRESET-FORMAT
  - UAT-05-CLI-PRECEDENCE
  - UAT-05-INTERPRET-LOOP
  - UAT-09-REGRESSION-PLOT
must_haves:
  truths:
    - "The four stale UAT-audit items use the canonical lowercase resolved status and carry dated evidence for the fixes that already landed"
    - "The Phase 05 README uv resolver item remains acknowledged and open for its separate quick fix"
    - "No file-level status falsely closes the remaining Phase 05 item, and no repository code, tests, product docs, or dependency configuration changes"
  artifacts:
    - ".planning/phases/27.1-address-tech-debt-in-01-in-06-census-lane-live-dispatch/deferred-items.md — resolved preset_loading format bullet with c3289a0 and live Ruff evidence"
    - ".planning/milestones/v1.1-phases/05-execution-harness-honest-gates-runner-feasibility/deferred-items.md — resolved CLI-precedence and dna_interpret bullets, with the README bullet untouched"
    - ".planning/milestones/v1.1-phases/09-ci-wiring-census-verification/deferred-items.md — resolved regression-plot bullet with quick-task, commit, and runner evidence"
  key_links:
    - "Canonical per-bullet resolved statuses -> UAT audit no longer reports the four stale items as outstanding"
    - "Dated resolution annotations -> commits, source lines, quick tasks, and runner run supply auditable closure evidence"
    - "Phase 05 README bullet retains status: acknowledged -> separate resolver fix remains visible and open"
---

<objective>
Close exactly four stale UAT-audit items whose fixes already landed by changing
their per-bullet statuses to the repository's canonical `status: resolved` form
and adding short 2026-10-30 evidence annotations in three deferred-item files.

Purpose: make the planning record reflect the verified fixes without hiding the
still-open Phase 05 README uv resolver item. This is planning-documentation
maintenance only; no repository code or product documentation needs modification.

Output: three scoped deferred-item edits covering UAT-27.1-PRESET-FORMAT,
UAT-05-CLI-PRECEDENCE, UAT-05-INTERPRET-LOOP, and UAT-09-REGRESSION-PLOT, plus
automated evidence that the README item remains acknowledged.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/phases/27.1-address-tech-debt-in-01-in-06-census-lane-live-dispatch/deferred-items.md
@.planning/milestones/v1.1-phases/05-execution-harness-honest-gates-runner-feasibility/deferred-items.md
@.planning/milestones/v1.1-phases/09-ci-wiring-census-verification/deferred-items.md

Planning baseline observed on branch v1.4: the worktree is clean; commits
c3289a0, 3fe80bf, and 550d311 exist with the expected fix subjects. The MCP
source has None sentinels for both CLI options at dnallm/mcp/server.py:3140-3154
and applies YAML values only when the supplied value is None at
dnallm/mcp/server.py:2797-2811. The post-change status gate below currently
fails at the still-open Phase 27.1 bullet, as expected; whitespace and
unchanged-source-tree baseline checks pass.

Source audit: GOAL and REQ are the four named stale items, all covered by Task
1. The orchestrator-supplied CONTEXT constraints are also covered by Task 1:
bullet structure is preserved, only four status lines and four annotations
change, the README resolver bullet remains acknowledged, and no file-level
status is introduced. No RESEARCH.md or phase CONTEXT.md exists for this quick
item. Other open deferred items are explicitly excluded; there are no unplanned
source items.
</context>

<tasks>

<task type="tracer">
  <name>Task 1: Flip the four evidenced bullets while preserving the open README item</name>
  <files>.planning/phases/27.1-address-tech-debt-in-01-in-06-census-lane-live-dispatch/deferred-items.md, .planning/milestones/v1.1-phases/05-execution-harness-honest-gates-runner-feasibility/deferred-items.md, .planning/milestones/v1.1-phases/09-ci-wiring-census-verification/deferred-items.md</files>
  <action>
    Use Edit for scoped replacements in the three existing files. Preserve every
    heading, bullet, nested detail line, and unrelated item. The complete change
    is four per-bullet status-line replacements plus one new annotation line
    immediately after each replaced status; do not add frontmatter or a
    file-level status.

    In the Phase 27.1 file, implement UAT-27.1-PRESET-FORMAT by changing only
    the preset_loading bullet's status to `status: resolved`. Add this single
    annotation immediately below it: `**Resolved 2026-10-30:** commit c3289a0
    formatted the guide's fenced Python examples; live repo-wide Ruff format
    verification passed with 374 files already formatted under dev Ruff
    0.16.10, with pyproject pinning 0.17.1, so the first-exposure warning is
    void.` Keep the original What and Scope decision text as historical
    context; the new annotation supersedes its stale warning.

    In the Phase 05 file, implement UAT-05-CLI-PRECEDENCE by changing only the
    dnallm-mcp-server CLI bullet's status to `status: resolved`. Add:
    `**Resolved 2026-10-30:** argparse now defaults --host and --port to None
    (dnallm/mcp/server.py:3140-3154), and host/port resolution consults YAML
    only when the supplied value is None (dnallm/mcp/server.py:2797-2811), so
    explicit CLI values win.`

    In the same Phase 05 file, implement UAT-05-INTERPRET-LOOP by canonicalizing
    the dna_interpret bullet's existing long uppercase resolution status to
    `status: resolved`, then adding: `**Resolved 2026-10-30:** UAT audit
    confirmed quick task 261003-ij4 and commit 3fe80bf (2026-10-03) moved
    interpretation into the default executor behind _interpret_thread_lock,
    with regression coverage in tests/mcp/test_interpret_tool.py.` Leave that
    bullet's historical What, Evidence, and Fix shape text intact.

    Do not edit the Phase 05 README uv resolver bullet. Its existing
    `status: acknowledged` must remain unchanged so the separate quick run
    still owns that open item; do not mark the whole Phase 05 file resolved.

    In the Phase 09 file, implement UAT-09-REGRESSION-PLOT by changing only the
    test_plot_for_regression bullet's primary status to `status: resolved`. Add:
    `**Resolved 2026-10-30:** UAT audit confirmed quick task 261006-cum and
    commit 550d311 (2026-10-06) retasked the test through benchmark.config and
    added a pinning test; runner run 37406829738 onward is green.` Preserve
    the existing nested Status line and its original resolution details.

    Keep all added prose in English and all items as bullets, not tables. Run
    the full verification gate and inspect the scoped diff to confirm exactly
    four status replacements and four annotations. Do not modify dnallm/,
    tests/, docs/, pyproject.toml, the README bullet, or any other deferred
    item; do not run pytest, Ruff, model downloads, installs, cache cleanup, or
    a new UAT audit for this documentation-only status reconciliation.
  </action>
  <verify>
    <automated>.venv/bin/python -I -c 'from pathlib import Path; p27=Path(".planning/phases/27.1-address-tech-debt-in-01-in-06-census-lane-live-dispatch/deferred-items.md").read_text(); p05=Path(".planning/milestones/v1.1-phases/05-execution-harness-honest-gates-runner-feasibility/deferred-items.md").read_text(); p09=Path(".planning/milestones/v1.1-phases/09-ci-wiring-census-verification/deferred-items.md").read_text(); readme,rest=p05.split("- dnallm-mcp-server CLI",1); cli,interp=rest.split("- MCP server `dna_interpret`",1); assert "status: acknowledged" in readme; assert all(x in p27 for x in ("status: resolved","**Resolved 2026-10-30:**","c3289a0","374 files already formatted")); assert all(x in cli for x in ("status: resolved","**Resolved 2026-10-30:**","3140-3154","2797-2811")); assert all(x in interp for x in ("status: resolved","**Resolved 2026-10-30:**","261003-ij4","3fe80bf")); assert all(x in p09 for x in ("status: resolved","**Resolved 2026-10-30:**","261006-cum","550d311","37406829738")); print("4 stale items resolved; README resolver item remains acknowledged")' && git diff --check -- .planning/phases/27.1-address-tech-debt-in-01-in-06-census-lane-live-dispatch/deferred-items.md .planning/milestones/v1.1-phases/05-execution-harness-honest-gates-runner-feasibility/deferred-items.md .planning/milestones/v1.1-phases/09-ci-wiring-census-verification/deferred-items.md && git diff --quiet -- dnallm docs tests pyproject.toml</automated>
  </verify>
  <done>All four named bullets have lowercase resolved statuses and 2026-10-30 evidence annotations; the README resolver bullet still has status acknowledged; no file-level Phase 05 status exists; the status/evidence gate, whitespace check, and unchanged-source-tree check pass; the scoped diff contains only four status replacements and four annotation lines.</done>
  <precondition>Run from the checkout root on branch v1.4 with the existing .venv; the three target files and product source trees start without uncommitted edits, and the cited fix commits are present.</precondition>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| deferred-item records -> UAT audit -> owner triage | per-bullet statuses and evidence determine which items remain actionable |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-qxo-01 | Tampering | Four stale deferred-item bullets | low | mitigate | Canonicalize only the named statuses, attach dated commit/source/runner evidence, and assert all four closures the scoped automated gate. |
| T-qxo-02 | Tampering | Shared Phase 05 deferred-items file | medium | mitigate | Leave the README resolver bullet untouched and acknowledged, add no file-level status, and assert its open state independently of the two resolved Phase 05 bullets. |
| T-qxo-SC | Tampering | npm/pip/cargo installs | low | accept | Reserved supply-chain row: this plan installs no packages and adds no dependencies. |
</threat_model>

<verification>
The single tracer task reconciles the complete audit path end to end: source
evidence already landed, each corresponding per-bullet status is canonicalized,
and the UAT-facing records retain enough evidence to justify closure.

The final automated gate positively checks all four resolved statuses and
resolution evidence, independently checks that the README resolver item remains
acknowledged, rejects whitespace errors, and proves dnallm/, docs/, tests/, and
pyproject.toml have no diff. Its baseline was run before planning completion and
failed at the still-open Phase 27.1 status as expected. No test suite or Ruff
run is needed because this task changes only .planning Markdown.
</verification>

<success_criteria>
- UAT-27.1-PRESET-FORMAT is resolved with c3289a0 and the 374-file Ruff result recorded.
- UAT-05-CLI-PRECEDENCE is resolved with both current server.py evidence ranges recorded.
- UAT-05-INTERPRET-LOOP is canonically resolved with 261003-ij4, 3fe80bf, and its original 2026-10-03 resolution preserved.
- UAT-09-REGRESSION-PLOT is resolved with 261006-cum, 550d311, and runner run 37406829738 recorded.
- The Phase 05 README uv resolver bullet remains unchanged and acknowledged; no file-level status closes it.
- Exactly three planning files change, limited to four status replacements and four evidence annotations; no repository code, tests, product docs, or dependency configuration changes.
- The automated status/evidence gate and both scoped diff checks pass.
</success_criteria>

<output>
Create `.planning/quick/261030-qxo-close-4-stale-uat-audit-items-flip-resol/261030-qxo-SUMMARY.md` when done. Record the four status flips and evidence annotations, explicit confirmation that the README resolver bullet remained acknowledged and untouched, the automated gate result, and confirmation that no repository code or product documentation changed.
</output>
