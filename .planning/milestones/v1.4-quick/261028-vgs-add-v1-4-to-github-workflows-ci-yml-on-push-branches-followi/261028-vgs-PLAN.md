---
phase: 261028-vgs
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - .github/workflows/ci.yml
autonomous: true
must_haves:
  truths:
    - "A push to v1.4 matches CI on.push.branches, so the existing test, test-windows, test-cuda, and coverage-gate jobs are eligible to run before the v1.4-to-main merge"
    - "The v1.4 entry carries a dated 2026-10-28 comment following the v1.3 milestone-closeout pattern, including the first-exposure rationale, ty hard-gate exposure, D-19 nightly posture, deploy restriction, and removal condition"
    - "pull_request.branches remains main/master/dev; workflow_dispatch inputs, cron schedules, every job definition, self-hosted nightly routing, and main/master-only deployment remain semantically identical to HEAD"
    - "ci.yml parses as YAML, the focused runner-infrastructure contract lane remains green, and the change has no whitespace errors"
  artifacts:
    - ".github/workflows/ci.yml — v1.4 added immediately after v1.3 under on.push.branches with an adjacent dated rationale comment"
  key_links:
    - "v1.4 push -> on.push.branches -> existing push-event job guards -> hosted matrix legs and coverage-gate ty hard gate"
    - "v1.4 push -> unchanged nightly job guards -> no schedule/workflow_dispatch-only self-hosted lane (D-19)"
    - "v1.4 push -> unchanged deploy ref guard -> no gh-pages deployment outside main/master"
---

<objective>
Add `v1.4` to the existing CI push triggers so the hosted matrix and coverage
gate, including its pinned ty hard gate, produce merge evidence throughout the
v1.4 milestone-closeout cycle instead of first running at the v1.4-to-main
merge.

Purpose: apply the documented 2026-10-07 first-exposure lesson using the v1.3
entry as the structural precedent. This is deliberately independent of the
sibling ty fixes: initial exposure may be red, and those siblings own turning
the gate green. Preserve the nightly lanes' cron/workflow_dispatch-only D-19
boundary and the main/master-only deploy boundary exactly.

Output: one scoped workflow-trigger edit and its dated operational rationale;
no Python source, test, dependency, or other workflow changes.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.github/workflows/ci.yml
@tests/test_runner_infra_contracts.py

Planning baseline (2026-10-28): PyYAML parsed `on.push.branches` as
`[main, master, dev, v1.3]` and `pull_request.branches` as
`[main, master, dev]`. The existing `test`, `test-windows`, `test-cuda`, and
`coverage-gate` job guards all admit push events; no job-body edit is needed to
expose them on v1.4. The three nightly jobs are schedule/dispatch-only on
`[self-hosted, dnallm-nightly]`, and deploy requires a push whose ref is main
or master.

The focused baseline `.venv/bin/python -m pytest
tests/test_runner_infra_contracts.py -q` passed 14 tests in 0.29 seconds. That
file reads ci.yml only for the example-nightly SSE probe; it does not pin the
branch lists. Reuse it unchanged as the workflow regression lane.

PyYAML 1.1 parses the bare `on:` key as boolean `True`; the verification must
use `doc.get(True) or doc.get("on")`, as the existing runner-infra contracts
do. Scope is only `.github/workflows/ci.yml`; all ty diagnostics and their
source files belong to sibling quick items.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add the v1.4 push trigger and prove all other CI routing is unchanged</name>
  <files>.github/workflows/ci.yml</files>
  <action>
    From the repository root, first run the Python assertion command in
    `<verify>` against the unedited file. It must stop on the missing v1.4 push
    branch; treat any different failure as a changed baseline and halt rather
    than editing around it. The focused runner-infra baseline is already green.

    Edit only the `on.push.branches` area near the top of
    `.github/workflows/ci.yml`. Insert `- v1.4` immediately after the existing
    `- v1.3`, retaining main, master, dev, and v1.3 in their current order.
    Directly adjacent to that entry, add a comment block whose first line is
    `# 2026-10-28: temporary v1.4 milestone-closeout vehicle (the 2026-10-07`.
    Continue the v1.3-style rationale across readable comment lines: long-lived
    working branches otherwise reach the merge with zero matrix exposure; the
    matrix legs, including the ty hard gate, need push exposure on v1.4 for
    first-exposure-free merge evidence; nightly legs stay
    cron/workflow_dispatch-gated per D-19 so v1.4 pushes never reach the
    self-hosted box; deploy stays main/master-only; and the entry stays until
    the actual v1.4-to-main merge or explicit owner removal. Include the exact
    phrases `matrix legs, including the ty hard gate, need push exposure` and
    `actual v1.4-to-main merge or explicit owner removal` so the operational
    contract is mechanically checkable.

    Do not add v1.4 to `pull_request.branches`, remove or reorder another push
    branch, alter dispatch inputs or cron entries, change any job guard/matrix/
    runner/permission/step, or touch the ty command. In particular, preserve
    every nightly job's D-19 schedule/dispatch-only guard and the deploy job's
    main/master ref guard. Do not edit tests, Python source, documentation, or
    sibling-owned files, and do not attempt to fix or suppress the diagnostics
    this trigger intentionally exposes.

    Before committing, run the complete `<verify>` command. Its HEAD comparison
    is the semantic scope guard: all jobs, PR/dispatch/schedule triggers, and
    permissions must equal their pre-item values while the push list gains only
    v1.4. Do not use a live push or workflow dispatch as local acceptance. The
    first real v1.4 run may have a red ty hard gate until the sibling fixes
    land; that expected exposure is not a failure of this item.
  </action>
  <verify>
    <automated>.venv/bin/python -c "import subprocess,yaml; p='.github/workflows/ci.yml'; text=open(p,encoding='utf-8').read(); d=yaml.safe_load(text); base=yaml.safe_load(subprocess.check_output(['git','show','HEAD:'+p],text=True)); t=d.get(True) or d.get('on'); b=base.get(True) or base.get('on'); push_event=\"github.event_name == 'push'\"; assert t['push']['branches']==['main','master','dev','v1.3','v1.4']; assert t['pull_request']==b['pull_request']; assert t['workflow_dispatch']==b['workflow_dispatch']; assert t['schedule']==b['schedule']; assert d['jobs']==base['jobs']; assert d['permissions']==base['permissions']; assert all(push_event in d['jobs'][n]['if'] for n in ('test','test-windows','test-cuda','coverage-gate')); assert all(push_event not in d['jobs'][n]['if'] for n in ('test-mamba','coverage-nightly','example-nightly')); assert '2026-10-28: temporary v1.4 milestone-closeout vehicle' in text; assert 'matrix legs, including the ty hard gate, need push exposure' in text; assert 'actual v1.4-to-main merge or explicit owner removal' in text; print('ci.yml v1.4 trigger contract: OK')" && .venv/bin/python -m pytest tests/test_runner_infra_contracts.py -q && git diff --check -- .github/workflows/ci.yml</automated>
  </verify>
  <done>The parsed push list is exactly main/master/dev/v1.3/v1.4; the dated v1.4 rationale and removal condition are present; all four hosted push jobs retain push guards; nightly jobs retain no push arm; PR, dispatch, schedule, jobs, and permissions equal HEAD; all 14 runner-infra tests pass; and git diff --check exits 0.</done>
  <precondition>Run from the repository root before the item commit with the existing `.venv` (PyYAML and pytest installed) and a HEAD whose ci.yml push list is exactly main/master/dev/v1.3; halt on any baseline mismatch.</precondition>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| v1.4 git push -> GitHub Actions | pushed repository content causes hosted CI jobs to execute |
| workflow trigger/job routing -> self-hosted nightly runner | event guards prevent push-authored code from crossing onto the persistent GPU box (D-19) |
| workflow deploy guard -> gh-pages | the ref condition restricts documentation write access to main/master pushes |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-vgs-01 | Tampering | ci.yml push trigger | medium | mitigate | Add only v1.4 after v1.3, preserve every existing branch, and require the parsed list plus dated rationale to match exactly. |
| T-vgs-02 | Elevation of Privilege | nightly job routing | high | mitigate | Keep D-19 schedule/workflow_dispatch-only guards and self-hosted routing unchanged; the acceptance assertion compares every job and schedule/dispatch trigger to HEAD and proves nightly guards contain no push arm. |
| T-vgs-03 | Tampering | deploy branch restriction | high | mitigate | Leave deploy and its contents:write permission untouched; compare all jobs and top-level permissions to HEAD so v1.4 cannot acquire gh-pages deployment or extra privileges. |
| T-vgs-SC | Tampering | package installs | low | accept | Reserved row: this plan installs no npm/pip/cargo packages and adds no dependencies; it uses the existing project environment. |
</threat_model>

<verification>
Run the task's complete `<automated>` command from the repository root before
committing. The Python check is both the red-to-green trigger oracle and the
semantic non-regression gate against `HEAD:.github/workflows/ci.yml`; the
focused pytest lane supplies 14 existing workflow/runner contract checks, and
`git diff --check` rejects whitespace damage.

A subsequent real GitHub Actions run is observational evidence owned by the
batch coordinator, not a prerequisite for this local plan. Before sibling ty
fixes merge, a red ty step is expected; after they merge, the same exposed
coverage-gate should become green without another trigger edit.
</verification>

<success_criteria>
- `v1.4` appears exactly once in `on.push.branches`, immediately after `v1.3`, with all previous entries retained.
- The adjacent 2026-10-28 comment records the milestone-closeout purpose, 2026-10-07 first-exposure lesson, matrix and ty-gate exposure, D-19 nightly restriction, deploy restriction, and removal at the actual v1.4-to-main merge or explicit owner removal.
- `pull_request.branches` remains exactly main/master/dev; schedule and workflow_dispatch configuration are unchanged.
- Every job and permission is semantically identical to HEAD, including the coverage-gate ty invocation, nightly self-hosted guards, and main/master-only deploy guard.
- The YAML/comment/HEAD-comparison assertion prints `ci.yml v1.4 trigger contract: OK`, all 14 focused runner-infra tests pass, and `git diff --check` exits 0.
- Only `.github/workflows/ci.yml` is modified; no sibling diagnostic is fixed or suppressed and no live CI result is treated as this item's local gate.
</success_criteria>

<output>
Create `.planning/quick/261028-vgs-add-v1-4-to-github-workflows-ci-yml-on-push-branches-followi/261028-vgs-SUMMARY.md` when done, recording the pre-edit missing-branch assertion, the successful parsed trigger/HEAD-equivalence check, the 14-test focused result, the diff-check result, and the distinction between trigger exposure and sibling-owned ty gate redness.
</output>
