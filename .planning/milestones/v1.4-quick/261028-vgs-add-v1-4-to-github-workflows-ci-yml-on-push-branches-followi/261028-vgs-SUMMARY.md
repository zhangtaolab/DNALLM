---
phase: 261028-vgs
plan: 01
type: execute
subsystem: ci
tags: [ci, github-actions, push-trigger, v1.4, milestone-closeout]
requires:
  - "HEAD ci.yml push list exactly main/master/dev/v1.3 (planning baseline 2026-10-28)"
provides:
  - "v1.4 push trigger on the CI workflow: hosted matrix legs (test, test-windows, test-cuda) and coverage-gate (incl. the pinned ty hard gate) are eligible to run on every v1.4 push before the v1.4-to-main merge"
affects:
  - ".github/workflows/ci.yml (on.push.branches only)"
key-files:
  created: []
  modified:
    - .github/workflows/ci.yml
decisions:
  - "v1.4 entry + dated 2026-10-28 rationale appended after the existing v1.3 comment block, mirroring the established chronological dated-comment pattern; the two mechanically-checkable contract phrases each sit contiguous on a single comment line"
metrics:
  duration: 4 min
  tasks: 1
  files: 1
  completed: 2026-10-28
status: complete
actuals:
  tasks: 1
  commits: 1
  commit: c96bbb1
---

# Quick Item 261028-vgs: Add v1.4 to GitHub workflows ci.yml push branches Summary

One scoped workflow-trigger edit: `v1.4` added to `on.push.branches` immediately after `v1.3` with an adjacent dated rationale comment, so the hosted matrix legs and the coverage-gate (including its pinned ty hard gate) produce merge evidence throughout the v1.4 milestone-closeout cycle instead of first running at the v1.4-to-main merge. Every other routing surface — PR branches, dispatch inputs, cron schedules, all job definitions, nightly self-hosted guards, main/master-only deploy — is proven semantically identical to HEAD.

## What Was Done

- `.github/workflows/ci.yml`: inserted `- v1.4` immediately after `- v1.3` under `on.push.branches` (main, master, dev, v1.3 order retained) and appended an 8-line `# 2026-10-28: temporary v1.4 milestone-closeout vehicle (the 2026-10-07` comment block directly adjacent, following the file's established chronological dated-comment pattern (the 2026-10-07 lesson block, then the 10-10/10-15 v1.3 additions, now the 10-28 v1.4 addition).
- The comment records: long-lived working branches otherwise reach the merge with zero matrix exposure; `matrix legs, including the ty hard gate, need push exposure` on v1.4 for first-exposure-free merge evidence (exact phrase contiguous on one line); nightly legs stay cron/workflow_dispatch-gated per D-19 so v1.4 pushes never reach the self-hosted box; deploy stays main/master-only; the entry stays until the `actual v1.4-to-main merge or explicit owner removal` (exact phrase contiguous on one line).
- No other change: 9 insertions, 0 deletions, single file (`git show --stat` confirms). pull_request.branches remains `[main, master, dev]`; workflow_dispatch inputs, both cron entries, every job guard/matrix/runner/permission/step, and the ty invocation are byte-identical outside the trigger block.

## Verification Evidence

### Pre-edit red baseline (trigger oracle)

The plan's full Python assertion run against the unedited file failed at its first check — `assert t['push']['branches'] == ['main','master','dev','v1.3','v1.4']` on the missing v1.4 entry (AssertionError, exit 1), with no other failure: the baseline was exactly as planned, so the edit proceeded. Precondition also verified read-only first: repo root, branch `v1.4`, existing `.venv` with PyYAML/pytest, HEAD push list exactly main/master/dev/v1.3.

### Post-edit green (all three verify legs)

1. **Parsed trigger/HEAD-equivalence contract:** `ci.yml v1.4 trigger contract: OK` — PyYAML-parsed push list is exactly `['main','master','dev','v1.3','v1.4']`; `pull_request`, `workflow_dispatch`, and `schedule` sections, the entire `jobs` map, and `permissions` all compare equal to `HEAD:.github/workflows/ci.yml` (the bare-`on:` YAML 1.1 boolean key handled via `doc.get(True) or doc.get('on')`); `github.event_name == 'push'` present in the `if` guards of `test`/`test-windows`/`test-cuda`/`coverage-gate` and absent from `test-mamba`/`coverage-nightly`/`example-nightly` (D-19 boundary intact); all three required comment phrases present in the file text.
2. **Focused runner-infra lane:** `.venv/bin/python -m pytest tests/test_runner_infra_contracts.py -q` → **14 passed in 0.29s** (matches the planning baseline of 14 tests / 0.29s; the lane was also green pre-edit at 0.28s).
3. **Whitespace:** `git diff --check -- .github/workflows/ci.yml` → exit 0 (clean).

## Trigger Exposure vs. Sibling-Owned Gate Redness

This item intentionally exposes existing jobs to a new event; it does not and must not change any job's content. The five sibling ty-fix items (261028-vgn/vgo/vgp/vgq/vgr) have already landed and the whole-tree ty gate reports zero diagnostics, so the newly exposed coverage-gate ty hard gate is expected green on the first real v1.4 run — but per the plan, even a red first run would not have been a failure of this item: trigger exposure and gate greenness are separate concerns, and no live GitHub Actions run was used as local acceptance.

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Self-Check: PASSED

- `.github/workflows/ci.yml` present on disk with the committed change (commit `c96bbb1`, verified `git merge-base --is-ancestor c96bbb1 HEAD`)
- This SUMMARY.md present with `status: complete` in frontmatter


---

*Item 261028-vgs · commit c96bbb1 · 2026-10-28*
