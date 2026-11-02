---
phase: 261102-kh9
plan: 01
subsystem: dependencies
tags: [deps, dependabot, extras-guard, ci]
requires: []
provides:
  - "nbclient>=0.11.0 floors in test and notebook extras (supersedes dependabot PR #76)"
  - "platformdirs>=4.14.0,<5 floor in the tui extra (supersedes dependabot PR #80)"
  - "dependabot.yml semver-major ignores for transformers and huggingface-hub with recorded revisit condition"
affects: []
tech-stack:
  added: []
  patterns: []
key-files:
  created: []
  modified:
    - pyproject.toml
    - tests/test_extras_guard.py
    - .github/dependabot.yml
decisions:
  - "The two unresolvable major proposals are encoded as dependabot.yml ignores, not floor raises: pydantic-ai-slim[huggingface] (1.107.x, via the mcp extra's pydantic-ai>=1.107.0,<2 pin) caps huggingface-hub >=1.3.4,<2.0.0, so hub>=2 (PR #79) and the transformers 5.x floor (PR #78, same constraint chain plus the declared 4.49-5.x compat span) fail CI at resolution; revisit when pydantic-ai ships hub-2-compatible releases"
  - "Band prose now states the platformdirs >=4.14.0,<5 band rides inside textual 8.2.8's declared >=3.6.0,<5 constraint (rewritten in all three places: pyproject tui comment, guard block comment, tui assert message)"
status: complete
metrics:
  duration: 7 min
  completed: 2026-11-02
actuals:
  tokens: 1192
  tasks: 3
  commits: 2
  plan_head_before: 92198591ddddcf485e28f820db2bb98198f87c2e
  plan_head_after: a594363562d5688a94d5b9028779d6ea0d3c68ec
---

# Quick Task 261102-kh9: Apply dependabot PRs #76/#80 manually plus ignore entries Summary

Landed the nbclient >=0.11.0 and platformdirs >=4.14.0,<5 floors with the
extras-guard literal sync in one atomic commit, added transformers and
huggingface-hub semver-major ignores to dependabot.yml with rationale
comments, and opened landing PR #81 against dev (supersedes dependabot
PRs #76 and #80).

## What Was Done

### Task 1: deps floors + guard literal sync (tracer, TDD RED/GREEN)

- pyproject.toml: nbclient floor raised 0.10 -> 0.11.0 at both occurrences
  (test extra and notebook extra); tui extra platformdirs floor raised to
  >=4.14.0,<5 (ceiling unchanged); tui comment rewritten to state the band
  rides inside textual's declared >=3.6.0,<5 constraint, floor raised
  2026-11-02 by dependabot PR #80.
- tests/test_extras_guard.py: EXPECTED_NOTEBOOK_MEMBERS nbclient member and
  EXPECTED_TUI_MEMBERS platformdirs member synced with dated audit comments
  (matching the EXPECTED_MCP_MEMBERS style); band comment above
  EXPECTED_TUI_MEMBERS and the test_tui_extra_members_exact assert message
  rewritten truthfully; no other comment, frozenset, or test touched.
- Commit: 3939308

### Task 2: dependabot.yml ignore entries

- Two pure-addition blocks inserted in the pip ecosystem's ignore list,
  between the existing transformers comment block ("...before merging the
  bump.") and the datasets comment line: a transformers semver-major ignore
  (PR #78 rationale: same pydantic-ai/huggingface-hub constraint chain plus
  the abandoned 4.49-5.x compat span; planned owner decision; accepted bumps
  still owe the mamba re-verify, census-lane re-run, and tf_compat_* refresh
  duties) and a huggingface-hub semver-major ignore (PR #79 rationale: mcp
  extra pins pydantic-ai>=1.107.0,<2, whose pydantic-ai-slim[huggingface]
  1.107.x caps huggingface-hub >=1.3.4,<2.0.0 and hf-xet<1.5.0; revisit when
  pydantic-ai ships hub-2-compatible releases).
- Every existing entry, comment, and the datasets/torch comment blocks
  untouched; 19 insertions, 0 deletions.
- Commit: a594363

### Task 3: gates, boundary proof, landing

- Repo-wide `ruff format --check .`: 374 files already formatted (matches
  the b42ea95 baseline).
- Repo-wide `ruff check . --statistics`: exit 0, zero findings.
- Guard lane: 20 passed.
- Boundary proof: changed paths since b42ea95 excluding .planning are
  exactly `.github/dependabot.yml pyproject.toml
  tests/test_extras_guard.py`.
- Landing: direct `git push origin dev` attempted first per orchestrator
  override — rejected (GH006, protected branch, PR + 2 required checks).
  Fell back to the plan's path: pushed HEAD to
  `deps/nbclient-platformdirs-manual` (branch absent on origin before the
  push, no divergence) and opened PR #81 against dev. PR NOT merged here
  (orchestrator handles merge).

## TDD Evidence (Task 1 tracer)

- Pre-sync RED (pyproject edited, guard literals not yet synced):
  `TestNotebookExtraMembers::test_pre_existing_members_preserved` failed
  reporting `nbclient>=0.10` missing from the notebook extra, and
  `TestTuiExtraMembers::test_tui_extra_members_exact` failed reporting
  `platformdirs>=3.6,<5` missing — 2 failed, 18 passed.
- Post-sync GREEN: 20 passed (count unchanged; no tests added or removed).

## Verification Results

- `python -m pytest tests/test_extras_guard.py -q`: 20 passed.
- tomllib structural assertions: nbclient>=0.11.0 exactly once in test and
  once in notebook extras; platformdirs>=4.14.0,<5 exactly once in tui; zero
  old member literals anywhere in the extras table.
- grep: `nbclient>=0.10` and `platformdirs>=3.6` absent from both managed
  files.
- dependabot.yml: YAML parses; pip ignore list is exactly mcp, mamba-ssm,
  causal-conv1d, transformers, huggingface-hub, torch with expected
  update-types per entry; PR #78 comment within 10 lines above the
  transformers entry; pydantic-ai comment within 10 lines above the
  huggingface-hub entry; diff pure addition; no residue-gate substrings.
- Repo-wide ruff format and ruff check: clean (see Task 3).
- Tracer feedback gate (post-commit re-verify): 20 passed, no tracked-file
  deletions in either commit.

## Commits

- 3939308 deps(deps-dev): update nbclient to >=0.11.0 and platformdirs to >=4.14.0,<5
- a594363 ci(dependabot): ignore huggingface-hub and transformers semver-major bumps

Both carried on branch `deps/nbclient-platformdirs-manual` (together with
the planner's PLAN.md commit 9219859) and offered as
PR #81: https://github.com/zhangtaolab/DNALLM/pull/81

## Deviations from Plan

1. **[Orchestrator adaptation — landing order]** The plan assumed a direct
   push to dev is rejected without trying; the orchestrator instructed
   trying `git push origin dev` FIRST. It was attempted and rejected
   (GH006: protected branch, changes must go through a pull request, 2
   required status checks), so the plan's branch + PR fallback executed
   exactly as written. The PR was NOT merged (orchestrator handles merge),
   matching the plan's intent that the owner merges after checks go green.
2. **[Orchestrator adaptation — PR hygiene]** Dependabot PRs #76, #78, #79,
   #80 were not closed, commented on, or otherwise touched; they are listed
   below as owner post-merge steps only.

No other deviations — plan executed exactly as written.

## Auth Gates

None.

## Known Stubs

None.

## Threat Flags

None. All mitigations from the plan's threat register (T-kh9-01, T-kh9-02,
T-kh9-03) were applied as planned; T-kh9-SC (no installs, declaration-only
edits, venv untouched under uv --no-sync) held throughout.

## Owner Post-Merge Steps

1. Merge PR #81 after its required checks go green.
2. Close #76 and #80 as superseded, citing the merge commit.
3. Close #78 and #79 as superseded by the ignore entries (closing them
   before the superseding floors land on origin/dev would let Dependabot
   recreate them).
4. Pull dev locally.
5. Expect no hub/transformers semver-major proposals on the next Monday
   dependabot run.

## Self-Check: PASSED

- Commits 3939308 and a594363 are ancestors of HEAD and pushed to
  deps/nbclient-platformdirs-manual.
- Modified files present: pyproject.toml, tests/test_extras_guard.py,
  .github/dependabot.yml.
- PR #81 open against dev from deps/nbclient-platformdirs-manual.
