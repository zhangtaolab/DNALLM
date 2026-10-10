---
phase: quick-261007-vxx
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - docs/user_guide/data_processing/format_conversion.md
  - docs/user_guide/fine_tuning/getting_started.md
  - docs/user_guide/models.md
autonomous: true
requirements:
  - QUICK-261007-VXX-01
user_setup: []

estimate:
  tokens: 12000
  raw_tokens: 8000
  tasks: 2
  confidence: high

must_haves:
  truths:
    - "`ruff format --check .` exits 0 repo-wide at the new dev HEAD, so the `ruff format --check .` steps in .github/workflows/ci.yml (lines 99 and 178) pass on every leg and PR #40's checks unblock"
    - "Exactly the three flagged docs pages changed and nothing else — the diff is confined to reflowed Python inside fenced code blocks (no prose edits, no source files)"
    - "The fix is committed to dev and pushed to origin/dev (no attribution trailers per owner rule; no .planning runtime artifacts staged)"
  artifacts:
    - "docs/user_guide/data_processing/format_conversion.md — the over-long dict literal in the fenced Python block at line ~118 wrapped by ruff format"
    - "docs/user_guide/fine_tuning/getting_started.md — the implicitly-concatenated f-string at line ~283 joined to one line by ruff format"
    - "docs/user_guide/models.md — the single-line load_model_and_tokenizer(...) call at line ~60 wrapped to stay within 100 columns"
  key_links:
    - "CI `ruff format --check .` gate (.github/workflows/ci.yml:99 test-windows / :178 test) -> repo-wide tree state -> fenced Python blocks in the three docs pages introduced by commit 1771991 (docs repair); ruff >=0.13 formats Python blocks inside Markdown, which is why a docs-only commit trips the format gate"
---

<objective>
Fix the CI `ruff format --check .` failure on dev: commit 1771991 (docs repair) introduced Python code blocks in three Markdown pages that ruff (pinned `ruff==0.16.10` in pyproject.toml) wants reformatted. CI runs for dev @1771991 failed in `test-windows (py3.12)` and `test (py3.13, numpy2.2.0)` with "3 files would be reformatted, 283 files already formatted". The fix is to run `ruff format` on the three flagged paths, prove the repo-wide check passes, and push to dev.

Purpose: The dev branch's format gate is red, blocking PR #40 checks. This is a formatting-only repair of fenced Python blocks — no `dnallm/` code changes, so the owner's pytest-with-change rule does not trigger.

Output: One style commit on dev containing only the three reformatted docs pages, pushed to origin/dev, with `ruff format --check .` green repo-wide.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@pyproject.toml
@.github/workflows/ci.yml

Live-tree facts observed at planning time (2026-10-07, branch dev @cfc8346):

- Local reproduction: `ruff format --check .` reports exactly 3 files would be reformatted, 283 already formatted. The flagged files and lines:
  1. docs/user_guide/data_processing/format_conversion.md:118 — over-long dict literal in a fenced Python block
  2. docs/user_guide/fine_tuning/getting_started.md:283 — implicit f-string concatenation ruff wants joined to one line
  3. docs/user_guide/models.md:60 — single-line `load_model_and_tokenizer(...)` call exceeding 100 columns (ruff wraps it to a 3-line call)
- Root cause: ruff >=0.13 formats Python code blocks inside Markdown; commit 1771991 introduced these blocks. Local ruff is 0.16.10 (matches the pyproject pin `ruff==0.16.10` at line 83) and reproduces the CI failure.
- CI gate locations: `.github/workflows/ci.yml:99` and `:178` both run `ruff format --check .`.
- Working tree at planning time is clean except untracked runtime dirs `.planning/graphs/`, `.planning/state.json`, `.planning/tmp/` — these must NEVER be staged.
- Docs-only change: no `dnallm/` code is touched, so no pytest run is required by the owner rule ("any dnallm/ code modification ships with pytest coverage").
- Out of scope: everything else. No ruff config changes, no CI workflow edits, no prose rewrites — the formatter is the only editor.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Ruff-format the three flagged docs pages and prove the repo-wide check is green</name>
  <files>docs/user_guide/data_processing/format_conversion.md, docs/user_guide/fine_tuning/getting_started.md, docs/user_guide/models.md</files>
  <action>
    Mutable-scope authority: the three-file set was observed live at planning time; re-observe before editing. Run `ruff --version` (expect 0.16.10, the pyproject pin — if the PATH ruff differs, use the matching tool; the formatter must be the version CI pins) and `ruff format --check .` from the repo root to capture the current flagged set. Expected: exactly the three paths above. If additional docs/ files are flagged, format them too (same root cause, same commit). If any non-docs file is flagged, STOP — that is a different defect than the one this plan authorizes; report it instead of formatting.

    Then run `ruff format` on exactly the flagged paths (pass the paths explicitly — do not run a repo-wide `ruff format .`): `ruff format docs/user_guide/data_processing/format_conversion.md docs/user_guide/fine_tuning/getting_started.md docs/user_guide/models.md`. Inspect `git diff` on the three files and confirm the hunks are pure reflow of Python inside fenced code blocks (dict-literal wrapping in format_conversion.md, f-string join in getting_started.md, call wrapping in models.md) with zero prose changes. Do not hand-edit anything the formatter did not touch.
  </action>
  <verify>
    <automated>ruff format --check . && [ "$(git diff --name-only | wc -l)" -eq 3 ] && [ -z "$(git diff --name-only -- . ':(exclude)docs')" ] && git diff --stat</automated>
  </verify>
  <done>
    `ruff format --check .` exits 0 reporting N files already formatted and zero "would be reformatted"; `git diff --name-only` lists exactly the three docs pages and nothing outside docs/ is modified; the diffstat shows only those three files.
  </done>
</task>

<task type="auto">
  <name>Task 2: Commit the three reformatted pages to dev and push to origin</name>
  <files>docs/user_guide/data_processing/format_conversion.md, docs/user_guide/fine_tuning/getting_started.md, docs/user_guide/models.md</files>
  <action>
    Pre-checks: `git branch --show-current` must print `dev` (observed live at planning time) — if on any other branch, STOP and report; `git status --porcelain` must show only the three modified docs files plus untracked `.planning/` entries.

    Stage the three files BY EXPLICIT PATH (never `git add -A` / `git add .` — the untracked `.planning/graphs/`, `.planning/state.json`, `.planning/tmp/` runtime dirs must never enter a commit). Commit with the message `style: ruff-format fenced python blocks in 3 docs pages (fixes CI ruff-format gate)` — no attribution trailers of any kind per owner rule. Then push: `git push origin dev`. This is a docs-only style change; no pytest run is required.
  </action>
  <verify>
    <automated>[ "$(git branch --show-current)" = "dev" ] && git push origin dev && [ "$(git rev-list --count origin/dev..HEAD)" -eq 0 ] && [ "$(git show --name-only --format= HEAD | wc -l)" -eq 3 ] && [ -z "$(git show --name-only --format= HEAD | grep '^\.planning/')" ] && git show --stat --oneline HEAD | head -6</automated>
  </verify>
  <done>
    HEAD is a single commit on dev containing exactly the three docs pages (no .planning paths in the commit), the message carries no attribution trailers, and `git rev-list --count origin/dev..HEAD` returns 0 after the push — dev on origin carries the fix, so PR #40's checks re-run against a green format gate.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| CI format gate ↔ repository content | The `ruff format --check .` steps in ci.yml read every Python block in the repo (including Markdown fences) and fail the build on any drift |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-261007-VXX-01 | Tampering | documented Python examples in the three docs pages | low | accept | ruff format is AST-preserving — the reflow changes layout only, never the semantics of the documented code; the Task 1 diff inspection confirms hunks are pure reflow, and the repo-wide `--check` gate pins the exact formatted shape so silent later drift fails CI |

No packages are installed and no code paths are added by this plan, so the package-legitimacy gate and the reserved `T-{phase}-SC` row do not apply.
</threat_model>

<verification>
- `ruff format --check .` exits 0 repo-wide (the exact command CI runs at ci.yml:99 and :178).
- `git diff --name-only` (pre-commit) / `git show --name-only` (post-commit) prove exactly the three docs pages changed and nothing else — no prose edits, no source files, no `.planning/` artifacts.
- `git rev-list origin/dev..HEAD` is empty after `git push origin dev` — origin carries the fix.
- The pushed commit message has no attribution trailers (owner rule: commit and push by default, never add trailers).
- Remote proof lands on the push: the dev CI workflow (and PR #40's checks) re-run the format step against the fixed tree.
</verification>

<success_criteria>
- The three flagged docs pages are ruff-formatted; repo-wide `ruff format --check .` is green.
- Exactly one style commit on dev with only those three files, no attribution trailers, pushed to origin/dev.
- CI format-gate failures (`test-windows (py3.12)`, `test (py3.13, numpy2.2.0)` at dev @1771991) are resolved by this push; PR #40 checks unblock.
</success_criteria>

<output>
Create `.planning/quick/261007-vxx-fix-ci-ruff-format-failure-run-ruff-form/261007-vxx-SUMMARY.md` when done
</output>
