---
phase: quick-261007-xbz
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - pyproject.toml
  - dnallm/version.py
  - CHANGELOG.md
autonomous: true
requirements:
  - QUICK-261007-XBZ-01
user_setup: []

estimate:
  tokens: 12000
  raw_tokens: 8000
  tasks: 2
  confidence: high

must_haves:
  truths:
    - "`import dnallm` reports `__version__` 0.7.1 and the tomllib sync assertion proves the pyproject.toml `[project]` version equals it — the two code-level version carriers agree at 0.7.1"
    - "CHANGELOG.md carries a new `## [0.7.1] - 2026-10-08` Keep-a-Changelog entry above the byte-identical `## [0.7.0] - 2026-10-07` entry, summarizing the post-0.7.0 stabilization already merged on dev (CI first-exposure fixes, docs accuracy repair, ruff-format gate fix, ruff bump)"
    - "The only changed content lines in the pyproject.toml diff are the old and new `[project]` version line — the three dependency floors that share the same digit string (line 30 captum, line 33 einops, line 40 loguru) are byte-identical"
    - "A fast pytest subset (tests/test_extras_guard.py + tests/utils/test_sequence.py, ~10s grounded) passes in the same change as the dnallm/version.py edit, per the owner rule that any dnallm/ code change ships with pytest"
    - "The bump is a trailer-free single-line release commit on dev pushed to origin/dev with `git rev-list origin/dev..HEAD` empty; untracked .planning runtime artifacts are never staged; dev HEAD becomes the v0.7.1 release tag target once PR #40 merges"
  artifacts:
    - "pyproject.toml — line 3 `version = \"0.7.1\"`; the only change in the file"
    - "dnallm/version.py — the single line `__version__ = \"0.7.1\"`"
    - "CHANGELOG.md — new `## [0.7.1] - 2026-10-08` section (Overview + Fixed + Changed) inserted between the Keep-a-Changelog header block and the 0.7.0 entry"
  key_links:
    - "dnallm/__init__.py:20 `from .version import __version__` (listed in `__all__` at line 12) → dnallm/version.py — the `python -c \"import dnallm; assert dnallm.__version__ == '0.7.1'\"` check depends on this re-export"
    - "tests/test_extras_guard.py → pyproject.toml (its 5 tests parse the extras declarations) — passing it proves the edited pyproject.toml still parses"
    - "future tag v0.7.1 (three-part semver product tag, applied at release time after PR #40 merges — never milestone/vN) → the dev HEAD produced by this task"
---

<objective>
Bump the dnallm package version 0.7.0 → 0.7.1 across the entire version surface — exactly three spots, repo-grep-verified: the `[project]` version field in pyproject.toml (line 3), `__version__` in dnallm/version.py (single-line file), and a new Keep-a-Changelog `## [0.7.1] - 2026-10-08` entry at the top of CHANGELOG.md summarizing the post-0.7.0 stabilization already merged on dev. Verify per the owner rule (any dnallm/ change ships with pytest in the same change — version.py counts), commit with a trailer-free single-line message, and push to dev.

Purpose: 0.7.0 shipped 2026-10-07 and dev has since absorbed a stabilization wave (Windows pybigwig marker, transformers>=5.19 device-query shim covering both torch signatures, OS-native test assertions, 49-page docs accuracy repair, CI ruff-format gate fix, dependabot ruff 0.16.9→0.16.10). The patch release records that stabilization; CI at dev@56f6ad5 is fully green, so the bump rides a proven baseline and this HEAD becomes the v0.7.1 release tag target once PR #40 merges.

Output: One release commit on dev touching exactly pyproject.toml, dnallm/version.py, CHANGELOG.md; all version gates green; pushed to origin/dev.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@pyproject.toml        # line 3 is the [project] version field; lines 30/33/40 are dependency floors that MUST stay untouched
@dnallm/version.py     # entire file is one line
@CHANGELOG.md          # the 0.7.0 entry (lines 8-31) is the tone/section-style reference for the compact 0.7.1 entry
</context>

<tasks>

<task type="auto">
  <name>Task 1: Bump all three version surfaces and write the 0.7.1 changelog entry</name>
  <files>pyproject.toml, dnallm/version.py, CHANGELOG.md</files>
  <action>
    Three edits, nothing else:
    1. pyproject.toml — change ONLY line 3 under `[project]` from `version = "0.7.0"` to `version = "0.7.1"`. WARNING: the digit string 0.7.0 also appears in three dependency floors — line 30 `captum>=0.7.0`, line 33 `einops>=0.7.0`, line 40 `loguru>=0.7.0` — which MUST stay byte-identical. Edit line 3 only; never run a global find/replace on the file (the diff-shape gate below catches any collateral damage).
    2. dnallm/version.py — the file is a single line; set it to `__version__ = "0.7.1"`.
    3. CHANGELOG.md — insert a new `## [0.7.1] - 2026-10-08` section between the header block (Keep-a-Changelog/SemVer statement, ends at line 6) and `## [0.7.0] - 2026-10-07` (line 8). Compact patch entry in the same Overview/Added/Fixed section style as the 0.7.0 entry, structured as Overview + Fixed + Changed:
       - Overview (one short paragraph): patch release recording the stabilization that landed on dev immediately after 0.7.0 — CI first-exposure fixes, a verifier-driven docs accuracy repair, and tooling refresh; no API changes.
       - Fixed bullets, one per item: (a) Windows installs failing on pybigwig — pygenometracks gated behind a non-Windows platform marker in the notebook extra; (b) transformers >= 5.19 device-type query crashing the import chain on CUDA-built torch without a visible GPU — compat shim answering both observed signatures (torch >= 2.6 RuntimeError and torch <= 2.5 AttributeError); (c) OS-native test assertions replacing Unix-only fd//proc assumptions; (d) CI ruff-format gate tripping on over-long fenced Python blocks in 3 docs pages — reformatted; (e) docs: 49-page verifier-driven accuracy repair retiring the stale VERIFICATION_REPORT snapshot.
       - Changed bullet: ruff dev dependency 0.16.9 → 0.16.10 (dependabot).
       The `## [0.7.0]` entry and everything below it stay byte-identical.
  </action>
  <verify>
    <automated>test "$(grep -c '^version = "0.7.1"$' pyproject.toml)" -ge 1 && test "$(grep -c '__version__ = "0.7.1"' dnallm/version.py)" -ge 1 && test "$(grep -cF '## [0.7.1] - 2026-10-08' CHANGELOG.md)" -ge 1 && test "$(grep -cF '## [0.7.0] - 2026-10-07' CHANGELOG.md)" -ge 1 && PYDIFF="$(git diff -U0 -- pyproject.toml)" && test -z "$(printf '%s' "$PYDIFF" | grep -E '^[+-][^+-]' | grep -vE '^[+-]version = \"0\.7\.[01]\"$')"</automated>
  </verify>
  <done>
    All three surfaces read 0.7.1 (anchored grep hits on pyproject.toml:3, dnallm/version.py:1, and the new CHANGELOG header); the 0.7.0 changelog header is still present; the only changed content lines in the pyproject.toml diff are the `[project]` version line pair — proving the dependency floors on lines 30/33/40 and every other line are untouched.
  </done>
</task>

<task type="auto">
  <name>Task 2: Owner-rule verification, trailer-free commit, push to dev</name>
  <files>pyproject.toml, dnallm/version.py, CHANGELOG.md</files>
  <precondition>origin remote is reachable for push (recent pushes to dev succeeded once the 2026-10-07 GitHub receive 500 outage cleared; retry transient 5xx once before treating as blocked).</precondition>
  <action>
    1. Run the owner-rule verification trio, all from the repo root (both commands and the subset are planning-time grounded):
       - Import assertion (exercises the re-export at dnallm/__init__.py:20): `python -c "import dnallm; assert dnallm.__version__ == '0.7.1'"`
       - Version-sync assertion between the edited pyproject and the package: `python -c "import tomllib, dnallm; assert tomllib.load(open('pyproject.toml','rb'))['project']['version'] == dnallm.__version__ == '0.7.1'"`
       - Fast pytest subset (~10s total): `python -m pytest tests/test_extras_guard.py tests/utils/test_sequence.py -q` — extras_guard parses the pyproject extras so passing it proves the edited file still parses; the sequence module adds a fast behavior lane. This satisfies the owner rule that a dnallm/ code change (version.py) ships with pytest in the same change.
    2. Commit by staging exactly the three paths — `git add pyproject.toml dnallm/version.py CHANGELOG.md` — never `git add -A` or `git add .`: the untracked .planning runtime artifacts (.planning/graphs/, .planning/state.json, .planning/tmp/) must never be staged. Message: the single line `chore: bump version to 0.7.1` with an empty body — no attribution trailers of any kind, per owner rule.
    3. Push: `git push origin dev`. Do not tag (the v0.7.1 three-part semver product tag is applied at release time after PR #40 merges, outside this task) and do not push any other branch.
  </action>
  <verify>
    <automated>python -c "import dnallm; assert dnallm.__version__ == '0.7.1'" && python -c "import tomllib, dnallm; assert tomllib.load(open('pyproject.toml','rb'))['project']['version'] == dnallm.__version__ == '0.7.1'" && python -m pytest tests/test_extras_guard.py tests/utils/test_sequence.py -q && ST="$(git status --porcelain -- pyproject.toml dnallm/version.py CHANGELOG.md)" && test -z "$ST" && MSG="$(git log -1 --format=%B)" && ! printf '%s' "$MSG" | grep -qiE 'co-authored|generated with' && UNPUSHED="$(git rev-list origin/dev..HEAD)" && test -z "$UNPUSHED"</automated>
  </verify>
  <done>
    The import assertion and the pyproject↔package sync assertion both pass at 0.7.1; the fast pytest subset (5 + 8 tests) passes in the same change as the version.py edit; the three files are fully committed (empty pathspec-scoped status) in a single trailer-free commit; `git rev-list origin/dev..HEAD` is empty, proving origin/dev carries the bump — dev HEAD is now the v0.7.1 release tag target pending PR #40.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| local repo → origin/dev push | the only boundary crossed; rides the pre-configured authenticated git remote — no new credentials, APIs, or user input are introduced by this plan |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-261007-XBZ-01 | Tampering | version carriers (pyproject.toml `[project]` version, dnallm/version.py) | low | mitigate | the exactly-2-content-line pyproject diff gate plus the tomllib sync assertion pin both carriers to 0.7.1 — a naive global replace that also rewrote the dependency floors sharing the digit string (lines 30/33/40) fails the diff-count gate before commit |
| T-261007-XBZ-02 | Tampering | origin/dev push | low | accept | push uses the already-configured authenticated remote (recent pushes succeeded post-outage); the plan introduces no secrets, installs no packages, and adds no code paths — the trailer-free-message and explicit-pathspec-staging steps keep the release commit exactly the three declared files |

No packages are installed and no code paths are added by this plan, so the package-legitimacy gate and the reserved `T-{phase}-SC` row do not apply.
</threat_model>

<verification>
- `python -c "import dnallm; assert dnallm.__version__ == '0.7.1'"` passes, and the tomllib sync assertion proves pyproject `[project]` version == `dnallm.__version__` == 0.7.1.
- `python -m pytest tests/test_extras_guard.py tests/utils/test_sequence.py -q` passes (~10s grounded) — owner rule satisfied for the version.py edit; extras_guard additionally proves the edited pyproject.toml still parses.
- pyproject.toml diff's only changed content lines are the `[project]` version line pair (nothing else in the file moves); CHANGELOG.md gains the one new 0.7.1 section with the 0.7.0-and-below history byte-identical.
- Exactly one release commit on dev containing only the three files (explicit pathspec staging — no .planning runtime artifacts staged), single-line trailer-free message, pushed: `git rev-list origin/dev..HEAD` is empty.
- Post-push (advisory observation): dev CI workflow re-runs on the new HEAD; CI was fully green at dev@56f6ad5 so only the version-string delta rides this run.
</verification>

<success_criteria>
- dnallm reports 0.7.1 from both version carriers, in sync; CHANGELOG documents the post-0.7.0 stabilization in the established Keep-a-Changelog style.
- Fast pytest subset green in the same change; dependency floors and historical changelog entries byte-identical.
- Trailer-free release commit pushed to origin/dev; dev HEAD is the v0.7.1 release tag target once PR #40 merges (tagging itself is out of scope).
</success_criteria>

<output>
Create `.planning/quick/261007-xbz-bump-package-version-0-7-0-to-0-7-1-pypr/261007-xbz-SUMMARY.md` when done
</output>
