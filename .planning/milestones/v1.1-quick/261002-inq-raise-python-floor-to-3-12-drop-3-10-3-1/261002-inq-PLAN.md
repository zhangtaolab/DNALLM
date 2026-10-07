---
quick_task: 261002-inq-raise-python-floor-to-3-12-drop-3-10-3-1
type: execute
wave: 1
depends_on: []
branch: phs
push: manual-only   # NEVER push — owner instruction; commits only
autonomous: true
files_modified:
  - pyproject.toml
  - dnallm/version.py
  - README.md
  - CONTRIBUTING.md
  - .claude/CLAUDE.md
  - CHANGELOG.md
  - docs/faq/index.md
  - docs/getting_started/installation.md
  - docs/getting_started/quick_start.md
  - docs/user_guide/getting_started.md
  - .github/workflows/ci.yml
  - .github/workflows/docs-validation.yml
  - .github/workflows/README.md
estimate:
  tokens: 48000
  raw_tokens: 30000
  tasks: 2
  confidence: med
must_haves:
  truths:
    - "pip/uv on Python 3.10/3.11 refuse to install dnallm 0.7.0 (requires-python = \">=3.12\"; trove classifiers list 3.12/3.13 only — floor is 3.12, NOT 3.13)"
    - "Version is 0.7.0 in BOTH pyproject.toml and dnallm/version.py (breaking change gets a minor bump per owner)"
    - "CHANGELOG [0.7.0] records the breaking drop of Python 3.10/3.11 and the checker unification"
    - "No README/CONTRIBUTING/.claude/CLAUDE.md/docs page still states a 3.10 or 3.11 floor"
    - "mypy python_version, ruff target-version, and a new [tool.ty] section all target Python 3.12"
    - "No CI job runs on Python below 3.12; ci.yml test matrix is ['3.12','3.13'] with the numpy 1.26.4/2.2.0 pairing logic untouched"
    - "ruff format --check . and ruff check . exit 0 under py312 target; fast pytest lane (-m 'not slow') passes; mypy completes without the numpy-stub parse abort"
  artifacts:
    - pyproject.toml (floor, classifiers, version 0.7.0, ruff py312, mypy 3.12, new [tool.ty] python-version 3.12)
    - dnallm/version.py (__version__ = "0.7.0")
    - CHANGELOG.md (new ## [0.7.0] - 2026-10-02 section)
    - .github/workflows/ci.yml (matrix 3.12/3.13; test-cuda/test-mamba/deploy legs off 3.11)
    - .github/workflows/docs-validation.yml (3.11 → 3.12)
  key_links:
    - "pyproject requires-python >=3.12  ↔  every Python leg in .github/workflows/ (ci.yml, docs-validation.yml, publish.yml) — any leg below 3.12 fails at `uv pip install -e .`"
    - "pyproject version ↔ dnallm/version.py ↔ CHANGELOG [0.7.0] — all three must say 0.7.0"
    - "ruff target-version py312 ↔ CI ruff format --check / ruff check steps — repo must stay clean after repo-wide UP re-lint"
    - "mypy python_version 3.12 ↔ numpy stubs in .venv — the PEP 695 stub parse abort must be gone"
---

# Quick Task: Raise Python floor to 3.12, drop 3.10/3.11 (v0.6.0 → v0.7.0)

<precondition>
Execution gate (owner, binding) — assert ALL before starting; HALT and report if any fails:

1. Phase 5 gap-closure executors fully finished:
   test -f .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-05-SUMMARY.md
   test -f .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-06-SUMMARY.md
   (Both must exist. Flipping mypy/ruff config mid-execution flips pre-commit hooks
   for in-flight 05-05/05-06 executors.)
2. git branch --show-current == phs
3. Local env ready for verification: .venv/bin/python reports >= 3.12 and
   .venv/bin/{pytest,ruff,mypy} exist. If the venv is missing or below 3.12:
   uv venv && uv pip install -e ".[base]" (the local dev venv is Python 3.13.15, which
   satisfies the new floor).
</precondition>

<objective>
Raise the supported Python floor from 3.10 to 3.12 (owner-approved BREAKING change:
pip/uv on 3.10/3.11 will refuse to install dnallm), bump 0.6.0 → 0.7.0, and unify all
three checkers (mypy / ruff / ty) plus every CI leg on Python 3.12.

Purpose: kill the mypy numpy PEP 695 stub parse crash at the root (python_version 3.10
cannot parse modern numpy stubs) and make 3.12+ syntax (PEP 695 type parameters, PEP 692
TypedDict kwargs, etc.) legal across dnallm/ — record this side benefit in CHANGELOG.

Output: two commits on branch phs (never pushed):
  1. support-surface commit (floor + classifiers + version bump + docs + CHANGELOG)
  2. checker/CI alignment commit (mypy/ruff/ty + ci.yml + workflows docs + ruff fallout)

Owner checklist traceability:
- Item 1 (pyproject floor/classifiers)  → Task 1
- Item 2 (mypy/ruff/ty unified on 3.12) → Task 2
- Item 3 (CI matrix 3.12/3.13, numpy pairing untouched) → Task 2
- Item 4 (README/CONTRIBUTING/.claude/CLAUDE.md + CHANGELOG + version 0.7.0) → Task 1 (+ Task 2 appends toolchain line to CHANGELOG)
- Item 5 (commit structure, execution gate, no-push, PEP 695 note) → precondition + per-task commit steps
</objective>

<context>
@pyproject.toml
@.github/workflows/ci.yml
@.github/workflows/docs-validation.yml
@CHANGELOG.md
@dnallm/version.py
@README.md
@CONTRIBUTING.md
@.claude/CLAUDE.md
@docs/faq/index.md
@docs/getting_started/installation.md
@docs/getting_started/quick_start.md
@docs/user_guide/getting_started.md
</context>

<rules>
- NEVER run `git push` — branch phs is manual-push-only (owner).
- Commit messages: no attribution trailers of any kind (owner standing rule).
- Do NOT edit anything under docs/example/ — it is a byte-identical mirror of example/
  enforced by scripts/check_docs_sync.py in CI. (Verified: no 3.10 mentions live there;
  the four editable docs pages are docs/faq/index.md, docs/getting_started/installation.md,
  docs/getting_started/quick_start.md, docs/user_guide/getting_started.md.)
- Do NOT install or invoke the `ty` binary. The ty change is config-only (a [tool.ty]
  section in pyproject.toml); no package install happens in this task.
- numpy matrix pairing (1.26.4 / 2.2.0 and its install logic in ci.yml) is explicitly
  UNTOUCHED per owner ("1.26 does not ship 3.13 wheels; keep the pairing rules").
- All commands below assume cwd at the repo checkout root.
</rules>

<tasks>

<task type="auto">
  <name>Task 1: Support-surface commit — floor 3.12, classifiers, v0.7.0, docs, CHANGELOG</name>
  <files>pyproject.toml, dnallm/version.py, README.md, CONTRIBUTING.md, .claude/CLAUDE.md, CHANGELOG.md, docs/faq/index.md, docs/getting_started/installation.md, docs/getting_started/quick_start.md, docs/user_guide/getting_started.md</files>
  <action>
  Single commit containing ONLY the support-surface decision (owner: do not mix with checker/CI changes):

  pyproject.toml:
  - Line 6: requires-python = ">=3.10" → ">=3.12". The floor is 3.12, NOT 3.13.
  - Classifiers: delete "Programming Language :: Python :: 3.10" and ":: 3.11" (lines 17-18); keep :: 3.12 and :: 3.13 (plus the base ":: 3" line).
  - Line 3: version = "0.6.0" → "0.7.0" (breaking change on 0.x gets a minor bump per owner).
  - Do NOT touch [tool.ruff] / [tool.mypy] / CI in this commit — that is Task 2 (owner commit-structure rule).

  dnallm/version.py: __version__ = "0.6.0" → "0.7.0".

  README.md:
  - Line 7 badge: [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)] → [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)] (both label and badge URL).
  - Line 71: "Python 3.11 or higher (recommended)" → "Python 3.12 or higher (recommended)".

  CONTRIBUTING.md line 26: "Python 3.10 or higher" → "Python 3.12 or higher".

  .claude/CLAUDE.md — update every SUPPORTED-PYTHON statement (leave ruff/mypy tool-config references for Task 2):
  - Line 14 (Compatibility constraint): "(Python 3.11/3.12/3.13, numpy 1.26.4 & 2.2.0)" → "(Python 3.12/3.13, numpy 1.26.4 & 2.2.0)".
  - Line 26: → "Python 3.12+ (requires-python `>=3.12`) - entire codebase; classifiers declare 3.12/3.13".
  - Line 33: "Python 3.11+ recommended for dev" → "Python 3.12+ recommended for dev" (keep the venv 3.13.15 note).
  - Line 34: "CI matrix tests Python 3.11 / 3.12 / 3.13" → "CI matrix tests Python 3.12 / 3.13".
  - Line 95: "Python 3.11+ with venv or conda" → "Python 3.12+ with venv or conda".
  - Line 111: change ONLY the floor parts: "Python **3.10+**" → "Python **3.12+**", `requires-python = ">=3.10"` → `">=3.12"`, "CI tests 3.11/3.12/3.13" → "CI tests 3.12/3.13". Leave `ruff target-version = "py310"` and `mypy python_version = "3.10"` verbatim — those stay factually true until Task 2 lands (transient by design; nothing is pushed between commits).

  CHANGELOG.md: insert a new section directly under the header (above ## [0.6.0]):
  "## [0.7.0] - 2026-10-02" with:
  "### Removed" containing: "- **BREAKING**: Dropped Python 3.10 and 3.11 support. `requires-python` is now `>=3.12`; pip/uv on 3.10/3.11 refuse to install dnallm 0.7.0+. Trove classifiers now list 3.12/3.13 only."
  (Task 2 will append the toolchain-unification lines under this same 0.7.0 section.)

  docs pages (the four original, non-mirror pages only):
  - docs/getting_started/installation.md:7: "Python 3.10 or higher (Python 3.13 recommended)" → "Python 3.12 or higher (Python 3.13 recommended)".
  - docs/getting_started/quick_start.md:7: "Python 3.10 or higher (Python 3.12 recommended)" → "Python 3.12 or higher (Python 3.13 recommended)".
  - docs/user_guide/getting_started.md:21: "Python 3.10 or higher" → "Python 3.12 or higher".
  - docs/faq/index.md:250: "Python: 3.10 or higher (Python 3.12 recommended)" → "Python: 3.12 or higher (Python 3.13 recommended)".

  Commit (only the files above): feat!: require Python >=3.12, drop 3.10/3.11 support (v0.7.0)
  </action>
  <verify>
    <automated>
grep -n 'requires-python = ">=3.12"' pyproject.toml && \
! grep -n 'Programming Language :: Python :: 3.1[01]' pyproject.toml && \
[ "$(grep -c 'Programming Language :: Python :: 3.1[23]' pyproject.toml)" -eq 2 ] && \
grep -n '^version = "0.7.0"' pyproject.toml && \
grep -n '__version__ = "0.7.0"' dnallm/version.py && \
grep -n '^## \[0.7.0\] - 2026-10-02' CHANGELOG.md && \
[ "$(sed -n '/## \[0.7.0\]/,/## \[0.6.0\]/p' CHANGELOG.md | grep -ci 'dropped python 3.10')" -ge 1 ] && \
! grep -rnE '3\.10\+|3\.10 or higher|>=3\.10|Python 3\.1[01]|python-3\.1[01]|3\.11 or higher|3\.11/' \
  README.md CONTRIBUTING.md .claude/CLAUDE.md \
  docs/faq/index.md docs/getting_started/installation.md docs/getting_started/quick_start.md docs/user_guide/getting_started.md && \
echo TASK1-VERIFIED
    </automated>
    <fails_when>
- The requires-python grep prints nothing (floor not raised or typo'd), or the negated
  classifier grep prints a surviving "Programming Language :: Python :: 3.10/3.11" line.
- The 3.12/3.13 classifier count is not exactly 2 (floor accidentally raised to 3.13, or a
  classifier was dropped).
- Either version string is not 0.7.0 (pyproject and dnallm/version.py must both bump).
- The CHANGELOG [0.7.0] section or its BREAKING/dropped-3.10 line is missing.
- The final negative grep prints ANY match: a remaining "3.10+", "3.10 or higher",
  ">=3.10", "Python 3.10/3.11", "python-3.10/3.11" badge, "3.11 or higher", or "3.11/"
  floor statement in the scoped docs. (Note: `python_version = "3.10"` and `py310` in
  .claude/CLAUDE.md line 111 are EXPECTED to remain until Task 2 — none of the gate
  patterns match them.)
    </fails_when>
  </verify>
  <done>
Support-surface commit landed on phs: requires-python ">=3.12", classifiers 3.12/3.13
only, version 0.7.0 in pyproject.toml AND dnallm/version.py, CHANGELOG [0.7.0] BREAKING
entry present, and zero remaining 3.10/3.11 floor statements across README.md,
CONTRIBUTING.md, .claude/CLAUDE.md, and the four in-scope docs pages. Nothing pushed.
  </done>
</task>

<task type="auto">
  <name>Task 2: Checker + CI alignment commit — mypy/ruff/ty on 3.12, CI legs, ruff fallout</name>
  <files>pyproject.toml, .github/workflows/ci.yml, .github/workflows/docs-validation.yml, .github/workflows/README.md, .claude/CLAUDE.md, CHANGELOG.md, dnallm/** (only files touched by ruff --fix py312 UP fallout, if any)</files>
  <precondition>Task 1 committed; still on branch phs; nothing pushed.</precondition>
  <action>
  Single alignment commit (plus an optional isolated style follow-up commit if fallout is large):

  pyproject.toml — checker targets:
  - [tool.ruff]: comment line "# Assume Python 3.10+" → "# Assume Python 3.12+"; target-version = "py310" → "py312".
  - [tool.mypy]: python_version = "3.10" → "3.12". This is the fix that kills the numpy
    PEP 695 stub parse abort (mypy parsing numpy's 3.12-syntax stubs under a 3.10 grammar).
  - Add a NEW [tool.ty] configuration (the repo has none today — owner checklist item 2):
    a table heading [tool.ty.environment] with python-version = "3.12" directly after the
    [tool.mypy] overrides block. ty's schema nests python-version under the environment
    table; this fulfills the owner's "add [tool.ty] python-version = 3.12" item. Config
    only — do not install or run ty.

  .github/workflows/ci.yml — move every Python leg to >= 3.12 (after the Task 1 floor
  raise, any 3.11 leg dies at `uv pip install -e` with a resolver error):
  - Line 28 test matrix: python-version: ['3.11', '3.12', '3.13'] → ['3.12', '3.13'].
    numpy-version: ['1.26.4', '2.2.0'] and the "Install specific numpy version" step
    logic stay EXACTLY as they are (owner: pairing rules unchanged).
  - Line 205 test-cuda matrix: ['3.11'] → ['3.12'].
  - Line 282 test-mamba matrix: ['3.11'] → ['3.12'].
  - Line 514 deploy job: python-version: 3.11 → 3.12.
  - test-windows / coverage-gate / coverage-nightly already run 3.12 — leave untouched.

  .github/workflows/docs-validation.yml lines 20/23: "Set up Python 3.11" and
  python-version: "3.11" → 3.12 (same install-failure rationale; this workflow was born
  in Phase 5 and currently runs below the new floor).

  .github/workflows/README.md: update the Python-version mentions (around lines 25, 55,
  61, 71, 76, 137): the matrix list "3.11, 3.12, 3.13" → "3.12, 3.13" and every
  "Python 3.11" reference → "Python 3.12". Only touch Python-version statements — the
  broader workflows-README staleness (ship-triage WR-03) is out of scope.

  .claude/CLAUDE.md — remaining tool-config references:
  - Line 49: "target py310" → "target py312".
  - Line 111 parenthetical: ruff `target-version = "py310"` → "py312" and mypy
    `python_version = "3.10"` → "3.12"; optionally note the new ty config so all three
    checkers are described consistently.

  Repo-wide ruff pass under the new target (expect zero-to-few changes; the codebase is
  already modern — 310 uses of `X | None` vs 3 of Optional):
  - Run .venv/bin/ruff check . --fix, then .venv/bin/ruff format .
  - Review git diff. Any dnallm/ fallout is auto-style pyupgrade fixes — no new tests
    required (owner rule; the behavior-suite gate below still must stay green).
  - If fallout touches more than ~10 files, isolate it as a separate follow-up commit
    (style: apply ruff py312 pyupgrade fixes) instead of mixing into the alignment commit.

  CHANGELOG.md: append under the existing ## [0.7.0] section (new "### Changed" list):
  toolchain unified on Python 3.12 — mypy python_version, ruff target-version, and a new
  [tool.ty] python-version all target 3.12 (resolves the mypy numpy-stub PEP 695 parse
  failure); CI matrix reduced to 3.12/3.13 with the numpy 1.26.4/2.2.0 pairing unchanged;
  dnallm/ source may now use Python 3.12+ syntax (PEP 695 type parameters, PEP 692
  TypedDict **kwargs, f-string improvements, etc.). This records the owner-requested
  side benefit.

  Commit: chore: align mypy/ruff/ty and CI on Python 3.12
  (+ optional separate style commit per the rule above). Never push.
  </action>
  <verify>
    <automated>
grep -n 'target-version = "py312"' pyproject.toml && \
grep -n 'python_version = "3.12"' pyproject.toml && \
grep -A2 '\[tool\.ty' pyproject.toml | grep 'python-version = "3.12"' && \
grep -n "python-version: \['3.12', '3.13'\]" .github/workflows/ci.yml && \
! grep -rnE '3\.1[01]|py310' pyproject.toml .github/workflows/ci.yml .github/workflows/docs-validation.yml .github/workflows/README.md .claude/CLAUDE.md README.md CONTRIBUTING.md && \
.venv/bin/ruff format --check . && .venv/bin/ruff check . --statistics && \
{ out=$(.venv/bin/mypy dnallm/ --show-error-codes --exclude=dnallm/tasks/metrics/ 2>&1); \
  echo "$out" | tail -3; \
  echo "$out" | grep -qiE 'traceback|internal error' && exit 1; \
  echo "$out" | grep -qE 'numpy[^ ]*\.pyi.*(error|invalid syntax)' && exit 1; \
  echo "$out" | grep -qE '^(Success: no issues found|Found [0-9]+ error)' || exit 1; \
  echo MYPY-COMPLETES; } && \
echo ALIGNMENT-VERIFIED
    </automated>
    <automated>
.venv/bin/python -m pytest -m "not slow" -q -p no:cacheprovider -p no:progress
    </automated>
    <fails_when>
- Any config grep prints nothing: ruff target not py312, mypy python_version not 3.12,
  or the [tool.ty] section lacks python-version = "3.12".
- The ci.yml matrix grep misses (matrix not reduced to ['3.12', '3.13']), or the negative
  grep prints ANY surviving "3.10"/"3.11"/"py310" token in pyproject.toml, any workflow
  file, workflows README, .claude/CLAUDE.md, README.md, or CONTRIBUTING.md (e.g. a
  forgotten test-cuda/test-mamba/deploy/docs-validation leg still on 3.11, or the
  "# Assume Python 3.10+" comment).
- ruff format --check or ruff check exits nonzero (format drift or lint findings under
  the py312 target — run the --fix/format pass again).
- mypy block: output contains "Traceback"/"INTERNAL ERROR", or an error line pointing at
  a numpy .pyi stub (the pre-change parse abort reproduces), or no terminal summary line
  ("Success: no issues found" / "Found N errors") meaning mypy aborted instead of
  completing. Ordinary pre-existing type errors do NOT fail this gate — mypy is advisory
  in CI (|| true); only the parse abort matters here.
- pytest exits nonzero: any fast-lane test fails or errors after the floor raise and any
  ruff pyupgrade fallout (regression — revert the offending style fix if it changed
  behavior, style fixes must be behavior-neutral).
    </fails_when>
  </verify>
  <done>
Alignment commit(s) landed on phs: mypy python_version 3.12, ruff target-version py312,
new [tool.ty.environment] python-version 3.12; ci.yml matrix ['3.12','3.13'] with numpy
pairing untouched and zero Python-3.11 legs anywhere in .github/workflows/ (including
docs-validation.yml and the workflows README); repo clean under ruff format --check and
ruff check at py312; mypy runs to completion with no numpy-stub parse abort; fast pytest
lane (-m "not slow") fully green; CHANGELOG 0.7.0 documents the unification and the
PEP 695 eligibility benefit. Nothing pushed.
  </done>
</task>

</tasks>

<threat_model>
Quick-task namespace T-QT-NN (no phase plans share this prefix).

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-QT-01 | Tampering / supply chain | ty checker config | medium | mitigate | Config-only: [tool.ty] python-version added to pyproject.toml; plan explicitly forbids installing/invoking the ty binary, so no unvetted package executes (package-legitimacy gate stays moot) |
| T-QT-02 | Denial of Service (CI) | CI legs vs requires-python | high | mitigate | After the 3.12 floor, every remaining 3.11 leg (test-cuda, test-mamba, deploy, docs-validation) hard-fails at install; Task 2 moves all of them and the negative grep gate (3\.1[01] across workflows) proves none survive |
| T-QT-03 | Tampering (repo integrity) | docs/example mirror | medium | mitigate | docs/example/ is a byte-identical mirror of example/ enforced by scripts/check_docs_sync.py; plan rules forbid editing it, and the floor-mention greps scope only the four original docs pages |
</threat_model>

<verification>
After both commits (still unpushed, on phs), the full end-state holds:
1. grep -n 'requires-python = ">=3.12"' pyproject.toml — floor raised; classifiers 3.12/3.13 only.
2. grep -rnE '3\.1[01]|py310' pyproject.toml .github/workflows/ README.md CONTRIBUTING.md .claude/CLAUDE.md — empty.
3. grep -rnE '3\.10|Python 3\.11|3\.11 or higher' README.md CONTRIBUTING.md .claude/CLAUDE.md docs/faq docs/getting_started docs/user_guide --include='*.md' — empty.
4. Version triangle consistent: pyproject 0.7.0 = dnallm/version.py 0.7.0 = CHANGELOG [0.7.0].
5. numpy matrix line unchanged: grep -n "numpy-version: \['1.26.4', '2.2.0'\]" .github/workflows/ci.yml still hits.
6. Both per-task automated gates green (config greps, ruff clean, mypy completes, fast pytest lane).
7. git log shows exactly the support-surface commit first, checker/CI alignment after — not mixed.
</verification>

<success_criteria>
- pip/uv on Python 3.10/3.11 will refuse to install dnallm 0.7.0 (declared floor, enforced by metadata).
- All three checkers (mypy, ruff, ty) target Python 3.12; the mypy numpy-stub parse abort is gone.
- CI runs Python 3.12/3.13 only, with the owner-mandated numpy pairing logic untouched.
- Every user-facing doc (README, CONTRIBUTING, .claude/CLAUDE.md, docs pages, workflows README) states the 3.12+ floor.
- Two clean commits on phs (support surface, then alignment), never pushed, no attribution trailers.
- Fast pytest lane green; ruff clean; zero behavior changes intended (any dnallm/ diff is auto-style pyupgrade only).
</success_criteria>

<output>
Create .planning/quick/261002-inq-raise-python-floor-to-3-12-drop-3-10-3-1/261002-inq-SUMMARY.md when done
(commits landed, gate evidence, any ruff fallout observed, confirmation nothing was pushed).
</output>
