---
phase: quick-261007-mhz
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - pyproject.toml
  - tests/test_extras_guard.py
autonomous: true
requirements:
  - QUICK-261007-MHZ-01
user_setup: []

estimate:
  tokens: 13000
  raw_tokens: 9000
  tasks: 1
  confidence: high

must_haves:
  truths:
    - "The notebook extra declares pygenometracks with a platform_system != 'Windows' marker, so Windows resolution skips the member entirely (no pybigwig sdist build attempt) while Linux keeps installing it"
    - "tests/test_extras_guard.py EXPECTED_NOTEBOOK_MEMBERS carries the same marker-qualified literal, so the guard suite stays green after the declaration edit"
    - "The base meta-extra dnallm[dev,test,notebook,mcp] resolves on Windows (test-windows leg install step at .github/workflows/ci.yml no longer dies in pyBigWig setup.py)"
    - "Linux resolution still includes pygenometracks (ubuntu legs and the self-hosted example-nightly, which install .[base] variants on Linux, are unaffected)"
  artifacts:
    - "pyproject.toml notebook-extra member string pygenometracks>=3.9; platform_system != 'Windows' (was bare pygenometracks>=3.9) with the adjacent comment block extended to record the Windows rationale"
    - "tests/test_extras_guard.py EXPECTED_NOTEBOOK_MEMBERS literal updated to the identical marker-qualified string"
  key_links:
    - "ci.yml test-windows 'uv pip install -e \".[base]\"' -> base meta-extra (pyproject.toml) -> notebook extra -> marker-qualified pygenometracks (requirement inactive under platform_system=Windows, so the transitive pybigwig 0.3.26 sdist never enters the Windows resolve)"
    - "tests/test_extras_guard.py test_pre_existing_members_preserved set-difference <-> tomllib-parsed notebook extra (both literals must move in the same commit or the guard goes red)"
---

<objective>
Fix the Windows CI leg (test-windows, py3.12) install failure: the base meta-extra `dnallm[dev,test,notebook,mcp]` transitively pulls `pygenometracks>=3.9` from the notebook extra, which depends on pybigwig 0.3.26 — a package with no Windows wheel whose sdist build dies in pyBigWig setup.py with `AttributeError: 'NoneType' object has no attribute 'split'`. The fix gates the pygenometracks member behind a `platform_system != 'Windows'` marker (exact in-file precedent: the dev-extra pybedtools member) and moves the extras-guard literal in the same change so the guard suite stays green.

Purpose: The Windows leg has claimed "Operating System :: OS Independent" coverage since 08-09 and currently fails at the dependency-install step before any test runs. `dnallm` package code never imports pygenometracks (verified live: zero `import pygenometracks`/`import pyBigWig` matches in `dnallm/` and `tests/`), so omitting it on Windows removes only the broken build, not any functionality. Linux legs and example-nightly install `.[base]` variants on Linux where the marker is a no-op.

Output: A two-line declaration change (pyproject.toml notebook extra + tests/test_extras_guard.py EXPECTED_NOTEBOOK_MEMBERS) with an extended rationale comment, proven by the guard test, a PEP 508 marker-semantics assertion against the parsed TOML, and a Linux dry-run resolve that still installs pygenometracks.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@pyproject.toml
@tests/test_extras_guard.py

Live-tree facts observed at planning time (2026-10-07, branch phs):

- pyproject.toml `[project.optional-dependencies]`: notebook extra spans lines 101-114; the target member is `"pygenometracks>=3.9",` at line 113 with a two-line GPL-3.0 comment above it (lines 111-112). The marker precedent is the dev-extra member `"pybedtools>=0.11.0; platform_system != 'Windows'",` at line 88 — copy that exact form (single quotes around Windows, semicolon-separated marker).
- pyproject.toml:137 defines `base = ["dnallm[dev,test,notebook,mcp]", ...]`; .github/workflows/ci.yml test-windows installs `uv pip install -e ".[base]"` (job at line 129, install at line 165) — that is the failing chain.
- tests/test_extras_guard.py: `EXPECTED_NOTEBOOK_MEMBERS` frozenset spans lines 50-55 with the literal `"pygenometracks>=3.9",` at line 54; `TestNotebookExtraMembers.test_pre_existing_members_preserved` (lines 103-107) does `EXPECTED_NOTEBOOK_MEMBERS - set(notebook)` over tomllib-parsed members, so the literal must match the full marker-qualified string or the test fails. The module docstring (line 11) and the ipython-pin assertion message (line 98) also mention `pygenometracks>=3.9` as prose — they are NOT compared against parsed members and must NOT be edited.
- Baseline proof on this box: `.venv/bin/python -m pytest tests/test_extras_guard.py -q` = 5 passed in 0.66s; pygenometracks 3.9 + pyBigWig 0.3.26 are installed in the project venv; `Marker("platform_system != 'Windows'")` evaluates False under `platform_system='Windows'` and True under `'Linux'`; `uv pip install --dry-run --reinstall-package pygenometracks -e ".[base]"` resolves clean on Linux and lists pygenometracks (a plain dry-run does NOT list it here because it is already satisfied — the `--reinstall-package` flag is load-bearing for the grep).
- Out of scope (no edits): .github/workflows/ci.yml (the marker fixes the install at the requirement level; nothing else in the Windows leg references pgt), the `ipython>=8.31,<9` pin (ipython ships Windows wheels), and any notebook/mirror content.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Gate pygenometracks behind the non-Windows platform marker (pyproject declaration + guard literal, one commit)</name>
  <files>pyproject.toml, tests/test_extras_guard.py</files>
  <action>
    In pyproject.toml, inside the notebook extra of `[project.optional-dependencies]`, change the member at line 113 from `pygenometracks>=3.9` to `pygenometracks>=3.9; platform_system != 'Windows'` — character-for-character the marker form of the dev-extra precedent `pybedtools>=0.11.0; platform_system != 'Windows'` at line 88 (semicolon separator, single quotes around Windows, marker after the version spec). Extend the GPL-3.0 comment block directly above the member (currently lines 111-112) with two or three comment lines recording why Windows is excluded: pygenometracks pulls pybigwig, which has no Windows wheel, and its sdist setup.py fails with AttributeError 'NoneType' object has no attribute 'split' on the test-windows leg installing .[base] (observed 2026-10-07); dnallm package code never imports pygenometracks so the Windows omission is behavior-safe. Keep every other notebook-extra member (jupyter, marimo, nbclient, ipython pin) and its comment untouched.

    In tests/test_extras_guard.py, inside the `EXPECTED_NOTEBOOK_MEMBERS` frozenset (lines 50-55), change the literal at line 54 from `pygenometracks>=3.9` to the identical marker-qualified string `pygenometracks>=3.9; platform_system != 'Windows'` so `test_pre_existing_members_preserved` compares the full requirement string against the tomllib-parsed member. Update the frozenset's lead comment (lines 48-49) only if needed for tense — do not rewrite it. Do NOT touch the module docstring or the ipython-pin assertion message that name pygenometracks>=3.9 as prose; neither is compared against parsed TOML members.

    Both file edits land in ONE commit — the declaration change ships with its guard test in the same change per the owner rule (a pyproject-only commit leaves `test_pre_existing_members_preserved` red because the bare literal is no longer a subset of the marker-qualified members).
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/test_extras_guard.py -q && .venv/bin/python -c 'import pathlib, tomllib; from packaging.requirements import Requirement; members = tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["optional-dependencies"]["notebook"]; pgt = [m for m in members if m.startswith("pygenometracks")][0]; r = Requirement(pgt); assert r.marker is not None, "platform marker missing from pygenometracks member"; assert not r.marker.evaluate({"platform_system": "Windows"}), "requirement must be skipped on Windows"; assert r.marker.evaluate({"platform_system": "Linux"}), "requirement must stay active on Linux"; print("marker OK:", pgt)' && uv pip install --dry-run --reinstall-package pygenometracks -e ".[base]" 2>&1 | grep -i pygenometracks</automated>
  </verify>
  <done>
    tests/test_extras_guard.py reports 5 passed (guard literal and pyproject declaration agree bidirectionally for the marker-qualified member); the Requirement proof prints marker OK with the marker-qualified string, establishing that the member parses as valid PEP 508, is inactive under platform_system=Windows (the pybigwig sdist never enters the Windows resolve) and active under platform_system=Linux; the uv dry-run of .[base] on this Linux box still lists pygenometracks in the would-install set (ubuntu legs and example-nightly unaffected); grep of pyproject.toml shows the marker-qualified notebook-extra member; both edits are in a single commit.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| CI dependency resolution | Requirement strings in pyproject.toml decide what code the Windows/Linux runners download and build from PyPI |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-261007-01 | Tampering | pyproject.toml notebook-extra member string | low | mitigate | The extras guard (tests/test_extras_guard.py test_pre_existing_members_preserved) set-compares the member literal against tomllib-parsed TOML on every fast-lane run, so any silent drop or rewording of the marker-qualified member fails CI; the PEP 508 marker-semantics assertion in the verify step pins skip-on-Windows/active-on-Linux behavior |
| T-261007-02 | Denial of Service | Windows leg install step (pybigwig sdist build failure) | medium | mitigate | The platform marker removes pygenometracks (and transitively pybigwig) from the Windows resolve entirely, eliminating the sdist build that crashed pyBigWig setup.py |

No new packages are introduced by this plan (an existing member gains a platform restriction only), so the package-legitimacy gate and the reserved `T-{phase}-SC` row do not apply.
</threat_model>

<verification>
- `.venv/bin/python -m pytest tests/test_extras_guard.py -q` passes (5 tests) — proves the guard literal matches the edited declaration.
- The packaging.Requirements marker assertion (in the automated chain) proves the exact published member string is skipped under platform_system=Windows and active under platform_system=Linux.
- `uv pip install --dry-run --reinstall-package pygenometracks -e ".[base]"` resolves on Linux and lists pygenometracks — proves the Linux CI legs and example-nightly (which install `.[base]`-family extras on Linux) are unaffected.
- `grep -n "pygenometracks" pyproject.toml` shows exactly one member line, marker-qualified; `git diff --stat` shows only pyproject.toml and tests/test_extras_guard.py changed, in one commit.
- The Windows leg itself is proven on the next push to phs (test-windows runs on push/PR); no local Windows runtime exists on this box, so runner evidence closes the loop.
</verification>

<success_criteria>
- pyproject.toml notebook extra declares `pygenometracks>=3.9; platform_system != 'Windows'` with the adjacent comment recording the Windows rationale; the dev-extra pybedtools marker form is followed exactly.
- tests/test_extras_guard.py EXPECTED_NOTEBOOK_MEMBERS carries the identical marker-qualified literal; the guard suite is green locally (5 passed).
- Marker semantics proven both directions (skipped on Windows, active on Linux) against the parsed TOML member; Linux base-extra dry-run still installs pygenometracks.
- Single atomic commit containing both files; no other files touched.
</success_criteria>

<output>
Create `.planning/quick/261007-mhz-fix-windows-ci-leg-failure-pybigwig-sdis/261007-mhz-SUMMARY.md` when done
</output>
