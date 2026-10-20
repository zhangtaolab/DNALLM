---
phase: 261010-rpa
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - pyproject.toml
  - CHANGELOG.md
  - .github/workflows/ci.yml
  - README.md
  - .github/workflows/README.md
  - tests/utils/test_transformers_compat_np.py
autonomous: true
requirements:
  - DEPS-NUMPY2-FLOOR
estimate:
  tokens: 24000
  raw_tokens: 24000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "pyproject.toml [project] dependencies: the four lines of the pyarrow cap (3-line comment at lines 33-35 + the entry at line 36) are deleted; the numpy line reads \"numpy>=2.0.0\", preceded by exactly ONE new comment line noting the 2026-10-10 numpy 1.x retirement; grep -c pyarrow over pyproject.toml returns 0; requires-python stays exactly \">=3.10\" with the 3.10 classifier intact; NO numpy ceiling is added and no other dependency line moves."
    - ".github/workflows/ci.yml test job: matrix numpy-version is ['2.2.0'] (6 legs -> 3); the \"Install specific numpy version\" step keeps checkout of the venv but its body is the single unconditional uv pip install \"numpy==${{ matrix.numpy-version }}\" line — the 1.26.4 branch and its scipy reinstall are gone; grep for 1.26.4 over the whole file returns 0; the coverage-gate and coverage-nightly numpy pins (numpy==2.2.0, exactly 2 occurrences) are byte-unchanged; the workflow still parses as YAML."
    - "No doc claims the retired leg: grep -rn 1.26.4 over README.md and .github/ returns nothing; README.md line ~577 reads Python 3.11/3.12/3.13 x numpy 2.2.0 plus a Windows leg; .github/workflows/README.md line ~29 reads NumPy versions: 2.2.0."
    - "tests/utils/test_transformers_compat_np.py loses EXACTLY two tests — test_patch_noops_when_numpy_provides_fromstring (numpy 1.x no-op rung, was lines 35-48) and test_binary_mode_str_input_matches_historical_behavior (historical numpy 1.x binary-mode framing, was lines 136-147); the module docstring's discipline enumeration is reworded to the surviving covered set; the lane command (.venv/bin/python -m pytest tests/utils/test_transformers_compat_np.py tests/test_extras_guard.py -q) reports \"18 passed\" (13 + 5); dnallm/utils/transformers_compat.py is byte-unchanged — the shim itself stays."
    - "CHANGELOG.md [Unreleased] carries a Changed breaking entry: numpy floor >=2.0.0, pyarrow cap removed (retirement condition met), numpy 1.x users stay on the 0.8.x series."
    - "Exactly ONE commit on revision carries all six files — atomic by necessity: with the floor raised, a surviving 1.26.4 matrix leg would resolver-conflict at its pin step (uv pip install -e \".[base]\" resolves numpy>=2.0.0, then the pin forces 1.26.4), so pyproject.toml and ci.yml must move together; no attribution trailers; staging by explicit pathspec leaves the pre-existing untracked .planning/graphs/ and .planning/tmp/ out of the commit; after push, git rev-parse HEAD == git rev-parse origin/revision."
  artifacts:
    - pyproject.toml — numpy floor >=2.0.0 + pyarrow cap deletion (+ one retirement comment line).
    - CHANGELOG.md — breaking-change entry under ## [Unreleased].
    - .github/workflows/ci.yml — matrix reduced to ['2.2.0'] + simplified numpy install step.
    - README.md + .github/workflows/README.md — CI matrix claims updated to numpy 2.2.0 only.
    - tests/utils/test_transformers_compat_np.py — two numpy-1.x-scenario tests removed, docstring synced, shim behavior tests retained.
    - 261010-rpa-SUMMARY.md — lane outputs, diff evidence, commit hash, push state, CI run URL.
  key_links:
    - "pyproject numpy>=2.0.0 floor -> ci.yml matrix pin numpy==2.2.0: the pin satisfies the floor; any surviving 1.26.4 leg conflicts at its pin step — the single atomic commit IS this link."
    - "The pyarrow cap's own comment (\"Cap until that leg retires\") -> the matrix retirement in this same change: the documented trigger and the cap removal land together."
    - "ci.yml matrix -> README.md:577 and .github/workflows/README.md:29 doc claims: no doc is left asserting the retired leg."
---

<objective>
Retire numpy 1.x support: raise the numpy floor to >=2.0.0, remove the pyarrow cap whose
documented retirement condition (the numpy 1.26.4 CI matrix leg) is now met, shrink the CI
matrix accordingly, trim the two numpy-1.x-scenario shim tests, and record the breaking
change — with every reference to the retired leg consistent across pyproject, CI, and docs.

Purpose: numpy 1.x is dead weight the CI matrix pays 3 legs to protect; the upstream
evaluation proved the retirement safe, so the floor, the cap, the matrix, the tests, and
the docs all move in one atomic change.

Output: six edited files, one commit on revision (pushed), 261010-rpa-SUMMARY.md with
the evidence.

## Owner-approved facts (executor cites, does NOT re-derive)

- Upstream deep evaluation already completed: 8/8 uv resolution proofs across py3.10-3.13
  including the CI-pinned numpy==2.2.0 scenario and the full extras union (426 packages);
  runtime smoke of numpy 2.2 + pyarrow 26 + datasets 3.2 passed; zero of the 42 packages
  in the numpy-constraint audit pin numpy<2. No re-verification of resolves is wanted.
- Verified at planning (live observation, 2026-10-10, HEAD 5e7c561 on branch revision):
  pyproject.toml lines 33-36 hold the pyarrow cap comment + entry and line 52 holds the
  numpy floor exactly as quoted in the tasks; ci.yml line 46 is the matrix line and lines
  87-94 the install step exactly as quoted; README.md:577 and
  .github/workflows/README.md:29 carry the matrix claims (the workflows README line was
  found by live grep and is included as the same claim class); CHANGELOG.md has an empty
  ## [Unreleased] section; .venv runs numpy 2.5.3 and the lane currently reports 20 passed
  (drops to 18 after the two trims); .venv/bin/ruff exists; tests/expected_skips.yaml has
  NO numpy-conditioned entries; nothing else under tests/, dnallm/, or scripts/ references
  pyarrow or numpy 1.26; no external file references either trimmed test by name.
- OUT OF SCOPE (red line): the requires-python floor bump to >=3.11 is a separate owner
  decision — do NOT bundle it (requires-python, the 3.10 classifier, and the ruff/mypy
  py310 targets stay untouched). Adding a numpy CEILING is likewise not this task's
  approved diff (it is a ledger item from the v1.2 close, not a target here).
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@pyproject.toml
@.github/workflows/ci.yml
@tests/utils/test_transformers_compat_np.py
</context>

<!-- planner-discipline-allow: pyarrow 1.26.4 numpy&gt;=1.26.0 -->
<!-- (the literals above appear in actions ONLY as deletion targets; every negative
     gate below is file-scoped to the artifact being cleaned, never to the plan) -->

<tasks>

<task type="auto">
  <name>Task 1: pyproject numpy floor + pyarrow cap removal + CHANGELOG breaking entry</name>
  <files>pyproject.toml, CHANGELOG.md</files>
  <action>
    Two scoped edits in pyproject.toml, nothing else in the file moves:
    1) Delete the four lines at 33-36: the three comment lines ("# pyarrow is transitive
    via datasets; ...", "# runtime floor that breaks ...", "# leg retires; 25.0.1
    verified ...") and the dependency entry that caps pyarrow below 26. The dependency
    list then reads "datasets&lt;=3.2.0", directly followed by "einops&gt;=0.7.0",.
    2) Replace the numpy floor line (52) "numpy&gt;=1.26.0", with exactly two lines: a
    single comment line reading "# numpy 1.x retired 2026-10-10; 1.x users stay on the
    0.8.x series" and the entry "numpy&gt;=2.0.0",. One comment line only; both lines
    within 100 columns; keep list ordering otherwise byte-identical.
    Then in CHANGELOG.md add under ## [Unreleased] a "### Changed" section whose single
    entry reads: **Breaking**: numpy support floor raised from `&gt;=1.26.0` to
    `&gt;=2.0.0` (numpy 1.x retired 2026-10-10; the CI matrix now runs numpy 2.2.0 only)
    and the `pyarrow&gt;=15,&lt;26` cap removed — its documented retirement condition
    (the numpy 1.26.4 CI matrix leg) is met. Environments on numpy 1.x must stay on the
    0.8.x series. (The CHANGELOG is the one file that SHOULD still name the retired
    leg.) Do not touch requires-python, classifiers, extras, tool sections, or any other
    dependency line; do not add any numpy ceiling. Do not commit — Task 3 owns the single
    atomic commit.
  </action>
  <verify>
    <automated>.venv/bin/python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); deps=d['project']['dependencies']; assert 'numpy>=2.0.0' in deps, deps; assert not any('pyarrow' in x for x in deps), deps; assert d['project']['requires-python'] == '>=3.10', d['project']['requires-python']" && ! grep -q pyarrow pyproject.toml && ! grep -qF 'numpy>=1.26.0' pyproject.toml && grep -qF '"numpy>=2.0.0",' pyproject.toml && grep -A4 '^## \[Unreleased\]' CHANGELOG.md | grep -q 'numpy' && .venv/bin/python -m pytest tests/test_extras_guard.py -q 2>&1 | tail -1 | grep -q '5 passed' && echo PYPROJECT-FLOOR-SET</automated>
  </verify>
  <done>
    pyproject parses, declares numpy&gt;=2.0.0 with the one-line retirement comment,
    carries no pyarrow constraint or comment, keeps requires-python &gt;=3.10 and every
    other line byte-identical; the extras-guard lane (which parses this file) stays 5
    passed; CHANGELOG [Unreleased] records the breaking change.
  </done>
</task>

<task type="auto">
  <name>Task 2: CI matrix retirement, install-step simplification, doc claims</name>
  <files>.github/workflows/ci.yml, README.md, .github/workflows/README.md</files>
  <action>
    In .github/workflows/ci.yml, test job only:
    1) Matrix line 46: numpy-version: ['1.26.4', '2.2.0'] becomes numpy-version:
    ['2.2.0'] — keep the python-version axis and the job-name template (which interpolates
    matrix.numpy-version) untouched, so the matrix stays re-widenable later.
    2) "Install specific numpy version" step (lines 87-94): keep the step name and the
    `source .venv/bin/activate` line, delete the whole if/elif/fi block (the 1.26.4
    branch with its scipy&gt;=1.15.2 reinstall and the 2.2.0 elif), leaving the single
    unconditional line uv pip install "numpy==${{ matrix.numpy-version }}" — the same
    shape the coverage-gate and coverage-nightly jobs already use for their pins. The
    scipy reinstall existed only to repair the 1.26.4 leg; scipy&gt;=1.15.2 is already a
    base dependency and needs no special-casing on numpy 2.
    RED LINE: the coverage-gate (line ~420) and coverage-nightly (line ~547) "Install
    specific numpy version" steps already pin numpy==2.2.0 — leave both byte-identical;
    do not touch test-windows, test-cuda, test-mamba, example-nightly, deploy, or any
    gate/cron/comment outside the two edits above.
    Docs: README.md line 577 — the parenthetical "(Python 3.11/3.12/3.13 × numpy
    1.26.4/2.2.0, plus a Windows leg)" (× is the U+00D7 multiplication sign already in
    the file) becomes "(Python 3.11/3.12/3.13 × numpy 2.2.0, plus a Windows leg)". .github/workflows/README.md line 29 (live-grep discovery, same
    claim class): "- NumPy versions: 1.26.4, 2.2.0" becomes "- NumPy versions: 2.2.0".
    Change nothing else in either doc. Do not commit — Task 3 owns the commit.
  </action>
  <verify>
    <automated>grep -qF "numpy-version: ['2.2.0']" .github/workflows/ci.yml && grep -qF 'uv pip install "numpy==${{ matrix.numpy-version }}"' .github/workflows/ci.yml && ! grep -rq '1\.26\.4' .github/ README.md && test "$(grep -c 'numpy==2.2.0' .github/workflows/ci.yml)" -eq 2 && .venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && grep -qF '3.11/3.12/3.13 × numpy 2.2.0' README.md && grep -qF '- NumPy versions: 2.2.0' .github/workflows/README.md && echo CI-MATRIX-RETIRED</automated>
  </verify>
  <done>
    The test matrix is 3 legs on numpy 2.2.0 only; the install step is one unconditional
    pin line with no version-conditional shell left; zero 1.26.4 occurrences remain under
    .github/ and README.md; the two pre-existing numpy==2.2.0 nightly/gate pins are
    untouched; the workflow still parses; both doc claims match the new matrix.
  </done>
</task>

<task type="auto">
  <name>Task 3: trim the two numpy-1.x shim tests, run the lane, atomic commit + push</name>
  <files>tests/utils/test_transformers_compat_np.py</files>
  <action>
    Delete EXACTLY two tests from tests/utils/test_transformers_compat_np.py and nothing
    else in the file:
    1) test_patch_noops_when_numpy_provides_fromstring (was lines 35-48, first class) —
    the numpy 1.x "native fromstring must stay untouched" rung. Under the &gt;=2.0.0
    floor a working native fromstring never exists in supported environments (real numpy
    2 ships the raising stub, whose replacement stays covered by
    test_patch_replaces_raising_numpy2_stub), and the shim's works-then-no-op branch
    stays exercised by test_patch_is_idempotent_via_sentinel (the installed fallback
    itself satisfies the probe on the second call).
    2) test_binary_mode_str_input_matches_historical_behavior (was lines 136-147, second
    class) — the historical numpy 1.x binary-mode framing test. The str-encode shim path
    it exercised remains covered end-to-end by its kept siblings
    test_binary_mode_str_count_is_honored, test_binary_mode_str_result_is_writable, and
    test_binary_mode_str_non_ascii_encodes_utf8, so the CR-01 regression guard (bare str
    from stripedhyena's CharLevelTokenizer) survives this trim.
    Reword the module docstring's discipline enumeration (lines 7-9, "absence gate,
    idempotency sentinel, no-op when the library already provides the API") to name the
    surviving covered set — absence gate (fallback install), raising-stub replacement,
    idempotency sentinel, missing-module no-op — noting the numpy 1.x no-op rung retired
    with the &gt;=2.0.0 floor (2026-10-10); keep lines within 100 columns. The first
    class's own docstring ("Absence-gated np.fromstring restore (the numpy rung,
    08-03)") stays as-is. RED LINE: dnallm/utils/transformers_compat.py is NOT edited —
    the shim code itself is unchanged, only its numpy-1.x-scenario tests are trimmed; do
    not delete or weaken any other test.
    Then run the owner-specified lane and gates (targeted only, no repo-wide runs):
    .venv/bin/python -m pytest tests/utils/test_transformers_compat_np.py
    tests/test_extras_guard.py -q (expect "18 passed"), .venv/bin/ruff format --check
    tests/utils/test_transformers_compat_np.py, .venv/bin/ruff check
    tests/utils/test_transformers_compat_np.py. Any failure: STOP and report — no noqa,
    no suppression.
    Commit once, staging by explicit pathspec only (git add pyproject.toml CHANGELOG.md
    .github/workflows/ci.yml README.md .github/workflows/README.md
    tests/utils/test_transformers_compat_np.py — never git add -A / . / .planning; the
    pre-existing untracked .planning/graphs/ and .planning/tmp/ must stay out). Subject:
    deps(quick-261010-rpa): retire numpy 1.x support (floor >=2.0.0, drop pyarrow cap,
    CI matrix numpy 2.2.0 only). Body records: breaking change per CHANGELOG [Unreleased];
    requires-python >=3.10 deliberately untouched (separate owner decision); upstream
    evaluation facts (8/8 uv proofs incl. CI-pinned numpy==2.2.0 and the 426-package
    extras union; numpy2.2+pyarrow26+datasets3.2 runtime smoke green; zero audited
    packages pin numpy&lt;2); the two trimmed tests named with their retained-coverage
    rationale. No attribution trailers (owner standing rule). Push origin revision (retry
    twice on egress failure); then verify git rev-parse HEAD equals git rev-parse
    origin/revision and record both. Pushing revision fires the 3-leg matrix on CI —
    record the run URL in the SUMMARY (remote proof; do not block on it locally). Write
    261010-rpa-SUMMARY.md per the output section.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/utils/test_transformers_compat_np.py tests/test_extras_guard.py -q 2>&1 | tail -1 | grep -q '18 passed' && ! grep -q 'def test_patch_noops_when_numpy_provides_fromstring\|def test_binary_mode_str_input_matches_historical_behavior' tests/utils/test_transformers_compat_np.py && .venv/bin/ruff format --check tests/utils/test_transformers_compat_np.py && .venv/bin/ruff check tests/utils/test_transformers_compat_np.py && B=$(git log -1 --format=%H --grep='quick-261010-rpa') && test "$(git diff --name-only "$B^" "$B" | wc -l)" -eq 6 && git diff --quiet "$B^" "$B" -- dnallm/ && test "$(git rev-parse HEAD)" = "$(git rev-parse origin/revision)" && echo RETIREMENT-COMPLETE</automated>
  </verify>
  <done>
    The lane reports 18 passed (13 shim tests + 5 extras guards) with the two numpy-1.x
    tests gone and every other test intact; ruff format --check and ruff check are green
    on the touched file; exactly one commit carries precisely the six files and nothing
    under dnallm/; origin/revision == HEAD; the SUMMARY carries lane output, commit hash,
    push state, and the CI run URL.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none changed) | This change edits dependency constraint metadata, CI matrix config, documentation claims, and test selection. No user-input surface, no network surface, and no runtime code path in dnallm/ changes (the compat shim source is untouched). |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-rpa-01 | Tampering | dependency constraints — removing the pyarrow cap admits pyarrow>=26 into future resolves | low | mitigate | Upstream owner evaluation already proved the uncapped resolve green (8/8 uv resolution proofs incl. the full 426-package extras union; runtime smoke numpy2.2+pyarrow26+datasets3.2); the pushed 3-leg CI matrix re-proves the resolve remotely. |
| T-rpa-02 | Tampering | CI install-step simplification could weaken the version pin | low | mitigate | The simplified step keeps the exact `numpy==` pin via the matrix expression; the task verify asserts the pin line is present and the two pre-existing numpy==2.2.0 nightly/gate pins are byte-unchanged (count exactly 2). |
| T-rpa-SC | Tampering | package installs | low | accept | No package installs occur in this task (constraint metadata, config, docs, and test trims only) — the package-legitimacy gate is not triggered. |
</threat_model>

<verification>
1. Floor authority: pyproject declares numpy>=2.0.0 (one retirement comment line), no pyarrow constraint remains anywhere in the file, requires-python stays >=3.10, no numpy ceiling added — proven by the Task 1 automated gate plus tomllib parse.
2. Matrix honesty: ci.yml test matrix is ['2.2.0'] with the single unconditional pin line; zero 1.26.4 strings under .github/ and README.md; the nightly/gate pins untouched; workflow parses as YAML.
3. Behavior preserved: the owner-specified lane (.venv/bin/python -m pytest tests/utils/test_transformers_compat_np.py tests/test_extras_guard.py -q) reports 18 passed with only the two named numpy-1.x-scenario tests removed; the shim source dnallm/utils/transformers_compat.py is byte-unchanged across the commit; ruff format --check and ruff check green on the touched Python file (no repo-wide local runs — owner rule).
4. Records: CHANGELOG [Unreleased] carries the breaking entry; the commit body records the out-of-scope requires-python decision and the upstream evaluation facts; no attribution trailers.
5. End state: one atomic commit of exactly six files on revision, origin/revision == HEAD, and the fired CI matrix run URL recorded in the SUMMARY.
</verification>

<success_criteria>
- numpy 1.x support is retired end to end: floor >=2.0.0 in pyproject, pyarrow cap gone with its documented trigger, CI runs 3 legs on numpy 2.2.0 only, no doc or workflow file claims the retired leg, the breaking change is recorded, and the shim test suite reflects numpy-2-only reality while keeping every shim behavior test that still matters green.
- The change lands as one atomic, fully enumerated commit on a pushed revision with the pre-existing untracked .planning dirt untouched.
</success_criteria>

<output>
Create .planning/quick/261010-rpa-numpy-1-x-pyproject-numpy-floor-2-0-0-py/261010-rpa-SUMMARY.md when done; push origin revision and verify origin == HEAD.
</output>
