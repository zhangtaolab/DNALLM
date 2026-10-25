---
phase: 261025-vvw
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/configuration/configs.py
  - tests/configuration/test_configs.py
autonomous: true
must_haves:
  truths:
    - "load_config on a YAML file whose root is not a mapping — empty file, comment-only file, list root, scalar root — raises ValueError with a matchable 'must contain a YAML mapping' message naming the actual root type, replacing the three opaque TypeErrors probed on v1.4 tip ('NoneType' object does not support item assignment / list indices must be integers or slices, not str / 'str' object does not support item assignment at dnallm/configuration/configs.py:777)"
    - "The four degenerate-root regression tests are red on v1.4 tip (pytest.raises(ValueError) fails because TypeError is raised first) and green after the guard commit"
    - "The guard lives only in load_config at the file boundary; validate_sections is untouched, so the 51-test equivalence suite tests/preflight/test_config_gen.py stays green (load_config ≡ validate_sections on every packaged config)"
    - "Unknown top-level sections are still silently dropped — a 'typo_section:' key in the YAML is absent from the returned config — pinned as documents-current-behavior so any future strictness is a conscious, test-updating decision"
    - "A relative config path is injected as an absolute config_path consumed by BenchmarkConfig (os.path.isabs true, path resolves to the written file, os.path.dirname gives the config's own directory — the dnallm/inference/benchmark.py:149 consumption), and 'config_path' never appears as a key of the returned config mapping (DNALLMConfig carries no such key, dnallm/configuration/configs.py:683)"
  artifacts:
    - dnallm/configuration/configs.py (root-type guard in load_config + Raises: docstring section)
    - tests/configuration/test_configs.py (TestLoadConfigDegenerateRoot with 4 tests + TestLoadConfigLoaderContracts with 2 tests, after the existing TestLoadConfig class)
  key_links:
    - "load_config open+yaml.safe_load -> isinstance-dict guard -> config_path=os.path.abspath injection -> validate_sections -> BenchmarkConfig(**full raw dict).config_path -> dnallm/inference/benchmark.py:149 os.path.dirname(benchmark_config.config_path)"
---

<objective>
Pin the config loader's unpinned contracts with regression tests, fixing the one
contract that is currently broken: `load_config` (dnallm/configuration/configs.py:763)
crashes with an opaque TypeError on any YAML whose root is not a mapping — an empty
file, a comment-only file, a list root, or a scalar root all fail at the
`config_dict["config_path"] = ...` line (configs.py:777) with messages like
"'NoneType' object does not support item assignment". The repo's error-handling
convention (CLAUDE.md: ValueError with a descriptive, matchable message for invalid
input/config) says these must be ValueError. Probed live on v1.4 tip (b188f6c):
empty -> TypeError NoneType, comment-only -> TypeError NoneType, list root ->
TypeError list-indices, scalar root -> TypeError str.

The 23-04 `validate_sections` factor-out pinned loader≡factored equivalence in
tests/preflight/test_config_gen.py, but three loader-level behaviors remain unpinned
anywhere: (1) degenerate-root handling, (2) silent drop of unknown top-level sections,
(3) relative-path abspath injection consumed downstream by benchmark.py:149. The
existing test at tests/configuration/test_configs.py:626 even carries a comment
admitting it "documents the current behavior" of config_path without asserting it.

Purpose: guard the loader surface that survived two refactors (23-04 factor-out,
261003-0p0 TypedDict) with no direct pins, and replace the convention-violating
crash per the same fix+red-to-green regression pattern as quick 261025-e4x.
Output: a 3-line guard in load_config, six new regression tests in the proper home
(tests/configuration/test_configs.py), all lanes green.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md

@dnallm/configuration/configs.py
@tests/configuration/test_configs.py
@tests/preflight/test_config_gen.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Red-to-green — load_config rejects non-mapping YAML roots with matchable ValueError</name>
  <files>tests/configuration/test_configs.py, dnallm/configuration/configs.py</files>
  <behavior>
    - Test 1 (empty file): a zero-byte YAML passed to load_config raises ValueError matching "must contain a YAML mapping" — red on tip, which raises TypeError('NoneType' object does not support item assignment)
    - Test 2 (comment-only file): a file containing only a comment line raises the same ValueError — red on tip (safe_load returns None)
    - Test 3 (list root): a file whose root is a YAML list raises the same ValueError — red on tip (TypeError: list indices must be integers or slices, not str)
    - Test 4 (scalar root): a file whose root is a bare scalar raises the same ValueError — red on tip (TypeError: 'str' object does not support item assignment)
    - All four assert via pytest.raises(ValueError, match=r"must contain a YAML mapping"); at least one also asserts the message names the offending root type (NoneType / list / str)
  </behavior>
  <action>
    RED first: add a TestLoadConfigDegenerateRoot class to tests/configuration/test_configs.py, directly after the existing TestLoadConfig class (line ~776). Use the tmp_path fixture to write the four degenerate files. Run the class — all four must fail with the TypeError shapes probed on tip; capture that output for the summary as the red proof.

    GREEN: in load_config (dnallm/configuration/configs.py:763), immediately after `config_dict = yaml.safe_load(f)` and BEFORE the `config_dict["config_path"] = ...` line, guard the root type: raise ValueError when config_dict is not a dict, with a message of the shape "Config file {os.path.abspath(config_path)} must contain a YAML mapping at the top level, got {type(config_dict).__name__}". Use plain isinstance(config_dict, dict) — no try/except around the item assignment (that would mask the distinction between a bad root and a bad injection).

    Scope fence — do NOT touch validate_sections: it receives already-constructed dicts from dnallm/preflight/config_gen.py and its plain-dict contract is separately pinned; the guard belongs only at the file boundary in load_config. Do not add yaml.SafeLoader arguments or switch loaders (safe_load stays). Update the load_config docstring with a Raises: section (Google style, untyped-param form `config_path:` per the IN-08 convention) documenting the ValueError for non-mapping roots. Per owner rule, the code change and its pytest coverage ship in the same commit.
  </action>
  <verify>
    <automated>.venv/bin/pytest tests/configuration/test_configs.py -q && .venv/bin/pytest tests/preflight/test_config_gen.py -q</automated>
  </verify>
  <done>Four degenerate-root tests green (110 total in the file: 106 baseline + 4); red proof (tip TypeError output) recorded in the summary; the 51-test equivalence suite in tests/preflight/test_config_gen.py still green; load_config docstring documents the ValueError.</done>
</task>

<task type="auto">
  <name>Task 2: Pin loader contracts — unknown-section drop and relative-path abspath injection</name>
  <files>tests/configuration/test_configs.py</files>
  <action>
    Test-only characterization of two loader contracts that no test currently asserts. Add a TestLoadConfigLoaderContracts class after TestLoadConfigDegenerateRoot:

    (1) test_unknown_top_level_sections_are_dropped — write a YAML carrying a valid `task:` section (task_type: binary) plus a `typo_section:` key with arbitrary content; assert the returned mapping's keys are exactly ["task"], i.e. the unknown key is silently absent. Docstring marks it documents-current-behavior (same style as the existing comment at tests/configuration/test_configs.py:626): a misspelled section name is masked today, and any future strictness/warning must consciously update this pin.

    (2) test_relative_path_injected_absolute_and_consumed_by_benchmark — using tmp_path and monkeypatch.chdir(tmp_path), write a COMPLETE minimal benchmark YAML to bm.yaml (benchmark: {name, description}; models: one entry with name+path; datasets: one entry with name+path+task; output: {path, format} — the exact shape of the existing test_load_config_benchmark at tests/configuration/test_configs.py:681-712, whose three required sections models/datasets/output were confirmed by a live probe). Call load_config("bm.yaml") with the bare relative filename. Assert: the returned benchmark section's config_path field is absolute (os.path.isabs), Path(config_path).resolve() equals (tmp_path / "bm.yaml").resolve() (robust to symlinked tmp dirs), and os.path.dirname(config_path) is the config's own directory — the exact consumption at dnallm/inference/benchmark.py:149. Also assert "config_path" is NOT a key of the returned config mapping (injection is internal only; DNALLMConfig declares no such key — dnallm/configuration/configs.py:683).
  </action>
  <verify>
    <automated>.venv/bin/pytest tests/configuration/ -q && .venv/bin/ruff check dnallm/configuration/configs.py tests/configuration/test_configs.py && .venv/bin/ruff format --check dnallm/configuration/configs.py tests/configuration/test_configs.py</automated>
  </verify>
  <done>Both pins green (112 total in test_configs.py: 106 baseline + 6 new; full tests/configuration/ lane green across its three files); no production file changed in this task; ruff check and format clean on both touched files (baselined clean on tip).</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| local YAML config file -> load_config | user-authored file content crosses into validated config objects via yaml.safe_load |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Severity | Mitigation Plan |
|-----------|----------|-----------|-------------|----------|-----------------|
| T-vvw-01 | Denial of Service | load_config on non-mapping YAML root | mitigate | low | Explicit isinstance-dict guard after safe_load raises ValueError naming the file and actual root type, replacing three distinct opaque TypeErrors (probed on tip); a user saving an empty or mis-rooted config now gets a diagnosable error instead of an interpreter internals message. |
| T-vvw-02 | Tampering | YAML file content | accept | low | Config files are local, owner-authored inputs at the same trust level as the process itself; yaml.safe_load (never yaml.load) is the existing pinned parsing path and this plan adds no new parsing surface. |
| T-vvw-SC | Tampering | package installs | accept | low | No package-manager installs (stdlib + existing pydantic/pytest deps only); the package-legitimacy gate is not triggered. |
</threat_model>

<verification>
- Red proof captured before the fix: the four degenerate-root tests fail on v1.4 tip with the exact TypeError shapes probed at planning time.
- Green proof after the fix: `.venv/bin/pytest tests/configuration/ -q` fully green (112 in test_configs.py: 106 baseline + 6 new; yaml_load and peft_presets unchanged).
- Equivalence non-regression: `.venv/bin/pytest tests/preflight/test_config_gen.py -q` stays at 51 passed — the guard sits before validate_sections and changes no validated path (load_config ≡ validate_sections still holds on every packaged config).
- Lint: `.venv/bin/ruff check` and `.venv/bin/ruff format --check` clean on both touched files (baselined clean on tip).
- New-code coverage: the guard's two branches (non-dict raise + dict pass-through) are both exercised by the new tests plus every existing green-path test, so the per-module >=96 percent rule for new code is met without a dedicated measurement; CI's fail_under=90 gate is unaffected.
</verification>

<success_criteria>
- load_config raises ValueError (matchable "must contain a YAML mapping", root type named) for empty, comment-only, list-root, and scalar-root YAML files; no TypeError path remains for those inputs.
- Six new regression tests in tests/configuration/test_configs.py (4 degenerate-root + 2 contract pins), all green, committed together with the guard in the same change per the owner rule.
- validate_sections, config_gen, and the equivalence suite untouched and green (51 passed).
- Unknown-section silent-drop and relative-path abspath injection are now pinned contracts with documents-current-behavior docstrings.
</success_criteria>

<output>
Create `.planning/quick/261025-vvw-add-regression-test-for-config-loader/261025-vvw-SUMMARY.md` when done
</output>
