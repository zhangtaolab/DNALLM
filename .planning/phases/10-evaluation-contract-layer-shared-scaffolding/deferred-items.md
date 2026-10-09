# Phase 10 Deferred Items

Out-of-scope discoveries logged by executor agents (per GSD scope boundary).
Do not fix in-phase; surfaced for owner disposition.

## From plan 10-02 (metric registry, 2026-10-09)

Pre-existing local-env failure: `pytest --cov=<dotted.dnallm.target>` (pytest-cov
7.1.0, torch 2.11.0+cu130, Python 3.13, venv unchanged since 2026-09-29) crashes at
conftest load with `RuntimeError: function '_has_torch_function' already has a
docstring` (torch.overrides double-execution) once pytest-cov resolves the dotted
cov target and imports the `dnallm` package early; the pandas/numpy ImportError
shown first is a masking cascade. Reproduces with ANY `--cov=dnallm.*` target
(e.g. `--cov=dnallm.utils.sequence`), i.e. unrelated to this plan's changes; bare
`--cov` (source_pkgs from pyproject) works, and `coverage run -m pytest` +
`coverage report` works. Workaround used for the 10-02 same-change coverage-row
proof (metric_registry.py 99%, metrics.py 100%): the coverage CLI. Not fixed
in-phase (environment-level, affects the whole repo, out of scope per rules).
Note: 10-03's table above lists tests/tasks/test_metrics.py old-terminology hits
as owned by 10-02 — those were swept in commit fad6a33 (module docstring).

## From plan 10-03 (docs/terminology/changelog, 2026-10-09)

Old "DNA language model(s)" terminology persists in repo surfaces OUTSIDE the
phase-level check surface (`grep -rlniE "DNA[ -]language[ -]model" dnallm/
docs/ README.md example/` is already zero) and outside every wave-1 plan's
file list. Prose/docstring-only hits, no functional impact:

| File | Hits | Note |
|------|------|------|
| pyproject.toml | 1 (project description) | Owned by agent 10-01 same-wave (D-08) — apparently not swept by them; owner to confirm |
| tests/models/test_model.py | 1 (module docstring) | No lane owns tests/ docstrings in phase 10 |
| tests/tasks/test_task.py | 2 (module docstring) | No lane owns tests/ docstrings in phase 10 |
| tests/tasks/test_metrics.py | 2 (module docstring) | test file owned by agent 10-02 same-wave |
| ui/model_config_generator_app.py | 1 (UI default string) | No lane owns ui/ in phase 10 |
| cli/model_config_generator.py | 1 (CLI default string) | Root legacy cli/ (backward-compat shims) |
| cli/examples/generated_finetune_config.yaml | 1 (comment) | Root legacy cli/ examples |
| cli/examples/generated_benchmark_config.yaml | 2 (comments) | Root legacy cli/ examples |
| CHANGELOG.md | historical entries | Append-only history — must NOT be rewritten |
| .planning/** | several | Planning artifacts, excluded from lint/gates |
