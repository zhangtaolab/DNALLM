# Phase 09 Deferred Items

## Deferred Items

- `tests/benchmark/test_benchmark.py::TestBenchmark::test_plot_for_regression` fails on the fast lane (pandas `TypeError: float() argument must be a string or a real number, not 'dict'` inside `_astype_nansafe` via the altair/pandas plotting path). **Pre-existing** — reproduced at the 09-02 plan-start commit `3557e0b` (worktree check, same venv: `1 failed, 1825 passed, 1 skipped`), not caused by any Phase 09 change (09-02 delta is +8 passed / +0 failed / +0 skipped). Likely fallout from quick task 13/14 Mapping-type changes (86022f7/16a9ffb) touching `Benchmark` config handling. Out of 09-02 scope per the scope boundary; surfaced to the owner for triage — candidate for a `/gsd-quick` fix or 09-04 awareness, since the fast-lane verify literal (`grep -E '^[0-9]+ passed'`) stays confounded until it is fixed.
  - **Status:** open
