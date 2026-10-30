## Deferred Items

- `docs/user_guide/preset_loading.md` fails the repo-wide `ruff format --check .` gate — pre-existing, untouched by plan 27.1-01
  status: resolved
  **Resolved 2026-10-30:** commit c3289a0 formatted the guide's fenced Python examples; live repo-wide Ruff format verification passed with 374 files already formatted under dev Ruff 0.16.10, with pyproject pinning 0.17.1, so the first-exposure warning is void.
  **What:** both ruff 0.16.10 (dev venv) and the pyproject-pinned 0.17.1 want the guide's fenced Python examples reformatted (comment alignment; one multi-line call collapsed onto one line). The file last changed in `b35f64d` (2026-10-22). The `v1.4` branch is not in `ci.yml` `on.push.branches`, so this CI step has never run against it; it will surface on the first `v1.4`→`main` exposure. Plan 27.1-01's four touched files are format-clean under the same pinned version.
  **Scope decision:** out-of-scope per the executor scope boundary (pre-existing issue in an unrelated file); not fixed inside this plan.
