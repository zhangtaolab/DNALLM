# Phase 19 Deferred Items (out-of-scope discoveries from review batches)

Code-level findings surfaced by the docs accuracy sweep that are OUTSIDE the
sweep's docs-only scope (the batches edit docs/ + phase artifacts only).
Routed here per the sweep's scope fence; owners decide disposition.
Both findings (#1 CLI crash, #2 task_guides hf-route) were RESOLVED by Phase 20 — see Resolutions below.

| # | Found by | Finding | Evidence | Suggested fix |
|---|----------|---------|----------|---------------|

## Resolutions

Item 1 — RESOLVED by Phase 20 (20-01 Tasks 1-2, commits c3c65cd + 48f9905): all four flag-only sites now raise a static click.UsageError naming --config as required (exit 2) before any lazy core import; regression suite tests/cli/test_config_required.py proves it; usage.md rewritten to the shipped behavior in 0655411. Verified by 20-VERIFICATION.md (passed 6/6, 2026-10-16).

Item 2 — RESOLVED by Phase 20 (20-01 Task 3, commit 0655411): all seven source= route sites in task_guides.md flipped to source="modelscope" per models.lock rows 21/29. Verified by 20-VERIFICATION.md (passed 6/6, 2026-10-16).
