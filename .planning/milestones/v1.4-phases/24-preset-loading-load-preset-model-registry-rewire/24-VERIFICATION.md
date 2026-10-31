---
phase: 24-preset-loading-load-preset-model-registry-rewire
verified: 2026-10-22T03:25:00Z
status: passed
score: 10/10 must-haves verified
covered_files:
  - .planning/phases/24-preset-loading-load-preset-model-registry-rewire/24-01-PLAN.md
  - .planning/phases/24-preset-loading-load-preset-model-registry-rewire/24-01-SUMMARY.md
  - dnallm/models/model.py
  - tests/models/test_model.py
covered_digest: v3:sha256:49189bcb439f9fa829eb599782d3f246b8336b0a0eb95e5180fb2fa0d52a9191
---

# Phase 24 Verification: Preset Loading — `load_preset_model` Registry Rewire

## Goal Assessment

**Goal:** Registry entry names resolve at runtime in `load_preset_model` — a finetuned entry loads with its head/labels/threshold patched into TaskConfig — filling the grep-verified dead `preset` path without changing any existing family behavior.

**VERIFIED.** `load_preset_model` resolves `try_resolve(model_name)` first (exact entry name, then repo id — one call covering D-01 step 1 and D-02), falls back to the unchanged `MODEL_INFO` family branch, and returns 0 on a miss. The dead preset-list flatten branch is gone (`grep -c preset_models dnallm/models/model.py` → 0, baseline 3). Family behavior is proven unchanged by a ten-family parametrized regression over the real, unpatched `MODEL_INFO` — covering all ten zhangtaolab-default families including the two pure-family names (`tRNADetector`, `tRNAPointer`). Live tracer re-run at verification time prints `tracer ok` against the real packaged registry.

## Must-Haves Verification (10/10)

| # | Must-have truth | Evidence |
|---|-----------------|----------|
| 1 | `load_preset_model('PlantHelixSeek CRE', ...)` returns the loader pair called with repo id `zhangtaolab/PlantHelixSeek-CRE`, source `'modelscope'`, `use_mirror=False`, and patched `2 / ['Not CRE','CRE'] / 0.5` | `test_load_preset_model_registry_entry_name_patches_config` (green, network-free) + fresh tracer run |
| 2 | Caller's TaskConfig never mutated after a patched load | same tracer + dedicated `test_load_preset_model_caller_config_not_mutated` |
| 3 | Raw repo id performs the identical patched load (D-02) | `test_load_preset_model_repo_id_reverse_lookup` |
| 4 | Pretrained entry loads with no field patched, config value-unchanged (D-06) | `test_load_preset_model_pretrained_entry_no_patch` (args[1] == caller's config) |
| 5 | task_type mismatch raises plain-language ValueError naming model + both task types; four long-form aliases compare as short forms | `test_load_preset_model_task_type_mismatch_raises`, `..._alias_normalized_match_loads`, `..._alias_normalized_mismatch_raises` (loader asserted uncalled) |
| 6 | Neither-entry-nor-family name returns 0 — including empty string and case-variants (exact, case-sensitive) | `test_load_preset_model_registry_and_family_miss_returns_zero`, `..._exact_match_edges`, existing `not_found`/`key_error` |
| 7 | All ten zhangtaolab-default families resolve their MODEL_INFO default repo with `'modelscope'`/`False`, identically to pre-rewire (D-09) | `test_load_preset_model_zhangtaolab_family_regression` (10 params, real MODEL_INFO unpatched) |
| 8 | Registry wins on collision: `'DNABERT-2'` resolves `zhihan1996/DNABERT-2-117M`, not family default `lgq12697/DNABERT-2-117M` | `test_load_preset_model_registry_wins_over_family_default` |
| 9 | Patched values verbatim (==) from the frozen entry — no arithmetic/re-validation | tracer assertions == entry fields; `model_copy(update=)` only (no `model_validate` in helper); regression labels preserved verbatim (`..._regression_label_names_preserved` — `['promoter strength in tobacco leaves']`, not `['value']`) |
| 10 | Exactly one lineage log ('trained on base …') + one info line per patched field | `test_load_preset_model_logs_overrides_and_lineage` (caplog INFO) |

## Requirement Traceability

| ID | Statement | Status |
|----|-----------|--------|
| LOAD-01 | Registry entry names resolve in `load_preset_model` (finetuned entries load with `num_labels`/`label_names`/`threshold` patched into TaskConfig); the existing family-name path is preserved and regression-tested | **Complete** — checkbox `[x]` and traceability row updated in REQUIREMENTS.md; delivered by plan 24-01 (sole declarer; ready-ids check passed) |

## Automated Checks (all exit 0, final tree)

1. `pytest tests/models/test_model.py -q` → **232 passed** (baseline 211; +21 net: −2 dead-branch tests, +23 new)
2. `pytest tests/models/test_registry.py -q` → **41 passed** (sole-mentioner + no-write-API guards hold; the registry data filename appears nowhere in `model.py`)
3. Scoped coverage on `dnallm/models/model.py` → **93%** (641 stmts / 44 miss; pre-change baseline 618/44 — every new line covered) at `--fail-under=93`
4. `ruff check` + `ruff format --check` on both touched files → clean
5. Dead-branch identifier `preset_models` in `model.py` → **0** occurrences (baseline 3)
6. Regression gate: prior-phase suites (9 files, Phases 22+23) → **257 passed**, no cross-phase regressions
7. `# ty: ignore[invalid-assignment]` preserved on the family-default assignment; `from .modeling_auto import MODEL_INFO` remains function-local

## Prohibition Checks (plan-frontmatter prohibitions, all honored)

- `task_type` never silently patched or coerced — mismatch raises; loader never rewrites the user's task type (loader-mock-uncalled assertions)
- No fuzzy/approximate matching — near-miss names return 0 (case-variant + empty-string tests)
- Resolution order + registry-wins-on-collision disclosed in the docstring — present, without naming the registry data file

## Human Verification

None required — every deliverable is proven by green network-free tests (SUMMARY coverage block: 4/4 deliverables `human_judgment: false` with all-pass verification entries).

## Findings

- Code review: 4 findings (0 critical) — 2 warnings deferred with recorded reasons (WR-01 ModelScope-hosting assumption for the 24 upstream collision repos, carried to Phases 25/26; WR-02 latent list-aliasing with no current mutator), 2 info skipped as intentional (24-REVIEW-DISPOSITION.md, ledger open: 0 of 4)
- Security: threats_open 0 (24-SECURITY.md, commit d81ab07)

## Verdict

**PASSED** — Phase 24 achieved its goal. The registry is now the runtime truth source for preset names with family behavior provably unchanged; the dead branch is deleted; all gates green.

## Re-verification 2026-10-22 (stale-digest refresh, #4682)

Covered source changed after the original verifier ran (later phases' edits to shared
files, plus the 2026-10-22 task-type audit's registry data fix `f94c50f` for phase 22's
covered model_info.yaml). Fresh evidence on the current tree this run; must-have verdicts
above re-checked and unchanged:

- Phase 22: tests/models/test_registry.py (42, +1 new task-type guard) +
  tests/models/test_model_info_registry.py green in the 290-test model-lane run;
  the nine data-layer fixes are pinned by the new permanent guard test.
- Phase 23: tests/preflight lane 214 passed (fresh run this date).
- Phase 24: preset-model test class 26 passed (fresh run this date).

Status remains **passed**; covered_digest regenerated below at the new tree state.
