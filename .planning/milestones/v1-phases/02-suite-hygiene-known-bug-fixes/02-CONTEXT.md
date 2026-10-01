# Phase 2: Suite Hygiene & Known-Bug Fixes - Context

**Gathered:** 2026-09-30
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — all 4 grey areas accepted as proposed

<domain>
## Phase Boundary

The suite reports true code behavior — no test is skipped because the code crashes, and every remaining skip is a typed, intentional network skip.

In scope (requirements FIX-01..FIX-04, ROADMAP success criteria 1-4):
1. Multiclass AUROC test at `tests/tasks/test_metrics.py:761` runs unskipped and passes; `compute_metrics` handles multiclass targets without crashing
2. CrossDNA handler results are returned instead of overwritten — regression test asserts the correct handler's result survives the dispatch chain
3. Every skip in the suite is typed (specific network exceptions) and matches an expected-skip allowlist; a new unexpected skip fails the run instead of passing silently
4. Running the PDF-marked tests leaves the git working tree clean (artifacts under `tmp_path`), and `.gitignore` ignores `tests/inference/pdf/` correctly

Out of scope: new coverage tests (Phase 3), CI gate (Phase 4), the CI-leg skip-audit wiring beyond what FIX-03's enforcement requires.

</domain>

<decisions>
## Implementation Decisions

### FIX-01: Multiclass AUROC fix
- Fix mechanism: pass `labels=np.arange(num_labels)` to `roc_auc_score` at `dnallm/tasks/metrics.py:283` — deterministic, keeps the metric honest (NOT try/except→NaN, which repeats the hide-failures pattern this milestone removes)
- Unskip BOTH AUROC skips: `tests/tasks/test_metrics.py:761` and the multiclass-plotting skip at `:298` (same root cause)
- When a class is genuinely absent from eval predictions: raise `ValueError` with a matchable message (honest failure, matches project error conventions)
- Add an edge regression test: batch missing a class → asserts the ValueError behavior deterministically

### FIX-02: CrossDNA dispatch fix
- Fix shape: early-return chain-of-responsibility — `result = handler(...)`; `if result is not None: return result` (REQUIREMENTS-specified contract)
- Regression test in `tests/models/test_model.py` (alongside existing dispatch tests)
- Assert depth: fault-injection with a sentinel object returned by the CrossDNA handler; assert the EXACT sentinel survives `load_model_and_tokenizer` (proves no later handler overwrites; not just non-None)
- Scope: audit all 12 `special/*` handlers for the same overwrite pattern; fix only confirmed instances; record audit findings in SUMMARY

### FIX-03: Typed skips + allowlist
- Enforcement: CI skip-audit step parsing the run's junit artifact against an allowlist; unexpected skip reason fails the step (no in-process self-judging)
- Allowlist lives at `tests/expected_skips.yaml` (data file, reviewable, read by CI and local scripts alike)
- Typed-exception set: narrow tuple — `requests.exceptions.ConnectionError/Timeout`, urllib3/socket connection errors, `huggingface_hub`/`modelscope` offline+rate-limit errors; `except Exception: pytest.skip` is banned
- Unexpected skip → CI failure (the whole point: a new crash-skip can never pass silently)

### FIX-04: PDF artifacts & .gitignore
- PDF test outputs parametrized through pytest `tmp_path` — zero repo writes
- `.gitignore`: fix the typo to the real path `tests/inference/pdf/` and KEEP the entry (belt-and-suspenders for stray writes)
- Delete the existing untracked `tests/inference/pdf/` generated files
- Keep the `@pytest.mark.pdf` marker (identification/deselection tool)

### Claude's Discretion
Implementation details beyond these decisions (exact test names, helper shapes, ordering within files) per codebase conventions.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 1's honest harness: bare `pytest` collects both roots under `pyproject.toml`; junit artifacts + census tooling from `01-AUDIT-REPORT.md` (skip-reason parsing already proven against junit-full.xml: 9 skips = 6 untyped TaskGroup network skips + 2 AUROC crash-skips + 1 benign example skip)
- `tests/conftest.py` shared mock fixtures (mock_model/mock_tokenizer/mock_config) for dispatch fault-injection
- `.planning/codebase/CONCERNS.md:55-58` documents the AUROC root cause: `roc_auc_score(labels, pred_probs, average="macro", multi_class="ovr")` fails when an eval batch lacks classes; names both fix approaches
- `tests/tasks/test_metrics.py` (32 tests) and `tests/models/test_model.py` (40 tests) are the existing test homes for FIX-01/FIX-02

### Established Patterns
- Error tests: `pytest.raises(ValueError, match=r"...")` with regex match
- Mock at the import site; shared fixtures preferred over re-rolled mocks
- Special handlers: `_handle_<family>_models(...) -> tuple | None` contract; returning None falls through (early-return chain is the documented intent at `dnallm/models/model.py:789-848`)
- CI: `ci.yml` test job now runs bare `pytest` (Phase 1); junit artifacts pattern established by the audit

### Integration Points
- `dnallm/tasks/metrics.py:283` (AUROC), `dnallm/models/model.py:873-887` (dispatch chain)
- `.github/workflows/ci.yml` gains the skip-audit step consuming the run's junit
- `.gitignore` typo fix; `tests/expected_skips.yaml` is a new data file

</code_context>

<specifics>
## Specific Ideas

No specific requirements beyond the accepted decisions above.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
