# Phase 2: Suite Hygiene & Known-Bug Fixes - Pattern Map

**Mapped:** 2026-09-30
**Files analyzed:** 12 (9 modified, 3 created) + 1 artifact deletion (9 untracked PDFs)
**Analogs found:** 11 / 12 (10 exact/role-match, 1 partial — see No Analog Found)

All analog paths below verified git-tracked (`git ls-files` non-empty) and read this session.
Line numbers are current `dev` HEAD (ada7f58).

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `dnallm/tasks/metrics.py` (edit ~283) | service (metrics factory) | transform | in-file: `dnallm/tasks/metrics.py:272-284` + error convention at `:660` | exact (in-file) |
| `dnallm/models/model.py` (edit 856-870) | service (loader dispatch) | request-response (chain-of-responsibility) | in-file: `dnallm/models/model.py:773-775` (early-return segments) + `:868-870` (None-check fall-through) | exact (in-file) |
| `tests/tasks/test_metrics.py` (edit 294-298, 743-791; +2 tests) | test | transform | in-file: `tests/tasks/test_metrics.py:243-274` (3-class data) + `:591-600` (error test) | exact (in-file) |
| `tests/models/test_model.py` (+1 regression test; − dead skips 97-130) | test | request-response (mocked) | in-file: `tests/models/test_model.py:473-501`, `:503-534`; sentinel `.to()` at `tests/conftest.py:55` | exact (in-file) |
| `tests/inference/test_plot.py` (edit 43-45; + fixture + marker) | test + fixture | file-I/O | `tests/mcp/test_interpret_tool.py:84-92` (autouse) + `tests/utils/test_cuda_compat.py:20-21` (monkeypatch.setattr) + `tests/models/test_model.py:92` (marker) | role-match (cross-file) |
| `dnallm/mcp/tests/test_sse_client.py` (edit 3 skip sites) | test (live integration) | request-response (network) | in-file: `dnallm/mcp/tests/test_sse_client.py:10-14` (existing *typed* skip — the model to copy) | exact (in-file) |
| `dnallm/mcp/tests/test_streamable_http_client.py` (edit 3 skip sites) | test (live integration) | request-response (network) | `dnallm/mcp/tests/test_sse_client.py` (structural twin; same edits at its `:54-55, 86-87, 111-112`) | exact (cross-file twin) |
| `dnallm/mcp/tests/conftest.py` (NEW shared skip helper) | test utility (conftest) | event-driven (exception classification) | `tests/conftest.py` (shared-fixture module layout) + `test_sse_client.py:10-14` | role-match |
| `tests/expected_skips.yaml` (NEW) | config (data file) | file-I/O (read-only) | `dnallm/configuration/configs.py:495-503` (yaml.safe_load consumer) — schema itself is novel | partial |
| `scripts/audit_skips.py` (NEW) | utility (CI gate script) | file-I/O parse → exit code | `scripts/check_docs_sync.py` (structural twin: main() -> int, exit 1 on findings) | exact |
| `.github/workflows/ci.yml` (edit test job) | config (CI) | batch | in-file: `:81-85` (fast-test step) + `:93-109` (canary gate step) | exact (in-file) |
| `.gitignore` (edit 132-138) | config (VCS) | — | in-file neighbors `:122-130` (`.mypy_cache/`, `.ruff_cache/`, `site`) | exact (in-file) |

## Pattern Assignments

### `dnallm/tasks/metrics.py` (service, transform) — FIX-01

**Analog:** in-file. The edit inserts a presence guard + `labels=` kwarg immediately before line 283.

**Current seam, verbatim (lines 283-284):**
```python
        metrics["AUROC"] = roc_auc_score(labels, pred_probs, average="macro", multi_class="ovr")
        metrics["AUPRC"] = average_precision_score(labels, pred_probs, average="macro")
```

**Error-convention analog (line 660, same file):** `raise ValueError(f"Unsupported task type for evaluation: {task_config.task_type}")` — descriptive, regex-matchable `ValueError`. The new guard must follow this shape; exact wording is executor discretion but the test's `match=` and the message must stay in sync.

**Scope facts for the edit:**
- `label_list` is the closure parameter of `multi_classification_metrics(label_list, plot)` at line 236; `len(label_list)` == `num_labels` for the dispatcher path (metrics.py:652 passes `task_config.label_names`; `TaskConfig.model_post_init` guarantees length-match — `dnallm/configuration/configs.py:132-136`).
- The guard must precede BOTH line 283 and line 284 (`average_precision_score` takes no `labels` kwarg) and also protects the `plot=True` branch (lines 317-362 use `roc_curve(np.array(labels) == i, pred_probs[:, i])` per class).
- Metrics at lines 272-282 (accuracy/precision/recall/F1/MCC) tolerate absent classes — do not move the guard above them (minimal diff, per RESEARCH Pattern 1).
- numpy usage restricted to `np.array/np.arange/np.unique/np.setdiff1d/np.array_equal` (stable across CI's numpy 1.26.4/2.2.0 span — RESEARCH Pitfall 8).

Recommended fix shape: RESEARCH.md Pattern 1 (verified against live sklearn 1.9.1 probes this cycle).

---

### `dnallm/models/model.py` (service, dispatch chain) — FIX-02

**Analog:** in-file. Two idioms already in this function:

**Idiom A — early-return segment (lines 773-775, the canonical correct handler):**
```python
    # Handle special case for EVO2 models
    evo2_result = _handle_evo2_models(model_name, source, head_config)  # type: ignore
    if evo2_result is not None:
        return evo2_result
```
(Same idiom repeats at 778-780, 786-788, 799-807, 810-818, 821-829. These run BEFORE `_get_model_path_and_imports`, so their early `return` is safe — the CrossDNA site is NOT in that position.)

**Idiom B — None-check fall-through (lines 868-870, the idiom the fix extends):**
```python
        model, tokenizer = _handle_dnabert2_models(downloaded_model_path, load_args)
        if model is None or tokenizer is None:
            model, tokenizer = _load_model_by_task_type(*load_args)
```

**Bug site, verbatim (lines 856-870):** the CrossDNA assignment at 857-867 is unconditionally overwritten by line 868 (see RESEARCH.md Pattern 2 for the verbatim block).

**Fix shape (RESEARCH Pattern 2 — copy this structure):**
```python
        model, tokenizer = None, None
        if "crossdna" in downloaded_model_path.lower():
            model, tokenizer = _handle_crossdna_models(...)  # unchanged args
        if model is None or tokenizer is None:
            model, tokenizer = _handle_dnabert2_models(downloaded_model_path, load_args)
        if model is None or tokenizer is None:
            model, tokenizer = _load_model_by_task_type(*load_args)
```
**Do NOT literally `return` at the crossdna site** — post-processing at lines 872-891 (mutbert/basenji2 tokenizers, `_model_path`/`source` attrs, `_configure_model_padding` at 883, `model = model.to(_get_device())` at 886, `_fix_bnb_quantized_layers` at 891) must still run for CrossDNA. "Return from the chain, not from the function."

**Anti-pattern (audit rows 3/6 in RESEARCH):** `_handle_gpn_models` (line 783) and `_handle_omnidna_models` (line 796) use `_ = handler(...)` and return `str | None` import-gates — NOT overwrite bugs; do not convert them to early returns. Confirmed fix count: exactly one (CrossDNA).

**Planner note:** the fix activates previously dead code — real CrossDNA models previously fell through to `_load_model_by_task_type`. State this as intended behavior change, not a regression.

---

### `tests/tasks/test_metrics.py` (test, transform) — FIX-01

**Analog:** in-file.

**3-class data analog (lines 243-252, `test_multi_classification_metrics_basic`):**
```python
        label_list = ["label1", "label2", "label3"]
        compute_func = multi_classification_metrics(label_list=label_list)

        logits = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
        labels = np.array([1, 0, 2])
        eval_pred = (logits, labels)
```
Use exactly this shape for (a) the unskipped plot test at 294-298 (assert `metrics["curve"]` keys `{"fpr","tpr","precision","recall"}` — verbatim from metrics.py:357-362) and (b) the rewritten multiclass branch of `test_compute_metrics_task_types`.

**Error-test analog (lines 591-600, `test_compute_metrics_unsupported_task`):**
```python
        with pytest.raises(ValueError, match="Unsupported task type for evaluation"):
            compute_metrics(task_config)
```
The new absent-class regression test follows this shape: `pytest.raises(ValueError, match=r"missing class id\(s\)")` (or whatever message the implementation lands on — keep both in sync).

**Parametrize block (lines 743-756):** change `("multiclass", 2, ["A", "B"])` → 3 classes; delete the skip at 759-761 (`if task_type == "multiclass": pytest.skip("Multiclass AUROC implementation has issues")`); rewrite the multiclass body branch at 782-788 (3-wide logits, ≥3 samples, all class ids present). 2-class multiclass is a probe-proven dead end (sklearn binary path demands 1-D scores) — RESEARCH Pitfall 1.

New tests go in `TestMultiClassificationMetrics` (class-per-function grouping convention, `Test*` prefix, one-line docstrings — match lines 240-244).

---

### `tests/models/test_model.py` (test, mocked request-response) — FIX-02 (+ FIX-03 dead-skip deletion)

**Analog:** in-file `TestLoadModelAndTokenizer` (line 470).

**Patch-stack analog (lines 473-501, `test_load_model_regular_huggingface`):**
```python
        task_config = TaskConfig(task_type="mask", num_labels=None)

        with (
            patch("dnallm.models.model._setup_huggingface_mirror"),
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=(
                    "/path",
                    {"AutoTokenizer": Mock(), "AutoModelForMaskedLM": Mock()},
                ),
            ),
            patch("dnallm.models.model._create_label_mappings", return_value=({}, {})),
            patch("dnallm.models.model._load_model_by_task_type", return_value=(Mock(), "tokenizer")),
            patch("dnallm.models.model._configure_model_padding"),
        ):
            model, tokenizer = load_model_and_tokenizer("test-model", task_config, source="huggingface")
```
Notes for the new sentinel test: patch handlers at the **import site** (`dnallm.models.model._handle_crossdna_models` / `_handle_dnabert2_models` — both imported into that namespace); the path must contain "crossdna" case-insensitively (e.g. `/models/CrossDNA-8.1M`); patch `_configure_model_padding` per existing style (line 494).

**Handler-patch analog (lines 510-514, `test_load_model_missing_num_labels_classification`):** shows `_handle_evo2_models`/`_handle_evo1_models`/`_handle_gpn_models` patched with `return_value=None` — the surrounding-chain mocks the new test may also need.

**Sentinel `.to()` analog — `tests/conftest.py:55` (critical, RESEARCH Pitfall 2):**
```python
    mock_model.to = Mock(return_value=mock_model)
```
model.py:886 rebinds `model = model.to(_get_device())`, so the sentinel MUST be constructed `sentinel_model = Mock(); sentinel_model.to = Mock(return_value=sentinel_model)` or the `assert model is sentinel_model` fails even with a correct fix. Assertion depth per decision: exact identity (`is`), not just non-None; optionally patch `_load_model_by_task_type` with `side_effect=AssertionError(...)` to prove no fall-through. Full test skeleton in RESEARCH.md Code Examples.

**Dead-skip deletion analog (lines 85-90):** the honest no-skip style — `try`-free body with `pytest.raises(ValueError, match="Model test-model download failed")`. Default for the dead scaffolding at 97-108 / 115-130 is deletion (RESEARCH Open Question 1, option a); whatever is kept must NOT retain the `"connection" in str(e).lower()` string-matching pattern (Pitfall 5).

---

### `tests/inference/test_plot.py` (test + fixture, file-I/O) — FIX-04

**Analog (autouse fixture):** `tests/mcp/test_interpret_tool.py:84-92`:
```python
    @pytest.fixture(autouse=True)
    def setup_mock_predict(self, mock_server):
        """Setup mock predict_sequence for auto target_class selection."""
```
plus session-scope autouse at `tests/conftest.py:9-12`. The new fixture is function-scoped autouse taking `(tmp_path, monkeypatch)`.

**Analog (monkeypatch.setattr on a module attribute):** `tests/utils/test_cuda_compat.py:20-21`:
```python
    monkeypatch.setattr(cuda_compat, "_preloaded", False)
    monkeypatch.setattr(cuda_compat, "_cuda13_wheels_available", lambda: False)
```
New fixture rebinds the module global: `monkeypatch.setattr("tests.inference.test_plot.PDF_OUTPUT_DIR", tmp_path)` (string-target form works because the module is importable; object form via `from tests.inference import test_plot` also fine).

**Edit sites:** delete import-time `PDF_OUTPUT_DIR.mkdir(exist_ok=True)` at line 45 (repo write at collection time — RESEARCH Pitfall 7); the in-function `mkdir` at line 105 stays (guarantees creation under tmp_path). `create_pdf_file` (92-111) resolves the global at call time, so all 26 call sites need zero changes. Helpers `cleanup_pdf_file` (114) and `assert_pdf_created` (141) operate on absolute paths — unchanged.

**Marker analog:** `@pytest.mark.slow` at `tests/models/test_model.py:92` (decorator above `@pytest.mark.asyncio`/test). Apply `@pytest.mark.pdf` at class level to PDF-writing classes; marker is already declared (`pyproject.toml:479`) and `--strict-markers` is active (`pyproject.toml:472`), so no registration work needed.

---

### `dnallm/mcp/tests/test_sse_client.py` and `test_streamable_http_client.py` (tests, network request-response) — FIX-03

**Analog:** in-file — the module-level ImportError guard is the repo's ONE existing *typed* skip, and the model to copy:

`test_sse_client.py:10-14`:
```python
try:
    from mcp.client.sse import sse_client
    from mcp.client.session import ClientSession
except ImportError as e:
    pytest.skip(f"MCP client modules not available: {e}", allow_module_level=True)
```
(same at `test_streamable_http_client.py:24` — this skip is already allowlist-worthy, message prefix `MCP client modules not available:`)

**Edit sites — broad except-skips to replace with the shared helper:**
- `test_sse_client.py:60-66` (`except Exception as e: ... pytest.skip(f"SSE connection failed: {e}")`), `:84-85`, `:109-110`
- `test_streamable_http_client.py:54-55`, `:86-87`, `:111-112`

Each becomes `except Exception as e: skip_if_unreachable(e, "<stable action label>")` — non-network leaves re-raise (honest failure). Skip messages get the stable `network-unavailable:` prefix for allowlist matching. Do NOT use `except*` syntax (3.10 floor — RESEARCH Pitfall on syntax). Exception classes: `httpx.TransportError` with ExceptionGroup unwrapping (probe-verified MRO; requests/urllib3 classes would never match — these clients fail through httpx).

---

### `dnallm/mcp/tests/conftest.py` (NEW — shared skip helper) — FIX-03

**Analog:** `tests/conftest.py` — module docstring (`"""Shared pytest fixtures for DNALLM test suite."""`), plain fixture/helper definitions, no conftest exists in `dnallm/mcp/tests/` today (this is the first; the directory already has `__init__.py`, so it is a package).

Contents: the `skip_if_unreachable` helper + `NETWORK_ERRORS = (httpx.TransportError,)` tuple + `_network_leaves` group-flattener — full vetted implementation in RESEARCH.md Pattern 3 (no in-repo analog for the flattening logic; copy from RESEARCH, not from anywhere in the codebase). Keep it importable from both client test modules; alternatively a `_network_skip.py` private module (leading-underscore private-module convention).

---

### `tests/expected_skips.yaml` (NEW — allowlist data file) — FIX-03

**Analog (partial):** no allowlist-style data file exists. Closest conventions:
- Consumed via `yaml.safe_load` like `dnallm/configuration/configs.py:503` (`config_dict = yaml.safe_load(f)`).
- Seed entries from the verbatim Phase-1 census messages (RESEARCH.md "census ground truth": 9 events — after FIX-01, rows 2-3 disappear; rows 4-9 become `network-unavailable:` prefixed; row 1 persists).
- Format design (prefix/exact/reason_like matching, `category` annotations) is specified in RESEARCH.md Pattern 4 — executor discretion on exact schema. Run the CI-shaped leg locally once before freezing (skipif reasons land in junit differently — RESEARCH Pitfall 6).

---

### `scripts/audit_skips.py` (NEW — CI gate utility) — FIX-03

**Analog:** `scripts/check_docs_sync.py` — exact structural twin (CI-invoked checker, exit-code gate):

Structure to copy (lines 1-6, 56-78):
```python
#!/usr/bin/env python3
"""Verify docs/example/ is a byte-identical mirror of example/ (except runtime artifacts)."""
...
def main() -> int:
    ...
    if errors:
        print("SYNC ERRORS:")
        for err in errors:
            print(f"  {err}")
        return 1

    print("OK: docs/example/ is in sync with example/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
Conventions: shebang, one-line module docstring, typed `main() -> int`, collect-then-print findings list, `return 1` on findings / `return 0` with an OK line, `sys.exit(main())`. `scripts/` is ruff-linted tracked code — full style rules apply (line-length 100, annotations).

Skeleton + junit parsing approach (`xml.etree.ElementTree`, fail-closed on unparseable input) in RESEARCH.md Pattern 4.

---

### `.github/workflows/ci.yml` (config, CI) — FIX-03

**Analog:** in-file.

**Edit target — "Run fast tests" step (lines 81-85, verbatim):**
```yaml
      - name: Run fast tests
        run: |
          source .venv/bin/activate
          pytest -m "not slow" --cov
          coverage xml -o coverage.xml
```
Add `--junitxml=pytest-junit.xml` to the pytest line.

**New-step analog — "Exit-code canary" step (lines 93-109):** a follow-on gate that activates the venv, runs a python check, and exits nonzero on failure; note its `run: |` + `source .venv/bin/activate` preamble convention used by every step. New step:
```yaml
      - name: Skip audit (unexpected skips fail the job)
        run: |
          source .venv/bin/activate
          python scripts/audit_skips.py pytest-junit.xml tests/expected_skips.yaml
```
Scope: wire ONLY the main `test` job (CONTEXT integration point is singular); `test-cuda`/`test-mamba` legs are Phase-4 GATE scope (RESEARCH Open Question 2). Do not touch the dead codecov step (line 88) — Phase-4 scope.

---

### `.gitignore` (config) — FIX-04

**Analog:** in-file directory-entry neighbors (lines 122-130: `site`, `.mypy_cache/`, `results/`, `.ruff_cache/`). Replace the seven lines at 132-138 (typo `test/inference/pdf/` + six enumerated filenames) with the single directory entry `tests/inference/pdf/`. Also delete the 9 untracked files under `tests/inference/pdf/` (runtime state, no history rewrite needed).

## Shared Patterns

### Honest ValueError with matchable message
**Source:** `dnallm/tasks/metrics.py:660`, `dnallm/models/model.py:880` (`raise ValueError(f"Failed to load model: {e}") from e`)
**Apply to:** FIX-01 guard; asserted everywhere via `pytest.raises(ValueError, match=...)` — analogs `tests/tasks/test_metrics.py:599`, `tests/models/test_model.py:530-534` and `:555-558`.

### Patch at the import site
**Source:** `tests/models/test_model.py:477-495` (`patch("dnallm.models.model._...")`)
**Apply to:** FIX-02 regression test (patch `_handle_crossdna_models`/`_handle_dnabert2_models` in the `dnallm.models.model` namespace, never their defining modules).

### Mock `.to()` self-return for identity assertions
**Source:** `tests/conftest.py:55` (`mock_model.to = Mock(return_value=mock_model)`)
**Apply to:** FIX-02 sentinel construction — required because `model.py:886` rebinds the model through `.to()`.

### Typed skips only (broad `except Exception: pytest.skip` banned)
**Source:** `dnallm/mcp/tests/test_sse_client.py:10-14` (ImportError guard — the existing correct example); new shared helper per RESEARCH Pattern 3
**Apply to:** all 6 MCP client sites; stable `network-unavailable:` message prefix for allowlist matching.

### CI gate script shape
**Source:** `scripts/check_docs_sync.py:56-78` (`main() -> int`, findings list, exit 1)
**Apply to:** `scripts/audit_skips.py`.

### Test module conventions
**Source:** `tests/tasks/test_metrics.py:240-244` (class-per-function `Test*` grouping, docstrings), `tests/inference/test_plot.py:1-10` (module docstring), `tests/conftest.py:1-12`
**Apply to:** all edited/new test code — descriptive behavior-name tests, Google-style docstrings, English comments only (existing files contain some Chinese; do not extend it).

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `tests/expected_skips.yaml` | config (data) | file-I/O | No allowlist-style data file exists in the repo; consumption pattern (yaml.safe_load) analog only. Use RESEARCH.md Pattern 4 schema + census ground truth. |
| ExceptionGroup flattening inside `dnallm/mcp/tests/conftest.py` | utility | event-driven | No group-unwrapping code exists anywhere in the repo. Copy RESEARCH.md Pattern 3 verbatim (probe-verified this cycle). |

Everything else has an exact in-file or cross-file analog.

## Metadata

**Analog search scope:** `dnallm/tasks/`, `dnallm/models/`, `dnallm/models/special/` (signatures only, via RESEARCH), `dnallm/mcp/tests/`, `tests/` (conftest, tasks, models, inference, mcp, utils), `scripts/`, `.github/workflows/`, `.gitignore`, `pyproject.toml`
**Files read this session:** 15 (all excerpts above verified in-context, non-overlapping reads)
**Tracked-source gate:** every analog named above passed `git ls-files -- <path>` (non-empty). No `.gsd/` or capability-mirror paths referenced.
**Pattern extraction date:** 2026-09-30
