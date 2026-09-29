# Phase 2: Suite Hygiene & Known-Bug Fixes - Research

**Researched:** 2026-09-30
**Domain:** pytest suite hygiene + two targeted library-code defect fixes (sklearn metrics, model-loading dispatch chain)
**Confidence:** HIGH (every load-bearing claim verified against live code executed this session: sklearn 1.9.1 / numpy 2.5.3 / httpx 0.28.1 / pytest 9.1.1 in `.venv`, plus verbatim file reads)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**FIX-01: Multiclass AUROC fix**
- Fix mechanism: pass `labels=np.arange(num_labels)` to `roc_auc_score` at `dnallm/tasks/metrics.py:283` — deterministic, keeps the metric honest (NOT try/except→NaN, which repeats the hide-failures pattern this milestone removes)
- Unskip BOTH AUROC skips: `tests/tasks/test_metrics.py:761` and the multiclass-plotting skip at `:298` (same root cause)
- When a class is genuinely absent from eval predictions: raise `ValueError` with a matchable message (honest failure, matches project error conventions)
- Add an edge regression test: batch missing a class → asserts the ValueError behavior deterministically

**FIX-02: CrossDNA dispatch fix**
- Fix shape: early-return chain-of-responsibility — `result = handler(...)`; `if result is not None: return result` (REQUIREMENTS-specified contract)
- Regression test in `tests/models/test_model.py` (alongside existing dispatch tests)
- Assert depth: fault-injection with a sentinel object returned by the CrossDNA handler; assert the EXACT sentinel survives `load_model_and_tokenizer` (proves no later handler overwrites; not just non-None)
- Scope: audit all 12 `special/*` handlers for the same overwrite pattern; fix only confirmed instances; record audit findings in SUMMARY

**FIX-03: Typed skips + allowlist**
- Enforcement: CI skip-audit step parsing the run's junit artifact against an allowlist; unexpected skip reason fails the step (no in-process self-judging)
- Allowlist lives at `tests/expected_skips.yaml` (data file, reviewable, read by CI and local scripts alike)
- Typed-exception set: narrow tuple — `requests.exceptions.ConnectionError/Timeout`, urllib3/socket connection errors, `huggingface_hub`/`modelscope` offline+rate-limit errors; `except Exception: pytest.skip` is banned
- Unexpected skip → CI failure (the whole point: a new crash-skip can never pass silently)

**FIX-04: PDF artifacts & .gitignore**
- PDF test outputs parametrized through pytest `tmp_path` — zero repo writes
- `.gitignore`: fix the typo to the real path `tests/inference/pdf/` and KEEP the entry (belt-and-suspenders for stray writes)
- Delete the existing untracked `tests/inference/pdf/` generated files
- Keep the `@pytest.mark.pdf` marker (identification/deselection tool)

### Claude's Discretion
Implementation details beyond these decisions (exact test names, helper shapes, ordering within files) per codebase conventions.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FIX-01 | Fix multiclass AUROC crash in `dnallm/tasks/metrics.py:283` and unskip `tests/tasks/test_metrics.py:761` | Live sklearn probe of the exact call; crash taxonomy per input shape; `label_list` identified as the in-scope `num_labels` source; discovery that the test parametrization itself must change to 3 classes |
| FIX-02 | Fix CrossDNA handler result overwrite in `dnallm/models/model.py:873-887` (early-return chain-of-responsibility) and add regression test | Verbatim read of the overwrite at line 868; all 12 handler call sites audited; sentinel-test construction hazards documented (`.to()` self-return requirement) |
| FIX-03 | Replace broad `except Exception: pytest.skip` with typed network skips; enforce an expected-skip allowlist | Complete skip-site census (file:line, catch shape, census events); live probe of actual exception classes (`ExceptionGroup` wrapping `httpx.ConnectError`); discovery that 2 of the named skip sites are dead code; CI wiring point identified (no junit in CI today) |
| FIX-04 | Point PDF test artifacts at `tmp_path`; fix `.gitignore` typo | Module-global `PDF_OUTPUT_DIR` mechanism read verbatim; 26 call sites counted; minimal monkeypatch-rebind fix with zero call-site churn; typo line + stray files confirmed |
</phase_requirements>

## Summary

Phase 2 is four surgical fixes, and live code probes this session materially sharpened three of them. For FIX-01, the AUROC crash at `dnallm/tasks/metrics.py:283` is now fully characterized: the call fails with three distinct errors depending on input shape, and — critically — the current test parametrization `("multiclass", 2, ["A","B"])` at `tests/tasks/test_metrics.py:748-751` can NEVER pass even with the planned `labels=np.arange(num_labels)` fix, because sklearn routes a 2-unique-class target to its binary path which demands 1-D scores (`ValueError: y should be a 1d array, got an array of shape (2, 2) instead.` — reproduced this session). Unskipping therefore requires changing that parametrize tuple to 3 classes, not just deleting the skip. Additionally, the probe shows `labels=np.arange(3)` on an absent-class batch silently returns `nan` (sklearn 1.9.1) rather than raising — which proves the decision's explicit presence-check→`ValueError` is load-bearing, not belt-and-suspenders: `labels=` alone would re-introduce the exact silent-degradation pattern this milestone exists to remove. The same single guard also protects line 284 (`average_precision_score`), which crashes with the same "y should be a 1d array" error on absent-class batches and does not accept a `labels` kwarg at all.

For FIX-02, the bug is one line: `dnallm/models/model.py:868` unconditionally overwrites the CrossDNA result computed at 857-867. The full 12-handler audit found exactly one other candidate family — `_ = _handle_gpn_models(...)` (line 783) and `_ = _handle_omnidna_models(...)` (line 796) — but those handlers return `str | None` (import-availability gates whose `ImportError` side effect is the point), NOT model tuples, so they are not overwrite bugs and must not be "fixed" into early returns. The regression test needs one non-obvious construction detail: line 886 `model = model.to(_get_device())` replaces the model object, so the sentinel Mock must have `sentinel.to = Mock(return_value=sentinel)` or the identity assertion fails even when the fix is correct.

For FIX-03, the census of every `pytest.skip` in both roots (23 sites) is complete, and two discoveries reshape the worklist: (a) the network-skip branches at `tests/models/test_model.py:105-108` and `:127-130` are dead code — `download_model` swallows every exception and re-raises only `ValueError(f"Model {model_name} download failed.")` whose message never matches the `"connection" in str(e).lower()` test, so those tests fail (not skip) when offline; (b) CI produces no junit artifact today (`pytest -m "not slow" --cov` at ci.yml:84), so FIX-03 must add `--junitxml` before an audit step can parse anything. Live probing the 6 MCP TaskGroup skips shows the catchable exception is `builtins.ExceptionGroup` wrapping `httpx.ConnectError` — the typed tuple is `httpx.TransportError` with group unwrapping, not the requests/urllib3 classes named in the decision (those apply to the dead download-model sites).

FIX-04 is the smallest: `tests/inference/test_plot.py` writes PDFs through one module global (`PDF_OUTPUT_DIR`, created by an import-time `mkdir` at line 45), so a single autouse fixture rebinding it to `tmp_path` redirects all 26 call sites with zero signature churn. No test currently carries `@pytest.mark.pdf` (the marker is only declared in `pyproject.toml:479`), so the success criterion "running the PDF-marked tests leaves the tree clean" requires also applying the marker to the PDF-writing tests.

**Primary recommendation:** Execute as four independent workstreams in one phase (no shared files except `pyproject.toml` untouched); FIX-01 first (its probe evidence changes test bodies most), FIX-02 second, then FIX-03 (which should re-run the census after FIX-01's skips disappear) and FIX-04 in any order. Every fix has a verified regression test shape in this document.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Multiclass metric computation (AUROC/AUPRC) | Library core (`dnallm/tasks/metrics.py`) | Test suite (regression tests only) | The defect is in library code; tests must not work around it (that was the old skip) |
| Model-load dispatch chain | Library core (`dnallm/models/model.py`) | Special-handler package (`dnallm/models/special/*`) | Chain-of-responsibility contract is already documented in CLAUDE.md; fix restores it at the one broken segment |
| Skip typing | Test code (both roots) | — | Skips live in tests; typing is a test-code change |
| Skip enforcement | CI workflow + tooling script | Test data file (`tests/expected_skips.yaml`) | Decision explicitly requires out-of-process judgment (junit parse), not in-process self-judging |
| PDF artifact isolation | Test code (`tests/inference/test_plot.py`) | VCS config (`.gitignore`) | pytest fixtures own temp-dir lifecycle; .gitignore is defense-in-depth |
| CI artifact production | CI workflow (`ci.yml` test job) | — | junit must exist before it can be audited |

## Standard Stack

No new packages. This phase adds zero dependencies; every tool needed is already installed and verified in `.venv` this session.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | 1.9.1 installed; `>=1.4.0` floor [VERIFIED: pyproject.toml:57] | AUROC/AUPRC behavior (the thing being fixed) | Already the metrics backend |
| pytest | 9.1.1 installed; `>=8.4` floor [VERIFIED: pyproject.toml:93] | skip mechanics, tmp_path, monkeypatch, junitxml | Harness already standardized in Phase 1 |
| httpx | 0.28.1 installed | Exception classes for typed MCP network skips | Transitive dep of `mcp>=1.3.0,<2` (core dep, pyproject.toml:42); the actual transport under the TaskGroup |
| PyYAML | `>=6.0` [VERIFIED: pyproject.toml:55] | `tests/expected_skips.yaml` parsing in audit script | Already a core dep (configs) |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `xml.etree.ElementTree` | stdlib | junit parsing in the skip-audit script | Phase-1 census already used exactly this (01-AUDIT-REPORT.md:13) |
| `exceptiongroup` backport | via anyio on py<3.11 | 3.10-safe ExceptionGroup handling | Only if the helper imports it explicitly; duck-typing avoids the need (see Pattern 3) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `httpx.TransportError` tuple | `except* httpx.ConnectError` syntax | `except*` is Python 3.11+ syntax → SyntaxError at collection on 3.10 (requires-python is `>=3.10`); rejected |
| `httpx.TransportError` tuple | `requests`/`urllib3` classes (named in CONTEXT decision) | Wrong transport: the MCP tests fail through httpx, not requests; requests classes would never match — see Pitfall 3 |
| monkeypatch-rebind of `PDF_OUTPUT_DIR` | threading a `pdf_dir` fixture through `create_pdf_file` signature | Signature change touches 26 call sites for identical behavior; rebind touches 1 fixture — see Pattern 5 |

## Package Legitimacy Audit

**No packages are installed by this phase.** All work is in-repo code, config, and CI YAML. The legitimacy gate is therefore trivially satisfied — nothing to check on any registry.

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

The phase touches four independent seams. The skip-audit data flow (FIX-03) is the only new pipeline:

```
                       ┌─────────────────────────────────────────────┐
                       │ CI test job (ci.yml, "Run fast tests")      │
                       │ pytest -m "not slow" --cov                  │
                       │        + --junitxml=pytest-junit.xml  (NEW) │
                       └──────────┬──────────────────────────┬───────┘
                                  │ pass/fail               │ junit artifact
                                  ▼                         ▼
                       ┌──────────────────┐    ┌──────────────────────────┐
                       │ job exit code    │    │ NEW step: scripts/       │
                       │ (pytest native)  │    │ audit_skips.py           │
                       └──────────────────┘    │  parse <skipped> msgs    │
                                               │  match vs tests/        │
                                               │  expected_skips.yaml    │
                                               │  unmatched ⇒ exit 1     │
                                               └──────────────────────────┘
   Local runs: same script, same YAML ──────────────────┘

Dispatch chain (FIX-02), segment inside load_model_and_tokenizer try-block:

  _get_model_path_and_imports → downloaded_model_path
        │
        ▼
  "crossdna" in path? ──yes──► _handle_crossdna_models ──(None,None)──► fall through
        │ no                        │ (model, tok)                        │
        │                           ▼  NEW: guard — result survives      ▼
        │                      [FIXED: today line 868 unconditionally
        │                       overwrites this with _handle_dnabert2_models]
        ▼
  _handle_dnabert2_models(path, load_args) ──(None,None)──► _load_model_by_task_type(*load_args)
        │ (model, tok)                                   (terminal generic fallback)
        ▼
  post-processing: mutbert/basenji2 tokenizers, ._model_path/.source attrs,
  _configure_model_padding, .to(device), _fix_bnb_quantized_layers → return
```

### Recommended Project Structure

```
tests/
├── expected_skips.yaml            # NEW (FIX-03): allowlist data file
├── models/test_model.py           # EDIT (FIX-02): + regression test in TestLoadModelAndTokenizer
├── tasks/test_metrics.py          # EDIT (FIX-01): unskip ×2, rewrite multiclass params/body, + edge test
├── inference/test_plot.py         # EDIT (FIX-04): tmp_path rebind fixture, + @pytest.mark.pdf, remove import-time mkdir
├── utils/test_cuda_compat.py      # no edit needed (already deterministic skips) — allowlist entry only
├── examples/test_examples.py      # no edit needed (content skips are benign) — allowlist entries only
dnallm/
├── tasks/metrics.py               # EDIT (FIX-01): presence guard + labels= at ~283
├── models/model.py                # EDIT (FIX-02): guarded chain at 856-870
└── mcp/tests/
    ├── test_sse_client.py         # EDIT (FIX-03): typed skip helper ×3 sites
    └── test_streamable_http_client.py  # EDIT (FIX-03): typed skip helper ×3 sites
scripts/
└── audit_skips.py                 # NEW (FIX-03): junit vs allowlist checker
.github/workflows/ci.yml           # EDIT (FIX-03): --junitxml + skip-audit step
.gitignore                         # EDIT (FIX-04): line 132 typo → tests/inference/pdf/
```

### Pattern 1: FIX-01 — presence guard before the AUROC/AUPRC pair
**What:** One explicit class-presence check in `multi_classification_metrics.compute_metrics`, raising `ValueError` per project conventions, THEN the `labels=`-anchored call.
**When to use:** Exactly here — the probe shows `labels=np.arange(3)` alone returns silent `nan` on absent-class batches (sklearn 1.9.1), which is the hide-failures anti-pattern the milestone removes.

Verbatim current code (the seam) [VERIFIED: dnallm/tasks/metrics.py:283-284, quoted with original indentation]:
```python
        metrics["AUROC"] = roc_auc_score(labels, pred_probs, average="macro", multi_class="ovr")
        metrics["AUPRC"] = average_precision_score(labels, pred_probs, average="macro")
```

`num_labels` source in scope: the enclosing factory's closure parameter [VERIFIED: dnallm/tasks/metrics.py:236 — `def multi_classification_metrics(label_list: list, plot: bool = False) -> Callable:`]. `TaskConfig.model_post_init` guarantees `label_names` is non-None and length-matched for multiclass [VERIFIED: dnallm/configuration/configs.py:132-136, quoted verbatim]:
```python
        elif task == "multiclass":
            if not self.num_labels or self.num_labels < 2:
                raise ValueError("num_labels must be at least 2 for multiclass classification")
            if not self.label_names or len(self.label_names) != self.num_labels:
                self.label_names = [f"class_{i}" for i in range(self.num_labels)]
```
So `len(label_list)` is always available and always equals `num_labels` for the `compute_metrics` dispatcher path (`multi_classification_metrics(task_config.label_names, plot=plot)` at metrics.py:652).

Recommended fix shape (executor discretion on exact message wording; regex-matchable per conventions):
```python
        # All classes must appear in the eval batch: roc_auc_score(multi_class="ovr")
        # and average_precision_score both degrade or crash otherwise.
        expected_classes = np.arange(len(label_list))
        present_classes = np.unique(labels)
        if not np.array_equal(present_classes, expected_classes):
            missing = np.setdiff1d(expected_classes, present_classes).tolist()
            raise ValueError(
                f"Multiclass metrics require every class in the eval batch; "
                f"missing class id(s) {missing} "
                f"({len(present_classes)}/{len(label_list)} classes present)."
            )
        metrics["AUROC"] = roc_auc_score(
            labels,
            pred_probs,
            average="macro",
            multi_class="ovr",
            labels=expected_classes,
        )
        metrics["AUPRC"] = average_precision_score(labels, pred_probs, average="macro")
```
Note `average_precision_score` takes NO `labels` kwarg (probe: `TypeError: got an unexpected keyword argument 'labels'`) — the guard is its only protection, which is why the guard must sit before BOTH calls. The guard also covers the `plot=True` branch below (lines 317-362 use `roc_curve(np.array(labels) == i, pred_probs[:, i])` per class — same all-present requirement) since line 283 precedes it.

Placement recommendation: immediately before the current line 283 (metrics at 272-282 — accuracy/precision/recall/F1/MCC — tolerate absent classes without crashing, so the minimal-diff placement keeps the guard tied to the calls that need it).

### Pattern 2: FIX-02 — guarded chain (None-check fall-through, not literal early return)
**What:** Chain-of-responsibility *within the try block*, preserving post-processing.
**When to use:** The segment at 856-870.

Verbatim current bug [VERIFIED: dnallm/models/model.py:856-870 — annotations on 869-870 in original]:
```python
        if "crossdna" in downloaded_model_path.lower():
            model, tokenizer = _handle_crossdna_models(
                task_type,
                downloaded_model_path,
                safe_num_labels,
                id2label,
                label2id,
                modules,
                head_config,
                custom_tokenizer,
                bnb_config,
            )
        model, tokenizer = _handle_dnabert2_models(downloaded_model_path, load_args)
        if model is None or tokenizer is None:
            model, tokenizer = _load_model_by_task_type(*load_args)
```

**Do NOT implement a literal `return crossdna_result` at the crossdna site.** Lines 872-891 (mutbert/basenji2 tokenizer post-processing, `model._model_path`/`model.source` attribution at 877-878, `_configure_model_padding` at 883, device placement at 885-886, `_fix_bnb_quantized_layers` at 890-891) must still run for a CrossDNA result — an early `return` from the function would silently skip padding/device placement for CrossDNA models. The decision's `if result is not None: return result` semantics must be realized as "return *from the chain*, not from the function":

```python
        model, tokenizer = None, None
        if "crossdna" in downloaded_model_path.lower():
            model, tokenizer = _handle_crossdna_models(
                # ... unchanged args ...
            )
        if model is None or tokenizer is None:
            model, tokenizer = _handle_dnabert2_models(downloaded_model_path, load_args)
        if model is None or tokenizer is None:
            model, tokenizer = _load_model_by_task_type(*load_args)
```

Handler contracts verified verbatim:
- [VERIFIED: dnallm/models/special/crossdna.py:473-484] signature `def _handle_crossdna_models(...) -> tuple[Any, Any]:` with docstring line "Returns ``(None, None)`` when *model_path* is not a CrossDNA snapshot so the normal DNALLM loading path can continue unchanged."
- [VERIFIED: dnallm/models/special/dnabert2.py:8, 15-16] `def _handle_dnabert2_models(model_path: str, load_args: list) -> tuple:` returns `return None, None` when the basename contains neither `"dnabert-2"` nor `"dnabert-s"` — so skipping the call for a CrossDNA path loses nothing (a CrossDNA snapshot was never going to match the basename check; today's wasted call had no side effect for such paths).

**Full 12-handler audit result (deliverable for the SUMMARY per the decision):**

| # | Handler | Call site | Current shape | Verdict |
|---|---------|-----------|---------------|---------|
| 1 | `_handle_evo2_models` | model.py:773-775 | `if ... is not None: return` | Correct (documented chain) |
| 2 | `_handle_evo1_models` | model.py:778-780 | same | Correct |
| 3 | `_handle_gpn_models` | model.py:783 | `_ = _handle_gpn_models(model_name)` | **Not a bug** — returns `str \| None` [VERIFIED: dnallm/models/special/gpn.py:10 — `def _handle_gpn_models(model_name: str, extra: str \| None = None) -> str \| None:`]; called for its `ImportError` side effect (optional-dep gate); the matched-name return is intentionally unused. Converting to an early return would return a *string* where a model tuple is expected — do not "fix" |
| 4 | `_handle_megadna_models` | model.py:786-788 | `if ... is not None: return` | Correct |
| 5 | `_handle_lucaone_models` | model.py:791-793 | commented out | Dead code, not a bug; leave |
| 6 | `_handle_omnidna_models` | model.py:796 | `_ = _handle_omnidna_models(model_name)` | **Not a bug** — same gate pattern as GPN [VERIFIED: dnallm/models/special/omnidna.py:11 — `def _handle_omnidna_models(model_name: str, extra: str \| None = None) -> str \| None:`] |
| 7 | `_handle_enformer_models` | model.py:799-807 | `if ... is not None: return` | Correct |
| 8 | `_handle_space_models` | model.py:810-818 | same | Correct |
| 9 | `_handle_borzoi_models` | model.py:821-829 | same | Correct |
| 10 | `_handle_crossdna_models` | model.py:856-867 | result overwritten at 868 | **THE bug — fix** |
| 11 | `_handle_dnabert2_models` | model.py:868 | None-check fallback to generic | Correct pattern; just must not run when crossdna already resolved |
| 12 | `_handle_mutbert_tokenizer` / `_handle_basenji2_tokenizer` | model.py:872-875 | tokenizer post-processors | Not result handlers; correct |

Confirmed instances to fix: exactly one (CrossDNA).

**Behavior-change note for the plan:** the fix *activates* previously dead code — today every real CrossDNA model silently loads through `_load_model_by_task_type` (generic path) because the handler result is always discarded. After the fix, CrossDNA models load through `_handle_crossdna_models` (this dead-path explains part of the 249 missing coverage lines in `dnallm/models/special/crossdna.py` from the Phase-1 audit, rank 7). This is the intended REQUIREMENTS behavior, but the plan should state it explicitly so the change is not mistaken for a regression.

### Pattern 3: FIX-03 — typed network skip with ExceptionGroup unwrapping
**What:** Replace `except Exception: pytest.skip(...)` in the 6 MCP sites with a narrow transport-error tuple plus group unwrapping; rewrite skip messages to a stable prefix so the allowlist can match deterministically.

Live probe evidence (this session, connecting to dead `localhost:8000` with the installed SDK) [VERIFIED: local probe, mcp SDK + httpx 0.28.1]:
```
SSE TOP-LEVEL: builtins.ExceptionGroup
  str: unhandled errors in a TaskGroup (1 sub-exception)
  SUB: httpx.ConnectError | All connection attempts failed
    SUB-CAUSE: httpcore.ConnectError | All connection attempts failed
STREAMABLE-HTTP TOP-LEVEL: builtins.ExceptionGroup
  str: unhandled errors in a TaskGroup (1 sub-exception)
  SUB: httpx.ConnectError | All connection attempts failed
```
`httpx.ConnectError` MRO [VERIFIED: local probe]: `httpx.ConnectError → httpx.NetworkError → httpx.TransportError → httpx.RequestError → httpx.HTTPError → builtins.Exception` — note it is NOT a builtin `ConnectionError` subclass, so `except ConnectionError` alone would not catch it.

Recommended helper (shared module under `dnallm/mcp/tests/`, e.g. `conftest.py` or `_network_skip.py` — no conftest exists there today [VERIFIED: directory listing]):
```python
import httpx
import pytest

# Network-level failures only. HTTPStatusError (server answered with an
# error status) is deliberately excluded: that is a real failure, not "no server".
NETWORK_ERRORS = (httpx.TransportError,)


def _network_leaves(exc: BaseException) -> list[BaseException]:
    """Flatten an ExceptionGroup tree to its leaves (duck-typed for 3.10)."""
    if hasattr(exc, "exceptions"):  # BaseExceptionGroup on 3.11+, exceptiongroup backport below
        out: list[BaseException] = []
        for sub in exc.exceptions:
            out.extend(_network_leaves(sub))
        return out
    return [exc]


def skip_if_unreachable(exc: BaseException, action: str) -> None:
    """Skip only when every leaf cause is a network-level failure; re-raise otherwise.

    Raises:
        BaseException: the original exception, when any leaf is not a network error.
    """
    leaves = _network_leaves(exc)
    if leaves and all(isinstance(leaf, NETWORK_ERRORS) for leaf in leaves):
        pytest.skip(f"network-unavailable: {action} (no server reachable: {type(leaves[0]).__name__})")
    raise exc
```
Call sites become:
```python
        except Exception as e:
            skip_if_unreachable(e, "SSE connection test")
```
Design properties (each maps to a locked decision):
- **Banned pattern gone:** no `except Exception: pytest.skip` — broad catch remains only as the unwrapping entry point, and non-network leaves re-raise (test fails honestly). E.g. a running server returning a tool error (`mcp` `McpError`, an `httpx.HTTPStatusError`, an assertion) now FAILS the test instead of skipping — this is the hygiene win.
- **3.10-safe:** `hasattr(exc, "exceptions")` duck-typing avoids importing `BaseExceptionGroup` (3.11+ builtin; on 3.10 the `exceptiongroup` backport that anyio installs provides the same attribute). `except*` syntax is 3.11+ only and would be a collection-time SyntaxError on 3.10 — do not use it.
- **All-leaves rule:** a mixed group (one network leaf + one real bug) re-raises — strictly safer than first-leaf-wins.
- **Stable skip message:** the `network-unavailable:` prefix makes junit messages deterministic enough for allowlist matching (current messages embed variable exception text like "unhandled errors in a TaskGroup (1 sub-exception)").

### Pattern 4: FIX-03 — allowlist + audit script + CI wiring

**Allowlist** `tests/expected_skips.yaml` (format is executor discretion; prefix-vs-exact is the key design axis — module-level ImportError skips embed variable messages so need prefix matching):
```yaml
# Expected skips: every junit <skipped message> must match exactly one entry.
# An unmatched skip fails CI (scripts/audit_skips.py exit 1).
allowed:
  - prefix: "network-unavailable:"
        category: network        # 6 MCP live-server probes (slow-marked)
  - prefix: "MCP client modules not available:"
        category: optional-dep   # module-level ImportError guard (never fires in CI: mcp is a core dep)
  - exact: "No import statements found"
        category: content        # tests/examples/test_examples.py:254 — deterministic for current repo content
  - exact: "No import cells found"
        category: content        # tests/examples/test_examples.py:139
  - exact: "CUDA 13 wheels not installed in this environment"
        category: environment    # tests/utils/test_cuda_compat.py:31
  - reason_like: "No marimo files found"   # skipif reasons land in junit differently — see Pitfall 6
        category: environment
```

**Audit script skeleton** (`scripts/audit_skips.py`; parses junit the same way the Phase-1 census did, `xml.etree` [CITED: .planning/phases/01-harness-integrity-measured-baseline/01-AUDIT-REPORT.md:13]; PyYAML `>=6.0` already a core dep [VERIFIED: pyproject.toml:55]):
```python
"""Fail when the pytest junit artifact contains a skip not in the allowlist."""
import sys
import xml.etree.ElementTree as ET

import yaml


def main(junit_path: str, allowlist_path: str) -> int:
    allowed = yaml.safe_load(open(allowlist_path))["allowed"]
    unexpected = []
    for tc in ET.parse(junit_path).iter("testcase"):  # CI-generated file, still parsed defensively
        sk = tc.find("skipped")
        if sk is None:
            continue
        msg = sk.get("message") or ""
        if not _matches(msg, allowed):
            unexpected.append(f"{tc.get('classname')}::{tc.get('name')} -> {msg!r}")
    if unexpected:
        print("UNEXPECTED SKIPS (not in allowlist):")
        print("\n".join(f"  {u}" for u in unexpected))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
```

**CI wiring:** the `test` job's fast-test step currently is, verbatim [VERIFIED: .github/workflows/ci.yml:84]:
```
          pytest -m "not slow" --cov
```
Change to `pytest -m "not slow" --cov --junitxml=pytest-junit.xml` and add a following step:
```yaml
      - name: Skip audit (unexpected skips fail the job)
        run: |
          source .venv/bin/activate
          python scripts/audit_skips.py pytest-junit.xml tests/expected_skips.yaml
```
Runs on all 6 matrix legs; deterministic skips are identical across legs. Scope boundary: the 6 network skips are `@pytest.mark.slow` and never run in this leg — their allowlist entries serve local/full runs and the Phase-4 GATE-02 slow leg (deeper slow-leg wiring is explicitly out of phase scope per CONTEXT). The script itself is leg-agnostic, so GATE-02 can reuse it unchanged.

### Pattern 5: FIX-04 — tmp_path rebind of the module global
**What:** One autouse fixture redirects `PDF_OUTPUT_DIR`; zero changes to the 26 `create_pdf_file` call sites.
**When to use:** When output paths flow through a module-level constant read at call time.

Verbatim current mechanism [VERIFIED: tests/inference/test_plot.py:43-45 and 104-109]:
```python
# Define PDF output directory
PDF_OUTPUT_DIR = Path(__file__).parent / "pdf"
PDF_OUTPUT_DIR.mkdir(exist_ok=True)
```
```python
    # Ensure the PDF directory exists
    PDF_OUTPUT_DIR.mkdir(exist_ok=True)

    # Create a unique filename based on test name
    filename = f"{test_name}_{suffix.lstrip('.')}.pdf"
    file_path = PDF_OUTPUT_DIR / filename
```
Because `create_pdf_file` resolves the module global at call time, this fixture redirects every write:
```python
@pytest.fixture(autouse=True)
def pdf_output_dir(tmp_path, monkeypatch):
    """Write all PDF artifacts under tmp_path — the repo tree stays clean."""
    monkeypatch.setattr("tests.inference.test_plot.PDF_OUTPUT_DIR", tmp_path)
```
Plus: delete the import-time `PDF_OUTPUT_DIR.mkdir(exist_ok=True)` at line 45 (it is itself a repo write on mere collection — the one remaining tree mutation after the rebind) — the mkdir inside `create_pdf_file` (line 105) still guarantees creation. `cleanup_pdf_file` and `assert_pdf_created` operate on absolute paths and need no change.

Marker: `@pytest.mark.pdf` is currently on ZERO tests — only declared [VERIFIED: pyproject.toml:479 — `    "pdf: marks tests that generate PDF files",`] and documented in `tests/TESTING.md:117`. `--strict-markers` is active in addopts (pyproject.toml:472) and the marker is declared, so applying it is safe. Apply at class level to the test classes whose methods call `create_pdf_file` (grep count: 26 call sites in the file) — the success criterion "running the PDF-marked tests leaves the tree clean" is only checkable if the marker is actually applied.

`.gitignore` fix, verbatim current lines [VERIFIED: .gitignore:132-138]:
```
test/inference/pdf/
tests/inference/pdf/demo_attention_pdf.pdf
tests/inference/pdf/demo_bars_pdf.pdf
tests/inference/pdf/demo_curves_pdf.pdf
tests/inference/pdf/demo_embeddings_pdf.pdf
tests/inference/pdf/demo_scatter_pdf.pdf
tests/inference/pdf/format_compatibility_test_pdf.pdf
```
Replace all seven lines with the single line `tests/inference/pdf/` (decision: fix typo AND keep the entry). The enumerated filenames are incomplete anyway — the directory entry supersedes them. Then delete the 9 stray untracked files currently in `tests/inference/pdf/` (`consistency_test_0_pdf.pdf`, `content_validation_test_pdf.pdf`, `demo_mutations_pdf.pdf` are not even enumerated — they are the visible untracked-dir noise in git status).

### Anti-Patterns to Avoid
- **Literal early `return` at the CrossDNA site** — skips padding/device placement (see Pattern 2).
- **`except*` syntax in test files** — SyntaxError on Python 3.10 (requires-python floor).
- **try/except→NaN around the AUROC call** — explicitly rejected by the locked decision; the probe proves sklearn would even do it "for free" via `labels=` — that is precisely why the presence guard must come first.
- **Applying `@pytest.mark.pdf` without the tmp_path rebind** (or vice versa) — the success criterion needs both: the marker makes the tests selectable, the rebind makes them clean.
- **Converting `_handle_gpn_models`/`_handle_omnidna_models` to early returns** — they return model-NAME strings, not model tuples (audit row 3/6).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Skip bookkeeping | Per-test flags/env vars judged in-process | junit artifact + external audit script + YAML allowlist | The locked decision explicitly bans in-process self-judging; junit is already the proven census medium |
| Exception-group flattening | Recursion-free ad-hoc `isinstance` checks per site | One shared `skip_if_unreachable` helper | Same tree-walk logic ×6 sites would drift; one helper keeps the all-leaves rule consistent |
| Temp-dir management in PDF tests | Per-test `os.makedirs`/`finally: rmtree` | pytest `tmp_path` + autouse rebind fixture | pytest manages lifecycle; failures cannot leak |
| AUROC absent-class semantics | Custom per-class AUC reimplementation | presence check + sklearn `labels=` kwarg | sklearn already computes macro-ovr correctly when classes are present; the defect is only the mismatch |

**Key insight:** every fix here is restoring an already-documented contract (handler None-fallthrough, honest ValueError, typed skips) — not designing new behavior. The tests should assert the contract, not the implementation.

## Common Pitfalls

### Pitfall 1: Unskipping `test_compute_metrics_task_types[multiclass-2-...]` without changing the parametrize tuple
**What goes wrong:** The unskipped test still crashes: `ValueError: y should be a 1d array, got an array of shape (2, 2) instead.` [VERIFIED: local probe, sklearn 1.9.1 — reproduced both with and without `labels=np.arange(2)`]. sklearn routes a target with exactly 2 unique values to its binary path regardless of the `labels` kwarg; the binary path demands 1-D scores. "Multiclass with exactly 2 classes" is unsupported by this metric path, full stop.
**Why it happens:** The parametrize tuple `("multiclass", 2, ["A", "B"])` at tests/tasks/test_metrics.py:748-751 was itself a workaround ("Fix: Use 2 classes to avoid AUROC issues" — verbatim comment in the test).
**How to avoid:** Change the tuple to 3 classes (e.g. `("multiclass", 3, ["A", "B", "C"])`) AND rewrite the multiclass test body (current body at 782-788 builds 2-sample 2-wide logits with labels `[1, 0]` — even 3-wide logits need ≥3 samples so every class id appears in `labels`).
**Warning signs:** The unskipped test fails with the "1d array" error; do not "fix" it by catching the error — fix the data.

### Pitfall 2: Sentinel Mock identity lost to `.to(_get_device())`
**What goes wrong:** The FIX-02 regression test asserts `result_model is sentinel_model` and fails even though the dispatch fix is correct.
**Why it happens:** model.py:885-886 (verbatim): `if bnb_config is None:` / `model = model.to(_get_device())` — a plain `MagicMock().to(...)` returns a fresh child mock, so the returned model is `.to()`'s return value, not the sentinel.
**How to avoid:** Construct the sentinel with `sentinel_model = Mock(); sentinel_model.to = Mock(return_value=sentinel_model)`. Also patch `_configure_model_padding` (it reads `model.config.pad_token_id` — Mocks tolerate it, but patching matches the existing test style at tests/models/test_model.py:494). Attribute writes (`model._model_path`, `model.source`) are harmless on Mocks.
**Warning signs:** Assertion `is not None` passes but `is sentinel` fails.

### Pitfall 3: Typing the MCP skips with requests/urllib3 exception classes
**What goes wrong:** The typed skip never matches; every connection failure re-raises and the 6 slow tests fail on any dev machine without a server.
**Why it happens:** The CONTEXT decision names `requests.exceptions.ConnectionError/Timeout, urllib3/socket` — correct for the (dead) download-model sites, wrong for the MCP clients: `sse_client`/`streamable_http_client` fail through **httpx** (probe: `httpx.ConnectError` under `builtins.ExceptionGroup`). httpx's `ConnectError` is not even a builtin `ConnectionError` subclass (MRO verified).
**How to avoid:** Use `httpx.TransportError` (parent of Connect/Read/Write errors and all httpx timeouts) with group unwrapping (Pattern 3). Note for the SUMMARY: the decision's exception-set bullet is satisfied in spirit (narrow tuple, broad-except banned) with httpx as the concrete class family for these sites.
**Warning signs:** `pytest -m slow dnallm/mcp/tests/test_sse_client.py` failing instead of skipping with no server up.

### Pitfall 4: Expecting `labels=np.arange(num_labels)` to produce the honest ValueError
**What goes wrong:** Implementing only the `labels=` change (no explicit presence check) yields silent `nan` AUROC + an `UndefinedMetricWarning` on absent-class batches [VERIFIED: local probe, sklearn 1.9.1] — and nan-poisoned metrics flow into Trainer logs. Worse, the behavior is sklearn-version-dependent (older sklearn in the `>=1.4.0` range raised "Only one class present" instead of nan).
**Why it happens:** sklearn's macro-ovr averages per-class AUCs; an absent class contributes an undefined AUC which sklearn 1.9 reports as nan-with-warning rather than an error.
**How to avoid:** The explicit presence check MUST execute before the sklearn call — it converts version-dependent degradation into a deterministic, matchable `ValueError` (and makes behavior identical across the supported sklearn range). This is the decision's own "honest failure" clause; the probe proves it is required, not optional.
**Warning signs:** A green test suite with `nan` showing up in multiclass eval logs.

### Pitfall 5: The two `tests/models/test_model.py` network-skip sites are dead code — do not "retype" them blindly
**What goes wrong:** Time spent crafting a typed tuple for sites that can never skip; or worse, "widening" them into real skips that newly mask failures.
**Why it happens:** [VERIFIED: dnallm/models/model.py:347-353, 374-375] `download_model` wraps EVERY downloader exception and after retries raises only `raise ValueError(f"Model {model_name} download failed.")` — the original exception text does not survive into the message. The tests' skip condition at tests/models/test_model.py:105 (verbatim): `if "connection" in str(e).lower() or "network" in str(e).lower():` can never match that ValueError, so the `else: raise` branch always runs: offline, these tests FAIL today (which is honest). The `pytest.skip` at :106/:128 is unreachable.
**How to avoid:** Two coherent options — (a) recommended: delete the dead try/except-skip scaffolding and let the wrapped ValueError fail loudly; CI's slow legs require network anyway (CLAUDE.md constraint), so no legitimate skip case exists; (b) if local-offline DX matters, probe network reachability explicitly with a typed check BEFORE calling `download_model` (e.g. attempt `huggingface_hub` metadata fetch and skip on `huggingface_hub.errors.OfflineModeIsEnabled`/`requests.exceptions.ConnectionError`). Flag in the plan as a decision point with option (a) as default; both satisfy "broad except-skip banned".
**Warning signs:** Any change that keeps the string-matching `"connection" in str(e)` pattern.

### Pitfall 6: skipif reasons vs runtime skip messages in junit
**What goes wrong:** The allowlist matches junit `<skipped message="...">`, but `@pytest.mark.skipif` failures-of-condition record the `reason=` string while `pytest.skip(msg)` records `msg` — and module-level `pytest.skip(..., allow_module_level=True)` records its message once per module, not per test.
**Why it happens:** junit representation differences; the census (junit-full.xml) only ever showed runtime `pytest.skip` messages because no skipif fired in that environment.
**How to avoid:** Seed the allowlist from the verbatim census messages (below), then run the CI-shaped leg (`pytest -m "not slow" --junitxml=...`) locally once to capture what actually appears in each environment before freezing the YAML. The static skipif sites (`test_cuda_compat.py:26`, the 8 file-existence skipifs in `test_examples.py`, `test_yaml_load.py:23`) did not fire in the census and are environment-deterministic — include their reason strings as defensive exact entries.
**Warning signs:** Audit script reports unexpected skips for skipif reasons that "obviously" should be allowed.

### Pitfall 7: PDF rebind fixture scoping and the import-time mkdir
**What goes wrong:** The repo tree still gets an empty `tests/inference/pdf/` directory created (or, if the fixture is function-scoped but `create_pdf_file` is called at module/class scope somewhere, writes land in the repo).
**Why it happens:** Line 45 `PDF_OUTPUT_DIR.mkdir(exist_ok=True)` runs at COLLECTION time, before any fixture; function-scoped autouse fixtures only cover test-time calls (all 26 call sites are test-time — verified by grep).
**How to avoid:** Remove line 45 as part of the change; keep the autouse fixture function-scoped (default). Empty dirs are not tracked by git, but creating one violates the "zero repo writes" criterion in spirit and leaves clutter.
**Warning signs:** `git status` clean but `tests/inference/pdf/` reappears after a run.

### Pitfall 8: numpy version span in the new tests
**What goes wrong:** Tests written against numpy 2.5.3 semantics behave differently under CI's numpy 1.26.4 leg.
**Why it happens:** CI matrix pins numpy 1.26.4 and 2.2.0 (CLAUDE.md); local venv has 2.5.3.
**How to avoid:** All new test code uses only `np.array`/`np.arange`/`np.unique`/`np.setdiff1d`/`np.array_equal` — stable across the span (no 2.x-only APIs). Verify by running the new tests once under the CI-shaped numpy pin if practical; the risk here is genuinely low (basic array ops only).
**Warning signs:** None expected; listed for the verifier's matrix check.

## Runtime State Inventory

Not a rename/refactor/migration phase — skipped per protocol (greenfield-style fixes; no stored data, service config, OS-registered state, secrets, or build artifacts reference the changed strings). One adjacent cleanup recorded under FIX-04: 9 untracked PDF artifacts in `tests/inference/pdf/` are runtime-generated state to delete (they are not tracked, so no history rewrite is needed).

## Code Examples

### FIX-01: unskipped parametrized test (test body rewrite)
```python
# tests/tasks/test_metrics.py — parametrize entry change (was ("multiclass", 2, ["A", "B"]))
        (
            "multiclass",
            3,
            ["A", "B", "C"],
        ),  # 3 classes: sklearn's ovr/macro path needs >2 unique target values
```
```python
# test body branch rewrite (all 3 class ids must appear in labels)
    elif task_type == "multiclass":
        logits = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
        labels = np.array([1, 0, 2])
```
And delete the skip at 759-761 (verbatim current: `if task_type == "multiclass":` / `pytest.skip("Multiclass AUROC implementation has issues")`).

### FIX-01: plot test (replaces the unconditional skip at :298)
```python
    def test_multi_classification_metrics_with_plot(self):
        """Test multi-class classification metrics with plotting data."""
        label_list = ["label1", "label2", "label3"]
        compute_func = multi_classification_metrics(label_list, plot=True)

        logits = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
        labels = np.array([1, 0, 2])  # every class present

        metrics = compute_func((logits, labels))

        assert "curve" in metrics
        assert set(metrics["curve"]) == {"fpr", "tpr", "precision", "recall"}
```
The `metrics["curve"]` keys are verbatim from the implementation [VERIFIED: dnallm/tasks/metrics.py:357-362 — `"fpr"`, `"tpr"`, `"precision"`, `"recall"`].

### FIX-01: absent-class edge regression test (decision-mandated)
```python
    def test_multi_classification_metrics_missing_class_raises(self):
        """A batch lacking a class must fail honestly, not return nan metrics."""
        compute_func = multi_classification_metrics(["label1", "label2", "label3"])

        logits = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1]])
        labels = np.array([0, 1, 0])  # class 2 absent

        with pytest.raises(ValueError, match=r"missing class id\(s\)"):
            compute_func((logits, labels))
```
(The match string must be kept in sync with the implemented message — executor discretion on wording, regex-matchable per project convention.)

### FIX-02: sentinel fault-injection regression test (TestLoadModelAndTokenizer style)
```python
    def test_load_model_crossdna_result_not_overwritten(self):
        """CrossDNA handler result must survive the dispatch chain verbatim."""
        task_config = TaskConfig(task_type="mask", num_labels=None)
        sentinel_model = Mock()
        sentinel_model.to = Mock(return_value=sentinel_model)  # model.py:886 rebinds via .to()
        sentinel_tokenizer = Mock()
        other = (Mock(), Mock())

        with (
            patch("dnallm.models.model._setup_huggingface_mirror"),
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=(
                    "/models/CrossDNA-8.1M",  # contains "crossdna" case-insensitively
                    {"AutoTokenizer": Mock(), "AutoModelForMaskedLM": Mock()},
                ),
            ),
            patch("dnallm.models.model._create_label_mappings", return_value=({}, {})),
            patch(
                "dnallm.models.model._handle_crossdna_models",
                return_value=(sentinel_model, sentinel_tokenizer),
            ),
            patch(
                "dnallm.models.model._handle_dnabert2_models",
                return_value=other,  # would have overwritten pre-fix
            ),
            patch("dnallm.models.model._configure_model_padding"),
        ):
            model, tokenizer = load_model_and_tokenizer(
                "CrossDNA-8.1M", task_config, source="local"
            )

            assert model is sentinel_model
            assert tokenizer is sentinel_tokenizer
```
Patch targets verified: both handlers are imported into `dnallm.models.model` namespace [VERIFIED: dnallm/models/model.py:20-33 — `from .special import (` includes `_handle_dnabert2_models` and `_handle_crossdna_models`]; patching at the import site is the established style (existing tests at tests/models/test_model.py:512-514). Optional stronger variant: also patch `_load_model_by_task_type` with `side_effect=AssertionError("generic loader must not run")` to prove the chain never falls through.

### FIX-03: census ground truth (verbatim junit skip messages, for allowlist seeding)
All 9 skip events from the Phase-1 full census [VERIFIED: .planning/phases/01-harness-integrity-measured-baseline/junit-full.xml, parsed this session with ElementTree]:
```
tests.examples.test_examples.TestNotebookExamples::test_notebook_imports[notebooks/data_prepare/predict/predict_data.ipynb]  ->  'No import statements found'
tests.tasks.test_metrics.TestMultiClassificationMetrics::test_multi_classification_metrics_with_plot  ->  'Multi-class plotting with AUROC requires complex implementation'
tests.tasks.test_metrics::test_compute_metrics_task_types[multiclass-2-label_names1]  ->  'Multiclass AUROC implementation has issues'
dnallm.mcp.tests.test_sse_client.TestSSEClient::test_sse_connection  ->  'SSE connection failed: unhandled errors in a TaskGroup (1 sub-exception)'
dnallm.mcp.tests.test_sse_client.TestSSEClient::test_health_check_tool  ->  'Health check test failed: unhandled errors in a TaskGroup (1 sub-exception)'
dnallm.mcp.tests.test_sse_client.TestSSEClient::test_dna_prediction_tool  ->  'DNA prediction test failed: unhandled errors in a TaskGroup (1 sub-exception)'
dnallm.mcp.tests.test_streamable_http_client.TestStreamableHTTPClient::test_streamable_http_connection  ->  'Streamable HTTP connection failed: unhandled errors in a TaskGroup (1 sub-exception)'
dnallm.mcp.tests.test_streamable_http_client.TestStreamableHTTPClient::test_streamable_http_session_reuse  ->  'Streamable HTTP session reuse test failed: unhandled errors in a TaskGroup (1 sub-exception)'
dnallm.mcp.tests.test_streamable_http_client.TestStreamableHTTPClient::test_streamable_http_custom_url  ->  'Streamable HTTP custom URL test failed: unhandled errors in a TaskGroup (1 sub-exception)'
```
After Phase 2: rows 2-3 disappear (FIX-01); rows 4-9 persist but with rewritten `network-unavailable:` messages (FIX-03); row 1 persists unchanged. Expected post-phase skip population on a full local run: 7 (6 network + 1 content); on the CI fast leg (`-m "not slow"`): 1 (content only).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `except Exception: pytest.skip` in live-network tests | Typed transport errors + ExceptionGroup unwrapping | anyio/anyio-based SDKs (mcp 1.x) always wrapped in TaskGroups; Python 3.11 native ExceptionGroup (2022) | Skip logic must flatten groups; binary `except X` never matches the top-level type |
| String-matching exception text (`"connection" in str(e)`) | Exception-type tuples | long-standing best practice | String matching is what made the test_model.py sites dead code — exception text does not survive wrapping |
| `pytest.skip` for known-broken code paths | Fix the code; skip only for environment | This milestone's whole thesis | Phase-2 success criteria 1-2 are the application |

**Deprecated/outdated:**
- `pytest.ini`-based config: already deleted in Phase 1 (HARN-01) — do not reintroduce per-file config for these tests.
- The dead `codecov-action@v3` step (ci.yml:88): Phase-4 GATE-03 scope; do not touch in Phase 2.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Multiclass labels are always integer ids `0..C-1` in eval batches (so `np.arange(len(label_list))` is the right expected-class set) | Pattern 1 | If some task produced non-contiguous label ids, the presence check would reject valid batches; HF Trainer conventions make this near-impossible in this codebase (labels come from `np.argmax`/tokenized class ids) |
| A2 | Skipping (not failing) remains the desired behavior for the 6 MCP live-server tests when no server runs | Pattern 3 | Locked by CONTEXT decision (typed network skips + allowlist) — listed only because the tests connect to `http://localhost:8000` with no fixture starting a server, which is unusual; if the owner actually wants them to self-host a server, that is a different (larger) change |
| A3 | sklearn macro-ovr `labels=` behavior on absent classes (nan on 1.9.1, raise on older) may vary across `>=1.4.0` | Pitfall 4 | Neutralized by placing the presence check before the sklearn call — behavior becomes version-independent either way; residual risk only if the guard is omitted |
| A4 | The `pdf` marker should be applied at class level to classes whose methods write PDFs (exact granularity is executor discretion) | Pattern 5 | If applied too narrowly, `pytest -m pdf` deselects some PDF writers and the "tree stays clean" check under-tests; too broadly is harmless |
| A5 | junit `<skipped message>` carries the skip reason string verbatim for both runtime skips and skipif conditions in all pytest 8.4+ | Pitfall 6 | ElementTree census parse proved it for runtime skips this session; skipif-reason representation is inferred — Pitfall 6 mitigations (local CI-shaped run before freezing the allowlist) cover it |

**All other claims were verified this session** by direct file reads (paths + line ranges quoted) or live probes executed in `.venv` (sklearn 1.9.1 / numpy 2.5.3 / httpx 0.28.1 / pytest 9.1.1 environment).

## Open Questions

1. **Dead skip sites in `tests/models/test_model.py:105-108/127-130` — delete or restructure?**
   - What we know: they are unreachable (Pitfall 5); offline runs already fail honestly through the wrapped ValueError.
   - What's unclear: whether local-offline developer experience matters enough to keep a skip (option b) versus clean deletion (option a).
   - Recommendation: option (a) delete — CI slow legs require network by owner decision (CLAUDE.md), so no CI-legitimate skip case exists; smallest honest diff. Planner should encode one option explicitly; neither contradicts the locked decision.
2. **Should the skip-audit step also gate the `test-cuda`/`test-mamba` legs?**
   - What we know: those legs run `pytest tests/ -m "not slow"` without junit (ci.yml:172, 218); mamba leg is `continue-on-error: true`.
   - What's unclear: owner appetite for per-leg audits in Phase 2 vs Phase 4 GATE consolidation.
   - Recommendation: Phase 2 wires only the main `test` job (matches the CONTEXT integration point "ci.yml gains the skip-audit step consuming the run's junit" — singular); note the other legs in the SUMMARY for GATE-02.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| pytest + plugins (cov/timeout/asyncio) | all fixes | ✓ | 9.1.1 / cov 7.1.0 / timeout 2.4.0 / asyncio 1.4.0 | — |
| scikit-learn | FIX-01 | ✓ | 1.9.1 (floor >=1.4.0) | — |
| httpx | FIX-03 typing | ✓ | 0.28.1 (via mcp) | — |
| mcp SDK | FIX-03 sites | ✓ | 1.30.0 installed (range >=1.3.0,<2) | — |
| PyYAML | audit script | ✓ | >=6.0 (core dep) | — |
| numpy | tests | ✓ | 2.5.3 local; CI legs pin 1.26.4/2.2.0 | — |
| Live localhost:8000 MCP server | 6 network tests | ✗ (by design) | — | typed skip (that is the feature) |
| Network (HF/ModelScope) | slow download tests | ✓ local (warm caches from Phase-1 audit); CI required | — | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** live MCP server — the typed skip IS the designed fallback.

## Security Domain

ASVS level 1 (config: `security_asvs_level: 1`, `security_enforcement: true`). This phase modifies tests, two library functions, CI YAML, and a data file — no new attack surface, but three touchpoints deserve explicit treatment:

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth changes (MCP tests connect to localhost dev server; the CONCERNS.md 0.0.0.0 finding is out of scope) |
| V3 Session Management | no | No session code touched |
| V4 Access Control | no | No authorization code touched |
| V5 Input Validation | yes (weakly) | The FIX-01 presence check is input validation on eval batches — reject-and-raise (`ValueError`) per project convention; the audit script treats junit XML and YAML as data, validated by structure (ElementTree parse errors surface as script failure, which fails the step — fail-closed) |
| V6 Cryptography | no | No crypto touched |
| V10 Malicious Code | yes (attention point) | Pre-existing `exec()` in `tests/examples/test_examples.py:144,258` executes notebook-derived import statements — NOT touched by this phase; the skip sites there are only allowlisted. `trust_remote_code=True` paths untouched. No new `exec`/`eval` introduced anywhere in this phase's patterns |

### Known Threat Patterns for this change set

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed junit artifact feeding the audit step (CI-tampered input) | Tampering | ElementTree parses defensively; unparseable/absent file → step failure (fail-closed); matches Phase-1 T-01-04 untrusted-artifact stance |
| Allowlist as a suppression vector (someone adds a broad prefix to hide a crash-skip) | Repudiation | `tests/expected_skips.yaml` is reviewable data in git; prefixes kept narrow (`network-unavailable:`, not `.*`); audit output prints every matched/unmatched skip for the log |
| `.gitignore` directory entry hiding future stray artifacts | — | Accepted by decision (belt-and-suspenders); the tmp_path rebind removes the write, the ignore covers the residue |

## Sources

### Primary (HIGH confidence)
- Direct reads this session (all quoted with path:line in situ): `dnallm/tasks/metrics.py` (236, 261-284, 317-362, 631-660), `dnallm/models/model.py` (317-391, 770-893), `dnallm/models/special/{gpn,omnidna,dnabert2,crossdna}.py`, `dnallm/configuration/configs.py` (122-150), `tests/tasks/test_metrics.py` (215-300, 700-800), `tests/models/test_model.py` (90-131, 470-559), `tests/inference/test_plot.py` (1-50, 85-150), `tests/utils/test_cuda_compat.py`, `dnallm/mcp/tests/{test_sse_client,test_streamable_http_client}.py`, `.gitignore` (128-141), `.github/workflows/ci.yml`, `pyproject.toml` (pytest/dep sections)
- Live probes executed this session in `.venv`: sklearn AUROC/AUPRC behavior matrix (all-present / absent-class / 2-class × with/without `labels=`); MCP client exception shapes against dead localhost:8000 (ExceptionGroup → httpx.ConnectError, MRO); httpx version/MRO
- `.planning/phases/01-harness-integrity-measured-baseline/junit-full.xml` — re-parsed this session (9 skip events, verbatim messages/test-ids above)
- `.planning/phases/01-harness-integrity-measured-baseline/01-AUDIT-REPORT.md` — census context, coverage worklist, cold/warm evidence

### Secondary (MEDIUM confidence)
- `.planning/codebase/CONCERNS.md:55-58, 166-168` — prior root-cause notes (confirmed by this session's probes; the "try/except→NaN or labels=arange" framing is refined by the nan finding in Pitfall 4)
- Knowledge graph `.planning/graphs/graph.json` — queried for the dispatch chain; confirmed call structure already established by direct reads; **note: commit-stale (22 commits behind, built at 3df056c)** — treated as approximate, nothing load-bearing rests on it

### Tertiary (LOW confidence)
- None — no training-data claims were load-bearing; every external-behavior claim (sklearn/httpx/pytest semantics) was replaced by a live probe this session

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; everything verified installed
- Architecture (fix shapes): HIGH — verbatim code reads + live probes for every claim; the two counterintuitive findings (2-class dead end, nan-not-raise) are probe-proven
- Pitfalls: HIGH — each pitfall is the direct negative result of a probe or verbatim read performed this session
- Allowlist/junit details: MEDIUM — runtime-skip messages verified from artifact; skipif-reason junit representation inferred (A5, mitigated by a local CI-shaped run before freezing)

**Research date:** 2026-09-30
**Valid until:** 2026-10-30 (stable domain — pinned versions and file lines; re-verify line numbers only if `dnallm/tasks/metrics.py` or `dnallm/models/model.py` change before planning)
