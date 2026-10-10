# Phase 3: Test Authoring to >90% Coverage - Pattern Map

**Mapped:** 2026-09-30
**Files analyzed:** 23 (17 new test files, 6 extended/touchpoint files)
**Analogs found:** 19 / 23 (4 files or file-portions rely on RESEARCH.md patterns with no codebase analog)

All analog paths verified git-tracked (`git ls-files` non-empty). No gitignored/mirror paths referenced.

## File Classification

Ordered by the locked wave sequence (ranked worklist: inference → models → mcp → datahandling/finetune → cli/compat + orphans).

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `tests/inference/test_inference.py` (extend) | test | transform (logits→preds), batch | itself + `tests/conftest.py` | exact |
| `tests/inference/test_mutagenesis.py` (new) | test | batch (saturation scan), transform | `tests/inference/test_inference.py` | role-match |
| `tests/inference/test_interpret.py` (new) | test | batch (captum attributions) | `tests/inference/test_inference.py` + `tests/mcp/test_interpret_tool.py` | role-match |
| `tests/inference/test_plot.py` (extend) | test | transform (altair spec), file-I/O (pdf) | itself (pdf fixture lines 154-157) | exact |
| `tests/benchmark/test_benchmark.py` (extend) | test | batch (multi-model × multi-dataset aggregation) | itself (existing shallow coverage — extend, do not duplicate) | exact |
| `tests/models/test_model.py` (extend) | test | transform (fault-injection) | itself (sentinel test lines 487-532) | exact |
| `tests/models/test_tokenizer.py` (new) | test | transform (staged-failure chain), file-I/O | `tests/models/test_model.py` + `tests/datahandling/test_dna_dataset.py` | role-match |
| `tests/models/test_head.py` (new) | test | transform (real torch forwards) | `tests/tasks/test_metrics.py` (real numerical asserts) | partial |
| `tests/models/test_losses.py` (new) | test | transform (real torch tensors) | `tests/tasks/test_metrics.py` | partial |
| `tests/models/test_special/test_crossdna.py` + family files (new) | test | transform (dispatch + real forwards) | `tests/models/test_model.py:487-532` | role-match |
| `tests/mcp/test_server_transports.py` (new) | test | request-response (in-memory ASGI) | `dnallm/mcp/tests/test_server_integration.py`; in-memory pair = RESEARCH Pattern 1 (no codebase analog) | role-match |
| `tests/mcp/test_server_streaming.py` (new) | test | streaming (progress coroutines) | `tests/mcp/test_timeout.py` | exact-role |
| `tests/mcp/test_model_manager.py` (new) | test | request-response (async load/unload) | `dnallm/mcp/tests/test_server_integration.py:129-141` | role-match |
| `tests/mcp/test_start_server.py` (new) | test | file-I/O (loguru handler), request-response | `tests/utils/test_cuda_compat.py` | partial |
| `tests/datahandling/test_dna_dataset.py` (extend) | test | file-I/O (format round-trips) | itself (loaders lines 92-211) | exact |
| `tests/finetune/test_trainer.py` (new) | test | batch (Trainer wiring, mocked boundary) | `tests/finetune/test_trainer_real_model.py` (construction only) | role-match |
| `tests/cli/test_cli.py` (new) | test | request-response (CliRunner) | NONE — no CliRunner usage in repo; use RESEARCH + `tests/mcp/test_client_sdk.py` style | none |
| `tests/utils/test_transformers_compat.py` (new) | test | transform (behavior contract) | `tests/utils/test_cuda_compat.py` | exact-role |
| `tests/utils/test_logger.py` (new) | test | transform (logging config) | `tests/utils/test_sequence.py` | role-match |
| `tests/tasks/test_metrics.py` (extend, orphan) | test | transform (metrics) | itself (parametrize line 770) | exact |
| `tests/configuration/test_configs.py` (extend, orphan) | test | transform (Pydantic validation) | itself | exact |
| `tests/conftest.py` (touchpoint) | test-infra | — | itself | exact |
| `tests/expected_skips.yaml` (touchpoint) | test-infra | — | itself | exact |

Note: `tests/models/test_special/` needs NO `__init__.py` — no `tests/*` subdirectory has one (only `dnallm/mcp/tests/__init__.py` exists).

## Pattern Assignments

### Wave A — inference

#### `tests/inference/test_inference.py` (extend) + `tests/inference/test_mutagenesis.py` (new) + `tests/inference/test_interpret.py` (new)

**Analog:** `tests/inference/test_inference.py`

**Engine construction pattern** (lines 31-58): build a real config YAML in a temp dir, `load_config` it, pass mock model/tokenizer into the class under test:

```python
def setUp(self):
    """Set up test fixtures."""
    self.test_dir = tempfile.mkdtemp()
    self.config_path = os.path.join(self.test_dir, "test_config.yaml")
    ...
    self.inference_engine = DNAInference(
        model=self.mock_model,
        tokenizer=self.mock_tokenizer,
        config=self.load_test_config(),
    )
```

Config YAML content (lines 62-76): `inference:` block (batch_size/device/max_length/num_workers/output_dir/use_fp16) + `task:` block (label_names/num_labels/task_type/threshold). `Mutagenesis`/`DNAInterpret`/`Benchmark` tests reuse this exact config shape.

**Logits→predictions assertion style** (lines 214-226): real `torch.tensor` inputs, assert both shapes AND semantic labels:

```python
def test_logits_to_preds_binary(self):
    """Test logits to predictions conversion for binary classification."""
    logits = torch.tensor([[1.0, 2.0], [0.5, 1.5], [2.0, 1.0]])
    probs, labels = self.predictor.logits_to_preds(logits)

    assert len(labels) == 3
    assert probs.shape == (3, 2)
    expected_labels = ["Core promoter", "Core promoter", "Not promoter"]
    assert labels == expected_labels
```

**Mock model shape** (lines 95-125): `Mock()` with `.config` attrs, `parameters()` iterator, a `mock_forward` returning `.logits` tensor, `eval`, and `.to = Mock(return_value=mock_model)` (self-return — REQUIRED for identity assertions after device moves). For new tests prefer the shared `tests/conftest.py` fixtures (see Shared Patterns) over re-declaring these.

For `test_interpret.py` / `test_mutagenesis.py` real-torch paths: use tiny real `torch.nn.Module`s on CPU (RESEARCH: ~28ms per captum run; outputs shaped `(batch, n_classes)` — scalar-summed outputs crash captum target selection). Mock-only engine paths reuse the conftest fixtures. `tests/mcp/test_interpret_tool.py` (lines 9-50, 94-119) documents the attribute surface a fake engine must expose for interpret (`pad_token_id`, `all_special_ids`, `mask_token_id`, `convert_ids_to_tokens`, `__call__` returning tensor dict).

---

#### `tests/inference/test_plot.py` (extend)

**Analog:** itself

**PDF artifact discipline** (lines 44-46 + 154-157) — every PDF-writing test class gets `@pytest.mark.pdf` (existing at lines 263, 423, 599, 767, 947, 1177, 1520, 1663, 1888) and artifacts go to `tmp_path` via an autouse fixture rebinding a module-global:

```python
# Define PDF output directory; rebound per test to tmp_path by the pdf_output_dir
# autouse fixture below. Directory creation happens inside create_pdf_file.
PDF_OUTPUT_DIR = Path(__file__).parent / "pdf"

@pytest.fixture(autouse=True)
def pdf_output_dir(tmp_path, monkeypatch):
    """Write all PDF artifacts under tmp_path so the repo working tree stays clean."""
    monkeypatch.setattr(sys.modules[__name__], "PDF_OUTPUT_DIR", tmp_path)
```

File-existence assertions (lines 150-151): `assert os.path.exists(file_path)` + `os.path.getsize(file_path) > 0`. New plot tests that only build altair chart objects need no marker; only actual PDF writers join the `pdf` class marker.

---

### Wave B — models

#### `tests/models/test_model.py` (extend) + `tests/models/test_special/*` (new)

**Analog:** `tests/models/test_model.py`

**Retry fault-injection idiom** (lines 40-54) — `side_effect` list on the downloader + `patch("time.sleep")` on EVERY retry-path test (Pitfall 6; `time.sleep(1)` at `dnallm/models/model.py:372`):

```python
def test_download_success_after_retries(self):
    """Test successful download after multiple retries."""
    mock_downloader = Mock(
        side_effect=[
            Exception("connection error"),
            Exception("connection error"),
            "/path/to/model",
        ]
    )

    with patch("time.sleep"):  # Mock sleep to speed up test
        result = download_model("test-model", mock_downloader, max_try=3)

    assert result == "/path/to/model"
    assert mock_downloader.call_count == 3
```

Always pair with call-count assertions (`mock_downloader.call_count == N`) — this is the observable-behavior evidence TEST-06 requires.

**Sentinel fault-injection dispatch template** (lines 487-532) — THE pattern for per-family selection and patch-all-but-one fall-through tests (RESEARCH Pattern 3). Note the `.to()` self-return comment:

```python
def test_load_model_crossdna_result_not_overwritten(self):
    """CrossDNA handler result must survive the dispatch chain verbatim."""
    task_config = TaskConfig(task_type="mask", num_labels=None)
    sentinel_model = Mock()
    # load_model_and_tokenizer rebinds the model via .to(device); a plain
    # Mock would return a fresh child mock and break the identity check.
    sentinel_model.to = Mock(return_value=sentinel_model)
    sentinel_tokenizer = Mock()
    other_model, other_tokenizer = Mock(), Mock()

    with (
        patch("dnallm.models.model._setup_huggingface_mirror"),
        patch("dnallm.models.model._handle_evo2_models", return_value=None),
        patch("dnallm.models.model._handle_evo1_models", return_value=None),
        patch("dnallm.models.model._handle_gpn_models", return_value=None),
        patch(
            "dnallm.models.model._get_model_path_and_imports",
            return_value=(
                "/models/CrossDNA-8.1M",
                {"AutoTokenizer": Mock(), "AutoModelForMaskedLM": Mock()},
            ),
        ),
        ...
        patch(
            "dnallm.models.model._handle_crossdna_models",
            return_value=(sentinel_model, sentinel_tokenizer),
        ),
        ...
        patch(
            "dnallm.models.model._load_model_by_task_type",
            side_effect=AssertionError("generic loader must not run"),
        ),
        patch("dnallm.models.model._configure_model_padding"),
    ):
        model, tokenizer = load_model_and_tokenizer(
            "CrossDNA-8.1M", task_config, source="local"
        )

        assert model is sentinel_model
        assert tokenizer is sentinel_tokenizer
```

Two techniques to reuse: (a) `side_effect=AssertionError(...)` on the loader that must NOT run; (b) patch handlers at `dnallm.models.model._handle_<family>_models` (the import site in the dispatch module). Dispatch-chain line map (RESEARCH-verified): early-return handlers evo2:773 / evo1:778 / megadna:786 / enformer:799 / space:810 / borzoi:821; import-gate discards gpn:783 / omnidna:796; first-resolved-wins chain crossdna:863-874 → dnabert2:875-876 → generic:877-878; mutbert/basenji2 post-processing :880-883.

For special-family handler bodies whose deps are absent (evo, evo2, enformer, borzoi, seqmodels, megadna, gpn, multiomics): `monkeypatch.setitem(sys.modules, "evo2", fake_module)` with shaped `types.ModuleType` fakes — monkeypatch auto-restores (Pitfall 10). `test_crossdna.py` needs NO stubs (module imports torch/transformers only).

**Parametrize convention** (lines 647-655) — tuple-of-names style, one behavior per row:

```python
@pytest.mark.parametrize(
    ("task_type", "expected_problem_type"),
    [
        ("binary", "single_label_classification"),
        ("multiclass", "single_label_classification"),
        ("multilabel", "multi_label_classification"),
        ("regression", "regression"),
    ],
)
def test_load_model_by_task_type_problem_types(task_type, expected_problem_type):
```

**Imports pattern** (lines 8-25): absolute imports from `dnallm.models.model` including private functions (`_load_model_by_task_type` etc.) — tests import privates directly; stdlib `unittest.mock` Mock/patch/MagicMock.

---

#### `tests/models/test_tokenizer.py` (new)

**Analog:** `tests/models/test_model.py` (staged-failure idiom) + `tests/datahandling/test_dna_dataset.py` (round-trip file tests)

Staged failures use the `side_effect` exception list exactly as `TestDownloadModel` does; tier assertions per RESEARCH (tier-2 = assert `"loaded fast tokenizer"` warning; tier-3 = `isinstance(tok, DNAOneHotTokenizer)` + `"using DNAOneHotTokenizer"` warning). `DNAOneHotTokenizer` round-trips (`save_pretrained`/`from_pretrained`) follow the datahandling tmp-file pattern:

```python
with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
    ...
    temp_file = f.name
try:
    ...  # exercise + assert
finally:
    os.unlink(temp_file)
```

(`tests/datahandling/test_dna_dataset.py:164-176`; newer tests may use pytest `tmp_path` directly — both exist in repo; prefer `tmp_path` for new files.)

#### `tests/models/test_head.py` + `tests/models/test_losses.py` (new)

No existing test exercises real torch `nn.Module` forwards. Closest numerical-behavior analog: `tests/tasks/test_metrics.py` (real tensors in, exact/`pytest.approx` values out; parametrize at line 770 with `("task_type", "num_labels", "label_names")` tuple-names style). Head tests construct the 7 real head classes with small dims, run `forward(torch.randn(batch, seq_len, hidden))`, assert output shape + differentiability (`loss.backward()`; `param.grad is not None`) — mocks cannot produce gradients (RESEARCH anti-pattern).

---

### Wave C — mcp

#### `tests/mcp/test_server_streaming.py` (new)

**Analog:** `tests/mcp/test_timeout.py` — same class under test, same fixture strategy.

**mock_server fixture** (lines 20-49) — patch BOTH collaborators at their import site, then build a real `DNALLMMCPServer`:

```python
@pytest.fixture
def mock_server(self):
    """Create a mock server with minimal setup for timeout tests."""
    with patch("dnallm.mcp.server.MCPConfigManager") as mock_cm:
        with patch("dnallm.mcp.server.ModelManager"):
            mock_config = MagicMock()
            mock_config.mcp.name = "Test Server"
            ...
            mock_cm_instance = MagicMock()
            mock_cm_instance.get_server_config.return_value = mock_config
            mock_cm_instance.get_timeout_config.return_value = {"tool_timeout_seconds": 30}
            mock_cm_instance.get_logging_config.return_value = {"log_format": "text"}
            mock_cm.return_value = mock_cm_instance

            server = DNALLMMCPServer("dummy_config.yaml")
            server._tool_timeout_seconds = 30
            server._log_format = "text"
            return server
```

**Streaming-tool invocation with AsyncMock context** (lines 100-125, 244-265):

```python
mock_server.model_manager = MagicMock()
mock_server.model_manager.predict_sequence = AsyncMock(
    return_value={"probabilities": [0.5, 0.5]}
)

mock_context = AsyncMock()

result = await mock_server._dna_stream_predict(
    sequence="ATCG",
    model_name="test_model",
    stream_progress=True,
    context=mock_context,
)

assert result.get("isError") is not True
assert result["streamed"] is True
```

New tests extend this to RESEARCH Pattern 2: assert ORDERED `mock_context.report_progress.call_args_list` (0→25→75→100 single; `i/total` batch), assert final dict, and inject mid-stream faults via `side_effect` lists on `predict_sequence` (raise on Nth call; `None` return builds the `{"result": None, "error": f"Prediction failed for sequence {i + 1}", "index": i}` dict — server.py:935-940). Also close: generic exceptions PROPAGATE out of `_with_timeout_wrapper` (only `asyncio.TimeoutError` is caught) and the `_structured_log` json branch (server.py:370-385). Async tests keep the explicit `@pytest.mark.asyncio` decorator used throughout this file (config also has `--asyncio-mode=auto`).

A leaner variant of the fixture (Mock instead of MagicMock chains) is at `tests/mcp/test_interpret_tool.py:53-77` — use it when only `get_inference_engine`/`predict_sequence` matter.

#### `tests/mcp/test_server_transports.py` (new)

**Analog (construction level):** `dnallm/mcp/tests/test_server_integration.py`

Real config files written via `yaml.dump` into a temp dir, then real server init with the load boundary patched (lines 121-141):

```python
with patch("dnallm.mcp.model_manager.load_model_and_tokenizer") as mock_load:
    mock_model = Mock()
    mock_tokenizer = Mock()
    mock_load.return_value = (mock_model, mock_tokenizer)

    server = DNALLMMCPServer(config_path)
    await server.initialize()

    assert server._initialized is True
    assert server.app is not None
```

Config dict shape: `server`/`mcp`/`models`/`multi_model`/`sse`/`streamable_http`/`logging` blocks (lines 186-227; note `streamable_http: {host, port, path}`). Note the inline `# ruff: ignore[hardcoded-bind-all-interfaces]` comments on `0.0.0.0` values.

**In-memory client/server pair: NO codebase analog** — copy RESEARCH Pattern 1 verbatim (httpx `ASGITransport` factory + `asgi_app.router.lifespan_context(asgi_app)` + `streamablehttp_client(...)` + `ClientSession`); base_url MUST be `http://localhost:8000` (421 pitfall). One smoke test asserting all 13 tools listable.

#### `tests/mcp/test_model_manager.py` (new)

**Analog:** `dnallm/mcp/tests/test_server_integration.py:129-132` — patch `dnallm.mcp.model_manager.load_model_and_tokenizer` returning `(Mock(), Mock())`; async lifecycle tests (`initialize`/`shutdown` style at lines 158-170) for load/unload/status routing.

#### `tests/mcp/test_start_server.py` (new)

**Analog:** `tests/utils/test_cuda_compat.py` (module-contract tests; no closer match exists)

```python
def test_preload_is_idempotent():
    """Repeated calls must be safe no-ops."""
    cuda_compat.preload_cuda13_libs()
    cuda_compat.preload_cuda13_libs()
    assert cuda_compat._preloaded is True


def test_preload_noops_without_cuda13_wheels(monkeypatch):
    """On cpu/cu12/rocm environments nothing must load or raise."""
    monkeypatch.setattr(cuda_compat, "_preloaded", False)
    monkeypatch.setattr(cuda_compat, "_cuda13_wheels_available", lambda: False)
```

Module-state assertions + `monkeypatch.setattr` for flags. MANDATORY: `monkeypatch.chdir(tmp_path)` in a fixture — `setup_logging` creates `logs/mcp_server.log` under CWD (start_server.py:34-43; RESEARCH Pitfall 5, twice-run tree-clean gate is the tripwire).

---

### Wave D — datahandling/finetune

#### `tests/datahandling/test_dna_dataset.py` (extend)

**Analog:** itself

**Format round-trip template** (lines 92-107 for csv; tsv/json/parquet/pkl/fasta/txt through 211):

```python
def test_load_csv_file(self):
    """Test loading CSV file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("sequence,label\n")
        f.write("ATCG,0\n")
        f.write("GCTA,1\n")
        f.write("TAGC,0\n")
        temp_file = f.name

    try:
        dna_ds = DNADataset.load_local_data(temp_file, seq_col="sequence", label_col="label")
        assert len(dna_ds) == 3
        assert "sequence" in dna_ds.dataset.column_names
        assert "labels" in dna_ds.dataset.column_names
    finally:
        os.unlink(temp_file)
```

Cheap real-behavior tests via tmp files — no mocks. Validation-error style (lines 72-86): `pytest.raises(ValueError, match="max_length must be positive")`. New tests extend to: tokenization, reverse-complement augmentation, splitting, stats, HF/ModelScope loaders (mock at the `datasets`/`modelscope` import site like `tests/models/test_model.py:214-231` does for modelscope).

#### `tests/finetune/test_trainer.py` (new)

**Analog:** `tests/finetune/test_trainer_real_model.py` (construction sequence ONLY — lines 37-44 config load, 100-102 `DNATrainer(model=..., config=..., datasets=...)`)

New fast unit tests mock at the HF `Trainer` boundary (patch `dnallm.finetune.trainer.Trainer` / `TrainingArguments` at their import site — same site-patching rule as `dnallm.models.model._handle_*`). Reuse `tests/finetune/test_finetune_config.yaml` (tracked fixture asset) via `load_config`. Do NOT copy the real-model file's `try/except Exception: self.skipTest(...)` broad-skip — that is the hand-rolled skip anti-pattern (RESEARCH); fast tests need no skips. Guard optional deps with the established try/except-import + module-level `pytest.skip(allow_module_level=True)` shape only if genuinely required (see `tests/mcp/test_client_sdk.py` header convention).

---

### Wave E — cli/compat + orphans

#### `tests/utils/test_transformers_compat.py` (new)

**Analog:** `tests/utils/test_cuda_compat.py` — the direct sibling contract-test file (34 lines, full read above)

Same structure: idempotency test via object identity (`before is after`), monkeypatched module flags for negative paths, `pytest.mark.skipif(sys.platform != "linux", ...)` only for genuinely platform-bound checks. Contract surface from RESEARCH Pattern 5: (a) `apply_patches()` idempotency; (b) patched `get_parameter_or_buffer` returning `_QuantStatProxy`; (c) proxy attribute forwarding + `_is_hf_initialized` drop (transformers_compat.py:209-212); (d) `_iter_uninitialized_quantized_weights` classification; (e) `initialize_weights` passthrough/swap/restore with `bitsandbytes.functional.dequantize_4bit`/`quantize_4bit` monkeypatched.

#### `tests/utils/test_logger.py` (new)

**Analog:** `tests/utils/test_sequence.py` — plain module-level `test_*` functions (no class grouping needed for simple pure functions), direct value assertions. Logger tests asserting handler state after `setup_logging(...)` should clean up loguru handlers per-test (loguru `logger.remove()` semantics) to avoid cross-test handler accumulation.

#### `tests/cli/test_cli.py` (new) — see No Analog Found

Style anchor while authoring: `tests/mcp/test_client_sdk.py` header (lines 8-21) — the repo's modern test-file preamble:

```python
from __future__ import annotations

import ...
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
```

Lazy imports inside CLI command bodies mean tests must patch at the command's import site (e.g. `dnallm.cli.cli.DNATrainer` resolves inside `train()` — patch `dnallm.cli.cli.DNATrainer` or the origin module depending on where the function-local import binds; verify per command).

#### `tests/tasks/test_metrics.py` + `tests/configuration/test_configs.py` (extend, orphans)

Both extend themselves — follow each file's existing class grouping and the tuple-names parametrize style (`tests/tasks/test_metrics.py:770-776`).

## Shared Patterns

### Mock model/tokenizer fixtures (use, don't rebuild)

**Source:** `tests/conftest.py`
**Apply to:** all inference/models/mcp engine-consuming tests

`mock_model` (lines 16-59, `.to()` self-return at 55), `mock_tokenizer` (62-102), `mock_config` (105-125), `mock_inference_engine` (138-161), `mock_dataset` (164-189), `sample_dna_sequence` (128-135). New cross-file fixture needs (e.g. a tiny real torch backbone, a shared mock-server) go HERE — one place, session-visible.

### DNALLMMCPServer mock fixture

**Source:** `tests/mcp/test_timeout.py:20-49` (full) / `tests/mcp/test_interpret_tool.py:53-77` (lean)
**Apply to:** all new `tests/mcp/` files that instantiate the server

Patch `dnallm.mcp.server.MCPConfigManager` + `dnallm.mcp.server.ModelManager`, build real server with `"dummy_config.yaml"`, then override `server.model_manager` per test.

### Site-patched mocking + fault injection

**Source:** `tests/models/test_model.py:40-54, 487-532`
**Apply to:** every wave

Patch where the SUT imports the name (`dnallm.models.model._handle_*`, `dnallm.mcp.model_manager.load_model_and_tokenizer`, `time.sleep`). Fault injection = `Mock(side_effect=[...])` lists; negative control = `side_effect=AssertionError("... must not run")`; always assert `call_count` / `call_args` as the observable behavior.

### Error-assertion idiom

**Source:** repo-wide, e.g. `tests/models/test_model.py:60-65`, `tests/datahandling/test_dna_dataset.py:72-86`
**Apply to:** all new tests

```python
with pytest.raises(ValueError, match=r"Model test-model download failed."):
```

Regex `match=` against a stable substring of the real message.

### Artifact hygiene (tmp_path)

**Source:** `tests/inference/test_plot.py:154-157` (autouse rebind), `tests/datahandling/test_dna_dataset.py:92-107` (try/finally unlink)
**Apply to:** any test writing files (pdf/json/parquet/logs)

Prefer pytest `tmp_path`; `monkeypatch.chdir(tmp_path)` for CWD-sensitive code (start_server).

### Test anatomy conventions

**Source:** `tests/TESTING.md` + observed repo practice
**Apply to:** all new tests

- One behavior per test; docstring on every test function (one line, `"""Test ..."""` / `"""Verify ..."""`)
- `Test*` class grouping one unit under test; module docstring at file top
- Absolute imports in tests (`from dnallm.models.model import ...`), private functions imported directly
- English comments only; ruff format, 100 cols
- Markers registered in pyproject (`slow`, `pdf`, ...); `--strict-markers` is on — never invent unregistered markers
- Parametrize with tuple-of-names: `@pytest.mark.parametrize(("task_type", "expected"), [...])`
- Async tests: explicit `@pytest.mark.asyncio` + `AsyncMock` (repo style, even though auto mode is configured)

### Skip discipline (fail-closed)

**Source:** `dnallm/mcp/tests/_network_skip.py:43-62` + `tests/expected_skips.yaml`
**Apply to:** any new intentional skip

No broad `except Exception: skip`. Network-gated tests use `skip_if_unreachable(exc, action)` (message prefix `network-unavailable:`). Every new skip message MUST be allowlisted in `tests/expected_skips.yaml` (`exact`/`prefix`/`reason_like` + `category`) BEFORE the audit runs — `scripts/audit_skips.py` exits 1 on unmatched skips, and wave DoD requires exit 0. Target state: new fast tests need NO skips at all.

## No Analog Found

Files/portions where the planner should bind RESEARCH.md patterns (not codebase excerpts) into plan actions:

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `tests/cli/test_cli.py` | test | request-response | Zero CliRunner usage in the repo (verified by grep across tests/, dnallm/, ui/). Use `click.testing.CliRunner` (invoke, `result.exit_code`, `result.output`) per RESEARCH; lazy CLI imports need per-command patch-site verification. Style anchor: `tests/mcp/test_client_sdk.py` |
| `tests/mcp/test_server_transports.py` (in-memory pair portion) | test | request-response | In-memory ASGI client/server pair is new to the codebase — RESEARCH Pattern 1 is the verified source (localhost base_url, `lifespan_context`, `httpx_client_factory`). Construction-level halves DO have the `test_server_integration.py` analog |
| `tests/mcp/test_start_server.py` | test | file-I/O | No argparse-main or loguru-handler tests exist. Nearest structural analog `tests/utils/test_cuda_compat.py` (module-contract + monkeypatch flags); `monkeypatch.chdir(tmp_path)` from RESEARCH Pitfall 5 |
| `tests/models/test_head.py`, `tests/models/test_losses.py`, real-torch forwards in `test_special/` | test | transform | No existing test runs real torch module forwards with autograd asserts. Numerical-assert style from `tests/tasks/test_metrics.py`; shapes/differentiability assertions are new (RESEARCH: tiny modules, `(batch, n_classes)` outputs) |

## Metadata

**Analog search scope:** `tests/**` (all subdirs), `dnallm/mcp/tests/**`, `tests/TESTING.md`, `tests/expected_skips.yaml`; grep across `dnallm/` and `ui/` for CliRunner (none found)
**Files scanned:** 34 test files enumerated; 13 read in full or targeted ranges; git-tracked status verified for every named analog
**Pattern extraction date:** 2026-09-30
**Key verifications:** `.to()` self-return in both conftest (line 55) and sentinel test (line 493); `patch("time.sleep")` at test_model.py:50; pdf autouse fixture at test_plot.py:154-157; mock-server fixture at test_timeout.py:20-49; `load_model_and_tokenizer` patch site at test_server_integration.py:129
