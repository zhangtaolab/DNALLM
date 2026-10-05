---
last_mapped_commit: 9ec6bf532fca3d999dfb42329ac6c5cfacff21fb
last_mapped_at: 2026-10-05
---
# Testing Patterns

**Analysis Date:** 2026-10-05

## Test Framework

**Runner:**
- pytest `>=8.4` with pytest-asyncio (auto mode), pytest-cov `>=7.0`, pytest-timeout `>=2.3.1,<2.5` (300s default), pytest-progress, coverage[toml], nbclient `>=0.10` (example-notebook execution)
- Config: `[tool.pytest.ini_options]` in `pyproject.toml` — `minversion = "8.4"`, `testpaths = ["tests", "dnallm/mcp/tests"]`, `python_files = "test_*.py"`, `python_classes = "Test*"`, `python_functions = "test_*"`
- addopts: `-v --tb=short --strict-markers --strict-config --asyncio-mode=auto --timeout=300`
- `filterwarnings` ignores Deprecation/PendingDeprecation/UserWarning plus specific legacy noise

**Assertion Library:**
- Plain `assert` plus `pytest.raises(..., match=...)`, `pytest.fail(...)`, `pytest.skip(...)`; unittest.mock for doubles

**Registered markers** (strict — unregistered marks error):
`slow`, `pdf`, `performance`, `integration`, `unit`, `inference`, `utils`, `data`, `legacy`

**Run Commands:**

```bash
pytest                                    # full suite (includes slow real-model lanes)
pytest -m "not slow" --cov                # fast leg (the CI shape) with coverage gate
pytest -m "not slow" --no-cov             # scoped run without the fail_under=90 gate
pytest tests/utils/test_sequence.py -v    # one module
pytest --cov --junitxml=pytest-junit.xml  # CI shape with junit for skip audit
python scripts/check_code.py --with-tests # lint + format + tests wrapper
```

## Test File Organization

**Location:**
- Two suites: `tests/` (mirrors the package layout) and `dnallm/mcp/tests/` (packaged with the module; omitted from coverage via `[tool.coverage.run]` omit)
- Shared fixtures in `tests/conftest.py` (root); MCP suite has no conftest — helpers live in `dnallm/mcp/tests/_network_skip.py`
- Real test data under `tests/test_data/<task_type>/` (binary_classification, multiclass, multilabel, regression, token_classification, embedding, language_model)
- ~1622 test functions total; 133 async

**Naming:**
- `test_<module>.py`; real-model integration tests suffixed `_real_model.py`; helper (non-test) modules prefixed `_` (`tests/examples/_execution.py`)

**Structure:**

```
tests/
├── conftest.py                  # shared fixtures (tokenizers, tiny models, factories, mocks)
├── expected_skips.yaml          # junit skip allowlist (audit contract)
├── TESTING.md                   # in-tree testing guide
├── benchmark/ cli/ configuration/ datahandling/ examples/ finetune/
├── inference/ mcp/ models/ scripts/ tasks/ utils/
│   └── (test_<module>.py per source module; models/test_special/ nests deeper)
├── examples/_execution.py       # private nbclient harness (see Example Execution Layer)
└── test_data/                   # per-task-type fixture data
dnallm/mcp/tests/                # packaged second suite + _network_skip.py helper
```

## Test Structure

**Suite Organization (canonical pattern):**

```python
class TestDownloadModel:
    """Tests for the model download retry logic."""

    def test_download_success_after_retries(self):
        """Model resolves after transient failures exhaust the retry ladder."""
        mock_downloader = Mock(side_effect=[Exception("conn"), "/path/to/model"])
        with patch("time.sleep"):  # Mock sleep to speed up test
            ...
```

- Class-based grouping `class TestX:` per area under test; module-level functions also accepted (`tests/utils/test_sequence.py`)
- Every test class and function carries a docstring
- `unittest.TestCase` style survives only in legacy real-model scripts (`tests/finetune/test_trainer_real_model.py`, `tests/inference/test_inference_real_model.py`) — do not copy into new tests
- Parametrization with readable ids: `@pytest.mark.parametrize("py_file", MARIMO_FILES, ids=lambda p: str(p.relative_to(EXAMPLE_DIR)))` (`tests/examples/test_examples.py`)
- Discovery guards: `@pytest.mark.skipif(not MARIMO_FILES, reason="No marimo files found")` for content-derived parametrize lists

**Patterns:**
- Assertion-first bodies with per-case inline comments (`# Test with lowercase:`)
- Boundary-failure tests: `with pytest.raises(ValueError, match="Input tensor must be 2D"):` (`tests/models/test_head.py`)
- Delta-zero baselines: compare against state captured at session start, never absolute zero (`_PREEXISTING_DIRTY` in `tests/examples/_execution.py`, `_kernel_count` in `tests/examples/test_notebook_execution.py`)
- Probe-then-execute gates: environment probes return `(installed, evidence)` and only then does the real execution run, else a typed skip fires with the evidence embedded

## Mocking

**Framework:** `unittest.mock` (`Mock`, `MagicMock`, `AsyncMock`, `patch`, `ANY`) + `pytest.MonkeyPatch`/`sys.modules` manipulation

**Patterns:**

```python

# Async MCP session mocking (tests/mcp/test_client_sdk.py)

session = MagicMock()
session.call_tool = AsyncMock(return_value=MagicMock(isError=False,
                              content=[MagicMock(text='{"result": "ok"}')]))

# Patch at the import site of the unit under test (tests/mcp/test_model_manager.py)

with patch("dnallm.mcp.model_manager.load_model_and_tokenizer"):
    ...

# Speed up retry/backoff loops (tests/models/test_model.py)

with patch("time.sleep"):
    ...
```

**What to Mock:**
- Network boundaries (`huggingface_hub.snapshot_download`), heavy loaders (`load_model_and_tokenizer` at its import site), `time.sleep` in retry paths, `builtins.print` to silence legacy output
- Boundary objects only — prefer real deterministic doubles for domain behavior

**What NOT to Mock:**
- Tokenizer/model behavior: use the real fakes from `tests/conftest.py` — `SimpleDNATokenizer` (callable character-level tokenizer returning real `BatchEncoding` under `return_tensors="pt"`, `N` → mask id) and `TinyDNAModel` (real `torch.nn.Module`, weights from a local `torch.Generator` seed, `.logits`/`.hidden_states` outputs, `pooled=True/False` for sequence- vs token-level shapes)
- Config validation: `inference_config_factory` writes a real YAML under `tmp_path` and loads it through `load_config` so every test exercises real Pydantic validation
- Mock-based fixtures (`mock_model`, `mock_tokenizer`, `mock_config`, `mock_inference_engine`, `mock_dataset` in `tests/conftest.py`) exist for boundary-shaped objects; new tests should reach for the real fakes first

## Fixtures and Factories

**Test Data:**

```python
@pytest.fixture
def inference_config_factory(tmp_path):
    """Factory building real loaded inference configs under tmp_path."""
    def _make(task_type="binary", num_labels=None, ...):
        path = tmp_path / f"config-{task_type}-{uuid.uuid4().hex[:8]}.yaml"
        path.write_text(yaml.safe_dump({"task": task, "inference": inference}))
        return load_config(path)
    return _make
```

**Location:**
- `tests/conftest.py` (all shared fixtures); per-file fixtures at top of the consuming test module under a `# Fixtures` banner
- Static datasets in `tests/test_data/`; example-input sandboxes are copied from `example/` at runtime (never committed copies)
- MCP model-manager tests write real YAML config dirs via `_write_configs(temp_dir, ...)` helpers (`tests/mcp/test_model_manager.py`)

## Coverage

**Requirements:** hard gate — `[tool.coverage.report] fail_under = 90` (Phase 4 ratchet GATE-01; suite landed at 96.30%). Applies to EVERY `--cov` invocation; use `--no-cov` for scoped runs.

**Scope config:** `[tool.coverage.run] source_pkgs = ["dnallm"]` with omit list: `*/dnallm/tasks/metrics/*` (vendored HF evaluate), `*/dnallm/models/special/enformer_model/*` (ported), `*/dnallm/finetune/megatron.py` and `*/dnallm/models/special/mamba_npu.py` (unimportable adapters), `*/dnallm/mcp/tests/*`, `*/dnallm/mcp/run_tests.py`, `*/dnallm/mcp/example_sse_usage.py`

**View Coverage:**

```bash
pytest -m "not slow" --cov                    # enforced gate (fails under 90)
pytest -m "not slow" --cov --cov-report=term-missing
```

## Test Types

**Unit Tests:**
- Default lane; `pytest -m "not slow"`; deterministic CPU-only; no kernel spawns, no network

**Integration Tests:**
- Real-model lanes behind `@pytest.mark.slow` (model downloads, real training, live MCP servers); run on nightly/self-hosted legs only
- Timeout ladder via `@pytest.mark.timeout(...)`: 600–900s inference/downloads, 1800s evo/generation-LoRA, 3600s finetune classes, 7200s whole-suite class marks — inner budgets (per-cell, per-subprocess) must stay strictly below the outer pytest mark

**E2E / Example Execution Tests:**
- `tests/examples/` executes the repo's own notebooks, marimo apps and helper scripts for real (see Common Patterns)

## Test Lanes & CI

- **Fast leg** (push/PR, hosted): `pytest -m "not slow" --cov --junitxml=pytest-junit.xml` + skip audit + exit-code canary + advisory mypy; matrix py3.11/3.12/3.13 × numpy 1.26.4/2.2.0; separate windows (py3.12) and CUDA 12.1/12.4 legs
- **coverage-gate** (push/PR): `pytest -m "not slow" -ra --durations=0 --junitxml=... --cov -p no:cacheprovider -p no:progress` + skip audit
- **Nightlies** (schedule/dispatch only, self-hosted `dnallm-nightly` runner — never PR code): coverage-nightly (full suite incl. slow, model caches keyed on `models.lock`), example-nightly (staged-serial example census, `timeout-minutes: 2700`, `HF_ENDPOINT=hf-mirror.com`), mamba kernel-build leg
- **Exit-code canary:** every fast leg writes a deliberately failing test and asserts pytest exits non-zero (guards against exit-code masking regressions)
- Real-model census status at closeout: 196 passed / 1 skipped / 0 failed

## Common Patterns

**Async Testing (pytest-asyncio auto mode — no markers needed):**

```python
async def test_client_roundtrip(mock_connect, mock_session):
    async with DNALLMMCPClient(...) as client:
        result = await client.call_tool("predict", {"sequence": "ATGC"})
        assert result == {"result": "ok"}
```

**Error Testing:**

```python
with pytest.raises(ValueError, match=r"should be a directory"):
    load_model_and_tokenizer("evo-2-7b", ...)
with pytest.raises(ImportError, match="gpn package is required"):
    _handle_gpn_models(...)
```

**Typed skips (the ONLY sanctioned skip paths):**
- Helpers in `tests/examples/_execution.py`: `environment_unavailable_skip(action, evidence)`, `optional_dep_skip(action, evidence)`, `network_unavailable_skip(action, evidence)` — each emits a stable junit-greppable prefix
- MCP live-server variant: `skip_if_unreachable(exc, action)` in `dnallm/mcp/tests/_network_skip.py` — flattens `ExceptionGroup` trees and skips only when every leaf is `httpx.TransportError`; anything else re-raises (honest failure)
- Every skip message must match an entry in `tests/expected_skips.yaml` (matchers: `exact`, `prefix`, `reason_like`; exactly one matcher + `category` per entry; empty/wildcard entries rejected). `scripts/audit_skips.py` fails CI (exit 1, fail-closed) on any unmatched skip. Never add a skip without an allowlist entry.
- `pytest.skip(..., allow_module_level=True)` guards module imports for optional deps (`"MCP client modules not available:"` prefix)

**Example Execution Layer (`tests/examples/`):**
- `tests/examples/_execution.py` is the private harness: `NOTEBOOK_EXEC_SPECS` (per-notebook per-cell timeout, extra sandbox inputs, per-notebook env, isolated kernelspec routing), `MARIMO_EXEC_SPECS`, `seed_sandbox` (whole-dir `copytree` into pytest `tmp_path` so kernels never touch the repo tree; `../` escapes beyond tmp rejected), `run_notebook` (nbclient, `allow_errors=False`, `shutdown_kernel="immediate"`, partial-failure artifacts `.executed.ipynb` + `.error.txt`), `run_marimo_app` (`marimo export html`, artifact >1000 bytes asserted), `run_example_script` (run log always written), `assert_tree_clean` (scoped `git status --porcelain` tripwire vs import-time baseline over `example/` + `docs/example/`)
- `ACTIVE_NOTEBOOKS` in `tests/examples/test_notebook_execution.py` gates which notebooks execute for real; census FAIL items stay out until repaired and evidenced — the rollout never widens silently
- Isolated kernelspec lanes for notebooks whose install cells must never touch the project venv: `dnallm-mcp-langchain`, `dnallm-megadna`, `dnallm-evo-kernel` — each with `kernel.json` env `VIRTUAL_ENV` pinned to a throwaway `.scratch/` venv; provisioning helpers are idempotent and raise `RuntimeError` (never skip) once a gate is green
- Global deterministic env for spawned kernels: `MPLBACKEND=Agg`, `TOKENIZERS_PARALLELISM=true`, `WANDB_MODE=disabled` via save/restore sandwich

**Testing the infrastructure itself:**
- `tests/scripts/test_audit_skips.py` unit-tests the skip-audit script (malformed allowlist, matcher semantics)
- `tests/examples/test_notebook_execution.py` includes kernel-lifecycle kill tests, partial-failure artifact tests, sandbox-escape rejection tests, probe-honesty and gate-matrix tests

**Owner rules baked into the suite:**
- Any `dnallm/` code change ships with pytest coverage in the same change
- Notebook + `docs/` Markdown mirror + wrapper changes commit atomically; drift caught by `scripts/check_notebook_md_sync.py` (AST-based, in `scripts/ci_checks.sh`)

---

*Testing analysis: 2026-10-05*
