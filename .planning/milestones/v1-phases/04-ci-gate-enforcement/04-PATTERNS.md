# Phase 4: CI Gate Enforcement - Pattern Map

**Mapped:** 2026-09-30
**Files analyzed:** 8 (7 modified, 1 new)
**Analogs found:** 8 / 8 (2 partial — no in-repo precedent for `schedule`/cron triggers or a remote-artifact lockfile manifest)

> **Shape note (owner amendment 2026-09-30, supersedes RESEARCH single-job sketch):** GATE-02 lands as **two jobs** in `ci.yml` —
> (a) `coverage-gate` on PRs/pushes: single leg (py3.12), **fast leg `-m "not slow"`** (measured 96% alone), `fail_under` enforced via pytest rc, skip audit;
> (b) **nightly scheduled job**: same `fail_under` + full census command (slow included), cache keyed on `models.lock`, `workflow_dispatch` for calibration, visible but not PR-blocking.
> Per-test `@pytest.mark.timeout` marks + job `timeout-minutes` backstop on **both** jobs. RESEARCH Code Examples remain the step library; only the job split is amended.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `pyproject.toml` | config | transform (coverage total → pytest rc) | itself — `[tool.coverage.report]` :512-514 | exact (in-place edit) |
| `.github/workflows/ci.yml` | CI workflow config | batch (CI pipeline) | `test` job in itself :19-119; `workflow_dispatch` from `.github/workflows/publish.yml:6` | exact (extend existing file) |
| `models.lock` (NEW) | config / manifest | file-I/O (consumed only by `hashFiles`) | `tests/expected_skips.yaml` (CI-consumed data file w/ comment discipline) | partial (format/consumer differ) |
| `tests/finetune/test_trainer_real_model.py` | test | batch (long training runs) | itself — `@pytest.mark.slow` at :20, :403, :697 | exact (in-place edit) |
| `tests/models/test_model.py` | test | batch (network download) | itself — `@pytest.mark.slow` at :178, :188 | exact (in-place edit) |
| `tests/inference/test_inference_real_model.py` | test | batch (real-model inference) | itself — class-level `@pytest.mark.slow` at :22 | exact (in-place edit) |
| `tests/inference/test_inference.py` | test | batch (integration) | itself — `@pytest.mark.slow` at :459 | exact (in-place edit) |
| `.github/workflows/README.md` | docs | n/a (documentation) | itself (WR-05 — update stale claims) | exact (in-place edit) |

**Tracked-source gate:** every analog above verified via `git ls-files -- <path>` (all printed, none gitignored). No `.gsd/capabilities/` mirrors involved.

## Pattern Assignments

### `pyproject.toml` (config, transform)

**Analog:** the file's own coverage block — `pyproject.toml:499-514`. This is a one-line in-place edit; the pattern to copy is the existing comment discipline (the "NO fail_under" comment was written to be replaced by this phase).

**Current state** (lines 499-514):
```toml
# Coverage configuration
[tool.coverage.run]
source_pkgs = ["dnallm"]
omit = [
    "*/dnallm/tasks/metrics/*",                    # vendored HF evaluate
    "*/dnallm/models/special/enformer_model/*",    # ported Enformer
    "*/dnallm/finetune/megatron.py",               # unimportable adapter
    "*/dnallm/models/special/mamba_npu.py",        # unimportable adapter
    "*/dnallm/mcp/tests/*",                        # packaged test files
    "*/dnallm/mcp/run_tests.py",                   # helper script
    "*/dnallm/mcp/example_sse_usage.py",           # example script
]

[tool.coverage.report]
show_missing = true
# NO fail_under in this phase — the enforcement threshold is added in Phase 4, ratcheted
```

**Target shape** (per RESEARCH GATE-01 example, verified exit codes):
```toml
[tool.coverage.report]
show_missing = true
fail_under = 90   # Phase 4 ratchet (GATE-01). Suite landed at 96.30% (Phase 3).
                  # Applies to every `--cov` invocation — use `--no-cov` for scoped runs.
```

**Marker-registration corollary** (lines 477-487): the `markers` list registers `slow` etc. Do NOT add `timeout` to this list — `pytest.mark.timeout` is registered by the pytest-timeout plugin (RESEARCH verified `--strict-markers`-safe; shown in `pytest --markers`). Adding it here is redundant, not required.

**Cross-cutting consequence to encode in plans:** all six existing matrix legs run `pytest -m "not slow" --cov` (`ci.yml:84`) — once `fail_under` lands they are all gated too. This is intentional ("identical command locally and in CI") and green (fast leg measured 96%); plans must not special-case the matrix legs.

---

### `.github/workflows/ci.yml` (CI workflow config, batch)

**Analog:** the `test` job in the same file (`.github/workflows/ci.yml:19-119`) — the new jobs clone its step skeleton; `publish.yml:3-6` supplies the `workflow_dispatch` trigger form.

**Trigger block** (lines 3-10) — new jobs inherit these; the nightly needs `schedule` + `workflow_dispatch` added to `on:`:
```yaml
on:
  push:
    branches:
      - main
      - master
      - dev
  pull_request:
    branches: [ main, master, dev ]
```

`workflow_dispatch` form (`.github/workflows/publish.yml:3-6`):
```yaml
on:
  release:
    types: [published]
  workflow_dispatch:
```

**Job skeleton to clone** (lines 19-70) — checkout → free disk → setup-python → uv → caches → venv:
```yaml
  test:
    name: test (py${{ matrix.python-version }}, numpy${{ matrix.numpy-version }})
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12', '3.13']
        numpy-version: ['1.26.4', '2.2.0']
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Free disk space
        run: |
          sudo rm -rf /usr/local/lib/android
          sudo rm -rf /usr/share/dotnet
          sudo rm -rf /opt/ghc
          sudo rm -rf /usr/local/share/powershell
          sudo rm -rf /usr/local/share/chromium
          sudo rm -rf /usr/local/.ghcup

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v7
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Cache pattern** (lines 47-53) — copy this shape for the new `models.lock`-keyed cache (new paths + new key, same action/version/restore-keys style):
```yaml
      - name: Cache uv dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/uv
          key: ${{ runner.os }}-uv-${{ hashFiles('**/pyproject.toml') }}
          restore-keys: |
            ${{ runner.os }}-uv-
```

**Install pattern** (lines 55-70) — single-leg variant pins numpy directly (the `elif` branch at :68-69 is the py3.12/numpy-2.2.0 shape):
```yaml
      - name: Create virtual environment and install dependencies
        env:
          UV_HTTP_TIMEOUT: 300
          UV_CONCURRENT_DOWNLOADS: 4
        run: |
          uv venv
          uv pip install -e ".[base]"

      - name: Install specific numpy version
        run: |
          source .venv/bin/activate
          if [ "${{ matrix.numpy-version }}" = "1.26.4" ]; then
            uv pip install "numpy==${{ matrix.numpy-version }}" "scipy>=1.15.2"
          elif [ "${{ matrix.numpy-version }}" = "2.2.0" ]; then
            uv pip install "numpy==${{ matrix.numpy-version }}"
          fi
```

**Test + skip-audit steps** (lines 81-90) — the gated jobs copy this pair (census command of record adds `-ra --durations=0 -p no:cacheprovider -p no:progress`; nightly omits `-m "not slow"`); every `run:` step begins with `source .venv/bin/activate`:
```yaml
      - name: Run fast tests
        run: |
          source .venv/bin/activate
          pytest -m "not slow" --cov --junitxml=pytest-junit.xml
          coverage xml -o coverage.xml

      - name: Skip audit (unexpected skips fail the job)
        run: |
          source .venv/bin/activate
          python scripts/audit_skips.py pytest-junit.xml tests/expected_skips.yaml
```

**GATE-03 target — the codecov step to remove or bump** (lines 92-96):
```yaml
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: false
```

**Canary — do NOT duplicate** (lines 98-114): the exit-code canary already lives on the fast `test` job and runs every push/PR; the new jobs must not copy it (RESEARCH anti-pattern). The nightly/gate jobs need the skip audit (they produce junit) but not a second canary.

**Minimal-touch constraints:** `deploy.needs: [test, test-cuda, test-mamba]` (:235) stays unchanged (Open Question 5: multi-hour gate must not delay docs deploys); `permissions: contents: read` (:15-16) is inherited by new jobs — do not widen (Security Domain V4), except nothing in the amended shape needs `id-token: write` since codecov removal is the recommendation.

---

### `models.lock` (NEW — config/manifest, file-I/O)

**Analog:** `tests/expected_skips.yaml` — the repo's precedent for a small, reviewable, CI-consumed data file whose header comment documents semantics and editing rules. Copy that comment discipline; do NOT copy the YAML format or fail-closed parsing — models.lock is plain text consumed **only** by `hashFiles('models.lock')` (never parsed/executed by tests or scripts; RESEARCH Security V5).

**Header-comment pattern to emulate** (`tests/expected_skips.yaml:1-13`):
```yaml
# Expected skips: every junit <skipped message> must match exactly one entry.
# An unmatched skip fails CI (scripts/audit_skips.py exit 1).
#
# ...
# Never add an empty or wildcard entry —
# this file is a reviewable suppression record, not a silencer.
```

**Content to write** (RESEARCH Pattern 2 — 9 entries, verified against test sources + `du`):
```
# models.lock — remote artifacts fetched by the slow test suite.
# Keys the gated CI job's model cache (actions/cache hashFiles).
# Edit an entry to rotate the cache key.
hf  microsoft/DialoGPT-small                         # tests/models/test_model.py::test_download_real_huggingface_connection
hf  zhangtaolab/plant-dnagpt-BPE-promoter            # tests/inference/test_inference_real_model.py (from_pretrained, HF route)
ms  zhangtaolab/plant-dnabert-BPE                    # tests/finetune/test_trainer_real_model.py (12 call sites, source=modelscope)
ms  zhangtaolab/plant-dnabert-BPE-promoter           # dnallm/mcp/tests/configs/promoter_inference_config.yaml
ms  zhangtaolab/plant-dnabert-BPE-conservation       # dnallm/mcp/tests/configs/conservation_inference_config.yaml
ms  zhangtaolab/plant-dnamamba-BPE-open_chromatin    # dnallm/mcp/tests/configs/open_chromatin_inference_config.yaml
ms  zhangtaolab/plant-dnagpt-BPE-promoter            # tests/inference/test_inference.py::test_real_model_integration (modelscope route)
ms  ZhejiangLab-LifeScience/DNA_bert_4               # tests/models/test_model.py::test_download_real_modelscope_connection
ms  dataset:zhangtaolab/plant-multi-species-core-promoters  # DNADataset.from_modelscope (trainer tests)
```

Excluded by research (referenced by no test): `plant-dnamamba-BPE-H3K27ac`/`H3K27me3` MCP configs. Cache step consumes it as (RESEARCH Pattern 2):
```yaml
- name: Restore model caches (keyed on models.lock)
  uses: actions/cache@v4
  with:
    path: |
      ~/.cache/huggingface/hub
      ~/.cache/modelscope/hub
    key: ${{ runner.os }}-models-${{ hashFiles('models.lock') }}
    restore-keys: |
      ${{ runner.os }}-models-
```

---

### `tests/finetune/test_trainer_real_model.py` (test, batch)

**Analog:** itself — three levels of existing `slow` marking; the new `@pytest.mark.timeout(N)` marks stack directly on/next to these decorators.

**Class-level mark** (lines 20-21) — all methods inside are slow; individual timeout marks go on the heavy methods only:
```python
@pytest.mark.slow
class TestTrainerRealModel(unittest.TestCase):
    """Test DNATrainer with real model."""
```

**Method-level mark** (lines 403-404) — the stack-order convention to copy (`slow` first, `timeout` joins beside it):
```python
    @pytest.mark.slow
    def test_early_stopping_stops_before_full_epochs(self):
        """Test that early stopping stops training before num_train_epochs."""
```

**Module-level function mark** (lines 697-698):
```python
@pytest.mark.slow
def test_with_config_file():
    """Test with the provided finetune config file."""
```

**Target shape** (RESEARCH timeout table — mark values initial, recalibrate from first CI `--durations=0`):
- `test_complete_training_workflow` (:54), `test_training` (:312), `test_with_config_file` (:698) → `@pytest.mark.timeout(7200)`
- `test_early_stopping_stops_before_full_epochs` (:404), `test_no_early_stopping_runs_full_epochs` (:482), `test_qlora_training` (:556) → `@pytest.mark.timeout(3600)`
- Decorator order: `@pytest.mark.slow` then `@pytest.mark.timeout(N)` (matches RESEARCH Pattern 3 example).

Note: `test_prediction` (:358) inherits the class-level slow mark but ran <300s-class locally — leave unmarked unless calibration says otherwise (RESEARCH: "unmarked for the <15s stratum").

---

### `tests/models/test_model.py` (test, batch/network)

**Analog:** itself — the two slow network tests (lines 178-200), both the models.lock provenance sources:
```python
    @pytest.mark.slow
    def test_download_real_huggingface_connection(self):
        """Test real HuggingFace connection (requires network)."""
        from huggingface_hub import snapshot_download

        # Try to download a small test model
        result = download_model("microsoft/DialoGPT-small", snapshot_download, max_try=1)
        assert result is not None
        assert os.path.exists(result)

    @pytest.mark.slow
    def test_download_real_modelscope_connection(self):
        """Test real ModelScope connection (requires network)."""
        from modelscope.hub.snapshot_download import snapshot_download
```

These are download-bound, not compute-bound (DialoGPT-small is tiny) — timeout marks likely unnecessary here; add only if nightly calibration shows otherwise. This file is also the **GATE-04 deletion target on the ephemeral probe branch only** (`git rm` → predicted ~81% < 90 → red check → branch deleted; no edit lands on dev).

---

### `tests/inference/test_inference_real_model.py` (test, batch)

**Analog:** itself — class-level slow mark (lines 22-23):
```python
@pytest.mark.slow
class TestRealModelInference(unittest.TestCase):
    """Test class for real model inference."""
```

Heavy work is in `setUpClass` (:34-80, model load) shared by all methods — a per-test timeout mark only bounds the individual test method, not `setUpClass`; keep marks conservative and let the job-level `timeout-minutes` backstop cover setup.

---

### `tests/inference/test_inference.py` (test, batch/integration)

**Analog:** itself — single method-level slow mark (lines 459-460), the ModelScope provenance source for models.lock:
```python
    @pytest.mark.slow
    def test_real_model_integration(self):
        """Test with real model loading from ModelScope."""
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            # Load real model and tokenizer from ModelScope
            model_name = "zhangtaolab/plant-dnagpt-BPE-promoter"
```

Add `@pytest.mark.timeout(N)` beside `@pytest.mark.slow` only if calibration shows it exceeding 300s on CI (research measured it in the unmarked <15s stratum locally, but CI is 4-core CPU — initial value `timeout(3600)` is the safe call per RESEARCH code examples).

---

### `.github/workflows/README.md` (docs, WR-05 disposition)

**Analog:** itself. Stale claims this phase's work flips (must not be left lying once the gate lands):
- Line 97: "Slow Tests: Tests that take longer to execute (excluded from CI)" — **false after this phase** (nightly runs them; the gated PR job deliberately excludes them)
- Line 204: "Slow tests are excluded from CI to maintain reasonable execution times" — same flip
- Lines 22-24: matrix "3.10, 3.11, 3.12" (actual: 3.11/3.12/3.13 × 2 numpy) — minimal-touch fix while in the file
- Lines 31-35, 107-109: Black/isort/Flake8 claims (actual: ruff) — owner decision on depth (WR-05); do not silently rewrite the whole file

**Format conventions to keep:** emoji-headed sections (`## 🚀`, `## 🔧`, `## 🧪`), numbered job subsections (`### 1. Test Job (`test`)`), a "Local Testing" bash block (:159-180) — extend it with the `--no-cov` scoped-run guidance (RESEARCH Pitfall 4) and the census command of record.

## Shared Patterns

### CI job step skeleton
**Source:** `.github/workflows/ci.yml:27-90` (checkout → free disk → setup-python → install uv → cache → `uv venv` + `uv pip install -e ".[base]"` → `source .venv/bin/activate` preamble on every run step)
**Apply to:** both new jobs (coverage-gate + nightly). The nightly additionally inserts the models.lock cache step and pins `numpy==2.2.0` (the :68-69 elif shape, inlined).

### Fail-closed CI audit tooling
**Source:** `scripts/audit_skips.py:13-46` (raises `ValueError` on absent/unparseable/malformed allowlist — "An empty matcher would silently allow every skip") + step wiring at `ci.yml:87-90`
**Apply to:** both new jobs, each against its own junit file. `tests/expected_skips.yaml:26-29` comment already anticipates the Phase-4 slow leg — no allowlist edit expected; a new skip reason = audit exit 1 = job red (Pitfall 7).

### Exit-code propagation (the mechanism the gate rides on — do not regress)
**Source:** `conftest.py:27-33`:
```python
def pytest_sessionfinish(session, exitstatus):
    """Whole-run cleanup; `exitstatus` propagates untouched because the exit is never forced."""
    cleanup_multiprocessing()
    cleanup_pytorch_resources()
    gc.collect()
    # Never force the process exit here: returning propagates `exitstatus` unchanged.
```
**Apply to:** any conftest touch in this phase (there should be none). `fail_under` only bites because this Phase-1 fix lets pytest's rc — including coverage-failure rc=1 — reach the Actions step. Guarded by the canary at `ci.yml:98-114`.

### Cache keying
**Source:** `ci.yml:47-53` (`actions/cache@v4`, `hashFiles` key, branch-tolerant `restore-keys`)
**Apply to:** the new models cache — same action and key shape, `${{ runner.os }}-models-${{ hashFiles('models.lock') }}`, paths `~/.cache/huggingface/hub` + `~/.cache/modelscope/hub` whole-dir (never fragments — symlinked blob trees).

### Timeout layering
**Source:** `pyproject.toml:475` (`--timeout=300` global) — marker `@pytest.mark.timeout(N)` overrides it (RESEARCH verified precedence); job-level `timeout-minutes` bounds wall clock.
**Apply to:** slow tests listed above; both new jobs get explicit `timeout-minutes` (no existing job sets one — verified). Do NOT raise the global 300s in addopts (weakens the hang-guard for 1600+ fast tests).

## No Analog Found

| Item | Needed By | Reason | Fallback |
|------|-----------|--------|----------|
| `schedule:` (cron) trigger | nightly slow job | No workflow in `.github/workflows/` uses `schedule` (verified by grep) | Standard GitHub Actions syntax: `schedule: - cron: "0 3 * * *"` under `on:`; RESEARCH Architecture Diagram shows the intended wiring |
| `timeout-minutes` job setting | both new jobs | No existing job sets it (verified by grep) | Standard syntax; initial 480 per RESEARCH Pitfall 3 |
| `@pytest.mark.timeout` usage | test files | Zero existing uses in `tests/` (verified by grep) — clean addition | RESEARCH Pattern 3 verbatim example (precedence behavior verified live) |
| Remote-artifact lockfile | `models.lock` | Repo has no committed lockfile of any kind (verified: no `*.lock` tracked) | RESEARCH Pattern 2 full manifest content, reproduced above |

## Metadata

**Analog search scope:** `.github/workflows/` (all 4 files), `pyproject.toml`, `conftest.py`, `scripts/audit_skips.py`, `tests/` (finetune/models/inference test files, `tests/expected_skips.yaml`); greps for `schedule`, `cron`, `timeout-minutes`, `mark.timeout`, `pytest.mark.slow` across `tests/` + `dnallm/` + workflows
**Files scanned:** 12 read, 6 grep sweeps
**Tracked-source gate:** all analogs pass `git ls-files` (tracked); no mirror paths emitted
**Pattern extraction date:** 2026-09-30
