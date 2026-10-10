---
last_mapped_commit: 9ec6bf532fca3d999dfb42329ac6c5cfacff21fb
last_mapped_at: 2026-10-05
---
# Technology Stack

**Analysis Date:** 2026-10-05

## Languages

**Primary:**
- Python 3.10+ — entire package (`requires-python = ">=3.10"` in `pyproject.toml`; classifiers declare 3.10/3.11/3.12/3.13; ruff `target-version = "py310"`; mypy `python_version = "3.10"`). CI matrix exercises 3.11/3.12/3.13 (`.github/workflows/ci.yml`)

**Secondary:**
- Bash — helper scripts in `scripts/` (`install_mamba.sh`, `setup_uv.sh`, `ci_checks.sh`, `publish.sh`, `check_code.sh`), root `install_deps.sh`, inline CI step scripts
- YAML — all configuration: `configs/*.yaml`, `dnallm/mcp/configs/*.yaml`, model registry `dnallm/models/model_info.yaml`, CI workflows, `models.lock` comments
- systemd unit syntax — `scripts/runner/ollama.service` (runner-host ollama definition, D-12)
- CSS/HTML — none; docs site is Markdown + MkDocs

## Runtime

**Environment:**
- Package version: `dnallm` **0.6.0** (`dnallm/version.py`); local dev venv Python 3.13 with torch 2.11.0+cu130 installed
- Self-hosted CI runner `dnallm-nightly` is an aarch64 (GB10) GPU box (`.github/workflows/ci.yml` coverage-nightly comment)

**Package Manager:**
- uv (primary; `uv venv` + `uv pip install -e ".[extra]"`), pip supported
- Lockfile: **missing** — no `uv.lock` committed; dependency ranges pinned in `pyproject.toml`; CI caches `~/.cache/uv` keyed on `hashFiles('**/pyproject.toml')`
- PyTorch index pinning via `[tool.uv.index]` / `[tool.uv.sources]` in `pyproject.toml`: `pypi`, `torch-cpu`, `torch-cuda121/124/126/128/130` (rocm index present but its source/conflict lines commented out); each `cudaNNN`/`cpu` extra maps torch to its index; `[tool.uv] conflicts` declares all cpu/cuda extras mutually exclusive so they cannot co-resolve

## Frameworks

**Core:**
- PyTorch (`torch>=2.4.0,<2.12`, mirrored across every cpu/cuda/rocm/mamba extra; installed 2.11.0+cu130) — all model training/inference
- Hugging Face `transformers` (`>=4.49.0,<6`; installed 5.17.0) — model loading (`AutoModel*`, `trust_remote_code=True`), `Trainer`/`TrainingArguments`. **Known risk (`.github/dependabot.yml`):** ModelScope mamba remote code imports `MambaCache`, removed in transformers 5.17 line — a transformers bump must re-verify the mamba load path before merge
- Hugging Face `datasets` (`<=3.2.0` pinned, tokenization pipeline API; installed 3.2.0) — dataset loading/tokenization in `dnallm/datahandling/data.py`
- Pydantic v2 (`>=2.10.6`; installed 2.13.5) — config validation in `dnallm/configuration/configs.py` (`DNALLMConfig` TypedDict + `BaseModel` sections)
- MCP SDK (`mcp>=1.3.0,<2`; installed 1.30.0) — FastMCP server `dnallm/mcp/server.py`, client `dnallm/mcp/client.py`. The `<2` ceiling is deliberate (commit 3490b71; 2.x breaks the server API)
- Starlette (`>=0.40.0,<1.8.0`) + Uvicorn (`>=0.24.0`) — HTTP/SSE/streamable-HTTP hosting for the MCP server

**Testing:**
- pytest (`>=8.4`; installed 9.1.1) with `pytest-asyncio` (auto mode), `pytest-cov` (installed 7.1.0), `pytest-timeout` (`>=2.3.1,<2.5`, default `--timeout=300`), `pytest-progress`, `coverage[toml]`
- `nbclient>=0.10` (installed 0.11.0) — notebook execution harness in `tests/examples/_execution.py` (`NotebookClient` with per-cell timeout, partial-failure artifacts)
- Config in `[tool.pytest.ini_options]` of `pyproject.toml`: testpaths `tests/` + `dnallm/mcp/tests/`, `--strict-markers`, `--strict-config`, markers `slow/pdf/performance/integration/unit/inference/utils/data/legacy`

**Build/Dev:**
- ruff (`==0.16.9`) — lint AND format (line-length 100, preview mode); configured in `pyproject.toml` `[tool.ruff.*]`; legacy `.flake8` (79-char) still present for the MCP module
- mypy (`>=1.15.0`; installed 2.3.1) — advisory only; CI runs `mypy dnallm/ ... || true`
- **ty (Astral)** — `[tool.ty.src]` in `pyproject.toml` excludes vendored code (`dnallm/tasks/metrics/**`, `mamba_npu.py`, `megatron.py`); invoked as `uvx ty check dnallm/` (config comment); deliberately no severity overrides. Owner decision (commit 9ec6bf5) is ty-hard-gate / mypy-retire — ty is the going-forward type checker
- pre-commit — local hooks: `ruff format` → `ruff check` → `mypy dnallm/` (`.pre-commit-config.yaml`)
- setuptools build backend (`[tool.setuptools.packages.find] include = ["dnallm*"]`); `setup.py` is a backward-compat stub only
- MkDocs Material + mkdocs-jupyter + mkdocstrings-python — docs site (`mkdocs.yml`), deployed to GitHub Pages

## Key Dependencies

**Critical:**
- `peft>=0.14.0` — LoRA/QLoRA fine-tuning (`dnallm/finetune/trainer.py`)
- `accelerate>=1.4.0` — device placement / distributed training
- `bitsandbytes>=0.43.0` — 4-bit quantization; patched via `dnallm/utils/transformers_compat.py` and `dnallm/utils/cuda_compat.py`
- `huggingface-hub>=0.29.0` — `snapshot_download` model fetching with `allow_patterns` passthrough (`dnallm/models/model.py:317` `download_model`)
- `modelscope[framework]>=1.23.2` (installed 1.34.0) — alternative model hub; `modelscope.hub.snapshot_download` + ModelScope `Auto*` classes (`dnallm/models/model.py:455`); `oss2>=2.18.0` declared for its storage flows
- `optuna>=3.6.0` — hyperparameter search (`Trainer.hyperparameter_search`, `dnallm/finetune/trainer.py`)
- `captum>=0.7.0` — interpretability attributions (`dnallm/inference/interpret.py`)
- `wandb>=0.19.8` + `tensorboardx>=2.6.2.2` — experiment tracking, selected via `report_to` config field (`dnallm/configuration/configs.py:266,298`; valid: tensorboard/wandb/none/all)
- `loguru>=0.7.0` — logging under `dnallm/utils/logger.py`
- `click` — CLI framework (`dnallm/cli/cli.py`); `click<8.5.1` pinned only in `docs` extra
- `altair[all]>=5.5.0` — plotting backend (`dnallm/inference/plot.py`, `dnallm/utils/training_plots.py`)
- `scikit-learn`, `scipy`, `seqeval`, `evaluate` — metrics (`dnallm/tasks/metrics.py` + vendored `dnallm/tasks/metrics/`)
- `mambapy>=1.2.0` — pure-PyTorch Mamba fallback (no compilation)

**Exact-pinned native kernel stack (do not float):**
- `mamba-ssm==2.3.2.post1` + `causal_conv1d==1.7.0` (the `mamba` extra) — CUDA kernels compiled against the local torch; Dependabot ignores them entirely (`.github/dependabot.yml`)
- `flash-attn==2.8.3.post1` — built once (~50 min sm_120 wheel build) into a cached wheelhouse for the evo isolated venv (example-nightly stage 0)
- torch ceiling `<2.12` is an **ABI claim, not a version preference** (`.github/dependabot.yml`): the pinned kernels compile against installed torch; widening the ceiling re-resolves torch and breaks them. Bump torch only as a coordinated kernel-rebuild + nightly-validation cycle

**Infrastructure:**
- `flash-linear-attention>=0.5.2,<0.6` (`fla` extra) — KDA kernels for PlantHelixSeek remote code (HelixSeekDelta layers); bare spec rides the already-installed torch/triton — never use fla's own `[cuda]/[rocm]` backend extras
- `jax>=0.5.2` — declared; tensor-format option only, no direct jax compute in `dnallm/`
- `python-dotenv>=1.0.0` — declared but `load_dotenv` never called inside `dnallm/`
- `numba>0.56.2`, `einops>=0.7.0`, `umap-learn>=0.5.7`, `openpyxl`, `sentencepiece>=0.2.0`, `tokenizers>=0.21.0` — data/numeric plumbing

## Extras Structure (`pyproject.toml` `[project.optional-dependencies]`)

| Extra | Contents |
|-------|----------|
| `dev` | `dnallm[test,notebook]` + ruff/pre-commit/mypy/pandas-stubs/logomaker/pybedtools(non-Windows)/pyfastx/seaborn |
| `test` | pytest stack + `coverage[toml]` + `nbclient>=0.10` |
| `notebook` | jupyter, marimo, nbclient, `ipython>=8.31,<9` (REPAIR-01: IPython 9 breaks pygenometracks' matplotlib<3.9), `pygenometracks>=3.9` (GPL-3.0, owner-approved 2026-10-04 for showcase figures only) |
| `docs` | mkdocs-material/jupyter/mkdocstrings-python, `click<8.5.1` |
| `ui` | `gradio>=4.0.0` (web apps in `ui/`) |
| `mcp` | mcp, langchain, langchain_mcp_adapters, `langchain-ollama>=1.1.0` (REPAIR-04 floor), nest-asyncio, `pydantic-ai<3` |
| `base` | `dnallm[dev,test,notebook,mcp]` + isort + types-transformers — the standard CI install target |
| `all` | `dnallm[base,dev,test,notebook,docs,ui,mcp,fla]` |
| `cpu`, `cuda121`, `cuda124`, `cuda126`, `cuda128`, `cuda130`, `rocm` | torch index selectors with per-extra minimum torch versions (`cuda130` needs `torch>=2.9.0`) |
| `mamba` | `causal_conv1d==1.7.0`, `mamba-ssm==2.3.2.post1`, `torch>=2.6.0,<2.12` — source-compile against local CUDA (`scripts/install_mamba.sh`; CI installs with `--no-cache-dir --no-build-isolation`) |
| `fla` | `flash-linear-attention>=0.5.2,<0.6` — KDA kernels |

## Configuration

**Environment:**
- No `.env` files committed or required; no env-var auth in package code
- Env vars set programmatically: `HF_ENDPOINT=https://hf-mirror.com` (use_mirror toggle, `dnallm/models/model.py:395-405`), `TOKENIZERS_PARALLELISM=true` (`dnallm/inference/inference.py:57`, `mutagenesis.py:28`, `benchmark.py:31`), `GRADIO_TEMP_DIR` (`ui/run_config_app.py:13`), `OMP_NUM_THREADS=1` in metrics sandbox (`dnallm/tasks/metrics/code_eval/execute.py:195`)
- CI env: `HF_ENDPOINT=hf-mirror.com` job-wide on example-nightly (origin unreachable on runner box), `HF_HUB_CACHE=~/models-giants/hub` + `HF_HUB_OFFLINE=1` for the evo giants lane (`tests/examples/_execution.py:143-155`), `PYTHONUTF8=1` on Windows leg, `UV_HTTP_TIMEOUT`/`UV_CONCURRENT_DOWNLOADS`, `LDFLAGS="-L$sys.prefix/lib"` (pyBigWig aarch64 sdist link fix)

**Build:**
- `pyproject.toml` — single source of truth for deps, extras, uv indexes, ruff, ty, mypy, pytest, coverage
- `[tool.coverage.report] fail_under = 90` (GATE-01 ratchet; suite landed 96.30%) — applies to every `--cov` invocation; use `--no-cov` for scoped runs. Omit list in `[tool.coverage.run]`: vendored metrics, ported Enformer, megatron/mamba_npu adapters, `dnallm/mcp/tests/*`
- `models.lock` (repo root) — pinned remote artifacts (`<hf|ms> <repo-id>@<revision-sha>`) fetched by the slow suite; keys the CI `actions/cache` model cache on both nightly jobs
- `tests/expected_skips.yaml` + `scripts/audit_skips.py` — every junit skip must match exactly one allowlisted entry or CI fails
- `.pre-commit-config.yaml` — ruff format/check + mypy local hooks
- `mkdocs.yml` — docs site (watches `example/notebooks`, `example/marimo`, `example/mcp_example`)

## Platform Requirements

**Development:**
- Python 3.11+ with venv/conda; uv for installs; `uv pip install -e '.[base]'` is the standard dev/CI install
- GPU optional. Devices: CPU, NVIDIA CUDA 12.1–13.0 wheels, AMD ROCm 6.2, Apple MPS, Intel Arc XPU, Huawei Ascend NPU (README "Supported Platforms")
- CUDA 13: `dnallm/utils/cuda_compat.py` preloads `libnvJitLink.so.13` (RTLD_GLOBAL) so bitsandbytes 4-bit works with cu130 wheels
- Mamba native kernels need `.[mamba]` (compile against local CUDA toolchain)

**Production/CI:**
- Published to PyPI as `dnallm` via `.github/workflows/publish.yml` (GitHub release trigger; uv + build + twine; `secrets.PYPI_API_TOKEN`; Python 3.12)
- MCP server ships as `dnallm-mcp-server` console script (stdio/SSE/streamable-HTTP)
- CI (`.github/workflows/ci.yml`): `test` (ubuntu, py3.11–3.13 × numpy 1.26.4/2.2.0), `test-windows` (py3.12 fast leg), `test-cuda` (12.1/12.4), `coverage-gate` (py3.12 fast, `--cov` hard gate), `coverage-nightly` + `test-mamba` + `example-nightly` (self-hosted `[self-hosted, dnallm-nightly]` GPU box, schedule/dispatch-only so fork PR code never runs there), `deploy` (mkdocs gh-pages on push to main)
- example-nightly (05:30 UTC, 2.5h after coverage-nightly's 03:00) executes the example census via the nbclient harness in staged-serial order (D-07): torch-heavy notebooks → VRAM settle → MCP live-server probes on :8000 → ollama pair last (D-13); fail-soft per-stage exit codes with a stage-4 summary that fails the job on any non-zero (D-08)
- Docs hosted at https://zhangtaolab.github.io/DNALLM/ (GitHub Pages)
- `feasibility.yml` — dispatch-only spike job on the self-hosted box (security boundary: workflow_dispatch reachable only by write-access users)

---

*Stack analysis: 2026-10-05*
