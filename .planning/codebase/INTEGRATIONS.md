---
last_mapped_commit: 9ec6bf532fca3d999dfb42329ac6c5cfacff21fb
last_mapped_at: 2026-10-05
---
# External Integrations

**Analysis Date:** 2026-10-05

## APIs & External Services

**Model hubs (the two primary sources for all 150+ pretrained models):**
- Hugging Face Hub — `snapshot_download(model_name, revision=..., allow_patterns=...)` for models; `AutoModel*`/`AutoTokenizer` from_pretrained loading
  - SDK/Client: `huggingface_hub>=0.29.0` (installed 1.33.0)
  - Auth: none (public repos); no HF token env var used in package code
  - Implementation: `download_model()` retry loop with reason classification (`dnallm/models/model.py:317`); `allow_patterns` passthrough keeps giant fetches safetensors-only (CI-05)
  - Mirror: `use_mirror` config flag sets `HF_ENDPOINT=https://hf-mirror.com` (`_setup_huggingface_mirror`, `dnallm/models/model.py:395-405`). Required on the dnallm-nightly runner — huggingface.co is unreachable there while hf-mirror.com serves the same artifacts (job-wide `HF_ENDPOINT` in example-nightly)
  - Offline mode: evo giants lane pins `HF_HUB_CACHE=~/models-giants/hub` + `HF_HUB_OFFLINE=1` (`tests/examples/_execution.py:143-155`)
- ModelScope — alternative hub for the same registry (`source="modelscope"` route)
  - SDK/Client: `modelscope[framework]>=1.23.2` (installed 1.34.0) + `oss2>=2.18.0` (its storage backend)
  - Auth: none in package code
  - Implementation: `from modelscope.hub.snapshot_download import snapshot_download` and ModelScope `Auto*` classes (`dnallm/models/model.py:455-476`, lazy import with install hint on ImportError); `DNADataset.from_modelscope` (`dnallm/datahandling/`) for preset datasets (`PRESET_DATASETS` in `dnallm/datahandling/dataset_auto.py`)

**Pinned remote artifacts (the slow/nightly suite's contract):**
- `models.lock` (repo root) — D-15 pinned form `<hf|ms> <repo-id>@<revision-sha>`; keys the nightly `actions/cache` hub caches; prefix must match the consuming notebook's `source=` route
- Giants tier: `togethercomputer/evo-1-8k-base@a9be7b66` (12.913GB safetensors-only, `~/models-giants` outside the cache quota, D-14/CI-05) and `arcinstitute/evo2_1b_base@2279e1df` (2.7GB, in-cache)
- megaDNA prerequisites: pinned git clone `https://github.com/lingxusb/megaDNA.git` @ `cb2f5ab4cc88dc0effe05c5f23358862c837014a` + `MEGABYTE_pytorch==0.2.1` (`tests/examples/_execution.py` constants; CI stage 0 mirrors the recipe) — never floating

**LLM inference (examples/tests only, not package code):**
- ollama — local loopback service `http://127.0.0.1:11434` backing the two MCP client example notebooks (`example/mcp_example/`); model `qwen3.8:latest` (17.74GB)
  - In-repo systemd unit: `scripts/runner/ollama.service` (`OLLAMA_HOST=127.0.0.1:11434` — loopback bind IS the access control, D-12; never bind 0.0.0.0)
  - Ops doc: `scripts/runner/README.md`; readiness probe `_gate_ollama_stack` retries `/api/tags` ~60s, degrades to typed `network-unavailable:` skips when the service is down (D-13)
  - Client stack: `langchain-ollama>=1.1.0` (`ChatOllama`) in the `mcp` extra, run inside the isolated `dnallm-mcp-langchain` kernelspec venv

**CI infrastructure binaries (rootless, example-nightly stage 0):**
- micromamba/bioconda — bedtools install into job-local `.bedtools-env` prefix (no sudo; `curl https://micro.mamba.pm/api/micromamba/linux-$(uname -m)/latest`)
- PyTorch wheel indexes — `download.pytorch.org/whl/{cpu,cu121,cu124,cu126,cu128,cu130}` via `[tool.uv.index]`

## Data Storage

**Databases:**
- None. No SQL/NoSQL anywhere.

**File Storage:**
- Local filesystem only — datasets load from local files (csv/tsv/json/parquet/fasta/txt/pkl) in `dnallm/datahandling/data.py`; outputs land in configured `output` dirs
- Hub caches: `~/.cache/huggingface/hub`, `~/.cache/modelscope/hub` (CI-cached keyed on `hashFiles('models.lock')`, D-09 shared read-only key between both nightly jobs); giants at `~/models-giants` deliberately outside every actions/cache path
- Wheelhouses: `wheelhouse-flashattn` (keyed on arch + evo-venv torch version) and `wheelhouse-mamba` (keyed on arch + project torch version), CI-cached

**Caching:**
- None at application level (no redis/memcached). HF/ModelScope hub caches above; uv cache `~/.cache/uv`.

## Authentication & Identity

**Auth Provider:**
- None for end users ("Custom" — no auth in package code)
- MCP server binds `0.0.0.0:8000` with no authentication — treat as trusted-network deployment (config: `dnallm/mcp/configs/mcp_server_config.yaml`, `cors_origins: ["*"]`)
- CI-only secret: PyPI token (`secrets.PYPI_API_TOKEN`, `TWINE_USERNAME=__token__`) in `.github/workflows/publish.yml`
- Security boundary for self-hosted runner: nightly/feasibility jobs are event-gated to `schedule`/`workflow_dispatch` so fork-PR code never executes there; `workflow_dispatch` reachable only by write-access users; job-default `permissions: contents: read` (deploy overrides to `contents: write` for gh-pages)

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry et al.)

**Logs:**
- loguru-based `get_logger`/`setup_logging` (`dnallm/utils/logger.py`); MCP server logs to `./logs/mcp_server.log` with rotation (`max_size: 10MB`, `backup_count: 5` per server config)
- CI artifacts: junit XMLs + `mcp-server-*.log` + mamba build logs (`actions/upload-artifact@v4`)
- Experiment tracking: `wandb` / `tensorboardx` selected via `report_to` (`dnallm/configuration/configs.py:266`)

## CI/CD & Deployment

**Hosting:**
- PyPI package `dnallm` (release-triggered publish, twine)
- GitHub Pages docs (mkdocs gh-deploy on push to main)
- MCP server deployed by end users (console script `dnallm-mcp-server`; transports stdio/SSE/streamable-HTTP via uvicorn; config `dnallm/mcp/configs/mcp_server_config.yaml` — yaml host/port win, CLI `--host/--port` are dead)

**CI Pipeline:**
- GitHub Actions (`.github/workflows/`): `ci.yml` (test/test-windows/test-cuda/coverage-gate/coverage-gate/coverage-nightly/test-mamba/example-nightly/deploy), `docs-validation.yml`, `feasibility.yml` (dispatch-only), `publish.yml`
- Self-hosted runner `[self-hosted, dnallm-nightly]` (aarch64 GPU box) serves the three nightly jobs; runner ops note: restart the actions service with sanitized env (`env -i`) or uv installs into the wrong venv
- Dependabot weekly (pip + github-actions, target-branch `dev`); ignores: `mcp` semver-major, `mamba-ssm`/`causal-conv1d` (exact-pinned native kernels), `torch` minor+major (ABI ceiling — see STACK.md)

## Environment Configuration

**Required env vars:**
- None required to import/use the package. All are optional toggles:
  - `HF_ENDPOINT` — set/deleted by `use_mirror` (`dnallm/models/model.py:395-405`)
  - `HF_HUB_CACHE`, `HF_HUB_OFFLINE` — giants lane only (test harness)
  - `TOKENIZERS_PARALLELISM` — set to `true` at module import in `dnallm/inference/{inference,mutagenesis,benchmark}.py`
  - `GRADIO_TEMP_DIR` — `ui/run_config_app.py`, `ui/generation_task_app.py`
  - `OLLAMA_HOST` — pinned in `scripts/runner/ollama.service` (runner host, not package)
  - CI-only: `PYTHONUTF8`, `UV_HTTP_TIMEOUT`, `UV_CONCURRENT_DOWNLOADS`, `LDFLAGS`

**Secrets location:**
- GitHub Actions secrets (`PYPI_API_TOKEN`). No `.env` files, no secrets in package code or repo.

## Webhooks & Callbacks

**Incoming:**
- None. GitHub release event triggers `publish.yml` (not a webhook handler in the repo).

**Outgoing:**
- None initiated by package code. Outbound network is: HF Hub / hf-mirror.com, ModelScope (+oss2 OSS backend), wandb/tensorboard (opt-in via `report_to`), ollama loopback (example notebooks), PyTorch wheel indexes (install time).

**Local servers bound by tests/CI:**
- MCP server on `localhost:8000` (streamable-http at `/mcp`, SSE at `/sse`) — started/stopped by example-nightly stages 2/3, one transport at a time (D-07); the 6 live-server probes in `dnallm/mcp/tests/test_sse_client.py` / `test_streamable_http_client.py` target it and skip as typed `network-unavailable:` when no server is up
- ollama on `127.0.0.1:11434` (runner-host systemd service)

---

*Integration audit: 2026-10-05*
