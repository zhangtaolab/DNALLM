# Phase 23: Pre-flight Core — Pure Validation / Suitability / Config API - Research

**Researched:** 2026-10-20
**Domain:** pure-function data validation / model-suitability verdicts / hub availability probing (huggingface-hub 1.33.0 + modelscope 1.34.0) / config generation over Pydantic sections
**Confidence:** HIGH

## Summary

Phase 23 delivers `dnallm/preflight/` — a pure, synchronous, offline-safe core (`validate_data`, `check_compatibility`, ModelScope-first `probe_availability`, `generate_config`) — plus the one sanctioned refactor: `validate_sections` factored out of `load_config` (`dnallm/configuration/configs.py:698-747`) with zero behavior change. Zero new dependencies: everything the probe layer needs is exposed by the installed `huggingface_hub` 1.33.0 and `modelscope` 1.34.0, and the reader layer needs only pandas + pyarrow (both already present; pyarrow guaranteed transitively because `datasets` requires `pyarrow>=15.0.0`). The Phase 22 registry loader (`dnallm/models/registry.py`) already provides every metadata input the suitability verdict needs — `effective_max_length` resolves for 100% of the 236 entries, `effective_tokenizer_type` for 219/236 (17 third-party gaps → the tokenizer dimension degrades to UNKNOWN, never a crash).

Both ROADMAP-flagged spikes were resolved empirically in this session against the installed venv. **Spike 1 (HF_ENDPOINT):** `huggingface_hub/constants.py:69` reads `HF_ENDPOINT` exactly once at import; setting the env var after import leaves `constants.ENDPOINT` frozen at `https://huggingface.co` for both existing and freshly constructed `HfApi` objects, and `import dnallm.models.model` already pulls `huggingface_hub` into `sys.modules` — so `_setup_huggingface_mirror` (`dnallm/models/model.py:487-498`) is **inert in-process**. The probe must report the EFFECTIVE endpoint via `huggingface_hub.constants.ENDPOINT`, and `HfApi(endpoint=...)` is the working injection seam for endpoint override. **Spike 2 (ModelScope cache layouts):** installed modelscope 1.34.0 resolves the cache root via `get_modelscope_cache_dir()` (`MODELSCOPE_CACHE` env or `~/.cache/modelscope/hub`) and writes models at `{root}/models/{org}/{name}` (name dots → `___`); the legacy pre-restructure layout `{root}/{org}/{name}` persists until a download migrates it; a read-only resolver must isdir-check both. Additionally discovered: `snapshot_download(local_files_only=True)` **litters a `.mdl` file on a miss** (verified: a fabricated repo id left a directory behind), so cache detection must be pure path checks, and modelscope ships its own `scan_cache_dir()` (`modelscope.hub.cache_manager`) that this machine's 239 cached repos scan cleanly through (191 `zhangtaolab/*`).

One contract gap needs a planner decision: CONTEXT decision **D-11 assumes registry entries carry `source=modelscope` for lab models, but `RegistryEntry` has no `source` field** (schema is `name/model/task/kind/tokenizer_type/max_length/base_model`, `extra="forbid"`, and the packaged YAML carries no `source` keys). `is_lab_model(entry)` is the natural proxy (all `zhangtaolab/*` are dual-homed and ModelScope-first by owner directive); D-11's miss→other-hub fallback already covers the rest. Two further discrete-value traps are documented below: `tokenizer_type` records the HF tokenizer CLASS, not the k-mer size (all `-6mer`/`-3mer` plant variants are typed `EsmTokenizer`), so the D-06 length budget cannot be derived from `effective_tokenizer_type` alone; and `load_local_data` demotes headerless csv/tsv to txt parsing — a parity rule the double-read structure test must pin.

**Primary recommendation:** Build `dnallm/preflight/` as five pure modules (`validation.py`, `compatibility.py`, `availability.py`, `config_gen.py`, `report.py`) with zero torch/transformers/datasets imports at module level, an injectable `prober` seam shaped like the existing `download_model(..., downloader=...)` precedent (`dnallm/models/model.py:410-455`), a path-only ModelScope dual-layout resolver, and `validate_sections(config_dict) -> DNALLMConfig` factored verbatim out of `load_config` with a per-section collect-all wrapper used only by `generate_config`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Input = file path (`str | Path`, including pre-split dict-of-splits `{"train": ..., "test": ...}` aligned with `DNADataset.load_local_data`) **plus** plain `list[str]` sequences. Hub dataset names do NOT enter `validate_data` — the network boundary stays with the probe layer.
- **D-02:** Sequence-column detection = deterministic heuristic auto-detection (per-column charset/length-distribution scoring) + explicit parameter override. No LLM in this loop.
- **D-03:** Underlying reader = a lightweight standalone reader inside preflight (pandas/json/pyarrow + hand-written fasta parse), importing NO `datasets`/`torch`/`transformers`. Format-parity with `DNADataset` enforced by a structure test that double-reads the same file through both paths and pins row/column agreement.
- **D-04:** Format surface = csv/tsv/json/parquet/arrow/fasta/txt (full alignment with DNADataset minus pkl). `pkl`/`pickle` inputs return an explicit unsupported-format reason code — never `pickle.load` untrusted files. xlsx deferred.
- **D-05:** FAIL is reserved for hard incompatibility (charset-rejected characters, task-type mismatch on a finetuned entry). Anything mitigable by config or data cleanup is at most WARN + counts.
- **D-06:** Length/truncation: OK/WARN only — truncation NEVER fails. p50/p95/p99/max vs effective max_length with tokenizer-type-adjusted character budget (k-mer ÷ k, char 1:1, BPE conservative ratio) + truncated-row count/fraction. Exact thresholds are planner discretion.
- **D-07:** Charset: data ⊆ model charset → OK; N/ambiguity codes → WARN with per-character counts; charset-外 characters → that model FAILs (per-model verdict, never a global error). The report counts; it never drops rows.
- **D-08:** Task matching: finetuned entries hard-match (`task_type` ≠ user task → FAIL with reason code); base models OK for any supervised task; generation/MLM backbones for supervised tasks → WARN.
- **D-09:** Census conventions: stats normalized via `upper()` (lowercase fraction reported separately); empty/too-short sequences (<20, configurable) get error-level reason codes; all-N sequences get warn. Severity annotations only.
- **D-10:** `probe_availability(model)` single-model primary API; `check_compatibility` listings network-free by default with an OPTIONAL shortlist batch probe parameter (top N≈20). Never probe all 236 entries.
- **D-11:** Hub fallback = probe the registry entry's own `source` first; on miss/failure probe the other hub; when both hubs have it, report ModelScope as primary source.
- **D-12:** `cached` detection = three-way: HF `scan_cache_dir()` + ModelScope dual-layout resolver + local-path entries by direct existence. In-process memo cache; per-probe default timeout ≈5 s configurable; rate-limit/timeout/any exception → UNKNOWN, never "unavailable".
- **D-13:** `generate_config` emits task + model + finetune + inference + lora sections with model-derived defaults (truncation from `effective_max_length`; finetuned → `num_labels`/`label_names`/`threshold` prefilled from the registry entry; unknown model = field-level error, collect-all).
- **D-14:** Round-trip input = YAML path (`str | Path`) OR already-parsed dict; edits apply as a patch dict then revalidate through the shared `validate_sections` path.
- **D-15:** Output = return the full config object (with validation report); write to YAML only when an optional `output_path` is given.

### Claude's Discretion
- Report/verdict object class names and field tables; reason-code enum naming (machine-readable, agent-friendly).
- Exact truncation WARN thresholds (quantile/fraction), BPE conservative budget ratio, shortlist size N, timeout default, probe concurrency cap.
- Standalone reader internal structure; concrete lock points of the double-read parity structure test.
- Generation/regression/multilabel section differences and multilabel separator defaults (follow existing TaskConfig conventions).
- Whether preflight joins the `dnallm/__init__.py` root facade — leaning NO; planner decides.

### Deferred Ideas (OUT OF SCOPE)
- **Agent 壳·麻瓜数据引导** (future Track, NOT v1.4): conversational agent consuming the core via D-6 MCP tools. Phase 23's answer: agent-friendly report objects + deterministic detection heuristics only.
- **xlsx/Excel input support** for `validate_data` — needs openpyxl (new dep, outside budget).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PFC-01 | `validate_data()` read-only collect-all report; never drops/mutates rows | D-03/D-04 reader contract pinned against `load_local_data` (data.py:82-334); census helpers in `dnallm/utils/sequence.py`; double-read parity test design below |
| PFC-02 | `check_compatibility()` three-state verdicts over task × length × tokenizer/charset; WARN never blocks | Registry API verified (registry.py:224-407); tokenizer-budget kinds + k-mer trap; charset fact sources catalogued |
| PFC-03 | ModelScope-first availability probe: cached/remote-confirmed/unknown + size/revision, no weights; rate-limit → unknown; injectable | Spike 2 resolver design; HF/MS adapter signatures verified on installed versions; injection seam precedent |
| PFC-04 | `generate_config()` validated emission + load→edit→revalidate round-trip via shared `validate_sections`; collect-all field errors | `load_config` body mapped section-by-section (configs.py:698-747); section-class map and TaskConfig semantics verified |
| PFC-05 | Intent-routed listings over the base↔finetuned axis (`base_model`) | `get_by_base_model`/`get_by_task_type`/`kind` field verified; task_type census (binary 115, regression 42, …) |
| PFC-06 | Offline/degraded: any probe failure → UNKNOWN, flow continues | Spike 1 endpoint freeze; `OfflineModeIsEnabled` (ConnectionError subclass); error-class catalogue verified |
| PFC-07 | Lab models (`zhangtaolab/*`) rank first in every listing | `is_lab_model`/`sort_key_lab_first` (registry.py:317-343) — drop-in sort keys |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Data census / charset / length stats | Core library (`dnallm/preflight/`) — pure CPU | — | Must be callable from CLI, TUI (Phase 25), MCP (Phase 26), notebooks without dragging torch/transformers into the import graph |
| Format reading (csv/tsv/json/parquet/arrow/fasta/txt) | Core library (standalone reader) | `datasets` lib (parity reference only) | D-03: no `datasets`/`torch`/`transformers` imports; parity locked by structure test against `DNADataset.load_local_data` |
| Model-suitability verdicts | Core library (pure functions over registry) | — | Registry is immutable in-process data; verdicts must never mutate or re-parse |
| Intent-routed listings | Core library | — | `base_model` axis lives in the registry loader (Phase 22); shells only render |
| Availability probing (remote) | Core library (adapter module) | Hub libraries behind injectable seam | PFC-03: unit suite network-free; adapters are the only module permitted to import hub clients |
| Cache detection (local) | Core library (pure path checks + hub scan APIs) | — | Offline-safe; no downloads; no side effects |
| Config emission / round-trip validation | Core library (shares `validate_sections` with `configuration`) | — | The one sanctioned refactor; `load_config` behavior byte-identical |
| YAML dumping | Core library (`config_gen.py` only) | — | D-15: dump logic lives in the same module, same test suite |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic (v2, already core) | installed, `>=2.10.6` per pyproject | Report/verdict objects (`DataReport`, `Verdict`, `AvailabilityReport`) + the existing config sections | Registry entries precedent: typed, frozen where sensible; agent-friendly structured reports [VERIFIED: dnallm/models/registry.py:92-114] |
| pandas (already core) | `>=2.2.3` (installed 3.0.6) | Standalone reader for csv/tsv/json + census computation | Declared core dependency in pyproject [VERIFIED: pyproject.toml dependencies]; CI matrix spans pandas 2.x/3.x — use version-agnostic calls only |
| pyarrow (transitive, guaranteed) | installed 25.0.1 | Parquet/arrow reading without importing `datasets` | `datasets` (core dep, `<=5.1.0`) requires `pyarrow (>=15.0.0)` [VERIFIED: installed dist metadata] — always present wherever dnallm installs |
| huggingface-hub (already core) | `>=0.29.0`, installed 1.33.0 | HF probe adapter: `repo_exists`, `model_info(files_metadata=True)`, `scan_cache_dir`, `try_to_load_from_cache` | All verified live in this venv (see Code Examples) |
| modelscope (already core) | `>=1.23.2`, installed 1.34.0 | ModelScope probe adapter: `HubApi.repo_exists/get_model_files/get_model`, `cache_manager.scan_cache_dir` | Signatures verified offline in this venv; field shapes live-verified 2026-10-20 by milestone research [CITED: .planning/research/STACK.md §D] |
| PyYAML (already core) | — | Round-trip YAML load (`yaml.safe_load`) and dump (`yaml.safe_dump`) for `generate_config` | Same loader the configuration layer already uses [VERIFIED: dnallm/configuration/configs.py:705-706] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `dnallm.models.registry` (in-repo) | Phase 22 | Every suitability fact: resolve/task routing/lab-first ordering/effective values | Always — never re-parse `model_info.yaml` |
| `dnallm.utils.sequence` (in-repo) | — | `check_sequence` / `calc_gc_content` / `seq2kmer` census helpers and the `set(seq.upper()) - set(valid_chars)` charset idiom | Census computation in `validation.py` [VERIFIED: dnallm/utils/sequence.py:94-113] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Standalone pandas reader (D-03) | Reuse `DNADataset.load_local_data` | Rejected by D-03: it imports `datasets`+`transformers` (`PreTrainedTokenizerBase` in its signature, `load_dataset` in `_load_structured_data` [VERIFIED: dnallm/datahandling/data.py:90,243-247]) — exactly the heavy import preflight must avoid |
| `snapshot_download(local_files_only=True)` for ModelScope cache detection | — | Rejected: verified to CREATE `{root}/models/{org}/{name}/.mdl` on a miss (write side effect from a "read-only" probe) |
| Custom REST calls to hubs | Hub libraries | Never — the hub libs are pinned deps handling auth/endpoint/retries [CITED: .planning/research/STACK.md "Don't hand-roll"] |

**Installation:** None. `npm`-style version verification not applicable — Python phase, zero new packages. Installed versions confirmed in the dev venv: `huggingface_hub` 1.33.0, `modelscope` 1.34.0, `pyarrow` 25.0.1, `pandas` 3.0.6, Python 3.13.15.

## Package Legitimacy Audit

> No external packages are installed by this phase. All libraries in the Standard Stack are existing core/transitive dependencies of `dnallm` (verified in `pyproject.toml [project] dependencies` and installed dist metadata this session). Registry checks not applicable.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| (none new) | — | — | — | — | — | Approved (no installs) |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```text
User data (path | dict-of-splits | list[str])      User intent (task, model, "fine-tune"/"predict now")
        │                                                    │
        ▼                                                    ▼
┌─────────────────────────────┐    ┌──────────────────────────────────────────┐
│ preflight/validation.py     │    │ preflight/compatibility.py               │
│ standalone reader           │    │  task match (D-08)                       │
│ (pandas/pyarrow/fasta;      │    │  length budget (D-06: k÷k | 1:1 | BPE)   │◄──── registry.py (Phase 22)
│  NO datasets/torch)         │    │  charset verdict (D-07)                  │      resolve / get_by_task_type /
│  column autodetect (D-02)   │    │  intent listings (PFC-05, PFC-07)        │      get_by_base_model / is_lab_model /
└───────────┬─────────────────┘    └──────────────┬───────────────────────────┘      effective_max_length /
            ▼                                     │                                    effective_tokenizer_type
     DataReport ─────────────────────────────────►│  (pure; registry read-only)
                                                  ▼
                                  ┌─────────────────────────────┐   optional shortlist (N≈20)
                                  │ preflight/availability.py   │◄──────────────────────────┐
                                  │ local: HF scan_cache_dir +  │                           │
                                  │  MS dual-layout isdir +     │   injectable prober seam (PFC-03)
                                  │  local-path existence       │   (unit tests: fakes only)
                                  │ remote: MS-first → HF       │───────────────────────────┤
                                  │  rate-limit/timeout →       │   HfApi(endpoint=constants.ENDPOINT)
                                  │  UNKNOWN (PFC-06)           │   HubApi() — modelscope.cn
                                  └───────────┬─────────────────┘
                                              ▼
                                   AvailabilityReport (cached | remote-confirmed | unknown, size, revision)

User YAML path or dict ──► preflight/config_gen.py ──► patch ──► validate_sections (NEW, factored from
        │                                                          load_config, configs.py:698-747 — zero
        │                                                          behavior change) ──► DNALLMConfig + collect-all
        └────────────────────► load_config (existing) ──────────► same validate_sections path
                                              │
                                              ▼ (optional output_path) yaml.safe_dump
```

Future shells (Phase 25 TUI panels, Phase 26 MCP tools, notebooks) consume the four report objects and own zero verdict logic.

### Recommended Project Structure
```
dnallm/preflight/
├── __init__.py          # public API re-exports (validate_data, check_compatibility,
│                        #   probe_availability, generate_config, report types)
├── report.py            # Pydantic report/verdict objects + reason-code enums (agent-friendly)
├── validation.py        # standalone reader (D-03/D-04) + census + column autodetect (D-02)
├── compatibility.py     # three-state verdicts, tokenizer budgets, charset verdicts, intent listings
├── availability.py      # probe_availability + injectable prober seam + cache resolvers + memo
└── config_gen.py        # generate_config, round-trip patch, YAML dump (D-13/D-14/D-15)
dnallm/configuration/configs.py   # + validate_sections (factored out of load_config)
tests/preflight/                   # test_validation.py, test_compatibility.py,
                                   # test_availability.py, test_config_gen.py,
                                   # test_reader_parity.py (double-read structure test)
```
Structure-test obligations: zero `torch`/`transformers`/`datasets` imports inside `dnallm/preflight/` at module level (lazy imports only if ever needed); zero hub-client imports outside `availability.py`. [CITED: .planning/research/ARCHITECTURE.md "Import lock" + TUI-01 discipline]

### Pattern 1: Injectable probe seam (mirrors `download_model(..., downloader=...)`)
**What:** every remote capability enters as a callable parameter; the default adapter is constructed lazily inside `availability.py`.
**When to use:** `probe_availability(model, *, prober=None, timeout=5.0)`; `check_compatibility(..., prober=None, probe_shortlist=None)`.
**Example:**
```python
# Source: dnallm/models/model.py:410-455 (existing precedent, read this session)
def download_model(model_name: str, downloader: Any, ...) -> str:
    ...
    status = downloader(model_name, **download_kwargs)   # caller injects hf/ms snapshot_download
# Probe mirrors it: tests inject fakes, never the network
# Source: tests/models/test_model.py:66-80 (existing test pattern)
mock_downloader = Mock(side_effect=[Exception("connection error"), "/path/to/model"])
with patch("time.sleep"):
    result = download_model("test-model", mock_downloader, max_try=3)
```

### Pattern 2: `validate_sections` factor-out (the one sanctioned refactor)
**What:** `load_config` becomes file-read + `config_path` injection + `validate_sections(config_dict)`; the section-to-class mapping moves verbatim.
**Verified mapping to move** [VERIFIED: dnallm/configuration/configs.py:709-747, quoted field names]:
- `task` → `TaskConfig(**config_dict["task"])` (only if `"task" in config_dict`)
- `inference` → `InferenceConfig(**config_dict["inference"])`
- `model` → plain dict passthrough (`configs["model"] = config_dict["model"]` — deliberately unvalidated)
- `finetune` → `TrainingConfig`, `lora` → `LoraConfig`, `ia3` → `Ia3Config`, `vep` → `VepConfig`, `sweep` → `SweepConfig`
- `benchmark` → `BenchmarkConfig(config_dict)` — NOTE: consumes the FULL raw dict (including injected `config_path`), not the section
- Every key optional; `config_path` injected only on the file path (`config_dict["config_path"] = os.path.abspath(config_path)` at configs.py:707)
**Collect-all wrapper (new, `generate_config` only — `load_config` behavior unchanged):** iterate the same section map, construct each in `try/except ValidationError`, accumulate `ValidationError.errors()` entries into field-level error records; never let one section's failure hide another's.

### Pattern 3: Report objects are frozen, typed, self-describing
**What:** Pydantic models with `model_config = ConfigDict(frozen=True)` for reports (they are facts, not mutable state), machine-readable reason-code enums (`Severity`/`ReasonCode` string enums), JSON-serializable via `model_dump()`.
**Example precedent** [VERIFIED: dnallm/models/registry.py:106 — `model_config = ConfigDict(frozen=True, extra="forbid")`].

### Pattern 4: Tokenizer budget classifier (length dimension, D-06)
**What:** a preflight-owned table classifying each entry into a budget kind; char budget = `effective_max_length(entry) × factor`.
**Verified classification inputs:**
- k-mer: repo id / entry name carries the k (`zhangtaolab/plant-dnabert-6mer`, `plant-dnamamba-3mer`) — `tokenizer_type` is `EsmTokenizer` for ALL of these and does NOT carry k [VERIFIED: live registry query this session]
- char 1:1: `EsmTokenizer`/`CaduceusTokenizer`/`HyenaDNATokenizer` etc. on non-kmer names
- BPE conservative ratio: `DebertaV2Tokenizer`, `GemmaTokenizer`, `GPT2Tokenizer`, `GPTNeoXTokenizer`, `PreTrainedTokenizerFast` (DNABERT-2-class)
- `effective_tokenizer_type` is `None` for 17 third-party entries → budget kind UNKNOWN → length dimension verdict UNKNOWN (degrade, never crash) [VERIFIED: live registry census]
**Family-level cross-check available:** `MODEL_INFO[family]["model_tags"]` carries `BPE`/`6mer`/`singlebase`/`3mer`… [VERIFIED: dnallm/models/modeling_auto.py:47-68] — but the registry-name join is messy (33/49 pretrained names match MODEL_INFO keys exactly; finetuned names like `"Plant DNABERT BPE promoter"` do not). Prefer the repo-id/name k-mer suffix regex `(\d+)\s*mer`; MODEL_INFO tags as a tiebreaker.

### Anti-Patterns to Avoid
- **Building the report on `validate_sequences`** — it is a destructive whole-row filter (`self.dataset = self.dataset.filter(...)`, prints `[Warning]` with dropped counts) [VERIFIED: dnallm/datahandling/data.py:905-915]. Pre-flight computes what WOULD be dropped, per criterion, and drops nothing.
- **Reading `os.environ` for endpoint/cache reporting** — Spike 1 proves the env var is stale relative to library behavior; report library constants (`huggingface_hub.constants.ENDPOINT`, resolved `HF_HUB_CACHE`, `modelscope` `get_modelscope_cache_dir()`).
- **A second parser of `model_info.yaml`** — REG-01 grep guard fails CI on any new mentioner; preflight goes through `dnallm.models.registry` only [VERIFIED: dnallm/models/registry.py:4-10; tests/models/test_registry.py mentioner allowlist].
- **Mapping exceptions to `available=False`** — every failure maps to UNKNOWN with the reason attached (PFC-03/PFC-06).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Hub REST calls (existence/files/sizes) | `requests` → modelscope.cn `/api/v1/...` | `HfApi` / `HubApi` adapters | Pinned deps handle auth/endpoint/retries/response shapes; adapters are what each hub's own CLI uses [CITED: .planning/research/STACK.md] |
| HF cache detection | Path spelunking `~/.cache/huggingface` | `scan_cache_dir()` + `try_to_load_from_cache(repo_id, "config.json")` | Honors `HF_HOME`/`HF_HUB_CACHE`/XDG precedence resolved at import; both verified live this session |
| ModelScope cache detection | `snapshot_download(local_files_only=True)` | Pure `isdir` on the two resolved candidate layouts (+ optional `modelscope.hub.cache_manager.scan_cache_dir`) | snapshot_download writes `.mdl` on miss (verified) and emits a noisy warning [CITED: STACK.md §D notes] |
| Config validation | Custom field checks | The existing Pydantic sections via `validate_sections` | Single validation path; `ValidationError.errors()` already yields per-field records |
| Registry metadata | Re-reading YAML / re-implementing inheritance | `registry.effective_max_length` / `effective_tokenizer_type` | Inheritance + cycle guard already tested in Phase 22 [VERIFIED: registry.py:346-406] |
| GC content / charset checks | New math | `calc_gc_content`, the `set(seq.upper()) - set(valid_chars)` idiom | Matches training-time behavior exactly [VERIFIED: dnallm/utils/sequence.py:20-113] |

**Key insight:** every deceptively complex sub-problem here (hub error taxonomy, cache-layout variance, config semantics) already has a tested owner in the installed libs or the Phase 22 loader; the phase's net-new code is verdict logic, reason codes, and the parity-locked reader.

## Runtime State Inventory

> The `validate_sections` factor-out is a code-level refactor with zero behavior change — no runtime state, stored data, or OS registrations are renamed or migrated by this phase. Categories checked and explicitly empty:

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — preflight is read-only; reports are return values, never persisted | None |
| Live service config | None — no services touched; MCP server untouched (Phase 26) | None |
| OS-registered state | None | None |
| Secrets/env vars | `MODELSCOPE_CACHE` / `HF_ENDPOINT` / `HF_HUB_CACHE` / `HF_HOME` are READ (via library resolvers) but never written by preflight. Note: `_setup_huggingface_mirror` (model.py:487-498) writes `HF_ENDPOINT` at runtime — preflight must NOT replicate that pattern (inert per Spike 1) | None — read-only consumption |
| Build artifacts | None — no packaging changes this phase | None |

## Common Pitfalls

### Pitfall 1: `HF_ENDPOINT` set after import is inert (Spike 1 — RESOLVED)
**What goes wrong:** reporting the endpoint from `os.environ` lies; a TUI toggle promising in-process mirror switching lies.
**Empirical evidence (this venv, 2026-10-20):**
- `huggingface_hub/constants.py:69`: `ENDPOINT = os.getenv("HF_ENDPOINT", _HF_DEFAULT_ENDPOINT).rstrip("/")` — read once at import [VERIFIED: live source inspection]
- After `import huggingface_hub`, `os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"` leaves `constants.ENDPOINT == "https://huggingface.co"` AND fresh `HfApi()` instances still resolve `https://huggingface.co` (`self.endpoint = endpoint if endpoint is not None else constants.ENDPOINT` in `HfApi.__init__`) [VERIFIED: live probe]
- `import dnallm.models.model` puts `huggingface_hub` and `transformers` in `sys.modules` [VERIFIED: live probe] — so `_setup_huggingface_mirror` (model.py:487-498, invoked at model.py:1250) can never take effect in-process once dnallm is imported.
**How to avoid:** report `huggingface_hub.constants.ENDPOINT`; use `HfApi(endpoint=...)` for any endpoint override (verified working); document mirror toggling as restart-required/subprocess-bound (Phase 25 concern).

### Pitfall 2: ModelScope probe litters the cache (NEW, discovered this session)
**What goes wrong:** `snapshot_download(local_files_only=True)` on a cache miss CREATES `{root}/models/{org}/{name}/.mdl` (60-byte msgpack) — a write side effect from a "read-only" probe. Verified: the fabricated id `zhangtaolab/definitely-not-cached-xyz` left exactly this artifact.
**How to avoid:** cache detection = pure `os.path.isdir` on candidate paths (see resolver design); reserve `snapshot_download` for Phase 25's actual downloads.

### Pitfall 3: Rate limits / offline mode read as "unavailable"
**What goes wrong:** HF 429s (shared CI/nightly egress IPs; transformers ≥5 can burst `model_info` into 429) and `HF_HUB_OFFLINE=1` raising `OfflineModeIsEnabled` (subclasses `ConnectionError`) get mistranslated into false negatives [CITED: .planning/research/PITFALLS.md Pitfall 5].
**How to avoid:** exhaustive-catch per probe → UNKNOWN + reason string. HF: `HfHubHTTPError` instances carry `.response` (check `status_code == 429`); `RepositoryNotFoundError`/`GatedRepoError` are DISTINCT (a 404/gated repo is a real "not on this hub" signal → try the other hub, not UNKNOWN). ModelScope limits officially undocumented — treat every non-404 failure as UNKNOWN. [VERIFIED: `huggingface_hub.errors` class list inspected this session]

### Pitfall 4: `tokenizer_type` does not carry the k-mer size
**What goes wrong:** the D-06 "k-mer ÷ k" budget computed from `effective_tokenizer_type` alone is wrong by up to 6×: `zhangtaolab/plant-dnabert-6mer` and every `-3mer/4mer/5mer/6mer` variant are typed `EsmTokenizer` [VERIFIED: live registry query].
**How to avoid:** k comes from a repo-id/entry-name regex `(\d+)\s*mer`; `EsmTokenizer` on non-kmer names is char 1:1; BPE-class tokenizers get the conservative ratio (planner sets the constant).

### Pitfall 5: Registry has no `source` field (D-11 premise gap)
**What goes wrong:** implementing D-11 as "read entry.source" — the field does not exist (`RegistryEntry` fields: `name/model/task/kind/tokenizer_type/max_length/base_model`, `extra="forbid"`; the YAML carries no `source` keys) [VERIFIED: registry.py:92-114 + grep of model_info.yaml].
**How to avoid:** `is_lab_model(entry)` is the ModelScope-first proxy (all `zhangtaolab/*` are dual-homed; owner directive makes MS primary); third-party entries get MS-probe → HF-fallback per D-11's own fallback rule. If a real `source` field is wanted, that is a registry-schema edit — out of this phase's boundary (registry is complete and read-only-forever).

### Pitfall 6: Reader parity — the txt demotion rule
**What goes wrong:** a headerless csv is NOT parsed as csv by `DNADataset`: `_check_header_and_type` demotes csv/tsv/txt to txt parsing when the first line contains NEITHER `seq_col` NOR `label_col` [VERIFIED: data.py:220-235]. A standalone reader that "helpfully" parses headerless csv as csv breaks parity.
**Also pin:** fasta label = header split on `fasta_sep="|"`, last part [VERIFIED: data.py:259-279]; txt first-line-contains-both-columns → csv re-parse, else `split(sep)` / `split()` needing ≥2 fields [VERIFIED: data.py:282-300]; list-of-files accepted for structured formats, `ValueError` for fasta/txt/pkl [VERIFIED: data.py:177-188]; `os.path.expanduser` normalization [VERIFIED: data.py:201-209]; label formatting (str → float attempt, multilabel split on `multi_label_sep`) [VERIFIED: data.py:314-334]; rename only when target name absent [VERIFIED: data.py:303-311].

### Pitfall 7: Pydantic raises first-failure — collect-all needs a wrapper
**What goes wrong:** calling `load_config` on a generated config surfaces one traceback-shaped error, not the field-level list PFC-04 demands.
**How to avoid:** `validate_sections` preserves `load_config`'s raise-first behavior byte-for-byte; `generate_config` adds the per-section try/except collect-all around the SAME section map. Do not change `load_config`'s error contract (existing tests + all callers depend on it).

### Pitfall 8: Unknown model in `generate_config`
**What goes wrong:** `registry.resolve("bogus")` raises `ValueError(f"Model {name_or_repo_id} not found in registry.")` [VERIFIED: registry.py:224-244] — a top-level traceback, violating D-13's field-level error rule.
**How to avoid:** use `try_resolve()` (returns `None`) inside config generation; emit a field-level error record.

### Pitfall 9: Probe cost and the memo
**What goes wrong:** `check_compatibility` probing all 236 entries = a 429 storm [CITED: PITFALLS.md Pitfall 5]; ModelScope `HubApi` methods expose NO per-call timeout (session-level only) [VERIFIED: signature inspection].
**How to avoid:** network-free listings by default (D-10), optional shortlist N≈20, in-process memo (D-12), and the core's own bounded-wait guard (e.g., run the sync adapter call under a `concurrent.futures` timeout) rather than relying on library timeouts.

### Pitfall 10: pandas 3.x in the dev venv
**What goes wrong:** reader code written against pandas 2 idioms may differ under the installed pandas 3.0.6 (dev venv) while CI rides other versions.
**How to avoid:** stick to stable, long-public APIs (`read_csv`/`read_json`/`DataFrame` ops); the parity double-read test runs in CI across the matrix and catches drift.

## Code Examples

### HF availability adapter (verified live in this venv, 2026-10-20)
```python
# Source: live venv probes this session (huggingface_hub 1.33.0)
import huggingface_hub.constants as C
from huggingface_hub import HfApi, scan_cache_dir, try_to_load_from_cache
from huggingface_hub.errors import (
    GatedRepoError, HfHubHTTPError, OfflineModeIsEnabled, RepositoryNotFoundError,
)

EFFECTIVE_ENDPOINT = C.ENDPOINT            # import-time resolved; NEVER os.environ["HF_ENDPOINT"]
api = HfApi(endpoint=EFFECTIVE_ENDPOINT)   # explicit endpoint injection seam (verified)

info = api.model_info(repo_id, files_metadata=True)   # .sha (revision), .siblings[].rfilename/.size
cached = try_to_load_from_cache(repo_id, "config.json")  # snapshot file path str, or None — offline-safe
scan = scan_cache_dir()                                # .repos: each .repo_id/.size_on_disk/.repo_path
# observed live: try_to_load_from_cache('zhangtaolab/plant-dnabert-BPE', 'config.json')
#   -> '/home/forrest/.cache/huggingface/hub/models--zhangtaolab--plant-dnabert-BPE/snapshots/9cac.../config.json'
```

### ModelScope availability adapter + dual-layout resolver (verified live, modelscope 1.34.0)
```python
# Source: live venv probes + installed-source inspection this session
from modelscope.hub.api import HubApi
from modelscope.utils.file_utils import get_modelscope_cache_dir
# get_modelscope_cache_dir() == expanduser(getenv('MODELSCOPE_CACHE', '~/.cache/modelscope/hub'))
#   [VERIFIED: modelscope/utils/file_utils.py:37-54]
# HubApi.repo_exists(repo_id, *, repo_type=None, endpoint=None, re_raise=False, token=None) -> bool
# HubApi.get_model_files(model_id, revision='master', root=None, recursive=False, ...) -> List[dict]
#   entries carry Name/Path/Size(bytes)/Sha256/Revision/Type/IsLFS/CommittedDate
#   [CITED: STACK.md live-verified 2026-10-20; signatures re-verified this session]

def modelscope_cache_candidates(repo_id: str) -> list[str]:
    org, name = repo_id.split("/", 1)
    name = name.replace(".", "___")           # modelscope's own convention (file_download.py:376)
    root = get_modelscope_cache_dir()         # honors MODELSCOPE_CACHE
    return [
        f"{root}/models/{org}/{name}",        # current layout (verified: plant-dnabert-BPE lives here)
        f"{root}/{org}/{name}",               # legacy pre-restructure layout (migration source,
                                              #   move_legacy_cache_to_standard_dir file_download.py:316-348)
    ]
# cached = any(os.path.isdir(p) for p in candidates)  — pure, offline, zero side effects
# Optional bulk view: from modelscope.hub.cache_manager import scan_cache_dir  -> ModelScopeCacheInfo
#   (verified live: 239 repos, 191 zhangtaolab, revision 'master'; raises CacheNotFound if root absent)
```

### Registry-driven verdict inputs (verified live)
```python
# Source: dnallm/models/registry.py:224-406 + live census this session
from dnallm.models import registry

entry = registry.try_resolve("zhangtaolab/plant-dnabert-BPE-promoter")  # None on miss, never raises
registry.effective_max_length(entry)      # int — resolves base_model chain; 100% coverage (236/236)
registry.effective_tokenizer_type(entry)  # 'EsmTokenizer' | ... | None (17 third-party gaps)
registry.get_by_task_type("binary")       # intent routing: "predict now" (115 binary entries exist)
registry.get_by_base_model("zhangtaolab/plant-dnabert-BPE")  # lineage: "fine-tune" context
sorted(entries, key=registry.sort_key_lab_first)             # PFC-07 ordering, drop-in
# entry.task.num_labels / label_names / threshold  -> D-13 config prefill (registry.py:53-74)
```

### Census helpers (existing, reuse)
```python
# Source: dnallm/utils/sequence.py:20-113
calc_gc_content(seq)                    # uppercases, U->T, drops N; 0.0 on empty
set(seq.upper()) - set(valid_chars)     # charset census idiom — matches check_sequence exactly
# Census conventions (D-09): normalize via upper(); report lowercase fraction separately;
#   min length 20 configurable; all-N sequences -> warn severity
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Runtime `HF_ENDPOINT` toggling in-process (`_setup_huggingface_mirror`) | Endpoint frozen at huggingface_hub import; override via `HfApi(endpoint=...)` or subprocess | Observed on installed 1.33.0 (spike 1) | Preflight reports effective endpoint; mirror toggle becomes restart-required UX (Phase 25) |
| Assume single ModelScope cache layout (`model.py:1545` hardcodes `~/.cache/modelscope/hub`) | Root via `get_modelscope_cache_dir()` honoring `MODELSCOPE_CACHE`; dual layout check | modelscope 1.34.0 (spike 2) | The hardcoded `clear_model_cache` path at model.py:1543-1545 misses `MODELSCOPE_CACHE` machines — the preflight resolver is the correct pattern; do NOT copy the hardcode |

**Deprecated/outdated:**
- `HUGGINGFACE_HUB_CACHE` env var → `HF_HUB_CACHE` (deprecated upstream; resolved constants already handle precedence) [CITED: .planning/research/PITFALLS.md Pitfall 5]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `HubApi.get_model_files` field shape (`Name/Path/Size/Sha256/Revision/Type/IsLFS/CommittedDate`) — live-verified 2026-10-20 by milestone research against the same venv, not re-fetched this session (offline discipline) | Standard Stack / Code Examples | Field-name drift within a pinned modelscope minor — containable: adapter is one module, faked in tests |
| A2 | ModelScope `HubApi.repo_exists(re_raise=False)` returns a clean `False` for absent repos (signature verified; 404 path not exercised offline this session) | Code Examples | A raise on 404 → caught by the exhaustive handler → UNKNOWN instead of "not on this hub"; probe still never crashes |
| A3 | HF `HfHubHTTPError` exposes `.response.status_code` on instances (class list verified; instance attribute pattern per upstream design, not constructed live) | Pitfall 3 | Rate-limit detection falls back to the generic UNKNOWN catch — behavior identical (429 → UNKNOWN either way) |
| A4 | `zhangtaolab/*` repos exist on BOTH hubs (dual-homed) — asserted by D-11/architecture; MS cache evidence on this box supports it but hub membership was not re-probed this session | Pitfall 5 | If a lab repo were HF-only, MS-first probe order costs one extra miss before HF fallback — verdict still correct |
| A5 | pandas 3.x API compatibility of chosen reader calls across CI matrix (installed 3.0.6 dev / matrix pins other versions) | Pitfall 10 | Parity double-read test in CI catches drift per leg |
| A6 | Exact 13-family strict-charset membership — the count appears only in the `validate_sequences` docstring hazard note [VERIFIED: data.py:879-886 as a documented claim]; the concrete family list must be DERIVED by this phase (no in-repo table exists) | Compatibility design | List is data, owned by preflight's tokenizer-type→charset table; corrections are additive reason codes |

## Open Questions

1. **D-11 `source` proxy — confirm `is_lab_model` is the intended stand-in**
   - What we know: `RegistryEntry` has no `source` field; the packaged YAML carries none; all `zhangtaolab/*` entries are lab-built and ModelScope-hosted.
   - What's unclear: whether the owner wants a real `source` field added to the registry schema later (a Phase 22-schema edit) vs the proxy forever.
   - Recommendation: implement with `is_lab_model` proxy + D-11 fallback ordering now (zero registry change); note the proxy in code docs. Planners should not block on this.

2. **Report-object naming and reason-code vocabulary** (Claude's discretion per CONTEXT)
   - What we know: agent-friendly, machine-readable reason codes are the stated requirement; registry precedent is frozen Pydantic + descriptive ValueErrors.
   - Recommendation: `Severity(str, Enum)` = `info|warn|error` for data findings; verdict status `ok|warn|fail|unknown` per dimension; reason codes as `UPPER_SNAKE` string enums (e.g. `CHARSET_REJECTED_CHARS`, `TRUNCATION_ABOVE_THRESHOLD`, `TASK_TYPE_MISMATCH`, `RATE_LIMITED`, `UNSUPPORTED_FORMAT_PKL`). Planner finalizes.

3. **Does `dnallm/preflight` join the root facade?**
   - What we know: root `dnallm/__init__.py` eagerly imports torch-heavy subpackages today [VERIFIED: dnallm/__init__.py]; CONTEXT leans NO.
   - Recommendation: NO facade membership — `from dnallm.preflight import ...`. Keeps the phase honest about import-lightness and avoids touching the facade's load-order comment contract.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| huggingface-hub | HF probe adapter | ✓ | 1.33.0 (floor >=0.29.0) | — |
| modelscope | MS probe adapter | ✓ | 1.34.0 (floor >=1.23.2) | — |
| pydantic | report objects / config sections | ✓ | v2 (>=2.10.6) | — |
| pandas | standalone reader | ✓ | 3.0.6 (>=2.2.3) | — |
| pyarrow | parquet/arrow reading | ✓ | 25.0.1 (guaranteed via `datasets` core dep) | — |
| PyYAML | round-trip load/dump | ✓ | core dep | — |
| Network (for slow-lane probe tests only) | real-network probe acceptance | ✓ (runner box; fast lane stays network-free by design) | — | typed skip `network-unavailable:` [VERIFIED: tests/expected_skips.yaml] |

**Missing dependencies with no fallback:** none
**Missing dependencies with fallback:** none

## Validation Architecture

> `workflow.nyquist_validation` is explicitly `false` in `.planning/config.json` — section skipped per contract.

Test-infrastructure facts the planner still needs (project conventions, not this section's table): pytest config in `pyproject.toml [tool.pytest.ini_options]` (`testpaths=["tests","dnallm/mcp/tests"]`, strict markers, asyncio auto, timeout 300) [VERIFIED: pyproject.toml:405-425]; markers `slow`/`integration`/`unit`… registered; coverage gate `source_pkgs=["dnallm"]`, `fail_under=90`, omit list at pyproject.toml:442-452 (preflight must NOT join the omit list); typed skip prefixes live in `tests/expected_skips.yaml` (`network-unavailable:`, `environment-unavailable:`, `optional-dep:`) and unmatched skips fail CI via `scripts/audit_skips.py` [VERIFIED: file header]; owner rule: every `dnallm/` change ships with pytest in the same change; per-module ≥96% for new code; real-network probe tests ride the `slow` marker, unit suite network-free via injected fakes (Mock pattern at tests/models/test_model.py:57-103).

## Security Domain

> `security_enforcement: true`, `security_asvs_level: 1`, `security_block_on: high` (`.planning/config.json`).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No user auth in scope (hub probing uses the libraries' ambient token handling; no credentials handled by preflight) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | **yes** | The phase IS input validation: untrusted user data files read read-only; **pickle refused by design (D-04)** — `pickle.load` on user files is an RCE vector and `_load_dict_data`'s pattern (data.py:251-256) must NOT be replicated; round-trip YAML via `yaml.safe_load` only (never `yaml.load` without SafeLoader — registry precedent uses CSafeLoader/SafeLoader [VERIFIED: registry.py:214-218]); file-type dispatch by suffix with an explicit unsupported-format reason code |
| V6 Cryptography | no | — |
| V8 Logging | partial | Log summary counts and reason codes, never raw user sequences at scale (genomic data minimization); follow the no-bare-print rule (`T20` ruff) |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malicious `.pkl` dataset (arbitrary code execution on load) | Tampering / Elevation | D-04: unsupported-format reason code; never `pickle.load` user input [VERIFIED: the existing pkl path is the contrast case, data.py:251-256] |
| YAML deserialization attacks in round-trip edit | Tampering / Elevation | `yaml.safe_load` only (PyYAML safe loaders cannot construct arbitrary Python objects) |
| Path traversal / surprise writes via `output_path` | Tampering | `generate_config` writes only when the user passes `output_path`; single `yaml.safe_dump` to that exact path (no derived paths, no overwrites of inputs beyond what the user names — document in docstring) |
| Probe-layer SSRF / endpoint manipulation | Tampering | Probes hit fixed hub endpoints via pinned libraries (`constants.ENDPOINT`, modelscope default `https://www.modelscope.cn` [VERIFIED: `get_endpoint()` live]); no user-controlled URL construction anywhere in preflight |
| Cache-poisoning verdicts (false "cached") | Spoofing | Cache detection is presence-based (isdir) — a stale/partial cache yields `cached` with size metadata; the report carries size/revision so the TUI can surface anomalies; no code is executed from cache paths |

## Sources

### Primary (HIGH confidence)
- Live venv spikes this session (Python 3.13.15; huggingface_hub 1.33.0, modelscope 1.34.0): HF_ENDPOINT freeze (`constants.py:69`, `HfApi.__init__`, dnallm import ordering), ModelScope cache layouts (`file_utils.py:37-63`, `file_download.py:316-380`, `cache_manager.py:291-439`), `snapshot_download` miss side effect, `scan_cache_dir` both hubs, `try_to_load_from_cache`, HubApi signatures, `HfApi(endpoint=...)`, error-class catalogue
- In-repo reads this session: `dnallm/models/registry.py` (full), `dnallm/configuration/configs.py:104-180,270-330,556-648,660-747`, `dnallm/datahandling/data.py:82-334,848-915,1340-1400`, `dnallm/utils/sequence.py`, `dnallm/models/model.py:410-570,1530-1565`, `dnallm/models/tokenizer.py:13-62`, `dnallm/models/modeling_auto.py:47-68`, `dnallm/__init__.py`, `pyproject.toml`, `tests/expected_skips.yaml`, `tests/conftest.py`, `tests/models/test_model.py`, live registry census (236 entries, task/tokenizer/max-length coverage)

### Secondary (MEDIUM confidence)
- `.planning/research/STACK.md` §D (probe adapters live-verified 2026-10-20: `get_model_files` field shapes, cross-hub byte-identical sizes, snapshot_download cache behavior, timeout caveats)
- `.planning/research/PITFALLS.md` Pitfall 5 (env-var import semantics, rate limits, ModelScope layout history) — its "two layouts" claim now superseded in detail by this session's source-level resolver evidence

### Tertiary (LOW confidence)
- None — no training-data-only claims were load-bearing; the ModelScope rate-limit posture stays explicitly "undocumented → design for 429-class resilience"

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — zero new deps; every adapter verified against installed versions
- Architecture: HIGH — registry API, config section map, and reader contract read at source level this session
- Pitfalls: HIGH — both flagged spikes resolved empirically; one new pitfall (probe cache littering) discovered and verified

**Research date:** 2026-10-20
**Valid until:** 2026-11-20 (stable registry/libs; revisit if modelscope or huggingface-hub majors change)
