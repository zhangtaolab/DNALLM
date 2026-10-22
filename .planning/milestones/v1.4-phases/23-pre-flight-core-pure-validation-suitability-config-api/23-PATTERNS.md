# Phase 23: Pre-flight Core — Pure Validation / Suitability / Config API - Pattern Map

**Mapped:** 2026-10-20
**Files analyzed:** 11 (6 new source + 1 modified source + 5 new test)
**Analogs found:** 11 / 11

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `dnallm/preflight/__init__.py` | package facade | — | `dnallm/models/registry.py` (`__all__` + lazy pattern) | role-match |
| `dnallm/preflight/report.py` | model (Pydantic reports) | transform | `dnallm/models/registry.py:53-114` (`TaskInfo`/`RegistryEntry`) | exact |
| `dnallm/preflight/validation.py` | utility (reader + census) | file-I/O / batch | `dnallm/datahandling/data.py:82-334` (`load_local_data` + helpers) | role-match (parity reference, NOT import) |
| `dnallm/preflight/compatibility.py` | service (pure verdicts) | transform | `dnallm/models/registry.py:247-406` (query functions) + `dnallm/tasks/metric_registry.py` | exact |
| `dnallm/preflight/availability.py` | service (probe adapters) | request-response | `dnallm/models/model.py:410-484` (`download_model` injectable downloader) | exact |
| `dnallm/preflight/config_gen.py` | service (config emission) | transform | `dnallm/configuration/configs.py:698-747` (`load_config`) | exact |
| `dnallm/configuration/configs.py` (modify: factor out `validate_sections`) | config | transform | its own `load_config` body | exact |
| `tests/preflight/test_validation.py` | test | — | `tests/models/test_model.py` (Mock/patch idiom) | role-match |
| `tests/preflight/test_compatibility.py` | test | — | `tests/models/test_registry.py` conventions (pure-function tests) | role-match |
| `tests/preflight/test_availability.py` | test | — | `tests/models/test_model.py:57-103` (injected fakes, `patch("time.sleep")`) | exact |
| `tests/preflight/test_config_gen.py` + `test_reader_parity.py` | test | — | existing config tests + double-read structure-test idiom (`tests/examples/test_examples.py` section separators) | role-match |

## Pattern Assignments

### `dnallm/preflight/report.py` (model, transform)

**Analog:** `dnallm/models/registry.py:53-114` — frozen typed Pydantic data objects (research Pattern 3).

**Object pattern** (registry.py:68,106):
```python
class RegistryEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str = Field(min_length=1)
    tokenizer_type: str | None = Field(default=None, min_length=1)
```
Copy this shape for `DataReport` / `Verdict` / `AvailabilityReport`: `ConfigDict(frozen=True)`, `Field(...)` constraints, `X | None` PEP 604 unions, Google docstring with `Attributes:`. Reason codes as `UPPER_SNAKE` string enums (`enum.Enum` with `str` mixin) per research Open Question 2.

**Module header pattern** (registry.py:1-36): numbered design-invariant docstring + `logger = get_logger("dnallm.preflight.report")` + explicit `__all__` — copy verbatim style.

---

### `dnallm/preflight/validation.py` (utility, file-I/O + batch)

**Analog:** `dnallm/datahandling/data.py:82-334` — the parity contract, read as reference only. NEVER import from it (D-03); import only stdlib + pandas/pyarrow.

**Parity behaviors to replicate in the standalone reader (from data.py):**
- Path normalization (201-209): `os.path.expanduser` on each path; `file_type = os.path.basename(path).split(".")[-1].lower()`.
- Suffix dispatch + list-of-files rules (175-190): structured formats accept lists; fasta/txt reject lists with `ValueError`. Add pkl → explicit `UNSUPPORTED_FORMAT_PKL` reason code instead (D-04); do NOT replicate `_load_dict_data`'s `pickle.load` (251-256) — security ASVS V5.
- Headerless-csv demotion (220-235): csv/tsv/txt whose first line contains neither `seq_col` nor `label_col` → parse as txt. Pin this in the parity test (research Pitfall 6).
- Fasta parse (259-279): label = `header.split(fasta_sep)[-1] if fasta_sep in header else header`, sequence accumulation across lines.
- Txt parse (282-300): first line containing both `seq_col` and `label_col` → csv re-parse; else `line.split(sep) if sep else line.split()` keeping first two fields.
- Separator defaults (212-218): csv→`","`, tsv→`"\t"`, else caller's.
- Label formatting semantics (314-334): str→float attempt, multilabel split on `multi_label_sep` — mirror as report metadata, never mutate rows (PFC-01).

**Census helpers** — reuse `dnallm/utils/sequence.py` directly (it is import-light):
```python
# dnallm/utils/sequence.py:29-34, 113
seq = seq.upper().replace("U", "T").replace("N", "")
gc = (seq.count("G") + seq.count("C")) / len(seq)
set(seq.upper()) - set(valid_chars)   # charset census idiom (check_sequence, line 113)
```

---

### `dnallm/preflight/compatibility.py` (service, transform — pure functions over registry)

**Analog:** `dnallm/models/registry.py:224-406` + `dnallm/tasks/metric_registry.py:1-44`.

**Pure-function query style** (registry.py:247-258):
```python
def try_resolve(name_or_repo_id: str) -> RegistryEntry | None:
    """... fall-through seam ..."""
    registry_data = _load_registry()
    entry = registry_data.by_name.get(name_or_repo_id)
```
Module-level functions (not classes) with full Google docstrings, `try_*` variant returning `None` alongside raising `resolve`. Use `registry.try_resolve` (never `resolve` — Pitfall 8), `effective_max_length`, `effective_tokenizer_type`, `get_by_task_type`, `get_by_base_model`, `is_lab_model`, `sort_key_lab_first`.

**Import-light discipline** (metric_registry.py:21-24): "no torch or sklearn imports at module level — each callable lazily imports". Preflight goes further: zero torch/transformers/datasets imports anywhere (structure-test enforced). Copy metric_registry's numbered-invariants module docstring.

**K-mer budget classifier** (research Pattern 4): derive k from repo-id regex `(\d+)\s*mer` — `effective_tokenizer_type` is `EsmTokenizer` for ALL k-mer variants (Pitfall 4); `None` tokenizer → budget UNKNOWN, verdict UNKNOWN.

---

### `dnallm/preflight/availability.py` (service, request-response — probe adapters)

**Analog:** `dnallm/models/model.py:410-484` (`download_model`).

**Injectable seam pattern** (model.py:410-455) — signature + docstring contract:
```python
def download_model(model_name: str, downloader: Any, revision: str | None = None,
                   max_try: int = 10, allow_patterns: list[str] | None = None) -> str:
    """...
    Args:
        downloader: Download function to use (e.g., snapshot_download)
    ...
    """
    download_kwargs: dict[str, Any] = {"revision": revision}
    status = downloader(model_name, **download_kwargs)
```
Mirror as `probe_availability(model, *, prober=None, timeout=5.0)` — hub clients imported lazily inside adapter functions only; this is the ONLY module permitted to import hub clients.

**Exception taxonomy precedent** (model.py:460-478): classify by exception string/class before deciding verdict. Preflight version: `RepositoryNotFoundError`/`GatedRepoError` → real miss (try other hub); `HfHubHTTPError` 429 / `OfflineModeIsEnabled` / any other exception → UNKNOWN (Pitfall 3). Do NOT copy `_setup_huggingface_mirror` (model.py:487-498) — proven inert in-process (Spike 1); report `huggingface_hub.constants.ENDPOINT` instead.

**Cache resolvers** (research Pattern 2 / Code Examples): HF `scan_cache_dir()` + `try_to_load_from_cache`; ModelScope dual-layout pure `isdir` via `get_modelscope_cache_dir()`; never `snapshot_download(local_files_only=True)` (litters `.mdl`, Pitfall 2).

---

### `dnallm/preflight/config_gen.py` + `validate_sections` refactor in `dnallm/configuration/configs.py` (service, transform)

**Analog:** `dnallm/configuration/configs.py:698-747` — the exact body to factor out.

**Section map to move verbatim** (configs.py:709-747):
```python
configs: DNALLMConfig = {}
if "task" in config_dict:
    configs["task"] = TaskConfig(**config_dict["task"])
if "inference" in config_dict:
    configs["inference"] = InferenceConfig(**config_dict["inference"])
if "model" in config_dict:
    configs["model"] = config_dict["model"]      # plain dict passthrough
if "finetune" in config_dict:
    configs["finetune"] = TrainingConfig(**config_dict["finetune"])
if "lora" in config_dict:
    configs["lora"] = LoraConfig(**config_dict["lora"])
if "ia3" in config_dict:
    configs["ia3"] = Ia3Config(**config_dict["ia3"])
if "vep" in config_dict:
    configs["vep"] = VepConfig(**config_dict["vep"])
if "sweep" in config_dict:
    configs["sweep"] = SweepConfig(**config_dict["sweep"])
if "benchmark" in config_dict:
    configs["benchmark"] = BenchmarkConfig(config_dict)  # FULL raw dict, not the section
```
After the refactor `load_config` = file open + `config_path` injection (configs.py:705-707: `yaml.safe_load` + `config_dict["config_path"] = os.path.abspath(config_path)`) + call to `validate_sections(config_dict)`. Raise-first behavior preserved byte-for-byte.

**Collect-all wrapper (new, `generate_config` only)** — wrap the same map per-section:
```python
try:
    configs[section] = section_cls(**config_dict[section])
except ValidationError as e:
    errors.extend(e.errors())   # field-level records, keep going
```
**YAML loading safety** (registry.py:218 precedent): `yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))` with the `# ruff: ignore[unsafe-yaml-load]` comment; dump with `yaml.safe_dump` only when `output_path` given (D-15).

---

### `dnallm/preflight/__init__.py` (package facade)

**Analog:** `dnallm/models/registry.py:38-50` — explicit `__all__` + docstring stating the single-parse/read-only contract. Keep it re-export-only (`validate_data`, `check_compatibility`, `probe_availability`, `generate_config`, report types). Per research Open Question 3: do NOT add to root `dnallm/__init__.py`.

---

## Shared Patterns

### Frozen Pydantic data objects
**Source:** `dnallm/models/registry.py:68,106,117-123` — `ConfigDict(frozen=True, extra="forbid")`, `@dataclass(frozen=True)` for internal indexes.
**Apply to:** all of `report.py`.

### Descriptive matchable ValueError
**Source:** `dnallm/models/registry.py:243` — `raise ValueError(f"Model {name_or_repo_id} not found in registry.")`; collect-and-rethrow wrapping at 167-174 (`raise ValueError(...) from e`).
**Apply to:** every raise in preflight; tests assert `pytest.raises(ValueError, match=r"...")`.

### Injectable dependency seam
**Source:** `dnallm/models/model.py:410-455` (`downloader` param); test pattern `tests/models/test_model.py:57-103`:
```python
mock_downloader = Mock(side_effect=[Exception("connection error"), "/path/to/model"])
with patch("time.sleep"):
    result = download_model("test-model", mock_downloader, max_try=3)
```
**Apply to:** `probe_availability`/`check_compatibility` prober param; all availability unit tests inject `Mock`/fakes — suite stays network-free (PFC-03). Real-network acceptance tests ride the `slow` marker with typed skip prefix `network-unavailable:` (`tests/expected_skips.yaml`).

### Import-light pure module
**Source:** `dnallm/tasks/metric_registry.py:21-24` — lazy imports inside callables, numbered-invariants docstring, `MappingProxyType` frozen surface.
**Apply to:** every `dnallm/preflight/` module; structure test asserts zero torch/transformers/datasets module-level imports and zero hub-client imports outside `availability.py`.

### Google docstrings + logging
**Source:** `dnallm/models/registry.py` (every public function) — `Args:`/`Returns:`/`Raises:`; `logger = get_logger("dnallm.preflight.<mod>")`; no bare `print` (`T20`).
**Apply to:** all new modules; log counts/reason codes, never raw sequences (ASVS V8).

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `tests/preflight/test_reader_parity.py` | test | — | No existing double-read parity test in repo; construct per research Pitfall 6 lock points (txt demotion, fasta label rule, list-of-files rules, label formatting) comparing preflight reader vs `DNADataset.load_local_data` on the same fixtures |

All adapter API details (HF/ModelScope calls, cache layouts, endpoint freeze) have no in-repo analog — use the live-verified examples in `23-RESEARCH.md` §Code Examples verbatim.

## Metadata

**Analog search scope:** `dnallm/models/`, `dnallm/configuration/`, `dnallm/datahandling/`, `dnallm/utils/`, `dnallm/tasks/`, `tests/models/`
**Files scanned:** 8 analogs read (registry.py, configs.py, data.py, sequence.py, model.py, test_model.py, metric_registry.py)
**Pattern extraction date:** 2026-10-20
**Tracked-source check:** all analogs are files under `dnallm/` and `tests/` committed on branch v1.4 (verified via recent-commits/context; none reside in gitignored mirrors)
