# Phase 27: Transformers 4/5 Model Compatibility Census - Pattern Map

**Mapped:** 2026-10-27
**Files analyzed:** 9 (new + modified)
**Analogs found:** 8 / 9 (one partial: generated docs page)

All analog paths verified git-tracked (`git ls-files` non-empty for every named source).

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `scripts/census_compat.py` (new) | utility/script (CI audit tool) | batch (fetch + subprocess probe + emit JSON evidence) | `scripts/audit_skips.py` | exact |
| `dnallm/models/registry.py` (modify: RegistryEntry fields) | model (schema) | transform | itself + REG-02 `tokenizer_type`/`max_length` precedent | exact |
| `dnallm/models/model_info.yaml` (modify: + compat fields on 49 pretrained rows) | config/data | file-I/O (scripted text insertion, never round-trip) | REG-02 enrichment commits; enforced by `tests/models/test_model_info_registry.py:309` | exact |
| `tests/models/test_model_info_registry.py` (modify) | test | file-I/O assertion | itself (count pins, insertion-only check, provenance test) | exact |
| `tests/models/test_registry.py` (modify) | test | transform assertion | itself (sole-parser grep guard) | exact |
| `tests/utils/test_transformers_compat*.py` (new per rung + extend) | test | unit (contract tests over patched live classes) | `tests/utils/test_transformers_compat.py` | exact |
| `dnallm/utils/transformers_compat.py` (modify: new rungs) | utility (compat shim) | event-driven at import (probe → absence gate → sentinel attach) | itself; `_patch_mamba_cache` / `_patch_get_head_mask` shapes | exact |
| `.github/workflows/<census>.yml` (new) | config (CI workflow) | batch (dispatch-only re-verify, fail-closed diff) | `.github/workflows/feasibility.yml` | exact |
| `tests/<gate>` — registry-vs-docs drift gate test (new) | test | file-I/O consistency | `tests/mcp/test_config_registry_consistency.py` + `dnallm/mcp/consistency.py` | role-match |
| `docs/.../compatibility table page` (new, generated) | docs (generated static page) | transform (registry → markdown) | none exact (no existing generated-from-registry docs page) | none — see below |

## Pattern Assignments

### `scripts/census_compat.py` (script, batch)

**Analog:** `scripts/audit_skips.py` (whole file, 133 lines — copy this shape wholesale)

**Shape pattern** (whole file):
- Shebang + one-line module docstring stating the fail-closed contract
- `main(args) -> int` returning 0/1; `__main__` block parses argv, `sys.exit(main(...))`
- Fail closed on absent/unparseable inputs — an interrupted or tampered artifact must never read as "no drift" (`scripts/audit_skips.py:88-92`):
```python
except (OSError, ET.ParseError) as e:
    # Fail closed: an absent or unparseable junit (an interrupted or
    # tampered run) must never read as "no unexpected skips".
    print(f"ERROR: cannot parse junit artifact {junit_path}: {e}")
    return 1
```
- Trail of matched vs unexpected lines, `UNEXPECTED ...` list, exit 1 with per-item prints (`audit_skips.py:94-125`)
- Docstrings: Google style, `Args:`/`Returns:`/`Raises:` (see `load_allowlist`)

**Census-specific additions** (from 27-RESEARCH.md, not the analog):
- Code-only fetch: `snapshot_download(..., allow_patterns=["*.py", "config.json"], revision=<sha>)`, ModelScope-first, always `HF_ENDPOINT=https://hf-mirror.com`
- Subprocess-per-entry import probe: `import dnallm` FIRST inside the child (shim ordering guarantee), `python -I`-style, `traceback.format_exc()` → parse last import frame (RESEARCH Pattern 1 sketch)
- Record hub provenance `<hf|ms> <repo>@<sha>` per verdict (models.lock convention)

### `dnallm/models/registry.py` (model, transform)

**Analog:** itself — `RegistryEntry`, `dnallm/models/registry.py:92-114`

**Field-extension pattern** (lines 106-114) — new fields copy the REG-02 optional-nullable shape exactly (schema + data + tests in ONE commit; `extra="forbid"` makes split commits fatal — Pitfall 4):
```python
model_config = ConfigDict(frozen=True, extra="forbid")

name: str = Field(min_length=1)
model: str = Field(min_length=1)
task: TaskInfo
kind: Literal["pretrained", "finetuned"]
tokenizer_type: str | None = Field(default=None, min_length=1)
max_length: int | None = Field(default=None, gt=0)
base_model: str | None = Field(default=None, min_length=1)
# Phase 27 additions: same optional-nullable shape, e.g.
# compat_status: Literal[...] | None = Field(default=None, ...)
# compat_evidence / compat_tf_version: str | None
```
Update the class `Attributes:` docstring (Google style, lines 96-104) in the same edit. Extend `__all__` accessor exports only if new accessors are added. Keep `reload()` as the test hook.

### `tests/models/test_model_info_registry.py` (test, file-I/O assertion)

**Analog:** itself

- Count pins (lines 28, 203, 316-320):
```python
EXPECTED_PRETRAINED_COUNT = 49
...
def test_registry_section_counts_unchanged() -> None:
    """The enrichment is field-level insertion: counts stay 49 pretrained / 187 finetuned."""
```
- Enrichment coverage test pattern (lines 250-266): `test_all_pretrained_entries_carry_positive_int_max_length`, `test_lab_pretrained_entries_carry_tokenizer_type` — new compat fields get the same every-entry assertions
- Provenance blocks test (line 309 `test_provenance_blocks_present_verbatim`) — the reason write-back MUST be scripted text insertion, never `yaml.dump`
- Insertion-only / degenerate-value guard pattern (line 268 `test_third_party_enrichment_values_present_or_absent_never_degenerate`)

### `tests/utils/test_transformers_compat*.py` (test, unit)

**Analog:** `tests/utils/test_transformers_compat.py` (1,415 lines)

**File header + import pattern** (lines 1-45): `from __future__ import annotations`, import the real live transformers modules, import private `_patch_*`/`_impl` names directly from `dnallm.utils.transformers_compat`. Test classes `Test*` grouping one rung; descriptive behavior names (`test_attach_helper_sentinel_short_circuits_repeat_calls`).

**Per-rung test contract** (lines 118-157, 576-613) — every new rung needs at minimum:
- idempotency on the live class (`test_apply_patches_is_idempotent_on_live_class`, 121)
- sentinel short-circuit (`test_patch_functions_return_early_when_already_flagged`, 133)
- no-op when target absent (`test_patch_skips_when_target_method_absent`, 144 — uses `monkeypatch` + `Mock`/fake classes)
- attach-only-missing / never-overwrite (`test_attach_helper_attaches_only_missing_names_and_never_overwrites`, 576)
- vendored-impl behavior tests against upstream v4.49.0 semantics (see `TestRemoteCodePruningHelpers`, 431-613)
- TDD: the failing-import test lands BEFORE the fix rung (D-06 / established pattern)

### `dnallm/utils/transformers_compat.py` (utility, import-time patch)

**Analog:** itself — two canonical rung shapes (copy one, do not invent a third):

**Module-attach rung** — `_patch_mamba_cache` (lines 776-810): try-import gate → `hasattr` absence gate → sentinel check → `setattr(module, name, impl)` + `setattr(module, "_dnallm_<x>_patch", True)`.

**Class-attach rung** — `_patch_get_head_mask` (line 952) and hook-wrap rung `_patch_deberta_vocab_dict` (lines 813-865, wraps a classmethod and re-attaches via `classmethod(...)`).

**Vendoring convention** (lines 868-879): vendored code cites upstream tag verbatim — comment block naming `tag v4.49.0, file src/transformers/...`; semantics/docstrings kept verbatim; documented deviations listed.
**Registration** (lines 1384-1408): every new rung called from `apply_patches()` before the module-level `apply_patches()` invocation (line 1408).
Closed-map discipline: extend the per-name map — never a catch-all `__getattr__` (comment at line 517).

### `.github/workflows/<census>.yml` (config, batch)

**Analog:** `.github/workflows/feasibility.yml` (110 lines — copy posture wholesale)

**Dispatch-only security boundary** (lines 1-31):
```yaml
on: workflow_dispatch
permissions:
  contents: read
jobs:
  census-...:
    runs-on: [self-hosted, dnallm-nightly]
    if: github.event_name == 'workflow_dispatch'
    timeout-minutes: 240
```
Comment block at top of file explains WHY dispatch-only (PR-authored code must never run on the self-hosted box). Reuse: uv venv + `uv pip install -e ".[full]"` steps; GPU check step that FAILS loud on absence rather than silently skipping (lines 33-50, WR-05 pattern); `if: always()` artifact upload (lines 99-109). The census lane adds a final diff step: re-run census → compare vs committed `model_info.yaml` fields → disagreement exits non-zero (audit_skips fail-closed pattern).

### Registry-vs-docs drift gate test (new, test)

**Analog:** `tests/mcp/test_config_registry_consistency.py` (Phase 26 gate test) over `dnallm/mcp/consistency.py`

**Structure** (from file head): `from __future__ import annotations`, `pathlib.Path` on the committed data file, pytest parametrize over registry entries, imports of reason-code constants (`CONFIG_INVALID`, ...) from the checker module. `consistency.py` supplies the Finding/Severity/reason-code vocabulary (lines 26-133) and the strict-mode pass rule (`finding.severity == Severity.ERROR or (strict_registry and reason_code in STRICT_FATAL_REASON_CODES)`). The Phase 27 gate can be a lighter pytest-only shape: parse the generated docs page, re-derive the table from `dnallm.models.registry` accessors, assert equality — no second yaml parser (REG-01 sole-parser rule; grep guard in `tests/models/test_registry.py` fails CI on a second mentioner of `model_info.yaml`).

## Shared Patterns

### Fail-closed audits over committed claims
**Source:** `scripts/audit_skips.py:66-125`
**Apply to:** census drift gate (dispatch lane) and docs/registry gate test. Absent/unparseable input ⇒ exit 1, never "no drift found".

### Import-time shim rung (probe-gated, absence-gated, sentinel-idempotent, no-op on 4.x)
**Source:** `dnallm/utils/transformers_compat.py:776-810` (module attach) and `:952` (class attach)
**Apply to:** every census-driven fix (D-06 fix-first); registered in `apply_patches()` (`:1384-1408`); paired TDD test file.

### Registry enrichment (schema + data + tests in one commit)
**Source:** `dnallm/models/registry.py:92-114`, `tests/models/test_model_info_registry.py:250-322`
**Apply to:** compat field write-back; write via scripted text insertion only (provenance test `:309` fails on round-trip damage).

### Honest-evidence labeling
**Source:** CONTEXT D-04/D-05 + `feasibility.yml:33-50` (GPU-absent = loud failure, not silent green)
**Apply to:** census tool (unreachable repo = recorded loud marker, never silent skip), registry evidence fields, docs table bilingual labels.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| Generated plain-language docs compatibility table page | docs | transform | No existing MkDocs page is generated from the registry; all `docs/` pages are hand-written (no "GENERATED" markers found). Planner should define the generation script alongside `scripts/census_compat.py` (or a `--emit-docs` flag), mark the page with a regeneration header, and gate it with the consistency test above. Follow RESEARCH Pattern 4 + plain-language directive for biologists. |

## Metadata

**Analog search scope:** `scripts/`, `dnallm/utils/`, `dnallm/models/`, `dnallm/mcp/`, `tests/utils/`, `tests/models/`, `tests/mcp/`, `.github/workflows/`, `docs/`
**Files scanned:** ~14 (all analogs read this session; counts from targeted reads)
**Pattern extraction date:** 2026-10-27
