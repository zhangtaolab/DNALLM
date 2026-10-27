# Phase 27: Transformers 4/5 Model Compatibility Census - Research

**Researched:** 2026-10-27
**Domain:** transformers 4→5 remote-code compatibility; registry enrichment; census tooling; CI fail-closed gates
**Confidence:** HIGH (core breakage surface verified live against the installed transformers 5.17.0 this session)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01: Pretrained-only census.** Only the 49 `pretrained:` entries are censused. The 187 `finetuned:` entries are excluded — their compatibility is inherited from their base models. (Owner clarification at discussion open.)
- **D-02: Layered evidence — full import census + sampled loads.** All 49 entries receive import-level evidence (remote-code files fetched, import chains resolved under transformers 5.17.0). Real loads (config + tokenizer + weights) are performed for exactly two classes: (a) models that fail at import level — to verify shim fixes actually restore the load; (b) one representative per model family — to prove that passing imports also survives runtime.
- **D-03: Load-level pass bar = load + one short-sequence real forward.** Model construction and weight loading alone do not count; one short real forward must run. Rationale: historically real breakage surfaced inside forward (`get_extended_attention_mask`, `head_mask` class — v1.2 evidence), not at load.
- **D-04: Giants (evo-1, evo-2) excluded from load sampling.** Import-level evidence only; the matrix labels them honestly (import-verified only). They load through `dnallm/models/special/evo.py` (stripedhyena path), not the trust_remote_code surface.
- **D-05: Empirical census only under transformers 5.17.0; no separate 4.x environment.** (Owner: "就用 5.17 检测即可，如果不兼容 4, 自然会报错" — just detect under 5.17; 4-incompatibility will error naturally.) Reasoning: all 49 checkpoints are 4.x-era code; breakage only runs in one direction (transformers 5 removing 4.x APIs), so pass-under-5.17 implies 4-side compatibility. The matrix/docs must label this honestly: the 5 side reads "tested on transformers 5.17.0"; the 4 side reads "inferred (model code written in the transformers 4 era)" — never claim separate 4.x testing. Roadmap taxonomy (`both` / `4-only` / `5-only` / `remote-code-broken`) is retained; the `5-only` bucket is expected to stay empty. — **Reversibility:** reversible — a 4.x verification environment can be added later without undoing anything.
- **D-06: Census evidence first, per-failure solutions after.** No pre-committed alias-vs-vendor priority. Run the census, capture each failure's exact shape (which import, what cause), then choose the fix per failure. (Owner: "先查吧，查出来问题再看解决方案".) Owner expectation: the dominant class is class/function renames (e.g. `PreTrained*`/`Pretrained*` casing) — cheap to fix. Fix takes priority over skip (2026-10-02 owner precedent: "fix all non-gated census failures now"); only structurally unfixable failures go to the typed-skip + traceback path (locked in ROADMAP).
- **D-07: Census results become new fields on the `pretrained:` entries in `model_info.yaml`** — compatibility status + evidence + a transformers version stamp, following the REG-02 enrichment pattern (`tokenizer_type`/`max_length` precedents). `RegistryEntry` in `dnallm/models/registry.py` and the REG-03 integrity tests (`tests/models/test_registry.py`, `tests/models/test_model_info_registry.py`) extend in the same change. The docs table is generated from the registry in plain language for biologists, and a CI consistency check keeps the generated table and the registry from drifting (Phase 26 `dnallm/mcp/consistency.py` precedent), running on the standard lanes. — **Reversibility:** one-way — REG-03's field-level insertion-only check treats registry fields as add-only; removing fields after the census ships would violate that invariant and break any consumer that read them. Field naming and schema details are planner's to propose.
- **D-08: Census tool is committed and re-runnable via a dispatch-only lane on the self-hosted runner** (`.github/workflows/feasibility.yml` precedent). Re-run on transformers bumps (operationalizing the dependabot "bump must re-verify" note) or whenever needed. A re-run whose results disagree with the committed `model_info.yaml` fields turns the lane red — fail-closed, same pattern as `scripts/audit_skips.py`. No nightly residency. — **Reversibility:** reversible — a workflow file.

### Claude's Discretion
- Census tool implementation shape (script vs pytest-collected lane, artifact format for the recorded evidence)
- Exact registry field names/schema and the version-stamp representation
- Docs page placement and table layout (per-model rows per ROADMAP deliverable)
- Which entry represents each model family in load sampling
- Alias-to-5.x vs vendor-4.49.0 choice per individual failure, decided on census evidence per D-06
- How the ROADMAP "238" count gets corrected (via the proper roadmap update path during planning)

### Deferred Ideas (OUT OF SCOPE)
- **Load-time compat notice** (noted by Claude during discussion, not owner-raised): once the registry carries per-model compat fields, `load_model_and_tokenizer` could consult them and warn/raise with the exact failing import when a user requests a 4-only model under transformers 5. New runtime capability — belongs in its own future phase, not this census.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TBD (owner-directed 2026-10-27) | Per-model transformers 4/5 compatibility matrix over the 49 `pretrained:` entries with import-level evidence; taxonomy `both`/`4-only`/`5-only`/`remote-code-broken`; disposition via `transformers_compat.py` shim (fix-first) or typed-skip + traceback; results written back to `model_info.yaml`; plain-language docs table; CI drift gate; dispatch-only re-verification lane | Verified census target inventory (49 entries, 27 remote-code / 21 native, 48/49 cached); verified removed-API surface at transformers 5.17.0 (import-level + method-level + config-default); verified fetch mechanism (`snapshot_download` allow_patterns on both hubs); verified extension points (RegistryEntry schema, REG-03 tests, shim rung pattern, consistency-gate precedent, dispatch-lane precedent) |

No REQUIREMENTS.md exists for this project; the phase is owner-directed. Requirements traceability rides on the CONTEXT.md decisions above.
</phase_requirements>

## Summary

The census target is exactly 49 `pretrained:` entries (verified by section count this session: `grep -c` over `dnallm/models/model_info.yaml` returns 49 pretrained + 187 finetuned = 236; no evo entries exist anywhere in the yaml) [VERIFIED: dnallm/models/model_info.yaml sections, counted 2026-10-27]. Of the 49, **27 are `trust_remote_code` models (config.json carries a non-empty `auto_map`)** and 21 load through native transformers architectures — measured by reading every cached config.json on this box (48/49 cached; only `GenerTeam/GENERanno-eukaryote-1.2b-base` absent locally) [VERIFIED: local cache probe, /tmp/p27-census-probe/probe_remote_imports.py]. The breakage surface therefore concentrates in those 27 remote-code repos whose `modeling_*.py` import transformers internals that transformers 5 removed.

The transformers 5.17.0 removal surface was probed live (fresh interpreter, `-I`, **without** the dnallm shim active, so results reflect raw transformers): module-level names gone from `modeling_utils` (`find_pruneable_heads_and_indices`, `prune_linear_layer`, `SequenceSummary`, `PoolerEndLogits`, `PoolerStartLogits`, `top_k_top_p_filtering`, `PruneConv1D`, `conv1d`, `apply_chunking_to_fn`, `_tie_encoder_decoder_weights`), from `cache_utils` (`MambaCache`, `from_legacy_cache`, `to_legacy_cache`), from `utils.import_utils` (`is_torch_fx_available` — a documented v5.0.0 removal that breaks remote-code models upstream, HF issue #44561 closed not_planned), and from `modeling_rope_utils` (`compute_default_rope_parameters`; `ROPE_INIT_FUNCTIONS` no longer has a `"default"` key — verified keys are `['dynamic', 'linear', 'llama3', 'longrope', 'proportional', 'yarn']`); `PreTrainedModel` methods gone at runtime (`get_extended_attention_mask`, `get_head_mask`, `invert_attention_mask`, `create_extended_attention_mask_for_decoder`, `prune_heads`); config-instance defaults gone (`is_decoder`, `add_cross_attention`, `pad_token_id` on a fresh `PretrainedConfig()`); and `tie_weights` gained the `(missing_keys, recompute_mapping=True)` protocol [VERIFIED: direct probe of installed transformers 5.17.0, /tmp/p27-census-probe]. Eleven of these classes are already bridged by the existing rungs in `dnallm/utils/transformers_compat.py` [VERIFIED: dnallm/utils/transformers_compat.py:1-1408, read this session]; `invert_attention_mask` (a live call in the cached `modeling_esm.py` of `zhangtaolab/plant-nucleotide-transformer-BPE`), `prune_heads`, `is_torch_fx_available`, `from_legacy_cache`/`to_legacy_cache`, and the RoPE-init family are **not yet shimmed** and are the expected census finds.

Mechanically, only 3 of the 27 remote repos have their remote `.py` files cached on this box (PlantCAD2, FungiHelixSeek, plant-nucleotide-transformer-BPE), so the census must fetch code-only artifacts — both `huggingface_hub.snapshot_download` (1.33.0, has `allow_patterns` and `dry_run`) and `modelscope.hub.snapshot_download` (1.34.0, has `allow_file_pattern`/`allow_patterns`) support fetching `*.py` + `config.json` without weights [VERIFIED: signature inspection of installed packages]. Raw transformers 5.17.0 still executes the trust_remote_code config path (`AutoConfig.from_pretrained(..., trust_remote_code=True)` on the cached plant-NT repo returns `EsmConfig` without any dnallm shim) [VERIFIED: live probe]. The registry, gate, and lane extension points are all verified open: `RegistryEntry` is `frozen=True, extra="forbid"` with optional nullable fields as the REG-02 precedent; the Phase 26 consistency checker supplies the reason-code + tiered-findings + strict-mode gate structure; `feasibility.yml` supplies the dispatch-only self-hosted lane pattern with `if: github.event_name == 'workflow_dispatch'` as the security boundary.

**Primary recommendation:** Build the census as a committed Python tool (script or script+pytest lane) that, per entry: fetches `config.json` + `*.py` (allow_patterns, ModelScope-first with hf-mirror fallback), classifies native-vs-remote via `auto_map`, subprocess-imports each remote `auto_map` module under transformers 5.17.0 with the dnallm shim active (capturing the exact failing import line and exception), then runs D-02b/D-03 load+forward for family representatives and shim verifications. Results land as add-only `RegistryEntry` fields, a generated plain-language docs table, a Phase-26-style CI consistency gate, and a `feasibility.yml`-style dispatch lane whose re-run diffs against the committed fields fail-closed.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Census execution (fetch + import-resolve + sampled loads) | Offline tooling (committed script/lane, dev box + self-hosted runner) | CI (dispatch-only re-verification) | Evidence production is a batch job needing warm caches and GPU for forwards; CI only re-verifies drift |
| Compatibility verdict storage | Data (registry `model_info.yaml` + typed `RegistryEntry`) | — | D-07 locks registry-as-truth; REG-02 enrichment precedent; add-only per REG-03 |
| Import-chain resolution | Census tool subprocess (transformers 5.17.0 + dnallm shim) | — | Must run in isolated subprocesses: remote code is untrusted, and import failures must not poison the parent process |
| Shim fixes for removable-API breakage | Library (`dnallm/utils/transformers_compat.py` rungs) | — | D-06 fix-first; the 11-rung probe/absence/sentinel pattern is the established mechanism |
| Unfixable-breakage labeling | Test infra (typed-skip + traceback annotation path) | Registry fields (status label) | ROADMAP-locked disposition for structurally unfixable failures |
| Docs table | Static site (MkDocs, generated file) | CI consistency gate | Plain-language for biologists (owner directive 2026-10-21); generated-from-registry with drift gate |
| Drift gate | CI standard lanes (pytest-collected gate test) | — | Phase 26 consistency.py precedent runs on testpaths, not a separate workflow |
| Re-verification lane | CI dispatch-only workflow on self-hosted runner | — | D-8; feasibility.yml precedent; never on PR/push events |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| transformers | 5.17.0 (installed; declared `>=4.49,<6`) | The compatibility SUBJECT under test; its removed-API surface is the breakage map | Already the project's model layer; census runs against the installed version only (D-05) |
| huggingface-hub | 1.33.0 (installed) | Code-only artifact fetch: `snapshot_download(repo_id, allow_patterns=["*.py","config.json"])`; also `dry_run` for listing | Verified signature this session; hf-mirror reachable via `HF_ENDPOINT` |
| modelscope | 1.34.0 (installed) | Code-only fetch from ModelScope: `snapshot_download(model_id, allow_file_pattern=[...])` | Verified signature this session; ModelScope-first owner directive |
| pydantic | v2 (installed) | `RegistryEntry` field extension; frozen `extra="forbid"` schema | Existing registry loader is the sole parser |
| PyYAML (CSafeLoader) | existing | Registry parse (unchanged); census artifact serialization | Registry already parses via `importlib.resources` + `yaml.load(CSafeLoader)` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | existing (>=8.3.5) | Census lane collection (if pytest-shaped); gate tests for REG-03 extension and docs-table drift | Discretion: script vs pytest lane |
| torch | 2.11.0+cu130 (installed) | D-03 short-sequence forward for sampled loads | Family representatives + shim-verification loads only |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Subprocess import checks | In-process `importlib` in a pytest lane | Subprocesses isolate untrusted remote code and import-order pollution; in-process is faster but one poisoned `sys.modules` (compat shims mutate transformers classes globally) invalidates subsequent probes — subprocess is the honest shape |
| Static AST import scan | Dynamic subprocess import | Static scan cannot see attribute-level/method failures (D-03 evidence: breakage surfaced inside forward, not at import) — use dynamic as truth, static as a fast pre-pass only |
| A second 4.x venv (uv) | — | Rejected by D-05 (owner): single-environment census; 4-side is inferred, honestly labeled |

**Installation:** none — this phase installs no new packages. (Verified: every capability above is already a declared dependency in `pyproject.toml`.)

**Version verification:** performed against the project venv (`.venv`): transformers 5.17.0, huggingface-hub 1.33.0, modelscope 1.34.0 — all confirmed live this session via `python -I` imports and `inspect.signature`.

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| (none) | — | — | — | — | — | Phase installs zero new packages; all tooling builds on existing declared deps |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                      ┌────────────────────────────────────────────────────────┐
                      │  registry truth: dnallm/models/model_info.yaml (49)    │
                      └───────────▲────────────────────────────▲───────────────┘
                                  │ write-back (add-only       │ read (sole parser)
                                  │ compat fields, D-07)       │
                                  │                            │
 ┌──────────────────────┐   ┌─────┴──────────────┐   ┌─────────┴─────────────┐
 │ CENSUS TOOL (commit) │──▶│ evidence artifact   │──▶│ RegistryEntry schema │
 │ per entry:           │   │ (per-model record:  │   │ + REG-03 tests       │
 │  1. read registry    │   │  status, failing    │   │ extend (49/187 pins, │
 │  2. fetch *.py only  │   │  import, versions,  │   │ insertion-only)      │
 │     (ms first,       │   │  revision sha)      │   └─────────┬─────────────┘
 │     hf-mirror 2nd)   │   └─────┬───────────────┘             │
 │  3. auto_map?        │         │ drives                      ▼
 │     no → native:     │         │                   ┌──────────────────────┐
 │        arch resolves │         │                   │ generated docs table │
 │        in 5.17?      │         │                   │ (plain language)     │
 │     yes → remote:    │         └──────────────────▶│ + CI drift gate      │
 │        subprocess    │                             │ (consistency.py      │
 │        import under  │   ┌──────────────────┐      │  precedent)          │
 │        5.17 + shim   │──▶│ disposition      │      └──────────────────────┘
 │  4. loads: shim-     │   │ fix-first (D-06):│
 │     verify + family  │   │  rename-class →  │
 │     reps, load +     │   │  new compat rung │
 │     forward (D-03)   │   │  + TDD test      │
 └──────────┬───────────┘   │ else typed-skip  │
            │               │  + traceback     │
            │ re-run        └──────────────────┘
            ▼ (dispatch-only, self-hosted runner)
 ┌────────────────────────────┐
 │ .github/workflows lane     │
 │ on: workflow_dispatch      │
 │ re-run diffs vs committed  │
 │ fields → disagree = RED    │
 │ (fail-closed, audit_skips  │
 │  pattern)                  │
 └────────────────────────────┘
```

Reader trace: registry feeds the census; census produces evidence; evidence drives per-failure disposition (shim or skip); results write back to registry add-only; registry generates docs table with drift gate; the dispatch lane re-runs the census and fails red when the world drifts from the committed fields.

### Recommended Project Structure
```
scripts/
├── census_compat.py            # census tool (discretion: script shape) — fetch, import-resolve, load-sample, emit evidence
docs/
└── user_guide/ (or resources/) # generated compatibility table page (placement is planner discretion)
dnallm/models/
├── model_info.yaml             # + compat fields on the 49 pretrained entries (scripted text insertion, never YAML round-trip)
└── registry.py                 # RegistryEntry + new optional fields
tests/models/
├── test_registry.py            # REG-03 extension
└── test_model_info_registry.py # count pins (49/187) + insertion-only check extension
tests/utils/
└── test_transformers_compat*.py  # TDD-per-rung for every new shim rung
.github/workflows/
└── census-compat.yml (name TDB) # dispatch-only re-verification lane
```

### Pattern 1: Census import probe (subprocess-isolated, shim-active)
**What:** Per remote-code entry, fetch code files, then import each `auto_map` module in a `subprocess` running `python -I`-style isolation WITH `import dnallm` first (shim armed), capturing stderr/stdout and the exact failing line.
**When to use:** every one of the 27 remote entries.
**Example:**
```python
# Sketch (shape only — planner owns details). Source: local probe validated this session
# in /tmp/p27-census-probe; subprocess isolation required because compat shims mutate
# transformers classes process-wide and one import failure poisons sys.modules.
import subprocess, sys, json
PROBE = """
import traceback
import dnallm  # arms transformers_compat.apply_patches() at import
from dnallm.utils import transformers_compat  # explicit, mirrors consumer ordering
code = {entry_file!r}
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location({module_name!r}, code)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(json.dumps({{"ok": True}}))
except Exception:
    tb = traceback.format_exc()
    print(json.dumps({{"ok": False, "traceback": tb}}))
"""
result = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=120)
# Record: status, the last import line of the traceback (the "exact failing import"),
# transformers.__version__, and the fetched revision sha.
```

### Pattern 2: Code-only fetch (both hubs, revision-pinned)
**What:** `snapshot_download` restricted to `*.py` + `config.json` (+ `tokenizer.json`/`spm` files only when the tokenizer is remote) — never weights.
**When to use:** the ~24 remote repos whose `.py` files are not already cached.
**Example:**
```python
# Source: signatures verified live this session (huggingface-hub 1.33.0, modelscope 1.34.0)
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="LongSafari/hyenadna-small-32k-seqlen-hf",
    allow_patterns=["*.py", "config.json"],   # code-only, no weights
    revision=<pinned-sha>,                     # models.lock provenance pattern
    endpoint="https://hf-mirror.com",          # runner constraint (huggingface.co unreachable there)
)
# ModelScope equivalent: modelscope.hub.snapshot_download(model_id=..., allow_file_pattern=["*.py", "config.json"])
```

### Pattern 3: Registry write-back is scripted text insertion
**What:** New fields land in `model_info.yaml` by scripted insertion (per the `261020-kwt` discipline cited in registry.py's docstring), not a YAML round-trip; `RegistryEntry` gains optional nullable fields (`extra="forbid"` requires schema+data in the same change).
**When to use:** the D-07 write-back commit.
**Example:**
```python
# Source: registry.py REG-02 precedent (tokenizer_type/max_length), read this session
class RegistryEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    # ... existing fields verbatim ...
    tokenizer_type: str | None = Field(default=None, min_length=1)
    max_length: int | None = Field(default=None, gt=0)
    base_model: str | None = Field(default=None, min_length=1)
    # Phase 27 additions follow the same optional-nullable shape (names = planner discretion):
    # compat_status: Literal[...] | None / compat_evidence: str | None / compat_tf_version: str | None
```

### Pattern 4: Fail-closed drift gate (two layers)
**What:** (a) a pytest-collected gate on standard lanes asserting generated-docs-table == registry-derived truth (Phase 26 consistency structure: reason codes + strict signal); (b) the dispatch lane re-runs the census and diffs results against committed `model_info.yaml` fields — disagreement = non-zero exit (audit_skips pattern).
**When to use:** D-07 (docs/registry drift) and D-08 (world/registry drift).
**Example:**
```yaml
# Source: .github/workflows/feasibility.yml (dispatch-only posture, read this session)
on: workflow_dispatch            # NEVER push/pull_request/schedule: PR code must not run on the self-hosted box
permissions:
  contents: read                 # least privilege
jobs:
  census-compat:
    runs-on: [self-hosted, dnallm-nightly]
    if: github.event_name == 'workflow_dispatch'   # the security boundary
```

### Pattern 5: New shim rung (probe-gated, absence-gated, idempotent, no-op on 4.x)
**What:** Every census-driven fix extends `transformers_compat.py` with a rung following the exact 11-rung pattern: try-import gate → hasattr/absence gate per name → sentinel-guarded attach → registered in `apply_patches()`, plus a failing-test-first test file.
**When to use:** every fixable census failure (D-06 fix-first).
**Example:** the existing `_patch_mamba_cache` (module attach) and `_patch_get_head_mask` (class attach) are the two canonical shapes — copy one, do not invent a third mechanism. Vendored implementations must cite upstream tag v4.49.0 and keep semantics verbatim (file convention in `transformers_compat.py:85-97`).

### Anti-Patterns to Avoid
- **YAML round-trip write-back:** the registry is maintained by scripted text insertion; a `yaml.dump` would destroy comments, ordering, and the provenance blocks pinned by `test_provenance_blocks_present_verbatim` [VERIFIED: tests/models/test_model_info_registry.py:309].
- **In-process import census:** compat rungs mutate transformers classes process-wide; importing 27 repos' code in one process cross-contaminates probes. One subprocess per entry.
- **Assuming the owner's "casing mismatch" lead instead of measuring it:** probed this session — `PreTrainedModel`/`PretrainedConfig` (correct casing) all resolve at 5.17; `PretrainedModel` (wrong casing) is gone but would fail on 4.x too; BOTH `PretrainedConfig` and `PreTrainedConfig` casings resolve in `configuration_utils` at 5.17 [VERIFIED: live probe]. The empirical lead class is removed-name imports (pruning/cache/utils), not casing. Record evidence, then decide (D-06).
- **Fetching weights during the import census:** `allow_patterns=["*.py","config.json"]` keeps each repo's fetch at kilobytes; a stray full download for a 1B-param repo wastes gigabytes on 27 repos.
- **Green-on-silent-skip census:** the feasibility.yml GPU-absent precedent — a census that silently skips unreachable repos greens with zero evidence. Unreachable = loud failure with a recorded marker, not a skip.
- **Trust the first cache hit:** the SAME repo id can differ between HF and ModelScope (e.g. OmniNA's cached configs differ in `auto_map` between caches — observed this session). The census must record WHICH hub each verdict came from, per the models.lock `<hf|ms>` convention.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Hub artifact fetch + cache + resume | requests/HTTP downloads | `huggingface_hub.snapshot_download` / `modelscope.snapshot_download` | Both verified to support code-only patterns; they handle resume, etag, revision pinning |
| Import-chain failure capture | regex over stderr | `traceback.format_exc()` in the subprocess + parse the last `import` frame | Gives the exact failing import line the taxonomy requires |
| Registry parse/validation | ad-hoc yaml reads | `dnallm/models/registry.py` accessors (REG-01 sole-parser + grep guard) | A second mentioner fails CI (sole-mentioner allowlist pinned to registry.py and mcp/server.py) |
| Config-vs-docs drift detection | bespoke differ | Phase 26 `consistency.py` structure (reason codes, tiered findings, strict pass signal, frozen report) | Reuses the established Finding/Severity vocabulary and CI placement pattern |
| Re-verification trigger | nightly cron | dispatch-only workflow (D-8) | Owner decision: no nightly residency; PR-authored code must never run on the self-hosted box |
| 4.x API restoration | catch-all `__getattr__` on transformers modules | per-name absence-gated rungs | The closed-map discipline (comment at transformers_compat.py:517: "extend the closed map — never a catch-all") |

**Key insight:** every mechanism this phase needs — shim rungs, registry enrichment, integrity tests, fail-closed audits, dispatch lanes, revision pinning — already exists in the repo with tests. The census is a new ORCHESTRATION over proven parts, not a new mechanism.

## Common Pitfalls

### Pitfall 1: Import-pass does not mean load-pass (or forward-pass)
**What goes wrong:** a remote module imports cleanly under the shim but dies inside `from_pretrained` (meta-device build, `post_init` bookkeeping) or inside `forward` (`get_extended_attention_mask` was exactly this class — v1.2 evidence, D-03's rationale).
**Why it happens:** method-level removals and signature changes surface only at call time.
**How to avoid:** keep the three evidence layers strictly separate in the registry fields (import / load / forward) — D-03 makes load+forward the pass bar for sampled models.
**Warning signs:** a compat field that only records "imported OK".

### Pitfall 2: HF vs ModelScope repo divergence
**What goes wrong:** the census fetches repo X from ModelScope (per ModelScope-first) but the user later loads from HF (or vice versa) where `auto_map` or the remote code revision differs — verdict doesn't transfer.
**Why it happens:** mirrors are not snapshots of each other; observed live this session (OmniNA cached configs differ across the two caches).
**How to avoid:** record source hub + revision sha in the evidence field (models.lock `<hf|ms> <repo>@<sha>` convention); docs table labels the tested source.
**Warning signs:** census evidence lacking a hub/revision stamp.

### Pitfall 3: The shim ordering guarantee during census probes
**What goes wrong:** importing remote code before `dnallm` arms the shim produces false failures (or, if a probe imports dnallm second, ordering-dependent verdicts).
**Why it happens:** `apply_patches()` runs at `import dnallm` via the lazy meta-path activation; remote code imported first resolves raw transformers.
**How to avoid:** census subprocesses import `dnallm` FIRST, mirroring the consumer ordering guarantee (CLAUDE.md Architectural Constraints).
**Warning signs:** census code that calls `importlib` on a modeling file before any dnallm import.

### Pitfall 4: `extra="forbid"` schema/data atomicity
**What goes wrong:** adding fields to `model_info.yaml` without the matching `RegistryEntry` fields (or vice versa) makes EVERY registry access raise `ValueError` at first access — the whole package goes down, not just the census.
**Why it happens:** hard validation at first access (D-01b), frozen schema.
**How to avoid:** schema + data + REG-03 test extension land in one commit; run the registry test file immediately after.
**Warning signs:** any plan task that edits the yaml in a different commit than registry.py.

### Pitfall 5: ProkBERT's cross-repo auto_map
**What goes wrong:** `neuralbioinfo/prokbert-mini`'s cached config carries `auto_map` values of the shape `neuralbioinfo/nbrg-transformers--models.ProkBertConfig` — a reference to code in ANOTHER repo, not a local file [VERIFIED: cached config.json read this session].
**Why it happens:** the prokbert repo outsources its model code to a sibling repo.
**How to avoid:** the census importer must resolve `--`-style cross-repo auto_map paths (fetch the referenced repo too) or record the entry as a distinct evidence class rather than crashing.
**Warning signs:** census assumes every auto_map value is `<file>.<class>` local.

### Pitfall 6: Runner network topology differs from the dev box
**What goes wrong:** census runs green on the dev box (huggingface.co reachable directly — verified HTTP 200 this session) then fails on the runner, where huggingface.co is unreachable and only hf-mirror + ModelScope work.
**Why it happens:** documented runner constraint (CONTEXT: hf-mirror only, ModelScope works).
**How to avoid:** the census tool hard-codes the ModelScope-first + `HF_ENDPOINT=https://hf-mirror.com` posture everywhere (the `model.py:505-509` toggle exists but the census should set it explicitly).
**Warning signs:** any `huggingface.co` URL or default-endpoint hub call in census code.

### Pitfall 7: Roadmap count correction via the wrong path
**What goes wrong:** hand-editing the "238" in ROADMAP.md prose creates drift with the canonical registry count (236) and bypasses the roadmap-update discipline.
**Why it happens:** the ROADMAP text predates the registry count verification.
**How to avoid:** CONTEXT D-01/`<domain>` directs correction "via the proper roadmap update path during planning" (the `/gsd-phase` roadmap CRUD surface); verified count is 49 + 187 = 236 [VERIFIED: section counts this session].
**Warning signs:** a plan task editing ROADMAP.md with a text tool instead of the roadmap command.

### Pitfall 8: `5-only` bucket must stay honestly empty, not silently empty
**What goes wrong:** recording `both` for every pass implies a 4.x test that never happened (D-05 forbids the claim).
**Why it happens:** taxonomy pressure — `both` looks like the good answer.
**How to avoid:** the docs/registry labels carry the honest provenance: 5 side "tested on transformers 5.17.0", 4 side "inferred (model code written in the transformers 4 era)" — the exact wording owner-approved in CONTEXT `<specifics>` (Chinese originals suggested there).
**Warning signs:** any English-only or unqualified "compatible with 4 and 5" cell.

## Code Examples

### Verified: cached remote-code import shape (Plant NT, the known-good-with-shim case)
```python
# Source: ~/.cache/modelscope/hub/models/zhangtaolab/plant-nucleotide-transformer-BPE/modeling_esm.py
# (read this session; the only repo family with .py already cached on the dev box)
from transformers.file_utils import (
    add_code_sample_docstrings,        # OK at 5.17 (verified live)
    add_start_docstrings,              # OK at 5.17
    add_start_docstrings_to_model_forward,  # OK at 5.17
)
from transformers.modeling_utils import (
    PreTrainedModel,                   # OK at 5.17
    find_pruneable_heads_and_indices,  # GONE at 5.17 raw — shimmed (rung 4)
    prune_linear_layer,                # GONE from modeling_utils at 5.17 raw — shimmed (rung 4)
)
# plus, inside forward: self.get_extended_attention_mask(...), self.get_head_mask(...)
# (both shimmed) AND self.invert_attention_mask(...) — NOT yet shimmed: census candidate.
```

### Verified: the removed-API probe (the census's breakage oracle)
```python
# Source: executed this session against .venv transformers 5.17.0 (python -I, no dnallm shim)
GONE_import_level = {
    "transformers.modeling_utils": ["find_pruneable_heads_and_indices", "prune_linear_layer",
        "SequenceSummary", "PoolerEndLogits", "PoolerStartLogits", "top_k_top_p_filtering",
        "PruneConv1D", "conv1d", "apply_chunking_to_fn", "_tie_encoder_decoder_weights"],
    "transformers.pytorch_utils": ["find_pruneable_heads_and_indices", "PruneConv1D", "conv1d"],
    "transformers.cache_utils": ["MambaCache", "from_legacy_cache", "to_legacy_cache"],
    "transformers.modeling_rope_utils": ["compute_default_rope_parameters"],
    "transformers.utils.import_utils": ["is_torch_fx_available"],
    "transformers.utils": ["is_flash_attn_available"],
}
GONE_methods_on_PreTrainedModel = ["get_extended_attention_mask", "get_head_mask",
    "invert_attention_mask", "create_extended_attention_mask_for_decoder", "prune_heads"]
GONE_config_defaults = ["is_decoder", "add_cross_attention", "pad_token_id"]
# tie_weights at 5.17: (self, missing_keys: set[str] | None = None, recompute_mapping: bool = True)
# ROPE_INIT_FUNCTIONS keys at 5.17: ['dynamic', 'linear', 'llama3', 'longrope', 'proportional', 'yarn']
# Still-present (do NOT pre-fail these): ACT2FN, ClassInstantier, _prepare_4d_attention_mask
# (and _create_4d_causal_attention_mask, _prepare_4d_attention_mask_for_sdpa),
# DataCollatorWithPadding, PreTrainedTokenizerBase, PreTrainedTokenizer,
# PreTrainedConfig AND PretrainedConfig (both casings), PreTrainedModel, file_utils
# doc-decorators, transformers.data.metrics.glue_compute_metrics.
```

### Verified: registry count + remote/native split (the census roster)
```python
# Source: measured this session over dnallm/models/model_info.yaml + local caches
# 49 pretrained / 187 finetuned (236 total; ROADMAP's "238" is stale — correct via roadmap path)
REMOTE_CODE_27 = [  # auto_map non-empty in cached config.json (27 of 48 cached; 1 uncached)
    # zhangtaolab: plant-nucleotide-transformer-BPE/-6mer/-singlebase, plant-dnamamba-BPE
    #   + 7 dnamamba k-mer/singlebase variants, PlantHelixSeek, FungiHelixSeek
    # third-party: caduceus-ph, caduceus-ps, PlantCaduceus_l32, PlantCAD2-Small-l24-d0768,
    #   DNA_bert_6, DNABERT-2-117M, gena-lm-bert-base, gena-lm-bigbird-base-sparse,
    #   GENERator-eukaryote-1.2b-base, GENERanno (UNCACHED — verdict pending fetch),
    #   hyenadna-small-32k-seqlen-hf, LucaOne-default-step36M, MutBERT-Human-Ref,
    #   nucleotide-transformer-v2-100m-multi-species, Omni-DNA-116M, prokbert-mini,
    #   PlantBiMoE
]
NATIVE_21 = [  # no auto_map: resolve via transformers' own classes (already suite-covered)
    # plant-dnabert x3, plant-dnagemma x3, plant-dnagpt x3, plant-dnamodernbert x2,
    # agro-nucleotide-transformer-1b, GenomeOcean-100M, gpn-brassicales, GROVER,
    # Jamba-DNA-v1-114M-hg38, Mistral-DNA-v1-138M-hg38, ModernBERT-DNA-v1-37M-hg38,
    # OmniNA-66m, CrossDNA_pretrain
]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Generic upstream tolerance for 4.x remote code under transformers 5 | Upstream declined compat shims (`is_torch_fx_available` restore PR #44615 rejected; issue #44561 not_planned); downstream loaders shim per-name (vLLM PR #55909 restores the symbol itself) | transformers v5.0.0 era | Confirms dnallm's per-rung shim strategy is the ecosystem-standard answer, not a wart |
| `PreTrainedTokenizer` slow/fast dual files | `PythonBackend` consolidation; one `tokenization_<model>.py`; `SpecialTokensMixin` merged; `build_inputs_with_special_tokens`/`prepare_for_model` removed from base | v5 | Remote TOKENIZERS are a second census surface (LucaOne ships `tokenization_lucaone.py`) — import-check them too, not just modeling files |
| Head masking + head pruning in core | Removed entirely from modeling_utils/pytorch_utils | v5 | Already rung-shimmed; `prune_heads` method also gone (not yet shimmed) |
| Legacy cache API (`from_legacy_cache`/`to_legacy_cache`) | CacheUtils protocol only | v5 | Remote mamba-style code constructing legacy caches breaks — census candidate class |
| `config.rope_theta` attribute | `config.rope_parameters` dict; direct read may AttributeError | v5 (#39847/#42255) | RoPE-based remote models (MutBERT is "RoPEBert") are a candidate class |

**Deprecated/outdated:**
- Roadmap's "238 models" count — actual registry is 49+187=236 [VERIFIED: counted this session]; CONTEXT directs correcting it via the proper roadmap update path.
- ROADMAP's owner lead "failures likely concentrate in `PreTrained*`/`Pretrained*` casing mismatches and dataloader-adjacent imports" — probed this session: casing is NOT the failure class at 5.17 (both `PretrainedConfig` casings resolve; wrong-casing `PretrainedModel` failed on 4.x too), and dataloader-adjacent modules survive (`data.data_collator.DataCollatorWithPadding` OK; only `data.datasets.GlueDatasetV2`-era names are gone). The real concentration is removed-name imports from `modeling_utils`/`cache_utils`/`utils.import_utils` [VERIFIED: live probe]. The census must still record evidence per failure, not assume either lead.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The uncached `GenerTeam/GENERanno-eukaryote-1.2b-base` mirrors sibling GENERator's remote-code shape | Census roster | Minor — census fetches it; verdict may differ from sibling |
| A2 | Native-architecture entries (21) pass under 5.17 because transformers' own implementations are suite-covered | Census roster | Low — transformers' own classes are the most-tested path; but a native arch REMOVED from transformers 5 (if any of the 21 architectures were dropped) would need a census check; the roster check "architectures resolves in installed transformers" covers it |
| A3 | ProkBERT's cross-repo auto_map (`nbrg-transformers--models`) resolves via transformers' dynamic-module machinery after fetching the sibling repo | Pitfall 5 | Medium — if unresolvable, entry lands in a distinct evidence class; planner should treat as census edge case, not crash |
| A4 | The dispatch lane can reuse the runner's warm caches (hub caches warm; giants at ~/models-giants) so load sampling stays within lane timeout | Environment | Low — 240min feasibility precedent; sampled loads are small-family representatives |
| A5 | Subprocess-per-entry census completes in acceptable wall time (27 subprocess imports + ~20 sampled loads) | Pattern 1 | Low — each subprocess is seconds; loads dominated by weight fetches already cached |
| A6 | evo-1/evo-2 (D-04) are NOT among the 49 registry entries — verified: zero evo rows in model_info.yaml — so D-04 applies only if the planner opts to add giant rows beyond registry scope | Open Questions | Planning-shaped — planner must reconcile D-04's text with the verified roster |
| A7 | New registry field names (`compat_status`-family) — schema is planner discretion per CONTEXT; no name is pre-locked | Pattern 3 | None — discretion delegated |
| A8 | `transformers>=4.49,<6` open range means future 5.x bumps (e.g. 5.19+ device-type query rung precedent) can shift verdicts; the version stamp field mitigates | D-08 | Low — drift gate + dependabot note already encode the re-verify duty |

## Open Questions

1. **D-04 giants vs the verified roster**
   - What we know: evo-1/evo-2 appear nowhere in `model_info.yaml` (verified: zero case-insensitive matches); the 49-entry census contains no evo rows. D-04 speaks as if they are censused ("import-verified only" labels).
   - What's unclear: whether the owner wants evo-1/evo-2 appended as extra rows (they load via `special/evo.py`, outside the trust_remote_code surface and outside the registry), or D-04 was scoped to the broader pre-discussion framing.
   - Recommendation: planner treats the 49 registry entries as the census universe (D-01 is explicit and later) and records D-04 as "no-op for this roster; giants clause applies only if extra rows are added". Flag to owner at plan review if evo rows are wanted.

2. **Census tool shape: standalone script vs pytest-collected lane**
   - What we know: CONTEXT leaves it to discretion; both precedents exist (`scripts/audit_skips.py` standalone; example-nightly stage-0.5 pytest collection assertion inside a workflow).
   - What's unclear: whether the import census should be pytest-collected (giving per-entry test IDs and skip machinery) or a script emitting a JSON artifact consumed by the write-back step.
   - Recommendation: script for the fetch+probe+census core (subprocess isolation is awkward inside pytest; artifacts feed the write-back), plus a small pytest gate test that validates the COMMITTED registry fields/artifact for internal consistency on standard lanes (offline, no network). This matches D-08's two layers exactly.

3. **`5-only` bucket semantics under D-05 single-environment evidence**
   - What we know: D-05 retains the roadmap taxonomy but forbids claiming 4.x testing; `5-only` expected empty.
   - What's unclear: the exact label a "passes 5.17 import+sampled-load" entry gets in the registry field (`both`? `tf5-tested`?) so the docs table can carry the honest two-sided labeling.
   - Recommendation: registry status vocabulary should be evidence-descriptive (e.g. `verified-tf5` / `shim-required-tf5` / `remote-code-broken`) and the DOCS table maps it to the taxonomy with the honest inferred/tested footnote — keeps the registry free of the 4-side inference while the docs carry the owner-approved bilingual labels.

4. **Where sampled-load failures that are NOT import-level failures get dispositioned**
   - What we know: D-02b samples one representative per family; a family rep could fail at load/forward (class the import census never sees).
   - What's unclear: does that failure shim-fix the family entry, or mark just the representative?
   - Recommendation: same census-first discipline (D-06) — capture evidence, fix if fixable, and record per-entry (the representative's verdict + evidence) so family siblings carry an honest "inferred from family representative" note.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| transformers (installed venv) | census subject | ✓ | 5.17.0 | — |
| huggingface-hub | code-only fetch | ✓ | 1.33.0 | — |
| modelscope | ModelScope-first fetch | ✓ | 1.34.0 | hf-mirror route |
| torch | sampled forwards | ✓ | 2.11.0+cu130 (dev box) | runner GB10 GPU |
| hf-mirror.com | HF-route fetch (runner + dev) | ✓ HTTP 200 | — | — |
| modelscope.cn | primary fetch | ✓ HTTP 302→200 | — | — |
| huggingface.co direct | NOT usable on runner | ✓ on dev box only (HTTP 200) | — | always set `HF_ENDPOINT=https://hf-mirror.com` in census code |
| Local hub caches (dev box) | census pre-warm | ✓ 48/49 repos cached (GENERanno missing; only 3 repos carry remote `.py` cached) | — | census fetches code-only (~24 repos, KB-scale) |
| Self-hosted runner `[self-hosted, dnallm-nightly]` | dispatch lane | ✓ (existing lanes) | — | feasibility.yml posture |
| GPU on runner | D-03 forwards | ✓ (GB10; nightly lanes prove) | — | — |

**Missing dependencies with no fallback:** none.

**Missing dependencies with fallback:** huggingface.co direct access on the runner — fallback is the mandatory hf-mirror endpoint (and ModelScope-first ordering), which the census tool must hard-code.

## Security Domain

security_enforcement: true (ASVS level 1, block_on high) — assessed below; no new packages, no new network services, no auth surface.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | no auth surface added |
| V3 Session Management | no | no sessions |
| V4 Access Control | yes (CI lane) | dispatch-only trigger + `if: github.event_name == 'workflow_dispatch'` + `permissions: contents: read` (feasibility.yml posture) — PR-authored code never executes on the self-hosted box |
| V5 Input Validation | yes | remote `config.json`/`.py` treated as untrusted data: fetched into cache, never executed outside isolated census subprocesses; registry data passes frozen `extra="forbid"` Pydantic validation; census evidence artifacts validated on read |
| V6 Cryptography | no | no crypto added (revision shas are provenance, not security control) |

### Known Threat Patterns for census-over-remote-code

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malicious remote modeling code executes during import census | Elevation of Privilege / Tampering | trust_remote_code is RCE by design; census runs untrusted imports ONLY in subprocesses on the dispatch-triggered self-hosted lane and dev box (never PR-triggered); code fetched at pinned revisions (models.lock pattern) |
| Repo content swapped between census and re-run (mirror drift) | Tampering | revision-sha pinning recorded in evidence; drift gate re-verifies and fails red |
| Forged census artifact makes docs table lie | Repudiation | evidence artifact committed alongside registry fields; CI consistency gate regenerates table from registry (single source of truth) |
| Census output poisoning downstream YAML | Injection | registry write-back is scripted text insertion into a schema-validated frozen file (no eval, no template exec) |

## Sources

### Primary (HIGH confidence)
- Live probe of installed transformers 5.17.0 (`.venv`, `python -I`, no dnallm shim): removed import names, removed methods, config defaults, tie_weights signature, ROPE keys, still-present names — executed 2026-10-27
- Local cache census probe (`/tmp/p27-census-probe/probe_remote_imports.py`, `probe_py_files.py`): 49-entry roster, 27 remote / 21 native, auto_map contents, cached `.py` inventory, modeling_esm.py import blocks — executed 2026-10-27
- [huggingface/transformers MIGRATION_GUIDE_V5.md](https://github.com/huggingface/transformers/blob/429e6916/MIGRATION_GUIDE_V5.md) — remote-code incompatibility classes, tokenizer rewrite, config changes
- [huggingface/transformers issue #44561](https://github.com/huggingface/transformers/issues/44561) — `is_torch_fx_available` removal breaks trust_remote_code models; upstream not_planned; [PR #44615](https://github.com/huggingface/transformers/pull/44615) rejected; [vLLM PR #55909](https://app.semanticdiff.com/gh/vllm-project/vllm/pull/55909/overview#1) downstream-restore precedent
- Repo sources read this session: `dnallm/models/registry.py`, `dnallm/utils/transformers_compat.py`, `dnallm/models/model.py` (load chain, HF_ENDPOINT toggle, random-init path), `dnallm/mcp/consistency.py`, `scripts/audit_skips.py`, `.github/workflows/feasibility.yml`, `.github/workflows/ci.yml` (stage 0.5 census assertion), `.github/dependabot.yml`, `models.lock`, `tests/models/test_registry.py`, `tests/models/test_model_info_registry.py`, `pyproject.toml` (pytest/coverage config), `mkdocs.yml` (nav), `.planning/phases/27-.../27-CONTEXT.md`, `.planning/STATE.md`, `.planning/ROADMAP.md`

### Secondary (MEDIUM confidence)
- [transformers v5.0.0 release notes](https://github.com/huggingface/transformers/releases/tag/v5.0.0) — min-version bumps, quantization_config consolidation (cross-check context)

### Tertiary (LOW confidence)
- None — no claim in this document rests on unverified training memory; all API claims were probed or read this session

## Metadata

**Confidence breakdown:**
- Census target/roster: HIGH — measured directly from the registry file and local caches this session
- Breakage surface: HIGH — dual-sourced (live probe + official migration guide + upstream issue tracker agree)
- Shim/gate/lane extension points: HIGH — every file read this session; patterns quoted verbatim
- Expected census outcome distribution: MEDIUM — 24 of 27 remote repos have no locally cached code; their exact failure classes are evidence-shaped predictions, not measurements (that is the census's job per D-06)

**Research date:** 2026-10-27
**Valid until:** 2026-11-27 (stable registry + pinned transformers 5.17.0; a transformers bump invalidates the version-stamped verdicts — which is exactly what D-08 drift gate exists to catch)
