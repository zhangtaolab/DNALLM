# Project Research Summary

**Project:** DNALLM v1.4 — Pre-flight UX & Registry-as-Truth-Source (Track A only)
**Domain:** Quality-engineering / developer-experience milestone on an existing DNA-LLM toolkit (pytest + coverage-gated Python library)
**Researched:** 2026-10-20
**Confidence:** HIGH (all four researchers verified claims against this checkout, the installed venv, live hub APIs, and the production sibling repo `/home/forrest/Github/dnallmmark`)

## Executive Summary

DNALLM v1.4 Track A adds a pre-flight terminal flow (data validation → model suitability → config generation → select-to-download) plus runtime consumption of `model_info.yaml` (236 entries: 49 pretrained + 187 finetuned) as the registry truth source. All four researchers converged on the same architecture without coordination: **one-core-many-shells** — pure, synchronous, pytest-coverable functions (`validate_data` / `check_compatibility` / `generate_config` / `probe_availability`) in a new `dnallm/preflight/` package, a cached read-only registry loader (`dnallm/models/registry.py`) as the single parse owner, and a Textual TUI that is presentation-only over thread workers. The stack cost is minimal and proven: exactly two new runtime deps (`textual>=8.2,<9`, `platformdirs>=3.6,<5`) isolated in a `tui` extra that rides every CI pytest leg via `test = ["dnallm[tui]", ...]` with zero workflow edits; zero new test frameworks (Textual's headless `run_test()`/Pilot + existing pytest-asyncio auto mode); zero new deps for registry loading (pyyaml, CSafeLoader 12× faster) or availability probing (installed huggingface-hub 1.33.0 / modelscope 1.34.0 expose every needed probe, all live-verified against `zhangtaolab/plant-dnabert-BPE`).

The recommended build order is dependency-driven and identical across researchers: **registry loader → pre-flight core → `load_preset_model` rewire → TUI panels → MCP consistency check**. A key discovery makes the rewire safe: `load_preset_model`'s registry path (`MODEL_INFO[model].get("preset", [])`, model.py:1598) is grep-verified dead code — no entry has a `preset` key — so wiring registry names in is filling a hole, not changing behavior. The core risks are: (1) sync torch/dataset/network calls freezing the Textual event loop — mitigated by the sibling-proven `@work(thread=True, exclusive=True)` worker seam built before any panel; (2) coverage-gate erosion if `dnallm/tui/` lands in the denominator without the extra installed on CI legs — mitigated atomically (extra + CI install wiring + structure tests in the same change; no omit entries, ever); (3) registry provenance-comment destruction via a YAML round-trip dump — the loader is born read-only with integrity tests extending the 261020-kwt suite; (4) false "unavailable" verdicts from HF/ModelScope rate limits and lying cache layouts — mitigated by three-state verdicts (`cached` / `remote-confirmed` / `unknown`), injectable probe seams, and layout-aware cache resolvers.

**Residual contradictions the roadmapper must resolve explicitly:**

1. **Snapshot testing.** STACK.md recommends an SVG golden corpus via core `App.export_screenshot()` (zero deps, byte-deterministic per the sibling's months of CI). PITFALLS.md says snapshot infra is NOT adopted in v1.4 (cross-OS glyph false diffs on the Windows leg) and prescribes Pilot-functional tests only. **Recommendation: follow PITFALLS for v1.4** (Pilot-functional, no snapshot infra); the STACK SVG lane is a sound post-v1.4 add *if* single-OS golden discipline is enforced — it is not a blocker.
2. **Registry enrichment ordering.** FEATURES.md puts structured `tokenizer_type` + `max_length` enrichment on the critical path (TS-4's metadata paths depend on it; today tokenizer info lives only in free-text `describe` strings and no entry carries max-context/size). ARCHITECTURE.md's build order does not name an enrichment step — its TS-4 inputs come from `MODEL_INFO.model_tags` + hardcoded special-handler facts, with on-demand `config.json` probing as fallback (network-bound, cache-mandatory). **Recommendation: a registry enrichment workstream (data-layer, kwt-style scripted text insertion) lands with or immediately after Phase R**, before the suitability verdict's non-empirical paths; the model_tags/config.json fallbacks cover what enrichment can't express.

## Key Findings

### Recommended Stack

Exactly two new runtime dependencies, both in a `tui` extra; everything else reuses installed pins. Verified: textual 8.2.8 is current, pure-Python (zero Windows compile risk), and production-proven in the sibling repo against this exact codebase family.

**Core technologies:**
- **textual `>=8.2,<9`** — async TUI framework for the pre-flight wizard; sibling runs a 9-screen/16-service TUI on this pin in daily CI; `<9` cap because Textual majors are frequent and real
- **platformdirs `>=3.6,<5`** — XDG state/config split for TUI-side persistence; identical band to textual's own so the resolver never fights itself; declare it because dnallm/tui imports it directly
- **pyyaml (existing) + cached CSafeLoader** — registry loading; 5.8 ms vs 69 ms pure-Python on the 236-entry file; `lru_cache(maxsize=1)` load-once (packaged-immutable data); `getattr(yaml, "CSafeLoader", yaml.SafeLoader)` fallback
- **huggingface-hub 1.33.0 / modelscope 1.34.0 probe adapters** — `repo_exists`, `model_info(files_metadata=True)`, `scan_cache_dir()`, `HubApi.get_model_files` (per-file sizes, verified byte-identical across both hubs on lab models) — no weights downloaded by any probe
- **App.run_test() Pilot + pytest-asyncio (existing, auto mode)** — headless in-process TUI testing; measures coverage like any unit test; no textual-dev, no pytest-textual-snapshot (syrupy pin forces `pytest<9` — excluded by policy and capability)

**Critical wiring decisions:** `test = ["dnallm[tui]", ...]` propagates textual to every pytest leg with zero CI workflow edits; `dnallm/tui/` must NOT join the `dnallm/__init__.py` facade (eager subpackage imports would break bare installs) — entry is a `dnallm preflight` click subcommand with lazy import + friendly `pip install dnallm[tui]` ImportError; no coverage-omit entry for tui, no `pytest.importorskip("textual")` guards (both are the gate-dishonesty class).

### Expected Features

**Must have (table stakes, v1):**
- TS-10 API-first core (`validate_data`/`check_compatibility`/`generate_config` pure functions, no Textual imports) — the milestone's testability foundation
- TS-1 read-only data profile (stats + charset census + length distribution) — must NOT build on the destructive `validate_sequences` filter (it drops the rows the panel must count)
- TS-2 collect-all validation report with severities + reason codes — kills the raw-Pydantic-traceback failure mode (the survey's #1 pain)
- TS-4 suitability verdicts: task × length (truncation-fraction vs max context) × tokenizer/charset
- TS-5 availability: cached/reachable/size-before-download/revision, ModelScope-first
- TS-7 config generation with live validation + load-existing→edit→revalidate round-trip
- TS-3 three-state verdicts (OK/WARN/FAIL); WARN never blocks
- TS-9 sequential single-model download with progress + cancel (bridge over existing `download_model`)
- TS-6 env panel (device/VRAM/versions/disk); TS-8 Back/Next with prefill; registry runtime consumption via `load_preset_model`; D-4 lab-models-first ranking

**Should have (differentiators, v1.x):** D-1 truncation-fraction model ranking (highest value-per-LOC); D-2 tokenizer dry-run on a sample; D-3 base↔finetuned lineage routing (base_model axis is complete, 187/187); D-5 strict-family charset matrix; D-6 MCP tools over the same core; D-7 offline/degraded mode (UNKNOWN-not-raise from day one as a design rule)

**Defer (v2+ / out of scope):** benchmark matrices (Track B/C), full registry browser, persistent download queue (Track B merge target — AF-8), dataset download management, auto-applied data cleanup (AF-1), launching training from the wizard (AF-2), YAML text editor (AF-5), Track B/C features and defects #62/#63/#64 per owner scope pin.

### Architecture Approach

New code in two packages plus seam edits; the existing dispatch chain (`load_model_and_tokenizer` + 12 special handlers) is untouched — registry resolution produces a repo id + patched TaskConfig BEFORE the chain, so the CrossDNA handler-overwrite bug class cannot regress.

**Major components:**
1. **`dnallm/models/registry.py`** — cached single-parse loader of `model_info.yaml` (importlib.resources, `lru_cache`, `reload()` hook, frozen typed entries, lookups by name/repo_id/task_type/base_model, `is_lab_model` sort key, grep-guard test against second parsers)
2. **`dnallm/preflight/`** — pure sync verdict functions returning report objects (`DataReport`, `Verdict[]`, `AvailabilityReport`); injectable probe seam mirroring `download_model(..., downloader=...)`; collects scattered suitability facts (registry, `MODEL_INFO` tags, special-handler constraints, probe results)
3. **`dnallm/tui/`** — screens-only shell behind the `tui` extra; zero textual imports elsewhere (import-lock test); all I/O via `@work(exclusive=True, thread=True)` workers
4. **`load_preset_model` rewire** (model.py:1567-1614) — registry names resolve via the loader; family-name path preserved + regression-tested
5. **MCP consistency** — extend `MCPConfigManager.validate_model_references`: warn-first comparison of per-model YAMLs against registry, keyed on repo id (`model.path`), `strict_registry` opt-in; `_get_all_available_models` serves registry listing with configured/loaded overlay
6. **`validate_sections` factor-out** (configs.py:698-747) — shared validation path for `load_config` and `generate_config` (the one sanctioned refactor; no behavior change)

### Critical Pitfalls

1. **Event-loop freeze (Pitfall 1)** — sync torch/dataset/network calls in Textual handlers freeze the UI and hang CI under `--timeout=300`. Avoid: worker seam first, before any panel; `exit_on_error=False`; cooperative cancel in `download_model`'s retry loop; keep the core sync.
2. **Coverage-gate erosion (Pitfall 2)** — `dnallm/tui/` enters the `source_pkgs=["dnallm"]` denominator on every `--cov` leg (incl. test-windows) the moment it exists. Avoid: `tui` extra + CI install wiring + structure test (tui must not import torch/transformers/datasets) in the same change; per-module ≥96%; no omit entries; coordinate with the `full`-alias removal landing in the same milestone.
3. **Registry provenance destruction (Pitfall 3)** — any `safe_load→dump` round-trip erases the PlantHelixSeek freeze provenance block (~lines 1876-1890) silently. Avoid: loader born read-only (no write API); provenance-presence + no-round-trip integrity tests extend the 261020-kwt suite; registry maintenance stays scripted text insertion.
4. **MCP consistency check corruption (Pitfall 4)** — title-keyed joins, shallow field comparison, or auto-sync fixers manufacture silent prediction corruption. Avoid: pure `check_config_registry_consistency() -> list[Finding]`; key on repo id; compare full semantic set (task_type, num_labels, label_names ORDER, threshold, source); never write; fixture matrix per drift class; committed-configs check in CI.
5. **Availability probing traps (Pitfall 5)** — network in unit tests, shared-IP rate limits (429 ≠ unavailable), HF env vars read at import time (the runtime `HF_ENDPOINT` toggle at model.py:494 may be inert), two ModelScope cache layouts vs the hardcoded path at model.py:1545, `HF_HUB_OFFLINE` raising. Avoid: injectable probes, three-state verdicts, layout-aware resolvers, effective-value reporting via library constants; probe shortlists only, never all 236.

## Implications for Roadmap

All four researchers independently converged on the same dependency-ordered phase structure. Suggested phases (labels follow PITFALLS' R/C/T/M):

### Phase 1 (R): Registry Runtime Loader
**Rationale:** Zero new dependencies, consumes nothing new; unblocks everything else in parallel. Registry enrichment (structured `tokenizer_type`/`max_length`, kwt-style scripted insertion) belongs here or immediately after — FEATURES flags it as the critical-path gap for TS-4.
**Delivers:** `dnallm/models/registry.py` (cached, read-only, typed, importlib.resources, grep guard, `reload()`); extended registry integrity tests (provenance block, no write API).
**Addresses:** registry-as-truth-source foundation; supplies TS-7's prefill and TS-4's metadata.
**Avoids:** Pitfall 3 (read-only from birth); repo-id keying groundwork for Pitfall 4.

### Phase 2 (C): Pre-flight Core
**Rationale:** The TUI is blocked until this exists; pure functions make the ≥96% coverage gate cheap; MCP consistency reuses its finding-type conventions.
**Delivers:** `dnallm/preflight/` (validation, compatibility, config_gen, availability, report types); `validate_sections` factor-out; injectable probe seam; three-state availability; cache resolvers (both ModelScope layouts).
**Addresses:** TS-10, TS-1, TS-2, TS-4, TS-5, TS-3, TS-7 (core halves).
**Avoids:** Pitfalls 5 and 1's sync-core boundary; Pitfall 4's contract shape.

### Phase 3 (R2): `load_preset_model` Rewire
**Rationale:** Needs only the loader; file-disjoint from Phase 2 (wave-parallelizable under single-tree pathspec-commit discipline). Fills a grep-verified dead path — net-new behavior, regression-test the family-name path only.
**Delivers:** Registry names (e.g. "Plant DNABERT BPE promoter") resolve to repo id + num_labels/label_names/threshold patched TaskConfig; base↔finetuned axis live at runtime.
**Addresses:** registry runtime consumption milestone goal; D-3 groundwork.

### Phase 4 (T): TUI Panels + tui Extra
**Rationale:** Deepest dependency (core + registry). First act is infrastructure, not panels: extra wiring, CI install lines, structure tests, worker seam — then env → data → suitability → config panels (config last, needs `generate_config`), select-to-download.
**Delivers:** `dnallm/tui/`, `dnallm preflight` subcommand (lazy import), Pilot test suite.
**Addresses:** TS-6, TS-8, TS-9, D-4, TS-7's edit round-trip (R4 absorbed).
**Avoids:** Pitfalls 1, 2, 6, 7, 8 (panel→function table enforced as a test; no snapshot infra; no scope creep — anti-feature list pinned in the roadmap).

### Phase 5 (M): MCP Consistency + Honest Listing
**Rationale:** Deliberately last so the registry format stabilizes through two consumers before the comparison semantics freeze. Independent of the TUI — a compression/parallelization candidate.
**Delivers:** `check_config_registry_consistency` (warn-first, `strict_registry` opt-in), fixture matrix, committed-configs CI check; `_get_all_available_models` registry listing with overlay.
**Addresses:** D-6 seams; fixes the name/reality mismatch flagged at server.py:821.
**Avoids:** Pitfall 4; re-checks Pitfall 2 when `full`-alias removal touches the same install lines.

### Phase Ordering Rationale

- R is the root dependency — both C and M consume the loader; read-only discipline (Pitfall 3) is only cheap on day one.
- C and the `load_preset_model` rewire are file-disjoint and both depend only on R — parallelizable.
- C before T is the architecture rule: the TUI is a thin consumer, never the owner of verdict logic (survey §5.3); core landing with/first also keeps the coverage gate and future MCP/notebook shells honest.
- T's first act is CI/extra wiring, not panels (Pitfall 2's atomicity).
- M last: its fixture matrix wants the loader's keying and the core's finding types to exist.

### Research Flags

Phases likely needing deeper research during planning (`/gsd-plan-phase --research-phase`):
- **Phase T:** worker-seam shape (message types vs call_from_thread), download cancel semantics (cooperative vs abandon+resume), `tui`-extra membership vs `full`-alias removal ordering, TUI test-batch wall-time budget. Pilot-in-THIS-gate is documented but unproven in this repo (ARCHITECTURE flags MEDIUM).
- **Phase C:** two spikes — effective-endpoint reporting through huggingface_hub resolved constants (verify the runtime `HF_ENDPOINT` toggle's in-process effect at installed versions), and the ModelScope cache-layout resolver against installed modelscope.
- **Phase M:** owner call — does the consistency check cover only `dnallm/mcp/configs/` or also `configs/*.yaml` user-facing examples?
- **Registry enrichment (within Phase R):** schema decision for `tokenizer_type`/`max_length`/`size` fields and the scripted-insertion discipline.

Phases with standard patterns (skip research-phase):
- **Phase R (loader itself):** cached package-data loader, well-documented; integrity-test precedent exists (261020-kwt).
- **Phase R2 (rewire):** integration points grep-verified with line anchors; dead-path finding makes it low-risk.
- **Phase M (checker function):** pure-function pattern with in-repo precedents once Phases C/R conventions exist.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Every version/API claim live-verified in venv, PyPI JSON API, official docs, or the production sibling; one MEDIUM seam (ModelScope HubApi deprecation contract — adapter-isolated by design) |
| Features | HIGH | Internal facts read directly from checkout; ecosystem UX patterns cross-verified across ≥2 sources; one MEDIUM-LOW item (ModelScope per-file-size SDK path — STACK's live probes largely resolved it) |
| Architecture | HIGH | Every integration point read directly in this checkout with line anchors; registry schema parsed live; dead preset path grep-verified. MEDIUM: Pilot-in-this-gate, `validate_sections` refactor size |
| Pitfalls | HIGH | Repo-grounded facts verified at branch v1.4; external behaviors doc-backed. LOW: ModelScope rate limits (undocumented — design for resilience, encode nothing) |

**Overall confidence:** HIGH

### Gaps to Address

- **Pilot tests under this repo's coverage gate:** documented approach, unproven in this exact gate — first Phase-T PR must measure the batch and confirm ≥96% per tui module.
- **`HF_ENDPOINT` runtime toggle inertness:** import-time env reading is documented; in-process effect at pinned versions needs the Phase-C spike.
- **ModelScope HubApi stability:** no published deprecation contract — keep probes in one adapter module, faked in tests.
- **Registry enrichment schema:** tokenizer info only in free-text `describe`; max_length/size absent — schema + insertion discipline decided at Phase-R planning.
- **ModelScope rate limits:** undocumented (LOW) — design for 429-class resilience; never encode assumptions.
- **MCP consistency scope:** committed-configs-only vs also user-facing example configs — owner call at Phase-M planning.

## Sources

### Primary (HIGH confidence)
- This checkout at branch `v1.4` — all integration points (model.py, configs.py, data.py, config_manager.py, server.py, modeling_auto.py, model_info.yaml, pyproject.toml, ci.yml), read 2026-10-20 with line anchors
- Live venv probes (2026-10-20) — ModelScope `HubApi.get_model_files` field shapes + byte sizes; HF `model_info(files_metadata=True)`/`scan_cache_dir()`/`repo_exists`; pyyaml CSafeLoader timing; textual 8.2.8 API signatures; cross-hub byte-identical size agreement on `zhangtaolab/plant-dnabert-BPE`
- Sibling production repo `/home/forrest/Github/dnallmmark` — textual pin, worker patterns, Pilot + SVG-golden CI, months of green runs
- Official docs: Textual testing/workers guides; huggingface_hub environment-variables + rate-limits pages; PyPI JSON API

### Secondary (MEDIUM confidence)
- docker init / nf-schema / pandera lazy / Snakemake dry-run / wandb / Ultralytics UX precedents (cross-verified ≥2 sources each)
- PyYAML comment-dropping + ruamel round-trip caveats (multi-source); pytest-asyncio ≥0.25 Textual breakage history; anyio/pytest-asyncio auto-mode conflict; HF model_info 429 burst under transformers≥5 (LlamaFactory #10155); Windows conhost/mintty issues (textual#4974/#5456, msys2#22188)
- ModelScope cache-layout change + MODELSCOPE_CACHE quirks (issue #845)

### Tertiary (LOW confidence)
- ModelScope API rate limits (no official doc — design for resilience)
- In-process `HF_ENDPOINT` toggle inertness at installed versions (needs Phase-C spike)

---
*Research completed: 2026-10-20*
*Ready for roadmap: yes*
