# Requirements: DNALLM v1.4 — Pre-flight UX & Registry-as-Truth-Source

**Defined:** 2026-10-20
**Core Value:** One terminal flow takes a user from their own data to a validated, model-compatible config, with the model registry as the runtime single source of truth (Track A first).

## v1.4 Requirements

Scope pinned by owner 2026-10-20: Track A only; Track B (dnallmmark TUI substrate merge) and Track C (R2/R3/R5/R6) deferred; defects #62/#63/#64 fixed outside this milestone. Differentiators D-3/D-7 in; D-1/D-2 to Future.

### Registry Foundation (Phase 22)

- [x] **REG-01**: A cached, read-only registry loader (`dnallm/models/registry.py`) is the single parse owner of `model_info.yaml` (typed entries; lookups by name/repo_id/task_type/base_model; `reload()` hook; CSafeLoader with fallback; grep-guard against second parsers)
- [x] **REG-02**: Registry entries carry structured `tokenizer_type` and `max_length` (enrichment via scripted text insertion — provenance comments preserved; schema finalized at phase planning)
- [x] **REG-03**: Registry integrity tests extended from 261020-kwt — provenance-block presence, no write/round-trip API on the loader, enrichment-schema invariants

### Pre-flight Core (Phase 23)

- [ ] **PFC-01**: `validate_data()` produces a read-only collect-all report (stats, charset census, length distribution, severities + reason codes) and never drops or mutates rows
- [ ] **PFC-02**: `check_compatibility()` returns three-state verdicts (OK/WARN/FAIL) over task × length (truncation vs max context) × tokenizer/charset; WARN never blocks
- [ ] **PFC-03**: Availability probe is ModelScope-first and returns cached/remote-confirmed/unknown states with size and revision, without downloading weights; rate-limit responses map to unknown, never "unavailable"; probes are injectable so unit tests are network-free
- [ ] **PFC-04**: `generate_config()` emits a validated config and supports load-existing → edit → revalidate round-trip (shared `validate_sections` path with `load_config`; collect-all field-level errors, not tracebacks)
- [ ] **PFC-05**: Model listings route by user intent via the base↔finetuned axis (`base_model`): "fine-tune" lists bases, "predict now" lists task models with heads/labels (D-3)
- [ ] **PFC-06**: Offline/degraded mode: any probe or network failure degrades to UNKNOWN and the flow continues — no hard errors from unavailability (D-7)
- [ ] **PFC-07**: Lab models (`zhangtaolab/*`) rank first in every model listing the core produces (owner directive 2026-10-20)

### Preset Loading (Phase 24)

- [ ] **LOAD-01**: Registry entry names resolve in `load_preset_model` (finetuned entries load with `num_labels`/`label_names`/`threshold` patched into TaskConfig); the existing family-name path is preserved and regression-tested

### Terminal UI (Phase 25)

- [ ] **TUI-01**: A `tui` extra (`textual>=8.2,<9`, `platformdirs>=3.6,<5`) with `test = ["dnallm[tui]", ...]` rides every pytest CI leg; structure tests enforce zero textual imports outside `dnallm/tui/` and zero torch/transformers imports inside it — all landing atomically in one change
- [ ] **TUI-02**: `dnallm preflight` click subcommand launches the wizard via lazy import with a friendly `pip install dnallm[tui]` error when the extra is absent
- [ ] **TUI-03**: Environment panel shows device/VRAM/library versions/disk from the core (no logic in screens)
- [ ] **TUI-04**: Data panel renders the `validate_data()` report (stats + per-rule failures with counts and reason codes)
- [ ] **TUI-05**: Suitability panel renders `check_compatibility()`/availability verdicts with lineage routing and lab-first ordering; model selection feeds the next step
- [ ] **TUI-06**: Config panel generates or edits an existing YAML with live collect-all validation (absorbs survey R4) and emits the validated config
- [ ] **TUI-07**: Select-to-download runs `download_model` sequentially with progress and cancel; all blocking work runs in thread workers so the event loop never freezes (worker seam lands before any panel)
- [ ] **TUI-08**: TUI tested via Pilot functional tests (no snapshot infrastructure); a panel→core-function table test proves screens contain no domain logic; per-module coverage ≥96% with no omit entries

### MCP Consistency (Phase 26)

- [ ] **MCP-01**: `check_config_registry_consistency()` compares per-model MCP YAMLs against the registry keyed on repo id (task_type, num_labels, label_names ORDER, threshold, source), warn-first with `strict_registry` opt-in, and never writes
- [ ] **MCP-02**: `get_all_available_models` serves the registry listing with a configured/loaded overlay (closes the name/reality mismatch — issue #64's consumption half)
- [ ] **MCP-03**: A CI check runs the consistency checker over committed configs (`dnallm/mcp/configs/`; coverage of example configs decided at phase planning)

### Packaging (Phase 25)

- [ ] **PKG-01**: The one-cycle `full` extras alias is removed (pre-committed at v1.3 close), sequenced atomically with TUI-01's install-line changes

## Future Requirements

### Pre-flight Extensions (v1.x)

- **D-1**: Truncation-fraction model ranking from the user's actual length distribution
- **D-2**: Tokenizer dry-run on sampled sequences (real encoding compatibility evidence)
- **D-5**: Strict-family charset compatibility matrix
- **D-6**: MCP tools exposing the pre-flight core (`dnallm_preflight_check` class)

### Deferred Tracks (owner, 2026-10-20)

- **Track B**: dnallmmark TUI substrate merge into `dnallm/tui/` (selection/download queue, env_check, wizard framework, monitor machinery)
- **Track C**: R2 live fine-tune dashboard, R3 MCP control room (needs #63), R5 VEP batch runner, R6 in-terminal interpretation triage

## Out of Scope

| Feature | Reason |
|---------|--------|
| Launching training from the wizard (AF-2) | Wizard ends at validated config + downloaded model; training launch is a separate surface |
| Persistent multi-file download queue (AF-8) | Track B merge target — sequential single-model download suffices for Track A |
| Full registry browser / benchmark dataset management | dnallmmark territory (merge policy: merge later, never duplicate) |
| Auto-applied data cleanup (AF-1) | Validation is read-only by design; users fix their own data |
| YAML text editor inside the TUI (AF-5) | Config panel edits structured fields; raw editing stays outside |
| Snapshot/SVG golden testing in v1.4 | Cross-OS glyph diffs on the Windows leg; Pilot-functional tests only |
| Fixes for issues #62/#63/#64 (defect halves) | Owner disposition 2026-10-20: file issues, fix outside v1.4 (MCP-02 covers only #64's registry-consumption half, already in-theme) |
| Merging into `main` | Owner directive 2026-10-20: main frozen; all v1.4 development on branch `v1.4` |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| REG-01 | Phase 22 | Complete |
| REG-02 | Phase 22 | Complete |
| REG-03 | Phase 22 | Complete |
| PFC-01 | Phase 23 | Pending |
| PFC-02 | Phase 23 | Pending |
| PFC-03 | Phase 23 | Pending |
| PFC-04 | Phase 23 | Pending |
| PFC-05 | Phase 23 | Pending |
| PFC-06 | Phase 23 | Pending |
| PFC-07 | Phase 23 | Pending |
| LOAD-01 | Phase 24 | Pending |
| TUI-01 | Phase 25 | Pending |
| TUI-02 | Phase 25 | Pending |
| TUI-03 | Phase 25 | Pending |
| TUI-04 | Phase 25 | Pending |
| TUI-05 | Phase 25 | Pending |
| TUI-06 | Phase 25 | Pending |
| TUI-07 | Phase 25 | Pending |
| TUI-08 | Phase 25 | Pending |
| PKG-01 | Phase 25 | Pending |
| MCP-01 | Phase 26 | Pending |
| MCP-02 | Phase 26 | Pending |
| MCP-03 | Phase 26 | Pending |

**Coverage:**
- v1.4 requirements: 23 total
- Mapped to phases: 23
- Unmapped: 0 ✓

---
*Requirements defined: 2026-10-20*
*Last updated: 2026-10-20 after initial definition (research SUMMARY.md + owner scoping D-3/D-7)*
