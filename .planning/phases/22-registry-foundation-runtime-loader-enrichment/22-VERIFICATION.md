---
phase: 22-registry-foundation-runtime-loader-enrichment
verified: 2026-10-22T03:25:00Z
status: passed
score: 17/18 must-haves verified
covered_files:
  - .planning/phases/22-registry-foundation-runtime-loader-enrichment/22-01-PLAN.md
  - .planning/phases/22-registry-foundation-runtime-loader-enrichment/22-01-SUMMARY.md
  - .planning/phases/22-registry-foundation-runtime-loader-enrichment/22-02-PLAN.md
  - .planning/phases/22-registry-foundation-runtime-loader-enrichment/22-02-SUMMARY.md
  - dnallm/models/__init__.py
  - dnallm/models/model_info.yaml
  - dnallm/models/registry.py
  - tests/models/test_model_info_registry.py
  - tests/models/test_registry.py
covered_digest: v3:sha256:624296f330cf153b44428f9aaeb65f6a975aa2ba206c4460c114ed350093e99d
behavior_unverified: 0 # No PRESENT_BEHAVIOR_UNVERIFIED truths — every behavior-dependent truth has a passing behavioral test
overrides_applied: 0
gaps: []
human_verification:
  - test: "Confirm the committed enrichment values trace to real remote hub metadata (the 22-02 no-guessed-values prohibition, marked [unverified]/judgment-tier in the plan). Re-run the gitignored probe (.scratch/probe_registry_enrichment.py) or spot-check 2-3 load-bearing repos by fetching their config.json / tokenizer_config.json from ModelScope/HF — e.g. LongSafari/hyenadna-small-32k-seqlen-hf config.json max_seq_len == 32770 and zhangtaolab/plant-dnabert-BPE tokenizer_class == 'DebertaV2Tokenizer'."
    expected: "Remote metadata values equal the committed model_info.yaml values (and the recorded .scratch/enrichment_results.json derivations). No committed value turns out to be a plausible default rather than a measurement."
    why_human: "Requires network access to the model hubs, which this verifier does not perform. The full derivation chain lives in gitignored .scratch/ by owner convention (uncommitted by design), so the committed repo can prove committed==recorded (this verifier cross-checked 109/109 fields) but not recorded==remote. The executor's own SUMMARY already marks this coverage item (D5) human_judgment: true for the same reason."
---

# Phase 22: Registry Foundation — Runtime Loader & Enrichment Verification Report

**Phase Goal:** `model_info.yaml` becomes a runtime-consumable single source of truth — one cached, read-only, typed loader is the file's sole parser, entries gain structured tokenizer/max-length metadata, and integrity tests make both properties permanent
**Verified:** 2026-10-20T12:40:44Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Roadmap Success Criteria (the contract)

| # | Success Criterion | Status | Evidence |
|---|-------------------|--------|----------|
| 1 | Lookups by name/repo_id/task_type/base_model through `dnallm/models/registry.py`, typed entries from a single cached parse (CSafeLoader w/ SafeLoader fallback; `reload()`); grep-guard test fails on any second parser in `dnallm/` | ✓ VERIFIED | `registry.py:194-273` (lru_cache, `getattr(yaml, "CSafeLoader", yaml.SafeLoader)` at :218, reload at :266); all lookup tests green; e2e resolve command passes; mentioner gate CLEAN (only `dnallm/models/registry.py` + the allowlisted stale comment `dnallm/mcp/server.py:821`); `test_registry_module_is_sole_parser_of_model_info_yaml` green |
| 2 | Structured `tokenizer_type`/`max_length`, scripted text insertion with provenance preserved, enrichment-schema invariants tested across all 236 entries | ✓ VERIFIED | Enrichment commit `2b74aa7` numstat on `model_info.yaml`: 122 added / 0 deleted (insertion-only); sentinels each count 1; enrichment provenance block present at EOF; 57/57 registry+kwt tests green; verifier's independent YAML audit: 49/49 pretrained max_length, 23/23 lab tokenizer_type, 46/49 tokenizer_type (3 documented third-party gaps), zero degenerate values, 14 external-based plantcad all 8192, zero finetuned duplicating base values |
| 3 | Extended kwt integrity suite proves the loader born-and-stays read-only: no write/YAML round-trip API; every provenance block intact | ✓ VERIFIED | `test_registry_module_exposes_no_write_api`, `test_registry_module_source_has_no_yaml_emitter` (AST mode scan), `test_registry_entries_are_frozen` all green; kwt suite extended not modified (numstat on `test_model_info_registry.py`: 136 added / 0 deleted; 9 pre-existing kwt tests pass unmodified); kwt freeze sentinels verbatim exactly once each |

### Observable Truths (Plan 22-01 — 12/12 verified)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | resolve('Plant DNABERT BPE promoter') returns frozen RegistryEntry with model/task/base/kind pinned | ✓ VERIFIED | `test_resolve_by_entry_name_returns_typed_entry` green; e2e command prints ok |
| 2 | Parsed exactly once per process; repeated access returns identical objects; reload() re-parses | ✓ VERIFIED (behavioral) | `test_single_cached_parse_returns_identical_objects`, `test_reload_forces_reparse` green — state transition exercised by tests |
| 3 | Every lookup flavor across all 236 entries (resolve/try_resolve, get_by_task_type, get_by_base_model) | ✓ VERIFIED | `test_get_by_task_type_spans_both_sections`, `test_get_by_base_model_lists_finetuned_children`, `test_all_entries_count_and_file_order` green; 236-entry facade command passes |
| 4 | Empty/unknown inputs raise matchable ValueError; list queries return [] | ✓ VERIFIED | `test_resolve_empty_string_raises_not_found`, `test_empty_or_unknown_list_queries_return_empty_lists` green |
| 5 | Exact UTF-8 str equality, no casefolding/trimming; file opened encoding='utf-8' | ✓ VERIFIED | `test_lookup_is_exact_case_sensitive` green; `registry.py:213` opens with encoding="utf-8" |
| 6 | Malformed data fails loudly at FIRST access (schema violations naming offender, duplicates, missing sections) | ✓ VERIFIED (behavioral) | `test_malformed_entry_fails_loudly_naming_offender`, `test_duplicate_*_rejected`, `test_missing_sections_rejected`, `test_missing_resource_raises_value_error`, `test_yaml_parse_error_raises_value_error` all green |
| 7 | Inheritance resolved at query time: own value wins, base_model chain followed, cycle guard | ✓ VERIFIED (behavioral) | 6 synthetic `_effective_value` tests incl. `test_effective_value_cycle_returns_none` green |
| 8 | RegistryEntry frozen; module exposes no write/serialization API | ✓ VERIFIED | `test_frozen_entry_assignment_rejected`, `test_registry_module_exposes_no_write_api`, `test_registry_module_source_has_no_yaml_emitter` green; module source read confirms |
| 9 | Sole mentioner allowlist exactly {registry.py, mcp/server.py} | ✓ VERIFIED | Grep gate CLEAN; `test_registry_module_is_sole_parser_of_model_info_yaml` green; server.py:821 is the stale comment |
| 10 | max_length boundary: 1 valid; 0/negative rejected via gt=0 | ✓ VERIFIED | `test_max_length_accepts_one_and_rejects_nonpositive` green |
| 11 | No arithmetic/rounding on values (threshold 0.5 exact, max_length exact int) | ✓ VERIFIED | `test_resolve_by_entry_name_returns_typed_entry` asserts `threshold == 0.5`; loader carries YAML-parsed values verbatim (no transformation in source) |
| 12 | Concurrency safe by construction; docstring documents reload() as single-threaded hook | ✓ VERIFIED | Module docstring (registry.py:14-19) documents exactly this; immutable entries, no mutation API |

### Observable Truths (Plan 22-02 — 5/6 verified, 1 uncertain)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | All 49 pretrained carry positive-int max_length; all 23 zhangtaolab pretrained carry non-empty tokenizer_type | ✓ VERIFIED | `test_all_pretrained_entries_carry_positive_int_max_length`, `test_lab_pretrained_entries_carry_tokenizer_type` green; independent YAML audit confirms 49/49 and 23/23 |
| 2 | Third-party gaps are absent keys — never null/empty/zero; present values non-degenerate | ✓ VERIFIED | `test_third_party_enrichment_values_present_or_absent_never_degenerate` green; independent audit: zero degenerate values across all 236 |
| 3 | Every committed tokenizer_type is the exact class-name string measured from hub metadata | ? UNCERTAIN | Committed values match the recorded probe measurements exactly (verifier cross-check: 109/109 enrichment fields vs `.scratch/enrichment_results.json`, zero mismatches); 3 spot pins enforced by committed tests. But "measured from hub metadata" is a provenance claim whose full chain lives in gitignored `.scratch/` — confirming recorded==remote requires network access. See Human Verification #1 |
| 4 | Every finetuned entry resolves positive effective max_length (inherited or explicit on the 14 plantcad); explicit only when differing from base | ✓ VERIFIED (behavioral) | `test_effective_max_length_resolved_for_every_finetuned_entry` green; verify command prints `187/187 effective max_length resolve`; independent audit: zero finetuned max_length duplicating its base's committed value |
| 5 | Enrichment diff insertion-only (0 deleted), kwt sentinels verbatim exactly once each, enrichment provenance comment block present | ✓ VERIFIED | `git show --numstat 2b74aa7` → 122/0 on model_info.yaml; both sentinel greps count 1; provenance block read at EOF |
| 6 | Extended integrity suite proves across all 236 entries: provenance present, invariants hold, counts 49/187, kwt tests pass unmodified | ✓ VERIFIED | `test_provenance_blocks_present_verbatim`, `test_registry_section_counts_unchanged`, `test_finetuned_max_length_only_as_explicit_override`, `test_spot_pinned_enrichment_values` green; kwt file diff is pure insertion (136/0) |

**Score:** 17/18 plan truths verified (0 present, behavior-unverified; 1 uncertain → human verification). Roadmap SCs: 3/3 verified.

### Flagged Prohibition (ADR-550 D4 — judgment-tier, human review recommended)

> **unverified-prohibition — human review recommended**: 22-02 must_haves.prohibitions item 1 ("No guessed or plausible-default enrichment values may be committed — every committed value traces to a measured remote artifact or documented-constant fallback, D-01") cannot be closed without network access to the model hubs. LLM-judge verdict on committed evidence: **upheld** — all 109 committed enrichment fields match the recorded per-value probe measurements exactly; 5 pretrained stragglers carry sourced documented-constant derivations (PlantCAD2 paper bioRxiv 10.1101/2025.08.27.672609, PDLLMs training-script default, GENERanno 0.5b sibling config) recorded in `.scratch/enrichment_results.json`; the gaps file records honest gaps (24 third-party + 1 lab finetuned with a concrete no-measurable-value reason) rather than fabricated values; spot pins pin load-bearing values in committed tests. The residual risk — recorded values not faithful to actual hub artifacts — is exactly what Human Verification #1 closes. The executor's own SUMMARY marks the corresponding coverage item (D5) `human_judgment: true`.

All other prohibitions (22-01 ×3, 22-02 ×5) verified programmatically: no write/emitter API (structure tests + source scan), no network imports in the loader (grep NONE), sole parser in `dnallm/` (gate CLEAN), insertion-only landing (numstat 122/0), no inherited-value duplication (independent audit + test), kwt suite unmodified (numstat 136/0), integrity suite network-free (grep NONE), DERIVATION_EXEMPT still empty (`test_model_info_registry.py:64`).

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `dnallm/models/registry.py` | Cached, read-only, typed sole-parser loader | ✓ VERIFIED | 406 lines; frozen Pydantic models (extra="forbid"), lru_cache parse, CSafeLoader+fallback, UTF-8, loud validation, full query surface, D-02 inheritance with cycle guard; 100% line coverage (119/119, floor 96%) |
| `tests/models/test_registry.py` | Loader behavior suite + REG-03 guards | ✓ VERIFIED | 681 lines, 41 tests green: e2e resolve, cache/reload, loud validation, boundary, query surface, lab-first, synthetic inheritance matrix, facade import, 4 structure guards, 5 live-inheritance proofs |
| `dnallm/models/__init__.py` | Extended exports (11 registry names) | ✓ VERIFIED | Facade re-exports all 11 public names + `__all__`; import stays lazy (cache currsize==0 at import, verified) |
| `dnallm/models/model_info.yaml` | Enriched with tokenizer_type/max_length + provenance comment | ✓ VERIFIED | 49/49 pretrained max_length, 46/49 tokenizer_type, 14× explicit 8192 plantcad, enrichment provenance block at EOF, kwt sentinels intact |
| `tests/models/test_model_info_registry.py` | Extended integrity suite | ✓ VERIFIED | 136/0 insertion over the kwt suite (extension only); 7 new tests + 9 pre-existing kwt tests green |
| `.scratch/` probe + insertion tooling | Gitignored one-off artifacts (uncommitted by design) | ✓ VERIFIED (present) | `probe_registry_enrichment.py`, `apply_registry_enrichment.py`, `enrichment_results.json` (236 records, all with derivation except the 1 documented lab gap), `enrichment_gaps.json` all exist locally; plan verifies probe JSON: prints "probe complete: 49/49 max_length, 23/23 lab tokenizer_type" |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| registry.py | dnallm/models/model_info.yaml | importlib.resources (`resources.files("dnallm.models").joinpath(...)`) | ✓ WIRED | `registry.py:212`; e2e resolve returns real packaged data (236 entries) |
| dnallm/models/__init__.py | registry public API | `from .registry import ...` | ✓ WIRED | 11 names re-exported; facade test green; zero consumers outside the facade (grep: NONE — "zero consumers rewired" holds) |
| tests/models/test_registry.py | registry module | direct import + public/underscore API | ✓ WIRED | 57 tests pass against the real module |
| committed enrichment fields | RegistryEntry schema | Pydantic validation at first access | ✓ WIRED | `extra="forbid"` + gt=0/min_length=1; loader validates 49/49 pretrained (verify command) |
| insertion keying | unique `model: "<repo>"` lines | scripted text insertion | ✓ WIRED | numstat 122/0 (zero deleted lines — no round-trip) |
| effective_max_length / effective_tokenizer_type | enriched data | base_model chain at query time | ✓ WIRED | 187/187 finetuned resolve live (verify command + tests) |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| registry.py | `_load_registry()` result | packaged `model_info.yaml` via importlib.resources + CSafeLoader | Yes — 236 real entries resolve (236/189 counts measured) | ✓ FLOWING |
| model_info.yaml enrichment fields | `tokenizer_type` / `max_length` | one-time remote hub probes (D-01), committed as text insertion | Yes — values consumed by effective_* at query time; match recorded probe results 109/109 | ✓ FLOWING (remote provenance → Human Verification #1) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Registry + kwt suites | `.venv/bin/python -m pytest tests/models/test_registry.py tests/models/test_model_info_registry.py -q` | 57 passed in 4.79s | ✓ PASS |
| Whole models lane | `.venv/bin/python -m pytest tests/models/ -q` | 527 passed, 1 warning in 68.06s | ✓ PASS |
| E2E typed resolve | `.venv/bin/python -c "from dnallm.models.registry import resolve; ..."` | ok | ✓ PASS |
| 236 entries / 189 lab / lab-first ordering | facade import + assertions | ok | ✓ PASS |
| 187/187 finetuned effective max_length | registry.reload() + truthy scan | 187/187 effective max_length resolve | ✓ PASS |
| Loader validates 49/49 pretrained | Pydantic path over committed YAML | loader validates 49/49 | ✓ PASS |
| Facade import lazy (no parse at import) | cache_info().currsize == 0 after import | lazy ok: currsize==0 at import | ✓ PASS |
| Sole-mentioner gate | grep -rl 'model_info.yaml' dnallm --include='*.py' minus allowlist | CLEAN (only registry.py + server.py) | ✓ PASS |
| registry.py coverage | plain `coverage run -m pytest` + filtered report (pytest-cov broken locally, known deferred item) | 119/119 statements, 100% | ✓ PASS |
| Ruff lint + format on phase files | `.venv/bin/ruff check` / `format --check` | All checks passed; 4 files already formatted | ✓ PASS |
| Commit ancestry | all 8 documented commits (d6f0589…7a3331e) | all ancestors of HEAD | ✓ PASS |

### Probe Execution

Not applicable — no `scripts/*/tests/probe-*.sh` probes declared by the plans; this phase's "probe" is remote hub metadata fetching (gitignored `.scratch/` tooling, validated above via its results JSON).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| REG-01 | 22-01 | Cached read-only registry loader, single parse owner, typed lookups, reload, CSafeLoader+fallback, grep-guard | ✓ SATISFIED | SC-1 evidence above; 12/12 22-01 truths verified |
| REG-02 | 22-02 | Structured tokenizer_type + max_length via scripted insertion, provenance preserved | ✓ SATISFIED | SC-2 evidence above; 5/6 22-02 truths verified, 1 uncertain (provenance faithfulness — Human Verification #1) |
| REG-03 | 22-01 + 22-02 | Integrity tests extended from 261020-kwt: provenance presence, no write/round-trip API, enrichment invariants | ✓ SATISFIED | SC-3 evidence above; kwt extended not modified; all read-only guards green |

Orphaned requirements: none — REQUIREMENTS.md maps exactly REG-01/REG-02/REG-03 to Phase 22, and the plans declare exactly these IDs (22-01: REG-01, REG-03; 22-02: REG-02, REG-03).

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| tests/models/test_registry.py | REG-01, REG-03 | 41 | 0 | 0 | Value / behavioral (identity, pinned constants, pytest.raises match) | OK |
| tests/models/test_model_info_registry.py | REG-02, REG-03 | 16 | 0 | 0 | Value (pinned constants, sentinel counts, offender lists) | OK |

- Disabled tests on requirements: 0 (grep for skip markers: NONE)
- Circular patterns: none — spot pins and expected values are hardcoded constants with cited sources (probe records / paper citations), never derived by running the loader
- Expected-value provenance: spot pins PARTIAL-to-VALID — hardcoded from the probe measurements recorded in `.scratch/enrichment_results.json` (uncommitted; the remote chain is Human Verification #1). The kwt suite's pre-existing pins predate this phase.
- TDD evidence: `.planning/phases/22-*/tdd-records/` holds 8 record files (RED/GREEN per task); SUMMARYs honestly note the gsd classifier cannot parse pytest console output (`workflow.tdd_mode: false`) and provide semantic assessment instead

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | No TBD/FIXME/XXX, no TODO/HACK/PLACEHOLDER, no placeholder text, no empty returns, no console-only implementations in any phase file | - | - |

Code-review findings (advisory, per task instructions — recorded open in `22-REVIEW-DISPOSITION.md`): 3 warnings / 6 info, all disposition `open`. Notable: WR-01 (S506 preview-only directive) is moot in this repo — `pyproject.toml` has `preview = true` and `ruff check` passes on the file; WR-02/WR-03 and the info findings are hardening opportunities for later phases, none block the phase goal. The phase's own `deferred-items.md` records one open planning-tooling issue (WINDOWS.md ledger row 28 blocks `windows append`) — unrelated to phase code.

### Decision Coverage

All trackable CONTEXT.md decisions honored by shipped artifacts: 4/4 (gate output: "All trackable CONTEXT.md decisions are honored by shipped artifacts." — D-01 probe-then-commit, D-02 loader-resolved inheritance, D-03 lab-mandatory/third-party-best-effort, D-01b hard Pydantic validation all evidenced above).

### Human Verification Required

### 1. Enrichment value provenance (no-guessed-values prohibition, D-01)

**Test:** Re-run the gitignored probe (`.scratch/probe_registry_enrichment.py`) or spot-check 2-3 load-bearing repos by fetching only their `config.json` / `tokenizer_config.json` from ModelScope/HF — e.g. `LongSafari/hyenadna-small-32k-seqlen-hf` (expect `max_seq_len` 32770) and `zhangtaolab/plant-dnabert-BPE` (expect `tokenizer_class` "DebertaV2Tokenizer").
**Expected:** Remote metadata values equal the committed `model_info.yaml` values and the recorded `.scratch/enrichment_results.json` derivations. No committed value turns out to be a plausible default rather than a measurement.
**Why human:** Requires network access to the model hubs. The full derivation chain lives in gitignored `.scratch/` by owner convention (uncommitted by design), so the committed repo proves committed==recorded (cross-checked 109/109) but not recorded==remote. The executor's own SUMMARY flags the same item (D5, human_judgment: true).

### Gaps Summary

No gaps. All roadmap success criteria verified; all artifacts exist, are substantive, are wired, and flow real data; all requirements REG-01/02/03 satisfied; no blocker anti-patterns; decision coverage 4/4. The single open item is a judgment-tier data-provenance prohibition that requires one network spot-check to close — routed to the end-of-phase human checkpoint rather than a gap, per the ADR-550 D4 soft-gate. Notable executed-and-documented deviations from plan (all verified benign in the codebase): hyenadna pinned at measured 32770 vs the plan's 32768 name figure; the 22-02 Task 2 commit includes `tests/models/test_registry.py` (13/6) to keep the suite green at every commit — the pre-enrichment None-contract test was strengthened to the post-enrichment inherited-values contract, exactly as the prior-wave note prescribed; `label_names` bare-string wrapping via field_validator accepts the 36 packaged regression entries without mutating the file.

---

_Verified: 2026-10-20T12:40:44Z_
_Verifier: Claude (gsd-verifier)_
