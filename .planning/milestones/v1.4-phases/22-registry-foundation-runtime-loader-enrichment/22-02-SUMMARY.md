---
phase: 22-registry-foundation-runtime-loader-enrichment
plan: "02"
subsystem: models/registry
tags: [registry, model-info, enrichment, tokenizer-type, max-length, scripted-insertion, probe-then-commit]

requires:
  - phase: 22-registry-foundation-runtime-loader-enrichment plan 01
    provides: the typed read-only loader (RegistryEntry schema with tokenizer_type/max_length, effective-value inheritance) this enrichment feeds
provides:
  - tokenizer_type / max_length enrichment on all 49 pretrained entries (REG-02), committed from one-time remote probes (D-01)
  - explicit max_length 8192 on the 14 external-based plantcad finetuned entries; every other finetuned entry inherits via base_model at query time (D-02)
  - enrichment provenance comment block at end of model_info.yaml (method, date, D-02/D-03 rules, tooling-uncommitted note)
  - seven new kwt-file integrity tests + five loader-level live-inheritance tests (REG-03 data-side enforcement)
affects: [23-pre-flight-core, 24-preset-loading-rewire, 25-tui, 26-mcp-consistency]

actuals:
  tokens: 8171   # chars/4 over the realized diff (32,683 chars)
  tasks: 3
  commits: 2     # MEASURED: git rev-list --count d9f2304..HEAD

tech-stack:
  added: []      # zero new dependencies (modelscope + huggingface-hub already pinned, probes only)
  patterns:
    - "probe-then-commit enrichment: one-time remote metadata probes (config.json/tokenizer_config.json only, never weights) -> gitignored .scratch tooling -> scripted text insertion"
    - "config-key priority extraction with tokenizer model_max_length measured fallback and documented-constant straggler chain (per-entry source comments)"

key-files:
  created: []    # committed artifacts only; probe/insertion tooling is gitignored .scratch/ by design
  modified:
    - dnallm/models/model_info.yaml
    - tests/models/test_model_info_registry.py
    - tests/models/test_registry.py

key-decisions:
  - "hyenadna-small-32k committed at the MEASURED 32770 (config.json max_seq_len + tokenizer model_max_length agree), not the plan's 32768 — committing the repo-name figure would have violated the plan's own no-guessed-values prohibition"
  - "tokenizer_config.json model_max_length used as a measured fallback when no config-side key exists (D-01 names tokenizer_config.json as a measured source; the mamba family records its context only there)"
  - "Three unmeasurable pretrained repos resolved via documented constants with per-entry sources: PlantCAD2-Small 8192 / PlantCaduceus_l32 512 / gpn-brassicales 512, all from the PlantCAD2 paper (bioRxiv 10.1101/2025.08.27.672609); plant-dnamamba-BPE 512 from the PDLLMs upstream training-script default; GENERanno-1.2b 8192 from the surviving 0.5b sibling's config (1.2b-base repo absent from both hubs)"
  - "The GPTNeoXTokenizer class-template max_length:1024 found in mamba-BPE tokenizer configs was deliberately NOT used (a library default, not a model-specific value — plausible-default prohibition)"

patterns-established:
  - "Enrichment gaps are absent keys, never degenerate values; enforced across all 236 entries by a never-degenerate scan"
  - "Finetuned duplication ban: a committed finetuned max_length exists only where it differs from the base's (test enforces for any future override) plus the registry-unresolvable external-based set"

requirements-completed: [REG-02, REG-03]

coverage:
  - id: D1
    description: "All 49 pretrained entries carry positive-int max_length and all 23 zhangtaolab pretrained carry non-empty tokenizer_type, committed as insertion-only text (122 added / 0 deleted) with both provenance blocks verbatim"
    requirement: REG-02
    verification:
      - kind: unit
        ref: "tests/models/test_model_info_registry.py#test_all_pretrained_entries_carry_positive_int_max_length"
        status: pass
      - kind: unit
        ref: "tests/models/test_model_info_registry.py#test_lab_pretrained_entries_carry_tokenizer_type"
        status: pass
      - kind: unit
        ref: "tests/models/test_model_info_registry.py#test_provenance_blocks_present_verbatim"
        status: pass
      - kind: other
        ref: "command: git show --numstat 2b74aa7 -- dnallm/models/model_info.yaml -> 122 added 0 deleted; sentinel greps count 1 (PROVENANCE-INTACT)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Third-party gaps are absent keys (never null/empty/zero) and finetuned max_length exists only as explicit override or on the 14 external-based plantcad repos (explicit 8192)"
    requirement: REG-02
    verification:
      - kind: unit
        ref: "tests/models/test_model_info_registry.py#test_third_party_enrichment_values_present_or_absent_never_degenerate"
        status: pass
      - kind: unit
        ref: "tests/models/test_model_info_registry.py#test_finetuned_max_length_only_as_explicit_override"
        status: pass
    human_judgment: false
  - id: D3
    description: "Live inheritance over committed data: 187/187 finetuned entries resolve positive effective max_length through the public loader API; pretrained effective == own; promoter inherits DebertaV2Tokenizer; 236-entry load validation; external-based fall back to own 8192"
    requirement: REG-02
    verification:
      - kind: unit
        ref: "tests/models/test_registry.py#test_effective_max_length_resolved_for_every_finetuned_entry"
        status: pass
      - kind: unit
        ref: "tests/models/test_registry.py#test_effective_tokenizer_type_inherits_from_base"
        status: pass
      - kind: unit
        ref: "tests/models/test_registry.py#test_external_based_entries_carry_explicit_max_length"
        status: pass
      - kind: other
        ref: "command: registry.reload(); finetuned effective_max_length all truthy -> '187/187 effective max_length resolve'"
        status: pass
    human_judgment: false
  - id: D4
    description: "REG-03 data-side enforcement: kwt suite extended not modified (9 pre-existing tests untouched), section counts pinned 49/187, spot pins for load-bearing values, loader validates every committed line (Pydantic, registry.py 100% covered)"
    requirement: REG-03
    verification:
      - kind: unit
        ref: "tests/models/test_model_info_registry.py#test_registry_section_counts_unchanged"
        status: pass
      - kind: unit
        ref: "tests/models/test_model_info_registry.py#test_spot_pinned_enrichment_values"
        status: pass
      - kind: other
        ref: "command: registry.reload(); 49/49 pretrained validate with max_length -> 'loader validates 49/49'; coverage report dnallm/models/registry.py 100% (>=96 floor)"
        status: pass
    human_judgment: false
  - id: D5
    description: "Probe-derived values trace to measured artifacts or documented constants: every committed value's derivation recorded in .scratch/enrichment_results.json (repo, kind, source_hub, source_field, derivation)"
    requirement: REG-02
    verification:
      - kind: other
        ref: "command: .scratch probe verify — 236 records, 49 pretrained all positive-int max_length, 23/23 lab tokenizer_type ('probe complete')"
        status: pass
    human_judgment: true
    rationale: "The probe results and their derivation records live in gitignored .scratch/ (owner convention, intentionally uncommitted) — a verifier can re-run the probe to reproduce, but the derivation chain itself is not committed to the repo."

duration: 54 min
completed: 2026-10-20
status: complete
plan_head_before: d9f230407d15418fe55e3c171ff3bdd88b211d63
plan_head_after: 7a3331e
---

# Phase 22 Plan 02: Runtime Loader & Enrichment — Enrichment Landing Summary

**tokenizer_type / max_length enrichment committed for all 49 pretrained entries from one-time ModelScope-first metadata probes (never weights), 14 external-based plantcad entries pinned at 8192, inheritance live for 187/187 finetuned — landed as 122-line insertion-only text with both provenance blocks byte-preserved and 12 new enforcing tests.**

## Performance

- **Duration:** 54 min (11:09:29Z – 12:04:18Z)
- **Tasks:** 3 (Task 1 probe tooling uncommitted by design; Tasks 2–3 TDD)
- **Commits:** 2 (measured: `git rev-list --count d9f2304..HEAD`)
- **Files:** 3 committed; 4 gitignored one-off artifacts under `.scratch/`
- **Coverage:** dnallm/models/registry.py 119/119 statements, **100%** (floor 96%; plain `coverage run` + filtered report — pytest-cov still broken locally per the known deferred item)

## Accomplishments

- REG-02 delivered: every pretrained entry carries `max_length` (49/49, positive ints); 46/49 also carry `tokenizer_type` (the 3 gaps are third-party: GENERanno — repo absent from both hubs, OmniNA-66m, DNA_bert_6 — no tokenizer_class in any fetched artifact); all 23 lab pretrained carry tokenizer_type (D-03 mandatory).
- The enrichment diff is pure text insertion: 122 added / 0 deleted lines, keyed on unique `model:` lines (kwt discipline), with the kwt freeze-provenance block and both checkpoint-sha sentinels byte-unchanged and a new enrichment provenance block appended at end of file.
- D-02 inheritance proven live: 187/187 finetuned entries resolve a positive effective max_length through the public loader API; measured finetuned values equaled their bases' everywhere, so zero registry-based overrides were written (no duplication); the 14 external-based plantcad entries carry explicit 8192 (PlantCAD2 paper).
- REG-03 data-side: the 9 pre-existing kwt tests pass unmodified (extension only — 7 new tests appended below a section banner); section counts pinned at 49/187; spot pins hold the measured load-bearing values.
- Probe evidence (uncommitted tooling): `.scratch/enrichment_results.json` covers all 236 repos with per-value derivation (source hub + field, or the documented constant's source); `.scratch/enrichment_gaps.json` lists every gap with a concrete reason — 24 third-party tokenizer_type gaps (acceptable per D-03) plus 1 inherited lab finetuned (see Deviations).

## Task Commits

Each task was committed atomically:

1. **Task 1: one-time enrichment probe** — no commit (gitignored `.scratch/` tooling per owner convention; the COMMITTED OUTPUT in model_info.yaml is the deliverable)
2. **Task 2: scripted insertion + same-commit invariants** — `2b74aa7` (feat; RED observed 5 failed on planned assertions → GREEN 52 passed)
3. **Task 3: loader-level live-inheritance proofs** — `7a3331e` (test; RED proven against the pre-enrichment file via controlled swap: 4 of 5 failed on planned assertions)

**Plan metadata:** committed with this SUMMARY.

## TDD Record

- **Task 2 RED:** `.venv/bin/python -m pytest tests/models/test_model_info_registry.py -q` → 5 failed / 11 passed — `test_all_pretrained_entries_carry_positive_int_max_length`, `test_lab_pretrained_entries_carry_tokenizer_type`, `test_finetuned_max_length_only_as_explicit_override`, `test_provenance_blocks_present_verbatim`, `test_spot_pinned_enrichment_values` each failed on the planned assertion (e.g. spot pin `assert None == 32770`), never import/syntax/collection errors. The two passing new tests are vacuous-or-structural by design (counts 49/187 hold pre/post; no keys exist pre-enrichment for the never-degenerate scan). **GREEN:** after the insertion, 52 passed across both files.
- **Task 3 RED:** the five live-inheritance tests were executed against the pre-enrichment YAML restored from `HEAD~1` (controlled file swap, byte-identical restore verified) — 4 failed on planned assertions (None effective values, absent explicit 8192); `test_loader_validates_every_committed_entry` passes by design (236/49/187 + Pydantic load path hold pre and post). **GREEN:** 57 passed; plan verify prints `187/187 effective max_length resolve`.
- Evidence records: `.planning/phases/22-registry-foundation-runtime-loader-enrichment/tdd-records/22-02-task{2,3}-{RED,GREEN}.json` (real command, exit code, unmodified output tail, semantic assessment).
- **Classifier caveat (same as 22-01):** `gsd_run check tdd-red-evidence` adapters do not parse pytest console output and `workflow.tdd_mode` is false for this plan (`type: execute`) — evidence is the real command output with semantic assessment.

## Files Created/Modified

- `dnallm/models/model_info.yaml` — +122 lines: `tokenizer_type`/`max_length` on 49 pretrained entries (inserted between `model:` and `task:`), explicit `max_length: 8192` on the 14 external-based plantcad finetuned entries, enrichment provenance comment block at end of file
- `tests/models/test_model_info_registry.py` — +136 lines: constants (`EXPECTED_FINETUNED_COUNT`, `PROVENANCE_SENTINELS`, `SPOT_PINNED_MAX_LENGTHS`, `SPOT_PINNED_TOKENIZER_TYPES`, `EXPECTED_EXTERNAL_BASED_COUNT`) and seven new integrity tests; nothing above the section banner touched
- `tests/models/test_registry.py` — the 22-01 pre-enrichment None-contract test strengthened to the post-enrichment inherited-values contract (Task 2 commit, with the data); five 22-02 live-inheritance tests appended (Task 3 commit)
- Uncommitted (gitignored): `.scratch/probe_registry_enrichment.py`, `.scratch/apply_registry_enrichment.py`, `.scratch/enrichment_results.json`, `.scratch/enrichment_gaps.json`, `.scratch/probe_run*.log`

## Decisions Made

- **Measured beats named (hyenadna):** committed 32770 (config.json `max_seq_len`, tokenizer agrees) over the plan's 32768 repo-name figure — the plan's own prohibition on plausible-default values decides the conflict.
- **Tokenizer-side measured fallback:** `tokenizer_config.json:model_max_length` (sentinel ~1e30 ignored) is a measured remote artifact per D-01's own wording and is the ONLY place the mamba family records context — used when no config-side key exists.
- **Documented-constant straggler chain, sourced:** PlantCAD2-Small 8192 / PlantCaduceus_l32 512 / gpn-brassicales 512 (PlantCAD2 paper, bioRxiv 10.1101/2025.08.27.672609 — "PlantCAD2 takes 8,192 bp windows", "512-bp windows used in PlantCAD", "GPN … limited to context windows of 512 bp"); plant-dnamamba-BPE 512 (PDLLMs upstream training-script default, corroborated by all six variant tokenizers); GENERanno-1.2b 8192 (surviving 0.5b sibling's config; the 1.2b-base repo is absent from both hubs); CrossDNA 1024 (tokenizer model_max_length measured from the repo's size-variant subdirectories, which agree; the 140K_len variant 143360 is a separate checkpoint, recorded as a note).
- **GPTNeoX class default rejected:** the `max_length: 1024` in mamba-BPE tokenizer configs is a tokenizer-class template default, not a model-specific value — not committed as model context.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Task 1 probe hard-failed its first run: the plan's extraction spec missed two measured sources**
- **Found during:** Task 1 (probe run 1 exited 2: pretrained max_length 35/49)
- **Issue:** The plan's config-key priority list omits `max_sequence_length` (Omni-DNA-116M carries its 250 there), and the plan extracts max_length from config.json only — the zhangtaolab mamba family, PlantBiMoE, and the CrossDNA size variants record their context solely in `tokenizer_config.json:model_max_length` (D-01 names tokenizer_config.json as a measured source). 14 pretrained repos and 29 lab finetuned repos were unresolvable under the literal spec.
- **Fix:** probe v2 adds `max_sequence_length` to the key list, a tokenizer-side measured fallback, subdir file paths for `chengCCC/CrossDNA_pretrain` (its checkpoints live in size-named subdirectories), a value-level other-hub retry for pretrained repos, and a DERIVED_MAX_LENGTHS table extended with per-entry sourced constants (see Decisions). Run 2: 49/49 max_length, 23/23 lab tokenizer_type, 187/187 finetuned resolvable, gaps third-party-only (+1 inherited lab entry, below).
- **Files modified:** `.scratch/probe_registry_enrichment.py` (uncommitted tooling)
- **Verification:** the plan's Task 1 verify command passes and prints `probe complete: 49/49 max_length, 23/23 lab tokenizer_type`.
- **Committed in:** n/a (gitignored one-off tooling; the committed values trace through it)

**2. [Rule 1 - Bug] hyenadna spot pin corrected from the plan's 32768 to the measured 32770**
- **Found during:** Task 1 smoke probe (before any commit)
- **Issue:** the plan pins `LongSafari/hyenadna-small-32k-seqlen-hf` at 32768 (repo-name figure); the measured `config.json:max_seq_len` is 32770 and the tokenizer's `model_max_length` agrees (32770 = 32k + 2 special tokens).
- **Fix:** committed and spot-pinned the measured 32770 — committing 32768 would violate the plan's own prohibition on plausible-default values when a measured artifact exists.
- **Files modified:** `dnallm/models/model_info.yaml`, `tests/models/test_model_info_registry.py` (SPOT_PINNED_MAX_LENGTHS)
- **Verification:** `test_spot_pinned_enrichment_values` green; derivation recorded in `.scratch/enrichment_results.json`.
- **Committed in:** 2b74aa7

**3. [Rule 1 - Bug] Task 2 commit includes tests/models/test_registry.py (plan acceptance said "exactly two files")**
- **Found during:** Task 2 GREEN gate
- **Issue:** the plan's Task 2 verify requires BOTH registry test files green, but the 22-01 test `test_effective_value_live_registry_pre_enrichment_is_none` asserts None effective values over the live registry — it goes red the moment the data lands, and its file is not in the plan's Task 2 file list. The two acceptance criteria cannot hold simultaneously.
- **Fix:** strengthened that test in the same commit as the data (renamed `test_effective_value_live_registry_post_enrichment_inherits_from_base`, asserting the real inherited 512 / DebertaV2Tokenizer) — the resolution the prior-wave note prescribed ("prefer strengthening it to assert real inherited values over deleting coverage"), keeping the suite green at every commit (same-change pytest rule).
- **Files modified:** `tests/models/test_registry.py`
- **Verification:** 52 passed across both files at commit 2b74aa7; the 9 kwt tests remain unmodified.
- **Committed in:** 2b74aa7

### Plan-Expectation Notes (no code impact)

- **One lab finetuned gap is inherited, not blocking:** `zhangtaolab/plant-dnamamba-BPE-promoter` carries no measurable max_length on either hub (both hubs' tokenizer_configs hold only the sentinel; the GPTNeoX class-default 1024 was rejected per the plausible-default prohibition). It is listed in `.scratch/enrichment_gaps.json` with that reason and resolves its effective max_length (512) via inheritance — the D-02 contract the phase enforces (187/187 proven). Fabricating a value to make the gaps file third-party-only would have violated the no-guessed-values prohibition.
- **Zero registry-based finetuned overrides measured:** every probed finetuned value equaled its base's committed value, so no finetuned entry except the 14 external-based received explicit fields — the override-differs test still enforces the invariant for any future override.

---

**Total deviations:** 3 auto-fixed (1 Rule 3 blocking, 2 Rule 1 bug) + 2 expectation notes
**Impact on plan:** All fixes were required for the plan's own success criteria (49/49 measured coverage, honest values, green suite at every commit). No scope creep; the committed data contract is exactly the plan's D-01/D-02/D-03.

## Issues Encountered

- Probe run 1 hard-failed as designed (35/49) — resolved by the sourced extraction extensions in Deviation 1, not by relaxing any gate.
- The 14 plantcad repos turned out to be PEFT adapter checkpoints (no config.json at all on either hub; ModelScope org absent); their explicit 8192 derives from the PlantCAD2 paper via the registry's own base_model pointers (documented constant per the plan's straggler chain for registry-unresolvable bases).
- ruff PT018 composite-assertion findings fixed in both test files during their RED/GREEN cycles (split assertions — same fix class as the kwt precedent).

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Phase 22 complete: REG-01 (22-01) + REG-02/REG-03 (this plan, with 22-01) all delivered. The registry now answers tokenizer and max-context questions with zero network at runtime.
- Phases 23 (pre-flight core) and 24 (`load_preset_model` rewire) can consume `effective_max_length` / `effective_tokenizer_type` / the lab-first helpers immediately.
- Known third-party tokenizer_type gaps (absent keys, by design): GENERanno-eukaryote-1.2b-base, XLS/OmniNA-66m, zhihan1996/DNA_bert_6, and the 21 plantcad adapter repos (their tokenizers ship only with the external bases).

## Self-Check: PASSED

- Files exist: dnallm/models/model_info.yaml, tests/models/test_model_info_registry.py, tests/models/test_registry.py — all FOUND
- Commits 2b74aa7, 7a3331e — both ancestors of HEAD
- TDD gate commits present: `test(22-02)` (7a3331e) and `feat(22-02)` (2b74aa7) in git log
- evaluation-scope (commits-only, plan 22-02): resolved (plan-subjects source)
- Plan verification block re-run at HEAD: 57/57 registry+kwt suites, 527/527 models lane, enrichment commit numstat 122/0, sentinel greps count 1 (PROVENANCE-INTACT), loader validates 49/49, registry.py coverage 100% >= 96 floor

---
*Phase: 22-registry-foundation-runtime-loader-enrichment*
*Completed: 2026-10-20*
