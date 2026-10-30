---
gsd_state_version: "1.0"
milestone: v1.4
current_phase: 28
status: completed
stopped_at: Phase 28 complete — all phases complete
last_updated: "2026-10-30T12:28:32.198Z"
last_activity: 2026-10-30
last_activity_desc: Phase 28 complete
state_head: c08e8eeef7497bb6ccf5c26d52fef33359203f21
progress:
  total_phases: 2
  completed_phases: 5
  total_plans: 9
  completed_plans: 23
milestone_name: Pre-flight UX & Registry-as-Truth-Source
---

Total Phases: 2

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-30)

**Core value:** One terminal flow takes a user from their own data to a validated, model-compatible config, with the model registry as the runtime single source of truth (Track A first) — suite, coverage gate, and CI honest-green throughout.
**Current focus:** Milestone v1.4 complete — Phase 28 (optional extras) closed verified+secured; next: `/gsd-complete-milestone v1.4`

## Current Position

Phase: 28
Plan: Not started
Status: All phases complete
Last activity: 2026-10-30 - Completed quick task 261030-s4k: fix raw_reverse_complement no-op

## Performance Metrics

**Velocity:**
- v1: 13 plans / 3 days · v1.1: 24 plans / 6 days · v1.2: 12 plans / 2 days · v1.3: 22 plans / 6 days + Phase 21 reopen (2026-10-17)
- Recent trend: Stable

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 22 P22-01 | 26 min | 3 tasks | 3 files |
| Phase 22 P22-02 | 54 min | 3 tasks | 3 files |
| Phase 23 P01 | 34 min | 3 tasks | 7 files |
| Phase 23 P02 | 11 min | 3 tasks | 3 files |
| Phase 23 P03 | 18 min | 3 tasks | 2 files |
| Phase 23 P04 | 12 min | 3 tasks | 3 files |
| Phase 23 P05 | 6 min | 2 tasks | 3 files |
| Phase 24 P01 | 12 min | 3 tasks | 2 files |
| Phase 24 P01 | 12 min | 3 tasks | 2 files |
| Phase 25 P12 | 22 min | 2 tasks | 6 files |
| Phase 25 P13 | 8min | 2 tasks | 3 files |
| Phase 26 P01 | 10 min | 3 tasks | 3 files |
| Phase 26 P02 | 6 min | 2 tasks | 2 files |
| Phase 27 P01 | 74 min | 3 tasks | 3 files |
| Phase 27 P02 | 38min | 3 tasks | 6 files |
| Phase 27 P03 | 27min | 3 tasks | 6 files |
| Phase 27 P04 | 19min | 3 tasks | 6 files |
| Phase 27 P05 | 21 min | 3 tasks | 5 files |
| Phase 27.1 P01 | 8 min | 3 tasks | 4 files |
| Phase 27.1 P02 | 6 min | 2 tasks | 4 files |
| Phase 27.2 P01 | 34 min | 2 tasks | 10 files |
| Phase 27.2 P02 | 21 min | 2 tasks | 7 files |
| Phase 27.2 P03 | 21 min | 3 tasks | 10 files |
| Phase 27.2 P04 | 15 min | 3 tasks | 9 files |
| Phase 27.2 P05 | 7 min | 3 tasks | 8 files |
| Phase 28 P01 | 92min | 3 tasks | 5 files |
| Phase 28 P02 | 16min | 2 tasks | 14 files |
| Phase 28 P03 | 11min | 2 tasks | 10 files |
| Phase 28 P05 | 8min | 2 tasks | 7 files |
| Phase 28 P04 | 18min | 2 tasks | 4 files |
| Phase 28 P06 | 10 min | 2 tasks | 5 files |

## Accumulated Context

### Roadmap Evolution

- v1.4 roadmap created (2026-10-20): 5 phases (22–26) honoring the REQUIREMENTS traceability mapping and the research-converged dependency order — registry loader → pre-flight core → `load_preset_model` rewire → TUI → MCP consistency
- Phase 27 added: Transformers 4/5 Model Compatibility Census (owner-directed 2026-10-27; v1.4 line reopened post-archive for this phase — 238-model registry × transformers 4/5 import-level matrix, shim-or-typed-skip disposition)
- Phase 27.1 inserted after Phase 27: Address tech debt: IN-01..IN-06 + census lane live dispatch (URGENT)
- Phase 27.2 inserted after Phase 27: Transformers-5 compatibility remediation: all 8 unsupported presets + support story in docs — owner ruling 2026-10-29: execute ALL fixes inside v1.4 (merge delayed by design), severity major, scope includes the two author-defect repairs (GENA-LM buffer reset, MutBERT forward wrap); measured-green-probe discipline holds (replica-grade until census lane is live post-merge) (URGENT)
- Phase 28 added: Owner chose option 1 (2026-10-29): curated optional extras for deepspeed (GENA-LM-BigBird) and flash-attn (CrossDNA) following the mamba-extra precedent; measured census flips on green only

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. Roadmap decisions (2026-10-20):

- Phase numbering continues from 22 (never restarts); the owner-pinned REQUIREMENTS phase mapping (REG→22, PFC→23, LOAD→24, TUI/PKG→25, MCP→26) overrides coarse-granularity compression — the mapping already matches the dependency order
- Phase 24 is file-disjoint from Phase 23 (both depend only on Phase 22) — wave-parallelizable under single-tree pathspec-commit discipline; Phase 26 is independent of Phase 25 — second parallelization candidate
- Phase 25's FIRST act is CI/extra wiring, not panels: `tui` extra + `test=["dnallm[tui]", ...]` install lines + structure tests + thread-worker seam land atomically, sequenced with PKG-01's one-cycle `full`-alias removal (same install lines)
- No snapshot/golden TUI infra in v1.4 — Pilot-functional tests only (PITFALLS over STACK on the snapshot contradiction); registry loader born read-only; WARN never blocks; offline degrades to UNKNOWN
- Research flags for plan time: Phase 23 and Phase 25 warrant `--research-phase` (P23: HF_ENDPOINT in-process effect spike + ModelScope cache-layout resolver; P25: worker-seam shape, cancel semantics, extra-vs-alias ordering, Pilot wall-time budget). Phase 22 enrichment schema is a plan-time schema decision; Phase 26 consistency scope is an owner call at planning
- Owner constraints encoded roadmap-wide: main frozen (all v1.4 work on branch `v1.4`), ModelScope-first, lab-models-first (`zhangtaolab/*` ranks first in every listing), every `dnallm/` change ships with pytest in the same change, `fail_under=90` holds throughout with per-module ≥96% for new code and no omit entries
- Commits carry no Co-Authored-By trailers (repo convention)
- [Phase 22]: 22-01: TaskInfo.label_names accepts the packaged regression-entry bare-string form (36 entries) via field_validator wrap to a one-element list — published contract stays list[str] | None, matching TaskConfig's regression convention
- [Phase 22]: 22-01: sole-mentioner allowlist pinned to {dnallm/models/registry.py, dnallm/mcp/server.py} (stale :821 integration comment; Phase 26 rewire target)
- [Phase 22]: 22-01: local coverage measured via plain coverage run + filtered report — pytest-cov broken in dev venv by numpy 2.5.3/py3.13 (known deferred item); CI matrix unaffected
- [Phase 22]: 22-02: enrichment committed from measured artifacts only — hyenadna pinned at measured 32770 (not the 32768 repo-name figure), tokenizer model_max_length accepted as a measured source where configs carry no key, GPTNeoX class-default 1024 rejected as a plausible default
- [Phase 22]: 22-02: three unmeasurable pretrained repos resolved via documented constants (PlantCAD2 8192 / PlantCaduceus 512 / GPN 512 from the PlantCAD2 paper bioRxiv 10.1101/2025.08.27.672609; mamba-BPE 512 from the PDLLMs training script; GENERanno 8192 from the surviving 0.5b sibling) — 14 external-based plantcad entries explicit at 8192, zero registry-based overrides (all measured finetuned values equal their bases')
- [Phase 23]: 23-01: report inventory born with three discretionary metadata fields (SplitReport.length_counts exact-truncation channel; DataReport.label_dtype / multilabel_sep) — CONTEXT delegates field tables to implementer discretion
- [Phase 23]: 23-01: list[str] disambiguation — a list whose every element carries a supported data suffix is a file list, otherwise raw sequences; dict-of-splits values follow the same rule
- [Phase 23]: 23-01: label inference = candidate-name match (label/labels/y/target/class/category) then two-column fallback; fasta/txt carry canonical sequence/label names with no guess finding
- [Phase 23]: 23-01: preflight docstrings never name the registry YAML by filename — the Phase-22 sole-mentioner grep guard is substring-based and fails CI on documentation mentions too
- [Phase 23]: 23-02: ListingEntry carries no per-dimension verdicts (frozen 23-01 type has no field; report.py locked) — dimension functions applied via the finetune listing's warn-or-ok task filter instead
- [Phase 23]: 23-02: unknown tokenizer kind never fails charset — a gap cannot prove hard incompatibility (D-05); rejected chars warn CHARSET_NONSTANDARD_CHARS on unknown stances like BPE classes
- [Phase 23]: 23-02: default intent is finetune when only task_type is given; predict without a task degrades to an empty listing, never an error; k-mer budgets classify from entry names first (6mer 517, BPE x3 1536)
- [Phase 23]: 23-03: D-11 source-first hub order implemented via the is_lab_model() proxy (RegistryEntry has no source field) — ModelScope probed first for every entry; source_hub reports modelscope when both hubs carry the repo (Phase 26 review flag)
- [Phase 23]: 23-03: timeouts terminal (unknown/PROBE_TIMEOUT immediately); raising adapters fall through to the other hub; HF cache lookup failures are best-effort misses (HFValidationError on path-like ids), never probe failures
- [Phase 23]: 23-04: registry-derived max_length lands in inference.max_length (validated + runtime-consumed) and the raw finetune section — TrainingConfig has no max_length field (extra=ignore), so the plan's finetune.max_length attribute premise was impossible
- [Phase 23]: 23-04: _dump_yaml writes the raw assembled/patched dict, never normalized sections — zero data loss for the round-trip editor; write only when output_path is explicitly given
- [Phase 23]: 23-04: collect-all fast path — fully valid mappings validate through validate_sections itself; the per-section collect-all (fixed section order, section.field paths) runs only on validation failure
- [Phase 23]: edit_config round-trip input is the raw mapping derived from GeneratedConfig.config (section objects via model_dump) — the editor's documented input contract, verified live
- [Phase 23]: facade completed as pure re-exports (probe_availability/effective_endpoints/clear_probe_memo/generate_config/edit_config); integration suite proves the user story facade-only on happy + severed-network paths
- [Phase 23]: docstring-facade drift guard added — every public name documented in the package docstring must sit in __all__ (probe-verified red)
- [Phase 24]: [Phase 24] 24-01: OQ-1 STRICT — registry entries with task_type='classification' (6 PlantCAD2, third-party) are intentionally unloadable by name under D-05; documented known limitation, pinned by a real-entry test; registry data fix out of scope — Smallest reading of locked D-05; all six entries sit outside every zhangtaolab flow, and widening the mismatch definition to alias entry-side 'classification' would silently change D-05's contract
- [Phase 24]: [Phase 24] 24-01: patched preset configs are built with model_copy(update=...) only — never model_validate/reconstruction (model_post_init regression branch would clobber registry label_names to ['value']); caller's TaskConfig never mutated; returned un-copied when no field differs — Research Pitfall 2 reproduced against live data: 42 regression entries carry descriptive one-element label_names that a revalidation round-trip would destroy
- [Phase 25]: 25-12: task-type vocabulary lives copy-side in TASK_TYPE_GLOSSES; a drift test reads the live TaskConfig schema pattern so the core stays untouched
- [Phase 25]: 25-12: the eight-value suitability explanation is a static legend inside #suitability-body, not a placeholder that vanishes on typing
- [Phase 25]: 25-12: the LoRA target-modules hint is pinned at 226 characters, the measured 80x24 form-viewport ceiling; navigation sentences interpolate the destination screens' own constants
- [Phase 25]: Config error pane gives up its height cap and the side column carries the scroll, so every row the pane holds is laid out
- [Phase 25]: Layout tests assert rendered geometry against Textual's own Content.get_height rather than .content strings
- [Phase 25]: Field hints render under their Input for all twelve hinted fields, with no copy change
- [Phase 26]: MCP consistency checker (26-01): direct yaml.safe_load route; label_names head rule matches num_labels; ERROR findings fatal in both modes; coverage 98pct — Keeps the checker independent of validator schema drift; a None-entry-vs-config-value label claim is real drift; unreadable committed files are breakage not drift; branch test beats sitting on the coverage gate
- [Phase 26]: MCP listing (26-02): status word mapping loaded greater-than enabled greater-than disabled greater-than available; repo-id route before entry-name route; unmatched configured keys visible, never phantom rows; no truncation until 10x registry growth — Loaded wins over enabled in the status column because callability is what the menu promises; both routes keep the registry-name key space; visibility beats silent dropping because the 26-01 checker finding is the durable backstop
- [Phase 27]: 27-01: census probes execute each fetched snapshot dir as a synthetic package (relative + absolute sibling imports resolve like transformers dynamic modules); probe paths stay absolute without symlink resolution (HF snapshots are extensionless blob symlinks)
- [Phase 27]: 27-01: load layer passes torch_dtype (plan-literal) with a kwarg-free retry + fp32 cast for 4.x-era remote from_pretrained implementations that forward unknown kwargs to __init__; ModelScope fetches its default revision (commit shas are not fetchable) while the HF route pins the census sha
- [Phase 27]: 27-01: load failures report forward_ok=false (no successful forward ran) with ok=false + error naming the phase, so the D-03 bool gate holds; D-04 giants are a no-op (zero evo rows in the roster, research A6 confirmed)
- [Phase 27]: 27-01: two census-blocked verdicts committed honestly — GenerTeam/GENERanno-eukaryote-1.2b-base repo id absent on ModelScope and HF (direct 404; nearest siblings are 0.5b) and chengCCC/CrossDNA_pretrain ships no root config.json (checkpoints under 8.1M/28.6M/71.6M/519M)
- [Phase 27]: 27-02: census-driven rung scope - only MEASURED failure classes got rungs (apply_chunking_to_forward alias + load-boundary bookkeeping); research-predicted classes (is_torch_fx_available, from_legacy_cache, ROPE) never surfaced in the census and got none (D-06: census is the truth)
- [Phase 27]: 27-02: 6mer/PlantCAD2 load failures were census-tool misroutes (AutoModel loading native classes for task-keyed auto_map checkpoints), fixed by task-appropriate Auto* routing in the tool - recovered-green records carry no disposition by design
- [Phase 27]: 27-02: gena imports restored by _patch_apply_chunking_to_forward but dispositions stay skip - base's checkpoint token_type_ids buffer carries garbage (3.9e12 vs type_vocab_size 2, version-independent defect) and bigbird needs external deepspeed; the rung name rides inside each skip reason
- [Phase 27]: 27-02: _finalize_model_loading is STATIC at transformers 5.17 - the bookkeeping rung's wrapper is shape-adaptive and also installs the legacy tie_weights arity bridge at the same boundary (the skipped-post_init entry leaves both 5.x protocol pieces missing)
- [Phase 27]: 27-03: shim-required-tf5 derives from the D-03 verdict plus rung dependency, not disposition type alone - the final artifact carries only skip dispositions, so the writer records the one shim-dependent green load (PlantCaduceus_l32 via rung 13) in SHIM_RESTORED_LOADS citing 27-02 live evidence; the independent consistency gate encodes the same fact. — Plan mapping table assumed fix dispositions would exist; 27-02 settled gena dispositions as skip (D-03 verdict governs), and the artifact cannot stamp which rung a green load needed. Literal application would empty the shim-required bucket and mislabel PlantCaduceus_l32 as verified-tf5 - a false native-load claim.
- [Phase 27]: 27-03: remote-code-broken covers both failure depths - import-fail skips (evidence = exact failing import) and structurally failed load/forward skips (evidence = structural reason); import-restored-but-still-failing gena entries are remote-code-broken because the registry consumes the D-03 verdict. — Plan mapping table assumed fix dispositions would exist; 27-02 settled gena dispositions as skip (D-03 verdict governs), and the artifact cannot stamp which rung a green load needed. Literal application would empty the shim-required bucket and mislabel PlantCaduceus_l32 as verified-tf5 - a false native-load claim.
- [Phase 27]: 27-04: docs-page regeneration date is a pinned data-date constant (DOCS_DATA_DATE), never a wall-clock stamp - a wall clock would break the byte-identical regeneration gate on any later day; the constant is bumped alongside census re-runs
- [Phase 27]: 27-04: bare-symbol failing imports take the explanatory evidence clause (no 'fails at olmo' cells) and census-blocked reasons are mapped to plain-language repository reasons - raw local cache paths in the evidence never reach the public page
- [Phase 27]: 27-04: --verify composes with --run (world-drift oracle on the dispatch lane) and runs dry on the committed artifact (offline unit-testable path); VERSION DRIFT - installed transformers vs committed tf_compat_tested - fails the run by itself, never a silent green
- [Phase 27]: 27-05: census-lane mamba kernels install via the sanctioned sha256-pinned scripts/install_mamba.sh (pip bootstrap + TUNA index) between the .[full] install and the census run - never the plain .[mamba] extra (uv source build fails on this box); pinned offline by TestCensusLaneInstallSequence (ordering, exact invocation, no masking, D-08 posture; deliberate-red proven against the pre-change workflow text)
- [Phase 27]: 27-05: broken-row docs cells lead with one of four honest per-class plain sentences (damaged-files / own-defect / external-package / library-change) keyed on evidence substrings most-specific-first, fail-closed ValueError on unmapped shapes; technical cause demoted to a 100-char word-boundary parenthetical suffix; gate re-derives independently (checker independence); DOCS_DATA_DATE untouched (presentation-only, no census re-run)
- [Phase 27]: 27-05: G-27-1 validated by executing the lane's exact command on the runner-box replica venv (49 matched / 0 UNEXPECTED / exit 0 / 302 s wall-clock) - the previously-disagreeing mamba family all matched (PlantCaduceus_l32 rides the seeded disposition to shim-required-tf5); live dispatch stays deferred to the post-v1.4->main-merge operational act (GitHub registers dispatchable workflows from the default branch only)
- [Phase 27.1]: 27.1-01: docs_bound_technical_fragment and the gate's _bound_fragment cut back to a bracket-balanced prefix at max(rfind("("), rfind("[")) then rstrip("(:[,; ") -- the reviewer's rstrip-only fix was proven insufficient against the real DNABERT-2 evidence; the planned passes also normalize the MutBERT (unclosed paren) and GENA-LM-BigBird (trailing comma) suffixes, and the page stays byte-identical on re-emission
- [Phase 27.1]: 27.1-01: docs_blocked_reason fails closed with distinct ValueErrors (empty evidence; unmapped shape), mirrored by pytest.fail in the gate plus a registry-side pin over every committed census-blocked row -- raw census text and local cache paths can never reach the public page
- [Phase 27.1]: 27.1-01: the summary-sentence pin rebuilds the complete sentence from Counter over tf_compat_status in the independent gate -- no hardcoded counts, no generator import, and a mutated-count negative probe proves the pin discriminates
- [Phase 27.1]: 27.1-01: the pre-existing repo-wide ruff format failure on docs/user_guide/preset_loading.md (untouched; last changed b35f64d) is out of scope and recorded in the phase deferred-items.md + WINDOWS ledger; the plan's four touched files are format-clean under pinned ruff 0.17.1
- [Phase 27.1]: 27.1-02: load-boundary receiver selection resolves kwargs model/self before any positional shape -- a class-shaped positional never receives model bookkeeping, and no receiver delegates unchanged (IN-04)
- [Phase 27.1]: 27.1-02: census lane keeps the runner-provisioned uv (skip-if-present, command -v uv before curl) -- the wiped-box curl|sh fallback stays unpinned as accepted residual exposure T-27.1-06, and no UV_VERSION knob exists (IN-05)
- [Phase 27.1]: 27.1-03: owner selected option-a (minimal byte-identical workflow registration commit) -- the direct push to main was rejected by branch protection (GH006: PR required, 1 approving review, 2 required checks, enforce_admins=true); the single-file commit d661998 is preserved on wip/27.1-03-census-registration, nothing pushed
- [Phase 27.1]: 27.1-03: owner then chose defer-dispatch -- DISP-01 moves to the v1.4->main milestone merge (workflow lands naturally with the merge PR) plus exactly one post-merge 'gh workflow run census_compat.yml --ref v1.4' operator step; no main mutation and no dispatch now; plan summarized as halted
- [Phase 27.1]: Preserved registration commit d661998 predates the CR-01 fix (76ea80f) and lacks the skip-branch GITHUB_PATH write — never use it as the registration source; the v1.4-to-main merge lands the correct workflow bytes. — Verifier evidence-accuracy check (27.1-VERIFICATION.md): byte-identity with origin/v1.4 no longer holds after the CR-01 fix.
- [Phase 27.2]: 27.2-01: load-boundary author-defect repairs run POST-delegation - transformers 5's own _finalize_model_loading materializes non-persistent meta buffers (_move_missing_keys_from_meta_to_device inside it), so a pre-delegation repair sees an unreadable meta tensor and never fires; the shared wrapper hosts MutBERT in 27.2-02
- [Phase 27.2]: 27.2-01: gena reset is torch.zeros(position_ids.size(), dtype=long) on the buffer's device, defect-detected (value outside [0, type_vocab_size)) per D-08 - never a version gate; SHIM_RESTORED_LOADS carries gena ahead of the committed flip while the independent registry mirror holds until 27.2-03
- [Phase 27.2]: 27.2-01: registry writer is update-capable - changed canonical tf_compat triples are replaced in place (reverse position order, no duplication, non-canonical layouts fail closed) and merge_records drops a prior skip disposition on fresh green load+forward, so a green re-probe can never keep publishing remote-code-broken
- [Phase 27.2]: 27.2-02: MutBERT repair wraps the defective nested forward per-instance with call-time word_embeddings resolution, and DNABERT-2 shims are installed pre-construction by wrapping transformers.dynamic_module_utils.get_class_in_module - the census load program arms the same installer for the exact DNABERT-2 repo after source acquisition and before AutoConfig, so one implementation serves both paths (D-05 file choice preserved).
- [Phase 27.2]: Both SHIM_RESTORED_LOADS entries (MutBERT, DNABERT-2) land ahead of the committed flip per the plan-01 interim discipline: the committed skip dispositions still govern the committed artifact, so the census table and the independent registry mirror intentionally differ until plan 27.2-03's measured write-back.
- [Phase 27.2]: 27.2-03: CrossDNA lands as a MEASURED remote-code-broken row (authored external-package skip), not green: the FULL snapshot ships 8.1M/config.json so the handler resolver selects the preferred 8.1M checkpoint (the research reading came from a code-only snapshot fetched without nested configs), and 8.1M's remote modeling_crossdna.py constructs fla.layers.attn.Attention, which demands the separately compiled flash-attn package dnallm does not install — the census verdict flipped census-blocked -> import-ok with the real load failure recorded.
- [Phase 27.2]: 27.2-03: the census import probe's auto_map jobs resolve against the RESOLVED config directory, not the repo root — resolving only read_config left CrossDNA census-blocked (auto_map values do not resolve to fetched .py files); census_entry now threads the resolved dir into _remote_record.
- [Phase 27.2]: 27.2-03: the final distribution is 19 verified / 21 import-verified / 5 shim-required (PlantCaduceus + GENERanno + GENA-LM + MutBERT + DNABERT-2) / 4 remote-code-broken (GPN, Omni-DNA, GENA-LM-BigBird unchanged + CrossDNA measured) / 0 census-blocked; --verify reports 49 matched, the page re-emits byte-identically, and disposition pins now agree with the green-drop rule (a green load+forward supersedes historical previous_status failure signals).
- [Phase 27.2]: 27.2-04: the census generator reads shared DOCS_STATUS_CELLS through a lazy _docs_status_cells() helper, not a module-level import - census_compat must stay stdlib-only at module load so loading the tool can never import hub clients while the mirror endpoint env var is set (WR-02 loader guard)
- [Phase 27.2]: 27.2-04: yaml.safe_load cannot parse mkdocs.yml (pre-existing !!python/name: emoji tags, lines 281-282) - the parse leg of the docs verify runs with yaml.UnsafeLoader, the loader mkdocs config files require; strict mkdocs build is the independent structural check
- [Phase 27.2]: 27.2-04: dnallm/models/support.py is the single shared support-word source (five exact status words, three repo overrides, registry.resolve base inheritance with cycle-safe untested fallback, 100% coverage); the static guide's section order (families before the compact table) keeps the plan's GENA-LM span probes unambiguous
- [Phase 27.2]: 27.2-05: ListingEntry.support is an additive frozen field defaulting to 'untested'; _build_listing is the only TUI-path derivation point and calls support_word(entry) - no per-surface mapping.
- [Phase 27.2]: 27.2-05: the TUI Support column is appended LAST so _AVAILABILITY_COLUMN stays 4 - refresh, on-demand, and probe writers keep writing only the availability cell.
- [Phase 27.2]: 27.2-05: the MCP text line appends support as a fifth pipe segment after the unchanged callability ladder; row keys, envelope, counts, and client signatures stay additive-only.
- [Phase 27.2]: 27.2-05: pinned ruff 0.17.1 caught a trailing-newline rule the repo hook's ruff 0.16.10 does not; fixed in a style commit, repo-wide check clean.
- [Phase 28]: 28-01: fla promoted to base (D-09) — PlantHelixSeek out-of-the-box; fla extra deleted, all slims, CI rewired atomically; curated extras deepspeed >=0.19.6,<0.19.7 and flash-attn >=2.8.3.post2,<2.9 (no markers, no meta-extras)
- [Phase 28]: 28-01: flash-attn installer carries TWO occurrence-counted patches — sm_121 gencode branch AND four -std=c++17->c++20 flips (torch>=2.13 ATen C++20 floor, measured under lane torch 2.14.1+cu130); 64-min build at MAX_JOBS=8, kernel smoke green on GB10
- [Phase 28]: 28-01 MEASURED BLOCKER: BigBird forward structurally impossible — deepspeed 0.19.6 sparse LUT needs triton libtriton.superblock (absent in triton 3.6.0 AND 3.8.0; torch>=2.4 implies triton 3.x); extra greens the LOAD (old ModuleNotFoundError fixed); Plan 02 BigBird flip awaits owner ruling (options A crossDNA-only / B new semantics / C drop); CrossDNA fully green (load+forward (1,64,5) cuda) and NOT blocked
- [Phase 28]: 28-01: sparse BigBird needs probe seq_len % 64 == 0 — committed census LOAD_SEQUENCE_LENGTH=32 can never forward it; extras evidence measured at length 64 via one-line scratch tool copy (committed tool untouched); future extras lane must probe BigBird at a 64-multiple or the divisibility error masks the kernel verdict
- [Phase 28]: [Phase 28] 28-01 halt resolved by owner ruling A (2026-10-29): CrossDNA-only extra-required-tf5 flip in Plan 02; BigBird keeps its committed skip disposition with evidence naming the load-green/forward-red split (deepspeed extra greens load; forward structurally impossible via removed triton libtriton.superblock API — deepspeed 0.19.6 sparse LUT, symbol absent in triton 3.6/3.8, no installable resolution under torch>=2.4). No new status vocabulary. — Research A2 pre-authorized the contingency: measured forward-red means no disposition flip. Option A is the smallest honest reading, consistent with Phase 27 census-verdict-governs discipline; the load-green fact rides in evidence text only. CrossDNA flip, extras lane, and CrossDNA docs proceed unchanged.
- [Phase 28]: 28-02: owner ruling A executed - CrossDNA-only extra-required-tf5 flip (writer: exactly 2 replacements; 49 matched; distribution 19/21/5/1/3/0); BigBird keeps skip with split evidence naming deepspeed 0.19.6 + _patch_apply_chunking_to_forward + the load-green/forward-red split; docs lead reclassified library-change via new lockstep needle
- [Phase 28]: 28-02: BigBird support word stays 'needs deepspeed' (plan-literal works-word withheld - forward measured structurally dead, would overclaim and break out-of-plan pins); CrossDNA gains 'works with the flash-attn extra'; TestDispositionCompleteness vocabulary grows to fix/skip/extra with green-record extra judgment allowed
- [Phase 28]: 28-02: disposition.extra branch ordered before skip AND green so the extras-green seed and default-env failure shapes map to one extra-required-tf5 verdict; guidance-marker exception attribute (_dnallm_missing_extra_guidance) lets specific ImportErrors survive broad except-Exception wraps
- [Phase 28]: 28-03: ruling A governs the static guide — BigBird keeps 'needs deepspeed' in legend/summary/table; the verify-required phrase 'works with the deepspeed extra' appears only inside the honest split sentence quoted from the generated page (load works with the deepspeed extra but forward is blocked by an upstream-dead Triton API)
- [Phase 28]: 28-03: PlantHelixSeek guide section support word is 'works' (verified-tf5) — 'built-in' describes the dependency in the four-row summary only, never a new status vocabulary
- [Phase 28]: 28-03: notebook fla retirement = one-line occurrence-asserted text swap per surface (guard hint '.[fla]' to '.[full]'/'[full,notebook]', md fences drop ',fla'), byte-preserving, no JSON round-trip; sync gate holds the 24-pair baseline
- [Phase 28]: 28-05: all six remaining installation surfaces tell the base-provided fla story (README table + built-in statement, both getting-started guides, honesty matrix, FAQ fix, Windows KDA runbook); retirement is named everywhere, never silenced
- [Phase 28]: 28-05: residue-gate companion — test_extras_guard sanctioned entries track reworded README/troubleshooting lines in the same commits that reword them (28-01 gate-move precedent)
- [Phase 28]: 28-04: extras lane is ruling-A shaped end to end — gate 1 asserts CrossDNA fully green + BigBird load-green/forward-red AT the superblock cause (pre-ruling 'both green' wording impossible), gate 2 is the committed --verify over the 49-row artifact (CrossDNA extra-required-tf5 / BigBird remote-code-broken); probe runs through a runtime-generated diff-proven length-64 census copy, committed tool pinned never --run in this lane
- [Phase 28]: 28-04: seed-strip discipline — the lane nulls the two seeded loads before probing (apply_loads carries a prior passed load instead of re-fetching), so every run re-measures; ci.yml reaches the lane only via dispatch-gated workflow_call with no schedule arm, extras_compat.yml owns the 04:00 UTC cron (exactly-once nightly + dispatch); first live dispatch stays a post-merge operator step
- [Phase 28]: BigBird split-aware lead class (load-works-prediction-blocked) keyed on the existing upstream-dead-Triton needle; README special-model table gains BigBird + CrossDNA rows with guide anchors; G-28-2 closed with ruling A intact
- [Phase 28]: api-coverage verify:pre gate waived for Phase 28 (owner ruling 2026-10-30, 'Waive & record'): the gate's surface scan matched the word 'api' from the phrase 'upstream-dead Triton API' — a library programming interface, not an external HTTP API integration. Phase 28 integrates no external APIs; no COVERAGE.md owed. Recorded so future gate hits on this phase are traceable to this waiver. — Keyword false positive: 'Triton API' names a library interface in census evidence text; ai-integration capability gate cannot distinguish it from an HTTP API integration.

### Pending Todos

- [2026-10-29] [general] "Transformers-5 compatibility remediation campaign: all 8 unsupported presets + support story in docs" — [todo file](.planning/todos/pending/2026-10-29-transformers-5-compatibility-remediation-campaign-all-8.md)

Legacy carried ledger (owner-scoped, outside active requirements, not todo-system items): delete unreferenced doc_mocks.py entries; README:345 + server.py:97 `config/` path pattern; quick_start "NER" task-type line; MCP client `_call_tool` naming mismatch.

### Blockers/Concerns
- [Phase 27, owner-scoped] 3 Info code-review findings open by scope decision + cosmetic UI suggestions (27-REVIEW-DISPOSITION 6/10 open incl. carried rows; 27-UI-REVIEW) — no must-have violated
- [Phase 27, deferred operational act] one live `gh workflow run census_compat.yml` dispatch owed after the v1.4→main merge (workflow registers dispatchables from the default branch only); replica green run is the interim evidence

- [Phase 24 review WR-01, deferred] 24 third-party family names resolve repo ids via hardcoded `source="modelscope"` (hosting unprobed — research A2); Phase 26's MCP listing surfaces the names neutrally (repo_id/task_type/overlay only, no hosting claim) — the unprobed-source concern itself remains open for any future UX that names a download hub

Carried runner-box ops items (cache-quota, `$HOME` cleanup) live in `milestones/v1.1-MILESTONE-AUDIT.md`. v1.3 known gaps carried in the WINDOWS ledger: #25 PyPI trusted-publisher registration, #27 Windows KDA smoke, #26 pip-side alias resolve.
- Phase 27.1 plan 27.1-03 Task 2 HALTED: direct push of the owner-approved option-a registration commit to main was rejected by branch protection (GH006). main requires PRs (1 approving review) + 2 required status checks (coverage-gate (py3.12, fast leg), docs-validation); enforce_admins=true blocks even the owner from direct push. The byte-identical single-file commit d661998 (only .github/workflows/census_compat.yml == origin/v1.4) is preserved: local branch wip/27.1-03-census-registration (nothing pushed); the temp worktree was removed during close-out (branch alone anchors the commit). **RESOLVED 2026-10-28 by owner defer-dispatch:** DISP-01 is deferred to the post-v1.4→main-merge operator step (the Phase 27 deferred operational act above); plan 27.1-03 summarized as halted; no main mutation and no dispatch occurred.
- Phase 28 (28-01): BigBird (GENA-LM-BigBird) forward measured structurally impossible in any dnallm env — deepspeed 0.19.6 sparse kernel requires triton libtriton.superblock (removed before triton 3.x; verified absent in 3.6.0 + 3.8.0). **RESOLVED 2026-10-29 by owner ruling A:** CrossDNA-only extra-required-tf5 flip (28-02); BigBird keeps skip with the load-green/forward-red split evidence; the 28-06 gap plan made every public surface agree with that split.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261024-dtw | Relax mamba Predictor device guard for Mac MPS (cuda/mps passthrough, fp32-only kept) + real-device darwin MPS smoke + FAQ cpu-only fix | 2026-10-24 | 9952ddc | [261024-dtw-relax-mamba-device-guard-in-predictor-to](./quick/261024-dtw-relax-mamba-device-guard-in-predictor-to/) |
| 261025-e4v | mamba-ssm c++20 patch pipeline — sha256-pinned sdist patch+build pipeline in scripts/install_mamba.sh, CI test-mamba/example-nightly routing, docs sync | 2026-10-25 | ef91f8d | .planning/quick/261025-e4v-mamba-ssm-c-20-patch-pipeline-rework-scripts-install-mamba-s |
| 261025-e4w | torch ceiling raise — pyproject.toml seven torch bounds <2.12 to <2.15, one commit citing GitHub run 40140203118 + dependabot cycle-record comment | 2026-10-25 | f00e90d | .planning/quick/261025-e4w-torch-ceiling-raise-pyproject-toml-seven-torch-bounds-2-12-t |
| 261025-e4x | v1.4 lazy-facade shim regression — dnallm/_compat_activation.py meta-path lazy activation restores the import-dnallm compat contract; red-to-green regression tests; 100% module coverage | 2026-10-25 | fac22a0 | .planning/quick/261025-e4x-v1-4-lazy-facade-shim-regression-plain-import-dnallm-must-ag |
| 5 | bump version to 1.4.1 (pyproject + version.py + changelog; version-contract 19 tests green) | 2026-10-25 | baf3445 | — |
| 261025-vvv | Fix stale docs reference in README.md | 2026-10-25 | 967708e | /home/forrest/Github/DNALLM/.planning/quick/261025-vvv-fix-stale-docs-reference-in-readme-md |
| 261025-vvw | Add regression test for config loader | 2026-10-25 | c0c6476 | /home/forrest/Github/DNALLM/.planning/quick/261025-vvw-add-regression-test-for-config-loader |
| 261026-aij | Recalibrate example-nightly D-03 census assertion 216/217 -> 222/223 on v1.4 (+6 from 0885416) + dated 09 rollup note + D-12 path refresh | 2026-10-26 | 73f85bd | [261026-aij-recalibrate-the-example-nightly-d-03-cen](./quick/261026-aij-recalibrate-the-example-nightly-d-03-cen/) |
| 261026-b34 | Per-lane workflow_dispatch selection for nightly lanes (run_coverage_nightly/run_example_nightly/run_test_mamba booleans, default true) + workflows README dispatch docs | 2026-10-26 | 9bf4b54 | [261026-b34-add-per-lane-workflow-dispatch-selection](./quick/261026-b34-add-per-lane-workflow-dispatch-selection/) |
| 261026-g13 | Drop GitHub-side uv cache steps from self-hosted nightly lanes (27GB > 10GB cap + stalled uploads, local ~/.cache/uv persists) + record run 40247981840 census receipts in 09 rollup | 2026-10-26 | c626966 | [261026-g13-disable-github-side-uv-cache-save-on-sel](./quick/261026-g13-disable-github-side-uv-cache-save-on-sel/) |
| 261028-vgn | Fix all 18 ty@0.0.84 diagnostics in dnallm/tui/workers.py: 16x error[unresolved-attribute] "Self@<worker> has no attribute post_message" across collect_env/validate_data/generate_config/edit_config/check_compatibility/local_availability_batch/probe_availability/download workers, plus 1x warning[unsupported-base] tqdm-union class base at line 258 and its fallout; repo rule: real fixes or line-level "# ty: ignore[code]" pragmas for library dynamics only, no severity overrides, no [tool.ty] excludes | 2026-10-28 | aeb34fa | /home/forrest/Github/DNALLM/.planning/quick/261028-vgn-fix-all-18-ty-0-0-84-diagnostics-in-dnallm-tui-workers-py-16 |
| 261028-vgo | Fix 7 ty@0.0.84 error[invalid-assignment] diagnostics at dnallm/preflight/config_gen.py:190 (union-typed values assigned to DNALLMConfig TypedDict keys task/inference/finetune/lora/ia3/sweep/vep); single root cause, prefer a typed construction over per-line ignores | 2026-10-28 | b37017a | /home/forrest/Github/DNALLM/.planning/quick/261028-vgo-fix-7-ty-0-0-84-error-invalid-assignment-diagnostics-at-dnal |
| 261028-vgp | Fix 3 ty@0.0.84 diagnostics in dnallm/utils/transformers_compat.py lines 1597-1601: error[unresolved-attribute] named_children and error[invalid-assignment] all_tied_weights_keys on a Protocol-with-one-member model type, plus warning[unused-ignore-comment] on the stale pragma at 1601; these lines are 27.1-02 (IN-04) load-boundary code, keep behavior identical | 2026-10-28 | e93f067 | /home/forrest/Github/DNALLM/.planning/quick/261028-vgp-fix-3-ty-0-0-84-diagnostics-in-dnallm-utils-transformers-com |
| 261028-vgq | Fix 1 ty@0.0.84 error[invalid-assignment] at dnallm/tui/screens/download.py:291: DownloadOperation assigned to attribute _worker typed Worker[Any] \| None; align the declared type with what is actually stored | 2026-10-28 | 379d525 | /home/forrest/Github/DNALLM/.planning/quick/261028-vgq-fix-1-ty-0-0-84-error-invalid-assignment-at-dnallm-tui-scree |
| 261028-vgr | Fix 1 ty@0.0.84 error[invalid-argument-type] at dnallm/preflight/availability.py:485: int() called on int \| None; guard or narrow per actual runtime contract | 2026-10-28 | 55d5d3f | /home/forrest/Github/DNALLM/.planning/quick/261028-vgr-fix-1-ty-0-0-84-error-invalid-argument-type-at-dnallm-prefli |
| 261028-vgs | Add v1.4 to .github/workflows/ci.yml on.push.branches following the documented v1.3 entry pattern (2026-10-07 first-exposure lesson comment block) so the matrix legs incl. the ty hard gate run on v1.4 pushes before the v1.4-to-main merge | 2026-10-28 | c96bbb1 | /home/forrest/Github/DNALLM/.planning/quick/261028-vgs-add-v1-4-to-github-workflows-ci-yml-on-push-branches-followi |
| 261029-c95 | Retarget registry GENERanno to lgq12697/GENERanno-eukaryote-0.5b-base with measured census disposition (remote-code-broken: ROPE init KeyError default under tf5.17, evidence pinned at ms commit 0ae1f40f), regenerate compat docs page (7 broken / 1 blocked, DOCS_DATA_DATE 2026-10-29), align broken-count pin 6->7; v1.5 backlog note for lgq12697 bulk ingestion (~130 entries) recorded in SUMMARY | 2026-10-29 | 1aa2364 | [261029-c95-retarget-registry-generanno-to-lgq12697-](./quick/261029-c95-retarget-registry-generanno-to-lgq12697-/) |
| 261030-oix | Fix Phase-28 code-review findings WR-01+IN-01 in scripts/census_compat.py: 300-char status-cell cap + lead-map guard as fail-closed emit ValueErrors, with 2 targeted tests; page byte-identical | 2026-10-30 | e1ed477 | [261030-oix-fix-phase-28-code-review-findings-wr-01-](./quick/261030-oix-fix-phase-28-code-review-findings-wr-01-/) |
| 261030-qxo | Close 4 stale UAT-audit items: flip resolved statuses with evidence (27.1 preset_loading ruff, 05 MCP host/port + interpret executor, 09 test_plot_for_regression) | 2026-10-30 | 007bb56 | [261030-qxo-close-4-stale-uat-audit-items-flip-resol](./quick/261030-qxo-close-4-stale-uat-audit-items-flip-resol/) |
| 261030-r96 | UAT docs truth sweep: 11 prose hits to DNA large language model(s) (6 registry paper titles kept), README uv run pytest to --no-sync (6 commands), phase 10 + 05 README deferred items flipped | 2026-10-30 | 205223a | [261030-r96-uat-audit-docs-truth-sweep-rewrite-dna-l](./quick/261030-r96-uat-audit-docs-truth-sweep-rewrite-dna-l/) |
| 261030-rst | Close stale phase-12 mkdocs strict deferred item: flip to resolved with dated dual evidence (prior fix 47b9bc2 + fresh zero-warning MkDocs 1.6.1 strict build at 8b224ea) + hand-maintained-mirror preservation warning; HBG1/numpy bullets untouched | 2026-10-30 | aab4260 | [261030-rst-fix-marimo-mirror-broken-links-at-the-so](./quick/261030-rst-fix-marimo-mirror-broken-links-at-the-so/) |
| 261030-s4k | Fix DNADataset.raw_reverse_complement no-op: assign ds.map result (data.py one-liner), 4 red-to-green tests incl. non-palindromic fixtures (lane 157 passed), phase 03 rc bullet flipped | 2026-10-30 | c08e8ee | [261030-s4k-fix-dnadataset-raw-reverse-complement-no](./quick/261030-s4k-fix-dnadataset-raw-reverse-complement-no/) |
*Rows archived 2026-10-16: 6 late v1.2 tasks -> milestones/v1.2-quick/ (README-v1.2-late-additions.md); 2 v1.3 spikes -> milestones/v1.3-quick/ (README.md)*

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first. The six 2026-10-11 post-close rows were promoted into v1.3 active scope as PKGH/PUBS/DOCR/WPAR requirement IDs (shipped); rows below remain carried.

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| deferred_items | MOTIF-01 golden fixture — HBG1/BCL11A Fig 4a owner inputs (issue #44; activation fixture-files-only) — now FUTR-04 | owner-deferred (option B) | 2026-10-10 | v1.2 |
| deferred_items | numpy 2.5.x cannot be instrumented by coverage on py3.13 (numpy ceiling call when the 1.26.4 leg retires) | acknowledged | 2026-10-10 | v1.2 |
| deferred_items | old-terminology prose hits outside check surface (pyproject description, 3 test docstrings, ui/ default, root cli/ shims — example/ configs now fall under DOCR-01) | acknowledged | 2026-10-10 | v1.2 |
| deferred_items | uat_gaps carried from v1.1 close (1 item) | acknowledged (prior close) | 2026-10-07 | v1.1 |
| deferred_items | 05: `uv run pytest` resolver failure under uv 0.12.20 (use `--no-sync`) | acknowledged | 2026-10-06 | v1.1 |
| deferred_items | 03: import-time `logs/dnallm.log` sink recreated under pytest cwd | acknowledged | 2026-10-01 | v1 |
| deferred_items | 03: test_timeout.py fixed ~60s cost | acknowledged | 2026-10-01 | v1 |
| deferred_items | `DNADataset.raw_reverse_complement` — RESOLVED: fully implemented at data.py:1013 (verified 2026-10-15 during 19-03 docs sweep; was stale-recorded as a no-op) | resolved | 2026-10-15 | v1 |

## Session Continuity

Last session: 2026-10-30T12:13:34Z
Stopped at: Completed quick task 261030-rst — stale mkdocs strict deferred item closed
Resume file: None

### Prior session record (2026-10-24, Mac side — retained)

(1) Mac dev env rebuilt to the recorded spec — found `.venv` stale (py3.12.8, pre-numpy-2, no textual/scikit-allel); rebuilt py3.13.17 + `.[test]` + dev tools minus pygenometracks; 489 offline tests green, ruff clean. (2) ty FINDING: `v1.4` absent from ci.yml `on.push.branches` → the v1.4 cycle never ran the ty hard gate; faithful reproduction (ty@0.0.84 + pinned stub env) shows 28 diagnostics (18 `tui/workers.py`, 7 `preflight/config_gen.py`, 1 download.py, 1 availability.py, 1 pyfaidx mac-artifact) — expect coverage-gate red on first v1.4→main exposure; textual unpinned in ty-stub-constraints (gate-stability gap). (3) PlantHelixSeek Mac confirmation (per upstream docs/installation.md, owner request): load via ModelScope route green (729/729 params materialized, REG-02 frozen labels hold), CPU forward green ((1,2) logits, pure-PyTorch fallback path — rotated-seq logit delta 0.254, i.e. position-sensitive but NOT KDA-exact per upstream's mask-recovery evidence); MPS forward blocked twice over — remote code's MPS branch imports fla chunk_kda ungueded (ModuleNotFoundError: triton; macOS has no triton wheels, fla 0.5.2 chunk_kda is triton-bound) and past that torch 2.11 carries the upstream-documented MPS SDPA/MLA shape bug (502x1536 vs 1024x512 reproduced) — upstream's verified mac path needs torch>=2.12 (above dnallm's <2.12 ceiling) + `apply_mps_fix.py [--kda_fallback]` patched local checkpoint (outside the registry route). Residual test-design gap: smoke's WR-01 guard (`importorskip("fla")`) passes with fla installed-but-triton-absent; on a hypothetical CPU-only box that would shape-only-green the fallback path. fla uninstalled again — Mac env back to the documented spec (CRE checkpoint stays in ~/.cache/modelscope for future runs). (4) torch>=2.12 experiment (owner question): local venv raised 2.11.0→**2.14.1** — MPS forward now WORKS with no checkpoint patch (max|mps-cpu|=1.3e-5, matching the fallback path; correction to the earlier 2.11-era reading that the patch would be required — the unguarded-kernel crash was torch-2.11-path-specific), CPU logits byte-identical, 496 offline tests green (incl. frozen pairs). Wheel/deps facts: mac arm64 + cu130 both have wheels through 2.14.1; transformers/accelerate/bitsandbytes/peft declare no upper bounds. The pyproject `<2.12` ceiling origin: 5721ec3 (2026-05-08, RTX 5090 relaxation — a forward-looking frontier bound, never a 2.12-was-broken statement). Upgrade risk is the GPU side only: mamba-ssm 2.3.2.post1 + causal_conv1d 1.7.0 are source-built CUDA extensions on the nightly box (no declared torch upper bound — build/runtime compat with 2.12+ unverified) and nightly re-resolves unlocked, so raising the ceiling moves every nightly lane to 2.14.x implicitly; needs one dispatched test-mamba + example-nightly verification before a pyproject change (repo's own never-bump-mid-fire discipline). Mac venv deliberately ahead of the declared range (documented deviation, local-only). Runbook committed: `.planning/research/torch-214-gb10-verification.md`; spike branch `spike/torch-214-verification` prepared (7 places `<2.12` → `<2.15`).
(5) mamba-on-Mac answer (owner question, official-only lens): official mamba-ssm (state-spaces/mamba) has NO mac support and is deepening NVIDIA lock-in (tilelang/nvidia-cutlass-dsl/quack-kernels deps, sdist-only, triton>=3.5 unresolvable on mac); the official transformers Mamba implementation IS pure-PyTorch and mac-capable. Empirical (plant-dnamamba-BPE via generic route): loads as MambaForCausalLM, CPU + MPS forward green, MPS==CPU bitwise (0.000000) — the remote code is a vendored copy of transformers' official modeling_mamba with transformers' own tri-gate dispatch. BUT the checkpoints' own config pins `"use_mambapy": true` → transformers hard-requires mambapy (third-party pscan, officially-gated optional accelerator) with NO eager fallback (verified: uninstall → load fails "use_mambapy is set to True but the mambapy package is not installed"). OWNER RULING (2026-10-24): the transformers implementation is the accepted official path as-is — including its own mambapy gate — so NO checkpoint re-export is needed; mac mamba support is settled green at the pscan tier. GB10 unaffected (mambapy in base deps satisfies the gate; official mamba-ssm kernels serve compute). Fills the 2026-10-23 non-mamba audit gap: mamba family is mac-green.

## Operator Next Steps

- Pre-merge (v1.4→main): resolve the 27 first-party ty diagnostics (workers.py/config_gen.py/download.py/availability.py — see Session Continuity) or the coverage-gate ty hard gate goes red on first exposure; consider adding `v1.4` to ci.yml push triggers per the documented 2026-10-07 first-exposure lesson
- Phase 27.1 execution status: plans 01 and 02 complete; plan 27.1-03 HALTED by owner defer-dispatch (DISP-01 not delivered). After the v1.4→main milestone merge PR lands, run exactly one `gh workflow run census_compat.yml --ref v1.4` and verify 49 matched records, the OK registry-match line, no UNEXPECTED, no VERSION DRIFT, and the `census-fresh-evidence` artifact; the preserved registration commit d661998 stays anchored on `wip/27.1-03-census-registration`
- After Phase 27.1 closes, re-close the v1.4 line and start the next milestone with /gsd-new-milestone
