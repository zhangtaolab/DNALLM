---
phase: 06-model-registry-showcase-data-curation
verified: 2026-10-03T09:48:13Z
status: passed
score: 21/21 must-haves verified
covered_files:
  - .planning/phases/06-model-registry-showcase-data-curation/06-01-PLAN.md
  - .planning/phases/06-model-registry-showcase-data-curation/06-01-SUMMARY.md
  - .planning/phases/06-model-registry-showcase-data-curation/06-02-PLAN.md
  - .planning/phases/06-model-registry-showcase-data-curation/06-02-SUMMARY.md
  - .planning/phases/06-model-registry-showcase-data-curation/06-03-PLAN.md
  - .planning/phases/06-model-registry-showcase-data-curation/06-03-SUMMARY.md
  - README.md
  - dnallm/models/model_info.yaml
  - dnallm/utils/__init__.py
  - dnallm/utils/genomic_coords.py
  - docs/faq/models_troubleshooting.md
  - example/notebooks/plant_helixseek_anno/data/TAIR10_GFF3_chr1_5100001_5300000.gff3
  - example/notebooks/plant_helixseek_anno/data/chr1_5100001_5300000.fas
  - example/notebooks/plant_helixseek_cre/data/TAIR10_DHSs_chr1_5100001_5300000.gff
  - example/notebooks/plant_helixseek_cre/data/chr1_5100001_5300000.fas
  - example/notebooks/plant_helixseek_shared/.gitignore
  - example/notebooks/plant_helixseek_shared/data/TAIR10_DHSs_chr1_14953292_14973291.gff
  - example/notebooks/plant_helixseek_shared/data/TAIR10_DHSs_chr1_5351001_5371000.gff
  - example/notebooks/plant_helixseek_shared/data/TAIR10_GFF3_chr1_14953292_14973291.gff3
  - example/notebooks/plant_helixseek_shared/data/TAIR10_GFF3_chr1_5351001_5371000.gff3
  - example/notebooks/plant_helixseek_shared/data/chr1_14953292_14973291.fas
  - example/notebooks/plant_helixseek_shared/data/chr1_5351001_5371000.fas
  - example/notebooks/plant_helixseek_shared/data/selection.md
  - pyproject.toml
  - tests/models/test_plant_helixseek_fla_kernels.py
  - tests/models/test_plant_helixseek_registry.py
  - tests/models/test_plant_helixseek_smoke.py
  - tests/utils/test_genomic_coords.py

covered_digest: "v2:sha256:2bcf9868d94ae8efe8b3eca9e4a17cf31cd29321ea07a08daa73ed1ba246f4fc"
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "Owner reviews example/notebooks/plant_helixseek_shared/data/selection.md (plan 06-03 declared end-of-phase UAT): selected coordinates' biological plausibility (Chr1:5100001-5300000 for both loci; flanking Chr1:5351001-5371000; intergenic Chr1:14953292-14973291), the tolerance bands ([0.3,1.00] jaccard; >=3 genes; [0.00,0.05] flanking CRE fraction; [0.00,0.1] intergenic genic fraction), and the negative-control observed fractions (0.0325 flanking; 0.0000 intergenic genic; 0.1200 intergenic CRE recorded as evidence-only)"
    expected: "Owner confirms the selected loci are plausible showcase targets, the tolerance bands are acceptable for Phase 7 assertions, and the negative-control fractions demonstrate the intended near-zero behavior; or requests reselection"
    why_human: "No automated test asserts biological plausibility of the chosen loci or the adequacy of tolerance bands — this is a judgment call on empirical data the plan explicitly routed to end-of-phase human review (06-03 <human-check>, SUMMARY coverage item D4)"
---

# Phase 6: Model Registry & Showcase Data Curation Verification Report

**Phase Goal:** Both PlantHelixSeek checkpoints load through the existing generic dnallm route (label order frozen to the checkpoint, transformers-5 compat proven), and the showcase's committed Arabidopsis loci — truth slices, selection rationale, negative control — exist in-repo alongside a shared, unit-tested coordinate/chrom-name normalization helper
**Verified:** 2026-10-03T09:48:13Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

Merged from ROADMAP Phase 6 Success Criteria (3) and PLAN frontmatter truths (deduplicated; plan truths add detail under the roadmap contract).

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SC-1/REG-01: Both checkpoints load through `load_model_and_tokenizer`'s generic task-type route (source=modelscope first), no special handler, no dispatch-chain edits | ✓ VERIFIED | Verifier re-ran `pytest tests/models/test_plant_helixseek_smoke.py` — **2 passed in 34.57s** via ModelScope cache; `git diff 9648a53..HEAD -- dnallm/models/model.py dnallm/models/special/ dnallm/models/modeling_auto.py` is EMPTY (no handler/dispatch edits anywhere in the phase) |
| 2 | SC-1/REG-02: Post-load Anno `id2label == dict(enumerate(frozen 17-BILOU order))`, CRE `== {0:'Not CRE',1:'CRE'}` — asserted against hard-coded upstream constants, never the yaml fed to the load | ✓ VERIFIED | Asserts at test lines 93/121-122 target `CRE_LABELS`/`ANNO_LABELS` imported from `tests.models.test_plant_helixseek_registry` (line 30); both asserts passed in the verifier's own smoke run; fast leg pins yaml == constants (3 tests passed) |
| 3 | SC-1/REG-03: Smoke passes on transformers 5.x with version recorded; failure ladder modelscope → one HF retry → typed skip only after documented manual 4.57 venv attempt | ✓ VERIFIED | Verifier's run printed `transformers_version=5.17.0`, `torch_version=2.11.0+cu130`; `_load_with_fallback` implements the ladder; `environment-unavailable:` prefix registered at tests/expected_skips.yaml:35; manual-venv procedure documented in the smoke file docstring |
| 4 | model_info.yaml gains exactly two finetuned entries, append-only (byte-identical prefix), with committed provenance comments (source lines, constraint facts, checkpoint shas) | ✓ VERIFIED | `head -c 73992` of current file `cmp`-identical to `7ba43f1~1` version (byte-identical prefix OK); tail block carries provenance comments incl. shas CRE 7093de3b…/Anno 6d39386a…; registry test asserts exactly one entry per repo id |
| 5 | Freeze run proved checkpoint constraint facts (Anno id2label = 17-entry placeholder pattern, head [17,512]; CRE no id2label, head [2,512]) before writing | ✓ VERIFIED (committed record) | Constraint facts recorded verbatim in the yaml provenance comments; the scratch run itself is intentionally uncommitted by design — the committed provenance + the `LABEL_`-token guard test (registry test lines 89/103) are the auditable outcome |
| 6 | No intermediate tooling committed: scratch covered by committed `.gitignore` (.scratch/), scratch script never git-added | ✓ VERIFIED | `git ls-files -- example/notebooks/plant_helixseek_shared/.scratch/` → empty; `git check-ignore` covers freeze_registry.py, select_loci.py, probe_fla_separation.py via the committed `.gitignore` (single entry `.scratch/`); tooling exists on disk only |
| 7 | Fast leg collects zero new skips; both smoke tests carry `@pytest.mark.slow` + `@pytest.mark.timeout(1800)` | ✓ VERIFIED | Markers on both test functions (smoke lines 78-79, 106-107); `-m "not slow"` deselects both cleanly (2 deselected); full fast suite: **1705 passed, 1 pre-existing typed skip, 50 deselected, 0 failed** (verifier's own run, 90.49s) |
| 8 | SC-3/SHOW-02 (adjacency): half-open ↔ closed conversions exact, round-trip, length-1 → width-1, touching features → adjacent non-overlapping intervals | ✓ VERIFIED | `test_gff1_to_half_open_boundaries`, `test_touching_features_map_to_adjacent_intervals`, `test_half_open_to_gff1_round_trip` all passed in verifier's run (20/20 green) |
| 9 | SC-3/SHOW-02 (empty): empty/unknown inputs raise ValueError with matchable messages — empty chrom, unknown chrom fetch, empty fetch result, require_nonempty zero-match — never silent empty | ✓ VERIFIED | `test_normalize_chrom_rejects_unknown_forms`, `test_fetch_sequence_guards`, `test_fetch_sequence_empty_result_guard`, `test_slice_gff_rows_require_nonempty` all passed; guards live inside the helpers (genomic_coords.py lines 158-161, 235-236) |
| 10 | SC-3/SHOW-02 (encoding): length = sequence characters; exact string equality after style conversion; ChrC/ChrM pass through untouched | ✓ VERIFIED | `test_normalize_chrom_tair_style`/`ensembl_style` (organelle pass-through asserted) passed; module is pure-stdlib str/int domain — no byte/grapheme ambiguity exists |
| 11 | SC-3/SHOW-02 (ordering): parse_gff_attributes and slice_gff_rows preserve source order, never re-sort | ✓ VERIFIED | `test_parse_gff_attributes_whitespace_order_and_empty`, `test_slice_gff_rows_filters_and_preserves_order` passed; implementation appends in iteration order only |
| 12 | Import purity: importing dnallm.utils(.genomic_coords) never imports pyfastx — lazy inside fetch_sequence only | ✓ VERIFIED | Verifier's live check: `import dnallm.utils; … assert 'pyfastx' not in sys.modules` → surface-ok; `test_module_import_is_pyfastx_free` passed; pyfastx import is function-local (genomic_coords.py line 153) |
| 13 | dnallm/utils/__init__.py re-exports the public surface and extends __all__; full fast suite green, zero new skips | ✓ VERIFIED | Re-export block at __init__ line 9; verifier's full fast suite run: 1705 passed / 1 pre-existing skip / 0 failed |
| 14 | SC-2/SHOW-01: One CRE locus and one Anno locus committed meeting floors (jaccard ≥ 0.3; ≥ 3 genes exon-F1 ≥ 0.8) with observed values + tolerance bands recorded in selection.md | ✓ VERIFIED | selection.md records `jaccard=0.3247` (≥ 0.3), `exon_f1=0.7522`, `genes_above_floor=59` (≥ 3), tolerance-band table, decode + match-rule definitions; data integrity independently re-derived by the verifier (truths 15-19); owner plausibility review routed to Human Verification; Phase 7 SC-3 re-asserts the floors against these artifacts |
| 15 | SHOW-01 (adjacency): truth slices preserve source rows verbatim in source coordinate order — truth never merged | ✓ VERIFIED | Verifier's independent subsequence check against owner-local sources: all slices are verbatim source rows in source order (GFF3 vs TAIR10_GFF3_genes.gff 590,264 rows; DHS vs scratch PlantDHS 39,523 rows); .fas fragments are byte-exact `source[lo-1:hi]` slices of TAIR10_chr1.fas |
| 16 | SHOW-01 (empty): intergenic negative control commits empty truth slices as existing zero-row files; assertion metric is predicted-signal fraction, never jaccard-vs-empty | ✓ VERIFIED | Both `TAIR10_*_chr1_14953292_14973291.*` files committed at 0 bytes (git-tracked); selection.md pre-registers the metric as Anno genic fraction (band ≤ 0.1), CRE fraction marked evidence-only |
| 17 | SHOW-01 (ordering): deterministic candidate ranking (score-desc, coordinate-asc tie-break) with the rule recorded in selection.md | ✓ VERIFIED | selection.md "Ranking" section records the rule and the top-5 table |
| 18 | SHOW-01 (concurrency): curation tooling restart-safe and scratch-confined; interrupted/re-run leaves the committed tree clean | ✓ VERIFIED | Scratch tool has `VERIFY_CACHE = SCRATCH/"verify_cache.json"` restart-safe cache; scoped `git status` on the three plant_helixseek dirs is clean |
| 19 | Each region set commits ≤ 200,000 sequence bases; fragments carry .fas; download intermediates gitignored (clean tree) | ✓ VERIFIED | Verifier independently counted .fas sequence characters: 200000 / 200000 / 20000+20000 — pure ACGTN, exactly matching headers; all 12 tracked paths use safe suffixes (.fas/.gff/.gff3/.md/.gitignore); tree clean |
| 20 | Model verification ran through the dnallm public route only at CRE batch 4 / Anno batch 1 (registry as single source of repo id + label order) | ✓ VERIFIED | Scratch select_loci.py (on disk): `load_model_and_tokenizer(repo_id, config, source="modelscope")` line 532, registry read line 512, `CRE_BATCH = 4` / `ANNO_BATCH = 1` lines 101/110; all six genomic_coords helpers imported and used (21 usages) |
| 21 | 06-03: no intermediate tooling committed — repo receives only outcome artifacts | ✓ VERIFIED | `git ls-files` on the scratch home is empty; committed set = 11 data/rationale artifacts + the 06-01 .gitignore; curation surface clean |

**Score:** 21/21 truths verified (0 present, behavior-unverified)

In-phase owner-directed quick task (not a phase must-have, verified for completeness): flash-linear-attention declared as `fla` extra (`flash-linear-attention>=0.5.2,<0.6`) in pyproject.toml, reachable from `all` (line 128), fla 0.5.2 installed in .venv, and 3 guard tests (`tests/models/test_plant_helixseek_fla_kernels.py`) pass — extra declared with bounded range, reachable from all, `fla.ops.kda.chunk.chunk_kda` callable. Code-review fixes WR-04 (pyfastx mypy override, pyproject line 470) and WR-05 (tomllib 3.11+ collection guard in the fla tests) confirmed in place.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `dnallm/models/model_info.yaml` | Two finetuned entries with provenance | ✓ VERIFIED | CRE binary/2 + Anno token/17 appended byte-identically; provenance comments with shas |
| `tests/models/test_plant_helixseek_registry.py` | Fast-leg structure assertions + frozen constants | ✓ VERIFIED | 3 tests, no network, passed; constants carry committed provenance |
| `tests/models/test_plant_helixseek_smoke.py` | Slow-leg smoke loads with id2label equality | ✓ VERIFIED | 2 tests passed in verifier's own run; slow+timeout(1800) markers; modelscope-first ladder |
| `example/notebooks/plant_helixseek_shared/.gitignore` | `.scratch/` ignore coverage | ✓ VERIFIED | Single entry; check-ignore covers all scratch files |
| `dnallm/utils/genomic_coords.py` | Six shared normalization functions | ✓ VERIFIED | 237 lines, stdlib-only at module level, all six functions substantive |
| `tests/utils/test_genomic_coords.py` | Unit tests on tiny fixtures incl. every guard | ✓ VERIFIED | 20 tests, all passed in verifier's run |
| `dnallm/utils/__init__.py` | Re-export of public surface | ✓ VERIFIED | Import block + alphabetical __all__; pyfastx absent at package import |
| `example/notebooks/plant_helixseek_shared/data/selection.md` | Rationale: floors, bands, observed values, decode rules, audit, provenance | ✓ VERIFIED | All required keys present (jaccard=, exon_f1=, genes_above_floor=, neg fractions, tolerance, audit_budget_ok=true); audit values match disk exactly |
| `example/notebooks/plant_helixseek_cre/data/` | CRE locus .fas + DHS truth slice | ✓ VERIFIED | 200,000 bases byte-exact vs source; 94 verbatim in-bounds DHS rows |
| `example/notebooks/plant_helixseek_anno/data/` | Anno locus .fas + GFF3 truth slice | ✓ VERIFIED | 200,000 bases byte-exact vs source; 1,522 verbatim in-bounds rows |
| `example/notebooks/plant_helixseek_shared/data/` | Negative controls + slices (intergenic zero-row) | ✓ VERIFIED | Flanking (3 DHS + 77 GFF3 rows) + intergenic (both zero-row, tracked) committed |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| smoke test | `dnallm/models/model.py` | `load_model_and_tokenizer(repo_id, cfg, source="modelscope")` | ✓ WIRED | Line 68 inside `_load_with_fallback`; exercised in verifier's passing run |
| smoke test | registry test module | cross-dir import of frozen constants | ✓ WIRED | Line 30; collected and ran (root conftest on sys.path) |
| `dnallm/utils/__init__.py` | `genomic_coords.py` | eager re-export | ✓ WIRED | Line 9; surface-ok live check |
| `tests/utils/test_genomic_coords.py` | `genomic_coords.py` | absolute imports | ✓ WIRED | All 20 tests exercise the module |
| scratch select_loci.py | `genomic_coords.py` | all coordinate/attr math routed through helper | ✓ WIRED | Import line 50; 21 usages across all six functions; no inline conversion bypasses found |
| scratch select_loci.py | `dnallm/models/model.py` | public-route loads | ✓ WIRED | Line 532 |
| scratch select_loci.py | `model_info.yaml` | registry as single source | ✓ WIRED | Line 512-517 |
| selection.md | Phase 7/8 | thresholds/bands/decode for verbatim reuse | ✓ WIRED | Sections "Verification contracts (frozen for Phase 7 reuse)" + tolerance table present |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| .fas fragments (4) | sequence chars | owner-local TAIR10_chr1.fas via fetch_sequence | byte-exact `source[lo-1:hi]` (verifier-verified) | ✓ FLOWING |
| DHS truth slices (3) | rows | PlantDHS TAIR10_DHSs.gff (39,523 rows) | verbatim subsequence in source order | ✓ FLOWING |
| GFF3 truth slices (3) | rows | TAIR10_GFF3_genes.gff (590,264 rows) | verbatim subsequence in source order | ✓ FLOWING |
| selection.md observed values | jaccard / exon_f1 / fractions | real GPU curation run through dnallm route | recorded key=value values consistent with committed truth data (94 DHS rows, 91 genes, 1,696 audit rows all re-derived) | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Both checkpoints load through generic route on transformers 5.x with frozen id2label | `.venv/bin/python -m pytest tests/models/test_plant_helixseek_smoke.py -q -rP` | **2 passed in 34.57s**; `transformers_version=5.17.0`, `torch_version=2.11.0+cu130`; ModelScope cache loads | ✓ PASS |
| Registry structure tests (fast, no network) | `pytest tests/models/test_plant_helixseek_registry.py -q` | 3 passed | ✓ PASS |
| Genomic helper unit suite | `pytest tests/utils/test_genomic_coords.py -q` | 20 passed | ✓ PASS |
| fla guard tests (in-phase quick task) | `pytest tests/models/test_plant_helixseek_fla_kernels.py -q` | 3 passed | ✓ PASS |
| Import purity (pyfastx absent at package import) | `python -c "import dnallm.utils; … assert 'pyfastx' not in sys.modules"` | surface-ok, pyfastx absent | ✓ PASS |
| Full fast suite at HEAD | `pytest tests/ -m "not slow" -q` | 1705 passed / 1 pre-existing skip / 50 deselected / 0 failed (90.49s) | ✓ PASS |
| Registry append-only gate | byte-prefix cmp vs 7ba43f1~1 | byte-identical prefix (73,992 bytes) | ✓ PASS |
| Budget re-derivation | count .fas sequence chars | 200000/200000/20000/20000, pure ACGTN | ✓ PASS |
| Truth-row bounds + verbatim check | subsequence + containment vs owner-local sources | 1,696/1,696 rows fully in-bounds; all slices verbatim source order | ✓ PASS |

### Probe Execution

Not applicable — no `scripts/*/tests/probe-*.sh` declared by the plans; the phase's probe-style evidence (fla separation probe) is documented in selection.md/scratch per the owner tooling constraint.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|---------------------|----------|
| REG-01 | 06-01 | Both entries loadable through generic route, no special handler | ✓ SATISFIED | Verifier's smoke run (2 passed); zero dispatch/special edits in phase diff |
| REG-02 | 06-01 | Frozen label order + post-load id2label equality guard | ✓ SATISFIED | Frozen constants committed; post-load equality against constants passed in verifier's run. Documented interpretation: order frozen from the upstream training script after the freeze run proved the checkpoint config is a LABEL_i placeholder carrying no ordering semantics (06-CONTEXT post-research decision; flagged assumption reviewed at planning) — this satisfies the requirement's intent (the checkpoint's true semantic order) and is guarded against silent permutation on both legs |
| REG-03 | 06-01 | Smoke-load on transformers 5.x dev environment | ✓ SATISFIED | Verifier's run on 5.17.0/torch 2.11.0+cu130 |
| SHOW-01 | 06-03 | Committed loci ≤200kb with truth slices, rationale doc, negative control, gitignored intermediates | ✓ SATISFIED | All 11 artifacts committed and independently re-derived; floors met and recorded; owner plausibility review = the one pending human item |
| SHOW-02 | 06-02 | Shared unit-tested coordinate/chrom helper with non-emptiness assertions | ✓ SATISFIED | 20/20 tests green in verifier's run; guards in-helper; import-pure |

Orphaned requirements: none — REQUIREMENTS.md maps exactly REG-01/02/03 + SHOW-01/02 to Phase 6 and all five are claimed by plan frontmatters (06-01: REG-01/02/03; 06-02: SHOW-02; 06-03: SHOW-01).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | No TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER markers in any phase-modified file; no stub implementations; no empty returns in the helper | - | - |

Open review warnings (06-REVIEW.md, dispositioned in 06-REVIEW-DISPOSITION.md — informational, none is a phase must-have): WR-01 (no CI leg installs `fla` — nightly smokes would run the silent non-KDA fallback) is routed to Phase 9 CI wiring / quick task; WR-02 (skip-contract converts any exception to green skip) and WR-03 (CRLF input to slice_gff_rows) routed as quick-task candidates; IN-01..IN-06 open as info. WR-04/WR-05 confirmed fixed (5d354c9).

### Human Verification Required

### 1. Owner review of selection.md (plan 06-03 declared end-of-phase UAT)

**Test:** Review `example/notebooks/plant_helixseek_shared/data/selection.md`: the selected coordinates' plausibility (CRE+Anno Chr1:5100001-5300000; flanking negative Chr1:5351001-5371000; intergenic negative Chr1:14953292-14973291), the tolerance bands (jaccard [0.3, 1.00]; ≥3 genes at exon-F1 ≥ 0.8; flanking CRE fraction [0.00, 0.05]; intergenic genic fraction [0.00, 0.1]), and the negative-control observed fractions (0.0325 flanking; 0.0000 intergenic genic; 0.1200 intergenic CRE recorded as evidence-only given the pericentromeric location).
**Expected:** Owner confirms the loci are plausible showcase targets, the bands are acceptable for Phase 7 assertions, and the negative controls demonstrate near-zero behavior as designed — or requests reselection.
**Why human:** Biological plausibility of chosen loci and adequacy of tolerance bands are judgment calls on empirical data; no automated test can assert them. The plan explicitly routed this to end-of-phase human review (06-03 `<human-check>`; 06-03 SUMMARY coverage item D4 with `human_judgment: true`).

### Gaps Summary

No gaps. All 21 merged truths verified — including independent behavioral re-execution of both slow-lane smoke loads on transformers 5.17.0, and independent re-derivation of every committed-data guarantee (budget, verbatim truth slices against owner-local sources, in-bounds rows, zero-row negative controls, byte-exact FASTA fragments). The only open item is the plan-declared owner review of selection.md, which routes this phase to human_needed rather than passed. Reproduction of the recorded agreement metrics on a fresh run is Phase 7's explicit job (Phase 7 SC-3 asserts the floors against exactly these committed artifacts), not a Phase 6 gap.

---

_Verified: 2026-10-03T09:48:13Z_
_Verifier: Claude (gsd-verifier)_
