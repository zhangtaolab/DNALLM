---
phase: 06-model-registry-showcase-data-curation
verified: 2026-10-06T16:34:49Z
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
covered_digest: "v3:sha256:ebd5952a578016328cfee2596add0aff93dc80e5e33b90a497977030be3cbc96"
behavior_unverified: 0
overrides_applied: 1
overrides:
  - must_have: "example/notebooks/plant_helixseek_shared/.gitignore provides: Scratch-home ignore coverage (.scratch/) — the phase's one-shot tooling lives beneath it, uncommitted by design"
    reason: "The file was removed post-verification as redundant (quick-261003-ryz IN-06 closure, commit 188a4f6): masked/unmasked git check-ignore proved live that the unanchored root .gitignore:60 `.scratch/` pattern covers every file under the scratch home regardless of extension — disproving the 06-01 plan's claim that root patterns do not cover .py files there. The ignore-coverage function the artifact provided is fully intact at HEAD via the root pattern (verified this pass: check-ignore resolves freeze_registry.py and select_loci.py via .gitignore:60; git ls-files on the scratch home is empty). Keeping a second, redundant ignore file was judged noise."
    accepted_by: "owner (quick-261003-ryz, disposition IN-06 'fixed' in 06-REVIEW-DISPOSITION.md)"
    accepted_at: "2026-10-03T20:28:59+08:00"
re_verification:
  previous_status: passed
  previous_score: 21/21
  gaps_closed: []
  gaps_remaining: []
  regressions: []
---

# Phase 6: Model Registry & Showcase Data Curation Verification Report

**Phase Goal:** Both PlantHelixSeek checkpoints load through the existing generic dnallm route (label order frozen to the checkpoint, transformers-5 compat proven), and the showcase's committed Arabidopsis loci — truth slices, selection rationale, negative control — exist in-repo alongside a shared, unit-tested coordinate/chrom-name normalization helper
**Verified:** 2026-10-06T16:34:49Z (regenerated at HEAD `a949b96`)
**Status:** passed
**Re-verification:** Yes — stale-verification regeneration at HEAD. Originally verified 2026-10-03T09:48:13Z (21/21, one pending owner-review item); that item closed via 06-UAT.md (2026-10-03, result: pass — headline jaccard independently reproduced with bedtools 2.31.1). Since then Phases 7-9 and quick tasks legitimately evolved covered files; this pass re-verified every must-have against the live codebase.

## Goal Achievement

### Observable Truths

Must-haves carried forward from the 2026-10-03 verification (ROADMAP Phase 6 Success Criteria merged with PLAN frontmatter truths), re-verified at HEAD `a949b96`.

| # | Truth | Status | Evidence (at HEAD) |
|---|-------|--------|----------|
| 1 | SC-1/REG-01: Both checkpoints load through `load_model_and_tokenizer`'s generic task-type route (source=modelscope first), no special handler, no dispatch-chain edits | ✓ VERIFIED | Verifier re-ran the slow smoke suite at HEAD: **12 passed in 34.35s** (2 slow loads + 10 fast classification tests), warm ModelScope cache; ladder at smoke.py:177-179 iterates `("modelscope", "huggingface")` calling the unmodified generic loader. `grep -rn "PlantHelixSeek" dnallm/models/` matches nothing outside `model_info.yaml` — still no special handler, no dispatch edits |
| 2 | SC-1/REG-02: Post-load Anno `id2label == dict(enumerate(frozen 17-BILOU order))`, CRE `== {0:'Not CRE',1:'CRE'}` — asserted against hard-coded upstream constants, never the yaml fed to the load | ✓ VERIFIED | Asserts intact at smoke.py:347 (`== {0: CRE_LABELS[0], 1: CRE_LABELS[1]}`) and 385-386 (`== dict(enumerate(ANNO_LABELS))`, `[1] == "B-CDS"`) against constants imported cross-dir from `tests.models.test_plant_helixseek_registry` (line 35); both asserts passed in this pass's smoke run; fast leg pins yaml == constants (3 registry tests green) |
| 3 | SC-1/REG-03: Smoke passes on transformers 5.x with version recorded; failure ladder modelscope → one HF retry → typed skip only after documented manual 4.57 venv attempt | ✓ VERIFIED | This pass's run printed `transformers_version=5.17.0`, `torch_version=2.11.0+cu130`; `_load_with_fallback` implements the ladder with the WR-02-hardened classifier (dnallm-origin OSError/ImportError raise sites now FAIL instead of green-skipping); `environment-unavailable:` prefix registered at tests/expected_skips.yaml:35; manual-4.57-venv procedure in the smoke docstring; `_emit_env()` fires before the fla importorskip guards (IN-01) so skips carry version evidence |
| 4 | model_info.yaml gains exactly two finetuned entries, append-only (byte-identical prefix), with committed provenance comments (source lines, constraint facts, checkpoint shas) | ✓ VERIFIED | `head -c 73992` of the HEAD file `cmp`-identical to `git show 9648a53:...` (pre-phase baseline) — the pre-existing 1,624-line content is still a byte-identical prefix despite the later IN-04 re-quote of the phase's OWN appended block; tail block carries provenance (upstream source lines, constraint facts, shas CRE 7093de3b…/Anno 6d39386a…); registry tests assert exactly one entry per repo id |
| 5 | Freeze run proved checkpoint constraint facts (Anno id2label = 17-entry placeholder pattern, head [17,512]; CRE no id2label, head [2,512]) before writing | ✓ VERIFIED (committed record) | Constraint facts recorded verbatim in the yaml provenance comments (lines 1632-1635); the scratch run stays intentionally uncommitted by design; the committed `LABEL_`-token guard (registry test lines 89/103) plus the fast-leg freeze-order pins are the auditable outcome — all green at HEAD |
| 6 | No intermediate tooling committed: scratch covered by committed ignore, scratch script never git-added | ✓ VERIFIED | `git ls-files -- example/notebooks/plant_helixseek_shared/.scratch/` → empty; `git check-ignore` resolves freeze_registry.py AND select_loci.py via **root `.gitignore:60`** (`.scratch/`, unanchored). Mechanism note: the per-dir `.gitignore` this phase committed was removed post-verification as redundant (IN-06 closure, 188a4f6) after live proof the root pattern covers all scratch files — see override in frontmatter; the tooling-boundary guarantee itself is unchanged and re-proven |
| 7 | Fast leg collects zero new skips; both smoke tests carry `@pytest.mark.slow` + `@pytest.mark.timeout(1800)` | ✓ VERIFIED | Markers at smoke.py:319-320 and 360-361; verifier's full fast suite at HEAD: **1845 passed, 1 pre-existing typed skip, 53 deselected, 0 failed** (97.97s); phase dirs (tests/models + tests/utils, `-m "not slow"`): **572 passed, 0 skipped** |
| 8 | SC-3/SHOW-02 (adjacency): half-open ↔ closed conversions exact, round-trip, length-1 → width-1, touching features → adjacent non-overlapping intervals | ✓ VERIFIED | Boundary/round-trip/touching-feature tests green in this pass's run of tests/utils/test_genomic_coords.py (now 27 tests — grown from 20 by post-verification quick-task tests, all passing) |
| 9 | SC-3/SHOW-02 (empty): empty/unknown inputs raise ValueError with matchable messages — empty chrom, unknown chrom fetch, empty fetch result, require_nonempty zero-match — never silent empty | ✓ VERIFIED | Guards live inside the helpers at HEAD (genomic_coords.py:183-184 empty-fetch raise; 267-268 require_nonempty raise; 66-67, 171, 213, 261); corresponding pytest.raises tests green |
| 10 | SC-3/SHOW-02 (encoding): length = sequence characters; exact string equality after style conversion; ChrC/ChrM pass through untouched | ✓ VERIFIED | normalize_chrom organelle pass-through asserted in both styles; module remains pure-stdlib str/int domain; IN-02 hardening present (bare-numeric branch requires `isascii() and isdigit()`, genomic_coords.py:74) |
| 11 | SC-3/SHOW-02 (ordering): parse_gff_attributes and slice_gff_rows preserve source order, never re-sort | ✓ VERIFIED | Insertion-ordered dict + iteration-order append only; order-preservation tests green; WR-03 hardening present (CRLF terminators stripped, embedded `\r` raises — genomic_coords.py:254-258) |
| 12 | Import purity: importing dnallm.utils(.genomic_coords) never imports pyfastx — lazy inside fetch_sequence only | ✓ VERIFIED | Verifier's live check: `import dnallm.utils; … assert 'pyfastx' not in sys.modules` → surface-ok; pyfastx import is function-local inside the fetch_sequence path branch (line 160); IN-01 hardening present (path branch releases handles + cleans its `.fxi` sidecar); purity + module-identity tests green |
| 13 | dnallm/utils/__init__.py re-exports the public surface and extends __all__; full fast suite green, zero new skips | ✓ VERIFIED | Re-export block at `__init__.py:9-16`, all six names in `__all__` (lines 46/60/68 et al.); full fast suite 1845 passed / 1 pre-existing skip / 0 failed at HEAD |
| 14 | SC-2/SHOW-01: One CRE locus and one Anno locus committed meeting floors (jaccard ≥ 0.3; ≥ 3 genes exon-F1 ≥ 0.8) with observed values + tolerance bands recorded in selection.md | ✓ VERIFIED | selection.md records `jaccard=0.3247` (≥ 0.3), `exon_f1=0.7522`, `genes_above_floor=59` (≥ 3), tolerance-band table, frozen argmax decode + reciprocal-overlap ≥ 0.5 match rule; the headline metric was independently reproduced with bedtools 2.31.1 during the completed owner review (06-UAT.md); Phase 7 SC-3 re-asserts these floors against exactly these artifacts |
| 15 | SHOW-01 (adjacency): truth slices preserve source rows verbatim in source coordinate order — truth never merged | ✓ VERIFIED | Slices re-checked at HEAD: CRE DHS slice 94 rows all Chr1 and all within [5100001, 5300000] (0 out-of-bounds); row counts match the audit record exactly (94 DHS / 1,522 GFF3 / 3 DHS flanking / 77 GFF3 flanking); verbatim-subsequence property originally re-derived against owner-local sources and recorded in the audit + 06-UAT review |
| 16 | SHOW-01 (empty): intergenic negative control commits empty truth slices as existing zero-row files; assertion metric is predicted-signal fraction, never jaccard-vs-empty | ✓ VERIFIED | Both `TAIR10_*_chr1_14953292_14973291.*` files git-tracked at 0 bytes (wc -c confirms); selection.md pre-registers the intergenic metric as Anno genic fraction (band ≤ 0.1), CRE fraction marked evidence-only |
| 17 | SHOW-01 (ordering): deterministic candidate ranking (score-desc, coordinate-asc tie-break) with the rule recorded in selection.md | ✓ VERIFIED | selection.md line 20: "Ranking: score-descending with a coordinate-ascending tie-break (deterministic)" plus the top-5 ranking table |
| 18 | SHOW-01 (concurrency): curation tooling restart-safe and scratch-confined; interrupted/re-run leaves the committed tree clean | ✓ VERIFIED | Scratch select_loci.py (on disk) still carries `VERIFY_CACHE` json checkpointing (lines 94, 544-553) and atomic temp-then-`os.replace` fetches (lines 249, 308); scoped `git status` on the three plant_helixseek dirs is clean at HEAD |
| 19 | Each region set commits ≤ 200,000 sequence bases; fragments carry .fas; download intermediates gitignored (clean tree) | ✓ VERIFIED | Verifier re-counted sequence characters at HEAD: CRE 200,000 / Anno 200,000 / negative 20,000+20,000 = 40,000 — pure ACGT, headers match coordinates exactly; all tracked showcase data files use safe suffixes (.fas/.gff/.gff3/.md/.bedGraph/.gtf/.ipynb); scoped tree clean |
| 20 | Model verification ran through the dnallm public route only at CRE batch 4 / Anno batch 1 (registry as single source of repo id + label order) | ✓ VERIFIED | Scratch select_loci.py at HEAD: registry read lines 512-517, `load_model_and_tokenizer(repo_id, config, source="modelscope")` line 532, `CRE_BATCH = 4` (line 101) / `ANNO_BATCH = 1` (line 110); 21 usages of the genomic_coords helpers, no inline minus-one math |
| 21 | 06-03: no intermediate tooling committed — repo receives only outcome artifacts | ✓ VERIFIED | `git ls-files` on the scratch home is empty; the phase's committed set (11 outcome artifacts from 06-03 + the 06-01 test files + registry) all tracked; Phase 7 later added notebooks/derived files in the same tree — additive, no tooling leakage |

**Score:** 21/21 truths verified (0 present, behavior-unverified)

### Advisory (New Scope, Unevidenced)

Re-verification ran; none found.

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| — | None — no new debt markers, stubs, or unevidenced new-scope findings in any covered file at HEAD | — | — |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `dnallm/models/model_info.yaml` | Two finetuned entries with provenance | ✓ VERIFIED | CRE binary/2 + Anno token/17 (double-quoted per IN-04) appended byte-identically over the pre-phase prefix; provenance comments with shas intact; 3 structure tests green |
| `tests/models/test_plant_helixseek_registry.py` | Fast-leg structure assertions + frozen constants | ✓ VERIFIED | 3 tests green in this pass; constants carry committed provenance; LABEL_ placeholder guard present |
| `tests/models/test_plant_helixseek_smoke.py` | Slow-leg smoke loads with id2label equality | ✓ VERIFIED | 12 passed in this pass's run (2 slow loads on transformers 5.17.0 + 10 fast WR-02 classification tests); slow+timeout(1800) markers; modelscope-first ladder |
| `example/notebooks/plant_helixseek_shared/.gitignore` | `.scratch/` ignore coverage | PASSED (override) | File removed at HEAD (IN-06 closure 188a4f6) after live proof root `.gitignore:60` covers the scratch home; ignore-coverage function re-proven intact via check-ignore — see frontmatter override |
| `dnallm/utils/genomic_coords.py` | Six shared normalization functions | ✓ VERIFIED | All six functions substantive at HEAD (270 lines), stdlib-only at module level; post-verification hardening (IN-01 sidecar cleanup, IN-02 ASCII digits, WR-03 CRLF) present |
| `tests/utils/test_genomic_coords.py` | Unit tests on tiny fixtures incl. every guard | ✓ VERIFIED | 27 tests green in this pass (grown from 20 by quick-task regression tests) |
| `dnallm/utils/__init__.py` | Re-export of public surface | ✓ VERIFIED | Import block + alphabetical `__all__`; pyfastx absent at package import (live check) |
| `example/notebooks/plant_helixseek_shared/data/selection.md` | Rationale: floors, bands, observed values, decode rules, audit, provenance | ✓ VERIFIED | All required keys present (jaccard=0.3247, exon_f1=0.7522, genes_above_floor=59, neg fractions, tolerance, audit_budget_ok=true); audit values match disk exactly |
| `example/notebooks/plant_helixseek_cre/data/` | CRE locus .fas + DHS truth slice | ✓ VERIFIED | 200,000 bases (re-counted); 94 in-bounds verbatim DHS rows (re-checked) |
| `example/notebooks/plant_helixseek_anno/data/` | Anno locus .fas + GFF3 truth slice | ✓ VERIFIED | 200,000 bases (re-counted); 1,522 in-bounds rows |
| `example/notebooks/plant_helixseek_shared/data/` | Negative controls + slices (intergenic zero-row) | ✓ VERIFIED | Flanking (3 DHS + 77 GFF3 rows) + intergenic (both zero-row, tracked at 0 bytes) committed |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| smoke test | `dnallm/models/model.py` | `load_model_and_tokenizer(repo_id, cfg, source=...)` modelscope-first | ✓ WIRED | Line 179 inside `_load_with_fallback`; exercised in this pass's green run |
| smoke test | registry test module | cross-dir import of frozen constants | ✓ WIRED | Line 35; collected and ran |
| `dnallm/utils/__init__.py` | `genomic_coords.py` | eager re-export | ✓ WIRED | Line 9; surface-ok live check |
| `tests/utils/test_genomic_coords.py` | `genomic_coords.py` | absolute imports | ✓ WIRED | All 27 tests exercise the module |
| scratch select_loci.py | `genomic_coords.py` | all coordinate/attr math routed through helper | ✓ WIRED | 21 usages across all six functions; no inline conversion bypasses |
| scratch select_loci.py | `dnallm/models/model.py` | public-route loads | ✓ WIRED | Line 532 |
| scratch select_loci.py | `model_info.yaml` | registry as single source | ✓ WIRED | Lines 512-517 |
| selection.md | Phase 7/8 | thresholds/bands/decode for verbatim reuse | ✓ WIRED | Phase 7 notebooks (committed with executed outputs at HEAD) consume exactly these artifacts and re-assert the floors |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| .fas fragments (4) | sequence chars | owner-local TAIR10_chr1.fas via fetch_sequence | 200,000/200,000/20,000/20,000 pure-ACGT bases, headers match coordinates (re-counted at HEAD) | ✓ FLOWING |
| DHS truth slices (3) | rows | PlantDHS TAIR10_DHSs.gff | in-bounds verbatim rows (94/3/0; re-checked) | ✓ FLOWING |
| GFF3 truth slices (3) | rows | TAIR10_GFF3_genes.gff | in-bounds verbatim rows (1,522/77/0) | ✓ FLOWING |
| selection.md observed values | jaccard / exon_f1 / fractions | real GPU curation run through the dnallm route | recorded key=value values consistent with the committed truth data; headline jaccard independently reproduced with bedtools during owner review | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Both checkpoints load through the generic route on transformers 5.x with frozen id2label | `pytest tests/models/test_plant_helixseek_smoke.py -q -rP` | **12 passed in 34.35s**; `transformers_version=5.17.0`, `torch_version=2.11.0+cu130`; warm ModelScope cache | ✓ PASS |
| Registry structure tests (fast, no network) | `pytest tests/models/test_plant_helixseek_registry.py -q` | 3 passed | ✓ PASS |
| fla guard tests (in-phase quick task) | `pytest tests/models/test_plant_helixseek_fla_kernels.py -q` | 6 passed (grown from 3 by IN-05 exact-bracket-member tests) | ✓ PASS |
| Genomic helper unit suite | `pytest tests/utils/test_genomic_coords.py -q` | 27 passed | ✓ PASS |
| Import purity (pyfastx absent at package import) | `python -c "import dnallm.utils; … assert 'pyfastx' not in sys.modules"` | surface-ok, pyfastx absent | ✓ PASS |
| Full fast suite at HEAD | `pytest tests/ -m "not slow" -q` | 1845 passed / 1 pre-existing typed skip / 53 deselected / 0 failed (97.97s) | ✓ PASS |
| Registry append-only gate | byte-prefix cmp vs `9648a53:dnallm/models/model_info.yaml` | byte-identical prefix (73,992 bytes) | ✓ PASS |
| Budget re-derivation | count .fas sequence chars per region set | 200000 / 200000 / 20000+20000, pure ACGT | ✓ PASS |
| Truth-row bounds | chrom + coordinate check on committed slices | CRE DHS slice 94/94 rows in-bounds; row counts match audit record | ✓ PASS |
| Tooling boundary | `git ls-files` scratch home; `git check-ignore` scratch scripts | empty ls-files; both scripts ignored via root .gitignore:60 | ✓ PASS |

### Probe Execution

Not applicable — no `scripts/*/tests/probe-*.sh` declared by the plans; the phase's probe-style evidence (fla separation probe) is documented in selection.md/scratch per the owner tooling constraint, and the fla dependency is now guarded by committed fast tests (`test_plant_helixseek_fla_kernels.py`, 6 passed).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|---------------------|----------|----------|
| REG-01 | 06-01 | Both entries loadable through generic route, no special handler | ✓ SATISFIED | This pass's smoke run (12 passed); zero PlantHelixSeek references in dnallm/models/ outside the registry |
| REG-02 | 06-01 | Frozen label order + post-load id2label equality guard | ✓ SATISFIED | Frozen constants committed (registry tests green); post-load equality against constants passed in this pass's run. Order frozen from the upstream training script after the freeze run proved the checkpoint config is a LABEL_i placeholder carrying no ordering semantics (06-CONTEXT post-research decision; flagged assumption reviewed at planning) |
| REG-03 | 06-01 | Smoke-load on transformers 5.x dev environment | ✓ SATISFIED | This pass's run on 5.17.0 / torch 2.11.0+cu130; skip ladder now WR-02-hardened (dnallm regressions fail loud) |
| SHOW-01 | 06-03 | Committed loci ≤200kb with truth slices, rationale doc, negative control, gitignored intermediates | ✓ SATISFIED | All 11 artifacts committed and re-verified at HEAD; floors met and recorded; owner plausibility review completed (06-UAT.md, pass) |
| SHOW-02 | 06-02 | Shared unit-tested coordinate/chrom helper with non-emptiness assertions | ✓ SATISFIED | 27/27 tests green in this pass; guards in-helper; import-pure |

Orphaned requirements: none — REQUIREMENTS.md maps exactly REG-01/02/03 + SHOW-01/02 to Phase 6, all five claimed by plan frontmatters (06-01: REG-01/02/03; 06-02: SHOW-02; 06-03: SHOW-01), all marked Complete in the traceability table.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | No TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER markers in any covered file at HEAD; no stub implementations; no empty returns in the helper | - | - |

Code-review posture at HEAD: all 11 findings from the phase code review are dispositioned `fixed` with commit evidence (06-REVIEW-DISPOSITION.md, `open: 0`) — WR-01 (symbol-first model.py anchors, ad6a920), WR-02 (origin-check classifier + 3 new classification pins, 7134aa6/1219f0f), WR-03 (CRLF-robust slicing, 8d6bd3b), WR-04/WR-05 (5d354c9), IN-01..IN-06 (8669164, aaf6308, c19999a, 3662ea5, 7790920, 2a0ba40, 188a4f6). The regression gate over prior-phase test files passed at HEAD (this pass's phase-scoped fast run: 572 passed, 0 skipped).

### Decision Coverage

No trackable decisions in 06-CONTEXT.md (gate skipped cleanly; non-blocking by design).

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| test_plant_helixseek_registry.py | REG-01/02 | 3 | 0 | No | Value (== frozen constants, order-sensitive) | OK |
| test_plant_helixseek_smoke.py | REG-01/02/03 | 12 (2 slow + 10 fast) | 0 in this env | No | Value/behavioral (id2label equality, forward shapes, classifier pins) | OK |
| test_plant_helixseek_fla_kernels.py | fla quick task | 6 | 0 | No | Value (exact bracket member, callable check) | OK |
| test_genomic_coords.py | SHOW-02 | 27 | 0 | No | Value (exact conversions, pytest.raises on every guard) | OK |

Expected-value provenance for the frozen label order: external upstream source (zhangtaolab/PlantHelixSeek `scripts/gene_annotation/train_token_cls.py:78-96`, fetched 2026-10-02) — VALID, not system-under-test-generated. No disabled tests linked to any requirement.

### Human Verification Required

None — the phase's single declared human item (owner review of selection.md plausibility, bands, and negative-control fractions) was completed 2026-10-03 and recorded in 06-UAT.md: result **pass**, with the headline jaccard metric independently reproduced with bedtools 2.31.1 (truth 94 rows == committed slice; pred 51 peaks; jaccard 0.324727 → recorded 0.3247). No behavior-unverified truths remain in this pass.

### Gaps Summary

No gaps. All 21 carried-forward must-have truths verified against the live codebase at HEAD `a949b96`, including fresh behavioral re-execution of both slow-lane smoke loads on transformers 5.17.0 and re-derivation of every committed-data guarantee (append-only prefix, per-region-set base budget, in-bounds truth rows, zero-row negative controls, tooling-boundary ignore coverage). Post-verification evolution of covered files (Phases 7-9 work, the 11 code-review fixes, and the IN-06 removal of the redundant per-dir `.gitignore`) is owner-documented, strengthens or preserves every guarantee, and is recorded via the one frontmatter override.

---

_Verified: 2026-10-06T16:34:49Z at HEAD a949b96_
_Verifier: Claude (gsd-verifier) — stale-verification regeneration_
