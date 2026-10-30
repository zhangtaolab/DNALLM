---
phase: 27-transformers-4-5-model-compatibility-census
plan: 03
subsystem: registry
tags: [transformers, compatibility, census, registry, write-back, model-info-yaml, reg-03, tdd]

# Dependency graph
requires:
  - phase: 27-02
    provides: "tests/models/compat_census.json — the frozen post-shim artifact (49 records, dispositions complete) plus the two census-driven rungs whose dependency the write-back records"
provides:
  - "RegistryEntry.tf_compat_status / tf_compat_evidence / tf_compat_tested — three optional-nullable typed fields; the status is the five-value Literal verified-tf5 / import-verified-tf5 / shim-required-tf5 / remote-code-broken / census-blocked"
  - "The 49 pretrained rows of dnallm/models/model_info.yaml carrying their census verdicts (anchored insertion after each max_length line) + dated provenance block"
  - "scripts/census_compat.py --write-registry [--apply] — dry-run-default, fail-closed, idempotent anchored-insertion writer (committed and re-runnable)"
  - "REG-03 extensions: data-side coverage/mirror/evidence/rung/degenerate/sentinel tests + the independent artifact↔registry consistency gate"
affects: [27-04]

# Actuals (#2632) — chars/4 over the realized diff; commits measured from the ledger.
actuals:
  tokens: 17678
  tasks: 3
  commits: 4
  plan_head_before: "9a818bd31698cd3717dd0220b10b301d96ef5ff0"
  plan_head_after: "0231b4169f2ab82e6ec5e29a70aaddc18cd71edf"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Schema-before-data ordering under extra=forbid: optional-nullable fields with None defaults land first (registry-safe by construction), the yaml data commit follows carrying its own enforcement tests — data never precedes schema"
    - "Rung-dependency table in the writer: the artifact records green loads but not which rung a load needed, so SHIM_RESTORED_LOADS encodes the 27-02 evidence (PlantCaduceus_l32 → rung 13) — the one fact both the writer and the independent consistency gate must carry"
    - "json.dumps as a YAML-double-quoted escaper for evidence scalars (escaping is shared between JSON and YAML double-quoted style), keeping quotes/brackets/colons round-trip safe without a yaml emitter"

key-files:
  created: []
  modified:
    - dnallm/models/registry.py
    - dnallm/models/model_info.yaml
    - scripts/census_compat.py
    - tests/models/test_registry.py
    - tests/models/test_model_info_registry.py
    - tests/models/test_census_compat.py

key-decisions:
  - "Shim-required status derives from the D-03 verdict plus rung dependency, not from disposition type alone: the plan's mapping clause (import-fail + fix + recovered + forward true) fires on zero entries of the final artifact (27-02 settled gena dispositions as skip because their loads still fail), so the single shim-dependent green load — PlantCaduceus_l32, restored by rung 13 — is recorded in SHIM_RESTORED_LOADS with the 27-02 live evidence; mislabeling it verified-tf5 would claim a native load that does not exist without the shim."
  - "remote-code-broken covers both failure depths: import-fail skips (gpn, Omni-DNA — evidence carries the exact failing import) and structurally failed load/forward skips (DNABERT-2, gena x2, MutBERT — evidence carries the structural reason; import restored by rung 12 does not make an entry shim-required when the D-03 bar still fails)."
  - "Import-restored gena entries are remote-code-broken, not shim-required: the registry consumes the D-03 verdict (load + one real forward), and both gena loads fail for version-independent causes (checkpoint token_type_ids buffer garbage; external deepspeed) — the restoring rung stays visible inside each evidence string."
  - "Non-rep mapping is load-evidence-based: a non-representative carrying its own green load maps to verified-tf5 (honest — it was measured); real non-reps carry load null and map to import-verified-tf5 naming their family representative."
  - "tf_compat_tested stamps only 5.17.0 (D-05): the version field records what was ACTUALLY tested; the transformers-4 inference is provenance-comment and docs-layer wording only, never a registry claim."

patterns-established:
  - "Anchored text insertion writer as the only mutation path for model_info.yaml: exact model-line match (repo ids unique across sections, so base_model lines never collide), max_length anchor, idempotency skip on existing tf_compat_status, provenance block appended exactly once"

requirements-completed: ["TBD"]

# Coverage metadata (#1602)
coverage:
  - id: D1
    description: "RegistryEntry schema: three optional-nullable tf_compat_* fields (five-value Literal status, min_length-1 evidence/tested) + honest Attributes docstring; loader validates the vocabulary and serves the fields with None defaults on uncensused entries"
    verification:
      - kind: unit
        ref: "python -m pytest tests/models/test_registry.py -q (47 passed incl. 5 new compat-field tests; RED-first with machine-validated RED_EVIDENCE_OK)"
        status: pass
      - kind: other
        ref: "python -I -c \"import dnallm.models.registry as r; assert len(r.all_entries())==236\" (exit 0 pre- and post-write-back)"
        status: pass
    human_judgment: false
  - id: D2
    description: "--write-registry writer: compat_verdict mapping (fail-closed on unclassifiable shapes), anchored insertion planning, idempotency, comment/provenance preservation, dry-run default, provenance block appended once"
    verification:
      - kind: unit
        ref: "python -m pytest tests/models/test_census_compat.py -q (68 passed incl. 16 new mapping/writer tests over tmp_path fixtures, never the packaged file)"
        status: pass
      - kind: other
        ref: "python -I scripts/census_compat.py --write-registry on the real artifact: 49 planned insertions, exit 0, file untouched (git diff empty)"
        status: pass
    human_judgment: false
  - id: D3
    description: "The write-back data + REG-03 data-side enforcement in one commit: tf_compat_* on exactly the 49 pretrained rows (19 verified / 21 import-verified / 1 shim-required / 6 remote-code-broken / 2 census-blocked), finetuned rows untouched, all sentinels intact once, counts 49/187 unchanged"
    verification:
      - kind: unit
        ref: "python -m pytest tests/models/test_model_info_registry.py -q (21 passed incl. 6 new 27-03 tests + extended sentinel tuple)"
        status: pass
      - kind: integration
        ref: "python -I loader assertion: 49 pretrained with valid statuses / 187 finetuned with None / tested==5.17.0 (exit 0); sole-parser guard green with the UNCHANGED allowlist"
        status: pass
    human_judgment: false
  - id: D4
    description: "Independent artifact↔registry consistency gate: the test re-implements the mapping without importing the writer (Phase 26 checker-independence), fail-closed on absent/unparseable artifact; adjacent lanes undisturbed"
    verification:
      - kind: unit
        ref: "python -m pytest tests/models/test_model_info_registry.py::test_compat_census_artifact_matches_registry -q (1 passed)"
        status: pass
      - kind: other
        ref: "python -m pytest tests/models/ tests/utils/ -q (864 passed); python -m ruff check + format --check clean on all five touched code files"
        status: pass
    human_judgment: false

# Metrics
duration: 27min
completed: 2026-10-27
status: complete
---

# Phase 27 Plan 03: Registry Write-Back Summary

**Census verdicts became permanent registry truth: three typed tf_compat_* fields on all 49 pretrained rows (19 verified-tf5 / 21 import-verified-tf5 / 1 shim-required-tf5 / 6 remote-code-broken / 2 census-blocked) via anchored text insertion, with REG-03 extended in the same commit and an independent artifact↔registry consistency gate proving the data cannot drift from its evidence**

## Performance

- **Duration:** 27 min
- **Started:** 2026-10-27T08:57:50Z
- **Completed:** 2026-10-27T09:25:03Z
- **Tasks:** 3
- **Files modified:** 6 (registry schema, registry data, census tool, three test files)

## Accomplishments

- **Schema (Task 1, TDD RED→GREEN):** `RegistryEntry` gained `tf_compat_status` (five-value Literal), `tf_compat_evidence`, and `tf_compat_tested` in the exact REG-02 optional-nullable shape, with an honest Attributes docstring (import-verified = inferred-from-representative; tested = empirically under the stamped version only). Fields default None, so the schema commit is registry-safe before any data lands — the extra=forbid atomicity (T-27-08) holds by construction: schema (7498ff2) precedes data (b523d57).
- **Writer (Task 1):** `scripts/census_compat.py --write-registry [--apply]` — `compat_verdict` maps artifact status+disposition+load to registry verdicts fail-closed; `plan_compat_insertions` locates each pretrained block by its exact `model:` line (repo ids unique across sections; base_model lines never match) and anchors after `max_length`; idempotent (existing tf_compat_status → skip); dry-run default; json.dumps quoting for round-trip-safe YAML scalars; dated provenance block appended exactly once. 16 offline unit tests over tmp_path fixtures (never the packaged file); dry-run on the real artifact plans exactly 49 insertions and writes nothing.
- **Write-back (Task 2, one atomic data+tests commit):** 49 rows enriched — 19 verified-tf5 (green family-representative load+forward), 21 import-verified-tf5 (siblings, evidence names their family rep), 1 shim-required-tf5 (PlantCaduceus_l32 via `_patch_tied_weights_bookkeeping_on_load`, rung 13), 6 remote-code-broken (gpn + Omni-DNA import-fails; DNABERT-2 + gena ×2 + MutBERT structural load/forward failures — each evidence carries the exact failing import or structural reason), 2 census-blocked (GENERanno, CrossDNA). Every `tf_compat_tested` is `"5.17.0"`; zero finetuned rows touched (D-01 inherit); 159-line diff is pure insertion; all prior provenance sentinels intact verbatim.
- **REG-03 extension (Task 2):** six new data-side tests — all-pretrained coverage, zero-finetuned mirror, remote-code-broken evidence names its cause (transformers symbol or dotted path, never a bare hub stamp), shim-required evidence names a rung whose existence is verified against the shim source itself, degenerate-value sibling guard, and the new provenance sentinel pinned in `PROVENANCE_SENTINELS`. Counts stay guarded 49/187; the sole-parser guard passes with the UNCHANGED allowlist (the writer lives in scripts/, outside the scanned dnallm/ tree).
- **Consistency gate (Task 3):** `test_compat_census_artifact_matches_registry` re-implements the mapping inside the test (no import from the writer — Phase 26 checker-independence, T-27-07 mitigation) and compares every artifact record against the typed registry per repo id; absent/unparseable artifact is a `pytest.fail`, never a skip. Full-lane confirmation: tests/models/ + tests/utils/ 864 passed; ruff check + format clean on every touched file.

## Task Commits

Each task was committed atomically (TDD RED → GREEN for Task 1; data + enforcement in ONE commit for Task 2):

1. **Task 1 (RED): failing schema + writer tests** - `ee760ab` (test)
2. **Task 1 (GREEN): RegistryEntry tf_compat_* fields + --write-registry writer** - `7498ff2` (feat)
3. **Task 2: census verdict write-back into model_info.yaml + REG-03 data tests** - `b523d57` (feat)
4. **Task 3: independent artifact-registry consistency gate** - `0231b41` (test)

**Plan metadata:** committed after this SUMMARY (docs: complete plan).

## Files Created/Modified

- `dnallm/models/registry.py` — three tf_compat_* fields + Attributes docstring (no accessor/loader changes).
- `scripts/census_compat.py` — `--write-registry [--apply] [--registry-yaml]` mode: compat_verdict, plan_compat_insertions, compat_provenance_block, write_registry, SHIM_RESTORED_LOADS, COMPAT_PROVENANCE_SENTINEL (+262 lines).
- `dnallm/models/model_info.yaml` — 49 enriched pretrained rows (147 field lines) + 12-line provenance block (anchored insertion only; never a round-trip).
- `tests/models/test_registry.py` — 27-03 section: five loader-side schema tests.
- `tests/models/test_model_info_registry.py` — 27-03 section: six data-side tests + consistency gate + extended sentinel tuple.
- `tests/models/test_census_compat.py` — 16 mapping/writer tests (offline tmp_path fixtures).

## Decisions Made

- **Shim-required derives from the D-03 verdict + rung dependency, not disposition type alone.** The plan's mapping clause (`import-fail`+fix, recovered, forward true) matches zero entries of the final artifact — 27-02 deliberately used `skip` dispositions for the rung-recovered gena entries because their loads still fail, and no `fix` dispositions exist. The one genuinely shim-dependent green load, PlantCaduceus_l32 (bare Caduceus backbone skips post_init; rung 13 materializes the tied-weights bookkeeping at the from_pretrained boundary), is invisible in the artifact, so the writer records it in `SHIM_RESTORED_LOADS` citing the 27-02 live evidence; the consistency test encodes the same fact independently.
- **remote-code-broken spans both failure depths** (import-fail skips and structural load/forward skips) — the vocabulary bucket for "cannot be used under 5.17", with evidence carrying the exact cause either way.
- **Evidence strings are hub@revision stamps plus cause**: failing import + skip reason (broken), rung name (shim-required), family-rep repo (import-verified), blocking reason (census-blocked); the D-05 honesty rule lives in tf_compat_tested = 5.17.0 only.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical detail] shim-required mapping source absent from the artifact**
- **Found during:** Task 1 (mapping design against the real artifact)
- **Issue:** the plan's mapping table expected `fix` dispositions to drive shim-required-tf5, but the final 27-02 artifact carries only `skip` dispositions (by design — the D-03 verdict is what the registry consumes), and the artifact never stamps which rung a green load needed. Applied literally, shim-required-tf5 would be an empty bucket and PlantCaduceus_l32 would be mislabeled verified-tf5 — claiming a native load that fails without the shim.
- **Fix:** documented `SHIM_RESTORED_LOADS` table in the writer (PlantCaduceus_l32 → `_patch_tied_weights_bookkeeping_on_load`, 27-02 live evidence + rung tests) plus the `fix`-disposition branch for future artifacts; the Task-3 independent gate encodes the same fact so the writer cannot silently redefine it.
- **Files modified:** scripts/census_compat.py, tests/models/test_model_info_registry.py
- **Verification:** test_shim_required_evidence_names_rung (verifies the rung exists in the shim source), consistency test, distribution check (exactly 1 shim-required).
- **Committed in:** `7498ff2`, `b523d57`, `0231b41`

---

**Total deviations:** 1 auto-fixed (1 Rule 2)
**Impact on plan:** Honest labeling preserved (the alternative — an empty shim bucket and a false verified-tf5 on a shim-dependent model — would have defeated the plan's own vocabulary semantics); no scope change, no new surface.

## Issues Encountered

- Two vocabulary-rejection tests passed vacuously in the RED run (extra=forbid rejects every tf_compat_* value today, valid or not); their RED is carried by the paired acceptance tests in the same suite — recorded in the machine-validated RED evidence semantic assessment.
- One test-shape correction during GREEN (the synthetic non-rep record initially carried its own green load, which honestly maps to verified-tf5; real non-reps carry `load: null`) — standard RED→GREEN iteration, fixed before the GREEN commit.

## Authentication Gates

None — no authentication surfaces were touched.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- 27-04 (docs layer) reads the verdicts through the typed registry accessors: `tf_compat_status` / `tf_compat_evidence` / `tf_compat_tested` on all 49 pretrained entries, with the honest-labeling rule already encoded (tested = 5.17.0 only; the 4-side inference is wording, never data).
- The registry remains the single parse owner (unchanged allowlist) and read-only inside dnallm/; re-running the census + `--write-registry` is the documented maintenance path (the provenance block names the tool).
- The consistency gate keeps the committed artifact and the committed data provably in agreement, so a future census re-run that disagrees will surface as a test failure, not silent drift.

## Self-Check: PASSED

- `dnallm/models/registry.py` (three fields declared) — verified in HEAD
- `dnallm/models/model_info.yaml` (49 enriched rows + provenance) — verified in HEAD
- `scripts/census_compat.py` (--write-registry mode) — verified in HEAD
- Commits `ee760ab`, `7498ff2`, `b523d57`, `0231b41` — all ancestors of HEAD (verified post-SUMMARY below).
