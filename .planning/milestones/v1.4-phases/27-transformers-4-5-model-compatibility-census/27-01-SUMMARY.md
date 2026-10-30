---
phase: 27-transformers-4-5-model-compatibility-census
plan: 01
subsystem: testing
tags: [transformers, compatibility, census, remote-code, subprocess, modelscope, hf-mirror, registry]

# Dependency graph
requires: []
provides:
  - "scripts/census_compat.py — re-runnable census tool: ModelScope-first code-only fetch, auto_map classification, isolated import probes, family-representative load+forward sampling"
  - "tests/models/compat_census.json — 49-record D-06 evidence artifact, every record stamped hub+revision under transformers 5.17.0"
  - "tests/models/test_census_compat.py — 35 offline unit tests over the tool's pure logic"
affects: [27-02, 27-03, 27-04]

# Actuals (#2632) — chars/4 over the realized files; commits measured from the ledger.
actuals:
  tokens: 25202
  tasks: 3
  commits: 4
  plan_head_before: "1e6c7b2077fd7f783e53c79c1f4e6eb5a68f7d82"
  plan_head_after: "74144c5625dfe76b500c823854079c66ba8ba1a4"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Code-only hub fetch: ModelScope first, hf-mirror fallback, hub+revision stamped per record (models.lock convention)"
    - "Subprocess-isolated remote-code import probe: 'import dnallm' strictly first (shim ordering), each snapshot dir executed as a synthetic package"
    - "Family-representative load+forward sampling in a second isolated subprocess (D-02b/D-03)"

key-files:
  created:
    - scripts/census_compat.py
    - tests/models/test_census_compat.py
    - tests/models/compat_census.json
  modified: []

key-decisions:
  - "Census probes execute each fetched snapshot dir as a synthetic package (relative and absolute sibling imports resolve like transformers' dynamic modules); probe paths stay absolute WITHOUT symlink resolution (HF snapshots are extensionless blob symlinks)."
  - "Load layer uses torch_dtype (the plan-literal kwarg) with a kwarg-free retry + fp32 cast for 4.x-era remote from_pretrained implementations that forward unknown kwargs to __init__."
  - "ModelScope loads fetch its default revision (commit shas are not fetchable revisions there, verified live); the HF route pins the census sha."
  - "Load failures report forward_ok=false (no successful forward ran) with ok=false and the error naming the phase, so the D-03 bool gate holds without conflating load and forward evidence."
  - "D-04 giants are a no-op: zero evo rows exist in the 49-entry roster (research A6 confirmed)."
  - "Evidence-first retained: the census fixes nothing and records two honest census-blocked verdicts (GENERanno repo id absent on both hubs; CrossDNA ships no root config.json)."

patterns-established:
  - "Census evidence records: repo/name/class/hub/revision/transformers/status/failing_import/traceback_tail/family/is_family_rep/load/fetched_at with a four-value status vocabulary"
  - "Fail-closed artifact I/O: absent/unparseable/invalid records refuse to read or write"

requirements-completed: ["TBD"]

# Coverage metadata (#1602)
coverage:
  - id: D1
    description: "Re-runnable census tool: one evidence record per pretrained registry entry, isolated subprocess probes with the shim armed"
    verification:
      - kind: unit
        ref: "python -m pytest tests/models/test_census_compat.py -q (35 passed)"
        status: pass
      - kind: integration
        ref: "python -I scripts/census_compat.py --run --entries zhangtaolab/plant-nucleotide-transformer-BPE --artifact /tmp/p27-tracer.json (import-ok)"
        status: pass
      - kind: other
        ref: "python -m ruff check scripts/ tests/models/test_census_compat.py"
        status: pass
    human_judgment: false
  - id: D2
    description: "Committed 49-record evidence artifact in registry order with hub+revision provenance under transformers 5.17.0"
    verification:
      - kind: integration
        ref: "tests/models/compat_census.json assertion gate (49 records, registry order, status vocabulary, provenance)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Family-representative load+forward layer (D-02b/D-03) with honest load_note labeling for non-measured entries"
    verification:
      - kind: integration
        ref: "Task-3 artifact assertion gate (28 reps; every passing-status rep carries a bool forward_ok)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Adjudication of the two census-blocked entries and the failure classes for 27-02 disposition"
    verification: []
    human_judgment: true
    rationale: "GENERanno's nonexistent repo id is a registry-data decision and CrossDNA's size-subdirectory layout is a load-path decision — neither is asserted by a test; 27-02 dispositions the 4 import-fail and 5 load/forward failures."

# Metrics
duration: 74min
completed: 2026-10-27
status: complete
---

# Phase 27 Plan 01: Census Tool + Evidence Run Summary

**Re-runnable census tool: 49 pretrained entries import-probed under transformers 5.17.0 (25 import-ok / 18 arch-ok / 4 import-fail / 2 census-blocked) plus 22 family-representative load+forward proofs**

## Performance

- **Duration:** 74 min
- **Started:** 2026-10-27T06:26:14Z
- **Completed:** 2026-10-27T07:40:40Z
- **Tasks:** 3
- **Files modified:** 3 (2 new code/test files, 1 new evidence artifact)

## Accomplishments

- **Census tool committed and re-runnable** (`scripts/census_compat.py`): registry roster → code-only fetch (ModelScope first, hf-mirror fallback, weights never during the import census) → `auto_map` classification (native vs remote vs cross-repo `--` form) → isolated per-entry import probe with `import dnallm` strictly before remote execution → validated evidence record with hub + revision provenance.
- **D-06 evidence base committed** (`tests/models/compat_census.json`): 49 records in registry order, every one under transformers 5.17.0 with a hub (`ms` 25 / `hf` 24) and revision stamp. Observed import-level failure classes (measured, not predicted):
  - `import-fail` (4): removed-name import `apply_chunking_to_forward` ×2 (gena-lm base + bigbird), native architecture `ConvNetForMaskedLM` gone ×1 (gpn-brassicales), external `olmo` package missing ×1 (Omni-DNA-116M).
  - `census-blocked` (2): `GenerTeam/GENERanno-eukaryote-1.2b-base` — repo id absent on both hubs (direct HF also 404; nearest siblings are `*0.5b-base`); `chengCCC/CrossDNA_pretrain` — no root `config.json` (checkpoints live under `8.1M/ 28.6M/ 71.6M/ 519M/`).
  - Casing was NOT a failure class (research confirmed), matching the corrected owner lead.
- **D-02b/D-03 load layer committed**: 28 family representatives (architectures[0] with namespace fallback; lab-first selection); 22 loaded and forward-tested on CUDA/GB10, 17 fully green, 5 recorded load/forward failures with exact tracebacks:
  - `plant-nucleotide-transformer-6mer`: `EsmConfig` has no `rope_theta` (native ESM path under 5.17).
  - `PlantCaduceus_l32`, `PlantCAD2-Small`: `Caduceus` has no `all_tied_weights_keys` (transformers-5 protocol).
  - `DNABERT-2-117M`: meta-device tensor at load.
  - `MutBERT-Human-Ref`: remote forward `input_shape` unpack failure with 2-D ids.
  - 6 reps parked honestly: 4 import-fail "pending shim", 2 census-blocked.
- **35 offline unit tests** pin the pure logic (classification, cross-repo parsing, failing-import extraction, record validation, artifact round-trip, family grouping/rep selection, load-note labeling, load-program shape).

## Task Commits

Each task was committed atomically (TDD RED → GREEN, then the two evidence runs):

1. **Task 1 (RED): failing tests + planned surface** - `f69a400` (test)
2. **Task 1 (GREEN): census core tool** - `eabd018` (feat)
3. **Task 2: full 49-entry census + HF-snapshot probe fixes** - `374cccd` (fix)
4. **Task 3: family-representative load+forward sampling** - `74144c5` (feat)

**Plan metadata:** committed separately (docs: complete plan).

## Files Created/Modified

- `scripts/census_compat.py` — the census tool (fetch, classify, probe, load, artifact merge, CLI).
- `tests/models/test_census_compat.py` — 35 offline unit tests.
- `tests/models/compat_census.json` — the committed evidence artifact (49 records; the input for 27-02 dispositions and 27-03 write-back).

## Decisions Made

- Probe execution model: each snapshot directory runs as a synthetic package with its directory appended to `sys.path`; module paths stay absolute without symlink resolution.
- Load dtype: `torch_dtype=torch.float32` with a kwarg-free retry + `.to(torch.float32)` for remote `from_pretrained` implementations that forward unknown kwargs to `__init__`.
- Provenance: ModelScope loads the default revision (its API rejects commit shas as `revision`); the HF route pins the stamped sha; the census stamp remains the record's provenance either way.
- Load-failure semantics: `ok=false`, `forward_ok=false`, and the error traceback naming the phase; a bool `forward_ok` keeps the D-03 gate meaningful.
- Nothing was fixed in library code — census-first (D-06); shim rungs belong to 27-02.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Remote files failed on relative sibling imports in the probe**
- **Found during:** Task 1 tracer run
- **Issue:** `modeling_esm.py` does `from .esm_config import EsmConfig`; registering modules top-level produced "attempted relative import with no known parent package" for the known-good tracer entry.
- **Fix:** execute each snapshot directory as a synthetic package (`_census_remote_N` with `__path__`) and append the directory to `sys.path`, mirroring transformers' dynamic-module machinery.
- **Files modified:** `scripts/census_compat.py`
- **Verification:** tracer returns `import-ok`; 35 offline tests green.
- **Committed in:** `eabd018`

**2. [Rule 1 - Bug] HF snapshots are symlinks; resolving them stripped the `.py` suffix**
- **Found during:** Task 2 full run
- **Issue:** `Path.resolve()` followed HF-hub symlinks into `blobs/<sha>` (no extension), so `spec_from_file_location` returned `None` and all 15 hf remote verdicts became bogus `'NoneType' object has no attribute 'loader'` import-fail records.
- **Fix:** absolute paths without symlink resolution; symbol extraction now requires a `module '...'` receiver; the probe raises a clear error when a spec cannot be built.
- **Files modified:** `scripts/census_compat.py`, `tests/models/test_census_compat.py`
- **Verification:** hf entries measured properly (25 import-ok total); regression tests added.
- **Committed in:** `374cccd`

**3. [Rule 2 - Missing critical detail] Opaque absent-config verdicts**
- **Found during:** Task 2 investigation
- **Issue:** CrossDNA's blocked reason only said "No such file or directory".
- **Fix:** absent-config verdicts now name the snapshot's top-level layout, so the reason is self-contained.
- **Files modified:** `scripts/census_compat.py`, `tests/models/test_census_compat.py`
- **Verification:** record reads "no root config.json … snapshot top-level entries: ['28.6M','519M','71.6M','8.1M']"; tests added.
- **Committed in:** `374cccd`

**4. [Rule 1 - Bug] ModelScope rejects commit shas as `revision`**
- **Found during:** Task 3 first loads run
- **Issue:** all 7 ModelScope-stamped reps failed at download with `NotExistError: has no revision: <sha>`.
- **Fix:** ModelScope loads fetch the default revision; HF loads keep the sha pin.
- **Files modified:** `scripts/census_compat.py`, `tests/models/test_census_compat.py`
- **Verification:** lab reps now load+forward green; regression tests added.
- **Committed in:** `74144c5`

**5. [Rule 1 - Bug] `dtype` kwarg broke 4.x-era remote `from_pretrained`**
- **Found during:** Task 3 (prokbert-mini)
- **Issue:** remote classes forward unknown kwargs to `__init__`; `dtype` (and even `torch_dtype`) raised `TypeError` for prokbert's custom loader.
- **Fix:** plan-literal `torch_dtype` with a kwarg-free retry + fp32 cast when the remote loader rejects it.
- **Files modified:** `scripts/census_compat.py`, `tests/models/test_census_compat.py`
- **Verification:** prokbert-mini loads and forwards (`(1, 32, 4101)`); tests added.
- **Committed in:** `74144c5`

**6. [Rule 1 - Bug] `forward_ok` was null on load failure**
- **Found during:** Task 3 gate run
- **Issue:** the D-03 verification requires a bool `forward_ok` for every passing-status representative; load failures reported null.
- **Fix:** load failures report `forward_ok=false` (no successful forward ran) with `ok=false` and the load traceback in `error`.
- **Files modified:** `scripts/census_compat.py`
- **Verification:** Task-3 gate passes (28 reps).
- **Committed in:** `74144c5`

---

**Total deviations:** 6 auto-fixed (6 Rule 1 bugs, 1 of which also carried a Rule 2 detail fix)
**Impact on plan:** All six were probe/tooling correctness issues found by running the census for real; none changed the evidence scope or the census-first discipline. No library code was touched.

## Issues Encountered

- **Registry data defect surfaced (owner decision needed):** `GenerTeam/GENERanno-eukaryote-1.2b-base` exists on neither ModelScope nor HF (direct HF 404 too). The census correctly reports it loud and non-zero; the registry id likely needs correction/removal in 27-03's write-back pass.
- **CrossDNA load path:** `chengCCC/CrossDNA_pretrain` has no root `config.json`; checkpoints sit under size-named subdirectories, so the bare repo id is not loadable by `AutoConfig`/`from_pretrained` — an entry-level load-path decision for 27-02/27-03.
- Both blocked entries are surfaced here per Task 2's instruction to report non-zero blocked counts to the owner.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- 27-02 can dispose every failure from committed evidence: 4 import-level failures (3 shim-candidate classes: removed-name import, removed native architecture, external-package import), 5 load/forward failures (tied-weights protocol, config `rope_theta`, meta-device load, remote forward input contract), 2 census-blocked (register as honest labels / registry-data decisions).
- 27-03's write-back consumes `tests/models/compat_census.json` directly; the `family`/`is_family_rep` fields are already populated for all 49 entries.
- Re-run command for D-08's dispatch lane: `python -I scripts/census_compat.py --run [--with-loads] --artifact tests/models/compat_census.json` (exit 1 signals any unreachable entry).

## Self-Check: PASSED

- `scripts/census_compat.py` — FOUND
- `tests/models/test_census_compat.py` — FOUND
- `tests/models/compat_census.json` — FOUND
- Commits `f69a400`, `eabd018`, `374cccd`, `74144c5` — all ancestors of HEAD.
