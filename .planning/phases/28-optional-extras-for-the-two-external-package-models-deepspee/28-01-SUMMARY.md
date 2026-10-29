---
phase: 28-optional-extras-for-the-two-external-package-models-deepspee
plan: "01"
subsystem: packaging
tags: [pyproject, extras, deepspeed, flash-attn, fla, supply-chain, cuda-build, sm-121, census-evidence]

requires:
  - phase: 27.2-transformers-5-compatibility-remediation-all-8-unsupported-p
    provides: measured census dispositions, the install_mamba.sh pinned-installer discipline, the support-word single source
provides:
  - Base flash-linear-attention dependency (fla extra retired) — PlantHelixSeek works out-of-the-box (D-09)
  - Curated extras deepspeed>=0.19.6,<0.19.7 and flash-attn>=2.8.3.post2,<2.9 with no markers and no meta-extra membership (D-10/D-11/D-13)
  - scripts/install_flash_attn.sh — sha256-pinned sdist + sm_121 gencode patch + c++20 flag patch, all occurrence-counted (D-12)
  - Packaging contract tests (TestFlaBaseDependency, TestCuratedExtras) that fail against the pre-change pyproject
  - Measured extras-replica evidence in .scratch/p28-extras/ (compat.json): CrossDNA load+forward green on cuda; BigBird load green, forward structurally blocked (triton superblock removal)
affects: [28-02 flip plan, optional_dependencies.md rewrite, model_compatibility.md regen, extras_compat.yml lane]

actuals:
  tokens: 9717
  tasks: 3
  commits: 4

tech-stack:
  added: [deepspeed 0.19.6 (extra), flash-attn 2.8.3.post2 (extra, source-built)]
  patterns: [occurrence-counted dual patch (gencode + -std) in a sha256-pinned installer, exact-band single-member curated extras]

key-files:
  created: [scripts/install_flash_attn.sh]
  modified: [pyproject.toml, .github/workflows/ci.yml, tests/test_extras_guard.py, tests/models/test_plant_helixseek_fla_kernels.py]

key-decisions:
  - "TDD: RED (0d6407d, RED_EVIDENCE_OK, 10 planned failures) before GREEN (2f8af48); the sanctioned [cuda]/[rocm] residue line moved with the fla comment and now carries the base-deps 4-space indent"
  - "Installer grew a second occurrence-counted patch: the four -std=c++17 -> -std=c++20 flips (torch >=2.13 ATen C++20 floor; measured first-build failure under the lane's torch 2.14.1+cu130) — same defect class install_mamba.sh patches"
  - "Extras replica resolved torch 2.14.1+cu130 / triton 3.8.0 / transformers 5.17.0 (the current runner-lane resolution), python pinned 3.13 (bare uv venv would pick 3.16.0a1); flash-attn built in 64 min at MAX_JOBS=8, kernel smoke green on GB10 sm_121"
  - "MEASURED STRUCTURAL FACT (owner decision required, research A2 contingency): deepspeed 0.19.6 ops.sparse_attention calls triton._C.libtriton.superblock, an API absent from triton 3.6.0 AND 3.8.0 — BigBird's sparse forward cannot launch in any dnallm-compatible environment (torch>=2.4 implies triton 3.x). The deepspeed extra greens BigBird's LOAD (construction was the old failure); the forward stays red"
  - "Census probe length adaptation: sparse BigBird requires seq_len % block_size(64) == 0, so the committed tool's LOAD_SEQUENCE_LENGTH=32 structurally cannot forward it; evidence measured at length 64 via .scratch/p28-extras/census_compat_len64.py — a one-line copy of the committed tool (diff proven), committed tool untouched"

patterns-established:
  - "Dual occurrence-counted patch discipline in pinned installers: assert exact before/after counts for every edited token class, ast.parse the patched file, print a PATCH VERIFIED line naming every verified property"

requirements-completed: [D-09, D-10, D-11, D-12, D-13]
requirements-partial: [D-03]

coverage:
  - id: D1
    description: "Base fla install contract: exact 0.5.x band + WPAR-07 triton-windows marker in [project] dependencies, fla extra retired, all slims (D-09)"
    requirement: D-09
    verification:
      - kind: unit
        ref: "tests/models/test_plant_helixseek_fla_kernels.py::TestFlaBaseDependency (3 tests) + tests/test_extras_guard.py::TestCuratedExtras::test_fla_band_rides_base_dependencies / test_retired_fla_extra_stays_deleted"
        status: pass
      - kind: automated
        ref: "python -I tomllib assertion one-liner (plan verify 2) — exit 0 on 2026-10-29"
        status: pass
    human_judgment: false
  - id: D2
    description: "Curated extras contract: deepspeed >=0.19.6,<0.19.7 and flash-attn >=2.8.3.post2,<2.9, single-member, no markers, excluded from all/full (D-10/D-11/D-13)"
    requirement: D-10
    verification:
      - kind: unit
        ref: "tests/test_extras_guard.py::TestCuratedExtras (6 tests)"
        status: pass
      - kind: automated
        ref: "uv pip install -e '.[deepspeed]' resolved deepspeed 0.19.6 via the plain extra route (no fallback needed)"
        status: pass
    human_judgment: false
  - id: D3
    description: "CI cannot request an undefined extra: .[full] at the two former .[full,fla] sites, .[full,dev,mcp] at the former .[full,fla,dev,mcp] site, prose rewritten for base-provided KDA (D-09 fallout)"
    verification:
      - kind: automated
        ref: "assert 'full,fla' not in ci.yml + assert 'uv pip install -e \".[full,dev,mcp]\"' in ci.yml — exit 0"
        status: pass
    human_judgment: false
  - id: D4
    description: "sha256-pinned sm_121 flash-attn installer with fail-closed hash gate (D-12, T-28-01)"
    requirement: D-12
    verification:
      - kind: automated
        ref: "bash -n + --patch-only against the real TUNA sdist (hash match, occurrence assertions, PATCH VERIFIED line) — exit 0"
        status: pass
      - kind: automated
        ref: "corrupted-file negative test rejected with sha256 MISMATCH (re-run after the c++20 patch edit) — exit 1 as required"
        status: pass
      - kind: integration
        ref: "full build: flash_attn-2.8.3.post2-cp313-cp313-linux_aarch64.whl (66,010,760 bytes) built and installed; flash_attn_func kernel launched on NVIDIA GB10 sm_121"
        status: pass
    human_judgment: false
  - id: D5
    description: "Isolated extras replica environment with both measured package versions; default dev venv and committed census artifact/registry untouched"
    verification:
      - kind: integration
        ref: ".scratch/p28-extras/venv: deepspeed 0.19.6, flash-attn 2.8.3.post2 asserted via importlib.metadata; git status clean of tracked Task-3 changes"
        status: pass
    human_judgment: false
  - id: D6
    description: "CrossDNA green in the extras replica (D-03, flash-attn half)"
    requirement: D-03
    verification:
      - kind: integration
        ref: ".scratch/p28-extras/compat.json: chengCCC/CrossDNA_pretrain @cd118dc45eaa0f2debb05f03ed0290fe56a78ab1 load.ok=true forward_ok=true shape=(1, 64, 5) device=cuda 2.807s"
        status: pass
    human_judgment: false
  - id: D7
    description: "BigBird green in the extras replica (D-03, deepspeed half) — DID NOT SHIP AS PLANNED"
    requirement: D-03
    verification:
      - kind: integration
        ref: ".scratch/p28-extras/compat.json: AIRI-Institute/gena-lm-bigbird-base-sparse @3a9028d36137341c10df771c3bd93de530d74594 load.ok=true (construction green — the old default-env failure is fixed by the extra) but forward_ok=false: AttributeError 'triton._C.libtriton' has no attribute 'superblock' (deepspeed 0.19.6 sparse matmul LUT construction; symbol absent in triton 3.6.0 AND 3.8.0)"
        status: fail
    human_judgment: true
    rationale: "Measured structural impossibility, not an execution gap: torch>=2.4 implies triton 3.x which removed the 2020-era superblock API deepspeed's sparse kernel requires (module deleted upstream in 0.19.7 for the same reason). Research A2 pre-authorized this contingency: 'the disposition does NOT flip; the phase then needs a documented fallback decision from the owner.' Plan 02's BigBird flip is blocked on that ruling."

duration: 92min
completed: 2026-10-29
status: halted
commits: 4
plan_head_before: 459fcb36277499a0643ef1beb5a667e541ed9e36
plan_head_after: 39bc483
---

# Phase 28 Plan 01: Optional extras install contract + hardest-constraint proof Summary

**Base-fla packaging, two curated extras, and a dual-patched sha256-pinned sm_121 flash-attn installer all landed green; CrossDNA measured fully green (load+forward on cuda) in the extras replica, while BigBird's forward is measured STRUCTURALLY impossible (deepspeed's sparse kernel needs the removed triton `superblock` API) — its load greens with the extra, and the forward flip waits on an owner ruling.**

## Performance

- **Duration:** 92 min (2026-10-29T13:57:45Z → 15:29Z)
- **Tasks:** 3 (Tasks 1-2 complete; Task 3 evidence complete, one success criterion measured impossible)
- **Files modified:** 5 tracked + 1 new script
- **Flash-attn build cost:** first build failed in ~2 min (c++17/ATen C++20 floor); patched rebuild 64 min at MAX_JOBS=8 → `flash_attn-2.8.3.post2-cp313-cp313-linux_aarch64.whl` (66,010,760 bytes, sha256 `08a15cb099c05706959e29f82933f66fe092965d4c51a1c4e0257613baa8371e`)

## Measured Evidence (for Plan 02's DOCS_DATA_DATE)

| Fact | Value |
|---|---|
| Evidence date (UTC) | **2026-10-29** (probes 15:24Z, version record 15:28Z) |
| deepspeed | 0.19.6 (plain `.[deepspeed]` uv route; `ops.sparse_attention` importable) |
| flash-attn | 2.8.3.post2 (built via scripts/install_flash_attn.sh, TUNA sdist sha256 `d0f9f0b0…cc340b` verified) |
| torch / triton / transformers / fla | 2.14.1+cu130 / 3.8.0 / 5.17.0 / 0.5.2 (base dep) |
| CrossDNA probe | `chengCCC/CrossDNA_pretrain` @`cd118dc45eaa0f2debb05f03ed0290fe56a78ab1` — load.ok=true, forward_ok=true, shape **(1, 64, 5)**, device cuda, 2.807 s |
| BigBird probe | `AIRI-Institute/gena-lm-bigbird-base-sparse` @`3a9028d36137341c10df771c3bd93de530d74594` — **load.ok=true** (7.21 s, construction green: the extra fixed the old ModuleNotFoundError), forward_ok=false (`libtriton.superblock` absent) |
| triton cross-check | `superblock` absent in triton 3.8.0 (extras venv) AND 3.6.0 (dev venv, torch 2.11) — impossible across the dnallm torch range |
| Kernel smoke | `flash_attn_func` fp16 forward on NVIDIA GB10: OK (sm_121 gencode patch proven at runtime) |

Artifacts: `.scratch/p28-extras/compat.json` (evidence), `.scratch/p28-extras/census_compat_len64.py` (one-line length-64 probe copy, diff-proven), `.scratch/p28-extras/fa-build.log` (failed c++17 build), `fa-build2.log` (green build), `.scratch/tdd-red-28-01-task1.json` (RED evidence, RED_EVIDENCE_OK).

## Accomplishments

- D-09/D-10/D-11/D-13 landed atomically: fla rides base (PlantHelixSeek out-of-the-box), the fla extra is deleted, `all` slims, two curated extras carry research-measured bands with no markers and no meta-extra membership, and all four CI `full,fla` occurrences are gone (uv can never hit an undefined extra).
- D-12 installer: PEP 503 stdlib fetch, index-declared hash precheck, local sha256 gate before extraction, sm_121 gencode patch + c++20 flag patch (both occurrence-counted, ast-checked), `--no-build-isolation` build, `--no-deps` install; every retained mamba-script interface; corrupted bytes rejected before any extraction.
- TDD discipline held end to end (RED_EVIDENCE_OK → GREEN → tracer re-verify green; both TDD gate commits present).
- CrossDNA's hardest runtime path is measured green in a runner-class replica — Plan 02 can flip its row on this evidence.

## Task Commits

1. **Task 1 RED: failing packaging tests** — `0d6407d` (test)
2. **Task 1 GREEN: base-fla promotion + curated extras + CI rewiring** — `2f8af48` (feat)
3. **Task 2: sha256-pinned sm_121 installer** — `cd78d21` (feat)
4. **Task 2 fix: c++20 flag patch (measured torch-2.14 build failure)** — `39bc483` (fix)

Task 3 made no tracked-file changes by design (runtime evidence only) — no commit; its evidence lives in `.scratch/p28-extras/` and this SUMMARY. Plan metadata commit follows this file.

_Note: 4 foreign commits from a concurrent session interleaved during the build window (d0b0285, c3289a0, db76aff, a106e8e — test packaging/docs-format/ty fixes, none touching this plan's files); all plan verifies re-ran green on the post-interleave HEAD._

## Files Created/Modified

- `pyproject.toml` — fla band + WPAR-07 marker into base deps (comments verbatim, `[cuda]/[rocm]` line preserved for the residue gate); fla extra deleted; deepspeed/flash-attn extras added
- `.github/workflows/ci.yml` — three install commands + four comment blocks describe base-provided KDA support
- `scripts/install_flash_attn.sh` (new) — pinned installer, dual patch, fail-closed gates
- `tests/test_extras_guard.py` — WPAR-07 pin reads base deps; TestCuratedExtras (6 tests); residue-gate sanctioned line re-indented
- `tests/models/test_plant_helixseek_fla_kernels.py` — TestFlaExtraDeclared → TestFlaBaseDependency (3 tests); parser cases kept

## Decisions Made

Recorded in key-decisions frontmatter. The load-bearing one for the phase: **BigBird's forward cannot go green with any installable dependency set** — deepspeed 0.19.6's sparse kernel constructs its Triton LUT via `libtriton.superblock`, removed from Triton long before 3.x (and the module was deleted upstream in deepspeed 0.19.7 for exactly this rot). The extra still delivers a real, measured improvement: BigBird's model construction/load is green where the default env fails at `ModuleNotFoundError`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Sanctioned residue line moved with the fla comment**
- **Found during:** Task 1 GREEN
- **Issue:** the `[cuda]/[rocm]` pyproject comment (sanctioned by the 18-01 residue gate) moved from column 0 (extras block) into the indented base-deps list; the exact-match gate failed
- **Fix:** sanctioned set entry updated to the moved line's exact new text (4-space indent), gate comment documents the D-09 move
- **Files modified:** tests/test_extras_guard.py
- **Verification:** 27/27 packaging tests green
- **Committed in:** 2f8af48

**2. [Rule 1 - Bug] Installer needed the c++20 flag patch (torch 2.14 ATen floor)**
- **Found during:** Task 3 (first flash-attn build)
- **Issue:** `#error C++20 or later compatible compiler is required to use ATen` — flash-attn setup.py hardcodes `-std=c++17` in 4 places; the lane resolves torch 2.14.1+cu130
- **Fix:** second occurrence-counted patch flips the 4 flags to c++20 (mamba-script discipline); PATCH VERIFIED line extended with the c++20 clause
- **Files modified:** scripts/install_flash_attn.sh
- **Verification:** patched rebuild green (64 min, wheel installed, kernel smoke on sm_121); corrupted-file negative test still rejects
- **Committed in:** 39bc483

**3. [Rule 1 - Plan defect] Census probe length 32 cannot forward sparse BigBird**
- **Found during:** Task 3 (first probe run)
- **Issue:** BigBird's remote code requires `seq_len % block_size(64) == 0`; the committed tool's `LOAD_SEQUENCE_LENGTH = 32` structurally cannot exercise it (fails before any kernel)
- **Fix:** evidence measured at length 64 via `.scratch/p28-extras/census_compat_len64.py` — a copy of the committed tool with ONLY that constant changed (diff-proven); the committed tool is untouched (not in this plan's file scope). The length-64 run then reached the real kernel gate and produced the load-bearing superblock evidence
- **Verification:** CrossDNA green at 64 (shape (1,64,5)); BigBird reaches deepspeed's sparse matmul and fails at `libtriton.superblock`
- **Committed in:** n/a (scratch-only, by plan design)

**4. [Rule 3 - Blocking] Environment pinning deviations**
- Bare `uv venv` on this box picks python 3.16.0a1 → pinned `--python 3.13`; build run with `MAX_JOBS=8` (plan default 4, caller-raisable by design; 121 GB RAM, 20 cores, load 0.4)

---

**Total deviations:** 4 auto-fixed (2 Rule 1 code, 1 Rule 1 plan-defect, 1 Rule 3)
**Impact on plan:** Tasks 1-2 fully green; Task 3's evidence complete and honest, but one success criterion (BigBird forward green) is measured impossible — see Blocker.

## Blocker: BigBird forward is structurally impossible (owner decision required)

Research A2 pre-authorized exactly this contingency: *"If the forward reds, BigBird's disposition cannot flip; the phase then needs a documented fallback decision from the owner."*

- **Measured:** `AttributeError: module 'triton._C.libtriton' has no attribute 'superblock'` at `deepspeed/ops/sparse_attention/matmul.py:276` — LUT construction, before any kernel launch. Symbol absent in triton 3.8.0 (extras venv, torch 2.14.1) and 3.6.0 (dev venv, torch 2.11). dnallm requires torch>=2.4 ⇒ triton 3.x ⇒ no installable resolution exists.
- **What the extra DID buy (measured):** load.ok=true — construction green (7.21 s on cuda) where the default env dies at `ModuleNotFoundError: deepspeed.ops.sparse_attention` inside `BertSelfAttention.__init__`.
- **Options for the owner (Plan 02 gate):**
  - **A.** CrossDNA-only `extra-required-tf5` flip; BigBird keeps its committed skip disposition with evidence updated to name the load-green/forward-red split (honest: "load works with the deepspeed extra; forward blocked by an upstream-dead Triton API").
  - **B.** New/adjusted status semantics for load-green/forward-red (schema + docs + support-word changes — larger Plan 02).
  - **C.** Drop BigBird from the extras story (keep the extra as the load fix, document forward as upstream-dead).
- **Not blocked:** Plan 02's CrossDNA flip, the extras lane, and the docs rewrite for CrossDNA can proceed on this evidence under any option.

## Issues Encountered

- First flash-attn build failed on the c++17/ATen C++20 floor (deviation 2) — fixed, evidence in `.scratch/p28-extras/fa-build.log`.
- BigBird forward: deviation 3 (probe length) then the structural superblock finding (Blocker above).
- No auth gates; no skipped tests; no stubs shipped.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Plan 02 (flips + lane + docs): CrossDNA evidence is ready (`hub repo@rev; load+forward verified via the flash-attn extra (flash-attn 2.8.3.post2)` — include torch 2.14.1 + the installer route in the detail text; DOCS_DATA_DATE = 2026-10-29). BigBird's flip is blocked on the owner ruling above; the drift-gate seed design (Pattern 1) must be re-scoped for whatever option is chosen.
- The length-64 probe fact must ride into the lane design (D-07): a future extras lane probing BigBird at length 32 will always forward-red on divisibility, masking the real kernel verdict.
- The committed `census_compat.py` LOAD_SEQUENCE_LENGTH stays 32 (default-lane records unchanged); any change is a separate reviewed decision.

## Self-Check: PASSED

All 4 task commits are ancestors of HEAD; all evidence artifacts exist on disk (installer, compat.json, len64 probe copy, RED evidence record); all plan verifies re-ran green on the post-interleave HEAD (27/27 packaging tests, TOML/CI assertions, repo-wide ruff check, format check on both test files).

---
*Phase: 28-optional-extras-for-the-two-external-package-models-deepspee*
*Completed: 2026-10-29 (halted at the A2 designed stop — BigBird forward owner ruling pending)*
