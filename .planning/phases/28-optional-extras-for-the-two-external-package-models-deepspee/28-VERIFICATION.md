---
phase: 28-optional-extras-for-the-two-external-package-models-deepspee
verified: 2026-10-29T17:22:27Z
status: human_needed
score: 13/13 must-haves verified
covered_files: [".github/workflows/README.md", ".github/workflows/ci.yml", ".github/workflows/extras_compat.yml", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-01-PLAN.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-01-SUMMARY.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-02-PLAN.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-02-SUMMARY.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-03-PLAN.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-03-SUMMARY.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-04-PLAN.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-04-SUMMARY.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-05-PLAN.md", ".planning/phases/28-optional-extras-for-the-two-external-package-models-deepspee/28-05-SUMMARY.md", "README.md", "WINDOWS-KDA-VERIFY.md", "dnallm/models/model.py", "dnallm/models/model_info.yaml", "dnallm/models/registry.py", "dnallm/models/special/crossdna.py", "dnallm/models/support.py", "docs/example/notebooks/plant_helixseek_anno.md", "docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb", "docs/example/notebooks/plant_helixseek_combined.md", "docs/example/notebooks/plant_helixseek_cre.md", "docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb", "docs/example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb", "docs/faq/models_troubleshooting.md", "docs/getting_started/install_honesty.md", "docs/getting_started/installation.md", "docs/getting_started/quick_start.md", "docs/user_guide/model_compatibility.md", "docs/user_guide/optional_dependencies.md", "example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb", "example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb", "example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb", "pyproject.toml", "scripts/census_compat.py", "scripts/install_flash_attn.sh", "tests/models/compat_census.json", "tests/models/test_census_compat.py", "tests/models/test_compat_docs_consistency.py", "tests/models/test_model_info_registry.py", "tests/models/test_plant_helixseek_fla_kernels.py", "tests/models/test_registry.py", "tests/models/test_special/test_family_handlers.py", "tests/models/test_support.py", "tests/test_extras_guard.py", "tests/test_runner_infra_contracts.py"]
covered_digest: "v4:sha256:8a8d1ee3009d63d5de4c998cb0c7d5bdd730e884a0c8c7892bcbf31ac719591d"
behavior_unverified: 0 # Every behavior-dependent truth has behavioral evidence: TestMissingExtraGuidance (17 tests, re-run green), installer corrupted-file negative test (re-run: exit 1 + sha256 MISMATCH), TestCompatVerdictMapping extra-branch tests, and measured GPU probe receipts on disk (.scratch/p28-extras/compat.json, .scratch/p28-lane/extras-fresh.json)
overrides_applied: 4 # Owner ruling A (2026-10-29), recorded in 28-01-SUMMARY.md and .planning/STATE.md (lines 173-176, 201)
overrides:
  - must_have: "BigBird and CrossDNA each record load.ok=true and forward_ok=true in the same extras-equipped runner-replica artifact (Plan 01 truth 4)"
    reason: "Owner ruling A: BigBird's forward is measured structurally impossible (deepspeed 0.19.6 sparse-attention LUT calls triton libtriton.superblock, absent from triton 3.6.0 and 3.8.0; no installable resolution under torch>=2.4). CrossDNA-only extra-required-tf5 flip; BigBird keeps its committed skip disposition with evidence naming the load-green/forward-red split. The phase goal's 'dispositions flip only on a measured green probe' is satisfied by CrossDNA's flip; BigBird's non-flip IS the census discipline (research A2 pre-authorized this contingency)."
    accepted_by: "owner (Tao Zhang)"
    accepted_at: "2026-10-29T15:45:00Z"
  - must_have: "Only the two Plan-01-measured rows flip; BigBird's evidence names deepspeed and _patch_apply_chunking_to_forward (Plan 02 truth 4, pre-ruling wording)"
    reason: "Ruling A re-scope: exactly one row (CrossDNA) flips to extra-required-tf5; BigBird stays remote-code-broken with split evidence naming deepspeed 0.19.6, _patch_apply_chunking_to_forward, and the load-green/forward-red split. Verified at runtime via resolve(): CrossDNA=extra-required-tf5, BigBird=remote-code-broken."
    accepted_by: "owner (Tao Zhang)"
    accepted_at: "2026-10-29T15:45:00Z"
  - must_have: "BigBird and CrossDNA display the exact shared support words (Plan 02 truth 5, pre-ruling wording)"
    reason: "Ruling A re-scope: a 'works with the deepspeed extra' word would overclaim a model whose forward is measured structurally dead. BigBird keeps 'needs deepspeed'; CrossDNA alone carries 'works with the flash-attn extra'. Verified in support.py SUPPORT_WORD_BY_REPO and pinned by tests/mcp/test_server_streaming.py."
    accepted_by: "owner (Tao Zhang)"
    accepted_at: "2026-10-29T15:45:00Z"
  - must_have: "Both fresh records are load+forward green in lane gate 1 (Plan 04 truth 3, pre-ruling wording)"
    reason: "Ruling A re-scope: gate 1 asserts the split shape (CrossDNA fully green; BigBird load-green + forward-red AT the named superblock cause), pinned offline by TestExtrasLaneContract::test_gate1_asserts_the_ruling_a_expected_shape. Replica run of the exact gate sequence exits green."
    accepted_by: "owner (Tao Zhang)"
    accepted_at: "2026-10-29T15:45:00Z"
human_verification:
  - test: "Read the rewritten optional-dependency guide (docs/user_guide/optional_dependencies.md) as a biologist would: the 'What is an optional extra?' intro, the BigBird deepspeed guide, the CrossDNA flash-attn guide, and the PlantHelixSeek built-in section."
    expected: "Every section is understandable without engineering background; install commands are copy-pasteable; compile-time expectations (hours-scale CUDA build) and Linux+CUDA constraints are clear."
    why_human: "Prose adequacy for the biologist audience (owner plain-language directive 2026-10-21) is a judgment call; automated legs prove content presence, protected-section hashes, and build validity only. Both 28-03-SUMMARY and 28-05-SUMMARY flagged this as human_judgment: true."
  - test: "Read the rewritten README installation journey (extras table, 'Built-in kernels' statement, PlantHelixSeek 'No extra is needed' section) and the generated model_compatibility.md BigBird row."
    expected: "A biologist cross-reading the static guide and the generated page gets one consistent story: CrossDNA works after installing its extra; BigBird needs deepspeed to load but cannot run predictions (upstream-dead Triton API)."
    why_human: "Cross-page narrative consistency and tone are judgment calls; the byte-level gates prove the two pages share source strings, not that the combined reading is clear."
---

# Phase 28: Optional extras: deepspeed & flash-attn model support Verification Report

**Phase Goal:** Make the two fixable external-package models installable via curated optional extras, following the mamba-extra precedent: `.[deepspeed]` for GENA-LM-BigBird and a flash-attn extra for CrossDNA (uv index pinning / sha256-pinned installer as needed), with CI lanes covering the extras. Dispositions flip only on a measured green probe in an extras-equipped environment (census discipline); GPN/Omni-DNA stay version-locked guidance cases. Support words, docs (optional_dependencies.md, model_compatibility.md), and handler error messages update to name the extras.
**Verified:** 2026-10-29T17:22:27Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

ROADMAP.md Phase 28 carries no separate Success Criteria block; the goal paragraph is the contract, merged with the five plans' must_haves and the CONTEXT.md D-01..D-17 decision items. Four plan-literal truths were re-scoped under owner ruling A (2026-10-29, recorded in 28-01-SUMMARY.md and the STATE.md decision log) BEFORE the dependent plans executed; the shipped ruling-A form is what was verified. See `overrides:` above.

### Observable Truths

| # | Truth | Status | Evidence |
| - | ----- | ------ | -------- |
| 1 | `.[deepspeed]` curated extra carries the exact research-measured band, no platform markers, no all/full membership (D-10/D-11/D-13) | ✓ VERIFIED | `pyproject.toml`: `deepspeed = ["deepspeed>=0.19.6,<0.19.7"]`; no `;` marker; `all` = `dnallm[full,dev,test,notebook,docs,ui,mcp,tui]`, `full` = `dnallm[dev,test,notebook,mcp]` + isort — neither names deepspeed; tests/test_extras_guard.py TestCuratedExtras green |
| 2 | flash-attn curated extra + sha256-pinned installer verify the sdist hash before extraction, patch sm_121 gencode, build without isolation (D-10/D-12) | ✓ VERIFIED | `flash-attn = ["flash-attn>=2.8.3.post2,<2.9"]`; scripts/install_flash_attn.sh pins `FLASH_ATTN_SDIST_SHA256=d0f9f0b0…cc340b`, sha256sum gate at line 216-219 runs BEFORE `tar -xzf` at 230, occurrence-counted compute_121/sm_121 patch (0 before, 1 after) + 4x c++17→c++20 flips, `--no-build-isolation --no-deps` at line 335, PATCH VERIFIED line at 300. Behavioral: corrupted-file negative test re-run by verifier — exit 1 with `sha256 MISMATCH` |
| 3 | fla promoted to base with the Windows triton marker; the retired fla extra and every CI/doc/notebook use of it are gone (D-09) | ✓ VERIFIED | `flash-linear-attention>=0.5.2,<0.6` and `triton-windows>=3.7.1.post27,<3.8; platform_system == 'Windows'` both in `[project] dependencies`; no `fla` key in optional-dependencies; `full,fla` absent from all of `.github/workflows/`; `.[full,dev,mcp]` present at ci.yml:829; zero retired forms (`.[fla]`/`dnallm[fla]`/`fla extra`) across the six doc surfaces; zero `full,fla` in notebooks/mirrors; README carries `base dependenc` (5x) + `built-in flash-linear-attention` |
| 4 | A dedicated extras lane runs nightly and on demand on the self-hosted GPU runner, never on push/PR, least privilege (D-05/D-06) | ✓ VERIFIED | extras_compat.yml parsed: triggers exactly {schedule `0 4 * * *`, workflow_dispatch, workflow_call} — no push/pull_request; permissions `contents: read`; runs-on `[self-hosted, dnallm-nightly]`; timeout 360; gpu_absent.marker fail-closed; upload `if: always()`; TestExtrasLaneContract 15 tests green |
| 5 | ci.yml fourth selector + exactly-once nightly + census_compat.yml byte-untouched (D-06/D-02) | ✓ VERIFIED | `run_extras_compat` boolean default true; job gate `github.event_name == 'workflow_dispatch' && inputs.run_extras_compat`; `uses: ./.github/workflows/extras_compat.yml`; no secrets; ci.yml schedules only 03:00/05:30 (04:00 lives solely in extras_compat.yml); `git diff HEAD -- census_compat.yml` empty and last touch is pre-phase commit 76ea80f; TestExtrasDispatchSelection 5 tests green |
| 6 | Lane installs deepspeed via the curated extra and flash-attn only through the pinned installer, with two independent gates (D-07/D-08/D-12) | ✓ VERIFIED | extras_compat.yml:122 `uv pip install -e ".[deepspeed]"`; :125 calls only `bash scripts/install_flash_attn.sh`; distinct "Gate 1" (raw load booleans, ruling-A per-model shape) and "Gate 2" (`census_compat.py --verify`) steps in green-then-verify order; no continue-on-error or `\|\| true`; replica execution receipt `.scratch/p28-lane/extras-fresh.json` on disk matches (BigBird ok=True fwd=False with superblock; CrossDNA both True) |
| 7 | CrossDNA's disposition flipped to extra-required-tf5 on a measured green probe (D-03) | ✓ VERIFIED | Runtime `resolve('chengCCC/CrossDNA_pretrain')` → `extra-required-tf5`, evidence `hf chengCCC/CrossDNA_pretrain@cd118dc45eaa…; load+forward verified via the flash-attn extra (flash-attn 2.8.3.post2)`; measured record in `.scratch/p28-extras/compat.json` shows load.ok=True forward_ok=True at that revision; offline `--verify` re-run by verifier: 49 compared, all matched, OK |
| 8 | BigBird keeps its skip disposition with honest load-green/forward-red split evidence — census discipline (D-03/D-04, ruling A) | ✓ VERIFIED (override) | Runtime `resolve('AIRI-Institute/gena-lm-bigbird-base-sparse')` → `remote-code-broken` (NOT flipped), evidence names deepspeed extra + `_patch_apply_chunking_to_forward` + the split ("load works with the deepspeed extra but forward is blocked by an upstream-dead Triton API"); measured `.scratch/p28-extras/compat.json` shows load.ok=True forward_ok=False; lane replica forward error contains `superblock`. Owner ruling A recorded in 28-01-SUMMARY.md + STATE.md |
| 9 | extra-required-tf5 accepted by every fail-closed surface; symmetric disposition.extra seed maps both default failures and extras green (D-01/D-02) | ✓ VERIFIED | registry.py:141 Literal includes `extra-required-tf5`; census_compat.py compat_verdict `disposition.get("extra")` branch precedes the skip branch (index-verified); support.py SUPPORT_WORDS + DOCS_STATUS_CELLS; both independent TF_COMPAT_VOCABULARY mirrors; seed: CrossDNA carries green load record + `{"extra": …}`, BigBird carries `{"skip": …}` with default-env load.ok=False — 402 tests across six gate files green |
| 10 | GPN/Omni-DNA stay version-locked guidance cases | ✓ VERIFIED | Runtime resolve: both `remote-code-broken` (unchanged); GPN section sha256 `fddf29e4…b084267` and Omni-DNA `6c5bde28…17161f8` byte-identical to the pre-phase baseline in optional_dependencies.md |
| 11 | Support words name the extras from one vocabulary source (D-14) | ✓ VERIFIED (override) | support.py: generic `extra-required-tf5` → "works with an optional extra"; CrossDNA override "works with the flash-attn extra"; BigBird keeps "needs deepspeed" per ruling A; docs cell "Works after installing an optional extra (tested)"; flows to mcp/server.py and preflight consumers |
| 12 | Docs update: plain-language guide with working install commands; generated page regenerates byte-identically with the evidence date (D-15/D-16) | ✓ VERIFIED (override) | All 10 required strings present in optional_dependencies.md; model_compatibility.md carries the 19/21/5/1/3/0 of 49 summary sentence, CrossDNA extra cell, BigBird split fragment, DOCS_DATA_DATE 2026-10-29; verifier re-ran `--emit-docs` + `cmp` → byte-identical; `mkdocs build --strict` exit 0 ("Documentation built in 4.78 seconds"). Prose adequacy routed to human verification |
| 13 | Missing-extra errors are plain-language, name the exact curated-extra install commands, and preserve the original exception (D-17) | ✓ VERIFIED | `_raise_missing_extra_import_error` (model.py:1213, invoked at both wrap sites 1424/1495): cycle-safe cause traversal, deepspeed ModuleNotFoundError → ImportError naming `pip install 'dnallm[deepspeed]'` + <0.19.7 caveat + guide URL, `raise … from node`; `_from_pretrained_with_flash_attn_guidance` (crossdna.py:38, used at both from_pretrained paths 576/629): Flash-Attention-marker-only translation to `pip install --no-build-isolation 'dnallm[flash-attn]'` with `_dnallm_missing_extra_guidance` marker so the generic wrap re-raises it unchanged. Behavioral: TestMissingExtraGuidance 17 tests green (re-run by verifier) |

**Score:** 13/13 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `pyproject.toml` | Base fla + two curated extras | ✓ VERIFIED | All declarations exact; no markers; no meta-extra membership |
| `scripts/install_flash_attn.sh` | Hash-gated sm_121 source-build pipeline | ✓ VERIFIED | 17.5 KB, executable, bash -n clean, dual occurrence-counted patch, fail-closed gate behaviorally re-tested |
| `.github/workflows/extras_compat.yml` | Scheduled/dispatch/reusable extras lane | ✓ VERIFIED | Parsed YAML confirms all trigger/permission/runner/timeout/gate claims |
| `.github/workflows/ci.yml` | Fourth selector + gated reusable call | ✓ VERIFIED | Parsed YAML confirms input shape, job gate, no secrets, no schedule arm |
| `.github/workflows/README.md` | Operator docs, deferred first-dispatch step | ✓ VERIFIED | Four selectors, run_extras_compat=false example, Extras Compat Job section, post-merge operator step, zero full,fla |
| `tests/test_runner_infra_contracts.py` | Offline lane/dispatch contracts | ✓ VERIFIED | TestExtrasLaneContract (15) + TestExtrasDispatchSelection (5) green |
| `dnallm/models/registry.py` | extra-required-tf5 vocabulary | ✓ VERIFIED | Literal + docstring |
| `scripts/census_compat.py` | Ordered extra verdict branch, status vocab | ✓ VERIFIED | extra-before-skip ordering index-proven; six-value TF_COMPAT_STATUSES |
| `tests/models/compat_census.json` | Symmetric extra/skip dispositions over measured records | ✓ VERIFIED | CrossDNA {extra}+green; BigBird {skip}+default-failure; 49 entries; --verify 49 matched |
| `dnallm/models/model_info.yaml` | CrossDNA flip + BigBird split evidence | ✓ VERIFIED | Runtime resolve confirms both rows + GPN/Omni-DNA unchanged |
| `dnallm/models/support.py` | Single-source support words + docs cell | ✓ VERIFIED | All three new vocabulary entries present |
| `docs/user_guide/model_compatibility.md` | Regenerated 49-row page | ✓ VERIFIED | Summary sentence, both rows, byte-identical re-emission |
| `docs/user_guide/optional_dependencies.md` | Plain-language extras guide | ✓ VERIFIED | 10/10 required strings; protected sections hash-identical |
| README.md / installation / quick_start / install_honesty / troubleshooting / WINDOWS-KDA-VERIFY | Base-fla migration | ✓ VERIFIED | Zero retired forms; required base/built-in wording present; 4x `-e .` + `[project] dependencies` in the Windows runbook |
| Six notebooks + three Markdown mirrors | Base-install commands only | ✓ VERIFIED | Zero `full,fla`; JSON-valid; 24-pair sync gate green (per 28-03 receipts + showcase guard tests) |
| `tests/models/test_plant_helixseek_fla_kernels.py`, `tests/test_extras_guard.py` | Packaging contract tests | ✓ VERIFIED | TestFlaBaseDependency + TestCuratedExtras green against the shipped pyproject |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| pyproject.toml | ci.yml | Retired-extra removal from all install commands | ✓ WIRED | `full,fla` count 0 across workflows; `.[full,dev,mcp]` present |
| extras_compat.yml | scripts/install_flash_attn.sh | Sole flash-attn install route | ✓ WIRED | Lane line 125 invokes only the script; no bare pip/uv flash-attn install; forbidden-shape contract test green |
| extras_compat.yml | tests/models/compat_census.json | Seed fresh evidence from committed artifact | ✓ WIRED | Lane line 164 `cp tests/models/compat_census.json census-artifacts/extras-fresh.json` + seed-strip forcing |
| census-artifacts/extras-fresh.json | dnallm/models/model_info.yaml | Gate 1 green-record assertion + fail-closed --verify | ✓ WIRED | Both gate steps present in lane; replica receipt shows 49-matched with both probe rows matched |
| compat_census.json | model_info.yaml | census_compat.py --write-registry --apply | ✓ WIRED | Offline --verify re-run by verifier: 49 compared, all matched, OK |
| support.py | model_compatibility.md | Lazy DOCS_STATUS_CELLS + --emit-docs | ✓ WIRED | Re-emission byte-identical to the committed page |
| model.py | special/crossdna.py | Specific ImportError survives the generic wrap | ✓ WIRED | `_dnallm_missing_extra_guidance` marker checked at model.py:1219 before traversal; both CrossDNA from_pretrained paths routed |
| pyproject.toml | README.md / WINDOWS-KDA-VERIFY.md | Published tables and runbook reflect base fla | ✓ WIRED | Zero retired forms; base/built-in/marker wording present in all six surfaces |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| model_info.yaml CrossDNA row | tf_compat_status/evidence | Measured probe @cd118dc + writer | Yes — record exists in .scratch/p28-extras/compat.json with load.ok/forward_ok true | ✓ FLOWING |
| compat_census.json CrossDNA seed | disposition.extra + load record | Plan-01 measured record copied per 28-02 | Yes — record verified in artifact | ✓ FLOWING |
| model_compatibility.md rows | DOCS_STATUS_CELLS + registry | census_compat.py generator over the 49-record artifact | Yes — byte-identity regeneration proves generator drives the page | ✓ FLOWING |
| support words | SUPPORT_WORD_BY_REPO | registry resolve() | Yes — runtime resolve + consumer surfaces green | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Installer rejects corrupted bytes before extraction | corrupted tmp file + `--verify-file` | exit 1, `sha256 MISMATCH` in log | ✓ PASS |
| Missing-extra guidance (direct + nested deepspeed, CrossDNA both paths, cause preservation, nonmatching controls) | pytest TestMissingExtraGuidance (17 tests) | 17 passed | ✓ PASS |
| Registry verdicts at runtime | `resolve()` on the four repos | CrossDNA=extra-required-tf5, BigBird/GPN/Omni-DNA=remote-code-broken | ✓ PASS |
| Census drift gate offline | `census_compat.py --verify` | 49 compared, all matched, OK (incl. CrossDNA extra-required-tf5) | ✓ PASS |
| Generated page determinism | `--emit-docs` to tmp + `cmp` | byte-identical | ✓ PASS |
| Packaging + lane contracts | pytest extras_guard + fla_kernels + runner_infra | 61 passed | ✓ PASS |
| Fail-closed registry/support/docs battery | pytest six gate files | 402 passed | ✓ PASS |
| Strict docs build | `mkdocs build --strict` | exit 0, built in 4.78s | ✓ PASS |
| GPU probes (load+forward on cuda) | Not re-runnable on this verifier (needs GB10 + 64-min build) | Replica receipts inspected: .scratch/p28-extras/compat.json and .scratch/p28-lane/extras-fresh.json contents match SUMMARY claims exactly | ✓ PASS (replica evidence, owner-sanctioned interim class) |

### Probe Execution

No `scripts/*/tests/probe-*.sh` convention exists in this project; the phase's probe equivalents are the census/verify/lane commands above, which the verifier executed directly (offline ones) or inspected as on-disk replica receipts (GPU ones).

### Requirements Coverage

This project has no REQUIREMENTS.md; tracking rides PLAN/SUMMARY frontmatter over CONTEXT.md D-01..D-17. Shared-ID semantics: an ID completes when its LAST declaring plan finished.

| Requirement | Declared by (last declarer) | Status | Evidence |
| ----------- | --------------------------- | ------ | -------- |
| D-01 | 01? No — 02 | ✓ SATISFIED | Registry Literal + all six fail-closed surfaces; tests green |
| D-02 | 02, 04 (04) | ✓ SATISFIED | census_compat.yml byte-untouched (diff empty, last touch pre-phase); default census keeps measuring failures (BigBird seed load.ok=False); extras lane answers the mirror question |
| D-03 | 01, 02 (02) | ✓ SATISFIED (ruling A) | CrossDNA flip on measured green (truth 7); BigBird non-flip on measured red IS the discipline (truth 8); override recorded |
| D-04 | 02 | ✓ SATISFIED (ruling A) | Single status value; BigBird evidence names deepspeed 0.19.6 + rung + split; CrossDNA names flash-attn; no combined enums |
| D-05 | 04 | ✓ SATISFIED | extras_compat.yml posture verified (truth 4) |
| D-06 | 04 | ✓ SATISFIED | Fourth selector + exactly-once nightly (truth 5) |
| D-07 | 04 | ✓ SATISFIED | Census single-entry reuse via runtime-generated length-64 copy; committed tool pinned never `--run` in the lane |
| D-08 | 04 | ✓ SATISFIED | Two independent gates; disposition alone can never satisfy gate 1 (raw-booleans contract pinned) |
| D-09 | 01, 03, 05 (05) | ✓ SATISFIED | Base promotion verified across pyproject/CI/docs/notebooks (truth 3) |
| D-10 | 01 | ✓ SATISFIED | Extra names/bands verified |
| D-11 | 01 | ✓ SATISFIED | No markers on either extra |
| D-12 | 01, 04 (04) | ✓ SATISFIED | Pinned installer verified + lane-only route |
| D-13 | 01, 05 (05) | ✓ SATISFIED | No meta-extra membership; aggregate descriptions honest |
| D-14 | 02, 03 (03) | ✓ SATISFIED (ruling A) | CrossDNA exact word; BigBird keeps honest "needs deepspeed"; shared vocabulary single-source |
| D-15 | 02 | ✓ SATISFIED | Page regenerated; DOCS_DATA_DATE 2026-10-29 (probe date, not wall clock); byte-identity gate re-run |
| D-16 | 03 | ✓ SATISFIED | Full affected-surface rewrite; GPN/Omni-DNA byte-verbatim (sha256-proven) |
| D-17 | 02 | ✓ SATISFIED | Both load-chain interception points wired; 17 behavioral tests green |

All 17 declared IDs accounted for; no orphans (CONTEXT declares D-01..D-17; every ID appears in at least one plan's requirements field).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| docs/getting_started/quick_start.md | 347 | WR-01 (open review finding): edited line still lists removed `base` dependency group | ⚠️ Warning | A user following the list may run `pip install 'dnallm[base]'` and hit the warn-and-install-core-only outcome; docs-only inaccuracy adjacent to (not part of) a must-have |
| scripts/install_flash_attn.sh | 207,228,327-331 | WR-02 (open review finding): three `mktemp -d` dirs never cleaned on any exit path | ⚠️ Warning | Nightly lane accumulates hundreds of MB/night in /tmp on the single self-hosted runner; eventual disk exhaustion would red every lane on that box. Operational risk, not a must-have failure |
| scripts/census_compat.py | 1424-1431, 2046 | IN-01/IN-02 (info): TF_COMPAT_STATUSES dead constant; stale docstring count | ℹ️ Info | Maintainability only |
| docs/user_guide/model_compatibility.md | 42 | IN-03 (info): generic library-change lead says "cannot be loaded" while the suffix says load works with the extra | ℹ️ Info | Wording tension on the BigBird row; class assignment is owner-approved |
| dnallm/models/model.py | 1397 | Pre-existing `TODO: Add more special cases` (dated 2026-03-27, pre-phase) | ℹ️ Info | Not phase-introduced; no unreferenced TBD/FIXME/XXX introduced by this phase |

Debt-marker gate: the only TBD/FIXME/XXX-regex hits in phase files are the literal `cuXXX` CUDA-version placeholder in pre-existing installation.md prose (line 596, not touched by this phase's edits) and the 2026-03-27 TODO above. No unreferenced phase-introduced markers.

All five review findings are recorded as `open` in 28-REVIEW-DISPOSITION.md and are advisory per the phase's binding context; none blocks a must-have.

### Human Verification Required

### 1. Optional-dependency guide prose adequacy (biologist readability)

**Test:** Read the rewritten `docs/user_guide/optional_dependencies.md` as a biologist would: the "What is an optional extra?" intro, the BigBird deepspeed guide, the CrossDNA flash-attn guide, and the PlantHelixSeek built-in section.
**Expected:** Every section is understandable without engineering background; install commands are copy-pasteable; compile-time expectations (hours-scale CUDA build) and Linux+CUDA constraints are clear.
**Why human:** Prose adequacy under the 2026-10-21 owner plain-language directive is a judgment call; automated legs prove content presence, protected-section hashes, and build validity only. Both 28-03-SUMMARY and 28-05-SUMMARY flagged this as human_judgment: true.

### 2. Cross-page narrative consistency of the BigBird/CrossDNA story

**Test:** Read the rewritten README installation journey (extras table, "Built-in kernels" statement, PlantHelixSeek "No extra is needed" section) and the generated `model_compatibility.md` BigBird row together.
**Expected:** A biologist cross-reading the static guide and the generated page gets one consistent story: CrossDNA works after installing its extra; BigBird needs deepspeed to load but cannot run predictions (upstream-dead Triton API).
**Why human:** The byte-level gates prove the two pages share source strings, not that the combined reading reads clearly; IN-03's lead/suffix tension makes the human read worth doing.

### Gaps Summary

No gaps. All 13 consolidated truths verified, all artifacts present/substantive/wired with real data flowing, all key links connected, all 17 requirement IDs satisfied under the documented owner ruling A, and the census drift gate (49 matched) plus docs byte-identity re-confirmed by this verifier. The only open items are (a) the two human prose/readability checks above and (b) five advisory review findings recorded open in 28-REVIEW-DISPOSITION.md (WR-01 docs residue, WR-02 installer temp-dir leak on the nightly runner, three info). The live extras-lane dispatch remains a documented post-v1.4-to-main-merge operator step (GitHub registers dispatchable workflows from the default branch only; main is frozen during v1.4) — the committed lane bytes and the runner-replica receipts are the sanctioned interim evidence, same class as Phase 27's census-lane deferral.

---

_Verified: 2026-10-29T17:22:27Z_
_Verifier: Claude (gsd-verifier)_
