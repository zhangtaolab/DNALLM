---
phase: 27-transformers-4-5-model-compatibility-census
verified: 2026-10-27T10:54:26Z
status: human_needed
score: 25/25 must-haves verified
covered_files:
  - .github/dependabot.yml
  - .github/workflows/census_compat.yml
  - .planning/phases/27-transformers-4-5-model-compatibility-census/27-01-PLAN.md
  - .planning/phases/27-transformers-4-5-model-compatibility-census/27-01-SUMMARY.md
  - .planning/phases/27-transformers-4-5-model-compatibility-census/27-02-PLAN.md
  - .planning/phases/27-transformers-4-5-model-compatibility-census/27-02-SUMMARY.md
  - .planning/phases/27-transformers-4-5-model-compatibility-census/27-03-PLAN.md
  - .planning/phases/27-transformers-4-5-model-compatibility-census/27-03-SUMMARY.md
  - .planning/phases/27-transformers-4-5-model-compatibility-census/27-04-PLAN.md
  - .planning/phases/27-transformers-4-5-model-compatibility-census/27-04-SUMMARY.md
  - dnallm/models/model_info.yaml
  - dnallm/models/registry.py
  - dnallm/utils/transformers_compat.py
  - docs/user_guide/model_compatibility.md
  - mkdocs.yml
  - scripts/census_compat.py
  - tests/models/compat_census.json
  - tests/models/test_census_compat.py
  - tests/models/test_compat_docs_consistency.py
  - tests/models/test_model_info_registry.py
  - tests/models/test_registry.py
  - tests/utils/test_transformers_compat.py
  - tests/utils/test_transformers_compat_census.py
covered_digest: "v3:sha256:c402642817cdd7fdb97b25c554d254e348bdc406a6d4d70fc8dc4c9ee4d67ce4"
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "Dispatch the census lane live: gh workflow run census_compat.yml --ref v1.4, then inspect the run"
    expected: "The lane completes on [self-hosted, dnallm-nightly] with the census re-run matching all 49 committed tf_compat_status fields (exit 0, artifact uploaded), or turns red with a per-entry UNEXPECTED trail if the world has drifted"
    why_human: "The --verify fail-closed logic is fully verified offline (see Behavioral Spot-Checks: dry run, drift-injected, absent-artifact, and version-drift probes all exit correctly), but the live dispatch depends on external infrastructure (self-hosted runner registration, GPU, uv install) that cannot be exercised from the verifier; CR-01's fix is likewise verified offline only (seeded-run regression)"
  - test: "Read docs/user_guide/model_compatibility.md as the target audience (biologist) and sign off on plain language"
    expected: "Intro and the five pinned status wordings read plainly; decide whether the 3 WARNING-class plain-language findings recorded in 27-UI-REVIEW.md (engineering jargon in failure Details cells, missing inline footnote markers, long cells) warrant a generator+gate update"
    why_human: "Reading-level judgment for the biologist audience is subjective; the automated gate pins exact wordings but cannot judge audience comprehension; the UI review's findings are recorded advisory, owner triage pending"
---

# Phase 27: Transformers 4/5 Model Compatibility Census Verification Report

**Phase Goal:** Produce a per-model transformers 4/5 compatibility matrix for the registry's `pretrained:` section (49 of 236 entries; count corrected from a stale "238" via the roadmap update path), with import-level evidence; disposition every failure into `transformers_compat.py` shim fixes or the typed-skip + traceback path; deliverable: model-level compatibility registry table folded into docs.
**Verified:** 2026-10-27T10:54:26Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

Verified goal-backward against the ROADMAP goal and the locked decisions D-01..D-08 in 27-CONTEXT.md (the phase carries owner-directed `requirements: ["TBD"]`; no REQUIREMENTS.md exists in this project). Decision coverage gate: **8/8 decisions honored** (`check.decision-coverage-verify`: skipped=false, honored=8, total=8, not_honored=[]).

### Observable Truths

Goal-level truths:

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Per-model compatibility matrix exists for all 49 pretrained entries with per-entry import-level evidence under transformers 5.17.0 | ✓ VERIFIED | `tests/models/compat_census.json`: exactly 49 records in registry file order (roster equality vs `dnallm.models.registry.all_entries()` re-verified live), all `transformers == "5.17.0"`, statuses 27 import-ok / 18 arch-ok / 2 import-fail / 2 census-blocked; each import-fail carries an exact failing import (`ConvNetForMaskedLM not found in transformers 5.17.0`; `olmo`) |
| 2 | Every measured failure carries a disposition: shim fix (rung + RED-first test) or structural skip naming the exact cause | ✓ VERIFIED | 8 disposition records, all `skip` with concrete structural reasons >10 chars (checkpoint buffer defect, external deepspeed/olmo packages, removed native architecture, meta-device ALiBi, remote forward contract bug, absent repo id, no root config.json); the 2 fixable classes got rungs: `_patch_apply_chunking_to_forward` and `_patch_tied_weights_bookkeeping_on_load`, both registered in `apply_patches()` (transformers_compat.py:1697-1698) with RED commit 7d4a97e (16:19) preceding GREEN 759018a (16:30); the 2 tool-misroute load failures (6mer, PlantCAD2) were fixed in the measurement instrument and carry fresh green loads (documented in 27-02-SUMMARY) |
| 3 | Census verdicts are permanent typed registry fields: 49 pretrained only, 5-value vocabulary, tested stamp 5.17.0 only (D-05, D-07) | ✓ VERIFIED | Live loader assertion: 49/187 section split, statuses 19 verified-tf5 / 21 import-verified-tf5 / 1 shim-required-tf5 / 6 remote-code-broken / 2 census-blocked, zero finetuned entries carry any tf_compat_* field, `tf_compat_tested == "5.17.0"` on all 49, no empty evidence; schema declared in `dnallm/models/registry.py:133-144` (Literal + min_length=1 Fields); yaml carries exactly 49x3 field lines, every insertion anchored after a `max_length` line, dated provenance block appended |
| 4 | Model-level compatibility table folded into docs (plain-language for biologists, honest two-sided labels, generated + drift-gated) | ✓ VERIFIED | `docs/user_guide/model_compatibility.md`: GENERATED header, plain intro introducing transformers, computed count sentence (19/21/1/6/2 from the registry), 49-row table, both provenance footnotes verbatim incl. Chinese originals (lines 62-63); `mkdocs.yml:68` nav entry; byte-identical regeneration re-executed by verifier (`--emit-docs /tmp` + `cmp`: PASS) |

Plan-level truths (27-01):

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 5 | `--run` produces one evidence record per pretrained entry from registry accessors (never parsing model_info.yaml) | ✓ VERIFIED | Artifact: 49 records, order equals registry order (verified live); roster via `dnallm.models.registry`; sole-parser allowlist in tests/models/test_registry.py:514 still `{"dnallm/models/registry.py"}` — unchanged, and `dnallm/` mentions of model_info.yaml remain registry.py only |
| 6 | Every remote-code verdict from an isolated subprocess importing dnallm BEFORE any remote module exec | ✓ VERIFIED | `build_probe_program` (scripts/census_compat.py:563-620): program string places `import dnallm` before `spec_from_file_location`/`exec_module`; `subprocess.run` per entry (fixed argv, no shell); probe env sanitized per WR-03 (BLOCKED_PROBE_ENV_PREFIXES strips GITHUB_/RUNNER_/ACTIONS_) |
| 7 | Every record carries hub (ms\|hf) + non-empty revision | ✓ VERIFIED | 25 ms / 24 hf; zero empty revisions (live artifact scan) |
| 8 | Unreachable-on-both-hubs entries are loud, recorded, and fail-closed — never a silent skip | ✓ VERIFIED (documented deviation — CR-01) | Record + loud print hold: `UNREACHABLE <repo>` per entry plus `CENSUS BLOCKED (N unreachable on both hubs)` summary; 2 census-blocked records committed with reasons. **Deviation:** bare `--run` now exits 0 for committed-known unreachable entries (CR-01 critical fix, commit 16363d4, test-pinned by `test_unreachable_entry_reports_loudly_and_exits_zero`); the non-zero exit moved to the `--verify` verdict gate, where a NEW unreachable entry diffs as UNEXPECTED → exit 1 (verifier exercised this path live). Relocation was required by D-08: with 2 permanently census-blocked committed entries, the plan-literal exit-1 would make the dispatch lane permanently red and (pre-CR-01) actually masked verify results. Override block offered below if formal acceptance is wanted |
| 9 | Import census fetches code only — zero weight files | ✓ VERIFIED | `CODE_PATTERNS = ("*.py", "config.json", "tokenizer.json", "tokenizer_config.json", "*.model")` (scripts/census_compat.py:23) feeds both routes' allow-patterns; weights only in `--with-loads` |
| 10 | Every family representative carries load + one short-sequence forward (or a recorded failure) per D-03 | ✓ VERIFIED | 28 reps across 28 families; every passing-status rep carries a bool `forward_ok` (20 true / 4 false with exact tracebacks; zero missing); 4 parked honestly (2 import-fail pending shim — since recovered with measured loads, 2 census-blocked); every non-rep carries `load_note` naming its representative |

Plan-level truths (27-02):

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 11 | Every fixable failure has a compat rung in `apply_patches()` paired with a test that failed before the rung existed | ✓ VERIFIED | Rungs 12-13 defined at transformers_compat.py:1534/1604, registered at 1697-1698; RED commit 7d4a97e (16:19) precedes GREEN 759018a (16:30), both ancestors of HEAD; 19 per-rung contract tests in tests/utils/test_transformers_compat_census.py; `EXPECTED_PATCH_INSTALLERS` roster (tests/utils/test_transformers_compat.py:1343) extended with both names — pin test green |
| 12 | New rungs are probe-gated, absence-gated, sentinel-idempotent, no-op on 4.x | ✓ VERIFIED | Rung 12: try-import gate → `_attach_apply_chunking_to_forward` per-name absence attach + module sentinel; rung 13: `_post_init_computes_tied_weights_keys` probe gate + `_finalize_model_loading` existence gate + class sentinel, absence-gated per instance ("never overwrite" in `_materialize_tied_weights_bookkeeping`); contract tests passed in the verifier's run |
| 13 | Entries whose import a rung restored carry D-02a load + forward evidence | ✓ VERIFIED | The 2 `previous_status: import-fail` → `import-ok` entries (gena x2) both carry measured load records: gena-base `ok=true, forward_ok=false` (load succeeded; forward failed on the checkpoint token_type_ids buffer defect — honest split semantics), gena-bigbird `ok=false, forward_ok=false` (external deepspeed); both dispositioned skip with the rung named in the reason |
| 14 | Every remaining failure carries a disposition (fix rung name or structural reason) | ✓ VERIFIED | 8 dispositions; offline `TestDispositionCompleteness` passed; verifier re-ran the plan's disposition assertion shape against the committed artifact — no failure record (current or previous import-fail, or forward_ok false) lacks a well-formed disposition |
| 15 | No catch-all `__getattr__` on any transformers module (closed per-name map) | ✓ VERIFIED | No `setattr(..., "__getattr__", ...)` on transformers modules anywhere in transformers_compat.py; the only `__getattr__` definitions are dnallm's own `_QuantStatProxy` (line 308) and the pre-existing closed-map `PretrainedConfig.__getattr__` default (line 566, never-overwrite-guarded); rungs 12-13 install none; closed-map test green |

Plan-level truths (27-03):

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 16 | All 49 pretrained entries carry the three tf_compat_* fields; no finetuned entry carries any | ✓ VERIFIED | Live loader assertion + yaml scan (49x3 field lines, all in the pretrained section; the 3 `tf_compat` strings after the finetuned header are inside the dated provenance comment block, by design) |
| 17 | Every status in the 5-value vocabulary; every tested == 5.17.0 | ✓ VERIFIED | Live loader assertion over all 49; single distinct tested value |
| 18 | Schema fields landed at or before the yaml data (extra=forbid atomicity) | ✓ VERIFIED | Commit 7498ff2 (schema + loader tests) precedes b523d57 (data + data tests); both ancestors of HEAD in that order; the packaged registry parses (236 entries) |
| 19 | Committed artifact and committed yaml agree per entry via an independent consistency test | ✓ VERIFIED | `test_compat_census_artifact_matches_registry` re-implements the mapping in-test (no import from scripts/census_compat.py — verified by grep); passed in verifier's run; the `SHIM_RESTORED_LOADS` fact (PlantCaduceus_l32 → rung 13) is encoded independently in both writer and gate |
| 20 | REG-03 integrity: counts 49/187, provenance sentinels intact, loader sole parser, no degenerate values | ✓ VERIFIED | Full tests/models/test_registry.py + test_model_info_registry.py suites green (47 + 22 passed) incl. sentinel, count, degenerate-value, mentioner-allowlist guards |

Plan-level truths (27-04):

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 21 | Docs page renders one row per pretrained entry (49) derived from the typed registry; regeneration is byte-identical | ✓ VERIFIED | `cmp` of fresh `--emit-docs` output vs committed page: identical; generator reads `dnallm.models.registry` only (zero yaml.load in the emit path); determinism via pinned DOCS_DATA_DATE constant (wall-clock would break byte-identity — documented decision) |
| 22 | Every 4-side mention qualified as inferred; the two provenance footnotes verbatim; no unqualified "compatible with transformers 4 and 5" cell | ✓ VERIFIED | Footnotes at docs lines 62-63 verbatim incl. Chinese parentheticals; footnote 1 interpolates the committed tested stamp; the drift gate asserts the literal unqualified string absent from the table region (test green) |
| 23 | Standard CI lanes fail when the committed docs page drifts from the registry (offline gate) | ✓ VERIFIED | tests/models/test_compat_docs_consistency.py: 54 parametrized tests (6 functions), checker-independent of the generator, fail-closed on missing page/header/table; all passed in verifier's run; deliberate-red demonstrated twice in 27-02-SUMMARY/27-04-SUMMARY records |
| 24 | Dispatch lane re-runs the census and exits non-zero on any disagreement with committed registry fields (fail-closed) | ✓ VERIFIED | `verify_registry` fail-closed by construction; verifier exercised all four paths live: committed artifact → 49 matched, exit 0; drift-injected copy → `UNEXPECTED ... disagrees`, exit 1 (and refuses unmappable shapes — import-fail with no load and no disposition — rather than guessing); absent artifact → ERROR, exit 1; simulated transformers 5.19.0 → `VERSION DRIFT` trail, exit 1. Live dispatch on the self-hosted runner not yet run — routed to Human Verification |
| 25 | Lane is workflow_dispatch only, with event-name guard and contents: read; never push/PR/schedule | ✓ VERIFIED | Verifier re-ran the structural assertion: triggers `{workflow_dispatch}` only, permissions `{contents: read}`, runs-on `[self-hosted, dnallm-nightly]`, `if: github.event_name == 'workflow_dispatch'`, timeout 240; no `\|\| true` masking; `if: always()` artifact upload; loud gpu_absent marker + exit 1; zero huggingface.co URLs in the workflow |

**Score:** 25/25 truths verified (0 present-but-behavior-unverified)

**Score note (truth 8):** counted VERIFIED on the strength of the never-silent contract and the relocated-but-proven fail-closed semantics; the literal "bare `--run` exits non-zero" clause was superseded by the CR-01 critical review fix, which the phase's accepted review process dispositioned as `fixed` (27-REVIEW-DISPOSITION.md, commit 16363d4, seeded regression pinning the new exits). To formalize this deviation as an accepted override, add:

```yaml
overrides:
  - must_have: "A repo unreachable on both hubs turns the census run non-zero with a census-blocked record written - never a silent skip"
    reason: "CR-01 (critical): with 2 committed-known census-blocked entries, plan-literal exit-1 made the dispatch lane permanently red and masked --verify results; fail-closed semantics relocated to the --verify verdict gate (new unreachability still exits 1), loud never-silent reporting preserved and test-pinned"
    accepted_by: "{owner}"
    accepted_at: "{ISO timestamp}"
```

### Decision Coverage

`check.decision-coverage-verify`: **8/8 honored, 0 not honored** (D-01 pretrained-only through D-08 dispatch-only lane all trace to shipped artifacts). Non-blocking gate; recorded for review.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/census_compat.py` | Re-runnable census tool: fetch/classify/probe/load, `--write-registry`, `--emit-docs`, `--verify` | ✓ VERIFIED | 2065 lines; all 8 CLI flags present; all modes exercised by the verifier (dry --verify, emit-docs cmp, drift/absent/version probes) |
| `tests/models/compat_census.json` | 49-record committed evidence artifact | ✓ VERIFIED | 49 records, registry order, full provenance, dispositions complete |
| `tests/models/test_census_compat.py` | Offline unit tests over the tool's pure logic | ✓ VERIFIED | 1173 lines, 83 test functions, all passed |
| `dnallm/utils/transformers_compat.py` (rungs 12-13) | Census-driven compat rungs registered in apply_patches() | ✓ VERIFIED | Both present, registered, v4.49.0-cited vendored fallbacks |
| `tests/utils/test_transformers_compat_census.py` | Per-rung TDD contract tests | ✓ VERIFIED | 19 tests, RED-first commit evidence in git order |
| `dnallm/models/registry.py` (tf_compat_* fields) | Three optional-nullable typed fields, 5-value Literal | ✓ VERIFIED | Lines 133-144 + honest Attributes docstring |
| `dnallm/models/model_info.yaml` (49 enriched rows + provenance) | Anchored insertion, never a round-trip | ✓ VERIFIED | 49x3 field lines all after max_length anchors; zero yaml.dump in the writer; provenance block present |
| `tests/models/test_registry.py`, `tests/models/test_model_info_registry.py` | REG-03 extensions + independent consistency gate | ✓ VERIFIED | 47 + 22 tests passed |
| `docs/user_guide/model_compatibility.md` | Generated plain-language page, GENERATED header | ✓ VERIFIED | 49 rows + footnotes; byte-identical regen |
| `tests/models/test_compat_docs_consistency.py` | Offline drift gate | ✓ VERIFIED | 54 tests passed, checker-independent |
| `.github/workflows/census_compat.yml` | Dispatch-only re-verification lane | ✓ VERIFIED | Structural assertion re-run by verifier |
| `mkdocs.yml` nav entry | Model Compatibility under User Guide | ✓ VERIFIED | Line 68 |
| `.github/dependabot.yml` | KNOWN RISK pointer naming the lane re-run command | ✓ VERIFIED | Lines 28-30 |

All artifacts: exists ✓, substantive ✓, wired ✓ (see Key Links), data flowing ✓ (see Data-Flow Trace).

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| Census roster | `dnallm.models.registry.all_entries()` | kind == "pretrained" filter | ✓ WIRED | Artifact order equals registry order (live check); tool never parses model_info.yaml |
| Probe subprocess | shim ordering | `import dnallm` precedes remote exec in program string | ✓ WIRED | build_probe_program source order verified |
| Evidence artifact | 27-02 dispositions / 27-03 write-back | compat_census.json consumed by --write-registry and the independent consistency test | ✓ WIRED | Dry --verify maps all 49 records to committed statuses, matched |
| Rungs 12-13 | `apply_patches()` | registration before module-level invocation | ✓ WIRED | Lines 1697-1698 precede line 1703 `apply_patches()` call |
| Registry verdicts | Docs page | `dnallm.models.registry` accessors only (no yaml parse) | ✓ WIRED | Byte-identical regen proves the derivation; grep confirms no yaml.load in emit path |
| Registry fields | CI drift gate | test re-derives 49 expected rows independently | ✓ WIRED | 54 tests green; no import from the generator |
| Fresh census | Committed fields | `--verify` diff via typed loader | ✓ WIRED | All four exit paths exercised live by the verifier |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| compat_census.json | entries[49] | Real hub fetches + subprocess probes + CUDA load/forward runs (hub/revision/fetched_at stamps per record) | Yes | ✓ FLOWING |
| model_info.yaml tf_compat_* | per-row verdicts | Anchored insertion from the committed artifact via --write-registry | Yes | ✓ FLOWING |
| registry.py typed fields | tf_compat_status/evidence/tested | Parsed from the packaged yaml at load | Yes | ✓ FLOWING |
| model_compatibility.md | 49 table rows + counts | Generated from typed registry accessors | Yes | ✓ FLOWING |
| Workflow census step | census-fresh.json | `--run --with-loads --verify` fresh run vs committed fields | Yes (logic verified offline; live run pending — human item) | ✓ FLOWING (offline-proven) |

No static returns, hardcoded model lists, or mock-fed values anywhere in the chain. Docs count sentence computed from the registry (19/21/1/6/2 matches the live registry distribution exactly).

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Five phase test files | `pytest tests/models/test_census_compat.py tests/utils/test_transformers_compat_census.py tests/models/test_registry.py tests/models/test_model_info_registry.py tests/models/test_compat_docs_consistency.py -q` | 225 passed in 3.45s | ✓ PASS |
| Registry loader serves the fields | typed-loader assertion (49/187/None/stamp) | exit 0; distribution 19/21/1/6/2 | ✓ PASS |
| Byte-identical docs regeneration | `--emit-docs /tmp && cmp` | identical | ✓ PASS |
| Dry verify (green path) | `--verify` on committed artifact | 49 matched, exit 0 | ✓ PASS |
| Drift detection | `--verify` on drift-injected copy (status flipped, load nulled) | `UNEXPECTED ... disagrees`, exit 1 | ✓ PASS |
| Unmappable record refused | same probe (import-fail, no load, no disposition) | `UNEXPECTED ... unmappable`, exit 1 — refuses to guess | ✓ PASS |
| Absent artifact | `--verify --artifact /tmp/does-not-exist.json` | ERROR, exit 1 | ✓ PASS |
| Version drift | verify_registry with monkeypatched transformers_version → "5.19.0" | `VERSION DRIFT` trail, rc=1 | ✓ PASS |
| Workflow posture | structural YAML assertion | dispatch-only, contents:read, runner, guard, timeout 240 | ✓ PASS |
| Ruff on 8 phase code files | `ruff check` | All checks passed | ✓ PASS |

Note on exit-code capture: two initial probe readings of "exit 0" were pipe-contaminated (`$?` captured `tail`); all values above were re-measured with output redirection and are trustworthy.

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes declared for this phase (plans carry the spec-less probe fallback note; no requirement IDs to probe). The census tool itself is the phase's probe surface and was exercised directly — see Behavioral Spot-Checks.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| TBD (owner-directed 2026-10-27) | 27-01..27-04 | No formal REQ-IDs exist; judged against the phase GOAL and D-01..D-08 per the verification tasking | ✓ SATISFIED (goal + decisions) | Goal truths 1-4 VERIFIED; decision coverage 8/8; no orphaned requirements possible (no REQUIREMENTS.md in project) |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| dnallm/utils/transformers_compat.py | 693 | `TODO (joao): remove =None in non-optional args in v4.46` | ℹ️ Info | Pre-existing (blame: commit 98c39f8, 2026-10-02 — before Phase 27); text is a vendored upstream transformers maintainer comment inside a v4.49.0-cited block, not dnallm debt; no action |
| docs/user_guide/model_compatibility.md | 40-55 | Failure Details cells carry engineering jargon; long cells; no inline footnote markers | ⚠️ Warning | 3 WARNING-class plain-language findings recorded in 27-UI-REVIEW.md (advisory, owner triage pending) — does not violate the pinned-wording contract the gate enforces |
| Review ledger | — | 6 info findings open (IN-01..IN-06: ty-ignore convention, fix-branch omission in independent checker, machine-specific evidence text, kwargs-only wrapper edge, unpinned uv install, docs_blocked_reason fail-open edge) | ℹ️ Info | Recorded open in 27-REVIEW-DISPOSITION.md — out of code-review fix scope by owner triage; IN-02's latent branch (fix-disposition mapping in the independent checker) is currently vacuous: the committed artifact carries zero `fix` dispositions |

Debt-marker gate: zero TBD/FIXME/XXX in any phase-modified file. No stubs, no empty implementations, no console-log-only paths, no hardcoded model data.

### Test Quality Audit

| Test File | Linked Truths | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|--------------|--------|---------|----------|-----------------|---------|
| tests/models/test_census_compat.py | 5-10, 14, 24 | 83 | 0 | 0 | Value/behavioral (exit codes, record shapes, exact strings) | OK |
| tests/utils/test_transformers_compat_census.py | 11, 12, 15 | 19 | 0 | 0 | Value/behavioral (resolution, idempotency, alias identity, semantics) | OK |
| tests/models/test_registry.py | 18, 20 | 47 | 0 | 0 | Value (ValidationError, defaults, sentinels) | OK |
| tests/models/test_model_info_registry.py | 16, 17, 19, 20 | 22 | 0 | 0 | Value (coverage, mirrors, rung existence re-checked against shim source) | OK |
| tests/models/test_compat_docs_consistency.py | 21, 22 | 54 (6 parametrized) | 0 | 0 | Value (exact wordings, verbatim footnotes, order) | OK |

**Disabled tests on requirements:** 0. **Circular patterns:** 0 — both consistency gates re-implement their derivations independently of the writer/generator (grep-verified: no imports from scripts/census_compat.py); the three-seam design (empirical artifact ↔ typed registry ↔ generated docs) checks each seam against a different evidence source, so no seam self-approves. **Insufficient assertions:** 0.

### Human Verification Required

### 1. Live dispatch of the census lane

**Test:** Run `gh workflow run census_compat.yml --ref v1.4` and inspect the resulting run.
**Expected:** Lane completes on the self-hosted GPU runner with all 49 committed statuses matched (exit 0, census-fresh.json artifact uploaded), or turns red with a per-entry UNEXPECTED / VERSION DRIFT trail.
**Why human:** The `--verify` fail-closed logic is fully proven offline (all four exit paths exercised during this verification), but the live run depends on external infrastructure — runner registration, GPU availability, uv install on the box — that the verifier cannot exercise. This is the one caveat the phase itself carries honestly (CR-01 fix verified offline only).

### 2. Biologist-audience readability sign-off on the docs page

**Test:** Read `docs/user_guide/model_compatibility.md` as a biologist and decide on the 3 recorded plain-language warnings.
**Expected:** Intro and pinned wordings read plainly; owner decides whether the Details-cell jargon, missing inline footnote markers, and long cells warrant a generator+gate update (both must change in lockstep to stay byte-deterministic).
**Why human:** Reading-level judgment for the target audience is subjective; the automated gate pins exact wordings but cannot judge comprehension. 27-UI-REVIEW.md (9/12 applicable pillars) already records the findings; final disposition is the owner's.

### Gaps Summary

No gaps. All 25 truths verified, all 13 artifacts present/substantive/wired/flowing, all 7 key links wired, 225 tests green, all behavioral spot-checks pass, decision coverage 8/8. The phase goal — a per-model transformers 4/5 compatibility matrix with empirical import-level evidence, every failure dispositioned (2 shim rungs for the fixable classes, typed-skip for the 8 structural ones), verdicts written back as typed registry fields on exactly the 49 pretrained rows, and a generated plain-language docs table locked against drift from both directions — is achieved in the codebase, not merely claimed.

Two non-gap observations for the record: (1) truth 8's exit-code relocation (CR-01) is a documented, test-pinned deviation from the plan-literal wording — override block offered above for formal acceptance; (2) the roadmap taxonomy buckets `both`/`4-only`/`5-only` are realized as the honest-label vocabulary instead (verified-tf5 / import-verified-tf5 / shim-required-tf5 / remote-code-broken / census-blocked) per D-05's single-environment decision — the `5-only` bucket stays honestly empty and the 4-side is labeled inferred everywhere, exactly as the decision requires.

---

_Verified: 2026-10-27T10:54:26Z_
_Verifier: Claude (gsd-verifier)_
