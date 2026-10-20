---
phase: 16-windows-parity-rollout-dispatch-gates-frozen-value-parity
verified: 2026-10-14T11:05:00Z
status: passed
score: 5/5 must-haves verified
covered_files:
  - .gitignore
  - .planning/phases/16-windows-parity-rollout-dispatch-gates-frozen-value-parity/16-01-PLAN.md
  - .planning/phases/16-windows-parity-rollout-dispatch-gates-frozen-value-parity/16-01-SUMMARY.md
  - .planning/phases/16-windows-parity-rollout-dispatch-gates-frozen-value-parity/16-02-PLAN.md
  - .planning/phases/16-windows-parity-rollout-dispatch-gates-frozen-value-parity/16-02-SUMMARY.md
  - .planning/phases/16-windows-parity-rollout-dispatch-gates-frozen-value-parity/16-03-PLAN.md
  - .planning/phases/16-windows-parity-rollout-dispatch-gates-frozen-value-parity/16-03-SUMMARY.md
  - .planning/phases/16-windows-parity-rollout-dispatch-gates-frozen-value-parity/16-04-PLAN.md
  - .planning/phases/16-windows-parity-rollout-dispatch-gates-frozen-value-parity/16-04-SUMMARY.md
  - docs/example/notebooks/finetune_NER_task/data_generation_and_inference.ipynb
  - docs/example/notebooks/finetune_NER_task/generate_bpe_dataset.py
  - docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
  - docs/example/notebooks/plant_helixseek_shared/data/selection.md
  - example/notebooks/finetune_NER_task/data_generation_and_inference.ipynb
  - example/notebooks/finetune_NER_task/generate_bpe_dataset.py
  - example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
  - example/notebooks/plant_helixseek_shared/data/selection.md
  - tests/examples/test_examples.py
  - tests/examples/test_notebook_execution.py
  - tests/examples/test_plant_helixseek_showcase.py
  - tests/examples/test_script_execution.py
  - tests/fixtures/genomic_intervals/PROVENANCE.md
  - tests/fixtures/genomic_intervals/anno_pred.bed
  - tests/fixtures/genomic_intervals/anno_truth.bed
  - tests/fixtures/genomic_intervals/cre_pred.bed
  - tests/fixtures/genomic_intervals/cre_truth.bed
  - tests/fixtures/genomic_intervals/loj_a.bed
  - tests/fixtures/genomic_intervals/loj_b.bed
  - tests/fixtures/genomic_intervals/loj_bedtools_oracle.tsv
  - tests/test_dispatch_direction.py
  - tests/test_frozen_pairs.py
covered_digest: "v3:sha256:84e6c178f9d0f49137a2f4aee3d13d299aa9ee60082960c609ac0df71070b7c0"
behavior_unverified: 0
overrides_applied: 1
overrides:
  - must_have: "TRUTH-1 (SC4, 16-04): a manually dispatched nightly run — executing the COMPLETE phase change on the dispatched ref — concludes SUCCESS for the example-nightly job ... and for the coverage-nightly job"
    reason: "Both nightly jobs of run 38441258701 concluded failure, but the only reds are the 4th/5th manifestations of the rice.uga.edu external-host class already owner-accepted in phase 14 (14-EVIDENCE H.4/H.4.final, explicitly carried into phase-16 planning), both upstream of all phase-16 code (verified from junit/log tracebacks: _seed_rice_input/_download/copyfileobj and the wget genome-download cell). Every phase-16 surface was green in the run (census gate byte-identical, CRE band re-executed green, NER notebook end-to-end green, both root suites 19+4 green in the full census, live-parity executed). The ROADMAP SC4 contract itself — nightly re-execution holds the parsed selection.md bands — is met and verified without the override; this override covers only the plan truth's stricter conclusion-success encoding."
    accepted_by: "owner (per 16-EVIDENCE A.5 owner-directed accept-with-caveat; WINDOWS #24 resolved fixed)"
    accepted_at: "2026-10-14T09:37:03Z"
---

# Phase 16: Windows Parity Rollout — Dispatch, Gates & Frozen-Value Parity Verification Report

**Phase Goal:** The example tree acquires its intervals natively on Windows — platform dispatch at acquisition seams only, platform-aware test gates, test-windows executing the numpy evaluators against pinned tolerances — while the Linux bedtools truth chain is structurally protected from dispatch leakage
**Verified:** 2026-10-14T11:05:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

Every claim below was re-proved independently of SUMMARY.md: files read, diffs inspected against the phase baseline `ff95ef7`, suites run by this verifier, and the SC4 CI record re-fetched live from GitHub (run/junit/job logs) rather than trusted from 16-EVIDENCE.md.

### Observable Truths

The five ROADMAP success criteria are the contract (Step 2c); the 24 plan-level truths across 16-01..16-04 map onto them without reducing scope.

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | SC1 (WPAR-03): `generate_bpe_dataset.py` and `data_generation_and_inference.ipynb` fork at acquisition only — Linux keeps `pybedtools.BedTool.intersect(loj=True)`, Windows dispatches to the numpy equivalent through the uniform row adapter; notebook install cell skips pybedtools on Windows | ✓ VERIFIED | Script diff `ff95ef7..HEAD` = exactly 3 hunks: header import seam (`if sys.platform == "win32": from dnallm.utils import loj_intersect` / `else: from pybedtools import BedTool`), module-level `LojRow`+`_loj_rows` block, 1-line call-site swap `intersection = _loj_rows(tokens_bed, annotation_bed)` — consumer loop byte-identical (gene flush, `iv.fields[4]/[9]/[10]`, `name == "-1"` all untouched). Notebook JSON-compared cell-by-cell vs baseline: source changes in exactly cells 0/1/2/8, outputs untouched; cell 0 POSIX keeps `!brew install bedtools` + `!uv pip install pyfastx pybedtools` verbatim, Windows branch installs pyfastx alone (no pybedtools/bedtools token — asserted by structure test); cell 1 `!which intersectBed` inside the POSIX guard; cell 2 same conditional pair; cell 8 adapter textually identical to the script (whitespace-normalized equality asserted + verified). No try/except around any dispatch import (AST-asserted; code read confirms). |
| 2 | SC2 (WPAR-04): the three inventoried gates are platform-aware — Linux probe-then-execute unchanged, Windows executes the numpy path fail-loud with no ever-green skip — with Windows-branch contract tests exercising the dispatch via monkeypatched platform | ✓ VERIFIED | Showcase gate `test_plant_helixseek_showcase.py:525`: `if sys.platform != "win32":` wraps the shutil.which assert with the exact message retained; `_gate_script_pybedtools` and `_gate_pybedtools` carry the win32 early-return as first statement with probe + typed skip verbatim after; `_ACTIVE_NOTEBOOK_GATES` registration untouched (`git diff` over the map = empty). Gate diffs vs baseline confined to guard/docstring additions + POSIX platform pins in the 4 contract tests (a documented Rule-3 fix keeping those tests truthful on test-windows). Windows-branch dispatch behavior proven by execution: `test_win32_contract_cell11_runs_evaluator_zero_subprocess` (poisoned subprocess.run never called; output equals evaluator at 4 decimals), `test_posix_contract_cell11_fails_loud_when_cli_unusable` (sentinel propagates), `test_ner_script_win32_rows_match_frozen_loj_truth` / `_live_pybedtools_loj`, `test_ner_script_posix_seam_import_fails_loud` — all green in this verifier's local run (23 passed) AND all 19 in the dispatched CI census. Skip-machinery ban asserted inside all three gate regions. |
| 3 | SC3 (WPAR-05): test-windows executes the numpy evaluators over committed frozen interval pairs with pinned tolerances; frozen metrics hold in band on BOTH paths — jaccard=0.324727 and exon_f1=0.7522 with its band newly pinned — and selection.md stays the single truth source parsed on both paths | ✓ VERIFIED | 8 fixture files git-tracked under `tests/fixtures/genomic_intervals/` (ls-files = 8; scoped `.gitignore:92` exception for the tsv); `tests/test_frozen_pairs.py` at tests/ root, unmarked, no skip machinery, no bedtools binary — runs on every leg incl. test-windows (eligibility: `pytest -m "not slow"` on windows-latest/.[base] collects it; grep pytest.mark = 0). Suite green locally (4 passed) and in the dispatched coverage-nightly census (all 4 PASSED rows in the job log). Truth sides independently regenerated byte-identical by this verifier: `cre_truth.bed` (94 rows) reproduces exactly from the committed TAIR10 DHS GFF via the documented half-open+sorted procedure; `anno_truth.bed` 526 rows = the committed GFF3 CDS count; `loj_b.bed` is a byte-copy of `cre_truth.bed` as PROVENANCE states. Band row: exactly one line added to selection.md (`Anno pooled exon-F1 (frozen pair) \| = 692/920 \| [0.752173912, 0.752173914] \| 0.7522`), 9-decimal endpoints, both mirrors; existing parser labels single-hit (frozen suite parses the same file the nightly parses — both green in the dispatch run). Both paths: numpy asserted in-suite (round(6)==0.324727 vs the same-session bedtools 6-dp oracle; exact 692/920 rational; parsed tight band); bedtools path re-executed live on CI — `test_cre_notebook_executes_within_selection_bands` PASSED 254.961s over the dispatched cell-11 change. |
| 4 | SC4 (WPAR-06/08): same-change dispatch-direction structure test proves the numpy path cannot silently execute on Linux (fail-loud, no except-fallback ladder), and the Linux nightly re-execution holds the parsed selection.md bands in the same change | ✓ VERIFIED (plan-truth conclusion-success clause via override — see frontmatter) | `tests/test_dispatch_direction.py` born in 16-01 (commit 8367bf5, same phase change as the first seam) and extended by 16-02; 19 tests — structure (predicate equality, POSIX CLI branch with check=True, AST no-try-wrapped-import, textual adapter identity, md cells frozen), runtime A/B (win32 zero-subprocess; POSIX sentinel propagates), NER parity (frozen 12-col truth + live pybedtools where installed) and ImportError fail-loud — all green locally (this verifier, incl. the live-parity branch with the nightly bedtools prefix exported) and all 19 PASSED in the dispatched run's coverage-nightly census, including `test_ner_script_win32_rows_match_live_pybedtools_loj` (live truth-chain parity executed on CI). Nightly re-execution verified live via gh: run 38441258701, headSha `4863883890fe62c845e3c81da06c7e07d4e8e282` (= the complete 16-01/02/03 tree), test-mamba **success**; example-nightly: census hard gate `216/217 tests collected (1 deselected)` byte-identical, CRE band PASS 254.961s, NER notebook end-to-end PASS **4023.932s** on the POSIX bedtools chain, 5 skips = the historical allowlisted set, sole failure = rice.uga.edu `_seed_rice_input` timeout; coverage-nightly: `2 failed, 2607 passed, 17 skipped in 13390.29s`, all 19+4 root-suite tests PASSED, 2 failures both rice-class (NER CellTimeout on the wget genome-download cell; BPE script `_download` timeout) — both tracebacks verified upstream of all phase-16 code. Bands held everywhere in the run; the conclusion-success deviation is the documented owner-accepted rice class (4th/5th manifestations; 14-EVIDENCE H.4/H.4.final verified on file). |
| 5 | SC5 (WPAR-08): every example edit lands with its docs mirror byte-identical (`check_docs_sync` green) | ✓ VERIFIED | `scripts/check_docs_sync.py` → `OK: docs/example/ is in sync with example/` exit 0 (run by this verifier). Same-commit discipline verified from git: d05a3b9 (CRE nb + twin), 3ffa658 (NER script + twin), 87b0c13 (NER nb + twin), 92ac413 (selection.md + twin) — 4/4. |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

### Advisory (New Scope, Unevidenced)

Not a re-verification pass; section recorded for completeness of the honesty posture. One advisory-level observation (no deterministic evidence of a defect; does not affect status):

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| 1 | No real test-windows run has executed the phase-16 tree yet (v1.3 gets matrix legs only via PR to main/master/dev; none is open; the SC4 dispatch run skipped test-windows by its event gate). Eligibility is verified (root placement, unmarked, `.[base]`-importable, PYTHONUTF8/autocrlf mitigations in ci.yml), matching phase-15's verified precedent and the evidence file's honest "eligibility re-confirmed" wording — but the first genuine Windows execution lands with the v1.3 PR flow. | other | Design-level guarantee verified; no reproducible failure exists; matrix exposure is the standing branch workflow, not a phase gap |

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `example/notebooks/finetune_NER_task/generate_bpe_dataset.py` | acquisition-only fork: conditional import seam + inline LojRow/_loj_rows + call-site swap; consumer byte-identical | ✓ VERIFIED | 53+/3- diff in exactly 3 hunks; py_compile OK; seam probe green in-suite |
| `example/notebooks/finetune_NER_task/data_generation_and_inference.ipynb` | cells 0/1/2/8 forked, only those cells changed | ✓ VERIFIED | JSON cell-compare vs baseline: source deltas exactly cells 0/1/2/8; outputs byte-preserved |
| `example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb` | cell 11 platform dispatch only; md cells 0/10/22 untouched | ✓ VERIFIED | Only cell 11 changed; md cells byte-identical; win32 branch reads evaluator `["jaccard"]` full precision into `jaccard_value`; POSIX branch argv/check=True/parse preserved; shared 4-decimal print outside branches |
| `tests/examples/test_plant_helixseek_showcase.py` | gate at :523 platform-aware, nothing below touched | ✓ VERIFIED | 2 hunks (`import sys`, guard); band assertions/harness contract untouched |
| `tests/examples/test_script_execution.py` | `_gate_script_pybedtools` win32 early-return only | ✓ VERIFIED | Guard + docstring + contract-test POSIX pin; probe/typed skip verbatim |
| `tests/examples/test_notebook_execution.py` | `_gate_pybedtools` win32 early-return; registration untouched | ✓ VERIFIED | Guard + docstring + POSIX pins; `_ACTIVE_NOTEBOOK_GATES` diff-empty |
| `tests/examples/test_examples.py` | `_strip_magic_lines` pass-placeholder fix (Rule 3) | ✓ VERIFIED | Required by guarded magics; census unchanged; fast lane green |
| `tests/test_dispatch_direction.py` | NEW root structure + runtime-contract suite (WPAR-06) | ✓ VERIFIED | 19 tests, substantive (AST readers, monkeypatched-platform runtime contracts, live-parity); zero markers/skips; green locally + CI census |
| `tests/fixtures/genomic_intervals/` (8 files) | committed real pairs + dual oracles + PROVENANCE.md, one atomic commit | ✓ VERIFIED | ls-files = 8, all in commit a5c07cc; PROVENANCE carries env/commands/both oracle families/drift table; truth sides independently regenerated byte-identical by this verifier |
| `tests/test_frozen_pairs.py` | NEW root platform-agnostic frozen-pair suite | ✓ VERIFIED | 4 tests, bands parsed never copied, fail-red parse guard, tight band consumed here only; green locally + CI census |
| `example/.../selection.md` (+ docs twin) | exactly one additive band row | ✓ VERIFIED | 1-line diff; 9-decimal bracket endpoints; both copies line 74 |
| 16-EVIDENCE.md | SC4 record + 5/5 matrix + honesty ledger | ✓ VERIFIED | Every recorded id/conclusion/row re-checked live via gh and found accurate (incl. the honest failure records) |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| CRE cell 11 win32 branch | `dnallm.utils.jaccard` | `from dnallm.utils import jaccard as _np_jaccard`; `["jaccard"]` key | ✓ WIRED | Runtime contract A proves the call executes in-process with the evaluator value at 4 decimals |
| NER `_loj_rows` win32 branch | `dnallm.utils.loj_intersect` | module-level helper in script + cell 8, textually pinned identical | ✓ WIRED | Parity contracts green (frozen 12-col + live pybedtools on CI) |
| Both root suites → fixtures/selection.md | parsed bands + oracle tsv | repo-root path resolution, `_parse_band` duplicated (fail-red) | ✓ WIRED | Green locally and in the 2607-passed census |
| Gates' win32 early-return → dispatched execution | notebook/script numpy paths | guard precedes probe in all three gates (structure-tested) | ✓ WIRED | No skip machinery reachable on any Windows route |
| Nightly evidence → GitHub run 38441258701 | junit artifact + job logs | `gh run view/download` (re-run by this verifier) | ✓ WIRED | headSha, job ids, conclusions, per-test rows all match the record |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| CRE cell 11 | `jaccard_value` | win32: evaluator over real BED paths; POSIX: bedtools CLI stdout | ✓ | FLOWING |
| NER script/cell 8 | `intersection` | win32: `loj_intersect` over tokens/annotation beds; POSIX: BedTool loj | ✓ | FLOWING |
| frozen-pair suite | oracle values | recomputed from committed fixture bytes vs bedtools-measured oracles | ✓ | FLOWING |
| gates | probe outcomes | live find_spec/which probes (POSIX); platform predicate (win32) | ✓ | FLOWING |

No static returns, hardcoded empty data, or mock-fed renders anywhere in the phase surface.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Both new root suites | `.venv/bin/python -m pytest tests/test_dispatch_direction.py tests/test_frozen_pairs.py -q` | 23 passed in 3.67s | ✓ PASS |
| Live-parity branch (truth chain installed) | `PATH=<nightly-prefix> pytest tests/test_dispatch_direction.py::test_ner_script_win32_rows_match_live_pybedtools_loj` | 1 passed | ✓ PASS |
| Full fast lane | `.venv/bin/python -m pytest -m "not slow" --no-cov -q` | 2550 passed, 1 skipped, 75 deselected in 102.52s | ✓ PASS |
| Docs sync | `.venv/bin/python scripts/check_docs_sync.py` | OK, exit 0 | ✓ PASS |
| Live census | `pytest tests/examples --collect-only -q -m "not giants" \| tail -2` | `216/217 tests collected (1 deselected)` | ✓ PASS |
| SC4 run re-check | `gh run view 38441258701 --json conclusion,headSha` + per-job jq | failure @ 4863883; test-mamba success; example-nightly failure; coverage-nightly failure — matches record | ✓ PASS |
| example-nightly junit | `gh run download 38441258701 -n example-nightly-junit` → parsed | tests=216 failures=1 skipped=5; CRE band PASS 254.961s; NER nb PASS 4023.932s; sole FAIL = `_seed_rice_input` timeout | ✓ PASS |
| coverage-nightly log | `gh run view --job 115448126935 --log` → grep | `2 failed, 2607 passed, 17 skipped ... 13390.29s`; 19/19 dispatch + 4/4 frozen PASSED; both failures rice-class (wget-cell CellTimeout; `_download` timeout) | ✓ PASS |
| CRE truth determinism | verifier's own regeneration from committed GFF | 94/94 rows byte-identical | ✓ PASS |

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes exist or were declared for this phase; the SC4 verification vehicle is the CI dispatch (covered under Behavioral Spot-Checks). Not applicable.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| WPAR-03 | 16-02 | acquisition-only fork, uniform row adapter, install-cell skip | ✓ SATISFIED | Truth 1 |
| WPAR-04 | 16-01, 16-02 | three gates platform-aware, no ever-green skip | ✓ SATISFIED | Truth 2 |
| WPAR-05 | 16-03 | frozen pairs, pinned tolerances, both paths, single truth source | ✓ SATISFIED | Truth 3 |
| WPAR-06 | 16-01, 16-02, 16-04 | dispatch-direction structure test, fail-loud, no fallback ladder | ✓ SATISFIED | Truth 4 |
| WPAR-08 | 16-01, 16-02, 16-03, 16-04 | docs mirrors byte-identical + nightly holds parsed bands | ✓ SATISFIED | Truths 4-5 |

Orphaned requirements: none — ROADMAP assigns exactly WPAR-03/04/05/06/08 to phase 16; WPAR-07 is explicitly phase-18 scope (CONTEXT boundary). All five are marked Complete in REQUIREMENTS.md.

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| tests/test_dispatch_direction.py | WPAR-03/04/06 | 19 | 0 | 0 | Value (exact 12-col field lists; exception propagation; AST equality) | OK — oracles external (bedtools v2.31.1/pybedtools-measured literals) + live cross-check executed on CI |
| tests/test_frozen_pairs.py | WPAR-05 | 4 | 0 | 0 | Value (6-dp round equality, 1e-9 tolerance, exact rational, parsed intervals, line-by-line byte equality) | OK — provenance VALID (same-session bedtools dual oracle; truth sides independently regenerated by this verifier) |

Disabled tests on requirements: 0. Circular patterns: 0 (expected values derive from bedtools/pybedtools, the external truth chain — the system under test is the numpy path). Insufficient assertions: 0.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| (none) | - | No TBD/FIXME/XXX/TODO/HACK markers in any phase file; no stub returns; no console-log-only implementations; the gates' plain `return` is the deliberate platform early-return (documented, structure-tested, produces no skip entry) | - | - |

### Decision Coverage

No trackable decisions in CONTEXT.md (gate reports "no trackable decisions"); the owner locks (platform-split directive, frozen-value strategy, `tests/test_dispatch_direction.py` root placement) were all verified honored in the truths above.

### Human Verification Required

None. Infrastructure/foundation phase — no user-facing elements. All behavior-dependent invariants (fail-loud dispatch direction, no-fallback ladder, win32 zero-subprocess, null-sextet parity) have passing behavioral tests executed locally and on CI. The single judgment item in the phase record (SC4 green-where-executable acceptance of the rice.uga.edu external-host class) was already resolved by the owner and documented (16-EVIDENCE A.5; WINDOWS #24 fixed 2026-10-14T09:37:03Z; carried in this report's frontmatter as an override).

### Gaps Summary

No gaps. All five ROADMAP success criteria verified against the codebase and live CI evidence. The honesty ledger holds under independent re-measurement: census triple byte-identical (`216/217 tests collected (1 deselected)` in ci.yml and live collect-only), `expected_skips.yaml` / `pyproject.toml` / `uv.lock` / `ci.yml` diffs over `ff95ef7..HEAD` all EMPTY, fast lane 2550 passed, docs sync exit 0, phase footprint = exactly the plan's 23-file enumeration. The SC4 nightly run of record (38441258701) matches its evidence file on every re-checkable fact — job ids, conclusions, census line, per-test rows, and the two rice-class failure tracebacks (both upstream of phase-16 code; 4th/5th manifestations of the phase-14 owner-accepted class) — with the conclusion-success deviation carried as a documented, owner-accepted override.

---

_Verified: 2026-10-14T11:05:00Z_
_Verifier: Claude (gsd-verifier)_
