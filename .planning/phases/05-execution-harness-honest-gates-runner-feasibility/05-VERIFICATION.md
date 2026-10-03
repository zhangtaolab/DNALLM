---
phase: 05-execution-harness-honest-gates-runner-feasibility
verified: 2026-10-03T04:03:05Z
status: passed
score: 25/25 must-haves verified
covered_files:
  - .github/workflows/docs-validation.yml
  - .github/workflows/feasibility.yml
  - .gitignore
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-01-PLAN.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-01-SUMMARY.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-02-PLAN.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-02-SUMMARY.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-03-PLAN.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-03-SUMMARY.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-04-PLAN.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-04-SUMMARY.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-05-PLAN.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-05-SUMMARY.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-06-PLAN.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-06-SUMMARY.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-CENSUS.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-CONTEXT.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-FEASIBILITY.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW-DISPOSITION.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW-FIX.md
  - README.md
  - dnallm/configuration/configs.py
  - dnallm/inference/benchmark.py
  - dnallm/mcp/model_manager.py
  - dnallm/mcp/server.py
  - dnallm/utils/transformers_compat.py
  - example/notebooks/finetune_NER_task/generate_bpe_dataset.py
  - pyproject.toml
  - scripts/check_docs_sync.py
  - scripts/feasibility/spike_families.py
  - tests/examples/_execution.py
  - tests/examples/test_marimo_execution.py
  - tests/examples/test_notebook_execution.py
  - tests/examples/test_script_execution.py
  - tests/expected_skips.yaml
  - tests/mcp/test_interpret_tool.py
  - tests/mcp/test_model_manager.py
  - tests/models/test_model_remote_code.py
  - tests/utils/test_transformers_compat.py
covered_digest: "v2:sha256:c44ff0e2ce54e0b5c81595168a925450fc1be740340b1bca240c1bc03c4096ed"
behavior_unverified: 0
overrides_applied: 1
overrides:
  - must_have: "tests/examples/conftest.py provides the locally-scoped notebook_sandbox fixture (05-01 artifact)"
    reason: "Post-wave integration fix: tests/ is not a package, so a bare tests/examples/conftest.py won the conftest module-name race and broke 'from conftest import ...' in test_trainer/test_benchmark/test_dna_dataset. The fixture was relocated module-locally to tests/examples/test_notebook_execution.py with an explanatory comment; in-phase consumption and tree-clean teardown are delivered identically. Recreating the conftest would re-break three test files."
    accepted_by: "owner (Tao Zhang) — 05-UAT item 3, acknowledged 2026-10-02 at closure"
    accepted_at: "2026-10-02T04:50:00+08:00"
re_verification:
  previous_status: human_needed
  previous_score: 25/25
  gaps_closed:
    - "No gaps existed (prior round was 25/25 with 2 human items). This stale-digest re-run (#4682) regenerates the report against HEAD 80b40a5 after five owner-sanctioned quick tasks changed covered source files post-verification (se3 fdc4915, sl7 98c39f8..6073f72, gsd-fast d352c0e, 0p0 7b0d360..a0220d5, csd bd5496a..9453d23, plus the owner's notebook-output commit 94f0651)"
    - "Prior human item 1 (NT smoke RED on this box at the get_extended_attention_mask rung) RESOLVED by code: the se3+sl7 transformers-5.x shims closed the environment gap — this verifier re-ran the smoke fresh: 1 passed in 6.96s, a REAL load+forward with (1,2) logits through the previously-patched snapshot"
    - "Prior human item 2 (WR-04 rice-URL network lane never executed) RESOLVED by execution: this verifier ran the script lane fresh: 1 passed in 51.08s — real rice download (4xx re-raise branch not fired), generate_bpe_dataset.py executed end-to-end producing rice_gene_ner_BPE.pkl in-sandbox"
  gaps_remaining: []
  regressions: []
deferred:
  - truth: "05-FEASIBILITY.md Runner-confirmation column is filled from an actual dispatch run of feasibility.yml on the self-hosted GB10 runner (verdicts become official per D-04)"
    addressed_in: "post-merge integration window (phs → dev → main) — first dispatch opportunity after feasibility.yml lands on the default branch"
    evidence: "Platform constraint unchanged: dispatch API needs feasibility.yml on the default branch and the phs range is still unpushed (git log origin/dev..phs = 109 commits at this verification, manual-push-only rule honored). 05-UAT item 2 records the same deferral."
  - truth: "Census FAIL repair queue (12 rows) + 2 deferred-owner rows + Phase 7-9 rescoping"
    addressed_in: "Phase 8 (owner decisions, flagged in 05-CENSUS.md Hand-off §1-§5)"
    evidence: "Designed deliverable (an honest worklist) per D-09. Post-closure progress note: owner-sanctioned quick tasks have since moved 6 of the 12 FAIL rows to real green (sl7) and executed both deferred-owner mcp notebooks green in the gated lane (csd, official 2 passed in 365s) — tracked in the quick-task summaries and .planning/WINDOWS.md, not by mutating the campaign-time census record."
  - truth: "MCP server deferred bugs: --host/--port CLI flags silently overridden by yaml config (server.py:1696-1700); dna_interpret runs captum inline on the event loop (WR-01)"
    addressed_in: "owner-scope server changes (deferred-items.md, open)"
    evidence: "Recorded in deferred-items.md with evidence (observed 0.0.0.0 bind despite --host 127.0.0.1; 172s dna_interpret completing past the 30s wrapper cap); fix shapes documented there."
advisory:
  - finding: "CR-01 (05-REVIEW.md, open): ModelManager._infer_lock (asyncio.Lock held across run_in_executor) releases on tool-timeout cancellation while the orphaned infer_seqs thread keeps running, so a subsequent call can start a concurrent infer_seqs — the exact DataLoader-fork hazard single-flight was added to prevent"
    category: architectural
    reason: "Quick-task (261003-csd) code, recorded and triaged open in 05-REVIEW-DISPOSITION.md; not a phase-05 must-have. No deterministic failing test exists (fast leg 1716 green; tests/mcp/test_model_manager.py green). Dependence assessment: nothing verified in this phase rides the affected path — the gated mcp notebook tests drive sequential client requests and currently skip honestly (server down); no phase truth asserts concurrent-serving correctness. Would be resolved by re-acquiring or leak-guarding the lock against cancellation."
    evidence_status: "none provided (code-reading finding; recorded in the disposition ledger)"
  - finding: "WR-01 (05-REVIEW.md, open): dna_interpret runs blocking captum work on the event-loop thread — the 30s _with_timeout_wrapper cannot fire while blocked (observed 172s completion)"
    category: architectural
    reason: "Quick-task code, recorded open in the disposition ledger and deferred-items.md; mamba models are guarded (fa19675) but non-mamba interpretations still block. Dependence assessment: no phase-05 truth depends on interpret concurrency; the phase's mcp durable tests skip honestly with the server down. Fix shape (run_in_executor, ModelManager pattern) documented in deferred-items.md."
    evidence_status: "none provided (observed once in a bring-up log; recorded in the ledger)"
  - finding: "IN-01 (05-REVIEW.md, open, info): three of the new transformers_compat patch installers rely on the module's outer import context instead of per-installer try/except guards"
    category: other
    reason: "Info-level stylistic deviation from the module's documented guard pattern; transformers is a hard dependency of dnallm so absence is not a runtime scenario the phase contract covers. Verified this round: import dnallm succeeds cleanly and 9 import-guard constructs remain in the module; the D-07 installer keeps its documented guard/sentinel contract."
    evidence_status: "none provided (ledger-recorded info finding)"
---

# Phase 5: Execution Harness, Honest Gates & Runner Feasibility Verification Report

**Phase Goal:** A trustworthy private execution harness exists and is proven (including kernel-kill on hang); both false-green CI gates are closed together with the docs-mirror drift they were hiding; and the runner's real capabilities for the environment-gated model families are settled in writing before execution tests are written against them — PLUS the reopened fifth success criterion (D-07/D-08/D-09, 2026-10-02 post-closure gap closure)
**Verified:** 2026-10-03T04:03:05Z
**Status:** passed (25/25 truths verified; 0 behavior-unverified; 0 human-verification items — both prior human items resolved by fresh verifier execution this round)
**Re-verification:** Yes — stale-digest regeneration (#4682). Prior report (2026-10-02T12:05:00Z, human_needed, 25/25) went stale because five owner-sanctioned quick tasks changed covered source files after it was written. This report re-collects ALL evidence against HEAD 80b40a5; no SUMMARY PASS claim was used as evidence for any gate. The prior report's full text is preserved in git history (last state at e646dd3).

## Re-Verification Scope

Full-scope regeneration: every truth's evidence was re-collected live against the current tree because the quick-task delta touched the core covered surfaces — `dnallm/utils/transformers_compat.py` (grew from the D-07 dual-module pruning shim to 9 absence-gated patches), `tests/examples/_execution.py` + `test_notebook_execution.py` (ACTIVE 8→13, GATED 7→8, sibling-input seeding), `dnallm/mcp/{model_manager,server}.py`, `dnallm/configuration/configs.py`, `dnallm/inference/benchmark.py`, and the example/docs trees (owner committed the previously-dirty notebook outputs in 94f0651, changing the docs-sync baseline).

Standing context honored: manual-push-only milestone (109 unpushed phs commits expected); the MCP server is intentionally DOWN (owner killed it — gated mcp tests skip honestly with live probe evidence); the 9 live ipykernel processes on the box all predate this verification session (owner's Jupyter).

## Goal Achievement

### Observable Truths

Truths 1-19 are the original-phase set; 20-25 the reopened fifth criterion (D-07/D-08/D-09). All evidence below was collected fresh this round unless marked "carried" (document/file unchanged since the prior pass, proven by empty git log).

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Pilot notebook executes end-to-end through run_notebook() with kernel cwd inside a tmp_path sandbox copy (SC1/EXEC-01) | ✓ VERIFIED | **Re-run this round**: `test_notebook_executes_end_to_end[notebooks/inference/inference.ipynb]` — 1 passed in 11.58s |
| 2 | Scoped tree clean after execution (SC1/EXEC-01) | ✓ VERIFIED | `git status --porcelain -- example docs/example` empty after EVERY verifier run this round — stronger than prior rounds: the owner committed the notebook outputs (94f0651), so the tracked baseline is now fully clean |
| 3 | On cell error/timeout the harness writes partial executed notebook + exception text before re-raising (SC1/EXEC-01) | ✓ VERIFIED | **Re-run this round**: `TestPartialFailureArtifacts` — passed (2 passed in 4.84s with the kill test) |
| 4 | Timeout layering: per-cell strictly below per-test mark (SC1/EXEC-01) | ✓ VERIFIED | Spec cell_timeout range 600–3600, all strictly below the outer marks (ACTIVE class 7200; GATED class 3600 with the WR-02 `**`7200 overrides for the three 3600-cell entries — frozenset re-read in code). WR-03 fix verified: dead `test_timeout` spec fields removed; layering now lives solely in the class/param marks |
| 5 | Kernel shutdown guaranteed via shutdown_kernel="immediate" + plain client.execute() (SC1/EXEC-01) | ✓ VERIFIED | `client.execute()` count 1, `with NotebookClient` count 0 in `_execution.py`; zero kernel debris from this verifier's runs (all 9 live ipykernel processes started Oct 2 10:28–Oct 3 00:31, before this session — owner's live Jupyter) |
| 6 | Harness private to its test tree (EXEC-01) | ✓ VERIFIED | Underscore module; notebook_sandbox fixture module-local with the anti-conftest rationale comment; tests/examples/conftest.py still absent (override honored); no root-conftest/dnallm importer |
| 7 | Kill test: hung kernel killed, no surviving ipykernel_launcher (SC2/EXEC-06) | ✓ VERIFIED | **Re-run this round (fresh)**: `TestKernelLifecycle` — passed (4.84s combined) |
| 8 | check_docs_sync.py honest, wrapper-.md relaxation scoped to right_only (SC3/REPAIR-02) | ✓ VERIFIED | Script unchanged since the in-phase WR-02 fix (9876a0b). Honesty demonstrated LIVE this round: it exits 1 correctly detecting the owner's fresh untracked runtime artifacts (benchmark_results/, plot_metrics.pdf, plot_roc.pdf from the owner's Oct 3 00:18 local benchmark re-run) — the gate fails on real drift exactly as designed; zero DIFFER lines on tracked content |
| 9 | Mirror byte-identical after resync (SC3/REPAIR-02/D-03) | ✓ VERIFIED | Tracked mirror fully in sync: scoped porcelain empty, no DIFFER lines — the 4 prior owner-baseline DIFFER notebooks were committed with mirrors in 94f0651. The 3 current left_only hits are untracked, gitignored-class runtime outputs (pdf matched by .gitignore:118), not committed drift |
| 10 | REPAIR-02 edges: fail-closed absent dirs, single-prefix reporting (SC3/REPAIR-02) | ✓ VERIFIED | Carried (script byte-unchanged since the in-phase fix; behavior proven at closure and the prior pass) |
| 11 | docs-validation honest: zero continue-on-error, born green (SC3/CI-01/D-01) | ✓ VERIFIED | `grep -c continue-on-error` = 0; job `docs-validation` present (lines 13-14) |
| 12 | docs-validation installs .[test,dev,mcp]; README documents proven install line (SC3/CI-02) | ✓ VERIFIED | Workflow line 42 + README.md:497 re-grepped; validators re-run green this round (snippets: 142 files/328 blocks OK; yaml: 21 files OK) |
| 13 | Branch protection on dev and main lists BOTH required contexts (CI-01/D-02) | ✓ VERIFIED | **Live read-only re-verification by this verifier (2026-10-03T04:0xZ)**: `gh api` on BOTH branches returns `coverage-gate (py3.12, fast leg)` + `docs-validation` |
| 14 | Written verdict matrix, every row carries measured evidence (SC4/FEAS-01/D-05) | ✓ VERIFIED | 05-FEASIBILITY.md untouched since the prior pass (empty git log 821e184..HEAD); all 8 spike logs re-listed on disk |
| 15 | Verdicts against exact notebook variants via real forward; pyBigWig real round-trip (D-05) | ✓ VERIFIED | Carried (matrix + logs unchanged; census gated-ladder legs re-proven at campaign time) |
| 16 | Every non-FEASIBLE verdict shows both attempts with recorded failure text (D-06) | ✓ VERIFIED | Carried (unchanged documents) |
| 17 | Spike ran in throwaway venv; project env untouched (D-04) | ✓ VERIFIED | **Re-proven this round**: stripedhyena/evo2/MEGABYTE_pytorch/pyBigWig/langchain_ollama/mamba_ssm/megaDNA ALL absent from .venv; pyproject porcelain-empty |
| 18 | Dispatch-gated runner confirmation job exists + documented + hand-off (D-04) | ✓ VERIFIED | feasibility.yml re-grepped: workflow_dispatch-only, runs-on [self-hosted, dnallm-nightly], timeout 240, `if: always()` upload, permissions block (WR-06); official dispatch stays post-merge (deferred item 1) |
| 19 | marimo flavor decided with evidence; pyBigWig not added to pyproject (FEAS-01) | ✓ VERIFIED | pyBigWig count in pyproject = 0; export-html flavor re-proven live this round: inference_demo 1 passed in 9.81s |
| 20 | D-07 shim: vendored v4.49.0 pruning helpers, absence-gated attach (4.x no-op), wired into apply_patches() | ✓ VERIFIED | Full re-read: `_find_pruneable_heads_and_indices`/`_prune_linear_layer`/`_attach_remote_code_pruning_helpers`/`_patch_remote_code_pruning_helpers` wired in apply_patches() (now 9 patches, all absence-gated). **Live probe on transformers 5.17.0 by this verifier**: modeling_utils exposes BOTH helpers as vendored identity; pytorch_utils gained the vendored find_pruneable while its NATIVE prune_linear_layer is untouched (never-overwrite); sentinel set; idempotent under repeat apply_patches(); arithmetic spot-check ({1} / rows [0,1,4,5,6,7]). Contract tests re-run: **76 passed in 3.51s** (26 + 6 WR-01 + 10 se3 + 34 sl7). Fresh `import dnallm` clean |
| 21 | D-07 smoke: real-model load+forward — **STATUS UPGRADED from the ladder-terminal typed skip to a REAL GREEN PASS** | ✓ VERIFIED | **Re-run this round: 1 passed in 6.96s** — the se3 `get_extended_attention_mask` shim (fdc4915, attaches to PreTrainedModel per its docstring) plus the sl7 legacy-config-defaults shim (`is_decoder`/`add_cross_attention`, transformers_compat.py:508-542) closed the environment gap the prior round routed to human item 1. The smoke now executes the real load+forward asserting (1,2) logits through the previously-patched NT snapshot. The owner's superseding decision is recorded (.continue-here: "NER is_decoder shim supersedes 05-04 D-07 record") |
| 22 | D-08 census: every Table A item carries a verdict, nothing pending, nothing silently omitted | ✓ VERIFIED | **All gates re-run**: `grep -c '\| pending'` = 0; `pending-owner` string = 0; Table A ipynb rows 21 == live 21; marimo rows 3 == live 3; script rows 1; verdict tally 11 PASS / 12 FAIL / 2 deferred-owner unchanged; census doc untouched since the prior pass. ℹ️ Note: Table C's tracked-file arithmetic (57) predates the sl7 sibling-CSV commit (now 60 tracked); the executable census (Table A, the D-08 bar) remains exact — see Anti-Patterns |
| 23 | D-08 evidence: every verdict traces to recorded evidence; ladder visible in gated rows | ✓ VERIFIED | manifest.json re-read on disk: 28 item outcomes (14 pass / 12 fail / 2 pending-owner), zero None; 26 per-item logs present under .scratch/census-out/logs/; census untouched |
| 24 | D-08 durable wiring: green set active, gated set probe-then-skip, suite green-or-typed-skipped | ✓ VERIFIED | ACTIVE_NOTEBOOKS = **13** (pilot + 12 census-green; grown 8→13 by the sanctioned sl7/csd follow-through, all spec keys exist on disk); GATED_NOTEBOOKS = 8 with probe gates (mcp pair → _gate_ollama_stack, evo, megadna×3, mamba×2). **Re-run this round**: TestGatedNotebookExecution — 8 skipped in 0.79s, EVERY message carrying live probe results (ollama GREEN + MCP endpoint refused; find_spec None); audit_skips.py over a fresh junit — 8/8 allowed, exit 0; all three prefixes registered. **Fast leg (full, once)**: 1716 passed, 1 pre-existing skip, 55 deselected, 90.03s |
| 25 | D-09: plans 05-01..03 untouched; gap plans carry honest REQ claims | ✓ VERIFIED | `git log 8fa9e05..HEAD -- 05-0{1,2,3}-PLAN.md` empty; all six plans' summary commits re-confirmed present on phs (ec19794/4decbe4/f7b5fa9/ebdf482/7769f35/330134a/bf504e1 spot-checked OK); `git ls-files .scratch/` = 0; REQUIREMENTS.md states re-cross-checked: EXEC-01/06/REPAIR-02/CI-01/CI-02/FEAS-01 Complete, EXEC-03 Complete (dev-box basis), EXEC-04 + REPAIR-03 open/Pending-Phase-8 — matching every plan claim |

**Score:** 25/25 truths verified (0 present-behavior-unverified; 0 items routed to human verification — the prior round's two human items are both resolved with fresh verifier-executed evidence: the NT smoke now passes REAL (truth 21) and the WR-04 rice network lane was executed for real by this verifier (script lane 1 passed in 51.08s, real download, pkl produced in-sandbox — the 4xx re-raise branch did not fire))

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Matrix Runner-confirmation column filled from an actual dispatch run of feasibility.yml (D-04 official-verdict step) | Post-merge integration (phs → dev → main) | Platform constraint unchanged (dispatch needs the workflow on the default branch); 109-commit phs range still unpushed (manual-push-only honored). 05-UAT item 2 records the same deferral |
| 2 | Census FAIL repair queue (12 rows) + 2 deferred-owner rows + Phase 7-9 rescoping | Phase 8 (owner decisions, 05-CENSUS.md Hand-off §1-§5) | Designed deliverable per D-09. Progress note: 6 FAIL rows since moved to real green (sl7) and both mcp notebooks executed green in the gated lane (csd) — recorded in quick-task summaries/WINDOWS ledger, census left as the honest campaign-time record |
| 3 | MCP server deferred bugs (--host/--port CLI flags dead; dna_interpret blocks the event loop) | Owner-scope server changes (deferred-items.md, open) | Evidence and fix shapes recorded in deferred-items.md (0.0.0.0 bind despite --host 127.0.0.1; 172s interpret past the 30s cap) |

### Advisory (New Scope, Unevidenced)

Findings from the incremental code review of the quick-task delta (05-REVIEW.md @ a943411; disposition @ 80b40a5) — recorded and triaged open in the project's own ledger, not phase-plan must-haves, no deterministic failing test, and (per this verifier's dependence assessment) nothing verified above rides the affected code paths. Reported here for visibility; they do not block and do not revert any completed must-have.

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| 1 | CR-01: single-flight `_infer_lock` releases on tool-timeout cancellation while the orphaned executor thread runs — concurrent infer_seqs still possible (dnallm/mcp/model_manager.py:249,280) | architectural | Quick-task code, ledger-recorded open, no failing test; gated mcp tests are sequential clients and currently skip honestly (server down) — no phase truth asserts concurrent-serving correctness |
| 2 | WR-01: dna_interpret runs captum inline on the event loop; the 30s timeout wrapper cannot fire while blocked (dnallm/mcp/server.py:1440; mamba guarded at :1531) | architectural | Quick-task code, ledger-recorded open + deferred-items.md entry; no phase truth depends on interpret concurrency |
| 3 | IN-01: three new patch installers rely on the module's outer import context rather than per-installer try/except guards (transformers_compat.py) | other | Info-level stylistic deviation; `import dnallm` verified clean this round; the D-07 installer keeps its documented guard/sentinel contract |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `dnallm/utils/transformers_compat.py` | D-07 vendored helpers + gated dual-module attach wired into apply_patches() | ✓ VERIFIED | Re-read in full; now 9 absence-gated patches (D-07 + se3 + sl7 set), all wired at apply_patches():1015-1025; live-probed on 5.17.0 |
| `tests/models/test_model_remote_code.py` | slow real-model load+forward smoke | ✓ VERIFIED | **Now passes REAL** (1 passed 6.96s) — the designed self-healing typed skip healed into the real green smoke when the quick-task shims closed the environment gap |
| `tests/utils/test_transformers_compat.py` | behavior-contract tests for the vendored helpers | ✓ VERIFIED | 76 passed this round (26 + 6 + 10 se3 + 34 sl7) |
| `tests/examples/_execution.py` | notebook/marimo/script lanes, specs, typed-skip helpers | ✓ VERIFIED | NOTEBOOK_EXEC_SPECS 21 / MARIMO_EXEC_SPECS 3, all keys on disk; `str(EXAMPLE_DIR` count 24; execute()-only client contract; sibling-input seeding table (sl7) |
| `tests/examples/test_notebook_execution.py` | ACTIVE + GATED + kill + partial-failure | ✓ VERIFIED | Read + re-run: ACTIVE 13, GATED 8, kill+partial 2 passed 4.84s, gated 8 evidence-bearing skips, pilot 11.58s |
| `tests/examples/test_marimo_execution.py` | parametrized over census-green apps | ✓ VERIFIED | 3 specs; inference_demo re-run green 9.81s |
| `tests/examples/test_script_execution.py` | script lane with WR-04 4xx re-raise routing | ✓ VERIFIED | **Executed for real this round**: 1 passed 51.08s (rice download + NT model load + pkl produced) — resolves the prior round's never-executed-network-lane caveat |
| `05-CENSUS.md` | full-tree census, all verdicts filled, hand-off section | ✓ VERIFIED | All completeness gates re-run green (truth 22-23); doc untouched since prior pass |
| `05-FEASIBILITY.md` + spike-logs/ | verdict matrix + 8 evidence logs | ✓ VERIFIED | Untouched; 8 logs on disk |
| `.gitignore` | `.scratch/` entry | ✓ VERIFIED | `git ls-files .scratch/` = 0 |
| `dnallm/mcp/model_manager.py`, `dnallm/mcp/server.py` | (quick-task covered surface) single-flight inference + mamba interpret guard | ✓ VERIFIED (surface) | Code read; csd tests green in fast leg; CR-01/WR-01 residual holes advisory (ledger-recorded) |
| Prior-closure artifacts (docs-validation.yml, README, feasibility.yml, spike_families.py, expected_skips.yaml, check_docs_sync.py, pyproject, configs.py, benchmark.py) | unchanged-honest | ✓ VERIFIED | All greps/probes green (truths 8-19, 24); pyproject changed only by the 0p0 ty-excludes quick task (porcelain-clean, no spike deps) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| dnallm/utils/__init__.py | transformers_compat.apply_patches() | eager import chain | ✓ WIRED | Live-probed: attachment active from fresh `import dnallm`; apply_patches() at module import (line 1030) |
| _patch_remote_code_pruning_helpers | transformers.modeling_utils + pytorch_utils | per-name setattr | ✓ WIRED | Identity-verified against the vendored functions this round; native pytorch_utils.prune_linear_layer preserved |
| _patch_get_extended_attention_mask (se3) | transformers.PreTrainedModel | class attach | ✓ WIRED | Proven live by the NT smoke's real forward pass (6.96s) |
| tests/models/test_model_remote_code.py | load_model_and_tokenizer → ModelScope NT snapshot | real load+forward | ✓ WIRED | Exercised green by this verifier (weights load + (1,2) logits) |
| TestGatedNotebookExecution | registered prefixes in expected_skips.yaml | typed-skip helpers | ✓ WIRED | 8 skips this round, all matched by audit_skips.py (exit 0); live probe results in every message |
| census Table A rows | .scratch/census-out/ manifest + logs | evidence paths | ✓ WIRED | 26/26 logs present; manifest 28 outcomes; re-read this round |
| 05-04/quick-task shims | benchmark/NER/script NT items | one root cause, now CLOSED in code | ✓ WIRED | All three formerly NT-REMOTE-STRUCTURAL items now execute: benchmark notebook ACTIVE-green (owner run committed), NER notebook ACTIVE-green (sl7), script lane passed 51.08s (this verifier) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| NT smoke (D-07 live probe) | `pytest tests/models/test_model_remote_code.py -q` | **1 passed in 6.96s** (real load+forward, (1,2) logits) | ✓ PASS (prior-round RED resolved) |
| Compat contract tests | `pytest tests/utils/test_transformers_compat.py -q` | 76 passed in 3.51s | ✓ PASS |
| Pilot notebook (ACTIVE lane) | `pytest ...::test_notebook_executes_end_to_end[notebooks/inference/inference.ipynb]` | 1 passed in 11.58s | ✓ PASS |
| Kernel-kill + partial-failure | `pytest ...::TestKernelLifecycle ...::TestPartialFailureArtifacts -q` | 2 passed in 4.84s | ✓ PASS |
| Gated notebook probes | `pytest ...::TestGatedNotebookExecution -q -rs` | 8 skipped in 0.79s, every message carrying live probe evidence | ✓ PASS |
| Skip audit | `audit_skips.py` over fresh gated-lane junit | 8/8 allowed, exit 0 | ✓ PASS |
| Marimo export-html lane | `pytest ...::TestMarimoAppExecution -k inference_demo` | 1 passed in 9.81s | ✓ PASS |
| Script lane (prior human item 2) | `pytest tests/examples/test_script_execution.py -q` | **1 passed in 51.08s** (real rice download + pkl produced in-sandbox) | ✓ PASS (prior-round never-executed lane now executed) |
| Fast leg (full suite, once) | `pytest -m "not slow" -q` | 1716 passed, 1 pre-existing skip, 55 deselected, 90.03s | ✓ PASS |
| Shim attach (both modules, identity, sentinel, idempotence, native preserved) | fresh-process python probe | all True on transformers 5.17.0 | ✓ PASS |
| Census completeness gates | Table C command block | 0 pending; 21/3/1 exact; tally 11/12/2; manifest 28/0-None; 26 logs | ✓ PASS |
| Docs validators | validate_docs_snippets.py + validate_yaml.py | 328 blocks OK / 21 files OK, both exit 0 | ✓ PASS |
| Environment isolation | find_spec probes + pyproject porcelain | zero spike-only/forbidden packages; pyproject clean | ✓ PASS |
| Branch protection (live, read-only) | `gh api .../branches/{dev,main}/protection` | both branches: both required contexts | ✓ PASS |
| Docs mirror (tracked content) | scoped `git status` + sync-script DIFFER scan | tracked mirror fully clean; 3 left_only hits are untracked gitignored runtime artifacts | ✓ PASS (tracked) / ℹ️ runtime-artifact note |

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes are declared by this phase; the plans' verify legs are pytest/script/awk/gh commands, all re-executed directly above by this verifier (SUMMARY PASS claims were not used as evidence for any gate).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| EXEC-01 | 05-01, 05-04, 05-05, 05-06 | Private execution harness (nbclient-as-library, timeout layering, sandbox isolation, kernel shutdown, partial artifacts) | ✓ SATISFIED | Truths 1-7, 20-24; pilot/kill/partial/gated/marimo/script all re-run green this round |
| EXEC-06 | 05-01 | Deliberate-hang kill test proves kernel cleanup | ✓ SATISFIED | Truth 7 — re-run green this round |
| REPAIR-02 | 05-02 | Docs mirror closed + regenerated per repair | ✓ SATISFIED | Truths 8-10; tracked mirror fully clean (owner committed the baseline outputs with mirrors); runtime-artifact walker note below |
| CI-01 | 05-02 | Masking removed in same unit as drift closure; docs-validation required check | ✓ SATISFIED | Truths 11 + 13 (live re-read this round) |
| CI-02 | 05-02 | mcp extra installed; README install line corrected | ✓ SATISFIED | Truth 12 |
| FEAS-01 | 05-03 | Written verdict matrix, real variants, evidence-backed typed skips | ✓ SATISFIED | Truths 14-19 |
| EXEC-03 (dev-box leg, claimed complete) | 05-05, 05-06 | All 3 marimo apps execute headlessly | ✓ SATISFIED (dev-box basis) | 3/3 specs; inference_demo re-run green this round; census rows unchanged; nightly-runner leg arrives with Phase 8 SC1 |
| EXEC-04 (dev-box leg, claimed open) | 05-05, 05-06 | generate_bpe_dataset.py produces artifact in-sandbox | ◐→✓ dev-box leg now REAL (REQ stays open for Phase 8 nightly leg) | **This verifier executed the lane green (51.08s, pkl produced)** — the NT environment gap that made it a typed skip at campaign time is closed in code; REQUIREMENTS.md checkbox correctly still Pending (Phase-8 scope owns the nightly leg) |
| REPAIR-03 (partial claim) | 05-04 | dnallm bugs exposed by execution fixed with regression tests | ◐ PARTIAL → substantially advanced (honest, deliberate) | The shim family now spans 9 patches with 76 contract tests and the NT slice is fully green (smoke + script + benchmark/NER notebooks); remaining census FAIL rows stay in the Phase-8 repair queue exactly as claimed; REQUIREMENTS.md correctly unchecked |

Orphaned requirements: none — REQUIREMENTS.md maps exactly EXEC-01/EXEC-06/REPAIR-02/CI-01/CI-02/FEAS-01 to Phase 5 (all Complete), matching the plans' `requirements` fields; the gap plans' additional claims (EXEC-03/EXEC-04/REPAIR-03) match the REQUIREMENTS.md states (complete-dev-box / pending / pending).

### Decision Coverage

Phase 05-CONTEXT.md decisions D-01 through D-09 were previously verified as honored and none of the post-verification changes touch their artifacts (feasibility/census/workflow files unchanged since the prior pass; the D-07 superseding decision is owner-recorded in .continue-here). No decision vanished during execution.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| dnallm/utils/transformers_compat.py | 668 | `# TODO (joao):` inside vendored `_MambaCache` | ℹ️ Info | Verbatim upstream transformers comment preserved by the vendoring contract ("keep its docstrings and semantics", 05-04 pattern) — upstream debt, not project debt; not a blocker |
| (all other changed code files) | - | Zero TBD/FIXME/XXX matches across the 12 scanned changed files | - | - |

ℹ️ Info — census Table C arithmetic drift: the tracked-example-file count recorded at campaign time (57) is now 60 because the sl7 quick task committed 3 sibling CSV inputs under example/notebooks/data_prepare/finetune/ after the census was frozen. The D-08 acceptance surface (Table A executables: 21/3/1) remains exact against the live tree; the census is a campaign-time record and the CSV addition is owner-sanctioned follow-through recorded in the sl7 summary. No action required for phase closure; Phase 8's census refresh will re-baseline Table B/C.

ℹ️ Info — docs-sync walker vs untracked runtime artifacts: check_docs_sync.py currently exits 1 on 3 untracked items (benchmark_results/, plot_metrics.pdf, plot_roc.pdf) generated by the owner's Oct 3 local benchmark re-run. The PDFs are gitignored (.gitignore:118) but the script's own IGNORE set predates this artifact class. The tracked mirror is fully clean; if the owner wants exit 0 on an idle tree, extending the script's IGNORE (a one-line owner change, out of phase scope) would cover it. Same standing-context disposition as the prior round's owner-baseline DIFFERs.

Re-verification evidence gate (#3304): the census's evidence living in gitignored `.scratch/` remains BY DESIGN (owner standing rule) — committed rows carry terminal one-liners and this verifier re-confirmed the referenced manifest (28 outcomes) and all 26 logs exist on disk.

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| tests/examples/test_notebook_execution.py | EXEC-01/06 | ACTIVE 13 + gated 8 + kill/partial + fast harness classes | gated skip only with probe evidence | none | Behavioral (execution outcomes, artifact presence, tree-clean, kernel delta-zero) | OK |
| tests/utils/test_transformers_compat.py | EXEC-01/REPAIR-03 slice | 76 | 0 | none | Value/behavioral (identity, shapes, arithmetic, idempotence) | OK |
| tests/models/test_model_remote_code.py | EXEC-01/REPAIR-03 slice | 1 (real load+forward) | self-healing skip now moot | none | Value ((1,2) logits, num_labels==2) | OK |
| tests/examples/test_script_execution.py | EXEC-04 dev-box leg | 1 (real run) | network-typed only | none | Behavioral (fresh pkl mtime+size, returncode, tree-clean) | OK |

No disabled tests proving requirements; no circular generation; no existence-only assertions on value-level requirements.

### Human Verification Required

None. The prior round's two items are both closed by fresh verifier-executed evidence this round:

1. **NT snapshot disposition (was: smoke RED on this box)** — RESOLVED in code: the se3/sl7 shims (get_extended_attention_mask on PreTrainedModel + legacy is_decoder/add_cross_attention config defaults) let the previously-patched snapshot load and forward. Smoke re-run: 1 passed in 6.96s. No owner disposition among the old options is needed anymore.
2. **WR-04 rice-URL network lane (was: never executed)** — RESOLVED by execution: this verifier ran the lane against the real network: 1 passed in 51.08s (download OK; the 4xx re-raise branch correctly did not fire; artifact produced in-sandbox).

The infrastructure/foundation scoping rule applies: no user-facing elements require manual UAT; every acceptance criterion was verified programmatically.

### Gaps Summary

No failed truths, no missing/stub artifacts, no unwired links, no unreferenced debt markers, no human-verification items. The phase goal holds on the current tree (HEAD 80b40a5) — and materially stronger than at the prior pass: the quick-task follow-through closed the NT remote-code environment gap in code (9 absence-gated compat patches, 76 contract tests), which flipped the D-07 smoke from its designed terminal typed skip to a real green load+forward, made the script lane execute for real (this verifier: 51.08s, pkl produced), and grew the durable ACTIVE lane 8→13 with the suite fully green (fast leg 1716/1 in 90.03s; gated lane 8 evidence-bearing, audit-clean skips). Both false-green CI gates remain closed and live-verified (branch protection on dev+main lists both contexts; zero masking flags; validators green; tracked docs mirror fully in sync). The census remains the honest campaign-time record with every verdict evidence-traceable on disk. The D-04 runner-confirmation dispatch stays a documented post-merge deferral (109-commit phs range unpushed, manual-push-only honored). The open code-review findings (CR-01/WR-01 + infos) are quick-task scope, recorded and triaged open in the project's disposition ledger, and nothing verified here depends on their affected paths — surfaced as advisories, not gaps. Status: **passed**.

---

_Verified: 2026-10-03T04:03:05Z_
_Verifier: Claude (gsd-verifier)_
_Stale-digest regeneration (#4682) of: 05-VERIFICATION.md @ e646dd3 (2026-10-02T12:05:00Z, human_needed 25/25 — preserved in git history). Delta verified: e646dd3..80b40a5 (quick tasks se3/sl7/d352c0e/0p0/csd + owner notebook-output commit 94f0651 + review docs a943411/80b40a5)_
