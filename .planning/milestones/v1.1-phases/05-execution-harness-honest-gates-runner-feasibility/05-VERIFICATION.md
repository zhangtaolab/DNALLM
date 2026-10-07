---
phase: 05-execution-harness-honest-gates-runner-feasibility
verified: 2026-10-07T00:22:22Z
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
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW-DISPOSITION.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW-FIX.md
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW.md
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
covered_digest: "v3:sha256:f93be8fdd2e56e378ad6105629234d5bb9819fc9a65043841b864287b4c8d5cf"
behavior_unverified: 0
overrides_applied: 1
overrides:
  - must_have: "tests/examples/conftest.py provides the locally-scoped notebook_sandbox fixture (05-01 artifact)"
    reason: "Post-wave integration fix: tests/ is not a package, so a bare tests/examples/conftest.py won the conftest module-name race and broke 'from conftest import ...' in test_trainer/test_benchmark/test_dna_dataset. The fixture was relocated module-locally to tests/examples/test_notebook_execution.py with an explanatory comment; in-phase consumption and tree-clean teardown are delivered identically. Recreating the conftest would re-break three test files."
    accepted_by: "owner (Tao Zhang) — 05-UAT item 3, acknowledged 2026-10-02 at closure"
    accepted_at: "2026-10-02T04:50:00+08:00"
re_verification:
  previous_status: passed
  previous_score: 25/25
  gaps_closed:
    - "No gaps existed (prior round was passed 25/25). This stale-digest regeneration re-collected evidence against HEAD 3d27360 after 39 commits (phase 06/07/08 incremental code-review fix cycles and their verification regenerations: check_docs_sync .pdf exemption narrowed to path-shaped, transformers_compat np.fromstring str-encode shim, test_notebook_execution +TestEvoNotebookContentContracts fast contracts, and the phase-08 CR-01 re-execution of generation_evo_models with honest outputs in both mirror twins)."
  gaps_remaining: []
  regressions: []
deferred:
  - truth: "05-FEASIBILITY.md Runner-confirmation column is filled from an actual dispatch run of feasibility.yml on the self-hosted GB10 runner (verdicts become official per D-04)"
    addressed_in: "post-merge integration window (phs → dev → main) — first dispatch opportunity after feasibility.yml lands on the default branch"
    evidence: "Platform constraint unchanged: dispatch needs feasibility.yml on the default branch and the phs range is still unpushed (git rev-list --count origin/dev..phs = 362 commits at this verification; manual-push-only rule honored). 05-UAT item 2 records the same deferral; the matrix column still reads 'pending (post-merge)'."
  - truth: "Census FAIL repair queue (12 rows) + 2 deferred-owner rows + Phase 7-9 rescoping"
    addressed_in: "RESOLVED — Phase 8 (completed 2026-10-05) and Phase 9 (completed 2026-10-06) processed the queue"
    evidence: "08-CENSUS-ROLLUP.md records every family closure; REQUIREMENTS.md now marks EXEC-03/EXEC-04/REPAIR-03 Complete (re-read this round). The 05 census document intentionally remains the campaign-time record (byte-unchanged since bf504e1, re-confirmed)."
  - truth: "MCP server deferred bug: --host/--port CLI flags silently overridden by yaml config"
    addressed_in: "owner-scope server changes (deferred-items.md, open)"
    evidence: "deferred-items.md status: open, with evidence (observed 0.0.0.0 bind despite --host 127.0.0.1) and documented fix shape. The sibling dna_interpret event-loop item in the same ledger is RESOLVED (261003-ij4; 23 interpret tests re-run green this round)."
advisory:
  - finding: "IN-03 (05-REVIEW-DISPOSITION.md, open, info): the langchain notebook's ensure-cell spawns a detached MCP server process that is never shut down"
    category: other
    reason: "Ledger-recorded open info finding from the campaign-era review, not a phase-05 must-have; the mcp pair currently skips honestly (server down, live probe evidence re-collected this round), so no verified truth rides the affected path. Would be resolved by a teardown cell or an atexit hook in the notebook."
    evidence_status: "none provided (ledger-recorded info finding)"
---

# Phase 5: Execution Harness, Honest Gates & Runner Feasibility Verification Report

**Phase Goal:** A trustworthy private execution harness exists and is proven (including kernel-kill on hang); both false-green CI gates are closed together with the docs-mirror drift they were hiding; and the runner's real capabilities for the environment-gated model families are settled in writing before execution tests are written against them — PLUS the reopened fifth success criterion (D-07/D-08/D-09, 2026-10-02 post-closure gap closure)
**Verified:** 2026-10-07T00:22:22Z
**Status:** passed (25/25 truths verified; 0 behavior-unverified; 0 human-verification items)
**Re-verification:** Yes — stale-digest regeneration at HEAD 3d27360. The prior report (2026-10-06T15:55:30Z, passed 25/25, digest v3:c1d62caa…) went stale solely because phases 06-08 verification-regeneration cycles legitimately repaired covered files (39 commits, delta listed below). All evidence below was re-collected live against the current tree; no SUMMARY PASS claim was used as evidence for any gate. Prior reports are preserved in git history.

## Re-Verification Scope

Full-scope regeneration at HEAD 3d27360 (39 commits past the prior verified tree 67b0692). The covered-file delta in that range, verified by `git diff 67b0692..HEAD`: `dnallm/utils/transformers_compat.py` (+str-encode rung in the `np.fromstring` shim — 08 CR-01 fix cd90e73), `scripts/check_docs_sync.py` (.pdf exemption narrowed from a global suffix to a path-shaped `notebooks/<one-dir>/` exemption — 07 IN-02 fix d93fb24), `tests/examples/_execution.py` (comment-only num_ctx narrative clarification — no behavior change), `tests/examples/test_notebook_execution.py` (+57 lines: new `TestEvoNotebookContentContracts` fast JSON-level class — 08 CR-01 regression pin), and `example/notebooks/generation_evo_models/inference.ipynb` + its docs mirror twin changed in lockstep (08 CR-01 honest re-execution 3352eb8). Every one of those surfaces was re-verified live this round; the rest of the evidence was re-collected fresh as well.

Standing context honored: manual-push-only milestone (362 unpushed phs commits); the MCP server is intentionally DOWN (mcp gated tests skip honestly with live probe evidence — re-collected fresh); the owner's live Jupyter session holds 8 long-lived kernels (~13.4h elapsed at check time, predating this verification) plus fresh owner-session kernels spawning mid-check — none attributable to this verifier's runs (its last kernel-spawning test completed ~10 minutes before the check; the kill test's delta-zero assertion passed).

## Goal Achievement

### Observable Truths

Truths 1-19 are the original-phase set; 20-25 the reopened fifth criterion (D-07/D-08/D-09). All evidence was collected fresh this round against HEAD 3d27360.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Pilot notebook executes end-to-end through run_notebook() with kernel cwd inside a tmp_path sandbox copy (SC1/EXEC-01) | ✓ VERIFIED | **Re-run this round**: `test_notebook_executes_end_to_end[notebooks/inference/inference.ipynb]` — 1 passed in 11.55s; scoped tree clean after |
| 2 | Scoped tree clean after execution (SC1/EXEC-01) | ✓ VERIFIED | `git status --porcelain -- example docs/example` printed 0 lines after EVERY verifier run this round (checked after pilot, kill/partial, marimo, script, gated legs, and the full fast leg) |
| 3 | On cell error/timeout the harness writes partial executed notebook + exception text before re-raising (SC1/EXEC-01) | ✓ VERIFIED | **Re-run this round**: `TestPartialFailureArtifacts` — passed (with the kill test: 2 passed in 5.36s) |
| 4 | Timeout layering: per-cell strictly below per-test mark (SC1/EXEC-01) | ✓ VERIFIED | Re-derived programmatically at HEAD: spec `cell_timeout` range 600–3600; ACTIVE class mark 7200 with max ACTIVE cell 3600 < 7200; GATED class mark 3600 with `_TIMEOUT_7200_GATED` (5 entries: mcp pair, custom_head, finetune_generation, lora_finetune) overriding to 7200 — every gated entry strict (3600<7200 ×5, 1800<3600 ×2, 900<3600 ×1); marimo class 7200 with walls 1200/3600; kill 120, partial 300; giants entry 1800 < 3600. Spec-side `test_timeout` duplication remains absent (WR-03 fix holds — zero grep hits) |
| 5 | Kernel shutdown guaranteed via shutdown_kernel="immediate" + plain client.execute() (SC1/EXEC-01) | ✓ VERIFIED | `client.execute()` count 1, `with NotebookClient` count 0, `shutdown_kernel="immediate"` at `_execution.py:512`; zero kernel debris from this verifier's runs — the 8 long-lived ipykernel_launcher processes at check time are ~13.4h old (owner session, predates this verification) and the kill test's delta-zero assertion passed |
| 6 | Harness private to its test tree (EXEC-01) | ✓ VERIFIED | `tests/examples/conftest.py` still absent (override honored); notebook_sandbox fixture module-local at test_notebook_execution.py:140 with the anti-conftest rationale comment; grep for `tests.examples._execution` across `dnallm/` = 0; no root-conftest importer |
| 7 | Kill test: hung kernel killed, no surviving ipykernel_launcher (SC2/EXEC-06) | ✓ VERIFIED | **Re-run this round (fresh)**: `TestKernelLifecycle` — passed (5.36s combined with partial-failure) |
| 8 | check_docs_sync.py honest, wrapper-.md relaxation scoped to right_only (SC3/REPAIR-02) | ✓ VERIFIED | **Live run: exit 0, "OK: docs/example/ is in sync with example/"**; `DOCS_ONLY_SUFFIXES` at :30 consulted only via `_is_docs_only` in the right_only loop (:66); left_only (:59)/diff_files untouched. The 07-IN-02 delta re-checked: the .pdf exemption is now path-shaped (`len(parts)==2 and parts[0]=="notebooks"`, exactly the .gitignore pattern) and applies on both sides like the suffix exemptions — no tracked file can be masked; the sync run is green over the tree including the 08-CR-01 re-executed evo notebook twins |
| 9 | Mirror byte-identical after resync (SC3/REPAIR-02/D-03) | ✓ VERIFIED | Live sync run exits 0 — zero DIFFER/left_only lines on tracked content; the 08-CR-01 evo re-execution (3352eb8) landed in BOTH mirror twins atomically (1663-line symmetric diff) and the byte-identity gate proves them equal at HEAD |
| 10 | REPAIR-02 edges: fail-closed absent dirs, single-prefix reporting (SC3/REPAIR-02) | ✓ VERIFIED | **Live edge test this round**: isolated copy of the script against absent dirs prints "ERROR: example does not exist", exit 1 (fail-closed unchanged despite the 07-IN-02 exemption re-shaping) |
| 11 | docs-validation honest: zero continue-on-error, born green (SC3/CI-01/D-01) | ✓ VERIFIED | `grep -c continue-on-error` = 0; job `docs-validation` at line 13-14; five enforcement steps present (sync :46 / snippets :51 / yaml :56 / example tests :61 / yaml load :66) |
| 12 | docs-validation installs .[test,dev,mcp]; README documents proven install line (SC3/CI-02) | ✓ VERIFIED | Workflow line 42 `uv pip install -e ".[test,dev,mcp]"` + README.md:511 same line re-grepped; **both validators re-run green this round** (snippets: 147 files/348 blocks OK; yaml: 21 files OK) |
| 13 | Branch protection on dev and main lists BOTH required contexts (CI-01/D-02) | ✓ VERIFIED | **Live read-only re-verification by this verifier (2026-10-07T00:1xZ)**: `gh api` on BOTH branches returns exactly `["coverage-gate (py3.12, fast leg)","docs-validation"]` |
| 14 | Written verdict matrix, every row carries measured evidence (SC4/FEAS-01/D-05) | ✓ VERIFIED | 05-FEASIBILITY.md unchanged since the in-phase commit (last commit 1168e0e, re-read); matrix rows for evo-1/evo2/megaDNA/pyBigWig/marimo with load_s/forward_s/peak_vram_gb/disk_gb or exact failure text; all 8 spike logs re-listed on disk |
| 15 | Verdicts against exact notebook variants via real forward; pyBigWig real round-trip (D-05) | ✓ VERIFIED | Carried (matrix + logs byte-unchanged since 1168e0e); the megaDNA/evo families have since been executed for real by phases 08-09, superseding the spike verdicts in the strongest direction |
| 16 | Every non-FEASIBLE verdict shows both attempts with recorded failure text (D-06) | ✓ VERIFIED | Carried (unchanged document; evo-1 row carries the 4-attempt ladder + fallback leg verbatim) |
| 17 | Spike ran in throwaway venv; project env untouched (D-04) | ✓ VERIFIED | **Re-proven this round**: stripedhyena, evo2, vortex, flash_attn, transformer_engine, evo ALL absent from .venv (importlib probe, empty list); pyproject carries the family names only in mypy `ignore_missing_imports` entries (:480-489) — no dependency declarations; pyproject porcelain-clean. (megaDNA/MEGABYTE_pytorch/pyBigWig/mamba_ssm remain importable per the sanctioned Phase-8 un-gating, none in the 05-03 prohibition list, none committed to pyproject) |
| 18 | Dispatch-gated runner confirmation job exists + documented + hand-off (D-04) | ✓ VERIFIED | feasibility.yml re-grepped: `on: workflow_dispatch` only (0 push/PR/schedule triggers), `runs-on: [self-hosted, dnallm-nightly]`, `if: github.event_name == 'workflow_dispatch'`, `timeout-minutes: 240`, `permissions:` block, `if: always()` upload; YAML parses; documented in .github/workflows/README.md:19,148; official dispatch stays post-merge (deferred item 1 — phs now 362 commits unpushed) |
| 19 | marimo flavor decided with evidence; pyBigWig not added to pyproject (FEAS-01) | ✓ VERIFIED | pyBigWig count in pyproject = 0; export-html flavor re-proven live this round: inference_demo 1 passed in 9.56s |
| 20 | D-07 shim: vendored v4.49.0 pruning helpers, absence-gated attach (4.x no-op), wired into apply_patches() | ✓ VERIFIED | Full re-read: 10 `_patch_*` functions defined, ALL 10 wired in apply_patches() (incl. `_patch_remote_code_pruning_helpers` at :344 with absence gate + `_dnallm_remote_code_pruning_patch` sentinel). **Live probe on transformers 5.17.0 by this verifier**: modeling_utils exposes BOTH helpers as the vendored identities; sentinel set; pytorch_utils NATIVE prune_linear_layer untouched; arithmetic spot-check heads {1}/index [0,1,4,5,6,7], pruned shape (2,8); idempotent under repeat apply_patches(). The 08 delta (str-encode rung in `_np_fromstring`) probed live: `np.fromstring("ACGT", dtype=uint8)` → `[65 67 71 84]`. Contract tests re-run: **88 passed in 4.08s** plus the new phase-08 file `tests/utils/test_transformers_compat_np.py` **15 passed in 3.76s** (the str-path rung's dedicated contract). Fresh `import dnallm` clean |
| 21 | D-07 smoke: real-model load+forward through the previously-patched NT snapshot | ✓ VERIFIED | **Re-run this round: 1 passed in 7.09s** — real load through `load_model_and_tokenizer` (modelscope route) + forward asserting (1,2) logits |
| 22 | D-08 census: every Table A item carries a verdict, nothing pending, nothing silently omitted | ✓ VERIFIED | **All gates re-run**: pending rows = 0; Table A = 21 notebook + 3 marimo + 1 script rows (marimo exact 3==3, script exactly 1); Table A-scoped verdict tally 11 PASS / 12 FAIL / 2 deferred-owner; manifest.json re-read: 28 items (14 pass / 12 fail / 2 pending-owner), zero None outcomes; 26 per-item logs present under .scratch/census-out/logs/. ℹ️ The live tree is 24 git-tracked notebooks (the 25th `find` hit is a transient `.ipynb_checkpoints/` copy) vs the census's frozen 21 — post-census plant_helixseek additions each carry their own durable lane (NOTEBOOK_EXEC_SPECS = 24, all keys on disk) and Phase 9's live census gate owns the live nothing-omitted duty; the 05 census is by design the frozen campaign-time record (byte-unchanged since bf504e1) |
| 23 | D-08 evidence: every verdict traces to recorded evidence; ladder visible in gated rows | ✓ VERIFIED | manifest + 26 logs re-verified on disk; gated census rows each show variant-first then fallback/prereq evidence with log paths |
| 24 | D-08 durable wiring: green set active, gated set probe-then-skip, suite green-or-typed-skipped | ✓ VERIFIED | ACTIVE_NOTEBOOKS = **13** (pilot + 12 census/repair-green; all spec keys exist on disk); GATED_NOTEBOOKS = 8 with probe gates (`_TIMEOUT_7200_GATED` ×5, `_GIANTS_GATED` = evo only). **Legs executed by this verifier at HEAD**: mcp pair → 2 typed skips in 0.85s, each message carrying live probe results (ollama GREEN HTTP 200 attempt 1/30; MCP endpoint Connection refused) — audit_skips.py over a fresh junit: 2/2 allowed, exit 0; generation_megaDNA + finetune_custom_head → **2 passed by real execution in 587.99s** (first run, no retry); lora_inference → passed in 43.13s; marimo inference_demo → passed 9.56s; script lane → **4 passed in 38.22s** (all four now real-green — the NT structural rung healed by Phase 8); generation_evo → owner-policy giants deselection (D-01, registered marker). Carried with documented code-diff proof: finetune_generation + lora_finetune (real-execution green at the prior verified tree, 1231.12s; the 39-commit delta contains zero behavioral changes to their lanes — `_execution.py` diff is comment-only and the test-file diff adds only the fast contract class; the phase-08 verifier's runs at this HEAD range corroborate). **Fast leg (full suite, once)**: 1881 passed, 1 pre-existing skip (test_examples.py:257, unchanged across all rounds), 53 deselected, 97.90s |
| 25 | D-09: plans 05-01..03 untouched; gap plans carry honest REQ claims | ✓ VERIFIED | `git log 8fa9e05..HEAD -- 05-0{1,2,3}-PLAN.md` empty; all 19 task commits re-confirmed present on phs (ec19794/252baa8/681be5e/4decbe4/7dad6a5/f7b5fa9/e74b0ce/4f78a76/7b8bf8a/ebdf482/0329d63/42f3df8/50e32ba/e8013fb/7769f35/a85bb42/311d9a6/330134a/147ea9a/bf504e1 — all `merge-base --is-ancestor` OK); `git ls-files .scratch/` = 0; REQUIREMENTS.md re-cross-checked: EXEC-01/EXEC-03/EXEC-04/EXEC-06/REPAIR-02/REPAIR-03/CI-01/CI-02/FEAS-01 ALL Complete |

**Score:** 25/25 truths verified (0 present-behavior-unverified; 0 items routed to human verification — every behavioral claim above was exercised by a test this verifier ran at HEAD, or carries an explicit code-diff proof that the lane is behavior-unchanged since its last real execution, as documented in truth 24)

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Matrix Runner-confirmation column filled from an actual dispatch run of feasibility.yml (D-04 official-verdict step) | Post-merge integration (phs → dev → main) — STILL OPEN | Platform constraint unchanged (dispatch needs the workflow on the default branch); 362-commit phs range still unpushed (manual-push-only honored); matrix column still "pending (post-merge)" |
| 2 | Census FAIL repair queue (12 rows) + 2 deferred-owner rows + Phase 7-9 rescoping | Phase 8 + Phase 9 — RESOLVED | 08-CENSUS-ROLLUP.md records every family closure; REQUIREMENTS.md marks EXEC-03/EXEC-04/REPAIR-03 Complete (re-read this round); the 05 census intentionally stays the campaign-time record |
| 3 | MCP server --host/--port CLI flags silently overridden by yaml config | Owner-scope server changes (deferred-items.md, open) | Evidence and fix shape recorded in deferred-items.md; the sibling dna_interpret event-loop item is RESOLVED (261003-ij4; 23 tests green this round) |

### Advisory (New Scope, Unevidenced)

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| 1 | IN-03: the langchain notebook's ensure-cell spawns a detached MCP server process that is never shut down | other | Ledger-recorded open info finding (05-REVIEW-DISPOSITION.md), campaign-era; the mcp pair currently skips honestly (server down, live probes re-collected) and no phase truth rides the affected path; no deterministic failing test |

No new advisories this round: the 39-commit delta's covered-file changes (pdf-exemption reshaping, str-encode shim rung, content-contract class) were each re-verified live and green, with dedicated test files from phases 07/08 (tests/scripts/test_check_docs_sync.py, tests/utils/test_transformers_compat_np.py) pinning them.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `dnallm/utils/transformers_compat.py` | D-07 vendored helpers + gated attach wired into apply_patches() | ✓ VERIFIED | Re-read; 10 absence-gated patches all wired at apply_patches(); live-probed on 5.17.0 (identity, sentinel, native preserved, idempotent, str-encode rung) |
| `tests/models/test_model_remote_code.py` | slow real-model load+forward smoke | ✓ VERIFIED | 1 passed 7.09s — real load+forward, (1,2) logits |
| `tests/utils/test_transformers_compat.py` | behavior-contract tests for the vendored helpers | ✓ VERIFIED | 88 passed in 4.08s |
| `tests/examples/_execution.py` | notebook/marimo/script lanes, specs, typed-skip helpers, probe helpers | ✓ VERIFIED | NOTEBOOK_EXEC_SPECS 24 / MARIMO_EXEC_SPECS 3, all keys on disk; execute()-only client contract; delta since prior digest comment-only |
| `tests/examples/test_notebook_execution.py` | ACTIVE + GATED + kill + partial-failure + gate contracts | ✓ VERIFIED | ACTIVE 13, GATED 8; kill+partial 2 passed 5.36s; WR-02 yaml_patch forwarding present in the ACTIVE fixture (:153-158); new phase-08 TestEvoNotebookContentContracts pins the evo-1 load cell (fast lane green) |
| `tests/examples/test_marimo_execution.py` | parametrized over census-green apps | ✓ VERIFIED | 3 specs; inference_demo re-run green 9.56s; D-18 markers asserted in-file |
| `tests/examples/test_script_execution.py` | script lane with 4xx re-raise routing + rice seeding | ✓ VERIFIED | **Executed for real this round**: 4 passed in 38.22s; tree clean |
| `05-CENSUS.md` | full-tree census, all verdicts filled, hand-off section | ✓ VERIFIED | All completeness gates green (truths 22-23); hand-off section at :179; doc untouched since bf504e1 |
| `05-FEASIBILITY.md` + spike-logs/ | verdict matrix + 8 evidence logs | ✓ VERIFIED | Untouched (1168e0e); 8 logs on disk |
| `.gitignore` | `.scratch/` entry | ✓ VERIFIED | `.scratch/` at :60; `git ls-files .scratch/` = 0; `git check-ignore` green |
| `dnallm/mcp/model_manager.py`, `dnallm/mcp/server.py` | (covered surface) single-flight inference + non-blocking interpret | ✓ VERIFIED (surface) | Thread-lock pattern in code; 28 + 23 tests green fresh |
| Prior-closure artifacts (docs-validation.yml, README, feasibility.yml, spike_families.py, expected_skips.yaml, check_docs_sync.py, pyproject, configs.py, benchmark.py) | unchanged-honest | ✓ VERIFIED | All greps/probes/validators green (truths 8-19); pyproject changed only by later-phase sanctioned declarations — none spike-only |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| dnallm/utils/__init__.py | transformers_compat.apply_patches() | eager import chain | ✓ WIRED | Live-probed: attachment active from fresh `import dnallm`; apply_patches() invoked at module import |
| _patch_remote_code_pruning_helpers | transformers.modeling_utils + pytorch_utils | per-name setattr | ✓ WIRED | Identity-verified against the vendored functions; native pytorch_utils.prune_linear_layer preserved |
| tests/models/test_model_remote_code.py | load_model_and_tokenizer → ModelScope NT snapshot | real load+forward | ✓ WIRED | Exercised green by this verifier (7.09s) |
| TestGatedNotebookExecution gates | registered prefixes in expected_skips.yaml | typed-skip helpers | ✓ WIRED | network-unavailable:/environment-unavailable:/optional-dep: all registered; mcp-pair skips matched by audit_skips.py (2/2, exit 0) |
| census Table A rows | .scratch/census-out/ manifest + logs | evidence paths | ✓ WIRED | 26/26 logs present; manifest 28 outcomes re-read |
| Gated lanes → isolated kernelspecs | .scratch evo/megadna venvs | kernel_name specs | ✓ WIRED | specs carry kernel_name for evo/megadna/mcp-langchain lanes; the gated lanes that ran this round executed green through them |

### Data-Flow Trace (Level 4)

Not applicable — this phase delivers test/CI/compat infrastructure; no user-facing rendered data. All dynamic values exercised are test outcomes verified by direct execution above.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| NT smoke (D-07 live proof) | `pytest tests/models/test_model_remote_code.py -q` | 1 passed in 7.09s (real load+forward) | ✓ PASS |
| Compat contract tests | `pytest tests/utils/test_transformers_compat.py -q` | 88 passed in 4.08s | ✓ PASS |
| np-fromstring shim contracts (08 addition) | `pytest tests/utils/test_transformers_compat_np.py -q` | 15 passed in 3.76s | ✓ PASS |
| Kernel-kill + partial-failure | `pytest ...::TestKernelLifecycle ...::TestPartialFailureArtifacts -q` | 2 passed in 5.36s | ✓ PASS |
| Pilot notebook (ACTIVE lane) | `pytest ...[notebooks/inference/inference.ipynb]` | 1 passed in 11.55s; tree clean | ✓ PASS |
| Marimo export-html lane | `pytest ...::TestMarimoAppExecution -k inference_demo` | 1 passed in 9.56s | ✓ PASS |
| Script lane (rice inputs, real model) | `pytest tests/examples/test_script_execution.py -q` | 4 passed in 38.22s; tree clean | ✓ PASS |
| Gated mcp pair (probe → typed skip) | `pytest ...::TestGatedNotebookExecution -k mcp -rs` | 2 skipped in 0.85s, live probe evidence in both messages | ✓ PASS |
| Gated megaDNA + custom_head (probe green → real execution) | targeted gated run | 2 passed by real execution in 587.99s | ✓ PASS |
| Gated lora_inference | targeted run | 1 passed in 43.13s | ✓ PASS |
| Skip audit | `audit_skips.py` over fresh mcp junit | 2/2 allowed, exit 0 | ✓ PASS |
| Fast leg (full suite, once) | `pytest tests/ -m "not slow and not giants" -q` | 1881 passed, 1 pre-existing skip, 53 deselected, 97.90s | ✓ PASS |
| Shim attach (identity, sentinel, idempotence, native preserved, str-encode rung) | fresh-process python probe | all True on transformers 5.17.0 | ✓ PASS |
| Docs sync + fail-closed edges | `check_docs_sync.py` live + absent-dir edge test | exit 0 OK line; absent-dir ERROR + exit 1 | ✓ PASS |
| Docs validators | validate_docs_snippets.py + validate_yaml.py | 348 blocks OK / 21 files OK, exit 0 | ✓ PASS |
| Branch protection (live, read-only) | `gh api .../branches/{dev,main}/protection` | both branches: both required contexts | ✓ PASS |
| MCP covered-surface tests | test_interpret_tool.py + test_model_manager.py | 23 + 28 passed | ✓ PASS |
| Census completeness gates | census command block (re-derived) | 0 pending; 21/3/1 rows; tally 11/12/2; manifest 28/0-None; 26 logs | ✓ PASS |
| Spec-key disk existence | python probe over both spec dicts | 24/24 notebooks + 3/3 marimo keys exist | ✓ PASS |
| Decision coverage gate | `check.decision-coverage-verify` | 9/9 honored, 0 not honored | ✓ PASS |

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes are declared by this phase; the plans' verify legs are pytest/script/gh commands, all re-executed directly above by this verifier.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| EXEC-01 | 05-01, 05-04, 05-05, 05-06 | Private execution harness (nbclient-as-library, timeout layering, sandbox isolation, kernel shutdown, partial artifacts, full-tree lanes) | ✓ SATISFIED | Truths 1-7, 20-24; pilot/kill/partial/gated/marimo/script all re-run green this round |
| EXEC-06 | 05-01 | Deliberate-hang kill test proves kernel cleanup | ✓ SATISFIED | Truth 7 — re-run green 5.36s |
| REPAIR-02 | 05-02 | Docs mirror closed + regenerated per repair | ✓ SATISFIED | Truths 8-10 — live exit 0, fail-closed edges re-proven, mirror byte-identical incl. the phase-08 evo re-execution twins |
| CI-01 | 05-02 | Masking removed in same unit as drift closure; docs-validation required check | ✓ SATISFIED | Truths 11 + 13 (live gh read this round) |
| CI-02 | 05-02 | mcp extra installed; README install line corrected | ✓ SATISFIED | Truth 12 |
| FEAS-01 | 05-03 | Written verdict matrix, real variants, evidence-backed typed skips | ✓ SATISFIED | Truths 14-19 |
| EXEC-03 (dev-box leg claimed then; REQ now Complete) | 05-05, 05-06 | All 3 marimo apps execute headlessly | ✓ SATISFIED | 3/3 specs; inference_demo re-run green; REQUIREMENTS.md Complete (Phase 8/9 closed the nightly leg) |
| EXEC-04 (dev-box leg claimed then; REQ now Complete) | 05-05, 05-06 | generate_bpe_dataset.py produces artifact in-sandbox | ✓ SATISFIED | 4 passed in 38.22s executed by this verifier (the formerly-typed-skip lane now fully real-green after the Phase-8 NT shim family fix); REQUIREMENTS.md Complete |
| REPAIR-03 (partial claim then; REQ now Complete) | 05-04 | dnallm bugs exposed by execution fixed with regression tests | ✓ SATISFIED | Shim family 10 patches + 88+15 contract tests + real-green NT smoke this round; Phase 8 processed the remaining queue (REQUIREMENTS.md Complete) |

Orphaned requirements: none — REQUIREMENTS.md maps exactly the plans' `requirements` claims; every ID in the PLAN frontmatter (EXEC-01/03/04/06, CI-01/02, REPAIR-02/03, FEAS-01) is accounted for above and marked Complete in REQUIREMENTS.md.

### Decision Coverage

Decision-coverage gate run this round (`check.decision-coverage-verify`): **9/9 decisions honored, 0 not honored** ("All trackable CONTEXT.md decisions are honored by shipped artifacts."). D-01..D-09 all trace to shipped artifacts; the D-07 superseding owner decision remains recorded.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| dnallm/utils/transformers_compat.py | 671 | `# TODO (joao):` inside vendored `_MambaCache` | ℹ️ Info | Verbatim upstream transformers comment preserved by the vendoring contract — upstream debt, not project debt |
| (all other covered code files) | - | Zero TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER matches across every covered implementation file scanned | - | - |

ℹ️ Info — census arithmetic drift (continued, expected): the 05 census froze a 21-notebook/57-tracked-file tree; the live tree at HEAD is 24 git-tracked notebooks / 3 marimo / 1 script (the 25th `find` hit is a transient `.ipynb_checkpoints/` notebook, untracked). The D-08 acceptance surface (Table A executables with verdicts) remains exact for its campaign scope; the live nothing-omitted duty is owned by Phase 9's nightly census plus the grown NOTEBOOK_EXEC_SPECS (24, all keys on disk). Same standing disposition as the prior rounds.

ℹ️ Info — carried gated-lane evidence: finetune_generation and lora_finetune were not re-executed by this verifier this round (their last real executions: 1231.12s green at the prior verified tree 67b0692; the phase-08 verifier ran at this HEAD range an hour prior). The 39-commit delta contains zero behavioral changes to their lanes (`tests/examples/_execution.py` diff is comment-only; the test-file diff adds only a fast contract class; the transformers_compat delta touches only the stripedhyena str path, off-lane for both). Both lanes' kernel/env seams are exercised green by the sibling lanes this verifier did run (isolated megadna-kernel and hf-mirror lanes among them).

Re-verification evidence gate (#3304): the census's evidence living in gitignored `.scratch/` remains BY DESIGN (owner standing rule) — committed rows carry terminal one-liners and this verifier re-confirmed the manifest (28 outcomes, 0 None) and all 26 logs exist on disk.

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| tests/examples/test_notebook_execution.py | EXEC-01/06 | ACTIVE 13 + gated 8 + kill/partial + 13 fast contract classes (incl. the new evo content contracts) | gated skip only with live probe evidence | none | Behavioral (execution outcomes, artifacts, tree-clean, kernel delta-zero, probe evidence in skip messages) | OK |
| tests/utils/test_transformers_compat.py | EXEC-01/REPAIR-03 slice | 88 | 0 | none | Value/behavioral (identity, shapes, arithmetic, idempotence, absence contracts) | OK |
| tests/utils/test_transformers_compat_np.py (08 addition on covered surface) | REPAIR-03 slice | 15 | 0 | none | Value/behavioral (str-encode rung, count/sep semantics, error paths) | OK |
| tests/models/test_model_remote_code.py | EXEC-01/REPAIR-03 slice | 1 (real load+forward) | self-healing skip moot (healed) | none | Value ((1,2) logits, num_labels==2) | OK |
| tests/examples/test_script_execution.py | EXEC-04 | 4 (real run + seeding contracts) | network-typed only with evidence | none | Behavioral (pkl artifact, returncode, tree-clean) | OK |
| tests/examples/test_marimo_execution.py | EXEC-03 | 3 specs (1 re-run green) | 0 | none | Behavioral (HTML size, defaults markers, failure-artifact absence) | OK |

No disabled tests proving requirements; no circular generation; no existence-only assertions on value-level requirements. The skipped gated tests carry live probe results in their skip messages — the strongest honesty form available.

### Human Verification Required

None. Infrastructure/foundation phase with no user-facing elements; every acceptance criterion was verified programmatically, and every behavioral truth was exercised by a test this verifier ran at HEAD (including the formerly-human NT smoke and rice-network lane, both long resolved). The two carried-lane executions (truth 24 Info note) are covered by code-diff proofs plus the phase-08 verifier's same-HEAD runs, not by judgment.

### Gaps Summary

No failed truths, no missing/stub artifacts, no unwired links, no unreferenced debt markers, no human-verification items. The phase goal holds at HEAD 3d27360: the compat shim family remains 10 absence-gated patches with 88+15 contract tests green and the str-encode rung (this delta's only shim change) probed live; the NT smoke is a real-green load+forward; 5 of the 8 gated lanes plus the giants-deselected evo lane were re-proven this round by direct execution or audit-green typed skip (the remaining two carried with documented code-diff proof); the fast suite is green at 1881 tests; both false-green CI gates remain closed and were live-verified (branch protection on dev+main lists both contexts; zero masking flags; validators and the sync gate green with fail-closed edges re-proven — including over the phase-08 re-executed evo notebook twins); the census remains the honest campaign-time record with every verdict evidence-traceable on disk. Open items are correctly tracked elsewhere: the D-04 dispatch deferral (phs now 362 commits unpushed) and the MCP --host/--port bug (owner-scope ledger). Status: **passed**.

---

_Verified: 2026-10-07T00:22:22Z_
_Verifier: Claude (gsd-verifier)_
_Stale-digest regeneration at HEAD 3d27360 of: 05-VERIFICATION.md @ 67b0692-era report (2026-10-06T15:55:30Z, passed 25/25, digest v3:c1d62caa… — preserved in git history). Delta verified: 67b0692..3d27360 (39 commits: phase 06/07/08 incremental code-review fix cycles and their verification regenerations)._
